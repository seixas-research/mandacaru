//! The Born-von Karman supercell Hamiltonian of a crystal (the Bloch methods, Stage 1).
//!
//! A Gamma-centered `n1 x n2 x n3` Monkhorst-Pack mesh of a primitive cell is the Gamma point of
//! the `n1 x n2 x n3` supercell. The correlated solvers act on that supercell, so its
//! Hamiltonian has to be a crystal's, not a cluster's. Four terms change from the molecular one:
//!
//! - **Coulomb kernel.** The grid is the supercell, one period exactly, and the electron-electron
//!   kernel is `4 pi / G^2` with `G = 0` dropped (`Poisson::periodic`).
//! - **External potential and ion-ion energy.** The potential of the ion lattice is the grid's
//!   own Fourier series, `-4 pi / (Omega G^2)` times the structure factor with `G = 0` dropped
//!   (the zero-mean Ewald potential truncated at the grid, with no real-space sum and no
//!   softening); the ion-ion energy is the jellium-convention Ewald energy (`periodic::ewald`).
//!   Each of the three electrostatic terms drops its own `G = 0` against a uniform background;
//!   for a neutral cell the backgrounds cancel. A charged cell is refused
//!   (`MolecularIntegrals::particles`).
//! - **Madelung term.** Dropping `G = 0` also removes every point electron's interaction with its
//!   own periodic images and the background, which nothing cancels: `1/2 N_e v_M`, with `v_M`
//!   twice the Ewald energy of one unit charge in the supercell. It is a constant in a fixed
//!   particle-number sector and is carried in `MolecularIntegrals::nuclear`.
//! - **Basis.** Every orbital is replaced by its Fourier series over the supercell, truncated at
//!   `|G| < pi / (2 max h_i)` (`sample_images`): the series sums every periodic image, and the
//!   truncation keeps every product of two orbitals inside the grid's frequency box. The
//!   matrix elements are then exact grid quadratures of band-limited functions, and a rigid
//!   translation of the crystal by any vector, not only by whole grid steps, changes nothing but
//!   phases: the energy no longer depends on where the atoms sit on the grid.
//!
//! The grid holds `n_i` times a whole number of nodes per primitive vector, so a primitive
//! translation is a whole number of grid steps and the Hamiltonian keeps the translation
//! symmetry Bloch's theorem rests on. The kinetic operator is the spectral one, periodic on the
//! grid. The kernel is periodic along all three axes: a chain or a slab is a crystal of chains
//! or slabs separated by the cell's vacuum, whose neutral images interact through multipoles only;
//! a slab with a dipole would need a two-dimensional truncated kernel, which this module does not
//! build.
//!
//! The same module unfolds supercell matrices back onto the primitive cell: the Bloch creation
//! operators `c+_{k nu} = N^{-1/2} sum_R exp(2 pi i k.R) c+_{R nu}` the spectral function needs,
//! and the mean-field band structure at any `k` (exact at the mesh points, a Fourier
//! interpolation between them).
use super::integrals::{MolecularIntegrals, PeriodicElectrostatics};
use super::invalid;
use crate::basis::{Basis, Orbital};
use crate::math::linalg::eigen;
use crate::math::storage::{Precision, RealSamples};
use crate::math::SpatialGrid;
use crate::periodic::ewald::Ewald;
use crate::quantum::C64;
use crate::{product, Context, Geometry, LengthUnit, Result};
use nalgebra::{DMatrix, Matrix3, Vector3};
use std::f64::consts::PI;
use std::sync::Arc;

/// The radial quadrature of an orbital's Fourier transform stops where `|R(r)|` stays below this.
pub const RADIAL_TOLERANCE: f64 = 1e-12;
/// Accuracy of the Ewald sums (absolute, per unit charge squared).
pub const EWALD_ACCURACY: f64 = 1e-12;
/// Radial scan step and limit (Bohr) for an orbital's support, and the quadrature step.
const RADIUS_STEP: f64 = 0.05;
const RADIUS_LIMIT: f64 = 150.;
const QUADRATURE_STEP: f64 = 0.004;
/// Knots of the tabulated radial transform `F_l(k)` on `[0, k_c]`.
const TRANSFORM_KNOTS: usize = 1025;

/// The supercell of a Gamma-centered mesh and the map from its atoms to the primitive cell.
#[derive(Clone, Debug)]
pub struct Supercell {
    /// The mesh `(n1, n2, n3)`, also the repetition of the primitive cell.
    pub mesh: [usize; 3],
    pub primitive: Geometry,
    /// The primitive cell `(i, j, k)` of every supercell atom. Atoms are cell-major, the first
    /// index slowest, the primitive atoms in order inside each cell.
    pub cells: Vec<[usize; 3]>,
    /// The primitive atom of every supercell atom.
    pub atoms: Vec<usize>,
}
/// Where each supercell orbital sits: its primitive cell (flat index, first axis slowest) and
/// its primitive orbital.
#[derive(Clone, Debug)]
pub struct OrbitalCells {
    pub cell: Vec<usize>,
    pub orbital: Vec<usize>,
    pub n_primitive: usize,
}
/// A mean-field band structure unfolded onto the primitive cell.
pub struct UnfoldedBands {
    /// `(spins, nk, n_primitive)` eigenvalues, Hartree, ascending per k.
    pub eigenvalues: Vec<f64>,
    /// `(spins, nk, n_primitive band, n_primitive orbital)` Lowdin weights of each band on the
    /// primitive orbitals; each band's weights sum to 1.
    pub weights: Vec<f64>,
    pub spins: usize,
    pub n_primitive: usize,
}

fn lattice(cell: &[[f64; 3]; 3]) -> Matrix3<f64> {
    // Columns are the lattice vectors.
    Matrix3::from_fn(|row, axis| cell[axis][row])
}

impl Supercell {
    /// The `mesh` supercell of `primitive` and its geometry (Bohr).
    pub fn new(primitive: &Geometry, mesh: [usize; 3]) -> Result<(Self, Geometry)> {
        let pbc = primitive.pbc();
        if !pbc.iter().any(|&p| p) {
            return Err(invalid(
                "a Bloch calculation needs a periodic geometry: set pbc along at least one lattice vector",
            ));
        }
        for axis in 0..3 {
            if mesh[axis] == 0 {
                return Err(invalid("every mesh size must be a positive integer"));
            }
            if !pbc[axis] && mesh[axis] != 1 {
                return Err(invalid(format!(
                    "the mesh has {} k-points along lattice vector {}, which is not periodic: use 1 there",
                    mesh[axis], axis
                )));
            }
        }
        let a = lattice(primitive.cell());
        if !a.determinant().is_finite() || a.determinant().abs() <= 1e-10 {
            return Err(invalid(
                "the cell must have three nonzero, independent lattice vectors: the cell is also the vacuum box along the directions that are not periodic",
            ));
        }
        let cells_count = product(&mesh)?;
        let n_atoms = primitive.numbers().len();
        let mut numbers = Vec::with_capacity(cells_count * n_atoms);
        let mut positions = Vec::with_capacity(cells_count * n_atoms);
        let mut cells = Vec::with_capacity(cells_count * n_atoms);
        let mut atoms = Vec::with_capacity(cells_count * n_atoms);
        for i in 0..mesh[0] {
            for j in 0..mesh[1] {
                for k in 0..mesh[2] {
                    let shift = a * Vector3::new(i as f64, j as f64, k as f64);
                    for (atom, (&z, r)) in primitive
                        .numbers()
                        .iter()
                        .zip(primitive.positions())
                        .enumerate()
                    {
                        numbers.push(z);
                        positions.push([r[0] + shift[0], r[1] + shift[1], r[2] + shift[2]]);
                        cells.push([i, j, k]);
                        atoms.push(atom);
                    }
                }
            }
        }
        let p = primitive.cell();
        let cell: [[f64; 3]; 3] =
            std::array::from_fn(|axis| std::array::from_fn(|c| p[axis][c] * mesh[axis] as f64));
        let geometry = Geometry::new(numbers, positions, cell, pbc, LengthUnit::Bohr)?;
        Ok((
            Self {
                mesh,
                primitive: primitive.clone(),
                cells,
                atoms,
            },
            geometry,
        ))
    }
    pub fn n_cells(&self) -> usize {
        self.mesh.iter().product()
    }
    /// The flat index of primitive cell `(i, j, k)`, the first index slowest.
    pub fn cell_index(&self, cell: [usize; 3]) -> usize {
        (cell[0] * self.mesh[1] + cell[1]) * self.mesh[2] + cell[2]
    }
    /// The mesh points the supercell realizes, fractional coordinates of the primitive
    /// reciprocal lattice folded into `(-1/2, 1/2]`, in the order of the cell index.
    pub fn kpoints(&self) -> Vec<[f64; 3]> {
        let fold = |j: usize, n: usize| {
            let f = j as f64 / n as f64;
            if f > 0.5 + 1e-12 {
                f - 1.
            } else {
                f
            }
        };
        let mut out = Vec::with_capacity(self.n_cells());
        for i in 0..self.mesh[0] {
            for j in 0..self.mesh[1] {
                for k in 0..self.mesh[2] {
                    out.push([
                        fold(i, self.mesh[0]),
                        fold(j, self.mesh[1]),
                        fold(k, self.mesh[2]),
                    ]);
                }
            }
        }
        out
    }
    /// The commensurate periodic grid of the supercell `geometry`: `round(|a_i| / spacing)`
    /// nodes per primitive vector (at least one), times `n_i`, the origin at the lattice origin.
    pub fn grid(&self, geometry: &Geometry, spacing: f64) -> Result<SpatialGrid> {
        if !spacing.is_finite() || spacing <= 0. {
            return Err(invalid("grid spacing must be positive and finite"));
        }
        let primitive = self.primitive.cell();
        let mut shape = [0usize; 3];
        for axis in 0..3 {
            let length = primitive[axis].iter().map(|v| v * v).sum::<f64>().sqrt();
            let nodes = (length / spacing).round_ties_even().max(1.);
            if nodes > 1e6 {
                return Err(crate::Error::Overflow);
            }
            shape[axis] = (nodes as usize)
                .checked_mul(self.mesh[axis])
                .ok_or(crate::Error::Overflow)?;
        }
        let cell = geometry.cell();
        let mut step = [0.; 9];
        for axis in 0..3 {
            for row in 0..3 {
                step[row * 3 + axis] = cell[axis][row] / shape[axis] as f64;
            }
        }
        SpatialGrid::new(shape, [0.; 3], step, true)
    }
    /// The cell and primitive orbital of every supercell orbital, checked against the orbital
    /// centers and shapes rather than assumed from the build order.
    pub fn orbital_cells(&self, basis: &Basis) -> Result<OrbitalCells> {
        let n_atoms = self.primitive.numbers().len();
        if basis.orbitals.iter().any(|o| o.atom >= self.atoms.len()) {
            return Err(invalid("the basis does not belong to this supercell"));
        }
        // Orbitals of each supercell atom, in basis order.
        let mut by_atom: Vec<Vec<usize>> = vec![Vec::new(); self.atoms.len()];
        for (index, orbital) in basis.orbitals.iter().enumerate() {
            by_atom[orbital.atom].push(index);
        }
        // The primitive orbitals: those of the atoms of cell (0, 0, 0), primitive atom order.
        let mut offset = vec![0usize; n_atoms];
        let mut n_primitive = 0;
        for atom in 0..n_atoms {
            offset[atom] = n_primitive;
            n_primitive += by_atom[atom].len();
        }
        let a = lattice(self.primitive.cell());
        let mut cell = vec![0; basis.orbitals.len()];
        let mut orbital = vec![0; basis.orbitals.len()];
        for (atom, indices) in by_atom.iter().enumerate() {
            let primitive_atom = self.atoms[atom];
            let reference = &by_atom[primitive_atom];
            if indices.len() != reference.len() {
                return Err(invalid(
                    "every copy of a primitive atom must carry the same orbitals",
                ));
            }
            let c = self.cells[atom];
            let shift = a * Vector3::new(c[0] as f64, c[1] as f64, c[2] as f64);
            for (local, (&index, &source)) in indices.iter().zip(reference).enumerate() {
                let (o, s): (&Orbital, &Orbital) =
                    (&basis.orbitals[index], &basis.orbitals[source]);
                let moved = (0..3).all(|j| (o.center[j] - s.center[j] - shift[j]).abs() <= 1e-8);
                if o.l != s.l || o.m != s.m || !moved {
                    return Err(invalid(
                        "a supercell orbital is not a lattice translate of its primitive orbital",
                    ));
                }
                cell[index] = self.cell_index(c);
                orbital[index] = offset[primitive_atom] + local;
            }
        }
        Ok(OrbitalCells {
            cell,
            orbital,
            n_primitive,
        })
    }
    /// Bloch creation operators in a supercell orbital basis: for each `k`, the
    /// `(n_primitive, n)` matrix `w` with `c+_{k nu} = sum_p w[nu, p] a+_p`, where `a+_p`
    /// creates the column `p` of `orthonormal` (the supercell orbitals in the Lowdin basis,
    /// orthonormal columns) and `c+_{R nu}` the Lowdin orbital `nu` of cell `R`.
    pub fn bloch_operators(
        &self,
        cells: &OrbitalCells,
        orthonormal: &DMatrix<C64>,
        kpoints: &[[f64; 3]],
    ) -> Result<Vec<DMatrix<C64>>> {
        let m = cells.cell.len();
        if orthonormal.nrows() != m {
            return Err(invalid(
                "the orbital coefficients do not match the supercell basis",
            ));
        }
        let n = orthonormal.ncols();
        let norm = 1. / (self.n_cells() as f64).sqrt();
        let translations = self.cell_translations();
        kpoints
            .iter()
            .map(|k| {
                crate::finite(k, "k-point")?;
                let mut w = DMatrix::<C64>::zeros(cells.n_primitive, n);
                for mu in 0..m {
                    let t = translations[cells.cell[mu]];
                    let phase =
                        C64::from_polar(norm, 2. * PI * (k[0] * t[0] + k[1] * t[1] + k[2] * t[2]));
                    for p in 0..n {
                        // c+_mu = sum_p conj(C[mu, p]) a+_p for a unitary C.
                        w[(cells.orbital[mu], p)] += phase * orthonormal[(mu, p)].conj();
                    }
                }
                Ok(w)
            })
            .collect()
    }
    /// The integer translation of every primitive cell, in cell-index order.
    fn cell_translations(&self) -> Vec<[f64; 3]> {
        let mut out = Vec::with_capacity(self.n_cells());
        for i in 0..self.mesh[0] {
            for j in 0..self.mesh[1] {
                for k in 0..self.mesh[2] {
                    out.push([i as f64, j as f64, k as f64]);
                }
            }
        }
        out
    }
    /// Unfold supercell mean-field matrices onto the primitive cell and diagonalize at `kpoints`.
    ///
    /// `overlap` and `fock[s]` are `m x m` matrices of the supercell's (image-summed) atomic
    /// orbitals; `centers` their centers (Bohr). The matrix element of the pair
    /// `(0, nu), (R, nu')` is assigned to the image of `R` modulo the supercell that is closest
    /// to orbital `(0, nu)`, shared equally between equidistant images, and
    /// `H(k) = sum_R exp(2 pi i k.R) F_{(0,nu),(R,nu')}`. At a mesh point the phase of every
    /// supercell image is 1, so the result is the supercell spectrum exactly (the Born-von Karman
    /// identity); between mesh points it is a Fourier interpolation whose error falls with the
    /// supercell size as the matrix elements decay.
    pub fn unfold(
        &self,
        cells: &OrbitalCells,
        centers: &[[f64; 3]],
        overlap: &DMatrix<f64>,
        fock: &[DMatrix<f64>],
        kpoints: &[[f64; 3]],
    ) -> Result<UnfoldedBands> {
        let m = cells.cell.len();
        let np = cells.n_primitive;
        if centers.len() != m
            || overlap.shape() != (m, m)
            || fock.is_empty()
            || fock.iter().any(|f| f.shape() != (m, m))
        {
            return Err(invalid("unfolding needs m x m overlap and Fock matrices"));
        }
        // The orbitals of cell 0, by primitive orbital.
        let mut home = vec![usize::MAX; np];
        for mu in 0..m {
            if cells.cell[mu] == 0 {
                home[cells.orbital[mu]] = mu;
            }
        }
        if home.contains(&usize::MAX) {
            return Err(invalid("cell 0 lacks a primitive orbital"));
        }
        let translations = self.cell_translations();
        let super_lattice = lattice(&std::array::from_fn(|axis| {
            std::array::from_fn(|c| self.primitive.cell()[axis][c] * self.mesh[axis] as f64)
        }));
        // For each (nu, mu): the supercell images closest to orbital (0, nu), as integer
        // primitive translations of the pair.
        let mut images: Vec<Vec<[f64; 3]>> = Vec::with_capacity(np * m);
        for &origin in &home {
            for mu in 0..m {
                let d0 = Vector3::from(centers[mu]) - Vector3::from(centers[origin]);
                let mut best = f64::INFINITY;
                let mut chosen = Vec::new();
                for l0 in -2i32..=2 {
                    for l1 in -2i32..=2 {
                        for l2 in -2i32..=2 {
                            let l = [l0, l1, l2];
                            if (0..3)
                                .any(|a| self.mesh[a] == 1 && l[a] != 0 && !self.primitive.pbc()[a])
                            {
                                continue;
                            }
                            let d =
                                d0 + super_lattice * Vector3::new(l0 as f64, l1 as f64, l2 as f64);
                            let length = d.norm();
                            let t = translations[cells.cell[mu]];
                            let shift =
                                std::array::from_fn(|a| t[a] + (l[a] as f64) * self.mesh[a] as f64);
                            if length < best - 1e-6 {
                                best = length;
                                chosen.clear();
                                chosen.push(shift);
                            } else if (length - best).abs() <= 1e-6 {
                                chosen.push(shift);
                            }
                        }
                    }
                }
                images.push(chosen);
            }
        }
        let spins = fock.len();
        let nk = kpoints.len();
        let mut eigenvalues = Vec::with_capacity(product(&[spins, nk, np])?);
        let mut weights = Vec::with_capacity(product(&[spins, nk, np, np])?);
        let assemble = |source: &DMatrix<f64>, k: &[f64; 3]| {
            let mut out = DMatrix::<C64>::zeros(np, np);
            for (nu, &origin) in home.iter().enumerate() {
                for mu in 0..m {
                    let value = source[(origin, mu)];
                    if value == 0. {
                        continue;
                    }
                    let list = &images[nu * m + mu];
                    let share = value / list.len() as f64;
                    for t in list {
                        out[(nu, cells.orbital[mu])] += C64::from_polar(
                            share,
                            2. * PI * (k[0] * t[0] + k[1] * t[1] + k[2] * t[2]),
                        );
                    }
                }
            }
            // Hermitian part: equidistant images can leave a round-off asymmetry.
            (&out + out.adjoint()) * C64::new(0.5, 0.)
        };
        for f in fock {
            for k in kpoints {
                crate::finite(k, "k-point")?;
                let s = assemble(overlap, k);
                let h = assemble(f, k);
                let (values, vectors) = eigen(s)?;
                if values[0] <= 1e-10 {
                    return Err(invalid(
                        "the unfolded overlap is singular at this k-point: the basis is linearly dependent",
                    ));
                }
                let inverse_root = &vectors
                    * DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
                        np,
                        values.iter().map(|v| C64::new(1. / v.sqrt(), 0.)),
                    ))
                    * vectors.adjoint();
                let orthonormal = &inverse_root * h * &inverse_root;
                let (bands, states) =
                    eigen((&orthonormal + orthonormal.adjoint()) * C64::new(0.5, 0.))?;
                eigenvalues.extend_from_slice(&bands);
                for b in 0..np {
                    for nu in 0..np {
                        weights.push(states[(nu, b)].norm_sqr());
                    }
                }
            }
        }
        Ok(UnfoldedBands {
            eigenvalues,
            weights,
            spins,
            n_primitive: np,
        })
    }
}

/// The largest radius at which the radial factor of `orbital` exceeds `RADIAL_TOLERANCE`.
fn support_radius(orbital: &Orbital) -> Result<f64> {
    let steps = (RADIUS_LIMIT / RADIUS_STEP) as usize;
    let mut radius = RADIUS_STEP;
    for step in 1..=steps {
        let r = step as f64 * RADIUS_STEP;
        if orbital.radial_value(r).abs() > RADIAL_TOLERANCE {
            radius = r + RADIUS_STEP;
        }
    }
    if radius >= RADIUS_LIMIT {
        return Err(invalid(format!(
            "an orbital still exceeds {RADIAL_TOLERANCE:e} at {RADIUS_LIMIT} Bohr from its center"
        )));
    }
    Ok(radius)
}

/// The band limit of a periodic grid: half the radius `pi / max(h_i)` of the sphere inscribed in
/// its frequency box, so that a product of two band-limited orbitals stays inside the box.
pub fn band_limit(grid: &SpatialGrid) -> f64 {
    PI / (2. * grid.spacings().into_iter().fold(0f64, f64::max))
}

/// The grid's reciprocal-lattice vectors: `(flat index, G)` for every node of the FFT grid.
fn wavevectors(grid: &SpatialGrid) -> Result<Vec<(usize, Vector3<f64>)>> {
    let shape = grid.index.shape();
    let cell = grid.step
        * Matrix3::from_diagonal(&Vector3::new(
            shape[0] as f64,
            shape[1] as f64,
            shape[2] as f64,
        ));
    let reciprocal = cell
        .try_inverse()
        .ok_or_else(|| invalid("the periodic cell is singular"))?
        .transpose()
        * (2. * PI);
    let mut out = Vec::with_capacity(grid.index.points());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let m = Vector3::new(
                    crate::math::fft::frequency(i, shape[0]),
                    crate::math::fft::frequency(j, shape[1]),
                    crate::math::fft::frequency(k, shape[2]),
                );
                out.push(((i * shape[1] + j) * shape[2] + k, reciprocal * m));
            }
        }
    }
    Ok(out)
}

/// `sum_G c(G) exp(i G . r_n)` on the grid nodes `r_n = origin + step n` (real part).
fn synthesize(
    ctx: &Context,
    grid: &SpatialGrid,
    coefficients: &mut [C64],
    out: &mut [f64],
) -> Result<()> {
    let n = coefficients.len() as f64;
    crate::math::fft::Fft3::new(ctx, grid.index.shape())?.transform(coefficients, true)?;
    for (value, c) in out.iter_mut().zip(coefficients.iter()) {
        *value = c.re * n;
    }
    Ok(())
}

/// The Fourier transform of a radial factor, `F_l(k) = int R(r) j_l(k r) r^2 dr`, tabulated on
/// `[0, k_c]` (Simpson in `r`, a cubic spline in `k`).
fn radial_transform(
    ctx: &Context,
    orbital: &Orbital,
    cutoff: f64,
) -> Result<crate::math::spline::Cubic> {
    let radius = support_radius(orbital)?;
    let intervals = 2 * (radius / QUADRATURE_STEP / 2.).ceil() as usize;
    let dr = radius / intervals as f64;
    let weighted: Vec<f64> = (0..=intervals)
        .map(|j| {
            let r = j as f64 * dr;
            let simpson = if j == 0 || j == intervals {
                1.
            } else if j % 2 == 1 {
                4.
            } else {
                2.
            };
            simpson * dr / 3. * orbital.radial_value(r) * r * r
        })
        .collect();
    let mut knots = ctx.allocate::<f64>(TRANSFORM_KNOTS)?;
    let mut values = ctx.allocate::<f64>(TRANSFORM_KNOTS)?;
    for i in 0..TRANSFORM_KNOTS {
        let k = cutoff * i as f64 / (TRANSFORM_KNOTS - 1) as f64;
        knots.as_mut_slice()[i] = k;
        values.as_mut_slice()[i] = weighted
            .iter()
            .enumerate()
            .map(|(j, w)| {
                w * mandacaru_kernels::two_center::spherical_bessel(
                    orbital.l as i32,
                    k * j as f64 * dr,
                )
            })
            .sum();
    }
    crate::math::spline::Cubic::new(ctx, knots, values)
}

/// Band-limited, image-summed orbitals sampled on the periodic `grid` (float64).
///
/// Each orbital `R(r) Y_lm` is replaced by its Fourier series over the supercell truncated at
/// `|G| < k_c` (`band_limit`): `c(G) = 4 pi (-i)^l Y_lm(G^) F_l(|G|) exp(-i G . R) / Omega`.
/// The series sums every periodic image exactly, and sampling it is exact: a sub-grid shift of
/// an atom only changes the phases, so the matrix elements are invariant under any rigid
/// translation of the crystal, not only under whole grid steps. Point sampling of the
/// unfiltered orbital aliases its cusp and made the energy depend on where the atoms sat on the
/// grid (by up to 0.65 eV per cell on an H2 chain at grid_spacing 0.2 A). `primitive[a]` names
/// the primitive orbital of orbital `a`, whose copies share one radial transform.
pub fn sample_images(
    ctx: &Context,
    basis: &Basis,
    grid: &SpatialGrid,
    primitive: &[usize],
) -> Result<RealSamples> {
    if !grid.periodic {
        return Err(invalid("image sums need a periodic grid"));
    }
    if primitive.len() != basis.orbitals.len() {
        return Err(invalid("one primitive label per orbital"));
    }
    let g = grid.index.points();
    let m = basis.orbitals.len();
    let cutoff = band_limit(grid);
    let volume = grid.index.dv() * g as f64;
    let inside: Vec<(usize, Vector3<f64>)> = wavevectors(grid)?
        .into_iter()
        .filter(|(_, q)| q.norm() < cutoff)
        .collect();
    let origin = Vector3::from(grid.origin);
    let _memory = ctx.reserve(product(&[inside.len(), 16, 2])?)?;
    // The angular and radial factor of each primitive orbital on the inside wavevectors.
    let mut shapes: Vec<Option<Vec<C64>>> = vec![None; primitive.iter().max().map_or(0, |p| p + 1)];
    let mut out = RealSamples::zeros(ctx, Precision::Float64, product(&[m, g])?)?;
    let mut coefficients = ctx.allocate::<C64>(g)?;
    let mut row = ctx.allocate::<f64>(g)?;
    for (index, orbital) in basis.orbitals.iter().enumerate() {
        let slot = &mut shapes[primitive[index]];
        if slot.is_none() {
            let transform = radial_transform(ctx, orbital, cutoff)?;
            let phase = C64::new(0., -1.).powu(orbital.l as u32) * (4. * PI / volume);
            *slot = Some(
                inside
                    .iter()
                    .map(|(_, q)| {
                        let k = q.norm();
                        let direction = if k > 0. {
                            [q[0] / k, q[1] / k, q[2] / k]
                        } else {
                            [0., 0., 1.]
                        };
                        phase
                            * crate::math::angular::normalized(orbital.l, orbital.m, direction)
                            * transform.value_slope(k).0
                    })
                    .collect(),
            );
        }
        let shape = slot.as_ref().expect("filled above");
        let center = Vector3::from(orbital.center);
        coefficients.as_mut_slice().fill(C64::default());
        for ((at, q), value) in inside.iter().zip(shape) {
            coefficients.as_mut_slice()[*at] =
                value * C64::from_polar(1., q.dot(&(origin - center)));
        }
        synthesize(ctx, grid, coefficients.as_mut_slice(), row.as_mut_slice())?;
        out.fill_row(index, g, row.as_slice())?;
    }
    Ok(out)
}

/// `v_M`: twice the jellium Ewald energy of one unit charge per cell.
pub fn madelung_constant(ctx: &Context, cell: &[[f64; 3]; 3]) -> Result<f64> {
    let flat: [f64; 9] = std::array::from_fn(|i| cell[i / 3][i % 3]);
    Ok(2.
        * Ewald::new(ctx, &[[0.; 3]], &[1.], flat, EWALD_ACCURACY, None)?
            .evaluate(ctx)?
            .energy)
}

fn electrostatics(
    ctx: &Context,
    geometry: &Geometry,
    grid: &SpatialGrid,
    supercell: Arc<Supercell>,
) -> Result<PeriodicElectrostatics> {
    let cell = geometry.cell();
    let flat: [f64; 9] = std::array::from_fn(|i| cell[i / 3][i % 3]);
    let charges: Vec<f64> = geometry.numbers().iter().map(|&z| z as f64).collect();
    let ions = Ewald::new(
        ctx,
        geometry.positions(),
        &charges,
        flat,
        EWALD_ACCURACY,
        None,
    )?
    .evaluate(ctx)?
    .energy;
    // The potential of the ion lattice as the grid's own Fourier series, G = 0 dropped (the
    // uniform background): exact for the band-limited orbitals, with no real-space sum and no
    // softening, so it moves with the ions by phases only.
    let g = grid.index.points();
    let volume = grid.index.dv() * g as f64;
    let origin = Vector3::from(grid.origin);
    let mut coefficients = ctx.allocate::<C64>(g)?;
    for (at, q) in wavevectors(grid)? {
        let q2 = q.norm_squared();
        if q2 <= 0. {
            continue;
        }
        let structure: C64 = geometry
            .positions()
            .iter()
            .zip(&charges)
            .map(|(r, z)| C64::from_polar(*z, q.dot(&(origin - Vector3::from(*r)))))
            .sum();
        coefficients.as_mut_slice()[at] = structure * (-4. * PI / (volume * q2));
    }
    let mut external = ctx.allocate::<f64>(g)?;
    synthesize(
        ctx,
        grid,
        coefficients.as_mut_slice(),
        external.as_mut_slice(),
    )?;
    let electrons: f64 = charges.iter().sum();
    Ok(PeriodicElectrostatics {
        external,
        ions,
        madelung: 0.5 * electrons * madelung_constant(ctx, cell)?,
        supercell,
    })
}

/// The integrals of `supercell` on its commensurate `grid` from an assembled `basis`.
pub(super) fn build(
    ctx: &Context,
    supercell: Arc<Supercell>,
    geometry: Geometry,
    grid: SpatialGrid,
    basis: Basis,
    cholesky: Option<f64>,
) -> Result<MolecularIntegrals> {
    let shape = grid.index.shape();
    let spanned = grid.step
        * Matrix3::from_diagonal(&Vector3::new(
            shape[0] as f64,
            shape[1] as f64,
            shape[2] as f64,
        ));
    let cell = lattice(geometry.cell());
    if (spanned - cell).abs().max() > 1e-8 * cell.abs().max() {
        return Err(invalid(
            "the periodic grid does not span the supercell exactly",
        ));
    }
    for axis in 0..3 {
        if shape[axis] % supercell.mesh[axis] != 0 {
            return Err(invalid(
                "the periodic grid is not commensurate with the primitive cell",
            ));
        }
    }
    let cells = supercell.orbital_cells(&basis)?;
    let samples = sample_images(ctx, &basis, &grid, &cells.orbital)?;
    let terms = electrostatics(ctx, &geometry, &grid, supercell)?;
    MolecularIntegrals::assemble(
        ctx,
        geometry,
        grid,
        basis,
        samples,
        "spectral",
        cholesky,
        None,
        Some(terms),
    )
}

impl MolecularIntegrals {
    /// The Born-von Karman supercell Hamiltonian of the Gamma-centered `mesh` of `primitive`,
    /// on the commensurate periodic grid of `spacing` (Bohr). Float64 only: the orbital stack is
    /// image-summed once, and the float32 path re-evaluates single orbitals.
    pub fn supercell(
        ctx: &Context,
        primitive: &Geometry,
        mesh: [usize; 3],
        basis_spec: &str,
        spacing: f64,
        cholesky: Option<f64>,
        precision: Precision,
    ) -> Result<Self> {
        if precision != Precision::Float64 {
            return Err(invalid(
                "the periodic supercell Hamiltonian is built in float64 only: use precision='float64'",
            ));
        }
        ctx.run(|| {
            let (supercell, geometry) = Supercell::new(primitive, mesh)?;
            let grid = supercell.grid(&geometry, spacing)?;
            let basis = Basis::build(ctx, &geometry, basis_spec)?;
            build(ctx, Arc::new(supercell), geometry, grid, basis, cholesky)
        })
    }
}

/// The mean-field bands of a supercell calculation on the primitive cell, at any `kpoints`.
///
/// The Fock matrix of each spin channel is rebuilt from the converged orbitals and energies in
/// the Lowdin frame, `F = C diag(e) C+`, taken back to the atomic orbitals,
/// `F_AO = S^{1/2} F S^{1/2}`, and unfolded with [`Supercell::unfold`].
pub fn mean_field_bands(
    integrals: &MolecularIntegrals,
    mean_field: &super::MeanField,
    kpoints: &[[f64; 3]],
) -> Result<UnfoldedBands> {
    let supercell = integrals.supercell.as_ref().ok_or_else(|| {
        invalid("bands on the primitive cell need a periodic supercell calculation")
    })?;
    if mean_field.method == "ghf" {
        return Err(invalid(
            "unfolded bands are defined for RHF and UHF references",
        ));
    }
    let m = integrals.m();
    let cells = supercell.orbital_cells(&integrals.basis)?;
    let centers: Vec<[f64; 3]> = integrals.basis.orbitals.iter().map(|o| o.center).collect();
    let overlap = DMatrix::from_row_slice(m, m, integrals.overlap.as_slice());
    // S^{1/2} = S S^{-1/2}; `lowdin` is the symmetric S^{-1/2}.
    let root = &overlap * &integrals.lowdin;
    let mut fock = Vec::with_capacity(mean_field.coefficients.len());
    for (c, e) in mean_field.coefficients.iter().zip(&mean_field.energies) {
        if c.shape() != (m, m) || e.len() != m {
            return Err(invalid("the mean field does not span the supercell basis"));
        }
        let diagonal = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            m,
            e.iter().map(|v| C64::new(*v, 0.)),
        ));
        let lowdin_fock = c * diagonal * c.adjoint();
        let real = DMatrix::from_fn(m, m, |i, j| {
            0.5 * (lowdin_fock[(i, j)].re + lowdin_fock[(j, i)].re)
        });
        fock.push(&root * real * &root);
    }
    integrals
        .context
        .run(|| supercell.unfold(&cells, &centers, &overlap, &fock, kpoints))
}

/// The Lehmann spectral function of a correlated supercell state at the mesh points:
/// `A_{nu nu}(k, omega)` for every primitive orbital `nu` (the diagonal of the spectral
/// matrix in the Lowdin orbitals of the primitive cell), both spins.
pub struct BlochSpectral {
    pub kpoints: Vec<[f64; 3]>,
    pub n_primitive: usize,
    /// `parts[k * n_primitive + nu]`.
    pub parts: crate::quantum::spectral::Spectral,
}

/// `A(k, omega)` of `state` (occupation form, the model's spin-orbital order) in the supercell
/// model `model` built on the orbitals of `mean_field`. The model must span every supercell
/// orbital with no frozen core and no rotation, since the Bloch operators are expansions over
/// all of them; `kpoints` must be mesh points (the commensurate ones are the only ones a
/// supercell state resolves).
pub fn bloch_spectral(
    ctx: &Context,
    integrals: &MolecularIntegrals,
    mean_field: &super::MeanField,
    model: &super::transform::Model,
    state: &crate::quantum::spectral::OccupationState,
    kpoints: &[[f64; 3]],
    krylov: usize,
) -> Result<BlochSpectral> {
    let supercell = integrals.supercell.as_ref().ok_or_else(|| {
        invalid("a spectral function on the Brillouin zone needs a periodic supercell calculation")
    })?;
    let m = integrals.m();
    let full: Vec<usize> = (0..m).collect();
    if model.spinor
        || model.m != m
        || !model.frozen_indices.is_empty()
        || model.orbital_rotation.is_some()
        || model.orbital_indices != full
    {
        return Err(invalid(
            "the spectral function needs the full orbital space of the supercell: run without active_space",
        ));
    }
    if mean_field.model_orbitals.shape() != (m, m) {
        return Err(invalid("the mean field does not span the supercell basis"));
    }
    for k in kpoints {
        crate::finite(k, "k-point")?;
        for axis in 0..3 {
            let x = k[axis] * supercell.mesh[axis] as f64;
            if (x - x.round()).abs() > 1e-9 {
                return Err(invalid(format!(
                    "k = {k:?} is not a point of the {}x{}x{} mesh: a supercell state resolves the spectral function at its commensurate k-points only",
                    supercell.mesh[0], supercell.mesh[1], supercell.mesh[2]
                )));
            }
        }
    }
    let cells = supercell.orbital_cells(&integrals.basis)?;
    let operators = supercell.bloch_operators(&cells, &mean_field.model_orbitals, kpoints)?;
    let mut rows = Vec::with_capacity(kpoints.len() * cells.n_primitive);
    for w in &operators {
        for nu in 0..cells.n_primitive {
            rows.push((0..m).map(|p| w[(nu, p)]).collect::<Vec<_>>());
        }
    }
    let parts = ctx.run(|| {
        crate::quantum::spectral::lehmann(ctx, model, model.particles, state, &rows, krylov)
    })?;
    Ok(BlochSpectral {
        kpoints: kpoints.to_vec(),
        n_primitive: cells.n_primitive,
        parts,
    })
}

impl BlochSpectral {
    /// Per `(k, nu, spin)`, first axis slowest: the energy and weight of the strongest pole of
    /// `A_{nu nu}(k, omega)` (addition and removal together), and the total weight, which is
    /// 1 by the sum rule.
    pub fn summary(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let rows = self.parts.addition.len();
        let mut energy = Vec::with_capacity(rows * 2);
        let mut weight = Vec::with_capacity(rows * 2);
        let mut total = Vec::with_capacity(rows * 2);
        for (add, remove) in self.parts.addition.iter().zip(&self.parts.removal) {
            for s in 0..2 {
                let mut best = (f64::NAN, 0f64);
                for poles in [&add[s], &remove[s]] {
                    for (e, w) in poles.energies.iter().zip(&poles.weights) {
                        if *w > best.1 {
                            best = (*e, *w);
                        }
                    }
                }
                energy.push(best.0);
                weight.push(best.1);
                total.push(add[s].weight() + remove[s].weight());
            }
        }
        (energy, weight, total)
    }
}

impl BlochSpectral {
    /// The highest removal pole and the lowest addition pole carrying weight above
    /// `threshold`, over every k, orbital and spin: `-IP` and `-EA` relative to `E0`. Their
    /// midpoint is the chemical potential of the state.
    pub fn edges(&self, threshold: f64) -> (f64, f64) {
        let mut removal = f64::NEG_INFINITY;
        let mut addition = f64::INFINITY;
        for (add, remove) in self.parts.addition.iter().zip(&self.parts.removal) {
            for s in 0..2 {
                for (e, w) in remove[s].energies.iter().zip(&remove[s].weights) {
                    if *w > threshold {
                        removal = removal.max(*e);
                    }
                }
                for (e, w) in add[s].energies.iter().zip(&add[s].weights) {
                    if *w > threshold {
                        addition = addition.min(*e);
                    }
                }
            }
        }
        (removal, addition)
    }
}

impl BlochSpectral {
    /// `points` energies spanning every pole with `8 eta` of margin on each side.
    pub fn default_axis(&self, eta: f64, points: usize) -> Result<Vec<f64>> {
        if !eta.is_finite() || eta <= 0. || points < 2 {
            return Err(invalid(
                "the axis needs a positive broadening and at least two points",
            ));
        }
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for part in self.parts.addition.iter().chain(&self.parts.removal) {
            for poles in part {
                for e in &poles.energies {
                    low = low.min(*e);
                    high = high.max(*e);
                }
            }
        }
        if !low.is_finite() {
            return Err(invalid("the spectral function has no poles"));
        }
        let (low, high) = (low - 8. * eta, high + 8. * eta);
        Ok((0..points)
            .map(|i| low + (high - low) * i as f64 / (points - 1) as f64)
            .collect())
    }
    /// `(nk, axis)`: the trace of `A(k, omega)` over orbitals and spins, Lorentzian of
    /// half-width `eta` (per energy unit, per cell).
    pub fn broadened(&self, axis: &[f64], eta: f64) -> Result<Vec<f64>> {
        let n = self.n_primitive;
        let mut out = Vec::with_capacity(self.kpoints.len() * axis.len());
        for k in 0..self.kpoints.len() {
            let mut parts = Vec::with_capacity(4 * n);
            for row in k * n..(k + 1) * n {
                for s in 0..2 {
                    parts.push(&self.parts.addition[row][s]);
                    parts.push(&self.parts.removal[row][s]);
                }
            }
            out.extend(crate::quantum::spectral::lorentzian(&parts, axis, eta)?);
        }
        Ok(out)
    }
}
