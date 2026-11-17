//! Stage 4a of the Bloch methods: the PAW-LCAO supercell from the crystal setup
//! (`BLOCH_VQE_PLAN.md`, Stage 4 design 4B).
//!
//! The `n1 x n2 x n3` supercell of a Gamma-centered mesh is built from the crystal's own
//! [`PeriodicSetup`] at that mesh, so every PAW fact (datasets, filtered shells, projectors,
//! Bloch-style compensation, Gaussian ions) has one owner. The supercell function of cell `R`
//! and primitive function `mu` is the image sum `chi_{R mu} = (1/N) sum_k e^{-ik.R} chi_{mu k}`
//! of the Bloch sums `chi_{mu k} = sum_L e^{ik.L} chi_mu(r - L)`: the primitive function moved by
//! `R` and summed over the supercell's images. Its one-body blocks fold the per-k matrices,
//!
//! `X_{R mu, R' nu} = (1/N) sum_k e^{ik.(R - R')} X_{mu nu}(k)`,
//!
//! for the overlap `S`, the fixed one-body operator `fixed(k) = T + V_sr + C D_ion C^H` and the
//! projections `C_{R mu, R' i}` onto the projector `i` of the atoms of cell `R'`, folded the same
//! way. On a Gamma-centered mesh the mesh is closed under `k -> -k`; the basis functions are
//! real, so `S(-k) = S(k)^*` and `fixed(-k) = fixed(k)^*` and the folded `S` and `fixed` are
//! real (their imaginary parts are measured and refused above [`REAL_TOLERANCE`]). The
//! projectors carry complex harmonics (`C(-k) = C(k)^* Pi^T`, `Pi` the signed permutation
//! `m -> -m`), so the folded projections stay complex, in the projector layout of the crystal's
//! moment blocks. [`PawSupercell::unfold`] is the inverse at a mesh point,
//! `X(k) = sum_{R'} e^{-ik.(R - R')} X_{R mu, R' nu}` for any `R` (the blocks depend on `R - R'`
//! only), the gate S4-2 of the plan: the crystal's `S(k)` and `fixed(k)` recovered from the
//! supercell blocks to 1e-12.
//!
//! Stage 4b builds the Hamiltonian of the quantum methods on it
//! ([`MolecularIntegrals::paw_supercell`]):
//! - the one-body operator `h(k)` is the crystal's bare operator
//!   ([`crate::periodic::solver::bare_hamiltonian`]: `fixed(k)`, the field of the Gaussian ions on
//!   the grid and the couplings of the compensation charges to the ions), folded like `S`;
//! - the two-body integrals are the PAW pair densities of the supercell's Lowdin functions with
//!   the periodic kernel `4 pi / G^2`, `G = 0` dropped
//!   ([`crate::periodic::wannier::downfold::basis`]), factored by the pivoted Cholesky;
//! - the constants are the crystal's ion-ion energy per cell (`eion + ion_constant`) times the
//!   cells, and the Madelung term `1/2 N_e v_M` of the supercell (the exchange's `G = 0`, as on
//!   the hydrogen-like supercell of Stage 1).
//!
//! The dataset coupling in `fixed(k)` is screened by the reference atom (Kohn-Sham); `h` takes
//! the molecular PAW Hartree-Fock one-center operator in its place
//! ([`crate::periodic::wannier::downfold::basis::hartree_fock_onsite`], `C(k) Delta C(k)^H`: the
//! reference screening removed, the frozen core's exchange with the valence added), so a
//! frozen core without core states (or with a partially filled subshell) is refused by name, as in
//! the molecule.
//!
//! Not in the first version, refused by name: a spin-orbit dataset, DFT+U,
//! `precision="float32"`.
use super::super::{integrals::Translations, invalid, MolecularIntegrals, SpatialGrid};
use super::{madelung_constant, Supercell};
use crate::basis::{Basis, Orbital, Radial};
use crate::math::linalg::real_eigen;
use crate::math::spline::Cubic;
use crate::math::storage::{Precision, RealSamples};
use crate::periodic::setup::PeriodicSetup;
use crate::periodic::wannier::downfold::basis::{basis_samples, hartree_fock_onsite, BasisCoulomb};
use crate::periodic::{solver::bare_hamiltonian, Crystal, KPoint};
use crate::quantum::C64;
use crate::{product, Context, Geometry, LengthUnit, Result};
use nalgebra::DMatrix;
use std::f64::consts::PI;
use std::sync::Arc;

/// `U` with `chi^real = chi U`: per shell (the functions of one atom and one radial table), the
/// real harmonics `Y_l0`, `sqrt(2) (-1)^m Re Y_lm` (labeled `m > 0`) and `sqrt(2) (-1)^m Im Y_lm`
/// (labeled `-m`) from the complex Condon-Shortley `Y_{l,+-m}` (`Y_{l,-m} = (-1)^m Y_lm^*`).
fn real_harmonics(
    description: &crate::periodic::description::Description,
    m: usize,
) -> Result<DMatrix<C64>> {
    let (atom, table, label) = (
        &description.function_atom,
        &description.function_table,
        &description.function_m,
    );
    if atom.len() != m || table.len() != m || label.len() != m {
        return Err(invalid(
            "the crystal's basis description does not list its functions",
        ));
    }
    let partner = |mu: usize, wanted: i32| {
        (0..m).find(|&nu| atom[nu] == atom[mu] && table[nu] == table[mu] && label[nu] == wanted)
    };
    let half = std::f64::consts::FRAC_1_SQRT_2;
    let mut out = DMatrix::<C64>::zeros(m, m);
    for mu in 0..m {
        let value = label[mu];
        if value == 0 {
            out[(mu, mu)] = C64::new(1., 0.);
            continue;
        }
        let a = value.abs();
        let sign = if a % 2 == 0 { 1. } else { -1. };
        let (Some(plus), Some(minus)) = (partner(mu, a), partner(mu, -a)) else {
            return Err(invalid(
                "a shell of the crystal's basis lacks the m partner of a function",
            ));
        };
        if value > 0 {
            out[(plus, mu)] = C64::new(sign * half, 0.);
            out[(minus, mu)] = C64::new(half, 0.);
        } else {
            out[(minus, mu)] = C64::new(0., half);
            out[(plus, mu)] = C64::new(0., -sign * half);
        }
    }
    Ok(out)
}

/// The largest imaginary part a folded block may keep (relative to its largest entry).
pub const REAL_TOLERANCE: f64 = 1e-10;

/// Where a Hamiltonian comes from ([`MolecularIntegrals::kind`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HamiltonianKind {
    /// An all-electron molecule (hydrogen-like or Gaussian orbitals, softened nuclei).
    AllElectronMolecule,
    /// A PAW-LCAO molecule (`paw::molecular`).
    PawMolecule,
    /// The hydrogen-like atomic orbital supercell of Stage 1 (`molecular::periodic`).
    Hao,
    /// The PAW-LCAO supercell of the crystal setup (this module).
    PawSupercell,
}

/// The one-body blocks of the PAW-LCAO supercell of a Gamma-centered mesh.
pub struct PawSupercell {
    pub mesh: [usize; 3],
    /// The primitive cell of every supercell cell, the first index slowest.
    pub cells: Vec<[usize; 3]>,
    /// The mesh points, fractional, in the order of `cells` (`j_i / n_i`).
    pub kpoints: Vec<[f64; 3]>,
    /// Basis functions and projectors of the primitive cell.
    pub m: usize,
    pub p: usize,
    /// `(N m) x (N m)`, the supercell function `R mu` at row `R m + mu`.
    pub overlap: DMatrix<f64>,
    /// `fixed(k)` folded, `(N m) x (N m)`.
    pub fixed: DMatrix<f64>,
    /// `(N m) x (N p)`, the projector `i` of the atoms of cell `R'` at column `R' p + i`.
    pub projections: DMatrix<C64>,
    /// The crystal's bare one-body operator `h(k)` folded, `(N m) x (N m)`.
    pub one_body: DMatrix<f64>,
    /// The crystal's per-k blocks the supercell folds, one per mesh point, in the real rows
    /// (`overlap`, `fixed`, `projections`: `U^H X U`, `U^H C`; `psi` stays in the crystal's rows).
    pub points: Vec<KPoint>,
    /// `U`, `m x m`: the real harmonics of every shell from the crystal's complex
    /// (Condon-Shortley) ones, `chi^real_mu = sum_nu chi_nu U_{nu mu}`.
    pub harmonics: DMatrix<C64>,
    /// The primitive crystal (its electrostatics, moment blocks and Bloch sums).
    pub crystal: Crystal,
    /// The ion-ion energy of the supercell, `N (eion + ion_constant)` (Hartree).
    pub ions: f64,
    /// Valence electrons of the supercell.
    pub electrons: usize,
}

impl PawSupercell {
    /// The supercell of `mesh` from the crystal `setup` (its own k-points are not used: the
    /// blocks are formed at the Gamma-centered mesh). Refused: a spinor setup (the spin-orbit
    /// datasets are not in the first version, plan 4.3), a zero mesh size.
    pub fn new(ctx: &Context, setup: &PeriodicSetup, mesh: [usize; 3]) -> Result<Self> {
        if mesh.contains(&0) {
            return Err(invalid("every mesh size must be a positive integer"));
        }
        if setup.spinor {
            return Err(crate::Error::Unsupported(
                "the PAW-LCAO supercell does not take a spin-orbit dataset yet".into(),
            ));
        }
        let (cells, kpoints) = mesh_points(&mesh)?;
        let count = cells.len();
        if setup.hubbard.is_some() {
            return Err(crate::Error::Unsupported(
                "the PAW-LCAO supercell does not take DFT+U".into(),
            ));
        }
        if setup.precision != Precision::Float64 {
            return Err(invalid(
                "the PAW-LCAO supercell is built in float64 only: use precision='float64'",
            ));
        }
        let crystal = setup.build(ctx)?;
        let mut points = setup.matrices_at(ctx, &kpoints, &vec![1. / count as f64; count])?;
        let (m, p) = match points.first() {
            Some(point) => (point.overlap.nrows(), point.projections.ncols()),
            None => return Err(invalid("the mesh has no point")),
        };
        // The crystal's basis carries complex harmonics; the supercell's functions are real.
        let description = setup.description.as_ref().ok_or_else(|| {
            invalid("the PAW-LCAO supercell needs the description of the crystal's basis")
        })?;
        let harmonics = real_harmonics(description, m)?;
        // The bare operator of the crystal's rows, with the one-center Hartree-Fock operator in
        // place of the reference atom's screening (and the frozen cores' exchange with the
        // valence), then every block in the real rows.
        let onsite = hartree_fock_onsite(&crystal)?;
        let mut bare = Vec::with_capacity(count);
        for point in &mut points {
            let h = bare_hamiltonian(&crystal, point)?
                + &point.projections * &onsite * point.projections.adjoint();
            bare.push(harmonics.adjoint() * h * &harmonics);
            point.overlap = harmonics.adjoint() * &point.overlap * &harmonics;
            point.fixed = harmonics.adjoint() * &point.fixed * &harmonics;
            point.projections = harmonics.adjoint() * &point.projections;
        }
        // The dense blocks this holds outside the context's buffers.
        let _held = ctx.reserve(product(&[count, count, m, m + m + p, 8])?)?;
        let fold = |blocks: Vec<&DMatrix<C64>>, columns: usize| -> DMatrix<C64> {
            let mut out = DMatrix::<C64>::zeros(count * m, count * columns);
            for (r, cell) in cells.iter().enumerate() {
                for (s, other) in cells.iter().enumerate() {
                    let shift: [f64; 3] = std::array::from_fn(|a| cell[a] as f64 - other[a] as f64);
                    let mut block = DMatrix::<C64>::zeros(m, columns);
                    for (index, k) in kpoints.iter().enumerate() {
                        let phase = C64::from_polar(
                            1.,
                            2. * PI * (0..3).map(|a| k[a] * shift[a]).sum::<f64>(),
                        );
                        block += blocks[index] * phase;
                    }
                    block /= C64::new(count as f64, 0.);
                    out.view_mut((r * m, s * columns), (m, columns))
                        .copy_from(&block);
                }
            }
            out
        };
        let real = |block: DMatrix<C64>, what: &str| -> Result<DMatrix<f64>> {
            let imaginary = block.iter().fold(0f64, |a, v| a.max(v.im.abs()));
            let largest = block.iter().fold(0f64, |a, v| a.max(v.norm()));
            if imaginary > REAL_TOLERANCE * largest.max(1e-300) {
                return Err(invalid(format!(
                    "the folded supercell {what} keeps an imaginary part of {imaginary:e} (largest entry {largest:e}): its per-k blocks are not time-reversal symmetric"
                )));
            }
            Ok(block.map(|v| v.re))
        };
        let overlap = real(
            fold(points.iter().map(|p| &p.overlap).collect(), m),
            "overlap",
        )?;
        let fixed = real(
            fold(points.iter().map(|p| &p.fixed).collect(), m),
            "one-body operator",
        )?;
        let projections = fold(points.iter().map(|p| &p.projections).collect(), p);
        let one_body = real(fold(bare.iter().collect(), m), "bare one-body operator")?;
        let electrons = crystal.electrons * count as f64;
        if (electrons - electrons.round()).abs() > 1e-8 {
            return Err(invalid(
                "the supercell's valence electron count is not integral",
            ));
        }
        let ions = count as f64 * (crystal.eion + crystal.ion_constant);
        Ok(Self {
            mesh,
            cells,
            kpoints,
            m,
            p,
            overlap,
            fixed,
            projections,
            one_body,
            points,
            harmonics,
            crystal,
            ions,
            electrons: electrons.round() as usize,
        })
    }

    /// The kind of Hamiltonian this supercell is the seam of.
    pub fn kind(&self) -> HamiltonianKind {
        HamiltonianKind::PawSupercell
    }

    /// The number of cells of the supercell.
    pub fn n_cells(&self) -> usize {
        self.cells.len()
    }

    /// A supercell matrix `x` (`(N m) x (N m)`, real rows, a density) at every mesh point in the
    /// crystal's rows, `U [sum_{R,R'} e^{-ik.(R-R')} x_{R, R'}] U^H`: `tr(x X) = sum_k (1/N) Re
    /// tr[x(k) X(k)]` for every operator `X` this supercell folds.
    pub fn fold_density(&self, x: &DMatrix<f64>) -> Result<Vec<DMatrix<C64>>> {
        fold_to_mesh(&self.cells, &self.kpoints, &self.harmonics, x)
    }

    /// `X(k) = sum_{R'} e^{-ik.(R - R')} X_{R mu, R' nu}` at the fractional point `k`, from the
    /// row block of cell 0 of a folded supercell matrix `block` (`(N m) x (N columns)`, real or
    /// complex): the per-k matrix it folds, exactly at a mesh point.
    pub fn unfold<T: Copy + Into<C64> + nalgebra::Scalar>(
        &self,
        block: &DMatrix<T>,
        k: [f64; 3],
    ) -> Result<DMatrix<C64>> {
        let count = self.n_cells();
        if block.nrows() != count * self.m || block.ncols() % count != 0 {
            return Err(invalid(
                "a supercell block has N m rows and N columns per cell",
            ));
        }
        let columns = block.ncols() / count;
        let origin = self.cells[0];
        let mut out = DMatrix::<C64>::zeros(self.m, columns);
        for (s, other) in self.cells.iter().enumerate() {
            let shift: f64 = (0..3)
                .map(|a| k[a] * (origin[a] as f64 - other[a] as f64))
                .sum();
            let phase = C64::from_polar(1., -2. * PI * shift);
            for i in 0..self.m {
                for j in 0..columns {
                    out[(i, j)] += block[(i, s * columns + j)].into() * phase;
                }
            }
        }
        Ok(out)
    }
}

/// The cells of the supercell of `mesh` (the first index slowest) and the Gamma-centered mesh
/// points in the same order (`j_i / n_i`, fractional).
fn mesh_points(mesh: &[usize; 3]) -> Result<(Vec<[usize; 3]>, Vec<[f64; 3]>)> {
    let count = product(mesh)?;
    let mut cells = Vec::with_capacity(count);
    let mut kpoints = Vec::with_capacity(count);
    for i in 0..mesh[0] {
        for j in 0..mesh[1] {
            for k in 0..mesh[2] {
                cells.push([i, j, k]);
                kpoints.push([
                    i as f64 / mesh[0] as f64,
                    j as f64 / mesh[1] as f64,
                    k as f64 / mesh[2] as f64,
                ]);
            }
        }
    }
    Ok((cells, kpoints))
}

/// The supercell matrix `x` (`(N m) x (N m)`, real rows) folded to every mesh point in the
/// crystal's complex-harmonic rows, `D(k) = U [sum_{R,R'} e^{-ik.(R-R')} x_{R, R'}] U^H`: the
/// density whose `sum_k (1/N) Re tr[D(k) X(k)]` is `tr(x X)` for every supercell operator `X`
/// folded from the per-k `X(k)` ([`PawSupercell::new`]).
fn fold_to_mesh(
    cells: &[[usize; 3]],
    kpoints: &[[f64; 3]],
    harmonics: &DMatrix<C64>,
    x: &DMatrix<f64>,
) -> Result<Vec<DMatrix<C64>>> {
    let m = harmonics.nrows();
    let count = cells.len();
    if x.shape() != (count * m, count * m) {
        return Err(invalid("a supercell density has N m rows and columns"));
    }
    Ok(kpoints
        .iter()
        .map(|k| {
            let mut block = DMatrix::<C64>::zeros(m, m);
            for (r, cell) in cells.iter().enumerate() {
                for (s, other) in cells.iter().enumerate() {
                    let shift: f64 = (0..3)
                        .map(|a| k[a] * (cell[a] as f64 - other[a] as f64))
                        .sum();
                    let phase = C64::from_polar(1., -2. * PI * shift);
                    for i in 0..m {
                        for j in 0..m {
                            block[(i, j)] += phase * x[(r * m + i, s * m + j)];
                        }
                    }
                }
            }
            harmonics * block * harmonics.adjoint()
        })
        .collect())
}

/// The explicit derivative (`dE/dtau`, Hartree/Bohr, per primitive atom, every image moving
/// with it) of the PAW-LCAO supercell's one-body energy `tr(P h) - tr(W S) + E_ions` at the
/// fixed raw-AO density `density` and energy-weighted density `weighted` of the supercell of
/// `mesh` on `setup` (Bloch-VQE stage 6a, [`crate::periodic::response::supercell`]): the
/// crystal's force engines at the mesh points, term by term. The two-body terms are not in it.
pub fn one_body_gradient(
    ctx: &Context,
    setup: &PeriodicSetup,
    mesh: [usize; 3],
    density: &DMatrix<f64>,
    weighted: &DMatrix<f64>,
) -> Result<crate::periodic::gradient::Breakdown> {
    let (crystal, cells, kpoints, harmonics) = mesh_crystal(ctx, setup, mesh)?;
    let onsite = hartree_fock_onsite(&crystal)?;
    crate::periodic::response::supercell::one_body_gradient(
        &crystal,
        &fold_to_mesh(&cells, &kpoints, &harmonics, density)?,
        &fold_to_mesh(&cells, &kpoints, &harmonics, weighted)?,
        &onsite,
        cells.len(),
    )
}

/// The explicit derivative (`dE/dtau`, Hartree/Bohr, per primitive atom, every image moving
/// with it) of the PAW-LCAO supercell's two-body energy `1/2 sum_k lambda_k (rho_k|rho_k)` at
/// fixed raw-AO coefficients: the Coulomb forms `(lambda_k, x_k)` (`x_k` symmetric `r x r`) over
/// the orbitals of `frame` (`(N m) x r`, raw AOs, rows `R m + mu`), as in
/// [`crate::molecular::variational_gradient::FixedDensities`]; split into the functions' and
/// the compensation shapes' motion ([`crate::periodic::wannier::downfold::basis::two_body_gradient`]).
pub fn two_body_gradient(
    ctx: &Context,
    setup: &PeriodicSetup,
    mesh: [usize; 3],
    frame: &DMatrix<f64>,
    forms: &[(f64, DMatrix<f64>)],
) -> Result<crate::periodic::wannier::downfold::basis::TwoBodyGradient> {
    let (crystal, cells, _, harmonics) = mesh_crystal(ctx, setup, mesh)?;
    check_frame(&crystal, cells.len(), frame)?;
    crate::periodic::wannier::downfold::basis::two_body_gradient(
        &crystal,
        mesh,
        &crystal.points,
        &harmonics,
        frame,
        forms,
    )
}

/// The two-body energy of each form, `1/2 lambda_k (rho_k|rho_k)` (their sum is the energy
/// whose derivative [`two_body_gradient`] is; Hartree; same arguments).
pub fn two_body_energies(
    ctx: &Context,
    setup: &PeriodicSetup,
    mesh: [usize; 3],
    frame: &DMatrix<f64>,
    forms: &[(f64, DMatrix<f64>)],
) -> Result<Vec<f64>> {
    let (crystal, cells, _, harmonics) = mesh_crystal(ctx, setup, mesh)?;
    check_frame(&crystal, cells.len(), frame)?;
    crate::periodic::wannier::downfold::basis::two_body_energies(
        &crystal,
        mesh,
        &crystal.points,
        &harmonics,
        frame,
        forms,
    )
}

/// `d<phi_p|S|phi_q>/dtau_axis` of the functions of `frame` for the motion of the primitive
/// atom `atom` (three `r x r` matrices, from the derivative functions' pair densities), the same
/// from their `G = 0` charge, and the gap of the sampled Bloch sums: the gate of the derivative
/// functions against the overlap's own derivative.
#[allow(clippy::type_complexity)]
pub fn metric_gradient(
    ctx: &Context,
    setup: &PeriodicSetup,
    mesh: [usize; 3],
    frame: &DMatrix<f64>,
    atom: usize,
) -> Result<([DMatrix<C64>; 3], [DMatrix<C64>; 3], f64)> {
    let (crystal, cells, _, harmonics) = mesh_crystal(ctx, setup, mesh)?;
    check_frame(&crystal, cells.len(), frame)?;
    crate::periodic::wannier::downfold::basis::metric_gradient(
        &crystal,
        mesh,
        &crystal.points,
        &harmonics,
        frame,
        atom,
    )
}

/// The explicit derivative of the PAW-LCAO supercell Hamiltonian's energy of fixed raw-AO
/// densities (`d/dR [tr(P h) - tr(W S) + E2 + E_ions]`, the seam of
/// [`MolecularIntegrals::fixed_density_gradient`] for [`HamiltonianKind::PawSupercell`]):
/// [`one_body_gradient`] and [`two_body_gradient`] on the integrals' own setup and mesh. Both are
/// derivatives under a collective displacement (every image of a primitive atom moving with it:
/// the only motion the crystal's engines at the mesh have), so each image of a primitive atom
/// carries `1/N` of its atom's derivative (the image average; the derivative of the energy per
/// cell is the image's value). Hartree per Bohr, the supercell's atoms (cell-major) in rows of
/// three. A model's cluster-frame response (`cluster_frame`) is refused by name: the supercell's
/// exactly degenerate clusters are refused before (`require_unsplit`).
pub(in crate::molecular) fn fixed_density_gradient(
    integrals: &MolecularIntegrals,
    densities: &crate::molecular::variational_gradient::FixedDensities,
) -> Result<crate::molecular::forces::Gradient> {
    let source = integrals.paw_supercell.as_ref().ok_or_else(|| {
        invalid("the supercell's fixed-density gradient needs the PAW-LCAO supercell source")
    })?;
    if densities.cluster_frame.is_some() {
        return Err(crate::Error::Unsupported(
            "the analytic force of a PAW-LCAO supercell has no frame response of a cut              degenerate cluster: choose an active space that closes the cluster"
                .into(),
        ));
    }
    let m = integrals.m();
    let r = densities.frame.ncols();
    if densities.one.shape() != (m, m)
        || densities.weighted.shape() != (m, m)
        || densities.frame.nrows() != m
        || densities.two.iter().any(|(_, x)| x.shape() != (r, r))
    {
        return Err(invalid("fixed densities must match the basis"));
    }
    let ctx = &integrals.context;
    let (setup, mesh) = (source.setup.as_ref(), source.mesh);
    let one = one_body_gradient(ctx, setup, mesh, &densities.one, &densities.weighted)?;
    let two = if densities.two.is_empty() {
        None
    } else {
        Some(two_body_gradient(
            ctx,
            setup,
            mesh,
            &densities.frame,
            &densities.two,
        )?)
    };
    let atoms = setup.centers.len();
    let cells = product(&mesh)?;
    if integrals.geometry.numbers().len() != cells * atoms {
        return Err(invalid(
            "the supercell's atoms are not the mesh's copies of the setup's",
        ));
    }
    let images = |per_atom: &[[f64; 3]]| -> Result<crate::Buffer<f64>> {
        let mut out = ctx.allocate::<f64>(product(&[cells, atoms, 3])?)?;
        for cell in 0..cells {
            for (a, g) in per_atom.iter().enumerate() {
                for x in 0..3 {
                    out.as_mut_slice()[(cell * atoms + a) * 3 + x] = g[x] / cells as f64;
                }
            }
        }
        Ok(out)
    };
    let mut total = one.gradient();
    let mut terms = Vec::new();
    for (name, values) in one.terms() {
        if values.iter().flatten().any(|v| *v != 0.) {
            terms.push((name, images(values)?));
        }
    }
    if let Some(two) = &two {
        for (t, g) in total.iter_mut().zip(two.total()) {
            for x in 0..3 {
                t[x] += g[x];
            }
        }
        terms.push(("two_body/functions", images(&two.functions)?));
        terms.push(("two_body/shapes", images(&two.shapes)?));
    }
    Ok(crate::molecular::forces::Gradient {
        hellmann_feynman: None,
        pulay: None,
        total: images(&total)?,
        scheme: "analytic",
        stationarity: None,
        terms,
    })
}

/// The derivative of the energy per cell, per primitive atom (`3 A`, Hartree per Bohr), from a
/// gradient with one row of three per supercell atom (cell-major): the image average, the
/// derivative of the supercell's energy under a collective displacement divided by `N`.
pub(in crate::molecular) fn per_cell(
    integrals: &MolecularIntegrals,
    gradient: &[f64],
) -> Result<crate::Buffer<f64>> {
    let source = integrals
        .paw_supercell
        .as_ref()
        .ok_or_else(|| invalid("the per-cell gradient needs the PAW-LCAO supercell source"))?;
    let atoms = source.setup.centers.len();
    let cells = product(&source.mesh)?;
    if gradient.len() != 3 * atoms * cells {
        return Err(invalid(
            "a supercell gradient has three rows per supercell atom",
        ));
    }
    let mut out = integrals.context.allocate::<f64>(3 * atoms)?;
    for cell in 0..cells {
        for (slot, value) in out
            .as_mut_slice()
            .iter_mut()
            .zip(&gradient[3 * atoms * cell..3 * atoms * (cell + 1)])
        {
            *slot += value / cells as f64;
        }
    }
    Ok(out)
}

/// Refuses, by name, a reference whose Fermi level is degenerate on the supercell's mesh (a
/// metal on this mesh: the lowest empty level of a channel within
/// [`crate::molecular::response::DEGENERACY_TOLERANCE`] of its highest occupied one), whose
/// orbital response has vanishing denominators (Bloch-VQE stage 6a, G7).
pub(in crate::molecular) fn require_gap(state: &crate::molecular::MeanField) -> Result<()> {
    for (s, levels) in state.energies.iter().enumerate() {
        let occupied = state.particles.get(s).copied().unwrap_or(0);
        if occupied == 0 || occupied >= levels.len() {
            continue;
        }
        let gap = levels[occupied] - levels[occupied - 1];
        if gap < crate::molecular::response::DEGENERACY_TOLERANCE {
            return Err(crate::Error::Unsupported(format!(
                "the Fermi level of the PAW-LCAO supercell's reference is degenerate (gap {gap:e} \
                 Ha in channel {s}): a metal on this mesh has no orbital response, so its analytic \
                 force is not defined; use a mesh or a spin state that opens the gap"
            )));
        }
    }
    Ok(())
}

/// The analytic gradient of a Hartree-Fock determinant (RHF or UHF) of the PAW-LCAO supercell:
/// its fixed densities (`P`, `W = sum_i n_i eps_i c_i c_i^T`, the occupied raw-AO orbitals with
/// the Coulomb form of the density and the exchange forms of each spin) through
/// [`fixed_density_gradient`], folded to the derivative of the energy per cell per primitive
/// atom ([`per_cell`]). A determinant converged on its own surface is stationary, so the overlap
/// term is its whole orbital response.
pub(in crate::molecular) fn mean_field_gradient(
    integrals: &MolecularIntegrals,
    state: &crate::molecular::MeanField,
) -> Result<crate::molecular::forces::Gradient> {
    state.require_determinant("the analytic force of a PAW-LCAO supercell")?;
    require_gap(state)?;
    let doubly = match state.method.as_str() {
        "rhf" if state.functional.is_none() => true,
        "uhf" if state.functional.is_none() => false,
        _ => {
            return Err(crate::Error::Unsupported(format!(
            "the analytic force of a PAW-LCAO supercell takes an RHF or UHF determinant, not {}{}",
            state.method,
            state
                .functional
                .as_ref()
                .map_or(String::new(), |f| format!(" ({f})"))
        )))
        }
    };
    let m = integrals.m();
    let channels = if doubly { 1 } else { 2 };
    if state.coefficients.len() != channels || state.energies.len() != channels {
        return Err(invalid("SCF state dimensions do not match the supercell"));
    }
    let occupation = if doubly { 2. } else { 1. };
    let mut columns: Vec<nalgebra::DVector<f64>> = Vec::new();
    let mut spins = Vec::new();
    let mut weighted = DMatrix::<f64>::zeros(m, m);
    for s in 0..channels {
        let c = &state.coefficients[s];
        if c.iter().any(|v| v.im.abs() > 1e-12) || c.shape() != (m, m) {
            return Err(invalid(
                "the supercell's analytic force takes real orbitals of the supercell's basis",
            ));
        }
        let raw = &integrals.lowdin * c.map(|v| v.re);
        for i in 0..state.particles[s] {
            let column = raw.column(i).into_owned();
            weighted += &column * column.transpose() * (occupation * state.energies[s][i]);
            columns.push(column);
            spins.push(s);
        }
    }
    let r = columns.len();
    let frame = DMatrix::from_columns(&columns);
    let one = &frame * &frame.transpose() * occupation;
    // J: the density; K: each spin's pairs, `-(ij|ij)` (RHF: `-2 (ij|ij)` over the doubly
    // occupied orbitals, the exchange of both spins).
    let mut two = vec![(1., DMatrix::<f64>::identity(r, r) * occupation)];
    for i in 0..r {
        for j in 0..=i {
            if spins[i] != spins[j] {
                continue;
            }
            let mut x = DMatrix::<f64>::zeros(r, r);
            if i == j {
                x[(i, i)] = 1.;
            } else {
                x[(i, j)] = std::f64::consts::FRAC_1_SQRT_2;
                x[(j, i)] = std::f64::consts::FRAC_1_SQRT_2;
            }
            two.push((-occupation, x));
        }
    }
    let densities = crate::molecular::variational_gradient::FixedDensities {
        one,
        weighted,
        frame,
        two,
        cluster_frame: None,
    };
    let images = fixed_density_gradient(integrals, &densities)?;
    let mut terms = Vec::with_capacity(images.terms.len());
    for (name, values) in &images.terms {
        terms.push((*name, per_cell(integrals, values.as_slice())?));
    }
    Ok(crate::molecular::forces::Gradient {
        hellmann_feynman: None,
        pulay: None,
        total: per_cell(integrals, images.total.as_slice())?,
        scheme: "analytic",
        stationarity: None,
        terms,
    })
}

fn check_frame(crystal: &Crystal, cells: usize, frame: &DMatrix<f64>) -> Result<()> {
    if frame.nrows() != cells * crystal.m || frame.ncols() == 0 {
        return Err(invalid(
            "a supercell frame has N m rows (the raw AOs) and at least one orbital",
        ));
    }
    Ok(())
}

/// The crystal of the supercell's Gamma-centered mesh (one Bloch point per mesh point, in the
/// mesh's order, no symmetry), the cells and points of the mesh, and the real harmonics `U`.
#[allow(clippy::type_complexity)]
fn mesh_crystal(
    ctx: &Context,
    setup: &PeriodicSetup,
    mesh: [usize; 3],
) -> Result<(Crystal, Vec<[usize; 3]>, Vec<[f64; 3]>, DMatrix<C64>)> {
    if mesh.contains(&0) {
        return Err(invalid("every mesh size must be a positive integer"));
    }
    if setup.precision != Precision::Float64 {
        return Err(invalid(
            "the PAW-LCAO supercell is built in float64 only: use precision='float64'",
        ));
    }
    let (cells, kpoints) = mesh_points(&mesh)?;
    let count = cells.len();
    let crystal = setup
        .with_kpoints(ctx, &kpoints, &vec![1. / count as f64; count])?
        .build(ctx)?;
    let placed = crystal.points.len() == count
        && crystal
            .points
            .iter()
            .zip(&kpoints)
            .all(|(point, k)| (0..3).all(|a| (point.fractional[a] - k[a]).abs() < 1e-12));
    if !placed {
        return Err(invalid(
            "the crystal of the supercell's mesh does not keep the mesh points in order",
        ));
    }
    let description = crystal.description.as_ref().ok_or_else(|| {
        invalid("the PAW-LCAO supercell needs the description of the crystal's basis")
    })?;
    let harmonics = real_harmonics(description, crystal.m)?;
    Ok((crystal, cells, kpoints, harmonics))
}

/// The pivoted Cholesky tolerance (Hartree, the largest unused pair integral) of a PAW-LCAO
/// supercell whose caller names none: the calculator's `two_body` default, 1e-8 eV, so the
/// engine and the calculator have one default. The pair matrix of the Lowdin functions has no
/// low-rank structure (Si DZP 2 x 1 x 1 has 1293 of its 1378 eigenvalues above 1e-10, and the
/// pivoted rank is within 5 % of that count), so the rank follows the tolerance; the factors are
/// stored per translation orbit, which keeps Si DZP 2 x 2 x 2 (`m = 208`, rank 8896) at a
/// reserved peak of 1.24 GB (`tests/paw_supercell_memory.rs`). A looser tolerance moves the RHF
/// energy per cell by 1.5e-7 Hartree at 1e-8 Hartree (Si 2 x 2 x 2 SZ,
/// `tests/paw_supercell_tolerance.rs`).
pub const SUPERCELL_CHOLESKY_TOLERANCE: f64 = 1e-8 / crate::units::HARTREE_TO_EV;

impl MolecularIntegrals {
    /// The PAW-LCAO supercell Hamiltonian of the Gamma-centered `mesh` of the crystal `setup`
    /// (`BLOCH_VQE_PLAN.md` Stage 4b): the bare one-body operator and the overlap folded from the
    /// crystal's per-k blocks, the PAW two-body integrals of the Lowdin functions with the
    /// periodic kernel factored by the pivoted Cholesky (`cholesky`: its tolerance, by default
    /// [`SUPERCELL_CHOLESKY_TOLERANCE`]), and the
    /// ion-ion and Madelung constants. The supercell's atoms are cell-major, the first mesh axis
    /// slowest ([`Supercell`]); its functions are the crystal's filtered shells summed over the
    /// supercell's images. Refused by name: a spin-orbit dataset, DFT+U, float32.
    pub fn paw_supercell(
        ctx: &Context,
        setup: Arc<PeriodicSetup>,
        mesh: [usize; 3],
        cholesky: Option<f64>,
    ) -> Result<Self> {
        ctx.run(|| build(ctx, setup, mesh, cholesky))
    }
}

fn build(
    ctx: &Context,
    setup: Arc<PeriodicSetup>,
    mesh: [usize; 3],
    cholesky: Option<f64>,
) -> Result<MolecularIntegrals> {
    let description = setup.description.as_ref().ok_or_else(|| {
        invalid("the PAW-LCAO supercell needs the description of the crystal's basis")
    })?;
    let cells = PawSupercell::new(ctx, &setup, mesh)?;
    let size = cells.n_cells() * cells.m;
    let (values, vectors) = real_eigen(cells.overlap.clone())?;
    let overlap_min = values[0];
    if overlap_min <= 1e-10 {
        return Err(invalid(format!(
            "the supercell basis is linearly dependent (overlap eigenvalue {overlap_min:e})"
        )));
    }
    let lowdin = &vectors
        * DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            size,
            values.iter().map(|s| 1. / s.sqrt()),
        ))
        * vectors.transpose();
    let samples = |frame: &DMatrix<f64>| -> Result<RealSamples> {
        let built = basis_samples(&cells.crystal, mesh, &cells.points, &cells.harmonics, frame)?;
        if built.imaginary > REAL_TOLERANCE {
            return Err(invalid(format!(
                "the supercell functions keep an imaginary part of {:e} of their largest value",
                built.imaginary
            )));
        }
        let mut out = RealSamples::zeros(ctx, Precision::Float64, built.values.len())?;
        out.fill_range(0, &built.values)?;
        Ok(out)
    };
    let raw = samples(&DMatrix::identity(size, size))?;
    let orthonormal = samples(&lowdin)?;
    let columns = BasisCoulomb::new(
        &cells.crystal,
        mesh,
        &cells.points,
        &cells.harmonics,
        &lowdin,
    )?;
    let metric_error = (0..size)
        .flat_map(|a| (0..size).map(move |b| (a, b)))
        .map(|(a, b)| (columns.metric[(a, b)] - C64::new(f64::from(u8::from(a == b)), 0.)).norm())
        .fold(0f64, f64::max);
    let crystal_grid = &cells.crystal.grid;
    let shape = crystal_grid.index.shape();
    let grid = SpatialGrid::new(
        std::array::from_fn(|a| shape[a] * mesh[a]),
        crystal_grid.origin,
        crystal_grid.step_values(),
        true,
    )?;
    let numbers = setup
        .atoms_species
        .iter()
        .map(|&s| {
            u8::try_from(setup.datasets[s].atomic_number).map_err(|_| invalid("atomic number"))
        })
        .collect::<Result<Vec<_>>>()?;
    let lattice: [[f64; 3]; 3] =
        std::array::from_fn(|a| std::array::from_fn(|x| setup.lattice[(x, a)]));
    let primitive = Geometry::new(
        numbers,
        setup.centers.clone(),
        lattice,
        [true; 3],
        LengthUnit::Bohr,
    )?;
    let (supercell, geometry) = Supercell::new(&primitive, mesh)?;
    let atoms = primitive.numbers().len();
    let mut tables: Vec<Option<Arc<Cubic>>> = vec![None; description.tables.len()];
    let mut orbitals = Vec::with_capacity(size);
    for cell in 0..cells.n_cells() {
        for mu in 0..cells.m {
            let t = description.function_table[mu];
            let table = &description.tables[t];
            let radial = match &tables[t] {
                Some(radial) => radial.clone(),
                None => {
                    let mut knots = ctx.allocate::<f64>(table.r.len())?;
                    knots.as_mut_slice().copy_from_slice(&table.r);
                    let mut values = ctx.allocate::<f64>(table.values.len())?;
                    values.as_mut_slice().copy_from_slice(&table.values);
                    let radial = Arc::new(Cubic::new(ctx, knots, values)?);
                    tables[t] = Some(radial.clone());
                    radial
                }
            };
            let atom = cell * atoms + description.function_atom[mu];
            orbitals.push(Orbital {
                atom,
                l: table.l,
                m: description.function_m[mu],
                center: geometry.positions()[atom],
                radial: Radial::Tabulated(radial),
            });
        }
    }
    let basis = Basis::from_orbitals(ctx, orbitals)?;
    let translations = Arc::new(orbital_translations(&supercell, &basis)?);
    let madelung = 0.5 * cells.electrons as f64 * madelung_constant(ctx, geometry.cell())?;
    let field = &cells.one_body - &cells.fixed;
    let source = super::super::integrals::PawSupercellSource {
        setup: setup.clone(),
        mesh,
        electrons: cells.electrons,
        correction: Arc::new(columns.onsite()?),
        columns: Box::new(columns),
        metric_error,
        translations,
    };
    MolecularIntegrals::from_paw_supercell(
        ctx,
        geometry,
        grid,
        basis,
        (raw, orthonormal),
        (&cells.overlap, &cells.fixed, &field),
        (lowdin, overlap_min),
        (cells.ions, madelung),
        Some(cholesky.unwrap_or(SUPERCELL_CHOLESKY_TOLERANCE)),
        Arc::new(supercell),
        source,
    )
}

/// The supercell's lattice translations as permutations of its functions: translation `t` (the
/// cell index of its lattice vector) carries the function of primitive orbital `o` in cell `c`
/// to that of `o` in cell `c + t`, modulo the mesh. The Lowdin frame is a function of the
/// overlap, which they leave invariant, so they carry its functions the same way.
fn orbital_translations(supercell: &Supercell, basis: &Basis) -> Result<Translations> {
    let cells = supercell.orbital_cells(basis)?;
    let (mesh, n_cells, width) = (supercell.mesh, supercell.n_cells(), cells.n_primitive);
    let of = |flat: usize| {
        [
            flat / (mesh[1] * mesh[2]),
            (flat / mesh[2]) % mesh[1],
            flat % mesh[2],
        ]
    };
    let mut index = vec![usize::MAX; n_cells * width];
    for (p, (&c, &o)) in cells.cell.iter().zip(&cells.orbital).enumerate() {
        index[c * width + o] = p;
    }
    if index.contains(&usize::MAX) {
        return Err(invalid(
            "the supercell functions are not one copy per cell of the primitive functions",
        ));
    }
    Translations::new(
        (0..n_cells)
            .map(|t| {
                let shift = of(t);
                cells
                    .cell
                    .iter()
                    .zip(&cells.orbital)
                    .map(|(&c, &o)| {
                        let at = of(c);
                        let moved = supercell
                            .cell_index(std::array::from_fn(|a| (at[a] + shift[a]) % mesh[a]));
                        index[moved * width + o]
                    })
                    .collect()
            })
            .collect(),
    )
}

/// The PAW-LCAO supercell Hamiltonian of `integrals` at the supercell `geometry`. The supercell
/// is the crystal's, so every image of a primitive atom moves together: a geometry that moves
/// one image apart from the others is refused by name.
pub(in crate::molecular) fn continued(
    integrals: &MolecularIntegrals,
    source: &super::super::integrals::PawSupercellSource,
    geometry: Geometry,
) -> Result<MolecularIntegrals> {
    let supercell = integrals
        .supercell
        .as_ref()
        .ok_or_else(|| invalid("a PAW supercell Hamiltonian without its supercell"))?;
    let setup = &source.setup;
    let atoms = setup.centers.len();
    let positions = geometry.positions();
    if positions.len() != supercell.cells.len() || supercell.atoms.len() != positions.len() {
        return Err(invalid("the geometry is not this supercell's"));
    }
    for (index, cell) in supercell.cells.iter().enumerate() {
        let home = positions[supercell.atoms[index]];
        let shift = setup.lattice * nalgebra::Vector3::from(cell.map(|c| c as f64));
        let gap = (0..3)
            .map(|a| (positions[index][a] - home[a] - shift[a]).abs())
            .fold(0f64, f64::max);
        if gap > 1e-9 {
            return Err(crate::Error::Unsupported(format!(
                "a PAW-LCAO supercell moves every image of a primitive atom together (it is the \
                 crystal's {:?} mesh); atom {index} is displaced {gap:e} Bohr apart from its images",
                source.mesh
            )));
        }
    }
    let inverse = setup
        .lattice
        .try_inverse()
        .ok_or_else(|| invalid("singular crystal lattice"))?;
    let fractional: Vec<[f64; 3]> = (0..atoms)
        .map(|a| (inverse * nalgebra::Vector3::from(positions[a])).into())
        .collect();
    let moved = setup.with_positions(&integrals.context, &fractional)?;
    build(
        &integrals.context,
        Arc::new(moved),
        source.mesh,
        Some(integrals.cholesky_tolerance()),
    )
}
