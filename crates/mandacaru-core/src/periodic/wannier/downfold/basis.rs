//! The two-body integrals of the PAW-LCAO supercell Hamiltonian (`molecular::periodic::paw`,
//! `BLOCH_VQE_PLAN.md` Stage 4b): the pair-density engine of the downfolding ([`super::pairs`])
//! run on the supercell functions of the crystal's basis, with the periodic kernel.
//!
//! The function `a` of the supercell frame `X` (rows `R m + mu`, the primitive function `mu` of
//! the cell `R`) is `phi_a = sum_{R mu} X_{R mu, a} chi_{R mu}`, the image sums
//! `chi_{R mu} = (1/N) sum_k e^{-ik.R} chi_{mu k}` of the crystal's Bloch sums. On the supercell
//! grid it is the map of [`super::functions::bloch_functions`] with the gauge
//! `U_{mu a}(k) = sum_R X_{R mu, a} e^{-ik.R}`. Its pair densities carry the smooth product, the
//! compensation charges of every atom copy and the one-center remainder, and
//! `(pq|rs) = int int rho_pq v rho_rs` uses `4 pi / G^2` with `G = 0` dropped
//! ([`Kernel::Periodic`]). Atomic units.
use super::{
    coulomb::{Cell, Kernel},
    functions::{bloch_functions, Functions, Source},
    pairs::{coulomb_rows, Mode, Pairs},
    streamed::{Grid, Streamed},
};
use crate::molecular::integrals::PairColumns;
use crate::{
    periodic::{invalid, response, Crystal, KPoint},
    product,
    quantum::C64,
    Result,
};
use nalgebra::DMatrix;
use std::f64::consts::PI;

/// The smooth parts of the supercell functions of a frame on the supercell grid (C order of
/// [`Cell::total`]), `values[a * points + g]`, real; with the largest imaginary part the image
/// sums left (relative to the largest value).
pub struct BasisSamples {
    pub values: Vec<f64>,
    pub points: usize,
    pub imaginary: f64,
}

/// The gauge `U_{mu a}(k) = sum_R X_{R mu, a} e^{-2 pi i k.R}` of the frame `X` at each mesh point
/// of `cell` (`points[t]` at the mesh coordinates of the cell `t`).
fn gauges(cell: &Cell, points: &[KPoint], frame: &DMatrix<f64>) -> Result<Vec<DMatrix<C64>>> {
    let cells = cell.cells;
    if points.len() != cells || frame.nrows() % cells != 0 {
        return Err(invalid(
            "a supercell frame needs one Bloch point per cell and N m rows",
        ));
    }
    let m = frame.nrows() / cells;
    if points.iter().any(|p| p.overlap.nrows() != m) {
        return Err(invalid("the frame's rows are not the crystal's basis"));
    }
    Ok(points
        .iter()
        .map(|point| {
            let k = point.fractional;
            DMatrix::from_fn(m, frame.ncols(), |mu, a| {
                (0..cells)
                    .map(|r| {
                        let v = cell.cell_vector(r);
                        let angle = -2. * PI * (0..3).map(|x| k[x] * v[x] as f64).sum::<f64>();
                        C64::from_polar(frame[(r * m + mu, a)], angle)
                    })
                    .sum()
            })
        })
        .collect())
}

/// The functions of `frame`, whose rows are the primitive functions `chi U` (`harmonics`, `U`:
/// the crystal's rows to the supercell's), from the Bloch sums `psi` of the crystal's rows; the
/// projections of `points` are in the supercell's rows (`U^H C`).
fn functions(
    crystal: &Crystal,
    cell: &Cell,
    points: &[KPoint],
    harmonics: &DMatrix<C64>,
    frame: &DMatrix<f64>,
) -> Result<super::functions::Functions> {
    let gauges: Vec<DMatrix<C64>> = gauges(cell, points, frame)?
        .into_iter()
        .map(|g| harmonics * g)
        .collect();
    let projections: Vec<DMatrix<C64>> =
        points.iter().map(|p| harmonics * &p.projections).collect();
    let sources: Vec<Source<'_>> = points
        .iter()
        .zip(&gauges)
        .zip(&projections)
        .map(|((point, gauge), projections)| Source {
            fractional: point.fractional,
            samples: &point.psi,
            projections,
            gauge,
            components: 1,
        })
        .collect();
    let ids: Vec<usize> = (0..frame.ncols()).collect();
    bloch_functions(crystal, cell, &sources, &ids)
}

/// The supercell functions of `frame` from Bloch points in the crystal's own rows (`points` of a
/// crystal built at the mesh: their projections are `<chi_{mu k}|p_j>` of the complex harmonics),
/// with the gauges `U harmonics_frame(k)` of [`gauges`].
fn crystal_row_functions(
    crystal: &Crystal,
    cell: &Cell,
    points: &[KPoint],
    gauges: &[DMatrix<C64>],
) -> Result<Functions> {
    let sources: Vec<Source<'_>> = points
        .iter()
        .zip(gauges)
        .map(|(point, gauge)| Source {
            fractional: point.fractional,
            samples: &point.psi,
            projections: &point.projections,
            gauge,
            components: 1,
        })
        .collect();
    let ids: Vec<usize> = (0..gauges.first().map_or(0, |g| g.ncols())).collect();
    bloch_functions(crystal, cell, &sources, &ids)
}

/// The center derivative `d phi_a / d tau_axis` of the supercell functions of a frame when the
/// primitive atom `atom` and all its images move (Bloch-VQE stage 6a, two-body part): on the
/// grid the Bloch sums of the functions `atom` owns, `-d chi_{mu k} / d x_axis` (the grid fixed),
/// and on the projectors `dC = P_X G_C - G_C Q_X` (the functions of `atom` moving and its
/// projectors moving, [`crate::periodic::setup::pulay::projection_gradient`]), mapped by
/// [`bloch_functions`] with the frame's gauges: the pair densities of these functions with the
/// frame's carry the motion of the smooth products, of the compensation charges and of the
/// one-center remainder, all bilinear in the coefficients. Also returns the largest gap between
/// the sampled Bloch sums and the crystal's own `psi` (relative to the largest value): the two
/// must be the same functions for the derivative to be theirs.
fn moving_functions(
    crystal: &Crystal,
    setup: &response::setup::Setup,
    cell: &Cell,
    points: &[KPoint],
    gauges: &[DMatrix<C64>],
    atom: usize,
    axis: usize,
) -> Result<(Functions, f64)> {
    use crate::math::storage::ComplexSamples;
    use crate::periodic::setup::pulay::{block_samples, projection_gradient, Mesh};
    let ctx = &crystal.context;
    let n = crystal.grid.index.points();
    let m = crystal.m;
    let owners = setup.function_atoms();
    let projector_atoms = setup.projector_atoms();
    if owners.len() != m || axis > 2 {
        return Err(invalid(
            "the moving functions need the crystal's basis and an axis",
        ));
    }
    let mesh = Mesh::new(ctx, &crystal.grid)?;
    let (mut gap, mut largest) = (0f64, 0f64);
    let mut samples = Vec::with_capacity(points.len());
    let mut slopes = Vec::with_capacity(points.len());
    for point in points {
        let (values, derivatives) = block_samples(
            ctx,
            setup.shells(),
            setup.lattice(),
            &mesh,
            point.k,
            &(0..n),
            true,
        )?;
        let psi = point.psi.read_range(ctx, 0, m * n)?;
        for (a, b) in values.as_slice().iter().zip(psi.iter()) {
            gap = gap.max((a - b).norm());
            largest = largest.max(b.norm());
        }
        let mut moving = ctx.allocate::<C64>(product(&[m, n])?)?;
        let slope = derivatives[axis].as_slice();
        for mu in (0..m).filter(|&mu| owners[mu] == atom) {
            for (to, from) in moving.as_mut_slice()[mu * n..][..n]
                .iter_mut()
                .zip(&slope[mu * n..][..n])
            {
                *to = -from;
            }
        }
        samples.push(ComplexSamples::Float64(moving));
        let g = &projection_gradient(
            ctx,
            setup.shells(),
            setup.lattice(),
            point.k,
            setup.projectors(),
        )?[axis];
        slopes.push(DMatrix::from_fn(m, crystal.p, |mu, j| {
            let rows = if owners[mu] == atom {
                g[(mu, j)]
            } else {
                C64::default()
            };
            let columns = if projector_atoms[j] == atom {
                g[(mu, j)]
            } else {
                C64::default()
            };
            rows - columns
        }));
    }
    let sources: Vec<Source<'_>> = points
        .iter()
        .zip(gauges)
        .zip(samples.iter().zip(&slopes))
        .map(|((point, gauge), (samples, projections))| Source {
            fractional: point.fractional,
            samples,
            projections,
            gauge,
            components: 1,
        })
        .collect();
    let ids: Vec<usize> = (0..gauges.first().map_or(0, |g| g.ncols())).collect();
    Ok((
        bloch_functions(crystal, cell, &sources, &ids)?,
        gap / largest.max(1e-300),
    ))
}

/// The functions `[phi_0 .. phi_{r-1}, d phi_0 .. d phi_{r-1}]` as one set.
fn joined(ctx: &crate::Context, base: &Functions, moving: Functions) -> Result<Functions> {
    let r = base.ids.len();
    let mut values = Vec::with_capacity(2 * r);
    let mut coefficients = Vec::with_capacity(2 * r);
    for (v, c) in base.values.iter().zip(&base.coefficients) {
        values.push(ctx.copy_slice(v.as_slice())?);
        coefficients.push(ctx.copy_slice(c.as_slice())?);
    }
    values.extend(moving.values);
    coefficients.extend(moving.coefficients);
    Ok(Functions {
        ids: (0..2 * r).collect(),
        values,
        coefficients,
        projectors: base.projectors,
        components: base.components,
    })
}

/// What the two-body explicit derivative of the PAW-LCAO supercell needs of the crystal of its
/// mesh: the crystal (one Bloch point per mesh point, in the crystal's rows), its engines' view,
/// the supercell grid and the gauges of the frame.
struct Moving<'a> {
    crystal: &'a Crystal,
    points: &'a [KPoint],
    setup: response::setup::Setup,
    cell: Cell,
    gauges: Vec<DMatrix<C64>>,
    base: Functions,
    grid: std::sync::Arc<Grid>,
}

impl<'a> Moving<'a> {
    fn new(
        crystal: &'a Crystal,
        size: [usize; 3],
        points: &'a [KPoint],
        harmonics: &DMatrix<C64>,
        frame: &DMatrix<f64>,
    ) -> Result<Self> {
        if crystal.spinor {
            return Err(crate::Error::Unsupported(
                "the PAW-LCAO supercell does not take a spin-orbit dataset yet".into(),
            ));
        }
        let setup = response::setup::Setup::new(crystal)?;
        let cell = Cell::of(crystal, size)?;
        let gauges: Vec<DMatrix<C64>> = gauges(&cell, points, frame)?
            .into_iter()
            .map(|g| harmonics * g)
            .collect();
        let base = crystal_row_functions(crystal, &cell, points, &gauges)?;
        let grid = Grid::new(crystal, &setup, &cell, Kernel::Periodic)?;
        Ok(Self {
            crystal,
            points,
            setup,
            cell,
            gauges,
            base,
            grid,
        })
    }

    /// The functions `[phi, d phi]` for the motion of `atom` along `axis` on the supercell's
    /// grid, with the gap of the sampled Bloch sums.
    fn streamed(&self, atom: usize, axis: usize) -> Result<(Streamed, f64)> {
        let (moving, gap) = moving_functions(
            self.crystal,
            &self.setup,
            &self.cell,
            self.points,
            &self.gauges,
            atom,
            axis,
        )?;
        let all = joined(&self.crystal.context, &self.base, moving)?;
        Ok((
            Streamed::new(self.crystal, &self.setup, self.grid.clone(), all)?,
            gap,
        ))
    }

    /// The frame's own functions on the supercell's grid.
    fn base_streamed(&self) -> Result<Streamed> {
        let ctx = &self.crystal.context;
        let copy = Functions {
            ids: self.base.ids.clone(),
            values: self
                .base
                .values
                .iter()
                .map(|v| ctx.copy_slice(v.as_slice()))
                .collect::<Result<_>>()?,
            coefficients: self
                .base
                .coefficients
                .iter()
                .map(|c| ctx.copy_slice(c.as_slice()))
                .collect::<Result<_>>()?,
            projectors: self.base.projectors,
            components: self.base.components,
        };
        Streamed::new(self.crystal, &self.setup, self.grid.clone(), copy)
    }
}

/// The supercell's two-body density as Coulomb forms over the orbitals of a frame (the
/// convention of `molecular::variational_gradient::FixedDensities`): `E2 = 1/2 sum_k lambda_k
/// (rho_k|rho_k)`, `rho_k = sum_pq x_k[p, q] phi_p phi_q`, `x_k` symmetric `r x r`.
fn form_weights(forms: &[(f64, DMatrix<f64>)], r: usize, n: usize) -> Result<Vec<Vec<C64>>> {
    forms
        .iter()
        .map(|(_, x)| {
            if x.shape() != (r, r) {
                return Err(invalid("a Coulomb form is r x r over the frame's orbitals"));
            }
            let mut w = vec![C64::default(); n * n];
            for p in 0..r {
                for q in 0..r {
                    w[p * n + q] = C64::new(x[(p, q)], 0.);
                }
            }
            Ok(w)
        })
        .collect()
}

/// `1/2 lambda_k (rho_k|rho_k)` of each of the forms `forms` (their sum is `E2`) over the
/// functions of the real frame `frame` (`(N m) x r`, rows `R m + mu` of the real-harmonic
/// primitive functions) on the supercell of `size`, from `points`, the Bloch points of the
/// crystal of the mesh in the crystal's own rows: the pair engine of the supercell's integrals,
/// the periodic kernel.
pub fn two_body_energies(
    crystal: &Crystal,
    size: [usize; 3],
    points: &[KPoint],
    harmonics: &DMatrix<C64>,
    frame: &DMatrix<f64>,
    forms: &[(f64, DMatrix<f64>)],
) -> Result<Vec<f64>> {
    let moving = Moving::new(crystal, size, points, harmonics, frame)?;
    let r = frame.ncols();
    let streamed = moving.base_streamed()?;
    let weights = form_weights(forms, r, r)?;
    let rows: Vec<usize> = (0..r * r).collect();
    let values = streamed.contracted(&moving.setup, &rows, &weights)?;
    let f = forms.len();
    Ok(forms
        .iter()
        .enumerate()
        .map(|(k, (lambda, x))| {
            let mut energy = C64::default();
            for p in 0..r {
                for q in 0..r {
                    energy += 0.5 * lambda * x[(p, q)] * values[(p * r + q) * f + k];
                }
            }
            energy.re
        })
        .collect())
}

/// The explicit derivative of the two-body energy of the PAW-LCAO supercell at fixed raw-AO
/// coefficients, per primitive atom (all its images moving with it), Hartree per Bohr.
pub struct TwoBodyGradient {
    /// The functions' motion: the smooth products, the coefficients of the compensation charges
    /// and of the one-center remainder (`lambda_k (d rho_k|rho_k)` at fixed shapes).
    pub functions: Vec<[f64; 3]>,
    /// The compensation shapes' motion (`-iG` of the moving atom's channels).
    pub shapes: Vec<[f64; 3]>,
    /// The largest imaginary part of a term before its real part was taken.
    pub imaginary: f64,
    /// The largest gap between the sampled Bloch sums and the crystal's (relative).
    pub samples: f64,
}

impl TwoBodyGradient {
    /// `functions + shapes`.
    pub fn total(&self) -> Vec<[f64; 3]> {
        self.functions
            .iter()
            .zip(&self.shapes)
            .map(|(a, b)| std::array::from_fn(|x| a[x] + b[x]))
            .collect()
    }
}

/// `d/dtau [1/2 sum_k lambda_k (rho_k|rho_k)]` at fixed frame coefficients (arguments as in
/// [`two_body_energies`]): `sum_k lambda_k [sum_pq x_k[p, q] ((d phi_p phi_q|rho_k) + (phi_p d
/// phi_q|rho_k)) + (d_shapes rho_k|rho_k)]`, the first through the pair densities of
/// [`moving_functions`] with the frame's, the second through [`Streamed::shape_motions`].
pub fn two_body_gradient(
    crystal: &Crystal,
    size: [usize; 3],
    points: &[KPoint],
    harmonics: &DMatrix<C64>,
    frame: &DMatrix<f64>,
    forms: &[(f64, DMatrix<f64>)],
) -> Result<TwoBodyGradient> {
    let moving = Moving::new(crystal, size, points, harmonics, frame)?;
    let r = frame.ncols();
    let atoms = moving.setup.sites().atoms.len();
    let f = forms.len();
    let mut out = TwoBodyGradient {
        functions: vec![[0.; 3]; atoms],
        shapes: vec![[0.; 3]; atoms],
        imaginary: 0.,
        samples: 0.,
    };
    let base = moving.base_streamed()?;
    let base_weights = form_weights(forms, r, r)?;
    let motions = base.shape_motions(&moving.setup, &base_weights)?;
    for (atom, motion) in motions.iter().enumerate() {
        for axis in 0..3 {
            let sum: C64 = forms
                .iter()
                .zip(motion)
                .map(|((lambda, _), m)| m[axis] * *lambda)
                .sum();
            out.shapes[atom][axis] = sum.re;
            out.imaginary = out.imaginary.max(sum.im.abs());
        }
    }
    drop(base);
    let n = 2 * r;
    let weights = form_weights(forms, r, n)?;
    // Rows (d phi_p, phi_q) and (phi_p, d phi_q) of the joined set.
    let rows: Vec<usize> = (0..r)
        .flat_map(|p| (0..r).flat_map(move |q| [(r + p) * n + q, p * n + r + q]))
        .collect();
    for atom in 0..atoms {
        for axis in 0..3 {
            let (streamed, gap) = moving.streamed(atom, axis)?;
            out.samples = out.samples.max(gap);
            let values = streamed.contracted(&moving.setup, &rows, &weights)?;
            let mut sum = C64::default();
            for (k, (lambda, x)) in forms.iter().enumerate() {
                for p in 0..r {
                    for q in 0..r {
                        let at = 2 * (p * r + q);
                        sum +=
                            *lambda * x[(p, q)] * (values[at * f + k] + values[(at + 1) * f + k]);
                    }
                }
            }
            out.functions[atom][axis] = sum.re;
            out.imaginary = out.imaginary.max(sum.im.abs());
        }
    }
    Ok(out)
}

/// The center derivative of the frame's overlap, `d<phi_p|S|phi_q> / d tau_axis` for the motion
/// of `atom` (arguments as in [`two_body_energies`]), from the pair densities of
/// [`moving_functions`] with the frame's: `M[d phi_p, phi_q] + M[phi_p, d phi_q]` of
/// [`Streamed::norms`]; with the same from the pair densities' `G = 0` charge (the compensation
/// charges' monopoles), and the gap of the sampled Bloch sums. For the gates of the derivative
/// functions: the overlap derivative is the one the one-body part already has.
#[allow(clippy::type_complexity)]
pub fn metric_gradient(
    crystal: &Crystal,
    size: [usize; 3],
    points: &[KPoint],
    harmonics: &DMatrix<C64>,
    frame: &DMatrix<f64>,
    atom: usize,
) -> Result<([DMatrix<C64>; 3], [DMatrix<C64>; 3], f64)> {
    let moving = Moving::new(crystal, size, points, harmonics, frame)?;
    let r = frame.ncols();
    let mut metric: [DMatrix<C64>; 3] = std::array::from_fn(|_| DMatrix::zeros(r, r));
    let mut charge: [DMatrix<C64>; 3] = std::array::from_fn(|_| DMatrix::zeros(r, r));
    let mut samples = 0f64;
    for axis in 0..3 {
        let (streamed, gap) = moving.streamed(atom, axis)?;
        samples = samples.max(gap);
        let (m, c) = streamed.norms(&moving.setup);
        metric[axis] = DMatrix::from_fn(r, r, |p, q| m[(r + p, q)] + m[(p, r + q)]);
        charge[axis] = DMatrix::from_fn(r, r, |p, q| c[(r + p, q)] + c[(p, r + q)]);
    }
    Ok((metric, charge, samples))
}

/// The smooth parts of the functions of `frame` on the supercell of `size` (rows as in
/// [`basis_two_body`]).
pub fn basis_samples(
    crystal: &Crystal,
    size: [usize; 3],
    points: &[KPoint],
    harmonics: &DMatrix<C64>,
    frame: &DMatrix<f64>,
) -> Result<BasisSamples> {
    let cell = Cell::of(crystal, size)?;
    let built = functions(crystal, &cell, points, harmonics, frame)?;
    let total = cell.points();
    let mut values = Vec::with_capacity(product(&[frame.ncols(), total])?);
    let (mut imaginary, mut largest) = (0f64, 0f64);
    for f in &built.values {
        for v in f.as_slice() {
            imaginary = imaginary.max(v.im.abs());
            largest = largest.max(v.re.abs());
            values.push(v.re);
        }
    }
    Ok(BasisSamples {
        values,
        points: total,
        imaginary: imaginary / largest.max(1e-300),
    })
}

/// The packed pairs `(p, r)`, `p >= r`, of `n` functions in the order of [`PairColumns`], and the
/// departures of the integrals read so far from real symmetric ones: the columns' projection.
struct Packing {
    n: usize,
    indices: Vec<(usize, usize)>,
    /// The largest imaginary part and the largest `|(pr|qs) - (pr|sq)|` the columns read so far
    /// left before their projection onto real symmetric integrals.
    departures: std::sync::Mutex<(f64, f64)>,
}

impl Packing {
    fn new(n: usize) -> Self {
        Self {
            n,
            indices: (0..n).flat_map(|p| (0..=p).map(move |r| (p, r))).collect(),
            departures: std::sync::Mutex::new((0., 0.)),
        }
    }

    fn departures(&self) -> (f64, f64) {
        *self.departures.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn note(&self, imaginary: f64, asymmetry: f64) {
        let mut seen = self.departures.lock().unwrap_or_else(|e| e.into_inner());
        seen.0 = seen.0.max(imaginary);
        seen.1 = seen.1.max(asymmetry);
    }

    fn pivot(&self, pivot: usize) -> Result<(usize, usize)> {
        self.indices
            .get(pivot)
            .copied()
            .ok_or_else(|| invalid("a Coulomb pivot is out of range"))
    }

    /// The real parts of the diagonal `(pr|pr)` of every packed pair.
    fn diagonal(&self, values: &[C64]) -> Vec<f64> {
        let imaginary = values.iter().fold(0f64, |a, v| a.max(v.im.abs()));
        self.note(imaginary, 0.);
        values.iter().map(|v| v.re).collect()
    }

    /// The packed real symmetric column from `values[q * n + s]`, `(pr|qs)` of one pivot.
    fn column(&self, values: &[C64]) -> Vec<f64> {
        let n = self.n;
        let (mut imaginary, mut asymmetry) = (0f64, 0f64);
        let column = self
            .indices
            .iter()
            .map(|&(q, s)| {
                let (a, b) = (values[q * n + s], values[s * n + q]);
                imaginary = imaginary.max(a.im.abs()).max(b.im.abs());
                asymmetry = asymmetry.max((a.re - b.re).abs());
                0.5 * (a.re + b.re)
            })
            .collect();
        self.note(imaginary, asymmetry);
        column
    }
}

/// The one-center remainder the columns of a supercell leave out, as signed rank-one pair terms
/// ([`super::pairs::onsite_factors`]), reserved through `ctx`.
fn onsite_correction(
    ctx: &crate::Context,
    setup: &response::setup::Setup,
    cell: &Cell,
    pairs: &Pairs,
) -> Result<crate::molecular::integrals::PairCorrection> {
    let (factors, weights) = super::pairs::onsite_factors(ctx, setup, cell, pairs)?;
    let memory = ctx.reserve(product(&[factors.len().max(1), pairs.n, pairs.n, 8])?)?;
    Ok(crate::molecular::integrals::PairCorrection {
        factors,
        weights,
        memory,
    })
}

/// The PAW pair Coulomb integrals of the functions of a real frame, read one Cholesky column at a
/// time ([`PairColumns`]) and streamed: the `n` functions are held on the supercell grid, not the
/// `n^2` transforms of their products, and each column `(pr|..)` forms its left density's
/// transform and takes one product over the grid with every right pair ([`super::streamed`]), so
/// neither the `n^4` tensor nor the pair transforms are ever formed. The columns carry the smooth
/// products and the compensation charges, a positive semidefinite Coulomb form; the one-center
/// remainder, which is not one (the N2 chain's pair matrix has eigenvalues down to -4.7e-6 Ha with
/// it), is [`BasisCoulomb::onsite`].
pub struct BasisCoulomb {
    setup: response::setup::Setup,
    streamed: Streamed,
    packing: Packing,
    /// `<phi_p|S|phi_q>` from the grid and the projectors: the identity for an orthonormal frame.
    pub metric: DMatrix<C64>,
}

impl BasisCoulomb {
    /// The functions of the real frame `frame` (`(N m) x n`, rows `R m + mu` of the primitive
    /// functions `chi U`, `U = harmonics`) on the supercell of `size`, from the crystal's Bloch
    /// sums `points` at the mesh coordinates of its cells.
    pub fn new(
        crystal: &Crystal,
        size: [usize; 3],
        points: &[KPoint],
        harmonics: &DMatrix<C64>,
        frame: &DMatrix<f64>,
    ) -> Result<Self> {
        if crystal.spinor {
            return Err(crate::Error::Unsupported(
                "the PAW-LCAO supercell does not take a spin-orbit dataset yet".into(),
            ));
        }
        let setup = response::setup::Setup::new(crystal)?;
        let cell = Cell::of(crystal, size)?;
        let built = functions(crystal, &cell, points, harmonics, frame)?;
        let grid = Grid::new(crystal, &setup, &cell, Kernel::Periodic)?;
        let streamed = Streamed::new(crystal, &setup, grid, built)?;
        let (metric, _) = streamed.norms(&setup);
        Ok(Self {
            packing: Packing::new(frame.ncols()),
            setup,
            streamed,
            metric,
        })
    }

    /// The one-center remainder the columns leave out, as signed rank-one pair terms
    /// ([`super::pairs::onsite_factors`]).
    pub fn onsite(&self) -> Result<crate::molecular::integrals::PairCorrection> {
        onsite_correction(
            self.streamed.context(),
            &self.setup,
            self.streamed.cell(),
            self.streamed.charges(),
        )
    }
}

impl PairColumns for BasisCoulomb {
    fn m(&self) -> usize {
        self.packing.n
    }

    fn departures(&self) -> (f64, f64) {
        self.packing.departures()
    }

    fn diagonal(&self) -> Result<Vec<f64>> {
        Ok(self
            .packing
            .diagonal(&self.streamed.diagonal(&self.packing.indices)?))
    }

    fn column(&self, pivot: usize) -> Result<Vec<f64>> {
        let (p, r) = self.packing.pivot(pivot)?;
        Ok(self.packing.column(&self.streamed.column(p, r)?))
    }
}

/// The same columns from the stored pair densities of every pair (`n^2` transforms on the
/// supercell grid, [`super::pairs::coulomb_rows`]): the reference the streamed [`BasisCoulomb`]
/// is gated against (`paw_supercell_columns`), `n^2 x points x 16` bytes where the streamed
/// columns hold `n x points x 16`.
pub struct StoredColumns {
    context: crate::Context,
    setup: response::setup::Setup,
    cell: Cell,
    pairs: Pairs,
    packing: Packing,
    /// `<phi_p|S|phi_q>` from the stored pairs ([`Pairs::norms`]).
    pub metric: DMatrix<C64>,
}

impl StoredColumns {
    /// The pair densities of the functions of `frame` (arguments as [`BasisCoulomb::new`]).
    pub fn new(
        crystal: &Crystal,
        size: [usize; 3],
        points: &[KPoint],
        harmonics: &DMatrix<C64>,
        frame: &DMatrix<f64>,
    ) -> Result<Self> {
        let setup = response::setup::Setup::new(crystal)?;
        let cell = Cell::of(crystal, size)?;
        let built = functions(crystal, &cell, points, harmonics, frame)?;
        let n = frame.ncols();
        let modes: Vec<Mode> = (0..n).map(|at| Mode { at, cell: [0; 3] }).collect();
        let pairs = Pairs::new(crystal, &setup, &cell, &built, &modes)?;
        let (metric, _) = pairs.norms(&setup, &cell)?;
        Ok(Self {
            context: crystal.context.clone(),
            setup,
            cell,
            pairs,
            packing: Packing::new(n),
            metric,
        })
    }

    fn rows(&self, rows: &[usize], diagonal: bool) -> Result<Vec<C64>> {
        Ok(coulomb_rows(
            &self.context,
            &self.setup,
            &self.cell,
            Kernel::Periodic,
            &self.pairs,
            &self.pairs,
            false,
            false,
            Some(rows),
            diagonal,
        )?
        .values)
    }
}

impl PairColumns for StoredColumns {
    fn m(&self) -> usize {
        self.packing.n
    }

    fn departures(&self) -> (f64, f64) {
        self.packing.departures()
    }

    fn diagonal(&self) -> Result<Vec<f64>> {
        let n = self.packing.n;
        let rows: Vec<usize> = self
            .packing
            .indices
            .iter()
            .map(|&(p, r)| p * n + r)
            .collect();
        Ok(self.packing.diagonal(&self.rows(&rows, true)?))
    }

    fn column(&self, pivot: usize) -> Result<Vec<f64>> {
        let (p, r) = self.packing.pivot(pivot)?;
        Ok(self
            .packing
            .column(&self.rows(&[p * self.packing.n + r], false)?))
    }
}

/// The one-center Hartree-Fock correction of the crystal's projector operator, `P x P` in its
/// projector layout (block diagonal by atom, diagonal in `m`): the molecular PAW Hartree-Fock
/// one-center operator less the dataset's coupling `D_ion` (which `fixed(k)` holds, screened by
/// the reference atom for Kohn-Sham), `-xc - Delta W . reference + X_core` per channel
/// (`paw::molecular::hartree_fock_onsite_channels`): the reference atom's one-center screening
/// replaced by the interaction of the electrons themselves, and the frozen core's exchange with the
/// valence. One owner with the molecule's Hartree-Fock Hamiltonian, which refuses the same cores.
pub fn hartree_fock_onsite(crystal: &Crystal) -> Result<DMatrix<C64>> {
    let ctx = &crystal.context;
    let setup = response::setup::Setup::new(crystal)?;
    let p = crystal.p;
    let owners = setup.projector_atoms();
    let labels = setup.projector_labels();
    let mut out = DMatrix::<C64>::zeros(p, p);
    let mut cache: std::collections::BTreeMap<
        usize,
        Vec<crate::paw::molecular::HartreeFockChannel>,
    > = Default::default();
    for atom in 0..setup.sites().atoms.len() {
        let key = setup.species()[atom];
        if let std::collections::btree_map::Entry::Vacant(entry) = cache.entry(key) {
            entry.insert(crate::paw::molecular::hartree_fock_onsite_channels(
                ctx,
                setup.dataset(atom),
            )?);
        }
        let channels = &cache[&key];
        let own: Vec<usize> = (0..p).filter(|j| owners[*j] == atom).collect();
        for &i in &own {
            for &j in &own {
                let ((li, mi, ri), (lj, mj, rj)) = (labels[i], labels[j]);
                if li != lj || mi != mj {
                    continue;
                }
                let channel = channels.iter().find(|c| c.l == li).ok_or_else(|| {
                    invalid("a projector channel has no Hartree-Fock one-center block")
                })?;
                out[(i, j)] = C64::new(channel.onsite[(ri, rj)], 0.);
            }
        }
    }
    Ok(out)
}

/// `sum_cd Delta W_abcd reference_cd` of every atom in the crystal's projector layout (`P x P`):
/// the reference atom's one-center Hartree potential through the one-center remainder the
/// two-body integrals add ([`super::pairs`]), against which the molecule's own (inside
/// [`hartree_fock_onsite`]) is audited.
pub fn reference_hartree(crystal: &Crystal) -> Result<DMatrix<C64>> {
    use crate::paw::onecenter::{Basis, Projector};
    let ctx = &crystal.context;
    let setup = response::setup::Setup::new(crystal)?;
    let p = crystal.p;
    let owners = setup.projector_atoms();
    let labels = setup.projector_labels();
    let mut out = DMatrix::<C64>::zeros(p, p);
    for atom in 0..setup.sites().atoms.len() {
        let own: Vec<usize> = (0..p).filter(|j| owners[*j] == atom).collect();
        let channels =
            crate::paw::molecular::hartree_fock_onsite_channels(ctx, setup.dataset(atom))?;
        let projectors: Vec<Projector> = own
            .iter()
            .map(|&j| Projector {
                l: labels[j].0,
                m: labels[j].1,
                index: labels[j].2,
                real: false,
            })
            .collect();
        let w = super::pairs::coulomb_symmetrized(
            &crate::periodic::setup::remainder::onsite_remainder(
                ctx,
                setup.dataset(atom),
                Basis::Raw,
                &projectors,
            )?,
        );
        let n = own.len();
        let reference = DMatrix::from_fn(n, n, |a, b| {
            let (pa, pb) = (&projectors[a], &projectors[b]);
            if pa.l != pb.l || pa.m != pb.m {
                return C64::default();
            }
            let ch = channels
                .iter()
                .find(|c| c.l == pa.l)
                .expect("a reference channel per l");
            C64::new(ch.reference[(pa.index, pb.index)], 0.)
        });
        for a in 0..n {
            for b in 0..n {
                let mut sum = C64::default();
                for c in 0..n {
                    for d in 0..n {
                        sum += w.data[((a * n + b) * n + c) * n + d] * reference[(c, d)];
                    }
                }
                out[(own[a], own[b])] = sum;
            }
        }
    }
    Ok(out)
}
