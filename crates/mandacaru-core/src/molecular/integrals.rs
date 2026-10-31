use super::{invalid, SpatialGrid};
use crate::basis::Basis;
use crate::math::fft::{spectral_laplacian, Poisson};
use crate::math::linalg::real_eigen;
use crate::math::storage::{with_rows, Precision, RealSamples, Sample, SampleBlocks};
use crate::quantum::C64;
use crate::{product, Buffer, Context, Geometry, Reservation, Result};
use nalgebra::DMatrix;
use std::sync::{Arc, Mutex};

pub fn pair(a: usize, b: usize) -> usize {
    let (a, b) = (a.max(b), a.min(b));
    a * (a + 1) / 2 + b
}

pub(crate) fn packed_len(n: usize) -> Result<usize> {
    n.checked_mul(n.checked_add(1).ok_or(crate::Error::Overflow)?)
        .map(|v| v / 2)
        .ok_or(crate::Error::Overflow)
}
/// Signed separable corrections to a Coulomb pair matrix. Factors are in
/// the same orthonormal spatial basis as the grid samples.
pub struct PairCorrection {
    pub factors: Vec<DMatrix<f64>>,
    pub weights: Vec<f64>,
    pub memory: Reservation,
}
impl PairCorrection {
    pub fn value(&self, a: usize, b: usize, c: usize, d: usize) -> f64 {
        self.factors
            .iter()
            .zip(&self.weights)
            .map(|(g, w)| w * g[(a, b)] * g[(c, d)])
            .sum()
    }
}
/// Density-independent additions from a separable pseudopotential setup.
pub struct MolecularCorrections {
    pub external: Buffer<f64>,
    pub overlap: DMatrix<f64>,
    pub potential: DMatrix<f64>,
    pub pairs: Arc<PairCorrection>,
    pub charges: Vec<f64>,
    pub electrons: usize,
    pub dft: Option<PawKohnSham>,
}
/// Frozen atomic one-center KS model, separate from the interacting HF model.
pub struct PawKohnSham {
    pub one: DMatrix<f64>,
    pub pairs: PairCorrection,
    pub core: Buffer<f64>,
    pub core_tau: Buffer<f64>,
    pub constant: f64,
    pub relativistic_exchange: bool,
    pub memory: Reservation,
}
pub enum Eri {
    Augmented {
        m: usize,
        base: Arc<Eri>,
        correction: Arc<PairCorrection>,
    },
    Packed {
        m: usize,
        values: Buffer<f64>,
    },
    Cholesky {
        m: usize,
        rank: usize,
        factors: Buffer<f64>,
        /// Maximum unused diagonal when measured by adaptive pivot selection.
        /// Fixed-pivot response does not scan unused Coulomb columns.
        residual: Option<f64>,
        pivots: Buffer<usize>,
    },
}
/// How a two-body factorization is stored, for a log: the representation, the
/// rank of the augmentation correction when there is one, and for a Cholesky
/// base its rank and the largest unused diagonal (`None` when the pivots were
/// fixed and the diagonal was not scanned).
#[derive(Debug, Clone, PartialEq)]
pub struct EriReport {
    pub representation: &'static str,
    pub augmentation_rank: Option<usize>,
    pub cholesky: Option<(usize, Option<f64>)>,
}
impl Eri {
    pub fn report(&self) -> EriReport {
        match self {
            Self::Augmented {
                base, correction, ..
            } => EriReport {
                representation: "cholesky+augmentation",
                augmentation_rank: Some(correction.factors.len()),
                cholesky: base.report().cholesky,
            },
            Self::Packed { .. } => EriReport {
                representation: "packed",
                augmentation_rank: None,
                cholesky: None,
            },
            Self::Cholesky { rank, residual, .. } => EriReport {
                representation: "cholesky",
                augmentation_rank: None,
                cholesky: Some((*rank, *residual)),
            },
        }
    }
    pub fn pivots(&self) -> Option<&[usize]> {
        match self {
            Self::Cholesky { rank, pivots, .. } => Some(&pivots.as_slice()[..*rank]),
            Self::Augmented { base, .. } => base.pivots(),
            Self::Packed { .. } => None,
        }
    }
    pub fn m(&self) -> usize {
        match self {
            Self::Packed { m, .. } | Self::Cholesky { m, .. } | Self::Augmented { m, .. } => *m,
        }
    }
    pub fn value(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let a = pair(p, r);
        let b = pair(q, s);
        match self {
            Self::Augmented {
                base, correction, ..
            } => base.value(p, q, r, s) + correction.value(p, r, q, s),
            Self::Packed { values, .. } => values.as_slice()[pair(a, b)],
            Self::Cholesky { rank, factors, .. } => (0..*rank)
                .map(|k| factors.as_slice()[a * rank + k] * factors.as_slice()[b * rank + k])
                .sum(),
        }
    }
    pub fn dense(&self, ctx: &Context) -> Result<Buffer<f64>> {
        let m = self.m();
        let mut out = ctx.allocate(product(&[m, m, m, m])?)?;
        for p in 0..m {
            for q in 0..m {
                for r in 0..m {
                    for s in 0..m {
                        out.as_mut_slice()[((p * m + q) * m + r) * m + s] = self.value(p, q, r, s);
                    }
                }
            }
        }
        Ok(out)
    }
    pub fn cholesky(self, ctx: &Context, tolerance: f64) -> Result<Self> {
        if !tolerance.is_finite() || tolerance <= 0. {
            return Err(invalid("Cholesky tolerance must be finite and positive"));
        }
        let m = self.m();
        let n = packed_len(m)?;
        let mut residual = ctx.allocate::<f64>(n)?;
        let mut columns = ctx.allocate::<f64>(product(&[n, n])?)?;
        let indices: Vec<_> = (0..m).flat_map(|p| (0..=p).map(move |q| (p, q))).collect();
        for (a, &(p, q)) in indices.iter().enumerate() {
            residual.as_mut_slice()[a] = self.value(p, p, q, q);
        }
        let mut rank = 0;
        let mut pivots = ctx.allocate::<usize>(n)?;
        loop {
            let pivot = (0..n)
                .max_by(|&a, &b| residual.as_slice()[a].total_cmp(&residual.as_slice()[b]))
                .unwrap();
            let diagonal = residual.as_slice()[pivot];
            if diagonal <= tolerance || rank == n {
                break;
            }
            let (p, r) = indices[pivot];
            pivots.as_mut_slice()[rank] = pivot;
            for (a, &(q, s)) in indices.iter().enumerate() {
                let previous: f64 = (0..rank)
                    .map(|k| columns.as_slice()[a * n + k] * columns.as_slice()[pivot * n + k])
                    .sum();
                let value = (self.value(p, q, r, s) - previous) / diagonal.sqrt();
                columns.as_mut_slice()[a * n + rank] = value;
                residual.as_mut_slice()[a] -= value * value;
            }
            rank += 1;
        }
        let error = residual.as_slice().iter().copied().fold(0f64, f64::max);
        if residual.as_slice().iter().any(|d| *d < -1e-9) {
            return Err(invalid(
                "Coulomb pair matrix is not positive semidefinite at the requested accuracy",
            ));
        }
        let mut factors = ctx.allocate(product(&[n, rank])?)?;
        for a in 0..n {
            for k in 0..rank {
                factors.as_mut_slice()[a * rank + k] = columns.as_slice()[a * n + k];
            }
        }
        Ok(Self::Cholesky {
            m,
            rank,
            factors,
            residual: Some(error),
            pivots,
        })
    }
}
pub struct MolecularIntegrals {
    pub context: Context,
    pub geometry: Geometry,
    pub grid: SpatialGrid,
    pub basis: Basis,
    pub samples: RealSamples,
    pub orthonormal_samples: RealSamples,
    pub overlap: Buffer<f64>,
    pub kinetic: Buffer<f64>,
    pub potential: Buffer<f64>,
    pub lowdin: DMatrix<f64>,
    pub one: Buffer<f64>,
    eri: Mutex<Option<Arc<Eri>>>,
    cholesky_tolerance: f64,
    fixed_pivots: Option<Buffer<usize>>,
    pub nuclear: f64,
    pub kinetic_method: String,
    pub overlap_min: f64,
    pub resolution_ratios: Vec<f64>,
    pub corrections: Option<MolecularCorrections>,
    pub paw_setup: Option<Arc<crate::paw::molecular::Setup>>,
    _matrices: Reservation,
}
pub fn pair_integrals<T: Sample>(
    ctx: &Context,
    grid: &SpatialGrid,
    samples: &[T],
    m: usize,
) -> Result<Eri> {
    pair_integrals_with_solver(ctx, grid, samples, m, &mut Poisson::isolated(ctx, grid)?)
}
/// Pivoted Cholesky using individual Coulomb columns. Neither the pair matrix
/// nor a four-index tensor is allocated, including during factorization.
pub(super) fn cholesky_from_grid<T: Sample>(
    ctx: &Context,
    grid: &SpatialGrid,
    samples: &[T],
    m: usize,
    tolerance: f64,
    solver: &mut Poisson,
    fixed_pivots: Option<&[usize]>,
) -> Result<Eri> {
    if m == 0
        || samples.len() != product(&[m, grid.index.points()])?
        || !tolerance.is_finite()
        || tolerance <= 0.
    {
        return Err(invalid("invalid grid Cholesky dimensions or tolerance"));
    }
    let n = packed_len(m)?;
    if let Some(pivots) = fixed_pivots {
        let _validation_memory = ctx.reserve(n)?;
        let mut seen = vec![false; n];
        for &pivot in pivots {
            if pivot >= n || seen[pivot] {
                return Err(invalid(
                    "fixed Coulomb pivots must be distinct orbital pairs",
                ));
            }
            seen[pivot] = true;
        }
    }
    let g = grid.index.points();
    let mut diagonal = ctx.allocate::<f64>(n)?;
    let mut rho = ctx.allocate::<C64>(g)?;
    let mut potential = ctx.allocate::<C64>(g)?;
    let _index_memory = ctx.reserve(product(&[n, 32])?)?;
    let mut projector = if m >= 16 {
        Some(crate::math::projection::RealProjector::new(
            ctx, samples, m, g,
        )?)
    } else {
        None
    };
    let mut field = if projector.is_some() {
        Some(ctx.allocate::<f64>(g)?)
    } else {
        None
    };
    let indices: Vec<_> = (0..m).flat_map(|a| (0..=a).map(move |b| (a, b))).collect();
    if fixed_pivots.is_none() {
        for (i, &(a, b)) in indices.iter().enumerate() {
            for k in 0..g {
                rho.as_mut_slice()[k] =
                    C64::new(samples[a * g + k].widen() * samples[b * g + k].widen(), 0.);
            }
            diagonal.as_mut_slice()[i] = solver.quadratic_form(rho.as_slice())?;
        }
    }
    let mut columns: Vec<Buffer<f64>> = Vec::new();
    let mut pivots = ctx.allocate::<usize>(fixed_pivots.map_or(n, |p| p.len()))?;
    loop {
        if columns.len() == fixed_pivots.map_or(n, |p| p.len()) {
            break;
        }
        let pivot = fixed_pivots.map_or_else(
            || {
                (0..n)
                    .max_by(|&a, &b| diagonal.as_slice()[a].total_cmp(&diagonal.as_slice()[b]))
                    .unwrap()
            },
            |p| p[columns.len()],
        );
        let mut maximum = diagonal.as_slice()[pivot];
        if fixed_pivots.is_none() && maximum <= tolerance {
            break;
        }
        pivots.as_mut_slice()[columns.len()] = pivot;
        let (a, b) = indices[pivot];
        for k in 0..g {
            rho.as_mut_slice()[k] =
                C64::new(samples[a * g + k].widen() * samples[b * g + k].widen(), 0.);
        }
        solver.solve_into(rho.as_slice(), potential.as_mut_slice())?;
        if fixed_pivots.is_some() {
            maximum = rho
                .as_slice()
                .iter()
                .zip(potential.as_slice())
                .map(|(r, v)| r.re * v.re)
                .sum::<f64>()
                * grid.index.dv()
                - columns
                    .iter()
                    .map(|c| c.as_slice()[pivot].powi(2))
                    .sum::<f64>();
        }
        if !maximum.is_finite() || maximum <= 0. {
            return Err(invalid(
                "a retained Coulomb pivot lost positive rank at the displaced geometry",
            ));
        }
        let mut column = match ctx.allocate::<f64>(n) {
            Err(crate::Error::Budget { .. })
                if projector
                    .as_mut()
                    .is_some_and(|p| p.release_extra_workspaces()) =>
            {
                ctx.allocate::<f64>(n)?
            }
            allocation => allocation?,
        };
        let projected = if let (Some(projector), Some(field)) = (projector.as_mut(), field.as_mut())
        {
            for (value, potential) in field.as_mut_slice().iter_mut().zip(potential.as_slice()) {
                *value = potential.re;
            }
            Some(projector.project(field.as_slice())?)
        } else {
            None
        };
        for (i, &(p, q)) in indices.iter().enumerate() {
            let integral = projected.map_or_else(
                || {
                    (0..g)
                        .map(|k| {
                            samples[p * g + k].widen()
                                * samples[q * g + k].widen()
                                * potential.as_slice()[k].re
                        })
                        .sum::<f64>()
                },
                |matrix| matrix[p + q * m],
            ) * grid.index.dv();
            let previous = columns
                .iter()
                .map(|c| c.as_slice()[i] * c.as_slice()[pivot])
                .sum::<f64>();
            let value = (integral - previous) / maximum.sqrt();
            column.as_mut_slice()[i] = value;
            if fixed_pivots.is_none() {
                diagonal.as_mut_slice()[i] -= value * value;
            }
        }
        columns.push(column);
    }
    if diagonal
        .as_slice()
        .iter()
        .any(|v| !v.is_finite() || *v < -1e-9)
    {
        return Err(invalid(
            "Coulomb operator is not positive semidefinite at this grid accuracy",
        ));
    }
    let residual = fixed_pivots
        .is_none()
        .then(|| diagonal.as_slice().iter().copied().fold(0f64, f64::max));
    let rank = columns.len();
    // Projection scratch is no longer needed while columns are copied into
    // their compact retained representation.
    drop(projector);
    drop(field);
    drop(rho);
    drop(potential);
    let mut factors = ctx.allocate::<f64>(product(&[n, rank])?)?;
    for i in 0..n {
        for (k, column) in columns.iter().enumerate() {
            factors.as_mut_slice()[i * rank + k] = column.as_slice()[i];
        }
    }
    Ok(Eri::Cholesky {
        m,
        rank,
        factors,
        residual,
        pivots,
    })
}
pub(super) fn pair_integrals_with_solver<T: Sample>(
    ctx: &Context,
    grid: &SpatialGrid,
    samples: &[T],
    m: usize,
    poisson: &mut Poisson,
) -> Result<Eri> {
    let g = grid.index.points();
    if samples.len() != product(&[m, g])? || m == 0 {
        return Err(invalid("orbital samples must match the molecular grid"));
    }
    let n = packed_len(m)?;
    let mut values = ctx.allocate(packed_len(n)?)?;
    let mut density = ctx.allocate::<C64>(g)?;
    let mut potential = ctx.allocate::<C64>(g)?;
    for b in 0..m {
        for d in 0..=b {
            let bd = pair(b, d);
            for k in 0..g {
                density.as_mut_slice()[k] =
                    C64::new(samples[b * g + k].widen() * samples[d * g + k].widen(), 0.);
            }
            poisson.solve_into(density.as_slice(), potential.as_mut_slice())?;
            for a in 0..m {
                for c in 0..=a {
                    let ac = pair(a, c);
                    if ac > bd {
                        continue;
                    }
                    let integral: f64 = (0..g)
                        .map(|k| {
                            samples[a * g + k].widen()
                                * samples[c * g + k].widen()
                                * potential.as_slice()[k].re
                        })
                        .sum();
                    values.as_mut_slice()[pair(ac, bd)] = integral * grid.index.dv();
                }
            }
        }
    }
    Ok(Eri::Packed { m, values })
}
fn laplacian_of(
    ctx: &Context,
    grid: &SpatialGrid,
    kinetic: &str,
    field: &[f64],
) -> Result<Buffer<f64>> {
    if kinetic == "spectral" {
        spectral_laplacian(ctx, grid, field)
    } else {
        crate::kernels::laplacian(ctx, &grid.index, 1, field)
    }
}
/// Kinetic matrix `t[a * m + b] = -1/2 <psi_a, laplacian psi_b>` of exact
/// float64 orbitals for a float32 stack, whose stored rows would put a
/// rounding error of 1e-7 relative on each side. Laplacian rows are resident
/// one third of the orbitals at a time (about 2.7 stack-lengths of float64 per
/// third), and the other side is re-evaluated in point blocks. Pairs with the
/// row group at or below the Laplacian group are computed once and mirrored.
fn exact_kinetic(
    ctx: &Context,
    basis: &Basis,
    grid: &SpatialGrid,
    kinetic: &str,
    t: &mut [f64],
) -> Result<()> {
    let m = basis.orbitals.len();
    let g = grid.index.points();
    let group = m.div_ceil(3).max(1);
    let stride = g.min(SampleBlocks::BLOCK);
    let mut laplacians = ctx.allocate::<f64>(product(&[group, g])?)?;
    let mut row = ctx.allocate::<f64>(g)?;
    let mut block = ctx.allocate::<f64>(product(&[group, stride])?)?;
    let mut raw = vec![0.; group];
    for b0 in (0..m).step_by(group) {
        let nb = group.min(m - b0);
        for j in 0..nb {
            for (k, value) in row.as_mut_slice().iter_mut().enumerate() {
                *value = basis.orbitals[b0 + j].evaluate(grid.point(k));
            }
            let lap = laplacian_of(ctx, grid, kinetic, row.as_slice())?;
            laplacians.as_mut_slice()[j * g..(j + 1) * g].copy_from_slice(lap.as_slice());
        }
        for a0 in (0..=b0).step_by(group) {
            let na = group.min(m - a0);
            for start in (0..g).step_by(stride) {
                let count = stride.min(g - start);
                for i in 0..count {
                    basis.evaluate_at(a0, grid.point(start + i), &mut raw[..na]);
                    for (a, value) in raw[..na].iter().enumerate() {
                        block.as_mut_slice()[a * stride + i] = *value;
                    }
                }
                for a in 0..na {
                    let left = &block.as_slice()[a * stride..a * stride + count];
                    for j in 0..nb {
                        let right = &laplacians.as_slice()[j * g + start..j * g + start + count];
                        let mut sum = t[(a0 + a) * m + b0 + j];
                        for i in 0..count {
                            sum += left[i] * right[i];
                        }
                        t[(a0 + a) * m + b0 + j] = sum;
                    }
                }
            }
        }
    }
    let scale = -0.5 * grid.index.dv();
    for (a, b) in (0..m).flat_map(|a| (0..m).map(move |b| (a, b))) {
        if a / group <= b / group {
            t[a * m + b] *= scale;
        }
    }
    // Mirror the pairs that were evaluated with the left orbital in an
    // earlier group than the Laplacian row.
    for a in 0..m {
        for b in 0..m {
            if a / group < b / group {
                t[b * m + a] = t[a * m + b];
            }
        }
    }
    Ok(())
}
/// The exact float64 values of all orbitals on a block of grid points. A
/// float64 stack IS the exact value, so its block is widened from storage; a
/// float32 stack has already been rounded, so the block is re-evaluated from
/// the orbitals. Everything that must not inherit rounding noise (overlap,
/// potential, the orthonormal rows) reads this, never the stored float32 rows.
struct ExactBlocks<'a> {
    basis: &'a Basis,
    grid: &'a SpatialGrid,
    stride: usize,
    points: usize,
    raw: Vec<f64>,
    source: ExactSource<'a>,
}
enum ExactSource<'a> {
    Stored(SampleBlocks<'a>),
    Evaluated(Buffer<f64>),
}
impl<'a> ExactBlocks<'a> {
    fn new(
        ctx: &Context,
        basis: &'a Basis,
        grid: &'a SpatialGrid,
        samples: &'a RealSamples,
    ) -> Result<Self> {
        let (m, points) = (basis.orbitals.len(), grid.index.points());
        let stride = points.min(SampleBlocks::BLOCK);
        let source = match samples.precision() {
            Precision::Float64 => ExactSource::Stored(SampleBlocks::new(ctx, samples, m, points)?),
            Precision::Float32 => ExactSource::Evaluated(ctx.allocate(product(&[m, stride])?)?),
        };
        Ok(Self {
            basis,
            grid,
            stride,
            points,
            raw: vec![0.; m],
            source,
        })
    }
    /// Orbital `a` of the loaded block is at `a * stride`; returns the point
    /// count and all values.
    fn load(&mut self, start: usize) -> Result<(usize, &[f64])> {
        let count = self.stride.min(self.points - start);
        match &mut self.source {
            ExactSource::Stored(blocks) => {
                blocks.load(start)?;
                Ok((count, blocks.values()))
            }
            ExactSource::Evaluated(block) => {
                for i in 0..count {
                    self.basis
                        .evaluate_at(0, self.grid.point(start + i), &mut self.raw);
                    for (a, value) in self.raw.iter().enumerate() {
                        block.as_mut_slice()[a * self.stride + i] = *value;
                    }
                }
                Ok((count, block.as_slice()))
            }
        }
    }
}
impl MolecularIntegrals {
    /// Rebuild on the same fixed integration grid with the same basis recipe
    /// and PAW species. Numerical setup objects stay owned in Rust.
    pub fn at_geometry(&self, geometry: Geometry) -> Result<Self> {
        if geometry.numbers() != self.geometry.numbers() || geometry.pbc() != self.geometry.pbc() {
            return Err(invalid(
                "geometry continuation needs the same atomic composition and boundary conditions",
            ));
        }
        self.context.run(|| {
            if let Some(setup) = &self.paw_setup {
                return setup.at_geometry(
                    &self.context,
                    geometry,
                    self.grid.clone(),
                    &self.kinetic_method,
                    self.cholesky_tolerance,
                    self.precision(),
                );
            }
            if self.corrections.is_some() {
                return Err(invalid(
                    "custom molecular corrections have no geometry-dependent setup",
                ));
            }
            let mut orbitals = self.basis.orbitals.clone();
            for orbital in &mut orbitals {
                orbital.center = geometry.positions()[orbital.atom];
            }
            let basis = Basis::from_orbitals(&self.context, orbitals)?;
            let samples = basis.sample(&self.context, &self.grid, self.precision())?;
            Self::from_basis(
                &self.context,
                geometry,
                self.grid.clone(),
                basis,
                samples,
                &self.kinetic_method,
                Some(self.cholesky_tolerance),
                None,
            )
        })
    }
    pub fn has_eri(&self) -> bool {
        self.eri.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }
    /// Keep the same low-rank Coulomb approximation while differentiating a
    /// geometry. Fresh calculations continue to select their own pivots.
    pub fn at_geometry_with_fixed_pivots(&self, geometry: Geometry) -> Result<Self> {
        let source = self.eri()?;
        let plan = source
            .pivots()
            .ok_or_else(|| invalid("geometry response needs a Coulomb factorization"))?;
        let mut rebuilt = self.at_geometry(geometry)?;
        let mut pivots = self.context.allocate::<usize>(plan.len())?;
        pivots.as_mut_slice().copy_from_slice(plan);
        rebuilt.fixed_pivots = Some(pivots);
        Ok(rebuilt)
    }
    /// Build Coulomb factors only when a solver or Hamiltonian asks for them.
    /// Failed allocations do not poison the cache and can be retried.
    pub fn eri(&self) -> Result<Arc<Eri>> {
        let mut cache = self.eri.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(value) = cache.as_ref() {
            return Ok(value.clone());
        }
        let mut solver = Poisson::isolated(&self.context, &self.grid)?;
        let base = Arc::new(with_rows!(self.orthonormal_samples.rows(), |samples| {
            cholesky_from_grid(
                &self.context,
                &self.grid,
                samples,
                self.m(),
                self.cholesky_tolerance,
                &mut solver,
                self.fixed_pivots.as_ref().map(|p| p.as_slice()),
            )?
        }));
        let factors = if let Some(correction) = self.corrections.as_ref() {
            Arc::new(Eri::Augmented {
                m: self.m(),
                base,
                correction: correction.pairs.clone(),
            })
        } else {
            base
        };
        *cache = Some(factors.clone());
        Ok(factors)
    }
    pub fn particles(
        &self,
        charge: i32,
        moment: f64,
        electrons: Option<usize>,
    ) -> Result<[usize; 2]> {
        if !moment.is_finite() {
            return Err(invalid("initial magnetic moment must be finite"));
        }
        let total = electrons.map(|n| n as i128).unwrap_or_else(|| {
            self.corrections.as_ref().map_or_else(
                || {
                    self.geometry
                        .numbers()
                        .iter()
                        .map(|&z| z as i128)
                        .sum::<i128>()
                },
                |c| c.electrons as i128,
            ) - charge as i128
        });
        if total < 0 || total > 2 * self.m() as i128 {
            return Err(invalid("electron count does not fit the spatial basis"));
        }
        let n = total as usize;
        let unpaired = if moment.abs() > 1e-8 {
            moment.abs().round_ties_even() as usize
        } else {
            n % 2
        };
        if unpaired > n || (n - unpaired) % 2 != 0 {
            return Err(invalid(
                "electron count and initial magnetic moment have inconsistent parity",
            ));
        }
        let particles = [(n + unpaired) / 2, (n - unpaired) / 2];
        if particles.iter().any(|p| *p > self.m()) {
            return Err(invalid("spin occupation does not fit the spatial basis"));
        }
        Ok(particles)
    }
    /// The precision is fixed at construction: the orbital stacks are built
    /// directly in it and never exist in float64 beside it.
    pub fn new(
        ctx: &Context,
        geometry: Geometry,
        grid: SpatialGrid,
        basis_spec: &str,
        kinetic: &str,
        cholesky: Option<f64>,
        precision: Precision,
    ) -> Result<Self> {
        let basis = Basis::build(ctx, &geometry, basis_spec)?;
        let samples = basis.sample(ctx, &grid, precision)?;
        Self::from_basis(ctx, geometry, grid, basis, samples, kinetic, cholesky, None)
    }
    /// Integrals from a basis and its raw orbital samples on `grid`; the
    /// samples' precision is the calculation's precision.
    #[allow(clippy::too_many_arguments)]
    pub fn from_basis(
        ctx: &Context,
        geometry: Geometry,
        grid: SpatialGrid,
        basis: Basis,
        samples: RealSamples,
        kinetic: &str,
        cholesky: Option<f64>,
        mut corrections: Option<MolecularCorrections>,
    ) -> Result<Self> {
        if grid.periodic {
            return Err(invalid("molecular integrals require an isolated grid; periodic electrostatics is a separate native primitive"));
        }
        if kinetic != "spectral" && kinetic != "fd" {
            return Err(invalid("kinetic must be spectral or fd"));
        }
        let m = basis.orbitals.len();
        let g = grid.index.points();
        if let Some(c) = &corrections {
            if c.external.as_slice().len() != g
                || c.overlap.shape() != (m, m)
                || c.potential.shape() != (m, m)
                || c.pairs.weights.len() != c.pairs.factors.len()
                || c.pairs.factors.iter().any(|f| f.shape() != (m, m))
                || c.charges.len() != geometry.numbers().len()
            {
                return Err(invalid(
                    "molecular correction dimensions disagree with the basis/grid",
                ));
            }
            crate::finite(c.external.as_slice(), "external potential")?;
            crate::finite(c.overlap.as_slice(), "overlap correction")?;
            crate::finite(c.potential.as_slice(), "potential correction")?;
            crate::finite(&c.charges, "ionic charges")?;
            crate::finite(&c.pairs.weights, "Coulomb correction weights")?;
            for factor in &c.pairs.factors {
                crate::finite(factor.as_slice(), "Coulomb correction factors")?;
            }
        }
        let matrices = ctx.reserve(product(&[m, m, 8, 12])?)?;
        if samples.len() != product(&[m, g])? {
            return Err(invalid("orbital samples must match the basis and grid"));
        }
        let precision = samples.precision();
        let mut overlap = ctx.allocate::<f64>(product(&[m, m])?)?;
        let mut t = ctx.allocate::<f64>(product(&[m, m])?)?;
        let mut v = ctx.allocate::<f64>(product(&[m, m])?)?;
        let mut external = ctx.allocate::<f64>(g)?;
        let softening = 0.5 * grid.spacings().into_iter().fold(f64::INFINITY, f64::min);
        for k in 0..g {
            if let Some(c) = &corrections {
                external.as_mut_slice()[k] = c.external.as_slice()[k];
                continue;
            }
            let point = grid.point(k);
            external.as_mut_slice()[k] = geometry
                .numbers()
                .iter()
                .zip(geometry.positions())
                .map(|(&z, r)| {
                    let distance = (0..3)
                        .map(|j| (point[j] - r[j]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        .max(softening);
                    -(z as f64) / distance
                })
                .sum();
        }
        // Matrix elements are contracted from the EXACT float64 orbital values,
        // not from rounded stored rows: the nuclear potential weights a few
        // points so heavily that rounding noise in stored values would move the
        // spectrum by 1e-6 Ha. A float64 stack IS the exact value; a float32
        // stack re-evaluates the orbitals, so the stack is only ever read to
        // build the orthonormal rows. Overlap and potential run over point
        // blocks (all orbitals of a block at once), the kinetic matrix over
        // rows (the Laplacian is a whole-row operation).
        let mut exact = ExactBlocks::new(ctx, &basis, &grid, &samples)?;
        let stride = exact.stride;
        for start in (0..g).step_by(stride) {
            let (count, phi) = exact.load(start)?;
            let ext = &external.as_slice()[start..start + count];
            for a in 0..m {
                let left = &phi[a * stride..a * stride + count];
                for b in a..m {
                    let right = &phi[b * stride..b * stride + count];
                    let mut s = overlap.as_slice()[a * m + b];
                    let mut potential = v.as_slice()[a * m + b];
                    for i in 0..count {
                        s += left[i] * right[i];
                        potential += left[i] * right[i] * ext[i];
                    }
                    overlap.as_mut_slice()[a * m + b] = s;
                    v.as_mut_slice()[a * m + b] = potential;
                }
            }
        }
        match precision {
            // A float64 stack is exact: one resident Laplacian row at a time.
            Precision::Float64 => {
                let mut row = ctx.allocate::<f64>(g)?;
                for b in 0..m {
                    samples.widen_range(b * g, row.as_mut_slice())?;
                    let lap = laplacian_of(ctx, &grid, kinetic, row.as_slice())?;
                    with_rows!(samples.rows(), |stored| {
                        for a in 0..m {
                            let mut sum = 0.;
                            for (k, &psi) in stored[a * g..(a + 1) * g].iter().enumerate() {
                                sum += psi.widen() * lap.as_slice()[k];
                            }
                            t.as_mut_slice()[a * m + b] = -0.5 * sum * grid.index.dv();
                        }
                    });
                }
            }
            Precision::Float32 => exact_kinetic(ctx, &basis, &grid, kinetic, t.as_mut_slice())?,
        }
        for a in 0..m {
            for b in a..m {
                let s = overlap.as_slice()[a * m + b] * grid.index.dv();
                let potential = v.as_slice()[a * m + b] * grid.index.dv();
                overlap.as_mut_slice()[a * m + b] = s;
                overlap.as_mut_slice()[b * m + a] = s;
                v.as_mut_slice()[a * m + b] = potential;
                v.as_mut_slice()[b * m + a] = potential;
            }
        }
        if let Some(c) = &corrections {
            for a in 0..m {
                for b in 0..m {
                    overlap.as_mut_slice()[a * m + b] += c.overlap[(a, b)];
                    v.as_mut_slice()[a * m + b] += c.potential[(a, b)];
                }
            }
        }
        let (values, vectors) = real_eigen(DMatrix::from_row_slice(m, m, overlap.as_slice()))?;
        let overlap_min = values[0];
        if overlap_min <= 1e-10 {
            return Err(invalid(format!(
                "basis is linearly dependent on this grid (overlap eigenvalue {overlap_min:e})"
            )));
        }
        let lowdin = &vectors
            * DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
                m,
                values.iter().map(|s| 1. / s.sqrt()),
            ))
            * vectors.transpose();
        if let Some(c) = corrections.as_mut() {
            if let Some(dft) = c.dft.as_mut() {
                dft.one = lowdin.transpose() * &dft.one * &lowdin;
                for factor in &mut dft.pairs.factors {
                    *factor = lowdin.transpose() * &*factor * &lowdin;
                }
            }
            for factor in &mut Arc::get_mut(&mut c.pairs)
                .ok_or_else(|| invalid("setup Coulomb corrections require unique ownership"))?
                .factors
            {
                *factor = lowdin.transpose() * &*factor * &lowdin;
            }
        }
        let h = DMatrix::from_fn(m, m, |a, b| {
            0.5 * (t.as_slice()[a * m + b]
                + v.as_slice()[a * m + b]
                + t.as_slice()[b * m + a]
                + v.as_slice()[b * m + a])
        });
        let h = lowdin.transpose() * h * &lowdin;
        let mut one = ctx.allocate(m * m)?;
        for a in 0..m {
            for b in 0..m {
                one.as_mut_slice()[a * m + b] = h[(a, b)];
            }
        }
        // Orthonormal rows by point blocks: widen a raw block to float64,
        // multiply by the Lowdin matrix there and round the block once into
        // the second stack. Only O(m B) float64 scratch exists.
        let mut orthonormal_samples = RealSamples::zeros(ctx, precision, product(&[m, g])?)?;
        {
            let stride = exact.stride;
            let mut block = ctx.allocate::<f64>(product(&[m, stride])?)?;
            let _packing = if m >= 16 {
                Some(ctx.reserve(product(&[m, 256, 16])?)?)
            } else {
                None
            };
            for start in (0..g).step_by(stride) {
                let (count, raw) = exact.load(start)?;
                if m >= 16 {
                    let input = nalgebra::DMatrixView::from_slice(raw, stride, m);
                    let mut output =
                        nalgebra::DMatrixViewMut::from_slice(block.as_mut_slice(), stride, m);
                    output.gemm(1., &input, &lowdin, 0.);
                } else {
                    block.as_mut_slice().fill(0.);
                    for p in 0..m {
                        for a in 0..m {
                            for k in 0..count {
                                block.as_mut_slice()[p * stride + k] +=
                                    lowdin[(a, p)] * raw[a * stride + k];
                            }
                        }
                    }
                }
                for p in 0..m {
                    orthonormal_samples.fill_range(
                        p * g + start,
                        &block.as_slice()[p * stride..p * stride + count],
                    )?;
                }
            }
        }
        drop(exact);
        let cholesky_tolerance = cholesky.unwrap_or(1e-10);
        if !cholesky_tolerance.is_finite() || cholesky_tolerance <= 0. {
            return Err(invalid("Cholesky tolerance must be finite and positive"));
        }
        let mut nuclear = 0.;
        for a in 0..geometry.numbers().len() {
            for b in 0..a {
                let distance = (0..3)
                    .map(|j| (geometry.positions()[a][j] - geometry.positions()[b][j]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if distance < 1e-12 {
                    return Err(invalid("nuclei must not coincide"));
                }
                let charges = |i| {
                    corrections
                        .as_ref()
                        .map_or(geometry.numbers()[i] as f64, |c| c.charges[i])
                };
                nuclear += charges(a) * charges(b) / distance;
            }
        }
        let resolution_ratios = basis
            .orbitals
            .iter()
            .enumerate()
            .map(|(a, o)| {
                t.as_slice()[a * m + a] / overlap.as_slice()[a * m + a] / o.exact_kinetic()
            })
            .collect();
        Ok(Self {
            context: ctx.clone(),
            geometry,
            grid,
            basis,
            samples,
            orthonormal_samples,
            overlap,
            kinetic: t,
            potential: v,
            lowdin,
            one,
            eri: Mutex::new(None),
            cholesky_tolerance,
            fixed_pivots: None,
            nuclear,
            kinetic_method: kinetic.into(),
            overlap_min,
            resolution_ratios,
            corrections,
            paw_setup: None,
            _matrices: matrices,
        })
    }
    /// Precision of the stored orbital stacks.
    pub fn precision(&self) -> Precision {
        self.samples.precision()
    }
    pub fn m(&self) -> usize {
        self.basis.orbitals.len()
    }
}
