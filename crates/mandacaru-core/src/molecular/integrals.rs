use super::{invalid, SpatialGrid};
use crate::basis::Basis;
use crate::math::fft::{spectral_laplacian, Poisson};
use crate::math::linalg::real_eigen;
use crate::quantum::C64;
use crate::{product, Buffer, Context, Geometry, Reservation, Result};
use nalgebra::DMatrix;
use std::sync::{Arc, Mutex};

pub fn pair(a: usize, b: usize) -> usize {
    let (a, b) = (a.max(b), a.min(b));
    a * (a + 1) / 2 + b
}
fn packed_len(n: usize) -> Result<usize> {
    n.checked_mul(n.checked_add(1).ok_or(crate::Error::Overflow)?)
        .map(|v| v / 2)
        .ok_or(crate::Error::Overflow)
}
pub enum Eri {
    Packed {
        m: usize,
        values: Buffer<f64>,
    },
    Cholesky {
        m: usize,
        rank: usize,
        factors: Buffer<f64>,
        residual: f64,
    },
}
impl Eri {
    pub fn m(&self) -> usize {
        match self {
            Self::Packed { m, .. } | Self::Cholesky { m, .. } => *m,
        }
    }
    pub fn value(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let a = pair(p, r);
        let b = pair(q, s);
        match self {
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
        loop {
            let pivot = (0..n)
                .max_by(|&a, &b| residual.as_slice()[a].total_cmp(&residual.as_slice()[b]))
                .unwrap();
            let diagonal = residual.as_slice()[pivot];
            if diagonal <= tolerance || rank == n {
                break;
            }
            let (p, r) = indices[pivot];
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
            residual: error,
        })
    }
}
pub struct MolecularIntegrals {
    pub context: Context,
    pub geometry: Geometry,
    pub grid: SpatialGrid,
    pub basis: Basis,
    pub samples: Buffer<f64>,
    pub orthonormal_samples: Buffer<f64>,
    pub overlap: Buffer<f64>,
    pub kinetic: Buffer<f64>,
    pub potential: Buffer<f64>,
    pub lowdin: DMatrix<f64>,
    pub one: Buffer<f64>,
    eri: Mutex<Option<Arc<Eri>>>,
    cholesky_tolerance: f64,
    pub nuclear: f64,
    pub kinetic_method: String,
    pub overlap_min: f64,
    pub resolution_ratios: Vec<f64>,
    _matrices: Reservation,
}
pub fn pair_integrals(ctx: &Context, grid: &SpatialGrid, samples: &[f64], m: usize) -> Result<Eri> {
    pair_integrals_with_solver(ctx, grid, samples, m, &mut Poisson::isolated(ctx, grid)?)
}
/// Pivoted Cholesky using individual Coulomb columns. Neither the pair matrix
/// nor a four-index tensor is allocated, including during factorization.
pub(super) fn cholesky_from_grid(
    ctx: &Context,
    grid: &SpatialGrid,
    samples: &[f64],
    m: usize,
    tolerance: f64,
    solver: &mut Poisson,
) -> Result<Eri> {
    if m == 0
        || samples.len() != product(&[m, grid.index.points()])?
        || !tolerance.is_finite()
        || tolerance <= 0.
    {
        return Err(invalid("invalid grid Cholesky dimensions or tolerance"));
    }
    let n = packed_len(m)?;
    let g = grid.index.points();
    let mut diagonal = ctx.allocate::<f64>(n)?;
    let mut rho = ctx.allocate::<C64>(g)?;
    let mut potential = ctx.allocate::<C64>(g)?;
    let _index_memory = ctx.reserve(product(&[n, 32])?)?;
    let indices: Vec<_> = (0..m).flat_map(|a| (0..=a).map(move |b| (a, b))).collect();
    for (i, &(a, b)) in indices.iter().enumerate() {
        for k in 0..g {
            rho.as_mut_slice()[k] = C64::new(samples[a * g + k] * samples[b * g + k], 0.);
        }
        solver.solve_into(rho.as_slice(), potential.as_mut_slice())?;
        diagonal.as_mut_slice()[i] = rho
            .as_slice()
            .iter()
            .zip(potential.as_slice())
            .map(|(r, v)| (r * v).re)
            .sum::<f64>()
            * grid.index.dv();
    }
    let mut columns: Vec<Buffer<f64>> = Vec::new();
    loop {
        let pivot = (0..n)
            .max_by(|&a, &b| diagonal.as_slice()[a].total_cmp(&diagonal.as_slice()[b]))
            .unwrap();
        let maximum = diagonal.as_slice()[pivot];
        if maximum <= tolerance || columns.len() == n {
            break;
        }
        let (a, b) = indices[pivot];
        for k in 0..g {
            rho.as_mut_slice()[k] = C64::new(samples[a * g + k] * samples[b * g + k], 0.);
        }
        solver.solve_into(rho.as_slice(), potential.as_mut_slice())?;
        let mut column = ctx.allocate::<f64>(n)?;
        for (i, &(p, q)) in indices.iter().enumerate() {
            let integral = (0..g)
                .map(|k| samples[p * g + k] * samples[q * g + k] * potential.as_slice()[k].re)
                .sum::<f64>()
                * grid.index.dv();
            let previous = columns
                .iter()
                .map(|c| c.as_slice()[i] * c.as_slice()[pivot])
                .sum::<f64>();
            let value = (integral - previous) / maximum.sqrt();
            column.as_mut_slice()[i] = value;
            diagonal.as_mut_slice()[i] -= value * value;
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
    let residual = diagonal.as_slice().iter().copied().fold(0f64, f64::max);
    let rank = columns.len();
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
    })
}
pub(super) fn pair_integrals_with_solver(
    ctx: &Context,
    grid: &SpatialGrid,
    samples: &[f64],
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
                density.as_mut_slice()[k] = C64::new(samples[b * g + k] * samples[d * g + k], 0.);
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
                            samples[a * g + k] * samples[c * g + k] * potential.as_slice()[k].re
                        })
                        .sum();
                    values.as_mut_slice()[pair(ac, bd)] = integral * grid.index.dv();
                }
            }
        }
    }
    Ok(Eri::Packed { m, values })
}
impl MolecularIntegrals {
    pub fn has_eri(&self) -> bool {
        self.eri.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }
    /// Build Coulomb factors only when a solver or Hamiltonian asks for them.
    /// Failed allocations do not poison the cache and can be retried.
    pub fn eri(&self) -> Result<Arc<Eri>> {
        let mut cache = self.eri.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(value) = cache.as_ref() {
            return Ok(value.clone());
        }
        let mut solver = Poisson::isolated(&self.context, &self.grid)?;
        let factors = Arc::new(cholesky_from_grid(
            &self.context,
            &self.grid,
            self.orthonormal_samples.as_slice(),
            self.m(),
            self.cholesky_tolerance,
            &mut solver,
        )?);
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
            self.geometry
                .numbers()
                .iter()
                .map(|&z| z as i128)
                .sum::<i128>()
                - charge as i128
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
    pub fn new(
        ctx: &Context,
        geometry: Geometry,
        grid: SpatialGrid,
        basis_spec: &str,
        kinetic: &str,
        cholesky: Option<f64>,
    ) -> Result<Self> {
        if grid.periodic {
            return Err(invalid("molecular integrals require an isolated grid; periodic electrostatics is a separate native primitive"));
        }
        if kinetic != "spectral" && kinetic != "fd" {
            return Err(invalid("kinetic must be spectral or fd"));
        }
        let basis = Basis::build(ctx, &geometry, basis_spec)?;
        let m = basis.orbitals.len();
        let g = grid.index.points();
        let matrices = ctx.reserve(product(&[m, m, 8, 12])?)?;
        let samples = basis.sample(ctx, &grid)?;
        let mut overlap = ctx.allocate::<f64>(product(&[m, m])?)?;
        let mut t = ctx.allocate::<f64>(product(&[m, m])?)?;
        let mut v = ctx.allocate::<f64>(product(&[m, m])?)?;
        let mut external = ctx.allocate::<f64>(g)?;
        let softening = 0.5 * grid.spacings().into_iter().fold(f64::INFINITY, f64::min);
        for k in 0..g {
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
        for b in 0..m {
            let row = &samples.as_slice()[b * g..(b + 1) * g];
            let lap = if kinetic == "spectral" {
                spectral_laplacian(ctx, &grid, row)?
            } else {
                crate::kernels::laplacian(ctx, &grid.index, 1, row)?
            };
            for a in 0..m {
                let mut s = 0.;
                let mut kinetic = 0.;
                let mut potential = 0.;
                for (k, &psi) in samples.as_slice()[a * g..(a + 1) * g].iter().enumerate() {
                    s += psi * row[k];
                    kinetic += psi * lap.as_slice()[k];
                    potential += psi * row[k] * external.as_slice()[k];
                }
                overlap.as_mut_slice()[a * m + b] = s * grid.index.dv();
                t.as_mut_slice()[a * m + b] = -0.5 * kinetic * grid.index.dv();
                v.as_mut_slice()[a * m + b] = potential * grid.index.dv();
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
        let mut orthonormal_samples = ctx.allocate::<f64>(product(&[m, g])?)?;
        for p in 0..m {
            for a in 0..m {
                for k in 0..g {
                    orthonormal_samples.as_mut_slice()[p * g + k] +=
                        lowdin[(a, p)] * samples.as_slice()[a * g + k];
                }
            }
        }
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
                nuclear += geometry.numbers()[a] as f64 * geometry.numbers()[b] as f64 / distance;
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
            nuclear,
            kinetic_method: kinetic.into(),
            overlap_min,
            resolution_ratios,
            _matrices: matrices,
        })
    }
    pub fn m(&self) -> usize {
        self.basis.orbitals.len()
    }
}
