//! Wannier gauge localization from periodic-part overlaps on a k-point mesh.
//! Atomic units. Neighbor links include the boundary sewing phases and PAW
//! metric before entering this solver; no position-operator proxy is used.
use super::invalid;
use crate::{
    math::{
        linalg::eigen,
        storage::{MatrixSource, StoredMatrix},
    },
    product,
    quantum::C64,
    Buffer, Context, Reservation, Result,
};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;
type Matrix = DMatrix<C64>;
pub mod disentangle;
pub mod model;
pub mod sampling;
pub mod setup;

pub struct Mesh {
    pub points: usize,
    pub bands: usize,
    pub vectors: Buffer<[f64; 3]>,
    pub weights: Buffer<f64>,
    pub neighbors: Buffer<usize>,
    pub links: Vec<StoredMatrix>,
}
impl Mesh {
    pub fn validate(&self, ctx: &Context) -> Result<()> {
        let count = self.vectors.as_slice().len();
        if self.points == 0
            || self.bands == 0
            || count == 0
            || self.weights.as_slice().len() != count
            || self.neighbors.as_slice().len() != product(&[self.points, count])?
            || self.links.len() != product(&[self.points, count])?
            || self.neighbors.as_slice().iter().any(|k| *k >= self.points)
            || self
                .vectors
                .as_slice()
                .iter()
                .flatten()
                .chain(self.weights.as_slice())
                .any(|v| !v.is_finite())
        {
            return Err(invalid(
                "invalid Wannier neighbor mesh dimensions or values",
            ));
        }
        for a in 0..3 {
            for b in 0..3 {
                let moment = self
                    .weights
                    .as_slice()
                    .iter()
                    .zip(self.vectors.as_slice())
                    .map(|(w, v)| w * v[a] * v[b])
                    .sum::<f64>();
                if (moment - if a == b { 1. } else { 0. }).abs() > 1e-8 {
                    return Err(invalid(
                        "Wannier neighbor weights do not satisfy Cartesian completeness",
                    ));
                }
            }
        }
        for link in &self.links {
            let link = link.matrix(ctx)?;
            if link.shape() != (self.bands, self.bands)
                || link.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
            {
                return Err(invalid("Wannier links must match the Bloch band space"));
            }
        }
        Ok(())
    }
    fn rotated(&self, ctx: &Context, gauge: &[Matrix], k: usize, b: usize) -> Result<Matrix> {
        let index = k * self.vectors.as_slice().len() + b;
        let link = self.links[index].matrix(ctx)?;
        Ok(gauge[k].adjoint() * &*link * &gauge[self.neighbors.as_slice()[index]])
    }
}

pub struct Spread {
    pub centers: Buffer<[f64; 3]>,
    pub second_moments: Buffer<f64>,
    pub invariant: f64,
    pub off_diagonal: f64,
    pub diagonal: f64,
}
impl Spread {
    pub fn total(&self) -> f64 {
        self.invariant + self.off_diagonal + self.diagonal
    }
}
fn phase(value: C64, vector: [f64; 3], reference: Option<[f64; 3]>) -> f64 {
    let angle = value.arg();
    match reference {
        Some(center) => {
            angle
                - 2. * PI
                    * ((angle + (0..3).map(|a| vector[a] * center[a]).sum::<f64>()) / (2. * PI))
                        .round()
        }
        None => angle,
    }
}
fn validate_gauge(mesh: &Mesh, gauge: &[Matrix]) -> Result<usize> {
    let functions = gauge.first().map_or(0, |g| g.ncols());
    if gauge.len() != mesh.points
        || functions == 0
        || functions > mesh.bands
        || gauge.iter().any(|g| {
            g.shape() != (mesh.bands, functions)
                || g.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
        })
    {
        return Err(invalid("invalid Wannier gauge dimensions or values"));
    }
    for g in gauge {
        let gram = g.adjoint() * g;
        if (gram - Matrix::identity(functions, functions)).norm() > 1e-8 {
            return Err(invalid(
                "Wannier gauge columns must be orthonormal in band space",
            ));
        }
    }
    Ok(functions)
}
pub fn spread(
    ctx: &Context,
    mesh: &Mesh,
    gauge: &[Matrix],
    reference: Option<&[[f64; 3]]>,
) -> Result<Spread> {
    let functions = validate_gauge(mesh, gauge)?;
    if reference.is_some_and(|r| r.len() != functions || r.iter().flatten().any(|v| !v.is_finite()))
    {
        return Err(invalid(
            "Wannier phase references must match the requested centers",
        ));
    }
    let _work = ctx.reserve(product(&[mesh.bands, mesh.bands, 16, 4])?)?;
    let mut centers = ctx.allocate::<[f64; 3]>(functions)?;
    let mut second = ctx.allocate::<f64>(functions)?;
    let mut invariant = 0.;
    let mut off = 0.;
    let count = mesh.vectors.as_slice().len();
    for k in 0..mesh.points {
        for b in 0..count {
            let m = mesh.rotated(ctx, gauge, k, b)?;
            let weight = mesh.weights.as_slice()[b] / mesh.points as f64;
            invariant += weight * (functions as f64 - m.norm_squared());
            for i in 0..functions {
                let angle = phase(
                    m[(i, i)],
                    mesh.vectors.as_slice()[b],
                    reference.map(|r| r[i]),
                );
                for a in 0..3 {
                    centers.as_mut_slice()[i][a] -= weight * mesh.vectors.as_slice()[b][a] * angle;
                }
                second.as_mut_slice()[i] += weight * (1. - m[(i, i)].norm_sqr() + angle * angle);
                for j in 0..functions {
                    if i != j {
                        off += weight * m[(i, j)].norm_sqr();
                    }
                }
            }
        }
    }
    // The center-dependent term differentiates to zero at these centers.
    let mut diagonal = 0.;
    for k in 0..mesh.points {
        for b in 0..count {
            let m = mesh.rotated(ctx, gauge, k, b)?;
            for i in 0..functions {
                let angle = phase(
                    m[(i, i)],
                    mesh.vectors.as_slice()[b],
                    reference.map(|r| r[i]),
                );
                let q = angle
                    + (0..3)
                        .map(|a| mesh.vectors.as_slice()[b][a] * centers.as_slice()[i][a])
                        .sum::<f64>();
                diagonal += mesh.weights.as_slice()[b] * q * q / mesh.points as f64;
            }
        }
    }
    Ok(Spread {
        centers,
        second_moments: second,
        invariant,
        off_diagonal: off,
        diagonal,
    })
}

/// Riemannian derivative for U(k) -> U(k) exp(W(k)), W anti-Hermitian.
/// Both endpoints of every directed link participate, so no implicit pairing
/// or symmetry of the supplied neighbor list is required.
pub fn gradient(
    ctx: &Context,
    mesh: &Mesh,
    gauge: &[Matrix],
    centers: &[[f64; 3]],
) -> Result<GaugeGradient> {
    let functions = validate_gauge(mesh, gauge)?;
    if centers.len() != functions || centers.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("Wannier gradient needs matching centers"));
    }
    let memory = ctx.reserve(product(&[mesh.points, functions, functions, 16])?)?;
    let _work = ctx.reserve(product(&[mesh.bands, mesh.bands, 16, 8])?)?;
    let mut result = vec![Matrix::zeros(functions, functions); mesh.points];
    for k in 0..mesh.points {
        for b in 0..mesh.vectors.as_slice().len() {
            let neighbor = mesh.neighbors.as_slice()[k * mesh.vectors.as_slice().len() + b];
            let m = mesh.rotated(ctx, gauge, k, b)?;
            let weight = 2. * mesh.weights.as_slice()[b] / mesh.points as f64;
            let mut derivative = Matrix::zeros(functions, functions);
            for i in 0..functions {
                let value = m[(i, i)];
                if value.norm_sqr() < 1e-28 {
                    return Err(invalid(
                        "zero Wannier link diagonal has an undefined phase derivative",
                    ));
                }
                let q = phase(value, mesh.vectors.as_slice()[b], Some(centers[i]))
                    + (0..3)
                        .map(|a| mesh.vectors.as_slice()[b][a] * centers[i][a])
                        .sum::<f64>();
                derivative[(i, i)] = -value + C64::i() * q / value.conj();
            }
            result[k] -= &derivative * m.adjoint() * C64::new(weight, 0.);
            result[neighbor] += m.adjoint() * derivative * C64::new(weight, 0.);
        }
    }
    for matrix in &mut result {
        *matrix = (&*matrix - matrix.adjoint()) * C64::new(0.5, 0.);
    }
    Ok(GaugeGradient {
        values: result,
        _memory: memory,
    })
}
pub struct GaugeGradient {
    pub values: Vec<Matrix>,
    _memory: Reservation,
}
pub fn unitary_step(generator: &Matrix) -> Result<Matrix> {
    if generator.nrows() != generator.ncols() || (generator + generator.adjoint()).norm() > 1e-9 {
        return Err(invalid("Wannier gauge step must be anti-Hermitian"));
    }
    let (values, vectors) = eigen(generator * C64::i())?;
    Ok(&vectors
        * Matrix::from_diagonal(&DVector::from_iterator(
            values.len(),
            values.into_iter().map(|v| C64::from_polar(1., -v)),
        ))
        * vectors.adjoint())
}
pub struct Options {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub step: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_iterations: 2000,
            tolerance: 1e-10,
            step: 0.5,
        }
    }
}
pub struct Localized {
    pub gauge: Vec<Matrix>,
    pub spread: Spread,
    pub history: Vec<[f64; 3]>,
    pub converged: bool,
    _memory: Reservation,
}
pub fn localize(
    ctx: &Context,
    mesh: &Mesh,
    initial: &[Matrix],
    reference: Option<&[[f64; 3]]>,
    options: &Options,
) -> Result<Localized> {
    mesh.validate(ctx)?;
    let functions = validate_gauge(mesh, initial)?;
    if options.max_iterations == 0
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.
        || !options.step.is_finite()
        || options.step <= 0.
    {
        return Err(invalid("invalid Wannier localization options"));
    }
    let retained = product(&[mesh.points, mesh.bands, functions, 16])?
        .checked_add(product(&[
            options
                .max_iterations
                .checked_add(1)
                .ok_or(crate::Error::Overflow)?,
            24,
        ])?)
        .ok_or(crate::Error::Overflow)?;
    let mut memory = ctx.reserve(retained)?;
    let _work = ctx.reserve(
        product(&[mesh.points, mesh.bands, functions, 16, 3])?
            .checked_add(product(&[mesh.bands, mesh.bands, 16, 12])?)
            .ok_or(crate::Error::Overflow)?,
    )?;
    ctx.run(|| {
        let mut gauge = initial.to_vec();
        let mut value = spread(ctx, mesh, &gauge, reference)?;
        let mut history = vec![[value.invariant, value.off_diagonal, value.diagonal]];
        let scale = mesh.weights.as_slice().iter().map(|w| w.abs()).sum::<f64>();
        let mut converged = false;
        for _ in 0..options.max_iterations {
            let derivative = gradient(ctx, mesh, &gauge, value.centers.as_slice())?;
            let norm = derivative
                .values
                .iter()
                .map(|g| g.norm_squared())
                .sum::<f64>();
            if norm < 1e-24 {
                converged = true;
                break;
            }
            let mut step = options.step * mesh.points as f64 / (4. * scale);
            let mut next = None;
            for _ in 0..30 {
                let proposed = gauge
                    .iter()
                    .zip(&derivative.values)
                    .map(|(g, d)| unitary_step(&(d * C64::new(-step, 0.))).map(|u| g * u))
                    .collect::<Result<Vec<_>>>()?;
                let spread = spread(ctx, mesh, &proposed, Some(value.centers.as_slice()))?;
                if spread.total() <= value.total() - 1e-4 * step * norm {
                    next = Some((proposed, spread));
                    break;
                }
                step *= 0.5;
            }
            let Some((proposed, spread)) = next else {
                break;
            };
            let change = (spread.total() - value.total()).abs();
            gauge = proposed;
            value = spread;
            history.push([value.invariant, value.off_diagonal, value.diagonal]);
            if change < options.tolerance {
                converged = true;
                break;
            }
        }
        memory.shrink_to(
            product(&[mesh.points, mesh.bands, functions, 16])?
                .checked_add(product(&[history.len(), 24])?)
                .ok_or(crate::Error::Overflow)?,
        )?;
        Ok(Localized {
            gauge,
            spread: value,
            history,
            converged,
            _memory: memory,
        })
    })
}
