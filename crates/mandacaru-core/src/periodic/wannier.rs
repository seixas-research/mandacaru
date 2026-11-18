//! Wannier gauge localization from periodic-part overlaps on a k-point mesh.
//! Atomic units. Neighbor links include the boundary sewing phases and PAW
//! metric before entering this solver; no position-operator proxy is used.
use super::{
    berry::{links, LINK_TOLERANCE},
    bloch_mesh::MeshIndex,
    invalid,
};
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
pub mod downfold;
pub mod interpolation;
pub mod model;
pub mod populations;
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
    /// Each function's spread `<r^2> - <r>^2`.
    pub fn spreads(&self) -> Vec<f64> {
        self.second_moments
            .as_slice()
            .iter()
            .zip(self.centers.as_slice())
            .map(|(second, r)| second - r.iter().map(|v| v * v).sum::<f64>())
            .collect()
    }
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
    let count = mesh.vectors.as_slice().len();
    let _work = ctx.reserve(
        product(&[mesh.bands, mesh.bands, 16, 4])?
            .checked_add(product(&[mesh.points, count, functions, 8])?)
            .ok_or(crate::Error::Overflow)?,
    )?;
    let mut centers = ctx.allocate::<[f64; 3]>(functions)?;
    let mut second = ctx.allocate::<f64>(functions)?;
    let mut invariant = 0.;
    let mut off = 0.;
    // One pass over the links: the branch-chosen phases of the diagonal are kept for the
    // center-dependent term, so every stored link is read once per evaluation.
    let mut angles = Vec::with_capacity(product(&[mesh.points, count, functions])?);
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
                angles.push(angle);
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
            for i in 0..functions {
                let angle = angles[(k * count + b) * functions + i];
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
/// Objective minimized during periodic Wannier orbital localization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Localization {
    /// Marzari-Vanderbilt quadratic spread `Omega = Omega_I + Omega_OD + Omega_D`.
    #[default]
    Spread,
    /// Pipek-Mezey atomic-population localization maximizing `sum_n sum_{A,R} (q^{A,R}_{nn})^p`.
    Populations {
        scheme: populations::PopulationScheme,
        exponent: usize,
    },
}

/// The localization options. The minimization is converged, and stops, when the gradient norm
/// [`gradient_norm`] is below `gradient_tolerance` (Bohr^2) AND the last step changed the spread by
/// less than `tolerance` (Bohr^2), at a point that a small rotation and a new descent do not
/// lower (a saddle check, [`localize`]); it also stops, unconverged unless the first two hold,
/// after `max_iterations` steps in all or when no step lowers the spread. `step` scales the first
/// trial step `step N / (4 sum_b |w_b|)` of every line search.
pub struct Options {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub step: f64,
    pub gradient_tolerance: f64,
    pub localization: Localization,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_iterations: 2000,
            tolerance: 1e-10,
            step: 0.5,
            gradient_tolerance: 1e-6,
            localization: Localization::Spread,
        }
    }
}

/// `sqrt(N sum_k ||G_k||^2)`: the size of the gauge gradient, independent of the mesh size `N`
/// (every `G_k` carries the `1/N` of the mesh average).
pub fn gradient_norm(gradient: &[Matrix]) -> f64 {
    (gradient.len() as f64 * gradient.iter().map(|g| g.norm_squared()).sum::<f64>()).sqrt()
}

fn inner(a: &[Matrix], b: &[Matrix]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.adjoint() * y).trace().re)
        .sum()
}
pub struct Localized {
    pub gauge: Vec<Matrix>,
    pub spread: Spread,
    pub history: Vec<[f64; 3]>,
    pub converged: bool,
    pub objective: f64,
    _memory: Reservation,
}

struct ObjectiveSample {
    value: f64,
    record: [f64; 3],
    gradient: Vec<Matrix>,
    spread: Option<Spread>,
}

trait GaugeObjective {
    fn sample(
        &self,
        ctx: &Context,
        gauge: &[Matrix],
        reference: Option<&[[f64; 3]]>,
    ) -> Result<ObjectiveSample>;
}

struct SpreadObjective<'a>(&'a Mesh);
impl GaugeObjective for SpreadObjective<'_> {
    fn sample(
        &self,
        ctx: &Context,
        gauge: &[Matrix],
        reference: Option<&[[f64; 3]]>,
    ) -> Result<ObjectiveSample> {
        let s = spread(ctx, self.0, gauge, reference)?;
        let g = gradient(ctx, self.0, gauge, s.centers.as_slice())?.values;
        Ok(ObjectiveSample {
            value: s.total(),
            record: [s.invariant, s.off_diagonal, s.diagonal],
            gradient: g,
            spread: Some(s),
        })
    }
}

struct PopulationObjective<'a>(&'a populations::PopulationMesh);
impl GaugeObjective for PopulationObjective<'_> {
    fn sample(
        &self,
        ctx: &Context,
        gauge: &[Matrix],
        _reference: Option<&[[f64; 3]]>,
    ) -> Result<ObjectiveSample> {
        let (neg_pop, g) = self.0.evaluate(ctx, gauge)?;
        Ok(ObjectiveSample {
            value: neg_pop,
            record: [-neg_pop, 0., 0.],
            gradient: g,
            spread: None,
        })
    }
}

pub fn localize(
    ctx: &Context,
    mesh: &Mesh,
    initial: &[Matrix],
    reference: Option<&[[f64; 3]]>,
    options: &Options,
) -> Result<Localized> {
    localize_with_populations(ctx, mesh, None, initial, reference, options)
}

/// General localization driver shared across spread (`populations = None`) and Pipek-Mezey
/// atomic-population (`populations = Some(...)`) objectives.
pub fn localize_with_populations(
    ctx: &Context,
    mesh: &Mesh,
    populations: Option<&populations::PopulationMesh>,
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
        || !options.gradient_tolerance.is_finite()
        || options.gradient_tolerance <= 0.
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
            .and_then(|v| v.checked_add(product(&[mesh.points, functions, functions, 16, 4]).ok()?))
            .ok_or(crate::Error::Overflow)?,
    )?;
    ctx.run(|| {
        let scale = mesh.weights.as_slice().iter().map(|w| w.abs()).sum::<f64>();
        let spread_obj = SpreadObjective(mesh);
        let pop_obj = populations.map(PopulationObjective);
        let (objective, first_step): (&dyn GaugeObjective, f64) = match &options.localization {
            Localization::Spread => (
                &spread_obj,
                options.step * mesh.points as f64 / (4. * scale),
            ),
            Localization::Populations { .. } => {
                let p = pop_obj.as_ref().ok_or_else(|| {
                    invalid("population localization requires a prepared PopulationMesh")
                })?;
                (p, options.step * 0.5)
            }
        };
        let start = objective.sample(ctx, initial, reference)?;
        let mut history = vec![start.record];
        let mut steps = 0;
        let (mut gauge, mut current, mut converged) = descend(
            ctx,
            objective,
            initial.to_vec(),
            start,
            options,
            first_step,
            &mut steps,
            &mut history,
        )?;
        if converged {
            for attempt in 0..SADDLE_CHECKS {
                if steps >= options.max_iterations {
                    converged = false;
                    break;
                }
                let kicked = rotated_gauge(&gauge, attempt as u64, SADDLE_KICK)?;
                let kick_ref = current.spread.as_ref().map(|s| s.centers.as_slice());
                let kicked_value = objective.sample(ctx, &kicked, kick_ref)?;
                let mut trial_history = vec![];
                let (trial_gauge, trial_value, trial_converged) = descend(
                    ctx,
                    objective,
                    kicked,
                    kicked_value,
                    options,
                    first_step,
                    &mut steps,
                    &mut trial_history,
                )?;
                let gain = current.value - trial_value.value;
                if gain <= options.tolerance.max(1e-9 * current.value.abs()) {
                    converged = trial_converged;
                    break;
                }
                let floor = current.value;
                history.extend(trial_history.into_iter().filter(|h| {
                    let val = match options.localization {
                        Localization::Spread => h.iter().sum::<f64>(),
                        Localization::Populations { .. } => -h[0],
                    };
                    val < floor
                }));
                gauge = trial_gauge;
                current = trial_value;
                converged = trial_converged;
                if !converged {
                    break;
                }
            }
        }
        let final_spread = match current.spread {
            Some(s) => s,
            None => spread(ctx, mesh, &gauge, reference)?,
        };
        memory.shrink_to(
            product(&[mesh.points, mesh.bands, functions, 16])?
                .checked_add(product(&[history.len(), 24])?)
                .ok_or(crate::Error::Overflow)?,
        )?;
        Ok(Localized {
            gauge,
            spread: final_spread,
            history,
            converged,
            objective: match options.localization {
                Localization::Spread => current.value,
                Localization::Populations { .. } => -current.value,
            },
            _memory: memory,
        })
    })
}

/// Saddle checks after convergence ([`localize`]): successive escapes, not independent probes.
/// Each starts from the point the previous one reached; the last point of an escape that used
/// them all is reported as that escape's descent left it.
const SADDLE_CHECKS: usize = 3;
/// The size `|W_k|` of the rotation that tests a converged point for a saddle. It bounds what a
/// check can see: a direction of negative curvature weaker than about 6e-3 Bohr^2/rad^2 lowers
/// the spread by less than the default `tolerance` over a rotation of this size, and goes
/// undetected.
const SADDLE_KICK: f64 = 1e-2;

/// `U_k e^{W_k}` with `W_k` anti-Hermitian of Frobenius norm `size`, from a fixed generator seeded
/// by `seed` and `k` (the same on every processor count and layout).
fn rotated_gauge(gauge: &[Matrix], seed: u64, size: f64) -> Result<Vec<Matrix>> {
    gauge
        .iter()
        .enumerate()
        .map(|(k, u)| {
            let n = u.ncols();
            let mut state = 0x9e37_79b9_7f4a_7c15u64
                ^ (seed.wrapping_mul(0x2545_f491_4f6c_dd1d))
                ^ ((k as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9));
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            };
            let a = Matrix::from_fn(n, n, |_, _| C64::new(next(), next()));
            let w = (&a - a.adjoint()) * C64::new(0.5, 0.);
            let norm = w.norm();
            if norm == 0. {
                return Ok(u.clone());
            }
            Ok(u * unitary_step(&(w * C64::new(size / norm, 0.)))?)
        })
        .collect()
}

/// One conjugate-gradient descent from `gauge`, counting its steps in `steps` against
/// `options.max_iterations` and recording each in `history`.
#[allow(clippy::too_many_arguments)]
fn descend(
    ctx: &Context,
    objective: &dyn GaugeObjective,
    mut gauge: Vec<Matrix>,
    mut current: ObjectiveSample,
    options: &Options,
    first_step: f64,
    steps: &mut usize,
    history: &mut Vec<[f64; 3]>,
) -> Result<(Vec<Matrix>, ObjectiveSample, bool)> {
    let mut g = std::mem::take(&mut current.gradient);
    let mut change = 0f64;
    let restart = 10;
    let mut previous: Option<(Vec<Matrix>, Vec<Matrix>)> = None;
    let mut since_restart = 0;
    loop {
        let stationary = gradient_norm(&g) < options.gradient_tolerance;
        if stationary && change < options.tolerance {
            current.gradient = g;
            return Ok((gauge, current, true));
        }
        if *steps >= options.max_iterations {
            current.gradient = g;
            return Ok((gauge, current, false));
        }
        let steepest: Vec<Matrix> = g.iter().map(|x| -x).collect();
        let mut direction = match &previous {
            Some((old_gradient, old_direction)) if since_restart < restart => {
                let beta =
                    (inner(&g, &g) - inner(&g, old_gradient)) / inner(old_gradient, old_gradient);
                if beta > 0. {
                    steepest
                        .iter()
                        .zip(old_direction)
                        .map(|(s, d)| s + d * C64::new(beta, 0.))
                        .collect()
                } else {
                    since_restart = 0;
                    steepest.clone()
                }
            }
            _ => {
                since_restart = 0;
                steepest.clone()
            }
        };
        if inner(&g, &direction) >= 0. {
            since_restart = 0;
            direction = steepest.clone();
        }
        let mut accepted = None;
        for attempt in 0..2 {
            accepted = line_search(ctx, objective, &gauge, &current, &g, &direction, first_step)?;
            if accepted.is_some() || attempt == 1 {
                break;
            }
            since_restart = 0;
            direction = steepest.clone();
        }
        let Some(mut point) = accepted else {
            current.gradient = g;
            return Ok((gauge, current, stationary));
        };
        change = (point.sample.value - current.value).abs();
        let next_g = std::mem::take(&mut point.sample.gradient);
        gauge = point.gauge;
        current = point.sample;
        previous = Some((std::mem::replace(&mut g, next_g), direction));
        history.push(current.record);
        since_restart += 1;
        *steps += 1;
    }
}

/// A gauge along a search line, with its sampled objective and gradient.
struct LinePoint {
    gauge: Vec<Matrix>,
    sample: ObjectiveSample,
    /// `phi'(a) = <G(U e^{aD}), D>`.
    slope: f64,
}

fn line_search(
    ctx: &Context,
    objective: &dyn GaugeObjective,
    gauge: &[Matrix],
    current: &ObjectiveSample,
    g: &[Matrix],
    direction: &[Matrix],
    first: f64,
) -> Result<Option<LinePoint>> {
    let start = inner(g, direction);
    if start >= 0. {
        return Ok(None);
    }
    let noise = 1e-12 * current.value.abs().max(1.);
    let ref_centers = current.spread.as_ref().map(|s| s.centers.as_slice());
    let at = |step: f64| -> Result<LinePoint> {
        let moved = gauge
            .iter()
            .zip(direction)
            .map(|(u, d)| unitary_step(&(d * C64::new(step, 0.))).map(|e| u * e))
            .collect::<Result<Vec<_>>>()?;
        let sample = objective.sample(ctx, &moved, ref_centers)?;
        let slope = inner(&sample.gradient, direction);
        Ok(LinePoint {
            gauge: moved,
            sample,
            slope,
        })
    };
    let downhill = |p: &LinePoint| p.sample.value <= current.value + noise;
    let (mut lo, mut lo_slope, mut best): (f64, f64, Option<(f64, LinePoint)>) = (0., start, None);
    let mut hi: Option<(f64, f64, bool)> = None;
    let mut step = first;
    for _ in 0..12 {
        let point = at(step)?;
        if !downhill(&point) {
            hi = Some((step, point.slope.max(0.), true));
            break;
        }
        if point.slope >= 0. {
            hi = Some((step, point.slope, false));
            best = Some((step, point));
            break;
        }
        lo = step;
        lo_slope = point.slope;
        best = Some((step, point));
        step *= 2.;
    }
    if let Some((mut high, mut high_slope, mut uphill)) = hi {
        for _ in 0..30 {
            if best
                .as_ref()
                .is_some_and(|(_, p)| p.slope.abs() <= 0.1 * start.abs())
            {
                break;
            }
            let span = high - lo;
            if span <= 1e-12 * first {
                break;
            }
            let trial = if uphill {
                lo + 0.25 * span
            } else {
                let secant = lo - lo_slope * span / (high_slope - lo_slope);
                secant.clamp(lo + 0.1 * span, high - 0.1 * span)
            };
            let point = at(trial)?;
            if !downhill(&point) {
                high = trial;
                high_slope = point.slope.max(0.);
                uphill = true;
            } else if point.slope >= 0. {
                high = trial;
                high_slope = point.slope;
                uphill = false;
                if best
                    .as_ref()
                    .is_none_or(|(_, b)| point.slope.abs() < b.slope.abs())
                {
                    best = Some((trial, point));
                }
            } else {
                lo = trial;
                lo_slope = point.slope;
                best = Some((trial, point));
            }
        }
    }
    Ok(best.map(|(_, point)| point))
}

/// Refuses a band subspace that carries a Chern number: a subspace with `C != 0` on some lattice
/// plane has no exponentially localized Wannier functions (Thouless 1984; Brouder et al. 2007),
/// so its spread grows without bound as the mesh is refined. `along[k][j]` is the subspace link
/// from mesh point `k` to the next point along `G_j` (`None` for a direction of one point). The
/// Chern number of every plane with at least two points along both of its directions is taken on
/// every layer by the Berry plaquettes (`berry::links::lattice_field`).
pub fn chern_guard(mesh: &MeshIndex, along: &[[Option<Matrix>; 3]]) -> Result<()> {
    if along.len() != mesh.points() {
        return Err(invalid("the Chern guard needs links at every mesh point"));
    }
    for normal in 0..3 {
        let [a, b] = [(normal + 1) % 3, (normal + 2) % 3];
        let (na, nb) = (mesh.size[a], mesh.size[b]);
        if na < 2 || nb < 2 {
            continue;
        }
        for layer in 0..mesh.size[normal] {
            let mut along_a = Vec::with_capacity(na * nb);
            let mut along_b = Vec::with_capacity(na * nb);
            for i in 0..na {
                for j in 0..nb {
                    let mut at = [0; 3];
                    at[a] = i;
                    at[b] = j;
                    at[normal] = layer;
                    let k = mesh.index(at);
                    for (direction, out) in [(a, &mut along_a), (b, &mut along_b)] {
                        let link = along[k][direction]
                            .as_ref()
                            .ok_or_else(|| invalid("the Chern guard is missing a link"))?;
                        out.push(links::link_phase(link, LINK_TOLERANCE)?.0);
                    }
                }
            }
            let field = links::lattice_field(&along_a, &along_b, [na, nb])?;
            if field.chern != 0 {
                return Err(invalid(format!(
                    "the Wannier subspace has Chern number {} on the k plane normal to G_{normal} \
                     (layer {layer} of {}): it has no exponentially localized Wannier functions; \
                     use get_chern_number or the Wannier charge-center flow instead",
                    field.chern, mesh.size[normal]
                )));
            }
        }
    }
    Ok(())
}
