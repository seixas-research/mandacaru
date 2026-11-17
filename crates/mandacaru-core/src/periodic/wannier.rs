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
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_iterations: 2000,
            tolerance: 1e-10,
            step: 0.5,
            gradient_tolerance: 1e-6,
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
            // The conjugate-gradient state: the previous gradient and direction, the steepest
            // direction and the current one.
            .and_then(|v| v.checked_add(product(&[mesh.points, functions, functions, 16, 4]).ok()?))
            .ok_or(crate::Error::Overflow)?,
    )?;
    ctx.run(|| {
        let start = spread(ctx, mesh, initial, reference)?;
        let mut history = vec![[start.invariant, start.off_diagonal, start.diagonal]];
        let scale = mesh.weights.as_slice().iter().map(|w| w.abs()).sum::<f64>();
        let first_step = options.step * mesh.points as f64 / (4. * scale);
        let mut steps = 0;
        let (mut gauge, mut value, mut converged) = descend(
            ctx,
            mesh,
            initial.to_vec(),
            start,
            options,
            first_step,
            &mut steps,
            &mut history,
        )?;
        // A vanishing gradient does not tell a minimum from a saddle (the spread has both, and
        // the descent can stall at one). A converged point is turned by a small deterministic
        // rotation and minimized again; a lower spread replaces it, and the check repeats on the
        // new point. A point counts as converged only when a check confirms it: a check that
        // cannot run, or whose descent runs out of `max_iterations` without going lower, leaves it
        // unconfirmed, and a lower point the budget did not let converge is returned unconverged.
        if converged {
            for attempt in 0..SADDLE_CHECKS {
                if steps >= options.max_iterations {
                    converged = false;
                    break;
                }
                let kicked = rotated_gauge(&gauge, attempt as u64, SADDLE_KICK)?;
                let kicked_value = spread(ctx, mesh, &kicked, Some(value.centers.as_slice()))?;
                let mut trial_history = vec![];
                let (trial_gauge, trial_value, trial_converged) = descend(
                    ctx,
                    mesh,
                    kicked,
                    kicked_value,
                    options,
                    first_step,
                    &mut steps,
                    &mut trial_history,
                )?;
                let gain = value.total() - trial_value.total();
                if gain <= options.tolerance.max(1e-9 * value.total().abs()) {
                    // Not lower: the point is kept, confirmed if that descent finished and
                    // unconfirmed if it ran out of budget.
                    converged = trial_converged;
                    break;
                }
                // Keep the monotone record: only the steps below the point it escaped.
                let floor = value.total();
                history.extend(
                    trial_history
                        .into_iter()
                        .filter(|h| h.iter().sum::<f64>() < floor),
                );
                gauge = trial_gauge;
                value = trial_value;
                converged = trial_converged;
                if !converged {
                    // Lower, but the budget ran out before it converged.
                    break;
                }
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

/// One conjugate-gradient descent from `gauge` (spread `value`), counting its steps in `steps`
/// against `options.max_iterations` and recording each in `history`. Converged as [`Options`]
/// says: the gradient norm below `gradient_tolerance` and the last step below `tolerance`.
#[allow(clippy::too_many_arguments)]
fn descend(
    ctx: &Context,
    mesh: &Mesh,
    mut gauge: Vec<Matrix>,
    mut value: Spread,
    options: &Options,
    first_step: f64,
    steps: &mut usize,
    history: &mut Vec<[f64; 3]>,
) -> Result<(Vec<Matrix>, Spread, bool)> {
    let mut g = gradient(ctx, mesh, &gauge, value.centers.as_slice())?.values;
    let mut change = 0f64;
    // Polak-Ribiere conjugate gradient on the product of unitary groups: the direction
    // `D = -G + beta D_previous` (generators at the current gauge), restarted on steepest descent
    // every `restart` steps, when beta is negative and when D is not a descent direction. The
    // line search works on the exact directional derivative `phi'(a) = <G(U e^{aD}), D>`, not on
    // the spread itself: near a minimum the spread's decrease sinks below its round-off long
    // before the gradient does, and a search on the spread would stop on a plateau where descent
    // continues.
    let restart = 10;
    let mut previous: Option<(Vec<Matrix>, Vec<Matrix>)> = None;
    let mut since_restart = 0;
    loop {
        let stationary = gradient_norm(&g) < options.gradient_tolerance;
        if stationary && change < options.tolerance {
            return Ok((gauge, value, true));
        }
        if *steps >= options.max_iterations {
            return Ok((gauge, value, false));
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
            accepted = line_search(ctx, mesh, &gauge, &value, &g, &direction, first_step)?;
            if accepted.is_some() || attempt == 1 {
                break;
            }
            // No step along a conjugate direction: retry on steepest descent.
            since_restart = 0;
            direction = steepest.clone();
        }
        let Some(point) = accepted else {
            // Not even a steepest-descent step lowers the spread: the gradient is at the
            // round-off of the link matrices.
            return Ok((gauge, value, stationary));
        };
        change = (point.spread.total() - value.total()).abs();
        gauge = point.gauge;
        value = point.spread;
        previous = Some((std::mem::replace(&mut g, point.gradient), direction));
        history.push([value.invariant, value.off_diagonal, value.diagonal]);
        since_restart += 1;
        *steps += 1;
    }
}

/// A gauge along a search line, with its spread and gradient.
struct LinePoint {
    gauge: Vec<Matrix>,
    spread: Spread,
    gradient: Vec<Matrix>,
    /// `phi'(a) = <G(U e^{aD}), D>`: `D` commutes with `e^{aD}`, so this is the exact
    /// derivative of the spread along the line at `a`.
    slope: f64,
}

/// A step along `direction` from `gauge` to near the first minimum of the spread on the line:
/// the derivative is followed until it changes sign (doubling the step from `first`), the bracket
/// is narrowed by secants until `|phi'(a)| <= |phi'(0)| / 10`, and every accepted point must not
/// raise the spread beyond its round-off. `None` when no such point exists.
fn line_search(
    ctx: &Context,
    mesh: &Mesh,
    gauge: &[Matrix],
    value: &Spread,
    g: &[Matrix],
    direction: &[Matrix],
    first: f64,
) -> Result<Option<LinePoint>> {
    let start = inner(g, direction);
    if start >= 0. {
        return Ok(None);
    }
    let noise = 1e-12 * value.total().abs().max(1.);
    let at = |step: f64| -> Result<LinePoint> {
        let moved = gauge
            .iter()
            .zip(direction)
            .map(|(u, d)| unitary_step(&(d * C64::new(step, 0.))).map(|e| u * e))
            .collect::<Result<Vec<_>>>()?;
        let spread = spread(ctx, mesh, &moved, Some(value.centers.as_slice()))?;
        let gradient = gradient(ctx, mesh, &moved, spread.centers.as_slice())?.values;
        let slope = inner(&gradient, direction);
        Ok(LinePoint {
            gauge: moved,
            spread,
            gradient,
            slope,
        })
    };
    let downhill = |p: &LinePoint| p.spread.total() <= value.total() + noise;
    // Bracket the first sign change of the derivative: `lo` still descends, `hi` does not (its
    // derivative is nonnegative, or it is uphill: the step overshot).
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
            // An uphill end says nothing about the derivative's root: shrink fast toward `lo`.
            // Otherwise the secant of the derivative, kept inside the bracket.
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
