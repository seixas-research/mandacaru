//! Native optimizers for unconstrained variational parameters.
mod conjugate;
mod derivative_free;
mod sequential;
mod stochastic;
use crate::{product, Context, Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Bfgs,
    Lbfgs,
    Slsqp,
    Cg,
    Cobyla,
    NelderMead,
    Spsa,
}
impl Method {
    pub const NAMES: [&'static str; 7] = [
        "SLSQP",
        "BFGS",
        "L-BFGS",
        "CG",
        "COBYLA",
        "Nelder-Mead",
        "SPSA",
    ];
    /// The method's name as `NAMES` and [`Self::parse`] spell it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Slsqp => "SLSQP",
            Self::Bfgs => "BFGS",
            Self::Lbfgs => "L-BFGS",
            Self::Cg => "CG",
            Self::Cobyla => "COBYLA",
            Self::NelderMead => "Nelder-Mead",
            Self::Spsa => "SPSA",
        }
    }
    /// Whether the method keeps a dense curvature estimate that [`minimize_from`] can start
    /// from and [`Outcome::hessian`] returns: SLSQP (a Hessian approximation) and BFGS (an
    /// inverse Hessian approximation). L-BFGS keeps ten correction pairs that cover a growing
    /// ansatz poorly (measured: no gain), so it starts cold.
    pub fn carries_curvature(self) -> bool {
        matches!(self, Self::Slsqp | Self::Bfgs)
    }
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "BFGS" => Ok(Self::Bfgs),
            "L-BFGS" => Ok(Self::Lbfgs),
            "SLSQP" => Ok(Self::Slsqp),
            "CG" => Ok(Self::Cg),
            "COBYLA" => Ok(Self::Cobyla),
            "Nelder-Mead" => Ok(Self::NelderMead),
            "SPSA" => Ok(Self::Spsa),
            _ => Err(Error::Invalid(format!("unknown optimizer {name:?}"))),
        }
    }
}

/// Convergence tolerance of every method when `Options::tolerance` is `None`
/// (SPSA is the exception: `None` disables its convergence test).
pub const DEFAULT_TOLERANCE: f64 = 1e-9;

#[derive(Clone, Debug)]
pub struct Options {
    pub method: Method,
    pub max_iterations: usize,
    pub max_evaluations: usize,
    /// `None` means the method default ([`DEFAULT_TOLERANCE`]); for SPSA it
    /// disables the five-calm-steps test so the run uses its whole iteration
    /// budget.
    pub tolerance: Option<f64>,
    pub seed: u64,
    /// SPSA only. `true` (default) evaluates the objective at every updated
    /// point (three evaluations per iteration); `false` is the strict
    /// two-evaluation form used on hardware. Other methods ignore it.
    pub track_best: bool,
    pub patience: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            method: Method::Bfgs,
            max_iterations: 300,
            max_evaluations: 5000,
            tolerance: Some(DEFAULT_TOLERANCE),
            seed: 0,
            track_best: true,
            patience: 5,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        if self.max_evaluations == 0 || self.tolerance.is_some_and(|t| !t.is_finite() || t <= 0.) {
            return Err(Error::Invalid(
                "optimizer needs positive finite tolerance and evaluation budget".into(),
            ));
        }
        Ok(())
    }
    /// The tolerance a method that always tests convergence uses.
    fn tolerance_or_default(&self) -> f64 {
        self.tolerance.unwrap_or(DEFAULT_TOLERANCE)
    }
    /// Most objective evaluations the method can request, or `None` when an
    /// objective call is not one energy evaluation (gradient methods).
    ///
    /// SPSA is bounded by its iteration count; the derivative-free simplex
    /// methods only by `max_evaluations`, because a Nelder-Mead shrink or a
    /// COBYLA geometry refresh spends several evaluations in one iteration.
    pub fn evaluation_bound(&self) -> Option<usize> {
        match self.method {
            Method::Spsa => Some(stochastic::evaluation_bound(self)),
            Method::Cobyla | Method::NelderMead => Some(self.max_evaluations),
            Method::Bfgs | Method::Lbfgs | Method::Slsqp | Method::Cg => None,
        }
    }
}
pub struct Outcome {
    pub parameters: Vec<f64>,
    pub energy: f64,
    pub history: Vec<f64>,
    pub steps: usize,
    pub success: bool,
    pub message: String,
    /// Set by [`minimize_interruptible`] when the objective ended the run;
    /// `parameters`, `energy` and `history` are then the best point reached.
    pub interrupted: Option<String>,
    /// The final curvature estimate of the ansatz parameters, row-major `n x n`, for the two
    /// quasi-Newton methods that keep a dense one: SLSQP its Hessian approximation, BFGS its
    /// INVERSE Hessian approximation (the two are not interchangeable, see
    /// [`Method::carries_curvature`]); `None` for every other method and for an interrupted
    /// run. A caller that re-optimizes the same parameters after growing them hands it back
    /// through [`minimize_from`] instead of starting from the identity.
    pub hessian: Option<Vec<f64>>,
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn validate(energy: f64, gradient: &[f64], n: usize) -> Result<()> {
    if !energy.is_finite() || gradient.len() != n || gradient.iter().any(|g| !g.is_finite()) {
        Err(Error::Kernel(
            "objective returned nonfinite or inconsistent energy/gradient".into(),
        ))
    } else {
        Ok(())
    }
}

/// One evaluation of the objective: its energy, and its gradient when the optimizer asks for it.
///
/// An optimizer's line search tests a trial point on its energy, and reads the gradient only at a
/// point it accepts (SLSQP) or keeps testing (conjugate gradients, the Wolfe tests of BFGS and
/// L-BFGS); a derivative-free method never reads it. A deferred gradient is computed from the
/// state the energy left behind, by the very code an eager evaluation runs, and a rejected
/// point's is never computed. The `history` of an [`Outcome`] still holds one energy per
/// evaluation: an entry no longer implies that a gradient was computed.
pub struct Probe<'p> {
    pub energy: f64,
    gradient: Gradient<'p>,
}
enum Gradient<'p> {
    Ready(Vec<f64>),
    Deferred(Box<dyn FnOnce() -> Result<Vec<f64>> + 'p>),
}
impl<'p> Probe<'p> {
    /// An evaluation whose gradient is already known.
    pub fn ready(energy: f64, gradient: Vec<f64>) -> Self {
        Self {
            energy,
            gradient: Gradient::Ready(gradient),
        }
    }
    /// An evaluation whose gradient `gradient` computes on demand (at most once).
    pub fn deferred(energy: f64, gradient: impl FnOnce() -> Result<Vec<f64>> + 'p) -> Self {
        Self {
            energy,
            gradient: Gradient::Deferred(Box::new(gradient)),
        }
    }
    /// The same evaluation with `map` applied to its gradient (a block of the parameters).
    pub fn map_gradient(self, map: impl FnOnce(Vec<f64>) -> Vec<f64> + 'p) -> Self {
        let energy = self.energy;
        Self {
            energy,
            gradient: match self.gradient {
                Gradient::Ready(g) => Gradient::Ready(map(g)),
                Gradient::Deferred(compute) => {
                    Gradient::Deferred(Box::new(move || compute().map(map)))
                }
            },
        }
    }
    /// The gradient, computed now if it was deferred.
    pub fn into_gradient(self) -> Result<Vec<f64>> {
        match self.gradient {
            Gradient::Ready(g) => Ok(g),
            Gradient::Deferred(compute) => compute(),
        }
    }
    /// The energy, which must be finite.
    fn checked_energy(&self) -> Result<f64> {
        validate(self.energy, &[], 0)?;
        Ok(self.energy)
    }
    /// The gradient of `n` parameters, which must be finite.
    fn checked_gradient(self, n: usize) -> Result<Vec<f64>> {
        let gradient = self.into_gradient()?;
        validate(0., &gradient, n)?;
        Ok(gradient)
    }
}
/// A closure that returns the energy and the gradient together, as a [`Probe`] whose gradient is
/// ready. It is checked at every call, whether or not the method reads the gradient, as it
/// always was.
fn eager<'p>(
    mut objective: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
) -> impl FnMut(&[f64]) -> Result<Probe<'p>> {
    move |x| {
        let (energy, gradient) = objective(x)?;
        validate(energy, &gradient, x.len())?;
        Ok(Probe::ready(energy, gradient))
    }
}
/// A trial point's [`Probe`] whose gradient a line search reads at most when a test needs it, and
/// then once: the first read computes it, later reads and the final hand-over reuse it.
struct Pending<'p> {
    probe: Option<Probe<'p>>,
    gradient: Option<Vec<f64>>,
    n: usize,
}
impl<'p> Pending<'p> {
    fn new(probe: Probe<'p>, n: usize) -> Self {
        Self {
            probe: Some(probe),
            gradient: None,
            n,
        }
    }
    fn gradient(&mut self) -> Result<&[f64]> {
        if self.gradient.is_none() {
            let probe = self.probe.take().expect("a pending probe holds its gradient");
            self.gradient = Some(probe.checked_gradient(self.n)?);
        }
        Ok(self.gradient.as_deref().expect("just computed"))
    }
    fn into_gradient(mut self) -> Result<Vec<f64>> {
        self.gradient()?;
        Ok(self.gradient.take().expect("just computed"))
    }
}
/// Like [`minimize`], but an objective that fails with
/// [`Error::Interrupted`] ends the run cleanly: the lowest energy among the
/// completed evaluations is returned with `success == false`, `interrupted`
/// set and `steps == 0` (the loop was cut short). When no evaluation
/// completed there is no best point and the error propagates.
pub fn minimize_interruptible(
    ctx: &Context,
    initial: &[f64],
    options: &Options,
    objective: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
) -> Result<Outcome> {
    minimize_interruptible_from(ctx, initial, options, None, objective)
}
/// [`minimize_interruptible`] starting from a Hessian approximation (see [`minimize_from`]).
pub fn minimize_interruptible_from(
    ctx: &Context,
    initial: &[f64],
    options: &Options,
    hessian: Option<&[f64]>,
    objective: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
) -> Result<Outcome> {
    minimize_interruptible_deferred_from(ctx, initial, options, hessian, eager(objective))
}
/// [`minimize_interruptible_from`] for an objective that defers its gradient ([`Probe`]).
pub fn minimize_interruptible_deferred_from<'p>(
    ctx: &Context,
    initial: &[f64],
    options: &Options,
    hessian: Option<&[f64]>,
    mut objective: impl FnMut(&[f64]) -> Result<Probe<'p>>,
) -> Result<Outcome> {
    let mut best: Option<(Vec<f64>, f64)> = None;
    let mut history = Vec::new();
    let result = minimize_deferred_from(ctx, initial, options, hessian, |x| {
        let probe = objective(x)?;
        if probe.energy.is_finite() {
            history.push(probe.energy);
            if best.as_ref().is_none_or(|(_, e)| probe.energy < *e) {
                best = Some((x.to_vec(), probe.energy));
            }
        }
        Ok(probe)
    });
    match (result, best) {
        (Err(Error::Interrupted(message)), Some((parameters, energy))) => Ok(Outcome {
            parameters,
            energy,
            history,
            steps: 0,
            success: false,
            message: format!("run budget exhausted: {message}"),
            hessian: None,
            interrupted: Some(message),
        }),
        (result, _) => result,
    }
}
pub fn minimize(
    ctx: &Context,
    initial: &[f64],
    options: &Options,
    objective: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
) -> Result<Outcome> {
    minimize_from(ctx, initial, options, None, objective)
}
/// [`minimize`] from a given curvature estimate, row-major `n x n` with `n = initial.len()` (an
/// [`Outcome::hessian`] of the same method, extended by the caller for any parameter it
/// appended). Only SLSQP (Hessian) and BFGS (inverse Hessian) keep one: another method given
/// one raises rather than ignoring it.
pub fn minimize_from(
    ctx: &Context,
    initial: &[f64],
    options: &Options,
    hessian: Option<&[f64]>,
    objective: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
) -> Result<Outcome> {
    minimize_deferred_from(ctx, initial, options, hessian, eager(objective))
}
/// [`minimize_from`] for an objective that defers its gradient ([`Probe`]): the one body every
/// closure form runs. The gradient of a point is computed only where the method reads it.
pub fn minimize_deferred_from<'p>(
    ctx: &Context,
    initial: &[f64],
    options: &Options,
    hessian: Option<&[f64]>,
    mut objective: impl FnMut(&[f64]) -> Result<Probe<'p>>,
) -> Result<Outcome> {
    options.validate()?;
    if initial.iter().any(|t| !t.is_finite()) {
        return Err(Error::Invalid("initial parameters must be finite".into()));
    }
    if hessian.is_some() && !options.method.carries_curvature() {
        return Err(Error::Invalid(
            "only SLSQP and BFGS carry a curvature estimate between optimizations".into(),
        ));
    }
    match options.method {
        Method::Slsqp => return sequential::minimize(ctx, initial, options, hessian, objective),
        Method::Cg => return conjugate::minimize(ctx, initial, options, objective),
        Method::NelderMead | Method::Cobyla => {
            return derivative_free::minimize(ctx, initial, options, objective);
        }
        Method::Spsa => return stochastic::minimize(ctx, initial, options, objective),
        Method::Bfgs | Method::Lbfgs => {}
    }
    let limited_memory = options.method == Method::Lbfgs;
    let n = initial.len();
    let _workspace = ctx.reserve(product(&[n, 8, 12])?)?;
    let _history = ctx.reserve(product(&[options.max_evaluations, 8])?)?;
    let _corrections = ctx.reserve(if limited_memory {
        product(&[n, 20, 8])?
    } else {
        0
    })?;
    let mut corrections: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new();
    let mut inverse = ctx.allocate::<f64>(if limited_memory { 0 } else { product(&[n, n])? })?;
    match hessian {
        Some(values) => {
            if values.len() != n * n || values.iter().any(|v| !v.is_finite()) {
                return Err(Error::Invalid(
                    "the carried inverse Hessian must be finite and n x n for the n parameters".into(),
                ));
            }
            inverse.as_mut_slice().copy_from_slice(values);
        }
        None => {
            for k in 0..if limited_memory { 0 } else { n } {
                inverse.as_mut_slice()[k * n + k] = 1.;
            }
        }
    }
    let mut x = initial.to_vec();
    let first = objective(&x)?;
    let mut energy = first.checked_energy()?;
    let mut gradient = first.checked_gradient(n)?;
    let mut history = vec![energy];
    let mut steps = 0;
    let mut reset_after_failure = false;
    let mut message = "iteration budget exhausted".to_owned();
    loop {
        let max_gradient = gradient.iter().fold(0f64, |a, g| a.max(g.abs()));
        if max_gradient <= options.tolerance_or_default() {
            return Ok(Outcome {
                parameters: x,
                energy,
                history,
                steps,
                success: true,
                message: "gradient converged".into(),
                hessian: (!limited_memory).then(|| inverse.as_slice().to_vec()),
                interrupted: None,
            });
        }
        if steps >= options.max_iterations {
            break;
        }
        if history.len() >= options.max_evaluations {
            message = "evaluation budget exhausted".into();
            break;
        }
        let mut direction: Vec<_> = if limited_memory {
            let mut q = gradient.clone();
            let mut alphas = Vec::new();
            for (s, y, sy) in corrections.iter().rev() {
                let alpha = dot(s, &q) / sy;
                alphas.push(alpha);
                for (q, y) in q.iter_mut().zip(y) {
                    *q -= alpha * y;
                }
            }
            if let Some((_, y, sy)) = corrections.last() {
                let scale = sy / dot(y, y);
                for q in &mut q {
                    *q *= scale;
                }
            }
            for ((s, y, sy), alpha) in corrections.iter().zip(alphas.into_iter().rev()) {
                let beta = dot(y, &q) / sy;
                for (q, s) in q.iter_mut().zip(s) {
                    *q += s * (alpha - beta);
                }
            }
            q.into_iter().map(|q| -q).collect()
        } else {
            (0..n)
                .map(|i| -dot(&inverse.as_slice()[i * n..(i + 1) * n], &gradient))
                .collect()
        };
        if dot(&direction, &gradient) >= -1e-14 * dot(&gradient, &gradient) {
            direction = gradient.iter().map(|g| -g).collect();
            inverse.as_mut_slice().fill(0.);
            corrections.clear();
            for k in 0..if limited_memory { 0 } else { n } {
                inverse.as_mut_slice()[k * n + k] = 1.;
            }
        }
        let slope = dot(&direction, &gradient);
        let mut alpha = 1.;
        let mut accepted = None;
        let mut lower = 0.;
        let mut upper = None;
        let mut lower_value = energy;
        for _ in 0..60 {
            if history.len() >= options.max_evaluations {
                break;
            }
            let trial: Vec<_> = x
                .iter()
                .zip(&direction)
                .map(|(x, p)| x + alpha * p)
                .collect();
            let probe = objective(&trial)?;
            let value = probe.checked_energy()?;
            history.push(value);
            let noise = 4. * f64::EPSILON * energy.abs().max(1.);
            // The gradient is computed where a test first reads it, and not for a trial that the
            // energy alone sends to `upper`.
            let mut pending = Pending::new(probe, n);
            // Approximate Wolfe (Hager–Zhang, 2006, eqs. 23/27): near
            // stationarity, energy subtraction loses digits before the
            // analytic directional derivatives do. This accepts a step,
            // never a convergence certificate; the requested gradient
            // tolerance remains the sole successful stopping test.
            if value <= energy + 1e-12 * energy.abs().max(1.) {
                let derivative = dot(pending.gradient()?, &direction);
                if derivative >= 0.9 * slope && derivative <= (2e-4 - 1.) * slope {
                    accepted = Some((trial, value, pending.into_gradient()?));
                    break;
                }
            }
            if value > energy + 1e-4 * alpha * slope + noise
                || (lower > 0. && value >= lower_value + noise)
            {
                upper = Some(alpha);
            } else {
                let derivative = dot(pending.gradient()?, &direction);
                if derivative.abs() <= -0.9 * slope {
                    accepted = Some((trial, value, pending.into_gradient()?));
                    break;
                }
                if derivative >= 0. {
                    upper = Some(alpha);
                } else {
                    lower = alpha;
                    lower_value = value;
                }
            }
            let next_alpha = upper.map_or(alpha * 2., |high| 0.5 * (lower + high));
            if next_alpha == alpha || !next_alpha.is_finite() {
                break;
            }
            alpha = next_alpha;
        }
        let Some((next, value, g)) = accepted else {
            if !reset_after_failure && history.len() < options.max_evaluations {
                inverse.as_mut_slice().fill(0.);
                for k in 0..if limited_memory { 0 } else { n } {
                    inverse.as_mut_slice()[k * n + k] = 1.;
                }
                corrections.clear();
                reset_after_failure = true;
                continue;
            }
            message = if history.len() >= options.max_evaluations {
                "evaluation budget exhausted".into()
            } else {
                format!(
                    "line search failed (max gradient {:.3e})",
                    gradient.iter().map(|g| g.abs()).fold(0., f64::max)
                )
            };
            break;
        };
        let s: Vec<_> = next.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<_> = g.iter().zip(&gradient).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt() {
            if limited_memory {
                if corrections.len() == 10 {
                    corrections.remove(0);
                }
                corrections.push((s, y, sy));
            } else {
                let hy: Vec<_> = (0..n)
                    .map(|i| dot(&inverse.as_slice()[i * n..(i + 1) * n], &y))
                    .collect();
                let factor = (1. + dot(&y, &hy) / sy) / sy;
                for i in 0..n {
                    for j in 0..n {
                        inverse.as_mut_slice()[i * n + j] +=
                            factor * s[i] * s[j] - (hy[i] * s[j] + s[i] * hy[j]) / sy;
                    }
                }
            }
        }
        x = next;
        energy = value;
        gradient = g;
        steps += 1;
        reset_after_failure = false;
    }
    Ok(Outcome {
        parameters: x,
        energy,
        history,
        steps,
        success: false,
        message,
        hessian: (!limited_memory).then(|| inverse.as_slice().to_vec()),
        interrupted: None,
    })
}
