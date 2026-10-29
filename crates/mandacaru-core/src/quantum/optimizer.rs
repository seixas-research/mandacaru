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
/// Like [`minimize`], but an objective that fails with
/// [`Error::Interrupted`] ends the run cleanly: the lowest energy among the
/// completed evaluations is returned with `success == false`, `interrupted`
/// set and `steps == 0` (the loop was cut short). When no evaluation
/// completed there is no best point and the error propagates.
pub fn minimize_interruptible(
    ctx: &Context,
    initial: &[f64],
    options: &Options,
    mut objective: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
) -> Result<Outcome> {
    let mut best: Option<(Vec<f64>, f64)> = None;
    let mut history = Vec::new();
    let result = minimize(ctx, initial, options, |x| {
        let value = objective(x)?;
        if value.0.is_finite() {
            history.push(value.0);
            if best.as_ref().is_none_or(|(_, e)| value.0 < *e) {
                best = Some((x.to_vec(), value.0));
            }
        }
        Ok(value)
    });
    match (result, best) {
        (Err(Error::Interrupted(message)), Some((parameters, energy))) => Ok(Outcome {
            parameters,
            energy,
            history,
            steps: 0,
            success: false,
            message: format!("run budget exhausted: {message}"),
            interrupted: Some(message),
        }),
        (result, _) => result,
    }
}
pub fn minimize(
    ctx: &Context,
    initial: &[f64],
    options: &Options,
    mut objective: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
) -> Result<Outcome> {
    options.validate()?;
    if initial.iter().any(|t| !t.is_finite()) {
        return Err(Error::Invalid("initial parameters must be finite".into()));
    }
    match options.method {
        Method::Slsqp => return sequential::minimize(ctx, initial, options, objective),
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
    for k in 0..if limited_memory { 0 } else { n } {
        inverse.as_mut_slice()[k * n + k] = 1.;
    }
    let mut x = initial.to_vec();
    let (mut energy, mut gradient) = objective(&x)?;
    validate(energy, &gradient, n)?;
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
            let (value, g) = objective(&trial)?;
            validate(value, &g, n)?;
            history.push(value);
            let noise = 4. * f64::EPSILON * energy.abs().max(1.);
            let derivative = dot(&g, &direction);
            // Approximate Wolfe (Hager–Zhang, 2006, eqs. 23/27): near
            // stationarity, energy subtraction loses digits before the
            // analytic directional derivatives do. This accepts a step,
            // never a convergence certificate; the requested gradient
            // tolerance remains the sole successful stopping test.
            if value <= energy + 1e-12 * energy.abs().max(1.)
                && derivative >= 0.9 * slope
                && derivative <= (2e-4 - 1.) * slope
            {
                accepted = Some((trial, value, g));
                break;
            }
            if value > energy + 1e-4 * alpha * slope + noise
                || (lower > 0. && value >= lower_value + noise)
            {
                upper = Some(alpha);
            } else {
                if derivative.abs() <= -0.9 * slope {
                    accepted = Some((trial, value, g));
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
        interrupted: None,
    })
}
