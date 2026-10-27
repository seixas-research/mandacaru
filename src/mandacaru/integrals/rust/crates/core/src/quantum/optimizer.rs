//! Deterministic BFGS/L-BFGS with a bounded strong-Wolfe line search.
use crate::{product, Context, Error, Result};

#[derive(Clone, Debug)]
pub struct Options {
    pub limited_memory: bool,
    pub max_iterations: usize,
    pub max_evaluations: usize,
    pub tolerance: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            limited_memory: false,
            max_iterations: 300,
            max_evaluations: 5000,
            tolerance: 1e-9,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        if self.max_evaluations == 0 || !self.tolerance.is_finite() || self.tolerance <= 0. {
            return Err(Error::Invalid(
                "optimizer needs positive finite tolerance and evaluation budget".into(),
            ));
        }
        Ok(())
    }
}
pub struct Outcome {
    pub parameters: Vec<f64>,
    pub energy: f64,
    pub history: Vec<f64>,
    pub steps: usize,
    pub success: bool,
    pub message: String,
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
    let n = initial.len();
    let _workspace = ctx.reserve(product(&[n, 8, 12])?)?;
    let _history = ctx.reserve(product(&[options.max_evaluations, 8])?)?;
    let _corrections = ctx.reserve(if options.limited_memory {
        product(&[n, 20, 8])?
    } else {
        0
    })?;
    let mut corrections: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new();
    let mut inverse = ctx.allocate::<f64>(if options.limited_memory {
        0
    } else {
        product(&[n, n])?
    })?;
    for k in 0..if options.limited_memory { 0 } else { n } {
        inverse.as_mut_slice()[k * n + k] = 1.;
    }
    let mut x = initial.to_vec();
    let (mut energy, mut gradient) = objective(&x)?;
    validate(energy, &gradient, n)?;
    let mut history = vec![energy];
    let mut steps = 0;
    let mut message = "iteration budget exhausted".to_owned();
    loop {
        let max_gradient = gradient.iter().fold(0f64, |a, g| a.max(g.abs()));
        if max_gradient <= options.tolerance {
            return Ok(Outcome {
                parameters: x,
                energy,
                history,
                steps,
                success: true,
                message: "gradient converged".into(),
            });
        }
        if steps >= options.max_iterations {
            break;
        }
        if history.len() >= options.max_evaluations {
            message = "evaluation budget exhausted".into();
            break;
        }
        let mut direction: Vec<_> = if options.limited_memory {
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
            for k in 0..if options.limited_memory { 0 } else { n } {
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
            if value > energy + 1e-4 * alpha * slope + noise
                || (lower > 0. && value >= lower_value + noise)
            {
                upper = Some(alpha);
            } else {
                let derivative = dot(&g, &direction);
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
            message = if history.len() >= options.max_evaluations {
                "evaluation budget exhausted"
            } else {
                "line search failed"
            }
            .into();
            break;
        };
        let s: Vec<_> = next.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<_> = g.iter().zip(&gradient).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt() {
            if options.limited_memory {
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
    }
    Ok(Outcome {
        parameters: x,
        energy,
        history,
        steps,
        success: false,
        message,
    })
}
