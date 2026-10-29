//! Simplex search and unconstrained COBYLA linear-model trust regions.
use super::*;
use nalgebra::{DMatrix, DVector};

struct Sample {
    x: Vec<f64>,
    value: f64,
}
fn evaluate(
    x: Vec<f64>,
    options: &Options,
    history: &mut Vec<f64>,
    objective: &mut impl FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
) -> Result<Option<Sample>> {
    if history.len() >= options.max_evaluations {
        return Ok(None);
    }
    let (value, g) = objective(&x)?;
    validate(value, &g, x.len())?;
    history.push(value);
    Ok(Some(Sample { x, value }))
}
pub(super) fn minimize(
    ctx: &Context,
    initial: &[f64],
    options: &Options,
    mut objective: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
) -> Result<Outcome> {
    let n = initial.len();
    let _work = ctx.reserve(product(&[
        n.checked_add(1).ok_or(Error::Overflow)?,
        n.max(1),
        8,
        12,
    ])?)?;
    let _history = ctx.reserve(product(&[options.max_evaluations, 8])?)?;
    let mut history = Vec::new();
    let first = evaluate(initial.to_vec(), options, &mut history, &mut objective)?.unwrap();
    if n == 0 || options.max_iterations == 0 {
        return Ok(Outcome {
            parameters: first.x,
            energy: first.value,
            history,
            steps: 0,
            success: n == 0,
            message: if n == 0 {
                "empty parameter vector"
            } else {
                "iteration budget exhausted"
            }
            .into(),
        });
    }
    let cobyla = options.method == Method::Cobyla;
    let mut radius = 0.5;
    let mut simplex = vec![first];
    for i in 0..n {
        let mut x = initial.to_vec();
        x[i] += if cobyla {
            radius
        } else if x[i] == 0. {
            0.00025
        } else {
            0.05 * x[i].abs()
        };
        if let Some(point) = evaluate(x, options, &mut history, &mut objective)? {
            simplex.push(point);
        } else {
            break;
        }
    }
    let mut steps = 0;
    let mut success = false;
    let mut failures = 0;
    let mut message = "iteration budget exhausted".to_owned();
    while simplex.len() == n + 1 && steps < options.max_iterations {
        simplex.sort_by(|a, b| a.value.total_cmp(&b.value));
        if history.len() >= options.max_evaluations {
            message = "evaluation budget exhausted".into();
            break;
        }
        if cobyla {
            if radius <= options.tolerance {
                success = true;
                message = "linear-model trust region converged".into();
                break;
            }
            let mut distances: Vec<_> = simplex[1..]
                .iter()
                .map(|s| {
                    s.x.iter()
                        .zip(&simplex[0].x)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            let mut matrix = DMatrix::from_fn(n, n, |i, j| simplex[i + 1].x[j] - simplex[0].x[j]);
            let mut delta =
                DVector::from_iterator(n, simplex[1..].iter().map(|s| s.value - simplex[0].value));
            let mut model = matrix.clone().lu().solve(&delta);
            if model.is_none()
                || distances
                    .iter()
                    .any(|d| *d < 0.1 * radius || *d > 4. * radius)
            {
                let center = simplex[0].x.clone();
                for i in 0..n {
                    let mut x = center.clone();
                    x[i] += radius;
                    if let Some(point) = evaluate(x, options, &mut history, &mut objective)? {
                        simplex[i + 1] = point;
                    } else {
                        break;
                    }
                }
                if history.len() >= options.max_evaluations {
                    message = "evaluation budget exhausted".into();
                    break;
                }
                matrix = DMatrix::from_fn(n, n, |i, j| simplex[i + 1].x[j] - simplex[0].x[j]);
                delta = DVector::from_iterator(
                    n,
                    simplex[1..].iter().map(|s| s.value - simplex[0].value),
                );
                model = matrix.clone().lu().solve(&delta);
                distances.clear();
            }
            let Some(model) = model else {
                radius *= 0.5;
                continue;
            };
            let norm = model.norm();
            if !norm.is_finite() {
                return Err(Error::Kernel("nonfinite COBYLA linear model".into()));
            }
            if norm <= f64::MIN_POSITIVE {
                radius *= 0.5;
                continue;
            }
            let step = -model * (radius / norm);
            let x: Vec<_> = simplex[0]
                .x
                .iter()
                .zip(step.iter())
                .map(|(x, d)| x + d)
                .collect();
            let Some(trial) = evaluate(x, options, &mut history, &mut objective)? else {
                message = "evaluation budget exhausted".into();
                break;
            };
            let improvement = simplex[0].value - trial.value;
            let coefficients = matrix.transpose().lu().solve(&step);
            let replace = if improvement > 0. {
                n
            } else if let Some(c) = coefficients {
                (0..n)
                    .max_by(|&a, &b| c[a].abs().total_cmp(&c[b].abs()))
                    .unwrap()
                    + 1
            } else {
                n
            };
            simplex[replace] = trial;
            steps += 1;
            if improvement > 0. {
                failures = 0;
            } else {
                failures += 1;
            }
            if failures > n {
                radius *= 0.5;
                failures = 0;
            }
        } else {
            let spread = simplex[1..]
                .iter()
                .flat_map(|s| s.x.iter().zip(&simplex[0].x).map(|(a, b)| (a - b).abs()))
                .fold(0_f64, f64::max);
            if spread <= options.tolerance
                && (simplex[n].value - simplex[0].value).abs() <= options.tolerance
            {
                success = true;
                message = "simplex converged".into();
                break;
            }
            let center: Vec<_> = (0..n)
                .map(|j| simplex[..n].iter().map(|s| s.x[j]).sum::<f64>() / n as f64)
                .collect();
            let trial = |factor: f64| {
                center
                    .iter()
                    .zip(&simplex[n].x)
                    .map(|(a, b)| a + factor * (a - b))
                    .collect::<Vec<_>>()
            };
            let Some(reflected) = evaluate(trial(1.), options, &mut history, &mut objective)?
            else {
                message = "evaluation budget exhausted".into();
                break;
            };
            if reflected.value < simplex[0].value {
                let expanded = evaluate(trial(2.), options, &mut history, &mut objective)?;
                simplex[n] = expanded
                    .filter(|s| s.value < reflected.value)
                    .unwrap_or(reflected);
            } else if reflected.value < simplex[n - 1].value {
                simplex[n] = reflected;
            } else {
                let outside = reflected.value < simplex[n].value;
                let contracted = evaluate(
                    trial(if outside { 0.5 } else { -0.5 }),
                    options,
                    &mut history,
                    &mut objective,
                )?;
                if contracted.as_ref().is_some_and(|c| {
                    c.value
                        < if outside {
                            reflected.value
                        } else {
                            simplex[n].value
                        }
                }) {
                    simplex[n] = contracted.unwrap();
                } else {
                    let best = simplex[0].x.clone();
                    for point in simplex.iter_mut().skip(1) {
                        let x = point
                            .x
                            .iter()
                            .zip(&best)
                            .map(|(x, b)| 0.5 * (x + b))
                            .collect();
                        if let Some(new) = evaluate(x, options, &mut history, &mut objective)? {
                            *point = new;
                        } else {
                            break;
                        }
                    }
                }
            }
            steps += 1;
        }
    }
    simplex.sort_by(|a, b| a.value.total_cmp(&b.value));
    if !success && history.len() >= options.max_evaluations {
        message = "evaluation budget exhausted".into();
    }
    let best = simplex.remove(0);
    Ok(Outcome {
        parameters: best.x,
        energy: best.value,
        history,
        steps,
        success,
        message,
    })
}
