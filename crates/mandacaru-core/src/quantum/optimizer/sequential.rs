//! Zero-constraint SLSQP: least-squares quadratic steps and damped BFGS Hessians.
use super::*;
use nalgebra::{DMatrix, DVector};

pub(super) fn minimize<'p>(
    ctx: &Context,
    initial: &[f64],
    options: &Options,
    start: Option<&[f64]>,
    mut objective: impl FnMut(&[f64]) -> Result<Probe<'p>>,
) -> Result<Outcome> {
    let n = initial.len();
    let _work = ctx.reserve(
        product(&[n, n, 8, 5])?
            .checked_add(product(&[n, 8, 16])?)
            .ok_or(Error::Overflow)?,
    )?;
    let _history = ctx.reserve(product(&[options.max_evaluations, 8])?)?;
    let mut x = initial.to_vec();
    let first = objective(&x)?;
    let mut energy = first.checked_energy()?;
    let mut gradient = first.checked_gradient(n)?;
    let mut history = vec![energy];
    let mut hessian = match start {
        None => DMatrix::<f64>::identity(n, n),
        Some(values) => {
            if values.len() != n * n || values.iter().any(|v| !v.is_finite()) {
                return Err(Error::Invalid(
                    "the carried Hessian must be finite and n x n for the n parameters".into(),
                ));
            }
            DMatrix::from_row_slice(n, n, values)
        }
    };
    let mut steps = 0;
    let mut success = false;
    let mut message = "iteration budget exhausted".to_owned();
    loop {
        if gradient.iter().fold(0_f64, |a, g| a.max(g.abs())) <= options.tolerance_or_default() {
            success = true;
            message = "stationarity converged".into();
            break;
        }
        if steps >= options.max_iterations {
            break;
        }
        if history.len() >= options.max_evaluations {
            message = "evaluation budget exhausted".into();
            break;
        }
        let factor = match hessian.clone().cholesky() {
            Some(f) => f,
            None => {
                hessian = DMatrix::identity(n, n);
                hessian.clone().cholesky().unwrap()
            }
        };
        let direction = factor.solve(&DVector::from_iterator(n, gradient.iter().map(|v| -v)));
        let slope = dot(direction.as_slice(), &gradient);
        let mut alpha = 1.;
        let mut accepted = None;
        for _ in 0..60 {
            if history.len() >= options.max_evaluations {
                break;
            }
            let next: Vec<_> = x
                .iter()
                .zip(direction.iter())
                .map(|(x, p)| x + alpha * p)
                .collect();
            let probe = objective(&next)?;
            let value = probe.checked_energy()?;
            history.push(value);
            let roundoff = 4. * f64::EPSILON * energy.abs().max(1.);
            // The test reads the energy alone: only an accepted point's gradient is computed, and
            // a rejected trial's state is dropped here, before the next one is prepared.
            if value <= energy + 0.1 * alpha * slope + roundoff {
                accepted = Some((next, value, probe.checked_gradient(n)?));
                break;
            }
            alpha *= 0.5;
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
        let s = DVector::from_iterator(n, next.iter().zip(&x).map(|(a, b)| a - b));
        let y = DVector::from_iterator(n, g.iter().zip(&gradient).map(|(a, b)| a - b));
        let bs = &hessian * &s;
        let sbs = s.dot(&bs);
        let sy = s.dot(&y);
        if sbs > f64::MIN_POSITIVE {
            let theta = if sy >= 0.2 * sbs {
                1.
            } else {
                0.8 * sbs / (sbs - sy)
            };
            let r = y * theta + &bs * (1. - theta);
            let sr = s.dot(&r);
            if sr > f64::MIN_POSITIVE {
                hessian += (&r * r.transpose()) / sr - (&bs * bs.transpose()) / sbs;
            }
        }
        steps += 1;
        let small = (value - energy).abs() <= options.tolerance_or_default()
            && s.amax() <= options.tolerance_or_default().sqrt();
        x = next;
        energy = value;
        gradient = g;
        if small {
            success = true;
            message = "function and step converged".into();
            break;
        }
    }
    Ok(Outcome {
        parameters: x,
        energy,
        history,
        steps,
        success,
        message,
        hessian: Some(
            (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .map(|index| hessian[index])
                .collect(),
        ),
        interrupted: None,
    })
}
