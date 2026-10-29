//! Polak–Ribiere+ nonlinear conjugate gradients with strong-Wolfe search.
use super::*;
pub(super) fn minimize(
    ctx: &Context,
    initial: &[f64],
    options: &Options,
    mut objective: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
) -> Result<Outcome> {
    let n = initial.len();
    let _work = ctx.reserve(product(&[n, 8, 16])?)?;
    let _history = ctx.reserve(product(&[options.max_evaluations, 8])?)?;
    let mut x = initial.to_vec();
    let (mut energy, mut gradient) = objective(&x)?;
    validate(energy, &gradient, n)?;
    let mut direction: Vec<_> = gradient.iter().map(|g| -g).collect();
    let mut history = vec![energy];
    let mut steps = 0;
    let mut success = false;
    let mut message = "iteration budget exhausted".to_owned();
    loop {
        if gradient.iter().fold(0_f64, |a, g| a.max(g.abs())) <= options.tolerance {
            success = true;
            message = "gradient converged".into();
            break;
        }
        if steps >= options.max_iterations {
            break;
        }
        if history.len() >= options.max_evaluations {
            message = "evaluation budget exhausted".into();
            break;
        }
        if dot(&direction, &gradient) >= -1e-12 * dot(&gradient, &gradient) {
            direction = gradient.iter().map(|g| -g).collect();
        }
        let slope = dot(&direction, &gradient);
        let mut alpha = 1.;
        let mut lower = 0.;
        let mut upper = None;
        let mut low_value = energy;
        let mut accepted = None;
        for _ in 0..80 {
            if history.len() >= options.max_evaluations {
                break;
            }
            let next: Vec<_> = x
                .iter()
                .zip(&direction)
                .map(|(x, p)| x + alpha * p)
                .collect();
            let (value, g) = objective(&next)?;
            validate(value, &g, n)?;
            history.push(value);
            let noise = 4. * f64::EPSILON * energy.abs().max(1.);
            if value > energy + 1e-4 * alpha * slope + noise
                || (lower > 0. && value > low_value + noise)
            {
                upper = Some(alpha);
            } else {
                let derivative = dot(&g, &direction);
                if derivative.abs() <= -0.1 * slope {
                    accepted = Some((next, value, g));
                    break;
                }
                if derivative >= 0. {
                    upper = Some(alpha);
                } else {
                    lower = alpha;
                    low_value = value;
                }
            }
            let next = upper.map_or(alpha * 2., |hi| 0.5 * (lower + hi));
            if next == alpha || !next.is_finite() {
                break;
            }
            alpha = next;
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
        let beta = (g
            .iter()
            .zip(&gradient)
            .map(|(a, b)| a * (a - b))
            .sum::<f64>()
            / dot(&gradient, &gradient))
        .max(0.);
        direction = g
            .iter()
            .zip(&direction)
            .map(|(g, p)| -g + beta * p)
            .collect();
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
        success,
        message,
    })
}
