//! Simultaneous-perturbation stochastic approximation with seeded gains.
//!
//! Two forms, chosen by `Options::track_best`:
//!
//! * `true` (default, the state-vector form): one initial evaluation, then
//!   three per iteration, the two perturbed points and the updated point. The
//!   returned point is the lowest energy among the initial and the updated
//!   points. At most `3 * max_iterations + 1` evaluations.
//! * `false` (the strict two-evaluation form for hardware): one initial
//!   evaluation, then the two perturbed points only; the updated point is
//!   never evaluated. The returned point is the lowest energy among the
//!   points actually evaluated, the initial point and the perturbed trial
//!   points. At most `2 * max_iterations + 1` evaluations.
//!
//! Both are further capped by `Options::max_evaluations`. With
//! `Options::tolerance == None` the five-calm-steps convergence test is off,
//! so a run always uses its whole iteration budget.
use super::*;
/// Most evaluations one run can request. Lives beside the loop so the two
/// cannot drift apart.
pub(super) fn evaluation_bound(options: &Options) -> usize {
    let per_iteration = if options.track_best { 3 } else { 2 };
    options
        .max_iterations
        .saturating_mul(per_iteration)
        .saturating_add(1)
        .min(options.max_evaluations)
}
fn random(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    if (z ^ (z >> 31)) >> 63 == 0 {
        1.
    } else {
        -1.
    }
}
pub(super) fn minimize(
    ctx: &Context,
    initial: &[f64],
    options: &Options,
    mut objective: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
) -> Result<Outcome> {
    let n = initial.len();
    let _work = ctx.reserve(product(&[n, 8, 12])?)?;
    let _history = ctx.reserve(product(&[options.max_evaluations, 8])?)?;
    let mut x = initial.to_vec();
    let (mut best_energy, g) = objective(&x)?;
    validate(best_energy, &g, n)?;
    let mut best = x.clone();
    let mut history = vec![best_energy];
    let mut seed = options.seed;
    let mut calm = 0;
    let mut steps = 0;
    let mut success = n == 0;
    let mut message = if success {
        "empty parameter vector"
    } else {
        "iteration budget exhausted"
    }
    .to_owned();
    for k in 0..if n == 0 { 0 } else { options.max_iterations } {
        let per_iteration = if options.track_best { 3 } else { 2 };
        if history.len().saturating_add(per_iteration) > options.max_evaluations {
            message = "evaluation budget exhausted".into();
            break;
        }
        let ak = 0.2 / (k as f64 + 1. + 0.1 * options.max_iterations as f64).powf(0.602);
        let ck = 0.1 / (k as f64 + 1.).powf(0.101);
        let delta: Vec<_> = (0..n).map(|_| random(&mut seed)).collect();
        let mut values = [0.; 2];
        for (sign, out) in [1., -1.].into_iter().zip(&mut values) {
            let trial: Vec<_> = x
                .iter()
                .zip(&delta)
                .map(|(x, d)| x + sign * ck * d)
                .collect();
            let (value, g) = objective(&trial)?;
            validate(value, &g, n)?;
            history.push(value);
            *out = value;
            if !options.track_best && value < best_energy {
                best_energy = value;
                best = trial;
            }
        }
        let estimate = (values[0] - values[1]) / (2. * ck);
        let step = ak * estimate;
        for (x, d) in x.iter_mut().zip(delta) {
            *x -= step * d;
        }
        steps += 1;
        if options.track_best {
            let (value, g) = objective(&x)?;
            validate(value, &g, n)?;
            history.push(value);
            if value < best_energy {
                best_energy = value;
                best.clone_from(&x);
            }
        }
        calm = if options
            .tolerance
            .is_some_and(|t| step.abs() * (n as f64).sqrt() < t)
        {
            calm + 1
        } else {
            0
        };
        if calm >= 5 {
            success = true;
            message = "five consecutive SPSA updates converged".into();
            break;
        }
    }
    Ok(Outcome {
        parameters: best,
        energy: best_energy,
        history,
        steps,
        success,
        message,
        interrupted: None,
    })
}
