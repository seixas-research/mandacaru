use mandacaru_core::{
    quantum::optimizer::{self, Method, Options},
    Context,
};

#[test]
fn quasi_newton_reaches_the_requested_gradient_accuracy_on_an_energy_plateau() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    let objective = |x: &[f64]| {
        let residual = x[1] - x[0] * x[0];
        Ok((
            -1e6 + (1. - x[0]).powi(2) + 100. * residual.powi(2),
            vec![-2. * (1. - x[0]) - 400. * x[0] * residual, 200. * residual],
        ))
    };
    for method in [Method::Bfgs, Method::Lbfgs] {
        let options = Options {
            method,
            max_iterations: 600,
            max_evaluations: 8000,
            tolerance: Some(1e-11),
            ..Default::default()
        };
        let result = optimizer::minimize(&context, &[-1.2, 1.], &options, objective).unwrap();
        let (_, gradient) = objective(&result.parameters).unwrap();
        assert!(result.success, "{method:?}: {}", result.message);
        assert!(gradient.iter().map(|g| g.abs()).fold(0., f64::max) <= options.tolerance.unwrap());
        assert!(result.history.len() <= options.max_evaluations);
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn every_named_optimizer_preserves_budgets_and_solves_a_shifted_quadratic() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    for name in Method::NAMES {
        let method = Method::parse(name).unwrap();
        let options = Options {
            method,
            max_iterations: 3000,
            max_evaluations: 20000,
            tolerance: Some(1e-8),
            ..Default::default()
        };
        let objective = |x: &[f64]| {
            Ok((
                (x[0] - 0.7).powi(2) + 2. * (x[1] + 0.4).powi(2),
                vec![2. * (x[0] - 0.7), 4. * (x[1] + 0.4)],
            ))
        };
        let result = optimizer::minimize(&ctx, &[-0.8, 0.9], &options, objective).unwrap();
        assert!(
            result.energy < if method == Method::Spsa { 1e-6 } else { 1e-10 },
            "{name}: {} {} {:?}",
            result.message,
            result.energy,
            result.parameters
        );
        if method != Method::Spsa {
            assert!(result.success, "{name}: {}", result.message);
        }
        assert!(result.history.len() <= options.max_evaluations);
        let budget = Options {
            max_evaluations: 1,
            ..options.clone()
        };
        let result = optimizer::minimize(&ctx, &[-0.8, 0.9], &budget, objective).unwrap();
        assert!(!result.success, "{name}");
        assert_eq!(result.history.len(), 1, "{name}");
        let empty = optimizer::minimize(&ctx, &[], &options, |_| Ok((2., vec![]))).unwrap();
        assert!(empty.success, "{name}");
        assert_eq!(empty.history, [2.]);
        let zero = Options {
            max_iterations: 0,
            ..options.clone()
        };
        let result = optimizer::minimize(&ctx, &[-0.8, 0.9], &zero, objective).unwrap();
        assert!(!result.success, "{name}");
        assert_eq!(result.steps, 0);
        assert_eq!(ctx.memory_used(), 0, "{name}");
    }
}

#[test]
fn gradient_and_simplex_methods_solve_a_nonseparable_rosenbrock_problem() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let objective = |x: &[f64]| {
        Ok((
            (1. - x[0]).powi(2) + 100. * (x[1] - x[0] * x[0]).powi(2),
            vec![
                -2. * (1. - x[0]) - 400. * x[0] * (x[1] - x[0] * x[0]),
                200. * (x[1] - x[0] * x[0]),
            ],
        ))
    };
    for method in [Method::Slsqp, Method::Cg, Method::NelderMead] {
        let result = optimizer::minimize(
            &ctx,
            &[-1.2, 1.],
            &Options {
                method,
                max_iterations: 4000,
                max_evaluations: 20000,
                tolerance: Some(1e-10),
                ..Default::default()
            },
            objective,
        )
        .unwrap();
        assert!(result.success, "{method:?}: {}", result.message);
        assert!(result.energy < 1e-12, "{method:?}: {}", result.energy);
    }
    assert_eq!(ctx.memory_used(), 0);
}

fn shifted_quadratic(x: &[f64]) -> mandacaru_core::Result<(f64, Vec<f64>)> {
    Ok((
        (x[0] - 0.7).powi(2) + 2. * (x[1] + 0.4).powi(2),
        vec![2. * (x[0] - 0.7), 4. * (x[1] + 0.4)],
    ))
}

/// Run `options` on the quadratic and return every evaluated point.
fn traced(ctx: &Context, options: &Options) -> (optimizer::Outcome, Vec<(Vec<f64>, f64)>) {
    let mut points = Vec::new();
    let outcome = optimizer::minimize(ctx, &[-0.8, 0.9], options, |x| {
        let value = shifted_quadratic(x)?;
        points.push((x.to_vec(), value.0));
        Ok(value)
    })
    .unwrap();
    (outcome, points)
}

#[test]
fn spsa_default_form_reproduces_the_frozen_energy_history() {
    // Captured before `track_best` existed: seed 3, 12 iterations, tolerance 1e-9.
    const HISTORY: [f64; 37] = [
        5.630000000000001,
        4.840000000000001,
        6.48,
        0.38668609210050475,
        0.6065645493360855,
        0.21896830847032536,
        0.08549359251072716,
        0.034327898591859786,
        0.18471806810494862,
        0.0410337510076151,
        0.027609901090830086,
        0.09980319877388694,
        0.031095564563282864,
        0.03317420320547101,
        0.07236395224273742,
        0.028260573255076398,
        0.0898081796543694,
        0.008492607269806506,
        0.016438202478485726,
        0.011294793620221597,
        0.06208034524555779,
        0.01196503650223498,
        0.048257441269646595,
        0.015093580549304507,
        0.010111639148458662,
        0.018142367618794,
        0.040575018946779125,
        0.009286311004905555,
        0.0068991088269262655,
        0.04935701471168392,
        0.006404322937761055,
        0.03458348200921965,
        0.015190097259416077,
        0.005817345636990979,
        0.039831321557187035,
        0.008124272833243127,
        0.004283648516797378,
    ];
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let options = Options {
        method: Method::Spsa,
        max_iterations: 12,
        seed: 3,
        ..Default::default()
    };
    assert!(options.track_best);
    let result = optimizer::minimize(&ctx, &[-0.8, 0.9], &options, shifted_quadratic).unwrap();
    assert_eq!(result.history, HISTORY);
    assert_eq!(result.energy, 0.004283648516797378);
    assert_eq!(result.steps, 12);
    assert!(!result.success);
    assert_eq!(options.evaluation_bound(), Some(37));
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn strict_spsa_uses_two_evaluations_per_iteration_and_keeps_the_best_evaluated_point() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let options = Options {
        method: Method::Spsa,
        max_iterations: 25,
        max_evaluations: 5000,
        tolerance: None,
        seed: 3,
        track_best: false,
        ..Default::default()
    };
    let (result, points) = traced(&ctx, &options);
    assert_eq!(points.len(), 2 * 25 + 1);
    assert_eq!(result.history.len(), points.len());
    assert_eq!(result.steps, 25);
    assert_eq!(options.evaluation_bound(), Some(51));
    let lowest = points
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("evaluated points");
    assert_eq!(result.energy, lowest.1);
    assert_eq!(result.parameters, lowest.0);
    // The returned point was evaluated, never an unmeasured update.
    assert!(points.iter().any(|(x, _)| *x == result.parameters));
    // The tracked form on the same seed makes 3 * iterations + 1 evaluations.
    let tracked = Options {
        track_best: true,
        ..options.clone()
    };
    let (_, points) = traced(&ctx, &tracked);
    assert_eq!(points.len(), 3 * 25 + 1);
    assert_eq!(tracked.evaluation_bound(), Some(76));
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn spsa_without_a_tolerance_never_declares_convergence() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let flat = |_: &[f64]| Ok((1., vec![0., 0.]));
    for track_best in [true, false] {
        let options = Options {
            method: Method::Spsa,
            max_iterations: 30,
            tolerance: Some(1e-9),
            track_best,
            ..Default::default()
        };
        let result = optimizer::minimize(&ctx, &[0.1, 0.2], &options, flat).unwrap();
        assert!(result.success && result.steps == 5, "{}", result.message);
        let options = Options {
            tolerance: None,
            ..options
        };
        let result = optimizer::minimize(&ctx, &[0.1, 0.2], &options, flat).unwrap();
        assert!(!result.success && result.steps == 30, "{}", result.message);
        assert_eq!(result.history.len(), if track_best { 91 } else { 61 });
    }
    assert!(Options {
        tolerance: Some(0.),
        ..Default::default()
    }
    .validate()
    .is_err());
}

#[test]
fn missing_tolerance_means_the_default_for_every_other_method() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    for method in [
        Method::Bfgs,
        Method::Lbfgs,
        Method::Slsqp,
        Method::Cg,
        Method::Cobyla,
        Method::NelderMead,
    ] {
        let base = Options {
            method,
            max_iterations: 400,
            max_evaluations: 4000,
            ..Default::default()
        };
        assert_eq!(base.tolerance, Some(optimizer::DEFAULT_TOLERANCE));
        let omitted = Options {
            tolerance: None,
            ..base.clone()
        };
        let a = optimizer::minimize(&ctx, &[-0.8, 0.9], &base, shifted_quadratic).unwrap();
        let b = optimizer::minimize(&ctx, &[-0.8, 0.9], &omitted, shifted_quadratic).unwrap();
        assert_eq!(a.history, b.history, "{method:?}");
        assert_eq!(a.success, b.success, "{method:?}");
    }
}

#[test]
fn evaluation_bound_matches_the_instrumented_call_count() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    for (track_best, per_iteration) in [(true, 3), (false, 2)] {
        for iterations in [0, 1, 7, 40] {
            let options = Options {
                method: Method::Spsa,
                max_iterations: iterations,
                tolerance: None,
                track_best,
                ..Default::default()
            };
            let (_, points) = traced(&ctx, &options);
            let bound = options.evaluation_bound().unwrap();
            assert_eq!(points.len(), per_iteration * iterations + 1);
            assert_eq!(
                bound,
                points.len(),
                "track_best={track_best} it={iterations}"
            );
        }
        // max_evaluations clips both the loop and the bound.
        let clipped = Options {
            method: Method::Spsa,
            max_iterations: 40,
            max_evaluations: 20,
            tolerance: None,
            track_best,
            ..Default::default()
        };
        let (_, points) = traced(&ctx, &clipped);
        assert_eq!(clipped.evaluation_bound(), Some(20));
        assert!(points.len() <= 20);
        assert!(points.len() > 20 - per_iteration);
    }
    for method in [Method::Cobyla, Method::NelderMead] {
        for max_evaluations in [3, 17, 60, 5000] {
            let options = Options {
                method,
                max_iterations: 200,
                max_evaluations,
                ..Default::default()
            };
            let (_, points) = traced(&ctx, &options);
            let bound = options.evaluation_bound().unwrap();
            assert_eq!(bound, max_evaluations, "{method:?}");
            assert!(
                points.len() <= bound,
                "{method:?}: {} > {bound}",
                points.len()
            );
            if max_evaluations <= 60 {
                assert_eq!(points.len(), bound, "{method:?} spends the whole budget");
            }
        }
    }
    for method in [Method::Bfgs, Method::Lbfgs, Method::Slsqp, Method::Cg] {
        let options = Options {
            method,
            ..Default::default()
        };
        assert_eq!(options.evaluation_bound(), None, "{method:?}");
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn convergence_needs_patience_calm_steps_a_finite_cost_and_works_without_parameters() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let flat = |_: &[f64]| Ok((1., vec![0.]));
    // `patience` consecutive calm steps are required before success, not one.
    for patience in [1, 5] {
        let options = Options {
            method: Method::Spsa,
            max_iterations: 30,
            tolerance: Some(1e-9),
            patience,
            ..Default::default()
        };
        let result = optimizer::minimize(&ctx, &[0.3], &options, flat).unwrap();
        assert!(result.success && result.steps == patience, "patience {patience}: {} steps", result.steps);
    }
    // A non-finite cost is refused, never minimized.
    let options = Options { method: Method::Cobyla, max_iterations: 5, ..Default::default() };
    let nan = |_: &[f64]| Ok((f64::NAN, vec![0.]));
    assert!(optimizer::minimize(&ctx, &[0.1], &options, nan).is_err());
    // A problem without parameters is one evaluation and a success.
    let result = optimizer::minimize(&ctx, &[], &options, |_: &[f64]| Ok((1., vec![]))).unwrap();
    assert!(result.success && result.history.len() == 1, "{}", result.message);
    assert_eq!(ctx.memory_used(), 0);
}

/// `f = 0.5 x' A x + b' x` on three parameters, `A` coupled.
fn coupled_quadratic(x: &[f64]) -> mandacaru_core::Result<(f64, Vec<f64>)> {
    const A: [[f64; 3]; 3] = [[4., 1., 0.5], [1., 3., -0.4], [0.5, -0.4, 2.]];
    const B: [f64; 3] = [1., -2., 0.5];
    let gradient: Vec<f64> = (0..3)
        .map(|i| (0..3).map(|j| A[i][j] * x[j]).sum::<f64>() + B[i])
        .collect();
    let energy = 0.5 * x.iter().zip(&gradient).map(|(x, g)| x * g).sum::<f64>()
        + 0.5 * x.iter().zip(&B).map(|(x, b)| x * b).sum::<f64>();
    Ok((energy, gradient))
}

#[test]
fn slsqp_returns_its_hessian_and_restarts_from_it() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let options = Options {
        method: Method::Slsqp,
        tolerance: Some(1e-12),
        ..Default::default()
    };
    let cold = optimizer::minimize(&ctx, &[0.3, -0.2, 0.1], &options, coupled_quadratic).unwrap();
    let hessian = cold.hessian.clone().expect("SLSQP returns its Hessian approximation");
    assert_eq!(hessian.len(), 9);
    for i in 0..3 {
        for j in 0..i {
            assert_eq!(hessian[i * 3 + j], hessian[j * 3 + i], "the approximation is symmetric");
        }
    }
    // Started at a displaced point with the learned curvature, the quadratic is solved in the
    // fewest steps; from the identity it takes more.
    let start = [1.1, 0.7, -0.9];
    let warm = optimizer::minimize_from(&ctx, &start, &options, Some(&hessian), coupled_quadratic).unwrap();
    let identity = optimizer::minimize(&ctx, &start, &options, coupled_quadratic).unwrap();
    assert!(warm.success && identity.success);
    assert!(warm.history.len() < identity.history.len(), "{} vs {}", warm.history.len(), identity.history.len());
    assert!((warm.energy - identity.energy).abs() < 1e-10);
    // The identity is the default: an explicit one is the same run, bit for bit.
    let explicit: Vec<f64> = (0..9).map(|k| if k % 4 == 0 { 1. } else { 0. }).collect();
    let same = optimizer::minimize_from(&ctx, &start, &options, Some(&explicit), coupled_quadratic).unwrap();
    assert_eq!(same.history, identity.history);
    assert_eq!(same.parameters, identity.parameters);
    // No other method but BFGS keeps one, and one given to it is refused, not ignored.
    for method in [Method::Lbfgs, Method::Cg, Method::Cobyla, Method::NelderMead, Method::Spsa] {
        let other = Options { method, ..options.clone() };
        let out = optimizer::minimize(&ctx, &start, &other, coupled_quadratic).unwrap();
        assert!(out.hessian.is_none(), "{method:?}");
        assert!(optimizer::minimize_from(&ctx, &start, &other, Some(&hessian), coupled_quadratic).is_err(), "{method:?}");
    }
    // A Hessian of the wrong size or with a nonfinite entry is refused.
    assert!(optimizer::minimize_from(&ctx, &start, &options, Some(&hessian[..4]), coupled_quadratic).is_err());
    let mut bad = hessian.clone();
    bad[1] = f64::NAN;
    assert!(optimizer::minimize_from(&ctx, &start, &options, Some(&bad), coupled_quadratic).is_err());
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn bfgs_returns_its_inverse_hessian_and_restarts_from_it() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let options = Options {
        method: Method::Bfgs,
        tolerance: Some(1e-12),
        max_iterations: 2000,
        ..Default::default()
    };
    let cold = optimizer::minimize(&ctx, &[0.3, -0.2, 0.1], &options, coupled_quadratic).unwrap();
    let inverse = cold.hessian.clone().expect("BFGS returns its inverse Hessian approximation");
    assert_eq!(inverse.len(), 9);
    // The estimate is symmetric with a positive diagonal (an inverse Hessian approximation).
    for i in 0..3 {
        assert!(inverse[i * 3 + i] > 0.);
        for j in 0..i {
            assert!((inverse[i * 3 + j] - inverse[j * 3 + i]).abs() < 1e-12);
        }
    }
    let start = [1.1, 0.7, -0.9];
    let warm = optimizer::minimize_from(&ctx, &start, &options, Some(&inverse), coupled_quadratic).unwrap();
    let identity = optimizer::minimize(&ctx, &start, &options, coupled_quadratic).unwrap();
    assert!(warm.success && identity.success);
    assert!(warm.history.len() < identity.history.len(), "{} vs {}", warm.history.len(), identity.history.len());
    let explicit: Vec<f64> = (0..9).map(|k| if k % 4 == 0 { 1. } else { 0. }).collect();
    let same = optimizer::minimize_from(&ctx, &start, &options, Some(&explicit), coupled_quadratic).unwrap();
    assert_eq!(same.history, identity.history);
    assert_eq!(same.parameters, identity.parameters);
    assert!(optimizer::minimize_from(&ctx, &start, &options, Some(&inverse[..4]), coupled_quadratic).is_err());
    assert_eq!(ctx.memory_used(), 0);
}

/// An objective whose gradient is computed only when the optimizer reads it, counting the reads.
fn deferred<'p>(
    objective: &'p impl Fn(&[f64]) -> mandacaru_core::Result<(f64, Vec<f64>)>,
    reads: &'p std::cell::Cell<usize>,
) -> impl FnMut(&[f64]) -> mandacaru_core::Result<optimizer::Probe<'p>> {
    move |x| {
        let energy = objective(x)?.0;
        let point = x.to_vec();
        Ok(optimizer::Probe::deferred(energy, move || {
            reads.set(reads.get() + 1);
            Ok(objective(&point)?.1)
        }))
    }
}

fn rosenbrock(x: &[f64]) -> mandacaru_core::Result<(f64, Vec<f64>)> {
    let residual = x[1] - x[0] * x[0];
    Ok((
        (1. - x[0]).powi(2) + 100. * residual.powi(2),
        vec![-2. * (1. - x[0]) - 400. * x[0] * residual, 200. * residual],
    ))
}

#[test]
fn a_deferred_gradient_leaves_every_method_run_unchanged_and_is_read_only_where_a_test_needs_it() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    for name in Method::NAMES {
        let method = Method::parse(name).unwrap();
        let options = Options {
            method,
            max_iterations: 400,
            max_evaluations: 3000,
            tolerance: Some(1e-9),
            ..Default::default()
        };
        let cases: [(bool, fn(&[f64]) -> mandacaru_core::Result<(f64, Vec<f64>)>, &[f64]); 2] =
            [(true, rosenbrock, &[-1.2, 1.]), (false, coupled_quadratic, &[0.3, -0.2, 0.1])];
        for (is_rosenbrock, objective, start) in cases {
            let eager = optimizer::minimize(&ctx, start, &options, objective).unwrap();
            let reads = std::cell::Cell::new(0);
            let lazy = optimizer::minimize_deferred_from(&ctx, start, &options, None, deferred(&objective, &reads)).unwrap();
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&lazy.parameters), bits(&eager.parameters), "{name}");
            assert_eq!(lazy.energy.to_bits(), eager.energy.to_bits(), "{name}");
            assert_eq!(bits(&lazy.history), bits(&eager.history), "{name}");
            assert_eq!(lazy.steps, eager.steps, "{name}");
            assert_eq!(lazy.success, eager.success, "{name}");
            assert_eq!(lazy.message, eager.message, "{name}");
            assert_eq!(lazy.hessian.as_deref().map(bits), eager.hessian.as_deref().map(bits), "{name}");
            let evaluations = lazy.history.len();
            match method {
                // The three derivative-free methods never read a gradient.
                Method::Cobyla | Method::NelderMead | Method::Spsa => assert_eq!(reads.get(), 0, "{name}"),
                // Every iteration accepts one trial on its energy alone: one gradient per
                // iteration and one at the start, whatever the line search rejected.
                Method::Slsqp => assert_eq!(reads.get(), lazy.steps + 1, "{name}"),
                // The Wolfe tests read a gradient only for a trial the energy does not reject.
                Method::Bfgs | Method::Lbfgs | Method::Cg => {
                    assert!(reads.get() <= evaluations, "{name}");
                    assert!(reads.get() > lazy.steps, "{name}");
                }
            }
            if is_rosenbrock && method == Method::Slsqp {
                assert!(reads.get() < evaluations, "Rosenbrock rejects line-search trials: {} reads of {evaluations}", reads.get());
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_gradient_that_a_run_never_reads_is_never_checked_and_one_it_reads_is() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let broken = |x: &[f64]| -> mandacaru_core::Result<optimizer::Probe<'static>> {
        let energy = (x[0] - 0.7).powi(2);
        Ok(optimizer::Probe::deferred(energy, || Ok(vec![f64::NAN])))
    };
    let derivative_free = Options { method: Method::NelderMead, ..Default::default() };
    assert!(optimizer::minimize_deferred_from(&ctx, &[0.1], &derivative_free, None, broken).is_ok());
    let quasi_newton = Options { method: Method::Slsqp, ..Default::default() };
    assert!(optimizer::minimize_deferred_from(&ctx, &[0.1], &quasi_newton, None, broken).is_err());
}
