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
