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
            tolerance: 1e-11,
            ..Default::default()
        };
        let result = optimizer::minimize(&context, &[-1.2, 1.], &options, objective).unwrap();
        let (_, gradient) = objective(&result.parameters).unwrap();
        assert!(result.success, "{method:?}: {}", result.message);
        assert!(gradient.iter().map(|g| g.abs()).fold(0., f64::max) <= options.tolerance);
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
            tolerance: 1e-8,
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
                tolerance: 1e-10,
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
