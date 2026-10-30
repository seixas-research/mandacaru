use mandacaru_core::{
    molecular::correlation,
    quantum::{
        dynamics::{qpe, Propagator},
        evolution, optimizer, Basis, CachedProblem, Encoding, Operator, SolveOptions, C64,
    },
    Context,
};

fn context() -> Context {
    Context::new(Some(1), 64 << 20).unwrap()
}
fn operator(ctx: &Context, n: usize, terms: &[(&str, f64, f64)]) -> Operator {
    Operator::new(
        ctx,
        n,
        &terms
            .iter()
            .map(|(l, r, i)| (l.to_string(), C64::new(*r, *i)))
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
#[test]
fn standalone_propagation_qpe_and_cleanup() {
    let ctx = context();
    {
        let prop = Propagator::new(
            &ctx,
            1,
            &[
                ("X".into(), C64::new(1., 0.)),
                ("I".into(), C64::new(-0.5, 0.)),
            ],
            1,
        )
        .unwrap();
        let state = [C64::new(1., 0.), C64::default()];
        let evolved = prop.evolve(&ctx, &state, 0.7, 3, false, true).unwrap();
        let phase = C64::from_polar(1., 0.35);
        assert!((evolved.as_slice()[0] - phase * 0.7f64.cos()).norm() < 1e-13);
        assert!((evolved.as_slice()[1] - phase * C64::new(0., -0.7f64.sin())).norm() < 1e-13);
        let recovered = prop
            .evolve(&ctx, evolved.as_slice(), 0.7, 3, true, true)
            .unwrap();
        assert!((recovered.as_slice()[0] - state[0]).norm() < 1e-13);
        let h = operator(&ctx, 1, &[("Z", 0.5, 0.)]);
        let distribution = qpe(&ctx, &h, &state, 3, [-0.5, 1.5], true).unwrap();
        assert!((distribution.probabilities.as_slice()[4] - 1.).abs() < 1e-13);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn arbitrary_reference_exact_derivatives() {
    let ctx = context();
    let basis = Basis::register(&ctx, 1).unwrap();
    let h = operator(&ctx, 1, &[("Z", 0.7, 0.), ("X", 0.2, 0.)]);
    let g = operator(&ctx, 1, &[("Y", 0., 1.)]);
    let initial = [C64::new(0.6, 0.), C64::new(0., 0.8)];
    let eval = |t| {
        evolution::evaluate_from(
            &ctx,
            &basis,
            &h,
            None,
            None,
            &[&g],
            &[t],
            false,
            Some(&initial),
        )
        .unwrap()
    };
    let result = eval(0.31);
    let finite = (eval(0.310001).energy - eval(0.309999).energy) / 0.000002;
    assert!((result.gradient[0] - finite).abs() < 1e-9);
}

#[test]
fn product_adjoint_gradients_match_forward_derivatives_for_noncommuting_generators() {
    let ctx = context();
    {
        let basis = Basis::register(&ctx, 2).unwrap();
        let h = operator(
            &ctx,
            2,
            &[("II", 19., 0.), ("ZX", 0.7, 0.), ("YY", -0.4, 0.)],
        );
        let a = operator(&ctx, 2, &[("XI", 0., 0.7), ("YZ", 0., 0.3)]);
        let b = operator(&ctx, 2, &[("ZY", 0., 0.4), ("IZ", 0., -0.6)]);
        let initial = [
            C64::new(0.5, 0.),
            C64::new(0., 0.5),
            C64::new(-0.5, 0.),
            C64::new(0., -0.5),
        ];
        for n in [0, 1, 7, 30] {
            let ops: Vec<_> = (0..n).map(|i| if i % 2 == 0 { &a } else { &b }).collect();
            let theta: Vec<_> = (0..n).map(|i| 0.31 * (i as f64 + 0.2).cos()).collect();
            let eval = |t: &[f64]| {
                evolution::evaluate_from(
                    &ctx,
                    &basis,
                    &h,
                    None,
                    None,
                    &ops,
                    t,
                    true,
                    Some(&initial),
                )
                .unwrap()
            };
            let result = eval(&theta);
            let forward =
                evolution::state_from(&ctx, &basis, &ops, &theta, true, true, Some(&initial))
                    .unwrap();
            let hpsi = h.apply(&ctx, &basis, &forward.as_slice()[..4]).unwrap();
            for k in 0..n {
                let direct = 2.
                    * evolution::inner(
                        &forward.as_slice()[(k + 1) * 4..(k + 2) * 4],
                        hpsi.as_slice(),
                    )
                    .re;
                assert!((result.gradient[k] - direct).abs() < 2e-12);
                if n == 7 {
                    let mut t = theta.clone();
                    t[k] += 1e-5;
                    let high = eval(&t).energy;
                    t[k] -= 2e-5;
                    let low = eval(&t).energy;
                    assert!((result.gradient[k] - (high - low) / 2e-5).abs() < 2e-9);
                }
            }
            assert!(result
                .state
                .complex128()
                .unwrap()
                .iter()
                .zip(&forward.as_slice()[..4])
                .all(|(a, b)| (a - b).norm() < 1e-14));
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn subspace_weights_sorting_and_one_state_limit() {
    let ctx = context();
    {
        let h = operator(
            &ctx,
            4,
            &[("ZIII", 0.3, 0.), ("XXII", 0.2, 0.), ("YYII", 0.2, 0.)],
        );
        let p = CachedProblem::new(
            &ctx,
            Encoding::new("jordan_wigner", 4, [1, 0]).unwrap(),
            h,
            true,
            false,
            true,
        )
        .unwrap();
        for adapt in [false, true] {
            let options = SolveOptions {
                adapt,
                optimizer: optimizer::Options {
                    tolerance: Some(1e-9),
                    ..Default::default()
                },
                ..Default::default()
            };
            let result = p.solve_subspace(&options, &[2., 1.], None).unwrap();
            assert!(result.success, "{}", result.message);
            assert!((result.energies[0] + 0.5).abs() < 1e-11);
            assert!((result.energies[1] - 0.5).abs() < 1e-11);
            let one = p.solve_subspace(&options, &[1.], None).unwrap();
            assert!((one.energies[0] - p.solve(&options, None).unwrap().energy).abs() < 1e-11);
            let states = result.states.complex128().unwrap();
            assert!(evolution::inner(&states[..2], &states[2..]).norm() < 1e-12);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn adaptive_growth_continues_after_an_inner_budget_failure_without_false_convergence() {
    let ctx = context();
    {
        let h = operator(
            &ctx,
            4,
            &[("ZIII", 0.3, 0.), ("XXII", 0.2, 0.), ("YYII", 0.2, 0.)],
        );
        let p = CachedProblem::new(
            &ctx,
            Encoding::new("jordan_wigner", 4, [1, 0]).unwrap(),
            h,
            true,
            false,
            true,
        )
        .unwrap();
        let options = SolveOptions {
            adapt: true,
            max_growth: 3,
            optimizer: optimizer::Options {
                max_iterations: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = p.solve(&options, None).unwrap();
        assert!(!result.success);
        assert!(result.final_gradient > 0.1);
        assert_eq!(result.iterations.len(), 3);
        assert!(result.iterations.iter().all(|it| !it.success));
        assert_eq!(result.message, "growth budget exhausted");
        let shared = p.solve_subspace(&options, &[2., 1.], None).unwrap();
        assert!(!shared.success);
        assert!(shared.final_gradient > 0.1);
        assert_eq!(shared.optimizer_failures.len(), 3);
        assert_eq!(shared.indices.len(), 3);
        assert_eq!(shared.message, "growth budget exhausted");
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn mp2_known_amplitude_density_and_degenerate_rejection() {
    let ctx = context();
    {
        let result =
            correlation::from_block(&ctx, &[0.2], &[-1., 1.], 1, &[1., 0., 0., 1.]).unwrap();
        assert!((result.correlation_energy + 0.01).abs() < 1e-15);
        assert!((result.occupied_occupations.as_slice()[0] - 1.995).abs() < 1e-15);
        assert!((result.virtual_occupations.as_slice()[0] - 0.005).abs() < 1e-15);
        assert!(correlation::from_block(&ctx, &[0.2], &[0., 0.], 1, &[1., 0., 0., 1.]).is_err());
    }
    assert_eq!(ctx.memory_used(), 0);
}
