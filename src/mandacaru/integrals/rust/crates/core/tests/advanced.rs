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
                    tolerance: 1e-9,
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
            let states = result.states.as_slice();
            assert!(evolution::inner(&states[..2], &states[2..]).norm() < 1e-12);
        }
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
