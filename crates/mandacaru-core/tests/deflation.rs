use mandacaru_core::{
    quantum::{evolution, optimizer, CachedProblem, Encoding, Operator, SolveOptions, C64},
    Context,
};

#[test]
fn sequential_deflation_finds_unbiased_orthogonal_levels_with_independent_restarts() {
    let ctx = Context::new(Some(1), 16 << 20).unwrap();
    {
        let h = Operator::new(
            &ctx,
            4,
            &[
                ("ZIII".into(), C64::new(0.3, 0.)),
                ("XXII".into(), C64::new(0.2, 0.)),
                ("YYII".into(), C64::new(0.2, 0.)),
            ],
        )
        .unwrap();
        let p = CachedProblem::new(
            &ctx,
            Encoding::new("jordan_wigner", 4, [1, 0]).unwrap(),
            h,
            true,
            false,
            true,
        )
        .unwrap();
        for method in [optimizer::Method::Bfgs, optimizer::Method::Slsqp] {
            let options = SolveOptions {
                optimizer: optimizer::Options {
                    method,
                    tolerance: 1e-12,
                    ..Default::default()
                },
                ..Default::default()
            };
            let result = p.energy_levels(&options, 2, 4, None, 11).unwrap();
            assert!((result.energies[0] + 0.5).abs() < 1e-10);
            assert!((result.energies[1] - 0.5).abs() < 1e-10);
            let states = result.states.as_slice();
            assert!(evolution::inner(&states[..2], &states[2..]).norm() < 1e-5);
            for state in states.chunks_exact(2) {
                assert!((evolution::inner(state, state).re - 1.).abs() < 1e-12);
            }
            assert!(result.evaluations > 0);
        }
        assert!(p.energy_levels(&Default::default(), 3, 1, None, 0).is_err());
        assert!(p.energy_levels(&Default::default(), 2, 0, None, 0).is_err());
        assert!(p
            .energy_levels(&Default::default(), 2, 1, Some(-1.), 0)
            .is_err());
    }
    assert_eq!(ctx.memory_used(), 0);
}
