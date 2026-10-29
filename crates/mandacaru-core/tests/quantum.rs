use mandacaru_core::{
    quantum::{optimizer, Basis, CachedProblem, Encoding, Operator, SolveOptions, C64},
    Context, Error,
};

fn context() -> Context {
    Context::new(Some(1), 64 * 1024 * 1024).unwrap()
}
#[test]
fn energy_gradients_are_invariant_under_large_identity_shifts_in_every_storage() {
    use mandacaru_core::quantum::sparse::SparseMatrix;
    let ctx = context();
    let mut expected: Option<(f64, f64)> = None;
    for storage in 0..3 {
        for constant in [0., 1e8, -1e8] {
            let h = Operator::new(
                &ctx,
                4,
                &[
                    ("IIII".into(), C64::new(constant, 0.)),
                    ("ZIII".into(), C64::new(0.3, 0.)),
                    ("XXII".into(), C64::new(0.2, 0.)),
                    ("YYII".into(), C64::new(0.2, 0.)),
                ],
            )
            .unwrap();
            let mut problem = CachedProblem::new(
                &ctx,
                Encoding::new("jordan_wigner", 4, [1, 0]).unwrap(),
                h,
                true,
                storage == 1,
                true,
            )
            .unwrap();
            if storage == 2 {
                problem.sparse =
                    Some(SparseMatrix::new(&ctx, &problem.basis, &problem.hamiltonian).unwrap());
            }
            let value = problem.evaluate(&[0], &[0.21], true).unwrap();
            if let Some((energy, gradient)) = expected {
                assert!((value.energy - constant - energy).abs() < 4e-8);
                assert!((value.gradient[0] - gradient).abs() < 1e-12);
            } else {
                expected = Some((value.energy, value.gradient[0]));
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
fn bounded_sector_lookup_matches_sorted_indices_and_preserves_cancellation_checks() {
    let ctx = context();
    for mapping in ["jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced"] {
        let encoding = Encoding::new(mapping, 16, [4, 4]).unwrap();
        {
            let basis = Basis::new(&ctx, &encoding, true).unwrap();
            for ket in 0..(1u64 << encoding.n_qubits()) {
                assert_eq!(
                    basis.index(ket),
                    basis.indices.as_slice().binary_search(&ket).ok()
                );
            }
            assert!(basis.index(u64::MAX).is_none());
            let jw = Encoding::new("jordan_wigner", 16, [4, 4]).unwrap();
            let first = jw.monomial(&ctx, &[(0, true), (1, false)]).unwrap();
            let hopping = first
                .add(&ctx, &first.dagger(&ctx).unwrap(), C64::new(1., 0.))
                .unwrap();
            let hopping = encoding.map_operator(&ctx, &hopping).unwrap();
            basis.check_invariant(&hopping).unwrap();
            let leaking = jw.monomial(&ctx, &[(0, true), (1, true)]).unwrap();
            // Single creation pairs change tapered parity, so mapping itself
            // may reject them before the sector check.
            if let Ok(leaking) = encoding.map_operator(&ctx, &leaking) {
                assert!(basis.check_invariant(&leaking).is_err());
            }
        }
        assert_eq!(ctx.memory_used(), 0);
    }
}
#[test]
fn bfgs_and_lbfgs_solve_rosenbrock_and_report_budgets() {
    let ctx = context();
    let objective = |x: &[f64]| {
        Ok((
            (1. - x[0]).powi(2) + 100. * (x[1] - x[0] * x[0]).powi(2),
            vec![
                -2. * (1. - x[0]) - 400. * x[0] * (x[1] - x[0] * x[0]),
                200. * (x[1] - x[0] * x[0]),
            ],
        ))
    };
    for limited_memory in [false, true] {
        let options = optimizer::Options {
            method: if limited_memory {
                optimizer::Method::Lbfgs
            } else {
                optimizer::Method::Bfgs
            },
            tolerance: 1e-8,
            ..Default::default()
        };
        let out = optimizer::minimize(&ctx, &[-1.2, 1.], &options, objective).unwrap();
        assert!(
            out.success,
            "{} ({}, {:?})",
            out.message, out.energy, out.parameters
        );
        assert!(out.energy < 1e-16);
        assert!(out.parameters.iter().all(|x| (x - 1.).abs() < 1e-7));
        let options = optimizer::Options {
            max_evaluations: 1,
            ..options
        };
        let out = optimizer::minimize(&ctx, &[-1.2, 1.], &options, objective).unwrap();
        assert!(!out.success);
        assert_eq!(out.history.len(), 1);
        assert_eq!(out.message, "evaluation budget exhausted");
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn lbfgs_respects_a_budget_that_cannot_hold_a_dense_hessian() {
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    let initial = vec![1.; 1000];
    let objective = |x: &[f64]| {
        Ok((
            x.iter().map(|x| x * x).sum(),
            x.iter().map(|x| 2. * x).collect(),
        ))
    };
    assert!(matches!(
        optimizer::minimize(&ctx, &initial, &Default::default(), objective),
        Err(Error::Budget { .. })
    ));
    let out = optimizer::minimize(
        &ctx,
        &initial,
        &optimizer::Options {
            method: optimizer::Method::Lbfgs,
            ..Default::default()
        },
        objective,
    )
    .unwrap();
    assert!(out.success);
    assert!(out.energy < 1e-20);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn encoding_roundtrips_and_sector_enumeration_avoid_the_full_register() {
    let ctx = context();
    for mapping in ["jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced"] {
        let encoding = Encoding::new(mapping, 40, [1, 1]).unwrap();
        let basis = Basis::new(&ctx, &encoding, true).unwrap();
        assert_eq!(basis.len(), 400);
        for &index in basis.indices.as_slice() {
            let occupation = encoding.decode(index);
            assert_eq!(occupation.count_ones(), 2);
            assert_eq!(encoding.encode(occupation), index);
        }
        assert!(matches!(
            Basis::new(&ctx, &encoding, false),
            Err(Error::Budget { .. })
        ));
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn sector_invariance_combines_cancelling_pauli_terms() {
    let ctx = context();
    let encoding = Encoding::new("jordan_wigner", 6, [1, 1]).unwrap();
    let basis = Basis::new(&ctx, &encoding, true).unwrap();
    let op = Operator::new(
        &ctx,
        6,
        &[
            ("XXIIII".into(), C64::new(1., 0.)),
            ("YYIIII".into(), C64::new(1., 0.)),
        ],
    )
    .unwrap();
    basis.check_invariant(&op).unwrap();
    let leak = Operator::new(&ctx, 6, &[("XXIIII".into(), C64::new(1., 0.))]).unwrap();
    assert!(basis.check_invariant(&leak).is_err());
}

#[test]
fn cached_h2_vqe_and_adapt_are_python_independent() {
    let ctx = context();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference/h2_hamiltonian.json");
    for sector in [false, true] {
        for dense in [false, true] {
            let problem = CachedProblem::load(&ctx, &fixture, sector, dense, true).unwrap();
            let live = ctx.memory_used();
            for adapt in [false, true] {
                for limited_memory in [false, true] {
                    let options = SolveOptions {
                        adapt,
                        optimizer: optimizer::Options {
                            method: if limited_memory {
                                optimizer::Method::Lbfgs
                            } else {
                                optimizer::Method::Bfgs
                            },
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    let result = problem.solve(&options, None).unwrap();
                    assert!(result.success, "{}", result.message);
                    assert!((result.energy + 0.9507746213870719).abs() < 1e-12);
                    assert!((result.reference + 0.9371240086056045).abs() < 1e-13);
                    if adapt {
                        assert_eq!(result.labels, ["D(0,2->1,3)"]);
                    }
                    drop(result);
                    assert_eq!(ctx.memory_used(), live);
                }
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn nonfinite_objectives_and_invalid_quantum_inputs_are_errors() {
    let ctx = context();
    assert!(
        optimizer::minimize(&ctx, &[1.], &Default::default(), |_| Ok((
            f64::NAN,
            vec![0.]
        )))
        .is_err()
    );
    assert!(Operator::new(&ctx, 2, &[("QQ".into(), C64::new(1., 0.))]).is_err());
    assert!(Operator::new(&ctx, 2, &[("XX".into(), C64::new(f64::NAN, 0.))]).is_err());
    assert!(Encoding::new("parity", 4, [3, 0]).is_err());
    assert!(Encoding::new("other", 4, [1, 1]).is_err());
    assert!(Operator::new(
        &ctx,
        1,
        &[
            ("I".into(), C64::new(f64::MAX, 0.)),
            ("I".into(), C64::new(f64::MAX, 0.))
        ]
    )
    .is_err());
    assert_eq!(ctx.memory_used(), 0);
}
