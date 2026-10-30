use mandacaru_core::{
    quantum::{pools, CachedProblem, Encoding, Operator, SolveOptions, C64},
    Context,
};

#[test]
fn every_pool_is_antihermitian_and_conserving_pools_stay_in_the_physical_sector() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    for mapping in ["jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced"] {
        let encoding = Encoding::new(mapping, 6, [1, 1]).unwrap();
        for name in ["fermionic", "qeb", "ceo", "ceo-ovp"] {
            let pool = pools::build(&ctx, &encoding, name, true).unwrap();
            let basis = mandacaru_core::quantum::Basis::new(&ctx, &encoding, true).unwrap();
            assert!(pool.screenable.iter().any(|v| *v));
            for generator in pool.operators {
                basis.check_invariant(&generator.operator).unwrap();
                let sum = generator
                    .operator
                    .add(
                        &ctx,
                        &generator.operator.dagger(&ctx).unwrap(),
                        C64::new(1., 0.),
                    )
                    .unwrap();
                assert!(sum.terms.is_empty());
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn qubit_generators_require_an_untapered_full_register() {
    let ctx = Context::new(Some(1), 16 << 20).unwrap();
    let encoding = Encoding::new("jordan_wigner", 6, [1, 1]).unwrap();
    let pool = pools::build(&ctx, &encoding, "qubit", true).unwrap();
    let basis = mandacaru_core::quantum::Basis::new(&ctx, &encoding, true).unwrap();
    assert!(pool
        .operators
        .iter()
        .any(|g| basis.check_invariant(&g.operator).is_err()));
    let reduced = Encoding::new("parity_reduced", 4, [1, 1]).unwrap();
    assert!(pools::build(&ctx, &reduced, "qubit", true).is_err());
}

#[test]
fn spin_orbit_rotations_conserve_total_number_and_include_complex_spin_flips() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    for mapping in ["jordan_wigner", "parity", "bravyi_kitaev"] {
        let encoding = Encoding::new(mapping, 4, [1, 1]).unwrap();
        let pool = pools::build(&ctx, &encoding, "spin-orbit", true).unwrap();
        assert_eq!(pool.operators.len(), 10);
        let mut number = Operator::new(&ctx, 4, &[]).unwrap();
        for p in 0..4 {
            number = number
                .add(
                    &ctx,
                    &encoding.monomial(&ctx, &[(p, true), (p, false)]).unwrap(),
                    C64::new(1., 0.),
                )
                .unwrap();
        }
        let sector = mandacaru_core::quantum::Basis::new(&ctx, &encoding, true).unwrap();
        assert!(pool
            .operators
            .iter()
            .any(|g| sector.check_invariant(&g.operator).is_err()));
        for g in &pool.operators {
            assert!(g.operator.is_antihermitian());
            let left = g.operator.compose(&ctx, &number).unwrap();
            let right = number.compose(&ctx, &g.operator).unwrap();
            assert!(left
                .add(&ctx, &right, C64::new(-1., 0.))
                .unwrap()
                .terms
                .is_empty());
        }
        assert!(pool.operators.iter().any(|g| g.label == "iS(0->3)"));
        assert_eq!(
            pools::build(&ctx, &encoding, "spin-orbit", false)
                .unwrap()
                .operators
                .len(),
            2
        );
    }
    let reduced = Encoding::new("parity_reduced", 4, [1, 1]).unwrap();
    assert!(pools::build(&ctx, &reduced, "spin-orbit", true).is_err());
    drop(reduced);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn complex_spin_mixing_reaches_the_independent_one_electron_ground_state() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    for mapping in ["jordan_wigner", "parity", "bravyi_kitaev"] {
        let encoding = Encoding::new(mapping, 4, [1, 0]).unwrap();
        let mut h = Operator::new(&ctx, 4, &[]).unwrap();
        for (p, e) in [-0.7, 0.8, 0.3, 0.9].into_iter().enumerate() {
            h = h
                .add(
                    &ctx,
                    &encoding.monomial(&ctx, &[(p, true), (p, false)]).unwrap(),
                    C64::new(e, 0.),
                )
                .unwrap();
        }
        for (p, q, c) in [(2, 0, C64::new(0., 0.2)), (0, 2, C64::new(0., -0.2))] {
            h = h
                .add(
                    &ctx,
                    &encoding.monomial(&ctx, &[(p, true), (q, false)]).unwrap(),
                    c,
                )
                .unwrap();
        }
        let mut p = CachedProblem::new(&ctx, encoding, h, false, false, true).unwrap();
        p.use_pool("spin-orbit", true).unwrap();
        let result = p
            .solve(
                &SolveOptions {
                    adapt: true,
                    max_growth: 8,
                    gradient_tolerance: Some(1e-9),
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        assert!(result.success, "{mapping}: {}", result.message);
        assert!((result.energy - (-0.2 - 0.29_f64.sqrt())).abs() < 1e-11);
        assert!(result.labels.iter().any(|l| l.starts_with("iS")));
        let mut beta = 0.;
        for (&encoded, amplitude) in p
            .basis
            .indices
            .as_slice()
            .iter()
            .zip(result.state.complex128().unwrap())
        {
            let occupation = p.encoding.decode(encoded);
            if occupation.count_ones() != 1 {
                assert!(amplitude.norm() < 1e-12);
            }
            beta += (occupation & 3).count_ones() as f64 * amplitude.norm_sqr();
        }
        assert!(beta > 0.01);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn adaptive_exchange_pools_solve_a_rotated_two_site_ground_state() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    for name in ["fermionic", "qeb", "ceo", "ceo-ovp"] {
        let encoding = Encoding::new("jordan_wigner", 4, [1, 1]).unwrap();
        let one = [[-0.7, -0.2], [-0.2, 0.4]];
        let mut terms = vec![];
        for spin in 0..2 {
            for (i, row) in one.iter().enumerate() {
                for (j, &value) in row.iter().enumerate() {
                    terms.push((
                        vec![(2 * spin + i, true), (2 * spin + j, false)],
                        C64::new(value, 0.),
                    ));
                }
            }
        }
        let mut h = Operator::new(&ctx, encoding.n_qubits(), &[]).unwrap();
        for (ladders, coefficient) in terms {
            h = h
                .add(
                    &ctx,
                    &encoding.monomial(&ctx, &ladders).unwrap(),
                    coefficient,
                )
                .unwrap();
        }
        let mut problem = CachedProblem::new(&ctx, encoding, h, true, false, true).unwrap();
        problem.use_pool(name, true).unwrap();
        let result = problem
            .solve(
                &SolveOptions {
                    adapt: true,
                    max_growth: 8,
                    gradient_tolerance: Some(1e-8),
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        let exact = -0.3 - (1.21_f64 + 0.16).sqrt();
        assert!(result.success, "{name}: {}", result.message);
        assert!(
            (result.energy - exact).abs() < 1e-9,
            "{name}: {} vs {exact}",
            result.energy
        );
    }
}

#[test]
fn coupled_exchange_growth_expands_live_members_with_independent_parameters() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let encoding = Encoding::new("jordan_wigner", 4, [1, 1]).unwrap();
    let mut h = Operator::new(&ctx, 4, &[]).unwrap();
    for spin in 0..2 {
        for (i, j) in [(2 * spin, 2 * spin + 1), (2 * spin + 1, 2 * spin)] {
            h = h
                .add(
                    &ctx,
                    &encoding.monomial(&ctx, &[(i, true), (j, false)]).unwrap(),
                    C64::new(-0.8, 0.),
                )
                .unwrap();
        }
    }
    for site in 0..2 {
        h = h
            .add(
                &ctx,
                &encoding
                    .monomial(
                        &ctx,
                        &[
                            (site, true),
                            (site, false),
                            (site + 2, true),
                            (site + 2, false),
                        ],
                    )
                    .unwrap(),
                C64::new(3., 0.),
            )
            .unwrap();
    }
    let mut problem = CachedProblem::new(&ctx, encoding, h, true, false, true).unwrap();
    problem.use_pool("ceo", true).unwrap();
    let result = problem
        .solve(
            &SolveOptions {
                adapt: true,
                max_growth: 12,
                gradient_tolerance: Some(1e-8),
                ..Default::default()
            },
            None,
        )
        .unwrap();
    assert!(result.success, "{}", result.message);
    assert!((result.energy - (3. - (9. + 16. * 0.8_f64.powi(2)).sqrt()) / 2.).abs() < 1e-9);
    assert!(result.indices.iter().any(|&i| !problem.pool_screenable[i]));
    assert!(result.parameters.len() > result.iterations.len());
}
