use mandacaru_core::{
    math::{assignment, rigid, SpatialGrid},
    molecular::{response, MolecularIntegrals},
    quantum::{properties, SolveOptions, C64},
    Context, Geometry, LengthUnit,
};
use nalgebra::DMatrix;

#[test]
fn assignment_is_globally_optimal_and_internal_directions_annihilate_rigid_motions() {
    let context = Context::new(Some(1), 32 << 20).unwrap();
    {
        let cost = DMatrix::from_row_slice(
            4,
            4,
            &[
                4., 1., 3., 2., 2., 0., 5., 3., 3., 2., 2., 1., 7., 4., 2., 5.,
            ],
        );
        let result = assignment::minimum(&context, &cost).unwrap();
        let total = (0..4).map(|i| cost[(i, result[i])]).sum::<f64>();
        let mut best = f64::INFINITY;
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let indices = [a, b, c, d];
                        if (0..4).any(|i| (0..i).any(|j| indices[i] == indices[j])) {
                            continue;
                        }
                        best = best.min((0..4).map(|i| cost[(i, indices[i])]).sum());
                    }
                }
            }
        }
        assert_eq!(total, best);
        for (positions, count) in [
            (vec![[0., 0., 0.], [0., 0., 1.]], 1),
            (vec![[0., 0., 0.], [1., 0., 1.], [-1., 0., 1.]], 3),
        ] {
            let directions = rigid::internal(&context, &positions).unwrap();
            let q = &directions.matrix;
            assert_eq!(q.ncols(), count);
            assert!((q.transpose() * q - DMatrix::identity(count, count)).norm() < 1e-12);
            for c in 0..count {
                for axis in 0..3 {
                    assert!(
                        (0..positions.len())
                            .map(|a| q[(3 * a + axis, c)])
                            .sum::<f64>()
                            .abs()
                            < 1e-12
                    );
                    let first = (axis + 1) % 3;
                    let second = (axis + 2) % 3;
                    let torque = (0..positions.len())
                        .map(|a| {
                            positions[a][first] * q[(3 * a + second, c)]
                                - positions[a][second] * q[(3 * a + first, c)]
                        })
                        .sum::<f64>();
                    assert!(torque.abs() < 1e-12);
                }
            }
        }
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn native_geometry_response_matches_independently_reoptimized_total_energies() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let geometry = Geometry::new(
            vec![1, 1],
            vec![[4., 4., 3.3], [4., 4., 4.7]],
            [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
            [false; 3],
            LengthUnit::Bohr,
        )
        .unwrap();
        let grid = SpatialGrid::from_geometry(&geometry, [0.5; 3], false, false).unwrap();
        let integrals = MolecularIntegrals::new(
            &context,
            geometry,
            grid,
            r#"{"name":"HAO"}"#,
            "spectral",
            Some(1e-12),
            Default::default(),
        )
        .unwrap();
        let reference = integrals
            .mean_field("rhf", [1, 1], &Default::default())
            .unwrap();
        let model = reference.model(&integrals).unwrap();
        let problem = model
            .problem(&context, "parity_reduced", true, false, true)
            .unwrap();
        let solved = problem
            .solve(
                &SolveOptions {
                    adapt: true,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        let gradient = response::gradient(
            &integrals,
            &reference,
            &model,
            &problem,
            solved.state.complex128().unwrap(),
            &response::Options::default(),
        )
        .unwrap();
        assert!((gradient.energy - solved.energy).abs() < 1e-12);
        let central = response::gradient(
            &integrals,
            &reference,
            &model,
            &problem,
            solved.state.complex128().unwrap(),
            &response::Options {
                step: 1e-4,
                extrapolate: false,
                ..Default::default()
            },
        )
        .unwrap();
        for (a, b) in central
            .gradient
            .as_slice()
            .iter()
            .zip(gradient.gradient.as_slice())
        {
            assert!((a - b).abs() < 1e-7, "central {a} vs Richardson {b}");
        }
        assert!(central.estimate.is_none());
        let rdm = properties::spatial_rdms(
            &context,
            &problem.encoding,
            problem.register("test").unwrap(),
            solved.state.complex128().unwrap(),
        )
        .unwrap();
        assert_eq!(rdm.m, 2);
        let energy = |delta: f64| {
            let geometry = &integrals.geometry;
            let mut positions = geometry.positions().to_vec();
            positions[1][2] += delta;
            let shifted = Geometry::new(
                geometry.numbers().to_vec(),
                positions,
                *geometry.cell(),
                [false; 3],
                LengthUnit::Bohr,
            )
            .unwrap();
            let rebuilt = integrals.at_geometry(shifted).unwrap();
            let reference = rebuilt
                .mean_field("rhf", [1, 1], &Default::default())
                .unwrap();
            let problem = reference
                .model(&rebuilt)
                .unwrap()
                .problem(&context, "jordan_wigner", true, false, true)
                .unwrap();
            problem
                .solve(
                    &SolveOptions {
                        adapt: true,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
                .energy
        };
        let derivative = (energy(2e-4) - energy(-2e-4)) / 4e-4;
        assert!(
            (gradient.gradient.as_slice()[5] - derivative).abs() < 2e-6,
            "{} vs {derivative}",
            gradient.gradient.as_slice()[5]
        );
        let mut reversed = DMatrix::identity(3, 3).map(|v| C64::new(v, 0.));
        reversed.swap_columns(0, 1);
        reversed.column_mut(2).scale_mut(-1.);
        let (indices, phases) = response::match_orbitals(
            &context,
            &DMatrix::identity(3, 3).map(|v| C64::new(v, 0.)),
            &reversed,
            [2, 2],
            0.9,
        )
        .unwrap();
        assert_eq!(indices, [1, 0, 2]);
        assert_eq!(phases[2], C64::new(-1., 0.));
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn a_degenerate_pair_is_aligned_by_rotation_and_a_split_pair_keeps_its_own() {
    // Four orbitals, one occupied; orbitals 1 and 2 are an exactly degenerate empty pair (as the
    // pi virtuals of LiH). The displaced SCF returns that pair rotated by 0.9 rad, which an
    // assignment cannot undo: the aligned frame must be the reference one. When the pair is split,
    // its rotation is physical and stays (only the order and phase are matched).
    let context = Context::new(Some(1), 32 << 20).unwrap();
    let identity = DMatrix::identity(4, 4).map(|v| C64::new(v, 0.));
    let (c, s) = (0.9f64.cos(), 0.9f64.sin());
    let mut rotated = identity.clone();
    rotated[(1, 1)] = C64::new(c, 0.);
    rotated[(2, 1)] = C64::new(s, 0.);
    rotated[(1, 2)] = C64::new(-s, 0.);
    rotated[(2, 2)] = C64::new(c, 0.);
    let aligned = |labels: &[f64]| {
        let alignment = response::align_orbitals(
            &context, &identity, &rotated, [1, 1], 0.5, Some(labels), Some(labels),
        )
        .unwrap();
        // The aligned frame: column p is sum_a rotated_a W[a, p].
        DMatrix::from_fn(4, 4, |row, p| {
            alignment.columns[p].iter().map(|&(a, w)| rotated[(row, a)] * w).sum::<C64>()
        })
    };
    let degenerate = aligned(&[-0.5, 0.26, 0.26, 1.2]);
    assert!((&degenerate - &identity).norm() < 1e-12, "{degenerate}");
    let split = aligned(&[-0.5, 0.20, 0.26, 1.2]);
    assert!((&split - &identity).norm() > 0.5, "a split pair keeps its physical rotation");
    // Without labels the frame is matched by order and phase only, as before.
    let unlabeled = response::align_orbitals(&context, &identity, &rotated, [1, 1], 0.5, None, None).unwrap();
    assert!(unlabeled.columns.iter().all(|column| column.len() == 1));
    // An occupied and an empty orbital of equal energy are never mixed.
    let crossing = response::align_orbitals(
        &context, &identity, &identity, [1, 1], 0.5, Some(&[0.3, 0.3, 0.5, 0.7]), Some(&[0.3, 0.3, 0.5, 0.7]),
    )
    .unwrap();
    assert!(crossing.columns.iter().all(|column| column.len() == 1));
    assert_eq!(context.memory_used(), 0);
}

/// H2 on a coarse grid, solved with ADAPT in `parity_reduced`: the response route's fixture.
fn h2_adapt(
    context: &Context,
) -> (MolecularIntegrals, mandacaru_core::molecular::MeanField, std::sync::Arc<mandacaru_core::molecular::transform::Model>) {
    let geometry = Geometry::new(
        vec![1, 1],
        vec![[4., 4., 3.3], [4., 4., 4.7]],
        [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
        [false; 3],
        LengthUnit::Bohr,
    )
    .unwrap();
    let grid = SpatialGrid::from_geometry(&geometry, [0.5; 3], false, false).unwrap();
    let integrals = MolecularIntegrals::new(
        context,
        geometry,
        grid,
        r#"{"name":"HAO"}"#,
        "spectral",
        Some(1e-12),
        Default::default(),
    )
    .unwrap();
    let reference = integrals.mean_field("rhf", [1, 1], &Default::default()).unwrap();
    let model = reference.model(&integrals).unwrap();
    (integrals, reference, model)
}

#[test]
fn the_lagrangian_correction_makes_the_force_of_an_unconverged_state_second_order() {
    use mandacaru_core::quantum::lagrangian::Ansatz;
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let (integrals, reference, model) = h2_adapt(&context);
        let problem = model.problem(&context, "parity_reduced", true, false, true).unwrap();
        let solved = problem
            .solve(&SolveOptions { adapt: true, ..Default::default() }, None)
            .unwrap();
        assert!(!solved.parameters.is_empty());
        let force = |theta: &[f64], with_ansatz: bool| {
            let state = problem.prepare_state(&solved.indices, theta, true).unwrap();
            response::gradient(
                &integrals,
                &reference,
                &model,
                &problem,
                state.complex128().unwrap(),
                &response::Options {
                    ansatz: with_ansatz.then_some(Ansatz {
                        indices: &solved.indices,
                        parameters: theta,
                        product_form: true,
                    }),
                    step: 1e-3,
                    internal: true,
                    extrapolate: false,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        // The converged state: the exact optimum, whose force is the reference.
        let at_optimum = force(&solved.parameters, true);
        let exact = at_optimum.gradient.as_slice().to_vec();
        let miss = |gradient: &[f64]| {
            gradient.iter().zip(&exact).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt()
        };
        // Off the optimum by delta, along the first parameter and along all of them.
        let mut errors = Vec::new();
        for size in [0.005, 0.01] {
            let theta: Vec<f64> = solved
                .parameters
                .iter()
                .enumerate()
                .map(|(i, t)| t + size * if i % 2 == 0 { 1. } else { -0.6 })
                .collect();
            let moved = force(&theta, true);
            let response::Stationarity::Lagrangian { gradient_norm, z_norm, correction_norm, truncated, .. } =
                moved.stationarity
            else {
                panic!("an off-optimum state must be corrected");
            };
            assert!(!truncated && gradient_norm > 1e-3);
            // Newton: |z| is |delta| to first order.
            assert!((z_norm / (size * theta.len() as f64).max(size)).is_finite());
            assert!(correction_norm > 1e-5);
            // The uncorrected gradient is what the same state gives with no ansatz at all.
            let bare = force(&theta, false);
            for (a, b) in bare.gradient.as_slice().iter().zip(moved.uncorrected.as_slice()) {
                assert!((a - b).abs() < 1e-12);
            }
            errors.push((miss(moved.uncorrected.as_slice()), miss(moved.gradient.as_slice()), gradient_norm));
        }
        let [(first_plain, first_corrected, first_g), (second_plain, second_corrected, second_g)] =
            errors[..]
        else {
            unreachable!()
        };
        // |g| doubles: the uncorrected miss is first order in |g|, the corrected one second order.
        assert!((second_g / first_g - 2.).abs() < 0.1, "{first_g} {second_g}");
        assert!((second_plain / first_plain - 2.).abs() < 0.3, "{first_plain} {second_plain}");
        assert!(second_corrected / first_corrected > 3. && second_corrected / first_corrected < 5.,
            "{first_corrected} {second_corrected}");
        assert!(first_corrected < first_plain / 10. && second_corrected < second_plain / 5.);
        // The optimum itself: the correction is below the force's own step noise.
        // A converged state sits under the floor: nothing is solved and nothing is added.
        assert!(matches!(
            at_optimum.stationarity,
            response::Stationarity::Stationary { gradient_norm } if gradient_norm < 1e-10
        ));
        assert_eq!(at_optimum.gradient.as_slice(), at_optimum.uncorrected.as_slice());
        // Forcing the solve at the optimum adds nothing measurable either (< 1e-9 eV/A).
        let state = problem.prepare_state(&solved.indices, &solved.parameters, true).unwrap();
        let forced = response::gradient(
            &integrals,
            &reference,
            &model,
            &problem,
            state.complex128().unwrap(),
            &response::Options {
                ansatz: Some(Ansatz {
                    indices: &solved.indices,
                    parameters: &solved.parameters,
                    product_form: true,
                }),
                lagrangian: mandacaru_core::quantum::lagrangian::Options { floor: 0., ..Default::default() },
                step: 1e-3,
                internal: true,
                extrapolate: false,
                ..Default::default()
            },
        )
        .unwrap();
        let response::Stationarity::Lagrangian { correction_norm, .. } = forced.stationarity else {
            panic!("a zero floor forces the solve");
        };
        assert!(correction_norm < 1e-11, "{correction_norm}");
        // A state that is not the ansatz's own is refused by name.
        let moved: Vec<f64> = solved.parameters.iter().map(|t| t + 0.3).collect();
        let error = response::gradient(
            &integrals,
            &reference,
            &model,
            &problem,
            state.complex128().unwrap(),
            &response::Options {
                ansatz: Some(Ansatz { indices: &solved.indices, parameters: &moved, product_form: true }),
                ..Default::default()
            },
        )
        .err()
        .expect("a mismatched ansatz is refused");
        assert!(error.to_string().contains("not the one the operator sequence"), "{error}");
        // Past the maximum of the parameter surface the Hessian is negative: refused by name.
        let past: Vec<f64> = solved.parameters.iter().map(|t| t + 1.4).collect();
        let state = problem.prepare_state(&solved.indices, &past, true).unwrap();
        let error = response::gradient(
            &integrals,
            &reference,
            &model,
            &problem,
            state.complex128().unwrap(),
            &response::Options {
                ansatz: Some(Ansatz { indices: &solved.indices, parameters: &past, product_form: true }),
                ..Default::default()
            },
        )
        .err()
        .expect("an indefinite parameter Hessian is refused");
        assert!(error.to_string().contains("negative curvature"), "{error}");
    }
    assert_eq!(context.memory_used(), 0);
}
