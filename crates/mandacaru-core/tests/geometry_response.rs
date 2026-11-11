//! The analytic variational force (`molecular::response`): its refusals, the orbital matching
//! the transfer of an ansatz relies on, and the Lagrangian (Z-vector) correction of a state off
//! its optimum, on the synthetic PAW hydrogen molecule (`paw_forces_support`).
mod paw_forces_support;

use mandacaru_core::{
    math::{assignment, rigid, SpatialGrid},
    molecular::{response, MolecularIntegrals},
    quantum::{SolveOptions, C64},
    Context, Geometry, LengthUnit,
};
use nalgebra::DMatrix;
use paw_forces_support::synthetic::{integrals, solve, Method, H2};

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
fn an_all_electron_variational_force_is_refused_by_name_and_orbital_matching_follows_order_and_phase(
) {
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
        let error = response::gradient(
            &integrals,
            &reference,
            &model,
            &problem,
            solved.state.complex128().unwrap(),
            &response::Options::default(),
        )
        .err()
        .expect("an all-electron variational force is refused");
        assert!(
            matches!(error, mandacaru_core::Error::Unsupported(_)),
            "{error}"
        );
        assert!(error.to_string().contains("PAW-LCAO only"), "{error}");
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

/// The synthetic PAW H2 in its two lowest canonical orbitals (one ADAPT generator), the response
/// route's fixture.
fn h2_adapt(
    context: &Context,
) -> (
    MolecularIntegrals,
    mandacaru_core::molecular::MeanField,
    std::sync::Arc<mandacaru_core::molecular::transform::Model>,
) {
    let integrals = integrals(context, &H2, "lda");
    let reference = solve(&integrals, Method::Hf("rhf"), [1, 1]);
    let model = std::sync::Arc::new(
        reference
            .active_model(&integrals, r#"{"orbitals": 2}"#)
            .unwrap(),
    );
    (integrals, reference, model)
}

#[test]
fn the_lagrangian_correction_makes_the_force_of_an_unconverged_state_second_order() {
    use mandacaru_core::quantum::lagrangian::Ansatz;
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let (integrals, reference, model) = h2_adapt(&context);
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
                    internal: true,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        // The converged state: the exact optimum, whose force is the reference.
        let at_optimum = force(&solved.parameters, true);
        let at_optimum_bare = force(&solved.parameters, false);
        let exact = at_optimum.gradient.as_slice().to_vec();
        let miss = |gradient: &[f64]| {
            gradient
                .iter()
                .zip(&exact)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
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
            let response::Stationarity::Lagrangian {
                gradient_norm,
                z_norm,
                solve,
            } = &moved.stationarity
            else {
                panic!("an off-optimum state must be corrected");
            };
            assert!(solve.dropped == 0 && *gradient_norm > 1e-3);
            // Newton: |z| is |delta| to first order.
            assert!((z_norm / (size * theta.len() as f64).max(size)).is_finite());
            // The uncorrected gradient: the same state with no ansatz at all.
            let bare = force(&theta, false);
            let distance = |a: &[f64], b: &[f64]| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f64>()
                    .sqrt()
            };
            assert!(distance(moved.gradient.as_slice(), bare.gradient.as_slice()) > 1e-5);
            errors.push((
                miss(bare.gradient.as_slice()),
                miss(moved.gradient.as_slice()),
                *gradient_norm,
            ));
        }
        let [(first_plain, first_corrected, first_g), (second_plain, second_corrected, second_g)] =
            errors[..]
        else {
            unreachable!()
        };
        // |g| doubles: the uncorrected miss is first order in |g|, the corrected one second order.
        assert!(
            (second_g / first_g - 2.).abs() < 0.1,
            "{first_g} {second_g}"
        );
        assert!(
            (second_plain / first_plain - 2.).abs() < 0.3,
            "{first_plain} {second_plain}"
        );
        assert!(
            second_corrected / first_corrected > 3. && second_corrected / first_corrected < 5.,
            "{first_corrected} {second_corrected}"
        );
        assert!(first_corrected < first_plain / 10. && second_corrected < second_plain / 5.);
        // A converged state: under the floor nothing is solved and nothing is added; above it the
        // correction is below the force's noise.
        match at_optimum.stationarity {
            response::Stationarity::Stationary { gradient_norm } => {
                assert!(gradient_norm < 1e-10);
                assert_eq!(
                    at_optimum.gradient.as_slice(),
                    at_optimum_bare.gradient.as_slice()
                );
            }
            response::Stationarity::Lagrangian { gradient_norm, .. } => {
                let correction = (|a: &[f64], b: &[f64]| {
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| (x - y) * (x - y))
                        .sum::<f64>()
                        .sqrt()
                })(
                    at_optimum.gradient.as_slice(),
                    at_optimum_bare.gradient.as_slice(),
                );
                assert!(
                    gradient_norm < 1e-7 && correction < 1e-9,
                    "{gradient_norm:.3e} {correction:.3e}"
                );
            }
            response::Stationarity::Assumed => panic!("the ansatz was given"),
        }
        // Forcing the solve at the optimum adds nothing measurable either (< 5e-9 eV/A).
        let state = problem
            .prepare_state(&solved.indices, &solved.parameters, true)
            .unwrap();
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
                lagrangian: mandacaru_core::quantum::lagrangian::Options {
                    floor: 0.,
                    ..Default::default()
                },
                internal: true,
                ..Default::default()
            },
        )
        .unwrap();
        let response::Stationarity::Lagrangian { .. } = forced.stationarity else {
            panic!("a zero floor forces the solve");
        };
        let correction = (|a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt()
        })(
            forced.gradient.as_slice(),
            at_optimum_bare.gradient.as_slice(),
        );
        assert!(correction < 1e-10, "{correction}");
        // A state that is not the ansatz's own is refused by name.
        let moved: Vec<f64> = solved.parameters.iter().map(|t| t + 0.3).collect();
        let error = response::gradient(
            &integrals,
            &reference,
            &model,
            &problem,
            state.complex128().unwrap(),
            &response::Options {
                ansatz: Some(Ansatz {
                    indices: &solved.indices,
                    parameters: &moved,
                    product_form: true,
                }),
                ..Default::default()
            },
        )
        .err()
        .expect("a mismatched ansatz is refused");
        assert!(
            error
                .to_string()
                .contains("not the one the operator sequence"),
            "{error}"
        );
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
                ansatz: Some(Ansatz {
                    indices: &solved.indices,
                    parameters: &past,
                    product_form: true,
                }),
                ..Default::default()
            },
        )
        .err()
        .expect("an indefinite parameter Hessian is refused");
        assert!(error.to_string().contains("negative curvature"), "{error}");
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn a_redundant_parameter_is_dropped_and_leaves_the_force_of_the_single_parameter_ansatz() {
    use mandacaru_core::quantum::lagrangian::Ansatz;
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let (integrals, reference, model) = h2_adapt(&context);
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
        assert_eq!(solved.indices.len(), 1);
        let force = |indices: &[usize], theta: &[f64]| {
            let state = problem.prepare_state(indices, theta, true).unwrap();
            response::gradient(
                &integrals,
                &reference,
                &model,
                &problem,
                state.complex128().unwrap(),
                &response::Options {
                    ansatz: Some(Ansatz {
                        indices,
                        parameters: theta,
                        product_form: true,
                    }),
                    internal: true,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        // The same generator twice: only the sum of the two angles matters, so the parameter
        // Hessian has an exact null direction (1, -1).
        let total = solved.parameters[0] + 0.01;
        let single = force(&solved.indices, &[total]);
        let double = force(&[solved.indices[0]; 2], &[0.3 * total, 0.7 * total]);
        let (
            response::Stationarity::Lagrangian { solve: one, .. },
            response::Stationarity::Lagrangian { solve: two, .. },
        ) = (&single.stationarity, &double.stationarity)
        else {
            panic!("both states are off the optimum");
        };
        assert_eq!((one.parameters, one.dropped), (1, 0));
        assert_eq!((two.parameters, two.dropped), (2, 1));
        assert!(two.exact && two.lowest.abs() <= two.cutoff && two.dropped_gradient < 1e-6);
        for (a, b) in double
            .gradient
            .as_slice()
            .iter()
            .zip(single.gradient.as_slice())
        {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }
    assert_eq!(context.memory_used(), 0);
}
