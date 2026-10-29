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
            solved.state.as_slice(),
            &response::Options::default(),
        )
        .unwrap();
        assert!((gradient.energy - solved.energy).abs() < 1e-12);
        let rdm = properties::spatial_rdms(
            &context,
            &problem.encoding,
            &problem.basis,
            solved.state.as_slice(),
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
