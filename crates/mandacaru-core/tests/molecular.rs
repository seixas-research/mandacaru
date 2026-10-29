use mandacaru_core::{
    math::SpatialGrid,
    molecular::{MolecularIntegrals, ScfOptions},
    quantum::{SolveOptions, C64},
    Context, Error, Geometry, LengthUnit,
};
fn geometry() -> Geometry {
    Geometry::new(
        vec![1, 1],
        vec![[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]],
        [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
        [false; 3],
        LengthUnit::Angstrom,
    )
    .unwrap()
}
fn integrals(ctx: &Context) -> MolecularIntegrals {
    let geometry = geometry();
    let grid =
        SpatialGrid::from_geometry(&geometry, [0.35 / 0.529177210903; 3], false, false).unwrap();
    MolecularIntegrals::new(ctx, geometry, grid, r#"{"name":"HAO"}"#, "spectral", None).unwrap()
}
#[test]
fn displaced_coulomb_factors_follow_reference_pivots_and_match_independent_columns() {
    use mandacaru_core::molecular::integrals::pair_integrals;
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    {
        let geometry = geometry();
        let grid = SpatialGrid::from_geometry(&geometry, [0.9; 3], false, false).unwrap();
        let reference = MolecularIntegrals::new(
            &ctx,
            geometry,
            grid,
            r#"{"name":"HAO","virtual_orbitals":1}"#,
            "spectral",
            Some(1e-2),
        )
        .unwrap();
        let original = reference.eri().unwrap();
        let pivots = original.pivots().unwrap();
        let m = reference.m();
        let pairs: Vec<_> = (0..m).flat_map(|a| (0..=a).map(move |b| (a, b))).collect();
        assert!(!pivots.is_empty());
        assert!(
            pivots.len() < pairs.len(),
            "exercise an incomplete factorization"
        );
        for shift in [0., 2e-3, -2e-3] {
            let g = &reference.geometry;
            let mut positions = g.positions().to_vec();
            positions[1][2] += shift;
            let displaced = Geometry::new(
                g.numbers().to_vec(),
                positions,
                *g.cell(),
                g.pbc(),
                LengthUnit::Bohr,
            )
            .unwrap();
            let rebuilt = reference.at_geometry_with_fixed_pivots(displaced).unwrap();
            let eri = rebuilt.eri().unwrap();
            assert_eq!(eri.pivots().unwrap(), pivots);
            let exact = pair_integrals(
                &ctx,
                &rebuilt.grid,
                &rebuilt.orthonormal_samples.read(&ctx).unwrap(),
                m,
            )
            .unwrap();
            let mut columns: Vec<Vec<f64>> = Vec::new();
            for &pivot in pivots {
                let (a, b) = pairs[pivot];
                let diagonal =
                    exact.value(a, a, b, b) - columns.iter().map(|v| v[pivot].powi(2)).sum::<f64>();
                let column = pairs
                    .iter()
                    .enumerate()
                    .map(|(i, &(p, q))| {
                        (exact.value(p, a, q, b)
                            - columns.iter().map(|v| v[i] * v[pivot]).sum::<f64>())
                            / diagonal.sqrt()
                    })
                    .collect();
                columns.push(column);
            }
            for (i, &(a, b)) in pairs.iter().enumerate() {
                for (j, &(p, q)) in pairs.iter().enumerate() {
                    let expected = columns.iter().map(|v| v[i] * v[j]).sum::<f64>();
                    assert!((eri.value(a, p, b, q) - expected).abs() < 1e-11);
                    if shift == 0. {
                        assert!((eri.value(a, p, b, q) - original.value(a, p, b, q)).abs() < 1e-11);
                    }
                }
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
fn geometry_to_rhf_and_quantum_h2_matches_the_portable_reference() {
    let ctx = Context::new(Some(1), 64 * 1024 * 1024).unwrap();
    let integrals = integrals(&ctx);
    assert_eq!(integrals.grid.index.shape(), [15; 3]);
    assert!((integrals.overlap.as_slice()[0] - 1.0234862568141794).abs() < 1e-12);
    let reference = integrals
        .mean_field("rhf", [1, 1], &Default::default())
        .unwrap();
    assert!(reference.converged);
    assert!((reference.electronic_energy + integrals.nuclear + 0.9371240086056045).abs() < 1e-12);
    for mapping in ["jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced"] {
        let problem = reference
            .model(&integrals)
            .unwrap()
            .problem(&ctx, mapping, true, false, true)
            .unwrap();
        for adapt in [false, true] {
            let result = problem
                .solve(
                    &SolveOptions {
                        adapt,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            assert!(result.success);
            assert!((result.energy + 0.9507746213870719).abs() < 1e-12);
        }
    }
}

#[test]
fn lazy_hf_factorization_and_scf_fit_the_same_budget_as_explicit_sequential_operations() {
    let measure = Context::new(Some(1), 64 << 20).unwrap();
    let explicit = integrals(&measure);
    explicit.eri().unwrap();
    let reference = explicit
        .mean_field("rhf", [1, 1], &Default::default())
        .unwrap();
    let expected = reference.electronic_energy;
    let limit = measure.memory_peak();
    drop(reference);
    drop(explicit);
    assert_eq!(measure.memory_used(), 0);
    let context = Context::new(Some(1), limit).unwrap();
    {
        let lazy = integrals(&context);
        assert!(!lazy.has_eri());
        let actual = lazy.mean_field("rhf", [1, 1], &Default::default()).unwrap();
        assert!(actual.converged);
        assert!((actual.electronic_energy - expected).abs() < 1e-12);
        let continued = lazy
            .continue_mean_field(&actual, &Default::default())
            .unwrap();
        assert!(continued.converged);
        assert!((continued.electronic_energy - expected).abs() < 1e-11);
    }
    assert_eq!(context.memory_used(), 0);
    assert!(context.memory_peak() <= limit);
}
#[test]
fn lowdin_orbitals_and_eri_obey_independent_symmetries() {
    let ctx = Context::new(Some(1), 64 * 1024 * 1024).unwrap();
    let integrals = integrals(&ctx);
    let m = integrals.m();
    let g = integrals.grid.index.points();
    let phi = integrals.orthonormal_samples.read(&ctx).unwrap();
    for p in 0..m {
        for q in 0..m {
            let overlap: f64 = (0..g).map(|k| phi[p * g + k] * phi[q * g + k]).sum::<f64>()
                * integrals.grid.index.dv();
            assert!((overlap - f64::from(p == q)).abs() < 1e-12);
            for r in 0..m {
                for s in 0..m {
                    let a = integrals.eri().unwrap().value(p, q, r, s);
                    assert!((a - integrals.eri().unwrap().value(r, q, p, s)).abs() < 1e-14);
                    assert!((a - integrals.eri().unwrap().value(q, p, s, r)).abs() < 1e-14);
                }
            }
        }
    }
}
#[test]
fn frozen_core_and_cholesky_keep_the_reference_energy() {
    let ctx = Context::new(Some(1), 64 * 1024 * 1024).unwrap();
    let integrals = integrals(&ctx);
    let reference = integrals
        .mean_field("rhf", [1, 1], &Default::default())
        .unwrap();
    let reduced = reference
        .model(&integrals)
        .unwrap()
        .reduce(&ctx, &[0], &[1])
        .unwrap();
    assert_eq!(reduced.particles, [0, 0]);
    assert_eq!(reduced.orbital_indices, [1]);
    let energy = reduced
        .problem(&ctx, "jordan_wigner", true, false, true)
        .unwrap()
        .solve(&Default::default(), None)
        .unwrap()
        .energy;
    assert!((energy - reference.electronic_energy - integrals.nuclear).abs() < 1e-12);
    let geometry = geometry();
    let grid =
        SpatialGrid::from_geometry(&geometry, [0.35 / 0.529177210903; 3], false, false).unwrap();
    let chol = MolecularIntegrals::new(
        &ctx,
        geometry,
        grid,
        r#"{"name":"HAO"}"#,
        "spectral",
        Some(1e-11),
    )
    .unwrap();
    let result = chol.mean_field("rhf", [1, 1], &Default::default()).unwrap();
    assert!((result.electronic_energy - reference.electronic_energy).abs() < 1e-11);
}
#[test]
fn selected_models_keep_their_rotation_and_release_budget_after_the_reference_is_dropped() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    {
        let geometry = geometry();
        let grid = SpatialGrid::from_geometry(&geometry, [0.95; 3], false, false).unwrap();
        let integrals = MolecularIntegrals::new(
            &ctx,
            geometry,
            grid,
            r#"{"name":"HAO","virtual_orbitals":1}"#,
            "spectral",
            Some(1e-10),
        )
        .unwrap();
        let reference = integrals
            .mean_field("rhf", [1, 1], &Default::default())
            .unwrap();
        let used = ctx.memory_used();
        {
            let reservation = ctx.reserve(ctx.memory_limit() - used - 1).unwrap();
            let held = ctx.memory_used();
            assert!(matches!(
                reference.active_model(&integrals, r#"{"method":"mp2","orbitals":3}"#),
                Err(Error::Budget { .. })
            ));
            assert_eq!(ctx.memory_used(), held);
            assert!(!reference.has_model());
            drop(reservation);
        }
        assert_eq!(ctx.memory_used(), used);
        let selected = reference
            .active_model(&integrals, r#"{"method":"mp2","orbitals":3}"#)
            .unwrap();
        assert!(!reference.has_model());
        assert_eq!(selected.m, 3);
        let reduced = selected.reduce(&ctx, &[0], &[1, 2]).unwrap();
        drop(selected);
        drop(reference);
        drop(integrals);
        let rotation = reduced.orbital_rotation.as_ref().unwrap();
        assert_eq!(rotation.matrix.shape(), (4, 4));
        assert_eq!(reduced.orbital_indices, [1, 2]);
        assert_eq!(reduced.frozen_indices, [0]);
        assert!(ctx.memory_used() > 0);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn open_shell_and_spinor_mean_fields_have_consistent_energies() {
    let ctx = Context::new(Some(1), 64 * 1024 * 1024).unwrap();
    let integrals = integrals(&ctx);
    for method in ["uhf", "ghf"] {
        let result = integrals
            .mean_field(method, [1, 1], &Default::default())
            .unwrap();
        assert!(result.converged);
        assert!((result.electronic_energy + integrals.nuclear + 0.9371240086056045).abs() < 1e-12);
        if method == "ghf" {
            assert!(result.kramers_pairing < 1e-10);
            assert!(result
                .model(&integrals)
                .unwrap()
                .problem(&ctx, "jordan_wigner", true, false, true)
                .is_err());
        }
    }
}
#[test]
fn complex_hermitian_eigensolver_preserves_the_matrix_equation() {
    let h = nalgebra::DMatrix::from_row_slice(
        2,
        2,
        &[
            C64::new(1., 0.),
            C64::new(0., 1.),
            C64::new(0., -1.),
            C64::new(2., 0.),
        ],
    );
    let (energies, c) = mandacaru_core::math::linalg::eigen(h.clone()).unwrap();
    assert!((energies[0] - (3. - 5f64.sqrt()) / 2.).abs() < 1e-13);
    assert!(
        (&h * &c
            - &c * nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
                2,
                energies.into_iter().map(|e| C64::new(e, 0.))
            )))
        .norm()
            < 1e-12
    );
    assert!((c.adjoint() * c - nalgebra::DMatrix::identity(2, 2)).norm() < 1e-12);
}

#[test]
fn factor_mo_transform_matches_full_contraction_for_complex_orbitals() {
    use mandacaru_core::molecular::{integrals::Eri, transform};
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    let mut values = ctx.allocate::<f64>(6).unwrap();
    values
        .as_mut_slice()
        .copy_from_slice(&[2., 0.3, 1., 0.1, 0.2, 0.8]);
    let packed = Eri::Packed { m: 2, values };
    let c = nalgebra::DMatrix::from_row_slice(
        2,
        2,
        &[
            C64::new(1., 0.),
            C64::new(0., 1.),
            C64::new(0., 1.),
            C64::new(1., 0.),
        ],
    ) * C64::new(1. / 2f64.sqrt(), 0.);
    let expected = transform::four(&ctx, &packed, [&c; 4]).unwrap();
    let occupied = c.columns(0, 1).into_owned();
    let virtuals = c.columns(1, 1).into_owned();
    let rectangular = transform::four(&ctx, &packed, [&occupied, &c, &virtuals, &c]).unwrap();
    let factors = packed.cholesky(&ctx, 1e-13).unwrap();
    let actual = transform::four(&ctx, &factors, [&c; 4]).unwrap();
    assert!(expected
        .as_slice()
        .iter()
        .zip(actual.as_slice())
        .all(|(a, b)| (a - b).norm() < 1e-12));
    for eri in [&factors] {
        let block = transform::four(&ctx, eri, [&occupied, &c, &virtuals, &c]).unwrap();
        assert_eq!(block.as_slice().len(), 4);
        assert!(block
            .as_slice()
            .iter()
            .zip(rectangular.as_slice())
            .all(|(a, b)| (a - b).norm() < 1e-12));
        for q in 0..2 {
            for s in 0..2 {
                assert!(
                    (block.as_slice()[q * 2 + s] - expected.as_slice()[(q * 2 + 1) * 2 + s]).norm()
                        < 1e-12
                );
            }
        }
    }
}
#[test]
fn errors_and_budget_exhaustion_release_molecular_storage() {
    let tiny = Context::new(Some(1), 1024).unwrap();
    let geometry = geometry();
    let grid = SpatialGrid::from_geometry(&geometry, [0.5; 3], false, false).unwrap();
    assert!(matches!(
        MolecularIntegrals::new(&tiny, geometry, grid, r#"{"name":"HAO"}"#, "spectral", None),
        Err(Error::Budget { .. })
    ));
    assert_eq!(tiny.memory_used(), 0);
    let ctx = Context::new(Some(1), 64 * 1024 * 1024).unwrap();
    let integrals = integrals(&ctx);
    let live = ctx.memory_used();
    assert!(integrals
        .mean_field("rhf", [1, 0], &Default::default())
        .is_err());
    assert_eq!(ctx.memory_used(), live);
    let failed = integrals
        .mean_field(
            "rhf",
            [1, 1],
            &ScfOptions {
                max_iterations: 1,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(!failed.converged);
    assert_eq!(failed.iterations.len(), 1);
    drop(failed);
    assert!(integrals.has_eri());
    assert!(ctx.memory_used() > live); // Reusable factors belong to the geometry.
    drop(integrals);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn semilocal_scf_and_snapshots_do_not_construct_a_hamiltonian() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let integrals = integrals(&ctx);
    assert!(!integrals.has_eri());
    let state = integrals
        .kohn_sham("pbe", [1, 1], false, &Default::default())
        .unwrap();
    assert!(state.converged);
    assert!(!integrals.has_eri());
    assert!(!state.has_model());
    let model = state.model(&integrals).unwrap();
    assert!(state.has_model());
    assert!(integrals.has_eri());
    assert_eq!(model.two.as_slice().len(), 16);
    assert!(std::sync::Arc::ptr_eq(
        &model,
        &state.model(&integrals).unwrap()
    ));
    drop(model);
    drop(state);
    drop(integrals);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn column_cholesky_matches_independently_built_coulomb_pairs() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let integrals = integrals(&ctx);
    let exact = mandacaru_core::molecular::integrals::pair_integrals(
        &ctx,
        &integrals.grid,
        &integrals.orthonormal_samples.read(&ctx).unwrap(),
        integrals.m(),
    )
    .unwrap();
    let factorized = integrals.eri().unwrap();
    assert!(matches!(
        factorized.as_ref(),
        mandacaru_core::molecular::integrals::Eri::Cholesky { .. }
    ));
    for p in 0..2 {
        for q in 0..2 {
            for r in 0..2 {
                for s in 0..2 {
                    assert!((exact.value(p, q, r, s) - factorized.value(p, q, r, s)).abs() < 1e-10);
                }
            }
        }
    }
}
