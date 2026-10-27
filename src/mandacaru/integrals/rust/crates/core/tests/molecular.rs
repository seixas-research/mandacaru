use mandacaru_core::{
    molecular::{scf, MolecularIntegrals, ScfOptions, SpatialGrid},
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
            .model
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
fn lowdin_orbitals_and_eri_obey_independent_symmetries() {
    let ctx = Context::new(Some(1), 64 * 1024 * 1024).unwrap();
    let integrals = integrals(&ctx);
    let m = integrals.m();
    let g = integrals.grid.index.points();
    for p in 0..m {
        for q in 0..m {
            let overlap: f64 = (0..g)
                .map(|k| {
                    integrals.orthonormal_samples.as_slice()[p * g + k]
                        * integrals.orthonormal_samples.as_slice()[q * g + k]
                })
                .sum::<f64>()
                * integrals.grid.index.dv();
            assert!((overlap - f64::from(p == q)).abs() < 1e-12);
            for r in 0..m {
                for s in 0..m {
                    let a = integrals.eri.value(p, q, r, s);
                    assert!((a - integrals.eri.value(r, q, p, s)).abs() < 1e-14);
                    assert!((a - integrals.eri.value(q, p, s, r)).abs() < 1e-14);
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
    let reduced = reference.model.reduce(&ctx, &[0], &[1]).unwrap();
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
                .model
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
    let (energies, c) = scf::eigen(h.clone()).unwrap();
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
    let factors = packed.cholesky(&ctx, 1e-13).unwrap();
    let actual = transform::four(&ctx, &factors, [&c; 4]).unwrap();
    assert!(expected
        .as_slice()
        .iter()
        .zip(actual.as_slice())
        .all(|(a, b)| (a - b).norm() < 1e-12));
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
    assert_eq!(ctx.memory_used(), live);
    drop(integrals);
    assert_eq!(ctx.memory_used(), 0);
}
