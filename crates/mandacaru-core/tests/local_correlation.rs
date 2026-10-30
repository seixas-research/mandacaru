use mandacaru_core::{
    math::SpatialGrid,
    molecular::{correlation, local_correlation, MolecularIntegrals},
    Context, Geometry, LengthUnit,
};

#[test]
fn full_domains_and_zero_pno_cutoff_recover_canonical_mp2_energy_and_density() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    {
        let geometry = Geometry::new(
            vec![1; 4],
            vec![
                [3., 3., 1.95],
                [3., 3., 2.65],
                [3., 3., 3.35],
                [3., 3., 4.05],
            ],
            [[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]],
            [false; 3],
            LengthUnit::Angstrom,
        )
        .unwrap();
        let grid = SpatialGrid::from_geometry(&geometry, [0.9; 3], false, false).unwrap();
        let integrals = MolecularIntegrals::new(
            &ctx,
            geometry,
            grid,
            r#"{"name":"HAO"}"#,
            "spectral",
            Some(1e-10),
            Default::default(),
        )
        .unwrap();
        let reference = integrals
            .mean_field("rhf", [2, 2], &Default::default())
            .unwrap();
        assert!(reference.converged);
        let options = local_correlation::Options {
            full_domains: true,
            pno_cutoff: 0.,
            ..Default::default()
        };
        let local = local_correlation::calculate(&integrals, &reference, &[], &options).unwrap();
        assert!(!reference.has_model());
        let model = reference.model(&integrals).unwrap();
        let canonical =
            correlation::natural_orbitals(&ctx, 4, model.one.as_slice(), model.two.as_slice(), 2)
                .unwrap();
        assert!(
            (local.mp2.correlation_energy - canonical.correlation_energy).abs() < 1e-7,
            "local {} canonical {}",
            local.mp2.correlation_energy,
            canonical.correlation_energy
        );
        assert!(local
            .mp2
            .virtual_occupations
            .as_slice()
            .iter()
            .zip(canonical.virtual_occupations.as_slice())
            .all(|(a, b)| (a - b).abs() < 1e-6));
        assert!(local
            .mp2
            .occupied_density
            .as_slice()
            .iter()
            .zip(canonical.occupied_density.as_slice())
            .all(|(a, b)| (a - b).abs() < 1e-6));
        assert_eq!(local.pno_counts, [2, 2, 2]);
        assert!(local.iterations > 0);
        let frozen = local_correlation::calculate(&integrals, &reference, &[0], &options).unwrap();
        let reduced = model.reduce(&ctx, &[0], &[1, 2, 3]).unwrap();
        let canonical = correlation::natural_orbitals(
            &ctx,
            3,
            reduced.one.as_slice(),
            reduced.two.as_slice(),
            1,
        )
        .unwrap();
        assert!((frozen.mp2.correlation_energy - canonical.correlation_energy).abs() < 1e-7);
        assert!((frozen.mp2.occupied_occupations.as_slice()[0] - 2.).abs() < 1e-12);
    }
    assert_eq!(ctx.memory_used(), 0);
}
