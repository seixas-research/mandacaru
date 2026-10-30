use mandacaru_core::{
    periodic::{
        analysis,
        ewald::Ewald,
        spectrum::{self, Smearing},
        Crystal, Options,
    },
    quantum::C64,
    Context,
};
use std::f64::consts::PI;

#[test]
fn archived_periodic_state_preserves_energy_and_projected_dos_weight() {
    use mandacaru_core::math::storage::{ComplexSamples, MatrixSource, Precision, StoredMatrix};
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test/migration/phase5/si_crystal.json");
        let mut crystal = Crystal::load(&ctx, path.to_str().unwrap()).unwrap();
        let reference = crystal.solve(&Options::default()).unwrap();
        for point in &mut crystal.points {
            let archived = ComplexSamples::archived_copy(
                &ctx,
                &point.psi.read(&ctx).unwrap(),
                Precision::Float64,
            )
            .unwrap();
            point.psi = archived;
        }
        crystal.hats = ComplexSamples::archived_copy(
            &ctx,
            &crystal.hats.read(&ctx).unwrap(),
            Precision::Float64,
        )
        .unwrap();
        let mut state = crystal.solve(&Options::default()).unwrap();
        assert!(state.converged);
        assert!((state.free_energy - reference.free_energy).abs() < 1e-12);
        let axis: Vec<_> = (0..1601).map(|i| -1. + i as f64 / 400.).collect();
        let original = analysis::projected_dos(&crystal, &state, &axis, 0.04).unwrap();
        for matrix in state.coefficients.iter_mut().chain(&mut state.matrices) {
            let archived =
                StoredMatrix::new(&ctx, matrix.matrix(&ctx).unwrap().clone(), true).unwrap();
            *matrix = archived;
        }
        let archived = analysis::projected_dos(&crystal, &state, &axis, 0.04).unwrap();
        assert_eq!(original.as_slice(), archived.as_slice());
        let weights: Vec<_> = crystal.points.iter().map(|p| p.weight).collect();
        let total = spectrum::broaden(
            &ctx,
            &axis,
            &state.eigenvalues,
            &weights,
            0.04,
            crystal.degeneracy(),
            None,
        )
        .unwrap();
        for (partial, total) in archived
            .as_slice()
            .chunks(crystal.bands())
            .zip(total.as_slice())
        {
            assert!((partial.iter().sum::<f64>() - total).abs() < 2e-11);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn periodic_float32_sample_storage_preserves_float64_band_matrices() {
    use mandacaru_core::math::storage::Precision;
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test/migration/phase5/si_crystal.json");
        let source = std::fs::read_to_string(path).unwrap();
        let float64 = Crystal::from_json(&ctx, &source).unwrap();
        let compact = source.replacen('{', "{\"precision\":\"float32\",", 1);
        let float32 = Crystal::from_json(&ctx, &compact).unwrap();
        assert_eq!(float32.precision, Precision::Float32);
        for (a, b) in float64.points.iter().zip(&float32.points) {
            assert_eq!(b.psi.precision(), Precision::Float32);
            assert_eq!(a.fixed, b.fixed);
            assert_eq!(a.overlap, b.overlap);
        }
        let a = float64.solve(&Options::default()).unwrap();
        let b = float32.solve(&Options::default()).unwrap();
        assert!(a.converged && b.converged);
        assert!((a.extrapolated_energy - b.extrapolated_energy).abs() * 27.211386245988 < 1e-4);
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
fn silicon_crystal_solves_without_python_and_releases_owned_storage() {
    let ctx = Context::new(Some(1), 128 * 1024 * 1024).unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/phase5/si_crystal.json");
    let crystal = Crystal::load(&ctx, path.to_str().unwrap()).unwrap();
    let state = crystal
        .solve(&Options {
            width: 0.01 / 27.211386245988,
            ..Default::default()
        })
        .unwrap();
    assert!(state.converged);
    assert!((state.extrapolated_energy * 27.211386245988 + 211.60193975895757).abs() < 1e-6);
    let count = state
        .occupations
        .iter()
        .zip(&crystal.points)
        .map(|(f, k)| k.weight * f.iter().sum::<f64>())
        .sum::<f64>();
    assert!((count - 8.).abs() < 1e-10);
    drop(state);
    drop(crystal);
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
fn ewald_madelung_and_derivatives_follow_the_same_lattice_energy() {
    let ctx = Context::new(Some(1), 32 * 1024 * 1024).unwrap();
    let cell = [6., 0.3, 0.1, 0.2, 7., 0.4, 0.1, 0.2, 8.];
    let positions = [[0.7, 1., 1.2], [2.1, 3., 3.1]];
    let charges = [1., -2.];
    let energy = |r: &[[f64; 3]]| {
        Ewald::new(&ctx, r, &charges, cell, 1e-12, None)
            .unwrap()
            .evaluate(&ctx)
            .unwrap()
            .energy
    };
    let state = Ewald::new(&ctx, &positions, &charges, cell, 1e-12, None)
        .unwrap()
        .evaluate(&ctx)
        .unwrap();
    let mut plus = positions;
    let mut minus = positions;
    let eps = 1e-5;
    plus[0][1] += eps;
    minus[0][1] -= eps;
    assert!(
        (state.forces.as_slice()[1] + (energy(&plus) - energy(&minus)) / (2. * eps)).abs() < 1e-9
    );
    let unit = Ewald::new(
        &ctx,
        &[[0.; 3]],
        &[1.],
        [5., 0., 0., 0., 5., 0., 0., 0., 5.],
        1e-12,
        None,
    )
    .unwrap()
    .evaluate(&ctx)
    .unwrap();
    assert!((2. * unit.energy * 5. + 2.83729747948062).abs() < 1e-10);
}
#[test]
fn metal_smearing_conserves_particle_number_for_every_scheme() {
    let eps = vec![vec![-0.3, -0.01, 0.12, 0.4], vec![-0.2, 0.03, 0.17, 0.5]];
    let weights = [0.25, 0.75];
    for scheme in [
        Smearing::FermiDirac,
        Smearing::Gaussian,
        Smearing::MethfesselPaxton,
    ] {
        let mu = spectrum::fermi(&eps, &weights, 3.4, scheme, 0.06, 2.).unwrap();
        let count = eps
            .iter()
            .zip(weights)
            .map(|(e, w)| {
                2. * w
                    * e.iter()
                        .map(|e| scheme.occupation((e - mu) / 0.06))
                        .sum::<f64>()
            })
            .sum::<f64>();
        assert!((count - 3.4).abs() < 1e-10);
        assert!((scheme.occupation(1.7) + scheme.occupation(-1.7) - 1.).abs() < 1e-14);
    }
}
#[test]
fn reciprocal_mesh_reduction_preserves_orbit_weights() {
    let ctx = Context::new(Some(1), 1024 * 1024).unwrap();
    let rotation = [0, 1, 0, 1, 0, 0, 0, 0, 1];
    let mesh = spectrum::mesh(&ctx, [3, 3, 1], true, &[rotation], true).unwrap();
    assert_eq!(mesh.full.len(), 9);
    assert!(mesh.points.len() < 9);
    assert!((mesh.weights.iter().sum::<f64>() - 1.).abs() < 1e-14);
    assert!(spectrum::mesh(&ctx, [2, 3, 1], false, &[rotation], true).is_err());
}
#[test]
fn wilson_phase_detects_topological_winding_and_singular_links() {
    let ctx = Context::new(Some(1), 1024 * 1024).unwrap();
    let n = 300;
    for (hopping, topological) in [(0.5, true), (1.5, false)] {
        let states: Vec<_> = (0..n)
            .map(|i| {
                let k = 2. * PI * i as f64 / n as f64;
                let phase = (C64::new(hopping, 0.) + C64::from_polar(1., k)).arg();
                [
                    -C64::from_polar(1. / 2f64.sqrt(), phase),
                    C64::new(1. / 2f64.sqrt(), 0.),
                ]
            })
            .collect();
        let links: Vec<_> = (0..n)
            .map(|i| {
                states[i][0].conj() * states[(i + 1) % n][0]
                    + states[i][1].conj() * states[(i + 1) % n][1]
            })
            .collect();
        let (phase, minimum) = analysis::wilson_loop(&ctx, &links, n, 1, 1e-10).unwrap();
        assert!((phase.abs() - if topological { PI } else { 0. }).abs() < 1e-10);
        assert!(minimum > 0.99);
    }
    assert!(analysis::wilson_loop(&ctx, &[C64::default()], 1, 1, 1e-10).is_err());
}
