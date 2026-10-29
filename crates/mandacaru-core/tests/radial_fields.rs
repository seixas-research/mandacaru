use mandacaru_core::{
    math::{spline::Cubic, SpatialGrid},
    paw::fields,
    Context,
};
use nalgebra::{Matrix3, Vector3};

#[test]
fn core_kinetic_density_matches_an_independent_quadratic_density_limit() {
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    {
        let r: Vec<_> = (0..101).map(|i| i as f64 / 50.).collect();
        let rho: Vec<_> = r.iter().map(|r| (3. - r).powi(2)).collect();
        let profile = Cubic::new(
            &ctx,
            ctx.copy_slice(&r).unwrap(),
            ctx.copy_slice(&rho).unwrap(),
        )
        .unwrap();
        let kinetic = fields::core_kinetic(&ctx, &profile).unwrap();
        for at in [0., 0.123, 0.8, 1.9, 2.] {
            assert!((kinetic.value_slope(at).0 - 0.5).abs() < 1e-12);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn skew_periodic_radial_placement_matches_an_explicit_image_sum_and_is_lattice_periodic() {
    let ctx = Context::new(Some(1), 2 << 20).unwrap();
    {
        let shape = [5, 6, 7];
        let grid = SpatialGrid::new(
            shape,
            [-0.2, 0.1, 0.3],
            [0.6, 0.2, 0.1, 0., 0.5, 0.1, 0., 0., 0.5],
            true,
        )
        .unwrap();
        let cell = grid.step * Matrix3::from_diagonal(&Vector3::from_fn(|a, _| shape[a] as f64));
        let r: Vec<_> = (0..901).map(|i| i as f64 / 100.).collect();
        let rho: Vec<_> = r.iter().map(|r| (-r * r).exp()).collect();
        let profile = Cubic::new(
            &ctx,
            ctx.copy_slice(&r).unwrap(),
            ctx.copy_slice(&rho).unwrap(),
        )
        .unwrap();
        let center = Vector3::new(0.31, 0.27, 0.63);
        let support = r
            .iter()
            .zip(&rho)
            .filter_map(|(&r, &v)| (v > 1e-14).then_some(r))
            .next_back()
            .unwrap();
        let result = fields::place(&ctx, &grid, &profile, &[center.into()]).unwrap();
        let translated = fields::place(
            &ctx,
            &grid,
            &profile,
            &[(center + cell.column(0) - 2. * cell.column(2)).into()],
        )
        .unwrap();
        for at in 0..grid.index.points() {
            let point = Vector3::from(grid.point(at));
            let mut expected = 0.;
            for i in -4..=4 {
                for j in -4..=4 {
                    for k in -4..=4 {
                        let image = center + cell * Vector3::new(i as f64, j as f64, k as f64);
                        let distance = (point - image).norm();
                        if distance <= support {
                            expected += profile.value_slope(distance).0;
                        }
                    }
                }
            }
            assert!((result.as_slice()[at] - expected).abs() < 1e-12);
            assert!((translated.as_slice()[at] - expected).abs() < 1e-12);
        }
        let plane = SpatialGrid::new(
            [1, 3, 3],
            [0.; 3],
            [0.2, 0., 0., 0., 0.2, 0., 0., 0., 0.2],
            false,
        )
        .unwrap();
        let sampled = fields::place(&ctx, &plane, &profile, &[[0.; 3]]).unwrap();
        for at in 0..plane.index.points() {
            let distance = Vector3::from(plane.point(at)).norm();
            assert!((sampled.as_slice()[at] - profile.value_slope(distance).0).abs() < 1e-14);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}
