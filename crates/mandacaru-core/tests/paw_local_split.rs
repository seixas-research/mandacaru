use mandacaru_core::paw::local_split::{
    angular_rule, grid_spacing, invariant_angular_rule, lattice_angular_rule, linspace,
    long_range_potential, long_range_potential_derivative, radial_rule, short_range_potential,
    short_range_potential_derivative, short_range_radius, tail_radius,
};
use mandacaru_core::{math::reciprocal::lattice_rotations, Context};
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;

#[test]
fn the_gaussian_ion_is_finite_at_the_origin_and_coulombic_far_away() {
    let sigma = 0.5;
    let at_origin = long_range_potential(3., sigma, &[0.])[0];
    assert!((at_origin + 3. * (2. / PI).sqrt() / sigma).abs() < 1e-14);
    assert!((long_range_potential(3., sigma, &[10.])[0] + 0.3).abs() < 1e-12);
    // The branch below 1e-12 sigma joins the closed form.
    let just_off = long_range_potential(3., sigma, &[1e-9])[0];
    assert!((just_off - at_origin).abs() < 1e-9);
}

#[test]
fn the_derivative_is_the_derivative() {
    let (charge, sigma) = (6., 0.66);
    let h = 1e-6;
    for r in [0.05, 0.3, 1., 2.5, 7.] {
        let numeric = (long_range_potential(charge, sigma, &[r + h])[0]
            - long_range_potential(charge, sigma, &[r - h])[0])
            / (2. * h);
        let analytic = long_range_potential_derivative(charge, sigma, &[r])[0];
        assert!(
            (numeric - analytic).abs() < 1e-7 * (1. + analytic.abs()),
            "r = {r}"
        );
    }
    assert_eq!(long_range_potential_derivative(charge, sigma, &[0.])[0], 0.);
}

#[test]
fn the_halves_add_up_to_the_sampled_potential() {
    let radius = linspace(0., 12., 401);
    let sampled = radius.iter().map(|r| -2. / r.max(0.1)).collect::<Vec<_>>();
    let slopes = radius
        .iter()
        .map(|r| 2. / r.max(0.1).powi(2))
        .collect::<Vec<_>>();
    let sigma = 0.4;
    let short = short_range_potential(&sampled, 2., sigma, &radius).unwrap();
    let long = long_range_potential(2., sigma, &radius);
    for i in 0..radius.len() {
        assert!((short[i] + long[i] - sampled[i]).abs() < 1e-14);
    }
    let short_slope = short_range_potential_derivative(&slopes, 2., sigma, &radius).unwrap();
    let long_slope = long_range_potential_derivative(2., sigma, &radius);
    for i in 0..radius.len() {
        assert!((short_slope[i] + long_slope[i] - slopes[i]).abs() < 1e-14);
    }
    assert!(short_range_potential(&sampled[..3], 2., sigma, &radius).is_err());
    assert!(short_range_potential_derivative(&slopes[..3], 2., sigma, &radius).is_err());
}

#[test]
fn the_tail_starts_where_the_table_stops_deviating() {
    let radius = linspace(0.3, 8., 7701);
    let table = radius
        .iter()
        .map(|r| -2. / r + if *r < 1.5 { 0.1 } else { 0. })
        .collect::<Vec<_>>();
    let tail = tail_radius(&radius, &table, 2., 1e-5).unwrap();
    assert!((1.5..=1.52).contains(&tail), "{tail}");
    let exact = radius.iter().map(|r| -2. / r).collect::<Vec<_>>();
    assert_eq!(tail_radius(&radius, &exact, 2., 1e-5).unwrap(), 0.);
    assert!(tail_radius(&radius, &exact[..10], 2., 1e-5).is_err());
    // The sphere holds the larger of the Gaussian, the cutoff and the tail.
    assert_eq!(short_range_radius(0.1, 6.5, 2.34, 2.2), 2.34);
    assert_eq!(short_range_radius(0.5, 6.5, 2.34, 2.2), 3.25);
    assert_eq!(short_range_radius(0.1, 6.5, 1.98, 2.215), 2.215);
}

#[test]
fn linspace_ends_exactly() {
    let x = linspace(0.3, 8., 7701);
    assert_eq!(x[0], 0.3);
    assert_eq!(x[7700], 8.);
    assert!((x[1] - x[0] - 7.7 / 7700.).abs() < 1e-15);
}

#[test]
fn the_radial_rule_integrates_r_squared_on_both_panels() {
    for (radius, panel) in [(3., 1.2), (3., 0.), (2.2, 2.2), (0.7, 1.5), (4.5, 1.)] {
        let (r, w) = radial_rule(radius, panel, 32).unwrap();
        let panels = if 0. < panel && panel < radius { 2 } else { 1 };
        assert_eq!(r.len(), 32 * panels);
        // int_0^R r^2 dr, and int r^2 * r^3 dr (the weights carry r^2).
        let volume: f64 = w.iter().sum();
        assert!((volume - radius.powi(3) / 3.).abs() < 1e-13 * radius.powi(3));
        let moment: f64 = w.iter().zip(&r).map(|(w, r)| w * r.powi(3)).sum();
        assert!((moment - radius.powi(6) / 6.).abs() < 1e-12 * radius.powi(6));
        assert!(r.iter().all(|x| *x > 0. && *x < radius));
        if panels == 2 {
            assert!(r[..32].iter().all(|x| *x < panel) && r[32..].iter().all(|x| *x > panel));
        }
    }
}

#[test]
fn the_angular_rule_covers_the_sphere() {
    for (polar, azimuthal) in [(32, 64), (16, 36), (8, 12)] {
        let (d, w) = angular_rule(polar, azimuthal).unwrap();
        assert_eq!(w.len(), polar * azimuthal);
        assert!((w.iter().sum::<f64>() - 4. * PI).abs() < 1e-12);
        for i in 0..w.len() {
            let norm = d[0][i].powi(2) + d[1][i].powi(2) + d[2][i].powi(2);
            assert!((norm - 1.).abs() < 1e-14);
        }
        // <z^2> = 1/3, <x y> = 0, and the mirror x -> -x maps the rule onto itself.
        let zz: f64 = (0..w.len()).map(|i| w[i] * d[2][i].powi(2)).sum();
        assert!((zz - 4. * PI / 3.).abs() < 1e-12);
        let xy: f64 = (0..w.len()).map(|i| w[i] * d[0][i] * d[1][i]).sum();
        assert!(xy.abs() < 1e-12);
        let x: f64 = (0..w.len()).map(|i| w[i] * d[0][i]).sum();
        assert!(x.abs() < 1e-12);
    }
}

#[test]
fn the_grid_spacing_is_the_longest_lattice_step() {
    let diagonal = [0.37, 0., 0., 0., 0.37, 0., 0., 0., 0.37];
    assert!((grid_spacing(&diagonal) - 0.37).abs() < 1e-15);
    // Columns are the steps along the grid axes.
    let skewed = [0.4, 0.1, 0., 0., 0.3, 0.05, 0., 0., 0.5];
    let expected = (0.1f64 * 0.1 + 0.3 * 0.3).sqrt().max(0.5f64.hypot(0.05));
    assert!((grid_spacing(&skewed) - expected).abs() < 1e-15);
}

/// `sum_x w_x f(R x)` of a rule for the operation `R`.
fn rotated_integral(
    rule: &([Vec<f64>; 3], Vec<f64>),
    r: &Matrix3<f64>,
    f: impl Fn(&Vector3<f64>) -> f64,
) -> f64 {
    let (d, w) = rule;
    (0..w.len())
        .map(|i| w[i] * f(&(r * Vector3::new(d[0][i], d[1][i], d[2][i]))))
        .sum()
}

/// The cells of the three cases of the invariant rule: cubic (the product rule's axis is not the
/// cube's 3-fold axis), hexagonal about the pole axis, and no symmetry beyond inversion.
fn lattices() -> [(Matrix3<f64>, usize); 3] {
    let h = 7.6524 / 2.;
    let a = 5.877;
    [
        (
            Matrix3::from_columns(&[
                Vector3::new(0., h, h),
                Vector3::new(h, 0., h),
                Vector3::new(h, h, 0.),
            ]),
            3,
        ),
        (
            Matrix3::from_columns(&[
                Vector3::new(a, 0., 0.),
                Vector3::new(-a / 2., a * 3f64.sqrt() / 2., 0.),
                Vector3::new(0., 0., 9.411),
            ]),
            1,
        ),
        (Matrix3::new(9., 1.1, 0.7, 0., 10., 0.9, 0., 0., 11.), 1),
    ]
}

#[test]
fn the_invariant_rule_replicates_the_product_rule_by_the_cosets_of_its_own_symmetry() {
    let ctx = Context::new(Some(1), 8 << 20).unwrap();
    for (cell, copies) in lattices() {
        let (directions, weights) = lattice_angular_rule(&ctx, &cell, 24, 48).unwrap();
        let expected = if copies == 3 { 770 } else { copies * 24 * 48 };
        assert_eq!(weights.len(), expected);
        assert!((weights.iter().sum::<f64>() - 4. * PI).abs() < 1e-12);
        if copies == 1 {
            // A group the product rule already has returns it unchanged, bit for bit.
            assert_eq!((directions, weights), angular_rule(24, 48).unwrap());
        }
    }
}

#[test]
fn the_invariant_rule_integrates_a_function_of_the_direction_alike_in_every_orientation() {
    let ctx = Context::new(Some(1), 8 << 20).unwrap();
    // A smooth function with more angular content than the 24 x 48 rule resolves (its error is
    // then orientation dependent), off every symmetry axis of the lattices.
    let f = |x: &Vector3<f64>| (100. * (0.31 * x[0] + 0.52 * x[1] + 0.79 * x[2])).exp();
    for (cell, _) in lattices() {
        let operations = lattice_rotations(&ctx, &cell).unwrap();
        let rule = lattice_angular_rule(&ctx, &cell, 24, 48).unwrap();
        let plain = angular_rule(24, 48).unwrap();
        let (mut invariant, mut product) = (0f64, 0f64);
        let reference = rotated_integral(&rule, &Matrix3::identity(), f);
        let mut scale = 0f64;
        for r in &operations {
            let moved = rotated_integral(&rule, r, f);
            scale = scale.max(moved.abs());
            invariant = invariant.max((moved - reference).abs());
            product = product.max(
                (rotated_integral(&plain, r, f)
                    - rotated_integral(&plain, &Matrix3::identity(), f))
                .abs(),
            );
        }
        println!(
            "{} operations: invariant rule {:.3e}, plain product rule {:.3e} (of {scale:.3e})",
            operations.len(),
            invariant / scale,
            product / scale
        );
        assert!(invariant < 1e-13 * scale, "{invariant:e} of {scale:e}");
        // The cubic cell is where the product rule breaks the group (the control).
        if operations.len() == 48 {
            assert!(
                product > 1e-8 * scale,
                "the plain rule is invariant already: {product:e}"
            );
        }
    }
}

#[test]
fn an_invariant_rule_of_a_group_the_rule_does_not_have_is_exact_where_the_rule_is() {
    // The union over the cosets integrates what the product rule integrates: a low-order
    // polynomial, and the identity group returns the product rule itself.
    let identity = invariant_angular_rule(8, 12, &[Matrix3::identity()]).unwrap();
    let plain = angular_rule(8, 12).unwrap();
    assert_eq!(identity.1, plain.1);
    assert_eq!(identity.0, plain.0);
}

#[test]
fn the_invariant_rule_of_a_strained_lattice_keeps_its_directions_and_moves_continuously() {
    // A generic strain of the fcc cell, at the finite-difference steps of the stress tiers (the
    // strain manifests step by 5e-4, 1e-3 and 2e-3): the polar factors of its near-symmetries miss
    // O_h by about the square of the strain, which must not change the cosets nor jump the
    // directions.
    let ctx = Context::new(Some(1), 8 << 20).unwrap();
    let (cubic, _) = lattices()[0];
    let (ideal_directions, ideal_weights) = lattice_angular_rule(&ctx, &cubic, 24, 48).unwrap();
    for step in [5e-4, 1e-3, 2e-3] {
        let strain = Matrix3::new(1., 0.3, -0.2, 0.3, -0.5, 0.4, -0.2, 0.4, 0.7) * step;
        let cell = (Matrix3::identity() + strain) * cubic;
        let (directions, weights) = lattice_angular_rule(&ctx, &cell, 24, 48).unwrap();
        assert_eq!(
            weights, ideal_weights,
            "strain {step}: the weights or their number changed"
        );
        let mut moved = 0f64;
        for j in 0..weights.len() {
            let d = Vector3::new(
                directions[0][j] - ideal_directions[0][j],
                directions[1][j] - ideal_directions[1][j],
                directions[2][j] - ideal_directions[2][j],
            );
            moved = moved.max(d.norm());
        }
        println!("strain {step:e}: directions move by {moved:.3e}");
        assert!(
            moved < 10. * step * step,
            "strain {step}: directions move by {moved:e}"
        );
    }
}
