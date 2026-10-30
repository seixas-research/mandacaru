use mandacaru_core::paw::local_split::{
    angular_rule, grid_spacing, linspace, long_range_potential, long_range_potential_derivative,
    radial_rule, short_range_potential, short_range_potential_derivative, short_range_radius,
    tail_radius,
};
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
