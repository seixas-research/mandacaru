use mandacaru_core::math::radial::gauss_legendre;
use mandacaru_core::paw::multipole::{
    angular_grid, gaunt, gaunt_grid, multipole_range, shape_function, shape_norm, shape_potential,
};
use mandacaru_core::paw::multipole_pairs::{
    radial_moments, radial_remainder, smooth_waves, ChannelWaves, Pair,
};
use mandacaru_core::Context;
use std::f64::consts::PI;

fn moment(radius: f64, l: usize, power: i32) -> f64 {
    gauss_legendre(200)
        .unwrap()
        .iter()
        .map(|(x, w)| {
            let r = 0.5 * radius * (x + 1.);
            0.5 * radius * w * shape_function(&[r], radius, l).unwrap()[0] * r.powi(power)
        })
        .sum()
}

#[test]
fn every_shape_has_a_unit_moment() {
    for l in 0..6 {
        assert!((moment(1.7, l, l as i32 + 2) - 1.).abs() < 1e-13, "L = {l}");
        assert!(shape_norm(1.7, l) > 0.);
    }
}

#[test]
fn the_shape_potential_is_continuous_and_coulombic_outside() {
    let r_g = 1.4;
    for l in 0..5 {
        let outside = shape_potential(&[r_g, 2. * r_g], r_g, l).unwrap();
        let just_inside = shape_potential(&[r_g * (1. - 1e-9)], r_g, l).unwrap()[0];
        let prefactor = 4. * PI / (2 * l + 1) as f64;
        assert!((outside[0] - just_inside).abs() < 1e-7 * outside[0]);
        assert!((outside[1] - prefactor / (2. * r_g).powi(l as i32 + 1)).abs() < 1e-14);
    }
    // Finite at the origin for L = 0, zero for higher L.
    assert!(shape_potential(&[0.], r_g, 0).unwrap()[0].is_finite());
    assert_eq!(shape_potential(&[0.], r_g, 2).unwrap()[0], 0.);
}

#[test]
fn the_shape_potential_obeys_poisson() {
    // v_L'' + 2 v_L'/r - L(L+1) v_L / r^2 = -4 pi g_L, by central differences.
    let (r_g, h) = (1.6, 1e-4);
    for l in 0..4 {
        let norm = shape_norm(r_g, l);
        for r in [0.3, 0.8, 1.3] {
            let v = |x: f64| shape_potential(&[x], r_g, l).unwrap()[0];
            let second = (v(r + h) - 2. * v(r) + v(r - h)) / (h * h);
            let first = (v(r + h) - v(r - h)) / (2. * h);
            let lhs = second + 2. * first / r - (l * (l + 1)) as f64 * v(r) / (r * r);
            let g = norm * r.powi(l as i32) * (1. - (r / r_g).powi(2)).powi(3);
            assert!(
                (lhs + 4. * PI * g).abs() < 2e-5 * (1. + g),
                "L = {l}, r = {r}"
            );
        }
    }
}

#[test]
fn the_angular_rule_integrates_the_sphere_and_a_triple_product() {
    let (theta, _phi, weight) = angular_grid(24, 48).unwrap();
    assert_eq!(theta.len(), 24 * 48);
    assert!((weight.iter().sum::<f64>() - 4. * PI).abs() < 1e-12);
    let grid = gaunt_grid().unwrap();
    // int Y00 Y00 Y00 = 1 / sqrt(4 pi).
    let g = gaunt([0, 0, 0], [0, 0, 0], true, &grid);
    assert!((g.re - 1. / (4. * PI).sqrt()).abs() < 1e-14);
}

#[test]
fn gaunt_obeys_the_selection_rules_and_the_real_complex_relation() {
    let grid = gaunt_grid().unwrap();
    assert_eq!(gaunt([1, 0, 0], [0, 0, 0], true, &grid).norm(), 0.); // parity
    assert_eq!(gaunt([3, 1, 1], [0, 0, 0], true, &grid).norm(), 0.); // triangle
    assert_eq!(gaunt([1, 1, 1], [0, 0, 0], false, &grid).norm(), 0.); // M != m2 - m1
                                                                      // int Y*_20 Y*_10 Y_10 = sqrt(5 / (4 pi)) (2/5)  (standard 3j value).
    let value = gaunt([2, 1, 1], [0, 0, 0], false, &grid);
    assert!((value.re - (5. / (4. * PI)).sqrt() * 2. / 5.).abs() < 1e-13);
    assert!(value.im.abs() < 1e-14);
    // The range of a pair obeys parity and the triangle rule.
    assert_eq!(multipole_range(1, 1), vec![0, 2]);
    assert_eq!(multipole_range(0, 2), vec![2]);
    assert_eq!(multipole_range(2, 3), vec![1, 3, 5]);
}

fn grid() -> Vec<f64> {
    (1..=1200).map(|i| i as f64 * 0.005).collect()
}

#[test]
fn smooth_waves_equal_the_all_electron_wave_beyond_the_cutoff() {
    let r = grid();
    let ae = r.iter().map(|x| (-x).exp()).collect::<Vec<_>>();
    let waves = ChannelWaves {
        l: 0,
        r_cut: 2.,
        ae: &ae,
        coefficients: &[1., 0.5],
        wavevectors: &[1., 2.],
        waves: 1,
    };
    let smooth = smooth_waves(&r, &waves).unwrap();
    for (i, x) in r.iter().enumerate() {
        if *x > 2. {
            assert_eq!(smooth[i], ae[i]);
        } else {
            let expected = (x.sin() / x) + 0.5 * (2. * x).sin() / (2. * x);
            assert!((smooth[i] - expected).abs() < 1e-13);
        }
    }
}

#[test]
fn identical_waves_leave_no_moment_and_no_remainder() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let r = grid();
    let ae = r.iter().map(|x| x * (-x).exp()).collect::<Vec<_>>();
    let pair = Pair {
        r: &r,
        r_cut: 2.,
        ae: [&ae, &ae],
        smooth: [&ae, &ae],
        waves: [1, 1],
    };
    assert_eq!(radial_moments(&pair, 2).unwrap(), vec![0.]);
    let q = [0.3, 1.1];
    let out = radial_remainder(&ctx, &pair, 1, &q, &[0.], &[5., 7.]).unwrap();
    assert_eq!(out, vec![0., 0.]);
    // A smooth wave with less weight leaves exactly the missing multipole moment.
    let smooth = ae.iter().map(|v| 0.9 * v).collect::<Vec<_>>();
    let pair = Pair {
        r: &r,
        r_cut: 2.,
        ae: [&ae, &ae],
        smooth: [&smooth, &smooth],
        waves: [1, 1],
    };
    let delta = radial_moments(&pair, 0).unwrap()[0];
    let inside = r.iter().take_while(|x| **x <= 2.).count();
    let direct = (0..inside - 1)
        .map(|i| {
            0.5 * 0.005 * (ae[i].powi(2) * r[i].powi(2) + ae[i + 1].powi(2) * r[i + 1].powi(2))
        })
        .sum::<f64>()
        * 0.19;
    assert!(
        (delta - direct).abs() < 1e-5 * direct,
        "{delta} vs {direct}"
    );
    assert_eq!(ctx.memory_used(), 0);
}
