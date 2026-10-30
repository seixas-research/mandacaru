//! Diffuse shells: the incomplete gamma function, the closed-form width and
//! support, the extended grid, the one-center projection and the box geometry.
use mandacaru_core::basis::{diffuse, pao::radial_norm};
use mandacaru_core::math::special::{gamma_q, gamma_q_inverse};
use mandacaru_core::Context;
use std::f64::consts::PI;

fn context() -> Context {
    Context::new(Some(1), 256 << 20).unwrap()
}
fn relative(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

#[test]
fn the_incomplete_gamma_function_matches_its_closed_forms() {
    for x in [1e-3, 0.3, 1., 2.5, 7., 20., 40.] {
        assert!(
            relative(gamma_q(1., x).unwrap(), (-x).exp()) < 1e-13,
            "a=1 x={x}"
        );
        assert!(
            relative(gamma_q(2., x).unwrap(), (1. + x) * (-x).exp()) < 1e-13,
            "a=2 x={x}"
        );
        assert!(
            relative(gamma_q(0.5, x).unwrap(), libm::erfc(x.sqrt())) < 1e-12,
            "a=1/2 x={x}"
        );
        let three_halves = libm::erfc(x.sqrt()) + 2. * (x / PI).sqrt() * (-x).exp();
        assert!(
            relative(gamma_q(1.5, x).unwrap(), three_halves) < 1e-12,
            "a=3/2 x={x}"
        );
    }
    assert_eq!(gamma_q(3., 0.).unwrap(), 1.);
    assert!(gamma_q(0., 1.).is_err() && gamma_q(1., -1.).is_err());
}

#[test]
fn the_inverse_undoes_the_function() {
    for a in [0.5, 1.5, 2.5, 3.5, 4.5, 8.] {
        for y in [1e-10, 1e-6, 1e-4, 1e-2, 0.3, 0.9] {
            let x = gamma_q_inverse(a, y).unwrap();
            assert!(relative(gamma_q(a, x).unwrap(), y) < 1e-11, "a={a} y={y}");
        }
    }
    // Q(1, x) = exp(-x): the inverse is -ln y.
    assert!(relative(gamma_q_inverse(1., 1e-4).unwrap(), -(1e-4f64).ln()) < 1e-12);
    for bad in [0., 1., -0.5, f64::NAN] {
        assert!(gamma_q_inverse(1.5, bad).is_err());
    }
}

#[test]
fn the_support_leaves_the_requested_tail() {
    for l in 0..4 {
        for tail in [1e-2, 1e-4, 1e-6] {
            let c = diffuse::gaussian_cut(l, tail).unwrap();
            assert!(relative(gamma_q(l as f64 + 1.5, 2. * c * c).unwrap(), tail) < 1e-11);
        }
    }
    // A tighter tail is a shorter support.
    assert!(diffuse::gaussian_cut(0, 1e-4).unwrap() < diffuse::gaussian_cut(0, 1e-6).unwrap());
}

#[test]
fn width_and_support_scale_together() {
    for l in 0..3 {
        let (alpha, r_cut) = diffuse::parameters(l, 3., 1e-5).unwrap();
        let (alpha2, r_cut2) = diffuse::parameters(l, 6., 1e-5).unwrap();
        assert!(relative(r_cut2, 2. * r_cut) < 1e-13);
        assert!(relative(alpha2, alpha / 4.) < 1e-13);
    }
    assert!(diffuse::parameters(0, -1., 1e-4).is_err());
    assert!(diffuse::parameters(0, 3., 1.).is_err());
    assert!(diffuse::parameters(0, 3., 0.).is_err());
}

#[test]
fn the_radial_function_has_unit_norm_the_requested_rms_and_a_clean_cut() {
    let r: Vec<f64> = (0..=40000).map(|i| i as f64 * 0.001).collect();
    for l in 0..3 {
        for tail in [1e-4, 1e-6] {
            let (values, _alpha, r_cut) = diffuse::radial(&r, l, 4., tail).unwrap();
            assert!((radial_norm(&r, &values) - 1.).abs() < 1e-12);
            assert!((diffuse::rms(&r, &values) - 4.).abs() < 4e-3);
            for (x, v) in r.iter().zip(&values) {
                if *x >= r_cut {
                    assert_eq!(*v, 0.);
                }
            }
            // r^l at the origin.
            let ratio = values[5] / r[5].powi(l as i32);
            let ratio2 = values[2] / r[2].powi(l as i32);
            assert!((ratio / ratio2 - 1.).abs() < 1e-3);
        }
    }
    assert!(diffuse::radial(&r[..2], 0, 4., 1e-4).is_err());
}

#[test]
fn the_grid_is_continued_with_its_last_spacing() {
    let r: Vec<f64> = (1..=100).map(|i| i as f64 * 0.3).collect();
    assert_eq!(diffuse::extended_grid(&r, 20.).unwrap(), r);
    let longer = diffuse::extended_grid(&r, 45.).unwrap();
    assert_eq!(&longer[..100], &r[..]);
    assert!(longer[longer.len() - 1] >= 45. && longer[longer.len() - 2] < 45.);
    assert!(longer.windows(2).all(|w| (w[1] - w[0] - 0.3).abs() < 1e-12));
    assert!(diffuse::extended_grid(&r[..1], 45.).is_err());
}

#[test]
fn a_table_is_resampled_exactly_on_its_own_grid_and_by_spline_elsewhere() {
    let ctx = context();
    let old: Vec<f64> = (1..=100).map(|i| i as f64 * 0.3).collect();
    let values: Vec<f64> = old.iter().map(|r| (-r).exp()).collect();
    let longer = diffuse::extended_grid(&old, 45.).unwrap();
    let same = diffuse::on_grid(&ctx, &longer, &old, &values).unwrap();
    assert_eq!(&same[..100], &values[..]);
    assert!(same[100..].iter().all(|v| *v == 0.));
    let shifted: Vec<f64> = (1..=150).map(|i| i as f64 * 0.17 + 0.05).collect();
    let resampled = diffuse::on_grid(&ctx, &shifted, &old, &values).unwrap();
    for (r, v) in shifted.iter().zip(&resampled) {
        if *r > old[99] {
            assert_eq!(*v, 0.);
        } else if *r >= old[3] {
            assert!((v - (-r).exp()).abs() < 5e-4, "{r}");
        }
    }
    assert!(diffuse::on_grid(&ctx, &longer, &old, &values[1..]).is_err());
}

fn grid() -> Vec<f64> {
    (0..=6000).map(|i| i as f64 * 0.005).collect()
}
fn function(r: &[f64], l: i32, alpha: f64) -> Vec<f64> {
    r.iter()
        .map(|x| x.powi(l) * (-alpha * x * x).exp())
        .collect()
}

#[test]
fn the_projection_leaves_a_unit_shell_orthogonal_to_its_channel() {
    let r = grid();
    let same = [function(&r, 1, 0.9), function(&r, 1, 0.3)];
    let views: Vec<&[f64]> = same.iter().map(Vec::as_slice).collect();
    let values = function(&r, 1, 0.05);
    let shell = diffuse::orthogonalize(&r, &views, &values, true, 1e-4).unwrap();
    assert!(shell.accepted && shell.residual > 1e-4 && shell.residual < 1.);
    assert!((radial_norm(&r, &shell.values) - 1.).abs() < 1e-12);
    for g in &same {
        let overlap: f64 = r
            .windows(2)
            .zip(g.windows(2).zip(shell.values.windows(2)))
            .map(|(r, (g, s))| {
                0.5 * (r[1] - r[0]) * (g[0] * s[0] * r[0] * r[0] + g[1] * s[1] * r[1] * r[1])
            })
            .sum();
        assert!(overlap.abs() < 1e-12, "{overlap}");
    }
    // The sign follows the unprojected shell.
    let dot: f64 = shell.values.iter().zip(&values).map(|(a, b)| a * b).sum();
    assert!(dot > 0.);
}

#[test]
fn a_shell_inside_the_span_of_its_channel_is_rejected() {
    let r = grid();
    let same = [function(&r, 0, 0.5)];
    let views: Vec<&[f64]> = same.iter().map(Vec::as_slice).collect();
    let nearly = function(&r, 0, 0.5001);
    let shell = diffuse::orthogonalize(&r, &views, &nearly, true, 1e-4).unwrap();
    assert!(!shell.accepted && shell.residual < 1e-4 && shell.values.is_empty());
    // Nothing to project against: the shell is kept as it is.
    let alone = diffuse::orthogonalize(&r, &[], &nearly, true, 1e-4).unwrap();
    assert!(alone.accepted && (alone.residual - 1.).abs() < 1e-12);
    assert!(diffuse::orthogonalize(&r, &views, &nearly[1..], true, 1e-4).is_err());
}

#[test]
fn the_face_distance_is_the_perpendicular_distance_to_the_nearest_face() {
    // 11 x 11 x 11 nodes at 0.5: a cube of side 5 from the origin.
    let step = [[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.5]];
    let d = diffuse::face_distance([0.; 3], step, [11; 3], [1., 2.5, 4.2]).unwrap();
    assert!((d - 0.8).abs() < 1e-12);
    // A sheared cell: faces are tilted, the distance is perpendicular to them.
    let step = [[0.5, 0., 0.], [0.25, 0.5, 0.], [0., 0.1, 0.5]];
    let origin = [-1., 0.5, 2.];
    let center = [0.4, 1.3, 3.1];
    let got = diffuse::face_distance(origin, step, [9, 11, 7], center).unwrap();
    // Brute force: the planes through the columns of `step @ diag(shape - 1)`.
    let m = [
        [0.5 * 8., 0., 0.],
        [0.25 * 8., 0.5 * 10., 0.],
        [0., 0.1 * 10., 0.5 * 6.],
    ];
    let col = |j: usize| [m[0][j], m[1][j], m[2][j]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let rel = [
        center[0] - origin[0],
        center[1] - origin[1],
        center[2] - origin[2],
    ];
    let mut best = f64::INFINITY;
    for (a, b, c) in [(1, 2, 0), (2, 0, 1), (0, 1, 2)] {
        let normal = cross(col(a), col(b));
        let length = dot(normal, normal).sqrt();
        let height = dot(normal, col(c)).abs() / length;
        let along = dot(normal, rel) * dot(normal, col(c)).signum() / length;
        best = best.min(along.min(height - along));
    }
    assert!((got - best).abs() < 1e-12, "{got} {best}");
    assert!(diffuse::face_distance([0.; 3], [[0.; 3]; 3], [5; 3], [1.; 3]).is_err());
    assert!(diffuse::face_distance([0.; 3], step, [1, 5, 5], [1.; 3]).is_err());
}
