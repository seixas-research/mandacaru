//! The split-valence multiple-zeta construction: grids that start at zero or at
//! the step, the two splitting schemes, the polynomial match and the refusals.
use mandacaru_core::basis::radial::{zeta_values, zeta_values_by_fraction, zetas, Table};
use mandacaru_core::Context;

fn context() -> Context {
    Context::new(Some(1), 256 << 20).unwrap()
}

fn grid(first: f64, step: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| first + step * i as f64).collect()
}

fn orbital(r: &[f64], l: i32) -> Vec<f64> {
    r.iter().map(|x| x.powi(l) * (-0.8 * x * x).exp()).collect()
}

fn norm(r: &[f64], v: &[f64]) -> f64 {
    (1..r.len())
        .map(|i| 0.5 * (v[i - 1].powi(2) * r[i - 1].powi(2) + v[i].powi(2) * r[i].powi(2)))
        .zip(r.windows(2))
        .map(|(f, w)| f * (w[1] - w[0]))
        .sum()
}

fn support(r: &[f64], v: &[f64]) -> f64 {
    let last = v.iter().rposition(|x| *x != 0.).unwrap();
    r[last]
}

#[test]
fn a_split_zeta_is_normalized_and_vanishes_beyond_the_split_radius() {
    let ctx = context();
    for l in 0..3 {
        for start in [0., 0.01] {
            let r = grid(start, 0.01, 900);
            let first = orbital(&r, l);
            let out = zeta_values(&ctx, &r, &first, l as usize, 3, 0.15, None).unwrap();
            assert_eq!(out.len(), 3);
            assert_eq!(out[0].as_slice(), first.as_slice());
            let mut previous = support(&r, &first);
            for zeta in &out[1..] {
                assert!((norm(&r, zeta.as_slice()) - 1.).abs() < 1e-12);
                let edge = support(&r, zeta.as_slice());
                assert!(
                    edge < previous,
                    "each zeta is shorter ranged: l={l} start={start}"
                );
                previous = edge;
            }
        }
    }
}

#[test]
fn the_split_radius_leaves_the_requested_tail_of_the_norm() {
    let ctx = context();
    let r = grid(0.02, 0.02, 600);
    let first = orbital(&r, 1);
    let total = norm(&r, &first);
    for tail in [0.2, 0.4] {
        let out = zeta_values(&ctx, &r, &first, 1, 2, 0.15, Some(&[tail])).unwrap();
        let edge = r
            .iter()
            .position(|x| *x >= support(&r, out[1].as_slice()))
            .unwrap();
        let beyond = total - norm(&r[..=edge], &first[..=edge]);
        // The split sits on the node nearest the fraction: within one step of it.
        assert!((beyond / total - tail * tail).abs() < 0.02, "tail {tail}");
    }
}

#[test]
fn the_polynomial_the_orbital_already_is_leaves_nothing_to_split() {
    // r^l (a - b r^2) is its own smooth part, and a not-a-knot cubic spline
    // reproduces it for l = 0 and 1: the origin of the grid enters the match.
    let ctx = context();
    for l in 0..2 {
        let r = grid(0.05, 0.05, 80);
        let first = r
            .iter()
            .map(|x| x.powi(l) * (1.5 - 0.3 * x * x))
            .collect::<Vec<_>>();
        let out = zeta_values(&ctx, &r, &first, l as usize, 4, 0.15, None).unwrap();
        assert_eq!(out.len(), 1, "l={l}");
    }
}

#[test]
fn the_grid_origin_moves_the_split() {
    let ctx = context();
    let from_step = grid(0.05, 0.05, 200);
    let from_zero = grid(0., 0.05, 200);
    let values = orbital(&from_step, 0);
    let a = zeta_values(&ctx, &from_step, &values, 0, 2, 0.15, None).unwrap();
    let b = zeta_values(&ctx, &from_zero, &values, 0, 2, 0.15, None).unwrap();
    assert_ne!(
        support(&from_step, a[1].as_slice()),
        support(&from_zero, b[1].as_slice())
    );
}

#[test]
fn a_table_from_zero_gives_the_same_zetas_as_its_grid() {
    let ctx = context();
    let h = 0.02;
    let r = grid(0., h, 500);
    let values = orbital(&r, 2);
    let direct = zeta_values(&ctx, &r, &values, 2, 3, 0.15, None).unwrap();
    let table = Table::new(&ctx, h, ctx.copy_slice(&values).unwrap(), 0., 2).unwrap();
    let tables = zetas(&ctx, table, 2, 3, 0.15, None).unwrap();
    assert_eq!(tables.len(), direct.len());
    for (t, d) in tables.iter().zip(&direct) {
        assert_eq!(t.values.as_slice(), d.as_slice());
    }
}

#[test]
fn unsupported_requests_raise() {
    let ctx = context();
    let r = grid(0.02, 0.02, 100);
    let v = orbital(&r, 0);
    let bad: [(usize, f64, Option<&[f64]>); 7] = [
        (0, 0.15, None),
        (5, 0.15, None),
        (2, 0., None),
        (2, 1., None),
        (2, f64::NAN, None),
        (3, 0.15, Some(&[0.2])),
        (2, 0.15, Some(&[f64::NAN])),
    ];
    for (count, split, tails) in bad {
        assert!(zeta_values(&ctx, &r, &v, 0, count, split, tails).is_err());
    }
    assert!(zeta_values(&ctx, &r, &v[..50], 0, 2, 0.15, None).is_err());
    assert!(zeta_values(&ctx, &grid(-0.1, 0.02, 100), &v, 0, 2, 0.15, None).is_err());
    let mut backward = r.clone();
    backward.swap(10, 11);
    assert!(zeta_values(&ctx, &backward, &v, 0, 2, 0.15, None).is_err());
    let mut broken = v.clone();
    broken[3] = f64::INFINITY;
    assert!(zeta_values(&ctx, &r, &broken, 0, 2, 0.15, None).is_err());
    assert!(zeta_values(&ctx, &r, &vec![0.; 100], 0, 2, 0.15, None).is_err());
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn tail_fractions_are_the_squares_of_tail_norms() {
    // The polarization zetas give the squared-norm fraction itself; the valence
    // zetas give its square root. Both must split at the same radius.
    let ctx = context();
    let r = grid(0.02, 0.02, 600);
    let first = orbital(&r, 1);
    let norms = [0.2, 0.4, 0.55];
    let fractions = norms.map(|x: f64| x.powi(2));
    let by_norm = zeta_values(&ctx, &r, &first, 1, 4, 0.15, Some(&norms)).unwrap();
    let by_fraction =
        zeta_values_by_fraction(&ctx, &r, &first, 1, 4, 0.15, Some(&fractions)).unwrap();
    assert_eq!(by_norm.len(), by_fraction.len());
    for (a, b) in by_norm.iter().zip(&by_fraction) {
        assert_eq!(a.as_slice(), b.as_slice());
    }
    assert!(zeta_values_by_fraction(&ctx, &r, &first, 1, 2, 0.15, Some(&[1.])).is_err());
    assert!(zeta_values_by_fraction(&ctx, &r, &first, 1, 3, 0.15, Some(&[0.1])).is_err());
}
