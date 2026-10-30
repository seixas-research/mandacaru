//! Tabulated radial functions: the spline, the continuation below the first node and
//! the piecewise-cubic coefficients.
use mandacaru_core::basis::table::{coefficients, evaluate};
use mandacaru_core::Context;

fn context() -> Context {
    Context::new(Some(1), 256 << 20).unwrap()
}

fn grid(first: f64, last: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| first + (last - first) * i as f64 / (n - 1) as f64)
        .collect()
}

#[test]
fn a_cubic_polynomial_is_reproduced_exactly_inside_the_table() {
    let ctx = context();
    let r = grid(0.2, 5., 40);
    let f = |x: f64| 0.3 - 1.2 * x + 0.7 * x * x - 0.1 * x.powi(3);
    let d = |x: f64| -1.2 + 1.4 * x - 0.3 * x * x;
    let values = r.iter().map(|x| f(*x)).collect::<Vec<_>>();
    let nodes = grid(0.2, 5., 333);
    let value = evaluate(&ctx, &r, &values, 0, &nodes, false).unwrap();
    let slope = evaluate(&ctx, &r, &values, 0, &nodes, true).unwrap();
    for (i, x) in nodes.iter().enumerate() {
        assert!((value[i] - f(*x)).abs() < 1e-11, "value at {x}");
        assert!((slope[i] - d(*x)).abs() < 1e-10, "slope at {x}");
    }
}

#[test]
fn the_function_is_zero_beyond_the_table_and_at_nan() {
    let ctx = context();
    let r = grid(0.1, 4., 30);
    let values = r.iter().map(|x| (-x).exp()).collect::<Vec<_>>();
    let nodes = [4.0 + 1e-9, 9., f64::NAN, 4.];
    for slope in [false, true] {
        let out = evaluate(&ctx, &r, &values, 1, &nodes, slope).unwrap();
        assert_eq!(&out[..3], &[0., 0., 0.]);
        assert!(out[3] != 0.);
    }
}

#[test]
fn below_the_first_node_the_function_continues_as_a_power_law() {
    let ctx = context();
    let r = grid(0.5, 4., 30);
    let values = r.iter().map(|x| x * (-x).exp()).collect::<Vec<_>>();
    for l in 0..4usize {
        let first = evaluate(&ctx, &r, &values, l, &[0.5], false).unwrap()[0];
        let nodes = [0.0, 0.1, 0.25, -1.];
        let value = evaluate(&ctx, &r, &values, l, &nodes, false).unwrap();
        let slope = evaluate(&ctx, &r, &values, l, &nodes, true).unwrap();
        for (i, x) in nodes.iter().enumerate() {
            let x = f64::max(*x, 0.);
            let expect = if l == 0 {
                first
            } else {
                first * (x / 0.5).powi(l as i32)
            };
            assert!((value[i] - expect).abs() < 1e-13, "l={l} x={x}");
            let expect = if l == 0 {
                0.
            } else {
                first * l as f64 * x.powi(l as i32 - 1) / 0.5f64.powi(l as i32)
            };
            assert!((slope[i] - expect).abs() < 1e-12, "slope l={l} x={x}");
        }
    }
}

#[test]
fn a_table_starting_at_the_origin_holds_its_first_value() {
    let ctx = context();
    let r = grid(0., 3., 25);
    let values = r.iter().map(|x| 1. + x).collect::<Vec<_>>();
    let out = evaluate(&ctx, &r, &values, 2, &[-0.5, 0.], false).unwrap();
    assert!((out[0] - 1.).abs() < 1e-13 && (out[1] - 1.).abs() < 1e-13);
}

#[test]
fn the_coefficients_rebuild_the_values_and_slopes() {
    let ctx = context();
    let r = (0..30)
        .map(|i| 0.1 + 0.05 * (i * i) as f64 / 10.)
        .collect::<Vec<_>>();
    let values = r
        .iter()
        .map(|x| (3. * x).sin() * (-x).exp())
        .collect::<Vec<_>>();
    let c = coefficients(&ctx, &r, &values).unwrap();
    let m = r.len() - 1;
    assert_eq!(c.len(), 4 * m);
    let at = (0..m)
        .map(|i| r[i] + 0.37 * (r[i + 1] - r[i]))
        .collect::<Vec<_>>();
    let value = evaluate(&ctx, &r, &values, 0, &at, false).unwrap();
    let slope = evaluate(&ctx, &r, &values, 0, &at, true).unwrap();
    for i in 0..m {
        let t = at[i] - r[i];
        let poly = ((c[i] * t + c[m + i]) * t + c[2 * m + i]) * t + c[3 * m + i];
        let derivative = (3. * c[i] * t + 2. * c[m + i]) * t + c[2 * m + i];
        assert!((poly - value[i]).abs() < 1e-13, "value, interval {i}");
        assert!((derivative - slope[i]).abs() < 1e-12, "slope, interval {i}");
        assert_eq!(c[3 * m + i], values[i]);
    }
}

#[test]
fn a_bad_table_is_refused() {
    let ctx = context();
    assert!(evaluate(&ctx, &[0., 1., 2.], &[1., 2., 3.], 0, &[0.5], false).is_err());
    assert!(coefficients(&ctx, &[0., 1., 1., 2., 3.], &[1.; 5]).is_err());
    assert!(coefficients(&ctx, &[0., 1., 2., 3., 4.], &[1., 2., f64::NAN, 4., 5.]).is_err());
    let tiny = Context::new(Some(1), 1024).unwrap();
    let r = grid(0., 1., 1000);
    assert!(evaluate(&tiny, &r, &r, 0, &[0.5], false).is_err());
}
