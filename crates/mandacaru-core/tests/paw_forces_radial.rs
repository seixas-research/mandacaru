//! The radial slopes of the PAW setup's functions (`paw::molecular::gradient`)
//! against central differences of the functions themselves.
//!
//! Each slope is the derivative of the exact function the setup evaluates, so
//! the gate is the central difference's own truncation: a fourth-order
//! five-point stencil at `h = 1e-3 radius` is good to `1e-9` of the scale.
use mandacaru_core::{
    math::spline::Cubic,
    paw::{
        molecular::gradient::{core_slope, long_range_slope, multipole_gradient, potential_slope},
        pair_tables::{harmonic, potential},
    },
    Context,
};

fn central(h: f64, f: impl Fn(f64) -> f64, x: f64) -> f64 {
    (8. * (f(x + h) - f(x - h)) - (f(x + 2. * h) - f(x - 2. * h))) / (12. * h)
}
const GATE: f64 = 2e-8;

#[test]
fn the_compensation_potential_slope_is_its_derivative_inside_and_outside_the_sphere() {
    let radius: f64 = 1.3;
    let mut worst = 0f64;
    let mut wrong = f64::INFINITY;
    for l in 0..=4usize {
        for r in [0.02, 0.2, 0.5, 0.9, 1.25, 1.35, 2., 4.] {
            let h = (1e-3 * radius).min(0.02 * r);
            if (r - radius).abs() < 3. * h {
                continue;
            }
            let (value, slope) = potential_slope(r, radius, l);
            assert!((value - potential(r, radius, l)).abs() < 1e-14 * value.abs().max(1.));
            let reference = central(h, |x| potential(x, radius, l), r);
            let scale = value.abs().max(reference.abs()).max(1e-3);
            worst = worst.max((slope - reference).abs() / scale);
            // A wrong slope: the logarithmic slope of the exterior form everywhere.
            wrong = wrong.min(((-((l + 1) as f64)) * value / r - reference).abs() / scale);
        }
    }
    println!("compensation potential slope: worst relative miss {worst:.2e}");
    assert!(worst < GATE, "{worst:.2e}");
    // The exterior form is only right outside: inside it misses by far more than 100 gates.
    let inside = potential_slope(0.5, radius, 1).1;
    let exterior = -2. * potential(0.5, radius, 1) / 0.5;
    assert!((inside - exterior).abs() > 100. * GATE * inside.abs());
    let _ = wrong;
}
#[test]
fn the_multipole_gradient_is_the_gradient_of_the_potential_times_the_harmonic() {
    let radius = 1.1;
    let mut worst = 0f64;
    for (l, m) in [
        (0usize, 0i32),
        (1, -1),
        (1, 0),
        (1, 1),
        (2, -2),
        (2, 0),
        (2, 2),
        (3, 1),
        (4, -3),
    ] {
        for delta in [
            [0.3, -0.2, 0.5],
            [-0.7, 0.4, 0.1],
            [1.4, 0.9, -0.6],
            [0.05, 0.02, -0.04],
        ] {
            let gradient = multipole_gradient(radius, l, m, delta);
            let field = |d: [f64; 3]| {
                potential((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt(), radius, l)
                    * harmonic(l, m, d)
            };
            for axis in 0..3 {
                let h = 1e-4;
                let reference = central(
                    h,
                    |x| {
                        let mut d = delta;
                        d[axis] = x;
                        field(d)
                    },
                    delta[axis],
                );
                worst = worst.max((gradient[axis] - reference).abs() / reference.abs().max(1e-2));
            }
        }
    }
    println!("multipole gradient: worst relative miss {worst:.2e}");
    assert!(worst < 1e-7, "{worst:.2e}");
}
#[test]
fn the_long_range_ionic_slope_is_the_derivative_of_the_error_function_potential() {
    let (charge, sigma) = (3., 0.35);
    let long_range = |r: f64| -charge * libm::erf(r / (2f64.sqrt() * sigma)) / r;
    let mut worst = 0f64;
    // The small radii exercise the series that replaces the cancelling difference.
    for r in [0.004, 0.01, 0.03, 0.1, 0.4, 1., 3., 8.] {
        let h = 1e-3 * r;
        let reference = central(h, long_range, r);
        let slope = long_range_slope(charge, sigma, r);
        worst = worst.max((slope - reference).abs() / reference.abs().max(1e-6));
    }
    println!("long-range ionic slope: worst relative miss {worst:.2e}");
    assert!(worst < 1e-6, "{worst:.2e}");
    assert_eq!(long_range_slope(charge, sigma, 0.), 0.);
}
#[test]
fn the_partial_core_slope_follows_the_clipped_table_the_grid_sums() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let knots: Vec<f64> = (1..=60).map(|k| 0.05 * k as f64).collect();
    let values: Vec<f64> = knots
        .iter()
        .map(|r| 0.8 * (-1.5 * r * r).exp() - 0.02 * r)
        .collect();
    let table = Cubic::new(
        &ctx,
        ctx.copy_slice(&knots).unwrap(),
        ctx.copy_slice(&values).unwrap(),
    )
    .unwrap();
    let clipped = |r: f64| {
        if r > 3. {
            0.
        } else {
            table.value_slope(r.max(0.05)).0.max(0.)
        }
    };
    let mut worst = 0f64;
    for r in [0.2, 0.7, 1.4, 2.2, 2.9] {
        let (value, slope) = core_slope(&table, r);
        assert!((value - clipped(r)).abs() < 1e-14);
        worst = worst.max((slope - central(1e-4, clipped, r)).abs());
    }
    // Below the first node the table is held constant, past the last it is zero,
    // and where the value is clipped at zero the slope is zero.
    assert_eq!(core_slope(&table, 0.01), (table.value_slope(0.05).0, 0.));
    assert_eq!(core_slope(&table, 3.5), (0., 0.));
    println!("partial core slope: worst miss {worst:.2e}");
    assert!(worst < 1e-8, "{worst:.2e}");
}
