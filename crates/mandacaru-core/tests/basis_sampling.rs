//! A tabulated radial orbital `R(r) Y_lm` and its gradient at Cartesian points
//! (`basis::sampling`), and the harmonics it shares with the angular bindings
//! (`math::harmonics`).
use mandacaru_core::basis::sampling::{gradient, values, Orbital, Sampled};
use mandacaru_core::math::harmonics::{harmonic, harmonic_gradient, polar, NODE_OFFSET};
use mandacaru_core::Context;
use std::f64::consts::PI;

fn table(l: usize) -> (Vec<f64>, Vec<f64>) {
    let r: Vec<f64> = (0..400).map(|i| 0.02 + 0.02 * i as f64).collect();
    let values = r
        .iter()
        .map(|x| x.powi(l as i32) * (-x).exp() * (1. + 0.3 * x.cos()))
        .collect();
    (r, values)
}

fn real(sampled: Sampled) -> Vec<f64> {
    match sampled {
        Sampled::Real(v) => v,
        Sampled::Complex(_) => panic!("expected a real field"),
    }
}

#[test]
fn an_s_orbital_is_its_radial_function_over_root_four_pi() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let (r, v) = table(0);
    let orbital = Orbital { r: &r, values: &v, l: 0, m: 0, center: [0.1, 0., 0.], real: true };
    let (x, y, z) = ([1.1, 0.1, 0.1], [0., 1., 0.], [0., 0., -2.]);
    let out = real(values(&ctx, &orbital, &x, &y, &z).unwrap());
    let radii = [1.0, 1.0, 2.0];
    let radial = mandacaru_core::basis::table::evaluate(&ctx, &r, &v, 0, &radii, false).unwrap();
    for i in 0..3 {
        assert_eq!(out[i], radial[i] * harmonic(0, 0, 0., 0., true).0);
        assert!((out[i] - radial[i] / (4. * PI).sqrt()).abs() < 1e-15);
    }
}

#[test]
fn the_gradient_is_the_derivative_of_the_orbital() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    for (l, m) in [(0, 0), (1, -1), (1, 0), (2, 1), (3, -2)] {
        let (r, v) = table(l);
        let orbital = Orbital { r: &r, values: &v, l, m: m as i32, center: [0.2, -0.1, 0.3], real: true };
        let point = [1.3, 0.4, -0.7];
        let g = real(gradient(&ctx, &orbital, &[point[0]], &[point[1]], &[point[2]]).unwrap());
        let h = 1e-5;
        for axis in 0..3 {
            let (mut up, mut down) = (point, point);
            up[axis] += h;
            down[axis] -= h;
            let f = |p: [f64; 3]| real(values(&ctx, &orbital, &[p[0]], &[p[1]], &[p[2]]).unwrap())[0];
            let difference = (f(up) - f(down)) / (2. * h);
            assert!((g[axis] - difference).abs() < 1e-7, "l = {l}, m = {m}, axis {axis}");
        }
    }
}

#[test]
fn a_point_on_the_center_is_moved_out_along_z_for_the_gradient() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let (r, v) = table(1);
    let orbital = Orbital { r: &r, values: &v, l: 1, m: 0, center: [0.5, 0.5, 0.5], real: true };
    let g = real(gradient(&ctx, &orbital, &[0.5], &[0.5], &[0.5]).unwrap());
    assert!(g.iter().all(|x| x.is_finite()));
    assert!(NODE_OFFSET > 0.);
}

#[test]
fn the_complex_harmonic_of_a_real_orbital_is_the_standard_one() {
    // Y_1^0 = sqrt(3/4pi) cos(theta); the polar angles of a point on the axis.
    let (r, theta, phi) = polar(0., 0., 2.);
    assert_eq!((r, theta, phi), (2., 0., 0.));
    let value = harmonic(1, 0, theta, phi, false).1;
    assert!((value.re - (3. / (4. * PI)).sqrt()).abs() < 1e-15 && value.im == 0.);
    let (_, _, complex, gradient) = harmonic_gradient(1, 0, 0., 0., 2., false);
    assert!((complex.re - (3. / (4. * PI)).sqrt()).abs() < 1e-15);
    assert!(gradient.iter().all(|g| g.re.is_finite() && g.im.is_finite()));
}
