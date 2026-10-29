use mandacaru_core::math::{angular, radial::gauss_legendre};
use std::f64::consts::PI;

#[test]
fn cartesian_harmonics_match_polar_values_and_exact_shell_normalization() {
    for l in 0..=8 {
        for m in -(l as i32)..=l as i32 {
            let mut norm = 0.;
            for (z, w) in gauss_legendre(20).unwrap() {
                let theta = z.acos();
                for k in 0..40 {
                    let phi = 2. * PI * k as f64 / 40.;
                    let point = [theta.sin() * phi.cos(), theta.sin() * phi.sin(), z];
                    let polar = angular::real(l, m, theta, phi);
                    for scale in [1e-8, 1., 10.] {
                        let cartesian = angular::cartesian(l, m, point.map(|v| v * scale));
                        assert!(
                            (polar - cartesian).abs() < 3e-14,
                            "l={l}, m={m}: {polar} vs {cartesian}"
                        );
                    }
                    let value = angular::cartesian(l, m, point);
                    norm += w * (2. * PI / 40.) * value * value;
                }
            }
            assert!((norm - 1.).abs() < 2e-13, "l={l}, m={m}, norm {norm}");
        }
    }
}
