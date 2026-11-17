//! Closed-form checks of the one-center densities (`paw::generate::core`): the smooth density
//! matches value, slope and curvature at the cutoff and the fit is exact on a polynomial, and a
//! hydrogenic core density integrates to its occupancy.

use mandacaru_core::math::radial::local_derivatives;
use mandacaru_core::paw::generate::core::{core_density, pseudize_density, CoreShell};
use std::f64::consts::PI;

#[test]
fn the_smooth_density_matches_value_slope_and_curvature_and_the_fit_is_exact_on_a_polynomial() {
    let r = (1..=3000).map(|i| i as f64 * 0.004).collect::<Vec<_>>();
    // 1 + 2 r^2 + 3 r^4 - r^6: the degree-8 fit is exact, so the derivatives at r_cut are.
    let density = r
        .iter()
        .map(|x| 1. + 2. * x * x + 3. * x.powi(4) - x.powi(6))
        .collect::<Vec<_>>();
    let cut = r[249]; // 1.0
    let at = local_derivatives(&r, &density, cut, 4, 25, 8).unwrap();
    let want = [
        1. + 2. + 3. - 1.,
        4. * cut + 12. * cut.powi(3) - 6. * cut.powi(5),
        4. + 36. * cut.powi(2) - 30. * cut.powi(4),
        72. * cut - 120. * cut.powi(3),
        72. - 360. * cut.powi(2),
    ];
    for (k, w) in want.iter().enumerate() {
        assert!(
            (at[k] - w).abs() <= 1e-8 * w.abs().max(1.),
            "derivative {k}: {} against {w}",
            at[k]
        );
    }
    let smooth = pseudize_density(&r, &density, cut + 1e-4).unwrap(); // snaps to the node
    for i in 0..r.len() {
        if r[i] > cut {
            assert_eq!(smooth[i], density[i]);
        }
    }
    // Inside it is the even quartic through the same value, slope and curvature.
    assert!((smooth[249] - density[249]).abs() < 1e-8);
    // Inside it is the even quartic p0 + p1 r^2 + p2 r^4: three inside nodes determine it, and
    // its value, slope and curvature at r_cut are those of the density.
    let nodes = [10usize, 100, 200];
    let system = nalgebra::Matrix3::from_fn(|i, j| r[nodes[i]].powi(2 * j as i32));
    let p = system
        .lu()
        .solve(&nalgebra::Vector3::from_fn(|i, _| smooth[nodes[i]]))
        .unwrap();
    for j in [20usize, 50, 150, 230] {
        let predicted = p[0] + p[1] * r[j].powi(2) + p[2] * r[j].powi(4);
        assert!(
            (predicted - smooth[j]).abs() < 1e-9,
            "{j}: {predicted} against {}",
            smooth[j]
        );
    }
    let value = p[0] + p[1] * cut.powi(2) + p[2] * cut.powi(4);
    let slope = 2. * p[1] * cut + 4. * p[2] * cut.powi(3);
    let curvature = 2. * p[1] + 12. * p[2] * cut.powi(2);
    for (got, want) in [(value, at[0]), (slope, at[1]), (curvature, at[2])] {
        assert!((got - want).abs() < 1e-7, "{got} against {want}");
    }
}

#[test]
fn a_hydrogenic_core_density_integrates_to_its_occupancy() {
    let r = (1..=12000).map(|i| i as f64 * 0.0025).collect::<Vec<_>>();
    let z: f64 = 3.;
    let u = r
        .iter()
        .map(|x| 2. * z.powf(1.5) * x * (-z * x).exp())
        .collect::<Vec<_>>();
    let density = core_density(
        &r,
        &[CoreShell {
            occupancy: 2.,
            orbital: &u,
        }],
    )
    .unwrap();
    for (x, rho) in r.iter().zip(&density).step_by(997) {
        assert!((rho - 2. * z.powi(3) * (-2. * z * x).exp() / PI).abs() < 1e-12 * z.powi(3));
    }
    let shell = r
        .iter()
        .zip(&density)
        .map(|(x, rho)| 4. * PI * x * x * rho)
        .collect::<Vec<_>>();
    let electrons = mandacaru_core::math::radial::integrate(&r, &shell).unwrap();
    // The trapezoid rule leaves out the first interval, 1.7e-6 electrons here.
    assert!((electrons - 2.).abs() < 1e-5, "{electrons}");
    let none = core_density(&r, &[]).unwrap();
    assert!(none.iter().all(|x| *x == 0.));
    assert!(core_density(
        &r,
        &[CoreShell {
            occupancy: 1.,
            orbital: &u[1..]
        }]
    )
    .is_err());
}
