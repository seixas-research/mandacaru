//! The Dirac quantum numbers of a subshell (`models::relativity`): `kappa`, `l`, `j`, the
//! degeneracy, the spin-orbit difference and the hydrogenic Dirac level, against their
//! definitions.
use mandacaru_core::models::relativity::{
    degeneracy, hydrogenic_dirac_energy, j_average, j_of, kappa_of, kappa_values, l_of,
    spin_orbit_difference, SPEED_OF_LIGHT,
};

#[test]
fn kappa_and_l_and_j_round_trip_for_every_subshell() {
    for l in 0..5usize {
        for kappa in kappa_values(l) {
            assert_eq!(l_of(kappa).unwrap(), l);
            assert_eq!(kappa_of(l, j_of(kappa)).unwrap(), kappa);
            assert_eq!(degeneracy(kappa) as f64, 2. * j_of(kappa) + 1.);
        }
    }
    assert_eq!(kappa_values(0), vec![-1]);
    assert_eq!(kappa_values(2), vec![2, -3]);
}

#[test]
fn an_impossible_j_or_a_zero_kappa_is_refused() {
    assert!(kappa_of(1, 0.7).is_err());
    assert!(kappa_of(0, -0.5).is_err());
    assert!(l_of(0).is_err());
}

#[test]
fn the_spin_orbit_difference_is_weighted_and_zero_for_s() {
    assert_eq!(spin_orbit_difference(0, 3., 1.), 0.);
    assert_eq!(spin_orbit_difference(1, 0.5, 0.2), 2. / 3. * (0.5 - 0.2));
}

#[test]
fn the_hydrogenic_dirac_level_has_its_limits_and_refuses_an_unbound_state() {
    // Far from relativity it is the Schroedinger level -Z^2 / 2n^2.
    let far = hydrogenic_dirac_energy(10., 3, -2, 1e7).unwrap();
    assert!((far + 100. / 18.).abs() < 1e-9, "{far}");
    // It depends on n and j only: 2s_1/2 and 2p_1/2 are degenerate.
    let s = hydrogenic_dirac_energy(20., 2, -1, SPEED_OF_LIGHT).unwrap();
    let p = hydrogenic_dirac_energy(20., 2, 1, SPEED_OF_LIGHT).unwrap();
    assert!((s - p).abs() <= 1e-14 * s.abs());
    assert!(hydrogenic_dirac_energy(1., 1, 2, SPEED_OF_LIGHT).is_err());
    assert!(hydrogenic_dirac_energy(140., 1, -1, SPEED_OF_LIGHT).is_err());
}

#[test]
fn the_j_average_weights_each_j_by_its_degeneracy() {
    assert_eq!(j_average(0, |_| -0.25), -0.25);
    let levels = |kappa: i32| if kappa > 0 { -1.0 } else { -0.4 };
    assert_eq!(j_average(1, levels), (2. * -1.0 + 4. * -0.4) / 6.);
}

#[test]
fn the_mass_terms_and_the_two_components_follow_their_definitions() {
    use mandacaru_core::math::radial::uniform_grid;
    use mandacaru_core::models::relativity::{
        large_component, mass_terms, small_component, spin_orbit_radial, PotentialTerms,
    };
    let r = uniform_grid(400, 10.0);
    assert_eq!(r.len(), 400);
    assert!((r[0] - 10.0 / 401.0).abs() < 1e-15 && r[399] < 10.0);
    let z = 4.0;
    let potential: Vec<f64> = r.iter().map(|r| -z / r * (-0.3 * r).exp()).collect();
    let terms = PotentialTerms::new(&r, &potential, z).unwrap();
    let (m, dm, d2m) = mass_terms(&r, &potential, -0.5, &terms, SPEED_OF_LIGHT);
    let two_c2 = 2.0 * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
    assert_eq!(m[10], (1.0 + (-0.5 - potential[10]) / two_c2).max(1e-3));
    assert_eq!(dm[10], -terms.dv[10] / two_c2);
    assert_eq!(d2m[10], -terms.second_derivative(&r)[10] / two_c2);
    let xi = spin_orbit_radial(&r, &potential, &terms, SPEED_OF_LIGHT, false, 0.0);
    assert_eq!(xi[5], terms.dv[5] / (two_c2 * r[5]));
    let w: Vec<f64> = r.iter().map(|r| r * (-r).exp()).collect();
    let p = large_component(&w, &m).unwrap();
    assert_eq!(p[3], w[3] * m[3].sqrt());
    let q = small_component(&r, &p, -1, &m, SPEED_OF_LIGHT).unwrap();
    assert_eq!(q.len(), r.len());
    assert!(large_component(&w[..3], &m).is_err());
    assert!(small_component(&r[..2], &p[..2], -1, &m[..2], SPEED_OF_LIGHT).is_err());
}
