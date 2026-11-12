//! The Dirac quantum numbers of a subshell (`models::relativity`): `kappa`, `l`, `j`, the
//! degeneracy and the spin-orbit difference, against their definitions.
use mandacaru_core::models::relativity::{
    degeneracy, j_of, kappa_of, kappa_values, l_of, spin_orbit_difference,
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
