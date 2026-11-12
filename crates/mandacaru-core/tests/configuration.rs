//! Electron configurations of a spherical reference atom (`models::configuration`): the aufbau
//! filling, the valence set and the one-electron rearrangements a reference atom is chosen among.
//! These are the invariants the retired Python copy (`basis/_config.py`) was tested for.
use mandacaru_core::models::configuration::{ground_state, rearrangements, valence_subshells};

/// The default number of unoccupied subshells a move may land in.
const VIRTUALS: usize = 3;

fn count(configuration: &[(usize, usize, f64)], state: (usize, usize)) -> f64 {
    configuration
        .iter()
        .find(|s| (s.0, s.1) == state)
        .map_or(0., |s| s.2)
}

#[test]
fn the_first_candidate_is_aufbau() {
    for z in [8, 26, 57, 90] {
        assert_eq!(rearrangements(z, VIRTUALS).unwrap()[0], ground_state(z).unwrap());
    }
}

#[test]
fn a_main_group_atom_has_nothing_to_rearrange() {
    // Only s, d and f compete; an s -> p move is an excitation.
    assert_eq!(
        rearrangements(8, VIRTUALS).unwrap(),
        vec![ground_state(8).unwrap()]
    );
}

#[test]
fn every_candidate_keeps_the_electron_count() {
    for z in [26u8, 57, 89, 90] {
        for candidate in rearrangements(z, VIRTUALS).unwrap() {
            let total: f64 = candidate.iter().map(|s| s.2).sum();
            assert_eq!(total, z as f64, "Z = {z}");
        }
    }
}

#[test]
fn lanthanum_can_move_its_4f_electron_to_5d() {
    let moved = rearrangements(57, VIRTUALS).unwrap().into_iter().any(|c| {
        count(&c, (5, 2)) == 1. && count(&c, (4, 3)) == 0.
    });
    assert!(moved);
}

#[test]
fn no_destination_opens_a_new_principal_shell() {
    // Lanthanum's 4f -> 7s would leave a one-electron valence.
    for z in [57u8, 89, 90] {
        let outer = ground_state(z).unwrap().iter().map(|s| s.0).max().unwrap();
        for candidate in rearrangements(z, VIRTUALS).unwrap() {
            assert!(candidate.iter().map(|s| s.0).max().unwrap() <= outer, "Z = {z}");
        }
    }
}

#[test]
fn thorium_still_reaches_6d() {
    // Only two empty subshells remain in the filling table at Z = 90.
    assert!(rearrangements(90, VIRTUALS)
        .unwrap()
        .iter()
        .any(|c| count(c, (6, 2)) > 0.));
}

#[test]
fn the_valence_follows_the_configuration_given() {
    let mut configuration = ground_state(57).unwrap();
    configuration.retain(|s| (s.0, s.1) != (4, 3));
    configuration.push((4, 3, 0.));
    configuration.push((5, 2, 1.));
    configuration.sort_by_key(|s| (s.0, s.1));
    let valence = valence_subshells(&configuration);
    assert!(valence.contains(&(5, 2)));
    assert!(!valence.contains(&(4, 3)));
}

#[test]
fn semicore_d_and_f_shells_belong_in_the_valence() {
    let valence = |z: u8| valence_subshells(&ground_state(z).unwrap());
    assert_eq!(valence(1), vec![(1, 0)]);
    assert_eq!(valence(6), vec![(2, 0), (2, 1)]);
    assert_eq!(valence(14), vec![(3, 0), (3, 1)]);
    assert_eq!(valence(20), vec![(4, 0)]);
    assert_eq!(valence(26), vec![(3, 2), (4, 0)]); // 3d must not be dropped
    assert_eq!(valence(30), vec![(3, 2), (4, 0)]);
    assert_eq!(valence(31), vec![(3, 2), (4, 0), (4, 1)]);
    assert!(valence(74).contains(&(4, 3))); // tungsten keeps its f shell
    // The regression: taking only the highest n gave Fe a 4s^2 valence.
    let iron = ground_state(26).unwrap();
    let electrons: f64 = valence(26).iter().map(|&s| count(&iron, s)).sum();
    assert_eq!(electrons, 8.);
}
