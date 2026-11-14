//! The numbers an SDK adapter consumes (`execution`): device-force sizing, job packing and gate
//! tallies, each against a count made by hand.
use mandacaru_core::execution::{device_force_cost, equal_chunks, gate_tally, pack_jobs, reversed_wire};

fn names(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

#[test]
fn a_device_force_costs_one_plus_two_rdm_circuits_per_rotation_and_their_shots() {
    // 5 rotations: 11 RDM circuits of 4 bases each, plus 21 Hessian circuits, at 1000 shots.
    assert_eq!(device_force_cost(5, 21, 4, 1000).unwrap(), (11, 65, 65_000));
    assert_eq!(device_force_cost(0, 0, 3, 10).unwrap(), (1, 3, 30));
    assert!(device_force_cost(u64::MAX, 1, 1, 1).is_err());
}

#[test]
fn requests_pack_into_jobs_in_order_and_an_oversized_one_fills_a_job_alone() {
    assert_eq!(
        pack_jobs(&[3, 3, 3, 3, 3], Some(7)),
        vec![(vec![0, 1], 6), (vec![2, 3], 6), (vec![4], 3)]
    );
    assert_eq!(
        pack_jobs(&[9, 2, 2], Some(4)),
        vec![(vec![0], 9), (vec![1, 2], 4)]
    );
    assert_eq!(pack_jobs(&[5, 5, 5], None), vec![(vec![0, 1, 2], 15)]);
    assert_eq!(pack_jobs(&[5, 5], Some(0)), vec![(vec![0, 1], 10)]);
    assert!(pack_jobs(&[], Some(4)).is_empty());
}

#[test]
fn gates_tally_by_arity_or_by_name_with_the_skipped_ones_left_out() {
    let gates = vec![
        ("h".to_string(), 1, 4),
        ("cx".to_string(), 2, 3),
        ("rzz".to_string(), 2, 2),
        ("barrier".to_string(), 3, 1),
        ("measure".to_string(), 1, 4),
    ];
    let skipped = names(&["barrier", "measure"]);
    // by arity: rzz counts as an entangler though it is not on the list
    assert_eq!(gate_tally(&gates, &skipped, None), (5, 4, 14));
    // by name: only cx is named, rzz is then single-qubit
    assert_eq!(
        gate_tally(&gates, &skipped, Some(&names(&["CX"]))),
        (3, 6, 14)
    );
    assert_eq!(gate_tally(&[], &[], None), (0, 0, 0));
}

#[test]
fn a_qubit_sits_on_the_wire_counted_from_the_other_end_and_chunks_are_equal_and_whole() {
    assert_eq!((0..4).map(|k| reversed_wire(4, k).unwrap()).collect::<Vec<_>>(), vec![3, 2, 1, 0]);
    assert!(reversed_wire(4, 4).is_err() && reversed_wire(0, 0).is_err());
    assert_eq!(equal_chunks(12, 3), vec![(0, 4), (4, 8), (8, 12)]);
    assert_eq!(equal_chunks(7, 3), vec![(0, 2), (2, 4), (4, 6)], "the remainder belongs to no chunk");
    assert_eq!(equal_chunks(5, 0), Vec::<(usize, usize)>::new());
    assert_eq!(equal_chunks(0, 2), vec![(0, 0), (0, 0)]);
}

