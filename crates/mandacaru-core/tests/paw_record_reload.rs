//! The read-back check of a generated dataset (`paw::record::reload_errors`): a bound level that
//! moved by more than the tolerance fails its channel, a box state or a small move does not.
use mandacaru_core::paw::record::{reload_errors, RELOAD_LEVEL_TOLERANCE};

#[test]
fn only_a_bound_level_that_moved_beyond_the_tolerance_is_reported() {
    let before = vec![(0, vec![-0.5, -0.1]), (1, vec![-0.3, 0.01]), (2, vec![-0.2])];
    let after = vec![
        (0, vec![-0.5 + 2. * RELOAD_LEVEL_TOLERANCE, -0.1 + 0.5 * RELOAD_LEVEL_TOLERANCE]),
        (1, vec![-0.3 + 0.5 * RELOAD_LEVEL_TOLERANCE, 5.0]),
        (2, vec![f64::NAN]),
    ];
    let moved = reload_errors(&before, &after);
    assert_eq!(moved.len(), 2);
    assert!((moved[&0] - 2. * RELOAD_LEVEL_TOLERANCE).abs() < 1e-15);
    assert!(moved[&2].is_nan());
    assert!(!moved.contains_key(&1));
}
