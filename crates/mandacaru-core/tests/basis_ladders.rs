//! The shell ladders of the correlation-consistent bases (`basis::ladders`): the counts the
//! published cc-pVXZ family stacks at each level.
use mandacaru_core::basis::ladders::{dunning, MAX_CONTRACTION};

#[test]
fn the_double_triple_and_quadruple_zeta_ladders_stack_one_shell_more_each_level() {
    let d = dunning(2, false).unwrap();
    assert_eq!(d.valence, vec![4, 1]);
    assert_eq!(d.polarization, vec![(2, 1)]);
    assert_eq!(d.polarization_h, vec![(1, 1)]);
    assert!(d.tight.is_empty());
    assert_eq!(d.core, 8);
    let t = dunning(3, false).unwrap();
    assert_eq!(t.valence, vec![4, 1, 1]);
    assert_eq!(t.polarization, vec![(2, 2), (3, 1)]);
    assert_eq!(t.polarization_h, vec![(1, 2), (2, 1)]);
    let q = dunning(4, false).unwrap();
    assert_eq!(q.polarization, vec![(2, 3), (3, 2), (4, 1)]);
    assert_eq!(q.polarization_h, vec![(1, 3), (2, 2), (3, 1)]);
}

#[test]
fn the_core_contraction_is_capped_and_cc_pcvxz_adds_tight_s_and_p_shells() {
    assert_eq!(dunning(3, false).unwrap().core, 10);
    assert_eq!(dunning(6, false).unwrap().core, MAX_CONTRACTION);
    assert_eq!(dunning(3, true).unwrap().tight, vec![(0, 2), (1, 2)]);
    assert!(dunning(1, false).is_none() && dunning(7, true).is_none());
}
