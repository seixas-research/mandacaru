//! The shell ladders of the named Gaussian families (`basis::ladders`): the counts the
//! cc-pVXZ family stacks at each level, the Pople marks and the def2 table.
use mandacaru_core::basis::ladders::{
    dunning, karlsruhe, polarization_spec, pople, MAX_CONTRACTION,
};

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

#[test]
fn pople_marks_add_d_on_heavy_atoms_p_on_hydrogen_and_diffuse_s_and_p() {
    let star = pople(6, &[3, 1], 0, 1, None).unwrap();
    assert_eq!(star.polarization, vec![(2, 1)]);
    assert!(star.polarization_h.is_empty() && star.diffuse.is_empty());
    assert_eq!(star.polarization_min_z, 3);
    let full = pople(3, &[2, 1], 2, 2, None).unwrap();
    assert_eq!(full.polarization_h, vec![(1, 1)]);
    assert_eq!(full.diffuse, vec![0, 1]);
    assert_eq!(full.diffuse_h, vec![0]);
    // 3-21G*: d functions on the second row only.
    assert_eq!(full.polarization_min_z, 11);
    assert_eq!(
        pople(3, &[0, 2, 1], 0, 1, None).unwrap().polarization_min_z,
        3
    );
    let explicit = pople(6, &[3, 1, 1], 1, 0, Some("2df, 2p")).unwrap();
    assert_eq!(explicit.polarization, vec![(2, 2), (3, 1)]);
    assert_eq!(explicit.polarization_h, vec![(1, 2)]);
    assert_eq!(explicit.diffuse, vec![0, 1]);
    assert!(explicit.diffuse_h.is_empty());
    let light_only = pople(6, &[3, 1], 0, 0, Some(",p")).unwrap();
    assert!(light_only.polarization.is_empty());
    assert_eq!(light_only.polarization_h, vec![(1, 1)]);
}

#[test]
fn a_polarization_spec_is_read_whole_or_refused() {
    assert_eq!(polarization_spec("3DF").unwrap(), vec![(2, 3), (3, 1)]);
    assert_eq!(polarization_spec("12d f").unwrap(), vec![(2, 12), (3, 1)]);
    assert!(polarization_spec("").unwrap().is_empty());
    for bad in ["x", "1x d", "dq", "0d", "3"] {
        assert!(polarization_spec(bad).is_err(), "{bad}");
    }
    assert!(pople(6, &[3, 1], 0, 0, Some("d,p,f")).is_err());
    assert!(pople(6, &[3, 1], 0, 1, Some("d")).is_err());
    assert!(pople(6, &[3, 1], 3, 0, None).is_err());
}

#[test]
fn the_def2_table_stacks_tight_core_and_valence_shells_by_zeta() {
    let svp = karlsruhe("SV(P)").unwrap();
    assert_eq!((svp.core, svp.valence), (vec![5], vec![3, 1]));
    assert!(svp.polarization_h.is_empty() && !svp.diffuse_present);
    assert_eq!(karlsruhe("SVP").unwrap().polarization_h, vec![(1, 1)]);
    assert_eq!(karlsruhe("TZDP"), karlsruhe("TZVP"));
    let tz = karlsruhe("TZVPPD").unwrap();
    assert_eq!((tz.core, tz.valence), (vec![6, 1], vec![3, 1, 1]));
    assert_eq!(tz.polarization, vec![(2, 2), (3, 1)]);
    assert_eq!(tz.polarization_h, vec![(1, 2), (2, 1)]);
    assert!(tz.diffuse_present && tz.diffuse_l_max == 2);
    let qz = karlsruhe("QZVPP").unwrap();
    assert_eq!(qz.core, vec![7, 1, 1]);
    assert_eq!(qz.polarization, vec![(2, 3), (3, 2), (4, 1)]);
    assert_eq!(qz.polarization_h, vec![(1, 3), (2, 2), (3, 1)]);
    assert!(karlsruhe("XYZ").is_none() && karlsruhe("svp").is_none());
}
