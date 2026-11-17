//! The choices a recipe pins and the text of a refusal: static checks that build no dataset.

use mandacaru_core::models::relativity::Treatment;
use mandacaru_core::paw::generate::{
    recipe::{Cutoffs, Recipe},
    search::{general, Pinned},
};
use std::collections::BTreeMap;

#[test]
fn what_a_recipe_pins() {
    let free = Recipe::for_dataset("O", "lda", Treatment::Scalar).unwrap();
    assert_eq!(Pinned::of(&free), Pinned::default());
    let radii = Recipe {
        cutoffs: Cutoffs::Uniform(1.5),
        local_radius: Some(1.2),
        ..free.clone()
    };
    assert_eq!(
        Pinned::of(&radii),
        Pinned {
            r_cut: true,
            r_cut_local: true,
            local_shift: false
        }
    );
    // The UPAW-LCAO table's raise is the caller's choice for the generator this replaces.
    let upaw = Recipe::upaw("Al", "lda", Treatment::Scalar).unwrap();
    assert!(Pinned::of(&upaw).local_shift && !Pinned::of(&upaw).r_cut);
    let plain = Recipe::upaw("O", "lda", Treatment::Scalar).unwrap();
    assert_eq!(Pinned::of(&plain), Pinned::default());
}

#[test]
fn ghost_depths_print_like_the_text_they_replace() {
    for (value, text) in [
        (-0.133, "-0.133"),
        (-0.0944, "-0.0944"),
        (-4.85, "-4.85"),
        (-0.8677, "-0.868"),
        (1234.5, "+1.23e+03"),
        (1.0e-5, "+1e-05"),
        (12.0, "+12"),
        (0.000123456, "+0.000123"),
        (0.0, "+0"),
        (-100.0, "-100"),
        (999.6, "+1e+03"),
    ] {
        assert_eq!(general(value, 3, true), text, "{value}");
    }
    assert_eq!(general(0.75, 6, false), "0.75");
    assert_eq!(general(40.0, 6, false), "40");
}

#[test]
fn a_missing_state_is_a_defect_worded_as_one() {
    use mandacaru_core::paw::diagnose::Diagnosis;
    use mandacaru_core::paw::generate::search::{Tried, Why};
    // A construction whose only defect is a bound reference the smooth Hamiltonian does not
    // bind is not clean, is reported after every other defect, and is worded for what it is.
    let only = Diagnosis {
        missing: BTreeMap::from([(2, 0.0123)]),
        ..Diagnosis::default()
    };
    assert!(!only.is_clean());
    let why = Why::of(&only).expect("a missing state fails the construction");
    assert_eq!(
        why.problem(),
        "a bound reference the smooth Hamiltonian does not bind (l=2 +0.0123 Ha)"
    );
    let tried = Tried {
        label: "own cutoffs, shift 5".into(),
        why,
    };
    assert_eq!(
        tried.to_string(),
        "own cutoffs, shift 5: missing state l=2 +0.0123 Ha"
    );
    // A ghost is reported before it, and a clean diagnosis is not a defect at all.
    let both = Diagnosis {
        ghosts: BTreeMap::from([(0, -0.5)]),
        ..only
    };
    assert!(matches!(Why::of(&both), Some(Why::Ghost(_))));
    assert!(Why::of(&Diagnosis::default()).is_none());
}
