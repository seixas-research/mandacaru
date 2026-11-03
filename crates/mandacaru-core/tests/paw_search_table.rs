//! The attempt table of the search (`paw::generate::search::plan`), the choices a recipe pins and
//! the text of a refusal: static checks that build no dataset. The table is compared with the 33
//! labels the Python search recorded for cerium and with the remedies it recorded for silicon.
mod paw_assemble_support;

use mandacaru_core::models::relativity::Treatment;
use mandacaru_core::paw::generate::{
    recipe::{Cutoffs, Recipe},
    search::{general, plan, Pinned},
};
use paw_assemble_support::Frozen;
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
fn the_attempt_table_is_the_one_the_python_search_recorded() {
    // Cerium's 33 attempts of the freeze: the first construction and 32 remedies, with the
    // balanced radius of its largest channel.
    let frozen = Frozen::open("lda-sr_Ce");
    let attempts = frozen.doc["ghost_free"]["attempts"].as_array().unwrap();
    assert_eq!(attempts.len(), 33);
    let selected = &attempts[8]["generated"];
    let radii = selected["r_cut"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(l, r)| (l.parse::<usize>().unwrap(), r.as_f64().unwrap()))
        .collect::<BTreeMap<_, _>>();
    let recipe = Recipe::for_dataset("Ce", "lda", Treatment::Scalar).unwrap();
    let table = plan(&recipe, &radii);
    assert_eq!(table.len(), 32);
    for (i, attempt) in table.iter().enumerate() {
        assert_eq!(
            attempt.label,
            attempts[i + 1]["label"].as_str().unwrap(),
            "attempt {}",
            i + 1
        );
        assert_eq!(attempt.recipe.norm_deficit, 0.0, "{}", attempt.label);
    }
    // The recorded remedies of the selected attempts of the freeze, as recipe fields.
    assert_eq!(table[7].recipe.local_shift, 40.0);
    let si = Recipe::for_dataset("Si", "lda", Treatment::Scalar).unwrap();
    let shorter = &plan(&si, &BTreeMap::from([(0, 2.3), (1, 2.8)]))[1];
    assert_eq!(shorter.recipe.local_shift, 0.0);
    assert_eq!(shorter.recipe.local_factor, si.local_factor * 0.75);
    let remedy = &Frozen::open("lda-sr_Si").doc["ghost_free"]["attempts"][2]["remedy"];
    assert_eq!(
        remedy["local_factor"].as_f64(),
        Some(shorter.recipe.local_factor)
    );
    // The balanced attempts give every channel the largest radius and a local radius to match.
    let balanced = &table[27];
    assert_eq!(balanced.label, "balanced at 4.993 Bohr, shift 0");
    match &balanced.recipe.cutoffs {
        Cutoffs::PerChannel(map) => {
            assert!(map.values().all(|&r| r == 4.992701233319157) && map.len() == 2)
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        balanced.recipe.local_radius,
        Some(recipe.local_factor * 4.992701233319157)
    );
    // The shorter s spheres scale l = 0 only.
    match &table[18].recipe.cutoffs {
        Cutoffs::PerChannel(map) => {
            assert_eq!(map[&0], 4.992701233319157 * 0.95);
            assert_eq!(map[&3], 0.944816726244526);
        }
        other => panic!("{other:?}"),
    }
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
