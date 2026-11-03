//! What a caller names of a generation (`paw::generate::request`): the tables answer for
//! everything unnamed, a named choice is pinned for the search whatever its value, the family
//! picks the recipe, and an unknown functional or treatment raises.
use mandacaru_core::models::relativity::Treatment;
use mandacaru_core::paw::generate::{
    recipe::{Cutoffs, Nlcc, Recipe},
    request::{resolve, Overrides},
};
use mandacaru_core::paw::record::Family;
use std::collections::BTreeMap;

#[test]
fn nothing_named_is_the_table() {
    let (recipe, pinned) =
        resolve("O", Family::PawLcao, "lda", "scalar", Overrides::default()).unwrap();
    assert_eq!(
        recipe,
        Recipe::for_dataset("O", "lda", Treatment::Scalar).unwrap()
    );
    assert!(!pinned.any(), "nothing was named and a PAW-LCAO table pins nothing");
    let (upaw, _) = resolve(
        "Cr",
        Family::UpawLcao,
        "lda",
        "scalar",
        Overrides::default(),
    )
    .unwrap();
    assert_eq!(upaw, Recipe::upaw("Cr", "lda", Treatment::Scalar).unwrap());
    assert_eq!(upaw.family, Family::UpawLcao);
}

#[test]
fn a_named_choice_is_applied_and_pinned_even_when_it_equals_the_table() {
    let table = Recipe::for_dataset("O", "lda", Treatment::Scalar).unwrap();
    let named = Overrides {
        local_shift: Some(table.local_shift),
        ..Overrides::default()
    };
    let (recipe, pinned) = resolve("O", Family::PawLcao, "lda", "scalar", named).unwrap();
    assert_eq!(recipe, table);
    assert!(pinned.local_shift, "the caller named it");
    assert!(!pinned.r_cut && !pinned.r_cut_local);

    let many = Overrides {
        r_cut: Some(Cutoffs::PerChannel(BTreeMap::from([(0, 1.3), (1, 1.4)]))),
        r_cut_local: Some(1.1),
        local_shift: Some(-3.0),
        n_bessel: Some(10),
        nlcc: Some(Nlcc::Off),
        extra_l: Some(1),
        ..Overrides::default()
    };
    let (recipe, pinned) = resolve("O", Family::PawLcao, "lda", "scalar", many).unwrap();
    assert_eq!(recipe.local_radius, Some(1.1));
    assert_eq!(recipe.local_shift, -3.0);
    assert_eq!(
        (recipe.n_bessel, recipe.extra_l, recipe.nlcc),
        (10, 1, Nlcc::Off)
    );
    assert!(matches!(recipe.cutoffs, Cutoffs::PerChannel(_)));
    assert!(pinned.r_cut && pinned.r_cut_local && pinned.local_shift);
}

#[test]
fn the_radii_a_family_table_lays_under_the_caller_are_pinned_too() {
    let (recipe, pinned) = resolve(
        "Cr",
        Family::UpawLcao,
        "lda",
        "scalar",
        Overrides::default(),
    )
    .unwrap();
    assert!(matches!(recipe.cutoffs, Cutoffs::PerChannel(_)));
    assert!(pinned.r_cut, "the UPAW-LCAO table assigns the radii of Cr");
}

#[test]
fn an_unknown_functional_or_treatment_raises() {
    for (functional, relativity) in [("hse06", "scalar"), ("lda", "relativistic-ish")] {
        let error = resolve(
            "O",
            Family::PawLcao,
            functional,
            relativity,
            Overrides::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains(functional) || error.contains(relativity),
            "{error}"
        );
    }
}
