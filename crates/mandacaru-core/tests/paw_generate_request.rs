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

/// A fully relativistic dataset is the scalar-relativistic construction plus the j-resolved
/// spin-orbit term, so its fr - sr difference is a spin-orbit effect only if both split core and
/// valence the same way. The recipe of every element shares its partition (frozen and semicore
/// subshells, reference occupancies) and every radius and pseudization choice across the two
/// treatments; only the treatment differs. (The shipped fr files that do not, Fe 16/8, Au 19/25,
/// Ta 13/27 valence electrons, are stale files, not a property of today's recipes.)
#[test]
fn the_scalar_and_fully_relativistic_recipes_share_one_partition() {
    use mandacaru_core::models::elements::SYMBOLS;
    for symbol in SYMBOLS.iter().take(103) {
        for functional in ["lda", "pbe"] {
            let sr = Recipe::for_dataset(symbol, functional, Treatment::Scalar).unwrap();
            let mut fr = Recipe::for_dataset(symbol, functional, Treatment::Dirac).unwrap();
            assert_eq!(fr.relativity, Treatment::Dirac);
            fr.relativity = Treatment::Scalar;
            assert_eq!(sr, fr, "{symbol} {functional}");
        }
    }
}

/// Radii laid over the policy table are consulted by the recipe's rule and are not pinned, so
/// the search keeps its remedies; the same radii passed as `r_cut` are pinned.
#[test]
fn a_cutoff_table_is_laid_over_the_policy_table_without_pinning() {
    let table = Recipe::for_dataset("Gd", "lda", Treatment::Scalar).unwrap();
    let radii = BTreeMap::from([(0, 2.4), (1, 2.4), (2, 2.4), (3, 1.6)]);
    let laid = Overrides {
        cutoff_table: Some(radii.clone()),
        ..Overrides::default()
    };
    let (recipe, pinned) = resolve("Gd", Family::PawLcao, "lda", "scalar", laid).unwrap();
    assert_eq!(recipe.cutoffs, Cutoffs::Rule);
    for (l, r) in &radii {
        assert_eq!(recipe.cutoff_table.get(l), Some(r));
    }
    assert_eq!(pinned, Default::default());
    assert_ne!(recipe, table);
    let named = Overrides {
        r_cut: Some(Cutoffs::PerChannel(radii)),
        ..Overrides::default()
    };
    let (_, pinned) = resolve("Gd", Family::PawLcao, "lda", "scalar", named).unwrap();
    assert!(pinned.r_cut);
}
