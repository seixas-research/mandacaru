//! The defaults and the refusals of the PAW-LCAO basis recipe (`paw::basis_recipe`), the values
//! the Python interface reads from Rust instead of repeating them.
use mandacaru_core::basis::confinement::{AMPLITUDE, INNER_FRACTION};
use mandacaru_core::paw::basis_recipe::{
    Recipe, DEFAULT_ENERGY_SHIFT, DEFAULT_WALL, MIN_ENERGY_SHIFT,
};
use serde_json::json;

#[test]
fn the_default_wall_is_the_confinement_potentials_own() {
    assert_eq!(DEFAULT_WALL, (AMPLITUDE, INNER_FRACTION));
    assert_eq!((AMPLITUDE, INNER_FRACTION), (12., 0.6));
    assert_eq!(DEFAULT_ENERGY_SHIFT, 0.1);
    assert_eq!(MIN_ENERGY_SHIFT, 1e-3);
}

#[test]
fn a_new_recipe_confines_at_the_default_shift_in_the_default_wall() {
    let recipe = Recipe::new(2, 1, None);
    assert_eq!(recipe.energy_shift, Some(DEFAULT_ENERGY_SHIFT));
    assert_eq!(recipe.wall, DEFAULT_WALL);
}

#[test]
fn an_energy_shift_below_the_minimum_is_refused_and_the_minimum_is_accepted() {
    let request = |shift: f64| json!({"zetas": 1, "polarizations": 0, "energy_shift": shift});
    let message = Recipe::from_json(&request(MIN_ENERGY_SHIFT / 2.), None)
        .unwrap_err()
        .to_string();
    assert!(message.contains("energy_shift must be at least"), "{message}");
    assert!(Recipe::from_json(&request(MIN_ENERGY_SHIFT), None).is_ok());
    let unconfined = json!({"zetas": 1, "polarizations": 0, "energy_shift": null});
    assert_eq!(Recipe::from_json(&unconfined, None).unwrap().energy_shift, None);
}
