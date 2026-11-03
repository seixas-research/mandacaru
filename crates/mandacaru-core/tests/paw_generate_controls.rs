//! Negative controls of the generator gates (`paw_generate.rs`): each recipe edit below changes
//! the dataset by a physically meaningless amount (1e-8 in an energy or a norm, or a whole
//! choice of the construction), and the comparison with the freeze must lose it by more than
//! 100 times its gate. A gate that passes these is blind to the quantity it claims to check.
mod paw_assemble_support;
mod paw_generate_support;

use mandacaru_core::models::relativity::Treatment;
use mandacaru_core::models::xc_scheme::AtomSolver;
use mandacaru_core::paw::generate::generate_with_atom;
use mandacaru_core::paw::generate::recipe::{Nlcc, Recipe};
use mandacaru_core::paw::generate::reference::solve_reference_with;
use mandacaru_core::Context;
use paw_assemble_support::Frozen;
use paw_generate_support::compare::{Comparison, Scheme};
use paw_generate_support::dataset::compare_dataset;
use paw_generate_support::{recipe_of, Decimated};

/// How far beyond its gate the worst numeric array or scalar of the edited recipe is.
fn bite(name: &str, edit: &dyn Fn(&mut Recipe)) -> (f64, String) {
    let f = Frozen::open(name);
    let data = Decimated::open(name);
    let mut recipe = recipe_of(&f);
    edit(&mut recipe);
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    // The atom is solved with the edited recipe too: the functional and the treatment are the
    // atom's.
    let atom = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
    let generated = generate_with_atom(&ctx, &recipe, &atom, &AtomSolver).unwrap();
    let comparison: Comparison = compare_dataset(&f, &data, &generated, Scheme::Legacy);
    let numeric = comparison.without(|n| {
        !(n.starts_with("array.")
            || n.starts_with("energy.")
            || n.starts_with("scalar.one_center")
            || n.starts_with("handoff."))
    });
    let worst = numeric.worst();
    (worst.ratio(), worst.name.clone())
}

fn bites(name: &str, what: &str, edit: &dyn Fn(&mut Recipe)) {
    let (ratio, at) = bite(name, edit);
    println!("{name} {what}: {ratio:.3e} x the gate at {at}");
    assert!(
        ratio > 100.,
        "{name} {what}: only {ratio:.3e} x the gate (at {at})"
    );
}

#[test]
fn the_gates_resolve_a_change_of_one_part_in_a_hundred_million() {
    bites("lda-sr_O", "energy_offset + 1e-8", &|r| {
        r.energy_offset += 1e-8
    });
    bites("lda-sr_O", "local_shift + 1e-8", &|r| r.local_shift += 1e-8);
    bites("lda-sr_O", "norm_deficit + 1e-8", &|r| {
        r.norm_deficit += 1e-8
    });
    bites("lda-sr_O", "q_cut + 1e-6", &|r| r.q_cut += 1e-6);
}

/// The control of the controls: a field this dataset does not read (the radius comes from the
/// table) must not move it, so the comparison is not simply blind to everything.
#[test]
fn a_field_the_dataset_does_not_read_does_not_move_it() {
    let (ratio, at) = bite("lda-sr_O", &|r| r.rc_factor += 1.);
    println!("rc_factor + 1 (oxygen has its radius in the table): {ratio:.3e} x the gate at {at}");
    assert!(ratio <= 1., "{at}: {ratio:e}");
}

#[test]
fn the_gates_resolve_a_wrong_choice_of_the_construction() {
    bites("lda-sr_O", "no relativistic treatment", &|r| {
        r.relativity = Treatment::None
    });
    bites("lda-sr_O", "the PBE functional", &|r| {
        r.functional = "pbe".into()
    });
    bites("lda-sr_O", "no nonlinear core correction", &|r| {
        r.nlcc = Nlcc::Off
    });
    bites("lda-sr_Mg", "no absolute second reference energy", &|r| {
        r.second_reference.clear()
    });
    bites("lda-sr_O", "a local radius one node away", &|r| {
        r.local_radius = Some(1.3049 + 2e-3);
    });
}

/// A NaN anywhere in a table must reach the gate: the folding maximum keeps it, and a comparison
/// item that carries it fails.
#[test]
fn a_nan_deviation_fails_its_gate() {
    use paw_generate_support::compare::nan_max;
    assert!([0.5, f64::NAN, 0.25].into_iter().fold(0., nan_max).is_nan());
    assert!([f64::NAN, 0.5].into_iter().fold(0., nan_max).is_nan());
    let mut comparison = Comparison::default();
    comparison.push(
        "array.nan",
        [0.5, f64::NAN].into_iter().fold(0., nan_max),
        1e-9,
    );
    assert!(
        comparison.deviation("array.nan") > 1e300
            || comparison.deviation("array.nan").is_infinite()
    );
}
