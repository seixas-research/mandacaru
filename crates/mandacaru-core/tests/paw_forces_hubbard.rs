//! The DFT+U term of the molecular PAW-LCAO gradient against independently converged total
//! energies, and the refusal of a +U request the molecule cannot carry.
//!
//! The species is the synthetic hydrogen-like one of `paw_forces_support::synthetic` with a `d`
//! channel added (a test species: the shell is the projection on its `d` partial wave). The
//! reference is the Richardson derivative of reconverged total energies on the same fixed grid,
//! which includes the +U energy and the response of the orbitals to it. The synthetic `d` shell is
//! nearly empty (n_d ~ 1e-6), so this gate covers the projection and Pulay structure of the term
//! (the spin channel, the `m` mapping, the weights, the empty beta channel) in the low-occupation
//! limit, where `V = dE/dn` is nearly the constant `(U - J)/2`. U is 400 eV so that the +U term is
//! 1e-3 to 4e-3 eV/A, a thousand times the 1e-6 eV/A gate. The occupation dependence of `V`
//! (CuH, n_d = 4.9, spin-polarized) is gated in `test/algorithms/test_hubbard.py`.
mod paw_forces_support;

use mandacaru_core::{math::storage::Precision, Context};
use paw_forces_support::synthetic::{
    energy_derivative, integrals_with, solve, try_integrals_with, try_integrals_with_metadata,
    Method, H2,
};

const HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM: f64 = 27.211386245988 / 0.529177210903;

fn dudarev() -> String {
    r#"{"H": {"l": "d", "U": 400.0, "J": 0.0, "form": "dudarev-fll"}}"#.into()
}
fn liechtenstein() -> String {
    r#"{"H": {"l": "d", "U": 400.0, "J": 60.0, "form": "liechtenstein-fll"}}"#.into()
}

fn refusal(request: &str, lmax: usize) -> String {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    match try_integrals_with(&ctx, &H2, "lda", Precision::Float64, lmax, Some(request)) {
        Ok(_) => panic!("the build must refuse {request}"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn a_request_the_molecule_cannot_carry_raises_by_name_and_never_runs_as_plain_dft() {
    // An element that is not in the molecule.
    let ni = r#"{"Ni": {"l": "d", "U": 4.0}}"#;
    assert!(refusal(ni, 2).contains("hubbard_u names Ni, which is not in the system"));
    // One element present and one absent: the request is not partly applied.
    let both = r#"{"H": {"l": "d", "U": 4.0}, "Ni": {"l": "d", "U": 4.0}}"#;
    assert!(refusal(both, 2).contains("hubbard_u names Ni, which is not in the system"));
    // A shell the dataset has no channel for.
    let f = r#"{"H": {"l": "f", "U": 4.0}}"#;
    assert!(refusal(f, 2).contains("does not have channel l = 3"));
    // The same dataset without the d channel.
    assert!(refusal(&dudarev(), 1).contains("does not have channel l = 2"));
}

#[test]
fn a_hubbard_metadata_value_that_is_not_the_request_text_raises_and_null_means_none() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    for value in [
        serde_json::json!({"H": {"l": "d", "U": 4.0}}),
        serde_json::json!(true),
        serde_json::json!(4),
    ] {
        let built =
            try_integrals_with_metadata(&ctx, &H2, "lda", Precision::Float64, 2, Some(value));
        let error = built
            .err()
            .expect("a value that is not text must raise")
            .to_string();
        assert!(
            error.contains("'hubbard' metadata must be the JSON text"),
            "{error}"
        );
    }
    let null = try_integrals_with_metadata(
        &ctx,
        &H2,
        "lda",
        Precision::Float64,
        2,
        Some(serde_json::Value::Null),
    );
    assert!(null
        .unwrap()
        .corrections
        .as_ref()
        .unwrap()
        .hubbard
        .is_none());
}

#[test]
fn the_hartree_fock_gradient_of_a_molecule_with_hubbard_u_raises() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let base = integrals_with(&ctx, &H2, "lda", Precision::Float64, 2, Some(&dudarev()));
    let state = solve(&base, Method::Hf("rhf"), [1, 1]);
    let error = base
        .gradient(&state)
        .err()
        .expect("no +U force for an energy without +U")
        .to_string();
    assert!(
        error.contains("Hartree-Fock gradient of a molecule with hubbard_u"),
        "{error}"
    );
}

#[test]
fn a_request_the_molecule_carries_builds_its_shells() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let base = integrals_with(&ctx, &H2, "lda", Precision::Float64, 2, Some(&dudarev()));
    let hubbard = base.corrections.as_ref().unwrap().hubbard.as_ref().unwrap();
    assert_eq!(hubbard.shells.len(), 2);
    assert!(hubbard.shells.iter().all(|s| s.l == 2 && s.symbol == "H"));
}

/// Largest |F_analytic - F_reference| in eV/Angstrom over `components`, and the size of the +U
/// term itself (so that the gate is not passed by a term that is zero).
fn gate(
    label: &str,
    request: &str,
    method: Method,
    particles: [usize; 2],
    components: &[(usize, usize)],
) -> (f64, f64) {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let base = integrals_with(&ctx, &H2, "lda", Precision::Float64, 2, Some(request));
    let state = solve(&base, method, particles);
    let gradient = base.gradient(&state).unwrap();
    let to_ev = HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM;
    let term = gradient
        .terms
        .iter()
        .find(|(name, _)| *name == "hubbard/projection")
        .expect("a +U molecule has a hubbard/projection term");
    let (mut worst, mut size) = (0f64, 0f64);
    for &(atom, axis) in components {
        let reference = energy_derivative(&base, method, particles, atom, axis);
        let analytic = gradient.total.as_slice()[3 * atom + axis];
        worst = worst.max((analytic - reference).abs() * to_ev);
        size = size.max(term.1.as_slice()[3 * atom + axis].abs() * to_ev);
    }
    println!("{label}: max |F - F_scf| = {worst:.3e} eV/A, +U term {size:.3e} eV/A");
    (worst, size)
}

#[test]
#[ignore = "slow: 47 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test paw_forces_hubbard -- --ignored"]
fn the_hubbard_force_follows_total_energies_for_both_forms_restricted_and_unrestricted() {
    for (label, request, particles) in [
        ("Dudarev restricted", dudarev(), [6, 6]),
        ("Liechtenstein restricted", liechtenstein(), [6, 6]),
        ("Dudarev unrestricted", dudarev(), [7, 5]),
        ("Liechtenstein unrestricted", liechtenstein(), [7, 5]),
        ("Dudarev unrestricted, no beta electron", dudarev(), [2, 0]),
    ] {
        let (worst, size) = gate(
            label,
            &request,
            Method::Ks("lda"),
            particles,
            &[(0, 2), (1, 2)],
        );
        assert!(
            size > 1e-4,
            "{label}: the +U term is {size:.3e} eV/A, too small to test"
        );
        assert!(worst < 1e-6, "{label}: {worst:.3e} eV/A");
    }
}
