//! Each term of the analytic PAW-LCAO gradient against its own oracle.
//!
//! The oracle of a term is the central difference (Richardson, steps `1e-3` and
//! `5e-4` Bohr) of the same energy term evaluated from the displaced integrals
//! at a FIXED atomic-orbital density (and, for the overlap terms, a fixed
//! energy-weighted density), so no term borrows from another through the
//! Lowdin carry. A term is gated at 100 times the measured floor of its oracle
//! (`paw_forces_support::floor`); each part of a term is then removed or sign-flipped in a
//! wrong variant, which must miss the oracle by more than 100 gates.
mod paw_forces_support;

use mandacaru_core::{
    molecular::{MeanField, MolecularIntegrals},
    Context,
};
use paw_forces_support::{
    ao_state, bite as bite_of, by_term, floor,
    synthetic::{integrals, solve, Method, H2},
    term_derivatives, GATE,
};
use std::collections::BTreeMap;

struct Case<'a> {
    label: &'a str,
    base: &'a MolecularIntegrals,
    state: &'a MeanField,
    components: &'a [(usize, usize)],
}
/// Returns, per term, the worst |analytic - oracle| and the worst bite of the
/// wrong variants in gates.
fn gate(ctx: &Context, case: Case<'_>) -> BTreeMap<String, (f64, f64, usize, usize)> {
    let ao = ao_state(case.base, case.state);
    let gradient = case.base.gradient(case.state).unwrap();
    let terms = by_term(&gradient);
    let parts: Vec<(&str, &[f64])> = gradient
        .terms
        .iter()
        .map(|(name, values)| (*name, values.as_slice()))
        .collect();
    let mut worst: BTreeMap<String, (f64, f64, usize, usize)> = BTreeMap::new();
    let mut total_sum = vec![0.; gradient.total.as_slice().len()];
    for values in terms.values() {
        for (t, v) in total_sum.iter_mut().zip(values) {
            *t += v;
        }
    }
    assert!(
        total_sum
            .iter()
            .zip(gradient.total.as_slice())
            .all(|(a, b)| (a - b).abs() < 1e-12),
        "the terms do not add up to the total"
    );
    for &(atom, axis) in case.components {
        let oracle = term_derivatives(ctx, case.base, case.state, &ao, atom, axis);
        let index = 3 * atom + axis;
        for (term, (reference, estimate)) in &oracle {
            println!(
                "  {} atom {atom} axis {axis} {term:<20} miss {:.2e} oracle step error {estimate:.2e}",
                case.label,
                (terms[*term][index] - reference).abs()
            );
            let analytic = terms
                .get(*term)
                .unwrap_or_else(|| panic!("{}: no analytic term {term}", case.label))[index];
            let miss = (analytic - reference).abs();
            let (bite, tested, skipped) = bite_of(&parts, term, index, analytic, *reference);
            let entry = worst
                .entry(term.to_string())
                .or_insert((0., f64::INFINITY, 0, 0));
            entry.0 = entry.0.max(miss);
            entry.1 = entry.1.min(bite);
            entry.2 += tested;
            entry.3 += skipped;
            assert!(
                miss.is_finite(),
                "{}: {term} atom {atom} axis {axis} not finite",
                case.label
            );
        }
        for term in terms.keys() {
            assert!(
                oracle.contains_key(term.as_str()),
                "{}: analytic term {term} has no oracle",
                case.label
            );
        }
    }
    for (term, (miss, bite, tested, skipped)) in &worst {
        println!(
            "{}: {term:<20} max |analytic - oracle| = {miss:.3e} Ha/Bohr ({:.1} gates), wrong variants ({tested} tested, {skipped} below 100 gates) miss by >= {bite:.1e} gates",
            case.label,
            miss / (GATE * floor(term))
        );
    }
    worst
}
fn assert_gated(worst: &BTreeMap<String, (f64, f64, usize, usize)>) {
    for (term, (miss, bite, _tested, _skipped)) in worst {
        assert!(*miss < GATE * floor(term), "{term}: {miss:.3e}");
        assert!(
            *bite > GATE,
            "{term}: its wrong variants only miss by {bite:.1} gates"
        );
    }
}

#[test]
fn every_term_of_the_closed_shell_gradient_has_its_own_oracle() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        // Off the bond axis so every direction carries a force.
        let positions = [[0.05, -0.03, -0.8], [0.02, 0.04, 0.8]];
        for (functional, method) in [("pbe", Method::Ks("pbe")), ("lda", Method::Hf("rhf"))] {
            let base = integrals(&ctx, &positions, functional);
            let state = solve(&base, method, [1, 1]);
            let worst = gate(
                &ctx,
                Case {
                    label: &format!(
                        "H2 {functional} {}",
                        matches!(method, Method::Hf(_))
                            .then_some("RHF")
                            .unwrap_or("KS")
                    ),
                    base: &base,
                    state: &state,
                    components: &[(0, 0), (0, 2), (1, 1)],
                },
            );
            assert_gated(&worst);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
#[ignore = "slow: 11 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_forces_terms -- --ignored every_term_of_the_open_shell_gradient_has_its_own_oracle"]
fn every_term_of_the_open_shell_gradient_has_its_own_oracle() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let bent = integrals(
            &ctx,
            &[[0., 0., 0.], [1.7, 0., 0.3], [0.4, 1.6, -0.5]],
            "pbe",
        );
        for method in [Method::Ks("pbe"), Method::Hf("uhf")] {
            let state = solve(&bent, method, [2, 1]);
            let worst = gate(
                &ctx,
                Case {
                    label: "H3 doublet",
                    base: &bent,
                    state: &state,
                    components: &[(0, 0), (1, 2), (2, 1)],
                },
            );
            assert_gated(&worst);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[allow(dead_code)]
fn unused() -> [[f64; 3]; 2] {
    H2
}
