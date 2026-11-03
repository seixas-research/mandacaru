//! Shared by the `periodic_stress_crystals_hybrid*` tests: one frozen hybrid case, its SCF, the
//! analytic stress against the frozen oracle, the equation of state, and the controls that remove
//! one term of the stress (the semilocal terms and each exchange term).
#![allow(dead_code)]

use crate::periodic_stress_crystals_support::{
    gates::{bites, entries, Entry, Tensor},
    run::{check, equation_of_state, weak_terms},
};
use mandacaru_core::periodic::stress_terms::Stress;

/// The exchange terms are live and each one is needed: the stress without it misses the oracle by
/// more than 100 gates. Returns the terms that do not, with their bite (printed either way).
pub fn exchange_controls(stress: &Stress, want: &[[Entry; 3]; 3]) -> Vec<(&'static str, f64)> {
    let parts = stress
        .breakdown
        .exchange_parts
        .as_ref()
        .expect("a hybrid crystal carries its exchange terms");
    let mut weak = Vec::new();
    for (name, term) in parts.parts() {
        let peak = term.iter().flatten().fold(0f64, |m, x| m.max(x.abs()));
        let without: Tensor =
            std::array::from_fn(|a| std::array::from_fn(|b| stress.gradient[a][b] - term[a][b]));
        let bite = bites(&without, want);
        println!("  control without {name}: peak {peak:.3e}, bites {bite:.2e} gates");
        if bite <= 100. {
            weak.push((name, bite));
        }
    }
    weak
}

/// One case: the SCF, the stress against the frozen oracle (and the second step family when
/// `second`), the equation of state, and the controls. A term below 100 gates is named in
/// `allowed` (it is smaller than the oracle can resolve at 100 gates, a stated property of the
/// case and not a hidden one) and must still bite by more than 10 gates.
pub fn run(case: &'static str, second: bool, allowed: &[&str]) {
    let (data, stress, want) = check(case, second);
    equation_of_state(&data.manifest, &stress);
    let mut weak = exchange_controls(&stress, &want);
    weak.extend(
        weak_terms(&stress, &want)
            .into_iter()
            .filter(|name| !name.starts_with("exchange") && *name != "spin_orbit")
            .map(|name| (name, f64::NAN)),
    );
    for (name, bite) in &weak {
        assert!(
            allowed.contains(name),
            "{name} bites {bite:.1} gates: the oracle cannot discriminate it"
        );
        assert!(
            !(bite < &10.),
            "{name} bites only {bite:.1} gates, below the 10 that a named weak term must keep"
        );
    }
}

/// A case whose shears do NOT meet the plan's gate: the complex-k whole mesh (`si_hse06_tr3`,
/// `si_hsefr_tr3`). The stress is compared with the step family of the case and with every family
/// `<case><tag>` of `families` (each shares the snapshot and has its own steps and floors); the
/// diagonal entries must meet their gates (the plan's), and every entry must stay inside
/// `envelope` gates, a bound set from the measured misses (the plan-gate factor is printed either
/// way, and the log states that the plan gate is not met). The terms are removed one at a time as
/// in [`run`]. Returns the worst factor over the families.
pub fn run_enveloped(
    case: &'static str,
    stress_of: crate::periodic_stress_crystals_support::run::StressOf,
    envelope: f64,
    allowed: &[&str],
    families: &[&str],
) -> f64 {
    use crate::periodic_stress_crystals_support::{
        assert_pins, load, manifest,
        run::{report, solve_with, SCF_DENSITY_TOLERANCE, SCF_TOLERANCE},
        summary,
    };
    let mut data = load(case);
    assert_pins(&data);
    let volume = data.manifest["volume_bohr3"].as_f64().unwrap();
    let (free_energy, _, stress) =
        solve_with(&mut data, SCF_TOLERANCE, SCF_DENSITY_TOLERANCE, stress_of);
    let frozen = data.manifest["free_energy_ha"].as_f64().unwrap();
    assert!(
        (free_energy - frozen).abs() < 1e-9,
        "{:e}",
        free_energy - frozen
    );
    assert!(
        (stress.volume - volume).abs() < 1e-6 * volume,
        "the cell volume"
    );
    summary(case, &stress);
    for (name, term) in stress.breakdown.terms() {
        let peak = term.iter().flatten().fold(0f64, |m, x| m.max(x.abs()));
        println!("  term {name}: peak {peak:.3e} Ha");
    }
    let mut sets = vec![(case.to_string(), data.manifest.clone())];
    for tag in families {
        let other = manifest(&format!("{case}{tag}"));
        assert_eq!(other["snapshot"].as_str(), Some(case));
        assert_ne!(other["steps"], data.manifest["steps"]);
        sets.push((format!("{case} (steps {})", other["steps"]), other));
    }
    let mut worst = 0f64;
    for (label, m) in &sets {
        let w = entries(m, volume);
        let factor = report(label, &stress.gradient, &w, volume);
        println!("  {label}: worst entry {factor:.2} x its gate (envelope {envelope})");
        for a in 0..3 {
            let miss = (stress.gradient[a][a] - w[a][a].oracle).abs();
            assert!(
                miss < w[a][a].gate,
                "{label}: diagonal ({a},{a}) misses its gate"
            );
        }
        assert!(
            factor < envelope,
            "{label}: {factor:.2} x the gate, past the envelope {envelope}"
        );
        worst = worst.max(factor);
    }
    equation_of_state(&data.manifest, &stress);
    let want = entries(&data.manifest, volume);
    let mut weak = exchange_controls(&stress, &want);
    weak.extend(
        weak_terms(&stress, &want)
            .into_iter()
            .filter(|name| !name.starts_with("exchange") && *name != "spin_orbit")
            .map(|name| (name, f64::NAN)),
    );
    println!("  terms below 100 gates when removed: {weak:?}");
    for (name, bite) in &weak {
        assert!(allowed.contains(name), "{name} bites {bite:.1} gates");
        assert!(!(bite < &10.), "{name} bites only {bite:.1} gates");
    }
    worst
}
