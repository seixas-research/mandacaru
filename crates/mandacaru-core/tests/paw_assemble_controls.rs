//! Controls of the A2 gates (`paw_assemble.rs`): the nearest-node fit against the frozen analytic
//! cases (`reference/paw_generate/misc.json`, which record no floor, so the Rust fit's own 1-ulp
//! sensitivity stands in for it), and the negative controls that show each gate resolves a real
//! error: a channel without its shell, a local potential moved by 1e-9, a wrong exchange scheme,
//! the older `derivatives` fit and a density moved by 1e-9 each miss the freeze by more than 100
//! times the gate.
mod paw_assemble_support;

use mandacaru_core::math::radial::{derivatives, local_derivatives};
use mandacaru_core::paw::generate::energies::{
    hartree_energy, xc_energies, AtomSolver, Variational, XcSource,
};
use mandacaru_core::Context;
use paw_assemble_support::call::{call, outputs, reals};
use paw_assemble_support::{directory, group_of, Frozen};
use serde_json::Value;
use std::collections::BTreeMap;

const GATE: f64 = 1e-12;

fn context() -> Context {
    Context::new(Some(1), 512 << 20).unwrap()
}

#[test]
fn the_nearest_node_fit_reproduces_the_frozen_analytic_cases() {
    let path = directory().join("misc.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let fits = &doc["_local_derivatives"];
    let (n, h) = (
        fits["grid"]["n"].as_u64().unwrap() as usize,
        fits["grid"]["h"].as_f64().unwrap(),
    );
    let r = (1..=n).map(|i| i as f64 * h).collect::<Vec<_>>();
    let function = |name: &str| -> Vec<f64> {
        match name {
            "gauss" => r
                .iter()
                .map(|x| (-0.5 * (x - 7.).powi(2)).exp() + 0.3 * (1.3 * x).sin())
                .collect(),
            "decay" => r
                .iter()
                .map(|x| (-x).exp() * x * x - 0.01 * x.powi(3))
                .collect(),
            other => panic!("unknown frozen function {other}"),
        }
    };
    let mut worst = 0f64;
    let cases = fits["cases"].as_array().unwrap();
    for case in cases {
        let get = |k: &str| case[k].as_u64().unwrap() as usize;
        let at = case["r0"].as_f64().unwrap();
        let values = function(case["function"].as_str().unwrap());
        let ours =
            local_derivatives(&r, &values, at, get("order"), get("window"), get("degree")).unwrap();
        let frozen = reals(&case["output"]);
        let scale = frozen.iter().fold(0f64, |m, x| m.max(x.abs()));
        assert_eq!(ours.len(), frozen.len());
        let deviation = ours
            .iter()
            .zip(&frozen)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max)
            / scale;
        // The freeze recorded no floor for these cases; the fit's own 1-ulp sensitivity is
        // what a different least-squares solver can disagree by.
        let moved = values
            .iter()
            .enumerate()
            .map(|(i, &v)| match (i * 2654435761usize >> 7) & 1 {
                0 => f64::from_bits(v.to_bits() + 1),
                _ => f64::from_bits(v.to_bits() - 1),
            })
            .collect::<Vec<_>>();
        let again =
            local_derivatives(&r, &moved, at, get("order"), get("window"), get("degree")).unwrap();
        let floor = ours
            .iter()
            .zip(&again)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max)
            / scale;
        let gate = (100. * floor).max(GATE);
        worst = worst.max(deviation / gate);
        assert!(
            deviation <= gate,
            "{} {}: {deviation:.2e} > {gate:.1e}",
            case["function"],
            case["case"]
        );
        let center = (0..n)
            .min_by(|&a, &b| (r[a] - at).abs().total_cmp(&(r[b] - at).abs()))
            .unwrap();
        assert_eq!(
            center as u64,
            case["nearest_node"].as_u64().unwrap(),
            "{}",
            case["case"]
        );
    }
    println!(
        "{} analytic cases, worst deviation {worst:.3e} of its gate",
        cases.len()
    );
}

/// Worst deviation of a channel against a recorded case over every non-diagnostic output.
fn channel_miss(f: &Frozen, case: &Value, ours: &BTreeMap<String, Vec<f64>>) -> f64 {
    case["outputs"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(name, _)| group_of(name) != "diagnostics")
        .map(|(name, spec)| f.output(spec).deviation(&ours[name]))
        .fold(0., f64::max)
}

#[test]
fn a_channel_without_its_shell_or_with_a_moved_potential_misses_the_freeze() {
    let ctx = context();
    let f = Frozen::open("lda-sr_Fe");
    let (key, case) = f
        .cases("assemble_paw_channel")
        .into_iter()
        .find(|(_, c)| {
            c["inputs"]["r_local"].as_f64().unwrap() > c["inputs"]["pw"]["r_cut"].as_f64().unwrap()
        })
        .expect("a frozen channel whose local radius exceeds its cutoff");
    let mut run = call(&f, case);
    let honest = channel_miss(&f, case, &outputs(&run.run(&ctx)));
    assert!(honest <= 1e-10, "{key}: {honest:.2e}");
    // The shell between r_cut and r_local is part of B, Delta V and the projectors.
    run.r_local = run.smooth.r_cut;
    let without_shell = channel_miss(&f, case, &outputs(&run.try_run(&ctx).unwrap()));
    assert!(without_shell > 100. * GATE, "no shell: {without_shell:.2e}");
    // One part in 1e9 of the local potential is far above the gate.
    let mut run = call(&f, case);
    run.v_loc.iter_mut().for_each(|v| *v *= 1. + 1e-9);
    let moved = channel_miss(&f, case, &outputs(&run.run(&ctx)));
    assert!(moved > 100. * GATE, "moved potential: {moved:.2e}");
    println!(
        "{key}: honest {honest:.2e}, no shell {without_shell:.2e}, v_loc x (1 + 1e-9) {moved:.2e}"
    );
}

#[test]
fn the_gate_resolves_a_wrong_scheme_a_wrong_fit_and_a_moved_density() {
    let ctx = context();
    let f = Frozen::open("lda-sr_O");
    // Dropping the relativistic exchange factor moves E_xc far above the gate.
    let (_, case) = f.cases("_xc_energies").remove(0);
    let (r, rho) = (
        f.input(&case["inputs"]["r"]),
        f.input(&case["inputs"]["rho"]),
    );
    let frozen = f.output(&case["outputs"]["E_xc"]);
    let wrong = xc_energies(&ctx, &AtomSolver, &r, &rho, "lda", false).unwrap();
    let nonrelativistic = frozen.deviation(&[wrong.energy]);
    assert!(
        nonrelativistic > 100. * GATE,
        "non-relativistic E_xc: {nonrelativistic:.2e}"
    );
    // The older fit of `derivatives` (center at the first node at or above the point, degree
    // order + 4) misses the frozen quartic matching.
    let (_, case) = f.cases("_local_derivatives").remove(0);
    let inp = &case["inputs"];
    let (r, values) = (f.input(&inp["r"]), f.input(&inp["values"]));
    let at = inp["r0"].as_f64().unwrap();
    let frozen = f.output(&case["outputs"]["derivatives"]);
    let older = derivatives(&r, &values, at, 2, 25).unwrap();
    let fit = frozen.deviation(&older);
    assert!(fit > 100. * GATE, "older fit: {fit:.2e}");
    // A density moved by 1e-9 of itself changes the Hartree energy by 2e-9 of itself.
    let (_, case) = f.cases("_hartree_energy").remove(0);
    let (r, rho) = (
        f.input(&case["inputs"]["r"]),
        f.input(&case["inputs"]["rho"]),
    );
    let frozen = f.output(&case["outputs"]["energy"]);
    let moved = rho.iter().map(|x| x * (1. + 1e-9)).collect::<Vec<_>>();
    let miss = frozen.deviation(&[hartree_energy(&ctx, &r, &moved).unwrap()]);
    assert!(miss > 100. * GATE, "moved density: {miss:.2e}");
    // The variational scheme does not reproduce the frozen PBE potential.
    let pbe = Frozen::open("pbe-sr_O");
    let (_, case) = pbe.cases("_xc_energies").remove(1);
    let (r, rho) = (
        pbe.input(&case["inputs"]["r"]),
        pbe.input(&case["inputs"]["rho"]),
    );
    let ours = Variational.evaluate(&ctx, &r, &rho, "pbe", true).unwrap();
    let scheme = pbe
        .output(&case["outputs"]["v_xc"])
        .deviation(&ours.potential);
    assert!(scheme > 100. * GATE, "variational PBE: {scheme:.2e}");
    println!(
        "controls: non-relativistic E_xc {nonrelativistic:.2e}, older fit {fit:.2e}, density x (1 + 1e-9) {miss:.2e}, variational PBE v_xc {scheme:.2e}"
    );
}
