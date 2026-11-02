//! The assemble/core/energies layer of the PAW generator (`paw::generate::{assemble, core,
//! energies}`, A2) against the A0 freeze of the Python functions it replaces
//! (`test/migration/reference/paw_generate`): `assemble_paw_channel`, `pseudize_density`,
//! `_local_derivatives`, `_hartree_energy` and `_xc_energies` on the recorded inputs
//! (`paw_assemble_controls.rs` holds the negative controls).
//!
//! Gate: every output within `max(1e-12, 100 x floor)` of the frozen number, relative to its
//! maximum (`floor` = the 1-ulp input sensitivity the freeze measured per case and group), the
//! round-off diagnostics by order of magnitude. The A0 freeze is Python and uses the atom-solver
//! radial exchange-correlation scheme, so every functional is gated through that scheme
//! (`AtomSolver`, the legacy seam). Its PBE tables are gated at `max(1e-12, ...)` wherever the
//! density is at least `TAIL_DENSITY`; below that the freeze and the Rust scheme part ways in
//! three named regions whose measured absolute bounds are written down in `PBE_TAIL`. The
//! variational scheme, the densities and the one-center constant are in `paw_one_center.rs`.
mod paw_assemble_support;

use mandacaru_core::math::radial::local_derivatives;
use mandacaru_core::paw::generate::core::pseudize_density;
use mandacaru_core::paw::generate::energies::{hartree_energy, xc_energies, AtomSolver};
use mandacaru_core::Context;
use paw_assemble_support::call::{call, outputs};
use paw_assemble_support::{datasets, directory, group_of, Frozen, Reference};
use serde_json::Value;
use std::collections::BTreeMap;

const GATE: f64 = 1e-12;
/// A density below which the PBE exchange-correlation tables of the freeze and of the Rust atom
/// scheme differ at the level of the exchange floor, not of round-off.
const TAIL_DENSITY: f64 = 1e-9;
/// `(lowest density, highest density, absolute bound in Hartree)` of the three regions of
/// the PBE tail, measured on pbe-sr {O, Cu, Ce} and pbe-fr Si (`PAW_MIGRATION_LOG.md`, A2): the
/// potential just above the Rust exchange floor (3.9e-9, Ce), in the band between the Python
/// threshold on the total density and the Rust one on the spin density (9.96e-5, the
/// uniform-gas exchange against the full PBE one) and below it (1.6e-9).
const PBE_TAIL: [(f64, f64, f64); 3] = [
    (2e-12, TAIL_DENSITY, 1e-8),
    (1e-12, 2e-12, 1.1e-4),
    (0., 1e-12, 1e-8),
];

/// Relative gate of the PBE integrals `E_xc` and `int rho v_xc`: the tail regions above change
/// them by up to 1.4e-12 of their value (pbe-sr Ce).
const PBE_INTEGRAL: f64 = 1e-11;

fn context() -> Context {
    Context::new(Some(1), 512 << 20).unwrap()
}

/// Worst deviation per `(function, group)` of the run, with its tolerance and where it was.
#[derive(Default)]
struct Ledger {
    rows: BTreeMap<(String, String), (f64, f64, String)>,
    failures: Vec<String>,
}

impl Ledger {
    fn record(&mut self, function: &str, label: &str, name: &str, deviation: f64, tolerance: f64) {
        let group = group_of(name).to_string();
        let row =
            self.rows
                .entry((function.into(), group))
                .or_insert((0., tolerance, String::new()));
        if deviation >= row.0 {
            *row = (deviation, tolerance, format!("{label} {name}"));
        }
        if !(deviation <= tolerance) {
            self.failures.push(format!(
                "{label} {function} {name}: {deviation:.3e} > {tolerance:.1e}"
            ));
        }
    }
    fn finish(self, title: &str) {
        println!("{title}");
        for ((function, group), (deviation, tolerance, at)) in &self.rows {
            println!(
                "  {function:<28} {group:<12} max {deviation:.3e}  (gate {tolerance:.1e})  at {at}"
            );
        }
        assert!(self.failures.is_empty(), "{}", self.failures.join("\n"));
    }
}

/// `max(1e-12, 100 x the 1-ulp floor of the group)`, never above the old looser value.
fn tolerance(case: &Value, group: &str) -> f64 {
    let cap = if group == "scalars" { 1e-8 } else { 1e-9 };
    let floor = case["floor"]["ulp_perturbation"]["groups"][group]["max_rel_to_max"]
        .as_f64()
        .unwrap_or(0.);
    (100. * floor).max(GATE).min(cap)
}

/// Compare one output with its frozen counterpart and record it.
fn check(
    ledger: &mut Ledger,
    function: &str,
    label: &str,
    case: &Value,
    frozen: &Reference,
    name: &str,
    ours: &[f64],
) {
    let group = group_of(name);
    // `norm_error` is a physical number for a norm-deficit dataset (0.11 for oxygen) and
    // round-off for a unitary one; the other diagnostics are always round-off. Those are
    // compared by order of magnitude: the ratio to ten times the frozen value plus 1e-12.
    let physical = name == "norm_error" && frozen.values[0].abs() > 1e-6;
    if group == "diagnostics" && !physical {
        let bound = 10. * frozen.values[0].abs() + 1e-12;
        ledger.record(
            &format!("{function}(diag)"),
            label,
            name,
            ours[0].abs() / bound,
            1.,
        );
        return;
    }
    let group = if physical { "scalars" } else { group };
    ledger.record(
        function,
        label,
        name,
        frozen.deviation(ours),
        tolerance(case, group),
    );
}

fn gate_channels(ctx: &Context, f: &Frozen, ledger: &mut Ledger) {
    for (key, case) in f.cases("assemble_paw_channel") {
        let ours = outputs(&call(f, case).run(ctx));
        let label = format!("{} {key}", f.name);
        for (name, spec) in case["outputs"].as_object().unwrap() {
            let frozen = f.output(spec);
            check(
                ledger,
                "assemble_paw_channel",
                &label,
                case,
                &frozen,
                name,
                &ours[name],
            );
        }
    }
}

fn gate_core(ctx: &Context, f: &Frozen, ledger: &mut Ledger) {
    for (key, case) in f.cases("_local_derivatives") {
        let inp = &case["inputs"];
        let (r, values) = (f.input(&inp["r"]), f.input(&inp["values"]));
        let at = inp["r0"].as_f64().unwrap();
        let get = |k: &str| inp[k].as_u64().unwrap() as usize;
        let ours =
            local_derivatives(&r, &values, at, get("order"), get("window"), get("degree")).unwrap();
        let center = (0..r.len())
            .min_by(|&a, &b| (r[a] - at).abs().total_cmp(&(r[b] - at).abs()))
            .unwrap();
        assert_eq!(center as u64, case["nearest_node"].as_u64().unwrap());
        let frozen = f.output(&case["outputs"]["derivatives"]);
        let label = format!("{} {key}", f.name);
        check(
            ledger,
            "_local_derivatives",
            &label,
            case,
            &frozen,
            "derivatives",
            &ours,
        );
    }
    for (key, case) in f.cases("pseudize_density") {
        let inp = &case["inputs"];
        let (r, density) = (f.input(&inp["r"]), f.input(&inp["density"]));
        let ours = pseudize_density(&r, &density, inp["r_cut"].as_f64().unwrap()).unwrap();
        let frozen = f.output(&case["outputs"]["smooth"]);
        let label = format!("{} {key}", f.name);
        check(
            ledger,
            "pseudize_density",
            &label,
            case,
            &frozen,
            "smooth",
            &ours,
        );
    }
    for (key, case) in f.cases("_hartree_energy") {
        let inp = &case["inputs"];
        let ours = hartree_energy(ctx, &f.input(&inp["r"]), &f.input(&inp["rho"])).unwrap();
        let frozen = f.output(&case["outputs"]["energy"]);
        let label = format!("{} {key}", f.name);
        check(
            ledger,
            "hartree_energy",
            &label,
            case,
            &frozen,
            "energy",
            &[ours],
        );
    }
}

/// An exchange-correlation table of a case against the freeze, region by region.
struct Table<'a> {
    label: &'a str,
    case: &'a Value,
    pbe: bool,
    rho: &'a [f64],
}

fn gate_table(ledger: &mut Ledger, at: &Table, name: &str, frozen: &Reference, ours: &[f64]) {
    let mut interior = 0f64;
    let mut tail = [0f64; 3];
    for (k, b) in frozen.values.iter().enumerate() {
        let i = (k + 1) * frozen.stride - 1;
        let d = (ours[i] - b).abs();
        let region = PBE_TAIL
            .iter()
            .position(|&(lo, hi, _)| at.rho[i] >= lo && at.rho[i] < hi);
        match region {
            Some(region) if at.pbe => tail[region] = tail[region].max(d),
            _ => interior = interior.max(d),
        }
    }
    let scale = frozen.scale.max(1e-300);
    let gate = tolerance(at.case, "tables");
    ledger.record("xc_energies(atom)", at.label, name, interior / scale, gate);
    if at.pbe {
        for (region, &(lo, _, bound)) in PBE_TAIL.iter().enumerate() {
            let function = format!("xc(atom) PBE tail >= {lo:.0e} abs");
            ledger.record(&function, at.label, name, tail[region], bound);
        }
    }
}

fn gate_xc(ctx: &Context, f: &Frozen, ledger: &mut Ledger) {
    for (key, case) in f.cases("_xc_energies") {
        let inp = &case["inputs"];
        let (r, rho) = (f.input(&inp["r"]), f.input(&inp["rho"]));
        let xc = inp["xc"].as_str().unwrap();
        let relativistic = inp["relativistic"].as_bool().unwrap();
        let ours = xc_energies(ctx, &AtomSolver, &r, &rho, xc, relativistic).unwrap();
        let label = format!("{} {key}", f.name);
        for (name, value) in [("E_xc", ours.energy), ("int_rho_v_xc", ours.rho_potential)] {
            let frozen = f.output(&case["outputs"][name]);
            if xc == "pbe" {
                // The tail band moves the integrals by up to 5e-12 Ha (measured, Ce): 1e-11.
                let deviation = frozen.deviation(&[value]);
                ledger.record(
                    "xc_energies(atom) PBE",
                    &label,
                    name,
                    deviation,
                    PBE_INTEGRAL,
                );
            } else {
                check(
                    ledger,
                    "xc_energies(atom)",
                    &label,
                    case,
                    &frozen,
                    name,
                    &[value],
                );
            }
        }
        let at = Table {
            label: &label,
            case,
            pbe: xc == "pbe",
            rho: &rho,
        };
        for (name, values) in [
            ("e_xc_density", &ours.field.per_particle),
            ("v_xc", &ours.field.potential),
        ] {
            gate_table(ledger, &at, name, &f.output(&case["outputs"][name]), values);
        }
    }
}

fn points(f: &Frozen) -> usize {
    f.doc["scalars"]["n_points"].as_u64().unwrap() as usize
}

fn gate_datasets(selected: impl Fn(usize) -> bool) {
    let ctx = context();
    let mut ledger = Ledger::default();
    let mut compared = 0;
    for name in datasets() {
        let f = Frozen::open(&name);
        if !selected(points(&f)) {
            continue;
        }
        compared += 1;
        gate_channels(&ctx, &f, &mut ledger);
        gate_core(&ctx, &f, &mut ledger);
        gate_xc(&ctx, &f, &mut ledger);
        assert_eq!(ctx.memory_used(), 0, "{name} leaks calculation memory");
    }
    assert!(
        compared > 0,
        "no frozen dataset in {}",
        directory().display()
    );
    ledger.finish(&format!("{compared} frozen datasets:"));
}

#[test]
fn light_frozen_datasets_reproduce_every_function_case() {
    gate_datasets(|n| n <= 45_000);
}

#[test]
fn heavy_frozen_datasets_reproduce_every_function_case() {
    gate_datasets(|n| n > 45_000);
}

#[test]
fn a_heavy_channel_peaks_at_its_three_tables_and_gives_everything_back() {
    let f = Frozen::open("lda-sr_Pb");
    let (key, case) = f
        .cases("assemble_paw_channel")
        .into_iter()
        .max_by_key(|(_, c)| c["inputs"]["pw"]["energies"].as_array().unwrap().len())
        .unwrap();
    let run = call(&f, case);
    let (n, count) = (run.r.len(), run.energies.len());
    let ctx = context();
    let channel = run.run(&ctx);
    // pseudo waves, projectors and raw projectors stay; the spline of a resample (four
    // arrays of n) and the 801-node work arrays are the transient.
    let bound = ((3 * count + 4) * n + (10 * count + 8) * 801) * 8;
    let peak = ctx.memory_peak();
    println!("{key}: n = {n}, {count} references, peak {peak} bytes, bound {bound}");
    assert!(peak <= bound, "{peak} > {bound}");
    assert_eq!(ctx.memory_used(), 3 * count * n * 8);
    drop(channel);
    assert_eq!(ctx.memory_used(), 0);
}
