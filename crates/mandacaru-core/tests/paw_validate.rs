//! `paw::validate`: every per-file gate (G1-G11 and G13) on a known-good record and on a record
//! damaged to fail exactly that gate: a ghost (G2), a repulsive bump that unbinds a state and
//! changes the phases (G3, G4), a missing core state, a core-density mismatch and a core record
//! that is not format 3 (G5), a conditioning figure (G6), a hard channel (G7), a sphere past a
//! bond and a twin with another partition (G8), a wrong smooth-core charge (G9), a grid that is not
//! `r_i = (i + 1) h` (G11) and a spin-orbit coupling that moves the splitting (G13).
mod paw_validate_support;

use mandacaru_core::paw::validate::{validate, GateReport, Options, Status, GATES};
use mandacaru_core::{paw::record::PawDataset, Context};
use paw_validate_support::{
    build_report, context, damaged, ghosted, good, row, status, table, try_damaged, with_well,
};
use serde_json::{json, Value};

fn run(ctx: &Context, dataset: &PawDataset) -> GateReport {
    validate(ctx, dataset, &Options::default()).unwrap()
}

#[test]
fn a_known_good_file_passes_every_gate_and_names_what_it_cannot_measure() {
    let ctx = context();
    let report = run(&ctx, &good(&ctx));
    assert!(report.passed(), "{}", report.lines().join("\n"));
    for (gate, _) in GATES {
        assert!(!report.gate(gate).is_empty(), "{gate} has no row");
    }
    // What a lone scalar-relativistic file cannot say is said, not skipped.
    assert_eq!(status(&report, "G1", "reload"), Status::NotApplicable);
    assert_eq!(status(&report, "G8", "twin"), Status::NotApplicable);
    assert_eq!(
        status(&report, "G10", "transferability"),
        Status::NotApplicable
    );
    assert_eq!(status(&report, "G13", "spin-orbit"), Status::NotApplicable);
    for gate in ["G2", "G3", "G4", "G6", "G11"] {
        assert_eq!(report.gate_status(gate), Status::Pass, "{gate}");
    }
    // G5 reports the core charge beyond the sphere, G7 a working spacing coarser than 0.15 A and G9
    // the smooth-core electrons: measurements, not refusals.
    for gate in ["G5", "G7", "G9"] {
        assert!(!report.gate_status(gate).refuses(), "{gate}");
    }
    // The core record measured as the numpy check did: Li has one stored 1s state.
    let density = row(&report, "G5", "core density");
    assert!(density.value.unwrap() < density.threshold.unwrap());
    assert!(
        row(&report, "G5", "beyond the widest sphere")
            .value
            .unwrap()
            >= 0.
    );
}

#[test]
fn the_generator_reports_the_same_gates_for_the_file_it_wrote() {
    let report = build_report();
    assert!(report.passed(), "{}", report.lines().join("\n"));
    // G1 is measured at the build, where the generated record is at hand.
    let reload = row(report, "G1", "reload");
    assert_eq!(reload.status, Status::Pass);
    assert!(reload.value.unwrap() <= reload.threshold.unwrap());
    assert_eq!(report.symbol, "Li");
}

#[test]
fn g1_a_reload_shift_past_the_tolerance_fails() {
    let ctx = context();
    let ds = good(&ctx);
    let with = |shift| {
        let report = validate(
            &ctx,
            &ds,
            &Options {
                reload_shift: Some(shift),
                ..Options::default()
            },
        )
        .unwrap();
        status(&report, "G1", "reload")
    };
    assert_eq!(with(2e-6), Status::Pass);
    assert_eq!(with(2e-5), Status::Fail);
    assert_eq!(with(f64::NAN), Status::Fail);
}

#[test]
fn g2_a_deep_local_well_binds_a_ghost() {
    let ctx = context();
    let (ghosted, depth) = ghosted(&ctx, &good(&ctx));
    let found = run(&ctx, &ghosted);
    assert!(!found.passed(), "a {depth} Ha well");
    let ghost = row(&found, "G2", "channels with projectors");
    assert_eq!(ghost.status, Status::Fail);
    assert!(ghost.value.unwrap() > ghost.threshold.unwrap());
    assert!(ghost.detail.contains("l=0"), "{}", ghost.detail);
}

/// The first bump of 10, 30, 100, ... Hartree on the local potential for which `fails` holds.
fn first_bump(
    ctx: &Context,
    ds: &PawDataset,
    fails: impl Fn(&GateReport) -> bool,
) -> (GateReport, f64) {
    let mut last = String::new();
    for height in [10., 30., 100., 300., 1000.] {
        let report = run(ctx, &with_well(ctx, ds, -height));
        if fails(&report) {
            return (report, height);
        }
        last = report.lines().join("\n");
    }
    panic!("no bump up to 1000 Ha fails the gate; the last report:\n{last}");
}

#[test]
fn g3_a_repulsive_bump_changes_the_phases() {
    let ctx = context();
    let ds = good(&ctx);
    let (report, height) = first_bump(&ctx, &ds, |r| status(r, "G3", "near") == Status::Fail);
    assert!(!report.passed());
    let near = row(&report, "G3", "near");
    assert!(
        near.value.unwrap() > near.threshold.unwrap(),
        "a {height} Ha bump: {:?}",
        near.value
    );
}

#[test]
fn g4_a_repulsive_bump_unbinds_a_state() {
    let ctx = context();
    let ds = good(&ctx);
    let (report, height) = first_bump(&ctx, &ds, |r| status(r, "G4", "missing") == Status::Fail);
    assert!(!report.passed());
    let missing = row(&report, "G4", "missing");
    assert!(
        missing.detail.contains("l="),
        "a {height} Ha bump: {}",
        missing.detail
    );
}

#[test]
fn g5_a_core_that_does_not_add_up_is_refused_by_the_reader_before_any_gate_runs() {
    // The reader refuses a version-3 core record that lists no state although its core density holds
    // charge, or whose occupation is out of range, so the gate's charge and density rows cannot be
    // failed by such a record: they are the audit of the files already on disk.
    let ctx = context();
    let ds = good(&ctx);
    let empty = try_damaged(&ctx, &ds, |payload, tables| {
        payload["core_states"] = json!([]);
        tables.retain(|(name, _)| !name.starts_with("core_wave_"));
    });
    assert!(empty.unwrap_err().to_string().contains("core density"));
    let full = try_damaged(&ctx, &ds, |payload, _| {
        let occupation = payload["core_states"][0]["occupation"].as_f64().unwrap();
        payload["core_states"][0]["occupation"] = json!(occupation + 0.5);
    });
    assert!(full.unwrap_err().to_string().contains("occupation"));
}

#[test]
fn g5_the_format_and_the_reader_defaults_are_part_of_the_gate() {
    let ctx = context();
    let mut old = good(&ctx);
    old.core_states = None;
    let report = run(&ctx, &old);
    assert_eq!(status(&report, "G5", "format"), Status::Fail);
    assert_eq!(status(&report, "G5", "core charge"), Status::NotApplicable);
    // A reader default (a key the file did not carry) refuses a file too.
    old.defaults_applied.push("semicore_subshells".into());
    let report = run(&ctx, &old);
    assert_eq!(status(&report, "G5", "no reader default"), Status::Fail);
}

#[test]
fn g5_a_core_density_that_the_orbitals_do_not_reproduce_fails() {
    let ctx = context();
    let ds = good(&ctx);
    let broken = damaged(&ctx, &ds, |_, tables| {
        let core = table(tables, "core_density");
        let peak = core.iter().fold(0f64, |m, x| m.max(x.abs()));
        core[7] += 1e-6 * peak;
    });
    let report = run(&ctx, &broken);
    let density = row(&report, "G5", "core density");
    assert_eq!(density.status, Status::Fail);
    assert!(
        (density.value.unwrap() - 1e-6).abs() < 1e-8,
        "{:?}",
        density.value
    );
    // Below the threshold it is the ten-digit storage and passes.
    let rounded = damaged(&ctx, &ds, |_, tables| {
        let core = table(tables, "core_density");
        let peak = core.iter().fold(0f64, |m, x| m.max(x.abs()));
        core[7] += 1e-9 * peak;
    });
    assert_eq!(
        status(&run(&ctx, &rounded), "G5", "core density"),
        Status::Pass
    );
}

#[test]
fn g6_a_conditioning_figure_past_its_threshold_fails() {
    let ctx = context();
    let ds = good(&ctx);
    for (key, value, check) in [
        ("duality_error", json!(1e-6), "duality"),
        ("asymmetry", json!(1e-2), "asymmetry"),
        ("overlap_minimum", json!(0.01), "overlap-operator"),
    ] {
        let broken = damaged(&ctx, &ds, |payload, _| {
            payload["channels"]["0"][key] = value.clone();
        });
        assert_eq!(
            status(&run(&ctx, &broken), "G6", check),
            Status::Fail,
            "{key}"
        );
    }
}

#[test]
fn g7_a_hard_channel_fails_and_a_coarse_spacing_is_only_reported() {
    let ctx = context();
    let ds = good(&ctx);
    let broken = damaged(&ctx, &ds, |payload, _| {
        payload["channels"]["0"]["residual_kinetic"] = json!([5.0, 5.0]);
    });
    let report = run(&ctx, &broken);
    let hard = row(&report, "G7", "hardness");
    assert_eq!(hard.status, Status::Fail);
    assert!(hard.detail.contains("hard channels"), "{}", hard.detail);
    // A spacing coarser than 0.15 Angstrom is recorded, never refused: the row can be Reported.
    let spacing = row(&report, "G7", "working spacing");
    assert!(matches!(spacing.status, Status::Pass | Status::Reported));
}

#[test]
fn g8_a_sphere_past_the_bond_is_reported_and_a_twin_with_another_partition_fails() {
    let ctx = context();
    let ds = good(&ctx);
    let report = validate(
        &ctx,
        &ds,
        &Options {
            bond: Some(1.0),
            ..Options::default()
        },
    )
    .unwrap();
    let sphere = row(&report, "G8", "sphere");
    assert_eq!(
        sphere.status,
        Status::Reported,
        "advisory, decided under the plan"
    );
    assert!(sphere.value.unwrap() >= 1.0);
    assert!(report.passed());
    // A twin of the same partition passes; one that moves a subshell into the valence fails.
    let same = good(&ctx);
    let ok = validate(
        &ctx,
        &ds,
        &Options {
            twin: Some(&same),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(status(&ok, "G8", "twin"), Status::Pass);
    let other = damaged(&ctx, &ds, |payload, _| {
        payload["semicore_subshells"] = json!([[1, 0]]);
    });
    let bad = validate(
        &ctx,
        &ds,
        &Options {
            twin: Some(&other),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(status(&bad, "G8", "twin"), Status::Fail);
    assert!(!bad.passed());
}

#[test]
fn g9_a_wrong_smooth_core_charge_fails() {
    let ctx = context();
    let ds = good(&ctx);
    let broken = damaged(&ctx, &ds, |_, tables| {
        for value in table(tables, "smooth_core_density") {
            *value *= 1.1;
        }
    });
    let report = run(&ctx, &broken);
    let charge = row(&report, "G9", "against partial_core_electrons");
    assert_eq!(charge.status, Status::Fail);
    // 10 percent of the partial core's 0.37 electrons.
    assert!(
        charge.value.unwrap() > 0.03 && charge.value.unwrap() < 0.045,
        "{:?}",
        charge.value
    );
    // A correction that is switched off holds no smooth core at all.
    let off = damaged(&ctx, &ds, |payload, _| {
        payload["nlcc"]["applied"] = Value::Bool(false);
    });
    assert_eq!(
        status(&run(&ctx, &off), "G9", "against partial_core_electrons"),
        Status::Fail
    );
}

#[test]
fn g11_a_grid_that_is_not_uniform_fails() {
    let ctx = context();
    let ds = good(&ctx);
    let good_grid = row(&run(&ctx, &ds), "G11", "radial grid").clone();
    // The ten-digit storage of the nodes at the library stride, 4e-7 of h at most.
    assert!(good_grid.value.unwrap() < 1e-5, "{:?}", good_grid.value);
    let h = ds.r.as_slice()[1] - ds.r.as_slice()[0];
    let jittered = damaged(&ctx, &ds, |_, tables| {
        table(tables, "r")[40] += 0.01 * h;
    });
    let report = run(&ctx, &jittered);
    let bad = row(&report, "G11", "radial grid");
    assert_eq!(bad.status, Status::Fail);
    assert!(bad.value.unwrap() > 1e-3, "{:?}", bad.value);
    // A grid that starts at a node other than h is not r_i = (i + 1) h either.
    let shifted = damaged(&ctx, &ds, |_, tables| {
        for value in table(tables, "r") {
            *value += 0.5 * h;
        }
    });
    assert_eq!(
        status(&run(&ctx, &shifted), "G11", "radial grid"),
        Status::Fail
    );
}

fn scaled(value: &mut Value, factor: f64) {
    match value {
        Value::Number(n) => *value = json!(n.as_f64().unwrap() * factor),
        Value::Array(items) => items.iter_mut().for_each(|v| scaled(v, factor)),
        _ => {}
    }
}

#[test]
#[ignore = "needs a Dirac library file (slow, about a minute for bismuth): MANDACARU_PAW_PATH=<mandacaru-paw> cargo test --profile release-test -p mandacaru-core --test paw_validate -- --ignored g13"]
fn g13_the_spin_orbit_splitting_is_the_engine_operators_and_a_wrong_coupling_fails() {
    use mandacaru_core::paw::engine_spin_orbit::splittings;
    let ctx = context();
    let root = std::env::var("MANDACARU_PAW_PATH")
        .expect("MANDACARU_PAW_PATH names the library with a pbe-fr set");
    let path = std::path::Path::new(&root)
        .join("pbe-fr")
        .join("Bi.parquet");
    let ds = PawDataset::read(&ctx, &path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let report = run(&ctx, &ds);
    // The gate is the engine function's verdict: one row per measured channel, nothing re-derived.
    let found = splittings(&ctx, &ds).unwrap();
    let rows = report.gate("G13");
    assert_eq!(rows.len(), found.len());
    for (row, s) in rows.iter().zip(&found) {
        assert_eq!(row.status == Status::Pass, s.passes(), "{}", row.line());
        assert!(row.detail.contains("eV"), "{}", row.detail);
    }
    assert!(
        !rows.is_empty() && rows.iter().all(|r| r.status == Status::Pass),
        "{}",
        report.lines().join("\n")
    );
    // A spin-orbit coupling 20 percent short moves the splitting out of 2 percent or 10 meV.
    let weak = damaged(&ctx, &ds, |payload, _| {
        scaled(&mut payload["spin_orbit"]["1"]["coupling"], 0.8);
    });
    let weak = run(&ctx, &weak);
    assert_eq!(
        weak.gate_status("G13"),
        Status::Fail,
        "{}",
        weak.lines().join("\n")
    );
    assert!(!weak.passed());
}
