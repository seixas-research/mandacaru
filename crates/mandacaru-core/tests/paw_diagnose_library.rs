//! THE CHEAPEST STRONG GATE of the PAW completion plan: diagnosing a shipped dataset again,
//! from its own stride-4 tables and a freshly solved reference atom, gives the `defects` and the
//! level data the generator stored in the file.
//!
//! One file at a time (a record, its smooth copy and the atom are dropped before the next).
//! `paw_diagnose_sample` runs a stratified sample in the default suite; `paw_diagnose_library`
//! runs all 431 files (368 PAW-LCAO in four sets and 63 UPAW-LCAO) and is `#[ignore]`d: the
//! reference atoms alone take hours (a Pb atom is about 100 s). Run it per set:
//!
//! ```text
//! MANDACARU_DIAGNOSE_SET=lda-sr MANDACARU_DIAGNOSE_REPORT=/path/report.jsonl \
//!   cargo test --release -p mandacaru-core --test paw_diagnose_library -- --ignored --nocapture
//! ```
//!
//! (`MANDACARU_DIAGNOSE_SET` is one of `lda-sr pbe-sr lda-fr pbe-fr upaw`; unset runs all five;
//! `MANDACARU_DIAGNOSE_ELEMENTS=Si,O` restricts the elements.) Every file appends a JSON line
//! to the report. The equality is of keys (which channel is flagged, which defect) and of the
//! recorded numbers to the gates below.

mod paw_assemble_support;
mod paw_diagnose_support;

use mandacaru_core::paw::diagnose::{
    ghosts::reference_offsets,
    kappa,
    reference::{check_reference, reference_atom, AeLogs, AllElectron},
    Diagnosis,
};
use paw_diagnose_support::*;
use serde_json::json;
use std::{io::Write, path::PathBuf, time::Instant};

/// Gates on the recorded numbers between a file's stored defects and the ones diagnosed from its
/// stride-4 tables, by kind (measured over the 367 files diagnosed when the A5 log was written):
/// the unprojected channels (a local-potential phase, stride-insensitive: worst 5.3e-4 rad), the
/// constructed channels of a flagged dataset (a sharp f resonance, which follows the tables: worst
/// 0.101 rad, lda-fr Yb), the intruding-1s miss (worst 2.7e-5) and the ghost depths (Hartree;
/// no shipped file stores a ghost, so that gate has no data and is provisional).
const UNPROJECTED_GATE: f64 = 2e-3;
const PHASE_GATE: f64 = 0.15;
const GHOST_GATE: f64 = 1e-2;
const MISS_GATE: f64 = 1e-3;
/// Gate on the level errors of the `j` branches (Hartree) between the stored ones (computed from
/// the full tables) and the ones recomputed from the shipped stride-4 tables: the same as
/// `paw_diagnose_kappa::LEVEL_GATE`, whose comment gives the measurement and the reason.
const LEVEL_GATE: f64 = 2e-4;

fn files(set: &str) -> Vec<(String, PathBuf)> {
    let (root, folder) = if set == "upaw" {
        (upaw_library(), "lda-sr")
    } else {
        (library(), set)
    };
    let mut found = std::fs::read_dir(root.join(folder))
        .unwrap_or_else(|e| panic!("{}: {e}", root.join(folder).display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "parquet"))
        .map(|p| (p.file_stem().unwrap().to_string_lossy().into_owned(), p))
        .collect::<Vec<_>>();
    // light elements first: a heavy atom takes minutes, and the early lines calibrate the gates
    found.sort_by_key(|(element, _)| mandacaru_core::models::elements::atomic_number(element));
    if let Ok(list) = std::env::var("MANDACARU_DIAGNOSE_ELEMENTS") {
        let wanted = list.split(',').map(str::to_string).collect::<Vec<_>>();
        found.retain(|(e, _)| wanted.contains(e));
    }
    found
}

struct Outcome {
    line: serde_json::Value,
    problems: Vec<String>,
}

fn diagnose_file(set: &str, element: &str, path: &std::path::Path) -> Outcome {
    let ctx = context();
    let ds = read(&ctx, path);
    let started = Instant::now();
    let atom = reference_atom(&ctx, &ds).unwrap_or_else(|e| panic!("{set} {element}: {e}"));
    check_reference(&ds, &atom).unwrap();
    let solved = started.elapsed().as_secs_f64();
    let smooth = smooth_of(&ctx, &ds);
    let ae = AllElectron::new(
        &atom,
        ds.atomic_number as u8,
        &smooth.frozen,
        &smooth.semicore,
    );
    let started = Instant::now();
    let diagnosis = Diagnosis::run(&ctx, &smooth, &ae, &mut AeLogs::new())
        .unwrap_or_else(|e| panic!("{set} {element}: {e}"));
    let offsets = reference_offsets(&ctx, &smooth).unwrap();
    let got = diagnosis.defects();
    let comparison = compare_defects(&ds.defects, &got);
    let mut problems = comparison
        .keys
        .iter()
        .map(|k| format!("{set} {element}: {k}"))
        .collect::<Vec<_>>();
    for (name, deviation, gate) in [
        ("unprojected", comparison.unprojected, UNPROJECTED_GATE),
        ("phases", comparison.phases, PHASE_GATE),
        ("ghosts", comparison.ghosts, GHOST_GATE),
        ("s_miss", comparison.s_miss, MISS_GATE),
    ] {
        if !(deviation <= gate) {
            problems.push(format!("{set} {element}: {name} differ by {deviation:.3e}"));
        }
    }
    let mut level_dev = 0f64;
    if !ds.spin_orbit.is_empty() {
        let levels = kappa::level_errors(&ctx, &ds).unwrap();
        for (l, so) in &ds.spin_orbit {
            for (b, &stored) in so.level_errors.data().iter().enumerate() {
                level_dev = nan_max(level_dev, (levels[&(*l as usize)][b] - stored).abs());
            }
        }
        if !(level_dev <= LEVEL_GATE) {
            problems.push(format!(
                "{set} {element}: level errors differ by {level_dev:.3e} Ha"
            ));
        }
    }
    let line = json!({
        "set": set, "element": element,
        "atom_seconds": solved, "diagnose_seconds": started.elapsed().as_secs_f64(),
        "stored": format!("{:?}", ds.defects), "found": format!("{got:?}"),
        "number_deviation": comparison.numbers(), "key_mismatches": comparison.keys,
        "level_deviation": level_dev,
        "s_miss": diagnosis.s_miss,
        "phases": diagnosis.phases.iter().map(|(k, v)| (format!("{}:{:?}", k.l, k.kappa), v.to_vec())).collect::<Vec<_>>(),
        "unprojected": diagnosis.unprojected.iter().map(|(l, v)| (*l, *v)).collect::<Vec<_>>(),
        "ghost_errors": diagnosis.ghosts.iter().map(|(l, v)| (*l, *v)).collect::<Vec<_>>(),
        "reference_offsets": offsets.iter().map(|(l, v)| (*l, *v)).collect::<Vec<_>>(),
        "missing": diagnosis.missing.iter().map(|(l, v)| (*l, *v)).collect::<Vec<_>>(),
        "context_peak_mb": ctx.memory_peak() as f64 / 1048576.,
        "clean": diagnosis.is_clean(),
        "problems": problems,
    });
    Outcome { line, problems }
}

fn sweep(sets: &[&str]) {
    let report = std::env::var("MANDACARU_DIAGNOSE_REPORT").ok().map(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .unwrap()
    });
    let mut report = report;
    let mut problems = Vec::new();
    let mut count = 0usize;
    for set in sets {
        for (element, path) in files(set) {
            let outcome = diagnose_file(set, &element, &path);
            eprintln!("{}", outcome.line);
            if let Some(file) = report.as_mut() {
                writeln!(file, "{}", outcome.line).unwrap();
            }
            problems.extend(outcome.problems);
            count += 1;
        }
    }
    eprintln!("diagnosed {count} files, {} problems", problems.len());
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn a_stratified_sample_of_the_library_diagnoses_to_its_stored_defects() {
    // light elements in the four sets that a default run can afford (the heavy atoms and the
    // flagged lanthanides are in the ignored sweep)
    for (set, element) in [
        ("lda-sr", "H"),
        ("lda-sr", "O"),
        ("lda-sr", "Si"),
        ("lda-sr", "Na"),
        ("lda-fr", "Na"),
        ("pbe-sr", "O"),
    ] {
        let outcome = diagnose_file(set, element, &dataset_path(set, element));
        eprintln!("{}", outcome.line);
        assert!(
            outcome.problems.is_empty(),
            "{}",
            outcome.problems.join("\n")
        );
    }
}

#[test]
#[ignore = "431 reference atoms: hours; see the module doc for the per-set run command"]
fn paw_diagnose_library() {
    match std::env::var("MANDACARU_DIAGNOSE_SET") {
        Ok(set) => sweep(&[set.as_str()]),
        Err(_) => sweep(&["lda-sr", "pbe-sr", "lda-fr", "pbe-fr", "upaw"]),
    }
}
