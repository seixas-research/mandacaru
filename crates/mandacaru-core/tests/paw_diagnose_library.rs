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
//!
//! The sweep is **resumable and non-fatal**. A file already in the report (status `ok` or
//! `refused`) is skipped on the next run, so a killed process loses at most the file it was on
//! (`MANDACARU_DIAGNOSE_REDO=1` diagnoses everything again); a file the diagnosis refuses (the
//! reference atom does not converge, the record is not the one the atom asks for) is written to
//! the report as `{"status": "refused", "cause": ...}` and the sweep goes on. The test ends by
//! failing on the problems found, and on a refusal that is not one of [`DOCUMENTED_REFUSALS`].
//!
//! **The oracle atom is the atom the shipped files were made on**: the 431 files were generated
//! with the atom-solver radial scheme (derivatives on subset nodes), so the sweep solves their
//! reference atoms through that legacy seam (`reference_atom_with(.., &AtomSolver)`); the
//! production scheme (variational, D-2b) would move the PBE atoms by the measured scheme
//! difference (`PAW_MIGRATION_LOG.md`, XC-SW) and the stored defects of a PBE file would then be
//! compared with a different atom. `MANDACARU_DIAGNOSE_SCHEME=variational` diagnoses the files
//! against the production atom instead; every report line names its scheme, and the resume key
//! includes it. The legacy atom of the heaviest PBE Dirac elements does not converge in the
//! generator's 200 iterations (the legacy potential is not the derivative of an energy), so those
//! files are refused by the legacy sweep and looked at by the variational one, labelled as such.

mod paw_assemble_support;
mod paw_diagnose_support;

use mandacaru_core::models::xc_scheme::{AtomSolver, Variational, XcSource};
use mandacaru_core::paw::diagnose::{
    ghosts::reference_offsets,
    kappa,
    reference::{check_reference, reference_atom_with, AeLogs, AllElectron},
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
    /// Set when the diagnosis refused the file: the cause, as reported.
    refusal: Option<String>,
}

/// The `(set, element)` pairs a report already holds with a final status, so a run can resume.
fn done(path: &str) -> std::collections::HashSet<(String, String)> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|v| matches!(v["status"].as_str(), Some("ok") | Some("refused")))
        // a line written before the scheme was named is the atom-solver one (the only one then)
        .filter(|v| v["scheme"].as_str().unwrap_or("atom-solver (legacy)") == scheme().0)
        .filter_map(|v| {
            Some((
                v["set"].as_str()?.to_string(),
                v["element"].as_str()?.to_string(),
            ))
        })
        .collect()
}

/// The refusals the sweep documents instead of failing on, with the cause that makes each one
/// not a property of the shipped file: `(set, element, cause prefix)`.
const DOCUMENTED_REFUSALS: &[(&str, &str, &str)] = &[];

/// A file the diagnosis cannot judge: the element, the stage and the cause.
fn refused(set: &str, element: &str, stage: &str, cause: impl std::fmt::Display) -> Outcome {
    let cause = format!("{cause}");
    Outcome {
        line: json!({"set": set, "element": element, "status": "refused", "scheme": scheme().0, "stage": stage, "cause": cause}),
        problems: Vec::new(),
        refusal: Some(format!("{set} {element}: {stage}: {cause}")),
    }
}

/// The scheme the sweep diagnoses against: the atom-solver one the files were made with, unless
/// `MANDACARU_DIAGNOSE_SCHEME=variational` asks for the production one.
fn scheme() -> (&'static str, &'static dyn XcSource) {
    match std::env::var("MANDACARU_DIAGNOSE_SCHEME").as_deref() {
        Ok("variational") => ("variational", &Variational),
        Ok("legacy") | Err(_) => ("atom-solver (legacy)", &AtomSolver),
        Ok(other) => panic!("MANDACARU_DIAGNOSE_SCHEME is legacy or variational, not {other:?}"),
    }
}

fn diagnose_file(set: &str, element: &str, path: &std::path::Path) -> Outcome {
    let ctx = context();
    let (scheme_name, scheme) = scheme();
    let ds = read(&ctx, path);
    let started = Instant::now();
    let atom = match reference_atom_with(&ctx, &ds, scheme) {
        Ok(atom) => atom,
        Err(e) => return refused(set, element, "reference atom", e),
    };
    if let Err(e) = check_reference(&ds, &atom) {
        return refused(set, element, "reference atom", e);
    }
    let solved = started.elapsed().as_secs_f64();
    let smooth = smooth_of(&ctx, &ds);
    let ae = AllElectron::new(
        &atom,
        ds.atomic_number as u8,
        &smooth.frozen,
        &smooth.semicore,
    );
    let started = Instant::now();
    let diagnosis = match Diagnosis::run(&ctx, &smooth, &ae, &mut AeLogs::new())
        .and_then(|d| d.with_branches(&ctx, &ds))
    {
        Ok(d) => d,
        Err(e) => return refused(set, element, "diagnosis", e),
    };
    let offsets = reference_offsets(&ctx, &smooth).unwrap();
    // The stored defects never had the branch ghosts or the missing states: the scalar
    // channels are compared as before (a branch ghost would also silence the other keys under
    // the recording rule), and the new keys are reported beside.
    let comparable = Diagnosis {
        branch_ghosts: Default::default(),
        missing: Default::default(),
        ..diagnosis.clone()
    };
    let got = comparable.defects();
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
        match kappa::level_errors(&ctx, &ds) {
            Ok(levels) => {
                for (l, so) in &ds.spin_orbit {
                    for (b, &stored) in so.level_errors.data().iter().enumerate() {
                        level_dev = nan_max(level_dev, (levels[&(*l as usize)][b] - stored).abs());
                    }
                }
            }
            Err(e) => return refused(set, element, "branch levels", e),
        }
        if !(level_dev <= LEVEL_GATE) {
            problems.push(format!(
                "{set} {element}: level errors differ by {level_dev:.3e} Ha"
            ));
        }
    }
    let line = json!({
        "set": set, "element": element, "status": "ok", "scheme": scheme_name,
        // The keys the file did not carry (a writer older than the current schema): a stored
        // `defects` of such a file predates the checks the record now has.
        "defaults_applied": ds.defaults_applied,
        "atom_seconds": solved, "diagnose_seconds": started.elapsed().as_secs_f64(),
        "stored": format!("{:?}", ds.defects), "found": format!("{got:?}"),
        "number_deviation": comparison.numbers(), "key_mismatches": comparison.keys,
        "level_deviation": level_dev,
        "s_miss": diagnosis.s_miss,
        "phases": diagnosis.phases.iter().map(|(k, v)| (format!("{}:{:?}", k.l, k.kappa), v.to_vec())).collect::<Vec<_>>(),
        "unprojected": diagnosis.unprojected.iter().map(|(l, v)| (*l, *v)).collect::<Vec<_>>(),
        "ghost_errors": diagnosis.ghosts.iter().map(|(l, v)| (*l, *v)).collect::<Vec<_>>(),
        "branch_ghosts": diagnosis.branch_ghosts.iter().map(|(k, v)| (format!("{}:{:?}", k.l, k.kappa), *v)).collect::<Vec<_>>(),
        "reference_offsets": offsets.iter().map(|(l, v)| (*l, *v)).collect::<Vec<_>>(),
        "missing": diagnosis.missing.iter().map(|(l, v)| (*l, *v)).collect::<Vec<_>>(),
        "context_peak_mb": ctx.memory_peak() as f64 / 1048576.,
        "clean": diagnosis.is_clean(),
        "problems": problems,
    });
    Outcome {
        line,
        problems,
        refusal: None,
    }
}

fn sweep(sets: &[&str]) {
    let path = std::env::var("MANDACARU_DIAGNOSE_REPORT").ok();
    let redo = std::env::var("MANDACARU_DIAGNOSE_REDO").is_ok_and(|v| v == "1");
    let finished = match (&path, redo) {
        (Some(p), false) => done(p),
        _ => Default::default(),
    };
    let mut report = path.as_ref().map(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .unwrap()
    });
    let mut problems = Vec::new();
    let mut refusals = Vec::new();
    let (mut count, mut skipped) = (0usize, 0usize);
    for set in sets {
        for (element, path) in files(set) {
            if finished.contains(&(set.to_string(), element.clone())) {
                skipped += 1;
                continue;
            }
            let outcome = diagnose_file(set, &element, &path);
            eprintln!("{}", outcome.line);
            if let Some(file) = report.as_mut() {
                writeln!(file, "{}", outcome.line).unwrap();
            }
            problems.extend(outcome.problems);
            refusals.extend(
                outcome
                    .refusal
                    .into_iter()
                    .map(|cause| (set.to_string(), element.clone(), cause)),
            );
            count += 1;
        }
    }
    eprintln!(
        "diagnosed {count} files ({skipped} already in the report), {} problems, {} refusals",
        problems.len(),
        refusals.len()
    );
    for (set, element, cause) in &refusals {
        let documented = DOCUMENTED_REFUSALS.iter().any(|(s, e, prefix)| {
            *s == set.as_str() && *e == element.as_str() && cause.contains(prefix)
        });
        if !documented {
            problems.push(format!("refused and not documented: {cause}"));
        }
    }
    assert!(
        count + skipped > 0,
        "no file was selected: check MANDACARU_DIAGNOSE_SET/_ELEMENTS"
    );
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
        assert!(outcome.refusal.is_none(), "{:?}", outcome.refusal);
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
