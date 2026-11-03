//! The search against the census of the shipped libraries (`census_<set>.json` of the A0 freeze):
//! for every element, which construction the shipped file holds, inferred from its parameters.
//! Where the inference names the attempt exactly (the first construction, zero norm deficit, a
//! shorter local radius, a raise at the own cutoffs, the balanced cutoffs), the search of the
//! same recipe selects that label; the Bessel-count attempts share a census label with the
//! shorter s spheres and are reported, not gated. The sets are generated with the atom-solver
//! radial scheme, so the search runs through that seam.
//!
//! `search_census_light_lda` and `_pbe` run H to Ar of lda-sr and pbe-sr in the default suite
//! (about 2 minutes each on a loaded machine).
//! `search_census_sweep` is ignored (the heavy atoms take minutes each) and runs per set:
//!
//! ```text
//! MANDACARU_SEARCH_SET=lda-sr MANDACARU_SEARCH_MAX_Z=54 MANDACARU_SEARCH_REPORT=/path/report.jsonl \
//!   cargo test --release -p mandacaru-core --test paw_search_library -- --ignored --nocapture
//! ```
//!
//! (`MANDACARU_SEARCH_SET` is one of `lda-sr pbe-sr lda-fr pbe-fr`; `MANDACARU_SEARCH_ELEMENTS=Si,O`
//! restricts the elements.) Every element appends a JSON line to the report.
mod paw_assemble_support;
mod paw_generate_support;
mod paw_search_support;

use mandacaru_core::models::xc_scheme::AtomSolver;
use mandacaru_core::models::{elements::atomic_number, relativity::Treatment};
use mandacaru_core::paw::generate::{
    recipe::Recipe,
    search::{search_with, Mode, Options},
};
use paw_search_support::context;
use serde_json::{json, Value};
use std::{collections::BTreeMap, io::Write};

fn census(set: &str) -> BTreeMap<u8, (String, String)> {
    let path = paw_assemble_support::directory().join(format!("census_{set}.json"));
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (tracked file: `git checkout` it)",
            path.display()
        )
    }))
    .unwrap();
    doc["files"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(symbol, row)| {
            let z = atomic_number(symbol).unwrap_or_else(|| panic!("{symbol}"));
            (
                z,
                (symbol.clone(), row["attempt"].as_str().unwrap().to_string()),
            )
        })
        .collect()
}

/// The census label when it names one attempt exactly.
fn exact(label: &str) -> Option<&str> {
    let exact = label == "first construction"
        || label == "norm_deficit=0"
        || label.starts_with("local radius x ")
        || label.starts_with("own cutoffs, shift ")
        || label.starts_with("balanced at ");
    exact.then_some(label)
}

fn settings(set: &str) -> (&'static str, Treatment) {
    match set {
        "lda-sr" => ("lda", Treatment::Scalar),
        "pbe-sr" => ("pbe", Treatment::Scalar),
        "lda-fr" => ("lda", Treatment::Dirac),
        "pbe-fr" => ("pbe", Treatment::Dirac),
        other => panic!("unknown set {other}"),
    }
}

/// Search one element and compare with its census label; returns the report line and whether the
/// label is a mismatch of an exact census class.
fn element(set: &str, symbol: &str, census_label: &str) -> (Value, bool) {
    let (functional, treatment) = settings(set);
    let ctx = context();
    let recipe = Recipe::for_dataset(symbol, functional, treatment).unwrap();
    let started = std::time::Instant::now();
    let searched = search_with(
        &ctx,
        &recipe,
        &Options::new(&recipe, Mode::Flag),
        &AtomSolver,
    )
    .unwrap_or_else(|e| panic!("{set} {symbol}: {e}"));
    let seconds = started.elapsed().as_secs_f64();
    let expected = exact(census_label);
    let wrong = expected.is_some_and(|label| label != searched.label);
    let line = json!({
        "set": set, "element": symbol, "census": census_label, "selected": searched.label,
        "index": searched.index, "tried": searched.tried.len(), "flagged": searched.flagged,
        "exact_class": expected.is_some(), "mismatch": wrong, "seconds": seconds,
        "missing": searched.diagnosis.as_ref().map(|d| d.missing.len()).unwrap_or(0),
        "peak_bytes": ctx.memory_peak(),
    });
    println!("{line}");
    (line, wrong)
}

fn report(line: &Value) {
    if let Ok(path) = std::env::var("MANDACARU_SEARCH_REPORT") {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        writeln!(file, "{line}").unwrap();
    }
}

/// The elements of `set` up to `max_z` searched, and what is wrong with them: a label that
/// disagrees with the census, and, when `clean` is asked for, any element returned flagged (the
/// light elements of the census are all unflagged). A run that searched nothing is a failure of
/// the run, not a pass: an empty selection (a typo in `MANDACARU_SEARCH_ELEMENTS`) must not be
/// read as zero mismatches.
fn run(set: &str, max_z: u8, clean: bool) -> Vec<String> {
    let only: Option<Vec<String>> = std::env::var("MANDACARU_SEARCH_ELEMENTS")
        .ok()
        .map(|s| s.split(',').map(str::to_string).collect());
    let mut problems = Vec::new();
    let mut searched = 0usize;
    for (z, (symbol, label)) in census(set) {
        if z > max_z || only.as_ref().is_some_and(|o| !o.contains(&symbol)) {
            continue;
        }
        let (line, wrong) = element(set, &symbol, &label);
        report(&line);
        searched += 1;
        if wrong {
            problems.push(format!(
                "{set} {symbol}: census {label}, search {}",
                line["selected"]
            ));
        }
        if clean && line["flagged"].as_bool() != Some(false) {
            problems.push(format!("{set} {symbol}: returned flagged"));
        }
    }
    if searched == 0 {
        problems.push(format!(
            "{set}: no element was searched (Z <= {max_z}, selection {only:?})"
        ));
    }
    problems
}

/// H to Ar: the first construction for most, zero norm deficit for Be, B and Ne, the shorter
/// local radius for Al and the third-row p block (PBE takes zero deficit for Si and P).
#[test]
fn search_census_light_lda() {
    let mismatches = run("lda-sr", 18, true);
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

#[test]
fn search_census_light_pbe() {
    let mismatches = run("pbe-sr", 18, true);
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

#[test]
#[ignore = "hours for the heavy atoms: see the module documentation"]
fn search_census_sweep() {
    let set = std::env::var("MANDACARU_SEARCH_SET").expect("MANDACARU_SEARCH_SET");
    let max_z: u8 = std::env::var("MANDACARU_SEARCH_MAX_Z")
        .map(|s| s.parse().unwrap())
        .unwrap_or(103);
    // The heavy elements include the flagged lanthanides: flags are reported in the JSON lines.
    let mismatches = run(&set, max_z, false);
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}
