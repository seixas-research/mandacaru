//! Generation tool, not a gate: re-runs ONLY the reconverged finite-difference oracles of the
//! HSE06 crystal cases (the snapshot and the manifest's `setup` sidecar are those
//! `periodic_freeze_snapshot.rs` wrote; it re-freezes both with the oracle in one pass) (`reference/periodic_gradients/<case>.json`, displaced atoms, and
//! `reference/periodic_strain/<case>.json`, strained cells) on the Rust energy. The oracle stays
//! independent of the analytic gradient and stress: it is the central difference of the converged
//! free energy of fresh SCFs, Richardson extrapolated over the manifest's three steps.
//!
//! Each SCF is a fresh crystal of the live setup (`parse_input_json`, `create_setup`, `build`)
//! from the request the case was set up with (`reference/periodic_requests/<snapshot>.json`, or
//! `strain_<snapshot>.json` for a stress case, one per frozen snapshot: the recipes, the mesh, the filter wave-vector, the node counts):
//!   * gradient: the same setup request with the displaced atom (fractional coordinates of the
//!     displaced Cartesian position), the ion width again `1.4 x` the coarsest step of the
//!     unchanged cell;
//!   * strain: the unstrained setup with `PeriodicSetup::with_cell((1 + eps) L)`, which holds the
//!     pins of the analytic stress (ion width, filter wave-vector, node counts, shape tables)
//!     and strains the lattice, the atoms, the grid step and the grid origin. THE PINS MATTER:
//!     a strain difference that lets the ion width follow the strained grid (the width is 1.4 x
//!     the coarsest grid step, here the step along the third lattice vector) differs from the
//!     analytic stress by a term proportional to the strain derivative of that step, 4e-6 eV/A^3
//!     in the zz stress of the skew LiH cell.
//!
//! The base `free_energy_ha` is the SCF of the frozen SNAPSHOT (what the tests solve and compare
//! with at 1e-9 Ha); the live setup differs from a snapshot frozen on an earlier day by a
//! constant of a few 1e-9 Ha (LiH 3.3e-9), which cancels in every difference.
//!
//! Every SCF energy is appended to `<case>.energies.jsonl` in `MANDACARU_FREEZE_WORK` as it is
//! found, so an interrupted run resumes; the manifest is rewritten (its keys sorted) when all are
//! in. One SCF at a time, one thread.
//!
//! ```text
//! MANDACARU_PAW_PATH=... MANDACARU_FREEZE_KIND=strain MANDACARU_FREEZE_CASE=lih_hse06 \
//!   MANDACARU_FREEZE_WORK=<scratch dir> \
//!   cargo test --profile release-test -p mandacaru-core --test periodic_freeze_hybrid -- --ignored --nocapture
//! ```
//! `KIND` is `gradient` or `strain`; `CASE` the manifest name (`si_hsefr_tr3_b`, ...).
mod periodic_freeze_support;
mod periodic_gradient_crystals_support;
mod periodic_stress_crystals_support;

use periodic_freeze_support::{
    free_energy, gradient_oracle, options_of, reference, request_of, strain_oracle, Case, Ledger,
};
use periodic_gradient_crystals_support::gates::PLAN_GATE;
use periodic_stress_crystals_support::{
    gates::{plan, PLAN_DIAGONAL},
    TO_EV_PER_A3,
};
use serde_json::{json, Value};
use std::{path::PathBuf, time::Instant};

#[test]
#[ignore = "generation tool: minutes to hours of SCFs; needs MANDACARU_FREEZE_KIND, _CASE and _WORK"]
fn refreeze_a_hybrid_crystal_oracle() {
    let (Ok(kind), Ok(name), Ok(work)) = (
        std::env::var("MANDACARU_FREEZE_KIND"),
        std::env::var("MANDACARU_FREEZE_CASE"),
        std::env::var("MANDACARU_FREEZE_WORK"),
    ) else {
        println!("nothing to do: MANDACARU_FREEZE_KIND, _CASE and _WORK are not all set");
        return;
    };
    let folder = match kind.as_str() {
        "gradient" => "periodic_gradients",
        "strain" => "periodic_strain",
        other => panic!("MANDACARU_FREEZE_KIND is gradient or strain, not {other:?}"),
    };
    let path = reference().join(folder).join(format!("{name}.json"));
    let mut manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let owner = manifest["snapshot"].as_str().unwrap_or(&name).to_string();
    let case = Case::load(request_of(&if kind == "strain" {
        format!("strain_{owner}")
    } else {
        owner.clone()
    }));
    let options = options_of(&manifest);
    let mut ledger =
        Ledger::open(PathBuf::from(&work).join(format!("{folder}_{name}.energies.jsonl")));
    let start = Instant::now();
    let old_energy = manifest["free_energy_ha"].as_f64().unwrap();
    let base = case.setup(None);
    // The frozen snapshot, built by the setup of the day it was frozen, differs from the live
    // setup by a constant at the 1e-9 Ha level (3.3e-9 for LiH): the SCF-energy gate of the tests
    // solves the snapshot, so the base free energy is the snapshot's, and the displaced and
    // strained energies (which only the live setup can build) carry the constant, which cancels
    // in every difference.
    let live = ledger.get("base", &|| free_energy(&case, &base, &options));
    let energy = ledger.get("base_snapshot", &|| {
        let leaked: &'static str = Box::leak(name.clone().into_boxed_str());
        let data = if kind == "gradient" {
            periodic_gradient_crystals_support::Data::load(leaked)
        } else {
            periodic_stress_crystals_support::load(leaked)
        };
        let result = data.crystal.solve(&options).unwrap();
        assert!(result.converged, "the snapshot SCF does not converge");
        result.free_energy
    });
    println!(
        "{name}: free energy {old_energy:.12} -> {energy:.12} (live setup {live:.12}, {:+.3e})",
        live - energy
    );
    manifest["free_energy_ha"] = json!(energy);
    if kind == "gradient" {
        gradient_oracle(&mut manifest, &case, &options, &mut ledger, PLAN_GATE);
    } else {
        let volume = manifest["volume_bohr3"].as_f64().unwrap();
        strain_oracle(
            &mut manifest,
            &case,
            &base,
            &options,
            &mut ledger,
            &|a, b| plan(a, b, volume),
            3. * PLAN_DIAGONAL / TO_EV_PER_A3 * volume,
        );
    }
    // The wall time of a run resumed from its ledger says nothing about the oracle: not recorded.
    manifest.as_object_mut().unwrap().remove("seconds");
    manifest["refrozen_2026_11_10"] = json!(format!(
        "HSE06 smooth-core offset fix: re-frozen on the Rust energy by periodic_freeze_hybrid.rs; free energy {old_energy} -> {energy} (snapshot SCF; the live setup is {:+.3e} Ha from it)",
        live - energy
    ));
    std::fs::write(&path, serde_json::to_string(&manifest).unwrap()).unwrap();
    println!("{name}: written, {:.0} s", start.elapsed().as_secs_f64());
}
