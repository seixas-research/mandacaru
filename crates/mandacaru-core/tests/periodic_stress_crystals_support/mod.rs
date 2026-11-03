//! Shared by the `periodic_stress_crystals*` tests: the frozen real-crystal strain cases of
//! `freeze_periodic_strain.py` (`reference/periodic_strain/<case>.json` with the unstrained
//! snapshot `<case>.snapshot.json`; a second step family `<case>_b.json` shares the snapshot), the
//! crystal built from them with the force tests' loader, the pins asserted, and the gate of one
//! strain entry from its measured floor.
//!
//! The oracle is `+dF/d eps_ab` of the Rust SCF's free energy (Hartree, one entry at a time,
//! unsymmetrized), Richardson over the three steps of the manifest.
#![allow(dead_code)]

use crate::periodic_gradient_crystals_support::{paw_root, Data};
use mandacaru_core::periodic::{setup::shapes::COMPENSATION_CUTOFF, stress_terms::Stress};
use serde_json::Value;
use std::path::PathBuf;

pub mod gates;
pub mod run;

/// `Ha/Bohr^3` in `eV/Angstrom^3` (`units.py` owns the constant).
pub const TO_EV_PER_A3: f64 = 183.63153645;

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/migration/reference/periodic_strain")
}

/// A frozen file; a missing one fails loudly.
pub fn read(name: &str) -> Value {
    let path = root().join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path:?}: {e} (run freeze_periodic_strain.py case)"));
    serde_json::from_str(&text).unwrap()
}

pub fn manifest(file: &str) -> Value {
    read(&format!("{file}.json"))
}

/// The crystal of `case` (its snapshot with the dataset files joined to `MANDACARU_PAW_PATH`).
pub fn load(case: &'static str) -> Data {
    let manifest = manifest(case);
    // A case that shares the snapshot of another (a divergence treatment, a second family) names it.
    let owner = manifest["snapshot"].as_str().unwrap_or(case);
    let mut snapshot = read(&format!("{owner}.snapshot.json"));
    for (atom, d) in snapshot["sites"]["datasets"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        let relative = manifest["setup"]["datasets"][atom]["file"]
            .as_str()
            .unwrap();
        d["file"] = Value::String(paw_root().join(relative).to_str().unwrap().into());
    }
    Data::build(case, manifest, snapshot)
}

/// The pins the manifest recorded, held by the crystal that is differentiated.
pub fn assert_pins(data: &Data) {
    let pins = &data.manifest["pins"];
    let shape = data.crystal.grid.index.shape();
    let want: Vec<usize> = pins["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(shape.to_vec(), want, "node counts");
    assert_eq!(
        data.sigma,
        pins["sigma_bohr"].as_f64().unwrap(),
        "the ion width"
    );
    assert_eq!(
        COMPENSATION_CUTOFF,
        pins["dense_cutoff"].as_f64().unwrap(),
        "the dense cutoff"
    );
    let operations = mandacaru_core::math::reciprocal::operation_count(
        &data.crystal.context,
        &data.crystal.grid,
    )
    .unwrap();
    assert_eq!(
        operations as u64,
        pins["operations"].as_u64().unwrap(),
        "the alias operation set"
    );
}

/// The frozen derivative of one entry: `(rr, floor_richardson, floor_extrapolated)` (Hartree).
pub fn entry(manifest: &Value, a: usize, b: usize) -> (f64, f64, f64) {
    let e = manifest["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["a"].as_u64() == Some(a as u64) && e["b"].as_u64() == Some(b as u64))
        .unwrap_or_else(|| panic!("a missing frozen entry ({a}, {b})"));
    (
        e["rr"].as_f64().unwrap(),
        e["floor_richardson"].as_f64().unwrap(),
        e["floor_extrapolated"].as_f64().unwrap(),
    )
}

/// `sigma = g / Omega` in eV/Angstrom^3 for a derivative `g` in Hartree.
pub fn to_sigma_ev(g: f64, volume: f64) -> f64 {
    g / volume * TO_EV_PER_A3
}

/// The analytic and frozen derivative tensors of a manifest, in Hartree.
pub fn oracle(manifest: &Value) -> [[f64; 3]; 3] {
    std::array::from_fn(|a| std::array::from_fn(|b| entry(manifest, a, b).0))
}

pub fn summary(name: &str, stress: &Stress) {
    println!(
        "{name}: volume {:.3} Bohr^3, pressure {:.6e} Ha/Bohr^3 = {:.6e} eV/A^3, antisymmetric {:?}",
        stress.volume,
        stress.pressure(),
        stress.pressure() * TO_EV_PER_A3,
        stress.antisymmetric
    );
}
