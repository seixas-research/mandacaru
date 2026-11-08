//! Shared by the `periodic_response*` tests: a snapshot that carries its own `setup` block (the
//! frozen sidecar of a force case turned into the block, or a snapshot the Python setup wrote with
//! the block and the symmetry operations of a reduced mesh), solved with the oracle's options.
#![allow(dead_code)]

use crate::periodic_gradient_crystals_support::{paw_root, root, setup_block, snapshot_of};
use mandacaru_core::{
    periodic::{
        screened_exchange::ExxDivTreatment, spectrum::Smearing, Crystal, Options, PeriodicResult,
    },
    Context,
};
use serde_json::Value;
use std::path::PathBuf;

/// A frozen force case: its manifest and its snapshot with the `setup` block of the sidecar.
pub struct ForceCase {
    pub manifest: Value,
    pub snapshot: Value,
}

impl ForceCase {
    pub fn load(case: &str) -> Self {
        let text = std::fs::read_to_string(root().join(format!("{case}.json")))
            .unwrap_or_else(|e| panic!("{case}.json: {e}"));
        let manifest: Value = serde_json::from_str(&text).unwrap();
        let mut snapshot = snapshot_of(case, &manifest);
        snapshot["setup"] = setup_block(&manifest);
        Self { manifest, snapshot }
    }
}

/// The SCF options of a manifest's `scf` object (the oracle's tolerances).
pub fn options_of(scf: &Value) -> Options {
    let exxdiv = match scf["exxdiv_treatment"].as_str() {
        Some(name) => ExxDivTreatment::parse(name).unwrap(),
        None => ExxDivTreatment::None,
    };
    Options {
        max_iterations: scf["max_iterations"].as_u64().unwrap() as usize,
        tolerance: scf["tolerance"].as_f64().unwrap(),
        density_tolerance: scf["density_tolerance"].as_f64().unwrap(),
        smearing: Smearing::FermiDirac,
        width: scf["width_ha"].as_f64().unwrap(),
        mixing: scf["mixing"].as_f64().unwrap_or(0.25),
        history: scf["history"].as_u64().unwrap_or(7) as usize,
        exxdiv_treatment: exxdiv,
        ..Options::default()
    }
}

/// The crystal of `snapshot`, solved with the options of `manifest["scf"]`.
pub fn solve_snapshot(snapshot: &Value, manifest: &Value) -> (Crystal, PeriodicResult) {
    solve_at(snapshot, manifest, 1, None)
}

/// [`solve_snapshot`] on `threads` threads and, with `tight`, at the tolerances of the stress
/// tests' reference state (`1e-13` Ha, `1e-11` in the density): a force or a stress is first
/// order in the SCF residual, so two SCFs that are compared are solved well below the gate.
pub fn solve_at(
    snapshot: &Value,
    manifest: &Value,
    threads: usize,
    tight: Option<(f64, f64)>,
) -> (Crystal, PeriodicResult) {
    let ctx = Context::new(Some(threads), 1 << 30).unwrap();
    let crystal = Crystal::from_json(&ctx, &snapshot.to_string()).unwrap();
    let mut options = options_of(&manifest["scf"]);
    if let Some((energy, density)) = tight {
        options.tolerance = energy;
        options.density_tolerance = density;
    }
    let result = crystal.solve(&options).unwrap();
    assert!(result.converged, "the oracle SCF must converge");
    (crystal, result)
}

pub fn symmetry_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference/periodic_symmetry")
}

/// A crystal written twice by the setup (`freeze_periodic_gradients.py symmetry`): the whole mesh
/// and the mesh reduced by its space group, both with their `setup` block.
pub struct Pair {
    pub manifest: Value,
    pub full: Value,
    pub reduced: Value,
}

impl Pair {
    pub fn load(case: &str) -> Self {
        let read = |name: String| -> Value {
            let path = symmetry_root().join(name);
            serde_json::from_str(
                &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}")),
            )
            .unwrap()
        };
        let snapshot = |kind: &str| {
            let mut value = read(format!("{case}.{kind}.snapshot.json"));
            for d in value["sites"]["datasets"].as_array_mut().unwrap() {
                let relative = d["file"].as_str().unwrap().to_string();
                d["file"] = Value::String(paw_root().join(relative).to_str().unwrap().into());
            }
            value
        };
        Self {
            manifest: read(format!("{case}.json")),
            full: snapshot("full"),
            reduced: snapshot("reduced"),
        }
    }
}

/// The largest of `values`, NaN when any is NaN: `f64::max` returns the other operand for a NaN, so a
/// fold with it lets a NaN miss pass every `miss < gate` that follows. Empty gives `0`.
pub fn worst(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(0., |a, b| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    })
}
