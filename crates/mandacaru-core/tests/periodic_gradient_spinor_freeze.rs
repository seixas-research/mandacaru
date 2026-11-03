//! The oracle of a spinor case whose SCF only the crate converges (F2b): solves the displaced
//! snapshots `freeze_periodic_gradients.py dump` wrote and records their free energies. A
//! generation tool, not a gate, so the test is `#[ignore]`d and does nothing unless it is told what
//! to solve:
//!
//! ```text
//! MANDACARU_PAW_PATH=... MANDACARU_FREEZE_DIR=<dump dir> MANDACARU_FREEZE_CASE=nne_ldafr \
//!   MANDACARU_FREEZE_OUT=<energies.json> MANDACARU_FREEZE_START=z \
//!   cargo test --release -p mandacaru-core --test periodic_gradient_spinor_freeze -- --ignored
//! ```
//!
//! Every displaced crystal is solved with the oracle tolerances (free energy 1e-11 Ha, density
//! 1e-9) from the same initial moment direction (`MANDACARU_FREEZE_START`: `z`, `x` or `y`, the
//! snapshot's own moment turned as `Spinor::rotated` does), the converged moment is printed, and a
//! displaced geometry that does not converge fails the run. `freeze_periodic_gradients.py assemble`
//! turns the energies into the Richardson manifest.
//!
//! Interim: the Python extension still carries the SCF without the orientation step, so it cannot
//! freeze `nne_ldafr`. When the extension is rebuilt (WP-G) `freeze_periodic_gradients.py case
//! nne_ldafr` runs the normal path, and this file and the `dump`/`assemble` actions are deleted.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;

use mandacaru_core::{
    periodic::{spectrum::Smearing, Crystal, Options},
    Context,
};
use periodic_gradient_crystals_support::paw_root;
use periodic_gradient_spinor_support::{manifest_of, norm};
use serde_json::{json, Value};

fn variable(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// The snapshot at `path` with its dataset files named from the manifest and its initial moment
/// along `start`.
fn snapshot(path: &std::path::Path, manifest: &Value, start: &str) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut snapshot: Value = serde_json::from_str(&text).unwrap();
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
    let initial = snapshot["initial_m"].as_array().unwrap().clone();
    let n = initial.len() / 3;
    let block = match start {
        "x" => 0,
        "y" => 1,
        "z" => 2,
        other => panic!("MANDACARU_FREEZE_START is x, y or z, not {other}"),
    };
    // The snapshots start along z; a turned start moves the z block.
    assert!(
        initial[..2 * n].iter().all(|v| v.as_f64() == Some(0.)),
        "the dumped snapshots start along z"
    );
    let mut turned = vec![json!(0.0); 3 * n];
    turned[block * n..(block + 1) * n].clone_from_slice(&initial[2 * n..]);
    snapshot["initial_m"] = json!(turned);
    snapshot
}

#[test]
#[ignore = "generation tool: needs MANDACARU_FREEZE_DIR, _CASE and _OUT (see the module doc); about 5 s per snapshot"]
fn solve_the_dumped_snapshots() {
    let (Some(dir), Some(case), Some(out)) = (
        variable("MANDACARU_FREEZE_DIR"),
        variable("MANDACARU_FREEZE_CASE"),
        variable("MANDACARU_FREEZE_OUT"),
    ) else {
        eprintln!("MANDACARU_FREEZE_DIR, _CASE and _OUT are not set: nothing to solve");
        return;
    };
    let start = variable("MANDACARU_FREEZE_START").unwrap_or_else(|| "z".into());
    let dir = std::path::PathBuf::from(dir);
    let index: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join(format!("{case}.index.json"))).unwrap(),
    )
    .unwrap();
    // The setup block (datasets, width) is the tracked manifest of the snapshot's owner.
    let owner = if case.ends_with("_b") {
        &case[..case.len() - 2]
    } else {
        &case[..]
    };
    let manifest = manifest_of(owner);
    let width = manifest["scf"]["width_ha"].as_f64().unwrap();
    let options = Options {
        max_iterations: 1000,
        tolerance: 1e-11,
        density_tolerance: 1e-9,
        smearing: Smearing::FermiDirac,
        width,
        mixing: 0.25,
        history: 7,
        ..Options::default()
    };
    let mut energies = serde_json::Map::new();
    for entry in index["files"].as_array().unwrap() {
        let (key, file) = (
            entry["key"].as_str().unwrap(),
            entry["file"].as_str().unwrap(),
        );
        let started = std::time::Instant::now();
        let ctx = Context::new(Some(1), 1 << 30).unwrap();
        let value = snapshot(&dir.join(file), &manifest, &start);
        let crystal = Crystal::from_json(&ctx, &value.to_string()).unwrap();
        let result = crystal.solve(&options).unwrap();
        assert!(
            result.converged,
            "{key}: the oracle SCF must converge ({file})"
        );
        let n = crystal.grid.index.points();
        let moment: [f64; 3] = std::array::from_fn(|a| {
            result.density.as_slice()[(a + 1) * n..(a + 2) * n]
                .iter()
                .sum::<f64>()
                * crystal.grid.index.dv()
        });
        eprintln!(
            "{key}: {:.12} Ha, {} iterations, moment {moment:.4?} (|M| {:.4}), {:.1} s",
            result.free_energy,
            result.history.len(),
            norm(moment),
            started.elapsed().as_secs_f64()
        );
        energies.insert(key.into(), json!(result.free_energy));
    }
    std::fs::write(out, Value::Object(energies).to_string()).unwrap();
}
