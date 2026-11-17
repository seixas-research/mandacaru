//! Generation tool, not a gate: re-freezes the symmetry pairs of `reference/periodic_symmetry/`
//! (`<case>.full.snapshot.json`, the whole mesh, and `<case>.reduced.snapshot.json`, the mesh
//! reduced by the space group of the cell, whose `setup` block lists the operations) and the
//! manifest `<case>.json` against the installed PAW library (`MANDACARU_PAW_PATH`), with the Rust
//! setup alone. There is no finite-difference oracle: the tests compare the force and the stress
//! of the reduced snapshot with those of the whole-mesh one.
//!
//! Each snapshot is the setup of its captured request
//! (`reference/periodic_requests/symmetry_<case>_<full|reduced>.json`) saved in the snapshot
//! schema, its dataset files named relative to the library and its `provenance` naming this tool,
//! the date and the library digests, then loaded back and solved with the manifest's SCF options.
//! The manifest keeps its geometry and options and gets the two free energies, the k-point and
//! operation counts (asserted equal to the frozen ones: the library does not move them) and a
//! `stamp`. The two free energies must agree to 1e-8 Ha.
//!
//! ```text
//! MANDACARU_PAW_PATH=... MANDACARU_FREEZE_CASE=si111 MANDACARU_FREEZE_WORK=<scratch dir> \
//!   [MANDACARU_FREEZE_OUT=<dir, default reference/periodic_symmetry>] \
//!   cargo test --profile release-test -p mandacaru-core --test periodic_freeze_symmetry -- --ignored --nocapture
//! ```
mod periodic_freeze_support;
mod periodic_gradient_crystals_support;

use mandacaru_core::{
    paw::digest::{dataset_content_sha256, sha256},
    periodic::Crystal,
};
use periodic_freeze_support::{options_of, reference, request_of, Case};
use periodic_gradient_crystals_support::paw_root;
use serde_json::{json, Value};
use std::{path::PathBuf, time::Instant};

#[test]
#[ignore = "generation tool: needs MANDACARU_FREEZE_CASE and _WORK (see the module doc); two SCFs, under a minute"]
fn refreeze_a_symmetry_pair() {
    let (Ok(name), Ok(work)) = (
        std::env::var("MANDACARU_FREEZE_CASE"),
        std::env::var("MANDACARU_FREEZE_WORK"),
    ) else {
        println!("nothing to do: MANDACARU_FREEZE_CASE and _WORK are not both set");
        return;
    };
    let started = Instant::now();
    let folder = reference().join("periodic_symmetry");
    let out = std::env::var("MANDACARU_FREEZE_OUT").map_or(folder.clone(), PathBuf::from);
    let mut manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(folder.join(format!("{name}.json"))).unwrap(),
    )
    .unwrap();
    let options = options_of(&manifest);
    std::fs::create_dir_all(&work).unwrap();
    std::fs::create_dir_all(&out).unwrap();
    let mut energies = serde_json::Map::new();
    let mut counts = serde_json::Map::new();
    let mut operations = serde_json::Map::new();
    let mut library = Vec::new();
    for which in ["full", "reduced"] {
        let case = Case::load(request_of(&format!("symmetry_{name}_{which}")));
        let setup = case.setup(None);
        let crystal = setup.build(&case.ctx).unwrap();
        let raw = PathBuf::from(&work).join(format!("{name}.{which}.rust.snapshot.json"));
        crystal.save(raw.to_str().unwrap()).unwrap();
        let mut snapshot: Value =
            serde_json::from_str(&std::fs::read_to_string(&raw).unwrap()).unwrap();
        std::fs::remove_file(&raw).unwrap();
        let set = case.request["_dataset_set"].as_str().unwrap();
        library.clear();
        for (entry, atom) in snapshot["sites"]["datasets"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .zip(case.request["atoms"].as_array().unwrap())
        {
            let symbol = atom["symbol"].as_str().unwrap();
            let relative = format!("{set}/{symbol}.parquet");
            let file = paw_root().join(&relative);
            let bytes = std::fs::read(&file).unwrap();
            let content = dataset_content_sha256(&case.ctx, &file).unwrap();
            assert_eq!(entry["content_sha256"].as_str().unwrap(), content);
            entry["file"] = json!(relative);
            entry["symbol"] = json!(symbol);
            library.push(
                json!({"file": relative, "symbol": symbol, "bytes": bytes.len(),
                                "sha256": sha256(&bytes), "content_sha256": content}),
            );
        }
        let numbers: Vec<u32> = case.request["atoms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| case.datasets[a["species"].as_u64().unwrap() as usize].atomic_number as u32)
            .collect();
        snapshot["provenance"] = json!({
            "family": "paw-lcao",
            "geometry": {"cell_angstrom": manifest["cell_angstrom"], "numbers": numbers,
                         "pbc": [true, true, true], "positions_angstrom": manifest["positions_angstrom"]},
            "mesh_gamma": manifest["kpts"]["gamma"], "mesh_size": manifest["kpts"]["size"],
            "setup_engine": "rust-setup", "library": library,
            "generator": "crates/mandacaru-core/tests/periodic_freeze_symmetry.rs",
        });
        let mut joined = snapshot.clone();
        for d in joined["sites"]["datasets"].as_array_mut().unwrap() {
            d["file"] = json!(paw_root().join(d["file"].as_str().unwrap()));
        }
        let loaded = Crystal::from_json(&case.ctx, &joined.to_string()).unwrap();
        let result = loaded.solve(&options).unwrap();
        assert!(
            result.converged,
            "{name} {which}: the SCF does not converge"
        );
        let n_ops = setup.symmetry.as_ref().map_or(1, |s| s.len());
        println!(
            "{name} {which}: {} k-points, {n_ops} operations, m {} p {} electrons {}, free energy {:.12} Ha (was {:.12}), {} iterations",
            loaded.points.len(),
            loaded.m,
            loaded.p,
            loaded.electrons,
            result.free_energy,
            manifest["free_energy_ha"][which].as_f64().unwrap(),
            result.history.len()
        );
        assert_eq!(
            json!(loaded.points.len()),
            manifest["k_points"][which],
            "{name} {which}: k-point count"
        );
        assert_eq!(
            json!(n_ops),
            manifest["operations"][which],
            "{name} {which}: operation count"
        );
        energies.insert(which.into(), json!(result.free_energy));
        counts.insert(which.into(), json!(loaded.points.len()));
        operations.insert(which.into(), json!(n_ops));
        std::fs::write(
            out.join(format!("{name}.{which}.snapshot.json")),
            serde_json::to_string(&snapshot).unwrap(),
        )
        .unwrap();
    }
    let gap = (energies["full"].as_f64().unwrap() - energies["reduced"].as_f64().unwrap()).abs();
    println!("{name}: full - reduced free energy {gap:.3e} Ha");
    assert!(
        gap < 1e-8,
        "{name}: the whole and the reduced mesh disagree"
    );
    manifest["free_energy_ha"] = Value::Object(energies);
    manifest["k_points"] = Value::Object(counts);
    manifest["operations"] = Value::Object(operations);
    manifest["stamp"] = json!({
        "engine": "mandacaru-core Rust setup (periodic::setup::create_setup) and SCF",
        "generator": "crates/mandacaru-core/tests/periodic_freeze_symmetry.rs",
        "library": library,
        "platform": format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        "extra": {"case": name},
    });
    std::fs::write(
        out.join(format!("{name}.json")),
        serde_json::to_string(&manifest).unwrap(),
    )
    .unwrap();
    println!(
        "{name}: written to {out:?}, {:.0} s",
        started.elapsed().as_secs_f64()
    );
}
