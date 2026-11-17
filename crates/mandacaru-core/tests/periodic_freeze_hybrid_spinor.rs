//! Generation tool, not a gate: re-freezes the spinor-hybrid snapshots of
//! `reference/hybrid_spinor/` (`si_so_tr3`, `si_so_tr4`, `si4_y2`, `si8_gamma`) against the
//! installed PAW library (`MANDACARU_PAW_PATH`), with the Rust setup alone and no SCF.
//!
//! Each fixture is the setup of its captured request
//! (`reference/periodic_requests/hybrid_spinor_<case>.json`, a `_whole_mesh` request: a hybrid
//! unfolds a time-reversal half mesh itself, but these fixtures hold every point of the mesh)
//! saved in the snapshot schema. The `provenance` keeps what `periodic_hybrid_spinor_support`
//! reads: the geometry, the mesh, `library` (the dataset set's name), `orbital_labels` (the
//! `(atom, l, m)` of each basis function, from the setup's function list) and `reference_case`,
//! with the dataset files and their digests under `datasets`. The report compares the new
//! fixture with the one on disk: the grid, the counts, the k-points and the orbital labels,
//! which must be equal (the library moves the radial data, not the layout).
//!
//! ```text
//! MANDACARU_PAW_PATH=... MANDACARU_FREEZE_CASE=si_so_tr3 MANDACARU_FREEZE_WORK=<scratch dir> \
//!   [MANDACARU_FREEZE_OUT=<dir, default reference/hybrid_spinor>] \
//!   cargo test --profile release-test -p mandacaru-core --test periodic_freeze_hybrid_spinor -- --ignored --nocapture
//! ```
mod periodic_freeze_support;
mod periodic_gradient_crystals_support;

use mandacaru_core::paw::digest::{dataset_content_sha256, sha256};
use periodic_freeze_support::{reference, request_of, Case};
use periodic_gradient_crystals_support::paw_root;
use serde_json::{json, Value};
use std::path::PathBuf;

#[test]
#[ignore = "generation tool: needs MANDACARU_FREEZE_CASE and _WORK (see the module doc); a few seconds"]
fn refreeze_a_spinor_hybrid_snapshot() {
    let (Ok(name), Ok(work)) = (
        std::env::var("MANDACARU_FREEZE_CASE"),
        std::env::var("MANDACARU_FREEZE_WORK"),
    ) else {
        println!("nothing to do: MANDACARU_FREEZE_CASE and _WORK are not both set");
        return;
    };
    let folder = reference().join("hybrid_spinor");
    let out = std::env::var("MANDACARU_FREEZE_OUT").map_or(folder.clone(), PathBuf::from);
    let old: Value = serde_json::from_str(
        &std::fs::read_to_string(folder.join(format!("{name}.json"))).unwrap(),
    )
    .unwrap();
    let case = Case::load(request_of(&format!("hybrid_spinor_{name}")));
    let setup = case.setup(None);
    let crystal = setup.build(&case.ctx).unwrap();
    std::fs::create_dir_all(&work).unwrap();
    let raw = PathBuf::from(&work).join(format!("{name}.rust.snapshot.json"));
    crystal.save(raw.to_str().unwrap()).unwrap();
    let mut snapshot: Value =
        serde_json::from_str(&std::fs::read_to_string(&raw).unwrap()).unwrap();
    std::fs::remove_file(&raw).unwrap();

    let set = case.request["_dataset_set"].as_str().unwrap().to_string();
    let mut datasets = Vec::new();
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
        datasets.push(
            json!({"file": relative, "symbol": symbol, "bytes": bytes.len(),
                             "sha256": sha256(&bytes), "content_sha256": content}),
        );
    }
    // The (atom, l, m) of every basis function, in the setup's function order.
    let functions = &snapshot["setup"]["functions"];
    let tables = snapshot["setup"]["tables"].as_array().unwrap();
    let labels: Vec<Value> = (0..functions["atom"].as_array().unwrap().len())
        .map(|i| {
            let table = functions["table"][i].as_u64().unwrap() as usize;
            json!([functions["atom"][i], tables[table]["l"], functions["m"][i]])
        })
        .collect();
    let geometry = old["provenance"]["geometry"].clone();
    snapshot["provenance"] = json!({
        "family": "paw-lcao", "geometry": geometry,
        "mesh_gamma": old["provenance"]["mesh_gamma"], "mesh_size": old["provenance"]["mesh_size"],
        "setup_engine": "rust-setup", "library": set, "datasets": datasets,
        "orbital_labels": labels, "reference_case": name,
        "generator": "crates/mandacaru-core/tests/periodic_freeze_hybrid_spinor.rs",
    });
    // The setup block and the symmetry operations belong to the setup, not to these fixtures.
    snapshot.as_object_mut().unwrap().remove("setup");

    // What the layout of the old fixture says and the library cannot move.
    assert_eq!(snapshot["spinor"], json!(true), "{name}: spinor");
    assert_eq!(
        snapshot["time_reversal"],
        json!(true),
        "{name}: time reversal"
    );
    assert_eq!(snapshot["permutations"], json!([]), "{name}: permutations");
    let weights: Vec<f64> = snapshot["kpoints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k["weight"].as_f64().unwrap())
        .collect();
    let whole = 1. / weights.len() as f64;
    assert!(
        weights.iter().all(|w| (w - whole).abs() < 1e-12),
        "{name}: the mesh is not whole: {weights:?}"
    );
    for key in [
        "grid",
        "kpoints",
        "channels",
        "m",
        "p",
        "spins",
        "functional",
    ] {
        let same = if key == "kpoints" {
            // The mesh points, to the precision of their coordinates.
            let fractional = |v: &Value| -> Vec<f64> {
                v.as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|k| {
                        k["fractional"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|x| x.as_f64().unwrap())
                    })
                    .collect()
            };
            let (a, b) = (fractional(&old[key]), fractional(&snapshot[key]));
            a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-12)
        } else if key == "grid" {
            old[key]["shape"] == snapshot[key]["shape"]
        } else {
            old[key] == snapshot[key]
        };
        println!(
            "{name}: {key} {} (old {}, new {})",
            if same { "equal" } else { "DIFFERS" },
            if key == "kpoints" || key == "grid" {
                json!("...")
            } else {
                old[key].clone()
            },
            if key == "kpoints" || key == "grid" {
                json!("...")
            } else {
                snapshot[key].clone()
            }
        );
    }
    println!(
        "{name}: orbital labels {}, electrons {} -> {}, reference {} -> {}",
        if labels == *old["provenance"]["orbital_labels"].as_array().unwrap() {
            "equal"
        } else {
            "DIFFER"
        },
        old["electrons"],
        snapshot["electrons"],
        old["reference"],
        snapshot["reference"]
    );
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(
        out.join(format!("{name}.json")),
        serde_json::to_string(&snapshot).unwrap(),
    )
    .unwrap();
    println!("{name}: written to {out:?}");
}
