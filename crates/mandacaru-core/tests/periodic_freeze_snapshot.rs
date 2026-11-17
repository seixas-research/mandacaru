//! Generation tool, not a gate: re-freezes a crystal snapshot of `reference/periodic_gradients/`
//! or `reference/periodic_strain/` and the finite-difference oracle of its manifest against the
//! installed PAW library (`MANDACARU_PAW_PATH`), with the Rust setup alone. Semilocal and hybrid
//! (HSE06) manifests, forces (`MANDACARU_FREEZE_KIND=gradient`, the default) and stress
//! (`strain`), scalar, collinear spin-polarized and spinor crystals go through the one path.
//!
//! The crystal is the setup of the request the case was set up with
//! (`reference/periodic_requests/<snapshot>.json` for a force case, `strain_<snapshot>.json` for a
//! stress case: the lattice, the atoms, the mesh, the basis recipes, the filter wave-vector, the
//! projector basis; a `_whole_mesh` request lists every point of its time-reversal mesh) on the
//! installed dataset files, built (`create_setup`, `build`) and saved in the snapshot schema, then
//! post-processed: the dataset files of its `sites` block are named relative to the library (the
//! tests join them with `MANDACARU_PAW_PATH`) and its `provenance` names this tool, the date and
//! the library digests. A manifest with a `snapshot` entry (a second step family `_b`, `_c`, a
//! divergence treatment `_gb`, a released pin `unpin_*`) shares the snapshot of that case: freeze
//! the owner first; the sharer reads the owner's new snapshot from `MANDACARU_FREEZE_OUT`.
//!
//! The manifest `<case>.json` keeps its geometry, discretization and SCF options and gets what
//! depends on the library: the `setup` sidecar (the basis tables, the function and projector
//! lists, the ion width, written in the shapes the test support reads from the snapshot's own
//! `setup` block, and checked equal to it), the free energy of the snapshot's SCF, the finite
//! differences of fresh SCFs of the displaced (force) or strained (stress) setups at the
//! manifest's coordinates or entries and steps (Richardson, `PAW_COMPLETION_PLAN.md` 14.4 item 1)
//! and a `stamp` naming the library. A stress manifest's `pins` (node counts, ion width, filter
//! wave-vector, alias operations, dense cutoff) are asserted equal to the setup's, and its
//! released pins (`pins.unpinned`) are released in the strained setups (`Case::strained`). Each
//! floor is printed against the plan's gate; a floor above it is the cue to halve the steps in
//! the manifest. The files are written once, at the end, after the snapshot has been solved
//! through the same `Data` the tests use (digests, ion width, sidecar all checked).
//!
//! The report compares the new snapshot with the one on disk (or with `MANDACARU_FREEZE_AGAINST`,
//! a directory holding `<snapshot>.snapshot.json`): sizes, the largest difference of every numeric
//! block, the basis tables by angular momentum. Every SCF energy is appended to
//! `<kind>_<case>.energies.jsonl` in `MANDACARU_FREEZE_WORK` as it is found, so an interrupted run
//! resumes (the ledger labels carry the step index, not the step: use a fresh `WORK` after
//! changing the steps).
//!
//! ```text
//! MANDACARU_PAW_PATH=... MANDACARU_FREEZE_CASE=auh_ldafr MANDACARU_FREEZE_WORK=<scratch dir> \
//!   [MANDACARU_FREEZE_KIND=gradient|strain] \
//!   [MANDACARU_FREEZE_OUT=<dir, default reference/periodic_gradients or periodic_strain>] \
//!   [MANDACARU_FREEZE_AGAINST=<dir>] [MANDACARU_FREEZE_ORACLE=skip] \
//!   [MANDACARU_FREEZE_MANIFEST=<manifest to start from, default <kind folder>/<case>.json>] \
//!   [MANDACARU_FREEZE_STEP_FACTOR=<f: every step of the manifest times f, e.g. 0.5 to halve>] \
//!   cargo test --profile release-test -p mandacaru-core --test periodic_freeze_snapshot -- --ignored --nocapture
//! ```
//! `MANDACARU_FREEZE_STEP_FACTOR` rescales the manifest's steps (`steps_angstrom` and `steps_bohr`, or
//! `steps`) before the oracle runs and is written with it: the halving of the steps that brings a
//! Richardson floor under the plan's gate (use a fresh `WORK`: the ledger labels carry the step
//! index). `MANDACARU_FREEZE_ORACLE=skip` writes the snapshot and solves it, without the displaced SCFs
//! (the manifest then keeps its old oracle and is not written).
mod periodic_freeze_support;
mod periodic_gradient_crystals_support;
mod periodic_stress_crystals_support;

use mandacaru_core::paw::digest::{dataset_content_sha256, sha256};
use periodic_freeze_support::{
    gradient_oracle, options_of, reference, request_of, strain_oracle, Case, Ledger,
};
use periodic_gradient_crystals_support::{gates::PLAN_GATE, paw_root, setup_block, Data};
use periodic_stress_crystals_support::gates::{plan, PLAN_DIAGONAL};
use serde_json::{json, Value};
use std::{path::PathBuf, time::Instant};

/// A basis table or lattice as the manifest's sidecar writes an array.
fn encoded(values: &Value, shape: &[usize]) -> Value {
    json!({"shape": shape, "dtype": "float64", "re": values})
}

/// The `setup` sidecar of a manifest from the `setup` and `sites` blocks of a snapshot: the shapes
/// `periodic_gradient_crystals_support::setup_block` reads back.
fn sidecar(snapshot: &Value, case: &Case) -> Value {
    let (setup, sites) = (&snapshot["setup"], &snapshot["sites"]);
    let tables = setup["tables"].as_array().unwrap();
    let functions = &setup["functions"];
    let centers = sites["centers"].as_array().unwrap();
    let atoms = functions["atom"].as_array().unwrap();
    let function_count = atoms.len();
    let basis_functions: Vec<Value> = (0..function_count)
        .map(|i| {
            let table = functions["table"][i].as_u64().unwrap() as usize;
            let atom = atoms[i].as_u64().unwrap() as usize;
            json!({"l": tables[table]["l"], "m": functions["m"][i], "atom": atom,
                   "center_bohr": centers[atom], "table": table})
        })
        .collect();
    // The sidecar's lattice is the request's, row by row (the support reads it with `from_row_slice`).
    let rows = case.request["lattice"].clone();
    json!({
        "basis_radial_tables": tables.iter().map(|t| json!({
            "l": t["l"],
            "r": encoded(&t["r"], &[t["r"].as_array().unwrap().len()]),
            "values": encoded(&t["values"], &[t["values"].as_array().unwrap().len()]),
        })).collect::<Vec<_>>(),
        "basis_functions": basis_functions,
        "centers_bohr": centers,
        "lattice_bohr": encoded(&rows, &[3, 3]),
        "datasets": sites["datasets"].as_array().unwrap().iter()
            .map(|d| json!({"symbol": d["symbol"], "file": d["file"]})).collect::<Vec<_>>(),
        "projector_basis": sites["projector_basis"],
        "sigma_bohr": setup["sigma_bohr"],
        "projector_orbitals": setup["projectors"],
        "orbital_atoms": functions["atom"],
    })
}

/// The atomic number of every atom of the request.
fn numbers_of(case: &Case) -> Vec<u32> {
    case.request["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| case.datasets[a["species"].as_u64().unwrap() as usize].atomic_number as u32)
        .collect()
}

/// Every number of `v`, flattened in document order.
fn numbers(v: &Value, out: &mut Vec<f64>) {
    match v {
        Value::Number(n) => out.push(n.as_f64().unwrap()),
        Value::Array(a) => a.iter().for_each(|x| numbers(x, out)),
        Value::Object(o) => o.values().for_each(|x| numbers(x, out)),
        _ => {}
    }
}

/// The largest difference and the largest magnitude of two blocks of numbers of equal length.
fn gap(old: &Value, new: &Value) -> Option<(usize, usize, f64, f64)> {
    let (mut a, mut b) = (Vec::new(), Vec::new());
    numbers(old, &mut a);
    numbers(new, &mut b);
    if a.is_empty() && b.is_empty() {
        return None;
    }
    let peak = a.iter().fold(0f64, |m, x| m.max(x.abs()));
    let diff = if a.len() == b.len() {
        a.iter()
            .zip(&b)
            .fold(0f64, |m, (x, y)| m.max((x - y).abs()))
    } else {
        f64::NAN
    };
    Some((a.len(), b.len(), diff, peak))
}

fn report_snapshot(old: &Value, new: &Value) {
    println!("--- snapshot blocks: old length, new length, largest |new - old|, largest |old|");
    let mut keys: Vec<&String> = new.as_object().unwrap().keys().collect();
    keys.sort();
    for key in keys {
        if matches!(key.as_str(), "provenance" | "sites" | "setup") {
            continue;
        }
        let (a, b) = (&old[key], &new[key]);
        match gap(a, b) {
            Some((n, m, diff, peak)) => {
                println!("{key:>18}: {n:>8} {m:>8}  {diff:.3e}  {peak:.3e}")
            }
            None if a == b => {}
            None => println!("{key:>18}: old {a} new {b}"),
        }
    }
    let table = |snapshot: &Value| -> Vec<(u64, usize, Value)> {
        snapshot["setup"]["tables"]
            .as_array()
            .map(|tables| {
                tables
                    .iter()
                    .map(|t| {
                        (
                            t["l"].as_u64().unwrap(),
                            t["values"].as_array().unwrap().len(),
                            t["values"].clone(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    println!(
        "--- basis tables (snapshot setup blocks): l, old points, new points, largest |new - old|"
    );
    for ((l, n, a), (_, m, b)) in table(old).into_iter().zip(table(new)) {
        let (_, _, diff, peak) = gap(&a, &b).unwrap();
        println!("  l {l}: {n} {m}  {diff:.3e} of {peak:.3e}");
    }
}

fn old_tables(manifest: &Value) {
    println!("--- basis tables of the old sidecar: l, points");
    for t in manifest["setup"]["basis_radial_tables"].as_array().unwrap() {
        println!("  l {}: {}", t["l"], t["r"]["re"].as_array().unwrap().len());
    }
}

/// `YYYY-MM-DD` of today (UTC), from the system clock.
fn today() -> String {
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        / 86400;
    // Civil-from-days (H. Hinnant).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

#[test]
#[ignore = "generation tool: needs MANDACARU_FREEZE_CASE and _WORK (see the module doc); AuH: 25 SCFs, 3 to 4 minutes at 3 threads"]
fn refreeze_a_crystal_snapshot_and_its_oracle() {
    let (Ok(name), Ok(work)) = (
        std::env::var("MANDACARU_FREEZE_CASE"),
        std::env::var("MANDACARU_FREEZE_WORK"),
    ) else {
        println!("nothing to do: MANDACARU_FREEZE_CASE and _WORK are not both set");
        return;
    };
    let started = Instant::now();
    let kind = std::env::var("MANDACARU_FREEZE_KIND").unwrap_or_else(|_| "gradient".into());
    let strain = match kind.as_str() {
        "gradient" => false,
        "strain" => true,
        other => panic!("MANDACARU_FREEZE_KIND is gradient or strain, not {other:?}"),
    };
    let folder = reference().join(if strain {
        "periodic_strain"
    } else {
        "periodic_gradients"
    });
    let out = std::env::var("MANDACARU_FREEZE_OUT").map_or(folder.clone(), PathBuf::from);
    let against = std::env::var("MANDACARU_FREEZE_AGAINST").map_or(folder.clone(), PathBuf::from);
    let skip_oracle = std::env::var("MANDACARU_FREEZE_ORACLE").is_ok_and(|v| v == "skip");
    let read = |path: PathBuf| -> Value {
        serde_json::from_str(
            &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}")),
        )
        .unwrap()
    };
    let mut manifest = read(
        std::env::var("MANDACARU_FREEZE_MANIFEST")
            .map_or(folder.join(format!("{name}.json")), PathBuf::from),
    );
    let owner = manifest["snapshot"].as_str().unwrap_or(&name).to_string();
    if let Ok(factor) = std::env::var("MANDACARU_FREEZE_STEP_FACTOR") {
        let factor: f64 = factor.parse().unwrap();
        for key in ["steps_angstrom", "steps_bohr", "steps"] {
            if let Some(steps) = manifest[key].as_array_mut() {
                for step in steps {
                    *step = json!(step.as_f64().unwrap() * factor);
                }
                println!("{name}: {key} times {factor}: {}", manifest[key]);
            }
        }
    }
    let request = request_of(&if strain {
        format!("strain_{owner}")
    } else {
        owner.clone()
    });
    let case = Case::load(request);
    let options = options_of(&manifest);

    // The snapshot: the live setup, saved and post-processed (an owner), or the owner's new one.
    let setup = case.setup(None);
    let crystal = setup.build(&case.ctx).unwrap();
    std::fs::create_dir_all(&work).unwrap();
    let raw = PathBuf::from(&work).join(format!("{name}.rust.snapshot.json"));
    crystal.save(raw.to_str().unwrap()).unwrap();
    let mut snapshot = read(raw.clone());
    std::fs::remove_file(&raw).unwrap();
    let set = case.request["_dataset_set"].as_str().unwrap();
    let atoms = case.request["atoms"].as_array().unwrap();
    let mut library = Vec::new();
    for (entry, atom) in snapshot["sites"]["datasets"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .zip(atoms)
    {
        let symbol = atom["symbol"].as_str().unwrap();
        let relative = format!("{set}/{symbol}.parquet");
        let file = paw_root().join(&relative);
        let bytes = std::fs::read(&file).unwrap();
        let sha256 = sha256(&bytes);
        let content = dataset_content_sha256(&case.ctx, &file).unwrap();
        assert_eq!(entry["content_sha256"].as_str().unwrap(), content);
        entry["file"] = json!(relative);
        entry["symbol"] = json!(symbol);
        library.push(
            json!({"file": relative, "symbol": symbol, "bytes": bytes.len(), "sha256": sha256,
                   "content_sha256": content}),
        );
    }
    let date = today();
    let geometry = json!({
        "cell_angstrom": manifest["cell_angstrom"], "numbers": numbers_of(&case),
        "pbc": [true, true, true], "positions_angstrom": manifest["positions_angstrom"],
    });
    snapshot["provenance"] = json!({
        "family": "paw-lcao", "geometry": geometry,
        "mesh_gamma": manifest["kpts"]["gamma"], "mesh_size": manifest["kpts"]["size"],
        "setup_engine": "rust-setup", "date": date, "library": library,
        "generator": "crates/mandacaru-core/tests/periodic_freeze_snapshot.rs",
    });
    if owner == name {
        let old_snapshot = read(against.join(format!("{name}.snapshot.json")));
        old_tables(&manifest);
        report_snapshot(&old_snapshot, &snapshot);
    } else {
        // The sharer's crystal is the owner's frozen one: the same live setup, saved again.
        let frozen = read(out.join(format!("{owner}.snapshot.json")));
        assert_eq!(
            frozen["setup"], snapshot["setup"],
            "{name}: the owner's snapshot {owner} is not the crystal of this request"
        );
        snapshot = frozen;
    }
    if strain {
        // The pins of the analytic stress are properties of the cell and the grid; the library
        // does not move them. The node counts, the alias operations and the dense cutoff must
        // equal the frozen ones; the ion width and the filter wave-vector are written again (the
        // tests assert them exactly against the snapshot), after a check that they moved by no
        // more than the last digit of the manifest's literal geometry.
        let same = |new: f64, old: f64| (new - old).abs() <= 1e-12 * old.abs();
        let pins = &mut manifest["pins"];
        assert_eq!(
            json!(setup.pins.grid_shape),
            pins["shape"],
            "{name}: node counts"
        );
        assert!(
            same(setup.pins.sigma, pins["sigma_bohr"].as_f64().unwrap()),
            "{name}: ion width {} against {}",
            setup.pins.sigma,
            pins["sigma_bohr"]
        );
        assert!(
            same(
                setup.pins.filter_cutoff,
                pins["filter_wavevector"].as_f64().unwrap()
            ),
            "{name}: filter wave-vector"
        );
        assert_eq!(
            setup.pins.alias_operations as u64,
            pins["operations"].as_u64().unwrap(),
            "{name}: alias operations"
        );
        assert_eq!(
            setup.pins.dense_cutoff,
            pins["dense_cutoff"].as_f64().unwrap(),
            "{name}: dense cutoff"
        );
        pins["sigma_bohr"] = json!(setup.pins.sigma);
        pins["filter_wavevector"] = json!(setup.pins.filter_cutoff);
    }

    // The sidecar, checked against the snapshot's own block, and the snapshot solved as the tests do.
    manifest["setup"] = sidecar(&snapshot, &case);
    assert_eq!(
        setup_block(&manifest),
        snapshot["setup"],
        "the sidecar is the snapshot's setup block"
    );
    let mut joined = snapshot.clone();
    for (d, entry) in joined["sites"]["datasets"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .zip(manifest["setup"]["datasets"].as_array().unwrap())
    {
        d["file"] = json!(paw_root().join(entry["file"].as_str().unwrap()));
    }
    let leaked: &'static str = Box::leak(name.clone().into_boxed_str());
    let data = Data::build(leaked, manifest.clone(), joined);
    let result = data.solve();
    let energy = result.free_energy;
    let old_energy = manifest["free_energy_ha"].as_f64().unwrap();
    println!(
        "{name}: m {} p {} channels {} electrons {} | free energy {old_energy:.12} -> {energy:.12} Ha ({:+.3e}), {} iterations",
        data.crystal.m,
        data.crystal.p,
        data.crystal.channels,
        data.crystal.electrons,
        energy - old_energy,
        result.history.len()
    );
    manifest["free_energy_ha"] = json!(energy);

    let write_snapshot = || {
        if owner == name {
            std::fs::create_dir_all(&out).unwrap();
            std::fs::write(
                out.join(format!("{name}.snapshot.json")),
                serde_json::to_string(&snapshot).unwrap(),
            )
            .unwrap();
        }
    };
    if skip_oracle {
        write_snapshot();
        println!(
            "{name}: snapshot written, oracle skipped, {:.0} s",
            started.elapsed().as_secs_f64()
        );
        return;
    }

    // The oracle: central differences of fresh SCFs of the displaced or strained setups.
    let mut ledger =
        Ledger::open(PathBuf::from(&work).join(format!("{kind}_{name}.energies.jsonl")));
    let base = ledger.get("base", &|| {
        periodic_freeze_support::free_energy(&case, &setup, &options)
    });
    println!(
        "live setup {base:.12} Ha, the snapshot {energy:.12} Ha ({:+.3e})",
        base - energy
    );
    if strain {
        let volume = manifest["volume_bohr3"].as_f64().unwrap();
        strain_oracle(
            &mut manifest,
            &case,
            &setup,
            &options,
            &mut ledger,
            &|a, b| plan(a, b, volume),
            3. * PLAN_DIAGONAL / periodic_stress_crystals_support::TO_EV_PER_A3 * volume,
        );
    } else {
        gradient_oracle(&mut manifest, &case, &options, &mut ledger, PLAN_GATE);
    }
    manifest["stamp"] = json!({
        "date": date,
        "engine": "mandacaru-core Rust setup (periodic::setup::create_setup) and SCF",
        "generator": "crates/mandacaru-core/tests/periodic_freeze_snapshot.rs",
        "library": library,
        "platform": format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        "extra": {"case": name},
    });
    // The wall time and memory of the Python-era freeze say nothing about this one.
    let object = manifest.as_object_mut().unwrap();
    object.remove("seconds");
    object.remove("peak_rss_mb");
    write_snapshot();
    std::fs::create_dir_all(&out).unwrap();
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
