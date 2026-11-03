//! CF-7 / AS-11 at the library entry: `periodic::response::{forces, stress}` build the Rust setup
//! from the crystal's own `setup` block and `sites` block, with no sidecar, and must give the
//! numbers the engine gave with the sidecar (the frozen Richardson oracles of
//! `freeze_periodic_gradients.py` and `freeze_periodic_strain.py`). A snapshot without the block
//! raises naming it; a block that disagrees with the snapshot is refused.
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_response_support;

use mandacaru_core::periodic::response::{forces, stress};
use periodic_gradient_crystals_support::gates::*;
use periodic_response_support::{solve_snapshot, ForceCase};
use serde_json::Value;

/// A frozen force case through the library entry: every frozen coordinate inside its gate.
fn check(case: &'static str) {
    let c = ForceCase::load(case);
    let (crystal, result) = solve_snapshot(&c.snapshot, &c.manifest);
    let start = std::time::Instant::now();
    let f = forces(&crystal, &result).unwrap();
    eprintln!("{case}: forces {:.1} s", start.elapsed().as_secs_f64());
    for entry in c.manifest["coordinates"].as_array().unwrap() {
        let (atom, axis) = (
            entry["atom"].as_u64().unwrap() as usize,
            entry["axis"].as_u64().unwrap() as usize,
        );
        let oracle = -entry["rr"].as_f64().unwrap();
        let floor = entry["floor_richardson"].as_f64().unwrap();
        let miss = (f[atom][axis] - oracle).abs();
        eprintln!(
            "{case} atom {atom} axis {axis}: force {:+.9e}, oracle {oracle:+.9e}, miss {miss:.2e} Ha/Bohr, gate {:.2e}",
            f[atom][axis],
            gate(floor)
        );
        assert!(
            miss < gate(floor),
            "{case} atom {atom} axis {axis}: {miss:e}"
        );
        assert!(
            oracle.abs() > 100. * gate(floor),
            "the oracle is not above its gate"
        );
    }
}

#[test]
fn lih_force_through_the_library() {
    check("lih");
}

#[test]
#[ignore = "112 s alone (displaced Si on a 2x2x2 mesh, one thread, load 15-20); run: MANDACARU_PAW_PATH=... cargo test --release -p mandacaru-core --test periodic_response -- --ignored"]
fn si_force_through_the_library() {
    check("si");
}

#[test]
fn a_snapshot_without_the_setup_block_raises_naming_it() {
    let c = ForceCase::load("lih");
    let mut old = c.snapshot.clone();
    old.as_object_mut().unwrap().remove("setup");
    let (crystal, result) = solve_snapshot(&old, &c.manifest);
    assert!(result.converged);
    for message in [
        forces(&crystal, &result).unwrap_err().to_string(),
        stress(&crystal, &result).unwrap_err().to_string(),
    ] {
        assert!(message.contains("'setup' block"), "{message}");
    }
}

/// A block the snapshot contradicts does not stop the energy: the crystal loads, solves, and the
/// force and the stress raise with the cause.
#[test]
fn an_inconsistent_block_is_refused_at_force_time_and_the_energy_stands() {
    let c = ForceCase::load("lih");
    let (crystal, result) = solve_snapshot(&c.snapshot, &c.manifest);
    forces(&crystal, &result).expect("the snapshot's own block is consistent");
    let refused = |edit: &dyn Fn(&mut Value)| -> String {
        let mut snapshot = c.snapshot.clone();
        edit(&mut snapshot["setup"]);
        let (crystal, result) = solve_snapshot(&snapshot, &c.manifest);
        assert!((result.free_energy - c.manifest["free_energy_ha"].as_f64().unwrap()).abs() < 1e-8);
        let message = forces(&crystal, &result).unwrap_err().to_string();
        assert_eq!(message, stress(&crystal, &result).unwrap_err().to_string());
        message
    };
    // One basis function too few, a table index past the list, a projector list of the wrong
    // size, a negative ion width and another version each raise naming the block.
    assert!(refused(&|b| {
        b["functions"]["atom"].as_array_mut().unwrap().pop();
    })
    .contains("basis functions"));
    assert!(refused(&|b| b["functions"]["table"][0] = 99.into()).contains("table"));
    assert!(refused(&|b| {
        b["projectors"].as_array_mut().unwrap().pop();
    })
    .contains("projectors"));
    assert!(refused(&|b| b["sigma_bohr"] = (-1.).into()).contains("ion width"));
    // Every angular momentum and index is bounded before a shell or a shape reads it.
    assert!(refused(&|b| b["tables"][0]["l"] = 9.into()).contains("l <= 8"));
    assert!(refused(&|b| b["projectors"][0] = serde_json::json!([0, 9, 0, 0])).contains("l <= 8"));
    assert!(
        refused(&|b| b["projectors"][0] = serde_json::json!([99, 0, 0, 0])).contains("outside the")
    );
    assert!(refused(&|b| {
        b["tables"][0]["values"].as_array_mut().unwrap().pop();
    })
    .contains("matching, strictly increasing"));
    assert!(refused(&|b| b["version"] = 2.into()).contains("version 1"));
    // A basis the setup could not describe says why.
    let message = refused(
        &|b| *b = serde_json::json!({"version": 1, "unavailable": "the basis function Gaussian has no radial table"}),
    );
    assert!(
        message.contains("Gaussian has no radial table"),
        "{message}"
    );
}

#[test]
fn a_wrong_projector_list_is_refused_at_force_time() {
    let c = ForceCase::load("lih");
    let mut snapshot = c.snapshot.clone();
    // Swap the radial indices of the first two projectors: the datasets give another order.
    let list = snapshot["setup"]["projectors"].as_array_mut().unwrap();
    let (a, b) = (list[0].clone(), list[1].clone());
    list[0] = b;
    list[1] = a;
    let (crystal, result) = solve_snapshot(&snapshot, &c.manifest);
    let message = forces(&crystal, &result).unwrap_err().to_string();
    assert!(message.contains("projector list"), "{message}");
}
