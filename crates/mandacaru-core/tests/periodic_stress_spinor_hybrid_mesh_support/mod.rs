//! Shared by `periodic_stress_spinor_hybrid_mesh` (AS-10): a spinor strain-freeze case on its own
//! k mesh as a fixed-state rig.
#![allow(dead_code)]

use crate::periodic_gradient_crystals_support::{paw_root, Data};
use crate::periodic_stress_crystals_support::{manifest, read};
use crate::periodic_stress_hybrid_support::Rig;
use crate::periodic_stress_spinor_hybrid_support::with_densities;
use mandacaru_core::periodic::screened_exchange::ExxDivTreatment;
use serde_json::json;

/// The rig of a spinor strain-freeze case of `reference/periodic_strain` (a real snapshot with its
/// own spin-orbit matrices, which the fixed-state exchange does not read) flagged
/// `time_reversal`, with hand-built `2m x 2m` densities as in [`rig`]: the exchange strain on the
/// k mesh of the case (`si_hsefr_tr3`: the points 0 and +-1/3, complex `k_diff`).
pub fn mesh_rig(case: &'static str, exxdiv: ExxDivTreatment, time_reversal: bool) -> Rig {
    let manifest = manifest(case);
    let mut snapshot = read(&format!("{case}.snapshot.json"));
    for (atom, d) in snapshot["sites"]["datasets"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        let relative = manifest["setup"]["datasets"][atom]["file"]
            .as_str()
            .unwrap();
        d["file"] = json!(paw_root().join(relative).to_str().unwrap());
    }
    let n = snapshot["initial"].as_array().unwrap().len();
    snapshot["time_reversal"] = json!(time_reversal);
    snapshot["initial_m"] = json!(vec![0.0; 3 * n]);
    let data = Data::build(case, manifest, snapshot.clone());
    with_densities(data, snapshot, exxdiv)
}
