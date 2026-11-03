//! Shared by `periodic_stress_spinor_hybrid_magnetic` (AS-10): the spin-polarized strain-freeze case
//! as a magnetic spinor crystal.
#![allow(dead_code)]

use crate::periodic_gradient_crystals_support::{paw_root, Data};
use crate::periodic_gradient_spinor_support::{union_list, Spinor};
use crate::periodic_stress_crystals_support::{manifest, read};
use mandacaru_core::paw::record::PawDataset;
use serde_json::json;

/// The strain-freeze case `scalar` (a SPIN-POLARIZED snapshot of `reference/periodic_strain`) as
/// a MAGNETIC spinor crystal with a zero spin-orbit matrix whose initial moment points along
/// `direction`. The discretized energy of a zero coupling is invariant under a global spin
/// rotation, so every converged state is the unrestricted one turned to `direction` and the
/// unrestricted free-energy differences of the manifest are the oracle of every direction.
pub fn magnetic_twin(scalar: &'static str, direction: [f64; 3]) -> Spinor {
    let manifest = manifest(scalar);
    let mut snapshot = read(&format!("{scalar}.snapshot.json"));
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
    assert_eq!(
        snapshot["spins"].as_u64(),
        Some(2),
        "{scalar} is spin-polarized"
    );
    let (m, n) = (
        snapshot["m"].as_u64().unwrap() as usize,
        snapshot["initial"].as_array().unwrap().len(),
    );
    let scalar_m: Vec<f64> = snapshot["initial_m"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
    snapshot["spinor"] = json!(true);
    snapshot["time_reversal"] = json!(false);
    snapshot["spins"] = json!(1);
    snapshot["initial_m"] = json!(direction
        .iter()
        .flat_map(|d| scalar_m.iter().map(move |v| v * d / norm))
        .collect::<Vec<f64>>());
    assert_eq!(snapshot["initial_m"].as_array().unwrap().len(), 3 * n);
    for point in snapshot["kpoints"].as_array_mut().unwrap() {
        point["spin_orbit"] = json!(vec![[0.0, 0.0]; 4 * m * m]);
    }
    let data = Data::build(scalar, manifest, snapshot);
    let ctx = &data.crystal.context;
    // The `lda-fr` Si record is a stand-in for the union projector list on every center (any list
    // gives `B = 0` at zero coupling); its coupling is zeroed.
    let library = PawDataset::read(ctx, &paw_root().join("lda-fr/Si.parquet")).unwrap();
    let refs: Vec<&PawDataset> = data.atoms.iter().map(|_| &library).collect();
    let (projectors, mut couplings) = union_list(ctx, &refs, &data.centers());
    for c in couplings.values_mut() {
        c.values.iter_mut().for_each(|v| *v = 0.0);
    }
    Spinor {
        data,
        projectors,
        couplings,
    }
}
