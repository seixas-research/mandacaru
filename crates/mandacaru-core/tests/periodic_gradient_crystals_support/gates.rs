//! The gates of the crystal force tests: the plan's number, the 14.4 rule for a coordinate's gate
//! (about 100x the measured floor, never looser than the plan's), and the bite of a variant.
#![allow(dead_code)]

use super::{root, Data};
use serde_json::Value;

/// Hartree/Bohr in eV/Angstrom.
pub const TO_EV_PER_A: f64 = 51.4220674763;
/// The plan's proposed crystal force gate, 1e-5 eV/Angstrom, in Hartree/Bohr.
pub const PLAN_GATE: f64 = 1e-5 / TO_EV_PER_A;

/// The gate of one coordinate from its measured Richardson floor: 100x the floor, clamped between
/// the noise of the oracle (1e-9, SCF 1e-11 Ha over a 0.0038 Bohr step) and the plan's number.
pub fn gate(floor: f64) -> f64 {
    (100. * floor.abs()).clamp(1e-9, PLAN_GATE)
}

/// The worst miss of `forces` over the frozen coordinates, in units of each coordinate's gate.
pub fn bite(data: &Data, forces: &[[f64; 3]]) -> f64 {
    data.manifest["coordinates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let (atom, axis) = (
                e["atom"].as_u64().unwrap() as usize,
                e["axis"].as_u64().unwrap() as usize,
            );
            (forces[atom][axis] + e["rr"].as_f64().unwrap()).abs()
                / gate(e["floor_richardson"].as_f64().unwrap())
        })
        .fold(0., f64::max)
}

/// The manifest of the SECOND step family of `case` (`<case>_b.json`, the same snapshot, other
/// steps): 14.4 asks for one before a gate near the floor is declared passed or failed, because a
/// floor measured with one family can be an underestimate of the oracle's own uncertainty.
pub fn second_family(case: &str) -> Value {
    let path = root().join(format!("{case}_b.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let other: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        other["snapshot"].as_str(),
        Some(case),
        "the second family must share the snapshot of {case}"
    );
    other
}

fn coordinate<'a>(manifest: &'a Value, atom: u64, axis: u64) -> &'a Value {
    manifest["coordinates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["atom"].as_u64() == Some(atom) && e["axis"].as_u64() == Some(axis))
        .unwrap_or_else(|| panic!("a missing frozen coordinate ({atom}, {axis})"))
}

/// The oracle's uncertainty of one coordinate and its gate. With two step families the
/// uncertainty of the extrapolated derivative is the largest of the extrapolation corrections
/// `RR - R(2,4)` of both families and of their disagreement `|RR_A - RR_B|` (all measured, none
/// assumed); the correction of a Richardson series that converges is the error of its LOWER order,
/// so it bounds the error of `RR` (measured: `R(2,4) - R(4,8)` overstated the uncertainty of `RR`
/// 10 to 15 times on these cells). Without the second family the CF-4 rule stands: the
/// Richardson floor `R(2,4) - R(4,8)` itself. The gate is 100x the uncertainty, between the noise
/// of the oracle and the plan's number.
pub fn family_gate(a: &Value, b: Option<&Value>) -> (f64, f64) {
    let floor = match b {
        None => a["floor_richardson"].as_f64().unwrap().abs(),
        Some(b) => a["floor_extrapolated"]
            .as_f64()
            .unwrap()
            .abs()
            .max(b["floor_extrapolated"].as_f64().unwrap().abs())
            .max((a["rr"].as_f64().unwrap() - b["rr"].as_f64().unwrap()).abs()),
    };
    (gate(floor), floor)
}

/// Every frozen coordinate of the manifest against `forces`: the miss is inside the coordinate's
/// gate AND the oracle itself is far above that gate (the component is not vacuous). Reports the
/// force, the oracle, the miss and the floors of each, and returns the worst miss in units of the
/// gate. With the second step family the two oracles must also agree inside the gate.
pub fn compare(case: &str, data: &Data, other: Option<&Value>, forces: &[[f64; 3]]) -> f64 {
    let mut worst = 0f64;
    for entry in data.manifest["coordinates"].as_array().unwrap() {
        let (atom, axis) = (
            entry["atom"].as_u64().unwrap(),
            entry["axis"].as_u64().unwrap(),
        );
        let second = other.map(|o| coordinate(o, atom, axis));
        let (gate, floor) = family_gate(entry, second);
        let oracle = -entry["rr"].as_f64().unwrap();
        let miss = (forces[atom as usize][axis as usize] - oracle).abs();
        let spread = second.map_or(0., |b| {
            (entry["rr"].as_f64().unwrap() - b["rr"].as_f64().unwrap()).abs()
        });
        eprintln!(
            "{case} atom {atom} axis {axis}: force {:+.9e}, oracle {oracle:+.9e}, miss {miss:.2e} Ha/Bohr = {:.2e} eV/A, floor {floor:.1e} (families differ by {spread:.1e}), gate {gate:.2e} ({:.1}x the floor)",
            forces[atom as usize][axis as usize],
            miss * TO_EV_PER_A,
            gate / floor.max(1e-300)
        );
        // A floor above the plan's gate would make the gate unreachable: the freeze must not
        // contain such a coordinate.
        assert!(
            floor <= PLAN_GATE,
            "{case} atom {atom} axis {axis}: the oracle floor {floor:e} is above the plan gate"
        );
        assert!(
            miss < gate,
            "{case} atom {atom} axis {axis}: {miss:e} against {gate:e}"
        );
        assert!(
            spread < gate,
            "{case} atom {atom} axis {axis}: the step families differ by {spread:e}"
        );
        assert!(
            oracle.abs() > 100. * gate,
            "{case} atom {atom} axis {axis}: the oracle {oracle:e} is not above its gate"
        );
        worst = worst.max(miss / gate);
    }
    worst
}

/// The worst miss of `forces` over the frozen coordinates in units of each one's family gate
/// (the same gates as [`compare`]); NaN propagates, so a NaN force fails every comparison.
pub fn bite_families(data: &Data, other: Option<&Value>, forces: &[[f64; 3]]) -> f64 {
    data.manifest["coordinates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let (atom, axis) = (e["atom"].as_u64().unwrap(), e["axis"].as_u64().unwrap());
            let (gate, _) = family_gate(e, other.map(|o| coordinate(o, atom, axis)));
            let miss = (forces[atom as usize][axis as usize] + e["rr"].as_f64().unwrap()).abs();
            if miss.is_nan() {
                f64::NAN
            } else {
                miss / gate
            }
        })
        .fold(0., |m, x| {
            if m.is_nan() || x.is_nan() {
                f64::NAN
            } else {
                m.max(x)
            }
        })
}

/// The largest entry of `values` (a per-atom, per-axis quantity such as the force a term adds) over
/// the frozen coordinates, in units of each coordinate's family gate; NaN propagates.
pub fn in_gates(data: &Data, other: Option<&Value>, values: &[[f64; 3]]) -> f64 {
    data.manifest["coordinates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let (atom, axis) = (e["atom"].as_u64().unwrap(), e["axis"].as_u64().unwrap());
            let (gate, _) = family_gate(e, other.map(|o| coordinate(o, atom, axis)));
            values[atom as usize][axis as usize].abs() / gate
        })
        .fold(0., |m, x| {
            if m.is_nan() || x.is_nan() {
                f64::NAN
            } else {
                m.max(x)
            }
        })
}

/// The largest absolute sum over the atoms of one term, over the three axes.
pub fn net(term: &[[f64; 3]]) -> f64 {
    (0..3)
        .map(|a| term.iter().map(|g| g[a]).sum::<f64>())
        .fold(0f64, |m, s| m.max(s.abs()))
}
