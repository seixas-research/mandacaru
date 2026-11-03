//! The gates of the real-crystal stress tests: the plan's numbers (`ANALYTIC_STRESS_PLAN.md`
//! section 9 rung 4: 3e-6 eV/A^3 on a diagonal entry, 5e-8 eV/A^3 on a shear), the rule of the
//! force freeze for an entry's own gate (about 100x the measured Richardson floor, never looser
//! than the plan's), and the bite of a wrong variant.
#![allow(dead_code)]

use super::{entry, TO_EV_PER_A3};
use serde_json::Value;

pub type Tensor = [[f64; 3]; 3];

/// The plan's proposed gates, eV/Angstrom^3 of the stress `g / Omega`.
pub const PLAN_DIAGONAL: f64 = 3e-6;
pub const PLAN_SHEAR: f64 = 5e-8;

/// The plan's gate on the derivative `g_ab` (Hartree) of a cell of `volume` Bohr^3.
pub fn plan(a: usize, b: usize, volume: f64) -> f64 {
    let sigma = if a == b { PLAN_DIAGONAL } else { PLAN_SHEAR };
    sigma / TO_EV_PER_A3 * volume
}

/// The gate of one entry (Hartree): 100x its Richardson floor, at least the noise of the free
/// energy over the smallest step (the SCF is converged to 1e-11 Ha), at most the plan's number.
pub fn gate(a: usize, b: usize, floor: f64, step: f64, volume: f64) -> f64 {
    (100. * floor.abs())
        .max(2e-11 / step)
        .min(plan(a, b, volume))
}

/// The gate of the isotropic series (the trace of the nine entries): the plan's diagonal gate on
/// each of the three diagonal entries.
pub fn trace_gate(floor: f64, step: f64, volume: f64) -> f64 {
    (100. * floor.abs())
        .max(2e-11 / step)
        .min(3. * plan(0, 0, volume))
}

/// One frozen entry with its gate.
pub struct Entry {
    pub oracle: f64,
    pub floor: f64,
    pub extrapolated: f64,
    pub gate: f64,
}

pub fn entries(manifest: &Value, volume: f64) -> [[Entry; 3]; 3] {
    let step = manifest["steps"][0].as_f64().unwrap();
    std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            let (oracle, floor, extrapolated) = entry(manifest, a, b);
            Entry {
                oracle,
                floor,
                extrapolated,
                gate: gate(a, b, floor, step, volume),
            }
        })
    })
}

/// Misses of `got` in units of each entry's gate (NaN is infinite), the worst first.
pub fn bites(got: &Tensor, want: &[[Entry; 3]; 3]) -> f64 {
    let mut worst = 0f64;
    for a in 0..3 {
        for b in 0..3 {
            let miss = (got[a][b] - want[a][b].oracle).abs() / want[a][b].gate;
            if miss.is_nan() {
                return f64::INFINITY;
            }
            worst = worst.max(miss);
        }
    }
    worst
}
