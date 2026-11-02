//! CS-3 (Kramers / time reversal): the force of a time-reversal-symmetric spinor on the mesh
//! HALVED by time reversal equals the force on the whole mesh, with no symmetrizer.
//!
//! The cell is the displaced `lda-fr` Si of `si_ldafr`, on a 3x1x1 Gamma-centered mesh, where `+k`
//! and `-k` differ (0 and +-1/3): the natural build keeps two points (weights 1/3 and 2/3) and a
//! nontrivial moment symmetrization, `si_ldafr_tr3_full` keeps all three. Both are solved and
//! differentiated; each is compared with its own frozen reconverged oracle, and with the other.
//! Eigenvalues are even in `k`, and Kramers-paired at the time-reversal-invariant point.
//!
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;

use mandacaru_core::periodic::gradient::gradient;
use periodic_gradient_crystals_support::gates::*;
use periodic_gradient_spinor_support::{oracle, Spinor};

struct Mesh {
    spinor: Spinor,
    forces: Vec<[f64; 3]>,
    free_energy: f64,
    eigenvalues: Vec<Vec<f64>>,
}

fn run(case: &'static str) -> Mesh {
    let spinor = Spinor::load(case);
    let result = spinor.data.solve();
    let cores = spinor.data.cores();
    let provider = spinor.provider(&cores);
    let forces = gradient(&provider, &spinor.data.state(&result), None)
        .unwrap()
        .forces();
    Mesh {
        forces,
        free_energy: result.free_energy,
        eigenvalues: result.eigenvalues.clone(),
        spinor,
    }
}

fn against_the_oracle(mesh: &Mesh, label: &str) {
    oracle::against_the_oracle(&mesh.spinor.data, mesh.free_energy, &mesh.forces, label);
}

#[test]
fn the_half_mesh_force_is_the_whole_mesh_force() {
    let half = run("si_ldafr_tr3");
    let full = run("si_ldafr_tr3_full");
    let (nh, nf) = (
        half.spinor.data.crystal.points.len(),
        full.spinor.data.crystal.points.len(),
    );
    eprintln!("k-points: half {nh}, whole {nf}");
    assert_eq!(
        (nh, nf),
        (2, 3),
        "time reversal halves 0, +1/3, -1/3 to 0, 1/3"
    );
    // The guard of the comparison: +k and -k are different points of the whole mesh, else halving
    // it proves nothing.
    let ks: Vec<[f64; 3]> = full
        .spinor
        .data
        .crystal
        .points
        .iter()
        .map(|p| p.k)
        .collect();
    let opposite = ks.iter().any(|a| {
        ks.iter().any(|b| {
            let norm = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            norm > 1e-6 && (0..3).all(|i| (a[i] + b[i]).abs() < 1e-9)
        })
    });
    assert!(opposite, "the whole mesh holds a pair +k, -k");
    // The halved build keeps the identity moment symmetrization (so the engine needs no hook for
    // it) and the weights 1/3 and 2/3: the half mesh stands for the whole one by weights alone.
    let weights: Vec<f64> = half
        .spinor
        .data
        .crystal
        .points
        .iter()
        .map(|p| p.weight)
        .collect();
    eprintln!("half-mesh weights {weights:?}");
    assert!(weights.iter().any(|w| (w - 1. / 3.).abs() < 1e-12));
    assert!(weights.iter().any(|w| (w - 2. / 3.).abs() < 1e-12));
    against_the_oracle(&full, "whole mesh");
    against_the_oracle(&half, "half mesh");
    let gap = half
        .forces
        .iter()
        .zip(&full.forces)
        .flat_map(|(a, b)| (0..3).map(move |i| (a[i] - b[i]).abs()))
        .fold(0f64, f64::max);
    let energy_gap = (half.free_energy - full.free_energy).abs();
    eprintln!("half against whole: free energy {energy_gap:.2e} Ha, force {gap:.2e} Ha/Bohr");
    assert!(
        energy_gap < 1e-9 && gap < PLAN_GATE,
        "{energy_gap:e} {gap:e}"
    );
}

#[test]
fn eigenvalues_are_even_in_k_and_paired_at_the_time_reversal_invariant_point() {
    let full = run("si_ldafr_tr3_full");
    let ks: Vec<[f64; 3]> = full
        .spinor
        .data
        .crystal
        .points
        .iter()
        .map(|p| p.k)
        .collect();
    // Kramers degeneracy needs a time-reversal-invariant k (here Gamma): at +-1/3 the displaced
    // cell has no inversion, so the pairs split by the spin-orbit coupling (reported, nonzero).
    let norm = |k: &[f64; 3]| (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
    let pair_gap = |e: &[f64]| {
        e.chunks(2)
            .map(|p| (p[0] - p[1]).abs())
            .fold(0f64, f64::max)
    };
    let mut trim = 0;
    let mut split = 0f64;
    for (k, e) in ks.iter().zip(&full.eigenvalues) {
        if norm(k) < 1e-9 {
            trim += 1;
            eprintln!("Gamma: Kramers pair gap {:.2e} Ha", pair_gap(e));
            assert!(pair_gap(e) < 1e-9);
        } else {
            split = split.max(pair_gap(e));
        }
    }
    eprintln!("away from Gamma the pairs split by up to {split:.2e} Ha (no inversion)");
    assert_eq!(trim, 1);
    assert!(
        split > 1e-9,
        "the guard: off Gamma the pairs are not degenerate"
    );
    // E_n(+k) = E_n(-k) on the whole mesh.
    let mut even_gap = 0f64;
    let mut pairs = 0;
    for (i, a) in ks.iter().enumerate() {
        for (j, b) in ks.iter().enumerate().skip(i + 1) {
            if (0..3).all(|x| (a[x] + b[x]).abs() < 1e-9) {
                pairs += 1;
                for (u, v) in full.eigenvalues[i].iter().zip(&full.eigenvalues[j]) {
                    even_gap = even_gap.max((u - v).abs());
                }
            }
        }
    }
    eprintln!("E(k) - E(-k): {even_gap:.2e} Ha over {pairs} pair(s)");
    assert_eq!(pairs, 1);
    assert!(even_gap < 1e-9);
}
