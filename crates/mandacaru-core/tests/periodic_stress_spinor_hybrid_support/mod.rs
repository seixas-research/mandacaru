//! Shared by the `periodic_stress_spinor_hybrid*` tests (AS-10): the exchange strain of a SPINOR
//! hybrid crystal at fixed density matrices.
//!
//! The rig is that of AS-9 ([`periodic_stress_hybrid_support::Rig`]) on a scalar frozen hybrid
//! case rewritten as a spinor with a zero spin-orbit matrix (the exchange energy at fixed `D`
//! does not depend on `h^SO`: it is a function of the density matrix, the geometry and the
//! spheres), with hand-built `2m x 2m` density matrices whose four spin blocks are populated.
//!
//! The twin is flagged MAGNETIC (`time_reversal = false`) for the fixed-state gates: a hand-built
//! non-collinear `D` has a sphere magnetization, which the guard of a time-reversal-flagged
//! crystal refuses, and the magnetic flag is the one whose one-center operator `O` carries the
//! non-collinear remainder of SH-2, so the sphere term (H-6) is gated with the full operator.
#![allow(dead_code)]

use crate::periodic_gradient_crystals_support::{snapshot_of, Data};
use crate::periodic_gradient_spinor_support::manifest_of;
use crate::periodic_stress_hybrid_support::{density, Rig};
use mandacaru_core::{
    periodic::{
        screened_exchange::{ExxDivTreatment, Screening},
        Crystal,
    },
    quantum::C64,
};
use nalgebra::DMatrix;
use serde_json::json;

/// The scalar hybrid case `scalar` as a spinor crystal with a zero spin-orbit matrix at every
/// k-point; `time_reversal` is its flag.
pub fn twin(scalar: &'static str, time_reversal: bool) -> (Data, serde_json::Value) {
    let manifest = manifest_of(scalar);
    let mut snapshot = snapshot_of(scalar, &manifest);
    let (m, n) = (
        snapshot["m"].as_u64().unwrap() as usize,
        snapshot["initial"].as_array().unwrap().len(),
    );
    snapshot["spinor"] = json!(true);
    snapshot["time_reversal"] = json!(time_reversal);
    snapshot["initial_m"] = json!(vec![0.0; 3 * n]);
    for point in snapshot["kpoints"].as_array_mut().unwrap() {
        point["spin_orbit"] = json!(vec![[0.0, 0.0]; 4 * m * m]);
    }
    (Data::build(scalar, manifest, snapshot.clone()), snapshot)
}

/// The block `(s, t)` of a `2m x 2m` matrix.
pub fn block(d: &DMatrix<C64>, s: usize, t: usize) -> DMatrix<C64> {
    let m = d.nrows() / 2;
    d.view((s * m, t * m), (m, m)).into_owned()
}

/// `blockdiag(D_uu, D_dd)`: the density without its cross-spin blocks.
pub fn without_cross_blocks(d: &DMatrix<C64>) -> DMatrix<C64> {
    let m = d.nrows() / 2;
    let mut out = DMatrix::<C64>::zeros(2 * m, 2 * m);
    for s in 0..2 {
        out.view_mut((s * m, s * m), (m, m))
            .copy_from(&d.view((s * m, s * m), (m, m)));
    }
    out
}

/// `(u (x) 1_m) D (u (x) 1_m)^+`: the spin rotation by the 2x2 matrix `u` (row-major).
pub fn rotated(d: &DMatrix<C64>, u: [[C64; 2]; 2]) -> DMatrix<C64> {
    let m = d.nrows() / 2;
    let big = DMatrix::<C64>::from_fn(2 * m, 2 * m, |i, j| {
        if i % m == j % m {
            u[i / m][j / m]
        } else {
            C64::default()
        }
    });
    &big * d * big.adjoint()
}

/// The SU(2) matrix of the rotation by `angle` about the unit vector `axis`.
pub fn su2(angle: f64, axis: [f64; 3]) -> [[C64; 2]; 2] {
    let norm = axis.iter().map(|x| x * x).sum::<f64>().sqrt();
    let (n, c, s) = (
        axis.map(|x| x / norm),
        (angle / 2.).cos(),
        (angle / 2.).sin(),
    );
    [
        [C64::new(c, -s * n[2]), C64::new(-s * n[1], -s * n[0])],
        [C64::new(s * n[1], -s * n[0]), C64::new(c, s * n[2])],
    ]
}

/// The transpose of an SU(2) matrix (not a rotation: the wrong spin transformation `u^T`).
pub fn transposed(u: [[C64; 2]; 2]) -> [[C64; 2]; 2] {
    [[u[0][0], u[1][0]], [u[0][1], u[1][1]]]
}

/// The rig of `scalar` rewritten as a spinor flagged `time_reversal`, with a `2m x 2m` density
/// matrix per k-point, rank four, every spin block populated: the cross blocks are asserted above
/// one percent of the matrix.
pub fn rig(scalar: &'static str, exxdiv: ExxDivTreatment, time_reversal: bool) -> Rig {
    let (data, snapshot) = twin(scalar, time_reversal);
    with_densities(data, snapshot, exxdiv)
}

/// The rig of a spinor `data` (built from `snapshot`) with hand-built densities.
pub fn with_densities(data: Data, snapshot: serde_json::Value, exxdiv: ExxDivTreatment) -> Rig {
    assert!(data.crystal.spinor);
    let work = Crystal::from_json(&data.crystal.context, &snapshot.to_string()).unwrap();
    let m2 = 2 * data.crystal.m;
    let d: Vec<DMatrix<C64>> = (0..data.crystal.points.len())
        .map(|k| density(m2, k))
        .collect();
    for matrix in &d {
        let cross = block(matrix, 0, 1).norm();
        assert!(
            cross > 1e-2 * matrix.norm(),
            "the cross-spin blocks must be populated: {cross:e} of {:e}",
            matrix.norm()
        );
    }
    let screening = Screening::of(&data.crystal).unwrap();
    Rig {
        data,
        work,
        d,
        screening,
        exxdiv,
    }
}

/// `V D V^+` with `V = blockdiag over mu of u_mu`: a spin rotation whose angle depends on the
/// spatial function (`angle (mu + 1)`), which is NOT a symmetry of the exchange (the wrong
/// transformation the SU(2) covariance must tell from the uniform rotation).
pub fn rotated_by_function(d: &DMatrix<C64>, angle: f64, axis: [f64; 3]) -> DMatrix<C64> {
    let m = d.nrows() / 2;
    let us: Vec<[[C64; 2]; 2]> = (0..m)
        .map(|mu| su2(angle * (mu + 1) as f64, axis))
        .collect();
    let big = DMatrix::<C64>::from_fn(2 * m, 2 * m, |i, j| {
        if i % m == j % m {
            us[i % m][i / m][j / m]
        } else {
            C64::default()
        }
    });
    &big * d * big.adjoint()
}
