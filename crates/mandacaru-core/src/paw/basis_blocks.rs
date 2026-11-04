//! Native basis transformations and projector quadratures for PAW datasets.
//!
//! Evaluates:
//! - Raw vs dual projector block transformations ($B^{-1} D B^{-T}$, $B^{-1} q B^{-T}$)
//! - Multi-channel radial multipole transformations ($B_1^{-1} \Delta B_2^{-T}$)
//! - Exact atom-centered projector product quadratures (radial panels and unit sphere angular rule)
//! - Atomic multipole block assembly via Gaunt quadratures

use super::{
    invalid,
    local_split::{angular_rule, radial_rule},
    multipole::{gaunt, gaunt_grid, multipole_range},
};
use crate::Result;
use nalgebra::DMatrix;
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};

/// Transforms coupling and overlap blocks from the dual projector basis to the raw projector basis:
/// $D_{raw} = B^{-1} D B^{-T}$ and $q_{raw} = B^{-1} q B^{-T}$.
pub fn projector_basis_transform(
    vanderbilt: &[f64],
    coupling: &[f64],
    overlap: &[f64],
    n: usize,
    basis: &str,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if vanderbilt.len() != n * n || coupling.len() != n * n || overlap.len() != n * n {
        return Err(invalid("projector matrix dimensions mismatch"));
    }
    match basis {
        "dual" => Ok((coupling.to_vec(), overlap.to_vec())),
        "raw" => {
            let b = DMatrix::from_row_slice(n, n, vanderbilt);
            let b_inv = b.try_inverse().ok_or_else(|| invalid("singular Vanderbilt matrix"))?;
            let d = DMatrix::from_row_slice(n, n, coupling);
            let q = DMatrix::from_row_slice(n, n, overlap);
            let d_raw = &b_inv * d * b_inv.transpose();
            let q_raw = &b_inv * q * b_inv.transpose();
            Ok((d_raw.as_slice().to_vec(), q_raw.as_slice().to_vec()))
        }
        _ => Err(invalid(format!("unknown projector basis {basis:?}"))),
    }
}

/// Transforms radial multipole moment blocks between channels $l_1$ and $l_2$:
/// $\Delta_{raw} = B_1^{-1} \Delta B_2^{-T}$.
pub fn multipole_transform(
    b1: &[f64],
    n1: usize,
    b2: &[f64],
    n2: usize,
    delta: &[f64],
) -> Result<Vec<f64>> {
    if b1.len() != n1 * n1 || b2.len() != n2 * n2 || delta.len() != n1 * n2 {
        return Err(invalid("multipole transform matrix dimensions mismatch"));
    }
    let mat_b1 = DMatrix::from_row_slice(n1, n1, b1);
    let mat_b2 = DMatrix::from_row_slice(n2, n2, b2);
    let b1_inv = mat_b1.try_inverse().ok_or_else(|| invalid("singular Vanderbilt matrix B1"))?;
    let b2_inv = mat_b2.try_inverse().ok_or_else(|| invalid("singular Vanderbilt matrix B2"))?;
    let d = DMatrix::from_row_slice(n1, n2, delta);
    let res = &b1_inv * d * b2_inv.transpose();
    Ok(res.as_slice().to_vec())
}

/// Evaluates atom-centered spherical quadrature points and weights over the sphere
/// of radius `r_cut` around `center`.
pub fn projector_sphere(
    center: [f64; 3],
    r_cut: f64,
    panel: f64,
    radial_points: usize,
    polar_points: usize,
    azimuthal_points: usize,
) -> Result<([Vec<f64>; 3], Vec<f64>)> {
    let (radii, radial_weights) = radial_rule(r_cut, panel, radial_points)?;
    let (directions, angular_weights) = angular_rule(polar_points, azimuthal_points)?;
    let n_rad = radii.len();
    let n_ang = angular_weights.len();
    let n_total = n_rad * n_ang;

    let mut points = [
        Vec::with_capacity(n_total),
        Vec::with_capacity(n_total),
        Vec::with_capacity(n_total),
    ];
    let mut weights = Vec::with_capacity(n_total);

    for r_idx in 0..n_rad {
        let r = radii[r_idx];
        let rw = radial_weights[r_idx];
        for a_idx in 0..n_ang {
            for axis in 0..3 {
                points[axis].push(center[axis] + r * directions[axis][a_idx]);
            }
            weights.push(rw * angular_weights[a_idx]);
        }
    }
    Ok((points, weights))
}

/// One projector specification within an atom.
#[derive(Clone, Copy, Debug)]
pub struct ProjectorSpec {
    pub l: usize,
    pub m: i32,
    pub index: usize,
}

/// Evaluates all non-zero `(L, M)` multipole moment blocks for a set of atomic projectors.
///
/// Returns `Vec<(L, M, flattened_matrix)>` where `flattened_matrix` has length `n * n`.
pub fn atom_multipole_blocks(
    projectors: &[ProjectorSpec],
    real: bool,
    deltas: &BTreeMap<(usize, usize, usize), Vec<f64>>,
    delta_strides: &BTreeMap<(usize, usize), (usize, usize)>,
) -> Result<Vec<(usize, i32, Vec<Complex64>)>> {
    let n = projectors.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut levels = BTreeSet::new();
    for a in projectors {
        for b in projectors {
            for l in multipole_range(a.l, b.l) {
                levels.insert(l);
            }
        }
    }

    let grid = gaunt_grid()?;
    let mut result = Vec::new();

    for &l in &levels {
        let l_i32 = l as i32;
        for m in -l_i32..=l_i32 {
            let mut matrix = vec![Complex64::new(0., 0.); n * n];
            let mut any_nonzero = false;

            for (a_idx, pa) in projectors.iter().enumerate() {
                for (b_idx, pb) in projectors.iter().enumerate() {
                    let angular = gaunt([l, pa.l, pb.l], [m, pa.m, pb.m], real, &grid);
                    if angular.norm() < 1e-15 {
                        continue;
                    }

                    if let Some(delta) = deltas.get(&(pa.l, pb.l, l)) {
                        let &(_n_rows, n_cols) = delta_strides
                            .get(&(pa.l, pb.l))
                            .ok_or_else(|| invalid("missing stride for channel pair"))?;
                        let d_val = delta[pa.index * n_cols + pb.index];
                        let val = angular * d_val;
                        matrix[a_idx * n + b_idx] = val;
                        if val.norm() > 1e-14 {
                            any_nonzero = true;
                        }
                    }
                }
            }

            if any_nonzero {
                result.push((l, m, matrix));
            }
        }
    }

    Ok(result)
}
