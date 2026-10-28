//! Symmetry-averaged reciprocal aliases of a discrete periodic grid.
use super::{fft::frequency, invalid, SpatialGrid};
use crate::{Buffer, Context, Result};
use nalgebra::{Matrix3, Vector3};
use std::{collections::HashSet, f64::consts::PI};

fn key(w: &Matrix3<f64>) -> [i64; 9] {
    std::array::from_fn(|k| w[(k / 3, k % 3)].round() as i64)
}
fn allowed(
    w: &Matrix3<f64>,
    cell: &Matrix3<f64>,
    inverse: &Matrix3<f64>,
    shape: [usize; 3],
) -> bool {
    if (w.determinant().abs() - 1.).abs() > 1e-9 {
        return false;
    }
    let r = cell * w * inverse;
    if (r.transpose() * r - Matrix3::identity()).norm() > 2e-2 {
        return false;
    }
    let v = w.try_inverse().unwrap().transpose();
    for i in 0..3 {
        for j in 0..3 {
            let nodes = w[(i, j)] * shape[i] as f64 / shape[j] as f64;
            let aliases = v[(i, j)] * shape[j] as f64 / shape[i] as f64;
            if (nodes - nodes.round()).abs() > 1e-9 || (aliases - aliases.round()).abs() > 1e-9 {
                return false;
            }
        }
    }
    true
}
pub fn norms(ctx: &Context, grid: &SpatialGrid) -> Result<Buffer<f64>> {
    let shape = grid.index.shape();
    let cell = grid.step * Matrix3::from_diagonal(&Vector3::from_fn(|i, _| shape[i] as f64));
    let inverse = cell
        .try_inverse()
        .ok_or_else(|| invalid("singular reciprocal cell"))?;
    let reciprocal = inverse.transpose() * (2. * PI);
    // Cubic grids have at most 48 operations; a generous bound also covers
    // mildly strained/skew cells while preventing an unbounded closure.
    let _storage = ctx.reserve(256 * 512)?;
    let mut group = Vec::new();
    let mut known = HashSet::new();
    for mut code in 0..3usize.pow(9) {
        let mut entries = [0.; 9];
        for value in &mut entries {
            *value = (code % 3) as f64 - 1.;
            code /= 3;
        }
        let w = Matrix3::from_row_slice(&entries);
        if allowed(&w, &cell, &inverse, shape) {
            known.insert(key(&w));
            group.push(w);
        }
    }
    loop {
        let n = group.len();
        for i in 0..n {
            for j in 0..n {
                let w = group[i] * group[j];
                if allowed(&w, &cell, &inverse, shape) && known.insert(key(&w)) {
                    if group.len() == 256 {
                        return Err(invalid(
                            "reciprocal symmetry closure exceeds 256 operations",
                        ));
                    }
                    group.push(w);
                }
            }
        }
        if group.len() == n {
            break;
        }
    }
    let operations: Vec<_> = group
        .iter()
        .map(|w| w.try_inverse().unwrap().transpose())
        .collect();
    let signed_permutations = operations
        .iter()
        .all(|v| (0..3).all(|i| (0..3).map(|j| v[(i, j)].abs()).sum::<f64>() == 1.));
    let mut out = ctx.allocate::<f64>(grid.index.points())?;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let indices = [i, j, k];
                let m = Vector3::from_fn(|a, _| frequency(indices[a], shape[a]));
                let norm = if signed_permutations {
                    (reciprocal * m).norm()
                } else {
                    let mut sum = 0.;
                    for (w, v) in group.iter().zip(&operations) {
                        let image = v * m;
                        let wrapped = Vector3::from_fn(|a, _| {
                            let n = shape[a] as f64;
                            (image[a] + (shape[a] / 2) as f64).rem_euclid(n) - (shape[a] / 2) as f64
                        });
                        // v = w^-T, so v^-1 = w^T, exactly for integer operations.
                        sum += (reciprocal * (w.transpose() * wrapped)).norm();
                    }
                    sum / group.len() as f64
                };
                out.as_mut_slice()[(i * shape[1] + j) * shape[2] + k] = norm;
            }
        }
    }
    Ok(out)
}
