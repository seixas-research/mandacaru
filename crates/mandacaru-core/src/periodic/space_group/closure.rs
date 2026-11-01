//! Group closure verification on cosets and tolerance shrink loop.
use super::search::{find_cosets, CosetDecomposition};
use crate::periodic::invalid;
use crate::Result;
use nalgebra::Matrix3;

/// Check group closure of pure translations and coset representatives.
pub fn is_group_closed(cosets: &CosetDecomposition, b: &Matrix3<f64>, tol: f64) -> bool {
    let pure = &cosets.pure_translations;
    let reps = &cosets.coset_representatives;
    let two_tol = 2.0 * tol + 1e-10;

    // 1. Pure translation subgroup closure
    for t1 in pure {
        for t2 in pure {
            let t = [
                (t1.translation[0] + t2.translation[0]).rem_euclid(1.0),
                (t1.translation[1] + t2.translation[1]).rem_euclid(1.0),
                (t1.translation[2] + t2.translation[2]).rem_euclid(1.0),
            ];
            let mut matched = false;
            for tau in pure {
                let dx = t[0] - tau.translation[0];
                let dy = t[1] - tau.translation[1];
                let dz = t[2] - tau.translation[2];
                let d = [dx - dx.round(), dy - dy.round(), dz - dz.round()];
                let vx = d[0] * b[(0, 0)] + d[1] * b[(1, 0)] + d[2] * b[(2, 0)];
                let vy = d[0] * b[(0, 1)] + d[1] * b[(1, 1)] + d[2] * b[(2, 1)];
                let vz = d[0] * b[(0, 2)] + d[1] * b[(1, 2)] + d[2] * b[(2, 2)];
                if (vx * vx + vy * vy + vz * vz).sqrt() <= two_tol {
                    matched = true;
                    break;
                }
            }
            if !matched {
                return false;
            }
        }
    }

    // 2. Coset representative multiplication closure
    for rep_i in reps {
        for rep_j in reps {
            let w = rep_i.rotation * rep_j.rotation;
            let t_prod = [
                rep_i.rotation[(0, 0)] as f64 * rep_j.translation[0]
                    + rep_i.rotation[(0, 1)] as f64 * rep_j.translation[1]
                    + rep_i.rotation[(0, 2)] as f64 * rep_j.translation[2]
                    + rep_i.translation[0],
                rep_i.rotation[(1, 0)] as f64 * rep_j.translation[0]
                    + rep_i.rotation[(1, 1)] as f64 * rep_j.translation[1]
                    + rep_i.rotation[(1, 2)] as f64 * rep_j.translation[2]
                    + rep_i.translation[1],
                rep_i.rotation[(2, 0)] as f64 * rep_j.translation[0]
                    + rep_i.rotation[(2, 1)] as f64 * rep_j.translation[1]
                    + rep_i.rotation[(2, 2)] as f64 * rep_j.translation[2]
                    + rep_i.translation[2],
            ];

            let rep_k = match reps.iter().find(|r| r.rotation == w) {
                Some(r) => r,
                None => return false,
            };

            // t_prod - t_{rep_k} must belong to pure translations T
            let delta_t = [
                (t_prod[0] - rep_k.translation[0]).rem_euclid(1.0),
                (t_prod[1] - rep_k.translation[1]).rem_euclid(1.0),
                (t_prod[2] - rep_k.translation[2]).rem_euclid(1.0),
            ];

            let mut in_pure = false;
            for tau in pure {
                let dx = delta_t[0] - tau.translation[0];
                let dy = delta_t[1] - tau.translation[1];
                let dz = delta_t[2] - tau.translation[2];
                let d = [dx - dx.round(), dy - dy.round(), dz - dz.round()];
                let vx = d[0] * b[(0, 0)] + d[1] * b[(1, 0)] + d[2] * b[(2, 0)];
                let vy = d[0] * b[(0, 1)] + d[1] * b[(1, 1)] + d[2] * b[(2, 1)];
                let vz = d[0] * b[(0, 2)] + d[1] * b[(1, 2)] + d[2] * b[(2, 2)];
                if (vx * vx + vy * vy + vz * vz).sqrt() <= two_tol {
                    in_pure = true;
                    break;
                }
            }
            if !in_pure {
                return false;
            }
        }
    }

    true
}

/// Find closed coset decomposition, shrinking tolerance up to 20 times if necessary.
pub fn find_closed_cosets(
    xb: &[[f64; 3]],
    labels: &[u32],
    b: &Matrix3<f64>,
    candidate_rotations: &[Matrix3<i32>],
    initial_tol: f64,
) -> Result<(CosetDecomposition, f64)> {
    let mut tol = initial_tol;
    let max_tries = 20;

    for _ in 0..max_tries {
        if let Ok(cosets) = find_cosets(xb, labels, b, candidate_rotations, tol) {
            if is_group_closed(&cosets, b, tol) {
                return Ok((cosets, tol));
            }
        }
        tol *= 0.95;
    }

    Err(invalid("spacegroup search failed"))
}
