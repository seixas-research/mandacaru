//! Lattice reduction and automorphism search via Selling reduction and shell matching.
use super::invalid;
use crate::Result;
use nalgebra::{Matrix3, SVD, Vector3};

/// Pre-check and validation parameters for a lattice basis.
pub fn validate_cell(cell: &Matrix3<f64>, tol: f64) -> Result<()> {
    if !cell.iter().all(|x| x.is_finite()) {
        return Err(invalid("cell components must be finite numbers"));
    }
    let det = cell.determinant();
    if det.abs() < 1e-14 {
        return Err(invalid(
            "space-group search failed: singular or degenerate lattice cell",
        ));
    }
    let max_len = (0..3)
        .map(|i| cell.row(i).norm())
        .fold(0.0f64, |a, b| a.max(b));
    if !tol.is_finite() || tol <= 0.0 {
        return Err(invalid("tolerance must be positive and finite"));
    }
    if tol < max_len * 1e-12 {
        return Err(invalid("tolerance is smaller than 1e-12 x lattice scale"));
    }
    Ok(())
}

/// Exact integer adjugate of a 3x3 integer matrix.
pub fn adjugate3(m: &Matrix3<i32>) -> Matrix3<i32> {
    let c00 = m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)];
    let c01 = -(m[(1, 0)] * m[(2, 2)] - m[(1, 2)] * m[(2, 0)]);
    let c02 = m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)];

    let c10 = -(m[(0, 1)] * m[(2, 2)] - m[(0, 2)] * m[(2, 1)]);
    let c11 = m[(0, 0)] * m[(2, 2)] - m[(0, 2)] * m[(2, 0)];
    let c12 = -(m[(0, 0)] * m[(2, 1)] - m[(0, 1)] * m[(2, 0)]);

    let c20 = m[(0, 1)] * m[(1, 2)] - m[(0, 2)] * m[(1, 1)];
    let c21 = -(m[(0, 0)] * m[(1, 2)] - m[(0, 2)] * m[(1, 0)]);
    let c22 = m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)];

    Matrix3::new(c00, c10, c20, c01, c11, c21, c02, c12, c22)
}

/// Exact determinant of a 3x3 integer matrix.
pub fn det3(m: &Matrix3<i32>) -> i32 {
    m[(0, 0)] * (m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)])
        - m[(0, 1)] * (m[(1, 0)] * m[(2, 2)] - m[(1, 2)] * m[(2, 0)])
        + m[(0, 2)] * (m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)])
}

/// Inverse of a unimodular 3x3 integer matrix (det must be +/-1).
pub fn unimodular_inverse(m: &Matrix3<i32>) -> Result<Matrix3<i32>> {
    let d = det3(m);
    if d == 1 {
        Ok(adjugate3(m))
    } else if d == -1 {
        Ok(-adjugate3(m))
    } else {
        Err(invalid(format!(
            "matrix is not integer unimodular (det = {})",
            d
        )))
    }
}

/// Selling / Delaunay reduction on the superbase (a1, a2, a3, -sum(a_i)).
/// Returns reduced basis B = U * A (rows) and unimodular transform U (det = +1).
pub fn selling_reduce(cell: &Matrix3<f64>) -> Result<(Matrix3<f64>, Matrix3<i32>)> {
    validate_cell(cell, 1e-6)?;

    // Superbase S as 4 rows in Z^3
    let mut s: [[i32; 3]; 4] = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, -1, -1],
    ];

    let eps = 1e-10;
    let maxit = 1000;
    let mut converged = false;

    for _ in 0..maxit {
        // Compute V = S * cell (4 rows)
        let mut v = [[0.0f64; 3]; 4];
        for i in 0..4 {
            for j in 0..3 {
                v[i][j] = s[i][0] as f64 * cell[(0, j)]
                    + s[i][1] as f64 * cell[(1, j)]
                    + s[i][2] as f64 * cell[(2, j)];
            }
        }

        let mut done = true;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let dot = v[i][0] * v[j][0] + v[i][1] * v[j][1] + v[i][2] * v[j][2];
                let norm_i = (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]).sqrt();
                let norm_j = (v[j][0] * v[j][0] + v[j][1] * v[j][1] + v[j][2] * v[j][2]).sqrt();
                if dot > eps * norm_i * norm_j {
                    // Find other two indices
                    let mut other = [0usize; 2];
                    let mut count = 0;
                    for m in 0..4 {
                        if m != i && m != j {
                            other[count] = m;
                            count += 1;
                        }
                    }
                    let k = other[0];
                    let l = other[1];

                    for c in 0..3 {
                        s[k][c] += s[i][c];
                        s[l][c] += s[i][c];
                        s[i][c] = -s[i][c];
                    }
                    done = false;
                    break;
                }
            }
            if !done {
                break;
            }
        }
        if done {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(invalid("Selling reduction failed to converge"));
    }

    // Compute final V norms and sort
    let mut v = [[0.0f64; 3]; 4];
    let mut lengths = [0.0f64; 4];
    for i in 0..4 {
        for j in 0..3 {
            v[i][j] = s[i][0] as f64 * cell[(0, j)]
                + s[i][1] as f64 * cell[(1, j)]
                + s[i][2] as f64 * cell[(2, j)];
        }
        lengths[i] = (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]).sqrt();
    }

    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|&a, &b| lengths[a].total_cmp(&lengths[b]));

    // Try trios of vectors in order of increasing length
    let trios = [
        [order[0], order[1], order[2]],
        [order[0], order[1], order[3]],
        [order[0], order[2], order[3]],
        [order[1], order[2], order[3]],
    ];

    for trio in trios {
        let mut u = Matrix3::new(
            s[trio[0]][0],
            s[trio[0]][1],
            s[trio[0]][2],
            s[trio[1]][0],
            s[trio[1]][1],
            s[trio[1]][2],
            s[trio[2]][0],
            s[trio[2]][1],
            s[trio[2]][2],
        );
        let d = det3(&u);
        if d.abs() == 1 {
            if d < 0 {
                // Flip first row to make det(U) = +1
                for c in 0..3 {
                    u[(0, c)] = -u[(0, c)];
                }
            }
            let u_f64 = Matrix3::from_fn(|r, c| u[(r, c)] as f64);
            let b = u_f64 * cell;
            return Ok((b, u));
        }
    }

    Err(invalid("no unimodular trio found in Selling reduction"))
}

/// Rotations of a reduced cell B that preserve the metric within Cartesian tolerance tol.
/// Returns integer rotation matrices W_B (in the reduced fractional basis).
pub fn lattice_rotations(b: &Matrix3<f64>, tol: f64) -> Result<Vec<Matrix3<i32>>> {
    validate_cell(b, tol)?;

    let b_inv = b
        .try_inverse()
        .ok_or_else(|| invalid("singular reduced basis"))?;
    let b_inv_t = b_inv.transpose(); // rows are reciprocal vectors b_i*
    let g = b * b.transpose();

    // Check height condition: tol < 0.1 * min(height_i)
    for i in 0..3 {
        let recip_len = b_inv_t.row(i).norm();
        let height = 1.0 / recip_len;
        if tol >= 0.1 * height {
            return Err(invalid(format!(
                "tolerance ({}) exceeds 0.1 x cell height ({})",
                tol, height
            )));
        }
    }

    // Shell matching for each basis vector b_i
    let mut shells: Vec<Vec<[i32; 3]>> = Vec::with_capacity(3);
    for i in 0..3 {
        let n_i = b.row(i).norm();
        let mut bounds = [0i32; 3];
        for j in 0..3 {
            let recip_len = b_inv_t.row(j).norm();
            bounds[j] = ((n_i + tol) * recip_len).ceil() as i32;
        }

        let mut shell = Vec::new();
        for m0 in -bounds[0]..=bounds[0] {
            for m1 in -bounds[1]..=bounds[1] {
                for m2 in -bounds[2]..=bounds[2] {
                    let v = Vector3::new(
                        m0 as f64 * b[(0, 0)] + m1 as f64 * b[(1, 0)] + m2 as f64 * b[(2, 0)],
                        m0 as f64 * b[(0, 1)] + m1 as f64 * b[(1, 1)] + m2 as f64 * b[(2, 1)],
                        m0 as f64 * b[(0, 2)] + m1 as f64 * b[(1, 2)] + m2 as f64 * b[(2, 2)],
                    );
                    if (v.norm() - n_i).abs() <= tol {
                        shell.push([m0, m1, m2]);
                    }
                }
            }
        }
        shells.push(shell);
    }

    let b_t = b.transpose();

    let mut out = Vec::new();
    let norm_b0 = b.row(0).norm();
    let norm_b1 = b.row(1).norm();
    let norm_b2 = b.row(2).norm();

    for n1 in &shells[0] {
        let v0 = Vector3::new(
            n1[0] as f64 * b[(0, 0)] + n1[1] as f64 * b[(1, 0)] + n1[2] as f64 * b[(2, 0)],
            n1[0] as f64 * b[(0, 1)] + n1[1] as f64 * b[(1, 1)] + n1[2] as f64 * b[(2, 1)],
            n1[0] as f64 * b[(0, 2)] + n1[1] as f64 * b[(1, 2)] + n1[2] as f64 * b[(2, 2)],
        );

        for n2 in &shells[1] {
            let v1 = Vector3::new(
                n2[0] as f64 * b[(0, 0)] + n2[1] as f64 * b[(1, 0)] + n2[2] as f64 * b[(2, 0)],
                n2[0] as f64 * b[(0, 1)] + n2[1] as f64 * b[(1, 1)] + n2[2] as f64 * b[(2, 1)],
                n2[0] as f64 * b[(0, 2)] + n2[1] as f64 * b[(1, 2)] + n2[2] as f64 * b[(2, 2)],
            );

            // Metric angle check on pair (b0, b1)
            let dot01 = v0.dot(&v1);
            if (dot01 - g[(0, 1)]).abs() > tol * (norm_b0 + norm_b1) + 1e-10 {
                continue;
            }

            for n3 in &shells[2] {
                let m = Matrix3::new(
                    n1[0], n1[1], n1[2],
                    n2[0], n2[1], n2[2],
                    n3[0], n3[1], n3[2],
                );
                if det3(&m).abs() != 1 {
                    continue;
                }

                let v2 = Vector3::new(
                    n3[0] as f64 * b[(0, 0)] + n3[1] as f64 * b[(1, 0)] + n3[2] as f64 * b[(2, 0)],
                    n3[0] as f64 * b[(0, 1)] + n3[1] as f64 * b[(1, 1)] + n3[2] as f64 * b[(2, 1)],
                    n3[0] as f64 * b[(0, 2)] + n3[1] as f64 * b[(1, 2)] + n3[2] as f64 * b[(2, 2)],
                );

                if (v0.dot(&v2) - g[(0, 2)]).abs() > tol * (norm_b0 + norm_b2) + 1e-10 {
                    continue;
                }
                if (v1.dot(&v2) - g[(1, 2)]).abs() > tol * (norm_b1 + norm_b2) + 1e-10 {
                    continue;
                }

                // Polar check: R = B^T * M^T * B^-T
                let m_f64 = Matrix3::from_fn(|r, c| m[(r, c)] as f64);
                let r_mat = b_t * m_f64.transpose() * b_inv_t;

                let svd = SVD::new(r_mat, true, true);
                let u = svd.u.ok_or_else(|| invalid("SVD U decomposition failed"))?;
                let v_t = svd.v_t.ok_or_else(|| invalid("SVD V decomposition failed"))?;
                let q = u * v_t;

                let diff = (r_mat - q) * b_t;
                let mut max_disp = 0.0f64;
                for c in 0..3 {
                    let col_norm = diff.column(c).norm();
                    if col_norm > max_disp {
                        max_disp = col_norm;
                    }
                }

                if max_disp <= tol {
                    out.push(m.transpose());
                }
            }
        }
    }

    Ok(out)
}

/// Automorphisms of the input cell in the input fractional basis.
/// Returns exact integer unimodular matrices W_in.
pub fn input_lattice_rotations(cell: &Matrix3<f64>, tol: f64) -> Result<Vec<Matrix3<i32>>> {
    let (b, u) = selling_reduce(cell)?;
    let w_b_list = lattice_rotations(&b, tol)?;

    let u_t = u.transpose();
    let u_inv_t = unimodular_inverse(&u)?.transpose();

    let mut out = Vec::with_capacity(w_b_list.len());
    let mut seen = std::collections::HashSet::new();

    for wb in w_b_list {
        // W_in = U^T * W_B * U^-T
        let win = u_t * wb * u_inv_t;
        let d = det3(&win);
        if d.abs() != 1 {
            return Err(invalid("mapped rotation is not unimodular"));
        }
        let key = [
            win[(0, 0)], win[(0, 1)], win[(0, 2)],
            win[(1, 0)], win[(1, 1)], win[(1, 2)],
            win[(2, 0)], win[(2, 1)], win[(2, 2)],
        ];
        if seen.insert(key) {
            out.push(win);
        }
    }

    // Sort operations: identity first, then by entries
    out.sort_by(|a, b| {
        let is_id_a = *a == Matrix3::identity();
        let is_id_b = *b == Matrix3::identity();
        match (is_id_a, is_id_b) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.as_slice().cmp(b.as_slice()),
        }
    });

    Ok(out)
}
