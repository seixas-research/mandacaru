//! Exact Smith Normal Form (SNF) origin shift solver and verification.
use nalgebra::Matrix3;
use std::collections::HashMap;

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs()
}

fn mod_inverse(a: i64, m: i64) -> Option<i64> {
    let mut t = 0i64;
    let mut newt = 1i64;
    let mut r = m;
    let mut newr = a.rem_euclid(m);

    while newr != 0 {
        let q = r / newr;
        let tmp_t = t - q * newt;
        t = newt;
        newt = tmp_t;
        let tmp_r = r - q * newr;
        r = newr;
        newr = tmp_r;
    }

    if r > 1 {
        return None;
    }
    Some(t.rem_euclid(m))
}

/// Compute Smith Normal Form of a K x 3 integer matrix A: U * A * V = D.
/// Returns (D_diag: [i64; 3], U: Vec<Vec<i64>>, V: Matrix3<i64>).
pub fn smith_normal_form_3col(
    a_in: &[[i64; 3]],
) -> ([i64; 3], Vec<Vec<i64>>, Matrix3<i64>) {
    let k = a_in.len();
    if k == 0 {
        return ([0, 0, 0], Vec::new(), Matrix3::identity());
    }

    let mut a: Vec<[i64; 3]> = a_in.to_vec();
    let mut u: Vec<Vec<i64>> = (0..k)
        .map(|i| {
            let mut row = vec![0i64; k];
            row[i] = 1;
            row
        })
        .collect();
    let mut v = Matrix3::<i64>::identity();

    let steps = k.min(3);
    for step in 0..steps {
        loop {
            let mut min_val = i64::MAX;
            let mut pivot_pos = None;
            for r in step..k {
                for c in step..3 {
                    let val = a[r][c].abs();
                    if val > 0 && val < min_val {
                        min_val = val;
                        pivot_pos = Some((r, c));
                    }
                }
            }

            let (pr, pc) = match pivot_pos {
                Some(pos) => pos,
                None => break,
            };

            if pr != step {
                a.swap(step, pr);
                u.swap(step, pr);
            }
            if pc != step {
                for r in 0..k {
                    let tmp = a[r][step];
                    a[r][step] = a[r][pc];
                    a[r][pc] = tmp;
                }
                for r in 0..3 {
                    let tmp = v[(r, step)];
                    v[(r, step)] = v[(r, pc)];
                    v[(r, pc)] = tmp;
                }
            }

            let mut changed = false;
            for c in (step + 1)..3 {
                if a[step][c] != 0 {
                    let q = a[step][c] / a[step][step];
                    if q != 0 {
                        for r in 0..k {
                            a[r][c] -= q * a[r][step];
                        }
                        for r in 0..3 {
                            v[(r, c)] -= q * v[(r, step)];
                        }
                    }
                    if a[step][c] != 0 {
                        changed = true;
                    }
                }
            }

            for r in (step + 1)..k {
                if a[r][step] != 0 {
                    let q = a[r][step] / a[step][step];
                    if q != 0 {
                        for c in 0..3 {
                            a[r][c] -= q * a[step][c];
                        }
                        for col in 0..k {
                            u[r][col] -= q * u[step][col];
                        }
                    }
                    if a[r][step] != 0 {
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        if a[step][step] < 0 {
            for c in 0..3 {
                a[step][c] = -a[step][c];
            }
            for col in 0..k {
                u[step][col] = -u[step][col];
            }
        }
    }

    let mut diag = [0i64; 3];
    for i in 0..steps {
        diag[i] = a[i][i];
    }

    (diag, u, v)
}

/// Compute invariant factors of the stacked integer matrix M_L * (I - W) for all rotations.
pub fn invariant_factors_for_group(
    rots: &[[[i8; 3]; 3]],
    m_l: &[[i32; 3]; 3],
) -> Vec<i64> {
    let mut a_rows = Vec::with_capacity(rots.len() * 3);
    for r in rots {
        let i_minus_w = [
            [1 - r[0][0] as i32, -r[0][1] as i32, -r[0][2] as i32],
            [-r[1][0] as i32, 1 - r[1][1] as i32, -r[1][2] as i32],
            [-r[2][0] as i32, -r[2][1] as i32, 1 - r[2][2] as i32],
        ];
        for i in 0..3 {
            let row = [
                (m_l[i][0] * i_minus_w[0][0] + m_l[i][1] * i_minus_w[1][0] + m_l[i][2] * i_minus_w[2][0]) as i64,
                (m_l[i][0] * i_minus_w[0][1] + m_l[i][1] * i_minus_w[1][1] + m_l[i][2] * i_minus_w[2][1]) as i64,
                (m_l[i][0] * i_minus_w[0][2] + m_l[i][1] * i_minus_w[1][2] + m_l[i][2] * i_minus_w[2][2]) as i64,
            ];
            a_rows.push(row);
        }
    }

    let (mut diag, _, _) = smith_normal_form_3col(&a_rows);
    for _ in 0..2 {
        for i in 0..2 {
            if diag[i] > 0 && diag[i + 1] > 0 && diag[i + 1] % diag[i] != 0 {
                let g = gcd(diag[i], diag[i + 1]);
                let l = (diag[i] / g) * diag[i + 1];
                diag[i] = g;
                diag[i + 1] = l;
            }
        }
    }
    diag.iter().copied().filter(|&d| d > 0).collect()
}

/// Verify origin shift on exact integer operations (units of 1/24).
fn verify_exact_shift(
    s: &[i32; 3],
    transformed_ops: &[(Matrix3<i32>, [i32; 3])],
    row_ops: &HashMap<[i32; 9], [i32; 3]>,
    c_trans: &[[i32; 3]],
) -> bool {
    for (w, t) in transformed_ops {
        let key = [
            w[(0, 0)], w[(0, 1)], w[(0, 2)],
            w[(1, 0)], w[(1, 1)], w[(1, 2)],
            w[(2, 0)], w[(2, 1)], w[(2, 2)],
        ];
        let t_row = match row_ops.get(&key) {
            Some(tr) => tr,
            None => return false,
        };

        let ws = [
            (w[(0, 0)] * s[0] + w[(0, 1)] * s[1] + w[(0, 2)] * s[2] - s[0] + t[0] - t_row[0]).rem_euclid(24),
            (w[(1, 0)] * s[0] + w[(1, 1)] * s[1] + w[(1, 2)] * s[2] - s[1] + t[1] - t_row[1]).rem_euclid(24),
            (w[(2, 0)] * s[0] + w[(2, 1)] * s[1] + w[(2, 2)] * s[2] - s[2] + t[2] - t_row[2]).rem_euclid(24),
        ];

        let mut matched = false;
        for c in c_trans {
            if ws[0] == c[0].rem_euclid(24)
                && ws[1] == c[1].rem_euclid(24)
                && ws[2] == c[2].rem_euclid(24)
            {
                matched = true;
                break;
            }
        }
        if !matched {
            return false;
        }
    }
    true
}

/// Solve origin shift for exact integer operations in units of 1/24.
pub fn solve_origin_shift_exact(
    transformed_ops: &[(Matrix3<i32>, [i32; 3])],
    row_ops: &HashMap<[i32; 9], [i32; 3]>,
    c_trans: &[[i32; 3]],
) -> Option<[i32; 3]> {
    let non_trivial: Vec<(Matrix3<i32>, [i32; 3])> = transformed_ops
        .iter()
        .filter(|(w, _)| *w != Matrix3::identity())
        .cloned()
        .collect();

    if non_trivial.is_empty() {
        let s = [0, 0, 0];
        if verify_exact_shift(&s, transformed_ops, row_ops, c_trans) {
            return Some(s);
        }
        return None;
    }

    let mut a_rows = Vec::with_capacity(non_trivial.len() * 3);
    for (w, _) in &non_trivial {
        for r in 0..3 {
            let row = [
                if r == 0 { 1 - w[(r, 0)] } else { -w[(r, 0)] } as i64,
                if r == 1 { 1 - w[(r, 1)] } else { -w[(r, 1)] } as i64,
                if r == 2 { 1 - w[(r, 2)] } else { -w[(r, 2)] } as i64,
            ];
            a_rows.push(row);
        }
    }

    let (diag, u_mat, v_mat) = smith_normal_form_3col(&a_rows);

    for c_vec in c_trans {
        let mut b_vec = Vec::with_capacity(non_trivial.len() * 3);
        for (w, t) in &non_trivial {
            let key = [
                w[(0, 0)], w[(0, 1)], w[(0, 2)],
                w[(1, 0)], w[(1, 1)], w[(1, 2)],
                w[(2, 0)], w[(2, 1)], w[(2, 2)],
            ];
            let t_row = match row_ops.get(&key) {
                Some(tr) => tr,
                None => continue,
            };
            for r in 0..3 {
                let diff = (t[r] - t_row[r] - c_vec[r]).rem_euclid(24) as i64;
                b_vec.push(diff);
            }
        }

        if b_vec.len() != a_rows.len() {
            continue;
        }

        let mut ub = [0i64; 3];
        for i in 0..3 {
            let mut sum = 0i64;
            for j in 0..b_vec.len() {
                sum = (sum + u_mat[i][j] * b_vec[j]).rem_euclid(24);
            }
            ub[i] = sum;
        }

        let mut coord_candidates: [Vec<i64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut solvable = true;

        for i in 0..3 {
            let d = diag[i];
            let rhs = ub[i];
            if d == 0 {
                if rhs % 24 != 0 {
                    solvable = false;
                    break;
                }
                coord_candidates[i] = vec![0];
            } else {
                let g = gcd(d, 24);
                if rhs % g != 0 {
                    solvable = false;
                    break;
                }
                let d_prime = d / g;
                let rhs_prime = rhs / g;
                let m_prime = 24 / g;
                let inv = match mod_inverse(d_prime, m_prime) {
                    Some(val) => val,
                    None => {
                        solvable = false;
                        break;
                    }
                };
                let base_y = (inv * rhs_prime).rem_euclid(m_prime);
                let mut vals = Vec::with_capacity(g as usize);
                for k in 0..g {
                    vals.push(base_y + k * m_prime);
                }
                coord_candidates[i] = vals;
            }
        }

        if !solvable {
            continue;
        }

        for &y0 in &coord_candidates[0] {
            for &y1 in &coord_candidates[1] {
                for &y2 in &coord_candidates[2] {
                    let s = [
                        (v_mat[(0, 0)] * y0 + v_mat[(0, 1)] * y1 + v_mat[(0, 2)] * y2).rem_euclid(24) as i32,
                        (v_mat[(1, 0)] * y0 + v_mat[(1, 1)] * y1 + v_mat[(1, 2)] * y2).rem_euclid(24) as i32,
                        (v_mat[(2, 0)] * y0 + v_mat[(2, 1)] * y1 + v_mat[(2, 2)] * y2).rem_euclid(24) as i32,
                    ];
                    if verify_exact_shift(&s, transformed_ops, row_ops, c_trans) {
                        return Some(s);
                    }
                }
            }
        }
    }

    None
}

/// Solve origin shift for floating point operations, verified to tolerance tol.
pub fn solve_origin_shift_f64(
    transformed_ops: &[(Matrix3<i32>, [f64; 3])],
    row_ops: &HashMap<[i32; 9], [i32; 3]>,
    c_trans: &[[i32; 3]],
    tol: f64,
) -> Option<[f64; 3]> {
    let non_trivial: Vec<(Matrix3<i32>, [f64; 3])> = transformed_ops
        .iter()
        .filter(|(w, _)| *w != Matrix3::identity())
        .cloned()
        .collect();

    if non_trivial.is_empty() {
        let s = [0.0, 0.0, 0.0];
        // Verify identity
        for (w, t) in transformed_ops {
            let key = [
                w[(0, 0)], w[(0, 1)], w[(0, 2)],
                w[(1, 0)], w[(1, 1)], w[(1, 2)],
                w[(2, 0)], w[(2, 1)], w[(2, 2)],
            ];
            let t_row = row_ops.get(&key)?;
            let t_row_f64 = [
                t_row[0] as f64 / 24.0,
                t_row[1] as f64 / 24.0,
                t_row[2] as f64 / 24.0,
            ];
            let ws = [
                (t[0] - t_row_f64[0]).rem_euclid(1.0),
                (t[1] - t_row_f64[1]).rem_euclid(1.0),
                (t[2] - t_row_f64[2]).rem_euclid(1.0),
            ];
            let mut matched = false;
            for c in c_trans {
                let c_f64 = [c[0] as f64 / 24.0, c[1] as f64 / 24.0, c[2] as f64 / 24.0];
                let d0 = (ws[0] - c_f64[0]).rem_euclid(1.0);
                let diff0 = if d0 > 0.5 { 1.0 - d0 } else { d0 };
                let d1 = (ws[1] - c_f64[1]).rem_euclid(1.0);
                let diff1 = if d1 > 0.5 { 1.0 - d1 } else { d1 };
                let d2 = (ws[2] - c_f64[2]).rem_euclid(1.0);
                let diff2 = if d2 > 0.5 { 1.0 - d2 } else { d2 };
                if diff0 <= tol && diff1 <= tol && diff2 <= tol {
                    matched = true;
                    break;
                }
            }
            if !matched {
                return None;
            }
        }
        return Some(s);
    }

    let m_l: [[i32; 3]; 3] = if c_trans.len() == 1 {
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    } else if c_trans.len() == 4 {
        [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    } else if c_trans.len() == 3 {
        [[2, 1, 1], [-1, 1, 1], [-1, -2, 1]]
    } else if c_trans.contains(&[12, 12, 12]) {
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    } else if c_trans.contains(&[12, 12, 0]) {
        [[1, 1, 0], [-1, 1, 0], [0, 0, 1]]
    } else if c_trans.contains(&[0, 12, 12]) {
        [[1, 0, 0], [0, 1, 1], [0, -1, 1]]
    } else {
        [[1, 0, 1], [0, 1, 0], [-1, 0, 1]]
    };

    let mut a_rows = Vec::with_capacity(non_trivial.len() * 3);
    let mut b_vec = Vec::with_capacity(non_trivial.len() * 3);

    for (w, t) in &non_trivial {
        let key = [
            w[(0, 0)], w[(0, 1)], w[(0, 2)],
            w[(1, 0)], w[(1, 1)], w[(1, 2)],
            w[(2, 0)], w[(2, 1)], w[(2, 2)],
        ];
        let t_row = row_ops.get(&key)?;
        let t_row_f64 = [
            t_row[0] as f64 / 24.0,
            t_row[1] as f64 / 24.0,
            t_row[2] as f64 / 24.0,
        ];

        let dt = [
            t[0] - t_row_f64[0],
            t[1] - t_row_f64[1],
            t[2] - t_row_f64[2],
        ];

        let i_minus_w = [
            [1 - w[(0, 0)], -w[(0, 1)], -w[(0, 2)]],
            [-w[(1, 0)], 1 - w[(1, 1)], -w[(1, 2)]],
            [-w[(2, 0)], -w[(2, 1)], 1 - w[(2, 2)]],
        ];

        for i in 0..3 {
            let row = [
                (m_l[i][0] * i_minus_w[0][0] + m_l[i][1] * i_minus_w[1][0] + m_l[i][2] * i_minus_w[2][0]) as i64,
                (m_l[i][0] * i_minus_w[0][1] + m_l[i][1] * i_minus_w[1][1] + m_l[i][2] * i_minus_w[2][1]) as i64,
                (m_l[i][0] * i_minus_w[0][2] + m_l[i][1] * i_minus_w[1][2] + m_l[i][2] * i_minus_w[2][2]) as i64,
            ];
            a_rows.push(row);
            let b_val = (m_l[i][0] as f64) * dt[0] + (m_l[i][1] as f64) * dt[1] + (m_l[i][2] as f64) * dt[2];
            b_vec.push(b_val);
        }
    }

    let (diag, u_mat, v_mat) = smith_normal_form_3col(&a_rows);

    let mut y_candidates: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for i in 0..3 {
        let ub_i: f64 = (0..b_vec.len()).map(|j| (u_mat[i][j] as f64) * b_vec[j]).sum();
        let d = diag[i];
        if d == 0 {
            y_candidates[i] = vec![0.0];
        } else {
            let base = ub_i.rem_euclid(1.0);
            let d_f64 = d as f64;
            y_candidates[i] = (0..d).map(|k| (base + k as f64) / d_f64).collect();
        }
    }

    for &y0 in &y_candidates[0] {
        for &y1 in &y_candidates[1] {
            for &y2 in &y_candidates[2] {
                let s_f64 = [
                    (v_mat[(0, 0)] as f64 * y0 + v_mat[(0, 1)] as f64 * y1 + v_mat[(0, 2)] as f64 * y2).rem_euclid(1.0),
                    (v_mat[(1, 0)] as f64 * y0 + v_mat[(1, 1)] as f64 * y1 + v_mat[(1, 2)] as f64 * y2).rem_euclid(1.0),
                    (v_mat[(2, 0)] as f64 * y0 + v_mat[(2, 1)] as f64 * y1 + v_mat[(2, 2)] as f64 * y2).rem_euclid(1.0),
                ];

                // Verify s_f64 on all transformed_ops
                let mut all_match = true;
                for (w, t) in transformed_ops {
                    let key = [
                        w[(0, 0)], w[(0, 1)], w[(0, 2)],
                        w[(1, 0)], w[(1, 1)], w[(1, 2)],
                        w[(2, 0)], w[(2, 1)], w[(2, 2)],
                    ];
                    let t_row = match row_ops.get(&key) {
                        Some(tr) => tr,
                        None => {
                            all_match = false;
                            break;
                        }
                    };
                    let t_row_f64 = [
                        t_row[0] as f64 / 24.0,
                        t_row[1] as f64 / 24.0,
                        t_row[2] as f64 / 24.0,
                    ];

                    let ws = [
                        (w[(0, 0)] as f64 * s_f64[0] + w[(0, 1)] as f64 * s_f64[1] + w[(0, 2)] as f64 * s_f64[2] - s_f64[0] + t[0] - t_row_f64[0]).rem_euclid(1.0),
                        (w[(1, 0)] as f64 * s_f64[0] + w[(1, 1)] as f64 * s_f64[1] + w[(1, 2)] as f64 * s_f64[2] - s_f64[1] + t[1] - t_row_f64[1]).rem_euclid(1.0),
                        (w[(2, 0)] as f64 * s_f64[0] + w[(2, 1)] as f64 * s_f64[1] + w[(2, 2)] as f64 * s_f64[2] - s_f64[2] + t[2] - t_row_f64[2]).rem_euclid(1.0),
                    ];

                    let mut matched = false;
                    for c in c_trans {
                        let c_f64 = [
                            c[0] as f64 / 24.0,
                            c[1] as f64 / 24.0,
                            c[2] as f64 / 24.0,
                        ];
                        let d0 = (ws[0] - c_f64[0]).rem_euclid(1.0);
                        let diff0 = if d0 > 0.5 { 1.0 - d0 } else { d0 };
                        let d1 = (ws[1] - c_f64[1]).rem_euclid(1.0);
                        let diff1 = if d1 > 0.5 { 1.0 - d1 } else { d1 };
                        let d2 = (ws[2] - c_f64[2]).rem_euclid(1.0);
                        let diff2 = if d2 > 0.5 { 1.0 - d2 } else { d2 };

                        if diff0 <= tol && diff1 <= tol && diff2 <= tol {
                            matched = true;
                            break;
                        }
                    }
                    if !matched {
                        all_match = false;
                        break;
                    }
                }

                if all_match {
                    return Some(s_f64);
                }
            }
        }
    }

    None
}
