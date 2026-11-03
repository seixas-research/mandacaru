//! Conventional frame standardization and centering detection.
use crate::math::lattice::{adjugate3, det3};
use crate::periodic::invalid;
use crate::periodic::space_group::catalog::centering_vectors;
use crate::periodic::space_group::classify::CrystalSystem;
use crate::periodic::space_group::{Cell, Operations};
use crate::Result;
use nalgebra::Matrix3;
use std::collections::HashSet;

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs()
}

/// Find a non-zero primitive integer nullvector of (W - I), if it exists.
pub fn int_nullvector(w: &Matrix3<i32>) -> Option<[i32; 3]> {
    let a = w - Matrix3::identity();
    let pairs = [(0, 1), (0, 2), (1, 2)];
    for (i, j) in pairs {
        let v0 = (a[(i, 1)] as i64) * (a[(j, 2)] as i64) - (a[(i, 2)] as i64) * (a[(j, 1)] as i64);
        let v1 = -((a[(i, 0)] as i64) * (a[(j, 2)] as i64) - (a[(i, 2)] as i64) * (a[(j, 0)] as i64));
        let v2 = (a[(i, 0)] as i64) * (a[(j, 1)] as i64) - (a[(i, 1)] as i64) * (a[(j, 0)] as i64);
        let g = gcd(gcd(v0.abs(), v1.abs()), v2.abs());
        if g > 0 {
            let mut v = [(v0 / g) as i32, (v1 / g) as i32, (v2 / g) as i32];
            if v[0] < 0 || (v[0] == 0 && v[1] < 0) || (v[0] == 0 && v[1] == 0 && v[2] < 0) {
                v = [-v[0], -v[1], -v[2]];
            }
            return Some(v);
        }
    }
    None
}

/// Metric dot product for two integer vectors: u^T * G * v.
fn dot_metric(u: &[i32; 3], v: &[i32; 3], g: &Matrix3<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            sum += (u[i] as f64) * g[(i, j)] * (v[j] as f64);
        }
    }
    sum
}

/// Find a 2D Gauss-reduced integer basis [u1, u2] for the plane n^T * u = 0.
fn plane_basis(n: [i32; 3], g_prim: &Matrix3<f64>) -> ([i32; 3], [i32; 3]) {
    let mut u_mat = Matrix3::<i32>::identity();
    let mut v = [n[0] as i64, n[1] as i64, n[2] as i64];

    while v[1] != 0 {
        let q = v[0] / v[1];
        v[0] -= q * v[1];
        for r in 0..3 {
            u_mat[(r, 0)] -= q as i32 * u_mat[(r, 1)];
        }
        v.swap(0, 1);
        for r in 0..3 {
            let tmp = u_mat[(r, 0)];
            u_mat[(r, 0)] = u_mat[(r, 1)];
            u_mat[(r, 1)] = tmp;
        }
    }

    while v[2] != 0 {
        let q = v[0] / v[2];
        v[0] -= q * v[2];
        for r in 0..3 {
            u_mat[(r, 0)] -= q as i32 * u_mat[(r, 2)];
        }
        v.swap(0, 2);
        for r in 0..3 {
            let tmp = u_mat[(r, 0)];
            u_mat[(r, 0)] = u_mat[(r, 2)];
            u_mat[(r, 2)] = tmp;
        }
    }

    let mut u1 = [u_mat[(0, 1)], u_mat[(1, 1)], u_mat[(2, 1)]];
    let mut u2 = [u_mat[(0, 2)], u_mat[(1, 2)], u_mat[(2, 2)]];

    // Gauss reduction in 2D using metric g_prim
    loop {
        let norm1 = dot_metric(&u1, &u1, g_prim);
        let norm2 = dot_metric(&u2, &u2, g_prim);
        if norm2 < norm1 - 1e-12 {
            std::mem::swap(&mut u1, &mut u2);
        }
        let n1 = dot_metric(&u1, &u1, g_prim);
        let n2 = dot_metric(&u2, &u2, g_prim);
        if n1 < 1e-12 {
            break;
        }
        let dot12 = dot_metric(&u1, &u2, g_prim);
        let q = (dot12 / n1).round() as i32;
        if q == 0 {
            break;
        }
        let cand = [u2[0] - q * u1[0], u2[1] - q * u1[1], u2[2] - q * u1[2]];
        let n_cand = dot_metric(&cand, &cand, g_prim);
        if n_cand >= n2 - 1e-12 {
            break;
        }
        u2 = cand;
    }

    (u1, u2)
}

/// 3D row Hermite Normal Form of integer matrix rows.
fn row_hnf_3d(mut rows: Vec<[i64; 3]>) -> Matrix3<i64> {
    for col in 0..3 {
        for r in (col + 1)..rows.len() {
            while rows[r][col] != 0 {
                let q = rows[col][col] / rows[r][col];
                for k in 0..3 {
                    rows[col][k] -= q * rows[r][k];
                }
                rows.swap(col, r);
            }
        }
        if rows[col][col] < 0 {
            for k in 0..3 {
                rows[col][k] = -rows[col][k];
            }
        }
        for r in 0..col {
            if rows[col][col] > 0 {
                let q = rows[r][col].div_euclid(rows[col][col]);
                for k in 0..3 {
                    rows[r][k] -= q * rows[col][k];
                }
            }
        }
    }
    Matrix3::new(
        rows[0][0], rows[0][1], rows[0][2],
        rows[1][0], rows[1][1], rows[1][2],
        rows[2][0], rows[2][1], rows[2][2],
    )
}

/// Compute primitive transformation matrices P and Q from pure translations.
pub fn primitive_matrices_from_pure_taus(
    pure_taus: &[[f64; 3]],
    lattice: &Matrix3<f64>,
) -> Result<(Matrix3<f64>, Matrix3<i32>)> {
    let n_pure = pure_taus.len().max(1) as i64;
    let m_scale = 24 * n_pure;
    let mut hnf_rows = Vec::with_capacity(3 + pure_taus.len());
    hnf_rows.push([m_scale, 0, 0]);
    hnf_rows.push([0, m_scale, 0]);
    hnf_rows.push([0, 0, m_scale]);
    for tau in pure_taus {
        let r0 = (tau[0] * m_scale as f64).round() as i64;
        let r1 = (tau[1] * m_scale as f64).round() as i64;
        let r2 = (tau[2] * m_scale as f64).round() as i64;
        hnf_rows.push([r0, r1, r2]);
    }

    let h_mat = row_hnf_3d(hnf_rows);
    let det_h = h_mat[(0, 0)] * h_mat[(1, 1)] * h_mat[(2, 2)];
    if det_h == 0 {
        return Err(invalid("singular translation lattice HNF"));
    }

    let mut p = Matrix3::new(
        h_mat[(0, 0)] as f64 / m_scale as f64, h_mat[(1, 0)] as f64 / m_scale as f64, h_mat[(2, 0)] as f64 / m_scale as f64,
        h_mat[(0, 1)] as f64 / m_scale as f64, h_mat[(1, 1)] as f64 / m_scale as f64, h_mat[(2, 1)] as f64 / m_scale as f64,
        h_mat[(0, 2)] as f64 / m_scale as f64, h_mat[(1, 2)] as f64 / m_scale as f64, h_mat[(2, 2)] as f64 / m_scale as f64,
    );

    let c00 = h_mat[(1, 1)] * h_mat[(2, 2)] - h_mat[(1, 2)] * h_mat[(2, 1)];
    let c01 = -(h_mat[(0, 1)] * h_mat[(2, 2)] - h_mat[(0, 2)] * h_mat[(2, 1)]);
    let c02 = h_mat[(0, 1)] * h_mat[(1, 2)] - h_mat[(0, 2)] * h_mat[(1, 1)];
    let c10 = -(h_mat[(1, 0)] * h_mat[(2, 2)] - h_mat[(1, 2)] * h_mat[(2, 0)]);
    let c11 = h_mat[(0, 0)] * h_mat[(2, 2)] - h_mat[(0, 2)] * h_mat[(2, 0)];
    let c12 = -(h_mat[(0, 0)] * h_mat[(1, 2)] - h_mat[(0, 2)] * h_mat[(1, 0)]);
    let c20 = h_mat[(1, 0)] * h_mat[(2, 1)] - h_mat[(1, 1)] * h_mat[(2, 0)];
    let c21 = -(h_mat[(0, 0)] * h_mat[(2, 1)] - h_mat[(0, 1)] * h_mat[(2, 0)]);
    let c22 = h_mat[(0, 0)] * h_mat[(1, 1)] - h_mat[(0, 1)] * h_mat[(1, 0)];

    let adj_h_t = Matrix3::new(
        c00, c10, c20,
        c01, c11, c21,
        c02, c12, c22,
    );

    let mut q = Matrix3::from_fn(|r, c| {
        let num = m_scale * adj_h_t[(r, c)];
        (num / det_h) as i32
    });

    let cart_p = lattice.transpose() * p;
    if cart_p.determinant() < 0.0 {
        for r in 0..3 {
            p[(r, 0)] = -p[(r, 0)];
            q[(0, r)] = -q[(0, r)];
        }
    }

    Ok((p, q))
}

pub struct OwnedCell {
    pub lattice: Matrix3<f64>,
    pub fractional: Vec<[f64; 3]>,
    pub labels: Vec<u32>,
}

impl OwnedCell {
    pub fn as_cell(&self) -> Cell<'_> {
        Cell {
            lattice: self.lattice,
            fractional: &self.fractional,
            labels: &self.labels,
        }
    }
}

/// Build primitive cell from pure translations and supercell.
pub fn primitive_cell_from_pure_translations(
    pure_taus: &[[f64; 3]],
    cell: &Cell,
    tol: f64,
) -> Result<OwnedCell> {
    let (p, q) = primitive_matrices_from_pure_taus(pure_taus, &cell.lattice)?;
    let prim_lattice = p.transpose() * cell.lattice;

    let mut frac_prim: Vec<[f64; 3]> = Vec::new();
    let mut labels_prim = Vec::new();

    for (p_in, &l) in cell.fractional.iter().zip(cell.labels.iter()) {
        let p_raw = [
            (q[(0, 0)] as f64 * p_in[0] + q[(0, 1)] as f64 * p_in[1] + q[(0, 2)] as f64 * p_in[2]).rem_euclid(1.0),
            (q[(1, 0)] as f64 * p_in[0] + q[(1, 1)] as f64 * p_in[1] + q[(1, 2)] as f64 * p_in[2]).rem_euclid(1.0),
            (q[(2, 0)] as f64 * p_in[0] + q[(2, 1)] as f64 * p_in[1] + q[(2, 2)] as f64 * p_in[2]).rem_euclid(1.0),
        ];

        let mut already_present = false;
        for (q_pos, &ql) in frac_prim.iter().zip(labels_prim.iter()) {
            if ql != l {
                continue;
            }
            let dx = p_raw[0] - q_pos[0];
            let dy = p_raw[1] - q_pos[1];
            let dz = p_raw[2] - q_pos[2];
            let d = [dx - dx.round(), dy - dy.round(), dz - dz.round()];
            let vx = d[0] * prim_lattice[(0, 0)] + d[1] * prim_lattice[(1, 0)] + d[2] * prim_lattice[(2, 0)];
            let vy = d[0] * prim_lattice[(0, 1)] + d[1] * prim_lattice[(1, 1)] + d[2] * prim_lattice[(2, 1)];
            let vz = d[0] * prim_lattice[(0, 2)] + d[1] * prim_lattice[(1, 2)] + d[2] * prim_lattice[(2, 2)];
            let dist = (vx * vx + vy * vy + vz * vz).sqrt();
            if dist < tol.max(1e-4) {
                already_present = true;
                break;
            }
        }

        if !already_present {
            frac_prim.push(p_raw);
            labels_prim.push(l);
        }
    }

    Ok(OwnedCell {
        lattice: prim_lattice,
        fractional: frac_prim,
        labels: labels_prim,
    })
}

/// Standard conventional frame description.
pub struct StandardFrame {
    pub centering: char,
    pub centering_vectors: Vec<[i32; 3]>,
    pub conv_operations: Vec<(Matrix3<i32>, [i32; 3])>,
    pub conv_operations_f64: Vec<(Matrix3<i32>, [f64; 3])>,
}

/// Compute centering vectors (mod 24) from conventional matrix M in primitive coordinates.
fn compute_centering_vectors(m: &Matrix3<i32>) -> Vec<[i32; 3]> {
    let det_m = det3(m);
    if det_m == 0 {
        return vec![[0, 0, 0]];
    }
    let adj_m = adjugate3(m);
    let scale = 24 / det_m.abs();
    let mut seen = HashSet::new();
    let mut out = Vec::new();

    for e0 in 0..3 {
        for e1 in 0..3 {
            for e2 in 0..3 {
                let v0 = (adj_m[(0, 0)] * e0 + adj_m[(0, 1)] * e1 + adj_m[(0, 2)] * e2) * scale;
                let v1 = (adj_m[(1, 0)] * e0 + adj_m[(1, 1)] * e1 + adj_m[(1, 2)] * e2) * scale;
                let v2 = (adj_m[(2, 0)] * e0 + adj_m[(2, 1)] * e1 + adj_m[(2, 2)] * e2) * scale;
                let key = [v0.rem_euclid(24), v1.rem_euclid(24), v2.rem_euclid(24)];
                if seen.insert(key) {
                    out.push(key);
                }
            }
        }
    }

    out.sort();
    out
}

/// Identify centering character from centering vectors in 1/24 units.
fn identify_centering_char(vecs: &[[i32; 3]]) -> char {
    let set: HashSet<[i32; 3]> = vecs.iter().copied().collect();
    for &ch in &['P', 'I', 'F', 'C', 'A', 'B', 'R'] {
        let expected = centering_vectors(ch);
        let exp_set: HashSet<[i32; 3]> = expected.iter().copied().collect();
        if set == exp_set {
            return ch;
        }
    }
    'P'
}

/// Standardize operations to the crystallographic conventional frame.
pub fn standardize_operations(
    ops: &Operations,
    cell: &Cell,
) -> Result<StandardFrame> {
    let system = ops.point_group.crystal_system;

    // 1. Extract pure translations and distinct rotation representatives
    let identity = Matrix3::identity();
    let mut pure_taus: Vec<[f64; 3]> = Vec::with_capacity(ops.n_pure_translations.max(1));
    let mut reps_in: Vec<(Matrix3<i32>, [f64; 3])> = Vec::with_capacity(ops.n_distinct_rotations);
    let mut seen_rots = HashSet::new();

    for (w, t) in ops.rotations.iter().zip(ops.translations.iter()) {
        if *w == identity {
            pure_taus.push(*t);
        }
        let key = [
            w[(0, 0)], w[(0, 1)], w[(0, 2)],
            w[(1, 0)], w[(1, 1)], w[(1, 2)],
            w[(2, 0)], w[(2, 1)], w[(2, 2)],
        ];
        if seen_rots.insert(key) {
            reps_in.push((*w, *t));
        }
    }

    if pure_taus.is_empty() {
        pure_taus.push([0.0, 0.0, 0.0]);
    }

    // Step 1: Primitive Frame Reduction via 3D row Hermite Normal Form (HNF)
    let (p, q) = primitive_matrices_from_pure_taus(&pure_taus, &cell.lattice)?;
    let det_q = det3(&q);
    if det_q == 0 {
        return Err(invalid("singular primitive transformation matrix Q"));
    }
    let adj_q = adjugate3(&q);

    // Primitive Cartesian lattice and metric
    let prim_lattice = p.transpose() * cell.lattice;
    let g_prim = prim_lattice * prim_lattice.transpose();

    // Transform representatives to primitive frame
    let mut reps_prim = Vec::with_capacity(reps_in.len());
    for (win, tin) in &reps_in {
        let num_w = q * win * adj_q;
        let w_prim = Matrix3::from_fn(|r, c| num_w[(r, c)] / det_q);
        let t_prim = [
            (q[(0, 0)] as f64 * tin[0] + q[(0, 1)] as f64 * tin[1] + q[(0, 2)] as f64 * tin[2]).rem_euclid(1.0),
            (q[(1, 0)] as f64 * tin[0] + q[(1, 1)] as f64 * tin[1] + q[(1, 2)] as f64 * tin[2]).rem_euclid(1.0),
            (q[(2, 0)] as f64 * tin[0] + q[(2, 1)] as f64 * tin[1] + q[(2, 2)] as f64 * tin[2]).rem_euclid(1.0),
        ];
        let mut t_snapped = t_prim;
        for c in 0..3 {
            if t_snapped[c] < 1e-6 || t_snapped[c] > 1.0 - 1e-6 {
                t_snapped[c] = 0.0;
            }
        }
        reps_prim.push((w_prim, t_snapped));
    }

    // Step 2: Conventional Basis M (det(M) > 0, columns are a_conv, b_conv, c_conv)
    let rep_rots: Vec<Matrix3<i32>> = reps_prim.iter().map(|(w, _)| *w).collect();

    let m_mat = match system {
        CrystalSystem::Triclinic => Matrix3::identity(),
        CrystalSystem::Monoclinic => {
            let mut r2_opt = None;
            for r in &rep_rots {
                let det = det3(r);
                let tr = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
                if det == 1 && tr == -1 {
                    r2_opt = Some(*r);
                    break;
                } else if det == -1 && tr == 1 {
                    r2_opt = Some(-*r);
                    break;
                }
            }
            let r2 = r2_opt.unwrap_or_else(|| Matrix3::new(-1, 0, 0, 0, 1, 0, 0, 0, -1));
            let b_axis = int_nullvector(&r2).unwrap_or([0, 1, 0]);

            // Normal to plane Pi = ker(R2 + I)
            let r_plus_i = r2 + Matrix3::identity();
            let mut n_row = [0i32; 3];
            for r in 0..3 {
                let g = gcd(gcd(r_plus_i[(r, 0)].abs() as i64, r_plus_i[(r, 1)].abs() as i64), r_plus_i[(r, 2)].abs() as i64);
                if g > 0 {
                    n_row = [
                        (r_plus_i[(r, 0)] as i64 / g) as i32,
                        (r_plus_i[(r, 1)] as i64 / g) as i32,
                        (r_plus_i[(r, 2)] as i64 / g) as i32,
                    ];
                    break;
                }
            }
            if n_row == [0, 0, 0] {
                n_row = b_axis;
            }

            let (u1, u2) = plane_basis(n_row, &g_prim);
            let mut b = b_axis;
            let mut a = u1;
            let mut c = u2;
            let mut m0 = Matrix3::new(a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]);
            if det3(&m0) < 0 {
                b = [-b[0], -b[1], -b[2]];
                m0 = Matrix3::new(a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]);
            }

            // If index is 2 and centering is 'I' or 'A', adapt to 'C'
            if det3(&m0).abs() == 2 {
                let cents = compute_centering_vectors(&m0);
                if cents.contains(&[12, 12, 12]) {
                    // 'I' centering: replace a with a + c
                    a = [a[0] + c[0], a[1] + c[1], a[2] + c[2]];
                    m0 = Matrix3::new(a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]);
                } else if cents.contains(&[0, 12, 12]) {
                    // 'A' centering: swap a and c, negate b
                    let tmp = a;
                    a = c;
                    c = tmp;
                    b = [-b[0], -b[1], -b[2]];
                    m0 = Matrix3::new(a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]);
                }
            }
            m0
        }
        CrystalSystem::Orthorhombic => {
            let mut axes = Vec::new();
            for r in &rep_rots {
                let det = det3(r);
                let tr = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
                if (det == 1 && tr == -1) || (det == -1 && tr == 1) {
                    let r_proper = if det == 1 { *r } else { -*r };
                    if let Some(v) = int_nullvector(&r_proper) {
                        if !axes.iter().any(|a: &[i32; 3]| a[0] == v[0] && a[1] == v[1] && a[2] == v[2]) {
                            axes.push(v);
                        }
                    }
                }
            }
            if axes.len() < 3 {
                Matrix3::identity()
            } else {
                let mut v0 = axes[0];
                let v1 = axes[1];
                let v2 = axes[2];
                let mut m = Matrix3::new(v0[0], v1[0], v2[0], v0[1], v1[1], v2[1], v0[2], v1[2], v2[2]);
                if det3(&m) < 0 {
                    v0 = [-v0[0], -v0[1], -v0[2]];
                    m = Matrix3::new(v0[0], v1[0], v2[0], v0[1], v1[1], v2[1], v0[2], v1[2], v2[2]);
                }
                // For mmm and 222, if index 2 with 'A' or 'B' centering, permute to 'C'
                if ops.point_group.symbol != "mm2" && det3(&m).abs() == 2 {
                    let cents = compute_centering_vectors(&m);
                    if cents.contains(&[0, 12, 12]) {
                        // Permute (y, z, x) -> (x, y, z)
                        m = Matrix3::new(v1[0], v2[0], v0[0], v1[1], v2[1], v0[1], v1[2], v2[2], v0[2]);
                    } else if cents.contains(&[12, 0, 12]) {
                        // Permute (z, x, y) -> (x, y, z)
                        m = Matrix3::new(v2[0], v0[0], v1[0], v2[1], v0[1], v1[1], v2[2], v0[2], v1[2]);
                    }
                }
                m
            }
        }
        CrystalSystem::Tetragonal => {
            let mut r4_opt = None;
            for r in &rep_rots {
                let det = det3(r);
                let tr = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
                if det == 1 && tr == 1 {
                    r4_opt = Some(*r);
                    break;
                } else if det == -1 && tr == -1 {
                    r4_opt = Some(-*r);
                    break;
                }
            }
            let r4 = r4_opt.unwrap_or_else(|| Matrix3::new(0, -1, 0, 1, 0, 0, 0, 0, 1));
            let c_axis = int_nullvector(&r4).unwrap_or([0, 0, 1]);

            // Plane Pi = ker(R4^2 + I)
            let r2 = r4 * r4;
            let r2_plus_i = r2 + Matrix3::identity();
            let mut n_row = [0i32; 3];
            for r in 0..3 {
                let g = gcd(gcd(r2_plus_i[(r, 0)].abs() as i64, r2_plus_i[(r, 1)].abs() as i64), r2_plus_i[(r, 2)].abs() as i64);
                if g > 0 {
                    n_row = [
                        (r2_plus_i[(r, 0)] as i64 / g) as i32,
                        (r2_plus_i[(r, 1)] as i64 / g) as i32,
                        (r2_plus_i[(r, 2)] as i64 / g) as i32,
                    ];
                    break;
                }
            }
            if n_row == [0, 0, 0] {
                n_row = c_axis;
            }

            let (u1, u2) = plane_basis(n_row, &g_prim);
            let mut best_m = None;
            let mut best_det = i32::MAX;

            let test_coeffs = [
                (1, 0), (0, 1), (1, 1), (1, -1),
                (2, 1), (1, 2), (-1, 1), (2, -1),
            ];

            for (c1, c2) in test_coeffs {
                let a = [c1 * u1[0] + c2 * u2[0], c1 * u1[1] + c2 * u2[1], c1 * u1[2] + c2 * u2[2]];
                let b = [
                    r4[(0, 0)] * a[0] + r4[(0, 1)] * a[1] + r4[(0, 2)] * a[2],
                    r4[(1, 0)] * a[0] + r4[(1, 1)] * a[1] + r4[(1, 2)] * a[2],
                    r4[(2, 0)] * a[0] + r4[(2, 1)] * a[1] + r4[(2, 2)] * a[2],
                ];
                let mut c = c_axis;
                let mut m_cand = Matrix3::new(a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]);
                if det3(&m_cand) < 0 {
                    c = [-c[0], -c[1], -c[2]];
                    m_cand = Matrix3::new(a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]);
                }
                let det_m = det3(&m_cand);
                if det_m <= 0 || det_m > 4 || det_m >= best_det {
                    continue;
                }

                // Check integer integrality of all operations in candidate frame
                let adj_m = adjugate3(&m_cand);
                let mut all_integral = true;
                for w in &rep_rots {
                    let num = adj_m * w * m_cand;
                    for r in 0..3 {
                        for col in 0..3 {
                            if num[(r, col)] % det_m != 0 {
                                all_integral = false;
                                break;
                            }
                        }
                    }
                    if !all_integral {
                        break;
                    }
                }
                if !all_integral {
                    continue;
                }

                let cents = compute_centering_vectors(&m_cand);
                let ch = identify_centering_char(&cents);
                if (det_m == 1 && ch == 'P') || (det_m == 2 && ch == 'I') {
                    best_m = Some(m_cand);
                    best_det = det_m;
                }
            }

            best_m.unwrap_or(Matrix3::identity())
        }
        CrystalSystem::Hexagonal | CrystalSystem::Trigonal => {
            let mut r3_opt = None;
            for r in &rep_rots {
                let det = det3(r);
                let tr = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
                if det == 1 && tr == 0 {
                    r3_opt = Some(*r);
                    break;
                } else if det == 1 && tr == 2 {
                    r3_opt = Some(*r * *r);
                    break;
                } else if det == -1 && tr == 0 {
                    r3_opt = Some(-*r);
                    break;
                }
            }
            let r3 = r3_opt.unwrap_or_else(|| Matrix3::new(0, -1, 0, 1, -1, 0, 0, 0, 1));
            let c_axis = int_nullvector(&r3).unwrap_or([0, 0, 1]);

            // Plane Pi = ker(I + R3 + R3^2)
            let r3_sq = r3 * r3;
            let sum_r = Matrix3::identity() + r3 + r3_sq;
            let mut n_row = [0i32; 3];
            for r in 0..3 {
                let g = gcd(gcd(sum_r[(r, 0)].abs() as i64, sum_r[(r, 1)].abs() as i64), sum_r[(r, 2)].abs() as i64);
                if g > 0 {
                    n_row = [
                        (sum_r[(r, 0)] as i64 / g) as i32,
                        (sum_r[(r, 1)] as i64 / g) as i32,
                        (sum_r[(r, 2)] as i64 / g) as i32,
                    ];
                    break;
                }
            }
            if n_row == [0, 0, 0] {
                n_row = c_axis;
            }

            let (u1, u2) = plane_basis(n_row, &g_prim);
            let mut best_m = None;
            let mut best_det = i32::MAX;

            let test_coeffs = [
                (1, 0), (0, 1), (1, 1), (1, -1),
                (2, 1), (1, 2), (-1, 1), (2, -1),
            ];

            for (c1, c2) in test_coeffs {
                let mut a = [c1 * u1[0] + c2 * u2[0], c1 * u1[1] + c2 * u2[1], c1 * u1[2] + c2 * u2[2]];
                let mut b = [
                    r3[(0, 0)] * a[0] + r3[(0, 1)] * a[1] + r3[(0, 2)] * a[2],
                    r3[(1, 0)] * a[0] + r3[(1, 1)] * a[1] + r3[(1, 2)] * a[2],
                    r3[(2, 0)] * a[0] + r3[(2, 1)] * a[1] + r3[(2, 2)] * a[2],
                ];
                let mut c = c_axis;
                let mut m_cand = Matrix3::new(a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]);
                if det3(&m_cand) < 0 {
                    c = [-c[0], -c[1], -c[2]];
                    m_cand = Matrix3::new(a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]);
                }
                let det_m = det3(&m_cand);
                if det_m <= 0 || det_m > 3 || det_m >= best_det {
                    continue;
                }

                let adj_m = adjugate3(&m_cand);
                let mut all_integral = true;
                for w in &rep_rots {
                    let num = adj_m * w * m_cand;
                    for r in 0..3 {
                        for col in 0..3 {
                            if num[(r, col)] % det_m != 0 {
                                all_integral = false;
                                break;
                            }
                        }
                    }
                    if !all_integral {
                        break;
                    }
                }
                if !all_integral {
                    continue;
                }

                let mut cents = compute_centering_vectors(&m_cand);
                if det_m == 3 && cents.contains(&[8, 16, 8]) {
                    a = [-a[0], -a[1], -a[2]];
                    b = [-b[0], -b[1], -b[2]];
                    m_cand = Matrix3::new(a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]);
                    cents = compute_centering_vectors(&m_cand);
                }
                let ch = identify_centering_char(&cents);
                if system == CrystalSystem::Hexagonal && det_m == 1 && ch == 'P' {
                    best_m = Some(m_cand);
                    best_det = det_m;
                } else if system == CrystalSystem::Trigonal {
                    if (det_m == 1 && ch == 'P') || (det_m == 3 && ch == 'R') {
                        best_m = Some(m_cand);
                        best_det = det_m;
                    }
                }
            }

            best_m.unwrap_or(Matrix3::identity())
        }
        CrystalSystem::Cubic => {
            let mut axes_4fold = Vec::new();
            for r in &rep_rots {
                let det = det3(r);
                let tr = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
                if (det == 1 && tr == 1) || (det == -1 && tr == -1) {
                    let r_proper = if det == 1 { *r } else { -*r };
                    if let Some(v) = int_nullvector(&r_proper) {
                        if !axes_4fold.iter().any(|a: &[i32; 3]| a[0] == v[0] && a[1] == v[1] && a[2] == v[2]) {
                            axes_4fold.push(v);
                        }
                    }
                }
            }
            let axes = if axes_4fold.len() >= 3 {
                axes_4fold
            } else {
                let mut axes_2fold = Vec::new();
                for r in &rep_rots {
                    let det = det3(r);
                    let tr = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
                    if det == 1 && tr == -1 {
                        if let Some(v) = int_nullvector(r) {
                            if !axes_2fold.iter().any(|a: &[i32; 3]| a[0] == v[0] && a[1] == v[1] && a[2] == v[2]) {
                                axes_2fold.push(v);
                            }
                        }
                    }
                }
                axes_2fold
            };

            if axes.len() < 3 {
                Matrix3::identity()
            } else {
                let mut v0 = axes[0];
                let v1 = axes[1];
                let v2 = axes[2];
                let mut m = Matrix3::new(v0[0], v1[0], v2[0], v0[1], v1[1], v2[1], v0[2], v1[2], v2[2]);
                if det3(&m) < 0 {
                    v0 = [-v0[0], -v0[1], -v0[2]];
                    m = Matrix3::new(v0[0], v1[0], v2[0], v0[1], v1[1], v2[1], v0[2], v1[2], v2[2]);
                }
                m
            }
        }
    };

    let det_m = det3(&m_mat);
    if det_m <= 0 {
        return Err(invalid("singular conventional basis matrix M"));
    }

    let adj_m = adjugate3(&m_mat);
    let centering_vectors = compute_centering_vectors(&m_mat);
    let centering = identify_centering_char(&centering_vectors);

    let mut conv_operations = Vec::with_capacity(reps_prim.len());
    let mut conv_operations_f64 = Vec::with_capacity(reps_prim.len());
    let mut seen = HashSet::new();

    let det_f64 = det_m as f64;

    for (w_prim, t_prim) in &reps_prim {
        let num_w = adj_m * w_prim * m_mat;
        let w_conv = Matrix3::from_fn(|r, c| (num_w[(r, c)] as f64 / det_f64).round() as i32);

        let t_conv = [
            ((adj_m[(0, 0)] as f64 * t_prim[0] + adj_m[(0, 1)] as f64 * t_prim[1] + adj_m[(0, 2)] as f64 * t_prim[2]) / det_f64).rem_euclid(1.0),
            ((adj_m[(1, 0)] as f64 * t_prim[0] + adj_m[(1, 1)] as f64 * t_prim[1] + adj_m[(1, 2)] as f64 * t_prim[2]) / det_f64).rem_euclid(1.0),
            ((adj_m[(2, 0)] as f64 * t_prim[0] + adj_m[(2, 1)] as f64 * t_prim[1] + adj_m[(2, 2)] as f64 * t_prim[2]) / det_f64).rem_euclid(1.0),
        ];

        let t24 = [
            (t_conv[0] * 24.0).round() as i32 % 24,
            (t_conv[1] * 24.0).round() as i32 % 24,
            (t_conv[2] * 24.0).round() as i32 % 24,
        ];
        let t24_pos = [
            t24[0].rem_euclid(24),
            t24[1].rem_euclid(24),
            t24[2].rem_euclid(24),
        ];

        let key = (
            [w_conv[(0, 0)], w_conv[(0, 1)], w_conv[(0, 2)],
             w_conv[(1, 0)], w_conv[(1, 1)], w_conv[(1, 2)],
             w_conv[(2, 0)], w_conv[(2, 1)], w_conv[(2, 2)]],
            t24_pos,
        );

        if seen.insert(key) {
            conv_operations.push((w_conv, t24_pos));
            conv_operations_f64.push((w_conv, t_conv));
        }
    }

    Ok(StandardFrame {
        centering,
        centering_vectors,
        conv_operations,
        conv_operations_f64,
    })
}
