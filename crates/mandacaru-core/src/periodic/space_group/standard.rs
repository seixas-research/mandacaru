//! Conventional frame standardization and centering detection.
use crate::math::lattice::{adjugate3, det3};
use crate::periodic::space_group::classify::CrystalSystem;
use crate::periodic::space_group::{Cell, Operations};
use crate::Result;
use nalgebra::Matrix3;
use std::collections::HashSet;

fn gcd(mut a: i32, mut b: i32) -> i32 {
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
        let v0 = a[(i, 1)] * a[(j, 2)] - a[(i, 2)] * a[(j, 1)];
        let v1 = -(a[(i, 0)] * a[(j, 2)] - a[(i, 2)] * a[(j, 0)]);
        let v2 = a[(i, 0)] * a[(j, 1)] - a[(i, 1)] * a[(j, 0)];
        let g = gcd(gcd(v0.abs(), v1.abs()), v2.abs());
        if g > 0 {
            let mut v = [v0 / g, v1 / g, v2 / g];
            if v[0] < 0 || (v[0] == 0 && v[1] < 0) || (v[0] == 0 && v[1] == 0 && v[2] < 0) {
                v = [-v[0], -v[1], -v[2]];
            }
            return Some(v);
        }
    }
    None
}

/// Standard conventional frame description.
pub struct StandardFrame {
    pub centering: char,
    pub conv_operations: Vec<(Matrix3<i32>, [i32; 3])>,
}

/// Standardize operations to the crystallographic conventional frame.
pub fn standardize_operations(
    ops: &Operations,
    _cell: &Cell,
) -> Result<StandardFrame> {
    let system = ops.point_group.crystal_system;

    let n_reps = ops.n_distinct_rotations;
    let rep_rotations: Vec<Matrix3<i32>> = ops.rotations[..n_reps].to_vec();

    // Determine conventional basis vectors M in integer fractional units
    let m_mat = match system {
        CrystalSystem::Cubic => {
            let mut axes = HashSet::new();
            for r in &rep_rotations {
                let tr = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
                let det = det3(r);
                if det == 1 && (tr == 1 || (ops.point_group.symbol == "23" && tr == -1)) {
                    if let Some(v) = int_nullvector(r) {
                        axes.insert(v);
                    }
                }
            }
            let axes_vec: Vec<[i32; 3]> = axes.into_iter().collect();
            if axes_vec.len() < 3 {
                Matrix3::identity()
            } else {
                let mut m = Matrix3::new(
                    axes_vec[0][0], axes_vec[0][1], axes_vec[0][2],
                    axes_vec[1][0], axes_vec[1][1], axes_vec[1][2],
                    axes_vec[2][0], axes_vec[2][1], axes_vec[2][2],
                );
                if det3(&m) < 0 {
                    m = Matrix3::new(
                        -axes_vec[0][0], -axes_vec[0][1], -axes_vec[0][2],
                        axes_vec[1][0], axes_vec[1][1], axes_vec[1][2],
                        axes_vec[2][0], axes_vec[2][1], axes_vec[2][2],
                    );
                }
                m
            }
        }
        CrystalSystem::Tetragonal => {
            let mut c_axis = None;
            for r in &rep_rotations {
                if det3(r) == 1 && r[(0, 0)] + r[(1, 1)] + r[(2, 2)] == 1 {
                    if let Some(v) = int_nullvector(r) {
                        c_axis = Some(v);
                        break;
                    }
                }
            }
            let c = c_axis.unwrap_or([0, 0, 1]);
            let a = if c[0] == 0 && c[1] == 0 {
                [1, 0, 0]
            } else {
                [-c[1], c[0], 0]
            };
            let b = [c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2], c[0] * a[1] - c[1] * a[0]];
            let mut m = Matrix3::new(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
            if det3(&m) < 0 {
                for i in 0..3 { m[(0, i)] = -m[(0, i)]; }
            }
            m
        }
        CrystalSystem::Hexagonal | CrystalSystem::Trigonal => {
            let mut c_axis = None;
            for r in &rep_rotations {
                let tr = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
                if det3(r) == 1 && (tr == 0 || tr == 2) {
                    if let Some(v) = int_nullvector(r) {
                        c_axis = Some(v);
                        break;
                    }
                }
            }
            let c = c_axis.unwrap_or([0, 0, 1]);
            let a = if c[0] == 0 && c[1] == 0 {
                [1, 0, 0]
            } else {
                [-c[1], c[0], 0]
            };
            let b = [c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2], c[0] * a[1] - c[1] * a[0]];
            let mut m = Matrix3::new(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
            if det3(&m) < 0 {
                for i in 0..3 { m[(0, i)] = -m[(0, i)]; }
            }
            m
        }
        CrystalSystem::Orthorhombic => {
            let mut axes = HashSet::new();
            for r in &rep_rotations {
                if det3(r) == 1 && r[(0, 0)] + r[(1, 1)] + r[(2, 2)] == -1 {
                    if let Some(v) = int_nullvector(r) {
                        axes.insert(v);
                    }
                }
            }
            let axes_vec: Vec<[i32; 3]> = axes.into_iter().collect();
            if axes_vec.len() < 3 {
                Matrix3::identity()
            } else {
                let mut m = Matrix3::new(
                    axes_vec[0][0], axes_vec[0][1], axes_vec[0][2],
                    axes_vec[1][0], axes_vec[1][1], axes_vec[1][2],
                    axes_vec[2][0], axes_vec[2][1], axes_vec[2][2],
                );
                if det3(&m) < 0 {
                    for i in 0..3 { m[(0, i)] = -m[(0, i)]; }
                }
                m
            }
        }
        CrystalSystem::Monoclinic => {
            let mut b_axis = None;
            for r in &rep_rotations {
                let tr = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
                if (det3(r) == 1 && tr == -1) || (det3(r) == -1 && tr == 1) {
                    if let Some(v) = int_nullvector(r) {
                        b_axis = Some(v);
                        break;
                    }
                }
            }
            let b = b_axis.unwrap_or([0, 1, 0]);
            let a = if b[0] == 0 && b[2] == 0 {
                [1, 0, 0]
            } else {
                [-b[2], 0, b[0]]
            };
            let c = [b[1] * a[2] - b[2] * a[1], b[2] * a[0] - b[0] * a[2], b[0] * a[1] - b[1] * a[0]];
            let mut m = Matrix3::new(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
            if det3(&m) < 0 {
                for i in 0..3 { m[(0, i)] = -m[(0, i)]; }
            }
            m
        }
        CrystalSystem::Triclinic => Matrix3::identity(),
    };

    let det_m = det3(&m_mat);
    let (m_actual, adj_m_t, det_actual) = if det_m == 0 {
        (Matrix3::identity(), Matrix3::identity(), 1)
    } else {
        (m_mat, adjugate3(&m_mat).transpose(), det_m)
    };

    let index = det_actual.abs() / (ops.n_pure_translations as i32).max(1);

    let centering = match (system, index) {
        (CrystalSystem::Cubic, 4) => 'F',
        (CrystalSystem::Cubic, 2) => 'I',
        (CrystalSystem::Orthorhombic, 4) => 'F',
        (CrystalSystem::Orthorhombic, 2) => 'C',
        (CrystalSystem::Tetragonal, 2) => 'I',
        (CrystalSystem::Trigonal, 3) => 'R',
        (CrystalSystem::Monoclinic, 2) => 'C',
        _ => 'P',
    };

    let m_t = m_actual.transpose();
    let det_f64 = det_actual as f64;

    // Convert distinct coset representatives to conventional frame:
    // W_conv = (adj_m_t * W * m_t) / det
    // t_conv = (adj_m_t * t) / det
    let mut conv_ops = Vec::with_capacity(n_reps);
    let mut seen = HashSet::new();

    for i in 0..n_reps {
        let w_orig = ops.rotations[i];
        let t_orig = ops.translations[i];

        let num_w = adj_m_t * w_orig * m_t;
        let w_conv = Matrix3::from_fn(|r, c| (num_w[(r, c)] as f64 / det_f64).round() as i32);

        let t_conv = [
            (adj_m_t[(0, 0)] as f64 * t_orig[0] + adj_m_t[(0, 1)] as f64 * t_orig[1] + adj_m_t[(0, 2)] as f64 * t_orig[2]) / det_f64,
            (adj_m_t[(1, 0)] as f64 * t_orig[0] + adj_m_t[(1, 1)] as f64 * t_orig[1] + adj_m_t[(1, 2)] as f64 * t_orig[2]) / det_f64,
            (adj_m_t[(2, 0)] as f64 * t_orig[0] + adj_m_t[(2, 1)] as f64 * t_orig[1] + adj_m_t[(2, 2)] as f64 * t_orig[2]) / det_f64,
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
            conv_ops.push((w_conv, t24_pos));
        }
    }

    Ok(StandardFrame {
        centering,
        conv_operations: conv_ops,
    })
}
