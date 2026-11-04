//! Crystal space group search, coset reduction, and point-group classification.
pub mod catalog;
pub mod classify;
pub mod closure;
pub mod matching;
pub mod search;
pub mod shift;
pub mod standard;

use super::invalid;
use crate::math::lattice::{
    lattice_rotations, selling_reduce, unimodular_inverse, validate_cell,
};
use crate::Result;
pub use classify::{classify_point_group, CrystalSystem, PointGroupInfo};
use closure::find_closed_cosets;
pub use matching::Identity;
use nalgebra::Matrix3;

/// A periodic crystal geometry for space-group detection.
pub struct Cell<'a> {
    pub lattice: Matrix3<f64>,
    pub fractional: &'a [[f64; 3]],
    pub labels: &'a [u32],
}

/// Space group operations and metadata found for a crystal.
#[derive(Clone, Debug)]
pub struct Operations {
    pub rotations: Vec<Matrix3<i32>>,
    pub translations: Vec<[f64; 3]>,
    pub n_distinct_rotations: usize,
    pub n_pure_translations: usize,
    pub tolerance_requested: f64,
    pub tolerance_used: f64,
    pub point_group: PointGroupInfo,
    pub representative_maps: Vec<Vec<usize>>,
    pub pure_translation_maps: Vec<Vec<usize>>,
    pub op_factors: Vec<(usize, usize)>,
}

pub trait ContextSource {
    fn get_context(&self) -> Result<&crate::Context>;
}

impl ContextSource for crate::Context {
    fn get_context(&self) -> Result<&crate::Context> {
        Ok(self)
    }
}

impl ContextSource for &crate::Context {
    fn get_context(&self) -> Result<&crate::Context> {
        Ok(*self)
    }
}

impl ContextSource for crate::Result<crate::Context> {
    fn get_context(&self) -> Result<&crate::Context> {
        self.as_ref().map_err(|e| e.clone())
    }
}

impl ContextSource for &crate::Result<crate::Context> {
    fn get_context(&self) -> Result<&crate::Context> {
        self.as_ref().map_err(|e| e.clone())
    }
}

/// Find all symmetry operations of the periodic cell using context memory reservation.
pub fn find_with_context<C: ContextSource>(ctx: C, cell: &Cell, tol: f64) -> Result<Operations> {
    let ctx_ref = ctx.get_context()?;
    let ops = find(cell, tol)?;
    let bytes = (ops.n_distinct_rotations + ops.n_pure_translations) * cell.fractional.len() * 4
        + ops.len() * 60;
    let _guard = ctx_ref.reserve(bytes)?;
    Ok(ops)
}

/// Identify the international space group number (1..=230) and Hermann-Mauguin symbol.
pub fn identify(ops: &Operations, cell: &Cell) -> Result<Identity> {
    if ops.n_pure_translations > 1 {
        let identity = Matrix3::identity();
        let mut pure_taus = Vec::with_capacity(ops.n_pure_translations);
        for (w, t) in ops.rotations.iter().zip(ops.translations.iter()) {
            if *w == identity {
                pure_taus.push(*t);
            }
        }
        if let Ok(prim_owned) = standard::primitive_cell_from_pure_translations(&pure_taus, cell, ops.tolerance_used) {
            let prim_cell = prim_owned.as_cell();
            if let Ok(prim_ops) = find(&prim_cell, ops.tolerance_requested) {
                if prim_ops.point_group.order > ops.point_group.order {
                    let standard = standard::standardize_operations(&prim_ops, &prim_cell)?;
                    return matching::match_standard_frame(
                        &standard,
                        prim_ops.point_group.crystal_system,
                        prim_ops.point_group.symbol,
                    );
                }
            }
        }
    }

    let standard = standard::standardize_operations(ops, cell)?;
    matching::match_standard_frame(
        &standard,
        ops.point_group.crystal_system,
        ops.point_group.symbol,
    )
}

impl Operations {
    /// Total number of operations.
    pub fn len(&self) -> usize {
        self.rotations.len()
    }

    /// Whether there are no operations.
    pub fn is_empty(&self) -> bool {
        self.rotations.is_empty()
    }

    /// Evaluate the full atom map for the operation at index `op_idx`, writing into `out`.
    pub fn atom_map(&self, op_idx: usize, out: &mut [usize]) {
        let (rep_idx, pure_idx) = self.op_factors[op_idx];
        let rep_map = &self.representative_maps[rep_idx];
        let pure_map = &self.pure_translation_maps[pure_idx];
        for (i, item) in out.iter_mut().enumerate() {
            *item = pure_map[rep_map[i]];
        }
    }
}

/// Find all symmetry operations of the periodic cell within Cartesian distance `tol`.
pub fn find(cell: &Cell, tol: f64) -> Result<Operations> {
    validate_cell(&cell.lattice, tol)?;

    let n = cell.fractional.len();
    if n == 0 {
        return Err(invalid("cell must contain at least one atom"));
    }
    if n != cell.labels.len() {
        return Err(invalid("fractional coordinates and labels length mismatch"));
    }
    for p in cell.fractional {
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            return Err(invalid("fractional coordinates must be finite"));
        }
    }

    // Pairwise distance check for overlapping atoms
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = cell.fractional[i][0] - cell.fractional[j][0];
            let dy = cell.fractional[i][1] - cell.fractional[j][1];
            let dz = cell.fractional[i][2] - cell.fractional[j][2];
            let d = [dx - dx.round(), dy - dy.round(), dz - dz.round()];
            let vx = d[0] * cell.lattice[(0, 0)]
                + d[1] * cell.lattice[(1, 0)]
                + d[2] * cell.lattice[(2, 0)];
            let vy = d[0] * cell.lattice[(0, 1)]
                + d[1] * cell.lattice[(1, 1)]
                + d[2] * cell.lattice[(2, 1)];
            let vz = d[0] * cell.lattice[(0, 2)]
                + d[1] * cell.lattice[(1, 2)]
                + d[2] * cell.lattice[(2, 2)];
            let dist = (vx * vx + vy * vy + vz * vz).sqrt();
            if dist < tol {
                return Err(invalid("too close distance between atoms"));
            }
        }
    }

    // 1. Selling reduction
    let (b, u) = selling_reduce(&cell.lattice)?;
    let u_inv = unimodular_inverse(&u)?;

    // 2. Fractional coordinates in reduced basis: x_B = x_in * U^-1
    let mut xb = Vec::with_capacity(n);
    for p in cell.fractional {
        let x0 = p[0] * u_inv[(0, 0)] as f64 + p[1] * u_inv[(1, 0)] as f64 + p[2] * u_inv[(2, 0)] as f64;
        let x1 = p[0] * u_inv[(0, 1)] as f64 + p[1] * u_inv[(1, 1)] as f64 + p[2] * u_inv[(2, 1)] as f64;
        let x2 = p[0] * u_inv[(0, 2)] as f64 + p[1] * u_inv[(1, 2)] as f64 + p[2] * u_inv[(2, 2)] as f64;
        xb.push([x0.rem_euclid(1.0), x1.rem_euclid(1.0), x2.rem_euclid(1.0)]);
    }

    // 3. Automorphisms of reduced lattice
    let w_b_list = lattice_rotations(&b, tol)?;

    // 4. Coset search with closure shrink loop
    let (cosets, tol_used) = find_closed_cosets(&xb, cell.labels, &b, &w_b_list, tol)?;

    // 5. Back-mapping to input basis
    let u_t = u.transpose();
    let u_inv_t = u_inv.transpose();

    let mut reps_in = Vec::with_capacity(cosets.coset_representatives.len());
    for rep in &cosets.coset_representatives {
        let win = u_t * rep.rotation * u_inv_t;
        let tin = [
            (u_t[(0, 0)] as f64 * rep.translation[0]
                + u_t[(0, 1)] as f64 * rep.translation[1]
                + u_t[(0, 2)] as f64 * rep.translation[2])
                .rem_euclid(1.0),
            (u_t[(1, 0)] as f64 * rep.translation[0]
                + u_t[(1, 1)] as f64 * rep.translation[1]
                + u_t[(1, 2)] as f64 * rep.translation[2])
                .rem_euclid(1.0),
            (u_t[(2, 0)] as f64 * rep.translation[0]
                + u_t[(2, 1)] as f64 * rep.translation[1]
                + u_t[(2, 2)] as f64 * rep.translation[2])
                .rem_euclid(1.0),
        ];
        reps_in.push((win, tin));
    }

    let mut pure_in = Vec::with_capacity(cosets.pure_translations.len());
    for p in &cosets.pure_translations {
        let tin = [
            (u_t[(0, 0)] as f64 * p.translation[0]
                + u_t[(0, 1)] as f64 * p.translation[1]
                + u_t[(0, 2)] as f64 * p.translation[2])
                .rem_euclid(1.0),
            (u_t[(1, 0)] as f64 * p.translation[0]
                + u_t[(1, 1)] as f64 * p.translation[1]
                + u_t[(1, 2)] as f64 * p.translation[2])
                .rem_euclid(1.0),
            (u_t[(2, 0)] as f64 * p.translation[0]
                + u_t[(2, 1)] as f64 * p.translation[1]
                + u_t[(2, 2)] as f64 * p.translation[2])
                .rem_euclid(1.0),
        ];
        pure_in.push(tin);
    }

    // 6. Expand all operations
    let n_reps = reps_in.len();
    let n_pure = pure_in.len();
    let mut all_ops = Vec::with_capacity(n_reps * n_pure);

    for (r_idx, (win, tin)) in reps_in.iter().enumerate() {
        for (p_idx, tau) in pure_in.iter().enumerate() {
            let mut t = [
                (tin[0] + tau[0]).rem_euclid(1.0),
                (tin[1] + tau[1]).rem_euclid(1.0),
                (tin[2] + tau[2]).rem_euclid(1.0),
            ];
            for c in 0..3 {
                if t[c] < 1e-6 || t[c] > 1.0 - 1e-6 {
                    t[c] = 0.0;
                }
            }
            all_ops.push((*win, t, r_idx, p_idx));
        }
    }

    // Sort: identity first, then by (rotation, translation)
    let identity = Matrix3::identity();
    all_ops.sort_by(|a, b| {
        let is_id_a = a.0 == identity
            && a.1[0].abs() < 1e-6
            && a.1[1].abs() < 1e-6
            && a.1[2].abs() < 1e-6;
        let is_id_b = b.0 == identity
            && b.1[0].abs() < 1e-6
            && b.1[1].abs() < 1e-6
            && b.1[2].abs() < 1e-6;
        match (is_id_a, is_id_b) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => {
                let rot_cmp = a.0.as_slice().cmp(b.0.as_slice());
                if rot_cmp != std::cmp::Ordering::Equal {
                    rot_cmp
                } else {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                }
            }
        }
    });

    let mut rotations = Vec::with_capacity(all_ops.len());
    let mut translations = Vec::with_capacity(all_ops.len());
    let mut op_factors = Vec::with_capacity(all_ops.len());

    for (w, t, r_idx, p_idx) in all_ops {
        rotations.push(w);
        translations.push(t);
        op_factors.push((r_idx, p_idx));
    }

    // 7. Point group classification
    let distinct_rotations: Vec<Matrix3<i32>> = reps_in.iter().map(|(w, _)| *w).collect();
    let point_group = classify_point_group(&distinct_rotations)?;

    let representative_maps = cosets
        .coset_representatives
        .into_iter()
        .map(|r| r.atom_map)
        .collect();
    let pure_translation_maps = cosets
        .pure_translations
        .into_iter()
        .map(|t| t.atom_map)
        .collect();

    Ok(Operations {
        rotations,
        translations,
        n_distinct_rotations: n_reps,
        n_pure_translations: n_pure,
        tolerance_requested: tol,
        tolerance_used: tol_used,
        point_group,
        representative_maps,
        pure_translation_maps,
        op_factors,
    })
}
