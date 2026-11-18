//! Crystallographic & magnetic (Shubnikov) double groups and projective little-group irreps.
//!
//! Every space-group operation `g = {R | v}` lifts to two SU(2) spinor elements `(g, +1)` and
//! `(g, -1)` differing by the central `2 pi` spin-flip `bar_E = (E, -1)`. At wavevector `k` the
//! Bloch action obeys a projective factor system combining the spinor sign cocycle and fractional
//! nonsymmorphic lattice translations `L_ab = (W_a tau_b + tau_a) - tau_c`:
//!
//! `D_k(a) D_k(b) = exp(-2 pi i k . L_ab) w_spin(a, b) D_k(c)`.
//!
//! Irreducible projective characters of any unitary little group `H_k` are computed directly from
//! sequential spectral splitting of the twisted center `Z(C^omega[H_k])`, automatically
//! separating single-valued irreps (`chi(bar_E) = +d`) from double-valued spinor irreps
//! (`chi(bar_E) = -d`) and nonsymmorphic boundary multiplier irreps, followed by Dimmock-Wheeler /
//! Herring-Wigner corepresentation classification (`Type a/b/c`) when anti-unitary operations
//! (`T * g`) are present.
use super::Operations;
use crate::math::linalg::eigen;
use crate::periodic::invalid;
use crate::quantum::C64;
use crate::Result;
use nalgebra::{DMatrix, Matrix3, Rotation3, UnitQuaternion, Vector3};
use std::f64::consts::PI;

const MATCH_TOL: f64 = 1e-4;
const SPLIT_TOL: f64 = 1e-6;

/// SU(2) spinor rotation matrix `U(R)` (`psi'_s = sum_t U_st psi_t`) for proper or improper
/// Cartesian orthogonal matrix `r` (an improper operation acts on axial spin as `det(r) * r`).
pub(crate) fn su2(r: &Matrix3<f64>) -> [[C64; 2]; 2] {
    let proper = if r.determinant() < 0. { -r } else { *r };
    let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(proper));
    let (w, x, y, z) = (q.w, q.i, q.j, q.k);
    let i = C64::i();
    [
        [C64::new(w, 0.) - i * z, -i * x - C64::new(y, 0.)],
        [-i * x + C64::new(y, 0.), C64::new(w, 0.) + i * z],
    ]
}

fn mat2_mul(a: &[[C64; 2]; 2], b: &[[C64; 2]; 2]) -> [[C64; 2]; 2] {
    let mut out = [[C64::default(); 2]; 2];
    for r in 0..2 {
        for c in 0..2 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c];
        }
    }
    out
}

fn mat2_diff(a: &[[C64; 2]; 2], b: &[[C64; 2]; 2], scale: f64) -> f64 {
    let mut max = 0.0_f64;
    for r in 0..2 {
        for c in 0..2 {
            max = max.max((a[r][c] - b[r][c] * scale).norm());
        }
    }
    max
}

fn spin_cocycle(a: &[[C64; 2]; 2], b: &[[C64; 2]; 2], target: &[[C64; 2]; 2]) -> Result<i8> {
    let prod = mat2_mul(a, b);
    if mat2_diff(&prod, target, 1.) < MATCH_TOL {
        Ok(1)
    } else if mat2_diff(&prod, target, -1.) < MATCH_TOL {
        Ok(-1)
    } else {
        Err(invalid(
            "SU(2) product does not match target double-group coset",
        ))
    }
}

/// Magnetic Herring-Wigner corepresentation class of a unitary little-group irrep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorepKind {
    /// No anti-unitary operations in the little group.
    UnitaryOnly,
    /// Type a (`sum = +1`): corepresentation adds no extra degeneracy.
    TypeA,
    /// Type b (`sum = -1`): Kramers-doubled degeneracy on the same irrep.
    TypeB,
    /// Type c (`sum = 0`): paired degenerate doublet with conjugate partner `partner`.
    TypeC { partner: usize },
}

/// One irreducible projective representation of a double little group `H_k`.
#[derive(Clone, Debug)]
pub struct DoubleIrrep {
    pub index: usize,
    pub dimension: usize,
    /// `true` for half-integer spinor irreps (`chi(bar_E) = -dimension`).
    pub spinor: bool,
    /// Character `chi(g)` ordered over `DoubleGroup::unitary` (`0..n_unitary`).
    pub characters: Vec<C64>,
    pub corep: CorepKind,
}

/// One element of the doubled space/point group cover (`SU(2)` lift of `{W | tau}`).
#[derive(Clone, Debug)]
pub struct DoubleElement {
    /// Index into source `Operations`.
    pub op_index: usize,
    /// Lift branch sign (`+1` or `-1`, where `(E, -1)` is `bar_E`).
    pub spin_sign: i8,
    pub rotation: Matrix3<i32>,
    pub cartesian: Matrix3<f64>,
    pub translation: [f64; 3],
    pub su2: [[C64; 2]; 2],
    pub anti_unitary: bool,
}

/// Unitary (and optional anti-unitary) crystallographic double group or little group `G_k`.
#[derive(Clone, Debug)]
pub struct DoubleGroup {
    /// Fractional wavevector (equals `[0, 0, 0]` for point double groups at `Gamma`).
    pub k_frac: [f64; 3],
    /// Unitary double-group elements (identity at index `0`, `bar_E` at `bar_e`).
    pub unitary: Vec<DoubleElement>,
    /// Optional anti-unitary coset elements `T * g` keeping `k` invariant (`-R k == k mod G`).
    pub anti_unitary: Vec<DoubleElement>,
    pub bar_e: usize,
    /// Multiplication table among `unitary` indices: `(i, j) -> product_idx`.
    pub product: Vec<Vec<usize>>,
    /// Projective Bloch translation multiplier `exp(-2 pi i k . L_ij)` on `(i, j)`.
    pub factor: Vec<Vec<C64>>,
    /// Inverse element index in `unitary`.
    pub inverse: Vec<usize>,
    /// Single-valued and double-valued projective irreps of `unitary`.
    pub irreps: Vec<DoubleIrrep>,
}

fn wrap_same_k(lattice: &Matrix3<f64>, a: &[f64; 3], b: &[f64; 3]) -> bool {
    let _ = lattice;
    (0..3).all(|i| {
        let d = a[i] - b[i];
        (d - d.round()).abs() < MATCH_TOL
    })
}

fn rotated_k(
    lattice: &Matrix3<f64>,
    lattice_inv_t: &Matrix3<f64>,
    r_cart: &Matrix3<f64>,
    k_frac: &[f64; 3],
) -> [f64; 3] {
    let k_cart = lattice_inv_t * (2. * PI * Vector3::from(*k_frac));
    let moved_frac = (lattice.transpose() * (r_cart * k_cart)) / (2. * PI);
    [moved_frac[0], moved_frac[1], moved_frac[2]]
}

fn selected_base_ops(
    rotations: &[Matrix3<i32>],
    translations: &[[f64; 3]],
    candidates: &[usize],
    lattice: &Matrix3<f64>,
    anti_flags: Option<&[bool]>,
    k_frac: [f64; 3],
) -> Result<(Vec<usize>, Vec<usize>)> {
    let inv = lattice
        .try_inverse()
        .ok_or_else(|| invalid("singular lattice matrix in double group"))?;
    let lattice_inv_t = inv.transpose();
    let mut u_ops = Vec::new();
    let mut a_ops = Vec::new();
    for &s in candidates {
        let w = rotations[s].map(f64::from);
        let r = lattice * w * inv;
        let rk = rotated_k(lattice, &lattice_inv_t, &r, &k_frac);
        let anti = anti_flags.is_some_and(|flags| flags[s]);
        let target = if anti {
            [-k_frac[0], -k_frac[1], -k_frac[2]]
        } else {
            k_frac
        };
        if wrap_same_k(lattice, &rk, &target) {
            if anti {
                a_ops.push(s);
            } else {
                u_ops.push(s);
            }
        }
    }
    // Ensure identity is first in `u_ops`.
    let id_w = Matrix3::<i32>::identity();
    let id_pos = u_ops
        .iter()
        .position(|&s| {
            rotations[s] == id_w
                && translations[s]
                    .iter()
                    .all(|t| (t - t.round()).abs() < MATCH_TOL)
        })
        .ok_or_else(|| invalid("unitary double little group must contain the identity"))?;
    u_ops.swap(0, id_pos);
    Ok((u_ops, a_ops))
}

impl DoubleGroup {
    /// Construct the crystallographic or magnetic double group (optionally at `k_frac`; `None`
    /// selects `Gamma = [0, 0, 0]`) from snapshot `description::Operations`.
    pub fn from_description(
        ops: &crate::periodic::description::Operations,
        lattice: &Matrix3<f64>,
        k_frac: Option<[f64; 3]>,
    ) -> Result<Self> {
        let rotations: Vec<Matrix3<i32>> = ops
            .rotations
            .iter()
            .map(|r| Matrix3::from_row_slice(r))
            .collect();
        let candidates: Vec<usize> = (0..rotations.len()).collect();
        Self::from_raw_operations(
            &rotations,
            &ops.translations,
            &candidates,
            lattice,
            ops.anti_unitary.as_deref(),
            k_frac.unwrap_or([0.; 3]),
        )
    }

    /// Construct the crystallographic or magnetic double group (optionally at `k_frac`; `None`
    /// selects `Gamma = [0, 0, 0]`) and compute all single- and double-valued projective irreps.
    pub fn from_operations(
        ops: &Operations,
        lattice: &Matrix3<f64>,
        anti_unitary: Option<&[bool]>,
        k_frac: Option<[f64; 3]>,
    ) -> Result<Self> {
        // Prefer distinct rotational cosets (`pure_idx == 0`) when closed (all primitive cells),
        // falling back to full operations when centering pure translations are required.
        let use_all = ops.n_pure_translations > 1;
        let mut candidates = Vec::new();
        let mut seen_rot = vec![false; ops.n_distinct_rotations];
        for s in 0..ops.rotations.len() {
            let (rot_idx, pure_idx) = ops.op_factors[s];
            if !use_all && (pure_idx != 0 || seen_rot[rot_idx]) {
                continue;
            }
            seen_rot[rot_idx] = true;
            candidates.push(s);
        }
        Self::from_raw_operations(
            &ops.rotations,
            &ops.translations,
            &candidates,
            lattice,
            anti_unitary,
            k_frac.unwrap_or([0.; 3]),
        )
    }

    pub(super) fn from_raw_operations(
        rotations: &[Matrix3<i32>],
        translations: &[[f64; 3]],
        candidates: &[usize],
        lattice: &Matrix3<f64>,
        anti_unitary: Option<&[bool]>,
        k_frac: [f64; 3],
    ) -> Result<Self> {
        if let Some(flags) = anti_unitary {
            if flags.len() != rotations.len() {
                return Err(invalid("anti_unitary flags must match operations length"));
            }
        }
        let inv = lattice
            .try_inverse()
            .ok_or_else(|| invalid("singular lattice matrix in double group"))?;
        let (u_indices, a_indices) = selected_base_ops(
            rotations,
            translations,
            candidates,
            lattice,
            anti_unitary,
            k_frac,
        )?;
        let lift = |indices: &[usize], anti: bool| -> Vec<DoubleElement> {
            let mut list = Vec::with_capacity(indices.len() * 2);
            for &sign in &[1_i8, -1_i8] {
                for &s in indices {
                    let w_i = rotations[s];
                    let w = w_i.map(f64::from);
                    let r = lattice * w * inv;
                    let mut u0 = su2(&r);
                    if sign < 0 {
                        for row in &mut u0 {
                            for entry in row {
                                *entry = -*entry;
                            }
                        }
                    }
                    list.push(DoubleElement {
                        op_index: s,
                        spin_sign: sign,
                        rotation: w_i,
                        cartesian: r,
                        translation: translations[s],
                        su2: u0,
                        anti_unitary: anti,
                    });
                }
            }
            list
        };
        let unitary = lift(&u_indices, false);
        let bar_e = u_indices.len();
        let anti_elems = lift(&a_indices, true);
        let n = unitary.len();
        let mut product = vec![vec![0usize; n]; n];
        let mut factor = vec![vec![C64::new(1., 0.); n]; n];
        for i in 0..n {
            for j in 0..n {
                let w_prod = unitary[i].rotation * unitary[j].rotation;
                let t_prod = Vector3::from(unitary[i].translation)
                    + unitary[i].rotation.map(f64::from) * Vector3::from(unitary[j].translation);
                // Find the matching coset element in `0..bar_e` whose translation differs by a lattice vector.
                let mut found_base = None;
                let mut shift_l = Vector3::zeros();
                for base in 0..bar_e {
                    if unitary[base].rotation != w_prod {
                        continue;
                    }
                    let delta = t_prod - Vector3::from(unitary[base].translation);
                    let rounded = delta.map(f64::round);
                    if (lattice * (delta - rounded)).norm() < 1e-3 {
                        found_base = Some(base);
                        shift_l = rounded;
                        break;
                    }
                }
                let base = found_base.ok_or_else(|| {
                    invalid("double little group failed closure; use primitive cell operations")
                })?;
                let sign = spin_cocycle(&unitary[i].su2, &unitary[j].su2, &unitary[base].su2)?;
                let target_idx = if sign > 0 { base } else { base + bar_e };
                product[i][j] = target_idx;
                let phase = -2.
                    * PI
                    * (k_frac[0] * shift_l[0] + k_frac[1] * shift_l[1] + k_frac[2] * shift_l[2]);
                factor[i][j] = C64::from_polar(1., phase);
            }
        }
        let mut inverse = vec![0usize; n];
        for i in 0..n {
            inverse[i] = (0..n)
                .find(|&j| product[i][j] == 0)
                .ok_or_else(|| invalid("missing inverse in double group"))?;
        }
        let mut group = Self {
            k_frac,
            unitary,
            anti_unitary: anti_elems,
            bar_e,
            product,
            factor,
            inverse,
            irreps: Vec::new(),
        };
        group.irreps = group.compute_projective_irreps()?;
        group.classify_corepresentations()?;
        Ok(group)
    }

    fn compute_projective_irreps(&self) -> Result<Vec<DoubleIrrep>> {
        let n = self.unitary.len();
        // Central operators `K(g) = (1/N) sum_h L(h) L(g) L(h)^H` commute with all `L(x)` and span Z(C^omega[G]).
        // Since each `L(a)` is monomial (`L(a) e_b = factor[a][b] e_{ab}`), computing `K(g)` takes O(N^2) scalar ops.
        let norm = 1. / n as f64;
        let mut basis = vec![DMatrix::<C64>::identity(n, n)];
        let mut seen_classes = Vec::<Vec<C64>>::new();
        for g in 1..n {
            let mut k_mat = DMatrix::<C64>::zeros(n, n);
            for c in 0..n {
                let gc = self.product[g][c];
                let fg_c = self.factor[g][c];
                for h in 0..n {
                    let col = self.product[h][c];
                    let row = self.product[h][gc];
                    k_mat[(row, col)] += self.factor[h][gc] * fg_c * self.factor[h][c].conj();
                }
            }
            k_mat *= C64::new(norm, 0.);
            let col0: Vec<C64> = (0..n).map(|r| k_mat[(r, 0)]).collect();
            if seen_classes
                .iter()
                .any(|prev| prev.iter().zip(&col0).all(|(a, b)| (a - b).norm() < 1e-10))
            {
                continue;
            }
            seen_classes.push(col0);
            let herm = (&k_mat + k_mat.adjoint()) * C64::new(0.5, 0.);
            let skew = (&k_mat - k_mat.adjoint()) * C64::new(0., -0.5);
            for op in [herm, skew] {
                if op.iter().map(|c| c.norm()).fold(0., f64::max) < 1e-10 {
                    continue;
                }
                let mut next = Vec::with_capacity(basis.len());
                for sub in &basis {
                    let k = sub.ncols();
                    if k == 1 {
                        next.push(sub.clone());
                        continue;
                    }
                    let proj = sub.adjoint() * &op * sub;
                    let mean = proj[(0, 0)].re;
                    let is_scalar = (0..k).all(|r| {
                        (0..k).all(|c| {
                            let target = if r == c {
                                C64::new(mean, 0.)
                            } else {
                                C64::default()
                            };
                            (proj[(r, c)] - target).norm() <= SPLIT_TOL
                        })
                    });
                    if is_scalar {
                        next.push(sub.clone());
                        continue;
                    }
                    let proj = (&proj + proj.adjoint()) * C64::new(0.5, 0.);
                    let (vals, vecs) = eigen(proj)?;
                    let mut start = 0;
                    while start < k {
                        let mut end = start + 1;
                        while end < k && (vals[end] - vals[start]).abs() <= SPLIT_TOL {
                            end += 1;
                        }
                        next.push(sub * vecs.columns(start, end - start));
                        start = end;
                    }
                }
                basis = next;
            }
        }
        let mut irreps = Vec::with_capacity(basis.len());
        for (idx, sub) in basis.into_iter().enumerate() {
            let mult_sq = sub.ncols();
            let d = ((mult_sq as f64).sqrt().round()) as usize;
            if d * d != mult_sq {
                return Err(invalid(
                    "projective isotypic subspace dimension is not a square",
                ));
            }
            let p_iso = &sub * sub.adjoint();
            // O(N) trace using monomial `L(g)`: `trace(P * L(g)) = sum_b P[b, g*b] * factor[g][b]`.
            let inv_d = 1. / d as f64;
            let characters: Vec<C64> = (0..n)
                .map(|g| {
                    let tr: C64 = (0..n)
                        .map(|b| p_iso[(b, self.product[g][b])] * self.factor[g][b])
                        .sum();
                    tr * inv_d
                })
                .collect();
            let spinor = (characters[self.bar_e] + C64::new(d as f64, 0.)).norm() < 1e-4;
            irreps.push(DoubleIrrep {
                index: idx,
                dimension: d,
                spinor,
                characters,
                corep: CorepKind::UnitaryOnly,
            });
        }
        irreps.sort_by(|a, b| {
            a.spinor
                .cmp(&b.spinor)
                .then(a.dimension.cmp(&b.dimension))
                .then(
                    b.characters[1.min(b.characters.len() - 1)]
                        .re
                        .total_cmp(&a.characters[1.min(a.characters.len() - 1)].re),
                )
        });
        for (idx, irrep) in irreps.iter_mut().enumerate() {
            irrep.index = idx;
        }
        Ok(irreps)
    }

    fn classify_corepresentations(&mut self) -> Result<()> {
        if self.anti_unitary.is_empty() {
            return Ok(());
        }
        let m_a = self.anti_unitary.len();
        let half_a = self.bar_e; // base anti-unitary count = anti_unitary.len() / 2
        let _ = half_a;
        // For each unitary irrep evaluate Dimmock-Wheeler sum over anti-unitary coset `a`:
        // `sum_a w(a, a) * chi(a^2) / |H|`, with `a = T * g_a` => `a^2 = (-I_2 if spin_1/2 else I_2?)`:
        // Note: `T^2 = bar_E`, so `(T g)^2 = T^2 g^2` multiplied by lattice shift phase!
        let mut sq_indices = Vec::with_capacity(m_a);
        let mut sq_phases = Vec::with_capacity(m_a);
        for a in &self.anti_unitary {
            let w_sq = a.rotation * a.rotation;
            let t_sq = Vector3::from(a.translation)
                + a.rotation.map(f64::from) * Vector3::from(a.translation);
            let mut found = None;
            let mut shift = Vector3::zeros();
            for b in 0..self.bar_e {
                if self.unitary[b].rotation != w_sq {
                    continue;
                }
                let delta = t_sq - Vector3::from(self.unitary[b].translation);
                let rounded = delta.map(f64::round);
                if (delta - rounded).norm() < 1e-3 {
                    found = Some(b);
                    shift = rounded;
                    break;
                }
            }
            let base =
                found.ok_or_else(|| invalid("anti-unitary square outside unitary subgroup"))?;
            // Include `T^2 = bar_E = -I_2` together with spatial `su2` product!
            let mut prod = mat2_mul(&a.su2, &a.su2);
            for row in &mut prod {
                for v in row {
                    *v = -*v;
                }
            }
            let sign = if mat2_diff(&prod, &self.unitary[base].su2, 1.) < MATCH_TOL {
                1
            } else {
                -1
            };
            let idx = if sign > 0 { base } else { base + self.bar_e };
            let phase = -2.
                * PI
                * (self.k_frac[0] * shift[0]
                    + self.k_frac[1] * shift[1]
                    + self.k_frac[2] * shift[2]);
            sq_indices.push(idx);
            sq_phases.push(C64::from_polar(1., phase));
        }
        let n_u = self.unitary.len() as f64;
        let mut indicators = Vec::with_capacity(self.irreps.len());
        for irrep in &self.irreps {
            let sum: C64 = sq_indices
                .iter()
                .zip(&sq_phases)
                .map(|(&idx, &ph)| ph * irrep.characters[idx])
                .sum();
            let val = (sum / n_u).re;
            indicators.push(if val > 0.5 {
                CorepKind::TypeA
            } else if val < -0.5 {
                CorepKind::TypeB
            } else {
                CorepKind::TypeC {
                    partner: irrep.index,
                }
            });
        }
        // Pair Type C irreps using conjugation by the representative anti-unitary element `a0 = T g0`.
        let a0 = &self.anti_unitary[0];
        let mut conj_map = vec![0usize; self.unitary.len()];
        let mut conj_phases = vec![C64::new(1., 0.); self.unitary.len()];
        let w0_inv = a0
            .rotation
            .map(f64::from)
            .try_inverse()
            .unwrap_or_else(Matrix3::identity);
        let t0 = Vector3::from(a0.translation);
        let mut u0_conj = [[C64::default(); 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                // `T^-1 U T = su2(R0)`? Conjugation on spinor: `(T g0)^-1 h (T g0)` acts spatially by
                // `g0^-1 h g0` and spin-wise by `a0.su2^H * su2_h^* * a0.su2` (T conjugates scalar matrices).
                u0_conj[r][c] = a0.su2[c][r].conj();
            }
        }
        let _ = u0_conj;
        for (h_idx, h_el) in self.unitary.iter().enumerate() {
            let w_conj = (w0_inv * h_el.rotation.map(f64::from) * a0.rotation.map(f64::from))
                .map(|v| v.round() as i32);
            let t_conj =
                w0_inv * (h_el.rotation.map(f64::from) * t0 + Vector3::from(h_el.translation) - t0);
            if let Some(base) = (0..self.bar_e).find(|&b| {
                self.unitary[b].rotation == w_conj
                    && (t_conj - Vector3::from(self.unitary[b].translation))
                        .map(|v| v - v.round())
                        .norm()
                        < 1e-3
            }) {
                let shift =
                    (t_conj - Vector3::from(self.unitary[base].translation)).map(f64::round);
                let u_rot = mat2_mul(
                    &[
                        [a0.su2[0][0].conj(), a0.su2[1][0].conj()],
                        [a0.su2[0][1].conj(), a0.su2[1][1].conj()],
                    ],
                    &mat2_mul(
                        &[
                            [h_el.su2[0][0].conj(), h_el.su2[0][1].conj()],
                            [h_el.su2[1][0].conj(), h_el.su2[1][1].conj()],
                        ],
                        &a0.su2,
                    ),
                );
                let sign = if mat2_diff(&u_rot, &self.unitary[base].su2, 1.) < MATCH_TOL {
                    1
                } else {
                    -1
                };
                conj_map[h_idx] = if sign > 0 { base } else { base + self.bar_e };
                let phase = -2.
                    * PI
                    * (self.k_frac[0] * shift[0]
                        + self.k_frac[1] * shift[1]
                        + self.k_frac[2] * shift[2]);
                conj_phases[h_idx] = C64::from_polar(1., phase);
            }
        }
        for i in 0..self.irreps.len() {
            if let CorepKind::TypeC { .. } = indicators[i] {
                let mut best_partner = i;
                let mut best_overlap = -1.0_f64;
                for j in 0..self.irreps.len() {
                    if j == i || self.irreps[j].dimension != self.irreps[i].dimension {
                        continue;
                    }
                    let overlap: f64 = (0..self.unitary.len())
                        .map(|h| {
                            let target_chi =
                                conj_phases[h] * self.irreps[i].characters[conj_map[h]].conj();
                            (target_chi.conj() * self.irreps[j].characters[h]).re
                        })
                        .sum::<f64>()
                        .abs();
                    if overlap > best_overlap {
                        best_overlap = overlap;
                        best_partner = j;
                    }
                }
                indicators[i] = CorepKind::TypeC {
                    partner: best_partner,
                };
            }
        }
        for (irrep, kind) in self.irreps.iter_mut().zip(indicators) {
            irrep.corep = kind;
        }
        Ok(())
    }

    /// Classify an arbitrary representation (`mats[g]` for every unitary element `g in 0..unitary.len()`)
    /// into `(irrep_index, multiplicity)` pairs.
    pub fn classify_representation(&self, mats: &[DMatrix<C64>]) -> Result<Vec<(usize, f64)>> {
        if mats.len() != self.unitary.len() {
            return Err(invalid(
                "representation matrices must match double group unitary elements",
            ));
        }
        let traces: Vec<C64> = mats.iter().map(DMatrix::trace).collect();
        let inv_n = 1. / self.unitary.len() as f64;
        let mut out = Vec::new();
        for irrep in &self.irreps {
            let mult = traces
                .iter()
                .zip(&irrep.characters)
                .map(|(t, c)| (c.conj() * t).re)
                .sum::<f64>()
                * inv_n;
            if mult > 1e-4 {
                out.push((irrep.index, mult));
            }
        }
        Ok(out)
    }

    /// Return `true` iff `initial_irrep x operator_irrep` contains `final_irrep` (selection rule).
    pub fn allows_transition(
        &self,
        initial_irrep: usize,
        operator_irrep: usize,
        final_irrep: usize,
    ) -> Result<bool> {
        let n = self.irreps.len();
        if initial_irrep >= n || operator_irrep >= n || final_irrep >= n {
            return Err(invalid("double-group irrep index out of range"));
        }
        let sum: f64 = (0..self.unitary.len())
            .map(|g| {
                (self.irreps[final_irrep].characters[g].conj()
                    * self.irreps[operator_irrep].characters[g]
                    * self.irreps[initial_irrep].characters[g])
                    .re
            })
            .sum::<f64>()
            / self.unitary.len() as f64;
        Ok(sum > 0.5)
    }

    /// Cartesian electric-dipole selection weights `[w_x, w_y, w_z]` for direct transitions
    /// `initial_irrep -> final_irrep` at wavevector `k`:
    /// `W_aa = Re (1/|G_k|) sum_g chi_i^*(g) chi_f(g) R(g)_{aa}`.
    /// Nonzero weight (`> 1e-6`) means polarization along Cartesian axis `a` is dipole-allowed
    /// at both symmorphic and nonsymmorphic `k`-points.
    pub fn dipole_axis_weights(
        &self,
        initial_irrep: usize,
        final_irrep: usize,
    ) -> Result<[f64; 3]> {
        let init = self
            .irreps
            .get(initial_irrep)
            .ok_or_else(|| invalid("initial double-group irrep index out of range"))?;
        let fin = self
            .irreps
            .get(final_irrep)
            .ok_or_else(|| invalid("final double-group irrep index out of range"))?;
        let inv_order = 1.0 / (self.unitary.len() as f64);
        let mut weights = [0.0_f64; 3];
        for (g, el) in self.unitary.iter().enumerate() {
            let phase = init.characters[g].conj() * fin.characters[g];
            for a in 0..3 {
                weights[a] += (phase * el.cartesian[(a, a)]).re * inv_order;
            }
        }
        Ok(weights)
    }
}
