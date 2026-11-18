//! Material, magnetic (Shubnikov), and spin-orbit double-group symmetry log reports for crystals.
use super::{
    catalog, classify_point_group,
    double_group::{CorepKind, DoubleGroup},
    find_with_context, identify, Cell, Operations,
};
use crate::{Context, Result};
use nalgebra::{Matrix3, Vector3};

const MOMENT_TOL: f64 = 1e-4;
const SYMPREC: f64 = 1e-5;

/// Partition crystal space-group operations into unitary and anti-unitary magnetic operations
/// keeping atomic axial moment vectors `moments` invariant:
/// unitary `R`: `det(R) * R * m_a == m_{s(a)}`; anti-unitary `T*R`: `-det(R) * R * m_a == m_{s(a)}`.
fn magnetic_coset_classification(
    ops: &Operations,
    lattice: &Matrix3<f64>,
    fractional: &[[f64; 3]],
    moments: &[[f64; 3]],
) -> Result<(Vec<usize>, Vec<bool>)> {
    let inv = lattice
        .try_inverse()
        .ok_or_else(|| crate::periodic::invalid("singular lattice matrix in symmetry report"))?;
    let mut kept = Vec::new();
    let mut anti_flags = Vec::new();
    for s in 0..ops.rotations.len() {
        let w = ops.rotations[s].map(f64::from);
        let t = Vector3::from(ops.translations[s]);
        let r = lattice * w * inv;
        let det = r.determinant();
        let mut atom_map = Vec::with_capacity(fractional.len());
        let mut valid_map = true;
        for f in fractional {
            let moved = w * Vector3::from(*f) + t;
            let matched = fractional.iter().position(|target| {
                (0..3).all(|axis| {
                    let d = moved[axis] - target[axis];
                    (d - d.round()).abs() < 1e-3
                })
            });
            match matched {
                Some(idx) => atom_map.push(idx),
                None => {
                    valid_map = false;
                    break;
                }
            }
        }
        if !valid_map {
            continue;
        }
        let check = |sign: f64| -> bool {
            (0..fractional.len()).all(|a| {
                let b = atom_map[a];
                let turned = (r * Vector3::from(moments[a])) * (det * sign);
                (turned - Vector3::from(moments[b])).norm() <= MOMENT_TOL
            })
        };
        if check(1.) {
            kept.push(s);
            anti_flags.push(false);
        } else if check(-1.) {
            kept.push(s);
            anti_flags.push(true);
        }
    }
    Ok((kept, anti_flags))
}

fn corep_summary(dg: &DoubleGroup) -> String {
    let spinor: Vec<_> = dg.irreps.iter().filter(|ir| ir.spinor).collect();
    let single: Vec<_> = dg.irreps.iter().filter(|ir| !ir.spinor).collect();
    let spinor_dims: Vec<String> = spinor.iter().map(|ir| ir.dimension.to_string()).collect();
    let single_dims: Vec<String> = single.iter().map(|ir| ir.dimension.to_string()).collect();
    let mut text = format!(
        "{} spinor irreps (dims [{}]), {} single-valued irreps (dims [{}])",
        spinor.len(),
        spinor_dims.join(","),
        single.len(),
        single_dims.join(","),
    );
    if !dg.anti_unitary.is_empty() {
        let mut ta = 0usize;
        let mut tb = 0usize;
        let mut tc = 0usize;
        for ir in &spinor {
            match ir.corep {
                CorepKind::TypeA => ta += 1,
                CorepKind::TypeB => tb += 1,
                CorepKind::TypeC { .. } => tc += 1,
                CorepKind::UnitaryOnly => {}
            }
        }
        text.push_str(&format!(
            "; Herring-Wigner spinor coreps: Type a={ta}, Type b={tb} (Kramers), Type c={tc}"
        ));
    }
    text
}

/// Ordered key-value fields for the `[SYMMETRY]` log block of a periodic crystal calculation.
#[allow(clippy::too_many_arguments)]
pub fn crystal_symmetry_fields(
    ctx: &Context,
    lattice_rows: [[f64; 3]; 3],
    positions_bohr: &[[f64; 3]],
    labels: &[u32],
    moments: Option<&[[f64; 3]]>,
    magnetic: bool,
    spinor: bool,
    time_reversal: bool,
    kmesh_reduced: bool,
    kept_operations: Option<usize>,
) -> Result<Vec<(String, String)>> {
    let lattice = Matrix3::from_columns(&[
        Vector3::from(lattice_rows[0]),
        Vector3::from(lattice_rows[1]),
        Vector3::from(lattice_rows[2]),
    ]);
    let inv = lattice
        .try_inverse()
        .ok_or_else(|| crate::periodic::invalid("singular crystal lattice in symmetry report"))?;
    let fractional: Vec<[f64; 3]> = positions_bohr
        .iter()
        .map(|p| {
            let f = inv * Vector3::from(*p);
            [f[0], f[1], f[2]]
        })
        .collect();
    let cell = Cell {
        lattice,
        fractional: &fractional,
        labels,
    };
    let ops = find_with_context(ctx, &cell, SYMPREC)?;
    let id = identify(&ops, &cell)?;
    let pg = catalog::lookup(id.number)
        .map(|row| row.point_group)
        .unwrap_or(ops.point_group.symbol);
    let system = ops.point_group.crystal_system.name();
    let laue = ops.point_group.laue_class;
    let total_ops = ops.rotations.len();
    let n_rot = ops.n_distinct_rotations;
    let n_pure = ops.n_pure_translations;
    let mut fields = vec![
        ("system_kind".into(), "periodic crystal".into()),
        (
            "space_group".into(),
            format!("{} (No. {})", id.symbol, id.number),
        ),
        ("crystal_system".into(), system.into()),
        ("point_group".into(), pg.into()),
        ("laue_class".into(), laue.into()),
        (
            "space_group_operations".into(),
            format!("{total_ops} ({n_rot} rotational cosets, {n_pure} pure translations)"),
        ),
        (
            "kmesh_symmetry".into(),
            match (kmesh_reduced, kept_operations) {
                (true, Some(n)) => format!("reduced by crystal symmetry ({n} kept operations)"),
                (true, None) => "reduced by crystal symmetry".into(),
                (false, _) => "whole mesh (symmetry reduction off)".into(),
            },
        ),
        (
            "material_symmetry_discussion".into(),
            format!(
                "Crystal belongs to the {system} space group {} (No. {}, point group {pg}, \
                 Laue class {laue}, {total_ops} operations); Bloch wavefunctions and Brillouin-zone \
                 stars transform under these crystallographic symmetries.",
                id.symbol, id.number
            ),
        ),
    ];

    let nonzero_moments = moments.is_some_and(|m| m.iter().flatten().any(|v| v.abs() > MOMENT_TOL));
    let mut anti_flags_for_dg: Option<Vec<bool>> = None;
    let mut kept_for_dg: Option<Vec<usize>> = None;
    if magnetic || nonzero_moments {
        let moment_vecs: Vec<[f64; 3]> = match moments {
            Some(m) if m.iter().flatten().any(|v| v.abs() > MOMENT_TOL) => m.to_vec(),
            // Collinear spin-polarized state with uniform net spin magnetization along z:
            _ => vec![[0., 0., 1.]; labels.len()],
        };
        let (kept, flags) =
            magnetic_coset_classification(&ops, &lattice, &fractional, &moment_vecs)?;
        let unitary_rots: Vec<Matrix3<i32>> = kept
            .iter()
            .zip(&flags)
            .filter(|&(_, &anti)| !anti)
            .map(|(&s, _)| ops.rotations[s])
            .collect();
        let n_u = unitary_rots.len();
        let n_a = flags.iter().filter(|&&anti| anti).count();
        let u_pg = classify_point_group(&unitary_rots)
            .map(|info| info.symbol)
            .unwrap_or("1");
        let (group_type, tr_text) = if n_a > 0 {
            (
                "Type III/IV black-white Shubnikov group (anti-unitary T*g cosets preserved)",
                "preserved via anti-unitary T*g operations",
            )
        } else {
            (
                "Type I unitary magnetic group (time reversal T broken)",
                "broken",
            )
        };
        fields.push(("magnetic_symmetry".into(), "present".into()));
        fields.push(("magnetic_group_type".into(), group_type.into()));
        fields.push(("magnetic_point_group".into(), u_pg.into()));
        fields.push((
            "magnetic_operations".into(),
            format!("{n_u} unitary, {n_a} anti-unitary"),
        ));
        fields.push(("time_reversal".into(), tr_text.into()));
        fields.push((
            "magnetic_symmetry_discussion".into(),
            format!(
                "Magnetization lowers the crystal symmetry to its {group_type} \
                 (unitary point group {u_pg}, {n_u} unitary and {n_a} anti-unitary operations \
                 leaving axial spin moments invariant; time reversal is {tr_text})."
            ),
        ));
        kept_for_dg = Some(kept);
        anti_flags_for_dg = Some(flags);
    } else if time_reversal {
        fields.push((
            "time_reversal".into(),
            "preserved (Type II gray crystallographic symmetry)".into(),
        ));
    }

    if spinor {
        let dg = match (kept_for_dg, anti_flags_for_dg) {
            (Some(kept), Some(flags)) => {
                let mut full_flags = vec![false; ops.rotations.len()];
                for (&idx, &f) in kept.iter().zip(&flags) {
                    full_flags[idx] = f;
                }
                let mut coset_kept = Vec::new();
                let mut seen_rot = vec![false; ops.n_distinct_rotations];
                for &s in &kept {
                    let (rot_idx, pure_idx) = ops.op_factors[s];
                    if pure_idx == 0 && !seen_rot[rot_idx] {
                        seen_rot[rot_idx] = true;
                        coset_kept.push(s);
                    }
                }
                let selected = if coset_kept.is_empty() {
                    &kept
                } else {
                    &coset_kept
                };
                DoubleGroup::from_raw_operations(
                    &ops.rotations,
                    &ops.translations,
                    selected,
                    &lattice,
                    Some(&full_flags),
                    [0.; 3],
                )?
            }
            _ => DoubleGroup::from_operations(&ops, &lattice, None, None)?,
        };
        let n_u_dg = dg.unitary.len();
        let n_a_dg = if !magnetic && !nonzero_moments && time_reversal {
            n_u_dg
        } else {
            dg.anti_unitary.len()
        };
        let irreps_line = corep_summary(&dg);
        fields.push((
            "spin_orbit_double_group".into(),
            "present (SU(2) double cover with 2*pi spin rotation bar_E)".into(),
        ));
        fields.push((
            "double_group_order".into(),
            format!("{n_u_dg} unitary, {n_a_dg} anti-unitary"),
        ));
        fields.push(("double_group_irreps".into(), irreps_line));
        fields.push((
            "double_group_discussion".into(),
            format!(
                "Spin-orbit coupling couples spatial operations R in O(3) to SU(2) spin rotations \
                 +-U(R), lifting the symmetry group to its SU(2) double group ({n_u_dg} unitary, \
                 {n_a_dg} anti-unitary elements including bar_E = (E, -1)). Bloch spinors \
                 transform in double-valued spinor irreps (chi(bar_E) = -d), splitting j = l +- 1/2 \
                 manifolds and enforcing Kramers pairing when time-reversal or T*g holds."
            ),
        ));
    }

    Ok(fields)
}
