//! Molecular and periodic crystal orbital symmetry representations and whole-multiplet selection.
use super::{invalid, MeanField, MolecularIntegrals};
use crate::{
    math::{angular, symmetry},
    periodic::space_group::{find, Cell},
    product,
    quantum::{CrystalPointSymmetry, C64},
    Reservation, Result,
};
use nalgebra::{DMatrix, Matrix3, Vector3};
pub struct OrbitalSymmetry {
    pub sets: Vec<Vec<usize>>,
    pub characters: Vec<Vec<f64>>,
    pub order: usize,
    _memory: Reservation,
}
pub struct PeriodicRepresentations {
    pub labels: Vec<String>,
    pub matrices: Vec<DMatrix<C64>>,
    pub orbital_symmetry: OrbitalSymmetry,
}
fn root(parent: &[usize], mut i: usize) -> usize {
    while parent[i] != i {
        i = parent[i];
    }
    i
}
fn finalize_orbital_symmetry(
    m: usize,
    representations: &[DMatrix<C64>],
    memory: Reservation,
) -> OrbitalSymmetry {
    let mut parent: Vec<_> = (0..m).collect();
    for representation in representations {
        for p in 0..m {
            for q in p + 1..m {
                if representation[(p, q)]
                    .norm()
                    .max(representation[(q, p)].norm())
                    > 0.3
                {
                    let a = root(&parent, p);
                    let b = root(&parent, q);
                    parent[a] = b;
                }
            }
        }
    }
    let mut sets: Vec<Vec<usize>> = vec![];
    for p in 0..m {
        if let Some(set) = sets
            .iter_mut()
            .find(|set| root(&parent, set[0]) == root(&parent, p))
        {
            set.push(p);
        } else {
            sets.push(vec![p]);
        }
    }
    let characters = sets
        .iter()
        .map(|set| {
            representations
                .iter()
                .map(|u| set.iter().map(|&p| u[(p, p)].re).sum())
                .collect()
        })
        .collect();
    OrbitalSymmetry {
        sets,
        characters,
        order: representations.len().max(1),
        _memory: memory,
    }
}
fn operation_label(r: &Matrix3<f64>) -> String {
    let det = r.determinant();
    let tr = r.trace();
    if (det - 1.).abs() < 1e-2 {
        if (tr - 3.).abs() < 1e-2 {
            "E".into()
        } else if (tr + 1.).abs() < 1e-2 {
            "C2".into()
        } else if tr.abs() < 1e-2 {
            "C3".into()
        } else if (tr - 1.).abs() < 1e-2 {
            "C4".into()
        } else if (tr - 2.).abs() < 1e-2 {
            "C6".into()
        } else {
            "R".into()
        }
    } else if (tr + 3.).abs() < 1e-2 {
        "I".into()
    } else if (tr - 1.).abs() < 1e-2 {
        "sigma".into()
    } else {
        "S".into()
    }
}
fn harmonic_rotation_block(
    l: usize,
    src_m: &[i32],
    dst_m: &[i32],
    r_cart: &Matrix3<f64>,
) -> DMatrix<f64> {
    let k = src_m.len();
    if l == 0 && k == 1 {
        return DMatrix::from_element(1, 1, 1.);
    }
    let n_dirs = (2 * l + 7).max(18);
    let golden = std::f64::consts::PI * (3. - 5.0_f64.sqrt());
    let dirs: Vec<[f64; 3]> = (0..n_dirs)
        .map(|i| {
            let y = 1. - (2. * i as f64 + 1.) / n_dirs as f64;
            let radius = (1. - y * y).max(0.).sqrt();
            let theta = golden * i as f64;
            [radius * theta.cos(), y, radius * theta.sin()]
        })
        .collect();
    let mut b = DMatrix::<f64>::zeros(n_dirs, k);
    let mut a = DMatrix::<f64>::zeros(n_dirs, k);
    let rt = r_cart.transpose();
    for (p, &u) in dirs.iter().enumerate() {
        let ru_v = rt * Vector3::from(u);
        let ru = [ru_v[0], ru_v[1], ru_v[2]];
        for c in 0..k {
            b[(p, c)] = angular::normalized(l, dst_m[c], u);
            a[(p, c)] = angular::normalized(l, src_m[c], ru);
        }
    }
    let btb = b.transpose() * &b;
    btb.try_inverse()
        .map_or_else(|| DMatrix::identity(k, k), |inv| inv * b.transpose() * a)
}
fn group_atom_shells(
    integrals: &MolecularIntegrals,
    n_atoms: usize,
) -> Vec<Vec<(usize, Vec<(usize, i32)>)>> {
    let mut by_atom: Vec<Vec<(usize, Vec<(usize, i32)>)>> = vec![Vec::new(); n_atoms];
    let mut per_atom: Vec<Vec<usize>> = vec![Vec::new(); n_atoms];
    for (idx, orb) in integrals.basis.orbitals.iter().enumerate() {
        per_atom[orb.atom].push(idx);
    }
    for (atom, indices) in per_atom.into_iter().enumerate() {
        let mut i = 0;
        while i < indices.len() {
            let l = integrals.basis.orbitals[indices[i]].l;
            let width = 2 * l + 1;
            let take = if i + width <= indices.len()
                && indices[i..i + width]
                    .iter()
                    .all(|&idx| integrals.basis.orbitals[idx].l == l)
            {
                width
            } else {
                1
            };
            let members: Vec<_> = indices[i..i + take]
                .iter()
                .map(|&idx| (idx, integrals.basis.orbitals[idx].m))
                .collect();
            by_atom[atom].push((l, members));
            i += take;
        }
    }
    by_atom
}
fn match_supercell_atoms(
    lattice: &Matrix3<f64>,
    lattice_inv: &Matrix3<f64>,
    positions: &[[f64; 3]],
    numbers: &[u8],
    pbc: [bool; 3],
    w: &Matrix3<i32>,
    tau: [f64; 3],
    r_cart: &Matrix3<f64>,
) -> Option<Vec<usize>> {
    let n_atoms = positions.len();
    let frac: Vec<Vector3<f64>> = positions
        .iter()
        .map(|&p| lattice_inv * Vector3::from(p))
        .collect();
    let w_f = w.map(|v| v as f64);
    let mut best_shifts: Vec<Vec<i32>> = vec![vec![0]; 3];
    for axis in 0..3 {
        if !pbc[axis] {
            best_shifts[axis] = vec![-1, 0, 1];
        }
    }
    for &s0 in &best_shifts[0] {
        for &s1 in &best_shifts[1] {
            for &s2 in &best_shifts[2] {
                let shift = Vector3::new(s0 as f64, s1 as f64, s2 as f64);
                let t_vec = Vector3::from(tau) + shift;
                let mut map = vec![0usize; n_atoms];
                let mut used = vec![false; n_atoms];
                let mut ok = true;
                for a in 0..n_atoms {
                    let img_f = w_f * frac[a] + t_vec;
                    let mut best = None;
                    let mut best_d2 = f64::INFINITY;
                    for b in 0..n_atoms {
                        if used[b] || numbers[b] != numbers[a] {
                            continue;
                        }
                        let mut df = img_f - frac[b];
                        for ax in 0..3 {
                            if pbc[ax] {
                                df[ax] -= df[ax].round();
                            }
                        }
                        let dc = lattice * df;
                        let d2 = dc.norm_squared();
                        if d2 < best_d2 {
                            best_d2 = d2;
                            best = Some(b);
                        }
                    }
                    if let Some(b) = best.filter(|_| best_d2 < 1e-6) {
                        map[a] = b;
                        used[b] = true;
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    let t_cart = lattice * t_vec;
                    let _ = (r_cart, t_cart);
                    return Some(map);
                }
            }
        }
    }
    None
}
/// Compute space-group representation matrices in the orthonormal supercell MO basis `coefficients`
/// (`m x ncols`, Lowdin AO representation) for periodic crystals (`integrals.supercell.is_some()`).
pub fn periodic_representations(
    integrals: &MolecularIntegrals,
    coefficients: &DMatrix<C64>,
) -> Result<PeriodicRepresentations> {
    let ctx = &integrals.context;
    let supercell = integrals
        .supercell
        .as_ref()
        .ok_or_else(|| invalid("periodic orbital symmetry requires a supercell"))?;
    let m = integrals.m();
    let ncols = coefficients.ncols();
    if coefficients.nrows() != m {
        return Err(invalid(
            "MO coefficients disagree with supercell orbital dimension",
        ));
    }
    let geom = &integrals.geometry;
    let cell_arr = geom.cell();
    let lattice = Matrix3::from_fn(|r, axis| cell_arr[axis][r]);
    let lattice_inv = lattice
        .try_inverse()
        .ok_or_else(|| invalid("singular supercell lattice matrix"))?;
    let n_atoms = geom.positions().len();
    let frac: Vec<[f64; 3]> = geom
        .positions()
        .iter()
        .map(|&p| {
            let f = lattice_inv * Vector3::from(p);
            [
                f[0] - f[0].floor(),
                f[1] - f[1].floor(),
                f[2] - f[2].floor(),
            ]
        })
        .collect();
    let labels: Vec<u32> = geom.numbers().iter().map(|&z| u32::from(z)).collect();
    let ops = find(
        &Cell {
            lattice,
            fractional: &frac,
            labels: &labels,
        },
        1e-4,
    )
    .or_else(|_| {
        find(
            &Cell {
                lattice,
                fractional: &frac,
                labels: &labels,
            },
            1e-3,
        )
    });
    let pbc = supercell.primitive.pbc();
    let shells = group_atom_shells(integrals, n_atoms);
    let overlap_f = DMatrix::from_fn(m, m, |i, j| integrals.overlap.as_slice()[i * m + j]);
    let lowdin_c = integrals.lowdin.map(|v| C64::new(v, 0.));
    let ao_coeffs = &lowdin_c * coefficients;
    let overlap_c = overlap_f.map(|v| C64::new(v, 0.));
    let mut matrices = Vec::new();
    let mut sym_labels = Vec::new();
    if let Ok(ops) = ops {
        for (w, &tau) in ops.rotations.iter().zip(&ops.translations) {
            let mut axes_ok = true;
            for i in 0..3 {
                for j in 0..3 {
                    if pbc[i] != pbc[j] && w[(i, j)] != 0 {
                        axes_ok = false;
                    }
                }
            }
            if !axes_ok {
                continue;
            }
            let r_cart = lattice * w.map(|v| v as f64) * &lattice_inv;
            if (r_cart.transpose() * r_cart - Matrix3::identity()).amax() > 1e-4 {
                continue;
            }
            let Some(atom_map) = match_supercell_atoms(
                &lattice,
                &lattice_inv,
                geom.positions(),
                geom.numbers(),
                pbc,
                w,
                tau,
                &r_cart,
            ) else {
                continue;
            };
            let mut d_ao = DMatrix::<f64>::zeros(m, m);
            let mut shell_ok = true;
            for (a, &b) in atom_map.iter().enumerate() {
                if shells[a].len() != shells[b].len() {
                    shell_ok = false;
                    break;
                }
                for ((l_a, src), (l_b, dst)) in shells[a].iter().zip(&shells[b]) {
                    if l_a != l_b || src.len() != dst.len() {
                        shell_ok = false;
                        break;
                    }
                    let src_m: Vec<i32> = src.iter().map(|x| x.1).collect();
                    let dst_m: Vec<i32> = dst.iter().map(|x| x.1).collect();
                    let block = harmonic_rotation_block(*l_a, &src_m, &dst_m, &r_cart);
                    for r in 0..dst.len() {
                        for c in 0..src.len() {
                            d_ao[(dst[r].0, src[c].0)] = block[(r, c)];
                        }
                    }
                }
                if !shell_ok {
                    break;
                }
            }
            if !shell_ok || (d_ao.transpose() * &overlap_f * &d_ao - &overlap_f).amax() > 1e-3 {
                continue;
            }
            let u = ao_coeffs.adjoint() * &overlap_c * d_ao.map(|v| C64::new(v, 0.)) * &ao_coeffs;
            if (&u * u.adjoint() - DMatrix::<C64>::identity(ncols, ncols))
                .iter()
                .map(|c| c.norm())
                .fold(0., f64::max)
                <= 1e-3
            {
                sym_labels.push(operation_label(&r_cart));
                matrices.push(u);
            }
        }
    }
    if matrices.is_empty() {
        sym_labels.push("E".into());
        matrices.push(DMatrix::<C64>::identity(ncols, ncols));
    }
    let memory = ctx.reserve(product(&[matrices.len(), ncols, ncols, 16])?)?;
    let orbital_symmetry = finalize_orbital_symmetry(ncols, &matrices, memory);
    Ok(PeriodicRepresentations {
        labels: sym_labels,
        matrices,
        orbital_symmetry,
    })
}
/// Resolve degeneracies among MOs in `coefficients` (grouped by equal energy/value clusters) so
/// each orbital becomes an eigenmode of commuting unitary symmetry operators in `matrices`.
pub fn symmetrize_degenerate_blocks(
    values: &[f64],
    coefficients: &mut DMatrix<C64>,
    matrices: &[DMatrix<C64>],
) -> Result<()> {
    let m = values.len();
    if m == 0 || coefficients.ncols() != m || matrices.is_empty() {
        return Ok(());
    }
    let mut start = 0;
    while start < m {
        let mut end = start + 1;
        while end < m && (values[end] - values[start]).abs() <= 1e-8 {
            end += 1;
        }
        let d = end - start;
        if d > 1 {
            let mut sub_ops: Vec<DMatrix<C64>> = Vec::new();
            for op in matrices {
                let block = op.view((start, start), (d, d)).into_owned();
                let leak: f64 = (0..m)
                    .flat_map(|r| (start..end).map(move |c| (r, c)))
                    .filter(|&(r, _)| r < start || r >= end)
                    .map(|(r, c)| op[(r, c)].norm())
                    .fold(0., f64::max);
                if leak > 1e-4 {
                    continue;
                }
                let commutes = sub_ops.iter().all(|prev| {
                    let diff = prev * &block - &block * prev;
                    diff.iter().map(|c| c.norm()).fold(0., f64::max) < 1e-4
                });
                if commutes {
                    sub_ops.push(block);
                }
            }
            if !sub_ops.is_empty() {
                let mut comb = DMatrix::<C64>::zeros(d, d);
                for (k, op) in sub_ops.iter().enumerate() {
                    let w1 = ((k + 1) as f64) * 0.618_033_988_7 + 0.13;
                    let w2 = ((k + 1) as f64) * 0.381_966_011_3 + 0.07;
                    let herm = (op + op.adjoint()) * C64::new(0.5 * w1, 0.);
                    let skew = (op - op.adjoint()) * C64::new(0., 0.5 * w2);
                    comb += herm + skew;
                }
                let comb = (&comb + comb.adjoint()) * C64::new(0.5, 0.);
                let (_, vecs) = crate::math::linalg::eigen(comb)?;
                let old_block = coefficients.columns(start, d).into_owned();
                let new_block = old_block * vecs;
                for r in 0..coefficients.nrows() {
                    for c in 0..d {
                        coefficients[(r, start + c)] = new_block[(r, c)];
                    }
                }
            }
        }
        start = end;
    }
    Ok(())
}
/// Extract non-trivial diagonal unit-modulus [`CrystalPointSymmetry`] characters verified against `model`.
pub fn verified_point_symmetry(
    labels: &[String],
    matrices: &[DMatrix<C64>],
    active: &[usize],
    model: &super::transform::Model,
) -> Result<Option<CrystalPointSymmetry>> {
    let n = active.len();
    if n == 0 || model.m != n {
        return Ok(None);
    }
    let mut kept_labels: Vec<String> = Vec::new();
    let mut kept_chars: Vec<Vec<C64>> = Vec::new();
    for (label, mat) in labels.iter().zip(matrices) {
        let mut is_diag = true;
        let mut row = Vec::with_capacity(n);
        for (a, &pa) in active.iter().enumerate() {
            let d = mat[(pa, pa)];
            if (d.norm() - 1.).abs() > 1e-4 {
                is_diag = false;
                break;
            }
            for (b, &pb) in active.iter().enumerate() {
                if a != b && mat[(pa, pb)].norm() > 1e-4 {
                    is_diag = false;
                    break;
                }
            }
            if !is_diag {
                break;
            }
            row.push(d / d.norm());
        }
        if !is_diag {
            continue;
        }
        let non_trivial = row.iter().any(|c| (*c - row[0]).norm() > 1e-4);
        if !non_trivial || !model.conserves_point_characters(&row) {
            continue;
        }
        let dup = kept_chars.iter().any(|existing: &Vec<C64>| {
            let same = existing
                .iter()
                .zip(&row)
                .all(|(x, y)| (x * y.conj() - existing[0] * row[0].conj()).norm() < 1e-4);
            let conj = existing
                .iter()
                .zip(&row)
                .all(|(x, y)| (x * y - existing[0] * row[0]).norm() < 1e-4);
            same || conj
        });
        if !dup {
            kept_labels.push(label.clone());
            kept_chars.push(row);
        }
    }
    if kept_chars.is_empty() {
        Ok(None)
    } else {
        Ok(Some(CrystalPointSymmetry::new(kept_labels, kept_chars)?))
    }
}
pub fn orbitals(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    rotation: &DMatrix<C64>,
) -> Result<OrbitalSymmetry> {
    if integrals.supercell.is_some() {
        let c = &reference.model_orbitals * rotation;
        return Ok(periodic_representations(integrals, &c)?.orbital_symmetry);
    }
    let ctx = &integrals.context;
    let m = integrals.m();
    let (center, operations) = symmetry::operations(
        integrals.geometry.positions(),
        integrals.geometry.numbers(),
        0.02,
    )?;
    let points = product(&[integrals.geometry.numbers().len(), 400])?;
    let memory = ctx.reserve(product(&[operations.len(), m, m, 16])?)?;
    let _work = ctx.reserve(
        product(&[points, m, 8, 5])?
            .checked_add(product(&[m, m, 16, 12])?)
            .ok_or(crate::Error::Overflow)?,
    )?;
    let mut samples = Vec::with_capacity(points);
    let mut seed = 0_u64;
    let mut uniform = || {
        seed = seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    for atom in integrals.geometry.positions() {
        for _ in 0..400 {
            let p = Vector3::from(*atom)
                + Vector3::from_fn(|_, _| {
                    (-2. * uniform().ln()).sqrt()
                        * (2. * std::f64::consts::PI * uniform()).cos()
                        * 1.5
                });
            samples.push(p);
        }
    }
    let phi = DMatrix::from_fn(points, m, |i, j| {
        integrals.basis.orbitals[j].evaluate(samples[i].into())
    });
    let svd = phi.svd(true, true);
    let scale = svd.singular_values.amax();
    let inverse = svd.pseudo_inverse(1e-12 * scale).map_err(invalid)?;
    let coefficients =
        integrals.lowdin.map(|v| C64::new(v, 0.)) * &reference.model_orbitals * rotation;
    let overlap = DMatrix::from_fn(m, m, |i, j| {
        C64::new(integrals.overlap.as_slice()[i * m + j], 0.)
    });
    let mut representations = Vec::new();
    for operation in &operations {
        let moved: Vec<_> = samples
            .iter()
            .map(|p| center + operation.transpose() * (p - center))
            .collect();
        let transformed = DMatrix::from_fn(points, m, |i, j| {
            integrals.basis.orbitals[j].evaluate(moved[i].into())
        });
        let ao = &inverse * transformed;
        let representation =
            coefficients.adjoint() * &overlap * ao.map(|v| C64::new(v, 0.)) * &coefficients;
        representations.push(representation);
    }
    Ok(finalize_orbital_symmetry(m, &representations, memory))
}
impl OrbitalSymmetry {
    pub fn select(
        &self,
        occupied: usize,
        doubly: usize,
        frozen: &[usize],
        plain: &[usize],
        occupations: &[f64],
        threshold: Option<f64>,
    ) -> Result<Vec<usize>> {
        for set in &self.sets {
            if set.iter().any(|&i| i < doubly) && set.iter().any(|&i| i >= occupied) {
                return Err(invalid("the mean-field reference mixes occupied and virtual orbitals in one symmetry multiplet"));
            }
            let cut = set.iter().filter(|i| frozen.contains(i)).count();
            if cut > 0 && cut < set.iter().filter(|&&i| i < occupied).count() {
                return Err(invalid("the frozen core splits a symmetry multiplet"));
            }
        }
        let cap = plain.iter().filter(|&&i| i >= occupied).count();
        let mut active: Vec<_> = plain.iter().copied().filter(|&i| i < occupied).collect();
        let virtuals: Vec<_> = self
            .sets
            .iter()
            .enumerate()
            .filter(|(_, set)| set.iter().all(|&i| i >= occupied))
            .filter(|(_, set)| threshold.is_none_or(|t| set.iter().any(|&i| occupations[i] >= t)))
            .collect();
        let reaches = |a: usize| {
            self.sets
                .iter()
                .enumerate()
                .filter(|(_, set)| set.iter().any(|&i| i < occupied && !frozen.contains(&i)))
                .any(|(i, _)| {
                    self.characters[i]
                        .iter()
                        .zip(&self.characters[a])
                        .map(|(x, y)| x * y)
                        .sum::<f64>()
                        / self.order as f64
                        > 0.5
                })
        };
        let mut chosen = Vec::new();
        for (index, set) in &virtuals {
            if chosen.len() + set.len() <= cap {
                chosen.extend_from_slice(set);
                let _ = index;
            }
        }
        if cap > 0
            && !active.is_empty()
            && !virtuals
                .iter()
                .any(|(index, set)| set.iter().all(|i| chosen.contains(i)) && reaches(*index))
        {
            let (_,required)=virtuals.iter().find(|(index,set)|set.len()<=cap && reaches(*index)).ok_or_else(||invalid("no complete virtual symmetry multiplet fits the requested ground-state active space"))?;
            chosen = required.to_vec();
            for (_, set) in &virtuals {
                if !set.iter().any(|i| chosen.contains(i)) && chosen.len() + set.len() <= cap {
                    chosen.extend_from_slice(set);
                }
            }
        }
        active.extend(chosen);
        active.sort_unstable();
        Ok(active)
    }
}

fn classify_molecular_point_group(
    points: &[Vector3<f64>],
    ops: &[symmetry::Operation],
    n_atoms: usize,
) -> (String, String) {
    if n_atoms == 1 {
        return (
            "K_h (spherical atom)".into(),
            "continuous SO(3) x C_i".into(),
        );
    }
    let tol = 1e-3;
    let outer = points
        .iter()
        .max_by(|a, b| a.norm_squared().total_cmp(&b.norm_squared()))
        .copied()
        .unwrap_or_else(Vector3::zeros);
    let has_inv = ops.iter().any(|r| (r + Matrix3::identity()).amax() < tol);
    if outer.norm() > tol
        && points
            .iter()
            .all(|p| outer.cross(p).norm() < tol * outer.norm())
    {
        let label = if has_inv { "D*h" } else { "C*v" };
        return (label.into(), format!("continuous axial ({label})"));
    }
    let order = ops.len();
    let proper: Vec<_> = ops.iter().filter(|r| r.determinant() > 0.).collect();
    let improper = order - proper.len();
    let mut n_max = 1usize;
    let mut c2_count = 0usize;
    for r in &proper {
        if (*r - Matrix3::identity()).amax() < tol {
            continue;
        }
        let cos_theta = ((r.trace() - 1.) * 0.5).clamp(-1., 1.);
        let angle = cos_theta.acos();
        if angle > tol {
            let fold = ((2. * std::f64::consts::PI) / angle).round() as usize;
            if fold >= 2 {
                n_max = n_max.max(fold);
            }
            if (angle - std::f64::consts::PI).abs() < 1e-2 {
                c2_count += 1;
            }
        }
    }
    let n_mirrors = ops
        .iter()
        .filter(|r| r.determinant() < 0. && (*r * *r - Matrix3::identity()).amax() < tol)
        .count();
    let label = match (order, improper > 0, has_inv) {
        (1, _, _) => "C1".to_string(),
        (2, true, true) => "Ci".to_string(),
        (2, true, false) => "Cs".to_string(),
        (120, true, true) => "Ih".to_string(),
        (60, false, false) => "I".to_string(),
        (48, true, true) => "Oh".to_string(),
        (24, true, false) => "Td".to_string(),
        (24, true, true) => "Th".to_string(),
        (24, false, false) => "O".to_string(),
        (12, false, false) if n_max == 3 => "T".to_string(),
        _ if proper.len() == order => {
            if c2_count >= n_max && order == 2 * n_max {
                format!("D{n_max}")
            } else {
                format!("C{n_max}")
            }
        }
        _ if order == 4 * n_max && has_inv && c2_count >= n_max => format!("D{n_max}h"),
        _ if order == 4 * n_max => format!("D{n_max}d"),
        _ if order == 2 * n_max && n_mirrors == 1 => format!("C{n_max}h"),
        _ if order == 2 * n_max && n_mirrors == n_max => format!("C{n_max}v"),
        _ if order == 2 * n_max => format!("S{}", 2 * n_max),
        _ => format!("order-{order}"),
    };
    (label, order.to_string())
}

/// Ordered key-value fields for the `[SYMMETRY]` log block of a finite molecular calculation.
pub fn molecular_symmetry_fields(
    positions_bohr: &[[f64; 3]],
    numbers: &[u8],
    moments: Option<&[[f64; 3]]>,
    magnetic: bool,
    spin_orbit: bool,
) -> Result<Vec<(String, String)>> {
    let n_atoms = positions_bohr.len();
    let (center, ops) = symmetry::operations(positions_bohr, numbers, 1e-4)?;
    let centered: Vec<Vector3<f64>> = positions_bohr
        .iter()
        .map(|p| Vector3::from(*p) - center)
        .collect();
    let (pg_symbol, order_text) = classify_molecular_point_group(&centered, &ops, n_atoms);
    let mut fields = vec![
        ("system_kind".into(), "molecule".into()),
        ("point_group".into(), pg_symbol.clone()),
        ("point_group_operations".into(), order_text.clone()),
        (
            "material_symmetry_discussion".into(),
            format!(
                "Molecule exhibits point group {pg_symbol} ({order_text} operations); molecular \
                 orbitals and degeneracies transform under its irreducible representations."
            ),
        ),
    ];

    let nonzero_moments = moments.is_some_and(|m| m.iter().flatten().any(|v| v.abs() > 1e-4));
    let mut n_unitary_mag = ops.len();
    let mut n_anti_mag = 0usize;
    if magnetic || nonzero_moments {
        let moment_vecs: Vec<[f64; 3]> = match moments {
            Some(m) if m.iter().flatten().any(|v| v.abs() > 1e-4) => m.to_vec(),
            _ => vec![[0., 0., 1.]; n_atoms],
        };
        let mut unitary_ops = Vec::new();
        let mut anti_count = 0usize;
        for r in &ops {
            let det = r.determinant();
            let mut atom_map = Vec::with_capacity(n_atoms);
            let mut valid = true;
            for (i, p) in centered.iter().enumerate() {
                let moved = *r * *p;
                let matched = centered
                    .iter()
                    .enumerate()
                    .find(|&(j, q)| numbers[j] == numbers[i] && (moved - q).norm() < 1e-3);
                match matched {
                    Some((j, _)) => atom_map.push(j),
                    None => {
                        valid = false;
                        break;
                    }
                }
            }
            if !valid {
                continue;
            }
            let check = |sign: f64| -> bool {
                (0..n_atoms).all(|a| {
                    let b = atom_map[a];
                    let turned = (*r * Vector3::from(moment_vecs[a])) * (det * sign);
                    (turned - Vector3::from(moment_vecs[b])).norm() <= 1e-4
                })
            };
            if check(1.) {
                unitary_ops.push(*r);
            } else if check(-1.) {
                anti_count += 1;
            }
        }
        n_unitary_mag = unitary_ops.len().max(1);
        n_anti_mag = anti_count;
        let (u_pg, _) = if unitary_ops.is_empty() {
            ("C1".into(), "1".into())
        } else {
            classify_molecular_point_group(&centered, &unitary_ops, n_atoms)
        };
        let (group_type, tr_text) = if n_anti_mag > 0 {
            (
                "Type III black-white magnetic point group (anti-unitary T*g cosets preserved)",
                "preserved via anti-unitary T*g operations",
            )
        } else {
            (
                "Type I unitary magnetic point group (time reversal T broken)",
                "broken",
            )
        };
        fields.push(("magnetic_symmetry".into(), "present".into()));
        fields.push(("magnetic_group_type".into(), group_type.into()));
        fields.push(("magnetic_point_group".into(), u_pg.clone()));
        fields.push((
            "magnetic_operations".into(),
            format!("{n_unitary_mag} unitary, {n_anti_mag} anti-unitary"),
        ));
        fields.push(("time_reversal".into(), tr_text.into()));
        fields.push((
            "magnetic_symmetry_discussion".into(),
            format!(
                "Spin polarization/magnetization lowers the molecular symmetry from {pg_symbol} to \
                 {group_type} (unitary subgroup {u_pg}, {n_unitary_mag} unitary and {n_anti_mag} \
                 anti-unitary operations; time reversal is {tr_text})."
            ),
        ));
    } else {
        fields.push((
            "time_reversal".into(),
            "preserved (Type II nonmagnetic symmetry)".into(),
        ));
    }

    if spin_orbit {
        let dg_u = if n_atoms == 1 || pg_symbol.starts_with("D*") || pg_symbol.starts_with("C*") {
            if !magnetic && !nonzero_moments {
                "continuous SU(2) double cover (bar_E present)".to_string()
            } else {
                format!(
                    "axial magnetic double cover ({} unitary)",
                    2 * n_unitary_mag
                )
            }
        } else {
            let a_str = if !magnetic && !nonzero_moments {
                format!("{}", 2 * n_unitary_mag)
            } else {
                format!("{}", 2 * n_anti_mag)
            };
            format!("{} unitary, {} anti-unitary", 2 * n_unitary_mag, a_str)
        };
        fields.push((
            "spin_orbit_double_group".into(),
            format!("present (SU(2) double cover of {pg_symbol} with bar_E = (E, -1))"),
        ));
        fields.push(("double_group_order".into(), dg_u.clone()));
        fields.push((
            "double_group_discussion".into(),
            format!(
                "Spin-orbit coupling lifts molecular point group {pg_symbol} to its SU(2) double \
                 group ({dg_u}), introducing the 2*pi spin-rotation bar_E. Spinor eigenstates \
                 transform in double-valued half-integer irreps (chi(bar_E) = -d), splitting \
                 spin-orbit coupled levels (j = l +- 1/2) and preserving Kramers degeneracy when \
                 time-reversal or anti-unitary T*g holds."
            ),
        ));
    }

    Ok(fields)
}
