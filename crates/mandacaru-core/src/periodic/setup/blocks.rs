//! Complex-harmonic multipole blocks and operator matrices for the periodic PAW setup.
//!
//! A channel `(atom, L, M)` carries a compensation multipole matrix `B_a` of size `P x P`.
//! Its entries between projectors `pa` and `pb` of the same atom are:
//!
//! ```text
//! B_{pa, pb} = Delta^{(L)}_{i, j} * int Y^*_{LM} Y^*_{l_a, m_a} Y_{l_b, m_b} dOmega
//! ```
//!
//! where `Delta^{(L)}` is the radial moment of the all-electron minus pseudo pair density,
//! and the angular integral is the complex Gaunt coefficient ([`crate::paw::multipole::gaunt`]).
//!
//! State-independent operator matrices `q_overlap` (the nonlocal overlap correction) and
//! `D_ion` (the ionic nonlocal coupling) are block-diagonal across the `(atom, l, m)` channels.

use super::{
    provider::Operators,
    pulay::Matrix,
    shapes::Channel,
    spheres::Projector,
};
use crate::{
    paw::{
        multipole::{gaunt, gaunt_grid},
        multipole_pairs::{radial_moments, Pair},
        record::PawDataset,
    },
    product, Buffer, Context, Result,
};
use nalgebra::DMatrix;
use num_complex::Complex64;
use std::{collections::HashMap, sync::Arc};

/// Build the `(channels, P, P)` multipole blocks buffer for [`crate::periodic::Crystal`].
pub fn multipole_blocks(
    ctx: &Context,
    projectors: &[Projector],
    datasets: &[Arc<PawDataset>],
    channels: &[Channel],
) -> Result<Buffer<Complex64>> {
    let p = projectors.len();
    let n_ch = channels.len();
    let total = product(&[n_ch, p, p])?;
    let mut out = ctx.allocate::<Complex64>(total)?;
    let slice = out.as_mut_slice();
    slice.fill(Complex64::new(0., 0.));

    let matrices = multipole_matrices(projectors, datasets, channels)?;
    for (a, mat) in matrices.into_iter().enumerate() {
        let offset = a * p * p;
        for i in 0..p {
            for j in 0..p {
                slice[offset + i * p + j] = mat[(i, j)];
            }
        }
    }
    Ok(out)
}

/// The `P x P` multipole matrix for each channel `(atom, L, M)` in `channels`.
pub fn multipole_matrices(
    projectors: &[Projector],
    datasets: &[Arc<PawDataset>],
    channels: &[Channel],
) -> Result<Vec<Matrix>> {
    let p = projectors.len();
    let grid = gaunt_grid()?;
    let mut matrices = Vec::with_capacity(channels.len());

    // Group projector global indices by atom
    let mut atom_projectors: Vec<Vec<usize>> = vec![Vec::new(); datasets.len()];
    for (idx, proj) in projectors.iter().enumerate() {
        if proj.atom < atom_projectors.len() {
            atom_projectors[proj.atom].push(idx);
        }
    }

    // Cache radial moments per (atom, l1, l2, L) -> DMatrix<f64>
    let mut moment_cache: HashMap<(usize, u32, u32, usize), DMatrix<f64>> = HashMap::new();

    for &(atom, big_l, m) in channels {
        let mut mat = Matrix::zeros(p, p);
        if atom >= datasets.len() {
            matrices.push(mat);
            continue;
        }
        let ds = &datasets[atom];
        let p_indices = &atom_projectors[atom];

        for &pa_idx in p_indices {
            let pa = &projectors[pa_idx];
            let la = pa.l;
            let ma = pa.m;
            let ia = pa.index;

            for &pb_idx in p_indices {
                let pb = &projectors[pb_idx];
                let lb = pb.l;
                let mb = pb.m;
                let ib = pb.index;

                let angular = gaunt([big_l, la, lb], [m, ma, mb], false, &grid);
                if angular.norm() < 1e-14 {
                    continue;
                }

                let key = (atom, la as u32, lb as u32, big_l);
                let radial_mat = if let Some(cached) = moment_cache.get(&key) {
                    cached
                } else {
                    let calculated = calculate_radial_moments(ds, la as u32, lb as u32, big_l)?;
                    moment_cache.entry(key).or_insert(calculated)
                };

                if ia < radial_mat.nrows() && ib < radial_mat.ncols() {
                    let val = radial_mat[(ia, ib)] * angular;
                    if val.norm() > 1e-15 {
                        mat[(pa_idx, pb_idx)] += val;
                    }
                }
            }
        }
        matrices.push(mat);
    }

    Ok(matrices)
}

fn calculate_radial_moments(
    ds: &PawDataset,
    l1: u32,
    l2: u32,
    big_l: usize,
) -> Result<DMatrix<f64>> {
    let ch1 = match ds.channels.get(&l1) {
        Some(c) => c,
        None => return Ok(DMatrix::zeros(0, 0)),
    };
    let ch2 = match ds.channels.get(&l2) {
        Some(c) => c,
        None => return Ok(DMatrix::zeros(0, 0)),
    };

    let n1 = ch1.ae_waves.len();
    let n2 = ch2.ae_waves.len();
    if n1 == 0 || n2 == 0 {
        return Ok(DMatrix::zeros(n1, n2));
    }

    if l1 == l2 && big_l == 0 {
        let overlap = ch1.overlap_correction.data();
        let mut mat = DMatrix::zeros(n1, n2);
        for i in 0..n1 {
            for j in 0..n2 {
                if i * n2 + j < overlap.len() {
                    mat[(i, j)] = overlap[i * n2 + j];
                }
            }
        }
        return Ok(mat);
    }

    let r = ds.r.as_slice();
    let nr = r.len();
    let mut ae1 = Vec::with_capacity(n1 * nr);
    let mut ps1 = Vec::with_capacity(n1 * nr);
    for w in &ch1.ae_waves {
        ae1.extend_from_slice(w.as_slice());
    }
    for w in &ch1.pseudo_waves {
        ps1.extend_from_slice(w.as_slice());
    }

    let mut ae2 = Vec::with_capacity(n2 * nr);
    let mut ps2 = Vec::with_capacity(n2 * nr);
    for w in &ch2.ae_waves {
        ae2.extend_from_slice(w.as_slice());
    }
    for w in &ch2.pseudo_waves {
        ps2.extend_from_slice(w.as_slice());
    }

    let pair = Pair {
        r,
        r_cut: ch1.r_cut.max(ch2.r_cut),
        ae: [&ae1, &ae2],
        smooth: [&ps1, &ps2],
        waves: [n1, n2],
    };

    let moments = radial_moments(&pair, big_l)?;
    let mut mat = DMatrix::zeros(n1, n2);
    for i in 0..n1 {
        for j in 0..n2 {
            if i * n2 + j < moments.len() {
                mat[(i, j)] = moments[i * n2 + j];
            }
        }
    }
    Ok(mat)
}

/// The `P x P` block-diagonal nonlocal overlap matrix `q_overlap`.
pub fn overlap_operator(projectors: &[Projector], datasets: &[Arc<PawDataset>]) -> Matrix {
    assemble_diagonal_blocks(projectors, datasets, |ch| ch.overlap_correction.data())
}

/// The `P x P` block-diagonal ionic nonlocal coupling matrix `D_ion`.
pub fn ion_operator(projectors: &[Projector], datasets: &[Arc<PawDataset>]) -> Matrix {
    assemble_diagonal_blocks(projectors, datasets, |ch| ch.coupling.data())
}

fn assemble_diagonal_blocks(
    projectors: &[Projector],
    datasets: &[Arc<PawDataset>],
    block_fn: impl Fn(&crate::paw::record::Channel) -> &[f64],
) -> Matrix {
    let p = projectors.len();
    let mut out = Matrix::zeros(p, p);

    // Group projectors by (atom, l, m)
    let mut groups: HashMap<(usize, usize, i32), Vec<(usize, usize)>> = HashMap::new();
    for (idx, proj) in projectors.iter().enumerate() {
        groups
            .entry((proj.atom, proj.l, proj.m))
            .or_default()
            .push((idx, proj.index));
    }

    for ((atom, l, _m), members) in groups {
        if atom >= datasets.len() {
            continue;
        }
        let ds = &datasets[atom];
        let ch = match ds.channels.get(&(l as u32)) {
            Some(c) => c,
            None => continue,
        };
        let table = block_fn(ch);
        let n = ch.projectors.len();

        for &(pa_idx, ia) in &members {
            for &(pb_idx, ib) in &members {
                if ia < n && ib < n && ia * n + ib < table.len() {
                    out[(pa_idx, pb_idx)] = Complex64::new(table[ia * n + ib], 0.);
                }
            }
        }
    }
    out
}

/// Assemble the complete `Operators` struct (`blocks`, `augmentation`, `ion_operator`).
pub fn operators(
    projectors: &[Projector],
    datasets: &[Arc<PawDataset>],
    channels: &[Channel],
) -> Result<Operators> {
    let blocks = multipole_matrices(projectors, datasets, channels)?;
    let augmentation = overlap_operator(projectors, datasets);
    let ion_operator = ion_operator(projectors, datasets);
    Ok(Operators {
        blocks,
        augmentation,
        ion_operator,
    })
}
