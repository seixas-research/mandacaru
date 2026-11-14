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

use super::{provider::Operators, pulay::Matrix, shapes::Channel, spheres::Projector};
use crate::{
    paw::{
        multipole::{gaunt, gaunt_grid},
        multipole_pairs::{
            pair_cut, radial_moments, radial_remainder, smooth_waves, ChannelWaves, Pair,
        },
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
    let total = product(&[channels.len(), p, p])?;
    let mut out = ctx.allocate::<Complex64>(total)?;
    let matrices = multipole_matrices(projectors, datasets, channels)?;
    let slice = out.as_mut_slice();
    for (a, mat) in matrices.iter().enumerate() {
        let offset = a * p * p;
        for i in 0..p {
            for j in 0..p {
                slice[offset + i * p + j] = mat[(i, j)];
            }
        }
    }
    Ok(out)
}

/// Store already assembled multipole matrices in the crystal's row-major layout.
pub fn store_multipole_blocks(
    ctx: &Context,
    matrices: &[Matrix],
    p: usize,
) -> Result<Buffer<Complex64>> {
    if matrices.iter().any(|m| m.nrows() != p || m.ncols() != p) {
        return Err(crate::periodic::invalid(
            "multipole block dimensions do not match projectors",
        ));
    }
    let total = product(&[matrices.len(), p, p])?;
    let mut out = ctx.allocate::<Complex64>(total)?;
    let slice = out.as_mut_slice();

    for (a, mat) in matrices.iter().enumerate() {
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
    channel_matrices(projectors, datasets, channels, calculate_radial_moments)
}

/// The one-center remainder of the plane wave `e^{-i b.(r - R_a)}` beyond its leading order, one
/// `P x P` matrix per channel `(atom, L, M)`:
///
/// ```text
/// R^{LM}_{pa,pb}(|b|) = [int (R_i R_j - R~_i R~_j) (j_L(|b| r) - (|b| r)^L / (2L+1)!!) r^2 dr]
///                       * int Y*_{LM} Y*_{l_a m_a} Y_{l_b m_b} dOmega
/// ```
///
/// With the multipole blocks `B^{LM}` (whose `L = 0` diagonal is the dataset's overlap
/// correction) the exact one-center matrix of the plane wave is
/// `sum_LM 4 pi (-i)^L Y_LM(b^) [|b|^L / (2L+1)!! B^{LM} + R^{LM}(|b|)]`
/// (`wannier::sampling::Augmentation::operator`). In the raw projector basis (`dual` false) the
/// matrices carry the same Vanderbilt transform as the blocks.
pub fn plane_wave_remainders(
    ctx: &Context,
    projectors: &[Projector],
    datasets: &[Arc<PawDataset>],
    channels: &[Channel],
    dual: bool,
    length: f64,
) -> Result<Vec<Matrix>> {
    if !length.is_finite() || length < 0. {
        return Err(crate::periodic::invalid(
            "a plane-wave remainder needs a finite nonnegative wave vector",
        ));
    }
    let mut matrices = channel_matrices(projectors, datasets, channels, |ds, l1, l2, big_l| {
        calculate_radial_remainder(ctx, ds, l1, l2, big_l, length)
    })?;
    if !dual {
        to_raw_basis(matrices.iter_mut(), projectors, datasets)?;
    }
    Ok(matrices)
}

/// One `P x P` matrix per channel `(atom, L, M)`: the radial table `radial(dataset, l_a, l_b, L)`
/// (rows and columns the radial indices of the two projector channels) times the complex Gaunt
/// coefficient `int Y*_LM Y*_{l_a m_a} Y_{l_b m_b}`.
fn channel_matrices(
    projectors: &[Projector],
    datasets: &[Arc<PawDataset>],
    channels: &[Channel],
    mut radial: impl FnMut(&PawDataset, u32, u32, usize) -> Result<DMatrix<f64>>,
) -> Result<Vec<Matrix>> {
    if projectors.iter().any(|p| p.atom >= datasets.len())
        || channels.iter().any(|&(atom, _, _)| atom >= datasets.len())
    {
        return Err(crate::periodic::invalid(
            "projector and multipole operators need one dataset per atom",
        ));
    }
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
                    let calculated = radial(ds, la as u32, lb as u32, big_l)?;
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

/// The all-electron and smooth radial waves of two projector channels of a dataset, with the
/// augmentation radius of the pair: `None` when a channel is missing or empty.
struct PairWaves {
    ae: [Vec<f64>; 2],
    smooth: [Vec<f64>; 2],
    r_cut: f64,
    waves: [usize; 2],
}

impl PairWaves {
    fn pair<'a>(&'a self, r: &'a [f64]) -> Pair<'a> {
        Pair {
            r,
            r_cut: self.r_cut,
            ae: [&self.ae[0], &self.ae[1]],
            smooth: [&self.smooth[0], &self.smooth[1]],
            waves: self.waves,
        }
    }
}

fn pair_waves(ds: &PawDataset, l1: u32, l2: u32) -> Result<Option<PairWaves>> {
    let (Some(ch1), Some(ch2)) = (ds.channels.get(&l1), ds.channels.get(&l2)) else {
        return Ok(None);
    };
    let (n1, n2) = (ch1.ae_waves.len(), ch2.ae_waves.len());
    if n1 == 0 || n2 == 0 {
        return Ok(None);
    }
    let r = ds.r.as_slice();
    let flatten = |ch: &crate::paw::record::Channel| -> Vec<f64> {
        ch.ae_waves
            .iter()
            .flat_map(|w| w.as_slice().iter().copied())
            .collect()
    };
    let reconstruct = |ch: &crate::paw::record::Channel, l: u32, ae: &[f64]| -> Result<Vec<f64>> {
        let coefficients: Vec<f64> = ch
            .wave_coefficients
            .iter()
            .flat_map(|a| a.data().iter().copied())
            .collect();
        let wavevectors: Vec<f64> = ch
            .wavevectors
            .iter()
            .flat_map(|a| a.data().iter().copied())
            .collect();
        smooth_waves(
            r,
            &ChannelWaves {
                l: l as usize,
                r_cut: ch.r_cut,
                ae,
                coefficients: &coefficients,
                wavevectors: &wavevectors,
                waves: ch.ae_waves.len(),
            },
        )
    };
    let (ae1, ae2) = (flatten(ch1), flatten(ch2));
    let (ps1, ps2) = (reconstruct(ch1, l1, &ae1)?, reconstruct(ch2, l2, &ae2)?);
    Ok(Some(PairWaves {
        ae: [ae1, ae2],
        smooth: [ps1, ps2],
        r_cut: pair_cut(ch1.r_cut, ch2.r_cut),
        waves: [n1, n2],
    }))
}

fn table(values: &[f64], rows: usize, columns: usize) -> DMatrix<f64> {
    DMatrix::from_fn(rows, columns, |i, j| {
        values.get(i * columns + j).copied().unwrap_or(0.)
    })
}

/// `int (R_i R_j - R~_i R~_j) (j_L(q r) - (q r)^L / (2L+1)!!) r^2 dr`: the radial part of the
/// plane wave beyond its leading order (zero at `q = 0`).
fn calculate_radial_remainder(
    ctx: &Context,
    ds: &PawDataset,
    l1: u32,
    l2: u32,
    big_l: usize,
    q: f64,
) -> Result<DMatrix<f64>> {
    let Some(waves) = pair_waves(ds, l1, l2)? else {
        let n = |l: u32| ds.channels.get(&l).map_or(0, |c| c.ae_waves.len());
        return Ok(DMatrix::zeros(n(l1), n(l2)));
    };
    let pair = waves.pair(ds.r.as_slice());
    let moments = radial_moments(&pair, big_l)?;
    let double_factorial = (1..=big_l).map(|k| (2 * k + 1) as f64).product::<f64>();
    let leading = q.powi(big_l as i32) / double_factorial;
    let values = radial_remainder(ctx, &pair, big_l, &[q], &moments, &[leading])?;
    Ok(table(&values, waves.waves[0], waves.waves[1]))
}

fn calculate_radial_moments(
    ds: &PawDataset,
    l1: u32,
    l2: u32,
    big_l: usize,
) -> Result<DMatrix<f64>> {
    let (Some(ch1), Some(ch2)) = (ds.channels.get(&l1), ds.channels.get(&l2)) else {
        return Ok(DMatrix::zeros(0, 0));
    };
    let (n1, n2) = (ch1.ae_waves.len(), ch2.ae_waves.len());
    if l1 == l2 && big_l == 0 && n1 > 0 {
        return Ok(table(ch1.overlap_correction.data(), n1, n2));
    }
    let Some(waves) = pair_waves(ds, l1, l2)? else {
        return Ok(DMatrix::zeros(n1, n2));
    };
    let moments = radial_moments(&waves.pair(ds.r.as_slice()), big_l)?;
    Ok(table(&moments, n1, n2))
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

/// Assemble the same operators in the raw projector basis. Each radial block
/// is transformed independently, so no additional `P x P` matrix is needed.
pub fn raw_operators(
    projectors: &[Projector],
    datasets: &[Arc<PawDataset>],
    channels: &[Channel],
) -> Result<Operators> {
    let mut result = operators(projectors, datasets, channels)?;
    to_raw_basis(
        std::iter::once(&mut result.augmentation)
            .chain(std::iter::once(&mut result.ion_operator))
            .chain(result.blocks.iter_mut()),
        projectors,
        datasets,
    )?;
    Ok(result)
}

/// Move dual-basis `P x P` matrices to the raw projector basis: every same-atom block `A`
/// becomes `V_a^{-1} A V_b^{-T}` with the channels' Vanderbilt matrices.
pub(crate) fn to_raw_basis<'m>(
    matrices: impl Iterator<Item = &'m mut Matrix>,
    projectors: &[Projector],
    datasets: &[Arc<PawDataset>],
) -> Result<()> {
    let mut groups: HashMap<(usize, usize, i32), Vec<(usize, usize)>> = HashMap::new();
    for (column, projector) in projectors.iter().enumerate() {
        groups
            .entry((projector.atom, projector.l, projector.m))
            .or_default()
            .push((column, projector.index));
    }
    let mut transforms = Vec::with_capacity(groups.len());
    for ((atom, l, _), members) in groups {
        let channel = datasets[atom]
            .channels
            .get(&(l as u32))
            .ok_or_else(|| crate::periodic::invalid("projector channel is missing"))?;
        let n = channel.projectors.len();
        if members.len() != n || channel.vanderbilt.data().len() != n * n {
            return Err(crate::periodic::invalid("incomplete projector channel"));
        }
        let mut ordered = vec![usize::MAX; n];
        for (column, index) in members {
            if index >= n || ordered[index] != usize::MAX {
                return Err(crate::periodic::invalid("invalid projector radial index"));
            }
            ordered[index] = column;
        }
        let inverse = DMatrix::from_row_slice(n, n, channel.vanderbilt.data())
            .try_inverse()
            .ok_or_else(|| crate::periodic::invalid("singular Vanderbilt matrix"))?;
        transforms.push((atom, ordered, inverse));
    }
    for matrix in matrices {
        for (atom_a, columns_a, inv_a) in &transforms {
            for (atom_b, columns_b, inv_b) in &transforms {
                if atom_a != atom_b {
                    continue;
                }
                let mut block = DMatrix::zeros(columns_a.len(), columns_b.len());
                for (i, &a) in columns_a.iter().enumerate() {
                    for (j, &b) in columns_b.iter().enumerate() {
                        block[(i, j)] = matrix[(a, b)];
                    }
                }
                let a = inv_a.map(|x| Complex64::new(x, 0.));
                let b = inv_b.map(|x| Complex64::new(x, 0.));
                let transformed = a * block * b.transpose();
                for (i, &a) in columns_a.iter().enumerate() {
                    for (j, &b) in columns_b.iter().enumerate() {
                        matrix[(a, b)] = transformed[(i, j)];
                    }
                }
            }
        }
    }
    Ok(())
}
