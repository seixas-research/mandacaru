//! The `kappa`-resolved channels of a fully relativistic dataset.
//!
//! A Dirac dataset stores, next to its `j`-averaged channels, one spin-orbit entry per `l >= 1`:
//! the projectors of the two `j` branches (`j = l - 1/2`, `kappa = l`, first; `j = l + 1/2`,
//! `kappa = -(l + 1)`, second) and the `(2j + 1)`-weighted average and `L.S` difference of
//! their screened couplings and overlap corrections, each a block-diagonal matrix over the
//! union of the projectors. Every branch is an ordinary channel of the diagnostics once its
//! own `D` and `q` are recovered from those blocks:
//!
//! ```text
//! down = -difference_dd / f,   up = difference_uu / f,   f = 2 / (2 l + 1),
//! ```
//! (and `average_dd = down w_down / (w_down + w_up)`, the same matrix: [`recovery_mismatch`]
//! measures how well the two recoveries agree). The branches are unitary by construction, so
//! their norm correction is their overlap correction.
use super::view::{Channel, Key, Smooth};
use crate::{
    paw::{invalid, record::PawDataset},
    Context, Result,
};
use std::collections::BTreeMap;

fn block(matrix: &[f64], size: usize, offset: usize, half: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(half * half);
    for i in 0..half {
        for j in 0..half {
            out.push(matrix[(offset + i) * size + offset + j]);
        }
    }
    out
}

/// `kappa` of the `j = l - 1/2` and `j = l + 1/2` branches.
pub fn kappas(l: usize) -> [i32; 2] {
    [l as i32, -(l as i32 + 1)]
}

/// The two branches of every spin-orbit entry of the record as an own channel set (keyed by
/// `(l, Some(kappa))`), on the record's table and with its screened local potential.
pub fn branches(ctx: &Context, ds: &PawDataset) -> Result<Smooth> {
    let scalar = Smooth::scalar(ctx, ds)?;
    let n = ds.points();
    let mut channels = BTreeMap::new();
    for (&l, so) in &ds.spin_orbit {
        let l = l as usize;
        let base = scalar.channel(l).ok_or_else(|| {
            invalid(format!(
                "diagnose: the spin-orbit entry l={l} has no j-averaged channel"
            ))
        })?;
        let union = so.projectors.len();
        if union == 0 || union % 2 != 0 {
            return Err(invalid(format!(
                "diagnose: the spin-orbit entry l={l} has {union} projectors, expected two equal branches"
            )));
        }
        let half = union / 2;
        let size = union;
        for (name, array) in [
            ("coupling_screened", &so.coupling_screened),
            ("overlap_coupling", &so.overlap_coupling),
        ] {
            if array.shape() != [size, size] {
                return Err(invalid(format!(
                    "diagnose: spin-orbit l={l} {name} has shape {:?}, expected {size} x {size}",
                    array.shape()
                )));
            }
        }
        if so.reference_energies.shape().first() != Some(&2) {
            return Err(invalid(format!(
                "diagnose: spin-orbit l={l} reference_energies must hold the two branches"
            )));
        }
        let factor = 2. / (2 * l + 1) as f64;
        let per = so
            .reference_energies
            .shape()
            .iter()
            .skip(1)
            .product::<usize>();
        for (b, &kappa) in kappas(l).iter().enumerate() {
            let sign = if b == 0 { -1. } else { 1. };
            let offset = b * half;
            let scaled = |m: &[f64]| {
                block(m, size, offset, half)
                    .into_iter()
                    .map(|v| sign * v / factor)
                    .collect::<Vec<_>>()
            };
            let overlap = scaled(so.overlap_coupling.data());
            let mut projectors = ctx.allocate::<f64>(half * n)?;
            for k in 0..half {
                projectors.as_mut_slice()[k * n..(k + 1) * n]
                    .copy_from_slice(so.projectors[offset + k].as_slice());
            }
            let copy = |waves: &[crate::Buffer<f64>]| -> Result<Vec<crate::Buffer<f64>>> {
                waves.iter().map(|w| ctx.copy_slice(w.as_slice())).collect()
            };
            let channel = Channel {
                l,
                kappa: Some(kappa),
                r_cut: base.r_cut,
                energies: so.reference_energies.data()[b * per..(b + 1) * per].to_vec(),
                occupations: base.occupations.clone(),
                rank: half,
                projectors,
                coupling: scaled(so.coupling_screened.data()),
                norm: overlap.clone(),
                overlap,
                pseudo_waves: copy(&base.pseudo_waves)?,
                ae_waves: copy(&base.ae_waves)?,
            };
            if channel.energies.len() != base.occupations.len() {
                return Err(invalid(format!(
                    "diagnose: spin-orbit l={l} has {} reference energies per branch, the channel {}",
                    channel.energies.len(),
                    base.occupations.len()
                )));
            }
            channels.insert(
                Key {
                    l,
                    kappa: Some(kappa),
                },
                channel,
            );
        }
    }
    Ok(Smooth {
        symbol: scalar.symbol.clone(),
        atomic_number: scalar.atomic_number,
        r: ctx.copy_slice(scalar.r.as_slice())?,
        v_local_screened: ctx.copy_slice(scalar.v_local_screened.as_slice())?,
        r_cut_local: scalar.r_cut_local,
        channels,
        frozen: scalar.frozen.clone(),
        semicore: scalar.semicore.clone(),
    })
}

/// `{l: [down, up]}`: each branch's lowest level less the Dirac atom's, recomputed from the
/// stored blocks (Hartree) -- the quantity a generator stores as `level_errors`.
pub fn level_errors(ctx: &Context, ds: &PawDataset) -> Result<BTreeMap<usize, [f64; 2]>> {
    let set = branches(ctx, ds)?;
    let mut out = BTreeMap::new();
    for (key, channel) in &set.channels {
        let slot = if key.kappa == Some(key.l as i32) {
            0
        } else {
            1
        };
        out.entry(key.l).or_insert([f64::NAN; 2])[slot] =
            super::spectrum::branch_level_error(ctx, &set, channel)?;
    }
    Ok(out)
}

/// Largest disagreement between the two ways to recover a branch's screened coupling from the
/// stored blocks (the difference block and the average block), relative to the largest entry.
pub fn recovery_mismatch(ds: &PawDataset) -> f64 {
    let mut worst = 0f64;
    for (&l, so) in &ds.spin_orbit {
        let l = l as usize;
        let size = so.projectors.len();
        let half = size / 2;
        let factor = 2. / (2 * l + 1) as f64;
        let (w_down, w_up) = (2. * l as f64, 2. * l as f64 + 2.);
        let total = w_down + w_up;
        let scale = so
            .coupling_screened
            .data()
            .iter()
            .fold(0f64, |m, v| m.max(v.abs()))
            .max(1e-300);
        for (b, weight) in [(0usize, w_down), (1, w_up)] {
            let sign = if b == 0 { -1. } else { 1. };
            let difference = block(so.coupling_screened.data(), size, b * half, half);
            let average = block(so.average_screened.data(), size, b * half, half);
            for (d, a) in difference.iter().zip(&average) {
                let from_difference = sign * d / factor;
                let from_average = a * total / weight;
                worst = worst.max((from_difference - from_average).abs() / scale);
            }
        }
    }
    worst
}
