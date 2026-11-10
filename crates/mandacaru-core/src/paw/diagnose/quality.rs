//! Verification gates beside the generator's recording rule: properties of a dataset that the
//! search does not act on (so adding a gate here changes no generated dataset) but that decide
//! whether a dataset is fit for chemistry.
//!
//! - [`unprojected_missing_states`]: a bound all-electron valence level in a channel *without*
//!   projectors that the local potential alone does not reproduce (Gd's 5d at -0.05 Ha in a
//!   dataset with s and f channels only). [`super::ghosts::missing_states`] only sees channels
//!   that have projectors.
//! - [`hardness`]: the residual kinetic energy above `q_cut` that each channel's smooth partial
//!   waves keep; a large residual is a wave the real-space grid cannot converge at practical
//!   spacings. [`grid_residual`] is the same measure at the wave vector a real-space grid of a
//!   given spacing resolves, weighted by the occupations: the kinetic energy per atom the grid
//!   cannot represent, an estimate of its total-energy error that needs no SCF.
//! - [`sphere_reach`]: the widest augmentation sphere against a bond length, so a neighbor's
//!   nucleus inside the sphere is reported.
//! - [`partition_mismatch`]: the core/valence partition of two datasets of one element (the
//!   scalar-relativistic and the fully relativistic set) must be the same, or their difference
//!   is not a spin-orbit effect.
//! - [`unrecorded_defects`]: what a fresh diagnosis finds that the record does not store (a
//!   stale file: pbe-fr Au's f phase of 0.445 rad).
use super::{
    ghosts::MISSING_TOLERANCE, reference::AllElectron, view::Smooth, Diagnosis, Error, Outcome,
};
use crate::{
    paw::{
        checks::{bound_levels, interpolate_linear},
        invalid,
        pseudize_tables::{inner_grid, q_grid, tail_transform, transform_table, Q_STEP},
        record::{Defects, PawDataset},
    },
    Context, Result,
};
use std::collections::BTreeMap;

/// A channel whose smooth partial waves keep more than this residual kinetic energy above
/// `q_cut` (Hartree) is hard. Measured on the shipped sets (2026-11-09 audit): O 2p keeps 0.017
/// Ha and its molecules converge at 0.2 Angstrom; Cl converges too (HCl moves 5 meV between 0.20
/// and 0.15 Angstrom). Br d keeps 8.2 Ha and HBr moves 186 eV over the same step; Xe 4d keeps
/// 2.4 Ha and sits 3.1 / 2.9 / 1.6 eV off its level at 0.2 / 0.15 / 0.1 Angstrom. One Hartree,
/// sixty times the oxygen value, separates every converging case from every failing one.
pub const HARDNESS_TOLERANCE: f64 = 1.0;

/// Levels of an unprojected channel shallower than this (Hartree) are not counted: within a few
/// millihartree of the continuum a level depends on the radial box, not on the atom.
pub const UNPROJECTED_LEVEL_FLOOR: f64 = -5e-3;

/// `{l: offset}` for every angular momentum without projectors (up to one above the highest
/// constructed channel) where the all-electron atom binds a valence level that the screened local
/// potential alone does not reproduce within [`MISSING_TOLERANCE`]. The offset is the smooth
/// level less the all-electron one, or, when the local potential binds nothing there, the
/// distance from the all-electron level up to the continuum (`-eps_ae`, positive). Both spectra
/// are counted on the reference atom's uniform grid, the core shells of that `l` excluded.
pub fn unprojected_missing_states(
    smooth: &Smooth,
    ae: &AllElectron<'_>,
) -> Outcome<BTreeMap<usize, f64>> {
    let r = ae.atom.r.as_slice();
    let local = interpolate_linear(r, smooth.r.as_slice(), smooth.v_local_screened.as_slice())?;
    let constructed = smooth.angular_momenta();
    let highest = constructed.iter().copied().max().unwrap_or(0);
    let levels = |v: &[f64], what: &str, l: usize| -> Outcome<Vec<f64>> {
        bound_levels(r, v, l, None, 0.).map_err(|e| {
            Error::Unevaluated(format!(
                "{}: {what} (l={l}) is not finite: {e}",
                smooth.symbol
            ))
        })
    };
    let mut out = BTreeMap::new();
    for l in 0..highest + 2 {
        if constructed.contains(&l) {
            continue;
        }
        let atom = levels(
            ae.atom.effective.as_slice(),
            "the all-electron potential",
            l,
        )?;
        let valence = atom
            .get(ae.core_count(l)..)
            .unwrap_or(&[])
            .iter()
            .copied()
            .filter(|&e| e < UNPROJECTED_LEVEL_FLOOR)
            .collect::<Vec<_>>();
        let Some(&wanted) = valence.first() else {
            continue;
        };
        let ps = levels(&local, "the screened local potential", l)?;
        let offset = ps
            .iter()
            .map(|&e| e - wanted)
            .min_by(|a, b| a.abs().total_cmp(&b.abs()))
            .unwrap_or(-wanted);
        if !(offset.abs() <= MISSING_TOLERANCE) {
            out.insert(l, offset);
        }
    }
    Ok(out)
}

/// `{l: residual}`: the largest residual kinetic energy above `q_cut` (Hartree) of the smooth
/// partial waves of each scalar channel's occupied references, as the generator stored it. Those
/// waves carry the valence density. A scattering reference only shapes the second projector and is
/// harder by construction (O's at +1 Ha keeps 0.12 Ha where its 2p keeps 0.02), and an empty
/// channel (hydrogen's p, a frozen shell's f) carries no electron: both count as 0 here.
pub fn residual_kinetic(dataset: &PawDataset) -> BTreeMap<usize, f64> {
    dataset
        .channels
        .iter()
        .map(|(&l, c)| {
            let worst = c
                .occupations
                .data()
                .iter()
                .zip(c.residual_kinetic.data())
                .filter(|(q, _)| **q > 0.)
                .map(|(_, &e)| e)
                .fold(0f64, |a, b| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                });
            (l as usize, worst)
        })
        .collect()
}

/// The channels of [`residual_kinetic`] above [`HARDNESS_TOLERANCE`]; a residual that is not a
/// number is hard.
pub fn hardness(dataset: &PawDataset) -> BTreeMap<usize, f64> {
    residual_kinetic(dataset)
        .into_iter()
        .filter(|(_, e)| !(*e <= HARDNESS_TOLERANCE))
        .collect()
}

/// The kinetic energy (Hartree) the smooth partial wave `index` of channel `l` carries above the
/// wave vector `q_min` (Bohr^-1): `1/2 int_{q >= q_min} q^4 |F(q)|^2 dq`, with `F` the Bessel
/// transform of the stored expansion inside the sphere plus that of the all-electron tail outside,
/// on the generator's own Fourier grid. At `q_min = q_cut` it is the `residual_kinetic` the
/// generator stored (the test of this function).
pub fn kinetic_above(
    ctx: &Context,
    dataset: &PawDataset,
    l: usize,
    index: usize,
    q_min: f64,
) -> Result<f64> {
    let channel = dataset
        .channels
        .get(&(l as u32))
        .ok_or_else(|| invalid(format!("the dataset has no l={l} channel")))?;
    let (Some(qs), Some(coefficients), Some(wave)) = (
        channel.wavevectors.get(index),
        channel.wave_coefficients.get(index),
        channel.ae_waves.get(index),
    ) else {
        return Err(invalid(format!(
            "the l={l} channel has no partial wave {index}"
        )));
    };
    let r = dataset.r.as_slice();
    let n = r.len();
    let wave = wave.as_slice();
    let energy = channel.reference_energies.data()[index];
    let bound = energy < 0. && (wave[n - 1] * r[n - 1]).abs() < 1e-6;
    let q = q_grid();
    let inner = transform_table(l, qs.data(), &inner_grid(channel.r_cut), &q)?;
    let tail = tail_transform(ctx, l, r, wave, channel.r_cut, bound, &q)?;
    let nq = q.len();
    let mut total = 0.;
    for k in 0..nq {
        if q[k] < q_min {
            continue;
        }
        let f = coefficients
            .data()
            .iter()
            .enumerate()
            .map(|(a, c)| c * inner[a * nq + k])
            .sum::<f64>()
            + tail[k];
        total += 0.5 * q[k].powi(4) * f * f * Q_STEP;
    }
    Ok(total)
}

/// Bohr per Angstrom.
const BOHR_PER_ANGSTROM: f64 = 1. / 0.529_177_210_903;

/// `{l: sum_i occupation_i * kinetic_above(pi / spacing)}` over the occupied bound references of
/// each channel, for a real-space grid of `spacing` Angstrom: the kinetic energy (Hartree per atom)
/// of the valence the grid cannot resolve.
pub fn grid_residual(
    ctx: &Context,
    dataset: &PawDataset,
    spacing: f64,
) -> Result<BTreeMap<usize, f64>> {
    let q_grid_limit = std::f64::consts::PI / (spacing * BOHR_PER_ANGSTROM);
    let mut out = BTreeMap::new();
    for (&l, channel) in &dataset.channels {
        let mut sum = 0.;
        for (index, &occupation) in channel.occupations.data().iter().enumerate() {
            if occupation > 0. {
                sum += occupation * kinetic_above(ctx, dataset, l as usize, index, q_grid_limit)?;
            }
        }
        out.insert(l as usize, sum);
    }
    Ok(out)
}

/// The grid residual (Hartree per atom) a dataset may leave at its working spacing. Calibrated on
/// the measured outcomes of the 2026-11-09 audit: predicted from the shipped pbe-sr files, Cl
/// changes by 2.3e-4 Ha (6 meV) between 0.20 and 0.15 Angstrom where HCl moved 5 meV, and Br by
/// 3.1 Ha (85 eV) where HBr moved 186 eV; O leaves 9.0 mHa at 0.15 and 0.9 mHa at 0.12.
pub const GRID_TOLERANCE: f64 = 1e-2;
/// The spacings (Angstrom) a working spacing is chosen from, coarsest first.
pub const GATE_SPACINGS: [f64; 4] = [0.2, 0.15, 0.12, 0.1];

/// The coarsest of [`GATE_SPACINGS`] at which the [`grid_residual`] summed over the channels is
/// within [`GRID_TOLERANCE`]; `None` when even 0.10 Angstrom is not.
pub fn working_spacing(ctx: &Context, dataset: &PawDataset) -> Result<Option<f64>> {
    for h in GATE_SPACINGS {
        let total = grid_residual(ctx, dataset, h)?.values().sum::<f64>();
        if total <= GRID_TOLERANCE {
            return Ok(Some(h));
        }
    }
    Ok(None)
}

/// The widest augmentation sphere (Bohr) against a bond of `bond` Bohr: `Some(r_cut_max)` when
/// the neighbor's nucleus lies inside it. The overlap of two spheres is allowed by the method; a
/// nucleus inside a sphere is not, because the one-center expansion then holds the neighbor's
/// cusp (the La/Ac and CuH precedents).
pub fn sphere_reach(dataset: &PawDataset, bond: f64) -> Option<f64> {
    let widest = dataset
        .channels
        .values()
        .map(|c| c.r_cut)
        .fold(dataset.r_cut_local, f64::max);
    (widest >= bond).then_some(widest)
}

/// The core/valence partition of a dataset: its valence charge and the `(n, l)` of the frozen and
/// semicore subshells, sorted.
#[derive(Debug, Clone, PartialEq)]
pub struct Partition {
    pub valence_charge: f64,
    pub frozen: Vec<Vec<i64>>,
    pub semicore: Vec<Vec<i64>>,
    pub channels: Vec<u32>,
}

impl Partition {
    pub fn of(dataset: &PawDataset) -> Self {
        let sorted = |list: &[Vec<i64>]| {
            let mut v = list.to_vec();
            v.sort();
            v
        };
        Self {
            valence_charge: dataset.valence_charge,
            frozen: sorted(&dataset.frozen_subshells),
            semicore: sorted(&dataset.semicore_subshells),
            channels: dataset.channels.keys().copied().collect(),
        }
    }
}

/// `Some(description)` when two datasets of one element split core and valence differently (a
/// scalar-relativistic set against its fully relativistic twin: "fr - sr" is a spin-orbit effect
/// only when the partitions agree).
pub fn partition_mismatch(a: &PawDataset, b: &PawDataset) -> Option<String> {
    let (pa, pb) = (Partition::of(a), Partition::of(b));
    if a.symbol != b.symbol {
        return Some(format!(
            "{} and {} are different elements",
            a.symbol, b.symbol
        ));
    }
    (pa != pb).then(|| {
        format!(
            "{}: {} {} has {} valence electrons, frozen {:?}, semicore {:?}, channels {:?}; {} {} has {}, frozen {:?}, semicore {:?}, channels {:?}",
            a.symbol,
            a.xc,
            a.relativity,
            pa.valence_charge,
            pa.frozen,
            pa.semicore,
            pa.channels,
            b.xc,
            b.relativity,
            pb.valence_charge,
            pb.frozen,
            pb.semicore,
            pb.channels
        )
    })
}

/// What `fresh` (a diagnosis of the record against today's engine, in the record's format)
/// holds that the record's stored defects do not: each map keeps the entries whose key the
/// stored one lacks, and `s_miss` when the record stores none.
pub fn unrecorded_defects(stored: &Defects, fresh: &Defects) -> Defects {
    fn extra<K: Ord + Clone, V: Clone>(
        fresh: &BTreeMap<K, V>,
        stored: &BTreeMap<K, V>,
    ) -> BTreeMap<K, V> {
        fresh
            .iter()
            .filter(|(k, _)| !stored.contains_key(*k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
    Defects {
        ghosts: extra(&fresh.ghosts, &stored.ghosts),
        phases: extra(&fresh.phases, &stored.phases),
        residuals: extra(&fresh.residuals, &stored.residuals),
        s_miss: fresh.s_miss.filter(|_| stored.s_miss.is_none()),
        unprojected: extra(&fresh.unprojected, &stored.unprojected),
        missing: extra(&fresh.missing, &stored.missing),
        branch_ghosts: extra(&fresh.branch_ghosts, &stored.branch_ghosts),
    }
}

/// Every verification gate of this module for one dataset, with the diagnosis it extends.
#[derive(Debug, Clone, Default)]
pub struct Quality {
    /// [`unprojected_missing_states`].
    pub unprojected_missing: BTreeMap<usize, f64>,
    /// [`hardness`].
    pub hard: BTreeMap<usize, f64>,
    /// [`sphere_reach`] against the bond the caller named, when it named one.
    pub sphere: Option<f64>,
    /// [`unrecorded_defects`] of the diagnosis against the record.
    pub unrecorded: Defects,
}

impl Quality {
    /// Evaluate the gates of `dataset` with the `diagnosis` already run against `ae`; `bond` is
    /// a typical bond length to a neighbor (Bohr), when the caller has one.
    pub fn run(
        dataset: &PawDataset,
        smooth: &Smooth,
        ae: &AllElectron<'_>,
        diagnosis: &Diagnosis,
        bond: Option<f64>,
    ) -> Outcome<Self> {
        Ok(Self {
            unprojected_missing: unprojected_missing_states(smooth, ae)?,
            hard: hardness(dataset),
            sphere: bond.and_then(|b| sphere_reach(dataset, b)),
            unrecorded: unrecorded_defects(&dataset.defects, &diagnosis.defects()),
        })
    }

    /// No gate of this module fires.
    pub fn is_clean(&self) -> bool {
        self.unprojected_missing.is_empty()
            && self.hard.is_empty()
            && self.sphere.is_none()
            && self.unrecorded == Defects::default()
    }
}
