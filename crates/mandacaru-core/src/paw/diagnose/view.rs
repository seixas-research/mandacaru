//! What the diagnostics read of a dataset: the smooth side of every channel, as owned arrays
//! on the dataset's radial table.
//!
//! A diagnostic never takes a [`PawDataset`]; it takes a [`Smooth`], so the same code serves
//! the scalar (`j`-averaged, `kappa = None`) channels of a record, the two `j` branches the
//! spin-orbit entry of a record stores, and a channel set a generator builds in memory. A
//! [`Channel`] carries its own `kappa`: the all-electron side of every comparison is integrated
//! with the equation that `kappa` names.
use crate::{
    paw::{invalid, record::PawDataset},
    Buffer, Context, Result,
};
use std::collections::BTreeMap;

/// Identity of a channel in a result: its `l` and the `kappa` of its all-electron equation
/// (`None` for the scalar-relativistic, `j`-averaged channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    pub l: usize,
    pub kappa: Option<i32>,
}

/// One channel of the smooth problem on the dataset's table.
#[derive(Debug)]
pub struct Channel {
    pub l: usize,
    pub kappa: Option<i32>,
    /// Radius of the augmentation sphere (Bohr).
    pub r_cut: f64,
    /// Reference energies: the occupied bound states first, then scattering references.
    pub energies: Vec<f64>,
    /// Electrons of each reference.
    pub occupations: Vec<f64>,
    pub rank: usize,
    /// `u`-less projector tables `p(r)`, row-major `(rank, n)`.
    pub projectors: Buffer<f64>,
    /// Screened coupling `D^scr`, row-major `(rank, rank)`.
    pub coupling: Vec<f64>,
    /// Overlap correction `q` (the metric of the generalized problem).
    pub overlap: Vec<f64>,
    /// Norm correction `q^norm` (the energy-dependent part of the scattering coupling).
    pub norm: Vec<f64>,
    /// The smooth partial waves, one table each (the bound ones first).
    pub pseudo_waves: Vec<Buffer<f64>>,
    /// The all-electron partial waves of the same references.
    pub ae_waves: Vec<Buffer<f64>>,
}

impl Channel {
    pub fn key(&self) -> Key {
        Key {
            l: self.l,
            kappa: self.kappa,
        }
    }

    /// The occupied references: the first one always, then the others with a positive
    /// occupation (a library channel without per-wave occupations has one bound reference).
    pub fn occupied_energies(&self) -> Vec<f64> {
        let mut out = vec![self.energies[0]];
        out.extend(
            self.energies
                .iter()
                .zip(&self.occupations)
                .skip(1)
                .filter(|(_, &o)| o > 0.)
                .map(|(&e, _)| e),
        );
        out
    }

    /// The bound references the channel holds electrons in: the physical states the smooth
    /// Hamiltonian has to bind (a scattering reference, however far below zero, is not one).
    pub fn bound_occupied_energies(&self) -> Vec<f64> {
        self.energies
            .iter()
            .zip(&self.occupations)
            .filter(|(&e, &o)| o > 0. && e < 0.)
            .map(|(&e, _)| e)
            .collect()
    }

    /// Radius beyond which the projectors vanish and the smooth wave obeys the all-electron
    /// equation again: the channel's own cutoff, or the local radius when that is larger.
    pub fn projector_radius(&self, r_cut_local: f64) -> f64 {
        self.r_cut.max(r_cut_local)
    }
}

/// The smooth side of a dataset: the table, the screened local potential and the channels.
#[derive(Debug)]
pub struct Smooth {
    pub symbol: String,
    pub atomic_number: u8,
    pub r: Buffer<f64>,
    pub v_local_screened: Buffer<f64>,
    pub r_cut_local: f64,
    /// Channels by `l` (the `j`-averaged set), or by `(l, kappa)` for the branch set.
    pub channels: BTreeMap<Key, Channel>,
    /// Occupied valence subshells generated into the frozen core.
    pub frozen: Vec<(usize, usize)>,
    /// Core subshells generated into the valence.
    pub semicore: Vec<(usize, usize)>,
}

fn matrix(array: &crate::paw::record::Array, size: usize, what: &str) -> Result<Vec<f64>> {
    if array.shape() != [size, size] {
        return Err(invalid(format!(
            "diagnose: {what} has shape {:?}, expected {size} x {size}",
            array.shape()
        )));
    }
    Ok(array.data().to_vec())
}

fn subshells(list: &[Vec<i64>]) -> Result<Vec<(usize, usize)>> {
    list.iter()
        .map(|pair| match pair.as_slice() {
            [n, l] if *n > 0 && *l >= 0 => Ok((*n as usize, *l as usize)),
            _ => Err(invalid("diagnose: a subshell is an (n, l) pair")),
        })
        .collect()
}

impl Smooth {
    /// The `j`-averaged channels of a record (`kappa = None`).
    pub fn scalar(ctx: &Context, ds: &PawDataset) -> Result<Self> {
        let n = ds.points();
        let mut channels = BTreeMap::new();
        for (&l, ch) in &ds.channels {
            let rank = ch.projectors.len();
            if rank == 0 || ch.pseudo_waves.len() != rank || ch.ae_waves.len() != rank {
                return Err(invalid(format!(
                    "diagnose: channel l={l} has {rank} projectors, {} pseudo waves and {} all-electron waves",
                    ch.pseudo_waves.len(),
                    ch.ae_waves.len()
                )));
            }
            let mut projectors = ctx.allocate::<f64>(rank * n)?;
            for (i, p) in ch.projectors.iter().enumerate() {
                projectors.as_mut_slice()[i * n..(i + 1) * n].copy_from_slice(p.as_slice());
            }
            let copy = |waves: &[Buffer<f64>]| -> Result<Vec<Buffer<f64>>> {
                waves.iter().map(|w| ctx.copy_slice(w.as_slice())).collect()
            };
            let channel = Channel {
                l: l as usize,
                kappa: None,
                r_cut: ch.r_cut,
                energies: ch.reference_energies.data().to_vec(),
                occupations: ch.occupations.data().to_vec(),
                rank,
                projectors,
                coupling: matrix(&ch.coupling_screened, rank, "coupling_screened")?,
                overlap: matrix(&ch.overlap_correction, rank, "overlap_correction")?,
                norm: matrix(&ch.norm_correction, rank, "norm_correction")?,
                pseudo_waves: copy(&ch.pseudo_waves)?,
                ae_waves: copy(&ch.ae_waves)?,
            };
            if channel.energies.len() < 2 || channel.occupations.len() != channel.energies.len() {
                return Err(invalid(format!(
                    "diagnose: channel l={l} needs two reference energies and one occupation each"
                )));
            }
            channels.insert(channel.key(), channel);
        }
        Ok(Self {
            symbol: ds.symbol.clone(),
            atomic_number: ds.atomic_number as u8,
            r: ctx.copy_slice(ds.r.as_slice())?,
            v_local_screened: ctx.copy_slice(ds.v_local_screened.as_slice())?,
            r_cut_local: ds.r_cut_local,
            channels,
            frozen: subshells(&ds.frozen_subshells)?,
            semicore: subshells(&ds.semicore_subshells)?,
        })
    }

    /// The channel of `l` of the `j`-averaged set.
    pub fn channel(&self, l: usize) -> Option<&Channel> {
        self.channels.get(&Key { l, kappa: None })
    }

    /// The angular momenta that have a `j`-averaged channel.
    pub fn angular_momenta(&self) -> Vec<usize> {
        self.channels
            .keys()
            .filter(|k| k.kappa.is_none())
            .map(|k| k.l)
            .collect()
    }

    /// Radius a channel's smooth wave is compared at: past the projectors.
    pub fn match_radius(&self, channel: &Channel) -> f64 {
        super::snap_to(
            self.r.as_slice(),
            channel.projector_radius(self.r_cut_local),
        )
    }
}
