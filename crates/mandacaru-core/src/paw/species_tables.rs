//! The radial tables of a PAW species from its dataset record, in the form the
//! species-level one-center code ([`super::pair_tables`], [`super::screened_atom`])
//! reads: the all-electron and the smooth partial waves of every channel in the
//! projector basis the Hamiltonian's projections were built in, and the reference
//! atom's density matrix over them.
//!
//! This is the construction of the molecular adapter (`paw::molecular::data`) for
//! the part the one-center exchange needs, taken from the typed record instead of
//! from adapter arrays, and without the molecular restrictions on the frozen core
//! (the one-center exchange is valence only). The smooth wave inside the channel
//! radius is the dataset's Bessel expansion, outside it the all-electron wave.
use super::{
    invalid,
    onecenter::Basis,
    pair_tables::{ChannelView, SpeciesView},
    record::PawDataset,
};
use crate::{kernels, product, Context, Result};
use nalgebra::DMatrix;

/// One channel: waves in the projector basis, `refs x grid` each.
struct ChannelTables {
    l: usize,
    radius: f64,
    ae: Vec<Vec<f64>>,
    ps: Vec<Vec<f64>>,
    reference: DMatrix<f64>,
}

/// The tables of one species, owned. [`SpeciesTables::view`] borrows them.
pub struct SpeciesTables {
    knots: Vec<f64>,
    compensation_radius: f64,
    channels: Vec<ChannelTables>,
}

impl SpeciesTables {
    /// The waves of `dataset` in `basis` (`Raw`: `B^-1` applied to the dual waves, the
    /// reference density `B^T D B`; `Dual`: the stored waves themselves).
    pub fn from_record(ctx: &Context, dataset: &PawDataset, basis: Basis) -> Result<Self> {
        let r = dataset.r.as_slice();
        let n = r.len();
        let waves: usize = dataset.channels.values().map(|c| c.ae_waves.len()).sum();
        let _memory = ctx.reserve(product(&[n, waves.max(1), 8, 4])?)?;
        let mut channels = Vec::new();
        for (&l, channel) in &dataset.channels {
            let l = l as usize;
            let refs = channel.ae_waves.len();
            let occupations = channel.occupations.data();
            if refs == 0
                || channel.pseudo_waves.len() != refs
                || occupations.len() != refs
                || channel.wavevectors.len() != refs
                || channel.wave_coefficients.len() != refs
                || channel.vanderbilt.len() != product(&[refs, refs])?
            {
                return Err(invalid(format!(
                    "the channel l = {l} of the dataset has inconsistent wave tables"
                )));
            }
            let b = DMatrix::from_row_slice(refs, refs, channel.vanderbilt.data());
            let transform = match basis {
                Basis::Raw => b
                    .clone()
                    .try_inverse()
                    .ok_or_else(|| invalid("singular PAW projector transform"))?,
                Basis::Dual => DMatrix::identity(refs, refs),
            };
            let mut ae: Vec<Vec<f64>> = channel
                .ae_waves
                .iter()
                .map(|w| w.as_slice().to_vec())
                .collect();
            let mut ps = vec![vec![0.; n]; refs];
            for i in 0..refs {
                let q = channel.wavevectors[i].data();
                let c = channel.wave_coefficients[i].data();
                if q.len() != c.len() || ae[i].len() != n {
                    return Err(invalid("PAW Bessel expansion dimensions disagree"));
                }
                let j = kernels::bessel(ctx, l as i32, r, q)?;
                for k in 0..n {
                    ps[i][k] = if r[k] <= channel.r_cut {
                        (0..c.len())
                            .map(|z| c[z] * j.as_slice()[k * c.len() + z])
                            .sum()
                    } else {
                        ae[i][k]
                    };
                }
            }
            let diagonal = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
                refs,
                occupations.iter().map(|o| o / (2 * l + 1) as f64),
            ));
            let reference = match basis {
                Basis::Raw => b.transpose() * diagonal * &b,
                Basis::Dual => diagonal,
            };
            let mix = |waves: &[Vec<f64>]| -> Vec<Vec<f64>> {
                (0..refs)
                    .map(|i| {
                        (0..n)
                            .map(|k| (0..refs).map(|j| transform[(i, j)] * waves[j][k]).sum())
                            .collect()
                    })
                    .collect()
            };
            if basis == Basis::Raw {
                ae = mix(&ae);
                ps = mix(&ps);
            }
            channels.push(ChannelTables {
                l,
                radius: channel.r_cut,
                ae,
                ps,
                reference,
            });
        }
        if channels.is_empty() || !(dataset.compensation_radius > 0.) {
            return Err(invalid(
                "a PAW species needs channels and a positive compensation radius",
            ));
        }
        Ok(Self {
            knots: r.to_vec(),
            compensation_radius: dataset.compensation_radius,
            channels,
        })
    }

    /// The borrowed form the one-center code takes.
    pub fn view(&self) -> SpeciesView<'_> {
        SpeciesView {
            knots: &self.knots,
            compensation_radius: self.compensation_radius,
            channels: self
                .channels
                .iter()
                .map(|ch| ChannelView {
                    l: ch.l,
                    radius: ch.radius,
                    ae: ch.ae.iter().map(Vec::as_slice).collect(),
                    ps: ch.ps.iter().map(Vec::as_slice).collect(),
                    reference: &ch.reference,
                })
                .collect(),
        }
    }
}
