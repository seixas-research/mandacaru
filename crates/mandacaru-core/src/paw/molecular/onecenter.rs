//! Multipole moments and exact valence Coulomb/exchange corrections.
use super::*;
use crate::{
    math::{linalg::real_eigen, spline::simpson},
    paw::pair_tables::{layout, pair_tensor, ChannelView, Projector, SpeciesView, Tables, POINTS},
};

/// The radial tables of `species` the species-level one-center code reads.
pub(super) fn view(species: &Species) -> SpeciesView<'_> {
    SpeciesView {
        knots: species.ionic.knots.as_slice(),
        smooth_core: species.xc_core.values.as_slice(),
        compensation_radius: species.compensation_radius,
        channels: species
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

pub struct Multipole {
    pub l: usize,
    pub m: i32,
    pub block: DMatrix<f64>,
}
pub struct AtomTerms {
    pub projectors: Vec<Projector>,
    pub coupling: DMatrix<f64>,
    pub ks_difference: DMatrix<f64>,
    pub overlap: DMatrix<f64>,
    pub multipoles: Vec<Multipole>,
    pub factors: Vec<DMatrix<f64>>,
    pub weights: Vec<f64>,
    pub _memory: crate::Reservation,
}
pub fn build(ctx: &Context, species: &Species) -> Result<AtomTerms> {
    let view = view(species);
    let projectors = layout(&view);
    let p = projectors.len();
    let pp = product(&[p, p])?;
    if p == 0 {
        return Err(invalid("PAW species has no projectors"));
    }
    let lmax = projectors.iter().map(|p| p.l).max().unwrap() * 2;
    let memory = ctx.reserve(
        product(&[pp, pp, 8, 6])?
            .checked_add(product(&[pp, POINTS, 16, lmax + 1])?)
            .ok_or(crate::Error::Overflow)?,
    )?;
    let tables = Tables::new(&view, &projectors)?;
    let gaunts = &tables.gaunts;
    let source = species.ionic.knots.as_slice();
    let mut multipoles = Vec::new();
    for (l, m, g) in gaunts {
        if g.iter().all(|&v| v == 0.) {
            continue;
        }
        let mut block = DMatrix::zeros(p, p);
        for (a, pa) in projectors.iter().enumerate() {
            for (b, pb) in projectors.iter().enumerate() {
                if g[(a, b)] == 0. {
                    continue;
                }
                let ca = &species.channels[pa.channel];
                let cb = &species.channels[pb.channel];
                let moment = if *l == 0 && pa.channel == pb.channel {
                    ca.overlap[(pa.radial, pb.radial)]
                } else {
                    let inside = source.partition_point(|r| *r <= ca.radius.max(cb.radius));
                    let values: Vec<_> = (0..inside)
                        .map(|k| {
                            (ca.ae[pa.radial][k] * cb.ae[pb.radial][k]
                                - ca.ps[pa.radial][k] * cb.ps[pb.radial][k])
                                * source[k].powi((*l + 2) as i32)
                        })
                        .collect();
                    simpson(&source[..inside], &values)?
                };
                block[(a, b)] = moment * g[(a, b)];
            }
        }
        if block.iter().any(|v| v.abs() > 1e-14) {
            multipoles.push(Multipole {
                l: *l,
                m: *m,
                block,
            });
        }
    }
    let w = pair_tensor(&tables, None)?;
    let mut coupling = DMatrix::zeros(p, p);
    let mut ks_coupling = DMatrix::zeros(p, p);
    let mut overlap = DMatrix::zeros(p, p);
    let mut reference = DMatrix::zeros(p, p);
    for (a, pa) in projectors.iter().enumerate() {
        for (b, pb) in projectors.iter().enumerate() {
            if pa.channel == pb.channel && pa.m == pb.m {
                let ch = &species.channels[pa.channel];
                coupling[(a, b)] = ch.coupling[(pa.radial, pb.radial)]
                    - ch.xc[(pa.radial, pb.radial)]
                    + ch.core_exchange[(pa.radial, pb.radial)];
                ks_coupling[(a, b)] = ch.coupling[(pa.radial, pb.radial)];
                overlap[(a, b)] = ch.overlap[(pa.radial, pb.radial)];
                reference[(a, b)] = ch.reference[(pa.radial, pb.radial)];
            }
        }
    }
    for a in 0..p {
        for b in 0..p {
            coupling[(a, b)] -= (0..p)
                .map(|c| {
                    (0..p)
                        .map(|d| w[(a * p + b, c * p + d)] * reference[(c, d)])
                        .sum::<f64>()
                })
                .sum::<f64>();
        }
    }
    let (values, vectors) = real_eigen(w)?;
    let floor = values.iter().map(|v| v.abs()).fold(0., f64::max) * 1e-10;
    let keep: Vec<_> = values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.abs() > floor)
        .map(|(i, _)| i)
        .collect();
    let factors = keep
        .iter()
        .map(|&k| {
            DMatrix::from_fn(p, p, |a, b| {
                0.5 * (vectors[(a * p + b, k)] + vectors[(b * p + a, k)])
            })
        })
        .collect();
    let weights = keep.iter().map(|&k| values[k]).collect();
    Ok(AtomTerms {
        ks_difference: ks_coupling - &coupling,
        projectors,
        coupling,
        overlap,
        multipoles,
        factors,
        weights,
        _memory: memory,
    })
}
