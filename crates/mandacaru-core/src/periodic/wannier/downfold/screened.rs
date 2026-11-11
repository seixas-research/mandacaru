//! The screened interaction of a downfolded fragment: constrained or full RPA, static, from the
//! Kohn-Sham states, not self-consistent.
//!
//! ```text
//! W = [1 - v chi_r]^-1 v,   chi_r = chi without the transitions inside the correlated subspace
//! ```
//!
//! on the vectors `|q + G| <= cutoff` of every mesh point `q`, with the periodic Coulomb
//! interaction `v = 4 pi / |q + G|^2` of the Dyson equation (`G = 0` left out at `q = 0`). The
//! divergent head is the long-wavelength limit `eps^-1_00(q -> 0)` of the same polarizability at a
//! small `q` along each Cartesian axis, averaged, with each state keeping the weight of its band;
//! it scales the bare integrals' own `G = 0` term (the truncated kernel's `2 pi R_c^2`); the wings
//! are dropped. The correction `W - v` is contracted with the fragment's pair densities
//! ([`correction`]) and added to the bare integrals.
use super::{
    coulomb::{truncated, Cell},
    pairs::{Integrals, Pairs},
    Screening,
};
use crate::{
    periodic::{
        invalid,
        response::setup::Setup,
        screening::{
            bloch::Bloch,
            dyson,
            medium::{within, Medium, Paw},
            transitions::{Augmentation, Engine, Remainders},
        },
        wannier::{model::Channel, sampling::Frame, setup::Wannier},
        Crystal,
    },
    quantum::C64,
    Error, Result,
};
use nalgebra::{DMatrix, Vector3};
use rayon::prelude::*;
use std::f64::consts::PI;

/// Cutoff of the screened vectors when none is given (Bohr^-1).
pub const SCREENING_CUTOFF: f64 = 3.;
/// The small wave vector of the long-wavelength head (Bohr^-1).
pub const HEAD_WAVEVECTOR: f64 = 0.02;

/// `W - v` on the vectors of one mesh point.
pub struct Block {
    /// Supercell frequencies `S g + q` of the vectors.
    pub m: Vec<[i64; 3]>,
    pub delta: DMatrix<C64>,
}

pub struct Screened {
    pub blocks: Vec<Block>,
    /// `eps^-1_00(q -> 0)`.
    pub head: f64,
}

/// The weights `p_a(k) = sum_{n in T} |U_an(k)|^2` of the first `bands` states of a channel's
/// frames in the Bloch subspace of the functions `targets` (zeros without targets).
fn weights(channel: &Channel, targets: &[usize], bands: usize) -> Vec<Vec<f64>> {
    channel
        .localized
        .gauge
        .iter()
        .map(|gauge| {
            (0..bands)
                .map(|a| {
                    targets
                        .iter()
                        .map(|&n| gauge[(a, n)].norm_sqr())
                        .sum::<f64>()
                        .min(1.)
                })
                .collect()
        })
        .collect()
}

fn view<'a>(frame: &'a Frame, target: Vec<f64>, bands: usize, points: usize) -> Bloch<'a> {
    Bloch {
        fractional: frame.fractional,
        energies: &frame.energies.as_slice()[..bands],
        target,
        samples: &frame.samples,
        projections: Some(&frame.projections),
        points,
    }
}

/// The index of the mesh point `k + q` (cells in C order).
fn partner(k: usize, q: [i64; 3], size: [usize; 3]) -> usize {
    let at = [
        k / (size[1] * size[2]),
        (k / size[2]) % size[1],
        k % size[2],
    ];
    let moved: [usize; 3] =
        std::array::from_fn(|a| (at[a] as i64 + q[a]).rem_euclid(size[a] as i64) as usize);
    (moved[0] * size[1] + moved[1]) * size[2] + moved[2]
}

/// The Kohn-Sham states of a Wannier calculation as the response reads them: every spin channel's
/// states on the mesh with their weights in the correlated subspace, and the medium.
pub(in crate::periodic) struct Rest<'a> {
    wannier: &'a Wannier,
    pub medium: Medium<'a>,
    states: Vec<Vec<Bloch<'a>>>,
    weights: Vec<Vec<Vec<f64>>>,
    pub cell: &'a Cell,
    pub cutoff: f64,
    pub bands: usize,
}

impl<'a> Rest<'a> {
    /// `ids`: the functions of the home cell whose Bloch subspace is the correlated one (constrained
    /// RPA; ignored for the full RPA).
    pub(in crate::periodic) fn new(
        wannier: &'a Wannier,
        setup: &'a Setup,
        remainders: &'a Remainders,
        cell: &'a Cell,
        ids: &[usize],
        screening: Screening,
        cutoff: Option<f64>,
        bands: Option<usize>,
    ) -> Result<Self> {
        let crystal: &Crystal = &wannier.crystal;
        let frame_bands = wannier.channels[0].frames[0].energies.as_slice().len();
        let bands = bands.unwrap_or(frame_bands);
        if bands > frame_bands {
            return Err(Error::Unsupported(format!(
                "screening_bands = {bands}, but the Wannier calculation kept {frame_bands} \
                 Kohn-Sham states per k-point: run wannier(..., windows=...) or bands=... with \
                 the states the screening should include"
            )));
        }
        let points = crystal.grid.index.points();
        let targets: &[usize] = if screening == Screening::Crpa {
            ids
        } else {
            &[]
        };
        let weights: Vec<Vec<Vec<f64>>> = wannier
            .channels
            .iter()
            .map(|c| weights(c, targets, bands))
            .collect();
        let states = wannier
            .channels
            .iter()
            .zip(&weights)
            .map(|(channel, weight)| {
                channel
                    .frames
                    .iter()
                    .zip(weight)
                    .map(|(frame, w)| view(frame, w.clone(), bands, points))
                    .collect()
            })
            .collect();
        let medium = Medium {
            ctx: &crystal.context,
            grid: &crystal.grid,
            volume: cell.lattice.determinant().abs(),
            fermi: wannier.state.fermi,
            width: wannier.state.options.width,
            degeneracy: crystal.degeneracy(),
            smearing: wannier.state.options.smearing,
            constrained: screening == Screening::Crpa,
            workers: None,
            paw: Some(Paw {
                crystal,
                setup,
                remainders,
            }),
        };
        Ok(Self {
            wannier,
            medium,
            states,
            weights,
            cell,
            cutoff: cutoff.unwrap_or(SCREENING_CUTOFF),
            bands,
        })
    }

    /// `(chi, Q, G)` at the mesh point `q_cell` (integer mesh coordinates).
    pub(in crate::periodic) fn polarizability(
        &self,
        q_cell: [i64; 3],
    ) -> Result<(DMatrix<C64>, Vec<[f64; 3]>, Vec<[i64; 3]>)> {
        let cell = self.cell;
        let mesh = cell.cells;
        let q = cell.vector(q_cell);
        let g = within(&cell.lattice, q, self.cutoff);
        let mut pairs: Vec<(&Bloch<'_>, &Bloch<'_>)> = Vec::with_capacity(self.states.len() * mesh);
        for channel in &self.states {
            for k in 0..mesh {
                pairs.push((&channel[k], &channel[partner(k, q_cell, cell.size)]));
            }
        }
        let (chi, vectors) = self.medium.polarizability(q, &g, &pairs, mesh)?;
        Ok((chi, vectors, g))
    }

    /// `rho_ab(Q = 0)` of every pair of states of mesh point `k` of a spin channel.
    pub(in crate::periodic) fn norms(&self, spin: usize, k: usize) -> Result<DMatrix<C64>> {
        let ctx = self.medium.ctx;
        let paw = self.medium.paw.expect("the rest carries its PAW data");
        let state = &self.states[spin][k];
        let engine = Engine::new(ctx, self.medium.grid, [0.; 3], &[[0, 0, 0]])?;
        let mut scratch = engine.scratch(ctx)?;
        let augmentation =
            Augmentation::new(paw.crystal, paw.setup, paw.remainders, &engine.vectors)?;
        let n = state.points;
        let samples = state.samples.read_range(ctx, 0, state.bands() * n)?;
        let mut out = DMatrix::<C64>::zeros(state.bands(), state.bands());
        let mut density = [C64::default()];
        for a in 0..state.bands() {
            for b in 0..state.bands() {
                engine.smooth(
                    &mut scratch,
                    &samples[a * n..(a + 1) * n],
                    &samples[b * n..(b + 1) * n],
                    &mut density,
                )?;
                augmentation.add(&state.projection(a), &state.projection(b), &mut density);
                out[(a, b)] = density[0];
            }
        }
        Ok(out)
    }

    /// `W - v` of every mesh point and the long-wavelength head.
    pub(in crate::periodic) fn interaction(&self, radius: f64) -> Result<Screened> {
        let cell = self.cell;
        let ctx = self.medium.ctx;
        let mut blocks = Vec::with_capacity(cell.cells);
        let mut origin: Option<(usize, usize)> = None;
        for t in 0..cell.cells {
            let q_cell = cell.cell_vector(t);
            let (chi, vectors, g) = self.polarizability(q_cell)?;
            let interaction = dyson(ctx, &chi, &vectors)?;
            if t == 0 {
                let head = g
                    .iter()
                    .position(|v| *v == [0, 0, 0])
                    .ok_or_else(|| invalid("the screened vectors of q = 0 lack G = 0"))?;
                origin = Some((blocks.len(), head));
            }
            blocks.push(Block {
                m: g.iter()
                    .map(|v| std::array::from_fn(|a| cell.size[a] as i64 * v[a] + q_cell[a]))
                    .collect(),
                delta: interaction.correction.clone(),
            });
        }
        let head = long_wavelength_head(
            self.wannier,
            &self.medium,
            cell,
            &self.states,
            &self.weights,
            self.cutoff,
            self.bands,
        )?;
        let (at, g0) = origin.expect("the mesh has q = 0");
        blocks[at].delta[(g0, g0)] = C64::new((head - 1.) * truncated(0., radius), 0.);
        Ok(Screened { blocks, head })
    }
}

/// `eps^-1_00(q -> 0)` from the polarizability at `q = HEAD_WAVEVECTOR` along each Cartesian
/// axis, averaged: the states at `k + q` come from the setup's matrices at the shifted points,
/// each keeping the weight of its band at `k`.
fn long_wavelength_head(
    wannier: &Wannier,
    medium: &Medium<'_>,
    cell: &Cell,
    states: &[Vec<Bloch<'_>>],
    target_weights: &[Vec<Vec<f64>>],
    cutoff: f64,
    bands: usize,
) -> Result<f64> {
    let provider = wannier.provider.as_ref().ok_or_else(|| {
        Error::Unsupported(
            "the long-wavelength limit of the screening needs the periodic setup the Wannier \
             functions were built from; this result has none"
                .into(),
        )
    })?;
    let crystal: &Crystal = &wannier.crystal;
    let ctx = &crystal.context;
    let points = crystal.grid.index.points();
    let mesh = cell.cells;
    let spins = wannier.channels.len();
    let weights = vec![1. / mesh as f64; mesh];
    let pool = ctx.pool().filter(|_| medium.workers() > 1);
    // The states at `k + q` of one axis: one call sets the matrices of every shifted point up
    // (a call has a fixed cost, 0.25 s for the H2 chain against 0.008 s per point, that chunks
    // would multiply) and the eigensolves, one per point and spin, follow. The three axes are
    // independent, and so is every point, so they run on the context's pool; each frame is
    // computed alone, so nothing depends on the number of workers.
    let shifted_frames = |axis: usize| -> Result<Vec<Vec<Frame>>> {
        let mut q = [0.; 3];
        q[axis] = HEAD_WAVEVECTOR;
        let shift = Vector3::from(q);
        let fractional: Vec<[f64; 3]> = states[0]
            .iter()
            .map(|s| {
                std::array::from_fn(|a| {
                    s.fractional[a] + cell.lattice.column(a).dot(&shift) / (2. * PI)
                })
            })
            .collect();
        let kpoints = provider.matrices_at(ctx, &fractional, &weights)?;
        let one = |point: &crate::periodic::KPoint| -> Result<Vec<Frame>> {
            (0..spins)
                .map(|spin| Frame::from_point(crystal, &wannier.state, point, spin, &wannier.bands))
                .collect()
        };
        let per_point: Vec<Vec<Frame>> = match pool {
            Some(pool) => {
                pool.install(|| kpoints.par_iter().map(one).collect::<Result<Vec<_>>>())?
            }
            None => kpoints.iter().map(one).collect::<Result<Vec<_>>>()?,
        };
        let mut shifted: Vec<Vec<Frame>> = (0..spins).map(|_| Vec::with_capacity(mesh)).collect();
        for frames in per_point {
            for (spin, frame) in frames.into_iter().enumerate() {
                shifted[spin].push(frame);
            }
        }
        Ok(shifted)
    };
    let axes: Vec<Vec<Vec<Frame>>> = match pool {
        Some(pool) => pool.install(|| {
            (0..3usize)
                .into_par_iter()
                .map(shifted_frames)
                .collect::<Result<Vec<_>>>()
        })?,
        None => (0..3).map(shifted_frames).collect::<Result<Vec<_>>>()?,
    };
    let mut total = 0.;
    for (axis, shifted) in axes.iter().enumerate() {
        let mut q = [0.; 3];
        q[axis] = HEAD_WAVEVECTOR;
        let views: Vec<Vec<Bloch<'_>>> = shifted
            .iter()
            .zip(target_weights)
            .map(|(frames, weight)| {
                frames
                    .iter()
                    .zip(weight)
                    .map(|(frame, w)| view(frame, w.clone(), bands, points))
                    .collect()
            })
            .collect();
        let g = within(&cell.lattice, q, cutoff);
        let head = g
            .iter()
            .position(|v| *v == [0, 0, 0])
            .ok_or_else(|| invalid("the head vectors lack G = 0"))?;
        let pairs: Vec<(&Bloch<'_>, &Bloch<'_>)> = states
            .iter()
            .zip(&views)
            .flat_map(|(own, moved)| own.iter().zip(moved))
            .collect();
        let (chi, vectors) = medium.polarizability(q, &g, &pairs, mesh)?;
        let interaction = dyson(ctx, &chi, &vectors)?;
        total += interaction.inverse_dielectric[(head, head)].re;
    }
    Ok(total / 3.)
}

/// What the screening adds to the bare integrals,
/// `1/Omega_s sum_{q G G'} rho_qp(q+G)^* [W - v]_{GG'}(q) rho_rs(q+G')`, between the pair
/// densities of two spin channels (each with its augmentation remainder).
pub(in crate::periodic) fn correction(
    setup: &Setup,
    remainders: &Remainders,
    cell: &Cell,
    first: &Pairs,
    second: &Pairs,
    screened: &Screened,
) -> Result<Integrals> {
    let (n1, n2) = (first.n, second.n);
    let (a, b) = (n1 * n1, n2 * n2);
    let mut values = vec![C64::default(); a * b];
    for block in &screened.blocks {
        let count = block.m.len();
        let left = first.at(setup, cell, remainders, &block.m)?;
        let right = second.at(setup, cell, remainders, &block.m)?;
        // delta * right: (count x count) (count x b)
        for p in 0..n1 {
            for q in 0..n1 {
                // The pair (pq) enters as the conjugate of (qp), as in the bare integral.
                let swapped = &left[(q * n1 + p) * count..][..count];
                for r in 0..b {
                    let column = &right[r * count..][..count];
                    let mut sum = C64::default();
                    for (g, l) in swapped.iter().enumerate() {
                        let mut row = C64::default();
                        for (h, c) in column.iter().enumerate() {
                            row += block.delta[(g, h)] * c;
                        }
                        sum += l.conj() * row;
                    }
                    values[(p * n1 + q) * b + r] += sum;
                }
            }
        }
    }
    let scale = 1. / cell.volume;
    for v in &mut values {
        *v *= scale;
    }
    Ok(Integrals {
        first: n1,
        second: n2,
        values,
    })
}
