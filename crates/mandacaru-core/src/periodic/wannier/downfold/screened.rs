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
            interaction::solve,
            medium::{within, Medium, Paw},
            transitions::{components_of, Augmentation, Engine, Remainders},
        },
        wannier::{model::Channel, sampling::Frame, setup::Wannier},
        Crystal,
    },
    quantum::C64,
    Error, Result,
};
use nalgebra::{DMatrix, Vector3};
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
        components: frame.components,
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
        let components = state.components;
        let bands = state.bands();
        let samples = state.samples.read_range(ctx, 0, components * bands * n)?;
        let mut out = DMatrix::<C64>::zeros(bands, bands);
        let mut density = [C64::default()];
        for a in 0..bands {
            for b in 0..bands {
                engine.smooth_components(
                    &mut scratch,
                    &components_of(&samples, bands, n, components, a),
                    &components_of(&samples, bands, n, components, b),
                    &mut density,
                )?;
                for c in 0..components {
                    augmentation.add(
                        &state.component_projection(a, c),
                        &state.component_projection(b, c),
                        &mut density,
                    );
                }
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
        // The mesh points one after another through the shared loop of
        // `periodic::screening::interaction`, which keeps the lanes out of the medium's pool.
        // When the crystal is time-reversal invariant and the constrained target weights keep
        // `p_a(-k) = p_a(k)`, pair `(q, -q)` via `chi_{GG'}(-q)^* = chi_{-G,-G'}(q)`.
        let can_reduce_q = self.wannier.crystal.time_reversal
            && (!self.medium.constrained
                || (0..cell.cells).all(|k| {
                    let k_vec = cell.cell_vector(k);
                    let neg_k_vec =
                        std::array::from_fn(|a| (-k_vec[a]).rem_euclid(cell.size[a] as i64));
                    let neg_k = partner(0, neg_k_vec, cell.size);
                    self.weights.iter().all(|ch| {
                        ch[k]
                            .iter()
                            .zip(&ch[neg_k])
                            .all(|(a, b)| (a - b).abs() < 1e-10)
                    })
                }));
        let reciprocal = 2.0
            * std::f64::consts::PI
            * cell
                .lattice
                .try_inverse()
                .ok_or_else(|| invalid("degenerate cell lattice in screened downfolding"))?
                .transpose();
        let tr_cache: std::cell::RefCell<
            std::collections::HashMap<usize, (DMatrix<C64>, Vec<[i64; 3]>)>,
        > = std::cell::RefCell::new(std::collections::HashMap::new());
        let solved = solve(
            ctx,
            cell.cells,
            |t| {
                let q_cell = cell.cell_vector(t);
                if can_reduce_q {
                    let neg_q_cell =
                        std::array::from_fn(|a| (-q_cell[a]).rem_euclid(cell.size[a] as i64));
                    let neg_t = partner(0, neg_q_cell, cell.size);
                    if neg_t < t {
                        if let Some((neg_chi, neg_g)) = tr_cache.borrow_mut().remove(&t) {
                            let q_phys = cell.vector(q_cell);
                            let g = crate::periodic::screening::medium::within(
                                &cell.lattice,
                                q_phys,
                                self.cutoff,
                            );
                            let g_shift = std::array::from_fn(|a| {
                                (q_cell[a] + neg_q_cell[a]) / cell.size[a] as i64
                            });
                            if let Some(chi) =
                                crate::periodic::screening::interaction::conjugate_polarizability(
                                    &neg_chi, &neg_g, &g, g_shift,
                                )
                            {
                                let vectors: Vec<[f64; 3]> = g
                                    .iter()
                                    .map(|gi| {
                                        let g_cart = reciprocal
                                            * nalgebra::Vector3::new(
                                                gi[0] as f64,
                                                gi[1] as f64,
                                                gi[2] as f64,
                                            );
                                        [
                                            q_phys[0] + g_cart[0],
                                            q_phys[1] + g_cart[1],
                                            q_phys[2] + g_cart[2],
                                        ]
                                    })
                                    .collect();
                                return Ok((chi, vectors, g));
                            }
                        }
                    } else if t < neg_t {
                        let (chi, vectors, g) = self.polarizability(q_cell)?;
                        tr_cache.borrow_mut().insert(neg_t, (chi.clone(), g.clone()));
                        return Ok((chi, vectors, g));
                    }
                }
                self.polarizability(q_cell)
            },
            |_, solution, _, g| Ok((solution.correction, g)),
        )?;
        for (t, (correction, g)) in solved.into_iter().enumerate() {
            let q_cell = cell.cell_vector(t);
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
                delta: correction,
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
    let provider = &wannier.setup;
    let crystal: &Crystal = &wannier.crystal;
    let ctx = &crystal.context;
    let points = crystal.grid.index.points();
    let mesh = cell.cells;
    let spins = wannier.channels.len();
    let weights = vec![1. / mesh as f64; mesh];
    let parallel = medium.workers() > 1;
    // The states at `k + q` of one axis: one call sets the matrices of every shifted point up
    // (a call has a fixed cost, 0.25 s for the H2 chain against 0.008 s per point, that chunks
    // would multiply) and the eigensolves, one per point and spin, follow. Every point is
    // independent, so an axis's points run on the layout's lanes (P6); each frame is computed
    // alone, so nothing depends on the number of workers.
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
        let per_point: Vec<Vec<Frame>> = if parallel {
            ctx.map_independent(kpoints.len(), |k| one(&kpoints[k]))?
        } else {
            kpoints.iter().map(one).collect::<Result<Vec<_>>>()?
        };
        let mut shifted: Vec<Vec<Frame>> = (0..spins).map(|_| Vec::with_capacity(mesh)).collect();
        for frames in per_point {
            for (spin, frame) in frames.into_iter().enumerate() {
                shifted[spin].push(frame);
            }
        }
        Ok(shifted)
    };
    // The axes one after another (each sets its shifted matrices up through the setup, which may
    // use the pool), the points of an axis on the lanes: one level of lanes, never nested.
    let axes: Vec<Vec<Vec<Frame>>> = (0..3).map(shifted_frames).collect::<Result<Vec<_>>>()?;
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
