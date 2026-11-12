//! The static Kohn-Sham polarizability `chi_{GG'}(q)` of a mesh of Bloch states:
//!
//! ```text
//! chi_{GG'}(q) = 1/(N Omega) sum_{k,a,b} w_ab  rho_ab(q+G) rho_ab(q+G')^*,
//! w_ab = g (f_a - f_b) / (e_a - e_b) (1 - p_a p_b),
//! ```
//!
//! over the pairs `(a at k, b at k+q)` of every spin channel, `f` the Fermi-Dirac or other
//! smearing occupation of the SCF, `g` the spin degeneracy, `p` each state's weight in the target
//! subspace (constrained RPA; `p = 0` gives the full RPA) and `rho` the transition density with
//! its PAW augmentation ([`transitions`](super::transitions)). A degenerate pair takes the slope
//! of the smearing. Atomic units, Bohr^-3 Hartree^-1.
use super::{
    bloch::Bloch,
    transitions::{Augmentation, Engine, Remainders, Scratch},
    Options, Polarizability, State,
};
use crate::{
    math::SpatialGrid,
    periodic::{invalid, response::setup::Setup, spectrum::Smearing, Crystal},
    product,
    quantum::C64,
    Context, Result,
};
use nalgebra::{DMatrix, Matrix3, Vector3};
use rayon::prelude::*;
use std::f64::consts::PI;

/// Transition densities computed per dispatch to the pool.
const BATCH: usize = 256;

/// The PAW data the transition densities of a crystal are augmented with.
#[derive(Clone, Copy)]
pub struct Paw<'a> {
    pub(in crate::periodic) crystal: &'a Crystal,
    pub(in crate::periodic) setup: &'a Setup,
    pub(in crate::periodic) remainders: &'a Remainders,
}

/// What the response of a crystal's electrons needs besides its states.
pub struct Medium<'a> {
    pub ctx: &'a Context,
    pub grid: &'a SpatialGrid,
    /// The cell volume (Bohr^3).
    pub volume: f64,
    pub fermi: f64,
    pub width: f64,
    pub degeneracy: f64,
    pub smearing: Smearing,
    /// Remove the transitions inside the target subspace (`1 - p_a p_b`).
    pub constrained: bool,
    pub paw: Option<Paw<'a>>,
    /// Workers of the context's pool the transition loop uses; `None`: all of them. The result
    /// does not depend on it.
    pub workers: Option<usize>,
}

impl<'a> Medium<'a> {
    /// The workers of the context's pool a parallel region uses: `workers` capped at the pool,
    /// one without a pool.
    pub fn workers(&self) -> usize {
        if self.ctx.pool().is_some() {
            self.workers
                .unwrap_or(self.ctx.width())
                .clamp(1, self.ctx.width())
        } else {
            1
        }
    }
    /// `(chi, Q)` at `q` (Cartesian, Bohr^-1) on the vectors `Q = q + G` of the integer cell
    /// coordinates `g`, from `pairs` of `(k, k + q)` states, `mesh` the number of k-points the
    /// sum is averaged over.
    pub fn polarizability(
        &self,
        q: [f64; 3],
        g: &[[i64; 3]],
        pairs: &[(&Bloch<'_>, &Bloch<'_>)],
        mesh: usize,
    ) -> Result<(DMatrix<C64>, Vec<[f64; 3]>)> {
        let ctx = self.ctx;
        if mesh == 0 || g.is_empty() {
            return Err(invalid(
                "a polarizability needs a mesh and reciprocal vectors",
            ));
        }
        let engine = Engine::new(ctx, self.grid, q, g)?;
        let augmentation = self
            .paw
            .map(|p| Augmentation::new(p.crystal, p.setup, p.remainders, &engine.vectors))
            .transpose()?;
        let mut accumulator = Polarizability::new(
            ctx,
            g.len(),
            Options {
                cell_volume: self.volume,
                kpoints: mesh,
                fermi: self.fermi,
                width: self.width,
                degeneracy: self.degeneracy,
                smearing: self.smearing,
                constrained: self.constrained,
            },
        )?;
        // The transition densities are independent of each other, so they are computed on the
        // context's pool, one contiguous share of a batch per worker with its own scratch, and
        // then added to the accumulator one by one in the order of the loops below: the sum does
        // not depend on the thread count (no atomic or per-thread partial sums).
        let workers = self.workers();
        let mut scratch = (0..workers)
            .map(|_| engine.scratch(ctx))
            .collect::<Result<Vec<_>>>()?;
        let nv = g.len();
        let mut batch = ctx.allocate::<C64>(product(&[BATCH, nv])?)?;
        for (left, right) in pairs {
            left.check()?;
            right.check()?;
            let n = left.points;
            let l = left.samples.read_range(ctx, 0, left.bands() * n)?;
            let r = right.samples.read_range(ctx, 0, right.bands() * n)?;
            let state = |bloch: &Bloch<'_>, a: usize| State {
                energy: bloch.energies[a],
                target: bloch.target[a],
            };
            let mut todo: Vec<(usize, usize)> = vec![];
            for a in 0..left.bands() {
                for b in 0..right.bands() {
                    if accumulator
                        .transition_weight(state(left, a), state(right, b))?
                        .abs()
                        > 1e-12
                    {
                        todo.push((a, b));
                    }
                }
            }
            for chunk in todo.chunks(BATCH) {
                let share = chunk.len().div_ceil(workers);
                let out = &mut batch.as_mut_slice()[..chunk.len() * nv];
                let compute = |(pieces, own): (&[(usize, usize)], &mut Scratch),
                               rows: &mut [C64]|
                 -> Result<()> {
                    for (&(a, b), row) in pieces.iter().zip(rows.chunks_mut(nv)) {
                        engine.smooth(own, &l[a * n..(a + 1) * n], &r[b * n..(b + 1) * n], row)?;
                        if let Some(augmentation) = &augmentation {
                            augmentation.add(&left.projection(a), &right.projection(b), row);
                        }
                    }
                    Ok(())
                };
                match ctx.pool().filter(|_| workers > 1) {
                    Some(pool) => pool.install(|| {
                        chunk
                            .par_chunks(share)
                            .zip(scratch.par_iter_mut())
                            .zip(out.par_chunks_mut(share * nv))
                            .try_for_each(|((pieces, own), rows)| compute((pieces, own), rows))
                    })?,
                    None => compute((chunk, &mut scratch[0]), out)?,
                }
                for (&(a, b), row) in chunk.iter().zip(out.chunks(nv)) {
                    accumulator.add(state(left, a), state(right, b), row)?;
                }
            }
        }
        let (chi, _memory) = accumulator.finish()?;
        Ok((chi, engine.vectors))
    }
}

/// The integer cell coordinates `g` with `|q + B g| <= cutoff`, `B` the cell's reciprocal
/// vectors (columns of `2 pi A^-T`), `G = 0` first among equals by lexicographic order.
pub fn within(lattice: &Matrix3<f64>, q: [f64; 3], cutoff: f64) -> Vec<[i64; 3]> {
    let reciprocal = 2. * PI * lattice.try_inverse().expect("a lattice").transpose();
    let q = Vector3::from(q);
    let reach: [i64; 3] =
        std::array::from_fn(|a| (cutoff * lattice.column(a).norm() / (2. * PI)).ceil() as i64 + 1);
    let mut out = vec![];
    for x in -reach[0]..=reach[0] {
        for y in -reach[1]..=reach[1] {
            for z in -reach[2]..=reach[2] {
                let v = q + reciprocal * Vector3::new(x as f64, y as f64, z as f64);
                if v.norm() <= cutoff {
                    out.push([x, y, z]);
                }
            }
        }
    }
    out
}

/// The polarizability of the Gamma-point Kohn-Sham states of `crystal` alone, on the vectors
/// `Q = B m` of the integer coordinates `m` of its own reciprocal lattice (every pair of its
/// states at `q = 0`). For a crystal that is the Born-von Karman supercell of a mesh this is the
/// mesh's polarizability `chi_{GG'}(q)` on `Q = q + G`: the same transitions counted without the
/// pairing of `k` with `k + q`. An audit seam of the mesh sum, not a route to a result.
///
/// The states' energies are the raw eigenvalues plus `energy_shift` (a frame's own crystal
/// `reference` is taken out), so the caller puts them on the scale of the Fermi level it passes.
pub fn gamma_polarizability(
    crystal: &Crystal,
    result: &crate::periodic::PeriodicResult,
    spin: usize,
    energy_shift: f64,
    fermi: f64,
    width: f64,
    smearing: Smearing,
    m: &[[i64; 3]],
) -> Result<DMatrix<C64>> {
    use crate::periodic::wannier::sampling::Frame;
    let ctx = &crystal.context;
    let setup = Setup::new(crystal)?;
    let remainders = Remainders::new(ctx, &setup)?;
    let bands: Vec<usize> = (0..crystal.bands()).collect();
    let frame = Frame::from_point(crystal, result, &crystal.points[0], spin, &bands)?;
    let energies: Vec<f64> = frame
        .energies
        .as_slice()
        .iter()
        .map(|e| e - crystal.reference + energy_shift)
        .collect();
    let points = crystal.grid.index.points();
    let state = Bloch {
        fractional: frame.fractional,
        energies: &energies,
        target: vec![0.; energies.len()],
        samples: &frame.samples,
        projections: Some(&frame.projections),
        points,
    };
    let lattice = crate::periodic::wannier::sampling::lattice(&crystal.grid);
    let medium = Medium {
        ctx,
        grid: &crystal.grid,
        volume: lattice.determinant().abs(),
        fermi,
        width,
        degeneracy: crystal.degeneracy(),
        smearing,
        constrained: false,
        workers: None,
        paw: Some(Paw {
            crystal,
            setup: &setup,
            remainders: &remainders,
        }),
    };
    let (chi, _) = medium.polarizability([0.; 3], m, &[(&state, &state)], 1)?;
    Ok(chi)
}
