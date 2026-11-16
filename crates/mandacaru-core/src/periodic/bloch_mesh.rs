//! Bloch states of a converged crystal on a uniform k mesh: the one owner of the mesh indexing,
//! of the frames re-diagonalized at the mesh points, of the PAW one-center geometry of their
//! overlaps, and of the checks every consumer needs (a converged SCF, isolated band groups). The
//! Berry quantities (`periodic::berry`) and the Wannier functions (`periodic::wannier`) are built
//! on it. Atomic units.
//!
//! A mesh is `k = origin + sum_j (i_j / n_j) G_j` in C order (`i_2` fastest). A displacement that
//! leaves the zone reuses the frame of the wrapped point with the true `b`: the periodic gauge
//! `u_{k+G} = e^{-iG.r} u_k` closes every loop with no extra phase.
use super::{
    invalid,
    setup::PeriodicSetup,
    wannier::sampling::{self, Augmentation, Frame},
    Crystal, PeriodicResult,
};
use crate::{product, quantum::C64, Error, Result};
use nalgebra::{DMatrix, Matrix3};
use std::f64::consts::PI;

type Matrix = DMatrix<C64>;

/// The smallest gap (Hartree) that separates a band group from its neighbors.
pub const SEPARATION: f64 = 1e-6;

/// The indexing of a uniform mesh `origin + i / size` (fractional, C order).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshIndex {
    pub size: [usize; 3],
    pub origin: [f64; 3],
}

impl MeshIndex {
    pub fn new(size: [usize; 3], origin: [f64; 3]) -> Result<Self> {
        if size.contains(&0) || size.iter().any(|n| *n > i32::MAX as usize) {
            return Err(invalid("a k mesh needs a positive size in every direction"));
        }
        product(&size)?;
        if origin.iter().any(|v| !v.is_finite()) {
            return Err(invalid("a k mesh origin must be finite"));
        }
        Ok(Self { size, origin })
    }
    pub fn points(&self) -> usize {
        self.size.iter().product()
    }
    /// The mesh indices `(i_0, i_1, i_2)` of point `index`.
    pub fn indices(&self, index: usize) -> [usize; 3] {
        let [_, n1, n2] = self.size;
        [index / (n1 * n2), (index / n2) % n1, index % n2]
    }
    pub fn index(&self, at: [usize; 3]) -> usize {
        let [_, n1, n2] = self.size;
        (at[0] * n1 + at[1]) * n2 + at[2]
    }
    /// The index `steps` points further along `direction`, wrapping at the zone edge.
    pub fn advance(&self, index: usize, direction: usize, steps: usize) -> usize {
        let mut at = self.indices(index);
        at[direction] = (at[direction] + steps) % self.size[direction];
        self.index(at)
    }
    /// The index displaced by the integer mesh step `step`, wrapping at the zone edge.
    pub fn shift(&self, index: usize, step: [i32; 3]) -> usize {
        let origin = self.indices(index);
        let at: [usize; 3] = std::array::from_fn(|a| {
            (origin[a] as i64 + step[a] as i64).rem_euclid(self.size[a] as i64) as usize
        });
        self.index(at)
    }
    /// The fractional k-point of mesh indices `at`.
    pub fn fractional(&self, at: [usize; 3]) -> [f64; 3] {
        std::array::from_fn(|a| self.origin[a] + at[a] as f64 / self.size[a] as f64)
    }
    /// The fractional k-point of point `index`.
    pub fn point(&self, index: usize) -> [f64; 3] {
        self.fractional(self.indices(index))
    }
}

/// A converged crystal with the setup that makes its states at any k-point and the PAW
/// one-center geometry of their overlaps.
pub struct Sampler<'a> {
    pub crystal: &'a Crystal,
    pub state: &'a PeriodicResult,
    pub setup: &'a PeriodicSetup,
    /// Lattice vectors as columns (Bohr).
    pub lattice: Matrix3<f64>,
    /// Reciprocal lattice vectors `G_j` as columns (1/Bohr), `G_i . a_j = 2 pi delta_ij`.
    pub reciprocal: Matrix3<f64>,
    augmentation: Augmentation<'a>,
}

impl<'a> Sampler<'a> {
    /// Refuses an unconverged SCF (its states would belong to an unconverged Hamiltonian) and a
    /// setup that describes another cell.
    pub fn new(
        crystal: &'a Crystal,
        state: &'a PeriodicResult,
        setup: &'a PeriodicSetup,
    ) -> Result<Self> {
        if !state.converged {
            return Err(invalid(
                "the crystal's self-consistent field did not converge: its Bloch states would \
                 belong to an unconverged Hamiltonian",
            ));
        }
        let lattice = setup.lattice;
        let grid = sampling::lattice(&crystal.grid);
        if (lattice - grid).norm() > 1e-8 * lattice.norm() {
            return Err(invalid(
                "the setup lattice and the crystal grid describe different cells",
            ));
        }
        let reciprocal = 2.
            * PI
            * lattice
                .try_inverse()
                .ok_or_else(|| invalid("singular crystal lattice"))?
                .transpose();
        let augmentation = Augmentation::from_setup(setup, crystal)?;
        Ok(Self {
            crystal,
            state,
            setup,
            lattice,
            reciprocal,
            augmentation,
        })
    }

    /// Spin channels: two for a collinear spin-polarized crystal, one otherwise (a spinor
    /// crystal's frames carry both components).
    pub fn channels(&self) -> usize {
        self.crystal.spins
    }

    /// Electrons one band of a channel holds: 2 unpolarized, 1 per spin channel or spinor.
    pub fn degeneracy(&self) -> f64 {
        self.crystal.degeneracy()
    }

    /// `G_j` (1/Bohr).
    pub fn reciprocal_vector(&self, j: usize) -> [f64; 3] {
        self.reciprocal.column(j).into()
    }

    /// The PAW one-center geometry of the overlaps.
    pub fn augmentation(&self) -> &Augmentation<'a> {
        &self.augmentation
    }

    /// The frames of every point of `mesh`, per channel, for the bands `bands[channel]` (an empty
    /// list makes no frames for that channel). Each frame is handed to `inspect(channel, frame)`
    /// as it is made, so a check can refuse a point before the whole mesh is built.
    pub fn frames(
        &self,
        mesh: &MeshIndex,
        bands: &[Vec<usize>],
        mut inspect: impl FnMut(usize, &Frame) -> Result<()>,
    ) -> Result<Vec<Vec<Frame>>> {
        let mut out: Vec<Vec<Frame>> = (0..self.channels()).map(|_| vec![]).collect();
        self.visit(mesh, bands, |s, frame| {
            inspect(s, &frame)?;
            out[s].push(frame);
            Ok(())
        })?;
        Ok(out)
    }

    /// [`Self::frames`] without keeping them: each frame is handed to `visit(channel, frame)` in
    /// mesh order (point, then channel) and dropped after it, so a mesh-wide sum holds one chunk
    /// of points at a time whatever the mesh size.
    pub fn visit(
        &self,
        mesh: &MeshIndex,
        bands: &[Vec<usize>],
        mut visit: impl FnMut(usize, Frame) -> Result<()>,
    ) -> Result<()> {
        if bands.len() != self.channels() {
            return Err(invalid("a k mesh needs a band list for every spin channel"));
        }
        let ctx = &self.crystal.context;
        // The states come from the setup in chunks of points: one call per point repeats the
        // setup's per-call work (the sphere tables, the projector sums) for every point, and the
        // whole mesh at once can exceed the budget. A chunk takes at most a quarter of the free
        // budget (each point holds the AO samples and its matrices).
        let c = self.crystal;
        let per_point = product(&[c.m, c.grid.index.points(), 16])?
            .checked_add(product(&[c.m, c.m + c.p, 16, 4])?)
            .ok_or(Error::Overflow)?;
        let free = ctx.memory_limit().saturating_sub(ctx.memory_used());
        let chunk = (free / 4 / per_point.max(1)).clamp(1, mesh.points().max(1));
        ctx.run(|| {
            for first in (0..mesh.points()).step_by(chunk) {
                let last = (first + chunk).min(mesh.points());
                let fractional: Vec<[f64; 3]> = (first..last).map(|i| mesh.point(i)).collect();
                let weights = vec![1. / fractional.len() as f64; fractional.len()];
                let points = self.setup.matrices_at(ctx, &fractional, &weights)?;
                if points.len() != fractional.len() {
                    return Err(invalid("the setup made the wrong number of mesh k-points"));
                }
                for point in points {
                    for (s, selection) in bands.iter().enumerate() {
                        if selection.is_empty() {
                            continue;
                        }
                        visit(
                            s,
                            Frame::from_point(self.crystal, self.state, &point, s, selection)?,
                        )?;
                    }
                }
            }
            Ok(())
        })
    }

    /// The overlap `M_mn = <psi_m(left)| e^{-ib.r} |psi_n(right)>` with the true displacement `b`
    /// (1/Bohr), smooth part and exact one-center part ([`sampling::link`]).
    pub fn overlap(&self, left: &Frame, right: &Frame, b: [f64; 3]) -> Result<Matrix> {
        sampling::link(self.crystal, left, right, b, &self.augmentation)
    }
}

/// The band group `bands` (ascending indices into the frame's levels) of `channel` is separated
/// from every band outside it by more than [`SEPARATION`] at this frame's k-point. `context`
/// names the consumer in the error.
pub fn check_isolated(frame: &Frame, channel: usize, bands: &[usize], context: &str) -> Result<()> {
    let levels = frame.levels.as_slice();
    let inside = |b: usize| bands.contains(&b);
    for &band in bands {
        if band >= levels.len() {
            return Err(invalid("a band index is beyond the levels of its k-point"));
        }
        for neighbor in [band.checked_sub(1), Some(band + 1)].into_iter().flatten() {
            if neighbor < levels.len()
                && !inside(neighbor)
                && (levels[band] - levels[neighbor]).abs() <= SEPARATION
            {
                return Err(Error::Invalid(format!(
                    "band {band} of channel {channel} touches band {neighbor} at k = {:?}: the \
                     group is not isolated, {context}",
                    frame.fractional
                )));
            }
        }
    }
    Ok(())
}
