//! Berry phases, Wilson loops, Berry curvature, Chern numbers, the electric polarization and the
//! anomalous Hall conductivity of a converged crystal. Atomic units; `dev_plan/BERRY_PLAN.md`
//! holds the formulas and conventions.
//!
//! Everything is built on the Wannier sampler: [`Frame::from_point`] re-diagonalizes at a k-point
//! with the converged Hamiltonian (DFT+U operator, hybrid exchange `K(k)`, meta-GGA and spinor
//! potentials included), and [`sampling::link`] forms the PAW overlap
//! `M_mn(k, b) = <psi_mk| e^{-i b.r} |psi_n,k+b>`: the smooth part on the cell grid plus the
//! one-center monopole and dipole of every sphere with its phase `e^{-i b.R_a}` (first order in
//! `b`; the dropped second-order term is Hermitian and enters `arg det` only at `b^3`, so a string
//! of N points converges as `1/N^2`). No functional, spin treatment or correction needs a case here:
//! a spinor frame carries both components, a collinear channel is its own frame.
//!
//! A mesh is `k = origin + sum_j (i_j / n_j) G_j`; a link along `G_j` goes to the next point with
//! the true displacement `G_j / n_j`, reusing the frame of the wrapped point at the zone edge, so
//! every string closes in the periodic gauge.
pub mod curvature;
pub mod hall;
pub mod links;
pub mod polarization;
pub mod strings;
pub mod weighted;

use super::{
    invalid,
    setup::PeriodicSetup,
    wannier::sampling::{self, Augmentation, Frame},
    Crystal, PeriodicResult,
};
use crate::{product, quantum::C64, Error, Result};
use nalgebra::{DMatrix, Matrix3, Vector3};
use std::f64::consts::PI;

type Matrix = DMatrix<C64>;

/// A link whose smallest singular value is below this raises: its phase is undefined.
pub const LINK_TOLERANCE: f64 = 1e-6;
/// An occupation within this fraction of empty or full counts as that (an insulator).
pub const OCCUPATION_TOLERANCE: f64 = 1e-4;
/// The smallest gap (Hartree) that separates a band group from its neighbors.
pub const SEPARATION: f64 = 1e-6;

/// The bands a Berry quantity is taken over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bands {
    /// The occupied manifold of an insulator, per spin channel.
    Occupied,
    /// The half-open band range `lo..hi`, isolated from its neighbors at every k-point.
    Range(usize, usize),
}

/// How a metal answers a request for its occupied manifold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metal {
    /// The quantity does not exist for a metal (`Error::Invalid`).
    Undefined,
    /// The quantity exists for a metal but is not implemented (`Error::Unsupported`, naming what
    /// is missing).
    Unimplemented(&'static str),
}

impl Metal {
    fn error(self, reason: String) -> Error {
        match self {
            Self::Undefined => invalid(format!(
                "{reason}: the Berry phase of the occupied bands is defined for an insulator only"
            )),
            Self::Unimplemented(missing) => Error::Unsupported(format!("{reason}; {missing}")),
        }
    }
}

/// The band range of each channel a calculation is taken over.
#[derive(Clone, Debug)]
pub struct Manifold {
    /// Half-open band range per channel; `lo == hi` is an empty channel.
    pub ranges: Vec<(usize, usize)>,
    /// The occupied manifold of an insulator (its gaps checked at every point).
    pub occupied: bool,
    /// A band group that must be isolated from its neighbors at every point.
    pub isolated: bool,
    pub metal: Metal,
}

impl Manifold {
    /// The bands `0..cap` of every channel, unchecked: the window of a metal's weighted field,
    /// which cuts its own groups.
    pub fn window(channels: usize, cap: usize) -> Self {
        Self {
            ranges: vec![(0, cap); channels],
            occupied: false,
            isolated: false,
            metal: Metal::Undefined,
        }
    }
}

/// The converged crystal, its setup (which makes states at any k-point) and the PAW one-center
/// geometry of the links.
pub struct Sampler<'a> {
    pub crystal: &'a Crystal,
    pub state: &'a PeriodicResult,
    pub setup: &'a PeriodicSetup,
    /// Lattice vectors as columns (Bohr).
    pub lattice: Matrix3<f64>,
    /// Reciprocal lattice vectors `G_j` as columns (1/Bohr), `G_i . a_j = 2 pi delta_ij`.
    pub reciprocal: Matrix3<f64>,
    augmentation: Augmentation,
}

/// The frames of a mesh, per channel.
pub struct Lattice {
    pub size: [usize; 3],
    pub origin: [f64; 3],
    /// Frames of each channel in C order of the mesh; empty for an empty channel.
    pub frames: Vec<Vec<Frame>>,
    pub ranges: Vec<(usize, usize)>,
}

impl Lattice {
    pub fn points(&self) -> usize {
        self.size.iter().product()
    }
    /// The mesh indices `(i_0, i_1, i_2)` of point `index` (C order).
    pub fn indices(&self, index: usize) -> [usize; 3] {
        let [_, n1, n2] = self.size;
        [index / (n1 * n2), (index / n2) % n1, index % n2]
    }
    pub fn index(&self, at: [usize; 3]) -> usize {
        let [_, n1, n2] = self.size;
        (at[0] * n1 + at[1]) * n2 + at[2]
    }
    /// The index of the next point along `direction`, wrapping at the zone edge.
    pub fn next(&self, index: usize, direction: usize) -> usize {
        self.advance(index, direction, 1)
    }
    /// The index `steps` points further along `direction`, wrapping at the zone edge.
    pub fn advance(&self, index: usize, direction: usize, steps: usize) -> usize {
        let mut at = self.indices(index);
        at[direction] = (at[direction] + steps) % self.size[direction];
        self.index(at)
    }
    /// The fractional k-point of mesh index `at`.
    pub fn fractional(&self, at: [usize; 3]) -> [f64; 3] {
        std::array::from_fn(|a| self.origin[a] + at[a] as f64 / self.size[a] as f64)
    }
    /// The bands of `channel`.
    pub fn bands(&self, channel: usize) -> usize {
        let (lo, hi) = self.ranges[channel];
        hi - lo
    }
}

fn mesh_size(size: [usize; 3]) -> Result<()> {
    if size.contains(&0) {
        return Err(invalid(
            "a Berry k mesh needs a positive size in every direction",
        ));
    }
    product(&size)?;
    Ok(())
}

impl<'a> Sampler<'a> {
    pub fn new(
        crystal: &'a Crystal,
        state: &'a PeriodicResult,
        setup: &'a PeriodicSetup,
    ) -> Result<Self> {
        if !state.converged {
            return Err(invalid(
                "the crystal's self-consistent field did not converge: its Berry phases would \
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
        let augmentation = Augmentation {
            centers: setup.centers.clone(),
            labels: setup
                .channels
                .iter()
                .map(|&(atom, l, m)| {
                    Ok([
                        i32::try_from(atom).map_err(|_| Error::Overflow)?,
                        i32::try_from(l).map_err(|_| Error::Overflow)?,
                        m,
                    ])
                })
                .collect::<Result<Vec<_>>>()?,
        };
        augmentation.validate(crystal)?;
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

    /// The occupied bands of an insulator, from the SCF occupations: every one empty or full
    /// within [`OCCUPATION_TOLERANCE`], one count per channel at every SCF point, and the counts
    /// holding the cell's electrons.
    pub fn manifold(&self, bands: Bands, metal: Metal) -> Result<Manifold> {
        let c = self.crystal;
        let channels = self.channels();
        match bands {
            Bands::Range(lo, hi) => {
                if lo >= hi || hi > c.bands() {
                    return Err(invalid(format!(
                        "a Berry band group lo..hi needs 0 <= lo < hi <= {}; got {lo}..{hi}",
                        c.bands()
                    )));
                }
                Ok(Manifold {
                    ranges: vec![(lo, hi); channels],
                    occupied: false,
                    isolated: true,
                    metal,
                })
            }
            Bands::Occupied => {
                let full = self.degeneracy();
                let nk = c.points.len();
                if self.state.occupations.len() != channels * nk {
                    return Err(invalid("the crystal occupations do not match its k-points"));
                }
                let mut counts = Vec::with_capacity(channels);
                for s in 0..channels {
                    let mut count = None;
                    for (k, point) in c.points.iter().enumerate() {
                        let row = &self.state.occupations[s * nk + k];
                        if let Some((n, f)) = row.iter().enumerate().find(|(_, f)| {
                            **f > OCCUPATION_TOLERANCE * full
                                && **f < (1. - OCCUPATION_TOLERANCE) * full
                        }) {
                            return Err(metal.error(format!(
                                "band {n} of channel {s} at the SCF k-point {:?} holds {f:.6} of \
                                 {full} electrons (a metal, or a smearing wider than the gap: \
                                 narrow the smearing width)",
                                point.fractional
                            )));
                        }
                        let here = row.iter().filter(|f| **f > 0.5 * full).count();
                        if count.is_some_and(|n| n != here) {
                            return Err(metal.error(format!(
                                "channel {s} has a different number of occupied bands at the SCF \
                                 k-point {:?} (a metal or semimetal)",
                                point.fractional
                            )));
                        }
                        count = Some(here);
                    }
                    counts.push(count.unwrap_or(0));
                }
                let held = counts.iter().sum::<usize>() as f64 * full;
                if (held - c.electrons).abs() > 1e-6 {
                    return Err(metal.error(format!(
                        "the occupied bands hold {held} electrons and the cell {}",
                        c.electrons
                    )));
                }
                Ok(Manifold {
                    ranges: counts.into_iter().map(|n| (0, n)).collect(),
                    occupied: true,
                    isolated: false,
                    metal,
                })
            }
        }
    }

    /// The frames of the mesh `origin + i / size` for the bands of `manifold`, each point checked:
    /// a band group is separated from its neighbors by more than [`SEPARATION`]; the occupied
    /// manifold, in addition, lies wholly below the empty bands over the whole mesh (every
    /// channel). The Fermi level is not used: a narrow smearing only places it somewhere in the
    /// gap of the SCF points.
    pub fn lattice(
        &self,
        size: [usize; 3],
        origin: [f64; 3],
        manifold: &Manifold,
    ) -> Result<Lattice> {
        mesh_size(size)?;
        if origin.iter().any(|v| !v.is_finite()) {
            return Err(invalid("a Berry mesh origin must be finite"));
        }
        if manifold.ranges.len() != self.channels() {
            return Err(invalid("the Berry manifold must name every spin channel"));
        }
        let ctx = &self.crystal.context;
        ctx.run(|| {
            let mut out = Lattice {
                size,
                origin,
                frames: (0..self.channels()).map(|_| vec![]).collect(),
                ranges: manifold.ranges.clone(),
            };
            // The highest occupied and lowest empty level over every channel and point: an
            // insulator keeps them apart (a semimetal overlaps them with a direct gap everywhere).
            let (mut top, mut bottom) = (f64::NEG_INFINITY, f64::INFINITY);
            for index in 0..out.points() {
                let fractional = out.fractional(out.indices(index));
                let point = self
                    .setup
                    .matrices_at(ctx, &[fractional], &[1.])?
                    .into_iter()
                    .next()
                    .ok_or_else(|| invalid("the setup made no states at a Berry k-point"))?;
                for (s, &(lo, hi)) in manifold.ranges.iter().enumerate() {
                    if lo == hi {
                        continue;
                    }
                    let bands: Vec<usize> = (lo..hi).collect();
                    let frame = Frame::from_point(self.crystal, self.state, &point, s, &bands)?;
                    Self::check_point(&frame, s, lo, hi, manifold)?;
                    let levels = frame.levels.as_slice();
                    top = top.max(levels[hi - 1]);
                    if let Some(empty) = levels.get(hi) {
                        bottom = bottom.min(*empty);
                    }
                    out.frames[s].push(frame);
                }
            }
            if manifold.occupied && bottom - top <= SEPARATION {
                return Err(manifold.metal.error(format!(
                    "the highest occupied level of the mesh ({top:.6} Ha) is not below its \
                     lowest empty level ({bottom:.6} Ha): the bands overlap (a metal or semimetal)"
                )));
            }
            Ok(out)
        })
    }

    /// The band group `lo..hi` of `channel` is separated from its neighbors at this point by more
    /// than [`SEPARATION`]; for the occupied manifold a touching gap is a metal.
    fn check_point(
        frame: &Frame,
        channel: usize,
        lo: usize,
        hi: usize,
        manifold: &Manifold,
    ) -> Result<()> {
        let levels = frame.levels.as_slice();
        let at = frame.fractional;
        if !manifold.occupied && !manifold.isolated {
            return Ok(());
        }
        if manifold.occupied {
            if hi < levels.len() && levels[hi] - levels[hi - 1] <= SEPARATION {
                return Err(manifold.metal.error(format!(
                    "the occupied band {} of channel {channel} touches band {hi} at k = {at:?} \
                     (a metal)",
                    hi - 1
                )));
            }
            return Ok(());
        }
        if lo > 0 && levels[lo] - levels[lo - 1] <= SEPARATION {
            return Err(invalid(format!(
                "bands {lo}..{hi} of channel {channel} touch band {} at k = {at:?}: the group is \
                 not isolated",
                lo - 1
            )));
        }
        if hi < levels.len() && levels[hi] - levels[hi - 1] <= SEPARATION {
            return Err(invalid(format!(
                "bands {lo}..{hi} of channel {channel} touch band {hi} at k = {at:?}: the group \
                 is not isolated"
            )));
        }
        Ok(())
    }

    /// The link `M(k, k + G_j / n_j)` from point `index` of `lattice` to the next point along
    /// `direction`, for `channel`'s bands.
    pub fn link(
        &self,
        lattice: &Lattice,
        channel: usize,
        index: usize,
        direction: usize,
    ) -> Result<Matrix> {
        self.link_stride(lattice, channel, index, direction, 1)
    }

    /// The link over `stride` mesh steps along `direction`, with the true displacement
    /// `stride G_j / n_j` (a coarser sub-mesh of the same states).
    pub fn link_stride(
        &self,
        lattice: &Lattice,
        channel: usize,
        index: usize,
        direction: usize,
        stride: usize,
    ) -> Result<Matrix> {
        if direction > 2
            || channel >= lattice.frames.len()
            || index >= lattice.points()
            || stride == 0
        {
            return Err(invalid("invalid Berry link request"));
        }
        let frames = &lattice.frames[channel];
        if frames.is_empty() {
            return Err(invalid("an empty band channel has no links"));
        }
        let b = Vector3::from(self.reciprocal_vector(direction)) * stride as f64
            / lattice.size[direction] as f64;
        sampling::link(
            self.crystal,
            &frames[index],
            &frames[lattice.advance(index, direction, stride)],
            b.into(),
            &self.augmentation,
        )
    }
}

/// `(a, b)`: the in-plane reciprocal directions of the plane normal to `G_c`, in cyclic order.
pub fn plane_axes(normal: usize) -> Result<[usize; 2]> {
    if normal > 2 {
        return Err(invalid(
            "a k plane is named by its normal G_c, c in 0, 1, 2",
        ));
    }
    Ok([(normal + 1) % 3, (normal + 2) % 3])
}

/// The two directions transverse to strings along `direction`, ascending.
pub fn transverse_axes(direction: usize) -> Result<[usize; 2]> {
    match direction {
        0 => Ok([1, 2]),
        1 => Ok([0, 2]),
        2 => Ok([0, 1]),
        _ => Err(invalid("a Berry string runs along G_j, j in 0, 1, 2")),
    }
}
