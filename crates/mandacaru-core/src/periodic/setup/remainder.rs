//! What the compensation charges leave of the augmentation density of one atom's projector
//! pairs: the reciprocal-space transforms of the remainder `rho^1_ab - rho~^1_ab - n^_ab`
//! ([`Remainder`]) and the Coulomb energy inside the sphere that the smooth pair densities with
//! their compensation charges do not carry ([`onsite_remainder`]).
//!
//! The transform of a pair is `sum_LM 4 pi (-i)^L Y_LM(g^) E^(L)_ab(|g|) <Y_LM|ab> exp(-i g.R)`,
//! `E^(L)` the Fourier-Bessel transform of the all-electron minus smooth pair density less the
//! compensation shape that carries its multipole moment. `E^(L)` is tabulated once on a uniform
//! grid in `|g|` and read by linear interpolation: the table step is part of the number, a
//! discretization of the Python era that this port reproduces (two-by-ten-to-the-minus-four of
//! the largest entry against a table four times finer).
use super::shapes::Shapes;
use crate::{
    paw::{
        multipole::{gaunt, gaunt_grid, multipole_range},
        multipole_pairs::{radial_moments, radial_remainder, smooth_waves, ChannelWaves, Pair},
        onecenter::{one_center_tensor, Basis, Channel, Projector, Sphere, Tensor},
        record::PawDataset,
    },
    periodic::invalid,
    product, Context, Result,
};
use nalgebra::DMatrix;
use num_complex::Complex64;
use std::{collections::BTreeSet, f64::consts::PI};

/// Step and end (Bohr^-1) of the table of `E^(L)(q)`. The table holds `E^(L)` up to its last point
/// (`REMAINDER_TABLE_MAX` plus two steps, 12.025) and a vector past it raises. The Python era read
/// the last entry there; that is a wrong number, not a convention (extending the table to 13
/// moved the transform of the one frozen vector at 12.717 by 8.5e-3 of the largest entry), and the
/// beyond-multipole remainder has no production caller to depend on it.
pub const REMAINDER_TABLE_STEP: f64 = 0.025;
pub const REMAINDER_TABLE_MAX: f64 = 12.;
/// Inside the table the last cell is read at its upper end at most: a shift of `1e-6` of a step,
/// the number the Python era produced.
const CLAMP: f64 = 1.000001;

/// The columns of the projections `C[:, p]` that hold `atom`'s projectors, given the atom each
/// projector belongs to.
pub fn projector_columns(owners: &[usize], atom: usize) -> Vec<usize> {
    owners
        .iter()
        .enumerate()
        .filter(|(_, &owner)| owner == atom)
        .map(|(p, _)| p)
        .collect()
}

/// One channel's partial waves on the dataset grid: the all-electron waves, the smooth waves
/// rebuilt from their Bessel expansions (beyond `r_cut` they are the all-electron ones) and the
/// Vanderbilt matrix.
struct Partial {
    l: usize,
    waves: usize,
    ae: Vec<f64>,
    ps: Vec<f64>,
    vanderbilt: Vec<f64>,
    overlap: Vec<f64>,
    r_cut: f64,
}

fn partials(dataset: &PawDataset) -> Result<Vec<Partial>> {
    dataset
        .channels
        .iter()
        .map(|(&l, channel)| {
            let waves = channel.ae_waves.len();
            let flat = |tables: &[crate::Buffer<f64>]| -> Vec<f64> {
                tables
                    .iter()
                    .flat_map(|t| t.as_slice().iter().copied())
                    .collect()
            };
            let ae = flat(&channel.ae_waves);
            let coefficients: Vec<f64> = channel
                .wave_coefficients
                .iter()
                .flat_map(|a| a.data().iter().copied())
                .collect();
            let wavevectors: Vec<f64> = channel
                .wavevectors
                .iter()
                .flat_map(|a| a.data().iter().copied())
                .collect();
            let ps = smooth_waves(
                dataset.r.as_slice(),
                &ChannelWaves {
                    l: l as usize,
                    r_cut: channel.r_cut,
                    ae: &ae,
                    coefficients: &coefficients,
                    wavevectors: &wavevectors,
                    waves,
                },
            )?;
            Ok(Partial {
                l: l as usize,
                waves,
                ae,
                ps,
                vanderbilt: channel.vanderbilt.data().to_vec(),
                overlap: channel.overlap_correction.data().to_vec(),
                r_cut: channel.r_cut,
            })
        })
        .collect()
}

/// `DeltaW[a, b, c, d]` of an atom's projectors under the bare Coulomb kernel: the
/// all-electron pair-pair energy less the smooth-plus-compensation one inside the augmentation
/// sphere. Not symmetrized under the exchange of the two pairs (the radial integral is not
/// exact to that order, about 1e-5 relative).
pub fn onsite_remainder(
    ctx: &Context,
    dataset: &PawDataset,
    basis: Basis,
    projectors: &[Projector],
) -> Result<Tensor> {
    let partial = partials(dataset)?;
    let sphere = Sphere {
        r: dataset.r.as_slice(),
        channels: partial
            .iter()
            .map(|p| Channel {
                l: p.l,
                waves: p.waves,
                ae: &p.ae,
                ps: &p.ps,
                vanderbilt: &p.vanderbilt,
                r_cut: p.r_cut,
            })
            .collect(),
        compensation_radius: dataset.compensation_radius,
    };
    one_center_tensor(ctx, &sphere, basis, projectors, 0.)
}

/// One multipole level: the table of `E^(L)_ab(q)` (`points x n x n`) and the angular couplings
/// of each `M` that couples any pair.
struct Level {
    l: usize,
    table: Vec<f64>,
    couplings: Vec<(i32, Vec<Complex64>)>,
}

/// The transforms of one atom's augmentation remainders, built once.
pub struct Remainder {
    n: usize,
    points: usize,
    center: [f64; 3],
    levels: Vec<Level>,
}

impl Remainder {
    /// The tables of `projectors` (this atom's, in the order the pair indices follow) of
    /// `dataset` in the projector `basis`, centered at `center`. The compensation shapes' tables
    /// come from `shapes`, which must hold every `L` the pairs carry for the dataset's radius.
    pub fn new(
        ctx: &Context,
        dataset: &PawDataset,
        shapes: &Shapes,
        basis: Basis,
        projectors: &[Projector],
        center: [f64; 3],
    ) -> Result<Self> {
        if projectors.is_empty() || center.iter().any(|x| !x.is_finite()) {
            return Err(invalid("a remainder needs projectors and a finite center"));
        }
        let n = projectors.len();
        let points = (REMAINDER_TABLE_MAX / REMAINDER_TABLE_STEP) as usize + 2;
        let q: Vec<f64> = (0..points)
            .map(|i| i as f64 * REMAINDER_TABLE_STEP)
            .collect();
        let partial = partials(dataset)?;
        let channel = |l: usize| {
            partial
                .iter()
                .find(|p| p.l == l)
                .ok_or_else(|| invalid(format!("the dataset has no channel l = {l}")))
        };
        let distinct: BTreeSet<usize> = projectors.iter().map(|p| p.l).collect();
        let levels_needed: BTreeSet<usize> = projectors
            .iter()
            .flat_map(|a| projectors.iter().flat_map(|b| multipole_range(a.l, b.l)))
            .collect();
        let _held = ctx.reserve(product(&[levels_needed.len(), points, n * n, 8])?)?;
        let grid = gaunt_grid()?;
        let mut levels = Vec::new();
        for &big in &levels_needed {
            let shape = shapes.table(dataset.compensation_radius, big)?;
            let g: Vec<f64> = q.iter().map(|x| shape.at(*x)).collect();
            // E^(L)(q) for every (l1, l2) that can carry L, in the basis of the projectors.
            let mut radial = std::collections::BTreeMap::new();
            for &l1 in &distinct {
                for &l2 in &distinct {
                    if !multipole_range(l1, l2).contains(&big) {
                        continue;
                    }
                    let (c1, c2) = (channel(l1)?, channel(l2)?);
                    let pair = Pair {
                        r: dataset.r.as_slice(),
                        r_cut: c1.r_cut.max(c2.r_cut),
                        ae: [&c1.ae, &c2.ae],
                        smooth: [&c1.ps, &c2.ps],
                        waves: [c1.waves, c2.waves],
                    };
                    // The monopole between a channel and itself is the stored overlap
                    // correction; the rebuilt one agrees only to 1e-8, which the raw basis
                    // amplifies.
                    let delta = if l1 == l2 && big == 0 {
                        c1.overlap.clone()
                    } else {
                        radial_moments(&pair, big)?
                    };
                    let mut e = radial_remainder(ctx, &pair, big, &q, &delta, &g)?;
                    if basis == Basis::Raw {
                        let inverse = |c: &Partial| -> Result<DMatrix<f64>> {
                            DMatrix::from_row_slice(c.waves, c.waves, &c.vanderbilt)
                                .try_inverse()
                                .ok_or_else(|| invalid("a singular Vanderbilt matrix"))
                        };
                        let (b1, b2) = (inverse(c1)?, inverse(c2)?);
                        for block in e.chunks_mut(c1.waves * c2.waves) {
                            let m = DMatrix::from_row_slice(c1.waves, c2.waves, block);
                            let t = &b1 * m * b2.transpose();
                            for a in 0..c1.waves {
                                for b in 0..c2.waves {
                                    block[a * c2.waves + b] = t[(a, b)];
                                }
                            }
                        }
                    }
                    radial.insert((l1, l2), (e, c2.waves));
                }
            }
            let mut table = vec![0.; points * n * n];
            for (a, pa) in projectors.iter().enumerate() {
                for (b, pb) in projectors.iter().enumerate() {
                    if let Some((e, width)) = radial.get(&(pa.l, pb.l)) {
                        let waves1 = channel(pa.l)?.waves;
                        for i in 0..points {
                            table[(i * n + a) * n + b] =
                                e[(i * waves1 + pa.index) * width + pb.index];
                        }
                    }
                }
            }
            let mut couplings = Vec::new();
            for m in -(big as i32)..=big as i32 {
                let mut angular = vec![Complex64::new(0., 0.); n * n];
                for (a, pa) in projectors.iter().enumerate() {
                    for (b, pb) in projectors.iter().enumerate() {
                        if multipole_range(pa.l, pb.l).contains(&big) {
                            angular[a * n + b] =
                                gaunt([big, pa.l, pb.l], [m, pa.m, pb.m], pa.real, &grid);
                        }
                    }
                }
                if angular.iter().any(|x| x.norm() != 0.) {
                    couplings.push((m, angular));
                }
            }
            levels.push(Level {
                l: big,
                table,
                couplings,
            });
        }
        Ok(Self {
            n,
            points,
            center,
            levels,
        })
    }

    /// Number of projectors.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The transforms at `vectors` (Bohr^-1): `out[(a * n + b) * vectors.len() + v]`, `E_ab(g)`
    /// of the unit-coefficient pair density `phi_a^* phi_b`.
    pub fn transforms(&self, vectors: &[[f64; 3]], out: &mut [Complex64]) -> Result<()> {
        let (n, count) = (self.n, vectors.len());
        if out.len() != n * n * count {
            return Err(invalid("the output holds n * n * vectors entries"));
        }
        out.fill(Complex64::new(0., 0.));
        for (v, g) in vectors.iter().enumerate() {
            let norm = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            if norm > (self.points - 1) as f64 * REMAINDER_TABLE_STEP {
                return Err(invalid(format!(
                    "the remainder table ends at |g| = {:.3} Bohr^-1 and a vector of {norm:.3} \
                     was asked for: extend REMAINDER_TABLE_MAX",
                    (self.points - 1) as f64 * REMAINDER_TABLE_STEP
                )));
            }
            let scale = (norm / REMAINDER_TABLE_STEP).min(self.points as f64 - CLAMP);
            let low = scale.floor();
            let frac = scale - low;
            let low = low as usize;
            // The zero vector reads as +x; every L > 0 table vanishes there.
            let direction = if norm > 0. {
                [g[0] / norm, g[1] / norm, g[2] / norm]
            } else {
                [1., 0., 0.]
            };
            let angle = -(g[0] * self.center[0] + g[1] * self.center[1] + g[2] * self.center[2]);
            let phase = Complex64::new(angle.cos(), angle.sin());
            for level in &self.levels {
                let unit = [
                    Complex64::new(1., 0.),
                    Complex64::new(0., -1.),
                    Complex64::new(-1., 0.),
                    Complex64::new(0., 1.),
                ][level.l % 4];
                for (m, angular) in &level.couplings {
                    let harmonic = unit
                        * phase
                        * crate::math::angular::complex(level.l, *m, direction)
                        * (4. * PI);
                    for ab in 0..n * n {
                        let radial = level.table[low * n * n + ab] * (1. - frac)
                            + level.table[(low + 1) * n * n + ab] * frac;
                        out[ab * count + v] += angular[ab] * radial * harmonic;
                    }
                }
            }
        }
        Ok(())
    }
}
