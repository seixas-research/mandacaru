//! The electric polarization of an insulator (King-Smith and Vanderbilt; Resta): the electrons'
//! Berry phases along the three reciprocal vectors plus the ions' valence charges at their
//! positions, on the branch nearest zero, with its quantum. Atomic units (e/Bohr^2).
use super::{invalid, strings, Bands, Metal, Sampler};
use crate::Result;
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;

/// A string-phase spread (radians) above this is reported as a warning (owner decision
/// 2026-11-11: "Warn"): beyond it the pairwise differences of the string phases may exceed pi and
/// the branch of their average is no longer unique. It is that edge, not a bound on the error of
/// the transverse sampling, which only a comparison of two transverse meshes measures.
pub const SPREAD_WARNING: f64 = 0.5 * PI;

/// The warnings of a polarization whose string phases spread by more than [`SPREAD_WARNING`]
/// along some `G_j`: each names the direction, the spread and the remedy.
pub fn spread_warnings(spread: [f64; 3], size: [usize; 3]) -> Vec<String> {
    (0..3)
        .filter(|j| spread[*j] > SPREAD_WARNING)
        .map(|j| {
            format!(
                "the Berry phases of the strings along G_{j} spread by {:.3} rad (more than \
                 {SPREAD_WARNING:.3}) about their average on the transverse mesh of {size:?}, \
                 so the branch of the average is no longer unique: compare the result on a \
                 denser kpts transverse to G_{j}",
                spread[j]
            )
        })
        .collect()
}

pub struct Polarization {
    /// The string mesh `(n_1, n_2, n_3)`: `n_j` points per string along `G_j`.
    pub size: [usize; 3],
    /// Electrons per band: 2 unpolarized, 1 per spin channel or spinor.
    pub degeneracy: f64,
    /// `g = gcd(f, Z_1, Z_2, ...)`: the polarization is defined modulo `g e a_j / Omega`.
    pub quantum: u32,
    /// `[channel][j]`: the string-averaged Berry phase along `G_j` (radians, (-pi, pi]).
    pub phases: Vec<[f64; 3]>,
    /// `[j]`: the largest distance of a string phase from its channel's average.
    pub spread: [f64; 3],
    /// Coordinates in units of the quanta: electronic `-(f/g) sum_s gamma_s / 2 pi`, ionic
    /// `sum_a Z_a s_a / g`, `raw` their sum and `fractional` the sum folded onto (-1/2, 1/2].
    pub electronic: [f64; 3],
    pub ionic: [f64; 3],
    pub raw: [f64; 3],
    pub fractional: [f64; 3],
    /// Rows `g e a_j / Omega` (e/Bohr^2).
    pub quanta: [[f64; 3]; 3],
    /// `fractional . quanta`, the polarization on the branch nearest zero (e/Bohr^2).
    pub vector: [f64; 3],
    /// `volume`, Bohr^3.
    pub volume: f64,
    /// The smallest singular value of every link used.
    pub minimum: f64,
    /// [`spread_warnings`] of this result; the value is returned with them.
    pub warnings: Vec<String>,
}

impl Polarization {
    /// `coordinates . quanta` (e/Bohr^2).
    pub fn cartesian(&self, coordinates: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|a| (0..3).map(|j| coordinates[j] * self.quanta[j][a]).sum())
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// `gcd(f, Z_1, Z_2, ...)`, or `f` alone when a valence charge is not an integer.
pub fn quantum(degeneracy: f64, charges: &[f64]) -> Result<u32> {
    if !(degeneracy == 1. || degeneracy == 2.) || charges.iter().any(|z| !z.is_finite()) {
        return Err(invalid("invalid electrons per band or valence charges"));
    }
    let f = degeneracy as u32;
    if charges.iter().any(|z| (z - z.round()).abs() > 1e-8) {
        return Ok(f);
    }
    Ok(charges
        .iter()
        .fold(f, |g, z| gcd(g, z.round().abs() as u32)))
}

/// The polarization from per-channel Berry phases (`phases[s][j]`) and the ions: the
/// assembly, separate from the strings so its bookkeeping is testable without a crystal.
pub fn assemble(
    lattice: Matrix3<f64>,
    centers: &[[f64; 3]],
    charges: &[f64],
    degeneracy: f64,
    phases: Vec<[f64; 3]>,
) -> Result<Polarization> {
    if centers.len() != charges.len() || phases.is_empty() {
        return Err(invalid(
            "one valence charge per atom and one phase set per channel",
        ));
    }
    let volume = lattice.determinant().abs();
    let inverse = lattice
        .try_inverse()
        .ok_or_else(|| invalid("singular crystal lattice"))?;
    let g = quantum(degeneracy, charges)?;
    let gf = g as f64;
    let mut ionic = [0.; 3];
    for (center, z) in centers.iter().zip(charges) {
        let s = inverse * Vector3::from(*center);
        for j in 0..3 {
            ionic[j] += z * s[j] / gf;
        }
    }
    let electronic: [f64; 3] = std::array::from_fn(|j| {
        -(degeneracy / gf) * phases.iter().map(|p| p[j]).sum::<f64>() / (2. * PI)
    });
    let raw: [f64; 3] = std::array::from_fn(|j| ionic[j] + electronic[j]);
    let fractional = raw.map(|x| {
        let folded = x - x.round();
        if folded <= -0.5 + 1e-12 {
            folded + 1.
        } else {
            folded
        }
    });
    let quanta: [[f64; 3]; 3] =
        std::array::from_fn(|j| std::array::from_fn(|a| gf * lattice[(a, j)] / volume));
    let mut out = Polarization {
        size: [0; 3],
        degeneracy,
        quantum: g,
        phases,
        spread: [0.; 3],
        electronic,
        ionic,
        raw,
        fractional,
        quanta,
        vector: [0.; 3],
        volume,
        minimum: f64::INFINITY,
        warnings: vec![],
    };
    out.vector = out.cartesian(fractional);
    Ok(out)
}

/// The polarization of an insulating crystal from strings cut from the mesh `size`.
pub fn polarization(sampler: &Sampler, size: [usize; 3]) -> Result<Polarization> {
    if size.iter().any(|n| *n < 2) {
        return Err(invalid(format!(
            "the polarization needs at least 2 k-points per string in every direction; the mesh \
             is {size:?}"
        )));
    }
    if sampler.setup.charge.is_some_and(|q| q != 0.) {
        return Err(invalid("a charged cell has no polarization"));
    }
    let manifold = sampler.manifold(Bands::Occupied, Metal::Undefined)?;
    let lattice = sampler.lattice(size, [0.; 3], &manifold)?;
    let channels = lattice.frames.len();
    let mut phases = vec![[0.; 3]; channels];
    let mut spread = [0f64; 3];
    let mut minimum = f64::INFINITY;
    for j in 0..3 {
        let set = strings::strings(sampler, &lattice, j)?;
        for s in 0..channels {
            phases[s][j] = set.mean[s];
            spread[j] = spread[j].max(set.spread[s]);
        }
        minimum = minimum.min(set.minimum);
    }
    let charges = sampler
        .setup
        .atoms_species
        .iter()
        .map(|&species| {
            sampler
                .setup
                .datasets
                .get(species)
                .map(|d| d.valence_charge)
                .ok_or_else(|| invalid("atom dataset index is out of range"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut out = assemble(
        sampler.lattice,
        &sampler.setup.centers,
        &charges,
        sampler.degeneracy(),
        phases,
    )?;
    out.size = size;
    out.spread = spread;
    out.warnings = spread_warnings(spread, size);
    out.minimum = minimum;
    Ok(out)
}
