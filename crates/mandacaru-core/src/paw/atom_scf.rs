//! The isolated atom of a dataset through the engine's own periodic spinor SCF (gate G13 and the
//! spin-orbit calibration of the generator).
//!
//! One atom sits in the middle of a cubic periodic box, Gamma only, in the DZP basis the engine
//! builds for the dataset and under the dataset's own functional, with a Fermi-Dirac occupation.
//! This is the crystal setup, build and solve of `periodic::setup` and `periodic::solver`, not a
//! second implementation: the eigenvalues are those the engine gives a Bi atom in a crystal run.
//! A spherical atom puts its levels in `j` multiplets (`2j + 1` states, split by the cubic box by
//! a few meV): [`clusters`] groups them.
//!
//! The Fermi level of an open shell sits inside a multiplet and puts the same fraction of an
//! electron on each of its states, so the density stays spherical; a narrow smearing of a
//! half-filled `d` or `f` shell oscillates between the near-degenerate multiplets and does not
//! converge. [`levels`] therefore tries a ladder of widths, the narrowest first, and reports the
//! one that converged.
use super::{invalid, record::PawDataset};
use crate::{
    periodic::{spectrum::Smearing, Options},
    Context, Result,
};
use serde_json::json;
use std::sync::Arc;

/// The text an unconverged isolated atom's error begins with ([`not_converged`]).
const NOT_CONVERGED: &str = "the isolated atom did not converge";

/// Whether `error` is the one [`levels`] raises when no width and mixing of the ladder converges
/// the SCF (a half-filled open shell may not): the atom cannot be measured, which is not a
/// failure to run.
pub fn not_converged(error: &crate::Error) -> bool {
    error.to_string().contains(NOT_CONVERGED)
}

const HARTREE_EV: f64 = 27.211_386_245_988;
const ANGSTROM: f64 = 1.0 / 0.529_177_210_903;

/// Box and numerical controls of the atom (the values the calibration of the library was made at).
#[derive(Clone, Copy, Debug)]
pub struct Controls {
    /// Edge of the cubic cell (Angstrom).
    pub box_edge: f64,
    /// Grid spacing (Angstrom).
    pub spacing: f64,
    /// Fermi-Dirac widths (eV) tried in turn until the SCF converges.
    pub widths: &'static [f64],
    /// Linear mixing of the density, tried in turn at each width (a half-filled shell may need
    /// the smaller step).
    pub mixings: &'static [f64],
    pub max_iterations: usize,
    /// Only this width (eV) is tried, not the ladder: the rounds of one calibration measure an
    /// open shell at the occupations of its first round.
    pub only_width: Option<f64>,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            box_edge: 10.,
            spacing: 0.2,
            widths: &[0.02, 0.1, 0.4],
            mixings: &[0.25, 0.08],
            max_iterations: 150,
            only_width: None,
        }
    }
}

/// One spinor level of the atom at Gamma.
#[derive(Clone, Copy, Debug)]
pub struct Level {
    /// Hartree, in the engine's zero (the mean electrostatic potential of the box), not the
    /// vacuum or the all-electron one.
    pub energy: f64,
    /// Fermi-Dirac occupation, 0 to 1 per spinor state.
    pub occupation: f64,
}

/// The spinor levels of the atom at Gamma.
#[derive(Clone, Debug)]
pub struct Levels {
    /// Every level, ascending in energy.
    pub levels: Vec<Level>,
    /// Fermi-Dirac width (eV) of the converged SCF.
    pub width: f64,
    /// Iterations of the converged SCF.
    pub iterations: usize,
    /// The total energy of the cell (Hartree). Two box edges of the same atom differ by the
    /// energy per volume a dataset's constants put into a crystal ([`super::validate`] G15).
    pub energy: f64,
}

/// A copy of `ds` the setup can own.
fn shared(ctx: &Context, ds: &PawDataset) -> Result<Arc<PawDataset>> {
    let parts = ds.parts(ctx)?;
    let mut tables = Vec::with_capacity(parts.tables.len());
    for (name, values) in &parts.tables {
        let mut buffer = ctx.allocate::<f64>(values.len())?;
        buffer.as_mut_slice().copy_from_slice(values);
        tables.push((name.clone(), buffer));
    }
    Ok(Arc::new(PawDataset::from_parts(ctx, &parts.text, tables)?))
}

/// The levels of the isolated atom of `ds`, by the narrowest width of `how.widths` whose SCF
/// converges. A scalar dataset runs as a collinear calculation, a Dirac one as a spinor.
pub fn levels(ctx: &Context, ds: &PawDataset, how: &Controls) -> Result<Levels> {
    let edge = how.box_edge * ANGSTROM;
    let count = (how.box_edge / how.spacing).round_ties_even().max(1.) as u64;
    let request = json!({
        "lattice": [edge, 0., 0., 0., edge, 0., 0., 0., edge],
        "grid_shape": [count, count, count],
        "atoms": [{"species": 0, "symbol": ds.symbol, "fractional": [0.5, 0.5, 0.5], "moment": 0.}],
        "mesh": {"size": [1, 1, 1], "gamma": true},
        "functional": ds.xc,
        "precision": "float64",
        "basis_size": "DZP",
        "spinor": !ds.spin_orbit.is_empty(),
        "spin": false,
        "symmetry": false,
        "charge": 0.,
        "projector_basis": "raw",
    });
    let input = crate::periodic::setup::parse_input_json(&request)?;
    let setup = crate::periodic::setup::create_setup(ctx, input, vec![shared(ctx, ds)?])?;
    let crystal = setup.build(ctx)?;
    let only = [how.only_width.unwrap_or(0.)];
    let ladder: &[f64] = if how.only_width.is_some() { &only } else { how.widths };
    for &width in ladder {
        for &mixing in how.mixings {
            let options = Options {
                max_iterations: how.max_iterations,
                smearing: Smearing::FermiDirac,
                width: width / HARTREE_EV,
                mixing,
                ..Options::default()
            };
            let result = crystal.solve(&options)?;
            if !result.converged {
                continue;
            }
            let energies = result
                .eigenvalues
                .first()
                .ok_or_else(|| invalid("the isolated atom has no k-point"))?;
            let occupations = result
                .occupations
                .first()
                .filter(|o| o.len() == energies.len())
                .ok_or_else(|| invalid("the isolated atom's occupations do not match its levels"))?;
            let mut levels: Vec<Level> = energies
                .iter()
                .zip(occupations)
                .map(|(&energy, &occupation)| Level { energy, occupation })
                .collect();
            levels.sort_by(|a, b| a.energy.total_cmp(&b.energy));
            return Ok(Levels {
                levels,
                width,
                iterations: result.history.len(),
                energy: result.energy,
            });
        }
    }
    Err(invalid(format!(
        "{NOT_CONVERGED} ({}) at Fermi-Dirac widths {:?} eV and mixings {:?} in {} iterations",
        ds.symbol, how.widths, how.mixings, how.max_iterations
    )))
}

/// A group of degenerate levels of the spherical atom (a `j` multiplet).
#[derive(Clone, Copy, Debug)]
pub struct Cluster {
    /// The lowest level of the group (Hartree).
    pub energy: f64,
    pub count: usize,
    /// Mean occupation per state.
    pub occupation: f64,
}

/// Degenerate groups of ascending `levels`: a level within `tolerance` (Hartree) of the first of
/// its group belongs to it. The cubic box splits a `j` multiplet by a few meV.
pub fn clusters(levels: &[Level], tolerance: f64) -> Vec<Cluster> {
    let mut out: Vec<Cluster> = Vec::new();
    for level in levels {
        match out.last_mut() {
            Some(c) if (level.energy - c.energy).abs() < tolerance => {
                c.occupation = (c.occupation * c.count as f64 + level.occupation)
                    / (c.count + 1) as f64;
                c.count += 1;
            }
            _ => out.push(Cluster {
                energy: level.energy,
                count: 1,
                occupation: level.occupation,
            }),
        }
    }
    out
}
