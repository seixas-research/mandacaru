//! The `j` splitting of the isolated atom through the engine's own spin-orbit operator (gate G13).
//!
//! A Dirac PAW-LCAO dataset is the scalar (`j`-averaged) construction plus an `L.S` coupling on the
//! `j` branches' own projectors. The engine's atom in the DZP basis of the same dataset is, for a
//! channel `l`,
//!
//! ```text
//! H_j = T + v~ + sum |p~_a> D_ab <p~_b|  +  (L.S)_j  sum |p~so_a> coupling_ab <p~so_b|,
//! S   = 1 + sum |p~_a> q_ab <p~_b|,
//! ```
//!
//! on the radial basis functions the engine builds for `l` (`basis_recipe::species`): the scalar
//! `D` and `q` on the scalar projectors, the spin-orbit term on the union of the branches' projectors,
//! the metric spin free (`models::spin_orbit`, `periodic/setup/parse.rs`). This module diagonalizes
//! that one-center problem in the basis, in the dataset's screened local potential and without a
//! self-consistency loop (the frozen reference density; the engine's SCF atom agrees with it to a
//! few percent), for `j = l - 1/2` and `j = l + 1/2`, and returns the splitting next to the
//! all-electron Dirac one of the record. Everything is Rust: the generator uses it to calibrate and to
//! gate, and the diagnostics report it.
//!
//! The restriction to the engine's basis is the point: the inner shape of a basis function is that
//! of the scalar partial wave, so the `j` projectors see a wave they were not built for, and a heavy
//! `p` shell loses 10-15 percent of its splitting (bismuth 6p: 1.60 eV from the engine against the
//! record's 1.836). A complete radial basis would not show it (1.79 eV for the same record).
use super::{
    basis_recipe::{species, Recipe as BasisRecipe},
    diagnose::view::Smooth,
    invalid,
    record::PawDataset,
};
use crate::{Context, Result};
use nalgebra::{DMatrix, SymmetricEigen};

/// A splitting within this fraction of the Dirac one, or within [`ABSOLUTE`], passes the native
/// gate G13. The model has a frozen reference density: the engine's SCF atom (the jj occupation of
/// the open shell) lowers the heavy `p` splittings a further 4-8 percent (Bi 1.679 eV frozen against
/// 1.607 SCF, Pb 1.258 against 1.168, Tl 0.928 against 0.867), so this gate is a 10 percent screen of
/// the operator and the basis. The 2 percent gate G13 proper is on the engine's own SCF atom
/// (`PAW_REGENERATION_PLAN.md`, `soatom/so_engine.py`), which is what the calibration table is
/// measured on.
pub const RELATIVE: f64 = 0.10;
/// 10 meV in Hartree (the absolute floor of both gates).
pub const ABSOLUTE: f64 = 0.010 / 27.211_386_245_988;
/// Overlap eigenvalues below this (relative to the largest) are dropped as linear dependence.
const DEPENDENCE: f64 = 1e-9;

/// One channel's splitting in the engine's basis and in the all-electron Dirac atom.
#[derive(Debug, Clone, Copy)]
pub struct Splitting {
    pub l: usize,
    /// `E(j = l + 1/2) - E(j = l - 1/2)` of the engine's operator in its basis (Hartree).
    pub engine: f64,
    /// The same difference of the Dirac branch reference energies of the record (Hartree).
    pub dirac: f64,
    /// Which bound reference of the channel the level is (0 = the lowest).
    pub state: usize,
}

impl Splitting {
    /// Whether the engine splitting is within [`RELATIVE`] or [`ABSOLUTE`] of the Dirac one.
    pub fn passes(&self) -> bool {
        (self.engine - self.dirac).abs() <= (RELATIVE * self.dirac.abs()).max(ABSOLUTE)
    }
}

fn trapezoid(h: f64, values: impl Iterator<Item = f64>) -> f64 {
    values.sum::<f64>() * h
}

/// Radial integrals on the uniform table `r` of the basis functions `chi` (`R(r)`), for the
/// angular momentum `l`: overlap, kinetic energy plus the local potential, and the projections on
/// each projector table (`p(r)`, the factor `r^2` included in the measure).
struct Basis {
    overlap: DMatrix<f64>,
    hamiltonian: DMatrix<f64>,
    projections: Vec<DMatrix<f64>>,
}

fn integrals(
    r: &[f64],
    l: usize,
    v: &[f64],
    chi: &[&[f64]],
    projector_sets: &[&[&[f64]]],
) -> Basis {
    let n = r.len();
    let h = r[1] - r[0];
    let centrifugal = (l * (l + 1)) as f64;
    let slope = |f: &[f64], i: usize| -> f64 {
        let (a, b) = (i.saturating_sub(1), (i + 1).min(n - 1));
        (f[b] - f[a]) / (r[b] - r[a])
    };
    let m = chi.len();
    let mut overlap = DMatrix::zeros(m, m);
    let mut hamiltonian = DMatrix::zeros(m, m);
    for i in 0..m {
        for j in i..m {
            let s = trapezoid(h, (0..n).map(|k| chi[i][k] * chi[j][k] * r[k] * r[k]));
            let t = trapezoid(
                h,
                (0..n).map(|k| {
                    0.5 * (slope(chi[i], k) * slope(chi[j], k)
                        + centrifugal * chi[i][k] * chi[j][k] / (r[k] * r[k]))
                        * r[k]
                        * r[k]
                }),
            );
            let p = trapezoid(
                h,
                (0..n).map(|k| v[k] * chi[i][k] * chi[j][k] * r[k] * r[k]),
            );
            overlap[(i, j)] = s;
            overlap[(j, i)] = s;
            hamiltonian[(i, j)] = t + p;
            hamiltonian[(j, i)] = t + p;
        }
    }
    let projections = projector_sets
        .iter()
        .map(|set| {
            DMatrix::from_fn(m, set.len(), |i, a| {
                trapezoid(h, (0..n).map(|k| set[a][k] * chi[i][k] * r[k] * r[k]))
            })
        })
        .collect();
    Basis {
        overlap,
        hamiltonian,
        projections,
    }
}

/// The `state`-th eigenvalue of `H c = E S c`, linear dependence removed.
fn level(h: &DMatrix<f64>, s: &DMatrix<f64>, state: usize) -> Result<f64> {
    let eigen = SymmetricEigen::new(s.clone());
    let top = eigen.eigenvalues.iter().cloned().fold(0., f64::max);
    let keep: Vec<usize> = (0..eigen.eigenvalues.len())
        .filter(|&i| eigen.eigenvalues[i] > DEPENDENCE * top)
        .collect();
    let mut x = DMatrix::zeros(s.nrows(), keep.len());
    for (c, &i) in keep.iter().enumerate() {
        let scale = 1. / eigen.eigenvalues[i].sqrt();
        for r in 0..s.nrows() {
            x[(r, c)] = eigen.eigenvectors[(r, i)] * scale;
        }
    }
    let reduced = x.transpose() * h * &x;
    let mut values: Vec<f64> = SymmetricEigen::new(reduced)
        .eigenvalues
        .iter()
        .copied()
        .collect();
    values.sort_by(|a, b| a.total_cmp(b));
    values
        .get(state)
        .copied()
        .ok_or_else(|| invalid("the basis has fewer levels than the bound state asks for"))
}

/// The engine and the Dirac splitting of every channel `l >= 1` with a spin-orbit entry and a
/// bound occupied reference (its valence-most one), for the record as stored.
pub fn splittings(ctx: &Context, ds: &PawDataset) -> Result<Vec<Splitting>> {
    let smooth = Smooth::scalar(ctx, ds)?;
    let (shells, _) = species(ctx, ds, &BasisRecipe::new(2, 1, None))?;
    let r = ds.r.as_slice();
    let mut out = Vec::new();
    for (&l, so) in &ds.spin_orbit {
        let l = l as usize;
        let channel = smooth
            .channel(l)
            .ok_or_else(|| invalid(format!("the record has no scalar channel l={l}")))?;
        let Some(state) = channel.bound_occupied_energies().len().checked_sub(1) else {
            continue;
        };
        let chi: Vec<&[f64]> = shells
            .iter()
            .filter(|s| s.l == l && !s.diffuse && s.r.len() == r.len())
            .map(|s| s.values.as_slice())
            .collect();
        if chi.is_empty() {
            continue;
        }
        let scalar: Vec<&[f64]> = (0..channel.rank)
            .map(|a| &channel.projectors.as_slice()[a * r.len()..(a + 1) * r.len()])
            .collect();
        let branch: Vec<&[f64]> = so.projectors.iter().map(|t| t.as_slice()).collect();
        let basis = integrals(
            r,
            l,
            ds.v_local_screened.as_slice(),
            &chi,
            &[scalar.as_slice(), branch.as_slice()],
        );
        let (ps, pso) = (&basis.projections[0], &basis.projections[1]);
        let (rank, nso) = (channel.rank, so.projectors.len());
        let d = DMatrix::from_row_slice(rank, rank, &channel.coupling);
        let q = DMatrix::from_row_slice(rank, rank, &channel.overlap);
        let ls = DMatrix::from_fn(nso, nso, |i, j| {
            so.coupling.data()[i * nso + j]
                + so.overlap_coupling.data()[i * nso + j] * ds.hartree_screening
        });
        let metric = &basis.overlap + ps * q * ps.transpose();
        let scalar_part = &basis.hamiltonian + ps * d * ps.transpose();
        let mut levels = [0.; 2];
        for (slot, factor) in [-(l as f64 + 1.) / 2., l as f64 / 2.]
            .into_iter()
            .enumerate()
        {
            let h = &scalar_part + pso * (&ls * factor) * pso.transpose();
            levels[slot] = level(&h, &metric, state)?;
        }
        let energies = so.reference_energies.data();
        let count = energies.len() / 2;
        if so.reference_energies.shape() != [2, count] || state >= count {
            return Err(invalid(format!(
                "the spin-orbit entry l={l} has no branch reference {state}"
            )));
        }
        out.push(Splitting {
            l,
            engine: levels[1] - levels[0],
            dirac: energies[count + state] - energies[state],
            state,
        });
    }
    Ok(out)
}
