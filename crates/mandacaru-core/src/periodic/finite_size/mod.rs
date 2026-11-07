//! Finite-size corrections for supercell calculations (Hartree atomic units).
//!
//! Includes:
//! - `kzk`: The Kwee-Zhang-Krakauer size-dependent LDA (density functional).
//! - `mpc`: The Model Periodic Coulomb scheme with Madelung double-counting correction.
//! - `ccmh`: The Chiesa-Ceperley-Martin-Holzmann structure factor extrapolation.

pub mod ccmh;
pub mod kzk;
pub mod mpc;

pub use ccmh::{
    ccmh_extrapolate, structure_factor_from_rdms, Ccmh, CcmhShell, StructureFactor,
    DEFAULT_SHELLS, QUADRATIC_REGIME_SPREAD,
};
pub use kzk::{kzk, kzk_flexible, Fit, Kzk, Vacuum, DENSITY_FLOOR, HARTREE_PER_RYDBERG};
pub use mpc::{mpc_from_hole_energies, mpc_hole_contraction, verify_hole_trace, Mpc};

use crate::{Error, Result};

/// The two scopes of a supercell result, kept apart: `volume` and the totals are per
/// supercell, `n_cells` is the number of primitive cells it holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scope {
    pub n_cells: usize,
    pub volume: f64,
}

impl Scope {
    pub fn new(n_cells: usize, volume: f64) -> Result<Self> {
        if n_cells == 0 {
            return Err(Error::Invalid(
                "a supercell holds at least one primitive cell".into(),
            ));
        }
        if !volume.is_finite() || volume <= 0.0 {
            return Err(Error::Invalid(format!(
                "the supercell volume must be finite and positive, got {volume}"
            )));
        }
        Ok(Self { n_cells, volume })
    }

    /// A per-supercell total expressed per primitive cell.
    pub fn per_cell(&self, supercell_total: f64) -> f64 {
        supercell_total / self.n_cells as f64
    }
}
