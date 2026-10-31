//! Finite-size corrections for supercell calculations (Hartree atomic units).
//!
//! `kzk` is the Kwee-Zhang-Krakauer size-dependent LDA: a functional of the density
//! alone. It has no public entry point yet; the wiring package (FS-5) adds it.
//! The Wigner-Seitz kernel, MPC and CCMH belong to later packages.
pub mod kzk;

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
