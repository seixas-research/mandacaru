//! The typed PAW-LCAO / UPAW-LCAO dataset record and its file codec.
//!
//! A dataset file is either Parquet (the shipped library: every radial table is
//! a `double` column, everything else is the JSON text stored under the key
//! `mandacaru.pseudopotential`) or plain JSON (`<symbol>.json`, the same
//! payload with the tables inline under `radial_tables`). [`PawDataset`] is the
//! one typed shape of both. All shaping lives here, in `mandacaru-core`: the
//! bindings only marshal arrays.
//!
//! Layout of this module:
//!
//! - `mod.rs` (this file): the record types and the constants of the format.
//! - `codec.rs`: payload (JSON tree plus tables) to record.
//! - `parts.rs`: the small sub-records (nlcc, defects, subshells, spin-orbit).
//! - `json.rs`: typed access to the JSON tree, arrays reserved before allocation.
//! - `emit.rs`: record to payload text, the inverse of `codec`.
//! - `files.rs`: read a file under a `Context` budget, staged write with
//!   read-back.
//! - `digits.rs`: the ten-digit, stride-`n` storage rule of the tables.
//! - `values.rs`: the local potential and its slope from the table.
//!
//! Refusals, all of them `Error::Invalid("paw record: ...")`: a format other
//! than `mandacaru-pseudopotential`, a version above 2, a family other than
//! `paw-lcao` / `upaw-lcao`, a bare-matrix (first-order) spin-orbit entry,
//! spin-orbit projectors off the dataset's radial grid, a missing norm
//! correction (there is no recovery), an unknown or missing key. The only
//! defaults are the two the shipped library needs; the record lists them in
//! [`PawDataset::defaults_applied`].

mod codec;
pub mod digits;
mod emit;
mod files;
mod json;
mod parts;
pub mod values;

use std::collections::BTreeMap;

use crate::paw::invalid;
use crate::{Buffer, Reservation, Result};

pub use files::{tree_cost, WriteSpec, TREE_BYTES_PER_MEMBER, TREE_BYTES_PER_VALUE};
pub use values::LocalPotential;

/// `format` of every dataset file.
pub const FORMAT_TAG: &str = "mandacaru-pseudopotential";
/// Newest `version` this build reads and the one it writes.
pub const FORMAT_VERSION: u64 = 2;
/// Key of the Parquet metadata that holds the JSON payload.
pub const METADATA_KEY: &str = "mandacaru.pseudopotential";

/// The two pseudopotential families that share this layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    PawLcao,
    UpawLcao,
}

impl Family {
    pub fn name(self) -> &'static str {
        match self {
            Self::PawLcao => "paw-lcao",
            Self::UpawLcao => "upaw-lcao",
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "paw-lcao" => Ok(Self::PawLcao),
            "upaw-lcao" => Ok(Self::UpawLcao),
            other => Err(invalid(format!(
                "paw record: family {other:?} is not readable; the families are paw-lcao and upaw-lcao"
            ))),
        }
    }
}

/// A small row-major array: shape plus data, the data charged to the `Context`.
#[derive(Debug)]
pub struct Array {
    pub(crate) shape: Vec<usize>,
    pub(crate) data: Buffer<f64>,
}

impl Array {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The values in row-major order.
    pub fn data(&self) -> &[f64] {
        self.data.as_slice()
    }

    pub fn len(&self) -> usize {
        self.data.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The nonlinear core correction record. `r_nlcc` is `null` in the file when
/// the correction is not applied: that is the schema, not a Parquet null.
#[derive(Debug, Clone, PartialEq)]
pub struct Nlcc {
    pub applied: bool,
    pub r_nlcc: Option<f64>,
    pub source: Option<String>,
    pub core_electrons: Option<f64>,
    pub partial_core_electrons: Option<f64>,
    pub reason: Option<String>,
}

/// Generation defects, keyed by angular momentum.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Defects {
    pub ghosts: BTreeMap<u32, f64>,
    pub phases: BTreeMap<u32, [f64; 2]>,
    pub residuals: BTreeMap<u32, f64>,
    pub s_miss: Option<f64>,
    pub unprojected: BTreeMap<u32, f64>,
}

/// One angular-momentum channel. The radial tables are on the dataset's grid.
#[derive(Debug)]
pub struct Channel {
    pub n: u32,
    pub r_cut: f64,
    pub occupation: f64,
    pub occupations: Array,
    pub reference_energies: Array,
    pub wavevectors: Vec<Array>,
    pub wave_coefficients: Vec<Array>,
    pub coupling: Array,
    pub coupling_screened: Array,
    pub overlap_correction: Array,
    pub norm_correction: Array,
    pub kinetic_difference: Array,
    pub potential_difference: Array,
    pub vanderbilt: Array,
    pub duality_error: f64,
    pub overlap_minimum: f64,
    pub asymmetry: f64,
    pub consistency_error: f64,
    pub residual_kinetic: Array,
    pub q_cut: f64,
    pub ae_waves: Vec<Buffer<f64>>,
    pub pseudo_waves: Vec<Buffer<f64>>,
    pub projectors: Vec<Buffer<f64>>,
    pub raw_projectors: Vec<Buffer<f64>>,
}

/// One channel of the j-resolved spin-orbit construction. The projectors are
/// radial tables on the dataset's grid.
#[derive(Debug)]
pub struct SpinOrbit {
    pub projectors: Vec<Buffer<f64>>,
    pub average: Array,
    pub average_screened: Array,
    pub coupling: Array,
    pub coupling_screened: Array,
    pub overlap_average: Array,
    pub overlap_coupling: Array,
    pub reference_energies: Array,
    pub level_errors: Array,
}

/// A PAW dataset: scalars, the radial tables and the per-channel matrices.
///
/// Every field is public so a consumer reads what it needs without a copy;
/// construct one only through [`PawDataset::read`]. The memory of the tables
/// and arrays is charged to the `Context` that read it and released when the
/// record is dropped.
#[derive(Debug)]
pub struct PawDataset {
    pub family: Family,
    pub symbol: String,
    pub atomic_number: u32,
    pub valence_charge: f64,
    pub r_cut_local: f64,
    pub local_shift: f64,
    pub compensation_radius: f64,
    pub compensation_charge: f64,
    pub hartree_screening: f64,
    pub one_center_energy: f64,
    pub energies: BTreeMap<String, f64>,
    pub q_cut: f64,
    pub energy_offset: f64,
    pub norm_deficit: f64,
    pub xc: String,
    pub relativity: String,
    pub relativistic_exchange: bool,
    pub extra_l: u32,
    pub nlcc: Nlcc,
    pub frozen_subshells: Vec<Vec<i64>>,
    pub semicore_subshells: Vec<Vec<i64>>,
    pub defects: Defects,
    pub spin_orbit: BTreeMap<u32, SpinOrbit>,
    pub channels: BTreeMap<u32, Channel>,
    pub r: Buffer<f64>,
    pub v_local: Buffer<f64>,
    pub v_local_screened: Buffer<f64>,
    pub valence_density: Buffer<f64>,
    pub core_density: Buffer<f64>,
    pub smooth_core_density: Buffer<f64>,
    /// The keys the file did not carry and a default filled in (a file written
    /// before they existed): `"semicore_subshells"` and
    /// `"channels.<l>.occupations"`. Empty for a file with the full schema.
    pub defaults_applied: Vec<String>,
    pub(crate) _small: Reservation,
}

impl PawDataset {
    /// The generation grid starts one step `h` from the origin, so `h` is
    /// `r[0]`. It is never `points(Z) / 4`: `pbe-fr/Pb` was written at stride 2.
    pub fn step(&self) -> f64 {
        self.r.as_slice()[0]
    }

    /// Radial points of the tables.
    pub fn points(&self) -> usize {
        self.r.as_slice().len()
    }

    /// Every radial table with the name of its Parquet column, in file order:
    /// the six dataset tables, then per channel and projector index the
    /// all-electron wave, the pseudo wave, the projector and the raw projector.
    pub fn tables(&self) -> Vec<(String, &[f64])> {
        let mut out: Vec<(String, &[f64])> = vec![
            ("r".into(), self.r.as_slice()),
            ("v_local".into(), self.v_local.as_slice()),
            ("v_local_screened".into(), self.v_local_screened.as_slice()),
            ("valence_density".into(), self.valence_density.as_slice()),
            ("core_density".into(), self.core_density.as_slice()),
            (
                "smooth_core_density".into(),
                self.smooth_core_density.as_slice(),
            ),
        ];
        for (l, channel) in &self.channels {
            for i in 0..channel.pseudo_waves.len() {
                out.push((format!("ae_wave_l{l}_{i}"), channel.ae_waves[i].as_slice()));
                out.push((
                    format!("pseudo_wave_l{l}_{i}"),
                    channel.pseudo_waves[i].as_slice(),
                ));
                out.push((
                    format!("projector_l{l}_{i}"),
                    channel.projectors[i].as_slice(),
                ));
                out.push((
                    format!("raw_projector_l{l}_{i}"),
                    channel.raw_projectors[i].as_slice(),
                ));
            }
        }
        out
    }
}
