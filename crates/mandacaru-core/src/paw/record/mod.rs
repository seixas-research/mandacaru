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
//! than `mandacaru-pseudopotential`, a version above 3, a family other than
//! `paw-lcao` / `upaw-lcao`, a bare-matrix (first-order) spin-orbit entry,
//! spin-orbit projectors off the dataset's radial grid, a missing norm
//! correction (there is no recovery), a version-3 record whose core density holds charge but
//! lists no core state, an unknown or missing key. The only
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
use crate::{Buffer, Context, Reservation, Result};

pub use files::{
    reload_errors, reload_shift, tree_cost, Parts, WriteSpec, RELOAD_LEVEL_TOLERANCE, TREE_BYTES_PER_MEMBER,
    TREE_BYTES_PER_VALUE,
};
pub use values::LocalPotential;

/// `format` of every dataset file.
pub const FORMAT_TAG: &str = "mandacaru-pseudopotential";
/// Newest `version` this build reads, and the one it writes for a record that carries its
/// core states ([`PawDataset::core_states`]).
pub const FORMAT_VERSION: u64 = 3;
/// The version of a record without core states: what a version-2 file read back is written as.
pub const CORE_FREE_VERSION: u64 = 2;
/// The most core states a record may declare: every `(n, l)` with `n <= 7` and `l < n`.
pub const MAX_CORE_STATES: usize = 28;
/// The radial exchange-correlation schemes a record's core states may name as the one its
/// reference atom was solved with (`PawDataset::core_scheme`): the legacy `AtomSolver` of the
/// first library and the production `Variational` (`models::xc_scheme`).
pub const CORE_SCHEMES: [&str; 2] = ["atom-solver", "variational"];
/// The first bytes of what [`PawDataset::content_sha256`] hashes: a new content encoding gets a
/// new tag.
pub const CONTENT_TAG: &str = "mandacaru-paw-content/1\n";
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
///
/// `missing` and `branch_ghosts` are keys of the native generator (`ghosts="flag"` could not
/// say these before): a file written earlier lacks them, and they are written only when
/// non-empty, so every older record round-trips byte for byte.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Defects {
    pub ghosts: BTreeMap<u32, f64>,
    pub phases: BTreeMap<u32, [f64; 2]>,
    pub residuals: BTreeMap<u32, f64>,
    pub s_miss: Option<f64>,
    pub unprojected: BTreeMap<u32, f64>,
    /// `{l: nearest smooth level - reference}` (Hartree) of the channels that do not bind an
    /// occupied bound reference: a physical state the dataset lacks.
    pub missing: BTreeMap<u32, f64>,
    /// `{"l:kappa": eps_0 - eps_ref}` (Hartree) of the `j` branches of a fully relativistic
    /// dataset that hold a ghost state, e.g. `"1:-2"` for the `j = 3/2` p branch.
    pub branch_ghosts: BTreeMap<String, f64>,
}

impl Defects {
    /// The key of a `j` branch in [`Defects::branch_ghosts`].
    pub fn branch_key(l: usize, kappa: i32) -> String {
        format!("{l}:{kappa}")
    }
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

/// One occupied core subshell of the reference atom (format version 3).
///
/// `orbital` is `u = r R(r)` on the dataset's radial grid, the function the generator summed
/// into `core_density` (`sum occupation u^2 / (4 pi r^2)`, `paw::generate::core`); for a
/// Dirac atom it is the `(2j+1)`-weighted average of the two `j` levels, as that sum used. It is
/// the column `core_wave_n{n}_l{l}` ([`PawDataset::core_table`]).
#[derive(Debug)]
pub struct CoreState {
    pub n: u32,
    pub l: u32,
    /// Electrons in the subshell, `0 < occupation <= 2 (2l + 1)`.
    pub occupation: f64,
    /// The level's eigenvalue in the reference atom, Hartree.
    pub energy: f64,
    pub orbital: Buffer<f64>,
}

/// A PAW dataset: scalars, the radial tables and the per-channel matrices.
///
/// Every field is public so a consumer reads what it needs without a copy. A record comes from
/// [`PawDataset::read`] (a file) or from the generator (`paw::generate`), which fills the same
/// fields; neither is built field by field elsewhere. The memory of the tables and arrays is
/// charged to the `Context` that read or generated it and released when the record is dropped.
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
    /// The occupied core subshells of the reference atom, in the order the core density is
    /// summed (format version 3). `None` for a version-2 file, which never had them: it is not a
    /// default and is not listed in `defaults_applied`.
    pub core_states: Option<Vec<CoreState>>,
    /// The radial exchange-correlation scheme the core states were solved with, one of
    /// [`CORE_SCHEMES`], when the record says (an upgraded format-2 file: the scheme identified
    /// by its own stored core density, owner Q22). Bookkeeping like `core_states`: not part of the
    /// content digest.
    pub core_scheme: Option<String>,
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

    /// The column name of the core state `(n, l)`.
    pub fn core_table(n: u32, l: u32) -> String {
        format!("core_wave_n{n}_l{l}")
    }

    /// SHA-256 (lowercase hex) of the record's content: a tag, then the payload text without
    /// `version` and `core_states` (`emit::content_text`, stride 1, unrounded: the record as it
    /// is held), then every radial table except the core states, each as its name, its length
    /// and its values' little-endian bytes. Two records with the same physics content have the
    /// same digest whatever file, codec or format version they came from: a version-2 dataset
    /// and its version-3 upgrade, which only adds the core states, share it.
    pub fn content_sha256(&self, ctx: &Context) -> Result<String> {
        let (text, _memory) = emit::content_text(ctx, self)?;
        let mut hasher = crate::paw::digest::Hasher::new();
        hasher.update(CONTENT_TAG.as_bytes());
        hasher.update(&(text.len() as u64).to_le_bytes());
        hasher.update(text.as_bytes());
        let tables = self.tables();
        let core = self.core_states.as_ref().map_or(0, Vec::len);
        for (name, values) in &tables[..tables.len() - core] {
            hasher.update(&(name.len() as u64).to_le_bytes());
            hasher.update(name.as_bytes());
            hasher.update(&(values.len() as u64).to_le_bytes());
            for value in values.iter() {
                hasher.update(&value.to_le_bytes());
            }
        }
        Ok(hasher.finish())
    }

    /// The format version the record is written as: 3 when it carries its core states, 2 when
    /// it does not (a version-2 file read back is written unchanged).
    pub fn format_version(&self) -> u64 {
        if self.core_states.is_some() {
            FORMAT_VERSION
        } else {
            CORE_FREE_VERSION
        }
    }

    /// Every radial table with the name of its Parquet column, in file order:
    /// the six dataset tables, then per channel and projector index the
    /// all-electron wave, the pseudo wave, the projector and the raw projector,
    /// then the core states (format version 3) in their order.
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
        for state in self.core_states.iter().flatten() {
            out.push((Self::core_table(state.n, state.l), state.orbital.as_slice()));
        }
        out
    }
}
