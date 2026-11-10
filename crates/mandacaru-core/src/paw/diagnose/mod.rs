//! Diagnostics of a PAW-LCAO / UPAW-LCAO dataset: does its smooth Hamiltonian have the atom's
//! spectrum and scatter like the atom?
//!
//! - [`spectrum`]: the bound levels of each channel's generalized radial problem.
//! - [`scattering`]: the smooth logarithmic derivative against the all-electron one, and the
//!   phase errors over the energy windows around the references.
//! - [`ghosts`]: spurious bound states (below or among the physical ones, excited, and in the
//!   channels without projectors) and the opposite failure, a physical reference the
//!   Hamiltonian does not bind.
//! - [`intruding`]: the completeness of the s channel against a neighbor's 1s.
//! - [`kappa`]: the `j` branches of a fully relativistic dataset as channels of their own.
//! - [`reference`]: the all-electron side (atom, equation, core shells, cached logarithmic
//!   derivatives). [`view`]: the smooth side, the one thing the checks read.
//! - [`quality`]: verification gates the search does not act on (a bound level in a channel
//!   without projectors, hardness, sphere against bond, sr/fr partition, defects the record does
//!   not store).
//!
//! [`Diagnosis::run`] evaluates every check; [`Diagnosis::defects`] applies the recording rule
//! of the generator (`ghosts="flag"`): a dataset with a ghost records the ghosts alone, one
//! without records the channels that scatter outside their tolerance, the unprojected channels
//! outside theirs, a failed s completeness and the missing states. That list is the record's
//! [`Defects`]. The `j` branches of a fully relativistic dataset are examined too
//! ([`Diagnosis::with_branches`]): a branch that holds a ghost is recorded like a ghost of a
//! scalar channel, under its own `"l:kappa"` key.
//!
//! A check that cannot be evaluated (a level or phase that is not finite) is
//! [`Error::Unevaluated`], never a pass.
pub mod ghosts;
pub mod intruding;
pub mod kappa;
pub mod quality;
pub mod reconstruct;
pub mod reference;
pub mod scattering;
pub mod spectrum;
pub mod view;

use self::{
    ghosts::{branch_ghost_errors, ghost_errors, missing_states},
    intruding::{failed_acceptance, intruding_s_miss},
    reference::{AeLogs, AllElectron},
    scattering::{
        scattering_errors, unprojected_errors, PHASE_TOLERANCE, RESONANCE_TOLERANCE,
        UNPROJECTED_TOLERANCE,
    },
    view::{Key, Smooth},
};
use crate::{paw::record::Defects, Context};
use std::collections::BTreeMap;

/// Why a diagnostic did not produce a number.
#[derive(Debug)]
pub enum Error {
    /// A level, phase or potential is not finite, or the check has nothing to evaluate on:
    /// a failure of the construction, never a pass.
    Unevaluated(String),
    /// Any other failure (a malformed dataset, a context budget).
    Failed(crate::Error),
}

pub type Outcome<T> = std::result::Result<T, Error>;

impl From<crate::Error> for Error {
    fn from(error: crate::Error) -> Self {
        Self::Failed(error)
    }
}

impl Error {
    /// The same failure as a crate error (an unevaluated check is an invalid dataset).
    pub fn into_core(self) -> crate::Error {
        match self {
            Self::Unevaluated(message) => crate::Error::Invalid(message),
            Self::Failed(error) => error,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unevaluated(message) => f.write_str(message),
            Self::Failed(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for Error {}

/// The grid node nearest to `radius`.
pub fn snap_to(r: &[f64], radius: f64) -> f64 {
    crate::paw::pseudize::snap(r, radius)
}

/// Every number the checks produce for one dataset.
#[derive(Debug, Clone, Default)]
pub struct Diagnosis {
    /// `{l: eps_0 - eps_ref}` of the channels holding a ghost.
    pub ghosts: BTreeMap<usize, f64>,
    /// `(near, far)` phase errors (rad) of every channel.
    pub phases: BTreeMap<Key, [f64; 2]>,
    /// Phase error (rad) of every angular momentum without projectors.
    pub unprojected: BTreeMap<usize, f64>,
    /// The intruding-1s miss; `None` for a dataset without an s channel.
    pub s_miss: Option<f64>,
    /// `{l: nearest level - reference}` of the channels that do not bind a physical reference.
    pub missing: BTreeMap<usize, f64>,
    /// `{(l, kappa): eps_0 - eps_ref}` of the `j` branches of a fully relativistic dataset that
    /// hold a ghost ([`ghosts::branch_ghost_errors`]); empty for a dataset without spin-orbit
    /// entries and for a diagnosis that did not look ([`Diagnosis::with_branches`]).
    pub branch_ghosts: BTreeMap<Key, f64>,
}

impl Diagnosis {
    /// Evaluate every check of `smooth` against the all-electron atom `ae`; `cache` holds the
    /// all-electron logarithmic derivatives already computed for this atom.
    pub fn run(
        ctx: &Context,
        smooth: &Smooth,
        ae: &AllElectron<'_>,
        cache: &mut AeLogs,
    ) -> Outcome<Self> {
        let s_miss = intruding_s_miss(smooth);
        if let Some(miss) = s_miss {
            if !miss.is_finite() {
                return Err(Error::Unevaluated(format!(
                    "{}: the intruding-1s miss is {miss}",
                    smooth.symbol
                )));
            }
        }
        Ok(Self {
            ghosts: ghost_errors(ctx, smooth, Some(ae))?,
            phases: scattering_errors(ctx, smooth, ae, cache)?,
            unprojected: unprojected_errors(ctx, smooth, ae, cache)?,
            s_miss,
            missing: missing_states(ctx, smooth)?,
            branch_ghosts: BTreeMap::new(),
        })
    }

    /// Add the ghosts of the `j` branches of `dataset` (a no-op without spin-orbit entries): the
    /// branches are channels of their own ([`kappa::branches`]) and the same extra-state test
    /// applies to each against its own Dirac reference.
    pub fn with_branches(
        mut self,
        ctx: &Context,
        dataset: &crate::paw::record::PawDataset,
    ) -> Outcome<Self> {
        if dataset.spin_orbit.is_empty() {
            return Ok(self);
        }
        let branches = kappa::branches(ctx, dataset)?;
        self.branch_ghosts = branch_ghost_errors(ctx, &branches)?;
        Ok(self)
    }

    /// The channels of the `j`-averaged set whose phase is outside `PHASE_TOLERANCE` near the
    /// reference or `RESONANCE_TOLERANCE` farther out. A phase that is not a number is outside
    /// (a NaN fails every gate).
    pub fn wrong_phases(&self) -> BTreeMap<usize, [f64; 2]> {
        self.phases
            .iter()
            .filter(|(k, [near, far])| {
                k.kappa.is_none() && (!(*near <= PHASE_TOLERANCE) || !(*far <= RESONANCE_TOLERANCE))
            })
            .map(|(k, v)| (k.l, *v))
            .collect()
    }

    /// The defects in the record's format. A dataset with a ghost records the ghosts and
    /// nothing else (the generator never examines the rest of a ghosted construction); one
    /// without records the phases outside tolerance, the unprojected channels above
    /// `UNPROJECTED_TOLERANCE` and an s completeness miss at or above its tolerance.
    pub fn defects(&self) -> Defects {
        if self.has_ghost() {
            return Defects {
                ghosts: self.ghosts.iter().map(|(&l, &e)| (l as u32, e)).collect(),
                branch_ghosts: self
                    .branch_ghosts
                    .iter()
                    .map(|(k, &e)| (Defects::branch_key(k.l, k.kappa.unwrap_or(0)), e))
                    .collect(),
                ..Defects::default()
            };
        }
        Defects {
            ghosts: BTreeMap::new(),
            phases: self
                .wrong_phases()
                .into_iter()
                .map(|(l, v)| (l as u32, v))
                .collect(),
            residuals: BTreeMap::new(),
            s_miss: failed_acceptance(self.s_miss),
            unprojected: self
                .unprojected
                .iter()
                .filter(|(_, &e)| !(e <= UNPROJECTED_TOLERANCE))
                .map(|(&l, &e)| (l as u32, e))
                .collect(),
            missing: self.missing.iter().map(|(&l, &d)| (l as u32, d)).collect(),
            branch_ghosts: BTreeMap::new(),
        }
    }

    /// Whether a scalar channel or a `j` branch holds a ghost.
    pub fn has_ghost(&self) -> bool {
        !self.ghosts.is_empty() || !self.branch_ghosts.is_empty()
    }

    /// Whether the dataset has no defect at all: no ghost, no missing state, every channel
    /// inside its tolerances.
    pub fn is_clean(&self) -> bool {
        self.defects() == Defects::default()
    }
}

/// Diagnose a dataset record against the all-electron atom it was built on: solved again from
/// the record when `atom` is `None`, as a dataset read from a file needs. This is the entry the
/// bindings and the ghost search call; `cache` shares all-electron logarithmic derivatives across
/// the attempts of a search that use one atom.
pub fn diagnose_record(
    ctx: &Context,
    dataset: &crate::paw::record::PawDataset,
    atom: Option<&crate::models::atom::Atom>,
    cache: &mut AeLogs,
) -> Outcome<Diagnosis> {
    let solved;
    let atom = match atom {
        Some(atom) => atom,
        None => {
            solved = reference::reference_atom(ctx, dataset)?;
            &solved
        }
    };
    reference::check_reference(dataset, atom)?;
    let smooth = Smooth::scalar(ctx, dataset)?;
    let ae = AllElectron::new(atom, smooth.atomic_number, &smooth.frozen, &smooth.semicore);
    Diagnosis::run(ctx, &smooth, &ae, cache)?.with_branches(ctx, dataset)
}

/// [`diagnose_record`] and, on the same atom, the bound all-electron levels of the channels
/// without projectors that the dataset misses ([`quality::unprojected_missing_states`]), which
/// the recording rule does not act on.
pub fn diagnose_record_with_unprojected(
    ctx: &Context,
    dataset: &crate::paw::record::PawDataset,
    cache: &mut AeLogs,
) -> Outcome<(Diagnosis, BTreeMap<usize, f64>)> {
    let atom = reference::reference_atom(ctx, dataset)?;
    let diagnosis = diagnose_record(ctx, dataset, Some(&atom), cache)?;
    let smooth = Smooth::scalar(ctx, dataset)?;
    let ae = AllElectron::new(
        &atom,
        smooth.atomic_number,
        &smooth.frozen,
        &smooth.semicore,
    );
    let missing = quality::unprojected_missing_states(&smooth, &ae)?;
    Ok((diagnosis, missing))
}
