//! The search over constructions: build the dataset, look at its spectrum and its scattering,
//! and when a ghost state, a wrong scattering or a missing state shows up, try the remedies of
//! [`attempts`] in order until one is clean.
//!
//! A construction is accepted when it has no ghost, every constructed channel scatters like the
//! all-electron atom (`PHASE_TOLERANCE` near its reference, `RESONANCE_TOLERANCE` farther out),
//! every angular momentum without projectors does (`UNPROJECTED_TOLERANCE`), the s channel
//! spans an intruding hydrogen 1s (the miss below 1), and every occupied bound reference is
//! bound by the smooth Hamiltonian (`MISSING_TOLERANCE`). The search stops at the first attempt
//! that is, so a dataset that is clean to begin with is returned exactly as built, and one
//! reference atom, solved once, serves every attempt.
//!
//! The four [`Mode`]s say what happens to a construction that is not clean:
//!
//! - `Keep` returns the first construction unexamined (to study a ghost).
//! - `Refuse` raises, naming the ghost, the phase or the missing state.
//! - `Repair` searches and raises when no attempt is clean.
//! - `Flag` searches and, when none is, returns the least defective attempt
//!   ([`ranking`](super::ranking)) with its defects recorded in the record.
//!
//! A caller that fixed the cutoffs or the local potential has made the choice a remedy would
//! change, so a defect there is refused rather than overridden ([`Pinned`]); a construction
//! whose only defect is an incomplete one is then returned flagged instead. A check that cannot
//! be evaluated (a level or phase that is not finite) fails the construction like a defect, but
//! such a construction is never returned, flagged or not.
//!
//! The checks read the `j`-averaged scalar channels, which is what the generator this replaces
//! examined. The `j` branches of a fully relativistic dataset are not searched over; their level
//! errors are `diagnose::kappa::level_errors` of the selected record.
mod attempts;
mod message;

pub use attempts::{plan, radii_of, Attempt};
pub use message::{general, Error, Tried, Why};

use super::ranking::{Badness, Keeper};
use super::{
    energies::{Variational, XcSource},
    generate_with_atom,
    recipe::{Cutoffs, Recipe},
    reference::solve_reference,
    Generated,
};
use crate::{
    models::atom::Atom,
    paw::diagnose::{
        self,
        ghosts::{ghost_errors, missing_states},
        intruding::intruding_s_miss,
        reference::{check_reference, AeLogs, AllElectron},
        scattering::{scattering_errors, unprojected_errors},
        view::Smooth,
        Diagnosis,
    },
    Context,
};

/// What to do about a construction that is not clean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Search, and raise when no attempt is clean.
    Repair,
    /// Raise at the first construction that is not clean.
    Refuse,
    /// Return the first construction unexamined.
    Keep,
    /// Search; when no attempt is clean, return the least defective with its defects recorded.
    Flag,
}

/// Which choices the caller made, and the search therefore must not change.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pinned {
    /// The augmentation radii (anything but the recipe's own rule).
    pub r_cut: bool,
    /// The radius of the local potential.
    pub r_cut_local: bool,
    /// The raise of the local potential.
    pub local_shift: bool,
}

impl Pinned {
    /// The choices a recipe carries explicitly: the radii that are not the rule's, a local
    /// radius, and the radii and raise a UPAW-LCAO table lays under the caller's options (the
    /// generator treats those as the caller's).
    pub fn of(recipe: &Recipe) -> Self {
        let upaw = (recipe.family == crate::paw::record::Family::UpawLcao)
            .then(|| super::tables_upaw::recipe(&recipe.symbol, &recipe.functional));
        Self {
            r_cut: !matches!(recipe.cutoffs, Cutoffs::Rule),
            r_cut_local: recipe.local_radius.is_some(),
            local_shift: upaw.is_some_and(|u| u.local_shift.is_some()),
        }
    }

    /// The names of the pinned choices, as a refusal lists them.
    pub fn names(&self) -> Vec<&'static str> {
        [
            (self.r_cut, "r_cut"),
            (self.r_cut_local, "r_cut_local"),
            (self.local_shift, "local_shift"),
        ]
        .into_iter()
        .filter_map(|(set, name)| set.then_some(name))
        .collect()
    }

    pub fn any(&self) -> bool {
        self.r_cut || self.r_cut_local || self.local_shift
    }
}

/// How the search runs.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub mode: Mode,
    pub pinned: Pinned,
}

impl Options {
    /// `mode`, with the choices `recipe` carries pinned.
    pub fn new(recipe: &Recipe, mode: Mode) -> Self {
        Self {
            mode,
            pinned: Pinned::of(recipe),
        }
    }
}

/// A construction with the diagnosis that judged it.
#[derive(Debug)]
pub struct Candidate {
    pub generated: Generated,
    pub diagnosis: Diagnosis,
    /// Position in the attempt list; 0 is the first construction.
    pub index: usize,
    pub label: String,
}

/// The label of the first construction.
pub const FIRST: &str = "first construction";

/// The result of a search.
#[derive(Debug)]
pub struct Searched {
    /// The accepted construction, or the least defective one in `Flag` mode (then its record
    /// carries the defects).
    pub generated: Generated,
    /// Position of the selected attempt; 0 is the first construction.
    pub index: usize,
    pub label: String,
    /// The diagnosis of the selected construction; `None` for `Keep`.
    pub diagnosis: Option<Diagnosis>,
    /// Every attempt that was not accepted, in the order tried.
    pub tried: Vec<Tried>,
    /// Whether the selected construction still has defects (only `Flag` returns one).
    pub flagged: bool,
}

/// Judge one construction against the all-electron `atom`: the ghosts first, and only when there
/// is none the scattering, the unprojected channels, the s completeness and the missing states
/// (a ghosted construction is not examined further, and a phase that cannot be evaluated on it
/// must not hide the ghost). `cache` holds the all-electron logarithmic derivatives of `atom`.
pub fn examine(
    ctx: &Context,
    generated: &Generated,
    atom: &Atom,
    cache: &mut AeLogs,
) -> diagnose::Outcome<Diagnosis> {
    let dataset = &generated.dataset;
    check_reference(dataset, atom)?;
    let smooth = Smooth::scalar(ctx, dataset)?;
    let ae = AllElectron::new(atom, smooth.atomic_number, &smooth.frozen, &smooth.semicore);
    let ghosts = ghost_errors(ctx, &smooth, Some(&ae))?;
    if !ghosts.is_empty() {
        return Ok(Diagnosis {
            ghosts,
            ..Diagnosis::default()
        });
    }
    let phases = scattering_errors(ctx, &smooth, &ae, cache)?;
    let s_miss = intruding_s_miss(&smooth);
    if let Some(miss) = s_miss.filter(|m| !m.is_finite()) {
        return Err(diagnose::Error::Unevaluated(format!(
            "{}: the intruding-1s miss is {miss}",
            smooth.symbol
        )));
    }
    Ok(Diagnosis {
        ghosts,
        phases,
        unprojected: unprojected_errors(ctx, &smooth, &ae, cache)?,
        s_miss,
        missing: missing_states(ctx, &smooth)?,
    })
}

/// The first construction of `recipe` and the search over its remedies, with the variational
/// radial exchange-correlation scheme.
pub fn search(ctx: &Context, recipe: &Recipe, options: &Options) -> Result<Searched, Error> {
    search_with(ctx, recipe, options, &Variational)
}

#[doc(hidden)]
/// [`search`] with the radial exchange-correlation scheme as a parameter (the seam of
/// [`super::generate_with`], for the parity tests).
pub fn search_with(
    ctx: &Context,
    recipe: &Recipe,
    options: &Options,
    source: &dyn XcSource,
) -> Result<Searched, Error> {
    let atom = solve_reference(ctx, recipe).map_err(Error::Failed)?;
    search_with_atom(ctx, recipe, options, &atom, source)
}

/// Whether an error is a property of the attempt (its smooth waves are infeasible) and not of
/// the run.
fn is_attempt_failure(error: &crate::Error) -> bool {
    matches!(error, crate::Error::Invalid(_) | crate::Error::Kernel(_))
}

#[doc(hidden)]
/// [`search_with`] on a reference atom already solved with the recipe's functional and
/// treatment.
pub fn search_with_atom(
    ctx: &Context,
    recipe: &Recipe,
    options: &Options,
    atom: &Atom,
    source: &dyn XcSource,
) -> Result<Searched, Error> {
    let first = generate_with_atom(ctx, recipe, atom, source).map_err(Error::Failed)?;
    if options.mode == Mode::Keep {
        return Ok(Searched {
            generated: first,
            index: 0,
            label: FIRST.into(),
            diagnosis: None,
            tried: Vec::new(),
            flagged: false,
        });
    }
    let attempts = plan(recipe, &radii_of(&first.dataset));
    let mut cache = AeLogs::new();
    let mut tried = Vec::new();
    let mut keeper = Keeper::<Candidate>::new();
    // Every construction that failed, with its recipe: an attempt whose recipe equals an earlier
    // one cannot differ from it (the generator is deterministic and the atom is shared), so it
    // is reported with the earlier verdict instead of being built again.
    let mut seen: Vec<(Recipe, Why)> = Vec::new();

    let problem = match examine(ctx, &first, atom, &mut cache) {
        Ok(diagnosis) => match Why::of(&diagnosis) {
            None => {
                return Ok(Searched {
                    generated: first,
                    index: 0,
                    label: FIRST.into(),
                    diagnosis: Some(diagnosis),
                    tried,
                    flagged: false,
                })
            }
            Some(why) => {
                let acceptance_only =
                    diagnosis.ghosts.is_empty() && diagnosis.wrong_phases().is_empty();
                if options.pinned.any() && acceptance_only && options.mode != Mode::Refuse {
                    // Only the family's acceptance failed and the caller chose what a repair
                    // would change: keep the construction and record what it misses.
                    return Ok(flagged(first, diagnosis, 0, FIRST.into(), tried));
                }
                if options.mode == Mode::Flag {
                    keeper.offer(
                        Badness::of(&diagnosis),
                        Candidate {
                            generated: first,
                            diagnosis,
                            index: 0,
                            label: FIRST.into(),
                        },
                    );
                }
                why
            }
        },
        Err(diagnose::Error::Unevaluated(text)) => Why::Unevaluated(text),
        Err(diagnose::Error::Failed(error)) => return Err(Error::Failed(error)),
    };
    if options.mode == Mode::Refuse || options.pinned.any() {
        let why = if options.pinned.any() {
            format!(
                "with {} fixed by the caller",
                options.pinned.names().join(", ")
            )
        } else {
            "and ghosts='refuse'".to_string()
        };
        return Err(Error::Refused {
            symbol: recipe.symbol.clone(),
            problem: problem.problem(),
            why,
        });
    }
    seen.push((recipe.clone(), problem.clone()));

    for (offset, attempt) in attempts.into_iter().enumerate() {
        let index = offset + 1;
        if let Some((_, why)) = seen.iter().find(|(earlier, _)| *earlier == attempt.recipe) {
            tried.push(Tried {
                label: attempt.label,
                why: why.clone(),
            });
            continue;
        }
        let why = match generate_with_atom(ctx, &attempt.recipe, atom, source) {
            Err(error) if is_attempt_failure(&error) => Why::NotBuilt(error.to_string()),
            Err(error) => return Err(Error::Failed(error)),
            Ok(built) => match examine(ctx, &built, atom, &mut cache) {
                Err(diagnose::Error::Unevaluated(text)) => Why::Unevaluated(text),
                Err(diagnose::Error::Failed(error)) => return Err(Error::Failed(error)),
                Ok(diagnosis) => match Why::of(&diagnosis) {
                    None => {
                        return Ok(Searched {
                            generated: built,
                            index,
                            label: attempt.label,
                            diagnosis: Some(diagnosis),
                            tried,
                            flagged: false,
                        })
                    }
                    Some(why) => {
                        if options.mode == Mode::Flag {
                            keeper.offer(
                                Badness::of(&diagnosis),
                                Candidate {
                                    generated: built,
                                    diagnosis,
                                    index,
                                    label: attempt.label.clone(),
                                },
                            );
                        }
                        why
                    }
                },
            },
        };
        tried.push(Tried {
            label: attempt.label,
            why: why.clone(),
        });
        seen.push((attempt.recipe, why));
    }
    if let Some(best) = keeper.into_best() {
        return Ok(flagged(
            best.generated,
            best.diagnosis,
            best.index,
            best.label,
            tried,
        ));
    }
    Err(Error::Exhausted {
        symbol: recipe.symbol.clone(),
        problem: problem.problem(),
        tried,
    })
}

fn flagged(
    mut generated: Generated,
    diagnosis: Diagnosis,
    index: usize,
    label: String,
    tried: Vec<Tried>,
) -> Searched {
    generated.dataset.defects = diagnosis.defects();
    Searched {
        generated,
        index,
        label,
        diagnosis: Some(diagnosis),
        tried,
        flagged: true,
    }
}
