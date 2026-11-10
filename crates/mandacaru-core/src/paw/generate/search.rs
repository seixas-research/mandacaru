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
//! examined, and, for a fully relativistic dataset, the ghosts of its `j` branches too (a
//! branch with an occupied bound reference that holds an extra state is a ghost like any
//! other: it is repaired, refused or flagged by the same rules). The branch scattering and the
//! branch level errors (`diagnose::kappa::level_errors`) are not criteria: the empty f
//! branches sit 0.2 Ha off their reference because the box follows the wall, which is no
//! defect.
//!
//! `Searched.flagged` is the one statement that the dataset still has a defect; a defect the
//! record has no scattering key for (a missing state, a branch ghost) is in the record's
//! `defects` (`missing`, `branch_ghosts`), so a flagged dataset never carries an empty record.
mod attempts;
mod message;

pub use attempts::{plan, radii_of, Attempt};
pub use message::{general, Error, Tried, Why};

use super::ranking::{Badness, Keeper};
use super::{
    channel_radii, generate_with_atom,
    recipe::{Cutoffs, Recipe},
    reference::solve_reference_with,
    Generated,
};
use crate::{
    models::{
        atom::Atom,
        xc_scheme::{Variational, XcSource},
    },
    paw::diagnose::{
        self,
        ghosts::{ghost_errors, missing_states},
        intruding::intruding_s_miss,
        reference::{check_reference, AeLogs, AllElectron},
        scattering::{scattering_errors, unprojected_errors},
        view::Smooth,
        Diagnosis,
    },
    paw::record::Defects,
    Context,
};
use std::cmp::Ordering;

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
    /// radius, a raise of the local potential that is not the one the policy tables give the
    /// dataset (a caller assigned it: the generator this replaces pins `local_shift` whenever
    /// the caller passed one), and the radii and raise a UPAW-LCAO table lays under the
    /// caller's options (the generator treats those as the caller's). A caller who passes the
    /// table's own value cannot be told from the table: the binding sets [`Options::pinned`].
    pub fn of(recipe: &Recipe) -> Self {
        let upaw = (recipe.family == crate::paw::record::Family::UpawLcao)
            .then(|| super::tables_upaw::recipe(&recipe.symbol, &recipe.functional));
        let tabulated = match recipe.family {
            crate::paw::record::Family::PawLcao => {
                Recipe::for_dataset(&recipe.symbol, &recipe.functional, recipe.relativity)
            }
            crate::paw::record::Family::UpawLcao => {
                Recipe::upaw(&recipe.symbol, &recipe.functional, recipe.relativity)
            }
        }
        .map(|table| table.local_shift)
        .ok();
        Self {
            r_cut: !matches!(recipe.cutoffs, Cutoffs::Rule),
            r_cut_local: recipe.local_radius.is_some(),
            local_shift: upaw.is_some_and(|u| u.local_shift.is_some())
                || tabulated.is_some_and(|shift| shift != recipe.local_shift),
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
    /// Whether the selected construction still has defects (only `Flag`, or a pinned choice
    /// whose only defect is the family's acceptance, returns one). Its `generated.dataset.defects`
    /// is never empty then: every defect has a key in the record.
    pub flagged: bool,
    /// Constructions actually generated, the first one included (an attempt whose recipe equals
    /// an earlier one is reported with that verdict and not built again).
    pub built: usize,
}

/// The pieces of the search a control may replace, to show what each one buys. The faithful
/// search is [`Faithful`]; a control passes a variant to [`search_reviewed`] and compares its
/// result with the faithful one, so the wrong variant runs through the loop it is a variant of.
#[doc(hidden)]
pub trait Review {
    /// The remedies, from the table in order (a control skips one).
    fn remedies(&self, table: Vec<Attempt>) -> Vec<Attempt> {
        table
    }
    /// Whether a ghost stops the examination of a construction (the faithful search stops).
    fn stops_at_ghost(&self) -> bool {
        true
    }
    /// The diagnosis the search acts on (a control blinds one of its criteria).
    fn judge(&self, diagnosis: Diagnosis) -> Diagnosis {
        diagnosis
    }
    /// The order that picks what `Flag` returns (a control reverses it).
    fn rank(&self, a: &Badness, b: &Badness) -> Ordering {
        a.compare(b)
    }
    /// Whether an attempt equal to an earlier one is reported with that verdict instead of being
    /// built again.
    fn reuses_duplicates(&self) -> bool {
        true
    }
}

/// The search as it is.
#[doc(hidden)]
pub struct Faithful;

impl Review for Faithful {}

/// Judge one construction against the all-electron `atom`: the ghosts first, and only when there
/// is none the scattering, the unprojected channels, the s completeness and the missing states
/// (a ghosted construction is not examined further, and a phase that cannot be evaluated on it
/// must not hide the ghost). A fully relativistic dataset has its `j` branches examined for
/// ghosts after the scalar channels and before the scattering. `cache` holds the all-electron
/// logarithmic derivatives of `atom`.
pub fn examine(
    ctx: &Context,
    generated: &Generated,
    atom: &Atom,
    cache: &mut AeLogs,
) -> diagnose::Outcome<Diagnosis> {
    examine_through(ctx, generated, atom, cache, true)
}

#[doc(hidden)]
/// [`examine`] that, when `stop_at_ghost` is false, evaluates every criterion on a ghosted
/// construction too (a control measures what the other criteria alone would catch).
pub fn examine_through(
    ctx: &Context,
    generated: &Generated,
    atom: &Atom,
    cache: &mut AeLogs,
    stop_at_ghost: bool,
) -> diagnose::Outcome<Diagnosis> {
    let dataset = &generated.dataset;
    check_reference(dataset, atom)?;
    let smooth = Smooth::scalar(ctx, dataset)?;
    let ae = AllElectron::new(atom, smooth.atomic_number, &smooth.frozen, &smooth.semicore);
    let ghosts = ghost_errors(ctx, &smooth, Some(&ae))?;
    if !ghosts.is_empty() && stop_at_ghost {
        return Ok(Diagnosis {
            ghosts,
            ..Diagnosis::default()
        });
    }
    let with_branches = Diagnosis {
        ghosts,
        ..Diagnosis::default()
    }
    .with_branches(ctx, dataset)?;
    if with_branches.has_ghost() && stop_at_ghost {
        return Ok(with_branches);
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
        phases,
        unprojected: unprojected_errors(ctx, &smooth, &ae, cache)?,
        s_miss,
        missing: missing_states(ctx, &smooth)?,
        ..with_branches
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
    let atom = solve_reference_with(ctx, recipe, source).map_err(Error::Failed)?;
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
    search_reviewed(ctx, recipe, options, atom, source, &Faithful)
}

#[doc(hidden)]
/// [`search_with_atom`] with a [`Review`]: the one loop, with one piece of it replaced.
pub fn search_reviewed(
    ctx: &Context,
    recipe: &Recipe,
    options: &Options,
    atom: &Atom,
    source: &dyn XcSource,
    review: &dyn Review,
) -> Result<Searched, Error> {
    // An infeasible first construction (a j-branch whose overlap operator is not positive
    // definite: pbe-fr Hf, Hg and Ra) is a failed attempt like any other, and the remedies are
    // planned from the radii it would have had. `keep` builds one construction and reports it.
    let (first, first_failure) = match generate_with_atom(ctx, recipe, atom, source) {
        Ok(made) => (Some(made), None),
        Err(error) if is_attempt_failure(&error) && options.mode != Mode::Keep => {
            (None, Some(error.to_string()))
        }
        Err(error) => return Err(Error::Failed(error)),
    };
    let mut built = usize::from(first.is_some());
    if options.mode == Mode::Keep {
        let first = first.expect("keep returns the error of its only construction");
        return Ok(Searched {
            generated: first,
            index: 0,
            label: FIRST.into(),
            diagnosis: None,
            tried: Vec::new(),
            flagged: false,
            built,
        });
    }
    let radii = match &first {
        Some(made) => radii_of(&made.dataset),
        None => channel_radii(ctx, recipe, atom).map_err(Error::Failed)?,
    };
    let attempts = review.remedies(plan(recipe, &radii));
    let mut cache = AeLogs::new();
    let mut tried = Vec::new();
    let mut keeper = Keeper::<Candidate>::new();
    // Every construction that failed, with its recipe: an attempt whose recipe equals an earlier
    // one cannot differ from it (the generator is deterministic and the atom is shared), so it
    // is reported with the earlier verdict instead of being built again.
    let mut seen: Vec<(Recipe, Why)> = Vec::new();
    let judged = |built: &Generated, cache: &mut AeLogs| {
        examine_through(ctx, built, atom, cache, review.stops_at_ghost()).map(|d| review.judge(d))
    };

    let problem = match first {
        None => Why::NotBuilt(first_failure.expect("a failed first construction has its error")),
        Some(first) => match judged(&first, &mut cache) {
            Ok(diagnosis) => match Why::of(&diagnosis) {
                None => {
                    return Ok(Searched {
                        generated: first,
                        index: 0,
                        label: FIRST.into(),
                        diagnosis: Some(diagnosis),
                        tried,
                        flagged: false,
                        built,
                    })
                }
                Some(why) => {
                    let acceptance_only =
                        !diagnosis.has_ghost() && diagnosis.wrong_phases().is_empty();
                    if options.pinned.any() && acceptance_only && options.mode != Mode::Refuse {
                        // Only the family's acceptance failed and the caller chose what a repair
                        // would change: keep the construction and record what it misses.
                        return flagged(first, diagnosis, 0, FIRST.into(), tried, built);
                    }
                    if options.mode == Mode::Flag {
                        keeper.offer_by(
                            Badness::of(&diagnosis),
                            Candidate {
                                generated: first,
                                diagnosis,
                                index: 0,
                                label: FIRST.into(),
                            },
                            |a, b| review.rank(a, b),
                        );
                    }
                    why
                }
            },
            Err(diagnose::Error::Unevaluated(text)) => Why::Unevaluated(text),
            // A check the construction itself makes impossible (its overlap operator is not positive
            // definite: pbe-fr Hf and Hg) is a failed attempt, as it is when the generator meets it.
            Err(diagnose::Error::Failed(error)) if is_attempt_failure(&error) => {
                Why::Unevaluated(error.to_string())
            }
            Err(diagnose::Error::Failed(error)) => return Err(Error::Failed(error)),
        },
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
        if review.reuses_duplicates() {
            if let Some((_, why)) = seen.iter().find(|(earlier, _)| *earlier == attempt.recipe) {
                tried.push(Tried {
                    label: attempt.label,
                    why: why.clone(),
                });
                continue;
            }
        }
        let outcome = generate_with_atom(ctx, &attempt.recipe, atom, source);
        if outcome.is_ok() {
            built += 1;
        }
        let why = match outcome {
            Err(error) if is_attempt_failure(&error) => Why::NotBuilt(error.to_string()),
            Err(error) => return Err(Error::Failed(error)),
            Ok(made) => match judged(&made, &mut cache) {
                Err(diagnose::Error::Unevaluated(text)) => Why::Unevaluated(text),
                Err(diagnose::Error::Failed(error)) if is_attempt_failure(&error) => {
                    Why::Unevaluated(error.to_string())
                }
                Err(diagnose::Error::Failed(error)) => return Err(Error::Failed(error)),
                Ok(diagnosis) => match Why::of(&diagnosis) {
                    None => {
                        return Ok(Searched {
                            generated: made,
                            index,
                            label: attempt.label,
                            diagnosis: Some(diagnosis),
                            tried,
                            flagged: false,
                            built,
                        })
                    }
                    Some(why) => {
                        if options.mode == Mode::Flag {
                            keeper.offer_by(
                                Badness::of(&diagnosis),
                                Candidate {
                                    generated: made,
                                    diagnosis,
                                    index,
                                    label: attempt.label.clone(),
                                },
                                |a, b| review.rank(a, b),
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
        return flagged(
            best.generated,
            best.diagnosis,
            best.index,
            best.label,
            tried,
            built,
        );
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
    built: usize,
) -> Result<Searched, Error> {
    generated.dataset.defects = diagnosis.defects();
    // A flag without a record would be silent: every defect the criteria find has a key.
    if generated.dataset.defects == Defects::default() {
        return Err(Error::Failed(crate::Error::Invalid(format!(
            "{}: a construction flagged as defective has no defect to record; this is a bug of \
             the search, not a property of the element",
            generated.dataset.symbol
        ))));
    }
    Ok(Searched {
        generated,
        index,
        label,
        diagnosis: Some(diagnosis),
        tried,
        flagged: true,
        built,
    })
}
