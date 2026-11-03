//! What a failed attempt of the search says about itself, as data and as the text a person
//! reads: the cause of a refusal and the `tried` list of an exhausted search.
//!
//! The text is the one the generator this replaces wrote (`ghost_free`), so a calculation that
//! refuses an element reads the same words. Numbers use the `%g` family of that text: depths
//! `+.3g`, phases `.3f`, unprojected phases and misses `.2f`.
use crate::paw::diagnose::{
    intruding::failed_acceptance, scattering::UNPROJECTED_TOLERANCE, view::Key, Diagnosis,
};
use std::collections::BTreeMap;

/// `value` with `precision` significant digits in the `%g` style: trailing zeros dropped, an
/// exponent when the decimal exponent is below -4 or at least `precision`. `plus` prints the
/// sign of a positive number too.
pub fn general(value: f64, precision: usize, plus: bool) -> String {
    let sign = if value.is_sign_negative() && value != 0. {
        "-"
    } else if plus {
        "+"
    } else {
        ""
    };
    if value == 0. {
        return format!("{sign}0");
    }
    if !value.is_finite() {
        return format!("{value}");
    }
    let scientific = format!("{:.*e}", precision - 1, value.abs());
    let (mantissa, exponent) = scientific.split_once('e').expect("scientific notation");
    let exponent: i32 = exponent.parse().expect("an integer exponent");
    let trim = |text: &str| -> String {
        if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            text.to_string()
        }
    };
    if exponent < -4 || exponent >= precision as i32 {
        let mark = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{}e{mark}{:02}", trim(mantissa), exponent.abs())
    } else {
        let decimals = (precision as i32 - 1 - exponent).max(0) as usize;
        format!("{sign}{}", trim(&format!("{:.*}", decimals, value.abs())))
    }
}

/// `l=0 -0.131 Ha, l=1 -0.133 Ha`: levels of the channels that hold a ghost.
pub fn describe_levels(levels: &BTreeMap<usize, f64>) -> String {
    levels
        .iter()
        .map(|(l, e)| format!("l={l} {} Ha", general(*e, 3, true)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `l=1 kappa=-2 -3.74 Ha`: the `j` branches that hold a ghost.
pub fn describe_branches(branches: &BTreeMap<Key, f64>) -> String {
    branches
        .iter()
        .map(|(k, e)| {
            format!(
                "l={} kappa={} {} Ha",
                k.l,
                k.kappa.unwrap_or(0),
                general(*e, 3, true)
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// `l=1 0.123/0.456 rad`: the near and far phase errors of the channels outside tolerance.
pub fn describe_phases(phases: &BTreeMap<usize, [f64; 2]>) -> String {
    phases
        .iter()
        .map(|(l, [near, far])| format!("l={l} {near:.3}/{far:.3} rad"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The unprojected channels outside their tolerance and a failed s completeness.
pub fn describe_incomplete(diagnosis: &Diagnosis) -> String {
    let mut parts = diagnosis
        .unprojected
        .iter()
        .filter(|(_, &e)| !(e <= UNPROJECTED_TOLERANCE))
        .map(|(l, e)| format!("l={l} unprojected phase error {e:.2} rad"))
        .collect::<Vec<_>>();
    if let Some(miss) = failed_acceptance(diagnosis.s_miss) {
        parts.push(format!("intruding-1s miss {miss:.2}"));
    }
    parts.join(", ")
}

/// `l=1 +0.0123 Ha`: the channels whose nearest smooth level is far from a bound reference.
pub fn describe_missing(missing: &BTreeMap<usize, f64>) -> String {
    missing
        .iter()
        .map(|(l, d)| format!("l={l} {} Ha", general(*d, 3, true)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Why one construction was not accepted.
#[derive(Debug, Clone, PartialEq)]
pub enum Why {
    /// A bound level below the reference that the atom does not have (or an excited or local
    /// ghost): `{l: eps_0 - eps_ref}`.
    Ghost(BTreeMap<usize, f64>),
    /// A `j` branch of a fully relativistic dataset holds a ghost: `{(l, kappa): eps_0 - eps_ref}`.
    /// The scalar generator never looked at branches, so this text has no counterpart in it.
    BranchGhost(BTreeMap<Key, f64>),
    /// Channels whose scattering is outside `PHASE_TOLERANCE` near or `RESONANCE_TOLERANCE` far.
    Phase(BTreeMap<usize, [f64; 2]>),
    /// An unprojected channel outside its tolerance or a failed s completeness (carried as the
    /// text of [`describe_incomplete`], computed once).
    Incomplete(String),
    /// A physical bound reference the smooth Hamiltonian does not bind: `{l: level - reference}`.
    Missing(BTreeMap<usize, f64>),
    /// A check that could not be evaluated (a level or phase that is not finite).
    Unevaluated(String),
    /// The construction itself failed (an infeasible smooth wave, a singular system).
    NotBuilt(String),
}

impl Why {
    /// The reason a clean-or-not diagnosis is not accepted, in the order the search reports
    /// them: ghosts, then wrong scattering, then incomplete construction, then missing states.
    pub fn of(diagnosis: &Diagnosis) -> Option<Self> {
        if !diagnosis.ghosts.is_empty() {
            return Some(Self::Ghost(diagnosis.ghosts.clone()));
        }
        if !diagnosis.branch_ghosts.is_empty() {
            return Some(Self::BranchGhost(diagnosis.branch_ghosts.clone()));
        }
        let wrong = diagnosis.wrong_phases();
        if !wrong.is_empty() {
            return Some(Self::Phase(wrong));
        }
        let incomplete = describe_incomplete(diagnosis);
        if !incomplete.is_empty() {
            return Some(Self::Incomplete(incomplete));
        }
        (!diagnosis.missing.is_empty()).then(|| Self::Missing(diagnosis.missing.clone()))
    }

    /// The short form of a `tried` line (after the label).
    pub fn short(&self) -> String {
        match self {
            Self::Ghost(levels) => format!("ghost {}", describe_levels(levels)),
            Self::BranchGhost(branches) => format!("ghost {}", describe_branches(branches)),
            Self::Phase(phases) => format!("phase {}", describe_phases(phases)),
            Self::Incomplete(text) => text.clone(),
            Self::Missing(missing) => format!("missing state {}", describe_missing(missing)),
            Self::Unevaluated(text) | Self::NotBuilt(text) => text.clone(),
        }
    }

    /// The problem of the first construction, as a refusal states it.
    pub fn problem(&self) -> String {
        match self {
            Self::Ghost(levels) => {
                format!(
                    "ghost state below the reference ({})",
                    describe_levels(levels)
                )
            }
            Self::BranchGhost(branches) => format!(
                "ghost state below the reference ({})",
                describe_branches(branches)
            ),
            Self::Phase(phases) => format!("wrong scattering ({})", describe_phases(phases)),
            Self::Incomplete(text) => format!("an incomplete construction ({text})"),
            Self::Missing(missing) => format!(
                "a bound reference the smooth Hamiltonian does not bind ({})",
                describe_missing(missing)
            ),
            Self::Unevaluated(text) => format!("a check that could not be evaluated ({text})"),
            Self::NotBuilt(text) => format!("a construction that failed ({text})"),
        }
    }
}

/// One attempt that was not accepted, with the label of its remedy.
#[derive(Debug, Clone, PartialEq)]
pub struct Tried {
    pub label: String,
    pub why: Why,
}

impl std::fmt::Display for Tried {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.label, self.why.short())
    }
}

/// Why a search ended without a dataset.
#[derive(Debug)]
pub enum Error {
    /// The first construction is not clean and the mode, or a choice the caller pinned, forbids
    /// the search.
    Refused {
        symbol: String,
        problem: String,
        why: String,
    },
    /// Every attempt failed.
    Exhausted {
        symbol: String,
        problem: String,
        tried: Vec<Tried>,
    },
    /// The generator or a diagnostic failed for a reason that is not a property of the attempt
    /// (a budget, an unsupported feature, an invalid recipe).
    Failed(crate::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused {
                symbol,
                problem,
                why,
            } => write!(
                f,
                "{symbol}: {problem} {why}.  Leave the cutoffs and the local potential to the \
                 generator, or pass ghosts='keep' to study it."
            ),
            Self::Exhausted {
                symbol,
                problem,
                tried,
            } => {
                write!(
                f,
                "{symbol}: {problem} and no remedy removed it while keeping the scattering ({}).",
                tried.iter().map(|t| t.to_string()).collect::<Vec<_>>().join("; ")
            )
            }
            Self::Failed(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    /// The same failure as a crate error: a refusal is an invalid request, anything else is
    /// what it was.
    pub fn into_core(self) -> crate::Error {
        match self {
            Self::Failed(error) => error,
            other => crate::Error::Invalid(other.to_string()),
        }
    }
}
