//! The scale of the stored spin-orbit `coupling` that makes the engine's isolated atom reproduce
//! the Dirac splittings of the record (gate G13), derived from the atom and not tabulated.
//!
//! The engine evaluates the `j` projectors' coupling on the scalar smooth waves of its basis
//! ([`crate::paw::engine_spin_orbit`]), so its splitting of a heavy shell falls short of the
//! all-electron one (bismuth 6p 1.61 eV against 1.84). Each bound occupied reference `k` of a
//! channel gets one factor `f_k` on its projectors and on the empty references above it (both
//! `j` branches); the coupling entry `(a, b)` is multiplied by `f_a f_b`. A shell's factor is
//! found by Newton steps on `ln(engine / Dirac)` against `ln` of the cumulative splitting factor,
//! with the slope measured from the last two rounds (a shell whose splitting answers a change of
//! the coupling with less than its full weight, such as the 5d of an open-4f atom, would take
//! dozens of rounds at slope one), a step halved whenever an error changes sign, and the engine
//! asked again after each scaling at the Fermi-Dirac width of the first round. The loop stops when
//! every shell is within [`CLOSED`] of the Dirac value. Only the stored `coupling` is scaled: the
//! arrays the branch diagnostics read (`coupling_screened`, `overlap_coupling`, the averages) keep
//! the branch construction, whose levels the ghost checks verify.
//!
//! A shell that does not close in [`ROUNDS`] is given up when it is an open `d` or `f` shell
//! ([`Splitting::unreliable`]): its factor is taken back, its coupling stays as built and it is
//! reported; any other shell that does not close is an error.
use crate::{
    paw::{engine_spin_orbit::Splitting, invalid, record::PawDataset},
    Result,
};
use std::collections::BTreeMap;

/// The loop stops when every shell is this close (relative) to its Dirac splitting, or within
/// [`CLOSED_FLOOR`] of it.
pub const CLOSED: f64 = 0.003;
/// 1 meV (Hartree): the box splits a small multiplet by about this much, so a smaller splitting
/// is not asked to close to 0.3 percent.
pub const CLOSED_FLOOR: f64 = 0.001 / 27.211_386_245_988;
/// At most this many engine atoms are solved.
const ROUNDS: usize = 8;

/// What the calibration did for one shell.
#[derive(Clone, Copy, Debug)]
pub struct Scale {
    pub l: usize,
    /// The reference of the channel.
    pub state: usize,
    /// The factor now applied to this shell's splitting (1 when it was left alone or given up).
    pub factor: f64,
    /// `engine / Dirac - 1` before any scaling, and after the last one (NaN: not measured).
    pub before: f64,
    pub after: f64,
    /// The shell did not close and was taken back to the coupling as built.
    pub given_up: bool,
    /// The Fermi-Dirac width (eV) of the last SCF that measured this shell.
    pub width: Option<f64>,
}

/// Multiply the coupling entries `(a, b)` of channel `l` by `g_a g_b`, `g` per reference (the
/// projectors of the two `j` branches are listed branch after branch).
fn apply(ds: &mut PawDataset, l: usize, g: &[f64]) -> Result<()> {
    let entry = ds
        .spin_orbit
        .get_mut(&(l as u32))
        .ok_or_else(|| invalid(format!("the record has no spin-orbit entry l={l}")))?;
    let count = entry.reference_energies.data().len() / 2;
    let size = entry.coupling.shape.first().copied().unwrap_or(0);
    if count == 0 || size != 2 * count || entry.coupling.shape != [size, size] || g.len() != count {
        return Err(invalid(format!(
            "the spin-orbit coupling l={l} is not two branches of {count} projectors"
        )));
    }
    for (index, value) in entry.coupling.data.as_mut_slice().iter_mut().enumerate() {
        *value *= g[(index / size) % count] * g[(index % size) % count];
    }
    Ok(())
}

/// The references a shell's factor acts on: from its own up to the next measured shell.
fn range(states: &[usize], state: usize, count: usize) -> std::ops::Range<usize> {
    let end = states.iter().copied().find(|&s| s > state).unwrap_or(count);
    state..end
}

/// Scale the `coupling` of every spin-orbit channel of `ds` until `measure` (the engine's
/// splittings of a record as it will be released, the ladder of the SCF starting at the width it is
/// given in eV) is within [`CLOSED`] of the Dirac ones. A record without a spin-orbit shell is
/// left alone.
pub fn calibrate(
    ds: &mut PawDataset,
    mut measure: impl FnMut(&PawDataset, Option<f64>) -> Result<Vec<Splitting>>,
) -> Result<Vec<Scale>> {
    let mut done: BTreeMap<(usize, usize), Scale> = BTreeMap::new();
    // Per channel: how much of the Newton step is taken (halved when an error changes sign), the
    // errors and cumulative factors of the last round of each shell, and the cumulative factor
    // on every reference.
    let mut damping: BTreeMap<usize, f64> = BTreeMap::new();
    let mut last: BTreeMap<(usize, usize), (f64, f64)> = BTreeMap::new();
    let mut slope: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    let mut cumulative: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    let mut floor: Option<f64> = None;
    for round in 0..=ROUNDS {
        // The rounds measure at the width of the first. When the SCF does not converge there any
        // more (the 4f occupations of an open-4f atom are not stable under a change of the
        // coupling), the ladder is used again and that round moves the shells whose splitting is
        // the spin-orbit one (p shells) only: an open d or f shell is not judged at other
        // occupations than the first round's.
        let (found, changed) = match measure(ds, floor) {
            Ok(found) => (found, false),
            Err(error) if round > 0 && crate::paw::atom_scf::not_converged(&error) => {
                (measure(ds, None)?, true)
            }
            Err(error) => return Err(error),
        };
        if round == 0 {
            floor = found.first().map(|first| first.width);
        }
        let mut step: BTreeMap<usize, BTreeMap<usize, f64>> = BTreeMap::new();
        for s in &found {
            if changed && s.unreliable() {
                continue;
            }
            // A shell the atom does not show (an open d or f shell: the exchange of its
            // fractionally filled j multiplets is not its spin-orbit splitting) keeps its coupling.
            let Some(engine) = s.engine else {
                done.entry((s.l, s.state)).or_insert(Scale {
                    l: s.l,
                    state: s.state,
                    factor: 1.,
                    before: f64::NAN,
                    after: f64::NAN,
                    given_up: false,
                    width: None,
                });
                continue;
            };
            let error = engine / s.dirac - 1.;
            let entry = done.entry((s.l, s.state)).or_insert(Scale {
                l: s.l,
                state: s.state,
                factor: 1.,
                before: error,
                after: error,
                given_up: false,
                width: None,
            });
            entry.width = Some(s.width);
            entry.after = error;
            let y = (engine / s.dirac).ln();
            let x = entry.factor.ln();
            if let Some(&(y0, x0)) = last.get(&(s.l, s.state)) {
                if y0 * y < 0. {
                    *damping.entry(s.l).or_insert(1.) *= 0.5;
                }
                if (x - x0).abs() > 1e-9 {
                    let measured = (y - y0) / (x - x0);
                    if measured.is_finite() {
                        slope.insert((s.l, s.state), measured.clamp(0.25, 1.5));
                    }
                }
            }
            last.insert((s.l, s.state), (y, x));
            if (engine - s.dirac).abs() > (CLOSED * s.dirac.abs()).max(CLOSED_FLOOR) {
                let lambda = damping.get(&s.l).copied().unwrap_or(1.);
                let s_k = slope.get(&(s.l, s.state)).copied().unwrap_or(1.);
                let ratio = (-y * lambda / s_k).exp();
                step.entry(s.l).or_default().insert(s.state, ratio.sqrt());
                entry.factor *= ratio;
            }
        }
        if step.is_empty() {
            if found.iter().all(|s| s.engine.is_none() || s.passes()) {
                return Ok(done.into_values().collect());
            }
            return give_up(ds, &found, done, cumulative);
        }
        if round == ROUNDS {
            if found.iter().all(|s| s.engine.is_none() || s.passes()) {
                return Ok(done.into_values().collect());
            }
            return give_up(ds, &found, done, cumulative);
        }
        for (l, f) in &step {
            let count = ds
                .spin_orbit
                .get(&(*l as u32))
                .map(|so| so.reference_energies.data().len() / 2)
                .ok_or_else(|| invalid(format!("the record has no spin-orbit entry l={l}")))?;
            let states: Vec<usize> = found
                .iter()
                .filter(|t| t.l == *l)
                .map(|t| t.state)
                .collect();
            let mut g = vec![1.; count];
            for (&state, &value) in f {
                for a in range(&states, state, count) {
                    g[a] = value;
                }
            }
            // References below the lowest shell stepped keep their coupling.
            apply(ds, *l, &g)?;
            let total = cumulative.entry(*l).or_insert_with(|| vec![1.; count]);
            for a in 0..count {
                total[a] *= g[a];
            }
        }
    }
    unreachable!("the loop returns or errors in its last round")
}

/// The shells of `found` still outside the gate: an open `d` or `f` shell is taken back to the
/// coupling as built and reported, any other is an error. `cumulative` is the factor applied on every
/// reference of each channel so far.
fn give_up(
    ds: &mut PawDataset,
    found: &[Splitting],
    mut done: BTreeMap<(usize, usize), Scale>,
    mut cumulative: BTreeMap<usize, Vec<f64>>,
) -> Result<Vec<Scale>> {
    // In an atom with a fractionally filled d or f shell the occupations of that shell move every
    // splitting: any shell still out is given up.
    let open_atom = found.iter().any(|t| t.unreliable());
    for s in found {
        if s.engine.is_some() && !s.passes() {
            if !s.unreliable() && !open_atom {
                return Err(invalid(format!(
                    "the {} spin-orbit splitting (l={} reference {}) is not closed after the scalings",
                    ds.symbol, s.l, s.state
                )));
            }
            // A shell that was never stepped (every round that measured it was one at another
            // width, which moves no open shell) has no factor to take back.
            let Some(g) = cumulative.get(&s.l).cloned() else {
                if let Some(scale) = done.get_mut(&(s.l, s.state)) {
                    scale.given_up = true;
                }
                continue;
            };
            let states: Vec<usize> = found
                .iter()
                .filter(|t| t.l == s.l)
                .map(|t| t.state)
                .collect();
            let count = g.len();
            let mut back = vec![1.; count];
            for a in range(&states, s.state, count) {
                back[a] = 1. / g[a];
            }
            apply(ds, s.l, &back)?;
            if let Some(scale) = done.get_mut(&(s.l, s.state)) {
                scale.factor = 1.;
                scale.given_up = true;
            }
            if let Some(v) = cumulative.get_mut(&s.l) {
                for a in range(&states, s.state, count) {
                    v[a] = 1.;
                }
            }
        }
    }
    Ok(done.into_values().collect())
}
