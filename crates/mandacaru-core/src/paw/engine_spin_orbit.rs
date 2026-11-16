//! The `j` splitting of the isolated atom through the engine itself (gate G13).
//!
//! A Dirac PAW-LCAO dataset is the scalar (`j`-averaged) construction plus an `L.S` coupling on the
//! `j` branches' own projectors. The engine hands those projectors the scalar smooth waves of its
//! LCAO basis, whose inner shape is the scalar partial wave's, so a heavy `p` shell loses
//! 10-15 percent of the splitting the record's all-electron Dirac levels carry (bismuth 6p: 1.60 eV
//! from the engine against 1.836 eV in the record, the complete radial basis 1.79 eV). The loss is
//! a limit of the basis shape and of the self-consistent response of the atom, so it is measured
//! where it happens: in the engine's own isolated-atom spinor SCF ([`super::atom_scf`]).
//!
//! For every angular momentum `l` with a spin-orbit entry, every bound occupied reference (the
//! valence shell and a semicore shell of the same `l` alike) is compared: the splitting of the
//! engine's `j = l - 1/2` and `j = l + 1/2` multiplets (`2l` and `2l + 2` degenerate spinor states,
//! in the engine's energy zero, so only their difference means anything; a multiplet the box splits
//! is taken at its centroid) against the difference of the record's Dirac reference energies. The
//! pairs of multiplets with the right degeneracies whose lower member holds electrons, in ascending
//! energy, are the occupied references in ascending energy. The generator closes the difference by
//! scaling the stored coupling ([`super::generate::spin_orbit_scale`]) and the gate G13 holds the
//! result to [`RELATIVE`] or [`ABSOLUTE`].
use super::{
    atom_scf::{clusters, levels, Cluster, Controls, Levels},
    invalid,
    record::PawDataset,
};
use crate::{Context, Result};

/// A splitting within this fraction of the Dirac one, or within [`ABSOLUTE`], passes G13.
pub const RELATIVE: f64 = 0.02;
/// 10 meV in Hartree (the absolute floor of the gate).
pub const ABSOLUTE: f64 = 0.010 / 27.211_386_245_988;
/// Two spinor levels closer than this (6 meV, Hartree) are one `j` multiplet.
pub const MULTIPLET: f64 = 0.006 / 27.211_386_245_988;
/// An engine splitting outside this range of the Dirac one is a pair of levels that is not the
/// shell's, or a shell whose multiplets the jj filling of an open shell does not leave a spin-orbit
/// splitting (cobalt 3d: 0.36 eV against 0.19, the exchange of the fractionally filled d5/2): the
/// calibration of a shell the atom cannot measure would move the coupling the wrong way, so it
/// refuses.
const WRONG_PAIR: (f64, f64) = (0.6, 1.6);
/// The widest gap (0.4 eV, Hartree) between the groups of one `j` multiplet that the cubic box has split.
const SPAN: f64 = 0.4 / 27.211_386_245_988;
/// A multiplet occupied by less than this per state (the tail of the Fermi-Dirac smearing above
/// the Fermi level) does not hold the shell's electrons.
const OCCUPIED: f64 = 0.02;

/// One channel's splitting in the engine's atom and in the all-electron Dirac atom.
#[derive(Debug, Clone, Copy)]
pub struct Splitting {
    pub l: usize,
    /// Which reference of the channel the level is (its index among the references).
    pub state: usize,
    /// `E(j = l + 1/2) - E(j = l - 1/2)` of the engine's atom (Hartree); `None` when no pair of
    /// multiplets with the degeneracies `2l` and `2l + 2` holds electrons.
    pub engine: Option<f64>,
    /// The same difference of the Dirac reference energies of the record (Hartree).
    pub dirac: f64,
    /// Fermi-Dirac width (eV) of the SCF the engine splitting comes from.
    pub width: f64,
    /// A multiplet of the pair is fractionally filled (between [`OCCUPIED`] and one minus it).
    /// For a `d` or `f` shell the exchange of such a filling is part of the splitting the atom
    /// shows, which is then not the spin-orbit one ([`Splitting::unreliable`]).
    pub fractional: bool,
}

impl Splitting {
    /// An open `d` or `f` shell whose multiplets are fractionally filled: the engine's atom
    /// cannot tell its spin-orbit splitting from the exchange of the filling (cobalt 3d reads 1.86
    /// times the Dirac splitting; the 5d and 4f of the open-4f lanthanides 1.5 times). The
    /// calibration scales such a shell as far as it closes and gives it up when it does not; the
    /// gate reports it instead of refusing the file.
    pub fn unreliable(&self) -> bool {
        self.l >= 2 && self.fractional
    }

    /// Whether the engine splitting is within [`RELATIVE`] or [`ABSOLUTE`] of the Dirac one.
    pub fn passes(&self) -> bool {
        self.engine
            .is_some_and(|e| (e - self.dirac).abs() <= (RELATIVE * self.dirac.abs()).max(ABSOLUTE))
    }
}

/// Every bound occupied reference of every channel with a spin-orbit entry whose Dirac splitting
/// is at least [`ABSOLUTE`], `(l, state, Dirac splitting)`, ascending in `l` and in `state`.
fn shells(ds: &PawDataset) -> Result<Vec<(usize, usize, f64)>> {
    let mut out = Vec::new();
    for (&l, so) in &ds.spin_orbit {
        let channel = ds
            .channels
            .get(&l)
            .ok_or_else(|| invalid(format!("the record has no scalar channel l={l}")))?;
        let energies = channel.reference_energies.data();
        let occupations = channel.occupations.data();
        let branch = so.reference_energies.data();
        let count = branch.len() / 2;
        for state in 0..energies.len() {
            if !(occupations.get(state).is_some_and(|&o| o > 0.) && energies[state] < 0.) {
                continue;
            }
            if so.reference_energies.shape() != [2, count] || state >= count {
                return Err(invalid(format!(
                    "the spin-orbit entry l={l} has no branch reference {state}"
                )));
            }
            let dirac = branch[count + state] - branch[state];
            // A Dirac splitting under the absolute floor is inside the gate whatever the engine
            // gives, and the box splits a multiplet by about as much (boron 2p: 2 meV), so the
            // two multiplets cannot be told apart: the shell is not measured.
            if dirac.abs() >= ABSOLUTE {
                out.push((l as usize, state, dirac));
            }
        }
    }
    Ok(out)
}

/// The `want` spinor states of consecutive groups from `start` on, which are one `j` multiplet:
/// its centroid (count weighted, Hartree), its mean occupation and the index after it. The cubic
/// box splits `d5/2` into `4 + 2` and `f` multiplets into `4 + 2 (+ 2)`, so a multiplet is the run
/// of groups whose counts add up to `2j + 1`, each within [`SPAN`] of the one before.
fn multiplet(groups: &[Cluster], start: usize, want: usize) -> Option<(f64, f64, usize)> {
    let (mut count, mut weighted, mut occupied) = (0, 0., 0.);
    let mut next = start;
    while count < want {
        let group = groups.get(next)?;
        if next > start && group.energy - groups[next - 1].energy > SPAN {
            return None;
        }
        count += group.count;
        weighted += group.energy * group.count as f64;
        occupied += group.occupation * group.count as f64;
        next += 1;
    }
    (count == want).then(|| (weighted / want as f64, occupied / want as f64, next))
}

/// The engine's splitting of each shell of one `l`, `diracs` being the Dirac splittings of its
/// bound occupied references, lowest first. A pair is an occupied lower multiplet (`2l` states)
/// and an upper one (`2l + 2`) somewhere above it; the shells take the lower multiplets in
/// ascending order, and of the uppers a lower multiplet has the one whose splitting is nearest the
/// shell's Dirac one within [`WRONG_PAIR`] (an s level followed by a four-fold group,
/// or the next shell's pair, looks like a p or d pair: the splitting that is not within a factor
/// of [`WRONG_PAIR`] of the Dirac one is not the shell's; transition-metal `s` and `d` levels sit between
/// the multiplets of an `f` shell, so the upper multiplet is searched for, not assumed to be the
/// next one). A shell no pair fits gets `None`.
pub fn assign(groups: &[Cluster], l: usize, diracs: &[f64]) -> Vec<Option<(f64, bool)>> {
    let fractional = |occupation: f64| occupation > OCCUPIED && occupation < 1. - OCCUPIED;
    let lowers: Vec<(usize, f64, bool)> = (0..groups.len())
        .filter_map(|a| {
            let (centroid, occupation, next) = multiplet(groups, a, 2 * l)?;
            (occupation > OCCUPIED).then_some((next, centroid, fractional(occupation)))
        })
        .collect();
    let mut taken = 0;
    diracs
        .iter()
        .map(|&dirac| {
            for (i, &(next, lower, lower_open)) in lowers.iter().enumerate().skip(taken) {
                let best = (next..groups.len())
                    .filter_map(|j| {
                        multiplet(groups, j, 2 * l + 2)
                            .map(|u| (u.0 - lower, lower_open || fractional(u.1)))
                    })
                    .filter(|(splitting, _)| {
                        (WRONG_PAIR.0..WRONG_PAIR.1).contains(&(splitting / dirac))
                    })
                    .min_by(|x, y| (x.0 / dirac).ln().abs().total_cmp(&(y.0 / dirac).ln().abs()));
                if best.is_some() {
                    taken = i + 1;
                    return best;
                }
            }
            None
        })
        .collect()
}

/// The splittings of the levels `atom` an SCF gave for `ds`, one per shell of [`shells`]
/// ([`assign`] per `l`).
pub fn from_levels(ds: &PawDataset, atom: &Levels) -> Result<Vec<Splitting>> {
    let groups = clusters(&atom.levels, MULTIPLET);
    let all = shells(ds)?;
    let mut out = Vec::with_capacity(all.len());
    let mut l = 0;
    while l <= all.iter().map(|s| s.0).max().unwrap_or(0) {
        let of_l: Vec<&(usize, usize, f64)> = all.iter().filter(|s| s.0 == l).collect();
        if !of_l.is_empty() {
            let diracs: Vec<f64> = of_l.iter().map(|s| s.2).collect();
            for (shell, engine) in of_l.iter().zip(assign(&groups, l, &diracs)) {
                out.push(Splitting {
                    l,
                    state: shell.1,
                    engine: engine.map(|e| e.0),
                    dirac: shell.2,
                    width: atom.width,
                    fractional: engine.is_some_and(|e| e.1),
                });
            }
        }
        l += 1;
    }
    Ok(out)
}

/// The engine and the Dirac splitting of every bound occupied reference of every channel with a
/// spin-orbit entry, for the record as stored: one isolated-atom
/// SCF through the engine. A scalar record has no entry and returns an empty list.
pub fn splittings(ctx: &Context, ds: &PawDataset) -> Result<Vec<Splitting>> {
    splittings_from(ctx, ds, None)
}

/// [`splittings`] at the one Fermi-Dirac width `only_width` (eV) when it is given: the rounds of
/// one calibration measure at the width of the first, so the occupations of an open shell do not
/// change from one round to the next.
pub fn splittings_from(ctx: &Context, ds: &PawDataset, only_width: Option<f64>) -> Result<Vec<Splitting>> {
    if shells(ds)?.is_empty() {
        return Ok(Vec::new());
    }
    let controls = Controls {
        only_width,
        ..Controls::default()
    };
    from_levels(ds, &levels(ctx, ds, &controls)?)
}
