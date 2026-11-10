//! The all-electron side of the diagnostics: the reference atom a dataset was built on, the
//! equation its logarithmic derivatives are integrated with, and the core shells that do not
//! count as valence states.
//!
//! A dataset read from a file carries no atom; [`reference_atom`] solves it again from the
//! record (element, functional, relativity), at the generator's grid, tolerance and mixing,
//! and with the occupancies the policy tables lay over the ground state for that element.
use super::{view::Key, Outcome};
use crate::{
    models::{
        atom::Atom,
        configuration::valence_subshells,
        relativity::Treatment,
        xc_scheme::{Variational, XcSource},
    },
    paw::{
        checks::log_derivatives_ae,
        generate::{recipe::Recipe, reference::solve_reference_with},
        invalid,
        record::{Family, PawDataset},
    },
    Context, Result,
};
use std::collections::HashMap;

/// The recipe a record was generated from: its element, functional and treatment, with the
/// family's policy tables laid over the defaults.
fn recipe_of(ds: &PawDataset) -> Result<Recipe> {
    let treatment = Treatment::parse(&ds.relativity)?;
    match ds.family {
        Family::PawLcao => Recipe::for_dataset(&ds.symbol, &ds.xc, treatment),
        Family::UpawLcao => Recipe::upaw(&ds.symbol, &ds.xc, treatment),
    }
}

/// The self-consistent atom a record was generated on: the relaxed ground state with the
/// element's occupancy overrides, in the record's functional and relativistic treatment, solved
/// by the generator's own routine so a diagnosis and a generation cannot disagree on the atom.
pub fn reference_atom(ctx: &Context, ds: &PawDataset) -> Result<Atom> {
    reference_atom_with(ctx, ds, &Variational)
}

#[doc(hidden)]
/// [`reference_atom`] with the radial exchange-correlation scheme as a parameter: the shipped PBE
/// datasets were generated on an atom solved with the legacy `AtomSolver`, so diagnosing one of
/// them against its own atom takes that scheme (the library sweep); a freshly generated record
/// takes the production one.
pub fn reference_atom_with(ctx: &Context, ds: &PawDataset, scheme: &dyn XcSource) -> Result<Atom> {
    solve_reference_with(ctx, &recipe_of(ds)?, scheme)
}

/// The reference atom and the equation the comparisons integrate on it.
pub struct AllElectron<'a> {
    pub atom: &'a Atom,
    /// Nuclear charge.
    pub z: f64,
    /// The core subshells the dataset leaves out of its valence: the atom's own core, the
    /// frozen valence subshells, less the semicore ones.
    pub core: Vec<(usize, usize)>,
}

impl<'a> AllElectron<'a> {
    /// `frozen` and `semicore` are the dataset's own subshell lists.
    pub fn new(
        atom: &'a Atom,
        z: u8,
        frozen: &[(usize, usize)],
        semicore: &[(usize, usize)],
    ) -> Self {
        let occupied = atom
            .occupations
            .iter()
            .filter(|s| s.2 > 0.)
            .copied()
            .collect::<Vec<(usize, usize, f64)>>();
        let valence = valence_subshells(&occupied);
        let mut core = occupied
            .iter()
            .map(|s| (s.0, s.1))
            .filter(|s| !valence.contains(s))
            .collect::<Vec<_>>();
        core.extend(frozen.iter().copied());
        core.retain(|s| !semicore.contains(s));
        Self {
            atom,
            z: z as f64,
            core,
        }
    }

    /// Core shells of angular momentum `l`.
    pub fn core_count(&self, l: usize) -> usize {
        self.core.iter().filter(|s| s.1 == l).count()
    }

    /// The equation of a `j`-averaged channel: a Dirac atom's ordinary channels are compared
    /// with the scalar-relativistic wave (the spin-orbit part is a separate term).
    pub fn scalar_treatment(&self) -> Treatment {
        match self.atom.relativity {
            Treatment::Dirac => Treatment::Scalar,
            other => other,
        }
    }

    /// `(treatment, kappa)` of the all-electron equation of a channel.
    pub fn equation(&self, key: Key) -> (Treatment, Option<i32>) {
        match key.kappa {
            None => (self.scalar_treatment(), None),
            Some(kappa) => (Treatment::Dirac, Some(kappa)),
        }
    }
}

/// All-electron logarithmic derivatives already computed, by channel, energy and matching
/// radius. Valid for one reference atom: a search shares one across its attempts.
#[derive(Default)]
pub struct AeLogs {
    values: HashMap<(Key, u64, u64), f64>,
}

impl AeLogs {
    pub fn new() -> Self {
        Self::default()
    }

    /// `R'/R` of the all-electron atom at `r_match` for each energy.
    pub fn at(
        &mut self,
        ctx: &Context,
        ae: &AllElectron<'_>,
        key: Key,
        energies: &[f64],
        r_match: f64,
    ) -> Outcome<Vec<f64>> {
        let id = |e: f64| (key, e.to_bits(), r_match.to_bits());
        let missing = energies
            .iter()
            .copied()
            .filter(|&e| !self.values.contains_key(&id(e)))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            let (treatment, kappa) = ae.equation(key);
            let fresh = log_derivatives_ae(
                ctx,
                ae.atom.r.as_slice(),
                ae.atom.effective.as_slice(),
                key.l,
                &missing,
                r_match,
                ae.z,
                treatment,
                kappa,
            )?;
            for (&e, v) in missing.iter().zip(fresh) {
                self.values.insert(id(e), v);
            }
        }
        Ok(energies.iter().map(|&e| self.values[&id(e)]).collect())
    }
}

/// The reference atom must have converged and be the one the record was built on. The two
/// failures are told apart: an unconverged atom is a failure of the all-electron SCF, a
/// mismatched one a wrong input.
pub fn check_reference(ds: &PawDataset, atom: &Atom) -> Result<()> {
    let treatment = Treatment::parse(&ds.relativity)?;
    if !atom.converged {
        return Err(invalid(format!(
            "diagnose: {}: the all-electron reference atom (xc {:?}, relativity {:?}) did not converge in {} iterations; the dataset cannot be checked against it",
            ds.symbol,
            atom.functional,
            atom.relativity.name(),
            atom.iterations
        )));
    }
    if atom.functional != ds.xc || atom.relativity != treatment {
        return Err(invalid(format!(
            "diagnose: {}: the reference atom (xc {:?}, relativity {:?}) is not the one the record asks for (xc {:?}, relativity {:?})",
            ds.symbol,
            atom.functional,
            atom.relativity.name(),
            ds.xc,
            ds.relativity
        )));
    }
    Ok(())
}
