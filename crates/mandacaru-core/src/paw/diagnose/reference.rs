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
        atom::{self, Atom, AtomModel, AtomOptions},
        configuration::valence_subshells,
        relativity::Treatment,
        selection::relaxed_configuration,
    },
    paw::{
        checks::log_derivatives_ae,
        generate::{
            recipe::Recipe,
            tables_elements::{REFERENCE_MIXING, REFERENCE_TOLERANCE},
        },
        invalid,
        record::{Family, PawDataset},
    },
    Context, Result,
};
use std::collections::HashMap;

/// A configuration as the atom solver takes it.
type Occupations = Vec<(usize, usize, f64)>;

/// The recipe-independent facts of a record that fix its reference atom.
fn recipe_of(ds: &PawDataset) -> Result<Recipe> {
    // The tables know no Dirac recipe yet; only the grid, the radius and the occupancy overlay
    // are read from it, and the atom below is solved with the record's own treatment.
    let table_treatment = match Treatment::parse(&ds.relativity)? {
        Treatment::Dirac => Treatment::Scalar,
        other => other,
    };
    match ds.family {
        Family::PawLcao => Recipe::for_dataset(&ds.symbol, &ds.xc, table_treatment),
        Family::UpawLcao => Recipe::upaw(&ds.symbol, &ds.xc, table_treatment),
    }
}

/// The self-consistent atom a record was generated on: the relaxed ground state with the
/// element's occupancy overrides, in the record's functional and relativistic treatment.
pub fn reference_atom(ctx: &Context, ds: &PawDataset) -> Result<Atom> {
    let recipe = recipe_of(ds)?;
    let treatment = Treatment::parse(&ds.relativity)?;
    let z = recipe.atomic_number;
    let configuration = if recipe.reference_configuration.is_empty() {
        None
    } else {
        let mut occupations: Occupations = relaxed_configuration(
            ctx,
            z,
            &recipe.functional,
            treatment,
            recipe.r_max,
            Some(recipe.points),
        )?;
        for &((n, l), q) in &recipe.reference_configuration {
            match occupations.iter_mut().find(|s| (s.0, s.1) == (n, l)) {
                Some(state) => state.2 = q,
                None => occupations.push((n, l, q)),
            }
        }
        occupations.retain(|s| s.2 > 0.);
        Some(occupations)
    };
    atom::solve_model(
        ctx,
        z,
        &AtomOptions {
            points: recipe.points,
            r_max: recipe.r_max,
            max_iterations: 200,
            tolerance: REFERENCE_TOLERANCE,
            mixing: REFERENCE_MIXING,
        },
        &AtomModel {
            functional: recipe.functional.clone(),
            relativity: treatment,
            polish: atom::POLISH_ROUNDS,
        },
        configuration.as_deref(),
        None,
    )
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
            .collect::<Occupations>();
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

/// The reference atom must be the one the record was built on.
pub fn check_reference(ds: &PawDataset, atom: &Atom) -> Result<()> {
    let treatment = Treatment::parse(&ds.relativity)?;
    if !atom.converged || atom.functional != ds.xc || atom.relativity != treatment {
        return Err(invalid(format!(
            "diagnose: {}: the reference atom (xc {:?}, relativity {:?}, converged {}) is not the one the record asks for (xc {:?}, relativity {:?})",
            ds.symbol,
            atom.functional,
            atom.relativity.name(),
            atom.converged,
            ds.xc,
            ds.relativity
        )));
    }
    Ok(())
}
