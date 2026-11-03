//! The recipe of a PAW-LCAO or UPAW-LCAO dataset: every choice the generator makes before it
//! looks at the solved reference atom, resolved from the policy tables of the dataset.
//!
//! A [`Recipe`] is fully resolved: `for_dataset` / `upaw` lay the per-dataset repairs, the
//! per-element defaults and the generic defaults over one another in that order (the order of
//! the generator this replaces), and a caller who wants something else assigns the field. What
//! cannot be known without the atom (a cutoff from the peak of a reference orbital, the local
//! radius from the largest cutoff) stays a rule in the recipe and is evaluated by the reference
//! step. There is no switch that selects another calculation: a functional other than LDA and
//! PBE and an inconsistent option raise. A Dirac recipe is the scalar-relativistic construction
//! plus the j-resolved spin-orbit term ([`super::spin_orbit`]); it is the same recipe with the
//! reference atom solved with the Dirac equation.
use super::{
    tables_cutoffs, tables_elements as elements, tables_elements::Subshell,
    tables_repairs as repairs, tables_upaw,
};
use crate::{
    models::{elements::atomic_number, relativity::Treatment},
    paw::{invalid, record::Family},
    Result,
};
use std::collections::BTreeMap;

/// Where the augmentation radii come from.
#[derive(Debug, Clone, PartialEq)]
pub enum Cutoffs {
    /// The recipe's table, then `rc_factor` times the outermost peak of the first bound reference.
    Rule,
    /// One radius for every channel of the valence.
    Uniform(f64),
    /// One radius per valence channel; a missing channel raises.
    PerChannel(BTreeMap<usize, f64>),
}

/// The nonlinear core correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Nlcc {
    Off,
    /// The pseudized frozen core the dataset already carries.
    SmoothCore,
    /// The true core re-pseudized at this radius (Bohr).
    Radius(f64),
}

/// Every choice of the generator that does not depend on the solved atom.
#[derive(Debug, Clone, PartialEq)]
pub struct Recipe {
    pub symbol: String,
    pub atomic_number: u8,
    pub family: Family,
    /// `"lda"` or `"pbe"`.
    pub functional: String,
    /// The treatment of the reference atom: `None`, `Scalar` or `Dirac` (the partial waves of a
    /// Dirac dataset are scalar-relativistic; the spin-orbit term is built per `j`).
    pub relativity: Treatment,
    pub cutoffs: Cutoffs,
    /// The radii of the policy tables for this dataset (element, then dataset on top), consulted
    /// by [`Cutoffs::Rule`]; also the radii a semicore channel keeps (they win over its rule).
    pub cutoff_table: BTreeMap<usize, f64>,
    pub rc_factor: f64,
    /// Radius of the local potential; `None` is `local_factor` times the largest radius.
    pub local_radius: Option<f64>,
    pub local_factor: f64,
    /// Raise of the local potential at the origin (Hartree).
    pub local_shift: f64,
    pub q_cut: f64,
    /// Second reference energy above the bound state (Hartree).
    pub energy_offset: f64,
    /// Absolute second reference energies of single channels (Hartree).
    pub second_reference: BTreeMap<usize, f64>,
    pub n_bessel: usize,
    /// Fraction of the all-electron inner norm the smooth waves give up. Free waves (no norm
    /// condition at all) have no place in the dataset record, which stores the deficit, so there
    /// is no such option here.
    pub norm_deficit: f64,
    /// Radial points of the reference atom.
    pub points: usize,
    pub r_max: f64,
    pub nlcc: Nlcc,
    /// Empty channels added above the highest valence one.
    pub extra_l: usize,
    pub highest_channel: Option<usize>,
    /// Occupied valence subshells generated into the frozen core.
    pub frozen_subshells: Vec<Subshell>,
    /// Filled core subshells generated into the valence.
    pub semicore_subshells: Vec<Subshell>,
    /// Occupancies laid over the relaxed ground-state configuration of the reference atom.
    pub reference_configuration: Vec<(Subshell, f64)>,
    /// Radius at which the frozen core is pseudized, when not the compensation radius.
    pub core_radius: Option<f64>,
}

fn radii(entries: &[(usize, f64)]) -> BTreeMap<usize, f64> {
    entries.iter().copied().collect()
}

impl Recipe {
    /// The PAW-LCAO recipe of `symbol` in `functional` (`"lda"` or `"pbe"`).
    pub fn for_dataset(symbol: &str, functional: &str, relativity: Treatment) -> Result<Self> {
        let z = atomic_number(symbol)
            .ok_or_else(|| invalid(format!("{symbol:?} is not a chemical symbol")))?;
        if !matches!(functional, "lda" | "pbe") {
            return Err(invalid(format!(
                "PAW-LCAO generation supports the functionals lda and pbe, not {functional:?}"
            )));
        }
        let mut cutoff_table = radii(tables_cutoffs::by_element(symbol).unwrap_or(&[]));
        cutoff_table.extend(radii(
            tables_cutoffs::by_dataset(symbol, functional).unwrap_or(&[]),
        ));
        let semicore_subshells = elements::semicore_subshells(symbol).to_vec();
        let recipe = Self {
            symbol: symbol.to_string(),
            atomic_number: z,
            family: Family::PawLcao,
            functional: functional.to_string(),
            relativity,
            cutoffs: Cutoffs::Rule,
            cutoff_table,
            rc_factor: elements::RC_FACTOR,
            local_radius: None,
            local_factor: elements::LOCAL_FACTOR,
            local_shift: repairs::local_shift(symbol, functional)
                .or_else(|| elements::local_shift(symbol))
                .unwrap_or(0.),
            q_cut: elements::Q_CUT,
            energy_offset: repairs::energy_offset(symbol, functional)
                .or_else(|| elements::energy_offset(symbol))
                .unwrap_or(elements::ENERGY_OFFSET),
            second_reference: radii(elements::second_references(symbol, functional)),
            n_bessel: elements::n_bessel(symbol, functional)
                .or_else(|| repairs::s_channel(symbol, functional).map(|e| e.0))
                .unwrap_or(elements::N_BESSEL),
            norm_deficit: repairs::norm_deficit(symbol, functional)
                .or_else(|| elements::norm_deficit(symbol))
                .unwrap_or(elements::NORM_DEFICIT),
            points: elements::generation_points(z),
            r_max: elements::R_MAX,
            nlcc: Nlcc::SmoothCore,
            extra_l: 0,
            highest_channel: elements::highest_channel(symbol),
            frozen_subshells: elements::frozen_subshells(symbol).to_vec(),
            semicore_subshells,
            reference_configuration: elements::reference_configuration(symbol).to_vec(),
            core_radius: elements::core_radius(symbol),
        };
        Ok(recipe)
    }

    /// The UPAW-LCAO recipe: the PAW-LCAO one at norm deficit zero, with the family's own
    /// per-dataset options laid over the defaults.
    pub fn upaw(symbol: &str, functional: &str, relativity: Treatment) -> Result<Self> {
        let mut recipe = Self::for_dataset(symbol, functional, relativity)?;
        let own = tables_upaw::recipe(symbol, functional);
        recipe.family = Family::UpawLcao;
        recipe.norm_deficit = 0.;
        if let Some(extra_l) = own.extra_l {
            recipe.extra_l = extra_l;
        }
        if let Some(shift) = own.local_shift {
            recipe.local_shift = shift;
        }
        if let Some(offset) = own.energy_offset {
            recipe.energy_offset = offset;
        }
        if let Some(count) = own.n_bessel {
            recipe.n_bessel = count;
        }
        if let Some(semicore) = own.semicore_subshells {
            recipe.semicore_subshells = semicore.to_vec();
        }
        if let Some(table) = own.radii {
            recipe.cutoffs = Cutoffs::PerChannel(radii(table));
        }
        if let Some(configuration) = own.reference_configuration {
            recipe.reference_configuration = configuration.to_vec();
        }
        Ok(recipe)
    }

    /// Refuse a recipe the generator cannot honor.
    pub fn validate(&self) -> Result<()> {
        let positive = |name: &str, value: f64| -> Result<()> {
            if value.is_finite() && value > 0. {
                Ok(())
            } else {
                Err(invalid(format!("{name} must be a positive number")))
            }
        };
        if !matches!(self.functional.as_str(), "lda" | "pbe") {
            return Err(invalid(
                "PAW-LCAO generation supports the functionals lda and pbe",
            ));
        }
        positive("q_cut", self.q_cut)?;
        positive("rc_factor", self.rc_factor)?;
        positive("local_factor", self.local_factor)?;
        positive("r_max", self.r_max)?;
        if !self.energy_offset.is_finite() || !self.local_shift.is_finite() {
            return Err(invalid("energy_offset and local_shift must be finite"));
        }
        if self.points < 16 || self.n_bessel == 0 {
            return Err(invalid(
                "a PAW reference needs a radial grid and Bessel functions",
            ));
        }
        if !(0. ..1.).contains(&self.norm_deficit) {
            return Err(invalid("norm_deficit must lie in [0, 1)"));
        }
        if self.family == Family::UpawLcao && self.norm_deficit != 0. {
            return Err(invalid(
                "a UPAW-LCAO dataset is defined by norm_deficit = 0; a nonzero value is a PAW-LCAO dataset",
            ));
        }
        match &self.cutoffs {
            Cutoffs::Uniform(radius) => positive("r_cut", *radius)?,
            Cutoffs::PerChannel(table) => {
                for radius in table.values() {
                    positive("r_cut", *radius)?;
                }
            }
            Cutoffs::Rule => {}
        }
        if let Some(radius) = self.local_radius {
            positive("r_cut_local", radius)?;
        }
        if let Nlcc::Radius(radius) = self.nlcc {
            positive("nlcc radius", radius)?;
        }
        Ok(())
    }
}
