//! What a caller names of a generation, resolved against the policy tables into a [`Recipe`].
//!
//! The tables of the dataset give every choice a default ([`Recipe::for_dataset`],
//! [`Recipe::upaw`]); a caller overrides single fields by name. Whatever the caller names is
//! also *pinned* for the search: a remedy of [`super::search`] must not change a choice the
//! caller made, so a defect there is refused instead of repaired. A caller cannot be told from
//! the table when it passes the table's own value, so the pin follows the naming, not the
//! value ([`Overrides::pinned`]).
//!
//! Nothing here selects another calculation: the functional and the treatment are parsed
//! (an unknown one raises), the family chooses the recipe constructor, and
//! [`Recipe::validate`] refuses an inconsistent request when the generator starts.
use super::{
    recipe::{Cutoffs, Nlcc, Recipe},
    search::Pinned,
    tables_elements::Subshell,
};
use crate::{
    models::relativity::Treatment,
    paw::{invalid, record::Family},
    Result,
};
use std::collections::BTreeMap;

/// The fields of a [`Recipe`] a caller may assign; `None` keeps the table's value.
#[derive(Debug, Clone, Default)]
pub struct Overrides {
    /// One radius or one per channel (anything but the recipe's rule).
    pub r_cut: Option<Cutoffs>,
    /// Radii laid over the policy table the recipe's rule consults. Unlike `r_cut` they are not
    /// pinned: the search may still move them (a balanced sphere, a shorter s sphere) the way it
    /// moves a tabulated radius. A candidate recipe states its spheres this way without giving
    /// up the ghost remedies.
    pub cutoff_table: Option<BTreeMap<usize, f64>>,
    pub rc_factor: Option<f64>,
    pub r_cut_local: Option<f64>,
    pub local_factor: Option<f64>,
    pub local_shift: Option<f64>,
    pub q_cut: Option<f64>,
    pub energy_offset: Option<f64>,
    pub second_reference: Option<BTreeMap<usize, f64>>,
    pub n_bessel: Option<usize>,
    pub norm_deficit: Option<f64>,
    pub points: Option<usize>,
    pub r_max: Option<f64>,
    pub nlcc: Option<Nlcc>,
    pub extra_l: Option<usize>,
    pub frozen_subshells: Option<Vec<Subshell>>,
    pub semicore_subshells: Option<Vec<Subshell>>,
    pub reference_configuration: Option<Vec<(Subshell, f64)>>,
}

impl Overrides {
    /// The choices the caller named, which the search must not change.
    pub fn pinned(&self) -> Pinned {
        Pinned {
            r_cut: self.r_cut.is_some(),
            r_cut_local: self.r_cut_local.is_some(),
            local_shift: self.local_shift.is_some(),
        }
    }

    fn apply(self, recipe: &mut Recipe) {
        if let Some(v) = self.r_cut {
            recipe.cutoffs = v;
        }
        if let Some(table) = self.cutoff_table {
            recipe.cutoff_table.extend(table);
        }
        macro_rules! set {
            ($($field:ident),*) => {$(
                if let Some(v) = self.$field {
                    recipe.$field = v;
                }
            )*};
        }
        set!(
            rc_factor,
            local_factor,
            local_shift,
            q_cut,
            energy_offset,
            second_reference,
            n_bessel,
            norm_deficit,
            points,
            r_max,
            nlcc,
            extra_l,
            frozen_subshells,
            semicore_subshells,
            reference_configuration
        );
        if let Some(radius) = self.r_cut_local {
            recipe.local_radius = Some(radius);
        }
    }
}

/// The recipe of `symbol` in `family`, `functional` and `relativity`, with `overrides` laid over
/// the policy tables, and the choices the search must leave alone: those the caller named, and
/// those the family's own table lays under the caller ([`Pinned::of`]).
pub fn resolve(
    symbol: &str,
    family: Family,
    functional: &str,
    relativity: &str,
    overrides: Overrides,
) -> Result<(Recipe, Pinned)> {
    let treatment = Treatment::parse(relativity)?;
    if !matches!(functional, "lda" | "pbe") {
        return Err(invalid(format!(
            "PAW-LCAO generation supports the functionals lda and pbe, not {functional:?}"
        )));
    }
    let mut recipe = match family {
        Family::PawLcao => Recipe::for_dataset(symbol, functional, treatment)?,
        Family::UpawLcao => Recipe::upaw(symbol, functional, treatment)?,
    };
    let named = overrides.pinned();
    overrides.apply(&mut recipe);
    let tabulated = Pinned::of(&recipe);
    Ok((
        recipe,
        Pinned {
            r_cut: named.r_cut || tabulated.r_cut,
            r_cut_local: named.r_cut_local || tabulated.r_cut_local,
            local_shift: named.local_shift || tabulated.local_shift,
        },
    ))
}
