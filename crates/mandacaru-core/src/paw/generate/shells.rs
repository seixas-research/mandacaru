//! The subshells of the reference atom split into the valence the dataset pseudizes and the frozen
//! core, after the recipe's frozen and semicore moves, and the radii a recipe takes from its
//! table. Atomic units.
use super::{
    recipe::Recipe,
    tables_elements::{Subshell, SEMICORE_CUTOFF_FRACTION},
};
use crate::{
    models::{atom::Atom, configuration::valence_subshells},
    paw::invalid,
    Result,
};
use std::collections::BTreeMap;

/// A configuration as the atom solver takes it.
type Occupations = Vec<(usize, usize, f64)>;

/// The subshells of the reference atom split into the valence the dataset pseudizes and the
/// frozen core, after the recipe's frozen and semicore moves.
#[derive(Debug, Clone)]
pub struct Shells {
    pub valence: Vec<(Subshell, f64)>,
    /// In the order the core density is summed: the atom's own core, then the frozen shells.
    pub core: Vec<(Subshell, f64)>,
    pub frozen: Vec<Subshell>,
    pub semicore: Vec<Subshell>,
    /// Electrons of the valence.
    pub charge: f64,
}

fn occupancy(shells: &[(Subshell, f64)], key: Subshell) -> Option<usize> {
    shells.iter().position(|s| s.0 == key)
}

/// Split the atom's subshells and apply the frozen and semicore moves.
pub fn partition(atom: &Atom, recipe: &Recipe) -> Result<Shells> {
    let states = atom
        .occupations
        .iter()
        .filter(|s| s.2 > 0.)
        .map(|s| ((s.0, s.1), s.2))
        .collect::<Vec<_>>();
    let subshells = atom
        .occupations
        .iter()
        .filter(|s| s.2 > 0.)
        .copied()
        .collect::<Occupations>();
    let wanted = valence_subshells(&subshells);
    let mut valence = states
        .iter()
        .filter(|s| wanted.contains(&s.0))
        .copied()
        .collect::<Vec<_>>();
    let mut core = states
        .iter()
        .filter(|s| !wanted.contains(&s.0))
        .copied()
        .collect::<Vec<_>>();

    let mut frozen = Vec::new();
    for &key in &recipe.frozen_subshells {
        if occupancy(&valence, key).is_none() {
            return Err(invalid(format!(
                "{key:?} is not an occupied valence subshell"
            )));
        }
        if frozen.contains(&key) {
            return Err(invalid(format!("duplicate frozen subshell {key:?}")));
        }
        frozen.push(key);
    }
    if !frozen.is_empty() && frozen.len() == valence.len() {
        return Err(invalid(
            "at least one occupied valence subshell must remain",
        ));
    }
    frozen.sort_unstable();
    for &key in &frozen {
        let at = occupancy(&valence, key).expect("checked above");
        core.push(valence.remove(at));
    }

    let mut semicore = Vec::new();
    for &key in &recipe.semicore_subshells {
        if occupancy(&core, key).is_none() {
            return Err(invalid(format!("{key:?} is not an occupied core subshell")));
        }
        if semicore.contains(&key) {
            return Err(invalid(format!("duplicate semicore subshell {key:?}")));
        }
        let same_l = valence
            .iter()
            .filter(|s| s.0 .1 == key.1)
            .map(|s| s.0 .0)
            .collect::<Vec<_>>();
        if !same_l.is_empty() && same_l != [key.0 + 1] {
            return Err(invalid(format!(
                "semicore {key:?} is not directly below the valence subshell of the same l"
            )));
        }
        semicore.push(key);
    }
    semicore.sort_unstable();
    for &key in &semicore {
        let at = occupancy(&core, key).expect("checked above");
        valence.push(core.remove(at));
    }
    if valence.is_empty() {
        return Err(invalid(format!(
            "{} has no valence subshells to pseudize",
            recipe.symbol
        )));
    }
    let charge = valence.iter().map(|s| s.1).sum();
    Ok(Shells {
        valence,
        core,
        frozen,
        semicore,
        charge,
    })
}

/// The `u = r R` orbital of a subshell of the atom.
pub fn orbital<'a>(atom: &'a Atom, key: Subshell) -> Result<&'a [f64]> {
    atom.occupations
        .iter()
        .position(|s| (s.0, s.1) == key)
        .map(|i| atom.orbitals[i].as_slice())
        .ok_or_else(|| invalid(format!("the reference atom has no {key:?} subshell")))
}

/// The radii the channels take from the recipe: its table, with the radius of every channel
/// holding a semicore state defaulting to a fraction of the valence s orbital's peak (the
/// default rule would take it from the compact semicore shell).
pub fn cutoff_table(atom: &Atom, recipe: &Recipe, shells: &Shells) -> Result<BTreeMap<usize, f64>> {
    let mut table = recipe.cutoff_table.clone();
    if !shells.semicore.is_empty() {
        let n_valence = shells
            .valence
            .iter()
            .filter(|s| s.0 .1 == 0)
            .map(|s| s.0 .0)
            .max()
            .ok_or_else(|| invalid("a semicore dataset needs a valence s subshell"))?;
        let u = orbital(atom, (n_valence, 0))?;
        let mut peak = 0;
        for (i, v) in u.iter().enumerate() {
            if v.abs() > u[peak].abs() {
                peak = i;
            }
        }
        let radius = atom.r.as_slice()[peak];
        for &(_, l) in &shells.semicore {
            table.entry(l).or_insert(SEMICORE_CUTOFF_FRACTION * radius);
        }
    }
    Ok(table)
}
