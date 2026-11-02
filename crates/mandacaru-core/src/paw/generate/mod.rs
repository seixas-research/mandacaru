//! Native PAW-LCAO / UPAW-LCAO dataset generation (Bloechl, PRB 50, 17953) in its frozen-core,
//! one-center form with the one-center energies linearized at the reference atom. Atomic units.
//!
//! The construction, per species:
//!
//! 1. **Reference atom** ([`reference`]): the self-consistent spherical atom, split into the
//!    frozen core and the valence the recipe asks for; per channel the occupied bound states
//!    and a second scattering reference ([`reference::reference_waves`]).
//! 2. **Smooth partial waves** ([`waves`]): Bessel expansions matched to the all-electron waves
//!    in value and three derivatives at the cutoff, no norm conserved (the norm deficit becomes
//!    the overlap correction `q`).
//! 3. **Local potential** ([`waves::screened_local_potential`]): the even-polynomial
//!    continuation of the all-electron potential inside the local radius.
//! 4. **Projectors and one-center matrices** ([`assemble`]): dual projectors, `q`, `Delta T`,
//!    `Delta V` and the screened coupling.
//! 5. **Densities, unscreening and the frozen one-center constant** ([`core`], [`finish`],
//!    [`energies`]).
//!
//! The entry points are [`generate`] and the seam [`generate_with`]: the first is the production
//! path and is wired to the variational radial exchange-correlation scheme (the scheme of the
//! molecular PAW path); the second takes the scheme as a parameter so a test can reproduce the
//! numbers of the atom-solver scheme the shipped datasets were produced with. There is no
//! user-visible selector. A [`recipe::Recipe`] is fully resolved from the policy tables of the
//! dataset ([`tables_cutoffs`], [`tables_elements`], [`tables_repairs`], [`tables_upaw`]).
//!
//! This is the first construction of the ghost search: the result is returned unexamined. The
//! search that repairs a construction binding a ghost state or scattering wrongly and the spectrum
//! and scattering diagnostics are separate pieces. A Dirac recipe adds the j-resolved spin-orbit
//! term ([`spin_orbit`]) to the scalar construction.
pub mod assemble;
pub mod core;
pub mod energies;
mod finish;
pub mod recipe;
pub mod reference;
pub mod shells;
pub mod spin_orbit;
pub mod tables_cutoffs;
pub mod tables_elements;
pub mod tables_repairs;
pub mod tables_upaw;
pub mod waves;

use self::{
    assemble::{assemble, Atomic, Partial, PawChannel},
    energies::{Variational, XcSource},
    finish::{finish, Parts},
    recipe::Recipe,
    reference::{check_atom, solve_reference},
    shells::{cutoff_table, partition},
    waves::{pseudize_channels, screened_local_potential},
};
use crate::{
    models::{atom::Atom, relativity::Treatment},
    paw::pseudize::snap,
    paw::record::PawDataset,
    product, Context, Result,
};
use std::collections::BTreeMap;

/// The result of a generation: the dataset record and what the generator noted on the way.
#[derive(Debug)]
pub struct Generated {
    pub dataset: PawDataset,
    /// Observations that are not numerics (a fallback cutoff beyond the usual range).
    pub notes: Vec<String>,
    /// The first reference energy the empty channels ended on (`None`: the highest occupied
    /// level, or no empty channel); the Dirac branches reuse it.
    pub anchor: Option<f64>,
}

/// Generate the dataset of `recipe` with the variational radial exchange-correlation scheme.
pub fn generate(ctx: &Context, recipe: &Recipe) -> Result<Generated> {
    generate_with(ctx, recipe, &Variational)
}

#[doc(hidden)]
/// [`generate`] with the radial exchange-correlation scheme as a parameter. The production path
/// passes [`Variational`]; [`energies::AtomSolver`] reproduces the numbers of the shipped
/// datasets and exists for the parity tests.
pub fn generate_with(ctx: &Context, recipe: &Recipe, source: &dyn XcSource) -> Result<Generated> {
    let atom = solve_reference(ctx, recipe)?;
    generate_with_atom(ctx, recipe, &atom, source)
}

#[doc(hidden)]
/// [`generate_with`] on a reference atom already solved with the recipe's functional and
/// treatment (a search shares one atom across its constructions).
pub fn generate_with_atom(
    ctx: &Context,
    recipe: &Recipe,
    atom: &Atom,
    source: &dyn XcSource,
) -> Result<Generated> {
    recipe.validate()?;
    check_atom(atom, recipe)?;
    let shells = partition(atom, recipe)?;
    let r = atom.r.as_slice();
    let n = r.len();
    // The working vectors (references, densities, potentials) that are not drawn from the
    // context one by one: a bound on what they hold at once.
    let _workspace = ctx.reserve(product(&[n, 8, 48])?)?;

    // A frozen shell above the highest remaining valence l keeps a channel of its own, empty and
    // scattering at a positive energy; some elements always carry a few empty channels.
    let highest = shells.valence.iter().map(|s| s.0 .1).max().unwrap_or(0);
    let frozen_top = shells.frozen.iter().map(|s| s.1).max().unwrap_or(highest);
    let extra_l = recipe.extra_l.max(frozen_top.saturating_sub(highest)).max(
        recipe
            .highest_channel
            .unwrap_or(highest)
            .saturating_sub(highest),
    );

    let table = cutoff_table(atom, recipe, &shells)?;
    let pseudized = pseudize_channels(ctx, recipe, atom, &shells, &table, extra_l)?;
    let radii = pseudized.channels.values().map(|c| c.r_cut);
    // The local potential follows the largest cutoff: a compact channel's projector reaches out
    // to the local radius instead.
    let r_local = snap(
        r,
        recipe.local_radius.unwrap_or_else(|| {
            recipe.local_factor * radii.clone().fold(f64::NEG_INFINITY, f64::max)
        }),
    );
    let r_g = radii.fold(f64::INFINITY, f64::min);
    let r_core = recipe.core_radius.unwrap_or(r_g);
    let v_loc = screened_local_potential(ctx, recipe, atom, r_local)?;

    let mut channels = BTreeMap::<usize, PawChannel>::new();
    for (&l, reference) in &pseudized.channels {
        channels.insert(
            l,
            assemble(
                ctx,
                &Atomic {
                    r,
                    v_ae: atom.effective.as_slice(),
                    v_loc: v_loc.as_slice(),
                    r_local,
                },
                &Partial {
                    l,
                    n: reference.n(),
                    energies: &reference.energies,
                    waves: &reference.waves,
                    smooth: &pseudized.smooth[&l],
                    q_cut: recipe.q_cut,
                    occupations: &reference.occupations(),
                },
                true,
            )?,
        );
    }
    let mut dataset = finish(
        ctx,
        &Parts {
            recipe,
            atom,
            shells: &shells,
            references: &pseudized.channels,
            extra_l,
            v_loc: v_loc.as_slice(),
            r_local,
            r_g,
            r_core,
        },
        channels,
        source,
    )?;
    if recipe.relativity == Treatment::Dirac {
        let radii = pseudized
            .channels
            .iter()
            .map(|(&l, c)| (l, c.r_cut))
            .collect();
        dataset.spin_orbit = spin_orbit::channels(
            ctx,
            &spin_orbit::Inputs {
                recipe,
                atom,
                shells: &shells,
                radii: &radii,
                extra_l,
                anchor: pseudized.anchor,
                v_loc: v_loc.as_slice(),
                r_local,
                hartree_screening: dataset.hartree_screening,
            },
        )?;
    }
    Ok(Generated {
        dataset,
        notes: pseudized.notes,
        anchor: pseudized.anchor,
    })
}
