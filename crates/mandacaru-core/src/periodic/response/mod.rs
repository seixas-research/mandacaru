//! The force and the stress of a converged crystal, from the crystal itself (CF-7, AS-11).
//!
//! The engines ([`gradient`](super::gradient), [`stress_terms`](super::stress_terms)) differentiate
//! the discretized free energy of the SCF crystal given the Rust setup of the same structure. This
//! module builds that setup from what the snapshot carries: the `setup` block
//! ([`Description`](super::description::Description): the radial tables and the function and
//! projector lists), the `sites` block's atom centers and dataset files (read here, checked by
//! SHA-256), and the crystal's own grid, k-points and channel blocks. Nothing is taken from Python
//! at run time. A snapshot without the block raises, naming it; the energy never needed it.
//!
//! Units are atomic: [`forces`] returns `-dF/dtau` (Hartree/Bohr) per atom, [`stress`] the engine's
//! [`Stress`] (`sigma = (g + g^T) / (2 Omega)` in Hartree/Bohr^3, `g_ab = dF/d eps_ab`). A
//! state that is not converged is refused: the force is first order in the SCF residual.
//!
//! What is refused, with its cause (the engines' own words): meta-GGA and rVV10 functionals,
//! float32 data, DFT+U, a mesh reduced by symmetry in a snapshot without its operations, a spin-orbit
//! or hybrid crystal on a reduced mesh, a hybrid without its PAW spheres, projectors in the dual
//! basis and, for the stress, a spin-polarized crystal.
use super::{
    description::Description,
    gradient::{gradient, scf, Bands, Breakdown, State},
    invalid,
    point_group::PointGroup,
    setup::provider::SetupDerivatives,
    stress_terms::{
        crystal::{crystal_strain, crystal_stress, SpinOrbitInputs},
        stress as stress_of, Stress,
    },
    Crystal, PeriodicResult,
};
use crate::{math::storage::MatrixSource, product, Error, Result};

mod setup;
use setup::Setup;

fn description(c: &Crystal) -> Result<&Description> {
    if let Some(cause) = &c.description_unavailable {
        return Err(Error::Unsupported(format!(
            "crystal forces and stress are not available for this crystal: {cause}"
        )));
    }
    c.description.as_ref().ok_or_else(|| {
        Error::Unsupported(
            "this crystal snapshot has no 'setup' block (the radial tables of its basis, its \
             function and projector lists and its symmetry operations), which forces and stress \
             are rebuilt from; rebuild and save the crystal with this version (its energy does \
             not need the block)"
                .into(),
        )
    })
}

/// The converged bands as the engine's state: all bands of every `(spin, k)`, occupations per
/// spin channel. Refuses a state that did not converge.
fn state(c: &Crystal, r: &PeriodicResult) -> Result<State> {
    if !r.converged {
        return Err(Error::Unsupported(
            "crystal forces and stress need a converged SCF: the derivative is first order in \
             the residual of an unconverged state"
                .into(),
        ));
    }
    let nk = c.points.len();
    if r.coefficients.len() != c.spins * nk {
        return Err(invalid("the result does not belong to this crystal"));
    }
    let degeneracy = c.degeneracy();
    let bands = (0..c.spins)
        .map(|s| {
            (0..nk)
                .map(|k| {
                    Ok(Bands {
                        coefficients: (*r.coefficients[s * nk + k].matrix(&c.context)?).clone(),
                        eigenvalues: r.eigenvalues[s * nk + k].clone(),
                        occupations: r.occupations[s * nk + k]
                            .iter()
                            .map(|f| f / degeneracy)
                            .collect(),
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(State { bands })
}

/// The bytes the engine's copy of the bands takes outside the `Context`'s buffers.
fn reserve_state(c: &Crystal) -> Result<crate::Reservation> {
    c.context.reserve(product(&[
        c.spins,
        c.points.len(),
        c.bands(),
        c.bands(),
        16,
    ])?)
}

/// The operations of the crystal's reduced mesh, `None` for a whole mesh.
fn own_group(c: &Crystal) -> Option<PointGroup> {
    c.description.as_ref().and_then(|d| d.point_group.clone())
}

/// `-dF/dtau` per atom (Hartree/Bohr) of the converged crystal `r`, the engines' sum of every term
/// averaged over the symmetry operations of a reduced mesh.
pub fn forces(c: &Crystal, r: &PeriodicResult) -> Result<Vec<[f64; 3]>> {
    forces_with_group(c, r, own_group(c))
}

/// [`forces`] with the averaging over `group` instead of the crystal's own: the controls of the
/// symmetrizer (a wrong atom map, no average) call it; a caller of the engine does not. `None`
/// runs the k-sum of a reduced mesh UNAVERAGED, a symmetry-broken number: hidden from the docs like
/// the other test seams, and [`forces`] never takes it.
#[doc(hidden)]
pub fn forces_with_group(
    c: &Crystal,
    r: &PeriodicResult,
    group: Option<PointGroup>,
) -> Result<Vec<[f64; 3]>> {
    Ok(breakdown_with_group(c, r, group)?.forces())
}

fn breakdown_with_group(
    c: &Crystal,
    r: &PeriodicResult,
    group: Option<PointGroup>,
) -> Result<Breakdown> {
    scf::refuse(c)?;
    let _held = reserve_state(c)?;
    let setup = Setup::new(c)?;
    let cores = setup.cores()?;
    let (so_projectors, so_couplings) = setup.spin_orbit(c)?;
    let mut provider = SetupDerivatives::new(
        &c.context,
        setup.inputs(c, &cores),
        setup.layout(c),
        setup.operators(),
    )?
    .with_scf(c, r.options.clone())?;
    if c.spinor {
        provider = provider.with_spin_orbit(&so_projectors, &so_couplings)?;
    }
    provider.set_point_group(group);
    gradient(&provider, &state(c, r)?, c.hubbard_hook())
}

/// The per-term [`Breakdown`] (`dF/dtau`) behind [`forces`], for the tests that check a term is
/// not vacuous. A caller of the engine uses [`forces`].
#[doc(hidden)]
pub fn breakdown(c: &Crystal, r: &PeriodicResult) -> Result<Breakdown> {
    breakdown_with_group(c, r, own_group(c))
}

/// The analytic stress of the converged crystal `r` at the pins of its snapshot (node counts, ion
/// width, basis): `g_ab = dF/d eps_ab` and `sigma = (g + g^T) / (2 Omega)`, averaged over the
/// symmetry operations of a reduced mesh.
pub fn stress(c: &Crystal, r: &PeriodicResult) -> Result<Stress> {
    stress_with_group(c, r, own_group(c))
}

/// [`stress`] with the averaging over `group` instead of the crystal's own (see
/// [`forces_with_group`]); a hybrid takes none. Hidden for the same reason.
#[doc(hidden)]
pub fn stress_with_group(
    c: &Crystal,
    r: &PeriodicResult,
    group: Option<PointGroup>,
) -> Result<Stress> {
    crate::periodic::stress_terms::crystal::refuse(c)?;
    let _held = reserve_state(c)?;
    let setup = Setup::new(c)?;
    let cores = setup.cores()?;
    let (so_projectors, so_couplings) = setup.spin_orbit(c)?;
    let spin_orbit = c.spinor.then_some(SpinOrbitInputs {
        projectors: &so_projectors,
        couplings: &so_couplings,
    });
    let state = state(c, r)?;
    let (inputs, layout, operators) = (setup.inputs(c, &cores), setup.layout(c), setup.operators());
    if c.screened_exchange()?.is_some() {
        if group.is_some() {
            return Err(Error::Unsupported(
                "the analytic crystal stress is not available for a hybrid crystal on a k mesh \
                 reduced by space-group symmetry (use symmetry=False)"
                    .into(),
            ));
        }
        return crystal_stress(
            &c.context, inputs, layout, operators, c, &r.options, &state, spin_orbit,
        );
    }
    let mut strain = crystal_strain(
        &c.context, inputs, layout, operators, c, &r.options, &state, spin_orbit,
    )?;
    strain.strain.set_point_group(group);
    stress_of(&strain.strain, &strain.frozen)
}
