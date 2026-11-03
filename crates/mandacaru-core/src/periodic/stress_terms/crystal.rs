//! The stress of a REAL converged SCF crystal (AS-6): the entry that ties the strain provider to
//! the SCF crystal the way the force engine ties its provider ([`SetupDerivatives::with_scf`]).
//!
//! The state is frozen by the force engine's own [`freeze`]: the channel moments, the potentials
//! and the +U operator of the OUTPUT density of the converged bands, and the bands re-diagonalized
//! with them (the Fermi level over the whole mesh, Mermin occupations, eigenvalues unshifted). The
//! geometry tie of `with_scf` (sizes, k-points and weights, reference, functional, the lattice
//! that generates the crystal's cell, the Gaussian ions that reproduce the crystal's ion table,
//! the sphere rules) raises on any mismatch. What the strain terms read beyond the force fields is
//! formed here from the same density with the crystal's own functions: the reciprocal density
//! `rho_G` (`solver::reciprocal`), the xc energy and, for a GGA, the gradients and the flux of
//! `solver::exchange_correlation` (the arrays the xc potential is made of).
//!
//! A spinor crystal (AS-8) additionally needs the spin-orbit union projectors and couplings of its
//! datasets ([`SpinOrbitInputs`]): both the force provider (whose `with_spin_orbit` ties `h^SO(k)`
//! of these projections and this operator to the matrix the crystal diagonalized with, at every
//! k-point) and the strain provider are built from them. Without them a spinor request raises, it is
//! never differentiated without its spin-orbit term. A hybrid spinor (AS-10) adds the exchange terms of
//! [`hybrid`] on top.
//!
//! The stress is that of the discretized functional at the crystal's pins (node counts, ion
//! width, filter cutoff, alias operations and the dense list of the crystal's own cell): the
//! dense sums run over the ball of the crystal's reference cell, which is the list its `U`,
//! `U_ion` and `E_ion` were summed over.
use super::{check_hybrid_request, check_request, hybrid, stress, stress_hybrid, Stress};
use crate::{
    math::fft::Fft3,
    periodic::{
        gradient::{freeze, Frozen, Layout, State},
        invalid,
        setup::{
            provider::{Inputs, Operators, SetupDerivatives},
            spheres::Projector,
            spin_orbit::Coupling,
            strain_provider::{StrainFields, StrainSetup, XcFields},
        },
        solver, Crystal, Options,
    },
    quantum::C64,
    Context, Error, Reservation, Result,
};
use std::collections::BTreeMap;

/// The spin-orbit union projectors (every atom's, `Projector::atom` the owner) and the radial
/// couplings `D^SO` per `(atom, l)` of a spinor crystal's datasets.
#[derive(Clone, Copy)]
pub struct SpinOrbitInputs<'a> {
    pub projectors: &'a [Projector],
    pub couplings: &'a BTreeMap<(usize, usize), Coupling>,
}

/// The crystals this stress does not differentiate, from the crystal itself: DFT+U (AS-7) and a k
/// mesh reduced by space-group symmetry (the stress symmetrizer is CF-6; build with
/// `symmetry=False`). A hybrid, float32 data, a meta-GGA, rVV10 and `spins != 1` are refused by
/// [`check_request`] from the layout.
pub fn refuse(c: &Crystal) -> Result<()> {
    let no = |what: &str| {
        Err(Error::Unsupported(format!(
            "the analytic crystal stress is not available for {what}"
        )))
    };
    if c.hubbard.as_ref().is_some_and(|h| h.is_active()) {
        return no("a DFT+U crystal (the +U strain wiring is AS-7)");
    }
    if c.screened_exchange()?.is_some() && c.hybrid.is_none() {
        return no("a hybrid crystal without the PAW spheres of its sites block");
    }
    let identity = c.qsym.as_slice().iter().enumerate().all(|(i, v)| {
        let want = if i / c.channels == i % c.channels {
            1.
        } else {
            0.
        };
        (v - C64::new(want, 0.)).norm() < 1e-12
    });
    if !c.permutations.is_empty() || !identity {
        return no("a k mesh reduced by space-group symmetry (use symmetry=False)");
    }
    Ok(())
}

fn copy(o: &Operators) -> Operators {
    Operators {
        blocks: o.blocks.clone(),
        augmentation: o.augmentation.clone(),
        ion_operator: o.ion_operator.clone(),
    }
}

/// `rho_G`, the xc energy, the gradients and the flux of the density matrices of `state`.
fn output_data(
    c: &Crystal,
    state: &State,
) -> Result<(Vec<C64>, f64, Option<Vec<f64>>, Option<Vec<f64>>)> {
    let n = c.grid.index.points();
    let densities = crate::periodic::gradient::scf::densities(c, &state.bands)?;
    let (rho, _) = solver::density(c, &densities)?;
    let mut fft = Fft3::new(&c.context, c.grid.index.shape())?;
    let terms = solver::exchange_correlation(c, &mut fft, rho.as_slice(), None)?;
    let mut total = vec![0.; n];
    for s in 0..c.spins {
        for (t, r) in total.iter_mut().zip(&rho.as_slice()[s * n..(s + 1) * n]) {
            *t += r;
        }
    }
    let density = solver::reciprocal(c, &mut fft, &total)?;
    if density.as_slice().len() != n {
        return Err(invalid(
            "crystal stress: the reciprocal density has one value per node",
        ));
    }
    Ok((
        density.as_slice().to_vec(),
        terms.energy,
        terms.gradients.map(|b| b.as_slice().to_vec()),
        terms.flux.map(|b| b.as_slice().to_vec()),
    ))
}

/// The strain provider of a crystal's geometry with the frozen output-density state of its
/// converged bands: everything [`stress`] contracts. [`crystal_stress`] is `stress` of this; a
/// test that damages one ingredient (the bands, a field) contracts the damaged copy itself.
pub struct CrystalStrain<'a> {
    pub strain: StrainSetup<'a>,
    pub frozen: Frozen<StrainFields>,
    _held: Reservation,
}

/// Ties the geometry to the SCF crystal, freezes the output-density state of `state` and forms
/// the strain data beyond the force fields. Raises when the setup is not the crystal's, and for
/// everything [`check_request`] and [`refuse`] name. `spin_orbit` is required for a spinor crystal
/// and refused for a scalar one.
#[allow(clippy::too_many_arguments)]
pub fn crystal_strain<'a>(
    ctx: &'a Context,
    inputs: Inputs<'a>,
    layout: Layout,
    operators: Operators,
    crystal: &'a Crystal,
    options: &Options,
    state: &State,
    spin_orbit: Option<SpinOrbitInputs<'a>>,
) -> Result<CrystalStrain<'a>> {
    if hybrid::is_hybrid(&layout)? {
        check_hybrid_request(&layout)?;
    } else {
        check_request(&layout)?;
    }
    refuse(crystal)?;
    if layout.request.spinor != spin_orbit.is_some() {
        return Err(Error::Unsupported(if layout.request.spinor {
            "the analytic crystal stress of a spinor crystal needs its spin-orbit projectors \
             and couplings"
                .into()
        } else {
            "spin-orbit projectors belong to a spinor crystal".into()
        }));
    }
    let mut forces = SetupDerivatives::new(ctx, inputs, layout.clone(), copy(&operators))?;
    if let Some(so) = spin_orbit {
        forces = forces.with_spin_orbit(so.projectors, so.couplings)?;
    }
    let forces = forces.with_scf(crystal, options.clone())?;
    let frozen = freeze(&forces, state, None)?;
    drop(forces);
    let n = crystal.grid.index.points();
    // The copies the strain terms keep: the density, the xc gradients and flux (a GGA).
    let held = ctx.reserve(crate::product(&[n, 8 * 18])?)?;
    let (density, energy, gradients, flux) = output_data(crystal, state)?;
    let fields = StrainFields {
        local: frozen.fields.local.clone(),
        moments: frozen.fields.moments.clone(),
        weights: frozen.fields.weights.clone(),
        density,
        xc: XcFields {
            energy,
            potential: frozen.fields.xc.clone(),
            gradients,
            flux,
        },
    };
    let frozen = Frozen {
        state: frozen.state,
        fields,
        q: frozen.q,
        hubbard: None,
    };
    let mut strain = StrainSetup::new(ctx, inputs, layout, operators)?;
    if let Some(so) = spin_orbit {
        strain = strain.with_spin_orbit(so.projectors, so.couplings)?;
    }
    Ok(CrystalStrain {
        strain,
        frozen,
        _held: held,
    })
}

/// The analytic stress of the converged `state` of `crystal`, whose geometry is described by
/// `inputs`, `layout` and `operators` (the Rust setup of the same structure): [`stress`] of
/// [`crystal_strain`].
#[allow(clippy::too_many_arguments)]
pub fn crystal_stress<'a>(
    ctx: &'a Context,
    inputs: Inputs<'a>,
    layout: Layout,
    operators: Operators,
    crystal: &'a Crystal,
    options: &Options,
    state: &State,
    spin_orbit: Option<SpinOrbitInputs<'a>>,
) -> Result<Stress> {
    let hybrid = hybrid::is_hybrid(&layout)?;
    let c = crystal_strain(
        ctx, inputs, layout, operators, crystal, options, state, spin_orbit,
    )?;
    if hybrid {
        let tangents = hybrid::Tangents::new(ctx, &inputs)?;
        let terms = hybrid::exchange_terms(
            crystal,
            &c.frozen.state.bands,
            options.exxdiv_treatment,
            &tangents,
        )?;
        return stress_hybrid(&c.strain, &c.frozen, terms);
    }
    stress(&c.strain, &c.frozen)
}
