//! From the assembled channels to the dataset record: the densities of the reference atom, the
//! unscreening of the local potential, the frozen one-center constant and the typed record.
//! Atomic units.
//!
//! Unscreening follows the norm-conserving families: the ionic local potential is the screened
//! one less the Hartree potential of the neutral smooth density and the exchange-correlation
//! potential of the smooth valence density (plus the smooth core when the nonlinear core
//! correction is on), and each coupling loses the Hartree screening of the augmentation,
//! `D^ion = D^scr - q int v_H[n~ + n^] g`. The compensation charge of the spherical reference
//! atom is its monopole `Q g(r)`, which restores neutrality with the ion outside the sphere.
use super::{
    assemble::PawChannel,
    core::{
        augmented_density, core_density, pseudize_density, valence_sums, ChannelWaves, CoreShell,
    },
    energies::{one_center_energies, EnergyDensities, XcSource},
    recipe::{Nlcc as NlccChoice, Recipe},
    reference::ChannelReference,
    shells::{orbital, Shells},
};
use crate::{
    math::radial::integrate,
    models::{atom::Atom, spherical},
    paw::{
        core_correction::partial_core_density,
        invalid,
        pseudize::snap,
        radial::shape,
        record::{Array, Channel, Defects, Nlcc, PawDataset, SpinOrbit},
    },
    product, Buffer, Context, Result,
};
use nalgebra::DMatrix;
use std::{collections::BTreeMap, f64::consts::PI};

/// Everything the generator has settled before the densities are formed.
pub struct Parts<'a> {
    pub recipe: &'a Recipe,
    pub atom: &'a Atom,
    pub shells: &'a Shells,
    pub references: &'a BTreeMap<usize, ChannelReference>,
    pub extra_l: usize,
    /// The screened local potential and its radius.
    pub v_loc: &'a [f64],
    pub r_local: f64,
    /// The compensation radius (the smallest cutoff) and the radius the frozen core is
    /// pseudized at.
    pub r_g: f64,
    pub r_core: f64,
}

/// Crossover ratio of the explicit-radius core correction (`core_correction.py`).
const CROSSOVER_RATIO: f64 = 1.0;

fn array(ctx: &Context, shape: Vec<usize>, values: &[f64]) -> Result<Array> {
    let mut data = ctx.allocate::<f64>(product(&shape)?)?;
    data.as_mut_slice().copy_from_slice(values);
    Ok(Array { shape, data })
}

fn matrix(ctx: &Context, m: &DMatrix<f64>) -> Result<Array> {
    let mut values = Vec::with_capacity(m.nrows() * m.ncols());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            values.push(m[(i, j)]);
        }
    }
    array(ctx, vec![m.nrows(), m.ncols()], &values)
}

fn tables(ctx: &Context, flat: &[f64], count: usize, n: usize) -> Result<Vec<Buffer<f64>>> {
    (0..count)
        .map(|i| ctx.copy_slice(&flat[i * n..(i + 1) * n]))
        .collect()
}

fn trapezoid(r: &[f64], values: &[f64]) -> Result<f64> {
    integrate(r, values)
}

/// The nonlinear core correction: the density to unscreen with and the record of how it was made.
fn core_correction(
    parts: &Parts<'_>,
    core: &[f64],
    smooth_core: &[f64],
    smooth_valence: &[f64],
) -> Result<(Vec<f64>, Nlcc)> {
    let r = parts.atom.r.as_slice();
    let n = r.len();
    let absent = |reason: &str| Nlcc {
        applied: false,
        r_nlcc: None,
        source: None,
        core_electrons: None,
        partial_core_electrons: None,
        reason: Some(reason.to_string()),
    };
    match parts.recipe.nlcc {
        NlccChoice::Off => Ok((vec![0.; n], absent("not requested"))),
        _ if parts.shells.core.is_empty() => {
            Ok((vec![0.; n], absent("the atom has no core to correct for")))
        }
        NlccChoice::SmoothCore => {
            let shell = r.iter().map(|x| 4. * PI * x * x).collect::<Vec<_>>();
            let weighted = |density: &[f64]| {
                let integrand = density
                    .iter()
                    .zip(&shell)
                    .map(|(d, s)| d * s)
                    .collect::<Vec<_>>();
                trapezoid(r, &integrand)
            };
            let record = Nlcc {
                applied: true,
                r_nlcc: Some(snap(r, parts.r_core)),
                source: Some("smooth_core".to_string()),
                core_electrons: Some(weighted(core)?),
                partial_core_electrons: Some(weighted(smooth_core)?),
                reason: None,
            };
            Ok((smooth_core.to_vec(), record))
        }
        NlccChoice::Radius(radius) => {
            let partial =
                partial_core_density(r, core, smooth_valence, Some(radius), CROSSOVER_RATIO)?;
            let record = match partial.r_nlcc {
                None => absent("the core density never exceeds the valence density"),
                Some(at) => Nlcc {
                    applied: true,
                    r_nlcc: Some(at),
                    source: None,
                    core_electrons: Some(partial.core_electrons),
                    partial_core_electrons: Some(partial.partial_core_electrons),
                    reason: None,
                },
            };
            Ok((partial.density, record))
        }
    }
}

/// One record channel from an assembled channel and its unscreened coupling.
fn record_channel(
    ctx: &Context,
    channel: PawChannel,
    reference: &ChannelReference,
    hartree_screening: f64,
    n: usize,
) -> Result<Channel> {
    let count = channel.reference_energies.len();
    let terms = channel.wavevectors.len();
    let coupling = &channel.coupling_screened - &channel.overlap_correction * hartree_screening;
    let wavevectors = (0..count)
        .map(|_| array(ctx, vec![terms], &channel.wavevectors))
        .collect::<Result<Vec<_>>>()?;
    let wave_coefficients = (0..count)
        .map(|i| {
            let row = (0..terms)
                .map(|k| channel.coefficients[(i, k)])
                .collect::<Vec<_>>();
            array(ctx, vec![terms], &row)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Channel {
        n: channel.n as u32,
        r_cut: channel.r_cut,
        occupation: channel.occupation(),
        occupations: array(ctx, vec![count], &channel.occupations)?,
        reference_energies: array(ctx, vec![count], &channel.reference_energies)?,
        wavevectors,
        wave_coefficients,
        coupling: matrix(ctx, &coupling)?,
        coupling_screened: matrix(ctx, &channel.coupling_screened)?,
        overlap_correction: matrix(ctx, &channel.overlap_correction)?,
        norm_correction: matrix(ctx, &channel.norm_correction)?,
        kinetic_difference: matrix(ctx, &channel.kinetic_difference)?,
        potential_difference: matrix(ctx, &channel.potential_difference)?,
        vanderbilt: matrix(ctx, &channel.vanderbilt)?,
        duality_error: channel.duality_error,
        overlap_minimum: channel.overlap_minimum,
        asymmetry: channel.asymmetry,
        consistency_error: channel.consistency_error,
        residual_kinetic: array(ctx, vec![count], &channel.residual_kinetic)?,
        q_cut: channel.q_cut,
        ae_waves: tables(ctx, &reference.waves, count, n)?,
        pseudo_waves: tables(ctx, channel.pseudo_waves.as_slice(), count, n)?,
        projectors: tables(ctx, channel.projectors.as_slice(), count, n)?,
        raw_projectors: tables(ctx, channel.raw_projectors.as_slice(), count, n)?,
    })
}

/// The dataset record of the assembled `channels` of the reference atom in `parts`.
pub fn finish(
    ctx: &Context,
    parts: &Parts<'_>,
    channels: BTreeMap<usize, PawChannel>,
    source: &dyn XcSource,
) -> Result<PawDataset> {
    let Parts {
        recipe,
        atom,
        shells,
        references,
        extra_l,
        v_loc,
        r_local,
        r_g,
        r_core,
    } = *parts;
    let r = atom.r.as_slice();
    let n = r.len();
    let relativistic = recipe.relativity.relativistic_exchange();

    // Densities of the reference atom: the frozen core and its smooth counterpart, the
    // all-electron and smooth valence densities, the compensation charge.
    let core_shells = shells
        .core
        .iter()
        .map(|&(key, occupancy)| {
            Ok(CoreShell {
                occupancy,
                orbital: orbital(atom, key)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let core = core_density(r, &core_shells)?;
    let smooth_core = if shells.core.is_empty() {
        vec![0.; n]
    } else {
        pseudize_density(r, &core, r_core)?
    };
    let waves = channels
        .iter()
        .map(|(l, channel)| ChannelWaves {
            channel,
            ae_waves: &references[l].waves,
        })
        .collect::<Vec<_>>();
    let sums = valence_sums(n, &waves)?;
    let augmented = augmented_density(r, &sums.smooth_valence, sums.compensation_charge, r_g);
    let (xc_core, nlcc) = core_correction(parts, &core, &smooth_core, &sums.smooth_valence)?;

    // Unscreening.
    let v_hartree = spherical::hartree(ctx, r, &augmented)?;
    let xc_density = sums
        .smooth_valence
        .iter()
        .zip(&xc_core)
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    let v_xc = source
        .evaluate(ctx, r, &xc_density, &recipe.functional, relativistic)?
        .potential;
    let mut ionic = ctx.allocate::<f64>(n)?;
    for i in 0..n {
        ionic.as_mut_slice()[i] = v_loc[i] - v_hartree.as_slice()[i] - v_xc[i];
    }
    let screening_integrand = (0..n)
        .map(|i| v_hartree.as_slice()[i] * shape(r[i], r_g) * 4. * PI * r[i] * r[i])
        .collect::<Vec<_>>();
    let hartree_screening = trapezoid(r, &screening_integrand)?;
    drop(v_hartree);

    // The frozen one-center constant: the reference atom evaluated with the molecular machinery
    // must give the all-electron valence energy in the norm-conserving convention.
    let energies = one_center_energies(
        ctx,
        source,
        r,
        &EnergyDensities {
            band: sums.band,
            ae_valence: &sums.ae_valence,
            smooth_valence: &sums.smooth_valence,
            augmented: &augmented,
            core: &core,
            smooth_core: &smooth_core,
        },
        &recipe.functional,
        relativistic,
    )?;

    let mut record_channels = BTreeMap::new();
    for (l, channel) in channels {
        record_channels.insert(
            l as u32,
            record_channel(ctx, channel, &references[&l], hartree_screening, n)?,
        );
    }
    let table = |values: &[f64]| ctx.copy_slice(values);
    if !(sums.compensation_charge.is_finite() && hartree_screening.is_finite()) {
        return Err(invalid(
            "the PAW generator produced a non-finite compensation charge",
        ));
    }
    Ok(PawDataset {
        family: recipe.family,
        symbol: recipe.symbol.clone(),
        atomic_number: recipe.atomic_number as u32,
        valence_charge: shells.charge,
        r_cut_local: r_local,
        local_shift: recipe.local_shift,
        compensation_radius: r_g,
        compensation_charge: sums.compensation_charge,
        hartree_screening,
        one_center_energy: energies.one_center,
        energies: BTreeMap::from([
            ("band".to_string(), energies.band),
            ("reference_valence".to_string(), energies.reference_valence),
            ("pseudo_atom".to_string(), energies.pseudo_atom),
            ("one_center".to_string(), energies.one_center),
            ("hartree_ae".to_string(), energies.hartree_ae),
            ("hartree_ps".to_string(), energies.hartree_ps),
            ("xc_ae".to_string(), energies.xc_ae),
            ("xc_ps".to_string(), energies.xc_ps),
            (
                "core_valence_xc_omitted".to_string(),
                energies.core_valence_xc_omitted,
            ),
            ("atom_total".to_string(), atom.energy),
        ]),
        q_cut: recipe.q_cut,
        energy_offset: recipe.energy_offset,
        norm_deficit: recipe.norm_deficit,
        xc: recipe.functional.clone(),
        relativity: recipe.relativity.name().to_string(),
        relativistic_exchange: relativistic,
        extra_l: extra_l as u32,
        nlcc,
        frozen_subshells: shells
            .frozen
            .iter()
            .map(|s| vec![s.0 as i64, s.1 as i64])
            .collect(),
        semicore_subshells: shells
            .semicore
            .iter()
            .map(|s| vec![s.0 as i64, s.1 as i64])
            .collect(),
        defects: Defects::default(),
        spin_orbit: BTreeMap::<u32, SpinOrbit>::new(),
        channels: record_channels,
        r: table(r)?,
        v_local: ionic,
        v_local_screened: table(v_loc)?,
        valence_density: table(&sums.smooth_valence)?,
        core_density: table(&core)?,
        smooth_core_density: table(&smooth_core)?,
        defaults_applied: Vec::new(),
        _small: ctx.reserve(1 << 20)?,
    })
}
