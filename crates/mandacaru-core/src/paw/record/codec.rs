//! Payload to record: the JSON tree plus the radial tables become a typed
//! [`PawDataset`]. The inverse is `emit.rs`.
//!
//! The reader is strict. A key the schema does not name, a missing key, a null
//! where a number belongs, a ragged array or a table off the radial grid
//! raises, naming the path. The only defaults are the two the shipped library
//! needs (a census of all 431 files found `semicore_subshells` missing from 11
//! and `occupations` from 21 of 1,176 channel entries); every other absence
//! raises. Each default is listed in `PawDataset::defaults_applied`.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use super::json::*;
use super::parts::{defects, nlcc, spin_orbit, subshells};
use super::{Array, Channel, Family, PawDataset, FORMAT_TAG, FORMAT_VERSION};
use crate::{Buffer, Context, Result};

/// Radial tables by column name; `assemble` removes what it uses.
pub(crate) type Tables = BTreeMap<String, Buffer<f64>>;

/// Memory reserved for the strings, maps, shapes and headers of a record, all
/// of which the caps below bound (a hostile file reaches the caps, not this).
pub(crate) const SMALL_BYTES: usize = 1 << 20;
const MAX_PROJECTORS: usize = 64;

fn grab(tables: &mut Tables, name: &str) -> Result<Buffer<f64>> {
    tables
        .remove(name)
        .ok_or_else(|| fail("radial_tables", format!("the table {name:?} is missing")))
}

fn channel(
    ctx: &Context,
    l: u32,
    value: Value,
    tables: &mut Tables,
    defaults: &mut Vec<String>,
) -> Result<Channel> {
    let what = format!("channels.{l}");
    let mut map = object(value, &what)?;
    let (mut ae_waves, mut pseudo_waves, mut projectors, mut raw_projectors) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    while tables.contains_key(&format!("pseudo_wave_l{l}_{}", pseudo_waves.len())) {
        let i = pseudo_waves.len();
        if i >= MAX_PROJECTORS {
            return Err(fail(
                &what,
                format!("more than {MAX_PROJECTORS} projectors"),
            ));
        }
        ae_waves.push(grab(tables, &format!("ae_wave_l{l}_{i}"))?);
        pseudo_waves.push(grab(tables, &format!("pseudo_wave_l{l}_{i}"))?);
        projectors.push(grab(tables, &format!("projector_l{l}_{i}"))?);
        raw_projectors.push(grab(tables, &format!("raw_projector_l{l}_{i}"))?);
    }
    let p = pseudo_waves.len();
    if p == 0 {
        return Err(fail(&what, "the channel has no radial tables"));
    }
    let n = integer(&mut map, "n", &what)?;
    let r_cut = float(&mut map, "r_cut", &what)?;
    let occupation = float(&mut map, "occupation", &what)?;
    let reference_energies = {
        let v = take(&mut map, "reference_energies", &what)?;
        array(ctx, &v, &format!("{what}.reference_energies"))?
    };
    if reference_energies.shape != [p] {
        return Err(fail(
            &what,
            format!(
                "{p} projectors but reference_energies has shape {:?}",
                reference_energies.shape
            ),
        ));
    }
    let occupations = match map.remove("occupations") {
        Some(v) => array(ctx, &v, &format!("{what}.occupations"))?,
        None => {
            // A file written before semicore references: one bound reference.
            defaults.push(format!("channels.{l}.occupations"));
            let mut filled = vec![0.0; p];
            filled[0] = occupation;
            array(ctx, &Value::from(filled), &format!("{what}.occupations"))?
        }
    };
    let wavevectors = arrays(
        ctx,
        take(&mut map, "wavevectors", &what)?,
        &format!("{what}.wavevectors"),
        p,
    )?;
    let wave_coefficients = arrays(
        ctx,
        take(&mut map, "wave_coefficients", &what)?,
        &format!("{what}.wave_coefficients"),
        p,
    )?;
    let mut matrix = |key: &str| -> Result<Array> {
        let v = take(&mut map, key, &what)?;
        let a = array(ctx, &v, &format!("{what}.{key}"))?;
        square(&a, p, &format!("{what}.{key}"))?;
        Ok(a)
    };
    let coupling = matrix("coupling")?;
    let coupling_screened = matrix("coupling_screened")?;
    let overlap_correction = matrix("overlap_correction")?;
    // The norm correction is read as stored: a file without it is refused
    // (`take` raises), there is no recovery from the coupling.
    let norm_correction = matrix("norm_correction")?;
    let kinetic_difference = matrix("kinetic_difference")?;
    let potential_difference = matrix("potential_difference")?;
    let vanderbilt = matrix("vanderbilt")?;
    let duality_error = float(&mut map, "duality_error", &what)?;
    let overlap_minimum = float(&mut map, "overlap_minimum", &what)?;
    let asymmetry = float(&mut map, "asymmetry", &what)?;
    let consistency_error = float(&mut map, "consistency_error", &what)?;
    let residual_kinetic = {
        let v = take(&mut map, "residual_kinetic", &what)?;
        array(ctx, &v, &format!("{what}.residual_kinetic"))?
    };
    let q_cut = float(&mut map, "q_cut", &what)?;
    nothing_left(&map, &what)?;
    Ok(Channel {
        n,
        r_cut,
        occupation,
        occupations,
        reference_energies,
        wavevectors,
        wave_coefficients,
        coupling,
        coupling_screened,
        overlap_correction,
        norm_correction,
        kinetic_difference,
        potential_difference,
        vanderbilt,
        duality_error,
        overlap_minimum,
        asymmetry,
        consistency_error,
        residual_kinetic,
        q_cut,
        ae_waves,
        pseudo_waves,
        projectors,
        raw_projectors,
    })
}

// ---------------------------------------------------------------------------
// The record.
// ---------------------------------------------------------------------------

/// Builds a record from the scalar payload (without `radial_tables`) and the
/// radial tables. Consumes both; whatever neither side names raises.
pub(crate) fn assemble(
    ctx: &Context,
    mut payload: Map<String, Value>,
    mut tables: Tables,
) -> Result<PawDataset> {
    let what = "payload";
    let small = ctx.reserve(SMALL_BYTES)?;
    match take(&mut payload, "format", what)? {
        Value::String(tag) if tag == FORMAT_TAG => {}
        other => return Err(fail(what, format!("format {other} is not {FORMAT_TAG:?}"))),
    }
    let version = take(&mut payload, "version", what)?;
    match version.as_u64() {
        Some(v) if (1..=FORMAT_VERSION).contains(&v) => {}
        Some(v) if v > FORMAT_VERSION => {
            return Err(fail(
                what,
                format!(
                "pseudopotential format version {v} is newer than this build ({FORMAT_VERSION})"
            ),
            ))
        }
        _ => {
            return Err(fail(
                what,
                format!("version {version} is not a format version"),
            ))
        }
    }
    let family = Family::parse(&text(&mut payload, "family", what)?)?;
    let symbol = text(&mut payload, "symbol", what)?;
    let atomic_number = integer(&mut payload, "atomic_number", what)?;
    let valence_charge = float(&mut payload, "valence_charge", what)?;
    let r_cut_local = float(&mut payload, "r_cut_local", what)?;
    let local_shift = float(&mut payload, "local_shift", what)?;
    let compensation_radius = float(&mut payload, "compensation_radius", what)?;
    let compensation_charge = float(&mut payload, "compensation_charge", what)?;
    let hartree_screening = float(&mut payload, "hartree_screening", what)?;
    let one_center_energy = float(&mut payload, "one_center_energy", what)?;
    let energies = {
        let map = object(take(&mut payload, "energies", what)?, "energies")?;
        if map.len() > MAX_ENTRIES {
            return Err(fail("energies", format!("more than {MAX_ENTRIES} entries")));
        }
        map.into_iter()
            .map(|(k, v)| {
                let value = number(&v, &format!("energies.{k}"))?;
                Ok((string(Value::String(k), "energies key")?, value))
            })
            .collect::<Result<BTreeMap<_, _>>>()?
    };
    let q_cut = float(&mut payload, "q_cut", what)?;
    let energy_offset = float(&mut payload, "energy_offset", what)?;
    let norm_deficit = float(&mut payload, "norm_deficit", what)?;
    let xc = text(&mut payload, "xc", what)?;
    let relativity = text(&mut payload, "relativity", what)?;
    let relativistic_exchange = boolean(&mut payload, "relativistic_exchange", what)?;
    let extra_l = integer(&mut payload, "extra_l", what)?;
    let nlcc = nlcc(take(&mut payload, "nlcc", what)?)?;
    let frozen_subshells = subshells(
        take(&mut payload, "frozen_subshells", what)?,
        "frozen_subshells",
    )?;
    let mut defaults_applied = Vec::new();
    let semicore_subshells = match payload.remove("semicore_subshells") {
        Some(v) => subshells(v, "semicore_subshells")?,
        None => {
            // A file written before semicore shells existed has none.
            defaults_applied.push("semicore_subshells".to_string());
            Vec::new()
        }
    };
    let defects = defects(take(&mut payload, "defects", what)?)?;
    let spin_orbit_value = take(&mut payload, "spin_orbit", what)?;
    let channels_value = take(&mut payload, "channels", what)?;
    nothing_left(&payload, what)?;

    let r = grab(&mut tables, "r")?;
    let grid = r.as_slice();
    if grid.is_empty()
        || !(grid[0] > 0.0 && grid[0].is_finite())
        || grid.windows(2).any(|w| !(w[1] > w[0] && w[1].is_finite()))
    {
        return Err(fail(
            "radial_tables",
            "r is not a strictly increasing grid of positive radii",
        ));
    }
    let points = grid.len();
    let mut table = |name: &str| -> Result<Buffer<f64>> { grab(&mut tables, name) };
    let (v_local, v_local_screened) = (table("v_local")?, table("v_local_screened")?);
    let (valence_density, core_density) = (table("valence_density")?, table("core_density")?);
    let smooth_core_density = table("smooth_core_density")?;

    let mut spin_orbit_map = BTreeMap::new();
    for (l, entry) in by_l(spin_orbit_value, "spin_orbit")? {
        spin_orbit_map.insert(l, spin_orbit(ctx, entry, points, &symbol, l)?);
    }
    let mut channel_map = BTreeMap::new();
    for (l, entry) in by_l(channels_value, "channels")? {
        channel_map.insert(
            l,
            channel(ctx, l, entry, &mut tables, &mut defaults_applied)?,
        );
    }
    if let Some(name) = tables.keys().next() {
        return Err(fail(
            "radial_tables",
            format!("the table {name:?} belongs to no channel"),
        ));
    }
    let dataset = PawDataset {
        family,
        symbol,
        atomic_number,
        valence_charge,
        r_cut_local,
        local_shift,
        compensation_radius,
        compensation_charge,
        hartree_screening,
        one_center_energy,
        energies,
        q_cut,
        energy_offset,
        norm_deficit,
        xc,
        relativity,
        relativistic_exchange,
        extra_l,
        nlcc,
        frozen_subshells,
        semicore_subshells,
        defects,
        spin_orbit: spin_orbit_map,
        channels: channel_map,
        r,
        v_local,
        v_local_screened,
        valence_density,
        core_density,
        smooth_core_density,
        defaults_applied,
        _small: small,
    };
    for (name, values) in dataset.tables() {
        if values.len() != points {
            return Err(fail(
                "radial_tables",
                format!("{name} has {} rows, r has {points}", values.len()),
            ));
        }
    }
    Ok(dataset)
}
