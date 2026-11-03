//! The small sub-records of a payload: the nonlinear core correction, the
//! defects, the subshell lists and the j-resolved spin-orbit entries.

use std::collections::BTreeMap;

use serde_json::Value;

use super::json::*;
use super::{Array, Defects, Nlcc, SpinOrbit};
use crate::{Context, Result};

const MAX_SPIN_ORBIT_PROJECTORS: usize = 64;

// ---------------------------------------------------------------------------
// Sub-records.
// ---------------------------------------------------------------------------

pub(crate) fn nlcc(value: Value) -> Result<Nlcc> {
    let what = "nlcc";
    let mut map = object(value, what)?;
    let applied = boolean(&mut map, "applied", what)?;
    let r_nlcc = match take(&mut map, "r_nlcc", what)? {
        Value::Null => None,
        other => Some(number(&other, "nlcc.r_nlcc")?),
    };
    let mut optional_number = |key: &str| -> Result<Option<f64>> {
        map.remove(key)
            .map(|v| number(&v, &format!("nlcc.{key}")))
            .transpose()
    };
    let core_electrons = optional_number("core_electrons")?;
    let partial_core_electrons = optional_number("partial_core_electrons")?;
    let source = map
        .remove("source")
        .map(|v| string(v, "nlcc.source"))
        .transpose()?;
    let reason = map
        .remove("reason")
        .map(|v| string(v, "nlcc.reason"))
        .transpose()?;
    nothing_left(&map, what)?;
    Ok(Nlcc {
        applied,
        r_nlcc,
        source,
        core_electrons,
        partial_core_electrons,
        reason,
    })
}

/// `{"l:kappa": number}`: the keys are validated, not kept as free text.
fn branch_map(value: Value) -> Result<BTreeMap<String, f64>> {
    let what = "defects.branch_ghosts";
    let map = object(value, what)?;
    if map.len() > MAX_ENTRIES {
        return Err(fail(what, format!("more than {MAX_ENTRIES} entries")));
    }
    map.into_iter()
        .map(|(key, v)| {
            let valid = key.split_once(':').is_some_and(|(l, kappa)| {
                l.parse::<u8>().is_ok_and(|l| l <= 8) && kappa.parse::<i8>().is_ok_and(|k| k != 0)
            });
            if !valid {
                return Err(fail(what, format!("{key:?} is not an \"l:kappa\" key")));
            }
            Ok((key.clone(), number(&v, &format!("{what}.{key}"))?))
        })
        .collect()
}

pub(crate) fn defects(value: Value) -> Result<Defects> {
    let what = "defects";
    let mut map = object(value, what)?;
    let ghosts = float_map(take(&mut map, "ghosts", what)?, "defects.ghosts")?;
    let mut phases = BTreeMap::new();
    for (l, pair) in by_l(take(&mut map, "phases", what)?, "defects.phases")? {
        match pair.as_array().map(Vec::as_slice) {
            Some([a, b]) => {
                phases.insert(
                    l,
                    [number(a, "defects.phases")?, number(b, "defects.phases")?],
                );
            }
            _ => return Err(fail("defects.phases", "each entry is a pair of numbers")),
        }
    }
    let residuals = map
        .remove("residuals")
        .map(|v| float_map(v, "defects.residuals"))
        .transpose()?;
    let s_miss = map
        .remove("s_miss")
        .map(|v| number(&v, "defects.s_miss"))
        .transpose()?;
    let unprojected = map
        .remove("unprojected")
        .map(|v| float_map(v, "defects.unprojected"))
        .transpose()?;
    let missing = map
        .remove("missing")
        .map(|v| float_map(v, "defects.missing"))
        .transpose()?;
    let branch_ghosts = map.remove("branch_ghosts").map(branch_map).transpose()?;
    nothing_left(&map, what)?;
    Ok(Defects {
        ghosts,
        phases,
        residuals: residuals.unwrap_or_default(),
        s_miss,
        unprojected: unprojected.unwrap_or_default(),
        missing: missing.unwrap_or_default(),
        branch_ghosts: branch_ghosts.unwrap_or_default(),
    })
}

pub(crate) fn subshells(value: Value, what: &str) -> Result<Vec<Vec<i64>>> {
    let Value::Array(rows) = value else {
        return Err(fail(
            what,
            format!("expected an array, found {}", kind(&value)),
        ));
    };
    if rows.len() > MAX_ENTRIES {
        return Err(fail(what, format!("more than {MAX_ENTRIES} entries")));
    }
    rows.iter()
        .map(|row| match row {
            Value::Array(items) if items.len() <= MAX_DIMS * 2 => items
                .iter()
                .map(|v| v.as_i64().ok_or_else(|| fail(what, "entries are integers")))
                .collect(),
            _ => Err(fail(what, "each entry is a short list of integers")),
        })
        .collect()
}

pub(crate) fn spin_orbit(
    ctx: &Context,
    value: Value,
    points: usize,
    symbol: &str,
    l: u32,
) -> Result<SpinOrbit> {
    let what = format!("spin_orbit.{l}");
    let mut map = match value {
        Value::Object(map) => map,
        _ => {
            return Err(fail(
                &what,
                format!(
                    "the {symbol} dataset carries the first-order spin-orbit term (a bare \
                     matrix), which the j-resolved construction replaced: regenerate it with \
                     mandacaru-build --pp PAW --dirac"
                ),
            ))
        }
    };
    let Value::Array(rows) = take(&mut map, "projectors", &what)? else {
        return Err(fail(&what, "projectors is a list of radial tables"));
    };
    if rows.len() > MAX_SPIN_ORBIT_PROJECTORS {
        return Err(fail(
            &what,
            format!("more than {MAX_SPIN_ORBIT_PROJECTORS} projectors"),
        ));
    }
    let mut projectors = Vec::with_capacity(rows.len());
    for (k, row) in rows.iter().enumerate() {
        let table = table_from_json(ctx, row, &format!("{what}.projectors[{k}]"))?;
        if table.as_slice().len() != points {
            return Err(fail(
                &what,
                format!(
                    "the {symbol} dataset's spin-orbit projectors are not on its {points}-point \
                     radial grid: regenerate it with mandacaru-build --pp PAW --dirac"
                ),
            ));
        }
        projectors.push(table);
    }
    let mut get = |key: &str| -> Result<Array> {
        let v = take(&mut map, key, &what)?;
        array(ctx, &v, &format!("{what}.{key}"))
    };
    let so = SpinOrbit {
        projectors,
        average: get("average")?,
        average_screened: get("average_screened")?,
        coupling: get("coupling")?,
        coupling_screened: get("coupling_screened")?,
        overlap_average: get("overlap_average")?,
        overlap_coupling: get("overlap_coupling")?,
        reference_energies: get("reference_energies")?,
        level_errors: get("level_errors")?,
    };
    nothing_left(&map, &what)?;
    Ok(so)
}
