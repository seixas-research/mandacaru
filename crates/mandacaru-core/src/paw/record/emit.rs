//! Record to payload text: the JSON stored under `mandacaru.pseudopotential`.
//!
//! The text is the key order of Python's `to_payload` (`format`, `version`,
//! `family`, then the scalars, `spin_orbit`, `channels`); floats are written
//! with Rust's shortest round-trip form, integers as integers, so the parsed
//! JSON equals what Python wrote. The spin-orbit projectors are radial tables
//! stored on the same stride and digits as every other table.
//!
//! The text is produced in two passes over one writer: a counting pass sizes
//! the buffer, the budget is reserved for exactly that, and a second pass
//! fills it. A value that JSON cannot hold (NaN, an infinity) raises.

use std::collections::BTreeMap;
use std::fmt::Write;

use super::digits::round_digits;
use super::json::fail;
use super::{Array, Channel, PawDataset, SpinOrbit, FORMAT_TAG, FORMAT_VERSION};
use crate::{Context, Error, Reservation, Result};

struct Counter(usize);

impl Write for Counter {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0 += s.len();
        Ok(())
    }
}

/// The payload text of `ds` at `stride`, with the reservation that pays for it.
pub(crate) fn payload_text(
    ctx: &Context,
    ds: &PawDataset,
    stride: usize,
) -> Result<(String, Reservation)> {
    text_of(ctx, ds, stride, true)
}

/// The payload text of `ds` as it is in memory: stride 1 and the spin-orbit projectors
/// at full precision (the file rule of ten digits applies to what is stored, not to a record
/// handed to a caller).
pub(crate) fn payload_text_exact(
    ctx: &Context,
    ds: &PawDataset,
) -> Result<(String, Reservation)> {
    text_of(ctx, ds, 1, false)
}

fn text_of(
    ctx: &Context,
    ds: &PawDataset,
    stride: usize,
    round: bool,
) -> Result<(String, Reservation)> {
    let mut counter = Counter(0);
    payload(&mut counter, ds, stride, round)?;
    let memory = ctx.reserve(counter.0)?;
    let mut text = String::new();
    text.try_reserve_exact(counter.0)
        .map_err(|_| Error::Allocation)?;
    payload(&mut text, ds, stride, round)?;
    debug_assert_eq!(text.len(), counter.0);
    Ok((text, memory))
}

/// The text of a plain JSON dataset file: the payload with every radial table inline under
/// `radial_tables`, at `stride` and ten digits (the file rule).
pub(crate) fn json_file_text(
    ctx: &Context,
    ds: &PawDataset,
    stride: usize,
) -> Result<(String, Reservation)> {
    let (head, _head_memory) = payload_text(ctx, ds, stride)?;
    let mut counter = Counter(0);
    tables_text(&mut counter, ds, stride)?;
    let total = head
        .len()
        .checked_add(counter.0)
        .and_then(|n| n.checked_add(24))
        .ok_or(Error::Overflow)?;
    let memory = ctx.reserve(total)?;
    let mut text = String::new();
    text.try_reserve_exact(total)
        .map_err(|_| Error::Allocation)?;
    text.push_str(&head[..head.len() - 1]);
    text.push_str(",\"radial_tables\":");
    tables_text(&mut text, ds, stride)?;
    text.push('}');
    Ok((text, memory))
}

fn tables_text<W: Write>(w: &mut W, ds: &PawDataset, stride: usize) -> Result<()> {
    w.write_char('{').map_err(bad)?;
    for (i, (name, values)) in ds.tables().into_iter().enumerate() {
        if i > 0 {
            w.write_char(',').map_err(bad)?;
        }
        key(w, &name)?;
        list(
            w,
            values.iter().skip(stride - 1).step_by(stride),
            |w, v| num(w, round_digits(*v), &name),
        )?;
    }
    w.write_char('}').map_err(bad)
}

fn bad(_: std::fmt::Error) -> Error {
    Error::Allocation
}

fn num<W: Write>(w: &mut W, v: f64, what: &str) -> Result<()> {
    if !v.is_finite() {
        return Err(fail(
            what,
            format!("{v} cannot be stored in the JSON payload"),
        ));
    }
    write!(w, "{v:?}").map_err(bad)
}

fn json_str<W: Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_char('"').map_err(bad)?;
    for c in s.chars() {
        match c {
            '"' => w.write_str("\\\""),
            '\\' => w.write_str("\\\\"),
            c if (c as u32) < 0x20 => write!(w, "\\u{:04x}", c as u32),
            c => w.write_char(c),
        }
        .map_err(bad)?;
    }
    w.write_char('"').map_err(bad)
}

fn key<W: Write>(w: &mut W, name: &str) -> Result<()> {
    json_str(w, name)?;
    w.write_char(':').map_err(bad)
}

fn list<W: Write, T>(
    w: &mut W,
    items: impl IntoIterator<Item = T>,
    mut each: impl FnMut(&mut W, T) -> Result<()>,
) -> Result<()> {
    w.write_char('[').map_err(bad)?;
    for (i, item) in items.into_iter().enumerate() {
        if i > 0 {
            w.write_char(',').map_err(bad)?;
        }
        each(w, item)?;
    }
    w.write_char(']').map_err(bad)
}

fn object<W: Write, K: std::fmt::Display, V>(
    w: &mut W,
    items: &BTreeMap<K, V>,
    mut each: impl FnMut(&mut W, &V) -> Result<()>,
) -> Result<()> {
    w.write_char('{').map_err(bad)?;
    for (i, (k, v)) in items.iter().enumerate() {
        if i > 0 {
            w.write_char(',').map_err(bad)?;
        }
        key(w, &k.to_string())?;
        each(w, v)?;
    }
    w.write_char('}').map_err(bad)
}

fn floats<W: Write>(w: &mut W, values: &[f64], what: &str) -> Result<()> {
    list(w, values, |w, v| num(w, *v, what))
}

fn array<W: Write>(w: &mut W, a: &Array, what: &str) -> Result<()> {
    fn level<W: Write>(w: &mut W, shape: &[usize], data: &[f64], what: &str) -> Result<()> {
        match shape {
            [] => Ok(()),
            [_] => floats(w, data, what),
            [rows, rest @ ..] => {
                let stride = rest.iter().product::<usize>();
                list(w, 0..*rows, |w, i| {
                    level(w, rest, &data[i * stride..(i + 1) * stride], what)
                })
            }
        }
    }
    level(w, a.shape(), a.data(), what)
}

fn field_f64<W: Write>(w: &mut W, name: &str, v: f64) -> Result<()> {
    key(w, name)?;
    num(w, v, name)?;
    w.write_char(',').map_err(bad)
}

fn field_array<W: Write>(w: &mut W, name: &str, a: &Array, last: bool) -> Result<()> {
    key(w, name)?;
    array(w, a, name)?;
    if !last {
        w.write_char(',').map_err(bad)?;
    }
    Ok(())
}

fn channel<W: Write>(w: &mut W, c: &Channel, l: u32) -> Result<()> {
    let what = format!("channels.{l}");
    write!(w, "{{\"n\":{},", c.n).map_err(bad)?;
    field_f64(w, "r_cut", c.r_cut)?;
    field_f64(w, "occupation", c.occupation)?;
    field_array(w, "occupations", &c.occupations, false)?;
    field_array(w, "reference_energies", &c.reference_energies, false)?;
    for (name, items) in [
        ("wavevectors", &c.wavevectors),
        ("wave_coefficients", &c.wave_coefficients),
    ] {
        key(w, name)?;
        list(w, items, |w, a| array(w, a, &what))?;
        w.write_char(',').map_err(bad)?;
    }
    for (name, a) in [
        ("coupling", &c.coupling),
        ("coupling_screened", &c.coupling_screened),
        ("overlap_correction", &c.overlap_correction),
        ("norm_correction", &c.norm_correction),
        ("kinetic_difference", &c.kinetic_difference),
        ("potential_difference", &c.potential_difference),
        ("vanderbilt", &c.vanderbilt),
    ] {
        field_array(w, name, a, false)?;
    }
    field_f64(w, "duality_error", c.duality_error)?;
    field_f64(w, "overlap_minimum", c.overlap_minimum)?;
    field_f64(w, "asymmetry", c.asymmetry)?;
    field_f64(w, "consistency_error", c.consistency_error)?;
    field_array(w, "residual_kinetic", &c.residual_kinetic, false)?;
    key(w, "q_cut")?;
    num(w, c.q_cut, "q_cut")?;
    w.write_char('}').map_err(bad)
}

fn spin_orbit<W: Write>(
    w: &mut W,
    so: &SpinOrbit,
    l: u32,
    stride: usize,
    round: bool,
) -> Result<()> {
    w.write_str("{\"projectors\":").map_err(bad)?;
    let what = format!("spin_orbit.{l}.projectors");
    list(w, &so.projectors, |w, p| {
        list(
            w,
            p.as_slice().iter().skip(stride - 1).step_by(stride),
            |w, v| num(w, if round { round_digits(*v) } else { *v }, &what),
        )
    })?;
    w.write_char(',').map_err(bad)?;
    for (name, a) in [
        ("average", &so.average),
        ("average_screened", &so.average_screened),
        ("coupling", &so.coupling),
        ("coupling_screened", &so.coupling_screened),
        ("overlap_average", &so.overlap_average),
        ("overlap_coupling", &so.overlap_coupling),
        ("reference_energies", &so.reference_energies),
    ] {
        field_array(w, name, a, false)?;
    }
    field_array(w, "level_errors", &so.level_errors, true)?;
    w.write_char('}').map_err(bad)
}

fn payload<W: Write>(w: &mut W, ds: &PawDataset, stride: usize, round: bool) -> Result<()> {
    write!(
        w,
        "{{\"format\":\"{FORMAT_TAG}\",\"version\":{FORMAT_VERSION},\"family\":\"{}\",",
        ds.family.name()
    )
    .map_err(bad)?;
    key(w, "symbol")?;
    json_str(w, &ds.symbol)?;
    write!(w, ",\"atomic_number\":{},", ds.atomic_number).map_err(bad)?;
    for (name, v) in [
        ("valence_charge", ds.valence_charge),
        ("r_cut_local", ds.r_cut_local),
        ("local_shift", ds.local_shift),
        ("compensation_radius", ds.compensation_radius),
        ("compensation_charge", ds.compensation_charge),
        ("hartree_screening", ds.hartree_screening),
        ("one_center_energy", ds.one_center_energy),
    ] {
        field_f64(w, name, v)?;
    }
    key(w, "energies")?;
    {
        w.write_char('{').map_err(bad)?;
        for (i, (k, v)) in ds.energies.iter().enumerate() {
            if i > 0 {
                w.write_char(',').map_err(bad)?;
            }
            key(w, k)?;
            num(w, *v, "energies")?;
        }
        w.write_str("},").map_err(bad)?;
    }
    field_f64(w, "q_cut", ds.q_cut)?;
    field_f64(w, "energy_offset", ds.energy_offset)?;
    field_f64(w, "norm_deficit", ds.norm_deficit)?;
    for (name, v) in [("xc", &ds.xc), ("relativity", &ds.relativity)] {
        key(w, name)?;
        json_str(w, v)?;
        w.write_char(',').map_err(bad)?;
    }
    write!(
        w,
        "\"relativistic_exchange\":{},\"extra_l\":{},",
        ds.relativistic_exchange, ds.extra_l
    )
    .map_err(bad)?;
    nlcc(w, ds)?;
    for (name, rows) in [
        ("frozen_subshells", &ds.frozen_subshells),
        ("semicore_subshells", &ds.semicore_subshells),
    ] {
        key(w, name)?;
        list(w, rows, |w, row| {
            list(w, row, |w, v| write!(w, "{v}").map_err(bad))
        })?;
        w.write_char(',').map_err(bad)?;
    }
    defects(w, ds)?;
    key(w, "spin_orbit")?;
    w.write_char('{').map_err(bad)?;
    for (i, (l, so)) in ds.spin_orbit.iter().enumerate() {
        if i > 0 {
            w.write_char(',').map_err(bad)?;
        }
        key(w, &l.to_string())?;
        spin_orbit(w, so, *l, stride, round)?;
    }
    w.write_str("},\"channels\":{").map_err(bad)?;
    for (i, (l, c)) in ds.channels.iter().enumerate() {
        if i > 0 {
            w.write_char(',').map_err(bad)?;
        }
        key(w, &l.to_string())?;
        channel(w, c, *l)?;
    }
    w.write_str("}}").map_err(bad)
}

fn nlcc<W: Write>(w: &mut W, ds: &PawDataset) -> Result<()> {
    let n = &ds.nlcc;
    write!(w, "\"nlcc\":{{\"applied\":{},\"r_nlcc\":", n.applied).map_err(bad)?;
    match n.r_nlcc {
        Some(v) => num(w, v, "nlcc.r_nlcc")?,
        None => w.write_str("null").map_err(bad)?,
    }
    if let Some(v) = &n.source {
        w.write_str(",\"source\":").map_err(bad)?;
        json_str(w, v)?;
    }
    for (name, v) in [
        ("core_electrons", n.core_electrons),
        ("partial_core_electrons", n.partial_core_electrons),
    ] {
        if let Some(v) = v {
            write!(w, ",\"{name}\":").map_err(bad)?;
            num(w, v, "nlcc")?;
        }
    }
    if let Some(v) = &n.reason {
        w.write_str(",\"reason\":").map_err(bad)?;
        json_str(w, v)?;
    }
    w.write_str("},").map_err(bad)
}

fn defects<W: Write>(w: &mut W, ds: &PawDataset) -> Result<()> {
    let d = &ds.defects;
    w.write_str("\"defects\":{\"ghosts\":").map_err(bad)?;
    object(w, &d.ghosts, |w, v| num(w, *v, "defects.ghosts"))?;
    w.write_str(",\"phases\":").map_err(bad)?;
    object(w, &d.phases, |w, v| floats(w, v, "defects.phases"))?;
    if !d.residuals.is_empty() {
        w.write_str(",\"residuals\":").map_err(bad)?;
        object(w, &d.residuals, |w, v| num(w, *v, "defects.residuals"))?;
    }
    if let Some(v) = d.s_miss {
        w.write_str(",\"s_miss\":").map_err(bad)?;
        num(w, v, "defects.s_miss")?;
    }
    if !d.unprojected.is_empty() {
        w.write_str(",\"unprojected\":").map_err(bad)?;
        object(w, &d.unprojected, |w, v| num(w, *v, "defects.unprojected"))?;
    }
    if !d.missing.is_empty() {
        w.write_str(",\"missing\":").map_err(bad)?;
        object(w, &d.missing, |w, v| num(w, *v, "defects.missing"))?;
    }
    if !d.branch_ghosts.is_empty() {
        w.write_str(",\"branch_ghosts\":").map_err(bad)?;
        object(w, &d.branch_ghosts, |w, v| {
            num(w, *v, "defects.branch_ghosts")
        })?;
    }
    w.write_str("},").map_err(bad)
}
