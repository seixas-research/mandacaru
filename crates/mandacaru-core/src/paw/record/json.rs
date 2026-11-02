//! Typed access to the JSON tree of a payload: objects, numbers, strings,
//! rectangular numeric arrays, all with the path in the error and every
//! allocation reserved from the `Context` first.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use super::Array;
use crate::paw::invalid;
use crate::{product, Buffer, Context, Error, Result};

pub(crate) const MAX_TEXT: usize = 256;
pub(crate) const MAX_ENTRIES: usize = 256;
pub(crate) const MAX_L: u32 = 15;
pub(crate) const MAX_DIMS: usize = 4;

pub(crate) fn fail(what: &str, message: impl AsRef<str>) -> Error {
    invalid(format!("paw record: {what}: {}", message.as_ref()))
}

pub(crate) fn object(value: Value, what: &str) -> Result<Map<String, Value>> {
    match value {
        Value::Object(map) => Ok(map),
        other => Err(fail(
            what,
            format!("expected an object, found {}", kind(&other)),
        )),
    }
}

pub(crate) fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

pub(crate) fn take(map: &mut Map<String, Value>, key: &str, what: &str) -> Result<Value> {
    map.remove(key)
        .ok_or_else(|| fail(what, format!("the key {key:?} is missing")))
}

pub(crate) fn nothing_left(map: &Map<String, Value>, what: &str) -> Result<()> {
    match map.keys().next() {
        None => Ok(()),
        Some(key) => Err(fail(
            what,
            format!("the key {key:?} is not part of the schema"),
        )),
    }
}

pub(crate) fn number(value: &Value, what: &str) -> Result<f64> {
    match value {
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| fail(what, "not representable as a double")),
        other => Err(fail(
            what,
            format!("expected a number, found {}", kind(other)),
        )),
    }
}

pub(crate) fn float(map: &mut Map<String, Value>, key: &str, what: &str) -> Result<f64> {
    let value = take(map, key, what)?;
    number(&value, &format!("{what}.{key}"))
}

pub(crate) fn integer(map: &mut Map<String, Value>, key: &str, what: &str) -> Result<u32> {
    let value = take(map, key, what)?;
    value
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| {
            fail(
                &format!("{what}.{key}"),
                format!("expected a non-negative integer, found {value}"),
            )
        })
}

pub(crate) fn text(map: &mut Map<String, Value>, key: &str, what: &str) -> Result<String> {
    string(take(map, key, what)?, &format!("{what}.{key}"))
}

pub(crate) fn string(value: Value, what: &str) -> Result<String> {
    match value {
        Value::String(s) if s.len() <= MAX_TEXT => Ok(s),
        Value::String(_) => Err(fail(what, format!("longer than {MAX_TEXT} bytes"))),
        other => Err(fail(
            what,
            format!("expected a string, found {}", kind(&other)),
        )),
    }
}

pub(crate) fn boolean(map: &mut Map<String, Value>, key: &str, what: &str) -> Result<bool> {
    match take(map, key, what)? {
        Value::Bool(b) => Ok(b),
        other => Err(fail(
            &format!("{what}.{key}"),
            format!("expected a boolean, found {}", kind(&other)),
        )),
    }
}

pub(crate) fn angular_momentum(key: &str, what: &str) -> Result<u32> {
    key.parse::<u32>()
        .ok()
        .filter(|l| *l <= MAX_L)
        .ok_or_else(|| {
            fail(
                what,
                format!("{key:?} is not an angular momentum in 0..={MAX_L}"),
            )
        })
}

/// `{ "<l>": value }` with `l` an angular momentum.
pub(crate) fn by_l(value: Value, what: &str) -> Result<BTreeMap<u32, Value>> {
    let map = object(value, what)?;
    if map.len() > MAX_ENTRIES {
        return Err(fail(what, format!("more than {MAX_ENTRIES} entries")));
    }
    map.into_iter()
        .map(|(key, v)| Ok((angular_momentum(&key, what)?, v)))
        .collect()
}

pub(crate) fn float_map(value: Value, what: &str) -> Result<BTreeMap<u32, f64>> {
    by_l(value, what)?
        .into_iter()
        .map(|(l, v)| Ok((l, number(&v, &format!("{what}.{l}"))?)))
        .collect()
}

// ---------------------------------------------------------------------------
// Arrays.
// ---------------------------------------------------------------------------

pub(crate) fn shape_of(value: &Value, what: &str) -> Result<Vec<usize>> {
    let mut shape = Vec::new();
    let mut at = value;
    while let Value::Array(items) = at {
        shape.push(items.len());
        if shape.len() > MAX_DIMS {
            return Err(fail(what, format!("more than {MAX_DIMS} dimensions")));
        }
        match items.first() {
            Some(first @ Value::Array(_)) => at = first,
            _ => break,
        }
    }
    if shape.is_empty() {
        return Err(fail(
            what,
            format!("expected an array, found {}", kind(value)),
        ));
    }
    Ok(shape)
}

pub(crate) fn fill(
    value: &Value,
    depth: usize,
    shape: &[usize],
    out: &mut [f64],
    at: &mut usize,
    what: &str,
) -> Result<()> {
    let Value::Array(items) = value else {
        return Err(fail(what, "ragged array"));
    };
    if items.len() != shape[depth] {
        return Err(fail(
            what,
            format!(
                "ragged array: {} items where {} were expected",
                items.len(),
                shape[depth]
            ),
        ));
    }
    for item in items {
        if depth + 1 == shape.len() {
            out[*at] = number(item, what)?;
            *at += 1;
        } else {
            fill(item, depth + 1, shape, out, at, what)?;
        }
    }
    Ok(())
}

/// A rectangular numeric array, its memory reserved before it is allocated.
pub(crate) fn array(ctx: &Context, value: &Value, what: &str) -> Result<Array> {
    let shape = shape_of(value, what)?;
    let total = product(&shape)?;
    let mut data = ctx.allocate::<f64>(total)?;
    let mut at = 0usize;
    fill(value, 0, &shape, data.as_mut_slice(), &mut at, what)?;
    Ok(Array { shape, data })
}

pub(crate) fn arrays(
    ctx: &Context,
    value: Value,
    what: &str,
    expected: usize,
) -> Result<Vec<Array>> {
    let Value::Array(items) = value else {
        return Err(fail(
            what,
            format!("expected an array, found {}", kind(&value)),
        ));
    };
    if items.len() != expected {
        return Err(fail(
            what,
            format!("{} entries for {expected} projectors", items.len()),
        ));
    }
    items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let a = array(ctx, item, &format!("{what}[{i}]"))?;
            if a.shape.len() != 1 {
                return Err(fail(what, "each entry is a vector"));
            }
            Ok(a)
        })
        .collect()
}

pub(crate) fn square(a: &Array, side: usize, what: &str) -> Result<()> {
    if a.shape == [side, side] {
        Ok(())
    } else {
        Err(fail(
            what,
            format!("shape {:?} where [{side}, {side}] was expected", a.shape),
        ))
    }
}

/// A flat list of radial values; JSON files may carry `"inf"`/`"nan"` strings
/// for non-finite entries.
pub(crate) fn table_from_json(ctx: &Context, value: &Value, what: &str) -> Result<Buffer<f64>> {
    let Value::Array(items) = value else {
        return Err(fail(
            what,
            format!("expected an array, found {}", kind(value)),
        ));
    };
    let mut out = ctx.allocate::<f64>(items.len())?;
    for (slot, item) in out.as_mut_slice().iter_mut().zip(items) {
        *slot = match item {
            Value::String(s) if s == "inf" => f64::INFINITY,
            Value::String(s) if s == "-inf" => f64::NEG_INFINITY,
            Value::String(s) if s == "nan" => f64::NAN,
            other => number(other, what)?,
        };
    }
    Ok(out)
}
