//! Binding of the PAW-LCAO / UPAW-LCAO generator. The options a caller names are copied into
//! `mandacaru_core::paw::generate::request::Overrides`; the policy tables, the construction and
//! the ghost search run in the core crate. The result is the record in the layout of a dataset
//! file ([`super::paw_record`]) plus what the search noted; a refusal of the search is returned
//! as data, so the exceptions and their wording stay in Python.
use super::{error, paw_record, PyContext};
use mandacaru_core::paw::{
    generate::{
        recipe::{Cutoffs, Nlcc},
        request::{resolve, Overrides},
        search::{search, Error as SearchError, Mode, Options},
        tables_elements::Subshell,
    },
    record::Family,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict},
};
use std::collections::BTreeMap;

fn bad(name: &str, what: &str) -> PyErr {
    PyTypeError::new_err(format!("generation option {name} must be {what}"))
}

fn by_l(name: &str, value: &Bound<'_, PyAny>) -> PyResult<BTreeMap<usize, f64>> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| bad(name, "a dict {l: value}"))?;
    dict.iter()
        .map(|(k, v)| Ok((k.extract()?, v.extract()?)))
        .collect::<PyResult<_>>()
        .map_err(|_: PyErr| bad(name, "a dict {l: value}"))
}

fn subshells(name: &str, value: &Bound<'_, PyAny>) -> PyResult<Vec<Subshell>> {
    value
        .extract::<Vec<(usize, usize)>>()
        .map_err(|_| bad(name, "a list of (n, l) pairs"))
}

fn overrides(options: &Bound<'_, PyDict>) -> PyResult<Overrides> {
    let mut out = Overrides::default();
    for (key, value) in options.iter() {
        let name: String = key.extract()?;
        let float = || value.extract::<f64>().map_err(|_| bad(&name, "a number"));
        let count = || {
            value
                .extract::<usize>()
                .map_err(|_| bad(&name, "a non-negative integer"))
        };
        match name.as_str() {
            "family" | "xc" | "relativity" | "ghosts" => {}
            "r_cut" => {
                out.r_cut = Some(if value.cast::<PyDict>().is_ok() {
                    Cutoffs::PerChannel(by_l(&name, &value)?)
                } else {
                    Cutoffs::Uniform(float()?)
                })
            }
            "cutoff_table" => out.cutoff_table = Some(by_l(&name, &value)?),
            "rc_factor" => out.rc_factor = Some(float()?),
            "r_cut_local" => out.r_cut_local = Some(float()?),
            "local_factor" => out.local_factor = Some(float()?),
            "local_shift" => out.local_shift = Some(float()?),
            "q_cut" => out.q_cut = Some(float()?),
            "energy_offset" => out.energy_offset = Some(float()?),
            "second_reference" => out.second_reference = Some(by_l(&name, &value)?),
            "n_bessel" => out.n_bessel = Some(count()?),
            "norm_deficit" => out.norm_deficit = Some(float()?),
            "points" => out.points = Some(count()?),
            "r_max" => out.r_max = Some(float()?),
            "nlcc" => {
                out.nlcc = Some(if value.is_instance_of::<PyBool>() {
                    if value.extract::<bool>()? {
                        Nlcc::SmoothCore
                    } else {
                        Nlcc::Off
                    }
                } else {
                    Nlcc::Radius(float()?)
                })
            }
            "extra_l" => out.extra_l = Some(count()?),
            "frozen_subshells" => out.frozen_subshells = Some(subshells(&name, &value)?),
            "semicore_subshells" => out.semicore_subshells = Some(subshells(&name, &value)?),
            "reference_configuration" => {
                let dict = value
                    .cast::<PyDict>()
                    .map_err(|_| bad(&name, "a dict {(n, l): occupancy}"))?;
                out.reference_configuration = Some(
                    dict.iter()
                        .map(|(k, v)| Ok((k.extract::<(usize, usize)>()?, v.extract::<f64>()?)))
                        .collect::<PyResult<_>>()
                        .map_err(|_: PyErr| bad(&name, "a dict {(n, l): occupancy}"))?,
                );
            }
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "unknown generation option {name:?}"
                )))
            }
        }
    }
    Ok(out)
}

fn text<'py>(options: &Bound<'py, PyDict>, name: &str) -> PyResult<String> {
    options
        .get_item(name)?
        .ok_or_else(|| PyTypeError::new_err(format!("generation option {name} is missing")))?
        .extract()
        .map_err(|_| bad(name, "a string"))
}

/// Generate a dataset and return it with the outcome of the search.
///
/// `options` carries the family (`"paw-lcao"` / `"upaw-lcao"`), `xc`, `relativity`, `ghosts`
/// (`"repair"`, `"refuse"`, `"keep"`, `"flag"`) and the overrides of the recipe by name.
/// The result holds `payload` and `tables` (the record), `index`, `label`, `tried`, `flagged`,
/// `built`, `notes` and `missing`; when the search refuses it holds `refusal` (the text) and
/// `kind` instead.
#[pyfunction]
fn paw_generate<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    symbol: &str,
    options: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = &context.inner;
    let family = Family::parse(&text(options, "family")?).map_err(error)?;
    let mode = match text(options, "ghosts")?.as_str() {
        "repair" => Mode::Repair,
        "refuse" => Mode::Refuse,
        "keep" => Mode::Keep,
        "flag" => Mode::Flag,
        other => {
            return Err(PyValueError::new_err(format!(
                "ghosts must be one of ('repair', 'refuse', 'keep', 'flag'), not {other:?}"
            )))
        }
    };
    let (recipe, pinned) = resolve(
        symbol,
        family,
        &text(options, "xc")?,
        &text(options, "relativity")?,
        overrides(options)?,
    )
    .map_err(error)?;
    let found = py.detach(|| search(ctx, &recipe, &Options { mode, pinned }));
    let out = PyDict::new(py);
    match found {
        Ok(searched) => {
            let (payload, tables) = paw_record::export(py, ctx, &searched.generated.dataset)?;
            out.set_item("payload", payload)?;
            out.set_item("tables", tables)?;
            out.set_item("index", searched.index)?;
            out.set_item("label", &searched.label)?;
            out.set_item(
                "tried",
                searched
                    .tried
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>(),
            )?;
            out.set_item("flagged", searched.flagged)?;
            out.set_item("built", searched.built)?;
            out.set_item("notes", &searched.generated.notes)?;
            out.set_item(
                "missing",
                searched
                    .diagnosis
                    .as_ref()
                    .map(|d| d.missing.clone())
                    .unwrap_or_default(),
            )?;
        }
        Err(SearchError::Failed(failure)) => return Err(error(failure)),
        Err(refusal) => {
            let kind = if matches!(refusal, SearchError::Refused { .. }) {
                "refused"
            } else {
                "exhausted"
            };
            out.set_item("kind", kind)?;
            out.set_item("refusal", refusal.to_string())?;
        }
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(paw_generate, m)?)?;
    Ok(())
}
