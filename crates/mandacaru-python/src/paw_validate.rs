//! Bindings of the PAW dataset validation entry (`mandacaru_core::paw::validate`): every per-file
//! gate on one record, returned as data. No threshold and no comparison lives here; the rows come
//! back as the core made them.
use super::{error, paw_record, PyContext};
use mandacaru_core::paw::{
    record::PawDataset,
    validate::{validate, GateReport, Options, GATES},
};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use std::path::Path;

/// A gate report as `{"symbol", "xc", "relativity", "passed", "rows", "gates"}`: `rows` is a list
/// of `{"gate", "check", "value", "threshold", "relation", "status", "detail"}` (`value` and
/// `threshold` are `None` when the row has none; `relation` is `"<="`, `"<"`, `">="` or `""`) and
/// `gates` maps each gate to `{"name", "status"}`, its worst row.
pub(crate) fn report_dict<'py>(
    py: Python<'py>,
    report: &GateReport,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("symbol", &report.symbol)?;
    out.set_item("xc", &report.xc)?;
    out.set_item("relativity", &report.relativity)?;
    out.set_item("passed", report.passed())?;
    let rows = PyList::empty(py);
    for row in &report.rows {
        let item = PyDict::new(py);
        item.set_item("gate", row.gate)?;
        item.set_item("check", &row.check)?;
        item.set_item("value", row.value)?;
        item.set_item("threshold", row.threshold)?;
        item.set_item("relation", row.relation.symbol())?;
        item.set_item("status", row.status.name())?;
        item.set_item("detail", &row.detail)?;
        rows.append(item)?;
    }
    out.set_item("rows", rows)?;
    let gates = PyDict::new(py);
    for (gate, name) in GATES {
        let item = PyDict::new(py);
        item.set_item("name", name)?;
        item.set_item("status", report.gate_status(gate).name())?;
        gates.set_item(gate, item)?;
    }
    out.set_item("gates", gates)?;
    Ok(out)
}

fn run<'py>(
    py: Python<'py>,
    ctx: &mandacaru_core::Context,
    dataset: &PawDataset,
    twin: Option<&PawDataset>,
    bond: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let report = py
        .detach(|| {
            validate(
                ctx,
                dataset,
                &Options {
                    twin,
                    bond,
                    ..Options::default()
                },
            )
        })
        .map_err(error)?;
    report_dict(py, &report)
}

/// Every per-file gate (G1-G11 and G13) of a record `(payload, tables)`, against the all-electron
/// atom solved again from it. `twin` is the `(payload, tables)` of the same element's other set,
/// whose core/valence partition must equal this one's; `bond` (Bohr) replaces the element's own
/// bond to hydrogen in the sphere gate. G1 is not applicable to a record read alone.
#[pyfunction]
#[pyo3(signature = (context, payload, tables, *, twin = None, bond = None))]
fn paw_validate<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    payload: &str,
    tables: &Bound<'py, PyDict>,
    twin: Option<(String, Bound<'py, PyDict>)>,
    bond: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = &context.inner;
    let dataset = paw_record::import(ctx, payload, tables)?;
    let other = match &twin {
        Some((p, t)) => Some(paw_record::import(ctx, p, t)?),
        None => None,
    };
    run(py, ctx, &dataset, other.as_ref(), bond)
}

/// [`paw_validate`] on the dataset file at `path` (and the twin file at `twin`): the tables never
/// cross into Python.
#[pyfunction]
#[pyo3(signature = (context, path, *, twin = None, bond = None))]
fn paw_validate_file<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    path: &str,
    twin: Option<&str>,
    bond: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = &context.inner;
    let read = |path: &str| {
        py.detach(|| PawDataset::read(ctx, Path::new(path)))
            .map_err(error)
    };
    let dataset = read(path)?;
    let other = match twin {
        Some(path) => Some(read(path)?),
        None => None,
    };
    run(py, ctx, &dataset, other.as_ref(), bond)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(paw_validate, m)?)?;
    m.add_function(wrap_pyfunction!(paw_validate_file, m)?)?;
    Ok(())
}
