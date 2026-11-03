//! Bindings of the PAW dataset record: read and write a dataset file, and hand a record across
//! the boundary in the layout of the file itself (the payload text and the radial tables by
//! column name). All shaping, validation and the storage rule live in
//! `mandacaru_core::paw::record`; this file copies arrays in after a layout check and returns
//! owned arrays.
use super::{array, copy, error, layout, Array, PyContext};
use mandacaru_core::{
    parquet::Codec,
    paw::record::{LocalPotential, PawDataset, WriteSpec},
    product, Buffer, Context,
};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use std::path::Path;

/// The record as `(payload text, {column: float64 array})`, the tables copied out under a
/// reservation of their size.
pub(crate) fn export<'py>(
    py: Python<'py>,
    ctx: &Context,
    dataset: &PawDataset,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let parts = dataset.parts(ctx).map_err(error)?;
    let values: usize = parts.tables.iter().map(|(_, t)| t.len()).sum();
    let _copies = ctx
        .reserve(product(&[values, 8]).map_err(error)?)
        .map_err(error)?;
    let tables = PyDict::new(py);
    for (name, table) in &parts.tables {
        tables.set_item(name, array(py, &[table.len()], table.to_vec())?)?;
    }
    Ok((parts.text.clone(), tables))
}

/// The record from `(payload text, {column: float64 array})`: the tables are copied in after a
/// layout check and the core reader applies every refusal of a file.
pub(crate) fn import(
    ctx: &Context,
    payload: &str,
    tables: &Bound<'_, PyDict>,
) -> PyResult<PawDataset> {
    let mut columns: Vec<(String, Buffer<f64>)> = Vec::with_capacity(tables.len());
    for (name, value) in tables.iter() {
        let name: String = name
            .extract()
            .map_err(|_| PyTypeError::new_err("the names of the radial tables must be str"))?;
        let values = value.cast::<PyArray1<f64>>().map_err(|_| {
            PyValueError::new_err(format!(
                "radial table {name:?} must be a one-dimensional C-contiguous float64 ndarray"
            ))
        })?;
        let values = values.readonly();
        layout(&values)?;
        columns.push((name, ctx.copy_slice(values.as_slice()?).map_err(error)?));
    }
    PawDataset::from_parts(ctx, payload, columns).map_err(error)
}

#[pyfunction]
fn paw_record_read<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    path: &str,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let ctx = &context.inner;
    let dataset = py
        .detach(|| PawDataset::read(ctx, Path::new(path)))
        .map_err(error)?;
    export(py, ctx, &dataset)
}

#[pyfunction]
#[pyo3(signature = (context, path, payload, tables, *, stride = 1, format = "parquet", codec = "zstd"))]
fn paw_record_write(
    context: PyRef<'_, PyContext>,
    py: Python<'_>,
    path: &str,
    payload: &str,
    tables: &Bound<'_, PyDict>,
    stride: usize,
    format: &str,
    codec: &str,
) -> PyResult<()> {
    let ctx = &context.inner;
    let dataset = import(ctx, payload, tables)?;
    py.detach(|| match format {
        "parquet" => dataset.write(
            ctx,
            Path::new(path),
            &WriteSpec {
                stride,
                codec: Codec::parse(codec)?,
            },
        ),
        "json" => dataset.write_json(ctx, Path::new(path), stride),
        other => Err(mandacaru_core::Error::Invalid(format!(
            "unknown pseudopotential format {other:?}; use parquet or json"
        ))),
    })
    .map_err(error)
}

/// The ionic local potential of a dataset table at `radius` and its slope: the C^2 cubic spline
/// inside the table, flat below it, the ionic tail `-charge/r` beyond it.
#[pyfunction]
fn paw_local_value<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    v_local: PyReadonlyArray1<'py, f64>,
    charge: f64,
    radius: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
    let ctx = &context.inner;
    let (r, _r) = copy(ctx, &r)?;
    let (v, _v) = copy(ctx, &v_local)?;
    let (radius, _x) = copy(ctx, &radius)?;
    let (values, slopes) = py
        .detach(|| {
            LocalPotential::from_table(ctx, &r, &v, charge)?.evaluate(ctx, &radius)
        })
        .map_err(error)?;
    Ok((
        array(py, &[radius.len()], values.as_slice().to_vec())?,
        array(py, &[radius.len()], slopes.as_slice().to_vec())?,
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(paw_record_read, m)?)?;
    m.add_function(wrap_pyfunction!(paw_record_write, m)?)?;
    m.add_function(wrap_pyfunction!(paw_local_value, m)?)?;
    Ok(())
}
