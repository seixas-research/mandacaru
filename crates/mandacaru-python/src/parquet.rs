//! Bindings of the Parquet reader and writer. Inputs are copied in after a
//! layout check (the copy is charged to the context budget first), outputs are
//! owned arrays and lists, and the file I/O runs with the GIL released. The
//! format lives in `mandacaru_core::parquet`.
use super::{array, error, layout, PyContext};
use mandacaru_core::{
    parquet::{self, Codec, Column, ColumnKind, Table, Utf8Column, WriteOptions},
    Context, Error,
};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList, PyString},
};
use std::{collections::BTreeMap, path::Path};

/// Bytes the `Vec<Bound<PyString>>` and `Vec<&str>` of a string column take
/// per value while the column is copied.
const STRING_HANDLE_BYTES: usize = 24;

fn column_from(ctx: &Context, name: &str, value: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(values) = value.cast::<PyArray1<f64>>() {
        let values = values.readonly();
        layout(&values)?;
        return Ok(Column::Double(
            ctx.copy_slice(values.as_slice()?).map_err(error)?,
        ));
    }
    if let Ok(list) = value.cast::<PyList>() {
        let _handles = ctx
            .reserve(
                list.len()
                    .checked_mul(STRING_HANDLE_BYTES)
                    .ok_or_else(|| error(Error::Overflow))?,
            )
            .map_err(error)?;
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            let text = item.cast_into::<PyString>().map_err(|_| {
                PyValueError::new_err(format!(
                    "parquet column {name:?}: a list column must hold only str values"
                ))
            })?;
            items.push(text);
        }
        let mut texts = Vec::with_capacity(items.len());
        for item in &items {
            texts.push(item.to_str()?);
        }
        return Ok(Column::Utf8(
            Utf8Column::from_strs(ctx, &texts).map_err(error)?,
        ));
    }
    Err(PyValueError::new_err(format!(
        "parquet column {name:?} must be a one-dimensional C-contiguous float64 \
         ndarray or a list of str"
    )))
}

fn metadata_from(metadata: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for (key, value) in metadata.iter() {
        let (Ok(key), Ok(value)) = (key.extract::<String>(), value.extract::<String>()) else {
            return Err(PyValueError::new_err(
                "parquet metadata must map str keys to str values",
            ));
        };
        out.insert(key, value);
    }
    Ok(out)
}

fn kind_name(kind: ColumnKind) -> &'static str {
    kind.name()
}

/// Bytes a Python `str` of `text` takes (object, rounded to the allocator's
/// 16-byte step) plus the 8-byte slot that holds it in a list or dict. Wide
/// text is priced as four bytes a byte, an upper bound of any kind.
fn python_str_bytes(text: &str) -> usize {
    let object = if text.is_ascii() {
        56 + text.len()
    } else {
        88usize.saturating_add(text.len().saturating_mul(4))
    };
    (object.saturating_add(15) & !15).saturating_add(8)
}

/// Bytes the key-value metadata takes as the Rust map the reader returned
/// (it is no longer charged once the read returns) and as a Python dict.
fn metadata_bytes(metadata: &BTreeMap<String, String>) -> usize {
    metadata.iter().fold(0usize, |total, (key, value)| {
        total
            .saturating_add(key.len())
            .saturating_add(value.len())
            .saturating_add(96)
            .saturating_add(python_str_bytes(key))
            .saturating_add(python_str_bytes(value))
            .saturating_add(64)
    })
}

/// Reads a Parquet file: `{"num_rows", "columns": {name: float64 ndarray |
/// list[str]}, "metadata": {str: str}}`, columns in file order.
///
/// The float64 columns move into the arrays without a copy. The Python
/// objects built here (the `str` of a text column, the metadata dict) are
/// charged to the context before they are built.
#[pyfunction]
fn read_parquet<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    path: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = context.inner.clone();
    let Table { columns, metadata } = py
        .detach(|| parquet::read_table(&ctx, Path::new(path)))
        .map_err(error)?;
    let mut guards = Vec::with_capacity(columns.len() + 1);
    guards.push(ctx.reserve(metadata_bytes(&metadata)).map_err(error)?);
    let num_rows = columns.first().map_or(0, |(_, column)| column.len());
    let out_columns = PyDict::new(py);
    for (name, column) in columns {
        guards.push(
            ctx.reserve(python_str_bytes(&name).saturating_mul(2))
                .map_err(error)?,
        );
        match column {
            Column::Double(values) => {
                let (values, guard) = values.into_parts();
                guards.push(guard);
                out_columns.set_item(&name, array(py, &[values.len()], values)?)?;
            }
            Column::Utf8(values) => {
                let bytes = values
                    .iter()
                    .fold(0usize, |t, text| t.saturating_add(python_str_bytes(text)));
                guards.push(ctx.reserve(bytes).map_err(error)?);
                let list = PyList::new(py, (0..values.len()).map(|i| values.get(i)))?;
                out_columns.set_item(&name, list)?;
            }
        }
    }
    let out_metadata = PyDict::new(py);
    for (key, value) in &metadata {
        out_metadata.set_item(key, value)?;
    }
    let out = PyDict::new(py);
    out.set_item("num_rows", num_rows)?;
    out.set_item("columns", out_columns)?;
    out.set_item("metadata", out_metadata)?;
    Ok(out)
}

/// Trailer and footer only: `{"num_rows", "num_row_groups", "columns":
/// [(name, kind)], "metadata"}`. No data page is read.
#[pyfunction]
fn read_parquet_metadata<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    path: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = context.inner.clone();
    let footer = py
        .detach(|| parquet::read_metadata(&ctx, Path::new(path)))
        .map_err(error)?;
    let names = footer.columns.iter().fold(0usize, |total, (name, _)| {
        total
            .saturating_add(name.len())
            .saturating_add(python_str_bytes(name))
            .saturating_add(128)
    });
    let _guard = ctx
        .reserve(metadata_bytes(&footer.metadata).saturating_add(names))
        .map_err(error)?;
    let columns = PyList::empty(py);
    for (name, kind) in &footer.columns {
        columns.append((name, kind_name(*kind)))?;
    }
    let metadata = PyDict::new(py);
    for (key, value) in &footer.metadata {
        metadata.set_item(key, value)?;
    }
    let out = PyDict::new(py);
    out.set_item("num_rows", footer.num_rows)?;
    out.set_item("num_row_groups", footer.num_row_groups)?;
    out.set_item("created_by", footer.created_by)?;
    out.set_item("columns", columns)?;
    out.set_item("metadata", metadata)?;
    Ok(out)
}

/// Writes `columns` (name to float64 ndarray or list of str, in order) and the
/// footer `metadata` to `path`, atomically.
#[pyfunction]
#[pyo3(signature = (context, path, columns, metadata, compression = "zstd"))]
fn write_parquet(
    context: PyRef<'_, PyContext>,
    py: Python<'_>,
    path: &str,
    columns: &Bound<'_, PyDict>,
    metadata: &Bound<'_, PyDict>,
    compression: &str,
) -> PyResult<()> {
    let ctx = context.inner.clone();
    let codec = Codec::parse(compression).map_err(error)?;
    let metadata = metadata_from(metadata)?;
    let mut table = Table {
        columns: Vec::with_capacity(columns.len()),
        metadata,
    };
    for (key, value) in columns.iter() {
        let name = key
            .extract::<String>()
            .map_err(|_| PyValueError::new_err("parquet column names must be str"))?;
        let column = column_from(&ctx, &name, &value)?;
        table.columns.push((name, column));
    }
    let options = WriteOptions {
        codec,
        ..WriteOptions::default()
    };
    py.detach(|| parquet::write_table(&ctx, Path::new(path), &table, &options))
        .map_err(error)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(read_parquet_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(write_parquet, m)?)?;
    Ok(())
}
