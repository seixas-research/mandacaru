//! Reading and writing dataset files under a `Context` budget.
//!
//! A read reserves everything it holds before it allocates it: the Parquet
//! reader reserves the table (columns, footer), the JSON text is reserved by
//! its length, and the tree the JSON parser builds is reserved from a count of
//! the values and members the text can hold ([`tree_cost`]), so a hostile
//! payload raises `Error::Budget` before a byte of it is parsed. The tree is
//! consumed into the typed record and freed; the record's arrays are reserved
//! as they are created.
//!
//! A write goes to `<path>.new`, is read back (low-level table first, then
//! through the typed reader) and replaces `<path>` only if the read-back is
//! exactly what was written.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use super::codec::{assemble, Tables};
use super::digits;
use super::emit::payload_text;
use super::json::{fail, kind, table_from_json};
use super::{PawDataset, METADATA_KEY};
use crate::parquet::{self, Codec, Column, Table, WriteOptions};
use crate::paw::invalid;
use crate::{Context, Error, Reservation, Result};

/// Bytes charged per JSON value of the parse tree. A `serde_json::Value` is 32
/// bytes; arrays grow by doubling, so a finished array holds up to twice its
/// length and a growing one briefly holds old plus new (3x): 96 covers both.
pub const TREE_BYTES_PER_VALUE: usize = 96;
/// Bytes charged per object member: key `String` header, the B-tree node share
/// and the allocator rounding.
pub const TREE_BYTES_PER_MEMBER: usize = 160;

/// How [`PawDataset::write`] stores a dataset.
#[derive(Debug, Clone, Copy)]
pub struct WriteSpec {
    /// Keep every `stride`-th radial point (`stride - 1, 2 stride - 1, ...`),
    /// each rounded to ten significant digits. 1 stores the tables as given.
    pub stride: usize,
    /// Page compression of the Parquet file.
    pub codec: Codec,
}

/// Upper bound of the memory `serde_json` needs to parse `text` into a
/// `Value`: values counted from the separators outside strings, members from
/// the colons, string contents by the text length. Raises on overflow.
pub fn tree_cost(text: &str) -> Result<usize> {
    let (mut values, mut members) = (1usize, 0usize);
    let (mut in_string, mut escaped) = (false, false);
    for byte in text.bytes() {
        if in_string {
            match (escaped, byte) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_string = false,
                _ => {}
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b',' | b'[' | b'{' => values += 1,
            b':' => members += 1,
            _ => {}
        }
    }
    values
        .checked_mul(TREE_BYTES_PER_VALUE)
        .and_then(|v| {
            members
                .checked_mul(TREE_BYTES_PER_MEMBER)
                .and_then(|m| v.checked_add(m))
        })
        .and_then(|v| v.checked_add(text.len()))
        .ok_or(Error::Overflow)
}

/// Parses `text` into a JSON object, the tree reserved first.
fn parse(ctx: &Context, text: &str, what: &str) -> Result<(Map<String, Value>, Reservation)> {
    let tree = ctx.reserve(tree_cost(text)?)?;
    let value: Value =
        serde_json::from_str(text).map_err(|e| fail(what, format!("not valid JSON: {e}")))?;
    match value {
        Value::Object(map) => Ok((map, tree)),
        other => Err(fail(
            what,
            format!("the payload is {}, not an object", kind(&other)),
        )),
    }
}

fn read_parquet(ctx: &Context, path: &Path) -> Result<PawDataset> {
    let mut table = parquet::read_table(ctx, path)?;
    let Some(text) = table.metadata.remove(METADATA_KEY) else {
        return Err(fail(
            &path.display().to_string(),
            format!("not a Mandacaru pseudopotential Parquet file: no {METADATA_KEY:?} metadata"),
        ));
    };
    // The reader returns the metadata text outside its reservation.
    let _text = ctx.reserve(text.len())?;
    let mut tables = Tables::new();
    for (name, column) in std::mem::take(&mut table.columns) {
        match column {
            Column::Double(values) => {
                tables.insert(name, values);
            }
            Column::Utf8(_) => {
                return Err(fail(
                    "radial_tables",
                    format!("the column {name:?} is not a float64 column"),
                ))
            }
        }
    }
    drop(table);
    let (payload, _tree) = parse(ctx, &text, "payload")?;
    drop(text);
    assemble(ctx, payload, tables)
}

fn read_json(ctx: &Context, path: &Path) -> Result<PawDataset> {
    let open =
        |e: std::io::Error| invalid(format!("paw record: cannot read {}: {e}", path.display()));
    let size = std::fs::metadata(path).map_err(open)?.len();
    let size = usize::try_from(size).map_err(|_| Error::Overflow)?;
    let _text = ctx.reserve(size)?;
    let mut text = String::new();
    text.try_reserve_exact(size)
        .map_err(|_| Error::Allocation)?;
    let file = std::fs::File::open(path).map_err(open)?;
    file.take(size as u64 + 1)
        .read_to_string(&mut text)
        .map_err(open)?;
    if text.len() != size {
        return Err(fail(
            &path.display().to_string(),
            "the file changed while it was read",
        ));
    }
    let (mut payload, _tree) = parse(ctx, &text, "payload")?;
    drop(text);
    let Some(Value::Object(columns)) = payload.remove("radial_tables") else {
        return Err(fail(
            "payload",
            "radial_tables is missing or is not an object",
        ));
    };
    let mut tables = Tables::new();
    for (name, values) in columns {
        let table = table_from_json(ctx, &values, &format!("radial_tables.{name}"))?;
        tables.insert(name, table);
    }
    assemble(ctx, payload, tables)
}

impl PawDataset {
    /// Reads a dataset file: Parquet (detected by its magic) or plain JSON.
    /// Everything it holds is reserved from `ctx` before it is allocated and
    /// released when the record is dropped.
    pub fn read(ctx: &Context, path: &Path) -> Result<Self> {
        if parquet::is_parquet(path)? {
            read_parquet(ctx, path)
        } else {
            read_json(ctx, path)
        }
    }

    /// Writes a Parquet dataset file: the radial tables as `double` columns
    /// (`spec.stride` and ten digits applied), the payload as the JSON metadata.
    /// The file is staged as `<path>.new`, read back, and moved into place only
    /// if the read-back equals what was written; otherwise it is removed and
    /// the error names what differed.
    pub fn write(&self, ctx: &Context, path: &Path, spec: &WriteSpec) -> Result<()> {
        let staged = staged_path(path);
        let result = self.write_staged(ctx, &staged, spec);
        if result.is_err() {
            let _ = std::fs::remove_file(&staged);
            return result;
        }
        std::fs::rename(&staged, path).map_err(|e| {
            invalid(format!(
                "paw record: cannot move {} into place: {e}",
                staged.display()
            ))
        })
    }

    fn write_staged(&self, ctx: &Context, staged: &Path, spec: &WriteSpec) -> Result<()> {
        let mut columns = Vec::new();
        for (name, values) in self.tables() {
            let kept = digits::table(ctx, values, spec.stride)?;
            if kept.as_slice().is_empty() {
                return Err(invalid(format!(
                    "paw record: stride {} leaves no point of the {}-point table {name}",
                    spec.stride,
                    values.len()
                )));
            }
            columns.push((name, Column::Double(kept)));
        }
        let (text, _text_memory) = payload_text(ctx, self, spec.stride)?;
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(METADATA_KEY.to_string(), text);
        let table = Table { columns, metadata };
        let options = WriteOptions {
            codec: spec.codec,
            ..WriteOptions::default()
        };
        parquet::write_table(ctx, staged, &table, &options)?;
        verify(ctx, staged, &table, self, spec.stride)
    }
}

fn staged_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".new");
    PathBuf::from(name)
}

/// The staged file must hold exactly the columns and payload text written.
fn verify(
    ctx: &Context,
    staged: &Path,
    written: &Table,
    source: &PawDataset,
    stride: usize,
) -> Result<()> {
    let back = parquet::read_table(ctx, staged)?;
    if back.column_names() != written.column_names() {
        return Err(fail(
            "read-back",
            "the column names differ from what was written",
        ));
    }
    for ((name, want), (_, got)) in written.columns.iter().zip(&back.columns) {
        let (Some(want), Some(got)) = (want.as_f64(), got.as_f64()) else {
            return Err(fail(
                "read-back",
                format!("column {name} is not a float64 column"),
            ));
        };
        if want.len() != got.len()
            || want
                .iter()
                .zip(got)
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(fail(
                "read-back",
                format!("column {name} differs from what was written"),
            ));
        }
    }
    if back.metadata.get(METADATA_KEY) != written.metadata.get(METADATA_KEY) {
        return Err(fail(
            "read-back",
            "the payload text differs from what was written",
        ));
    }
    drop(back);
    // The typed reader must accept the file and reproduce the payload at
    // stride 1 (the tables are already on their final grid and digits).
    let again = PawDataset::read(ctx, staged)?;
    let (a, _ma) = payload_text(ctx, source, stride)?;
    let (b, _mb) = payload_text(ctx, &again, 1)?;
    if a != b {
        return Err(fail(
            "read-back",
            "the typed record read back does not reproduce the payload",
        ));
    }
    Ok(())
}
