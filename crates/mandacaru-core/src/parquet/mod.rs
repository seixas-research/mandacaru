//! Parquet read and write, hand-written, with no dependency on Python or NumPy.
//!
//! Layout of this module (one owner per file, see `dev_plan/PARQUET_RUST_PLAN.md`):
//!
//! - `thrift`: Thrift compact protocol reader and writer.
//! - `format`: the Parquet footer and page-header structs over `thrift`.
//! - `hybrid`, `plain`, `crc32`: value encodings and the page checksum.
//! - `codec`: page compression (`zstd`, `snappy`) and the dispatch.
//! - `column`: the in-memory `Table` and its columns.
//! - `read`, `write`: the file reader and the atomic writer.
//!
//! Every public entry point takes a `Context` first and reserves its buffers
//! from the context budget before allocating them. Anything unsupported or
//! malformed raises `Error::Invalid("parquet: ...")`; nothing is skipped
//! silently.

pub mod codec;
pub mod column;
pub mod crc32;
pub mod format;
pub mod hybrid;
pub mod plain;
pub mod read;
pub mod thrift;
pub mod write;

use std::collections::BTreeMap;

use crate::{Error, Result};

pub use column::{Column, Table, Utf8Column};
pub use read::{is_parquet, read_metadata, read_table};
pub use write::write_table;

/// Leading and trailing magic of every Parquet file.
pub const MAGIC: &[u8; 4] = b"PAR1";
/// Largest footer the reader accepts (the largest shipped footer is 4.5 MB).
pub const MAX_FOOTER_BYTES: usize = 64 << 20;
/// Most columns a file may have, read or written. The Hamiltonian and
/// pseudopotential layouts need a few dozen; the cap keeps a hostile footer
/// from making per-column work unbounded.
pub const MAX_COLUMNS: usize = 65_536;
/// Deepest Thrift struct/list/map nesting the reader follows or skips.
pub const MAX_THRIFT_DEPTH: usize = 16;
/// Largest zstd window the page decoder accepts (bytes).
pub const MAX_ZSTD_WINDOW_BYTES: usize = 128 << 20;
/// `created_by` string the writer stores.
pub const CREATED_BY: &str = "mandacaru-parquet 1";
/// Default row-group size of the writer (rows).
pub const DEFAULT_ROW_GROUP_ROWS: usize = 1 << 20;
/// Default data-page size of the writer (rows).
pub const DEFAULT_PAGE_ROWS: usize = 1 << 16;

/// Page compression the writer can emit. Snappy is read-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Uncompressed,
    Zstd,
}

impl Codec {
    /// Accepts `zstd`, `uncompressed` and `none`, case-insensitively.
    pub fn parse(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "zstd" => Ok(Self::Zstd),
            "uncompressed" | "none" => Ok(Self::Uncompressed),
            _ => Err(Error::Invalid(format!(
                "parquet: unknown compression {name:?}; accepted names are zstd, uncompressed, none"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Uncompressed => "uncompressed",
            Self::Zstd => "zstd",
        }
    }
}

/// Physical column types the reader and writer support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnKind {
    Double,
    Utf8,
}

impl ColumnKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Double => "double",
            Self::Utf8 => "utf8",
        }
    }
}

/// What `read_metadata` returns: the trailer and footer only, no data pages.
#[derive(Debug, Clone, PartialEq)]
pub struct Footer {
    pub num_rows: u64,
    pub num_row_groups: usize,
    pub created_by: String,
    pub columns: Vec<(String, ColumnKind)>,
    pub metadata: BTreeMap<String, String>,
}

/// Writer settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteOptions {
    pub codec: Codec,
    /// Rows per row group (at most 1,048,576 by default).
    pub row_group_rows: usize,
    /// Rows per data page (at most 65,536 by default).
    pub page_rows: usize,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            codec: Codec::Zstd,
            row_group_rows: DEFAULT_ROW_GROUP_ROWS,
            page_rows: DEFAULT_PAGE_ROWS,
        }
    }
}

/// `Error::Invalid` with the `parquet: ` prefix every error of this module carries.
pub(crate) fn invalid(message: impl AsRef<str>) -> Error {
    Error::Invalid(format!("parquet: {}", message.as_ref()))
}
