//! Parquet footer and page-header structs over the Thrift compact protocol.
//!
//! Enum-valued fields are stored as their raw `i32` so a struct round-trips
//! exactly and the caller decides which values it supports; the `physical`,
//! `repetition`, `encoding`, `codec` and `page_type` modules name the values
//! and render them for error messages. Unknown Thrift fields (statistics,
//! encoding stats, column orders, logical types beyond STRING, indexes,
//! encryption) are skipped on read and never written.

use std::collections::{BTreeMap, HashSet};

use crate::parquet::thrift::{Reader, Type, Writer};
use crate::parquet::{invalid, ColumnKind, MAGIC, MAX_COLUMNS, MAX_FOOTER_BYTES};
use crate::Result;

/// Physical types (`Type` in parquet.thrift).
pub mod physical {
    pub const BOOLEAN: i32 = 0;
    pub const INT32: i32 = 1;
    pub const INT64: i32 = 2;
    pub const INT96: i32 = 3;
    pub const FLOAT: i32 = 4;
    pub const DOUBLE: i32 = 5;
    pub const BYTE_ARRAY: i32 = 6;
    pub const FIXED_LEN_BYTE_ARRAY: i32 = 7;

    pub fn name(value: i32) -> String {
        match value {
            BOOLEAN => "BOOLEAN".into(),
            INT32 => "INT32".into(),
            INT64 => "INT64".into(),
            INT96 => "INT96".into(),
            FLOAT => "FLOAT".into(),
            DOUBLE => "DOUBLE".into(),
            BYTE_ARRAY => "BYTE_ARRAY".into(),
            FIXED_LEN_BYTE_ARRAY => "FIXED_LEN_BYTE_ARRAY".into(),
            other => format!("physical type {other}"),
        }
    }
}

/// Field repetition (`FieldRepetitionType`).
pub mod repetition {
    pub const REQUIRED: i32 = 0;
    pub const OPTIONAL: i32 = 1;
    pub const REPEATED: i32 = 2;
}

/// Value encodings (`Encoding`).
pub mod encoding {
    pub const PLAIN: i32 = 0;
    pub const PLAIN_DICTIONARY: i32 = 2;
    pub const RLE: i32 = 3;
    pub const BIT_PACKED: i32 = 4;
    pub const DELTA_BINARY_PACKED: i32 = 5;
    pub const DELTA_LENGTH_BYTE_ARRAY: i32 = 6;
    pub const DELTA_BYTE_ARRAY: i32 = 7;
    pub const RLE_DICTIONARY: i32 = 8;
    pub const BYTE_STREAM_SPLIT: i32 = 9;

    pub fn name(value: i32) -> String {
        match value {
            PLAIN => "PLAIN".into(),
            PLAIN_DICTIONARY => "PLAIN_DICTIONARY".into(),
            RLE => "RLE".into(),
            BIT_PACKED => "BIT_PACKED".into(),
            DELTA_BINARY_PACKED => "DELTA_BINARY_PACKED".into(),
            DELTA_LENGTH_BYTE_ARRAY => "DELTA_LENGTH_BYTE_ARRAY".into(),
            DELTA_BYTE_ARRAY => "DELTA_BYTE_ARRAY".into(),
            RLE_DICTIONARY => "RLE_DICTIONARY".into(),
            BYTE_STREAM_SPLIT => "BYTE_STREAM_SPLIT".into(),
            other => format!("encoding {other}"),
        }
    }
}

/// Compression codecs (`CompressionCodec`).
pub mod codec {
    pub const UNCOMPRESSED: i32 = 0;
    pub const SNAPPY: i32 = 1;
    pub const GZIP: i32 = 2;
    pub const LZO: i32 = 3;
    pub const BROTLI: i32 = 4;
    pub const LZ4: i32 = 5;
    pub const ZSTD: i32 = 6;
    pub const LZ4_RAW: i32 = 7;

    pub fn name(value: i32) -> String {
        match value {
            UNCOMPRESSED => "UNCOMPRESSED".into(),
            SNAPPY => "SNAPPY".into(),
            GZIP => "GZIP".into(),
            LZO => "LZO".into(),
            BROTLI => "BROTLI".into(),
            LZ4 => "LZ4".into(),
            ZSTD => "ZSTD".into(),
            LZ4_RAW => "LZ4_RAW".into(),
            other => format!("compression codec {other}"),
        }
    }
}

/// Page types (`PageType`).
pub mod page_type {
    pub const DATA_PAGE: i32 = 0;
    pub const INDEX_PAGE: i32 = 1;
    pub const DICTIONARY_PAGE: i32 = 2;
    pub const DATA_PAGE_V2: i32 = 3;

    pub fn name(value: i32) -> String {
        match value {
            DATA_PAGE => "DATA_PAGE".into(),
            INDEX_PAGE => "INDEX_PAGE".into(),
            DICTIONARY_PAGE => "DICTIONARY_PAGE".into(),
            DATA_PAGE_V2 => "DATA_PAGE_V2".into(),
            other => format!("page type {other}"),
        }
    }
}

/// Converted type UTF8 (`ConvertedType`).
pub const CONVERTED_UTF8: i32 = 0;
/// Converted types that annotate a BYTE_ARRAY as text: ENUM and JSON.
const CONVERTED_ENUM: i32 = 4;
const CONVERTED_JSON: i32 = 19;
/// `LogicalType` union members that annotate a BYTE_ARRAY as text.
const LOGICAL_STRING: i16 = 1;
const LOGICAL_ENUM: i16 = 4;
const LOGICAL_JSON: i16 = 12;

/// The name of a `ConvertedType` for error messages.
fn converted_name(value: i32) -> String {
    match value {
        0 => "UTF8".into(),
        1 => "MAP".into(),
        2 => "MAP_KEY_VALUE".into(),
        3 => "LIST".into(),
        4 => "ENUM".into(),
        5 => "DECIMAL".into(),
        6 => "DATE".into(),
        7 | 8 => "TIME".into(),
        9 | 10 => "TIMESTAMP".into(),
        11..=14 => "UINT".into(),
        15..=18 => "INT".into(),
        19 => "JSON".into(),
        20 => "BSON".into(),
        21 => "INTERVAL".into(),
        other => format!("converted type {other}"),
    }
}

/// The name of a `LogicalType` union member for error messages.
fn logical_name(member: i16) -> String {
    match member {
        2 => "MAP".into(),
        3 => "LIST".into(),
        5 => "DECIMAL".into(),
        6 => "DATE".into(),
        7 => "TIME".into(),
        8 => "TIMESTAMP".into(),
        10 => "INTEGER".into(),
        11 => "UNKNOWN".into(),
        13 => "BSON".into(),
        14 => "UUID".into(),
        15 => "FLOAT16".into(),
        other => format!("logical type {other}"),
    }
}

fn required<T>(value: Option<T>, owner: &str, field: &str) -> Result<T> {
    value.ok_or_else(|| invalid(format!("{owner} is missing the required field '{field}'")))
}

fn non_negative(value: i64, what: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid(format!("{what} is negative ({value})")))
}

/// `KeyValue`: one entry of the file or column key-value metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyValue {
    pub key: String,
    pub value: Option<String>,
}

impl KeyValue {
    pub fn read(r: &mut Reader) -> Result<Self> {
        let (mut key, mut value) = (None, None);
        r.read_struct(|r, id, ty| {
            match id {
                1 => key = Some(r.string(ty)?),
                2 => value = Some(r.string(ty)?),
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(Self {
            key: required(key, "KeyValue", "key")?,
            value,
        })
    }

    pub fn write(&self, w: &mut Writer) -> Result<()> {
        w.struct_begin();
        w.field_string(1, &self.key)?;
        if let Some(value) = &self.value {
            w.field_string(2, value)?;
        }
        w.struct_end()
    }
}

/// `SchemaElement`. `logical_string` is true when the LogicalType union holds
/// STRING; `logical_other` is the id of the first other union member (DECIMAL,
/// BSON, ...), which `validate` refuses on a text column. The rest of the
/// annotation is skipped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaElement {
    pub physical_type: Option<i32>,
    pub type_length: Option<i32>,
    pub repetition: Option<i32>,
    pub name: String,
    pub num_children: Option<i32>,
    pub converted_type: Option<i32>,
    pub logical_string: bool,
    pub logical_other: Option<i16>,
}

impl SchemaElement {
    /// A leaf under the root.
    pub fn leaf(name: &str, physical_type: i32, utf8: bool) -> Self {
        Self {
            physical_type: Some(physical_type),
            type_length: None,
            repetition: Some(repetition::REQUIRED),
            name: name.to_owned(),
            num_children: None,
            converted_type: utf8.then_some(CONVERTED_UTF8),
            logical_string: utf8,
            logical_other: None,
        }
    }

    /// The root group with `children` leaves.
    pub fn root(children: usize) -> Result<Self> {
        Ok(Self {
            physical_type: None,
            type_length: None,
            repetition: None,
            name: "schema".into(),
            num_children: Some(i32::try_from(children).map_err(|_| invalid("too many columns"))?),
            converted_type: None,
            logical_string: false,
            logical_other: None,
        })
    }

    pub fn read(r: &mut Reader) -> Result<Self> {
        let mut out = Self {
            physical_type: None,
            type_length: None,
            repetition: None,
            name: String::new(),
            num_children: None,
            converted_type: None,
            logical_string: false,
            logical_other: None,
        };
        let mut name = None;
        r.read_struct(|r, id, ty| {
            match id {
                1 => out.physical_type = Some(r.i32(ty)?),
                2 => out.type_length = Some(r.i32(ty)?),
                3 => out.repetition = Some(r.i32(ty)?),
                4 => name = Some(r.string(ty)?),
                5 => out.num_children = Some(r.i32(ty)?),
                6 => out.converted_type = Some(r.i32(ty)?),
                10 => {
                    // LogicalType is a union; STRING is member 1 (an empty struct).
                    if ty != Type::Struct {
                        return Ok(false);
                    }
                    let mut string = false;
                    let mut other = None;
                    r.read_struct(|_, member, _| {
                        if member == LOGICAL_STRING {
                            string = true;
                        } else {
                            other.get_or_insert(member);
                        }
                        Ok(false)
                    })?;
                    out.logical_string = string;
                    out.logical_other = other;
                }
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        out.name = required(name, "SchemaElement", "name")?;
        Ok(out)
    }

    pub fn write(&self, w: &mut Writer) -> Result<()> {
        w.struct_begin();
        if let Some(v) = self.physical_type {
            w.field_i32(1, v)?;
        }
        if let Some(v) = self.type_length {
            w.field_i32(2, v)?;
        }
        if let Some(v) = self.repetition {
            w.field_i32(3, v)?;
        }
        w.field_string(4, &self.name)?;
        if let Some(v) = self.num_children {
            w.field_i32(5, v)?;
        }
        if let Some(v) = self.converted_type {
            w.field_i32(6, v)?;
        }
        let member = if self.logical_string {
            Some(LOGICAL_STRING)
        } else {
            self.logical_other
        };
        if let Some(member) = member {
            // The union member is written as an empty struct: enough to carry
            // its id; the reader never looks inside.
            w.field(10, Type::Struct)?;
            w.struct_begin();
            w.field(member, Type::Struct)?;
            w.struct_begin();
            w.struct_end()?;
            w.struct_end()?;
        }
        w.struct_end()
    }
}

/// `ColumnMetaData`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnMetaData {
    pub physical_type: i32,
    pub encodings: Vec<i32>,
    pub path_in_schema: Vec<String>,
    pub codec: i32,
    pub num_values: i64,
    pub total_uncompressed_size: i64,
    pub total_compressed_size: i64,
    pub key_value_metadata: Vec<KeyValue>,
    pub data_page_offset: i64,
    pub index_page_offset: Option<i64>,
    pub dictionary_page_offset: Option<i64>,
}

impl ColumnMetaData {
    pub fn read(r: &mut Reader) -> Result<Self> {
        let o = "ColumnMetaData";
        let (mut physical_type, mut encodings, mut path, mut codec) = (None, None, None, None);
        let (mut num_values, mut uncompressed, mut compressed) = (None, None, None);
        let (mut data_page_offset, mut index, mut dictionary) = (None, None, None);
        let mut key_value_metadata = Vec::new();
        r.read_struct(|r, id, ty| {
            match id {
                1 => physical_type = Some(r.i32(ty)?),
                2 => encodings = Some(r.list_i32(ty)?),
                3 => path = Some(r.list_string(ty)?),
                4 => codec = Some(r.i32(ty)?),
                5 => num_values = Some(r.i64(ty)?),
                6 => uncompressed = Some(r.i64(ty)?),
                7 => compressed = Some(r.i64(ty)?),
                8 => {
                    key_value_metadata = r.list(ty, |r, t| {
                        expect_struct(t)?;
                        KeyValue::read(r)
                    })?
                }
                9 => data_page_offset = Some(r.i64(ty)?),
                10 => index = Some(r.i64(ty)?),
                11 => dictionary = Some(r.i64(ty)?),
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(Self {
            physical_type: required(physical_type, o, "type")?,
            encodings: required(encodings, o, "encodings")?,
            path_in_schema: required(path, o, "path_in_schema")?,
            codec: required(codec, o, "codec")?,
            num_values: required(num_values, o, "num_values")?,
            total_uncompressed_size: required(uncompressed, o, "total_uncompressed_size")?,
            total_compressed_size: required(compressed, o, "total_compressed_size")?,
            key_value_metadata,
            data_page_offset: required(data_page_offset, o, "data_page_offset")?,
            index_page_offset: index,
            dictionary_page_offset: dictionary,
        })
    }

    pub fn write(&self, w: &mut Writer) -> Result<()> {
        w.struct_begin();
        w.field_i32(1, self.physical_type)?;
        w.field(2, Type::List)?;
        w.list_header(Type::I32, self.encodings.len());
        for &e in &self.encodings {
            w.i32(e);
        }
        w.field(3, Type::List)?;
        w.list_header(Type::Binary, self.path_in_schema.len());
        for part in &self.path_in_schema {
            w.string(part);
        }
        w.field_i32(4, self.codec)?;
        w.field_i64(5, self.num_values)?;
        w.field_i64(6, self.total_uncompressed_size)?;
        w.field_i64(7, self.total_compressed_size)?;
        if !self.key_value_metadata.is_empty() {
            w.field(8, Type::List)?;
            w.list_header(Type::Struct, self.key_value_metadata.len());
            for kv in &self.key_value_metadata {
                kv.write(w)?;
            }
        }
        w.field_i64(9, self.data_page_offset)?;
        if let Some(v) = self.index_page_offset {
            w.field_i64(10, v)?;
        }
        if let Some(v) = self.dictionary_page_offset {
            w.field_i64(11, v)?;
        }
        w.struct_end()
    }

    /// Offset of the first byte of the chunk (dictionary page when present).
    pub fn start_offset(&self) -> i64 {
        match self.dictionary_page_offset {
            // pyarrow writes data_page_offset = 0 for a zero-row chunk that
            // still has a dictionary page
            Some(d) if d > 0 && (self.data_page_offset <= 0 || d < self.data_page_offset) => d,
            _ => self.data_page_offset,
        }
    }
}

fn expect_struct(ty: Type) -> Result<()> {
    if ty == Type::Struct {
        Ok(())
    } else {
        Err(invalid(format!(
            "thrift list element is {ty:?} where a struct is required"
        )))
    }
}

/// `ColumnChunk`. A chunk without `meta_data` (encrypted) is rejected on read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnChunk {
    pub file_path: Option<String>,
    pub file_offset: i64,
    pub meta_data: ColumnMetaData,
}

impl ColumnChunk {
    pub fn read(r: &mut Reader) -> Result<Self> {
        let (mut file_path, mut file_offset, mut meta_data) = (None, 0i64, None);
        let mut encrypted = false;
        r.read_struct(|r, id, ty| {
            match id {
                1 => file_path = Some(r.string(ty)?),
                2 => file_offset = r.i64(ty)?,
                3 => {
                    expect_struct(ty)?;
                    meta_data = Some(ColumnMetaData::read(r)?);
                }
                8 | 9 => {
                    encrypted = true;
                    return Ok(false);
                }
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        if encrypted {
            return Err(invalid("encrypted column chunks are not supported"));
        }
        Ok(Self {
            file_path,
            file_offset,
            meta_data: meta_data
                .ok_or_else(|| invalid("column chunk has no metadata (encrypted footer?)"))?,
        })
    }

    pub fn write(&self, w: &mut Writer) -> Result<()> {
        w.struct_begin();
        if let Some(path) = &self.file_path {
            w.field_string(1, path)?;
        }
        w.field_i64(2, self.file_offset)?;
        w.field(3, Type::Struct)?;
        self.meta_data.write(w)?;
        w.struct_end()
    }
}

/// `RowGroup`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowGroup {
    pub columns: Vec<ColumnChunk>,
    pub total_byte_size: i64,
    pub num_rows: i64,
}

impl RowGroup {
    pub fn read(r: &mut Reader) -> Result<Self> {
        let o = "RowGroup";
        let (mut columns, mut total_byte_size, mut num_rows) = (None, None, None);
        r.read_struct(|r, id, ty| {
            match id {
                1 => {
                    columns = Some(r.list(ty, |r, t| {
                        expect_struct(t)?;
                        ColumnChunk::read(r)
                    })?)
                }
                2 => total_byte_size = Some(r.i64(ty)?),
                3 => num_rows = Some(r.i64(ty)?),
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(Self {
            columns: required(columns, o, "columns")?,
            total_byte_size: required(total_byte_size, o, "total_byte_size")?,
            num_rows: required(num_rows, o, "num_rows")?,
        })
    }

    pub fn write(&self, w: &mut Writer) -> Result<()> {
        w.struct_begin();
        w.field(1, Type::List)?;
        w.list_header(Type::Struct, self.columns.len());
        for c in &self.columns {
            c.write(w)?;
        }
        w.field_i64(2, self.total_byte_size)?;
        w.field_i64(3, self.num_rows)?;
        w.struct_end()
    }
}

/// `FileMetaData`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileMetaData {
    pub version: i32,
    pub schema: Vec<SchemaElement>,
    pub num_rows: i64,
    pub row_groups: Vec<RowGroup>,
    pub key_value_metadata: Vec<KeyValue>,
    pub created_by: Option<String>,
}

/// One validated leaf column of a flat schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Leaf {
    pub name: String,
    pub kind: ColumnKind,
    pub optional: bool,
    /// Physical type code (`physical::DOUBLE` or `physical::BYTE_ARRAY`).
    pub physical_type: i32,
}

impl FileMetaData {
    /// Parse a whole footer; bytes after the struct are an error.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let out = Self::read(&mut r)?;
        r.finish()?;
        Ok(out)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut w = Writer::new();
        self.write(&mut w)?;
        w.finish()
    }

    pub fn read(r: &mut Reader) -> Result<Self> {
        let o = "FileMetaData";
        let (mut version, mut schema, mut num_rows, mut row_groups) = (None, None, None, None);
        let (mut key_value_metadata, mut created_by) = (Vec::new(), None);
        let mut encrypted = false;
        r.read_struct(|r, id, ty| {
            match id {
                1 => version = Some(r.i32(ty)?),
                2 => {
                    schema = Some(r.list(ty, |r, t| {
                        expect_struct(t)?;
                        SchemaElement::read(r)
                    })?)
                }
                3 => num_rows = Some(r.i64(ty)?),
                4 => {
                    row_groups = Some(r.list(ty, |r, t| {
                        expect_struct(t)?;
                        RowGroup::read(r)
                    })?)
                }
                5 => {
                    key_value_metadata = r.list(ty, |r, t| {
                        expect_struct(t)?;
                        KeyValue::read(r)
                    })?
                }
                6 => created_by = Some(r.string(ty)?),
                8 | 9 => {
                    encrypted = true;
                    return Ok(false);
                }
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        if encrypted {
            return Err(invalid("encrypted footers are not supported"));
        }
        Ok(Self {
            version: required(version, o, "version")?,
            schema: required(schema, o, "schema")?,
            num_rows: required(num_rows, o, "num_rows")?,
            row_groups: required(row_groups, o, "row_groups")?,
            key_value_metadata,
            created_by,
        })
    }

    pub fn write(&self, w: &mut Writer) -> Result<()> {
        w.struct_begin();
        w.field_i32(1, self.version)?;
        w.field(2, Type::List)?;
        w.list_header(Type::Struct, self.schema.len());
        for s in &self.schema {
            s.write(w)?;
        }
        w.field_i64(3, self.num_rows)?;
        w.field(4, Type::List)?;
        w.list_header(Type::Struct, self.row_groups.len());
        for g in &self.row_groups {
            g.write(w)?;
        }
        if !self.key_value_metadata.is_empty() {
            w.field(5, Type::List)?;
            w.list_header(Type::Struct, self.key_value_metadata.len());
            for kv in &self.key_value_metadata {
                kv.write(w)?;
            }
        }
        if let Some(created_by) = &self.created_by {
            w.field_string(6, created_by)?;
        }
        w.struct_end()
    }

    /// The file key-value metadata as a sorted map. A key without a value maps
    /// to the empty string; a repeated key is an error.
    pub fn metadata_map(&self) -> Result<BTreeMap<String, String>> {
        let mut out = BTreeMap::new();
        for kv in &self.key_value_metadata {
            let value = kv.value.clone().unwrap_or_default();
            if out.insert(kv.key.clone(), value).is_some() {
                return Err(invalid(format!(
                    "key-value metadata repeats the key {:?}",
                    kv.key
                )));
            }
        }
        Ok(out)
    }

    /// Check the footer against the supported subset and return the leaf
    /// columns. `data_end` is the file offset where the footer begins: every
    /// column chunk must lie inside `[4, data_end]`.
    ///
    /// Checks: footer version 1 or 2; one root group followed by exactly
    /// `num_children` flat leaves (no nesting, no REPEATED); leaves DOUBLE or
    /// BYTE_ARRAY (with the UTF8 annotation optional); unique column names;
    /// no external `file_path`; each row group has one chunk per leaf whose
    /// path, type and `num_values` match; row-group row counts sum to
    /// `num_rows`; chunk offsets and sizes are non-negative and in range.
    pub fn validate(&self, data_end: u64) -> Result<Vec<Leaf>> {
        if self.version != 1 && self.version != 2 {
            return Err(invalid(format!(
                "footer version {} is not supported (1 or 2)",
                self.version
            )));
        }
        let root = self
            .schema
            .first()
            .ok_or_else(|| invalid("schema is empty"))?;
        if root.physical_type.is_some() {
            return Err(invalid("the first schema element is not a group"));
        }
        let children = root.num_children.unwrap_or(0);
        let children = usize::try_from(children)
            .map_err(|_| invalid("schema root has a negative child count"))?;
        if children != self.schema.len() - 1 {
            return Err(invalid(
                "nested schemas are not supported (the schema is not a flat list of columns)",
            ));
        }
        if children > MAX_COLUMNS {
            return Err(invalid(format!(
                "the file has {children} columns; at most {MAX_COLUMNS} are supported"
            )));
        }
        let mut leaves: Vec<Leaf> = Vec::with_capacity(children);
        let mut names: HashSet<&str> = HashSet::with_capacity(children);
        for element in &self.schema[1..] {
            if element.num_children.is_some_and(|n| n != 0) {
                return Err(invalid(format!(
                    "column {:?} is a nested group; nested schemas are not supported",
                    element.name
                )));
            }
            let optional = match element.repetition.unwrap_or(repetition::REQUIRED) {
                repetition::REQUIRED => false,
                repetition::OPTIONAL => true,
                _ => {
                    return Err(invalid(format!(
                        "column {:?} is REPEATED; repeated columns are not supported",
                        element.name
                    )))
                }
            };
            let physical_type = element.physical_type.ok_or_else(|| {
                invalid(format!("column {:?} has no physical type", element.name))
            })?;
            let kind = match physical_type {
                physical::DOUBLE => ColumnKind::Double,
                physical::BYTE_ARRAY => ColumnKind::Utf8,
                other => {
                    return Err(invalid(format!(
                        "column {:?} has type {}; only DOUBLE and BYTE_ARRAY (UTF-8) columns are supported",
                        element.name,
                        physical::name(other)
                    )))
                }
            };
            if kind == ColumnKind::Utf8 {
                let text_converted = matches!(
                    element.converted_type,
                    None | Some(CONVERTED_UTF8 | CONVERTED_ENUM | CONVERTED_JSON)
                );
                let text_logical = element
                    .logical_other
                    .is_none_or(|m| m == LOGICAL_ENUM || m == LOGICAL_JSON);
                if !text_converted {
                    return Err(invalid(format!(
                        "column {:?} is a BYTE_ARRAY annotated {}, not text; only UTF-8 text columns are supported",
                        element.name,
                        converted_name(element.converted_type.unwrap_or_default())
                    )));
                }
                if !text_logical {
                    return Err(invalid(format!(
                        "column {:?} is a BYTE_ARRAY annotated {}, not text; only UTF-8 text columns are supported",
                        element.name,
                        logical_name(element.logical_other.unwrap_or_default())
                    )));
                }
            }
            if !names.insert(element.name.as_str()) {
                return Err(invalid(format!(
                    "column name {:?} is repeated",
                    element.name
                )));
            }
            leaves.push(Leaf {
                name: element.name.clone(),
                kind,
                optional,
                physical_type,
            });
        }
        let total_rows = non_negative(self.num_rows, "num_rows")?;
        let mut sum = 0u64;
        // Bytes the non-empty chunks claim. Chunks of an honest file do not
        // overlap, so they cannot claim more than the data area holds; a
        // footer whose chunks all point at one small range is refused instead
        // of being read (and decoded) once per row group.
        let mut claimed = 0u64;
        for (g, group) in self.row_groups.iter().enumerate() {
            let rows = non_negative(group.num_rows, "row group num_rows")?;
            sum = sum
                .checked_add(rows)
                .ok_or_else(|| invalid("row counts overflow"))?;
            if group.columns.len() != leaves.len() {
                return Err(invalid(format!(
                    "row group {g} has {} column chunks for {} columns",
                    group.columns.len(),
                    leaves.len()
                )));
            }
            non_negative(group.total_byte_size, "row group total_byte_size")?;
            for (chunk, leaf) in group.columns.iter().zip(&leaves) {
                let meta = &chunk.meta_data;
                if chunk.file_path.is_some() {
                    return Err(invalid(
                        "column chunks stored in an external file_path are not supported",
                    ));
                }
                if meta.path_in_schema.len() != 1 || meta.path_in_schema[0] != leaf.name {
                    return Err(invalid(format!(
                        "row group {g}: chunk path {:?} does not match column {:?}",
                        meta.path_in_schema, leaf.name
                    )));
                }
                if meta.physical_type != leaf.physical_type {
                    return Err(invalid(format!(
                        "row group {g}: chunk of column {:?} has type {} but the schema says {}",
                        leaf.name,
                        physical::name(meta.physical_type),
                        physical::name(leaf.physical_type)
                    )));
                }
                if non_negative(meta.num_values, "chunk num_values")? != rows {
                    return Err(invalid(format!(
                        "row group {g}: column {:?} has {} values for {} rows",
                        leaf.name, meta.num_values, group.num_rows
                    )));
                }
                non_negative(
                    meta.total_uncompressed_size,
                    "chunk total_uncompressed_size",
                )?;
                let compressed =
                    non_negative(meta.total_compressed_size, "chunk total_compressed_size")?;
                if rows == 0 && compressed == 0 {
                    // An empty chunk (pyarrow writes offset 0 when it has no
                    // dictionary page) holds no byte, so it has no range.
                    continue;
                }
                let start = non_negative(meta.start_offset(), "chunk offset")?;
                if start < MAGIC.len() as u64 {
                    return Err(invalid(format!(
                        "column {:?} starts at offset {start}, inside the file header",
                        leaf.name
                    )));
                }
                let end = start
                    .checked_add(compressed)
                    .ok_or_else(|| invalid("column chunk range overflows"))?;
                if end > data_end {
                    return Err(invalid(format!(
                        "column {:?} extends to byte {end}, past the data area ({data_end} bytes)",
                        leaf.name
                    )));
                }
                claimed = claimed
                    .checked_add(compressed)
                    .ok_or_else(|| invalid("column chunk sizes overflow"))?;
                if claimed > data_end {
                    return Err(invalid(format!(
                        "the column chunks claim {claimed} bytes but the data area holds {data_end} (overlapping chunks)"
                    )));
                }
            }
        }
        if sum != total_rows {
            return Err(invalid(format!(
                "num_rows is {total_rows} but the row groups hold {sum} rows"
            )));
        }
        Ok(leaves)
    }
}

/// Validate the last 8 bytes of a file (`footer length`, `PAR1`) and return
/// the footer length, checked against the file length and `MAX_FOOTER_BYTES`.
/// `tail` must be exactly the last 8 bytes; `file_len` the total file length.
pub fn footer_length(tail: &[u8], file_len: u64) -> Result<usize> {
    if tail.len() != 8 {
        return Err(invalid("the file trailer must be 8 bytes"));
    }
    if &tail[4..] != MAGIC {
        return Err(invalid("not a Parquet file (missing trailing PAR1 magic)"));
    }
    if file_len < 12 {
        return Err(invalid("not a Parquet file (shorter than 12 bytes)"));
    }
    let len = u64::from(u32::from_le_bytes(tail[..4].try_into().expect("4 bytes")));
    if len > MAX_FOOTER_BYTES as u64 {
        return Err(invalid(format!(
            "footer length {len} exceeds the {MAX_FOOTER_BYTES} byte limit"
        )));
    }
    if len + 12 > file_len {
        return Err(invalid(format!(
            "footer length {len} does not fit in a file of {file_len} bytes"
        )));
    }
    Ok(len as usize)
}

/// `PageHeader.data_page_header` (v1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataPageHeader {
    pub num_values: i32,
    pub encoding: i32,
    pub definition_level_encoding: i32,
    pub repetition_level_encoding: i32,
}

impl DataPageHeader {
    pub fn read(r: &mut Reader) -> Result<Self> {
        let o = "DataPageHeader";
        let (mut n, mut e, mut d, mut p) = (None, None, None, None);
        r.read_struct(|r, id, ty| {
            match id {
                1 => n = Some(r.i32(ty)?),
                2 => e = Some(r.i32(ty)?),
                3 => d = Some(r.i32(ty)?),
                4 => p = Some(r.i32(ty)?),
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(Self {
            num_values: required(n, o, "num_values")?,
            encoding: required(e, o, "encoding")?,
            definition_level_encoding: required(d, o, "definition_level_encoding")?,
            repetition_level_encoding: required(p, o, "repetition_level_encoding")?,
        })
    }

    pub fn write(&self, w: &mut Writer) -> Result<()> {
        w.struct_begin();
        w.field_i32(1, self.num_values)?;
        w.field_i32(2, self.encoding)?;
        w.field_i32(3, self.definition_level_encoding)?;
        w.field_i32(4, self.repetition_level_encoding)?;
        w.struct_end()
    }
}

/// `PageHeader.dictionary_page_header`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictionaryPageHeader {
    pub num_values: i32,
    pub encoding: i32,
    pub is_sorted: Option<bool>,
}

impl DictionaryPageHeader {
    pub fn read(r: &mut Reader) -> Result<Self> {
        let o = "DictionaryPageHeader";
        let (mut n, mut e, mut sorted) = (None, None, None);
        r.read_struct(|r, id, ty| {
            match id {
                1 => n = Some(r.i32(ty)?),
                2 => e = Some(r.i32(ty)?),
                3 => sorted = Some(r.boolean(ty)?),
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(Self {
            num_values: required(n, o, "num_values")?,
            encoding: required(e, o, "encoding")?,
            is_sorted: sorted,
        })
    }

    pub fn write(&self, w: &mut Writer) -> Result<()> {
        w.struct_begin();
        w.field_i32(1, self.num_values)?;
        w.field_i32(2, self.encoding)?;
        if let Some(v) = self.is_sorted {
            w.field_bool(3, v)?;
        }
        w.struct_end()
    }
}

/// `PageHeader.data_page_header_v2`. `is_compressed` defaults to true.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataPageHeaderV2 {
    pub num_values: i32,
    pub num_nulls: i32,
    pub num_rows: i32,
    pub encoding: i32,
    pub definition_levels_byte_length: i32,
    pub repetition_levels_byte_length: i32,
    pub is_compressed: bool,
}

impl DataPageHeaderV2 {
    pub fn read(r: &mut Reader) -> Result<Self> {
        let o = "DataPageHeaderV2";
        let (mut n, mut nulls, mut rows, mut e) = (None, None, None, None);
        let (mut dl, mut rl, mut compressed) = (None, None, true);
        r.read_struct(|r, id, ty| {
            match id {
                1 => n = Some(r.i32(ty)?),
                2 => nulls = Some(r.i32(ty)?),
                3 => rows = Some(r.i32(ty)?),
                4 => e = Some(r.i32(ty)?),
                5 => dl = Some(r.i32(ty)?),
                6 => rl = Some(r.i32(ty)?),
                7 => compressed = r.boolean(ty)?,
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(Self {
            num_values: required(n, o, "num_values")?,
            num_nulls: required(nulls, o, "num_nulls")?,
            num_rows: required(rows, o, "num_rows")?,
            encoding: required(e, o, "encoding")?,
            definition_levels_byte_length: required(dl, o, "definition_levels_byte_length")?,
            repetition_levels_byte_length: required(rl, o, "repetition_levels_byte_length")?,
            is_compressed: compressed,
        })
    }

    pub fn write(&self, w: &mut Writer) -> Result<()> {
        w.struct_begin();
        w.field_i32(1, self.num_values)?;
        w.field_i32(2, self.num_nulls)?;
        w.field_i32(3, self.num_rows)?;
        w.field_i32(4, self.encoding)?;
        w.field_i32(5, self.definition_levels_byte_length)?;
        w.field_i32(6, self.repetition_levels_byte_length)?;
        w.field_bool(7, self.is_compressed)?;
        w.struct_end()
    }
}

/// `PageHeader`. Exactly the sub-header that matches `page_type` is read; a
/// header that carries none (an INDEX page) leaves all three `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageHeader {
    pub page_type: i32,
    pub uncompressed_page_size: i32,
    pub compressed_page_size: i32,
    pub crc: Option<i32>,
    pub data_page: Option<DataPageHeader>,
    pub dictionary_page: Option<DictionaryPageHeader>,
    pub data_page_v2: Option<DataPageHeaderV2>,
}

impl PageHeader {
    pub fn read(r: &mut Reader) -> Result<Self> {
        let o = "PageHeader";
        let (mut page_type, mut uncompressed, mut compressed) = (None, None, None);
        let (mut crc, mut data_page, mut dictionary_page, mut v2) = (None, None, None, None);
        r.read_struct(|r, id, ty| {
            match id {
                1 => page_type = Some(r.i32(ty)?),
                2 => uncompressed = Some(r.i32(ty)?),
                3 => compressed = Some(r.i32(ty)?),
                4 => crc = Some(r.i32(ty)?),
                5 => {
                    expect_struct(ty)?;
                    data_page = Some(DataPageHeader::read(r)?);
                }
                7 => {
                    expect_struct(ty)?;
                    dictionary_page = Some(DictionaryPageHeader::read(r)?);
                }
                8 => {
                    expect_struct(ty)?;
                    v2 = Some(DataPageHeaderV2::read(r)?);
                }
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(Self {
            page_type: required(page_type, o, "type")?,
            uncompressed_page_size: required(uncompressed, o, "uncompressed_page_size")?,
            compressed_page_size: required(compressed, o, "compressed_page_size")?,
            crc,
            data_page,
            dictionary_page,
            data_page_v2: v2,
        })
    }

    /// Parse a header at the start of `bytes`; returns it and its byte length.
    pub fn from_bytes(bytes: &[u8]) -> Result<(Self, usize)> {
        let mut r = Reader::new(bytes);
        let header = Self::read(&mut r)?;
        Ok((header, r.position()))
    }

    pub fn write(&self, w: &mut Writer) -> Result<()> {
        w.struct_begin();
        w.field_i32(1, self.page_type)?;
        w.field_i32(2, self.uncompressed_page_size)?;
        w.field_i32(3, self.compressed_page_size)?;
        if let Some(crc) = self.crc {
            w.field_i32(4, crc)?;
        }
        if let Some(h) = &self.data_page {
            w.field(5, Type::Struct)?;
            h.write(w)?;
        }
        if let Some(h) = &self.dictionary_page {
            w.field(7, Type::Struct)?;
            h.write(w)?;
        }
        if let Some(h) = &self.data_page_v2 {
            w.field(8, Type::Struct)?;
            h.write(w)?;
        }
        w.struct_end()
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut w = Writer::new();
        self.write(&mut w)?;
        w.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::thrift::parse_memory_bound;

    fn chunk(name: &str, ty: i32, rows: i64, offset: i64, size: i64) -> ColumnChunk {
        ColumnChunk {
            file_path: None,
            file_offset: offset + size,
            meta_data: ColumnMetaData {
                physical_type: ty,
                encodings: vec![encoding::PLAIN, encoding::RLE],
                path_in_schema: vec![name.to_owned()],
                codec: codec::ZSTD,
                num_values: rows,
                total_uncompressed_size: size * 2,
                total_compressed_size: size,
                key_value_metadata: vec![],
                data_page_offset: offset,
                index_page_offset: None,
                dictionary_page_offset: None,
            },
        }
    }

    fn sample() -> FileMetaData {
        FileMetaData {
            version: 1,
            schema: vec![
                SchemaElement::root(2).unwrap(),
                SchemaElement::leaf("pauli", physical::BYTE_ARRAY, true),
                SchemaElement::leaf("real", physical::DOUBLE, false),
            ],
            num_rows: 30,
            row_groups: vec![
                RowGroup {
                    columns: vec![
                        chunk("pauli", physical::BYTE_ARRAY, 20, 4, 100),
                        chunk("real", physical::DOUBLE, 20, 104, 160),
                    ],
                    total_byte_size: 520,
                    num_rows: 20,
                },
                RowGroup {
                    columns: vec![
                        chunk("pauli", physical::BYTE_ARRAY, 10, 264, 50),
                        chunk("real", physical::DOUBLE, 10, 314, 80),
                    ],
                    total_byte_size: 260,
                    num_rows: 10,
                },
            ],
            key_value_metadata: vec![
                KeyValue {
                    key: "mandacaru.format".into(),
                    value: Some("hamiltonian".into()),
                },
                KeyValue {
                    key: "mandacaru.metadata".into(),
                    value: Some("{\"a\":[1,1]}".into()),
                },
                KeyValue {
                    key: "empty".into(),
                    value: None,
                },
            ],
            created_by: Some("mandacaru-parquet 1".into()),
        }
    }

    #[test]
    fn file_metadata_round_trips() {
        let meta = sample();
        let bytes = meta.to_bytes().unwrap();
        assert_eq!(FileMetaData::from_bytes(&bytes).unwrap(), meta);
        // Writing again gives the same bytes.
        assert_eq!(
            FileMetaData::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
    }

    #[test]
    fn every_truncation_of_an_encoded_footer_is_an_error() {
        let bytes = sample().to_bytes().unwrap();
        for len in 0..bytes.len() {
            assert!(
                FileMetaData::from_bytes(&bytes[..len]).is_err(),
                "footer prefix of {len} bytes parsed"
            );
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(FileMetaData::from_bytes(&extra).is_err());
    }

    #[test]
    fn every_truncation_of_an_encoded_page_header_is_an_error() {
        for header in page_headers() {
            let bytes = header.to_bytes().unwrap();
            for len in 0..bytes.len() {
                assert!(PageHeader::from_bytes(&bytes[..len]).is_err());
            }
        }
    }

    fn page_headers() -> Vec<PageHeader> {
        vec![
            PageHeader {
                page_type: page_type::DATA_PAGE,
                uncompressed_page_size: 800,
                compressed_page_size: 321,
                crc: Some(-12345),
                data_page: Some(DataPageHeader {
                    num_values: 100,
                    encoding: encoding::PLAIN,
                    definition_level_encoding: encoding::RLE,
                    repetition_level_encoding: encoding::RLE,
                }),
                dictionary_page: None,
                data_page_v2: None,
            },
            PageHeader {
                page_type: page_type::DICTIONARY_PAGE,
                uncompressed_page_size: 64,
                compressed_page_size: 64,
                crc: None,
                data_page: None,
                dictionary_page: Some(DictionaryPageHeader {
                    num_values: 8,
                    encoding: encoding::PLAIN_DICTIONARY,
                    is_sorted: Some(true),
                }),
                data_page_v2: None,
            },
            PageHeader {
                page_type: page_type::DATA_PAGE_V2,
                uncompressed_page_size: 10,
                compressed_page_size: 9,
                crc: None,
                data_page: None,
                dictionary_page: None,
                data_page_v2: Some(DataPageHeaderV2 {
                    num_values: 5,
                    num_nulls: 0,
                    num_rows: 5,
                    encoding: encoding::RLE_DICTIONARY,
                    definition_levels_byte_length: 0,
                    repetition_levels_byte_length: 0,
                    is_compressed: false,
                }),
            },
            PageHeader {
                page_type: page_type::INDEX_PAGE,
                uncompressed_page_size: 0,
                compressed_page_size: 0,
                crc: None,
                data_page: None,
                dictionary_page: None,
                data_page_v2: None,
            },
        ]
    }

    #[test]
    fn page_headers_round_trip_and_report_their_length() {
        for header in page_headers() {
            let mut bytes = header.to_bytes().unwrap();
            let len = bytes.len();
            bytes.extend_from_slice(b"page body follows");
            let (back, used) = PageHeader::from_bytes(&bytes).unwrap();
            assert_eq!(back, header);
            assert_eq!(used, len);
        }
    }

    #[test]
    fn v2_is_compressed_defaults_to_true() {
        let mut w = Writer::new();
        w.struct_begin();
        for (id, v) in [(1, 5), (2, 0), (3, 5), (4, 0), (5, 0), (6, 0)] {
            w.field_i32(id, v).unwrap();
        }
        w.struct_end().unwrap();
        let bytes = w.finish().unwrap();
        let header = DataPageHeaderV2::read(&mut Reader::new(&bytes)).unwrap();
        assert!(header.is_compressed);
    }

    #[test]
    fn unknown_fields_are_skipped_at_any_depth() {
        // Footer with extra fields (statistics-like nested structs, a list of
        // structs, a high field id) inserted between known fields.
        let meta = sample();
        let mut w = Writer::new();
        w.struct_begin();
        w.field_i32(1, meta.version).unwrap();
        w.field(2, Type::List).unwrap();
        w.list_header(Type::Struct, meta.schema.len());
        for (i, s) in meta.schema.iter().enumerate() {
            // Re-encode each element with an unknown trailing field.
            w.struct_begin();
            if let Some(v) = s.physical_type {
                w.field_i32(1, v).unwrap();
            }
            if let Some(v) = s.repetition {
                w.field_i32(3, v).unwrap();
            }
            w.field_string(4, &s.name).unwrap();
            if let Some(v) = s.num_children {
                w.field_i32(5, v).unwrap();
            }
            if let Some(v) = s.converted_type {
                w.field_i32(6, v).unwrap();
            }
            if s.logical_string {
                w.field(10, Type::Struct).unwrap();
                w.struct_begin();
                w.field(1, Type::Struct).unwrap();
                w.struct_begin();
                w.struct_end().unwrap();
                w.struct_end().unwrap();
            }
            w.field(99, Type::List).unwrap();
            w.list_header(Type::Struct, 1 + i);
            for _ in 0..=i {
                w.struct_begin();
                w.field(1, Type::Map).unwrap();
                w.map_header(Type::Binary, Type::Double, 1);
                w.string("k");
                w.double(0.5);
                w.struct_end().unwrap();
            }
            w.struct_end().unwrap();
        }
        w.field_i64(3, meta.num_rows).unwrap();
        w.field(4, Type::List).unwrap();
        w.list_header(Type::Struct, 0);
        w.field_string(6, "x").unwrap();
        w.field_i64(300, 1).unwrap();
        w.struct_end().unwrap();
        let bytes = w.finish().unwrap();
        let back = FileMetaData::from_bytes(&bytes).unwrap();
        assert_eq!(back.schema, meta.schema);
        assert_eq!(back.created_by.as_deref(), Some("x"));
        assert!(back.row_groups.is_empty());
    }

    #[test]
    fn missing_required_fields_are_errors() {
        let mut w = Writer::new();
        w.struct_begin();
        w.field_i32(1, 1).unwrap();
        w.struct_end().unwrap();
        let bytes = w.finish().unwrap();
        let e = FileMetaData::from_bytes(&bytes).unwrap_err().to_string();
        assert!(e.contains("schema"), "{e}");
        // A schema element without a name.
        let mut w = Writer::new();
        w.struct_begin();
        w.field_i32(1, 5).unwrap();
        w.struct_end().unwrap();
        let bytes = w.finish().unwrap();
        assert!(SchemaElement::read(&mut Reader::new(&bytes)).is_err());
    }

    #[test]
    fn validation_accepts_the_sample_and_returns_the_leaves() {
        let leaves = sample().validate(10_000).unwrap();
        assert_eq!(leaves.len(), 2);
        assert_eq!(leaves[0].name, "pauli");
        assert_eq!(leaves[0].kind, ColumnKind::Utf8);
        assert_eq!(leaves[1].kind, ColumnKind::Double);
        assert!(!leaves[1].optional);
    }

    #[test]
    fn an_empty_table_validates_with_zero_or_one_empty_row_group() {
        let mut meta = sample();
        meta.num_rows = 0;
        meta.row_groups.clear();
        assert!(meta.validate(100).is_ok());
        meta.row_groups = vec![RowGroup {
            columns: vec![
                chunk("pauli", physical::BYTE_ARRAY, 0, 4, 0),
                chunk("real", physical::DOUBLE, 0, 4, 0),
            ],
            total_byte_size: 0,
            num_rows: 0,
        }];
        assert!(meta.validate(100).is_ok());
    }

    fn rejects(mutate: impl FnOnce(&mut FileMetaData), needle: &str) {
        let mut meta = sample();
        mutate(&mut meta);
        let message = meta.validate(10_000).unwrap_err().to_string();
        assert!(message.starts_with("parquet: "), "{message}");
        assert!(message.contains(needle), "{message:?} lacks {needle:?}");
    }

    #[test]
    fn validation_names_every_unsupported_feature() {
        rejects(|m| m.version = 7, "version");
        rejects(
            |m| m.schema[2].repetition = Some(repetition::REPEATED),
            "REPEATED",
        );
        rejects(
            |m| m.schema[2].physical_type = Some(physical::INT32),
            "INT32",
        );
        rejects(
            |m| m.schema[1].physical_type = Some(physical::FIXED_LEN_BYTE_ARRAY),
            "FIXED_LEN_BYTE_ARRAY",
        );
        rejects(|m| m.schema[1].num_children = Some(2), "nested");
        rejects(|m| m.schema[0].num_children = Some(1), "nested");
        rejects(
            |m| m.schema[0].physical_type = Some(physical::DOUBLE),
            "group",
        );
        rejects(|m| m.schema[2].name = "pauli".into(), "repeated");
        rejects(
            |m| m.row_groups[0].columns[0].file_path = Some("other.parquet".into()),
            "file_path",
        );
        rejects(
            |m| m.row_groups[0].columns[1].meta_data.num_values = 19,
            "values",
        );
        rejects(|m| m.num_rows = 31, "num_rows");
        rejects(|m| m.num_rows = -1, "negative");
        rejects(
            |m| m.row_groups[1].columns.pop().map(|_| ()).unwrap(),
            "column chunks",
        );
        rejects(
            |m| m.row_groups[0].columns[0].meta_data.path_in_schema = vec!["x".into()],
            "path",
        );
        rejects(
            |m| m.row_groups[0].columns[0].meta_data.physical_type = physical::DOUBLE,
            "type",
        );
        rejects(
            |m| m.row_groups[0].columns[0].meta_data.data_page_offset = 2,
            "header",
        );
        rejects(
            |m| m.row_groups[0].columns[0].meta_data.total_compressed_size = 1 << 40,
            "past the data area",
        );
        rejects(
            |m| m.row_groups[0].columns[0].meta_data.data_page_offset = i64::MAX,
            "past the data area",
        );
        rejects(
            |m| m.row_groups[0].columns[0].meta_data.total_compressed_size = -5,
            "negative",
        );
        rejects(|m| m.row_groups[0].num_rows = -1, "negative");
    }

    #[test]
    fn a_dictionary_offset_before_the_data_offset_starts_the_chunk() {
        let mut meta = sample();
        let m = &mut meta.row_groups[0].columns[0].meta_data;
        m.dictionary_page_offset = Some(4);
        m.data_page_offset = 60;
        assert_eq!(m.start_offset(), 4);
        // A dictionary offset of 0 means "absent" in old writers.
        m.dictionary_page_offset = Some(0);
        assert_eq!(m.start_offset(), 60);
    }

    #[test]
    fn a_zero_row_chunk_with_only_a_dictionary_page_starts_at_the_dictionary() {
        let mut meta = sample();
        let m = &mut meta.row_groups[0].columns[0].meta_data;
        m.dictionary_page_offset = Some(4);
        m.data_page_offset = 0;
        assert_eq!(m.start_offset(), 4);
    }

    #[test]
    fn encrypted_footers_and_chunks_are_rejected_by_name() {
        let mut w = Writer::new();
        w.struct_begin();
        w.field_i32(1, 1).unwrap();
        w.field(2, Type::List).unwrap();
        w.list_header(Type::Struct, 0);
        w.field_i64(3, 0).unwrap();
        w.field(4, Type::List).unwrap();
        w.list_header(Type::Struct, 0);
        w.field(8, Type::Struct).unwrap();
        w.struct_begin();
        w.struct_end().unwrap();
        w.struct_end().unwrap();
        let bytes = w.finish().unwrap();
        let e = FileMetaData::from_bytes(&bytes).unwrap_err().to_string();
        assert!(e.contains("encrypted"), "{e}");
        // A chunk with no metadata.
        let mut w = Writer::new();
        w.struct_begin();
        w.field_i64(2, 0).unwrap();
        w.struct_end().unwrap();
        let bytes = w.finish().unwrap();
        let e = ColumnChunk::read(&mut Reader::new(&bytes))
            .unwrap_err()
            .to_string();
        assert!(e.contains("no metadata"), "{e}");
    }

    #[test]
    fn metadata_map_is_sorted_and_rejects_repeated_keys() {
        let map = sample().metadata_map().unwrap();
        assert_eq!(
            map.keys().cloned().collect::<Vec<_>>(),
            vec!["empty", "mandacaru.format", "mandacaru.metadata"]
        );
        assert_eq!(map["empty"], "");
        let mut meta = sample();
        meta.key_value_metadata.push(KeyValue {
            key: "empty".into(),
            value: None,
        });
        assert!(meta.metadata_map().is_err());
    }

    #[test]
    fn trailer_checks() {
        let mut tail = 100u32.to_le_bytes().to_vec();
        tail.extend_from_slice(b"PAR1");
        assert_eq!(footer_length(&tail, 1000).unwrap(), 100);
        assert!(footer_length(&tail, 111).is_err()); // 100 + 12 > 111
        assert_eq!(footer_length(&tail, 112).unwrap(), 100);
        let mut bad = tail.clone();
        bad[7] = b'E'; // PARE
        assert!(footer_length(&bad, 1000).is_err());
        let mut huge = (MAX_FOOTER_BYTES as u32 + 1).to_le_bytes().to_vec();
        huge.extend_from_slice(b"PAR1");
        assert!(footer_length(&huge, u64::MAX).is_err());
        let mut max = (MAX_FOOTER_BYTES as u32).to_le_bytes().to_vec();
        max.extend_from_slice(b"PAR1");
        assert!(footer_length(&max, 1 << 30).is_ok());
        let mut lie = u32::MAX.to_le_bytes().to_vec();
        lie.extend_from_slice(b"PAR1");
        assert!(footer_length(&lie, 1000).is_err());
        assert!(footer_length(&tail[..7], 1000).is_err());
        assert!(footer_length(&tail, 4).is_err());
    }

    #[test]
    fn footer_expansion_stays_inside_the_memory_bound() {
        // Many minimal column chunks: the bound must cover their in-memory size
        // without refusing a legitimate footer.
        let mut meta = sample();
        meta.schema.truncate(2);
        meta.schema[0] = SchemaElement::root(1).unwrap();
        meta.num_rows = 0;
        meta.row_groups = (0..2000)
            .map(|_| RowGroup {
                columns: vec![chunk("pauli", physical::BYTE_ARRAY, 0, 4, 0)],
                total_byte_size: 0,
                num_rows: 0,
            })
            .collect();
        let bytes = meta.to_bytes().unwrap();
        let mut r = Reader::new(&bytes);
        FileMetaData::read(&mut r).unwrap();
        assert!(r.charged() <= parse_memory_bound(bytes.len()));
        // The same bytes under a tiny bound are refused instead of allocated.
        let mut small = Reader::with_limit(&bytes, bytes.len());
        assert!(FileMetaData::read(&mut small).is_err());
    }

    #[test]
    fn a_hostile_footer_of_empty_chunks_is_refused() {
        // A list of 100k one-byte row groups claims far more memory than its size.
        let mut w = Writer::new();
        w.struct_begin();
        w.field_i32(1, 1).unwrap();
        w.field(2, Type::List).unwrap();
        w.list_header(Type::Struct, 0);
        w.field_i64(3, 0).unwrap();
        w.field(4, Type::List).unwrap();
        w.list_header(Type::Struct, 100_000);
        for _ in 0..100_000 {
            w.struct_begin();
            w.struct_end().unwrap();
        }
        w.struct_end().unwrap();
        let bytes = w.finish().unwrap();
        assert!(FileMetaData::from_bytes(&bytes).is_err());
    }

    /// Parse the real footer of every `.parquet` file under `MANDACARU_PAW_PATH`
    /// (and `MANDACARU_UPAW_PATH` when set). Skipped when neither is set.
    #[test]
    fn real_library_footers_parse_and_validate() {
        use std::io::{Read, Seek, SeekFrom};
        let mut roots = Vec::new();
        for var in ["MANDACARU_PAW_PATH", "MANDACARU_UPAW_PATH"] {
            if let Ok(dir) = std::env::var(var) {
                roots.push(std::path::PathBuf::from(dir));
            }
        }
        // Repo file: the Hamiltonian cache example.
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        roots.push(repo.join("examples/hamiltonian_cache/data"));
        if roots.iter().all(|r| !r.exists()) {
            eprintln!("skipped: no Parquet files found");
            return;
        }
        let mut files = Vec::new();
        let mut stack = roots;
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "parquet") {
                    files.push(path);
                }
            }
        }
        files.sort();
        let (mut ok, mut chunks) = (0usize, 0usize);
        for path in &files {
            let mut f = std::fs::File::open(path).unwrap();
            let len = f.metadata().unwrap().len();
            let mut tail = [0u8; 8];
            f.seek(SeekFrom::Start(len - 8)).unwrap();
            f.read_exact(&mut tail).unwrap();
            let footer_len = footer_length(&tail, len).unwrap();
            let mut footer = vec![0u8; footer_len];
            f.seek(SeekFrom::Start(len - 8 - footer_len as u64))
                .unwrap();
            f.read_exact(&mut footer).unwrap();
            let meta = FileMetaData::from_bytes(&footer)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let leaves = meta
                .validate(len - 8 - footer_len as u64)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            chunks += meta.row_groups.len() * leaves.len();
            ok += 1;
        }
        eprintln!("parsed and validated {ok} footers, {chunks} column chunks");
    }
}
