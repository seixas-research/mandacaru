//! In-memory table and columns.
//!
//! A `Table` is what the reader returns and the writer takes: named columns in
//! file order plus the sorted key-value metadata. Doubles live in a
//! context-accounted `Buffer<f64>`. Strings live in a flat layout (one byte
//! vector plus `rows + 1` end offsets) so the budget charges the real bytes; a
//! `Buffer<String>` would be charged 24 bytes per element and nothing for the
//! text.
//!
//! The memory of every column is reserved from the `Context` before it is
//! allocated and released when the column is dropped.

use std::collections::BTreeMap;

use crate::parquet::invalid;
use crate::{product, Buffer, Context, Error, Reservation, Result};

/// A UTF-8 string column in a flat layout so the budget charges real bytes.
///
/// `offsets` always holds `len() + 1` end positions and starts with 0, so an
/// empty column has the single offset `[0]`. Every constructor validates UTF-8
/// (a `&str` input is valid by construction), so a value is always valid text.
#[derive(Debug)]
pub struct Utf8Column {
    data: Vec<u8>,
    offsets: Vec<u64>,
    _offsets_memory: Option<Reservation>,
    _data_memory: Option<Reservation>,
}

impl Utf8Column {
    /// A column with no values.
    pub fn empty(ctx: &Context) -> Result<Self> {
        Utf8Builder::new(ctx, 0)?.finish(ctx)
    }

    /// Copy `values` into a new column; reserves `8 * (rows + 1) + bytes`
    /// before allocating.
    pub fn from_strs<S: AsRef<str>>(ctx: &Context, values: &[S]) -> Result<Self> {
        let mut bytes = 0usize;
        for value in values {
            bytes = bytes
                .checked_add(value.as_ref().len())
                .ok_or(Error::Overflow)?;
        }
        let mut builder = Utf8Builder::new(ctx, values.len())?;
        builder.reserve_data(ctx, bytes)?;
        for value in values {
            builder.push_str(value.as_ref())?;
        }
        builder.finish(ctx)
    }

    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `i`-th value. Panics when `i >= len()`, like slice indexing.
    pub fn get(&self, i: usize) -> &str {
        std::str::from_utf8(self.bytes(i)).expect("a Utf8Column only holds validated UTF-8")
    }

    /// The `i`-th value as bytes. Panics when `i >= len()`.
    pub fn bytes(&self, i: usize) -> &[u8] {
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        &self.data[start..end]
    }

    /// Values in order.
    pub fn iter(&self) -> impl Iterator<Item = &str> + '_ {
        (0..self.len()).map(|i| self.get(i))
    }

    /// All value bytes, concatenated.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// `len() + 1` end positions into [`Utf8Column::data`], the first being 0.
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }
}

#[derive(Debug)]
pub enum Column {
    Double(Buffer<f64>),
    Utf8(Utf8Column),
}

impl Column {
    /// Number of values.
    pub fn len(&self) -> usize {
        match self {
            Self::Double(values) => values.as_slice().len(),
            Self::Utf8(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f64(&self) -> Option<&[f64]> {
        match self {
            Self::Double(values) => Some(values.as_slice()),
            Self::Utf8(_) => None,
        }
    }

    pub fn as_utf8(&self) -> Option<&Utf8Column> {
        match self {
            Self::Utf8(values) => Some(values),
            Self::Double(_) => None,
        }
    }
}

#[derive(Debug)]
pub struct Table {
    /// Columns in file order.
    pub columns: Vec<(String, Column)>,
    /// Sorted, so equal tables produce equal bytes.
    pub metadata: BTreeMap<String, String>,
}

impl Table {
    /// Rows of the first column (0 for a table without columns).
    pub fn num_rows(&self) -> usize {
        self.columns.first().map_or(0, |(_, column)| column.len())
    }

    /// The column called `name`.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, column)| column)
    }

    /// Column names in file order.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|(name, _)| name.as_str()).collect()
    }
}

/// Builds a `Utf8Column` from pages. The `rows + 1` offsets are reserved when
/// the builder is created (that is the check that refuses a footer claiming
/// 2^40 rows); the byte vector grows geometrically, every growth reserving the
/// new capacity before the old one is released, so the accounted memory never
/// falls below what the allocator holds.
#[derive(Debug)]
pub(crate) struct Utf8Builder {
    data: Vec<u8>,
    offsets: Vec<u64>,
    rows: usize,
    offsets_memory: Option<Reservation>,
    data_memory: Option<Reservation>,
}

impl Utf8Builder {
    pub(crate) fn new(ctx: &Context, rows: usize) -> Result<Self> {
        let slots = rows.checked_add(1).ok_or(Error::Overflow)?;
        let bytes = product(&[slots, std::mem::size_of::<u64>()])?;
        if bytes > isize::MAX as usize {
            return Err(Error::Overflow);
        }
        let memory = ctx.reserve(bytes)?;
        let mut offsets = Vec::new();
        offsets
            .try_reserve_exact(slots)
            .map_err(|_| Error::Allocation)?;
        offsets.push(0);
        Ok(Self {
            data: Vec::new(),
            offsets,
            rows,
            offsets_memory: Some(memory),
            data_memory: None,
        })
    }

    /// Values appended so far.
    pub(crate) fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Make room for `extra` more payload bytes without a reallocation inside
    /// the append functions.
    pub(crate) fn reserve_data(&mut self, ctx: &Context, extra: usize) -> Result<()> {
        let needed = self.data.len().checked_add(extra).ok_or(Error::Overflow)?;
        if needed <= self.data.capacity() {
            return Ok(());
        }
        if needed > isize::MAX as usize {
            return Err(Error::Overflow);
        }
        let capacity = needed.max(self.data.capacity() + self.data.capacity() / 2);
        // The new capacity is reserved while the old reservation is still held:
        // during the copy both vectors exist.
        let memory = ctx.reserve(capacity)?;
        let mut grown = Vec::new();
        grown
            .try_reserve_exact(capacity)
            .map_err(|_| Error::Allocation)?;
        grown.extend_from_slice(&self.data);
        self.data = grown;
        self.data_memory = Some(memory);
        Ok(())
    }

    /// Vectors for `plain::append_*`, which require the spare capacity that
    /// [`Utf8Builder::reserve_data`] and the constructor provide.
    pub(crate) fn vectors(&mut self) -> (&mut Vec<u8>, &mut Vec<u64>) {
        (&mut self.data, &mut self.offsets)
    }

    fn push_str(&mut self, value: &str) -> Result<()> {
        if self.offsets.len() > self.rows || self.data.capacity() - self.data.len() < value.len() {
            return Err(invalid("string column received more values than reserved"));
        }
        self.data.extend_from_slice(value.as_bytes());
        self.offsets.push(self.data.len() as u64);
        Ok(())
    }

    pub(crate) fn finish(mut self, _ctx: &Context) -> Result<Utf8Column> {
        if self.offsets.len() != self.rows + 1 {
            return Err(invalid(format!(
                "string column holds {} values for {} rows",
                self.offsets.len() - 1,
                self.rows
            )));
        }
        self.data.shrink_to_fit();
        if let Some(memory) = self.data_memory.as_mut() {
            memory.shrink_to(self.data.capacity().min(memory.bytes()))?;
        }
        Ok(Utf8Column {
            data: self.data,
            offsets: self.offsets,
            _offsets_memory: self.offsets_memory,
            _data_memory: self.data_memory,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> Context {
        Context::new(Some(1), 1 << 24).unwrap()
    }

    #[test]
    fn from_strs_round_trips_and_releases_memory() {
        let c = ctx();
        {
            let column = Utf8Column::from_strs(&c, &["IXYZ", "", "ZZZZ", "héllo"]).unwrap();
            assert_eq!(column.len(), 4);
            assert_eq!(column.get(0), "IXYZ");
            assert_eq!(column.get(1), "");
            assert_eq!(column.get(3), "héllo");
            assert_eq!(column.offsets()[0], 0);
            assert_eq!(
                column.iter().collect::<Vec<_>>(),
                ["IXYZ", "", "ZZZZ", "héllo"]
            );
            // 8 bytes per offset (rows + 1) plus the real text bytes.
            assert_eq!(c.memory_used(), 8 * 5 + column.data().len());
        }
        assert_eq!(c.memory_used(), 0);
    }

    #[test]
    fn an_empty_column_has_the_single_offset_zero() {
        let c = ctx();
        let column = Utf8Column::empty(&c).unwrap();
        assert!(column.is_empty());
        assert_eq!(column.offsets(), [0]);
        drop(column);
        assert_eq!(c.memory_used(), 0);
    }

    #[test]
    fn the_budget_refuses_before_allocating() {
        let c = Context::new(Some(1), 1 << 10).unwrap();
        let huge = Utf8Builder::new(&c, 1usize << 40).unwrap_err();
        assert!(matches!(huge, Error::Budget { .. }), "{huge:?}");
        assert_eq!(c.memory_used(), 0);
        let none = Utf8Builder::new(&c, usize::MAX).unwrap_err();
        assert!(matches!(none, Error::Overflow), "{none:?}");
    }

    #[test]
    fn data_growth_is_charged_and_shrinks_at_the_end() {
        let c = ctx();
        let mut builder = Utf8Builder::new(&c, 3).unwrap();
        builder.reserve_data(&c, 10).unwrap();
        builder.reserve_data(&c, 100).unwrap();
        assert!(c.memory_used() >= 8 * 4 + 100);
        for text in ["a", "bb", "ccc"] {
            builder.push_str(text).unwrap();
        }
        assert!(
            builder.push_str("d").is_err(),
            "a fourth value is one too many"
        );
        let column = builder.finish(&c).unwrap();
        assert_eq!(column.get(2), "ccc");
        assert!(
            c.memory_used() < 8 * 4 + 100,
            "the data reservation was trimmed"
        );
        drop(column);
        assert_eq!(c.memory_used(), 0);
    }

    #[test]
    fn a_short_builder_cannot_finish() {
        let c = ctx();
        let builder = Utf8Builder::new(&c, 2).unwrap();
        assert!(builder.finish(&c).is_err());
        assert_eq!(c.memory_used(), 0);
    }

    #[test]
    fn table_lookups() {
        let c = ctx();
        let table = Table {
            columns: vec![
                (
                    "s".into(),
                    Column::Utf8(Utf8Column::from_strs(&c, &["a", "b"]).unwrap()),
                ),
                (
                    "d".into(),
                    Column::Double(c.copy_slice(&[1.0, 2.0]).unwrap()),
                ),
            ],
            metadata: BTreeMap::new(),
        };
        assert_eq!(table.num_rows(), 2);
        assert_eq!(table.column_names(), ["s", "d"]);
        assert_eq!(table.column("d").unwrap().as_f64().unwrap(), [1.0, 2.0]);
        assert!(table.column("s").unwrap().as_f64().is_none());
        assert!(table.column("missing").is_none());
        drop(table);
        assert_eq!(c.memory_used(), 0);
    }
}
