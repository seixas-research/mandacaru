//! File writer.
//!
//! One layout, always: `PAR1`, then for every row group one column chunk per
//! column (data pages only, `DATA_PAGE` v1, PLAIN, REQUIRED columns so no
//! levels, no dictionary, no statistics, the page CRC32 of the stored body), then the Thrift footer,
//! its length as a little-endian `u32` and `PAR1`. Equal input gives equal
//! bytes: `created_by` is fixed, key-value metadata comes out of a `BTreeMap`,
//! nothing carries a timestamp and the zstd encoder is deterministic.
//!
//! Every check that can fail on the input (equal column lengths, unique
//! non-empty names, page sizes the format can express, the footer cap) runs
//! before the stage file exists, so a rejected table creates nothing. The
//! bytes go through `persistence::atomic_write` (same-directory stage file,
//! sync, rename), wrapped in a counting writer that supplies the column
//! offsets. A failure in the middle removes the stage file and leaves the
//! previous file in place.
//!
//! A Parquet codec belongs to a whole column chunk, so the pages of a chunk
//! are compressed first and kept. A zstd chunk stores every page as a frame,
//! including a small page whose frame is no smaller than its raw bytes (the
//! last page of a chunk often holds a handful of rows); only when the frames
//! of the whole chunk together are not smaller than its raw pages is the chunk
//! stored `UNCOMPRESSED`, and its pages are encoded a second time, which is
//! deterministic. Peak memory is the table plus one chunk of frames.

use std::io::Write;
use std::path::Path;

use crate::parquet::codec;
use crate::parquet::crc32::crc32;
use crate::parquet::format::{
    self, encoding, page_type, physical, ColumnChunk, ColumnMetaData, DataPageHeader, FileMetaData,
    KeyValue, PageHeader, RowGroup, SchemaElement,
};
use crate::parquet::{
    invalid, Codec, Column, ColumnKind, Table, WriteOptions, CREATED_BY, MAGIC, MAX_FOOTER_BYTES,
};
use crate::persistence::atomic_write;
use crate::{Context, Error, Reservation, Result};

/// Footer version the writer stores.
const FOOTER_VERSION: i32 = 1;
/// Largest page body the format can express (`i32` sizes), headers excluded.
const MAX_PAGE_BYTES: u64 = (i32::MAX as u64) - (1 << 16);

/// Writes `table` to `path` atomically.
///
/// `Error::Invalid("parquet: ...")` for a malformed table or options;
/// `Error::Budget` when a buffer does not fit the context budget.
pub fn write_table(
    ctx: &Context,
    path: &Path,
    table: &Table,
    options: &WriteOptions,
) -> Result<()> {
    let rows = validate(table, options)?;
    if options.codec == Codec::Zstd {
        // The zstd encoder needs more stack than a small caller thread has.
        return codec::on_large_stack(|| write_validated(ctx, path, table, options, rows))?;
    }
    write_validated(ctx, path, table, options, rows)
}

fn write_validated(
    ctx: &Context,
    path: &Path,
    table: &Table,
    options: &WriteOptions,
    rows: usize,
) -> Result<()> {
    let groups = rows.div_ceil(options.row_group_rows);
    let kv_bytes: usize = table
        .metadata
        .iter()
        .map(|(k, v)| k.len() + v.len() + 16)
        .sum();
    let names_bytes: usize = table.columns.iter().map(|(n, _)| n.len() + 512).sum();
    let footer_estimate = kv_bytes
        .checked_mul(2)
        .and_then(|b| b.checked_add(names_bytes.checked_mul(groups.checked_add(1)?)?))
        .and_then(|b| b.checked_add(1024))
        .ok_or(Error::Overflow)?;
    let _footer_budget = ctx.reserve(footer_estimate)?;

    atomic_write(path, |sink| {
        let mut out = Counting::new(sink);
        out.put(MAGIC)?;
        let mut row_groups = Vec::with_capacity(groups);
        let mut start = 0usize;
        while start < rows {
            let end = start.saturating_add(options.row_group_rows).min(rows);
            row_groups.push(write_group(ctx, &mut out, table, start, end, options)?);
            start = end;
        }
        let meta = file_metadata(table, rows, row_groups)?;
        let data_end = out.position();
        let footer = meta.to_bytes()?;
        if footer.len() > MAX_FOOTER_BYTES {
            return Err(invalid(format!(
                "the footer would hold {} bytes, above the {MAX_FOOTER_BYTES}-byte limit",
                footer.len()
            )));
        }
        meta.validate(data_end)?;
        let footer_length = u32::try_from(footer.len())
            .map_err(|_| invalid("the footer length does not fit 32 bits"))?;
        out.put(&footer)?;
        out.put(&footer_length.to_le_bytes())?;
        out.put(MAGIC)
    })
}

/// Everything that can be wrong with the input, checked before any file is
/// created. Returns the row count.
fn validate(table: &Table, options: &WriteOptions) -> Result<usize> {
    if options.row_group_rows == 0 {
        return Err(invalid("row_group_rows must be positive"));
    }
    if options.page_rows == 0 || options.page_rows > i32::MAX as usize {
        return Err(invalid("page_rows must be in 1..=2147483647"));
    }
    if table.columns.is_empty() {
        return Err(invalid("a table needs at least one column"));
    }
    if table.columns.len() > crate::parquet::MAX_COLUMNS {
        return Err(invalid(format!(
            "a table has {} columns; at most {} are supported",
            table.columns.len(),
            crate::parquet::MAX_COLUMNS
        )));
    }
    let mut names = std::collections::BTreeSet::new();
    for (name, _) in &table.columns {
        if name.is_empty() {
            return Err(invalid("column names must not be empty"));
        }
        if !names.insert(name.as_str()) {
            return Err(invalid(format!("column name {name:?} is repeated")));
        }
    }
    let rows = column_len(&table.columns[0].1);
    for (name, column) in &table.columns {
        let len = column_len(column);
        if len != rows {
            return Err(invalid(format!(
                "column {name:?} holds {len} values but column {:?} holds {rows}",
                table.columns[0].0
            )));
        }
    }
    if i64::try_from(rows).is_err() {
        return Err(invalid("too many rows"));
    }
    let metadata_bytes: usize = table
        .metadata
        .iter()
        .map(|(k, v)| k.len().saturating_add(v.len()).saturating_add(16))
        .fold(0usize, usize::saturating_add);
    if metadata_bytes > MAX_FOOTER_BYTES / 2 {
        return Err(invalid(format!(
            "the key-value metadata holds {metadata_bytes} bytes, too large for the footer"
        )));
    }
    for (name, column) in &table.columns {
        if let Column::Utf8(strings) = column {
            for start in (0..rows).step_by(options.page_rows) {
                let end = start.saturating_add(options.page_rows).min(rows);
                let mut bytes = 0u64;
                for i in start..end {
                    let len = strings.bytes(i).len();
                    if u32::try_from(len).is_err() {
                        return Err(invalid(format!(
                            "column {name:?}, row {i}: a string of {len} bytes does not fit a 32-bit length"
                        )));
                    }
                    bytes += 4 + len as u64;
                }
                if bytes > MAX_PAGE_BYTES {
                    return Err(invalid(format!(
                        "column {name:?}: a page of {} rows would hold {bytes} bytes; lower page_rows",
                        end - start
                    )));
                }
            }
        } else if (options.page_rows.min(rows) as u64).saturating_mul(8) > MAX_PAGE_BYTES {
            return Err(invalid(format!(
                "column {name:?}: a page would exceed the size the format can express; lower page_rows"
            )));
        }
    }
    Ok(rows)
}

fn column_len(column: &Column) -> usize {
    match column {
        Column::Double(values) => values.as_slice().len(),
        Column::Utf8(strings) => strings.len(),
    }
}

fn column_kind(column: &Column) -> ColumnKind {
    match column {
        Column::Double(_) => ColumnKind::Double,
        Column::Utf8(_) => ColumnKind::Utf8,
    }
}

fn file_metadata(table: &Table, rows: usize, row_groups: Vec<RowGroup>) -> Result<FileMetaData> {
    let mut schema = Vec::with_capacity(table.columns.len() + 1);
    schema.push(SchemaElement::root(table.columns.len())?);
    for (name, column) in &table.columns {
        schema.push(match column_kind(column) {
            ColumnKind::Double => SchemaElement::leaf(name, physical::DOUBLE, false),
            ColumnKind::Utf8 => SchemaElement::leaf(name, physical::BYTE_ARRAY, true),
        });
    }
    Ok(FileMetaData {
        version: FOOTER_VERSION,
        schema,
        num_rows: i64::try_from(rows).map_err(|_| invalid("too many rows"))?,
        row_groups,
        key_value_metadata: table
            .metadata
            .iter()
            .map(|(key, value)| KeyValue {
                key: key.clone(),
                value: Some(value.clone()),
            })
            .collect(),
        created_by: Some(CREATED_BY.to_owned()),
    })
}

fn write_group(
    ctx: &Context,
    out: &mut Counting,
    table: &Table,
    start: usize,
    end: usize,
    options: &WriteOptions,
) -> Result<RowGroup> {
    let mut chunks = Vec::with_capacity(table.columns.len());
    let mut total = 0i64;
    for (name, column) in &table.columns {
        let chunk = write_chunk(ctx, out, name, column, start, end, options)?;
        total = total
            .checked_add(chunk.meta_data.total_uncompressed_size)
            .ok_or_else(|| invalid("row group size overflows"))?;
        chunks.push(chunk);
    }
    Ok(RowGroup {
        columns: chunks,
        total_byte_size: total,
        num_rows: i64::try_from(end - start).map_err(|_| invalid("too many rows"))?,
    })
}

/// Bytes of the PLAIN body of rows `start..end`.
fn body_len(column: &Column, start: usize, end: usize) -> usize {
    match column {
        Column::Double(_) => (end - start) * 8,
        Column::Utf8(strings) => (start..end).map(|i| 4 + strings.bytes(i).len()).sum(),
    }
}

/// Encodes rows `start..end` as one PLAIN page body into a charged buffer.
fn encode_page(
    ctx: &Context,
    column: &Column,
    start: usize,
    end: usize,
) -> Result<(Vec<u8>, Reservation)> {
    let len = body_len(column, start, end);
    let reservation = ctx.reserve(len)?;
    let mut body = Vec::new();
    body.try_reserve_exact(len).map_err(|_| Error::Allocation)?;
    match column {
        Column::Double(values) => {
            for value in &values.as_slice()[start..end] {
                body.extend_from_slice(&value.to_le_bytes());
            }
        }
        Column::Utf8(strings) => {
            for i in start..end {
                let text = strings.bytes(i);
                // Validated: every length fits 32 bits.
                body.extend_from_slice(&(text.len() as u32).to_le_bytes());
                body.extend_from_slice(text);
            }
        }
    }
    debug_assert_eq!(body.len(), len);
    Ok((body, reservation))
}

/// One data page ready to be written.
struct Page {
    uncompressed: usize,
    stored: Vec<u8>,
    _memory: Reservation,
}

fn write_chunk(
    ctx: &Context,
    out: &mut Counting,
    name: &str,
    column: &Column,
    start: usize,
    end: usize,
    options: &WriteOptions,
) -> Result<ColumnChunk> {
    let ranges: Vec<(usize, usize)> = (start..end)
        .step_by(options.page_rows)
        .map(|s| (s, s.saturating_add(options.page_rows).min(end)))
        .collect();

    // First pass: compress every page and keep the frames. The codec belongs
    // to the chunk, so one page whose frame is not smaller (the tail page of a
    // chunk may hold four rows) does not decide: the totals do.
    let mut frames: Vec<Page> = Vec::new();
    let mut compressed_chunk = options.codec != Codec::Uncompressed;
    if compressed_chunk {
        frames.reserve_exact(ranges.len());
        let (mut raw_total, mut frame_total) = (0usize, 0usize);
        for &(s, e) in &ranges {
            let (body, _body_memory) = encode_page(ctx, column, s, e)?;
            let Some(frame) = codec::compress(ctx, options.codec, &body)? else {
                compressed_chunk = false;
                break;
            };
            raw_total += body.len();
            frame_total += frame.len();
            let memory = ctx.reserve(frame.len())?;
            frames.push(Page {
                uncompressed: body.len(),
                stored: frame,
                _memory: memory,
            });
        }
        if frame_total >= raw_total && !frames.is_empty() {
            compressed_chunk = false;
        }
    }

    let codec_id = if compressed_chunk {
        codec::codec_id(options.codec)
    } else {
        format::codec::UNCOMPRESSED
    };
    let chunk_start = out.position();
    let mut total_uncompressed = 0u64;
    let mut total_compressed = 0u64;
    let mut write_page = |out: &mut Counting, uncompressed: usize, stored: &[u8], rows: usize| {
        let header = PageHeader {
            page_type: page_type::DATA_PAGE,
            uncompressed_page_size: i32::try_from(uncompressed)
                .map_err(|_| invalid("page size does not fit 32 bits"))?,
            compressed_page_size: i32::try_from(stored.len())
                .map_err(|_| invalid("page size does not fit 32 bits"))?,
            // The zstd frames carry no content checksum; this CRC is what
            // lets a reader refuse a bit-flipped page.
            crc: Some(crc32(stored) as i32),
            data_page: Some(DataPageHeader {
                num_values: i32::try_from(rows)
                    .map_err(|_| invalid("page row count does not fit 32 bits"))?,
                encoding: encoding::PLAIN,
                definition_level_encoding: encoding::RLE,
                repetition_level_encoding: encoding::RLE,
            }),
            dictionary_page: None,
            data_page_v2: None,
        }
        .to_bytes()?;
        out.put(&header)?;
        out.put(stored)?;
        total_uncompressed += (header.len() + uncompressed) as u64;
        total_compressed += (header.len() + stored.len()) as u64;
        Ok::<(), Error>(())
    };
    if compressed_chunk {
        for (page, &(s, e)) in frames.iter().zip(&ranges) {
            write_page(out, page.uncompressed, &page.stored, e - s)?;
        }
    } else {
        drop(frames);
        for &(s, e) in &ranges {
            let (body, _memory) = encode_page(ctx, column, s, e)?;
            write_page(out, body.len(), &body, e - s)?;
        }
    }

    let offset = i64::try_from(chunk_start).map_err(|_| invalid("file offset overflows"))?;
    Ok(ColumnChunk {
        file_path: None,
        file_offset: 0,
        meta_data: ColumnMetaData {
            physical_type: match column_kind(column) {
                ColumnKind::Double => physical::DOUBLE,
                ColumnKind::Utf8 => physical::BYTE_ARRAY,
            },
            encodings: vec![encoding::PLAIN],
            path_in_schema: vec![name.to_owned()],
            codec: codec_id,
            num_values: i64::try_from(end - start).map_err(|_| invalid("too many rows"))?,
            total_uncompressed_size: i64::try_from(total_uncompressed)
                .map_err(|_| invalid("chunk size overflows"))?,
            total_compressed_size: i64::try_from(total_compressed)
                .map_err(|_| invalid("chunk size overflows"))?,
            key_value_metadata: Vec::new(),
            data_page_offset: offset,
            index_page_offset: None,
            dictionary_page_offset: None,
        },
    })
}

/// Counts the bytes that pass through, so column offsets need no seeking.
struct Counting<'a> {
    inner: &'a mut dyn Write,
    written: u64,
}

impl<'a> Counting<'a> {
    fn new(inner: &'a mut dyn Write) -> Self {
        Self { inner, written: 0 }
    }

    fn position(&self) -> u64 {
        self.written
    }

    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        self.write_all(bytes)
            .map_err(|e| invalid(format!("cannot write the file: {e}")))
    }
}

impl Write for Counting<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::format::FileMetaData;
    use crate::parquet::{plain, Utf8Column};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "mandacaru-parquet-write-{tag}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }

        fn entries(&self) -> Vec<String> {
            let mut names: Vec<String> = std::fs::read_dir(&self.0)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            names
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn ctx() -> Context {
        Context::new(Some(1), 1 << 30).unwrap()
    }

    fn doubles(ctx: &Context, values: &[f64]) -> Column {
        Column::Double(ctx.copy_slice(values).unwrap())
    }

    fn sample(ctx: &Context, rows: usize) -> Table {
        let a: Vec<f64> = (0..rows).map(|i| (i as f64) * 0.25 - 3.0).collect();
        let b: Vec<f64> = (0..rows).map(|i| ((i % 5) as f64).sqrt()).collect();
        Table {
            columns: vec![
                ("a".into(), doubles(ctx, &a)),
                ("b".into(), doubles(ctx, &b)),
            ],
            metadata: BTreeMap::from([
                ("zeta".to_owned(), "last".to_owned()),
                ("alpha".to_owned(), "first".to_owned()),
            ]),
        }
    }

    /// Independent mini reader built on `format`, `codec` and `plain` only:
    /// returns the footer and the doubles of every column, group by group.
    fn parse(ctx: &Context, bytes: &[u8]) -> (FileMetaData, Vec<Vec<f64>>) {
        let (meta, doubles, _) = parse_all(ctx, bytes);
        (meta, doubles)
    }

    fn parse_all(ctx: &Context, bytes: &[u8]) -> (FileMetaData, Vec<Vec<f64>>, Vec<Vec<String>>) {
        assert_eq!(&bytes[..4], MAGIC);
        assert_eq!(&bytes[bytes.len() - 4..], MAGIC);
        let tail = &bytes[bytes.len() - 8..];
        let footer_len = u32::from_le_bytes(tail[..4].try_into().unwrap()) as usize;
        let footer_start = bytes.len() - 8 - footer_len;
        let meta =
            FileMetaData::from_bytes(&bytes[footer_start..footer_start + footer_len]).unwrap();
        let leaves = meta.validate(footer_start as u64).unwrap();
        let mut columns = vec![Vec::new(); leaves.len()];
        let mut strings: Vec<Vec<String>> = vec![Vec::new(); leaves.len()];
        for group in &meta.row_groups {
            for (c, chunk) in group.columns.iter().enumerate() {
                let m = &chunk.meta_data;
                let mut position = m.start_offset() as usize;
                let mut remaining = m.num_values as usize;
                let (mut up, mut cp) = (0i64, 0i64);
                while remaining > 0 {
                    let (header, header_len) = PageHeader::from_bytes(&bytes[position..]).unwrap();
                    assert_eq!(header.page_type, page_type::DATA_PAGE);
                    let data = header.data_page.unwrap();
                    assert_eq!(data.encoding, encoding::PLAIN);
                    let n = data.num_values as usize;
                    let stored = &bytes[position + header_len
                        ..position + header_len + header.compressed_page_size as usize];
                    let mut body = vec![0u8; header.uncompressed_page_size as usize];
                    codec::decompress(ctx, m.codec, stored, &mut body).unwrap();
                    if leaves[c].kind == ColumnKind::Double {
                        let mut values = vec![0f64; n];
                        let used = plain::decode_f64(&body, &mut values).unwrap();
                        assert_eq!(used, body.len());
                        columns[c].extend(values);
                    } else {
                        let mut at = 0usize;
                        for _ in 0..n {
                            let len =
                                u32::from_le_bytes(body[at..at + 4].try_into().unwrap()) as usize;
                            let text = std::str::from_utf8(&body[at + 4..at + 4 + len]).unwrap();
                            strings[c].push(text.to_owned());
                            at += 4 + len;
                        }
                        assert_eq!(at, body.len());
                    }
                    up += (header_len + body.len()) as i64;
                    cp += (header_len + stored.len()) as i64;
                    position += header_len + stored.len();
                    remaining -= n;
                }
                assert_eq!(up, m.total_uncompressed_size);
                assert_eq!(cp, m.total_compressed_size);
                assert_eq!(
                    (position - m.start_offset() as usize) as i64,
                    m.total_compressed_size
                );
            }
        }
        (meta, columns, strings)
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn output_is_deterministic_and_reads_back_bit_exact() {
        let c = ctx();
        let dir = Scratch::new("roundtrip");
        let special = [
            f64::NAN,
            f64::from_bits(0x7ff8_dead_beef_0001),
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            0.0,
            f64::MIN_POSITIVE / 4.0,
            5e-324,
            f64::MAX,
            f64::MIN,
            1.0 / 3.0,
        ];
        let mut values = Vec::new();
        for i in 0..1000 {
            values.push(special[i % special.len()]);
            values.push((i as f64).sin() * 1e-3);
        }
        let table = Table {
            columns: vec![
                ("x".into(), doubles(&c, &values)),
                (
                    "y".into(),
                    doubles(&c, &values.iter().rev().copied().collect::<Vec<_>>()),
                ),
            ],
            metadata: BTreeMap::from([("k".to_owned(), "v".to_owned())]),
        };
        // The table's own buffers stay charged; the write must add nothing.
        let held = c.memory_used();
        for codec in [Codec::Zstd, Codec::Uncompressed] {
            for (row_group_rows, page_rows) in [(1usize << 20, 1usize << 16), (700, 64), (1, 1)] {
                if row_group_rows == 1 && values.len() > 200 {
                    continue;
                }
                let options = WriteOptions {
                    codec,
                    row_group_rows,
                    page_rows,
                };
                let (p1, p2) = (dir.path("one.parquet"), dir.path("two.parquet"));
                write_table(&c, &p1, &table, &options).unwrap();
                write_table(&c, &p2, &table, &options).unwrap();
                let one = std::fs::read(&p1).unwrap();
                assert_eq!(
                    one,
                    std::fs::read(&p2).unwrap(),
                    "{codec:?} {row_group_rows}"
                );
                let (meta, columns) = parse(&c, &one);
                assert_eq!(bits(&columns[0]), bits(&values));
                let reversed: Vec<f64> = values.iter().rev().copied().collect();
                assert_eq!(bits(&columns[1]), bits(&reversed));
                assert_eq!(meta.version, 1);
                assert_eq!(meta.created_by.as_deref(), Some(CREATED_BY));
                assert_eq!(meta.num_rows as usize, values.len());
                assert_eq!(meta.row_groups.len(), values.len().div_ceil(row_group_rows));
                assert_eq!(
                    meta.metadata_map().unwrap(),
                    BTreeMap::from([("k".to_owned(), "v".to_owned())])
                );
                assert_eq!(c.memory_used(), held);
            }
        }
        assert_eq!(dir.entries(), ["one.parquet", "two.parquet"]);
    }

    #[test]
    fn schema_columns_and_footer_are_as_documented() {
        let c = ctx();
        let dir = Scratch::new("schema");
        let table = sample(&c, 10);
        let path = dir.path("t.parquet");
        write_table(&c, &path, &table, &WriteOptions::default()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let (meta, columns) = parse(&c, &bytes);
        assert_eq!(meta.schema[0].name, "schema");
        assert_eq!(meta.schema[0].num_children, Some(2));
        for leaf in &meta.schema[1..] {
            assert_eq!(leaf.repetition, Some(format::repetition::REQUIRED));
            assert_eq!(leaf.physical_type, Some(physical::DOUBLE));
            assert_eq!(leaf.type_length, None);
        }
        let keys: Vec<&str> = meta
            .key_value_metadata
            .iter()
            .map(|kv| kv.key.as_str())
            .collect();
        assert_eq!(keys, ["alpha", "zeta"]);
        assert_eq!(columns[0].len(), 10);
        for chunk in &meta.row_groups[0].columns {
            assert_eq!(chunk.meta_data.encodings, [encoding::PLAIN]);
            assert_eq!(chunk.meta_data.dictionary_page_offset, None);
        }
        assert_eq!(meta.row_groups[0].columns[0].meta_data.data_page_offset, 4);
        assert_eq!(
            meta.row_groups[0].total_byte_size,
            meta.row_groups[0]
                .columns
                .iter()
                .map(|c| c.meta_data.total_uncompressed_size)
                .sum::<i64>()
        );
    }

    #[test]
    fn compressible_columns_use_zstd_and_tiny_ones_fall_back_to_raw() {
        let c = ctx();
        let dir = Scratch::new("codec");
        let big = vec![1.5f64; 5000];
        let table = Table {
            columns: vec![("big".into(), doubles(&c, &big))],
            metadata: BTreeMap::new(),
        };
        let path = dir.path("t.parquet");
        write_table(&c, &path, &table, &WriteOptions::default()).unwrap();
        let (meta, columns) = parse(&c, &std::fs::read(&path).unwrap());
        assert_eq!(
            meta.row_groups[0].columns[0].meta_data.codec,
            format::codec::ZSTD
        );
        assert!(
            meta.row_groups[0].columns[0]
                .meta_data
                .total_compressed_size
                < meta.row_groups[0].columns[0]
                    .meta_data
                    .total_uncompressed_size
                    / 4
        );
        assert_eq!(bits(&columns[0]), bits(&big));
        let one = Table {
            columns: vec![("x".into(), doubles(&c, &[7.0]))],
            metadata: BTreeMap::new(),
        };
        write_table(&c, &path, &one, &WriteOptions::default()).unwrap();
        let (meta, columns) = parse(&c, &std::fs::read(&path).unwrap());
        assert_eq!(
            meta.row_groups[0].columns[0].meta_data.codec,
            format::codec::UNCOMPRESSED
        );
        assert_eq!(columns[0], [7.0]);
    }

    #[test]
    fn strings_round_trip_with_utf8_annotation_in_every_layout() {
        let c = ctx();
        let dir = Scratch::new("strings");
        let long = "IXYZ".repeat(2500);
        let mut labels: Vec<String> = vec![
            String::new(),
            "ação".into(),
            "日本語 \u{1F980}".into(),
            "a\0b".into(),
            long,
        ];
        for i in 0..300 {
            labels.push(format!("{}{i}", "XYZI".repeat(i % 9)));
        }
        let real: Vec<f64> = (0..labels.len()).map(|i| i as f64 * 0.5).collect();
        let table = Table {
            columns: vec![
                (
                    "pauli".into(),
                    Column::Utf8(Utf8Column::from_strs(&c, &labels).unwrap()),
                ),
                ("real".into(), doubles(&c, &real)),
            ],
            metadata: BTreeMap::new(),
        };
        let held = c.memory_used();
        for codec in [Codec::Zstd, Codec::Uncompressed] {
            for (row_group_rows, page_rows) in [(1usize << 20, 1usize << 16), (100, 32), (7, 3)] {
                let options = WriteOptions {
                    codec,
                    row_group_rows,
                    page_rows,
                };
                let path = dir.path("s.parquet");
                write_table(&c, &path, &table, &options).unwrap();
                let first = std::fs::read(&path).unwrap();
                write_table(&c, &path, &table, &options).unwrap();
                assert_eq!(first, std::fs::read(&path).unwrap());
                let (meta, doubles, strings) = parse_all(&c, &first);
                assert_eq!(strings[0], labels);
                assert_eq!(bits(&doubles[1]), bits(&real));
                let leaf = &meta.schema[1];
                assert_eq!(leaf.physical_type, Some(physical::BYTE_ARRAY));
                assert_eq!(leaf.converted_type, Some(format::CONVERTED_UTF8));
                assert!(leaf.logical_string);
                assert_eq!(c.memory_used(), held);
            }
        }
    }

    #[test]
    fn an_empty_table_has_no_row_groups_and_keeps_its_schema() {
        let c = ctx();
        let dir = Scratch::new("empty");
        let table = Table {
            columns: vec![("a".into(), doubles(&c, &[]))],
            metadata: BTreeMap::from([("n".to_owned(), "3".to_owned())]),
        };
        let path = dir.path("t.parquet");
        write_table(&c, &path, &table, &WriteOptions::default()).unwrap();
        let (meta, columns) = parse(&c, &std::fs::read(&path).unwrap());
        assert_eq!(meta.num_rows, 0);
        assert!(meta.row_groups.is_empty());
        assert_eq!(meta.schema.len(), 2);
        assert!(columns[0].is_empty());
        assert_eq!(meta.metadata_map().unwrap()["n"], "3");
    }

    #[test]
    fn invalid_input_is_refused_before_any_file_exists() {
        let c = ctx();
        let dir = Scratch::new("invalid");
        let path = dir.path("t.parquet");
        let ok = WriteOptions::default();
        let held = c.memory_used();
        let table = |columns: Vec<(&str, Column)>| Table {
            columns: columns
                .into_iter()
                .map(|(n, c)| (n.to_owned(), c))
                .collect(),
            metadata: BTreeMap::new(),
        };
        let cases: Vec<(Table, WriteOptions, &str)> = vec![
            (table(vec![]), ok, "at least one column"),
            (
                table(vec![
                    ("a", doubles(&c, &[1.0])),
                    ("b", doubles(&c, &[1.0, 2.0])),
                ]),
                ok,
                "holds 2 values",
            ),
            (
                table(vec![("a", doubles(&c, &[1.0])), ("a", doubles(&c, &[1.0]))]),
                ok,
                "repeated",
            ),
            (
                table(vec![("", doubles(&c, &[1.0]))]),
                ok,
                "must not be empty",
            ),
            (
                table(vec![("a", doubles(&c, &[1.0]))]),
                WriteOptions {
                    row_group_rows: 0,
                    ..ok
                },
                "row_group_rows",
            ),
            (
                table(vec![("a", doubles(&c, &[1.0]))]),
                WriteOptions { page_rows: 0, ..ok },
                "page_rows",
            ),
        ];
        for (table, options, needle) in cases {
            let message = format!(
                "{:?}",
                write_table(&c, &path, &table, &options).unwrap_err()
            );
            assert!(
                message.contains("parquet: ") && message.contains(needle),
                "{message}"
            );
            assert!(dir.entries().is_empty(), "{message}");
        }
        assert_eq!(c.memory_used(), held);
    }

    #[test]
    fn a_failure_midway_keeps_the_old_file_and_leaves_no_stage_file() {
        let big = ctx();
        let dir = Scratch::new("atomic");
        let path = dir.path("t.parquet");
        let table = sample(&big, 4000);
        write_table(&big, &path, &table, &WriteOptions::default()).unwrap();
        let before = std::fs::read(&path).unwrap();
        // The footer budget fits, the zstd encoder's working memory does not.
        let tight = Context::new(Some(1), 1 << 20).unwrap();
        let other = Table {
            columns: vec![("q".into(), doubles(&tight, &vec![2.0; 4000]))],
            metadata: BTreeMap::new(),
        };
        let err = write_table(&tight, &path, &other, &WriteOptions::default()).unwrap_err();
        assert!(matches!(err, Error::Budget { .. }), "{err:?}");
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(dir.entries(), ["t.parquet"]);
        assert_eq!(tight.memory_used(), 4000 * 8);
        // A path whose parent is a file fails at staging and creates nothing.
        let blocked = dir.path("t.parquet").join("inner.parquet");
        assert!(write_table(&big, &blocked, &table, &WriteOptions::default()).is_err());
        assert_eq!(dir.entries(), ["t.parquet"]);
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn counting_writer_counts_what_passes_through() {
        let mut sink = Vec::new();
        {
            let mut out = Counting::new(&mut sink);
            out.put(b"abc").unwrap();
            out.put(b"").unwrap();
            out.put(&[0u8; 10]).unwrap();
            assert_eq!(out.position(), 13);
        }
        assert_eq!(sink.len(), 13);
    }

    #[test]
    fn the_wp4_reader_returns_what_the_writer_stored() {
        let c = ctx();
        let dir = Scratch::new("reader");
        let special = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            5e-324,
            f64::MIN_POSITIVE / 3.0,
            f64::MAX,
        ];
        let values: Vec<f64> = (0..2500)
            .map(|i| special[i % special.len()] * if i % 11 == 0 { 1.0 } else { 0.5 })
            .collect();
        let labels: Vec<String> = (0..values.len())
            .map(|i| format!("{}ç日{i}", "XZ".repeat(i % 40)))
            .collect();
        let table = Table {
            columns: vec![
                (
                    "pauli".into(),
                    Column::Utf8(Utf8Column::from_strs(&c, &labels).unwrap()),
                ),
                ("real".into(), doubles(&c, &values)),
            ],
            metadata: BTreeMap::from([(
                "mandacaru.format".to_owned(),
                "qubit-hamiltonian".to_owned(),
            )]),
        };
        let path = dir.path("t.parquet");
        for codec in [Codec::Zstd, Codec::Uncompressed] {
            let options = WriteOptions {
                codec,
                row_group_rows: 1000,
                page_rows: 333,
            };
            write_table(&c, &path, &table, &options).unwrap();
            let read = crate::parquet::read_table(&c, &path).unwrap();
            assert_eq!(read.metadata, table.metadata);
            assert_eq!(read.column_names(), ["pauli", "real"]);
            let strings = read.column("pauli").unwrap().as_utf8().unwrap();
            assert_eq!(
                strings.iter().collect::<Vec<_>>(),
                labels.iter().map(String::as_str).collect::<Vec<_>>()
            );
            assert_eq!(
                bits(read.column("real").unwrap().as_f64().unwrap()),
                bits(&values)
            );
            let footer = crate::parquet::read_metadata(&c, &path).unwrap();
            assert_eq!(footer.num_rows, values.len() as u64);
            assert_eq!(footer.num_row_groups, 3);
            assert_eq!(footer.created_by, CREATED_BY);
            assert_eq!(
                footer.columns,
                [
                    ("pauli".to_owned(), ColumnKind::Utf8),
                    ("real".to_owned(), ColumnKind::Double)
                ]
            );
        }
    }
}
