//! File reader.
//!
//! `read_table` reads the trailer and footer, validates the footer against the
//! supported subset, reserves the output columns from the `Context` (a footer
//! that claims more rows than the budget allows fails here, before any data
//! page is touched), then streams the file one column chunk at a time: the
//! chunk bytes are read into a reserved scratch buffer, every page is
//! decompressed into a second reserved scratch buffer and decoded straight into
//! the preallocated column. The peak is the table plus one chunk plus one page.
//!
//! Supported: DATA_PAGE v1 and v2, DICTIONARY_PAGE with PLAIN fallback inside
//! one chunk, PLAIN and the two dictionary encodings, UNCOMPRESSED, SNAPPY and
//! ZSTD. Everything else raises `Error::Invalid("parquet: ...")` naming the
//! feature. A null (definition level 0, or a v2 page with `num_nulls > 0`) is
//! an error: the library has none.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::parquet::codec;
use crate::parquet::column::{Column, Table, Utf8Builder};
use crate::parquet::crc32::crc32;
use crate::parquet::format::{
    self, encoding, footer_length, page_type, ColumnMetaData, DataPageHeader, DataPageHeaderV2,
    DictionaryPageHeader, FileMetaData, Leaf, PageHeader,
};
use crate::parquet::hybrid::{require_all_present, skip_present_levels_v1};
use crate::parquet::plain::{
    append_byte_array, append_dictionary_byte_array, decode_dictionary_f64, decode_f64,
    scan_byte_array, scan_dictionary_byte_array,
};
use crate::parquet::thrift::Reader;
use crate::parquet::{invalid, ColumnKind, Footer, MAGIC};
use crate::{product, Buffer, Context, Error, Reservation, Result};

/// Encrypted-footer magic (modular encryption); a different file, not ours.
const ENCRYPTED_MAGIC: &[u8; 4] = b"PARE";
/// Largest page header the reader parses (statistics can make one long).
const MAX_PAGE_HEADER_BYTES: usize = 1 << 20;
/// Working-memory allowance of the footer parse, per footer byte.
const FOOTER_PARSE_FACTOR: usize = 8;
/// Fixed allowance on top of it, for tiny footers.
const FOOTER_PARSE_FIXED: usize = 4096;
/// Bytes charged per key-value entry on top of its text for the sorted map
/// (two `String` headers, the tree node share, allocator rounding).
const METADATA_ENTRY_BYTES: usize = 96;
/// Bytes charged per column for what is built from the schema after the
/// footer parse: the `Leaf`, the output column, the final name and column.
const COLUMN_WRAPPER_BYTES: usize = 512;

/// True when the file starts with the `PAR1` magic.
pub fn is_parquet(path: &Path) -> Result<bool> {
    let mut file = open_file(path)?;
    let mut head = [0u8; 4];
    let mut filled = 0usize;
    while filled < head.len() {
        match file.read(&mut head[filled..]) {
            Ok(0) => return Ok(false),
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(invalid(format!("cannot read {}: {e}", path.display()))),
        }
    }
    Ok(&head == MAGIC)
}

/// Trailer and footer only; never touches a data page.
pub fn read_metadata(ctx: &Context, path: &Path) -> Result<Footer> {
    let opened = open(ctx, path)?;
    let num_rows =
        u64::try_from(opened.meta.num_rows).map_err(|_| invalid("num_rows is negative"))?;
    Ok(Footer {
        num_rows,
        num_row_groups: opened.meta.row_groups.len(),
        created_by: opened.meta.created_by.clone().unwrap_or_default(),
        columns: opened
            .leaves
            .iter()
            .map(|leaf| (leaf.name.clone(), leaf.kind))
            .collect(),
        metadata: opened.meta.metadata_map()?,
    })
}

/// Reads a whole file into a `Table`.
pub fn read_table(ctx: &Context, path: &Path) -> Result<Table> {
    let Opened {
        file,
        meta,
        leaves,
        _memory,
        _derived,
    } = open(ctx, path)?;
    if leaves.is_empty() {
        return Err(invalid("the file has no columns"));
    }
    let rows = usize::try_from(meta.num_rows).map_err(|_| Error::Overflow)?;
    let mut outputs = Vec::with_capacity(leaves.len());
    for leaf in &leaves {
        outputs.push(match leaf.kind {
            ColumnKind::Double => Output::Double(ctx.allocate::<f64>(rows)?),
            ColumnKind::Utf8 => Output::Utf8(Utf8Builder::new(ctx, rows)?),
        });
    }
    let mut starts = Vec::with_capacity(meta.row_groups.len());
    let mut first_row = 0usize;
    for group in &meta.row_groups {
        starts.push(first_row);
        first_row += usize::try_from(group.num_rows).map_err(|_| Error::Overflow)?;
    }
    let job = Job {
        ctx,
        meta: &meta,
        leaves: &leaves,
        starts: &starts,
    };
    let workers = worker_count(ctx, &meta, leaves.len());
    if workers <= 1 {
        let mut serial = || -> Result<()> {
            let mut scratch = (Scratch::default(), Scratch::default());
            for (index, output) in outputs.iter_mut().enumerate() {
                job.column(&file, index, output, &mut scratch)?;
            }
            Ok(())
        };
        if uses_zstd(&meta) {
            // One large-stack thread for the whole read: a thread per page
            // would cost more than a small page takes to decode.
            codec::on_large_stack(serial)??;
        } else {
            serial()?;
        }
    } else {
        job.parallel(&file, &mut outputs, workers)?;
    }
    let mut columns = Vec::with_capacity(leaves.len());
    for (leaf, output) in leaves.into_iter().zip(outputs) {
        let column = match output {
            Output::Double(values) => Column::Double(values),
            Output::Utf8(builder) => Column::Utf8(builder.finish(ctx)?),
        };
        columns.push((leaf.name, column));
    }
    Ok(Table {
        columns,
        metadata: meta.metadata_map()?,
    })
}

/// Whether any column chunk of the file is zstd-compressed.
fn uses_zstd(meta: &FileMetaData) -> bool {
    meta.row_groups.iter().any(|group| {
        group
            .columns
            .iter()
            .any(|chunk| chunk.meta_data.codec == format::codec::ZSTD)
    })
}

/// Says that a parse or validation failure is about the footer.
fn in_footer(error: Error) -> Error {
    match error {
        Error::Invalid(message) => Error::Invalid(match message.strip_prefix("parquet: ") {
            Some(rest) => format!("parquet: footer is invalid: {rest}"),
            None => format!("parquet: footer is invalid: {message}"),
        }),
        other => other,
    }
}

/// Fills `buffer` from byte `offset` of `file` without touching the file
/// cursor, so threads can share one handle.
#[cfg(unix)]
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buffer, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buffer: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    while !buffer.is_empty() {
        match std::os::windows::fs::FileExt::seek_read(file, buffer, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buffer = &mut buffer[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn open_file(path: &Path) -> Result<File> {
    File::open(path).map_err(|e| invalid(format!("cannot open {}: {e}", path.display())))
}

/// A file whose trailer and footer were read, parsed and validated.
struct Opened {
    file: File,
    meta: FileMetaData,
    leaves: Vec<Leaf>,
    /// The footer parse allowance, held while the parsed footer is alive.
    _memory: Reservation,
    /// What is derived from the footer after the parse (see `derived_bytes`).
    _derived: Reservation,
}

fn open(ctx: &Context, path: &Path) -> Result<Opened> {
    let mut file = open_file(path)?;
    let file_len = file
        .metadata()
        .map_err(|e| invalid(format!("cannot stat {}: {e}", path.display())))?
        .len();
    if file_len < 12 {
        return Err(invalid("not a Parquet file (shorter than 12 bytes)"));
    }
    let mut head = [0u8; 4];
    file.read_exact(&mut head)
        .map_err(|e| invalid(format!("cannot read the file header: {e}")))?;
    if &head == ENCRYPTED_MAGIC {
        return Err(invalid("encrypted Parquet files are not supported"));
    }
    if &head != MAGIC {
        return Err(invalid("not a Parquet file (missing leading PAR1 magic)"));
    }
    let mut tail = [0u8; 8];
    file.seek(SeekFrom::End(-8))
        .and_then(|_| file.read_exact(&mut tail))
        .map_err(|e| invalid(format!("cannot read the file trailer: {e}")))?;
    if &tail[4..] == ENCRYPTED_MAGIC {
        return Err(invalid("encrypted Parquet footers are not supported"));
    }
    let footer_len = footer_length(&tail, file_len)?;
    // Footer bytes plus the allowance its parse may charge. The parser counts
    // what it allocates against this limit, so the reservation is an upper
    // bound of the real use.
    let parse_limit = footer_len
        .checked_mul(FOOTER_PARSE_FACTOR)
        .and_then(|bytes| bytes.checked_add(FOOTER_PARSE_FIXED))
        .ok_or(Error::Overflow)?;
    let mut memory = ctx.reserve(footer_len.checked_add(parse_limit).ok_or(Error::Overflow)?)?;
    let data_end = file_len - 8 - footer_len as u64;
    let mut footer = Vec::new();
    footer
        .try_reserve_exact(footer_len)
        .map_err(|_| Error::Allocation)?;
    footer.resize(footer_len, 0);
    file.seek(SeekFrom::Start(data_end))
        .and_then(|_| file.read_exact(&mut footer))
        .map_err(|e| invalid(format!("cannot read the footer: {e}")))?;
    let mut reader = Reader::with_limit(&footer, parse_limit);
    let meta = FileMetaData::read(&mut reader)
        .and_then(|meta| reader.finish().map(|()| meta))
        .map_err(in_footer)?;
    // Keep reserved only what the parsed footer holds.
    memory.shrink_to(reader.charged().min(memory.bytes()))?;
    drop(footer);
    // What is built from the parsed footer (the validated leaves, the sorted
    // metadata map, the output columns) is charged before it is built, so a
    // read never allocates more than the context allows.
    let derived = derived_bytes(&meta)?;
    let derived_memory = ctx.reserve(derived)?;
    let leaves = meta.validate(data_end).map_err(in_footer)?;
    Ok(Opened {
        file,
        meta,
        leaves,
        _memory: memory,
        _derived: derived_memory,
    })
}

/// Upper bound of the bytes `validate`, `metadata_map` and the table assembly
/// allocate from a parsed footer.
fn derived_bytes(meta: &FileMetaData) -> Result<usize> {
    let mut total = 0usize;
    for kv in &meta.key_value_metadata {
        let text = kv
            .key
            .len()
            .checked_add(kv.value.as_ref().map_or(0, String::len))
            .ok_or(Error::Overflow)?;
        total = total
            .checked_add(text)
            .and_then(|t| t.checked_add(METADATA_ENTRY_BYTES))
            .ok_or(Error::Overflow)?;
    }
    for element in meta.schema.iter().skip(1) {
        total = element
            .name
            .len()
            .checked_mul(2)
            .and_then(|t| t.checked_add(COLUMN_WRAPPER_BYTES))
            .and_then(|t| t.checked_add(total))
            .ok_or(Error::Overflow)?;
    }
    Ok(total)
}

/// Most bytes the buffers of one column worker hold at the same time, for the
/// largest chunk of the file: the chunk, its decompressed page, and the zstd
/// decoder working set (an upper bound of `codec::zstd`'s charge).
fn worker_bytes(meta: &FileMetaData) -> usize {
    let mut worst = 0u64;
    for group in &meta.row_groups {
        for chunk in &group.columns {
            let m = &chunk.meta_data;
            let compressed = u64::try_from(m.total_compressed_size).unwrap_or(0);
            let uncompressed = u64::try_from(m.total_uncompressed_size).unwrap_or(0);
            let decoder = codec::zstd::charge_bound(uncompressed).map_or(u64::MAX, |b| b as u64);
            let bytes = compressed
                .saturating_add(uncompressed)
                .saturating_add(decoder);
            worst = worst.max(bytes);
        }
    }
    usize::try_from(worst).unwrap_or(usize::MAX)
}

/// Threads that decode columns: at most `ctx.threads()` and one per column,
/// and few enough that the buffers of all of them fit what the budget has
/// left (one worker always runs, as before).
fn worker_count(ctx: &Context, meta: &FileMetaData, columns: usize) -> usize {
    let wanted = ctx.threads().min(columns);
    if wanted <= 1 {
        return 1;
    }
    let available = ctx.memory_limit().saturating_sub(ctx.memory_used());
    let each = worker_bytes(meta).max(1);
    (available / each).clamp(1, wanted)
}

/// What decoding the columns of one file needs.
struct Job<'a> {
    ctx: &'a Context,
    meta: &'a FileMetaData,
    leaves: &'a [Leaf],
    /// First table row of each row group.
    starts: &'a [usize],
}

impl Job<'_> {
    /// Decodes column `index` over every row group into `output`.
    fn column(
        &self,
        file: &File,
        index: usize,
        output: &mut Output,
        scratch: &mut (Scratch, Scratch),
    ) -> Result<()> {
        for (group, &first_row) in self.meta.row_groups.iter().zip(self.starts) {
            let rows = usize::try_from(group.num_rows).map_err(|_| Error::Overflow)?;
            let mut reader = ChunkReader {
                ctx: self.ctx,
                leaf: &self.leaves[index],
                meta: &group.columns[index].meta_data,
                rows,
                first_row,
                output: &mut *output,
            };
            reader.read(file, &mut scratch.0, &mut scratch.1)?;
        }
        Ok(())
    }

    /// Decodes the columns on `workers` threads that pull the next undecoded
    /// column. All threads read the one file the footer was validated against,
    /// through positional reads (no shared cursor), so a file replaced while
    /// this runs cannot mix its bytes with the old footer. Each thread has its
    /// own buffers. The error of the first failing column (in file order) is
    /// the one returned.
    fn parallel(&self, file: &File, outputs: &mut [Output], workers: usize) -> Result<()> {
        let slots: Vec<Mutex<&mut Output>> = outputs.iter_mut().map(Mutex::new).collect();
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let errors: Mutex<Vec<(usize, Error)>> = Mutex::new(Vec::new());
        let work = || {
            let mut scratch = (Scratch::default(), Scratch::default());
            while !failed.load(Ordering::Relaxed) {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(slot) = slots.get(index) else { break };
                let mut output = lock(slot);
                if let Err(error) = self.column(file, index, &mut output, &mut scratch) {
                    failed.store(true, Ordering::Relaxed);
                    lock(&errors).push((index, error));
                }
            }
        };
        let outcome = std::thread::scope(|scope| -> Result<()> {
            let mut handles = Vec::with_capacity(workers);
            for n in 0..workers {
                match codec::spawn_large(scope, &format!("mandacaru-parquet-{n}"), work) {
                    Ok(handle) => handles.push(handle),
                    // A thread that cannot start is not fatal while another runs.
                    Err(error) if handles.is_empty() => return Err(error),
                    Err(_) => break,
                }
            }
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| invalid("a column decoder failed (internal failure)"))?;
            }
            Ok(())
        });
        outcome?;
        let mut errors = errors.into_inner().unwrap_or_else(|e| e.into_inner());
        errors.sort_by_key(|(index, _)| *index);
        match errors.into_iter().next() {
            Some((_, error)) => Err(error),
            None => Ok(()),
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// The column being filled.
enum Output {
    Double(Buffer<f64>),
    Utf8(Utf8Builder),
}

/// A reusable byte buffer whose size is reserved from the `Context`.
#[derive(Default)]
struct Scratch {
    bytes: Vec<u8>,
    memory: Option<Reservation>,
}

impl Scratch {
    /// `n` bytes of storage, growing (and reserving the growth) when needed.
    fn ensure(&mut self, ctx: &Context, n: usize) -> Result<&mut [u8]> {
        if self.bytes.len() < n {
            if n > isize::MAX as usize {
                return Err(Error::Overflow);
            }
            // Reserve first, while the old reservation is still held.
            let memory = ctx.reserve(n)?;
            self.bytes = Vec::new();
            self.bytes
                .try_reserve_exact(n)
                .map_err(|_| Error::Allocation)?;
            self.bytes.resize(n, 0);
            self.memory = Some(memory);
        }
        Ok(&mut self.bytes[..n])
    }
}

/// A dictionary page, decoded; it lives until the end of its chunk.
enum Dictionary {
    Double {
        values: Vec<f64>,
        _memory: Reservation,
    },
    Utf8 {
        data: Vec<u8>,
        offsets: Vec<u64>,
        _memory: Reservation,
    },
}

struct ChunkReader<'a> {
    ctx: &'a Context,
    leaf: &'a Leaf,
    meta: &'a ColumnMetaData,
    /// Rows of the row group, hence values of the chunk.
    rows: usize,
    /// Row of the whole table where this chunk starts.
    first_row: usize,
    output: &'a mut Output,
}

/// A page body as stored in the chunk, with what its header says about it.
struct RawPage<'b> {
    codec_number: i32,
    body: &'b [u8],
    uncompressed: usize,
}

/// The decompressed bytes of a page body: the body itself for an uncompressed
/// chunk, `pages` otherwise.
///
/// `limit` is the most bytes the values of the page can need; a compressed
/// page that claims more is refused before any byte is decompressed, so a
/// small file cannot make the decoder produce far more than it can use.
fn payload<'b>(
    ctx: &Context,
    codec_number: i32,
    body: &'b [u8],
    uncompressed: usize,
    limit: Option<usize>,
    pages: &'b mut Scratch,
) -> Result<&'b [u8]> {
    if codec_number == format::codec::UNCOMPRESSED {
        if body.len() != uncompressed {
            return Err(invalid(format!(
                "uncompressed page holds {} bytes but its header says {uncompressed}",
                body.len()
            )));
        }
        return Ok(body);
    }
    if let Some(limit) = limit {
        if uncompressed > limit {
            return Err(invalid(format!(
                "a compressed page claims {uncompressed} uncompressed bytes where its values need at most {limit}"
            )));
        }
    }
    // The claim is checked against the body before any buffer of that size
    // exists: a few bytes of body cannot make the reader reserve and
    // zero-fill what the header says.
    let claim = codec::claim(codec_number, body, uncompressed)?;
    let output = pages.ensure(ctx, uncompressed)?;
    claim.decompress(ctx, body, output)?;
    Ok(&pages.bytes[..uncompressed])
}

/// Bytes a page can hold beyond its values: fastparquet pads with 8.
const PAGE_SLACK: usize = 64;
/// Most bytes one value takes in a page, levels or dictionary index included.
const VALUE_BYTES: usize = 16;

fn non_negative(value: i32, what: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid(format!("{what} is negative ({value})")))
}

fn unsupported_codec(number: i32) -> crate::Error {
    invalid(format!(
        "{} is not supported; re-save the file with zstd",
        format::codec::name(number)
    ))
}

impl ChunkReader<'_> {
    fn read(&mut self, file: &File, chunk_bytes: &mut Scratch, pages: &mut Scratch) -> Result<()> {
        let codec_number = self.meta.codec;
        if !matches!(
            codec_number,
            format::codec::UNCOMPRESSED | format::codec::SNAPPY | format::codec::ZSTD
        ) {
            return Err(unsupported_codec(codec_number));
        }
        let name = &self.leaf.name;
        if self.rows == 0 {
            // An empty chunk holds no value, so nothing is read or
            // decompressed (pyarrow gives it offset 0, or an empty dictionary
            // page). The footer check already placed its range in the file.
            return Ok(());
        }
        let start = u64::try_from(self.meta.start_offset())
            .map_err(|_| invalid(format!("column {name:?} has a negative offset")))?;
        let size = usize::try_from(self.meta.total_compressed_size)
            .map_err(|_| invalid(format!("column {name:?} has a negative size")))?;
        let buffer = chunk_bytes.ensure(self.ctx, size)?;
        read_exact_at(file, buffer, start)
            .map_err(|e| invalid(format!("cannot read column {name:?}: {e}")))?;
        let chunk: &[u8] = &chunk_bytes.bytes[..size];

        let mut position = 0usize;
        let mut done = 0usize;
        let mut dictionary: Option<Dictionary> = None;
        while position < chunk.len() {
            let window = &chunk[position..chunk.len().min(position + MAX_PAGE_HEADER_BYTES)];
            let (header, header_len) = PageHeader::from_bytes(window)?;
            position += header_len;
            let compressed = non_negative(header.compressed_page_size, "compressed_page_size")?;
            let uncompressed =
                non_negative(header.uncompressed_page_size, "uncompressed_page_size")?;
            let Some(body) = position
                .checked_add(compressed)
                .and_then(|end| chunk.get(position..end))
            else {
                return Err(invalid(format!(
                    "a page of column {name:?} extends past the end of its column chunk"
                )));
            };
            position += compressed;
            if let Some(expected) = header.crc {
                if crc32(body) as i32 != expected {
                    return Err(invalid(format!(
                        "page checksum mismatch in column {name:?}"
                    )));
                }
            }
            match header.page_type {
                page_type::DICTIONARY_PAGE => {
                    let page = header.dictionary_page.as_ref().ok_or_else(|| {
                        invalid("dictionary page without a dictionary page header")
                    })?;
                    if dictionary.is_some() || done > 0 {
                        return Err(invalid(format!(
                            "column {name:?} has a dictionary page that is repeated or follows data pages"
                        )));
                    }
                    let limit = match self.leaf.kind {
                        ColumnKind::Double => {
                            non_negative(page.num_values, "dictionary num_values")?
                                .checked_mul(8)
                                .and_then(|bytes| bytes.checked_add(PAGE_SLACK))
                        }
                        // Checked after the scan (`load_dictionary`).
                        ColumnKind::Utf8 => None,
                    };
                    let bytes = payload(self.ctx, codec_number, body, uncompressed, limit, pages)?;
                    dictionary = Some(self.load_dictionary(page, bytes)?);
                }
                page_type::DATA_PAGE => {
                    let page = header
                        .data_page
                        .as_ref()
                        .ok_or_else(|| invalid("data page without a data page header"))?;
                    done += self.data_page_v1(
                        page,
                        RawPage {
                            codec_number,
                            body,
                            uncompressed,
                        },
                        pages,
                        done,
                        dictionary.as_ref(),
                    )?;
                }
                page_type::DATA_PAGE_V2 => {
                    let page = header
                        .data_page_v2
                        .as_ref()
                        .ok_or_else(|| invalid("data page v2 without a data page header"))?;
                    done += self.data_page_v2(
                        page,
                        RawPage {
                            codec_number,
                            body,
                            uncompressed,
                        },
                        pages,
                        done,
                        dictionary.as_ref(),
                    )?;
                }
                other => {
                    return Err(invalid(format!(
                        "{} pages are not supported",
                        page_type::name(other)
                    )))
                }
            }
        }
        if done != self.rows {
            return Err(invalid(format!(
                "column {name:?} holds {done} values in its pages but its row group has {} rows",
                self.rows
            )));
        }
        Ok(())
    }

    fn data_page_v1(
        &mut self,
        page: &DataPageHeader,
        raw: RawPage<'_>,
        pages: &mut Scratch,
        done: usize,
        dictionary: Option<&Dictionary>,
    ) -> Result<usize> {
        let RawPage {
            codec_number,
            body,
            uncompressed,
        } = raw;
        let count = self.page_count(page.num_values, done)?;
        if count == 0 {
            // No value to decode: do not decompress what claims to hold none.
            return Ok(0);
        }
        let ctx = self.ctx;
        let optional = self.leaf.optional;
        let limit = self.page_limit(page.encoding, count);
        let page_bytes = payload(ctx, codec_number, body, uncompressed, limit, pages)?;
        let values = if optional {
            if page.definition_level_encoding != encoding::RLE {
                return Err(invalid(format!(
                    "definition levels in {} are not supported",
                    encoding::name(page.definition_level_encoding)
                )));
            }
            skip_present_levels_v1(page_bytes, count)?
        } else {
            page_bytes
        };
        let at = self.first_row + done;
        decode_values(
            ctx,
            self.output,
            at,
            count,
            page.encoding,
            values,
            dictionary,
        )?;
        Ok(count)
    }

    fn data_page_v2(
        &mut self,
        page: &DataPageHeaderV2,
        raw: RawPage<'_>,
        pages: &mut Scratch,
        done: usize,
        dictionary: Option<&Dictionary>,
    ) -> Result<usize> {
        let RawPage {
            codec_number,
            body,
            uncompressed,
        } = raw;
        let count = self.page_count(page.num_values, done)?;
        if page.num_nulls != 0 {
            return Err(invalid(format!(
                "null values are not supported (a data page v2 of column {:?} reports {} nulls)",
                self.leaf.name, page.num_nulls
            )));
        }
        if page.num_rows != page.num_values {
            return Err(invalid(
                "a data page v2 of a flat column must hold one value per row",
            ));
        }
        let repetition_bytes = non_negative(
            page.repetition_levels_byte_length,
            "repetition level length",
        )?;
        let definition_bytes = non_negative(
            page.definition_levels_byte_length,
            "definition level length",
        )?;
        if repetition_bytes != 0 {
            return Err(invalid(
                "repeated columns are not supported (repetition levels found)",
            ));
        }
        if body.len() < definition_bytes || uncompressed < definition_bytes {
            return Err(invalid(
                "a data page v2 is shorter than its definition levels",
            ));
        }
        if self.leaf.optional {
            if definition_bytes == 0 && count > 0 {
                return Err(invalid(
                    "an optional column page carries no definition levels",
                ));
            }
            require_all_present(&body[..definition_bytes], count)?;
        } else if definition_bytes != 0 {
            return Err(invalid("a required column page carries definition levels"));
        }
        let values = &body[definition_bytes..];
        let values_len = uncompressed - definition_bytes;
        let ctx = self.ctx;
        let compressed = page.is_compressed && codec_number != format::codec::UNCOMPRESSED;
        if count == 0 {
            // No value to decode: do not decompress what claims to hold none.
            return Ok(0);
        }
        let limit = self.page_limit(page.encoding, count);
        let payload = payload(
            ctx,
            if compressed {
                codec_number
            } else {
                format::codec::UNCOMPRESSED
            },
            values,
            values_len,
            limit,
            pages,
        )?;
        let at = self.first_row + done;
        decode_values(
            ctx,
            self.output,
            at,
            count,
            page.encoding,
            payload,
            dictionary,
        )?;
        Ok(count)
    }

    /// Most uncompressed bytes a data page of `count` values can need, or
    /// `None` for PLAIN text, whose size has no bound but its content (the
    /// decode checks that the page holds no more than its values).
    fn page_limit(&self, page_encoding: i32, count: usize) -> Option<usize> {
        if self.leaf.kind == ColumnKind::Utf8 && page_encoding == encoding::PLAIN {
            return None;
        }
        count
            .checked_mul(VALUE_BYTES)
            .and_then(|bytes| bytes.checked_add(PAGE_SLACK))
    }

    /// Values a data page declares, checked against the rows still missing.
    fn page_count(&self, declared: i32, done: usize) -> Result<usize> {
        let count = non_negative(declared, "page num_values")?;
        if count > self.rows - done {
            return Err(invalid(format!(
                "a page of column {:?} holds {count} values but only {} rows are missing",
                self.leaf.name,
                self.rows - done
            )));
        }
        Ok(count)
    }

    fn load_dictionary(&self, page: &DictionaryPageHeader, payload: &[u8]) -> Result<Dictionary> {
        if page.encoding != encoding::PLAIN && page.encoding != encoding::PLAIN_DICTIONARY {
            return Err(invalid(format!(
                "a dictionary page in {} is not supported",
                encoding::name(page.encoding)
            )));
        }
        let entries = non_negative(page.num_values, "dictionary num_values")?;
        match self.leaf.kind {
            ColumnKind::Double => {
                let bytes = entries
                    .checked_mul(8)
                    .filter(|&bytes| bytes <= payload.len())
                    .ok_or_else(|| {
                        invalid(format!(
                            "dictionary of {entries} doubles does not fit in a page of {} bytes",
                            payload.len()
                        ))
                    })?;
                let memory = self.ctx.reserve(bytes)?;
                let mut values = Vec::new();
                values
                    .try_reserve_exact(entries)
                    .map_err(|_| Error::Allocation)?;
                values.resize(entries, 0.0);
                decode_f64(payload, &mut values)?;
                Ok(Dictionary::Double {
                    values,
                    _memory: memory,
                })
            }
            ColumnKind::Utf8 => {
                // Every value needs at least its 4-byte length prefix.
                if entries > payload.len() / 4 {
                    return Err(invalid(format!(
                        "dictionary of {entries} strings does not fit in a page of {} bytes",
                        payload.len()
                    )));
                }
                let scan = scan_byte_array(payload, entries)?;
                no_surplus(payload.len(), scan.consumed)?;
                let slots = entries + 1;
                let bytes = product(&[slots, 8])?
                    .checked_add(scan.bytes)
                    .ok_or(Error::Overflow)?;
                let memory = self.ctx.reserve(bytes)?;
                let mut data = Vec::new();
                data.try_reserve_exact(scan.bytes)
                    .map_err(|_| Error::Allocation)?;
                let mut offsets = Vec::new();
                offsets
                    .try_reserve_exact(slots)
                    .map_err(|_| Error::Allocation)?;
                append_byte_array(payload, entries, &mut data, &mut offsets)?;
                Ok(Dictionary::Utf8 {
                    data,
                    offsets,
                    _memory: memory,
                })
            }
        }
    }
}

/// Refuses a page that holds more than `PAGE_SLACK` bytes beyond the values it
/// declares: a decompressed page that is mostly unused is a lying size.
fn no_surplus(page_len: usize, consumed: usize) -> Result<()> {
    if page_len.saturating_sub(consumed) > PAGE_SLACK {
        return Err(invalid(format!(
            "a text page holds {page_len} bytes but its values end at byte {consumed}"
        )));
    }
    Ok(())
}

/// Decode `count` values of one page into rows `at..at + count` of `output`.
fn decode_values(
    ctx: &Context,
    output: &mut Output,
    at: usize,
    count: usize,
    page_encoding: i32,
    payload: &[u8],
    dictionary: Option<&Dictionary>,
) -> Result<()> {
    let dictionary_encoded =
        page_encoding == encoding::PLAIN_DICTIONARY || page_encoding == encoding::RLE_DICTIONARY;
    if page_encoding != encoding::PLAIN && !dictionary_encoded {
        return Err(invalid(format!(
            "the {} encoding is not supported (PLAIN and dictionary encodings only)",
            encoding::name(page_encoding)
        )));
    }
    match output {
        Output::Double(buffer) => {
            let slot = buffer
                .as_mut_slice()
                .get_mut(at..at + count)
                .ok_or_else(|| invalid("page values extend past the column"))?;
            if !dictionary_encoded {
                decode_f64(payload, slot)?;
            } else {
                let Some(Dictionary::Double { values, .. }) = dictionary else {
                    return Err(invalid(
                        "a dictionary-encoded page has no dictionary page before it",
                    ));
                };
                decode_dictionary_f64(payload, values, slot)?;
            }
        }
        Output::Utf8(builder) => {
            if builder.len() != at {
                return Err(invalid("string pages arrived out of order"));
            }
            if !dictionary_encoded {
                let scan = scan_byte_array(payload, count)?;
                no_surplus(payload.len(), scan.consumed)?;
                builder.reserve_data(ctx, scan.bytes)?;
                let (data, offsets) = builder.vectors();
                append_byte_array(payload, count, data, offsets)?;
            } else {
                let Some(Dictionary::Utf8 { data, offsets, .. }) = dictionary else {
                    return Err(invalid(
                        "a dictionary-encoded page has no dictionary page before it",
                    ));
                };
                let bytes = scan_dictionary_byte_array(payload, count, data, offsets)?;
                builder.reserve_data(ctx, bytes)?;
                let (out_data, out_offsets) = builder.vectors();
                append_dictionary_byte_array(payload, count, data, offsets, out_data, out_offsets)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::format::{physical, ColumnChunk, KeyValue, RowGroup, SchemaElement};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn fixtures() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/migration/reference/parquet")
    }

    fn ctx() -> Context {
        Context::new(Some(1), 1 << 28).unwrap()
    }

    // ---- SHA-256, for the frozen digests ----

    fn sha256(data: &[u8]) -> String {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let mut message = data.to_vec();
        message.push(0x80);
        while message.len() % 64 != 56 {
            message.push(0);
        }
        message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
        for block in message.chunks_exact(64) {
            let mut w = [0u32; 64];
            for (i, word) in block.chunks_exact(4).enumerate() {
                w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let mut v = h;
            for i in 0..64 {
                let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
                let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
                let t1 = v[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
                let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
                let t2 = s0.wrapping_add(maj);
                v = [
                    t1.wrapping_add(t2),
                    v[0],
                    v[1],
                    v[2],
                    v[3].wrapping_add(t1),
                    v[4],
                    v[5],
                    v[6],
                ];
            }
            for (slot, value) in h.iter_mut().zip(v) {
                *slot = slot.wrapping_add(value);
            }
        }
        h.iter().map(|word| format!("{word:08x}")).collect()
    }

    #[test]
    fn the_sha256_helper_matches_known_digests() {
        assert_eq!(
            sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    fn double_digest(column: &Column) -> String {
        let bytes: Vec<u8> = column
            .as_f64()
            .unwrap()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        sha256(&bytes)
    }

    fn string_digest(column: &Column) -> String {
        let values: Vec<&str> = column.as_utf8().unwrap().iter().collect();
        sha256(values.join("\0").as_bytes())
    }

    fn references() -> serde_json::Value {
        let text = std::fs::read_to_string(fixtures().join("references.json")).unwrap();
        serde_json::from_str(&text).unwrap()
    }

    // ---- the frozen fixtures ----

    #[test]
    fn every_frozen_fixture_reads_or_raises_as_recorded() {
        let references = references();
        let mut reads = 0;
        let mut raised = 0;
        let mut hints_missed = Vec::new();
        for (name, entry) in references["fixtures"].as_object().unwrap() {
            let path = fixtures().join(name);
            let c = ctx();
            // Files that are valid Parquet and fail only the Hamiltonian schema
            // check (WP6): the generic reader reads them.
            let semantic = name.starts_with("negative/hamiltonian_")
                || name == "negative/column_double_pauli.parquet";
            let expect_ok = entry["expect_new_reader"] == "ok" || semantic;
            if name == "negative/footer_rows_2e40.parquet" {
                let small = Context::new(Some(1), 1 << 20).unwrap();
                let err = read_table(&small, &path).unwrap_err();
                assert!(matches!(err, Error::Budget { .. }), "{name}: {err:?}");
                assert_eq!(small.memory_used(), 0);
                raised += 1;
                continue;
            }
            let outcome = read_table(&c, &path);
            if !expect_ok {
                let err = outcome.map(|_| ()).expect_err(name);
                assert!(matches!(err, Error::Invalid(_)), "{name}: {err:?}");
                let text = err.to_string().to_lowercase();
                assert!(text.starts_with("parquet: "), "{name}: {text}");
                let hints = entry["new_reader_error_hint"].as_array().unwrap();
                if !hints
                    .iter()
                    .any(|h| text.contains(&h.as_str().unwrap().to_lowercase()))
                {
                    hints_missed.push(format!("{name}: {text}"));
                }
                raised += 1;
            } else {
                let table = outcome.unwrap_or_else(|e| panic!("{name}: {e}"));
                if let Some(decoded) = entry.get("decoded_by_pyarrow") {
                    let decoded = decoded.as_object().unwrap();
                    if !semantic {
                        assert_eq!(table.columns.len(), decoded.len(), "{name}");
                    }
                    for (column_name, expect) in decoded {
                        let column = table
                            .column(column_name)
                            .unwrap_or_else(|| panic!("{name}: missing {column_name}"));
                        assert_eq!(
                            column.len() as u64,
                            expect["rows"].as_u64().unwrap(),
                            "{name} {column_name}"
                        );
                        if let Some(want) = expect.get("sha256_float64_bytes") {
                            assert_eq!(
                                double_digest(column),
                                want.as_str().unwrap(),
                                "{name} {column_name}"
                            );
                        } else {
                            assert_eq!(
                                string_digest(column),
                                expect["sha256_utf8_nul_joined"].as_str().unwrap(),
                                "{name} {column_name}"
                            );
                        }
                    }
                }
                // Key-value metadata equals what pyarrow reports.
                let mut expected_kv = BTreeMap::new();
                for (key, value) in entry["kv"].as_object().unwrap() {
                    match value {
                        serde_json::Value::String(text) => {
                            expected_kv.insert(key.clone(), text.clone());
                        }
                        other => {
                            let got = &table.metadata[key];
                            assert_eq!(got.len() as u64, other["bytes"].as_u64().unwrap());
                            assert_eq!(sha256(got.as_bytes()), other["sha256"].as_str().unwrap());
                            expected_kv.insert(key.clone(), got.clone());
                        }
                    }
                }
                assert_eq!(table.metadata, expected_kv, "{name}");
                // read_metadata agrees with the table.
                let footer = read_metadata(&c, &path).unwrap();
                assert_eq!(footer.num_rows as usize, table.num_rows(), "{name}");
                assert_eq!(footer.metadata, table.metadata, "{name}");
                assert_eq!(footer.columns.len(), table.columns.len(), "{name}");
                reads += 1;
            }
            assert_eq!(c.memory_used(), 0, "{name}");
        }
        assert!(
            reads >= 40 && raised >= 40,
            "reads {reads}, raised {raised}"
        );
        // The hints in references.json are informative, not binding.
        for missed in &hints_missed {
            eprintln!("error text does not repeat the recorded hint: {missed}");
        }
    }

    #[test]
    fn data_page_variants_are_all_covered_by_the_fixtures() {
        // The fixtures must exercise v1, v2, dictionary pages, the PLAIN
        // fallback, several pages per chunk, several row groups and all three
        // codecs, so the digests above prove every reader path.
        let references = references();
        let mut seen = std::collections::BTreeSet::new();
        for (name, entry) in references["fixtures"].as_object().unwrap() {
            if entry["expect_new_reader"] != "ok" {
                continue;
            }
            for page in entry["page_summary"].as_array().unwrap() {
                seen.insert(page.as_str().unwrap().to_owned());
            }
            for chunk in entry["structure"]["chunks"].as_array().unwrap() {
                seen.insert(format!("codec:{}", chunk["codec"].as_str().unwrap()));
            }
            if entry["structure"]["num_row_groups"].as_u64().unwrap() > 1 {
                seen.insert("many row groups".into());
            }
            if entry["max_pages_per_chunk"].as_u64().unwrap_or(0) > 2 {
                seen.insert(
                    format!("many pages ({name})")
                        .split(" (")
                        .next()
                        .unwrap()
                        .into(),
                );
            }
        }
        for want in [
            "DATA_PAGE:PLAIN",
            "DATA_PAGE:RLE_DICTIONARY",
            "DATA_PAGE:PLAIN_DICTIONARY",
            "DATA_PAGE_V2:PLAIN",
            "DATA_PAGE_V2:RLE_DICTIONARY",
            "DICTIONARY_PAGE:PLAIN",
            "DICTIONARY_PAGE:PLAIN_DICTIONARY",
            "codec:ZSTD",
            "codec:SNAPPY",
            "codec:UNCOMPRESSED",
            "many row groups",
            "many pages",
        ] {
            assert!(seen.contains(want), "no fixture covers {want}: {seen:?}");
        }
    }

    #[test]
    fn the_zero_row_files_keep_their_columns_and_metadata() {
        for name in [
            "hamiltonian/fastparquet_empty.parquet",
            "hamiltonian/pyarrow_empty.parquet",
        ] {
            let c = ctx();
            let table = read_table(&c, &fixtures().join(name)).unwrap();
            assert_eq!(table.num_rows(), 0, "{name}");
            assert_eq!(table.column_names(), ["pauli", "real", "imag"]);
            assert!(
                table.metadata.contains_key("mandacaru.num_qubits"),
                "{name}"
            );
            let footer = read_metadata(&c, &fixtures().join(name)).unwrap();
            assert_eq!(footer.num_rows, 0);
        }
    }

    #[test]
    fn read_metadata_never_touches_a_data_page() {
        let source = fixtures().join("hamiltonian/fastparquet_zstd.parquet");
        let mut bytes = std::fs::read(&source).unwrap();
        let tail: [u8; 8] = bytes[bytes.len() - 8..].try_into().unwrap();
        let footer = u32::from_le_bytes(tail[..4].try_into().unwrap()) as usize;
        let data_end = bytes.len() - 8 - footer;
        for byte in &mut bytes[4..data_end] {
            *byte ^= 0xA5;
        }
        let dir =
            std::env::temp_dir().join(format!("mandacaru-parquet-read-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corrupt.parquet");
        std::fs::write(&path, &bytes).unwrap();
        let c = ctx();
        let header = read_metadata(&c, &path).unwrap();
        let clean = read_metadata(&c, &source).unwrap();
        assert_eq!(header, clean);
        assert!(
            read_table(&c, &path).is_err(),
            "the corrupted pages must not decode"
        );
        assert_eq!(c.memory_used(), 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_two_to_the_forty_row_footer_is_refused_by_the_budget() {
        let path = fixtures().join("negative/footer_rows_2e40.parquet");
        let c = Context::new(Some(1), 1 << 20).unwrap();
        let err = read_table(&c, &path).unwrap_err();
        assert!(matches!(err, Error::Budget { .. }), "{err:?}");
        assert_eq!(c.memory_used(), 0);
        // The footer alone is fine to read: no row is allocated.
        assert_eq!(read_metadata(&c, &path).unwrap().num_rows, 1 << 40);
    }

    #[test]
    fn a_tiny_budget_refuses_an_ordinary_file_and_leaves_no_residue() {
        let path = fixtures().join("hamiltonian/pyarrow_multi_page_plain.parquet");
        for limit in [1usize, 64, 1 << 10, 8 << 10] {
            let c = Context::new(Some(1), limit).unwrap();
            match read_table(&c, &path) {
                Err(Error::Budget { .. }) => {}
                Ok(table) => drop(table),
                Err(other) => panic!("limit {limit}: {other:?}"),
            }
            assert_eq!(c.memory_used(), 0, "limit {limit}");
        }
    }

    // ---- hand-built files ----

    /// One column chunk: its raw pages (header + body) and the chunk metadata.
    struct Chunk {
        name: &'static str,
        physical: i32,
        codec: i32,
        rows: i64,
        pages: Vec<(PageHeader, Vec<u8>)>,
        optional: bool,
    }

    fn page(
        page_type: i32,
        body: Vec<u8>,
        uncompressed: usize,
        data: Option<DataPageHeader>,
        dictionary: Option<DictionaryPageHeader>,
    ) -> (PageHeader, Vec<u8>) {
        (
            PageHeader {
                page_type,
                uncompressed_page_size: uncompressed as i32,
                compressed_page_size: body.len() as i32,
                crc: None,
                data_page: data,
                dictionary_page: dictionary,
                data_page_v2: None,
            },
            body,
        )
    }

    fn plain_page(rows: i32, body: Vec<u8>) -> (PageHeader, Vec<u8>) {
        let size = body.len();
        page(
            page_type::DATA_PAGE,
            body,
            size,
            Some(DataPageHeader {
                num_values: rows,
                encoding: encoding::PLAIN,
                definition_level_encoding: encoding::RLE,
                repetition_level_encoding: encoding::RLE,
            }),
            None,
        )
    }

    fn doubles(values: &[f64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn strings(values: &[&str]) -> Vec<u8> {
        let mut out = Vec::new();
        for value in values {
            out.extend_from_slice(&(value.len() as u32).to_le_bytes());
            out.extend_from_slice(value.as_bytes());
        }
        out
    }

    /// A single row group file from raw chunks.
    fn build(chunks: Vec<Chunk>, metadata: &[(&str, &str)]) -> Vec<u8> {
        let mut file = MAGIC.to_vec();
        let mut schema = vec![SchemaElement::root(chunks.len()).unwrap()];
        let mut columns = Vec::new();
        let rows = chunks.first().map_or(0, |c| c.rows);
        for chunk in &chunks {
            let mut leaf = SchemaElement::leaf(
                chunk.name,
                chunk.physical,
                chunk.physical == physical::BYTE_ARRAY,
            );
            if chunk.optional {
                leaf.repetition = Some(format::repetition::OPTIONAL);
            }
            schema.push(leaf);
            let start = file.len();
            let mut dictionary_offset = None;
            let mut first_data = None;
            for (header, body) in &chunk.pages {
                if header.page_type == page_type::DICTIONARY_PAGE {
                    dictionary_offset = Some(file.len() as i64);
                } else if first_data.is_none() {
                    first_data = Some(file.len() as i64);
                }
                file.extend_from_slice(&header.to_bytes().unwrap());
                file.extend_from_slice(body);
            }
            let size = (file.len() - start) as i64;
            columns.push(ColumnChunk {
                file_path: None,
                file_offset: file.len() as i64,
                meta_data: ColumnMetaData {
                    physical_type: chunk.physical,
                    encodings: vec![encoding::PLAIN, encoding::RLE],
                    path_in_schema: vec![chunk.name.to_owned()],
                    codec: chunk.codec,
                    num_values: chunk.rows,
                    total_uncompressed_size: size,
                    total_compressed_size: size,
                    key_value_metadata: vec![],
                    data_page_offset: first_data.unwrap_or(start as i64),
                    index_page_offset: None,
                    dictionary_page_offset: dictionary_offset,
                },
            });
        }
        let meta = FileMetaData {
            version: 1,
            schema,
            num_rows: rows,
            row_groups: vec![RowGroup {
                columns,
                total_byte_size: 0,
                num_rows: rows,
            }],
            key_value_metadata: metadata
                .iter()
                .map(|(k, v)| KeyValue {
                    key: (*k).into(),
                    value: Some((*v).into()),
                })
                .collect(),
            created_by: Some("test".into()),
        };
        let footer = meta.to_bytes().unwrap();
        file.extend_from_slice(&footer);
        file.extend_from_slice(&(footer.len() as u32).to_le_bytes());
        file.extend_from_slice(MAGIC);
        file
    }

    fn write_temp(name: &str, bytes: &[u8]) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "mandacaru-parquet-read-{}-{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.parquet");
        std::fs::write(&path, bytes).unwrap();
        (dir, path)
    }

    /// Reads `bytes` as a file; also returns the memory still reserved while
    /// the outcome is alive.
    fn read_bytes(name: &str, bytes: &[u8]) -> (Result<Table>, usize) {
        let (dir, path) = write_temp(name, bytes);
        let c = ctx();
        let outcome = read_table(&c, &path);
        let leaked = c.memory_used();
        std::fs::remove_dir_all(dir).unwrap();
        (outcome, leaked)
    }

    fn simple(rows: i64, pages: Vec<(PageHeader, Vec<u8>)>) -> Vec<u8> {
        build(
            vec![Chunk {
                name: "d",
                physical: physical::DOUBLE,
                codec: format::codec::UNCOMPRESSED,
                rows,
                pages,
                optional: false,
            }],
            &[("k", "v")],
        )
    }

    #[test]
    fn a_hand_built_plain_file_reads_bit_exact() {
        let values = [
            0.0,
            -0.0,
            1.5,
            f64::NAN,
            f64::INFINITY,
            f64::MIN_POSITIVE / 4.0,
        ];
        let bytes = simple(
            6,
            vec![
                plain_page(4, doubles(&values[..4])),
                plain_page(2, doubles(&values[4..])),
            ],
        );
        let (outcome, _) = read_bytes("plain", &bytes);
        let table = outcome.unwrap();
        let got = table.column("d").unwrap().as_f64().unwrap();
        for (a, b) in got.iter().zip(values) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(table.metadata["k"], "v");
    }

    #[test]
    fn trailing_pad_bytes_are_tolerated_and_missing_bytes_are_not() {
        let mut body = doubles(&[1.0, 2.0]);
        body.extend_from_slice(&[0u8; 8]);
        let (ok, _) = read_bytes("pad", &simple(2, vec![plain_page(2, body)]));
        assert_eq!(
            ok.unwrap().column("d").unwrap().as_f64().unwrap(),
            [1.0, 2.0]
        );
        let short = doubles(&[1.0, 2.0])[..15].to_vec();
        let (err, _) = read_bytes("short", &simple(2, vec![plain_page(2, short)]));
        assert!(err.is_err());
    }

    #[test]
    fn page_value_counts_must_add_up_to_the_rows() {
        // Fewer values than rows.
        let (err, _) = read_bytes("few", &simple(3, vec![plain_page(2, doubles(&[1.0, 2.0]))]));
        assert!(err.unwrap_err().to_string().contains("rows"));
        // More values than rows.
        let (err, _) = read_bytes(
            "many",
            &simple(2, vec![plain_page(3, doubles(&[1.0, 2.0, 3.0]))]),
        );
        assert!(err.is_err());
        // A page after the rows are complete.
        let (err, _) = read_bytes(
            "extra",
            &simple(
                1,
                vec![
                    plain_page(1, doubles(&[1.0])),
                    plain_page(1, doubles(&[2.0])),
                ],
            ),
        );
        assert!(err.is_err());
        // A negative count.
        let (err, _) = read_bytes(
            "negative",
            &simple(1, vec![plain_page(-1, doubles(&[1.0]))]),
        );
        assert!(err.is_err());
    }

    #[test]
    fn a_lying_page_size_cannot_exhaust_memory() {
        // compressed_page_size claims 2 GB.
        let mut header = plain_page(1, doubles(&[1.0]));
        header.0.compressed_page_size = i32::MAX;
        let (err, _) = read_bytes("huge-compressed", &simple(1, vec![header]));
        assert!(err.is_err());
        // uncompressed_page_size claims 2 GB for a ZSTD chunk: refused by the
        // budget (or by the frame header check) before any allocation.
        let bytes = build(
            vec![Chunk {
                name: "d",
                physical: physical::DOUBLE,
                codec: format::codec::ZSTD,
                rows: 1,
                pages: vec![{
                    let mut p = plain_page(1, vec![0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x00]);
                    p.0.uncompressed_page_size = i32::MAX;
                    p
                }],
                optional: false,
            }],
            &[],
        );
        let (dir, path) = write_temp("huge-uncompressed", &bytes);
        let small = Context::new(Some(1), 1 << 20).unwrap();
        let err = read_table(&small, &path).unwrap_err();
        assert!(
            matches!(err, Error::Budget { .. } | Error::Invalid(_)),
            "{err:?}"
        );
        assert_eq!(small.memory_used(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn dictionary_pages_and_the_plain_fallback_decode() {
        // Dictionary of three strings; one page of indices 2,0,0,1 (width 2,
        // bit-packed: 2 + 0<<2 + 0<<4 + 1<<6 = 66, one group of 8 values),
        // then a PLAIN page for the fallback.
        let dictionary = strings(&["XX", "YY", "ZZ"]);
        let indices = vec![2u8, (1 << 1) | 1, 0b0100_0010, 0b0000_0000];
        let mut index_page = Vec::new();
        index_page.extend_from_slice(&indices);
        let dictionary_size = dictionary.len();
        let pages = vec![
            page(
                page_type::DICTIONARY_PAGE,
                dictionary,
                dictionary_size,
                None,
                Some(DictionaryPageHeader {
                    num_values: 3,
                    encoding: encoding::PLAIN,
                    is_sorted: None,
                }),
            ),
            {
                let size = index_page.len();
                page(
                    page_type::DATA_PAGE,
                    index_page,
                    size,
                    Some(DataPageHeader {
                        num_values: 4,
                        encoding: encoding::RLE_DICTIONARY,
                        definition_level_encoding: encoding::RLE,
                        repetition_level_encoding: encoding::RLE,
                    }),
                    None,
                )
            },
            plain_page(2, strings(&["plain", "tail"])),
        ];
        let bytes = build(
            vec![Chunk {
                name: "s",
                physical: physical::BYTE_ARRAY,
                codec: format::codec::UNCOMPRESSED,
                rows: 6,
                pages,
                optional: false,
            }],
            &[],
        );
        let (outcome, _) = read_bytes("dictionary", &bytes);
        let table = outcome.unwrap();
        let column = table.column("s").unwrap().as_utf8().unwrap();
        assert_eq!(
            column.iter().collect::<Vec<_>>(),
            ["ZZ", "XX", "XX", "YY", "plain", "tail"]
        );
    }

    #[test]
    fn a_dictionary_index_out_of_range_and_a_missing_dictionary_raise() {
        // Width 2, one bit-packed group whose first index is 3, dictionary of 1.
        let dictionary = doubles(&[1.0]);
        let size = dictionary.len();
        let dict_page = page(
            page_type::DICTIONARY_PAGE,
            dictionary,
            size,
            None,
            Some(DictionaryPageHeader {
                num_values: 1,
                encoding: encoding::PLAIN,
                is_sorted: None,
            }),
        );
        let indices = vec![2u8, (1 << 1) | 1, 0b0000_0011, 0];
        let isize_ = indices.len();
        let index_page = |body: Vec<u8>, n: usize| {
            page(
                page_type::DATA_PAGE,
                body,
                n,
                Some(DataPageHeader {
                    num_values: 1,
                    encoding: encoding::RLE_DICTIONARY,
                    definition_level_encoding: encoding::RLE,
                    repetition_level_encoding: encoding::RLE,
                }),
                None,
            )
        };
        let (err, _) = read_bytes(
            "range",
            &simple(1, vec![dict_page, index_page(indices.clone(), isize_)]),
        );
        assert!(err.unwrap_err().to_string().contains("out of range"));
        let (err, _) = read_bytes("nodict", &simple(1, vec![index_page(indices, isize_)]));
        assert!(err.unwrap_err().to_string().contains("dictionary"));
    }

    #[test]
    fn invalid_utf8_and_unsupported_encodings_raise() {
        let mut body = Vec::new();
        body.extend_from_slice(&2u32.to_le_bytes());
        body.extend_from_slice(&[0xff, 0xfe]);
        let bytes = build(
            vec![Chunk {
                name: "s",
                physical: physical::BYTE_ARRAY,
                codec: format::codec::UNCOMPRESSED,
                rows: 1,
                pages: vec![plain_page(1, body)],
                optional: false,
            }],
            &[],
        );
        let (err, _) = read_bytes("utf8", &bytes);
        assert!(err.unwrap_err().to_string().contains("UTF-8"));
        let mut delta = plain_page(1, doubles(&[1.0]));
        delta.0.data_page.as_mut().unwrap().encoding = encoding::DELTA_BINARY_PACKED;
        let (err, _) = read_bytes("delta", &simple(1, vec![delta]));
        assert!(err.unwrap_err().to_string().contains("DELTA_BINARY_PACKED"));
    }

    #[test]
    fn optional_columns_with_a_null_raise_and_without_one_read() {
        // Definition levels, RLE width 1: one run of 3 ones is header 0b110,
        // value 1; the length prefix is 2.
        let present = vec![2, 0, 0, 0, 0b0000_0110, 1];
        let mut with_levels = present.clone();
        with_levels.extend_from_slice(&doubles(&[1.0, 2.0, 3.0]));
        let build_optional = |body: Vec<u8>| {
            build(
                vec![Chunk {
                    name: "d",
                    physical: physical::DOUBLE,
                    codec: format::codec::UNCOMPRESSED,
                    rows: 3,
                    pages: vec![plain_page(3, body)],
                    optional: true,
                }],
                &[],
            )
        };
        let (ok, _) = read_bytes("optional", &build_optional(with_levels));
        assert_eq!(
            ok.unwrap().column("d").unwrap().as_f64().unwrap(),
            [1.0, 2.0, 3.0]
        );
        // Bit-packed group, levels 1,0,1: header 0b11, byte 0b101.
        let mut with_null = vec![2, 0, 0, 0, 0b0000_0011, 0b0000_0101];
        with_null.extend_from_slice(&doubles(&[1.0, 3.0]));
        let (err, _) = read_bytes("null", &build_optional(with_null));
        assert!(err.unwrap_err().to_string().contains("null"));
    }

    #[test]
    fn a_checksum_mismatch_raises() {
        let mut p = plain_page(1, doubles(&[1.0]));
        p.0.crc = Some(crc32(&p.1) as i32);
        let (ok, _) = read_bytes("crc-ok", &simple(1, vec![p.clone()]));
        assert!(ok.is_ok());
        p.0.crc = Some(crc32(&p.1) as i32 ^ 1);
        let (err, _) = read_bytes("crc-bad", &simple(1, vec![p]));
        assert!(err.unwrap_err().to_string().contains("checksum"));
    }

    #[test]
    fn bytes_after_the_last_page_and_index_pages_raise() {
        // An INDEX page after the data page.
        let index = page(page_type::INDEX_PAGE, vec![1, 2, 3], 3, None, None);
        let (err, _) = read_bytes(
            "index",
            &simple(1, vec![plain_page(1, doubles(&[1.0])), index]),
        );
        assert!(err.unwrap_err().to_string().contains("INDEX_PAGE"));
    }

    #[test]
    fn truncating_a_real_file_at_every_length_never_panics() {
        let source =
            std::fs::read(fixtures().join("hamiltonian/fastparquet_zstd.parquet")).unwrap();
        let (dir, path) = write_temp("truncate", &source);
        let c = Context::new(Some(1), 1 << 26).unwrap();
        for len in 0..source.len() {
            std::fs::write(&path, &source[..len]).unwrap();
            assert!(read_table(&c, &path).is_err(), "prefix of {len} bytes read");
            assert!(
                read_metadata(&c, &path).is_err(),
                "prefix of {len} bytes opened"
            );
            assert_eq!(c.memory_used(), 0);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_seeded_mutation_sweep_never_panics_or_leaks() {
        let source = std::fs::read(fixtures().join("hamiltonian/pyarrow_zstd.parquet")).unwrap();
        let (dir, path) = write_temp("mutate", &source);
        let c = Context::new(Some(1), 1 << 26).unwrap();
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..1500 {
            let mut bytes = source.clone();
            for _ in 0..1 + next() % 4 {
                let at = (next() as usize) % bytes.len();
                bytes[at] = next() as u8;
            }
            std::fs::write(&path, &bytes).unwrap();
            let _ = read_table(&c, &path);
            let _ = read_metadata(&c, &path);
            assert_eq!(c.memory_used(), 0);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    // ---- the shipped libraries (skipped when the library is not set) ----

    fn corpus_root(key: &str) -> Option<PathBuf> {
        match std::env::var_os(key) {
            Some(value) if !value.is_empty() => Some(PathBuf::from(value)),
            _ => None,
        }
    }

    fn corpus_path(name: &str) -> Option<PathBuf> {
        if let Some(rest) = name.strip_prefix("paw/") {
            corpus_root("MANDACARU_PAW_PATH").map(|root| root.join(rest))
        } else if let Some(rest) = name.strip_prefix("upaw/") {
            corpus_root("MANDACARU_UPAW_PATH").map(|root| root.join(rest))
        } else {
            None
        }
    }

    fn check_corpus_file(name: &str, entry: &serde_json::Value, c: &Context) {
        let Some(path) = corpus_path(name) else {
            return;
        };
        if !path.exists() {
            return;
        }
        let table = read_table(c, &path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let expected = entry["columns"].as_array().unwrap();
        assert_eq!(table.columns.len(), expected.len(), "{name}");
        for ((got_name, column), want) in table.columns.iter().zip(expected) {
            assert_eq!(got_name, want[0].as_str().unwrap(), "{name}");
            assert_eq!(
                column.len() as u64,
                want[1].as_u64().unwrap(),
                "{name} {got_name}"
            );
            assert_eq!(
                double_digest(column),
                want[2].as_str().unwrap(),
                "{name} {got_name}"
            );
        }
        let text = &table.metadata["mandacaru.pseudopotential"];
        assert_eq!(
            text.len() as u64,
            entry["kv"]["bytes"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(
            sha256(text.as_bytes()),
            entry["kv"]["sha256"].as_str().unwrap(),
            "{name}"
        );
        let footer = read_metadata(c, &path).unwrap();
        assert_eq!(footer.num_rows, entry["rows"].as_u64().unwrap(), "{name}");
        assert_eq!(
            footer.created_by,
            entry["created_by"].as_str().unwrap(),
            "{name}"
        );
        drop(table);
        assert_eq!(c.memory_used(), 0, "{name}");
    }

    #[test]
    fn the_peak_of_a_big_file_is_the_table_plus_one_chunk() {
        let Some(root) = corpus_root("MANDACARU_PAW_PATH") else {
            return;
        };
        let path = root.join("pbe-fr/Pb.parquet");
        if !path.exists() {
            return;
        }
        let c = Context::new(Some(1), 1 << 31).unwrap();
        let table = read_table(&c, &path).unwrap();
        let table_bytes: usize = table.columns.iter().map(|(_, col)| col.len() * 8).sum();
        let kv: usize = table.metadata.values().map(String::len).sum();
        eprintln!(
            "table {table_bytes} B, metadata {kv} B, peak {} B",
            c.memory_peak()
        );
        assert!(
            c.memory_peak() <= table_bytes + 10 * kv + (4 << 20),
            "peak {}",
            c.memory_peak()
        );
        drop(table);
        assert_eq!(c.memory_used(), 0);
    }

    #[test]
    fn one_shipped_file_per_library_matches_the_frozen_digests() {
        let references = references();
        let files = references["corpus_files"].as_object().unwrap();
        let c = Context::new(Some(1), 1 << 31).unwrap();
        for name in [
            "paw/lda-sr/Cu.parquet",
            "paw/lda-fr/Ac.parquet",
            "paw/pbe-sr/H.parquet",
            "paw/pbe-fr/Pb.parquet",
            "upaw/lda-sr/Ag.parquet",
        ] {
            check_corpus_file(name, &files[name], &c);
        }
    }

    /// All 431 shipped files; slow in a debug build, run with
    /// `cargo test --release -p mandacaru-core --lib parquet::read -- --ignored`.
    #[test]
    #[ignore]
    fn every_shipped_file_matches_the_frozen_digests() {
        let references = references();
        let c = Context::new(Some(1), 1 << 31).unwrap();
        let mut checked = 0;
        for (name, entry) in references["corpus_files"].as_object().unwrap() {
            if corpus_path(name).is_some_and(|p| p.exists()) {
                check_corpus_file(name, entry, &c);
                checked += 1;
            }
        }
        eprintln!("checked {checked} shipped files");
    }
}
