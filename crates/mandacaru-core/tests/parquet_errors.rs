//! Hostile and malformed Parquet input: every failure is a clean
//! `Error::Invalid("parquet: ...")` (or `Error::Budget`), never a panic, never
//! an allocation beyond what the `Context` reserved, and never a reservation
//! left behind.
//!
//! The memory claim is measured, not assumed: this test binary installs a
//! counting global allocator that tracks the live bytes of the calling thread,
//! and every probe asserts `peak live bytes <= ctx.memory_peak() + SLACK`
//! (what the reader reserved from the budget, plus a constant for the small
//! structures it does not account). A reader that sized a buffer from an
//! untrusted number before charging it would exceed that bound by orders of
//! magnitude.
//!
//! Frozen inputs come from `test/migration/reference/parquet/`; hand-built
//! files use the public `format` and `thrift` layers.

mod parquet_support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use mandacaru_core::parquet::{
    self,
    codec::{self, zstd as zstd_codec},
    format::{
        codec as codec_id, encoding, page_type, physical, repetition, ColumnChunk, ColumnMetaData,
        DataPageHeader, DataPageHeaderV2, DictionaryPageHeader, FileMetaData, KeyValue, PageHeader,
        RowGroup, SchemaElement,
    },
    read_metadata, read_table, Codec, MAGIC,
};
use mandacaru_core::{Context, Error};
use parquet_support::{fixture, fixtures, mutate, target, Directory, Part, Rng, Target};

// ---------------------------------------------------------------------------
// Counting allocator (per thread, armed only inside `measured`).
// ---------------------------------------------------------------------------

struct Counting;

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
}

fn note_alloc(size: usize) {
    if ACTIVE.try_with(Cell::get).unwrap_or(false) {
        let _ = LIVE.try_with(|live| {
            let now = live.get() + size;
            live.set(now);
            let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
        });
    }
}

fn note_free(size: usize) {
    if ACTIVE.try_with(Cell::get).unwrap_or(false) {
        let _ = LIVE.try_with(|live| live.set(live.get().saturating_sub(size)));
    }
}

// SAFETY: this forwards every call unchanged to `System`, which upholds the
// `GlobalAlloc` contract; the only addition is bookkeeping in const-initialized
// thread-locals that never allocate (and are read with `try_with`, so a call
// made during thread teardown is ignored rather than reentrant).
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_alloc(layout.size());
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note_free(layout.size());
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note_free(layout.size());
        note_alloc(new_size);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// Peak live bytes allocated on this thread while `f` runs.
fn measured<T>(f: impl FnOnce() -> T) -> (T, usize) {
    LIVE.with(|l| l.set(0));
    PEAK.with(|p| p.set(0));
    ACTIVE.with(|a| a.set(true));
    let value = f();
    ACTIVE.with(|a| a.set(false));
    (value, PEAK.with(Cell::get))
}

/// What the reader may allocate without reserving it: error text, path
/// strings, `Vec` headers of the parsed footer.
const SLACK: usize = 1 << 20;

// ---------------------------------------------------------------------------
// Scratch files and fixtures.
// ---------------------------------------------------------------------------

/// Every `*.parquet` fixture as `(relative name, bytes)`, sorted by name.
fn all_fixtures() -> Vec<(String, Vec<u8>)> {
    let mut found = Vec::new();
    for folder in ["generic", "hamiltonian", "negative", "pseudopotential"] {
        let mut names: Vec<_> = fs::read_dir(fixtures().join(folder))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.ends_with(".parquet"))
            .collect();
        names.sort();
        for name in names {
            let relative = format!("{folder}/{name}");
            let bytes = fixture(&relative);
            found.push((relative, bytes));
        }
    }
    found
}

fn references() -> serde_json::Value {
    let text = fs::read_to_string(fixtures().join("references.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

// ---------------------------------------------------------------------------
// Probing a file: both entry points, with every invariant checked.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Outcome {
    read: Result<(usize, usize), Error>,
    metadata: Result<u64, Error>,
}

/// The most any probe allocated beyond what it had reserved.
static WORST_EXCESS: AtomicUsize = AtomicUsize::new(0);

/// Runs `read_table` and `read_metadata` on `path` under a context of `limit`
/// bytes and checks the invariants that hold for every input.
fn probe(path: &Path, limit: usize) -> Outcome {
    let ctx = Context::new(Some(1), limit).unwrap();
    let ((read, metadata), peak) = measured(|| {
        let read = read_table(&ctx, path).map(|table| {
            let rows = table.num_rows();
            for (name, column) in &table.columns {
                assert_eq!(column.len(), rows, "column {name} has the wrong length");
            }
            (rows, table.columns.len())
        });
        let metadata = read_metadata(&ctx, path).map(|footer| footer.num_rows);
        (read, metadata)
    });
    assert_eq!(ctx.memory_used(), 0, "a reservation outlived the call");
    WORST_EXCESS.fetch_max(peak.saturating_sub(ctx.memory_peak()), Ordering::Relaxed);
    assert!(
        peak <= ctx.memory_peak() + SLACK,
        "allocated {peak} bytes live but reserved only {} (limit {limit})",
        ctx.memory_peak()
    );
    assert!(ctx.memory_peak() <= limit);
    if read.is_ok() {
        assert!(
            metadata.is_ok(),
            "read_table succeeded where read_metadata failed"
        );
    }
    for error in [read.as_ref().err(), metadata.as_ref().err()]
        .into_iter()
        .flatten()
    {
        match error {
            Error::Invalid(message) => assert!(
                message.starts_with("parquet: "),
                "an error without the parquet prefix: {message}"
            ),
            Error::Budget { .. } | Error::Overflow | Error::Allocation => {}
            other => panic!("unexpected error kind {other:?}"),
        }
    }
    Outcome { read, metadata }
}

/// The error text of reading `bytes`, which must fail.
fn failure(bytes: &[u8]) -> String {
    failure_under(bytes, 64 << 20)
}

fn failure_under(bytes: &[u8], limit: usize) -> String {
    let dir = Directory::new();
    let path = dir.file("t.parquet");
    fs::write(&path, bytes).unwrap();
    let outcome = probe(&path, limit);
    match outcome.read {
        Ok(shape) => panic!("the file read as {shape:?}"),
        Err(error) => error.to_string(),
    }
}

fn reads(bytes: &[u8]) -> parquet::Table {
    let dir = Directory::new();
    let path = dir.file("t.parquet");
    fs::write(&path, bytes).unwrap();
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let table = read_table(&ctx, &path).unwrap_or_else(|e| panic!("read failed: {e}"));
    table
}

fn contains_all(message: &str, needles: &[&str]) {
    for needle in needles {
        assert!(message.contains(needle), "{message:?} lacks {needle:?}");
    }
}

// ---------------------------------------------------------------------------
// Hand-built files.
// ---------------------------------------------------------------------------

type Page = (PageHeader, Vec<u8>);

struct Chunk {
    name: &'static str,
    physical: i32,
    codec: i32,
    rows: i64,
    pages: Vec<Page>,
    optional: bool,
}

fn page(
    page_type: i32,
    body: Vec<u8>,
    uncompressed: usize,
    data: Option<DataPageHeader>,
    dictionary: Option<DictionaryPageHeader>,
    v2: Option<DataPageHeaderV2>,
) -> Page {
    (
        PageHeader {
            page_type,
            uncompressed_page_size: uncompressed as i32,
            compressed_page_size: body.len() as i32,
            crc: None,
            data_page: data,
            dictionary_page: dictionary,
            data_page_v2: v2,
        },
        body,
    )
}

fn data_header(rows: i32, page_encoding: i32) -> DataPageHeader {
    DataPageHeader {
        num_values: rows,
        encoding: page_encoding,
        definition_level_encoding: encoding::RLE,
        repetition_level_encoding: encoding::RLE,
    }
}

fn plain_page(rows: i32, body: Vec<u8>) -> Page {
    let size = body.len();
    page(
        page_type::DATA_PAGE,
        body,
        size,
        Some(data_header(rows, encoding::PLAIN)),
        None,
        None,
    )
}

fn page_with_encoding(rows: i32, body: Vec<u8>, page_encoding: i32) -> Page {
    let size = body.len();
    page(
        page_type::DATA_PAGE,
        body,
        size,
        Some(data_header(rows, page_encoding)),
        None,
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

fn build(chunks: Vec<Chunk>, metadata: &[(&str, &str)]) -> Vec<u8> {
    build_with(chunks, metadata, |_| {})
}

/// A single-row-group file from raw chunks; `tweak` edits the footer structs
/// before they are encoded.
fn build_with(
    chunks: Vec<Chunk>,
    metadata: &[(&str, &str)],
    tweak: impl FnOnce(&mut FileMetaData),
) -> Vec<u8> {
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
            leaf.repetition = Some(repetition::OPTIONAL);
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
    let mut meta = FileMetaData {
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
    tweak(&mut meta);
    let footer = meta.to_bytes().unwrap();
    file.extend_from_slice(&footer);
    file.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    file.extend_from_slice(MAGIC);
    file
}

/// One DOUBLE column named `d`, uncompressed, with the given pages.
fn double_file(rows: i64, pages: Vec<Page>) -> Vec<u8> {
    build(
        vec![Chunk {
            name: "d",
            physical: physical::DOUBLE,
            codec: codec_id::UNCOMPRESSED,
            rows,
            pages,
            optional: false,
        }],
        &[("k", "v")],
    )
}

fn string_file(rows: i64, pages: Vec<Page>) -> Vec<u8> {
    build(
        vec![Chunk {
            name: "s",
            physical: physical::BYTE_ARRAY,
            codec: codec_id::UNCOMPRESSED,
            rows,
            pages,
            optional: false,
        }],
        &[],
    )
}

fn compressed_file(codec_number: i32, rows: i64, pages: Vec<Page>) -> Vec<u8> {
    build(
        vec![Chunk {
            name: "d",
            physical: physical::DOUBLE,
            codec: codec_number,
            rows,
            pages,
            optional: false,
        }],
        &[],
    )
}

/// A whole file around raw footer bytes.
fn file_around(footer: &[u8]) -> Vec<u8> {
    let mut file = MAGIC.to_vec();
    file.extend_from_slice(footer);
    file.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    file.extend_from_slice(MAGIC);
    file
}

// ---------------------------------------------------------------------------
// The measuring instrument works.
// ---------------------------------------------------------------------------

#[test]
fn the_allocation_probe_sees_what_a_call_allocates() {
    let (len, peak) = measured(|| {
        let v = vec![1u8; 3 << 20];
        v.len()
    });
    assert_eq!(len, 3 << 20);
    assert!((3 << 20..(3 << 20) + 4096).contains(&peak), "{peak}");
    // Freed memory does not accumulate.
    let (_, peak) = measured(|| {
        for _ in 0..100 {
            let v = vec![0u8; 1 << 20];
            assert_eq!(v.len(), 1 << 20);
        }
    });
    assert!(peak < (2 << 20), "{peak}");
    // A reader that allocated a megabyte without reserving it would fail the
    // bound the probes assert.
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    let (_, peak) = measured(|| {
        let _r = ctx.reserve(1000).unwrap();
        let v = vec![0u8; 2 * SLACK];
        v.len()
    });
    assert!(peak > ctx.memory_peak() + SLACK);
}

// ---------------------------------------------------------------------------
// The hand-built control: the helpers above produce readable files.
// ---------------------------------------------------------------------------

#[test]
fn the_hand_built_control_files_read() {
    let table = reads(&double_file(2, vec![plain_page(2, doubles(&[1.5, -2.5]))]));
    assert_eq!(table.column("d").unwrap().as_f64().unwrap(), [1.5, -2.5]);
    let table = reads(&string_file(2, vec![plain_page(2, strings(&["a", "bc"]))]));
    let column = table.column("s").unwrap().as_utf8().unwrap();
    assert_eq!(column.iter().collect::<Vec<_>>(), ["a", "bc"]);
}

// ---------------------------------------------------------------------------
// Every frozen negative fixture raises and names what is wrong.
// ---------------------------------------------------------------------------

/// Negative fixtures that are valid Parquet and fail only the Hamiltonian
/// schema (exercised through `Document::read` in `persistence.rs`).
fn is_semantic(name: &str) -> bool {
    name.starts_with("negative/hamiltonian_") || name == "negative/column_double_pauli.parquet"
}

/// Other wordings, per fixture, that name the problem as well as the hint
/// recorded in `references.json` does.
fn wording(name: &str) -> &'static [&'static str] {
    match name {
        "negative/footer_chunk_size_huge.parquet"
        | "negative/footer_page_offset_outside_file.parquet" => &["past the data area"],
        _ => &[],
    }
}

#[test]
fn every_negative_fixture_raises_and_names_its_problem() {
    let references = references();
    let mut checked = 0;
    let mut misses = Vec::new();
    for (name, bytes) in all_fixtures() {
        if !name.starts_with("negative/") || is_semantic(&name) {
            continue;
        }
        let entry = &references["fixtures"][name.as_str()];
        assert_eq!(entry["expect_new_reader"], "error", "{name}");
        let dir = Directory::new();
        let path = dir.file("t.parquet");
        fs::write(&path, &bytes).unwrap();
        // The 2^40-row footer is a budget test: it needs the small context.
        let limit = if name == "negative/footer_rows_2e40.parquet" {
            1 << 20
        } else {
            64 << 20
        };
        let outcome = probe(&path, limit);
        let error = outcome.read.expect_err(&format!("{name} must not read"));
        if name == "negative/footer_rows_2e40.parquet" {
            assert!(matches!(error, Error::Budget { .. }), "{name}: {error:?}");
            continue;
        }
        let text = error.to_string();
        let lowered = text.to_lowercase();
        let mut hints: Vec<String> = entry["new_reader_error_hint"]
            .as_array()
            .unwrap_or_else(|| panic!("{name} has no hint"))
            .iter()
            .map(|h| h.as_str().unwrap().to_lowercase())
            .collect();
        // The recorded hints are informative; where the message says the same
        // thing in other words, that wording is pinned here.
        hints.extend(wording(&name).iter().map(|w| w.to_lowercase()));
        if !hints.iter().any(|h| lowered.contains(h)) {
            misses.push(format!("{name}: {text:?} names none of {hints:?}"));
        }
        checked += 1;
    }
    assert!(misses.is_empty(), "{}", misses.join("\n"));
    assert!(checked >= 40, "only {checked} negative fixtures checked");
}

#[test]
fn the_positive_fixtures_read_under_the_probe_invariants() {
    let mut read = 0;
    for (name, bytes) in all_fixtures() {
        if name.starts_with("negative/") {
            continue;
        }
        let dir = Directory::new();
        let path = dir.file("t.parquet");
        fs::write(&path, &bytes).unwrap();
        let outcome = probe(&path, 16 << 20);
        outcome.read.unwrap_or_else(|e| panic!("{name}: {e}"));
        outcome.metadata.unwrap_or_else(|e| panic!("{name}: {e}"));
        read += 1;
    }
    assert!(read >= 30, "{read}");
}

// ---------------------------------------------------------------------------
// The container: magic, trailer and footer length.
// ---------------------------------------------------------------------------

#[test]
fn bad_missing_and_encrypted_magic_are_refused() {
    let good = fixture("hamiltonian/fastparquet_zstd.parquet");
    // Head magic.
    let mut head = good.clone();
    head[0] = b'X';
    contains_all(&failure(&head), &["magic"]);
    // Tail magic.
    let mut tail = good.clone();
    let n = tail.len();
    tail[n - 1] = b'X';
    contains_all(&failure(&tail), &["magic"]);
    // Encrypted footer (PARE at the tail) and encrypted file (PARE at the head).
    let mut encrypted_tail = good.clone();
    encrypted_tail[n - 1] = b'E';
    contains_all(&failure(&encrypted_tail), &["encrypted"]);
    let mut encrypted_head = good.clone();
    encrypted_head[3] = b'E';
    contains_all(&failure(&encrypted_head), &["encrypted"]);
    // Empty, one magic only, and a JSON text.
    assert!(failure(b"").contains("Parquet"));
    assert!(failure(b"PAR1").contains("Parquet"));
    assert!(failure(b"PAR1PAR1").contains("Parquet"));
    assert!(failure(b"{\"format\": \"mandacaru-qubit-hamiltonian\"}").contains("Parquet"));
    // A missing path is an error too.
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    let err = read_table(&ctx, Path::new("/nonexistent/dir/file.parquet")).unwrap_err();
    assert!(matches!(&err, Error::Invalid(m) if m.starts_with("parquet: cannot open")));
    assert!(parquet::is_parquet(Path::new("/nonexistent/dir/file.parquet")).is_err());
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn is_parquet_looks_only_at_the_leading_magic() {
    let dir = Directory::new();
    let path = dir.file("t");
    for (bytes, expected) in [
        (&b"PAR1"[..], true),
        (b"PAR1junk", true),
        (b"PAR", false),
        (b"", false),
        (b"{\"a\":1}", false),
        (b"PARE", false),
        (b"par1", false),
    ] {
        fs::write(&path, bytes).unwrap();
        assert_eq!(parquet::is_parquet(&path).unwrap(), expected, "{bytes:?}");
    }
}

#[test]
fn footer_lengths_that_lie_are_refused() {
    let good = fixture("hamiltonian/fastparquet_zstd.parquet");
    let n = good.len();
    let patch = |length: u32| {
        let mut bytes = good.clone();
        bytes[n - 8..n - 4].copy_from_slice(&length.to_le_bytes());
        bytes
    };
    let true_length = u32::from_le_bytes(good[n - 8..n - 4].try_into().unwrap());
    for length in [
        0,
        1,
        2,
        true_length - 1,
        true_length + 1,
        (n - 8) as u32,
        (n - 7) as u32,
        (n - 4) as u32,
        n as u32,
        (64 << 20) + 1,
        u32::MAX / 2,
        u32::MAX - 1,
        u32::MAX,
    ] {
        let text = failure(&patch(length));
        assert!(
            text.contains("footer") || text.contains("Parquet"),
            "length {length}: {text}"
        );
    }
    // The same on a context too small to hold even an honest footer: the
    // budget refuses before the footer is read.
    let ctx = Context::new(Some(1), 64).unwrap();
    let dir = Directory::new();
    let path = dir.file("t.parquet");
    fs::write(&path, &good).unwrap();
    assert!(matches!(
        read_table(&ctx, &path).unwrap_err(),
        Error::Budget { .. }
    ));
    assert!(matches!(
        read_metadata(&ctx, &path).unwrap_err(),
        Error::Budget { .. }
    ));
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_footer_of_garbage_or_a_wrong_version_is_refused() {
    // Random bytes of every small length as the footer.
    let mut state = 0x1234_5678_9abc_def1u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for length in 1..200usize {
        let footer: Vec<u8> = (0..length).map(|_| next() as u8).collect();
        let text = failure(&file_around(&footer));
        assert!(text.starts_with("parquet: "), "{text}");
    }
    // A valid footer with an unsupported version.
    let bytes = double_file(1, vec![plain_page(1, doubles(&[1.0]))]);
    for version in [0, 3, 99, -1] {
        let mut patched = bytes.clone();
        let n = patched.len();
        let footer_len = u32::from_le_bytes(patched[n - 8..n - 4].try_into().unwrap()) as usize;
        let start = n - 8 - footer_len;
        let mut meta = FileMetaData::from_bytes(&patched[start..start + footer_len]).unwrap();
        meta.version = version;
        let footer = meta.to_bytes().unwrap();
        patched.truncate(start);
        patched.extend_from_slice(&footer);
        patched.extend_from_slice(&(footer.len() as u32).to_le_bytes());
        patched.extend_from_slice(MAGIC);
        contains_all(&failure(&patched), &["version"]);
    }
}

// ---------------------------------------------------------------------------
// Thrift: deep nesting, lying lengths.
// ---------------------------------------------------------------------------

#[test]
fn deeply_nested_thrift_is_refused_without_overflowing_the_stack() {
    // An unknown field (id 15) holding `depth` nested structs.
    let structs = |depth: usize| {
        let mut footer = vec![0xFC];
        footer.extend(std::iter::repeat_n(0x1C, depth));
        footer.extend(std::iter::repeat_n(0x00, depth + 2));
        footer
    };
    // An unknown field (id 15) holding `depth` nested lists of one element.
    let lists = |depth: usize| {
        let mut footer = vec![0xF9];
        footer.extend(std::iter::repeat_n(0x19, depth));
        footer.push(0x09);
        footer.push(0x00);
        footer
    };
    // A map of maps is not needed: lists and structs use the same guard.
    for depth in [20usize, 64, 1000, 100_000] {
        for footer in [structs(depth), lists(depth)] {
            let text = failure(&file_around(&footer));
            assert!(
                text.contains("depth") || text.contains("nest"),
                "depth {depth}: {text}"
            );
        }
    }
    // A footer of millions of levels is refused too (by the depth limit or by
    // the budget the footer charges), without recursing that far.
    for footer in [structs(5_000_000), lists(5_000_000)] {
        let text = failure(&file_around(&footer));
        assert!(
            text.starts_with("parquet: ") || text.contains("budget"),
            "{text}"
        );
    }
    // Within the limit the same unknown field is skipped; the footer then fails
    // for what it lacks, not for its nesting.
    let text = failure(&file_around(&structs(8)));
    assert!(!text.contains("depth"), "{text}");
}

#[test]
fn thrift_lengths_larger_than_the_footer_are_refused() {
    // A list header claiming 2^28 elements with nothing behind it.
    let mut footer = vec![0xF9]; // field 15, list
    footer.push(0xF9); // size 15 or more, element type list ...
    footer.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0x7F]); // ... length varint 2^28 - 1
    footer.push(0x00);
    let text = failure(&file_around(&footer));
    assert!(text.starts_with("parquet: "), "{text}");
    // A binary field claiming more bytes than remain: created_by (id 6) is a
    // string; after the previous field ids the header byte is built by hand.
    let mut footer = vec![0x18, 0x02]; // field 1 (version) is i32 type 5; this is deliberately malformed
    footer.extend_from_slice(&[0x68, 0xFF, 0xFF, 0xFF, 0x0F]);
    let text = failure(&file_around(&footer));
    assert!(text.starts_with("parquet: "), "{text}");
}

// ---------------------------------------------------------------------------
// Sizes and counts that lie.
// ---------------------------------------------------------------------------

#[test]
fn a_two_to_the_forty_row_footer_is_refused_by_the_budget() {
    let name = "negative/footer_rows_2e40.parquet";
    let bytes = fixture(name);
    let dir = Directory::new();
    let path = dir.file("t.parquet");
    fs::write(&path, &bytes).unwrap();
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    let (outcome, peak) = measured(|| read_table(&ctx, &path));
    match outcome {
        Err(Error::Budget { .. }) => {}
        other => panic!(
            "expected Error::Budget, got {:?}",
            other.map(|t| t.num_rows())
        ),
    }
    assert_eq!(ctx.memory_used(), 0);
    assert!(peak <= (1 << 20) + SLACK, "allocated {peak} bytes");
    // The footer alone is honest about the claim and cheap to read.
    let footer = read_metadata(&ctx, &path).unwrap();
    assert_eq!(footer.num_rows, 1 << 40);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn absurd_row_counts_are_refused_for_every_column_kind() {
    for rows in [
        1i64 << 40,
        1 << 33,
        1 << 31,
        i64::MAX / 16,
        i64::MAX / 8,
        i64::MAX,
    ] {
        for string_column in [false, true] {
            let pages = vec![plain_page(
                1,
                if string_column {
                    strings(&["x"])
                } else {
                    doubles(&[1.0])
                },
            )];
            let bytes = if string_column {
                string_file(rows, pages)
            } else {
                double_file(rows, pages)
            };
            let dir = Directory::new();
            let path = dir.file("t.parquet");
            fs::write(&path, &bytes).unwrap();
            let outcome = probe(&path, 1 << 20);
            let error = outcome.read.unwrap_err();
            assert!(
                matches!(
                    error,
                    Error::Budget { .. } | Error::Overflow | Error::Invalid(_)
                ),
                "rows {rows}: {error:?}"
            );
            // Only the budget families are acceptable for a footer that is
            // consistent with itself.
            if rows <= 1 << 40 {
                assert!(
                    matches!(error, Error::Budget { .. }),
                    "rows {rows}: {error:?}"
                );
            }
        }
    }
}

#[test]
fn page_sizes_and_counts_that_lie_are_refused_inside_the_budget() {
    // compressed_page_size past the chunk.
    for compressed in [i32::MAX, 1 << 30, 100_000] {
        let mut p = plain_page(1, doubles(&[1.0]));
        p.0.compressed_page_size = compressed;
        let text = failure(&double_file(1, vec![p]));
        assert!(text.contains("extends past"), "{text}");
    }
    // Negative sizes and counts.
    let mut p = plain_page(1, doubles(&[1.0]));
    p.0.compressed_page_size = -1;
    contains_all(&failure(&double_file(1, vec![p])), &["negative"]);
    let mut p = plain_page(1, doubles(&[1.0]));
    p.0.uncompressed_page_size = -5;
    contains_all(&failure(&double_file(1, vec![p])), &["negative"]);
    contains_all(
        &failure(&double_file(1, vec![plain_page(-1, doubles(&[1.0]))])),
        &["negative"],
    );
    // uncompressed_page_size that disagrees with an uncompressed body.
    for lie in [0usize, 1, 7, 9, 1 << 20, i32::MAX as usize] {
        let mut p = plain_page(1, doubles(&[1.0]));
        p.0.uncompressed_page_size = lie as i32;
        let text = failure(&double_file(1, vec![p]));
        assert!(text.contains("uncompressed"), "{lie}: {text}");
    }
    // The same lie on a compressed chunk must not allocate what it claims.
    let frame = valid_frame(&doubles(&[1.0, 2.0, 3.0, 4.0]));
    for lie in [1usize << 20, 1 << 28, i32::MAX as usize] {
        let mut p = plain_page(4, frame.clone());
        p.0.uncompressed_page_size = lie as i32;
        let bytes = compressed_file(codec_id::ZSTD, 4, vec![p.clone()]);
        let text = failure_under(&bytes, 4 << 20);
        assert!(!text.is_empty());
        let bytes = compressed_file(codec_id::SNAPPY, 4, vec![p]);
        let text = failure_under(&bytes, 4 << 20);
        assert!(!text.is_empty());
    }
    // A value count above the rows, below the rows, and a repeated page.
    contains_all(
        &failure(&double_file(1, vec![plain_page(2, doubles(&[1.0, 2.0]))])),
        &["rows"],
    );
    contains_all(
        &failure(&double_file(3, vec![plain_page(2, doubles(&[1.0, 2.0]))])),
        &["rows"],
    );
    // Chunk-level numbers: num_values, offsets and sizes.
    let good = || double_file(1, vec![plain_page(1, doubles(&[1.0]))]);
    let tweak = |edit: &dyn Fn(&mut ColumnMetaData)| {
        let bytes = good();
        retarget(&bytes, |meta| {
            edit(&mut meta.row_groups[0].columns[0].meta_data)
        })
    };
    contains_all(&failure(&tweak(&|m| m.num_values = 2)), &["values"]);
    contains_all(
        &failure(&tweak(&|m| m.data_page_offset = -1)),
        &["negative"],
    );
    contains_all(&failure(&tweak(&|m| m.data_page_offset = 0)), &["header"]);
    contains_all(
        &failure(&tweak(&|m| m.data_page_offset = 1 << 40)),
        &["past"],
    );
    contains_all(
        &failure(&tweak(&|m| m.total_compressed_size = 1 << 40)),
        &["past"],
    );
    contains_all(
        &failure(&tweak(&|m| m.total_compressed_size = i64::MAX)),
        &["past"],
    );
    contains_all(
        &failure(&tweak(&|m| m.total_compressed_size = -1)),
        &["negative"],
    );
    contains_all(
        &failure(&tweak(&|m| m.data_page_offset = i64::MAX)),
        &["past"],
    );
}

/// Re-encodes `bytes` with its footer edited by `edit`.
fn retarget(bytes: &[u8], edit: impl FnOnce(&mut FileMetaData)) -> Vec<u8> {
    let n = bytes.len();
    let footer_len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    let start = n - 8 - footer_len;
    let mut meta = FileMetaData::from_bytes(&bytes[start..start + footer_len]).unwrap();
    edit(&mut meta);
    let footer = meta.to_bytes().unwrap();
    let mut out = bytes[..start].to_vec();
    out.extend_from_slice(&footer);
    out.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    out.extend_from_slice(MAGIC);
    out
}

#[test]
fn inconsistent_schemas_and_row_groups_are_refused() {
    let good = || double_file(2, vec![plain_page(2, doubles(&[1.0, 2.0]))]);
    // num_rows differs from the row groups.
    contains_all(&failure(&retarget(&good(), |m| m.num_rows = 3)), &["rows"]);
    contains_all(
        &failure(&retarget(&good(), |m| m.num_rows = -1)),
        &["negative"],
    );
    contains_all(
        &failure(&retarget(&good(), |m| m.row_groups[0].num_rows = 5)),
        &["rows"],
    );
    // Root child count that disagrees with the schema list.
    contains_all(
        &failure(&retarget(&good(), |m| m.schema[0].num_children = Some(7))),
        &["nested"],
    );
    contains_all(
        &failure(&retarget(&good(), |m| m.schema[0].num_children = Some(-1))),
        &["negative"],
    );
    // A schema with no elements and a root that is a leaf.
    contains_all(
        &failure(&retarget(&good(), |m| m.schema.clear())),
        &["schema"],
    );
    contains_all(
        &failure(&retarget(&good(), |m| {
            m.schema[0].physical_type = Some(physical::DOUBLE)
        })),
        &["group"],
    );
    // Duplicate and mismatching column paths.
    contains_all(
        &failure(&retarget(&good(), |m| {
            m.row_groups[0].columns[0].meta_data.path_in_schema = vec!["other".into()]
        })),
        &["path"],
    );
    contains_all(
        &failure(&retarget(&good(), |m| {
            m.row_groups[0].columns[0].meta_data.physical_type = physical::INT32
        })),
        &["type"],
    );
    // A row group with too few chunks.
    contains_all(
        &failure(&retarget(&good(), |m| m.row_groups[0].columns.clear())),
        &["column"],
    );
    // A file without columns.
    let none = build(vec![], &[]);
    let text = failure(&none);
    assert!(text.starts_with("parquet: "), "{text}");
}

// ---------------------------------------------------------------------------
// Each unsupported feature names itself.
// ---------------------------------------------------------------------------

#[test]
fn unsupported_value_encodings_name_themselves() {
    for (encoding_number, name) in [
        (encoding::DELTA_BINARY_PACKED, "DELTA_BINARY_PACKED"),
        (encoding::DELTA_LENGTH_BYTE_ARRAY, "DELTA_LENGTH_BYTE_ARRAY"),
        (encoding::DELTA_BYTE_ARRAY, "DELTA_BYTE_ARRAY"),
        (encoding::BYTE_STREAM_SPLIT, "BYTE_STREAM_SPLIT"),
        (encoding::BIT_PACKED, "BIT_PACKED"),
        (encoding::RLE, "RLE"),
        (77, "encoding 77"),
    ] {
        let doubles_file = double_file(
            1,
            vec![page_with_encoding(1, doubles(&[1.0]), encoding_number)],
        );
        contains_all(&failure(&doubles_file), &[name, "not supported"]);
        let strings_file = string_file(
            1,
            vec![page_with_encoding(1, strings(&["x"]), encoding_number)],
        );
        contains_all(&failure(&strings_file), &[name, "not supported"]);
    }
    // A dictionary page that is not PLAIN.
    let dictionary = page(
        page_type::DICTIONARY_PAGE,
        doubles(&[1.0]),
        8,
        None,
        Some(DictionaryPageHeader {
            num_values: 1,
            encoding: encoding::DELTA_BINARY_PACKED,
            is_sorted: None,
        }),
        None,
    );
    contains_all(
        &failure(&double_file(1, vec![dictionary])),
        &["DELTA_BINARY_PACKED"],
    );
    // Definition levels that are not RLE on an OPTIONAL column.
    let mut levels = plain_page(1, [vec![1, 0, 0, 0, 0], doubles(&[1.0])].concat());
    levels
        .0
        .data_page
        .as_mut()
        .unwrap()
        .definition_level_encoding = encoding::BIT_PACKED;
    let bytes = build(
        vec![Chunk {
            name: "d",
            physical: physical::DOUBLE,
            codec: codec_id::UNCOMPRESSED,
            rows: 1,
            pages: vec![levels],
            optional: true,
        }],
        &[],
    );
    contains_all(&failure(&bytes), &["BIT_PACKED", "definition levels"]);
}

#[test]
fn unsupported_codecs_and_page_types_name_themselves() {
    let frame = plain_page(1, doubles(&[1.0]));
    for (number, name) in [
        (codec_id::GZIP, "GZIP"),
        (codec_id::LZO, "LZO"),
        (codec_id::BROTLI, "BROTLI"),
        (codec_id::LZ4, "LZ4"),
        (codec_id::LZ4_RAW, "LZ4_RAW"),
        (99, "compression codec 99"),
        (-3, "compression codec -3"),
    ] {
        let bytes = compressed_file(number, 1, vec![frame.clone()]);
        contains_all(&failure(&bytes), &[name, "re-save the file with zstd"]);
    }
    // Page types: INDEX_PAGE, an unknown type, and a data page without its header.
    let index = page(page_type::INDEX_PAGE, vec![1, 2, 3], 3, None, None, None);
    contains_all(
        &failure(&double_file(
            1,
            vec![plain_page(1, doubles(&[1.0])), index.clone()],
        )),
        &["INDEX_PAGE"],
    );
    contains_all(&failure(&double_file(1, vec![index])), &["INDEX_PAGE"]);
    let odd = page(42, vec![0], 1, None, None, None);
    contains_all(&failure(&double_file(1, vec![odd])), &["page type 42"]);
    let headerless = page(page_type::DATA_PAGE, doubles(&[1.0]), 8, None, None, None);
    contains_all(&failure(&double_file(1, vec![headerless])), &["header"]);
    let headerless = page(
        page_type::DATA_PAGE_V2,
        doubles(&[1.0]),
        8,
        None,
        None,
        None,
    );
    contains_all(&failure(&double_file(1, vec![headerless])), &["header"]);
    let headerless = page(
        page_type::DICTIONARY_PAGE,
        doubles(&[1.0]),
        8,
        None,
        None,
        None,
    );
    contains_all(&failure(&double_file(1, vec![headerless])), &["header"]);
}

#[test]
fn unsupported_types_and_layouts_name_themselves() {
    // Physical types in the schema.
    for (ty, name) in [
        (physical::BOOLEAN, "BOOLEAN"),
        (physical::INT32, "INT32"),
        (physical::INT64, "INT64"),
        (physical::INT96, "INT96"),
        (physical::FLOAT, "FLOAT"),
        (physical::FIXED_LEN_BYTE_ARRAY, "FIXED_LEN_BYTE_ARRAY"),
    ] {
        let bytes = build(
            vec![Chunk {
                name: "x",
                physical: ty,
                codec: codec_id::UNCOMPRESSED,
                rows: 1,
                pages: vec![plain_page(1, vec![0; 16])],
                optional: false,
            }],
            &[],
        );
        contains_all(&failure(&bytes), &[name, "supported"]);
    }
    // REPEATED and nested columns, an external file_path, repeated names.
    let good = || double_file(1, vec![plain_page(1, doubles(&[1.0]))]);
    contains_all(
        &failure(&retarget(&good(), |m| {
            m.schema[1].repetition = Some(repetition::REPEATED)
        })),
        &["REPEATED", "not supported"],
    );
    contains_all(
        &failure(&retarget(&good(), |m| m.schema[1].num_children = Some(2))),
        &["nested"],
    );
    contains_all(
        &failure(&retarget(&good(), |m| {
            m.row_groups[0].columns[0].file_path = Some("other.parquet".into())
        })),
        &["external", "file_path"],
    );
    let two = build(
        vec![
            Chunk {
                name: "x",
                physical: physical::DOUBLE,
                codec: codec_id::UNCOMPRESSED,
                rows: 1,
                pages: vec![plain_page(1, doubles(&[1.0]))],
                optional: false,
            },
            Chunk {
                name: "x",
                physical: physical::DOUBLE,
                codec: codec_id::UNCOMPRESSED,
                rows: 1,
                pages: vec![plain_page(1, doubles(&[1.0]))],
                optional: false,
            },
        ],
        &[],
    );
    contains_all(&failure(&two), &["repeated"]);
    // Repetition levels in a v2 page of a flat column.
    let v2 = DataPageHeaderV2 {
        num_values: 1,
        num_nulls: 0,
        num_rows: 1,
        encoding: encoding::PLAIN,
        definition_levels_byte_length: 0,
        repetition_levels_byte_length: 2,
        is_compressed: false,
    };
    let body = [vec![0u8, 0], doubles(&[1.0])].concat();
    let size = body.len();
    let p = page(page_type::DATA_PAGE_V2, body, size, None, None, Some(v2));
    contains_all(&failure(&double_file(1, vec![p])), &["repeated"]);
}

#[test]
fn nulls_raise_in_every_page_version() {
    // v1, OPTIONAL: definition levels 1,0,1 as one bit-packed group.
    let mut with_null = vec![2, 0, 0, 0, 0b0000_0011, 0b0000_0101];
    with_null.extend_from_slice(&doubles(&[1.0, 3.0]));
    let optional = |body: Vec<u8>, rows: i32| {
        build(
            vec![Chunk {
                name: "d",
                physical: physical::DOUBLE,
                codec: codec_id::UNCOMPRESSED,
                rows: i64::from(rows),
                pages: vec![plain_page(rows, body)],
                optional: true,
            }],
            &[],
        )
    };
    contains_all(&failure(&optional(with_null, 3)), &["null"]);
    // v1, OPTIONAL, a single RLE run of definition level 0.
    let all_null = vec![2, 0, 0, 0, 0b0000_0110, 0];
    contains_all(&failure(&optional(all_null, 3)), &["null"]);
    // The same OPTIONAL column without a null reads.
    let mut present = vec![2, 0, 0, 0, 0b0000_0110, 1];
    present.extend_from_slice(&doubles(&[1.0, 2.0, 3.0]));
    let table = reads(&optional(present, 3));
    assert_eq!(
        table.column("d").unwrap().as_f64().unwrap(),
        [1.0, 2.0, 3.0]
    );
    // v2: num_nulls > 0.
    for nulls in [1, 2, i32::MAX] {
        let v2 = DataPageHeaderV2 {
            num_values: 3,
            num_nulls: nulls,
            num_rows: 3,
            encoding: encoding::PLAIN,
            definition_levels_byte_length: 0,
            repetition_levels_byte_length: 0,
            is_compressed: false,
        };
        let body = doubles(&[1.0, 2.0]);
        let size = body.len();
        let p = page(page_type::DATA_PAGE_V2, body, size, None, None, Some(v2));
        contains_all(&failure(&double_file(3, vec![p])), &["null"]);
    }
    // Frozen null fixtures.
    for name in [
        "negative/null_in_real_pyarrow.parquet",
        "negative/null_in_real_fastparquet.parquet",
        "negative/null_in_pauli_pyarrow.parquet",
        "negative/nan_in_real_fastparquet.parquet",
    ] {
        contains_all(&failure(&fixture(name)), &["null"]);
    }
}

#[test]
fn invalid_text_and_broken_dictionaries_raise() {
    // Invalid UTF-8, in a PLAIN page and in a dictionary.
    let mut body = Vec::new();
    body.extend_from_slice(&2u32.to_le_bytes());
    body.extend_from_slice(&[0xff, 0xfe]);
    contains_all(
        &failure(&string_file(1, vec![plain_page(1, body.clone())])),
        &["UTF-8"],
    );
    let size = body.len();
    let dictionary = page(
        page_type::DICTIONARY_PAGE,
        body,
        size,
        None,
        Some(DictionaryPageHeader {
            num_values: 1,
            encoding: encoding::PLAIN,
            is_sorted: None,
        }),
        None,
    );
    let indices = page_with_encoding(1, vec![0, 3, 0], encoding::RLE_DICTIONARY);
    contains_all(
        &failure(&string_file(1, vec![dictionary, indices])),
        &["UTF-8"],
    );
    // Surrogate halves and overlong encodings are not UTF-8 either.
    for bad in [
        &[0xed, 0xa0, 0x80][..],
        &[0xc0, 0x80],
        &[0xf8, 0x88, 0x80, 0x80, 0x80],
    ] {
        let mut body = Vec::new();
        body.extend_from_slice(&(bad.len() as u32).to_le_bytes());
        body.extend_from_slice(bad);
        contains_all(
            &failure(&string_file(1, vec![plain_page(1, body)])),
            &["UTF-8"],
        );
    }
    // A string longer than its page, and a length prefix of 4 GB.
    let mut body = Vec::new();
    body.extend_from_slice(&100u32.to_le_bytes());
    body.extend_from_slice(b"short");
    assert!(failure(&string_file(1, vec![plain_page(1, body)])).starts_with("parquet: "));
    let mut body = Vec::new();
    body.extend_from_slice(&u32::MAX.to_le_bytes());
    body.extend_from_slice(b"short");
    assert!(failure(&string_file(1, vec![plain_page(1, body)])).starts_with("parquet: "));
    // A dictionary that claims more entries than its page can hold.
    for entries in [1_000_000i32, i32::MAX, -1] {
        let dictionary = page(
            page_type::DICTIONARY_PAGE,
            strings(&["a"]),
            5,
            None,
            Some(DictionaryPageHeader {
                num_values: entries,
                encoding: encoding::PLAIN,
                is_sorted: None,
            }),
            None,
        );
        let text = failure(&string_file(1, vec![dictionary]));
        assert!(text.starts_with("parquet: "), "{text}");
    }
    // An index outside the dictionary, and no dictionary at all.
    let dictionary = page(
        page_type::DICTIONARY_PAGE,
        strings(&["a"]),
        5,
        None,
        Some(DictionaryPageHeader {
            num_values: 1,
            encoding: encoding::PLAIN,
            is_sorted: None,
        }),
        None,
    );
    // width 2, one bit-packed group of 8, first index 3.
    let out_of_range =
        page_with_encoding(1, vec![2, (1 << 1) | 1, 0b11, 0], encoding::RLE_DICTIONARY);
    contains_all(
        &failure(&string_file(
            1,
            vec![dictionary.clone(), out_of_range.clone()],
        )),
        &["out of range"],
    );
    contains_all(
        &failure(&string_file(1, vec![out_of_range])),
        &["dictionary"],
    );
    // A dictionary index width above 32 bits.
    let wide = page_with_encoding(1, vec![40, 2, 0], encoding::RLE_DICTIONARY);
    assert!(failure(&string_file(1, vec![dictionary, wide])).starts_with("parquet: "));
}

#[test]
fn page_checksums_are_enforced() {
    let mut p = plain_page(1, doubles(&[1.0]));
    p.0.crc = Some(0x1234_5678);
    contains_all(&failure(&double_file(1, vec![p])), &["checksum"]);
    // A checksum that is right reads.
    let mut p = plain_page(1, doubles(&[1.0]));
    p.0.crc = Some(parquet::crc32::crc32(&p.1) as i32);
    reads(&double_file(1, vec![p]));
}

// ---------------------------------------------------------------------------
// Compressed pages: the zstd adapter and the snappy decoder.
// ---------------------------------------------------------------------------

const ZSTD_MAGIC: u32 = 0xFD2F_B528;

/// A frame with explicit header fields and one raw block holding `body`.
fn raw_frame(descriptor: u8, window: Option<u8>, header_tail: &[u8], body: &[u8]) -> Vec<u8> {
    let mut frame = ZSTD_MAGIC.to_le_bytes().to_vec();
    frame.push(descriptor);
    frame.extend(window);
    frame.extend_from_slice(header_tail);
    let block = ((body.len() as u32) << 3) | 1; // last block, raw
    frame.extend_from_slice(&block.to_le_bytes()[..3]);
    frame.extend_from_slice(body);
    frame
}

/// A frame the reader accepts: single segment with a one-byte content size.
fn valid_frame(body: &[u8]) -> Vec<u8> {
    assert!(body.len() < 256);
    raw_frame(0x20, None, &[body.len() as u8], body)
}

fn zstd_file(frame: Vec<u8>, rows: i32, uncompressed: usize) -> Vec<u8> {
    let mut p = plain_page(rows, frame);
    p.0.uncompressed_page_size = uncompressed as i32;
    compressed_file(codec_id::ZSTD, i64::from(rows), vec![p])
}

#[test]
fn zstd_frames_the_adapter_must_refuse() {
    let body = doubles(&[1.0, 2.0, 3.0, 4.0]);
    // The control reads.
    let table = reads(&zstd_file(valid_frame(&body), 4, 32));
    assert_eq!(
        table.column("d").unwrap().as_f64().unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );

    // Lying Frame_Content_Size, both directions.
    for fcs in [0u8, 1, 31, 33, 64, 255] {
        let frame = raw_frame(0x20, None, &[fcs], &body);
        let text = failure(&zstd_file(frame, 4, 32));
        assert!(
            text.contains("zstd") || text.contains("content size"),
            "fcs {fcs}: {text}"
        );
    }
    // A header whose uncompressed size differs from the frame's.
    for lie in [0usize, 8, 31, 33, 4096, 1 << 20] {
        failure(&zstd_file(valid_frame(&body), 4, lie));
    }
    // Dictionary id (descriptor bits 0..2), content checksum flag, reserved bit.
    let text = failure(&zstd_file(raw_frame(0x21, None, &[7, 32], &body), 4, 32));
    assert!(text.contains("dictionary"), "{text}");
    let text = failure(&zstd_file(raw_frame(0x24, None, &[32], &body), 4, 32));
    assert!(text.contains("checksum"), "{text}");
    let text = failure(&zstd_file(raw_frame(0x28, None, &[32], &body), 4, 32));
    assert!(text.contains("reserved"), "{text}");
    // No Frame_Content_Size at all.
    let text = failure(&zstd_file(raw_frame(0x00, Some(0x00), &[], &body), 4, 32));
    assert!(text.contains("content size"), "{text}");
    // A window above the 128 MiB limit, declared by the window descriptor
    // (exponent 31) with an 8-byte content size.
    let window = raw_frame(0xC0, Some(0xFF), &32u64.to_le_bytes(), &body);
    let text = failure(&zstd_file(window, 4, 32));
    assert!(text.contains("window"), "{text}");
    // A content size of 2^60 in the same slot.
    let window = raw_frame(0xC0, Some(0x00), &(1u64 << 60).to_le_bytes(), &body);
    failure_under(&zstd_file(window, 4, 32), 4 << 20);
    // Skippable frames, a wrong magic, a truncated header, an empty page.
    let mut skippable = 0x184D_2A50u32.to_le_bytes().to_vec();
    skippable.extend_from_slice(&0u32.to_le_bytes());
    let text = failure(&zstd_file(skippable, 4, 32));
    assert!(text.contains("skippable"), "{text}");
    let text = failure(&zstd_file(vec![1, 2, 3, 4, 5, 6, 7, 8], 4, 32));
    assert!(text.contains("magic") || text.contains("zstd"), "{text}");
    for cut in 0..valid_frame(&body).len() {
        let frame = valid_frame(&body)[..cut].to_vec();
        failure(&zstd_file(frame, 4, 32));
    }
    // Two frames in one page, and bytes after the frame.
    let mut two = valid_frame(&body);
    two.extend_from_slice(&valid_frame(&body));
    failure(&zstd_file(two, 8, 64));
    let mut trailing = valid_frame(&body);
    trailing.extend_from_slice(&[0, 0, 0, 0]);
    failure(&zstd_file(trailing, 4, 32));
    // A compressed block of garbage: the decoder may panic inside; the adapter
    // turns that into an error.
    let mut state = 0xdead_beef_cafe_f00du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    for case in 0..4000 {
        let length = (next() % 120) as usize;
        let junk: Vec<u8> = (0..length).map(|_| next() as u8).collect();
        let mut frame = ZSTD_MAGIC.to_le_bytes().to_vec();
        frame.push(0x20);
        frame.push(32);
        // Block header: compressed type (2), last block, size = junk length.
        let block = ((length as u32) << 3) | (2 << 1) | 1;
        frame.extend_from_slice(&block.to_le_bytes()[..3]);
        frame.extend_from_slice(&junk);
        let mut output = [0u8; 32];
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            zstd_codec::decompress(&ctx, &frame, &mut output)
        }));
        match outcome {
            Ok(_) => {}
            Err(_) => panic!("case {case}: the adapter let a panic through"),
        }
        assert_eq!(ctx.memory_used(), 0);
    }
}

#[test]
fn the_zstd_encoder_and_decoder_agree_on_every_size() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let mut state = 0x0123_4567_89ab_cdefu64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for length in [
        0usize, 1, 2, 7, 8, 255, 256, 257, 4096, 65_535, 65_536, 131_071, 131_072, 131_073,
        300_000, 1_000_000,
    ] {
        let compressible: Vec<u8> = (0..length).map(|i| ((i / 8) % 13) as u8).collect();
        let noise: Vec<u8> = (0..length).map(|_| next() as u8).collect();
        for data in [compressible, noise] {
            // The adapter always returns a frame; whether it pays is the
            // writer's decision per chunk.
            let frame = codec::compress(&ctx, Codec::Zstd, &data).unwrap().unwrap();
            let mut out = vec![0u8; data.len()];
            codec::decompress(&ctx, codec_id::ZSTD, &frame, &mut out).unwrap();
            assert_eq!(out, data);
            // The frame decodes to nothing else: a one-byte-short output is
            // an error.
            if !data.is_empty() {
                let mut short = vec![0u8; data.len() - 1];
                assert!(codec::decompress(&ctx, codec_id::ZSTD, &frame, &mut short).is_err());
            }
            let mut long = vec![0u8; data.len() + 1];
            assert!(codec::decompress(&ctx, codec_id::ZSTD, &frame, &mut long).is_err());
            assert_eq!(ctx.memory_used(), 0);
        }
    }
}

/// The snappy raw format: a varint length, then literals and copies.
fn snappy_literal(data: &[u8]) -> Vec<u8> {
    assert!(data.len() <= 60);
    let mut out = vec![data.len() as u8];
    out.push(((data.len() - 1) << 2) as u8);
    out.extend_from_slice(data);
    out
}

fn snappy_file(body: Vec<u8>, rows: i32, uncompressed: usize) -> Vec<u8> {
    let mut p = plain_page(rows, body);
    p.0.uncompressed_page_size = uncompressed as i32;
    compressed_file(codec_id::SNAPPY, i64::from(rows), vec![p])
}

#[test]
fn snappy_pages_the_decoder_must_refuse() {
    let body = doubles(&[1.0, 2.0, 3.0]); // 24 bytes
                                          // The control: a varint length (24) and one literal.
    let good = snappy_literal(&body);
    let table = reads(&snappy_file(good.clone(), 3, 24));
    assert_eq!(
        table.column("d").unwrap().as_f64().unwrap(),
        [1.0, 2.0, 3.0]
    );

    // A preamble that disagrees with the header, both directions, and huge.
    for preamble in [0u8, 1, 23, 25, 127] {
        let mut lie = good.clone();
        lie[0] = preamble;
        failure(&snappy_file(lie, 3, 24));
    }
    let mut huge = vec![0xff, 0xff, 0xff, 0xff, 0x0f];
    huge.extend_from_slice(&good[1..]);
    failure_under(&snappy_file(huge, 3, 24), 1 << 20);
    let endless = vec![0xff; 40];
    failure(&snappy_file(endless, 3, 24));
    // A literal longer than the input; a truncated literal; an empty page.
    let mut long_literal = vec![24u8, ((60 - 1) << 2) as u8];
    long_literal.extend_from_slice(&body);
    failure(&snappy_file(long_literal, 3, 24));
    for cut in 0..good.len() {
        failure(&snappy_file(good[..cut].to_vec(), 3, 24));
    }
    // Copies: an offset of zero, an offset before the start of the output, a
    // copy running past the end, for the three copy tags.
    // copy-1: tag 0b01 with len in bits 2..4 (len - 4) and offset high bits.
    let copy1 = |offset: u8, len: u8| vec![((len - 4) << 2) | 1, offset];
    let mut zero = vec![24u8];
    zero.extend(&snappy_literal(&body)[1..]);
    zero.extend(copy1(0, 4));
    failure(&snappy_file(zero, 3, 24));
    let mut before = vec![24u8];
    before.extend(&snappy_literal(&body)[1..]);
    before.extend(copy1(200, 4));
    failure(&snappy_file(before, 3, 24));
    // copy-2 (tag 0b10): 2-byte offset little-endian.
    let mut copy2 = vec![24u8];
    copy2.extend(&snappy_literal(&body)[1..]);
    copy2.extend([((8 - 1) << 2) | 2, 0xff, 0xff]);
    failure(&snappy_file(copy2, 3, 24));
    // copy-4 (tag 0b11): 4-byte offset.
    let mut copy4 = vec![24u8];
    copy4.extend(&snappy_literal(&body)[1..]);
    copy4.extend([((8 - 1) << 2) | 3, 0xff, 0xff, 0xff, 0xff]);
    failure(&snappy_file(copy4, 3, 24));
    // An overlapping copy that is valid reads: literal of 8 then copy of 16
    // from offset 8 (run-length style) gives 24 bytes.
    let eight = doubles(&[1.0]);
    let mut overlap = vec![24u8];
    overlap.extend(&snappy_literal(&eight)[1..]);
    overlap.extend([((16 - 1) << 2) | 2, 8, 0]);
    let table = reads(&snappy_file(overlap, 3, 24));
    assert_eq!(
        table.column("d").unwrap().as_f64().unwrap(),
        [1.0, 1.0, 1.0]
    );
    // Random bytes as a snappy page never panic and never leak.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let ctx = Context::new(Some(1), 16 << 20).unwrap();
    for case in 0..20_000 {
        let length = (next() % 64) as usize;
        let mut junk: Vec<u8> = (0..length).map(|_| next() as u8).collect();
        if case % 2 == 0 && !junk.is_empty() {
            junk[0] = 24; // a plausible preamble, so the elements run
        }
        let mut output = [0u8; 24];
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            codec::decompress(&ctx, codec_id::SNAPPY, &junk, &mut output)
        }));
        assert!(outcome.is_ok(), "case {case}: snappy panicked on {junk:?}");
        assert_eq!(ctx.memory_used(), 0);
    }
}

// ---------------------------------------------------------------------------
// Truncation and mutation sweeps.
// ---------------------------------------------------------------------------

#[test]
fn truncating_every_fixture_at_every_length_never_panics_or_leaks() {
    let dir = Directory::new();
    let path = dir.file("t.parquet");
    let mut cases = 0usize;
    for (name, bytes) in all_fixtures() {
        if name.starts_with("negative/") {
            continue;
        }
        for length in 0..bytes.len() {
            fs::write(&path, &bytes[..length]).unwrap();
            let outcome = catch_unwind(AssertUnwindSafe(|| probe(&path, 8 << 20)))
                .unwrap_or_else(|_| panic!("{name}: a prefix of {length} bytes panicked"));
            assert!(
                outcome.read.is_err() && outcome.metadata.is_err(),
                "{name}: a prefix of {length} of {} bytes opened",
                bytes.len()
            );
            cases += 1;
        }
    }
    eprintln!("truncation sweep: {cases} prefixes, all refused");
    assert!(cases > 100_000, "{cases}");
}

fn weight(part: Part) -> usize {
    match part {
        Part::Head => 1,
        Part::Trailer => 2,
        Part::Footer => 6,
        Part::PageHeader => 6,
        Part::ZstdBody | Part::SnappyBody => 6,
        Part::PlainBody => 3,
    }
}

const MUTATION_TARGETS: [&str; 16] = [
    "hamiltonian/pyarrow_zstd.parquet",
    "hamiltonian/fastparquet_zstd.parquet",
    "hamiltonian/pyarrow_zstd_data_page_v2.parquet",
    "hamiltonian/pyarrow_required_zstd.parquet",
    "hamiltonian/pyarrow_snappy_dictionary.parquet",
    "hamiltonian/pyarrow_snappy_data_page_v2_plain.parquet",
    "hamiltonian/fastparquet_snappy.parquet",
    "hamiltonian/pyarrow_dictionary_fallback.parquet",
    "hamiltonian/pyarrow_multi_page_dictionary.parquet",
    "hamiltonian/pyarrow_uncompressed_plain.parquet",
    "hamiltonian/fastparquet_multi_row_group.parquet",
    "hamiltonian/pyarrow_page_checksum.parquet",
    "hamiltonian/fastparquet_empty.parquet",
    "pseudopotential/fastparquet_zstd.parquet",
    "pseudopotential/pyarrow_snappy.parquet",
    "generic/pyarrow_special_values_zstd.parquet",
];

/// Cases of the main mutation sweep: well above the 100,000 the plan asks for.
const MUTATION_CASES: usize = 120_000;

#[test]
fn a_seeded_mutation_sweep_never_panics_or_allocates_beyond_the_budget() {
    let targets: Vec<Target> = MUTATION_TARGETS.iter().map(|n| target(n)).collect();
    // Every region kind must be present among the targets, or the sweep would
    // silently skip the part it is meant to cover.
    for part in [
        Part::Footer,
        Part::PageHeader,
        Part::ZstdBody,
        Part::SnappyBody,
        Part::PlainBody,
    ] {
        assert!(
            targets
                .iter()
                .any(|t| t.parts.iter().any(|(p, _)| *p == part)),
            "no target has a {part:?} region"
        );
    }
    let dir = Directory::new();
    let path = dir.file("m.parquet");
    let limits = [1usize << 18, 1 << 22, 1 << 25];
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let mut read_ok = 0usize;
    let mut refused = 0usize;
    let mut budget = 0usize;
    let mut per_part = std::collections::BTreeMap::<String, [usize; 2]>::new();
    for case in 0..MUTATION_CASES {
        let target = &targets[case % targets.len()];
        let total: usize = target.parts.iter().map(|(p, _)| weight(*p)).sum();
        let mut pick = rng.below(total);
        let (part, range) = target
            .parts
            .iter()
            .find(|(p, _)| {
                if pick < weight(*p) {
                    true
                } else {
                    pick -= weight(*p);
                    false
                }
            })
            .unwrap();
        let mut bytes = target.bytes.clone();
        let how = mutate(&mut rng, &mut bytes, range);
        fs::write(&path, &bytes).unwrap();
        let limit = limits[rng.below(limits.len())];
        let outcome = catch_unwind(AssertUnwindSafe(|| probe(&path, limit))).unwrap_or_else(|_| {
            // Keep the offending input for debugging.
            let keep = std::env::temp_dir().join("mandacaru-parquet-panic.parquet");
            let _ = fs::copy(&path, &keep);
            panic!(
                "case {case}: {how} in {part:?} of {} (limit {limit}) panicked; input kept at {}",
                target.name,
                keep.display()
            )
        });
        let entry = per_part.entry(format!("{part:?}")).or_default();
        match &outcome.read {
            Ok(_) => {
                read_ok += 1;
                entry[0] += 1;
            }
            Err(Error::Budget { .. }) => {
                budget += 1;
                refused += 1;
                entry[1] += 1;
            }
            Err(_) => {
                refused += 1;
                entry[1] += 1;
            }
        }
    }
    eprintln!(
        "mutation sweep: {MUTATION_CASES} cases, {read_ok} still read, {refused} refused ({budget} by the budget); per part [read, refused]: {per_part:?}; largest unaccounted allocation {} bytes",
        WORST_EXCESS.load(Ordering::Relaxed)
    );
    assert!(refused > MUTATION_CASES / 4, "{refused} refused");
    assert!(
        read_ok > 0,
        "no mutated file still read: the sweep is too destructive"
    );
}

#[test]
fn a_mutated_hamiltonian_file_never_panics_through_document_read() {
    use mandacaru_core::persistence::{Document, HAMILTONIAN};
    let names = [
        "hamiltonian/pyarrow_zstd.parquet",
        "hamiltonian/fastparquet_zstd.parquet",
        "hamiltonian/pyarrow_snappy_dictionary.parquet",
        "hamiltonian/fastparquet_uncompressed.parquet",
        "hamiltonian/pyarrow_empty.parquet",
    ];
    let targets: Vec<Target> = names.iter().map(|n| target(n)).collect();
    let dir = Directory::new();
    let path = dir.file("m.parquet");
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut loaded = 0usize;
    for case in 0..20_000 {
        let target = &targets[case % targets.len()];
        let (_, range) = &target.parts[rng.below(target.parts.len())];
        let mut bytes = target.bytes.clone();
        let how = mutate(&mut rng, &mut bytes, range);
        fs::write(&path, &bytes).unwrap();
        let ctx = Context::new(Some(1), 1 << 24).unwrap();
        let (outcome, peak) = measured(|| {
            catch_unwind(AssertUnwindSafe(|| {
                Document::read(&ctx, &path, HAMILTONIAN)
                    .map(|doc| doc.payload()["terms"].is_array())
            }))
        });
        let outcome = outcome.unwrap_or_else(|_| panic!("case {case}: {how} panicked"));
        if let Ok(true) = outcome {
            loaded += 1;
        }
        assert_eq!(ctx.memory_used(), 0, "case {case}: {how}");
        assert!(
            peak <= ctx.memory_peak() + SLACK,
            "case {case}: {how}: allocated {peak} live bytes, reserved {}",
            ctx.memory_peak()
        );
    }
    eprintln!("document mutation sweep: 20000 cases, {loaded} still loaded");
}

// ---------------------------------------------------------------------------
// Review fixes: charged copies, annotations, empty chunks.
// ---------------------------------------------------------------------------

/// A one-row double file whose footer carries `entries` short key-value pairs:
/// few footer bytes per entry, many allocated bytes per entry, which is the
/// shape that made the uncharged copies exceed the budget.
fn file_with_entries(entries: usize) -> Vec<u8> {
    let pairs: Vec<(String, String)> = (0..entries)
        .map(|i| (format!("{i:x}"), "v".to_owned()))
        .collect();
    let refs: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    build(
        vec![Chunk {
            name: "d",
            physical: physical::DOUBLE,
            codec: codec_id::UNCOMPRESSED,
            rows: 1,
            pages: vec![plain_page(1, doubles(&[1.0]))],
            optional: false,
        }],
        &refs,
    )
}

#[test]
fn what_is_built_after_the_footer_parse_is_charged_before_it_is_built() {
    // The sorted metadata map, the leaves and the output wrappers are
    // allocated after the footer parse; they used to escape the budget, so a
    // read could allocate more than the context allowed and still return Ok.
    let dir = Directory::new();
    let path = dir.file("many.parquet");
    fs::write(&path, file_with_entries(400_000)).unwrap();
    // The invariants of every probe, at the default budget.
    let outcome = probe(&path, 512 << 20);
    assert!(outcome.read.is_ok());
    // And at the smallest budget that reads: nothing allocated beyond it.
    let reads_in = |limit: usize| {
        let ctx = Context::new(Some(1), limit).unwrap();
        let (outcome, peak) = measured(|| read_table(&ctx, &path).map(|t| t.metadata.len()));
        assert_eq!(ctx.memory_used(), 0);
        assert!(
            peak <= limit + SLACK,
            "limit {limit}: allocated {peak} live bytes (outcome {outcome:?})"
        );
        outcome.is_ok()
    };
    let (mut low, mut high) = (1usize << 20, 512usize << 20);
    assert!(reads_in(high));
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if reads_in(middle) {
            high = middle;
        } else {
            low = middle;
        }
    }
    assert!(!reads_in(high - 1));
    // A footer whose copies cannot fit is refused by the budget, not read.
    let ctx = Context::new(Some(1), high / 2).unwrap();
    assert!(matches!(read_table(&ctx, &path), Err(Error::Budget { .. })));
}

#[test]
fn a_byte_array_annotated_as_something_other_than_text_is_refused() {
    let good = || string_file(2, vec![plain_page(2, strings(&["IXYZ", "ZZZZ"]))]);
    // Accepted: none, UTF8, ENUM, JSON (converted and logical forms).
    for (converted, logical) in [
        (None, None),
        (Some(0), None),
        (Some(4), None),
        (Some(19), None),
        (None, Some(4i16)),
        (None, Some(12)),
    ] {
        let bytes = retarget(&good(), |m| {
            m.schema[1].converted_type = converted;
            m.schema[1].logical_string = false;
            m.schema[1].logical_other = logical;
        });
        assert_eq!(reads(&bytes).num_rows(), 2, "{converted:?} {logical:?}");
    }
    // Refused, naming the annotation: DECIMAL, BSON, INTERVAL, UINT_8 and the
    // logical members DECIMAL, BSON, UUID.
    for (converted, name) in [(5, "DECIMAL"), (20, "BSON"), (21, "INTERVAL"), (11, "UINT")] {
        let bytes = retarget(&good(), |m| m.schema[1].converted_type = Some(converted));
        contains_all(&failure(&bytes), &[name, "not text"]);
    }
    for (member, name) in [(5i16, "DECIMAL"), (13, "BSON"), (14, "UUID")] {
        let bytes = retarget(&good(), |m| {
            m.schema[1].logical_string = false;
            m.schema[1].logical_other = Some(member);
        });
        contains_all(&failure(&bytes), &[name, "not text"]);
    }
}

#[test]
fn a_zero_row_chunk_without_offsets_reads_as_pyarrow_writes_it() {
    // pyarrow with use_dictionary=False stores zero rows as one row group
    // whose chunks have data_page_offset 0 and no bytes at all.
    let bytes = build_with(
        vec![
            Chunk {
                name: "pauli",
                physical: physical::BYTE_ARRAY,
                codec: codec_id::ZSTD,
                rows: 0,
                pages: vec![],
                optional: false,
            },
            Chunk {
                name: "real",
                physical: physical::DOUBLE,
                codec: codec_id::ZSTD,
                rows: 0,
                pages: vec![],
                optional: false,
            },
        ],
        &[("mandacaru.num_qubits", "4")],
        |meta| {
            for chunk in &mut meta.row_groups[0].columns {
                let m = &mut chunk.meta_data;
                m.data_page_offset = 0;
                m.dictionary_page_offset = None;
                m.total_compressed_size = 0;
                m.total_uncompressed_size = 0;
            }
        },
    );
    let table = reads(&bytes);
    assert_eq!(table.num_rows(), 0);
    assert_eq!(table.column_names(), vec!["pauli", "real"]);
    assert_eq!(table.metadata["mandacaru.num_qubits"], "4");
    let dir = Directory::new();
    let path = dir.file("t.parquet");
    fs::write(&path, &bytes).unwrap();
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    let footer = read_metadata(&ctx, &path).unwrap();
    assert_eq!((footer.num_rows, footer.columns.len()), (0, 2));
    // A chunk with rows but no bytes still fails, as before.
    let lying = build_with(
        vec![Chunk {
            name: "d",
            physical: physical::DOUBLE,
            codec: codec_id::UNCOMPRESSED,
            rows: 1,
            pages: vec![],
            optional: false,
        }],
        &[],
        |meta| {
            let m = &mut meta.row_groups[0].columns[0].meta_data;
            m.data_page_offset = 0;
            m.total_compressed_size = 0;
        },
    );
    contains_all(&failure(&lying), &["offset 0"]);
}

// ---------------------------------------------------------------------------
// Decompression work is bounded by the data a file holds, not by what its
// page headers claim.
// ---------------------------------------------------------------------------

/// A zstd frame of `blocks` RLE blocks of 128 KiB each: 4 bytes of file per
/// 128 KiB of output. The header claims `blocks * 128 KiB` bytes.
fn rle_frame(blocks: usize) -> Vec<u8> {
    const BLOCK: u32 = 128 << 10;
    let total = BLOCK * blocks as u32;
    let mut frame = ZSTD_MAGIC.to_le_bytes().to_vec();
    frame.push(0xA0); // single segment, 4-byte content size
    frame.extend_from_slice(&total.to_le_bytes());
    for block in 0..blocks {
        let last = u32::from(block + 1 == blocks);
        let header = (BLOCK << 3) | (1 << 1) | last; // RLE block
        frame.extend_from_slice(&header.to_le_bytes()[..3]);
        frame.push(0);
    }
    frame
}

/// A data page of `values` doubles that says it decompresses to 64 MiB.
fn lying_zstd_page(values: i32) -> Page {
    let frame = rle_frame(512);
    let mut p = plain_page(values, frame);
    p.0.uncompressed_page_size = 64 << 20;
    p
}

fn double_chunk(rows: i64, pages: Vec<Page>) -> Chunk {
    Chunk {
        name: "d",
        physical: physical::DOUBLE,
        codec: codec_id::ZSTD,
        rows,
        pages,
        optional: false,
    }
}

#[test]
fn row_groups_that_share_one_compressed_page_are_refused_without_decoding() {
    // 2000 row groups, each pointing at the same small zstd page that claims
    // 64 MiB of output. Decoding it per group would take minutes.
    for rows in [0i64, 1] {
        let bytes = build_with(
            vec![double_chunk(rows, vec![lying_zstd_page(rows as i32)])],
            &[],
            |meta| {
                let group = meta.row_groups[0].clone();
                meta.row_groups = vec![group; 2000];
                meta.num_rows = rows * 2000;
            },
        );
        assert!(bytes.len() < 1 << 20, "{} bytes", bytes.len());
        let started = std::time::Instant::now();
        let text = failure(&bytes);
        assert!(
            text.contains("overlapping"),
            "rows {rows}: unexpected message {text}"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "rows {rows}: refusing took {:?}",
            started.elapsed()
        );
    }
}

#[test]
fn pages_without_values_are_never_decompressed() {
    // 200 pages of zero values, each claiming 64 MiB, then the real page. The
    // empty pages must cost nothing; decoding them would take tens of seconds.
    let mut pages: Vec<Page> = (0..200).map(|_| lying_zstd_page(0)).collect();
    let body = valid_frame(&doubles(&[2.5]));
    let mut real = plain_page(1, body);
    real.0.uncompressed_page_size = 8;
    pages.push(real);
    let bytes = build(vec![double_chunk(1, pages)], &[]);
    let started = std::time::Instant::now();
    let table = reads(&bytes);
    assert_eq!(table.column("d").unwrap().as_f64().unwrap(), &[2.5]);
    assert!(
        started.elapsed() < std::time::Duration::from_secs(2),
        "empty pages took {:?}",
        started.elapsed()
    );
}

#[test]
fn a_page_that_claims_more_output_than_its_values_can_use_is_refused_first() {
    // One value, a 64 MiB claim: refused from the header, before any decoding.
    let bytes = build(vec![double_chunk(1, vec![lying_zstd_page(1)])], &[]);
    let started = std::time::Instant::now();
    let text = failure(&bytes);
    contains_all(&text, &["claims", "uncompressed bytes"]);
    assert!(started.elapsed() < std::time::Duration::from_secs(2));
}

#[test]
fn a_zero_row_chunk_with_an_empty_dictionary_page_still_reads() {
    // pyarrow with dictionaries on writes an empty dictionary page for zero
    // rows; the chunk has real bytes but nothing to decode.
    let dictionary = page(
        page_type::DICTIONARY_PAGE,
        vec![],
        0,
        None,
        Some(DictionaryPageHeader {
            num_values: 0,
            encoding: encoding::PLAIN,
            is_sorted: None,
        }),
        None,
    );
    let bytes = build(
        vec![Chunk {
            name: "d",
            physical: physical::DOUBLE,
            codec: codec_id::UNCOMPRESSED,
            rows: 0,
            pages: vec![dictionary],
            optional: false,
        }],
        &[],
    );
    assert_eq!(reads(&bytes).num_rows(), 0);
}

#[test]
fn a_large_count_entry_is_refused_before_it_becomes_a_json_tree() {
    use mandacaru_core::persistence::{Document, HAMILTONIAN};
    // A zero-term Hamiltonian whose particle-count entry is 1M small numbers:
    // parsed into a JSON tree it would take about 16 times its text.
    let big = format!("[{}0]", "0,".repeat(1_000_000));
    let chunks = |_: ()| {
        vec![
            Chunk {
                name: "pauli",
                physical: physical::BYTE_ARRAY,
                codec: codec_id::UNCOMPRESSED,
                rows: 0,
                pages: vec![],
                optional: false,
            },
            Chunk {
                name: "real",
                physical: physical::DOUBLE,
                codec: codec_id::UNCOMPRESSED,
                rows: 0,
                pages: vec![],
                optional: false,
            },
            Chunk {
                name: "imag",
                physical: physical::DOUBLE,
                codec: codec_id::UNCOMPRESSED,
                rows: 0,
                pages: vec![],
                optional: false,
            },
        ]
    };
    let dir = Directory::new();
    for key in ["mandacaru.num_particles", "mandacaru.n_spatial_orbitals"] {
        let bytes = build(
            chunks(()),
            &[
                ("mandacaru.format", HAMILTONIAN),
                ("mandacaru.version", "1"),
                ("mandacaru.num_qubits", "2"),
                (key, big.as_str()),
            ],
        );
        let path = dir.file("big.parquet");
        fs::write(&path, &bytes).unwrap();
        let ctx = Context::new(Some(1), 512 << 20).unwrap();
        let (outcome, peak) = measured(|| Document::read(&ctx, &path, HAMILTONIAN).map(|_| ()));
        let text = format!(
            "{:?}",
            outcome.expect_err("an absurd count entry must be refused")
        );
        assert!(text.contains(key) && text.contains("at most"), "{text}");
        assert_eq!(ctx.memory_used(), 0);
        assert!(
            peak <= ctx.memory_peak() + SLACK,
            "{key}: allocated {peak} live bytes, reserved {}",
            ctx.memory_peak()
        );
    }
}
