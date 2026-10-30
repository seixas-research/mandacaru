//! What the Parquet reader allocates, measured over the whole process.
//!
//! `parquet_errors.rs` counts the allocations of the calling thread only, so
//! it cannot see the zstd decoder (which runs on its own thread) or the column
//! workers. This binary installs a counting global allocator whose counters
//! are process-wide atomics, and every test takes `GATE` first, so exactly one
//! test allocates at a time and a measurement holds nothing but its own work.
//!
//! The allocator also enforces an emulated hard limit (`HARD_LIMIT`): an
//! allocation that would take the live bytes past it fails, so a regression
//! that lets a hostile frame expand to gigabytes stops the test process
//! instead of the machine.
//!
//! The invariant under test is the one the plan calls "never allocate beyond
//! the `Context` budget": the live bytes of a call never exceed what the call
//! reserved (`ctx.memory_peak()`) plus a small constant, for every input.

mod parquet_support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use mandacaru_core::parquet::{
    codec::zstd as zstd_codec,
    format::{
        codec as codec_id, encoding, page_type, physical, ColumnChunk, ColumnMetaData,
        DataPageHeader, FileMetaData, PageHeader, RowGroup, SchemaElement,
    },
    read_table, MAGIC,
};
use mandacaru_core::{Context, Error};
use parquet_support::{mutate, target, Directory, Part, Rng};

// ---------------------------------------------------------------------------
// The process-wide counting allocator.
// ---------------------------------------------------------------------------

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// Live bytes past which an allocation fails (the test process stops, the
/// machine does not swap).
const HARD_LIMIT: usize = 300 << 20;

fn grow(size: usize) -> bool {
    let now = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    if now > HARD_LIMIT {
        LIVE.fetch_sub(size, Ordering::Relaxed);
        return false;
    }
    PEAK.fetch_max(now, Ordering::Relaxed);
    true
}

fn shrink(size: usize) {
    LIVE.fetch_sub(size, Ordering::Relaxed);
}

// SAFETY: every call is forwarded unchanged to `System`, which upholds the
// `GlobalAlloc` contract; the additions are atomic counters that never
// allocate, and a refused growth returns null as the contract allows.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !grow(layout.size()) {
            return std::ptr::null_mut();
        }
        let pointer = System.alloc(layout);
        if pointer.is_null() {
            shrink(layout.size());
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !grow(layout.size()) {
            return std::ptr::null_mut();
        }
        let pointer = System.alloc_zeroed(layout);
        if pointer.is_null() {
            shrink(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        shrink(layout.size());
        System.dealloc(pointer, layout)
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() {
            if !grow(new_size - layout.size()) {
                return std::ptr::null_mut();
            }
        } else {
            shrink(layout.size() - new_size);
        }
        let moved = System.realloc(pointer, layout, new_size);
        if moved.is_null() {
            // Undo the bookkeeping of the call that failed.
            if new_size > layout.size() {
                shrink(new_size - layout.size());
            } else {
                LIVE.fetch_add(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        moved
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// One measurement at a time.
static GATE: Mutex<()> = Mutex::new(());

fn gate() -> MutexGuard<'static, ()> {
    GATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Peak live bytes above the level at entry, on every thread, while `f` runs.
/// The caller holds the gate.
fn measured<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let value = f();
    (value, PEAK.load(Ordering::Relaxed).saturating_sub(base))
}

/// What a call may allocate without reserving it: error text, path strings,
/// the `Vec` headers of the footer, thread bookkeeping.
const SLACK: usize = 1 << 20;

// ---------------------------------------------------------------------------
// Hostile zstd frames.
// ---------------------------------------------------------------------------

const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// A frame of `fcs` content bytes, a 4-byte content size and the given window
/// descriptor, around the given blocks (the last one must carry the flag).
fn frame_of(window: u8, fcs: u32, blocks: &[u8]) -> Vec<u8> {
    let mut frame = ZSTD_MAGIC.to_vec();
    frame.push(0x80);
    frame.push(window);
    frame.extend_from_slice(&fcs.to_le_bytes());
    frame.extend_from_slice(blocks);
    frame
}

fn block_header(last: bool, kind: u32, size: usize) -> [u8; 3] {
    let value = ((size as u32) << 3) | (kind << 1) | u32::from(last);
    let bytes = value.to_le_bytes();
    [bytes[0], bytes[1], bytes[2]]
}

/// One compressed block that expands to about 131 KB per sequence: raw
/// literals (`n` bytes), `n` sequences in RLE mode (literal-length code 1,
/// offset code 0, match-length code 52 with sixteen extra bits).
fn sequence_block(n: usize) -> Vec<u8> {
    let mut b = Vec::new();
    let header = (3u32 << 2) | ((n as u32) << 4);
    b.extend_from_slice(&header.to_le_bytes()[..3]);
    b.extend(std::iter::repeat_n(0x41u8, n));
    if n < 128 {
        b.push(n as u8);
    } else if n < 0x7F00 {
        b.push(((n >> 8) as u8) + 128);
        b.push((n & 255) as u8);
    } else {
        let m = n - 0x7F00;
        b.push(255);
        b.push((m & 255) as u8);
        b.push((m >> 8) as u8);
    }
    b.push(0b01_01_01_00);
    b.extend_from_slice(&[1, 0, 52]);
    for _ in 0..n {
        b.extend_from_slice(&0xFFFFu16.to_le_bytes());
    }
    b.push(0x01);
    assert!(b.len() < 131_072, "block of {} bytes", b.len());
    b
}

/// Honest content size, one compressed block of sequences that expand far
/// past it.
fn sequence_bomb(n: usize, fcs: u32) -> Vec<u8> {
    let block = sequence_block(n);
    let mut blocks = block_header(true, 2, block.len()).to_vec();
    blocks.extend_from_slice(&block);
    frame_of(0x00, fcs, &blocks)
}

/// Honest content size and window descriptor `window`, followed by `count`
/// 128 KiB run-length blocks: a decoder that trusts the window keeps
/// decoding until the window is full.
fn extra_block_bomb(window: u8, fcs: u32, count: usize) -> Vec<u8> {
    let mut blocks = Vec::new();
    for index in 0..count {
        blocks.extend_from_slice(&block_header(index + 1 == count, 1, 128 << 10));
        blocks.push(7);
    }
    frame_of(window, fcs, &blocks)
}

/// Decodes `frame` into `fcs` bytes under a 64 MiB context and returns the
/// outcome with the process-wide peak and what the call reserved.
fn decode_measured(frame: &[u8], fcs: usize) -> (Result<(), Error>, usize, usize) {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let mut out = vec![0u8; fcs];
    let (result, peak) = measured(|| zstd_codec::decompress(&ctx, frame, &mut out));
    assert_eq!(ctx.memory_used(), 0, "a reservation outlived the call");
    (result, peak, ctx.memory_peak())
}

#[test]
fn the_probe_sees_the_decoder_thread_and_the_column_workers() {
    let _gate = gate();
    // A valid frame of four 128 KiB blocks: the ring buffer lives on the
    // codec's own thread, which the per-thread probe of `parquet_errors.rs`
    // cannot count.
    let content = 4 * (128usize << 10);
    let frame = extra_block_bomb(0x58, content as u32, 4);
    let (result, peak, charged) = decode_measured(&frame, content);
    result.unwrap();
    eprintln!(
        "valid 4-block frame: peak {} KB, charged {} KB",
        peak >> 10,
        charged >> 10
    );
    assert!(
        peak >= content,
        "the probe saw only {peak} bytes of a {content}-byte decode"
    );
    assert!(peak <= charged + SLACK);
    // Four workers decoding the columns of a real file, each on its own thread.
    let bytes =
        fs::read(parquet_support::fixtures().join("hamiltonian/pyarrow_zstd.parquet")).unwrap();
    let (result, peak, reserved) = read_measured(&bytes, 4);
    result.unwrap();
    eprintln!(
        "four-worker read: peak {} KB, reserved {} KB",
        peak >> 10,
        reserved >> 10
    );
    assert!(peak > 0 && peak <= reserved + SLACK);
}

#[test]
fn a_block_of_sequences_cannot_expand_past_the_charge() {
    let _gate = gate();
    let mut failures = Vec::new();
    for n in [1usize, 16, 100, 1000, 20_000, 40_000] {
        let frame = sequence_bomb(n, 16);
        let (result, peak, charged) = decode_measured(&frame, 16);
        eprintln!(
            "sequence bomb n={n}: {} B frame, would expand to {} MB, peak {} KB, charged {} KB, {:?}",
            frame.len(),
            (n * 131_075) >> 20,
            peak >> 10,
            charged >> 10,
            result.as_ref().map_err(|e| e.to_string())
        );
        if !matches!(result, Err(Error::Invalid(_))) {
            failures.push(format!("n={n}: {result:?}"));
        }
        if peak > charged + SLACK {
            failures.push(format!("n={n}: peak {peak} B against {charged} B charged"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn blocks_behind_the_declared_content_cannot_fill_the_window() {
    let _gate = gate();
    let mut failures = Vec::new();
    // Window descriptors for 128 KiB x 2^k style windows up to the cap.
    for (window, blocks) in [
        (0x58u8, 64usize),
        (0x68, 128),
        (0x78, 256),
        (0x78, 1024),
        (0x88, 1024),
    ] {
        let frame = extra_block_bomb(window, 16, blocks);
        let (result, peak, charged) = decode_measured(&frame, 16);
        eprintln!(
            "extra blocks window {window:#04x} x {blocks}: {} B frame, peak {} KB, charged {} KB, {:?}",
            frame.len(),
            peak >> 10,
            charged >> 10,
            result.as_ref().map_err(|e| e.to_string())
        );
        if !matches!(result, Err(Error::Invalid(_))) {
            failures.push(format!("window {window:#04x} x {blocks}: {result:?}"));
        }
        if peak > charged + SLACK {
            failures.push(format!(
                "window {window:#04x} x {blocks}: peak {peak} B against {charged} B charged"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

// ---------------------------------------------------------------------------
// A page that claims far more than its body holds, read through a file.
// ---------------------------------------------------------------------------

/// A file with one REQUIRED column and one data page.
fn one_page_file(kind: i32, codec: i32, rows: i32, claimed: usize, body: &[u8]) -> Vec<u8> {
    let mut file = MAGIC.to_vec();
    let header = PageHeader {
        page_type: page_type::DATA_PAGE,
        uncompressed_page_size: claimed as i32,
        compressed_page_size: body.len() as i32,
        crc: None,
        data_page: Some(DataPageHeader {
            num_values: rows,
            encoding: encoding::PLAIN,
            definition_level_encoding: encoding::RLE,
            repetition_level_encoding: encoding::RLE,
        }),
        dictionary_page: None,
        data_page_v2: None,
    };
    let start = file.len() as i64;
    file.extend_from_slice(&header.to_bytes().unwrap());
    file.extend_from_slice(body);
    let size = file.len() as i64 - start;
    let leaf = SchemaElement::leaf("c", kind, kind == physical::BYTE_ARRAY);
    let meta = FileMetaData {
        version: 1,
        schema: vec![SchemaElement::root(1).unwrap(), leaf],
        num_rows: i64::from(rows),
        row_groups: vec![RowGroup {
            columns: vec![ColumnChunk {
                file_path: None,
                file_offset: file.len() as i64,
                meta_data: ColumnMetaData {
                    physical_type: kind,
                    encodings: vec![encoding::PLAIN, encoding::RLE],
                    path_in_schema: vec!["c".to_owned()],
                    codec,
                    num_values: i64::from(rows),
                    total_uncompressed_size: size,
                    total_compressed_size: size,
                    key_value_metadata: vec![],
                    data_page_offset: start,
                    index_page_offset: None,
                    dictionary_page_offset: None,
                },
            }],
            total_byte_size: 0,
            num_rows: i64::from(rows),
        }],
        key_value_metadata: vec![],
        created_by: Some("test".into()),
    };
    let footer = meta.to_bytes().unwrap();
    file.extend_from_slice(&footer);
    file.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    file.extend_from_slice(MAGIC);
    file
}

/// Reads `bytes` under a 500 MB context and returns the outcome, the
/// process-wide peak and what the read reserved.
fn read_measured(bytes: &[u8], threads: usize) -> (Result<(), Error>, usize, usize) {
    let dir = Directory::new();
    let path = dir.file("t.parquet");
    fs::write(&path, bytes).unwrap();
    let ctx = Context::new(Some(threads), 500 << 20).unwrap();
    let (result, peak) = measured(|| read_table(&ctx, &path).map(|_| ()));
    assert_eq!(ctx.memory_used(), 0, "a reservation outlived the call");
    (result, peak, ctx.memory_peak())
}

#[test]
fn a_tiny_compressed_text_page_cannot_make_the_reader_allocate_what_it_claims() {
    let _gate = gate();
    const CLAIMED: usize = 100_000_000;
    // One text value: the zstd frame says CLAIMED content bytes and holds a
    // few; the snappy block says the same in its preamble.
    let mut zstd = ZSTD_MAGIC.to_vec();
    zstd.push(0x80);
    zstd.push(0x00);
    zstd.extend_from_slice(&(CLAIMED as u32).to_le_bytes());
    zstd.extend_from_slice(&block_header(true, 0, 4));
    zstd.extend_from_slice(&[1, 2, 3, 4]);
    let mut snappy = Vec::new();
    let mut length = CLAIMED;
    while length >= 0x80 {
        snappy.push((length & 0x7F) as u8 | 0x80);
        length >>= 7;
    }
    snappy.push(length as u8);
    snappy.extend_from_slice(&[0x0C, 1, 2, 3, 4]);
    for (name, codec, body) in [
        ("zstd", codec_id::ZSTD, zstd),
        ("snappy", codec_id::SNAPPY, snappy),
    ] {
        let bytes = one_page_file(physical::BYTE_ARRAY, codec, 1, CLAIMED, &body);
        let (result, peak, reserved) = read_measured(&bytes, 1);
        eprintln!(
            "{name} text page claiming {CLAIMED} B in a {} B file: peak {} KB, reserved {} KB, {:?}",
            bytes.len(),
            peak >> 10,
            reserved >> 10,
            result.as_ref().map_err(|e| e.to_string())
        );
        assert!(
            matches!(result, Err(Error::Invalid(_))),
            "{name}: {result:?}"
        );
        assert!(
            reserved < (8 << 20),
            "{name}: the read reserved {reserved} B for a {} B file",
            bytes.len()
        );
        assert!(
            peak < (8 << 20),
            "{name}: the read allocated {peak} B for a {} B file",
            bytes.len()
        );
    }
}

// ---------------------------------------------------------------------------
// The mutation sweep, with every thread counted and four column workers.
// ---------------------------------------------------------------------------

/// Multi-column files whose columns carry zstd pages.
const SWEEP_TARGETS: [&str; 4] = [
    "hamiltonian/pyarrow_zstd.parquet",
    "hamiltonian/fastparquet_zstd.parquet",
    "pseudopotential/fastparquet_zstd.parquet",
    "generic/pyarrow_special_values_zstd.parquet",
];

const SWEEP_CASES: usize = 6_000;

#[test]
fn a_mutation_sweep_with_four_column_workers_stays_inside_the_budget() {
    let _gate = gate();
    let targets: Vec<_> = SWEEP_TARGETS.iter().map(|n| target(n)).collect();
    let dir = Directory::new();
    let path = dir.file("m.parquet");
    let limits = [1usize << 22, 1 << 25, 1 << 27];
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let (mut read_ok, mut refused, mut worst) = (0usize, 0usize, 0usize);
    for case in 0..SWEEP_CASES {
        let target = &targets[case % targets.len()];
        // Aim at the compressed bodies and page headers: that is where a
        // worker thread allocates.
        let regions: Vec<_> = target
            .parts
            .iter()
            .filter(|(part, _)| matches!(part, Part::ZstdBody | Part::PageHeader | Part::Footer))
            .collect();
        let (part, range) = regions[rng.below(regions.len())];
        let mut bytes = target.bytes.clone();
        let how = mutate(&mut rng, &mut bytes, range);
        fs::write(&path, &bytes).unwrap();
        let limit = limits[rng.below(limits.len())];
        let ctx = Context::new(Some(4), limit).unwrap();
        let (outcome, peak) = measured(|| {
            catch_unwind(AssertUnwindSafe(|| {
                read_table(&ctx, &path).map(|table| table.num_rows())
            }))
        });
        let outcome = outcome.unwrap_or_else(|_| {
            panic!(
                "case {case}: {how} in {part:?} of {} (limit {limit}) panicked",
                target.name
            )
        });
        assert_eq!(ctx.memory_used(), 0, "case {case}: a reservation leaked");
        worst = worst.max(peak.saturating_sub(ctx.memory_peak()));
        assert!(
            peak <= ctx.memory_peak() + SLACK,
            "case {case}: {how} in {part:?} of {} (limit {limit}) allocated {peak} B live but reserved {}",
            target.name,
            ctx.memory_peak()
        );
        match outcome {
            Ok(_) => read_ok += 1,
            Err(_) => refused += 1,
        }
    }
    eprintln!(
        "four-worker sweep: {SWEEP_CASES} cases, {read_ok} still read, {refused} refused, largest unreserved allocation {worst} bytes"
    );
    assert!(refused > SWEEP_CASES / 4, "{refused} refused");
    assert!(read_ok > 0, "no mutated file still read");
}
