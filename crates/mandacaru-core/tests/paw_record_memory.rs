//! What reading a PAW dataset allocates, measured over the whole process.
//!
//! This binary installs a counting global allocator (process-wide atomics, so
//! the zstd decoder thread and the column workers are seen) with an emulated
//! hard limit, and every test takes `GATE` first so exactly one measurement
//! runs at a time. The invariant is the reader rule of CLAUDE.md: the live
//! bytes of a read never exceed what the read reserved from its `Context`
//! (`ctx.memory_peak()`) plus a small constant, for every input, and a refusal
//! returns the budget to zero.
//!
//! One file at a time: `pbe-fr/Pb.parquet` (15.5 MB, 61,500 rows, a 4.48 MB
//! payload) is the large case; it also yields the measured cost of the JSON
//! tree per payload byte that `TREE_BYTES_PER_VALUE` is set from.

mod parquet_support;
mod paw_record_support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use mandacaru_core::parquet::format::FileMetaData;
use mandacaru_core::parquet::{self, Codec, WriteOptions};
use mandacaru_core::paw::record::{
    tree_cost, PawDataset, METADATA_KEY, TREE_BYTES_PER_MEMBER, TREE_BYTES_PER_VALUE,
};
use mandacaru_core::{Context, Error};
use paw_record_support::{dataset_path, TempDir};
use serde_json::Value;

// ---------------------------------------------------------------------------
// The process-wide counting allocator.
// ---------------------------------------------------------------------------

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// Live bytes past which an allocation fails: the test process stops, the
/// machine does not swap (the Mac has 16 GB and no swap budget).
const HARD_LIMIT: usize = 700 << 20;

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

static GATE: Mutex<()> = Mutex::new(());

fn gate() -> MutexGuard<'static, ()> {
    GATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Peak live bytes above the level at entry, on every thread, while `f` runs.
fn measured<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let value = f();
    (value, PEAK.load(Ordering::Relaxed).saturating_sub(base))
}

fn live() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// What a call may allocate without reserving it: error text, path strings,
/// `Vec` headers, thread bookkeeping.
const SLACK: usize = 1 << 20;

// ---------------------------------------------------------------------------
// The large case and the measured factor.
// ---------------------------------------------------------------------------

#[test]
fn pb_pbe_fr_stays_inside_its_reservation_and_under_a_gigabyte() {
    let _gate = gate();
    let path = dataset_path("pbe-fr", "Pb");
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let footer = parquet::read_metadata(&ctx, &path).unwrap();
    let text = footer.metadata[METADATA_KEY].clone();
    let (rows, columns) = (footer.num_rows as usize, footer.columns.len());
    drop(footer);
    let tables = rows * columns * 8;
    let before = live();
    let (record, peak) = measured(|| PawDataset::read(&ctx, &path).unwrap());
    let retained = live() - before;
    println!(
        "pbe-fr/Pb: {} B file, {rows} rows x {columns} columns = {tables} B of tables, payload {} B",
        std::fs::metadata(&path).unwrap().len(),
        text.len()
    );
    println!(
        "read: context peak {} B, real peak {peak} B (above the tables {} B = {:.2} per payload byte), retained {retained} B against {} B reserved",
        ctx.memory_peak(),
        peak.saturating_sub(tables),
        peak.saturating_sub(tables) as f64 / text.len() as f64,
        ctx.memory_used()
    );
    assert!(
        peak <= ctx.memory_peak() + SLACK,
        "live {peak} B exceeded the reservation {} B",
        ctx.memory_peak()
    );
    assert!(
        retained <= ctx.memory_used() + SLACK,
        "retained {retained} B against {} B reserved",
        ctx.memory_used()
    );
    assert!(peak < 1 << 30, "peak {peak} B must stay under 1 GB");
    assert_eq!(record.points(), rows);
    drop(record);
    assert_eq!(ctx.memory_used(), 0);

    // The JSON tree, measured on its own: this is the factor the reserve is
    // set from, and the reserve must bound it.
    let (tree, tree_peak) = measured(|| serde_json::from_str::<Value>(&text).unwrap());
    let cost = tree_cost(&text).unwrap();
    println!(
        "payload tree: measured peak {tree_peak} B = {:.2} per payload byte; reserved {cost} B = {:.2} per payload byte ({TREE_BYTES_PER_VALUE} B per value, {TREE_BYTES_PER_MEMBER} B per member, plus the text)",
        tree_peak as f64 / text.len() as f64,
        cost as f64 / text.len() as f64
    );
    assert!(
        tree_peak <= cost,
        "the tree took {tree_peak} B, the reserve is {cost} B"
    );
    drop(tree);
}

#[test]
fn a_small_budget_raises_budget_and_everything_is_released() {
    let _gate = gate();
    let path = dataset_path("pbe-fr", "Pb");
    for limit in [64usize << 10, 1 << 20, 16 << 20, 19 << 20] {
        let ctx = Context::new(Some(2), limit).unwrap();
        let (result, peak) = measured(|| PawDataset::read(&ctx, &path));
        let error = result.expect_err("Pb cannot fit this budget");
        assert!(
            matches!(error, Error::Budget { .. }),
            "limit {limit}: {error:?}"
        );
        assert_eq!(
            ctx.memory_used(),
            0,
            "limit {limit}: the budget returns to zero"
        );
        assert!(
            peak <= ctx.memory_peak() + SLACK,
            "limit {limit}: live {peak} B, reserved {}",
            ctx.memory_peak()
        );
        println!(
            "limit {limit} B: Budget, real peak {peak} B, context peak {} B",
            ctx.memory_peak()
        );
    }
    // A budget for the tables and the text but not for the tree: the refusal
    // comes at the tree reservation, before the parse allocates it. The limit
    // is what reading the table alone needs at its peak (scratch included),
    // plus the payload text.
    let probe = Context::new(Some(2), 512 << 20).unwrap();
    let footer = parquet::read_metadata(&probe, &path).unwrap();
    let text_len = footer.metadata[METADATA_KEY].len();
    let cost = tree_cost(&footer.metadata[METADATA_KEY]).unwrap();
    drop(footer);
    let table = parquet::read_table(&probe, &path).unwrap();
    let tables = table.num_rows() * table.columns.len() * 8;
    let table_peak = probe.memory_peak();
    drop(table);
    let limit = table_peak + text_len + (1 << 20);
    assert!(
        limit < tables + text_len + cost,
        "the case must starve only the tree"
    );
    let ctx = Context::new(Some(2), limit).unwrap();
    let (result, peak) = measured(|| PawDataset::read(&ctx, &path));
    let Err(Error::Budget {
        requested,
        available,
    }) = result
    else {
        panic!("expected Budget at the tree, got {result:?}")
    };
    assert_eq!(requested, cost, "the refusal is the tree reservation");
    assert_eq!(ctx.memory_used(), 0);
    println!(
        "tree starved: Budget asking {requested} B with {available} B available; real peak {peak} B for {tables} B of tables"
    );
    assert!(
        peak < table_peak + 3 * text_len + 2 * SLACK,
        "no tree was allocated: peak {peak} B"
    );
}

// ---------------------------------------------------------------------------
// Hostile declared sizes.
// ---------------------------------------------------------------------------

/// A copy of a dataset file with the payload replaced by `text`.
fn with_payload(dir: &TempDir, text: String) -> std::path::PathBuf {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let mut table = parquet::read_table(&ctx, &dataset_path("lda-sr", "H")).unwrap();
    table.metadata.insert(METADATA_KEY.into(), text);
    let path = dir.file("hostile.parquet");
    let options = WriteOptions {
        codec: Codec::Uncompressed,
        ..WriteOptions::default()
    };
    parquet::write_table(&ctx, &path, &table, &options).unwrap();
    path
}

#[test]
fn a_payload_that_declares_millions_of_values_is_refused_before_it_is_parsed() {
    let _gate = gate();
    let dir = TempDir::new("bomb");
    let text = format!("[{}1]", "1,".repeat(3_000_000));
    let path = with_payload(&dir, text.clone());
    let cost = tree_cost(&text).unwrap();
    let ctx = Context::new(Some(2), 100 << 20).unwrap();
    assert!(cost > ctx.memory_limit(), "{cost} B must exceed the budget");
    let (result, peak) = measured(|| PawDataset::read(&ctx, &path));
    assert!(matches!(result, Err(Error::Budget { .. })), "{result:?}");
    assert_eq!(ctx.memory_used(), 0);
    println!(
        "{} B payload of 3,000,001 values: Budget (tree {cost} B), real peak {peak} B",
        text.len()
    );
    assert!(peak <= ctx.memory_peak() + SLACK);
    assert!(
        peak < cost / 8,
        "nothing near the tree was allocated: {peak} B"
    );
    // Deep nesting is refused by the parser, still inside the reservation.
    let deep = format!("{}{}", "[".repeat(5_000), "]".repeat(5_000));
    let path = with_payload(&dir, deep);
    let ctx = Context::new(Some(2), 100 << 20).unwrap();
    let (result, peak) = measured(|| PawDataset::read(&ctx, &path));
    assert!(matches!(result, Err(Error::Invalid(_))), "{result:?}");
    assert!(peak <= ctx.memory_peak() + SLACK);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_footer_that_declares_two_to_the_forty_rows_raises_through_the_record_loader() {
    let _gate = gate();
    let dir = TempDir::new("rows");
    let bytes = std::fs::read(dataset_path("lda-sr", "H")).unwrap();
    let n = bytes.len();
    let footer_len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    let start = n - 8 - footer_len;
    let mut meta = FileMetaData::from_bytes(&bytes[start..n - 8]).unwrap();
    meta.num_rows = 1 << 40;
    for group in &mut meta.row_groups {
        group.num_rows = 1 << 40;
        for chunk in &mut group.columns {
            chunk.meta_data.num_values = 1 << 40;
        }
    }
    let footer = meta.to_bytes().unwrap();
    let mut patched = bytes[..start].to_vec();
    patched.extend_from_slice(&footer);
    patched.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    patched.extend_from_slice(b"PAR1");
    let path = dir.file("rows.parquet");
    std::fs::write(&path, &patched).unwrap();
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let (result, peak) = measured(|| PawDataset::read(&ctx, &path));
    let error = result.expect_err("2^40 rows cannot be read");
    println!(
        "2^40 rows declared by a {} B file: {error}; real peak {peak} B",
        patched.len()
    );
    assert!(
        matches!(
            error,
            Error::Budget { .. } | Error::Invalid(_) | Error::Overflow
        ),
        "{error:?}"
    );
    assert!(
        peak <= ctx.memory_peak() + SLACK && peak < 2 * SLACK,
        "peak {peak} B"
    );
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_json_file_is_bounded_by_its_size_before_it_is_read() {
    let _gate = gate();
    let dir = TempDir::new("jsonbound");
    let path = dir.file("big.json");
    std::fs::write(&path, format!("{{\"x\":[{}1]}}", "1,".repeat(2_000_000))).unwrap();
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    let (result, peak) = measured(|| PawDataset::read(&ctx, &path));
    assert!(matches!(result, Err(Error::Budget { .. })), "{result:?}");
    assert_eq!(ctx.memory_used(), 0);
    println!(
        "4 MB JSON file under a 1 MiB budget: Budget before the text was read, real peak {peak} B"
    );
    assert!(peak < SLACK, "the file was not read: {peak} B");
    // Within a larger budget the tree is the next gate: 4 MB of text, 2,000,002 values.
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let (result, peak) = measured(|| PawDataset::read(&ctx, &path));
    assert!(matches!(result, Err(Error::Budget { .. })), "{result:?}");
    assert!(peak <= ctx.memory_peak() + SLACK, "peak {peak} B");
}

// ---------------------------------------------------------------------------
// Mutated files through the record loader.
// ---------------------------------------------------------------------------

#[test]
fn a_byte_mutation_sweep_stays_inside_the_reservation() {
    let _gate = gate();
    let dir = TempDir::new("sweep");
    let path = dir.file("m.parquet");
    let mut rng = parquet_support::Rng(0x2545_f491_4f6c_dd1d);
    let limits = [1usize << 20, 1 << 24, 1 << 27];
    let sources = [dataset_path("lda-sr", "H"), dataset_path("lda-fr", "Na")];
    let (mut read, mut refused, mut worst) = (0usize, 0usize, 0usize);
    for source in &sources {
        let original = std::fs::read(source).unwrap();
        let n = original.len();
        let footer_len = u32::from_le_bytes(original[n - 8..n - 4].try_into().unwrap()) as usize;
        // Aim at the footer (it carries the payload text) three times in four.
        let footer = n - 8 - footer_len..n - 8;
        for case in 0..300 {
            let mut bytes = original.clone();
            let range = if case % 4 == 3 { 0..n } else { footer.clone() };
            let how = parquet_support::mutate(&mut rng, &mut bytes, &range);
            std::fs::write(&path, &bytes).unwrap();
            let limit = limits[rng.below(limits.len())];
            let ctx = Context::new(Some(2), limit).unwrap();
            let (outcome, peak) = measured(|| {
                catch_unwind(AssertUnwindSafe(|| {
                    PawDataset::read(&ctx, &path).map(|d| d.points())
                }))
            });
            let outcome = outcome.unwrap_or_else(|_| panic!("case {case} ({how}) panicked"));
            assert_eq!(
                ctx.memory_used(),
                0,
                "case {case} ({how}): a reservation leaked"
            );
            worst = worst.max(peak.saturating_sub(ctx.memory_peak()));
            assert!(
                peak <= ctx.memory_peak() + SLACK,
                "case {case} ({how}, limit {limit}): live {peak} B against {} B reserved",
                ctx.memory_peak()
            );
            match outcome {
                Ok(_) => read += 1,
                Err(_) => refused += 1,
            }
        }
    }
    println!("mutation sweep: 600 cases, {read} still read, {refused} refused, largest unreserved allocation {worst} B");
    assert!(refused > 100);
}
