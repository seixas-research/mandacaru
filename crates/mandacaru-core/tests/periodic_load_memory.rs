//! Loading a saved crystal reserves what it allocates: a counting allocator measures the heap
//! `Crystal::load` really takes and compares it with the `Context` peak, and the 512 MiB budget
//! of the guides is enough for a file that a fixed multiple of the text length once refused.
//! One test only: the counters are per thread, and a second test would make the baseline depend
//! on the harness.
mod hubbard_support;

use hubbard_support::load;
use mandacaru_core::{periodic::Crystal, Context};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    static LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
}

// Safety contract: `mandacaru-kernels/SAFETY.md`, "Test-only counting allocators".
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        let _ = LIVE.try_with(|c| c.set(c.get().saturating_sub(layout.size())));
        System.dealloc(pointer, layout)
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new: usize) -> *mut u8 {
        let _ = LIVE.try_with(|c| c.set(c.get().saturating_sub(layout.size())));
        grow(new);
        System.realloc(pointer, layout, new)
    }
}

/// Add `bytes` to this thread's live count and raise its peak. `try_with` ignores an
/// allocation made while the thread-locals are being torn down.
fn grow(bytes: usize) {
    let live = LIVE
        .try_with(|c| {
            c.set(c.get() + bytes);
            c.get()
        })
        .unwrap_or(0);
    let _ = PEAK.try_with(|p| p.set(p.get().max(live)));
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn the_reservation_of_a_crystal_load_covers_the_heap_and_fits_the_default_budget() {
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("load_memory_cuh.json");
    {
        let ctx = Context::new(Some(1), 1 << 30).unwrap();
        load(&ctx, "cuh_7").save(path.to_str().unwrap()).unwrap();
    }
    let bytes = std::fs::metadata(&path).unwrap().len() as usize;
    // The budget of the guides.
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let before = LIVE.with(Cell::get);
    PEAK.with(|p| p.set(before));
    let crystal = Crystal::load(&ctx, path.to_str().unwrap()).unwrap();
    let measured = PEAK.with(Cell::get) - before;
    let reserved = ctx.memory_peak();
    println!(
        "file {bytes} bytes: the load took {measured}, {reserved} were reserved (peak {:.2} of the file)",
        reserved as f64 / bytes as f64
    );
    // The reservation is a bound, not a tight fit: it must hold the real heap, and the real heap
    // must be a sizeable part of it (the bound is not vacuous).
    assert!(
        measured <= reserved,
        "the load took {measured} bytes, {reserved} were reserved"
    );
    assert!(
        3 * measured >= reserved,
        "{measured} of {reserved} reserved bytes were used"
    );
    assert!(reserved < 512 << 20);
    drop(crystal);
    std::fs::remove_file(&path).unwrap();
}
