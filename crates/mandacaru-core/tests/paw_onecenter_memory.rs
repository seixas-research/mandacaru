//! The memory reserved by `exact_atom` covers what it allocates: a counting allocator
//! measures the heap the call really takes and compares it with the `Context` peak.
//! One test only: the counters are per thread, but a second test would make the
//! baseline depend on the harness.
use mandacaru_core::paw::onecenter_hybrid::{exact_atom, exact_atom_peak_bytes};
use mandacaru_core::Context;
use nalgebra::DMatrix;
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

fn tensor(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    let m = n * n;
    let mut out = vec![0.; m * m];
    for ab in 0..m {
        for cd in 0..=ab {
            let value = next();
            out[ab * m + cd] = value;
            out[cd * m + ab] = value;
        }
    }
    out
}

#[test]
fn the_reservation_of_the_exact_atom_covers_the_heap_it_takes() {
    for n in [4usize, 6] {
        let m = n * n;
        let tensor = tensor(n, 11);
        let (v, k, d0) = (
            DMatrix::zeros(n, n),
            DMatrix::zeros(n, n),
            DMatrix::from_fn(n, n, |i, j| 1. / (1. + (i + j) as f64)),
        );
        let ctx = Context::new(Some(1), 64 << 20).unwrap();
        let before = LIVE.with(Cell::get);
        PEAK.with(|p| p.set(before));
        let atom = exact_atom(&ctx, n, &tensor, v, k, d0).unwrap();
        let measured = PEAK.with(Cell::get) - before;
        let reserved = ctx.memory_peak();
        assert_eq!(reserved, exact_atom_peak_bytes(n).unwrap());
        // Allowed: the reservation is the bound, not a tight fit. It must hold the real
        // heap, and the real heap must be a sizeable part of it (the bound is not vacuous).
        assert!(
            measured <= reserved,
            "n = {n}: the call took {measured} bytes, {reserved} were reserved"
        );
        assert!(
            2 * measured >= reserved,
            "n = {n}: {measured} of {reserved} reserved bytes were used (m = {m})"
        );
        drop(atom);
    }
}
