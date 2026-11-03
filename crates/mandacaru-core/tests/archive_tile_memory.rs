//! Streaming a tile from a scratch array allocates nothing, and a projection or a density over a
//! stack on disk never takes more heap than the one over the stack in memory: a counting
//! allocator measures the heap against the `Context` reservations. One test only: the counters
//! are per thread, and a second test would make the baseline depend on the harness.
use mandacaru_core::{
    math::{
        projection::{hermitian_density_tiles, ComplexProjector},
        storage::{ComplexSamples, ComplexTiles, Precision, TILE},
    },
    quantum::C64,
    Context,
};
use nalgebra::DMatrix;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    static LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
    static TOTAL: Cell<usize> = const { Cell::new(0) };
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
    let _ = TOTAL.try_with(|t| t.set(t.get() + bytes));
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// `(peak heap growth, bytes allocated)` of `work` on this thread.
fn measure<T>(work: impl FnOnce() -> T) -> (T, usize, usize) {
    let before = LIVE.with(Cell::get);
    PEAK.with(|p| p.set(before));
    let total = TOTAL.with(Cell::get);
    let out = work();
    (
        out,
        PEAK.with(Cell::get) - before,
        TOTAL.with(Cell::get) - total,
    )
}

fn stack(count: usize) -> Vec<C64> {
    (0..count)
        .map(|i| C64::new((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos()))
        .collect()
}

#[test]
fn a_tile_read_from_disk_allocates_nothing_and_the_streamed_products_stay_in_their_reservation() {
    let (m, n) = (8, 3 * TILE + 100);
    let original = stack(m * n);
    let field: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).cos()).collect();
    let d = DMatrix::from_fn(m, m, |i, j| {
        if i == j {
            C64::new(1. + i as f64, 0.)
        } else {
            C64::new(0.1, 0.05 * (i as f64 - j as f64))
        }
    });
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let resident = ComplexSamples::copy(&ctx, &original, Precision::Float64).unwrap();
    let archived = ComplexSamples::archived_copy(&ctx, &original, Precision::Float64).unwrap();
    let narrow = ComplexSamples::archived_copy(&ctx, &original, Precision::Float32).unwrap();

    // The tile read itself: no heap at all, whatever the precision on disk.
    for source in [&archived, &narrow] {
        let mut scratch = vec![C64::default(); source.scratch_len(m, TILE)];
        let (_, growth, total) = measure(|| {
            for start in (0..n).step_by(TILE) {
                let count = TILE.min(n - start);
                let tile = source
                    .tile(m, n, start, count, &mut scratch[..m * count])
                    .unwrap();
                assert_eq!(tile.count(), count);
                std::hint::black_box(tile.row(m - 1)[count - 1]);
            }
        });
        assert_eq!((growth, total), (0, 0), "a tile read allocated");
    }

    // The products: the projector reserves its working set from the Context before it runs;
    // the heap of the projections must fit that reservation, and a stack on disk must cost no
    // more heap than the stack in memory.
    let mut costs = vec![];
    for source in [&resident, &archived] {
        let (projector, _, _) = measure(|| ComplexProjector::streaming(&ctx, source, m, n).unwrap());
        let mut projector = projector;
        let reserved = ctx.memory_used();
        let (_, growth, total) = measure(|| {
            for _ in 0..3 {
                std::hint::black_box(projector.project(&field).unwrap()[0]);
            }
        });
        println!("project: archived={} growth={growth} total={total} reserved={reserved}", source.archived());
        assert!(growth <= reserved, "the projection took {growth} of {reserved} reserved bytes");
        let (density, dgrowth, _) = measure(|| hermitian_density_tiles(&ctx, source, &d, n).unwrap());
        let density_reserved = ctx.memory_peak();
        println!("density: archived={} growth={dgrowth}", source.archived());
        assert!(dgrowth <= density_reserved);
        std::hint::black_box(density.as_slice()[0]);
        costs.push((growth, total, dgrowth));
    }
    assert!(
        costs[1].0 <= costs[0].0 && costs[1].1 <= costs[0].1 && costs[1].2 <= costs[0].2 + m * TILE * 16,
        "the stack on disk took more heap than the stack in memory: {costs:?}"
    );
    // What is held on disk is accounted as the descriptor-sized reservations only, and all of it
    // leaves with the stacks.
    drop((resident, archived, narrow));
    assert_eq!(ctx.memory_used(), 0);
}
