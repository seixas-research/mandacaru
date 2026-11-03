//! The `setup` block of a crystal snapshot is reserved from the `Context` before it is allocated:
//! a counting allocator measures the heap `Description::parse` and `PointGroup::parse` really take
//! on a reduced-mesh snapshot that carries a block and compares it with the reservation, and a
//! `Context` too small for the block refuses it having allocated next to nothing (the control: a
//! parse that allocated first and reserved after would have taken the tables before it failed).
//! One test only: the counters are per thread, and a second test would make the baseline depend
//! on the harness. Needs `MANDACARU_PAW_PATH` for the snapshot's dataset paths only through the
//! support module.
mod periodic_gradient_crystals_support;
mod periodic_response_support;

use mandacaru_core::{
    periodic::{description::Description, point_group::PointGroup, Crystal},
    Context, Error,
};
use nalgebra::Matrix3;
use periodic_response_support::Pair;
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

/// Runs `work`, returning its value, the peak heap it added and the heap it left alive.
fn measured<T>(work: impl FnOnce() -> T) -> (T, usize, usize) {
    let before = LIVE.with(Cell::get);
    PEAK.with(|p| p.set(before));
    let out = work();
    let peak = PEAK.with(Cell::get) - before;
    let left = LIVE.with(Cell::get).saturating_sub(before);
    (out, peak, left)
}

#[test]
fn the_setup_block_is_reserved_before_it_is_allocated_and_the_reservation_covers_it() {
    let pair = Pair::load("si110");
    let root = pair.reduced;
    let loader = Context::new(Some(1), 1 << 30).unwrap();
    let crystal = Crystal::from_json(&loader, &root.to_string()).unwrap();
    assert!(
        crystal.description.is_some(),
        "the snapshot carries a block"
    );
    let shape = crystal.grid.index.shape();
    let cell = Matrix3::from_fn(|i, j| shape[j] as f64 * crystal.grid.step[(i, j)]);
    let parse = |ctx: &Context| {
        Description::parse(
            ctx,
            &root,
            crystal.m,
            crystal.p,
            &cell,
            shape,
            &crystal.permutations,
        )
    };

    // The description of a block with its operations, against its reservation.
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let (description, peak, left) = measured(|| parse(&ctx).unwrap());
    let (reserved, held) = (ctx.memory_peak(), ctx.memory_used());
    println!(
        "setup description: {peak} bytes at the peak, {left} left; reserved {reserved} at the peak, {held} held"
    );
    assert!(
        description.point_group.is_some() && description.point_group.as_ref().unwrap().len() >= 2
    );
    assert!(
        peak <= reserved,
        "the parse took {peak}, {reserved} were reserved"
    );
    assert!(
        left <= held,
        "the description keeps {left}, {held} are held"
    );
    // The reservation is a bound, not a tight fit, and not vacuous either.
    assert!(
        3 * peak >= reserved,
        "{peak} of {reserved} reserved bytes were used"
    );
    assert!(3 * left >= held, "{left} of {held} held bytes are alive");
    drop(description);
    assert_eq!(ctx.memory_used(), 0);

    // The control: a Context that cannot hold the block refuses it before allocating any of it.
    let small = Context::new(Some(1), held / 2).unwrap();
    let (refused, early, _) = measured(|| parse(&small));
    assert!(
        matches!(refused, Err(Error::Budget { .. })),
        "a too small budget must be a budget error, not {:?}",
        refused.as_ref().err()
    );
    println!(
        "refused by a budget of {} bytes after {early} bytes of heap",
        held / 2
    );
    assert_eq!(
        early, 0,
        "the refused parse allocated {early} bytes, the description is {left}"
    );

    // The same for the operations alone.
    let operations = &root["setup"]["operations"];
    let centers = &description_centers(&root);
    let group = |ctx: &Context| {
        PointGroup::parse(
            ctx,
            operations,
            &cell,
            centers,
            shape,
            &crystal.permutations,
        )
    };
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let (g, peak, _) = measured(|| group(&ctx).unwrap());
    let reserved = ctx.memory_peak();
    println!("point group: {peak} bytes at the peak, {reserved} reserved");
    assert!(
        peak <= reserved,
        "the group took {peak}, {reserved} were reserved"
    );
    assert!(3 * peak >= reserved);
    let small = Context::new(Some(1), reserved / 2).unwrap();
    let (refused, early, _) = measured(|| group(&small));
    assert!(
        matches!(refused, Err(Error::Budget { .. })),
        "{:?}",
        refused.err()
    );
    assert_eq!(
        early, 0,
        "{early} bytes allocated before the refusal (the group is {peak})"
    );
    drop(g);
}

fn description_centers(root: &serde_json::Value) -> Vec<[f64; 3]> {
    root["sites"]["centers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let c = c.as_array().unwrap();
            [
                c[0].as_f64().unwrap(),
                c[1].as_f64().unwrap(),
                c[2].as_f64().unwrap(),
            ]
        })
        .collect()
}
