//! Row-parallel state-vector actions on the calculation's worker pool.
//!
//! A gathering kernel writes each output row from inputs it only reads, so a row block is
//! independent of every other: the rows split into contiguous blocks, one per worker, with no
//! locks, and each row's arithmetic does not depend on the split. The blocks run on the
//! context's persistent pool ([`Context::pool`]): a dispatch wakes parked workers instead of
//! creating threads, which an evaluation does thousands of times.
use crate::Context;
use rayon::prelude::*;

/// The work (row-term products) below which an action stays on the calling thread.
pub(crate) const PARALLEL_WORK: usize = 1 << 18;
/// The fewest rows a block holds.
const MIN_BLOCK: usize = 256;

thread_local! {
    /// Set while this thread runs a block of a parallel region: an action called from inside one
    /// (a pool generator inside the pool screening) stays on its thread, so regions never nest.
    static INSIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn inside<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            INSIDE.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(INSIDE.with(|flag| flag.replace(true)));
    f()
}

/// The blocks a region of `work` over `units` independent pieces splits into: one below
/// [`PARALLEL_WORK`], inside another region, or without a pool; otherwise up to the context's
/// width (`Context::width`, whatever the environment says).
pub(crate) fn threads_for(ctx: &Context, work: usize, units: usize) -> usize {
    if work < PARALLEL_WORK || INSIDE.with(|flag| flag.get()) || ctx.pool().is_none() {
        1
    } else {
        ctx.width().min(units).max(1)
    }
}

/// Runs `block(first_row, rows)` over `out` in contiguous blocks on the context's pool when
/// `work` reaches [`PARALLEL_WORK`], else on the calling thread.
pub(crate) fn parallel_rows<T: Send>(
    ctx: &Context,
    rows: usize,
    work: usize,
    out: &mut [T],
    block: impl Fn(usize, &mut [T]) + Sync,
) {
    debug_assert_eq!(rows, out.len());
    let threads = threads_for(ctx, work, rows.div_ceil(MIN_BLOCK));
    let Some(pool) = ctx.pool().filter(|_| threads > 1) else {
        block(0, out);
        return;
    };
    let size = rows.div_ceil(threads);
    pool.install(|| {
        out.par_chunks_mut(size)
            .enumerate()
            .for_each(|(k, chunk)| inside(|| block(k * size, chunk)));
    });
}

/// Runs `task(first, items, scratch)` over `out` in `scratch.len()` contiguous blocks, one per
/// scratch entry, on the context's pool (`scratch.len()` comes from [`threads_for`]); each block
/// owns its scratch, so a task needs no lock.
pub(crate) fn parallel_tasks<T: Send, S: Send>(
    ctx: &Context,
    out: &mut [T],
    scratch: &mut [S],
    task: impl Fn(usize, &mut [T], &mut S) + Sync,
) {
    let threads = scratch.len().max(1);
    let size = out.len().div_ceil(threads).max(1);
    let pool = ctx.pool().filter(|_| threads > 1);
    let (Some(pool), false) = (pool, out.is_empty()) else {
        if let Some(first) = scratch.first_mut() {
            task(0, out, first);
        }
        return;
    };
    pool.install(|| {
        out.par_chunks_mut(size)
            .zip(scratch.par_iter_mut())
            .enumerate()
            .for_each(|(k, (items, own))| inside(|| task(k * size, items, own)));
    });
}

/// Runs `task(first_row, a_block, b_block, sum)` over `a` and `b` in fixed blocks of `block` rows
/// with one entry of `sums` per block, on the context's pool (on the calling thread without one,
/// or inside another region). The blocks are the same whatever the thread count, so a block's sum
/// is too; `sums.len()` is `a.len().div_ceil(block)`.
pub(crate) fn parallel_row_pairs<T: Send, S: Send>(
    ctx: &Context,
    block: usize,
    a: &mut [T],
    b: &mut [T],
    sums: &mut [S],
    task: impl Fn(usize, &mut [T], &mut [T], &mut S) + Sync,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(sums.len(), a.len().div_ceil(block));
    let pool = ctx.pool().filter(|_| !INSIDE.with(|flag| flag.get()) && sums.len() > 1);
    let Some(pool) = pool else {
        for (k, ((a, b), sum)) in a.chunks_mut(block).zip(b.chunks_mut(block)).zip(sums.iter_mut()).enumerate() {
            task(k * block, a, b, sum);
        }
        return;
    };
    pool.install(|| {
        a.par_chunks_mut(block)
            .zip(b.par_chunks_mut(block))
            .zip(sums.par_iter_mut())
            .enumerate()
            .for_each(|(k, ((a, b), sum))| inside(|| task(k * block, a, b, sum)));
    });
}

/// Runs `task(k, row)` for every `k` in `0..count` and collects the results in order, the
/// applications spread over the context's pool in contiguous shares when `work` (the cost of all
/// of them, in row-term products) reaches [`PARALLEL_WORK`]. Each worker owns one scratch row of
/// `rows` entries, reused from one task to the next: a task that needs a clean row clears it. A task is one serial computation (every region it opens stays on its
/// worker), so a result does not depend on the thread count or on which worker computed it.
pub(crate) fn parallel_each<T: Send>(
    ctx: &Context,
    count: usize,
    rows: usize,
    work: usize,
    task: impl Fn(usize, &mut [crate::quantum::C64]) -> T + Sync,
) -> crate::Result<Vec<T>> {
    let lanes = threads_for(ctx, work, count);
    let mut scratch = (0..lanes)
        .map(|_| ctx.allocate::<crate::quantum::C64>(rows))
        .collect::<crate::Result<Vec<_>>>()?;
    let mut out: Vec<Option<T>> = (0..count).map(|_| None).collect();
    parallel_tasks(ctx, &mut out, &mut scratch, |first, slots, row| {
        for (offset, slot) in slots.iter_mut().enumerate() {
            *slot = Some(task(first + offset, row.as_mut_slice()));
        }
    });
    Ok(out.into_iter().map(|slot| slot.expect("every application ran")).collect())
}
