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

pub(crate) fn is_inside() -> bool {
    INSIDE.with(|flag| flag.get())
}

pub(crate) fn inside<T>(f: impl FnOnce() -> T) -> T {
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

/// The products of one round of [`ordered_scatter`], sorted by the output block they add to.
pub(crate) struct Buckets {
    block: usize,
    lists: Vec<Vec<(usize, crate::quantum::C64)>>,
}

impl Buckets {
    /// Queues `value` for `out[row] += value`.
    #[inline]
    pub(crate) fn push(&mut self, row: usize, value: crate::quantum::C64) {
        self.lists[row / self.block].push((row, value));
    }
}

/// The products one round of [`ordered_scatter`] holds at most, over all workers.
const ROUND_PRODUCTS: usize = 1 << 20;

/// A serial scatter `for column in columns { emit(column) }` with `emit` pushing `(row, value)`
/// for `out[row] += value`, run on the context's workers with the same bits: each round of
/// columns is split into contiguous shares, each worker queues its share's products by output
/// block, then each output block replays the shares in column order. Every row therefore adds
/// the same values in the same order as the serial loop, whatever the number of workers.
/// `columns()` lists the `count` columns (ascending) and is called only when the region splits;
/// `products` bounds what one column pushes. Returns `false`, touching nothing, when the region
/// would run on one worker or its queues cannot be reserved: the caller runs the serial loop.
pub(crate) fn ordered_scatter(
    ctx: &Context,
    count: usize,
    columns: impl FnOnce() -> Vec<usize>,
    products: usize,
    out: &mut [crate::quantum::C64],
    emit: impl Fn(usize, &mut Buckets) + Sync,
) -> bool {
    let lanes = threads_for(
        ctx,
        count.saturating_mul(products),
        out.len().div_ceil(MIN_BLOCK),
    );
    let Some(pool) = ctx.pool().filter(|_| lanes > 1) else {
        return false;
    };
    let per_round = (ROUND_PRODUCTS / products.max(1)).max(1);
    let entry = std::mem::size_of::<(usize, crate::quantum::C64)>();
    // A queue grows by doubling, so a round's queues hold up to twice its products.
    let queued = 2 * per_round.saturating_mul(products).saturating_mul(entry);
    let listed = count.saturating_mul(std::mem::size_of::<usize>());
    let Ok(_held) = ctx.reserve(queued.saturating_add(listed)) else {
        return false;
    };
    let columns = columns();
    let block = out.len().div_ceil(lanes);
    let blocks = out.len().div_ceil(block);
    let mut queues: Vec<Buckets> = (0..lanes)
        .map(|_| Buckets {
            block,
            lists: (0..blocks).map(|_| Vec::new()).collect(),
        })
        .collect();
    for round in columns.chunks(per_round) {
        let share = round.len().div_ceil(lanes);
        pool.install(|| {
            queues.par_iter_mut().enumerate().for_each(|(w, queue)| {
                inside(|| {
                    queue.lists.iter_mut().for_each(Vec::clear);
                    let start = (w * share).min(round.len());
                    let end = ((w + 1) * share).min(round.len());
                    for &column in &round[start..end] {
                        emit(column, queue);
                    }
                })
            });
            let queues = &queues;
            out.par_chunks_mut(block).enumerate().for_each(|(b, rows)| {
                inside(|| {
                    let first = b * block;
                    for queue in queues {
                        for &(row, value) in &queue.lists[b] {
                            rows[row - first] += value;
                        }
                    }
                })
            });
        });
    }
    true
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
    let pool = ctx
        .pool()
        .filter(|_| !INSIDE.with(|flag| flag.get()) && sums.len() > 1);
    let Some(pool) = pool else {
        for (k, ((a, b), sum)) in a
            .chunks_mut(block)
            .zip(b.chunks_mut(block))
            .zip(sums.iter_mut())
            .enumerate()
        {
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
    Ok(out
        .into_iter()
        .map(|slot| slot.expect("every application ran"))
        .collect())
}
