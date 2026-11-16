//! The factors of a pivoted Cholesky while they grow: `n` rows (the packed pairs) that gain one
//! entry per accepted pivot. The rank is not known in advance, so the rows live in one zeroed
//! allocation that the operating system maps lazily ([`crate::Context::allocate_lazily`]), sized
//! for the most a rank can be (`n`, or what the memory budget and the machine hold if that is
//! fewer), and only its first `n x stride` entries are used, dense: each row is `stride` long,
//! and when the rows are full the stride grows by an eighth by sliding them up in place (last
//! row first). So
//! - the memory resident and charged is `n x stride`, within an eighth of the factors, and never
//!   a row's partially filled pages beyond it;
//! - the update `sum_k L[a,k] L[pivot,k]` of every row is one contiguous dot product, parallel
//!   over rows;
//! - the pair-major matrix `n x rank` of [`super::integrals::Eri::Cholesky`] is made at the end
//!   by sliding the rows down to stride `rank` (into pages they already occupy) and releasing the
//!   capacity beyond, so the factors are never held twice.
use crate::{Buffer, Context, Lazy, Result};
use rayon::prelude::*;

/// Entries per row the first allocation holds, and the least the stride grows by.
const FIRST_STRIDE: usize = 64;

pub(super) struct GrowingRows<'a> {
    ctx: &'a Context,
    rows: usize,
    /// Entries per row of the dense region in use.
    stride: usize,
    /// The most the stride can reach: what the budget and the machine hold, at most `rows`.
    limit: usize,
    data: Lazy<f64>,
    /// Entries per row filled so far.
    count: usize,
}

/// `sum_i a[i] b[i]`, four accumulators.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut lanes = [0f64; 4];
    let (ha, hb) = (a.chunks_exact(4), b.chunks_exact(4));
    let tail: f64 = ha
        .remainder()
        .iter()
        .zip(hb.remainder())
        .map(|(x, y)| x * y)
        .sum();
    for (x, y) in ha.zip(hb) {
        for l in 0..4 {
            lanes[l] += x[l] * y[l];
        }
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) + tail
}

impl<'a> GrowingRows<'a> {
    pub(super) fn new(ctx: &'a Context, rows: usize) -> Result<Self> {
        let machine = mandacaru_kernels::heap::installed_memory_bytes()
            .and_then(|bytes| usize::try_from(bytes).ok())
            .unwrap_or(usize::MAX);
        let held = ctx.memory_limit().min(machine);
        let limit = rows
            .min(held / crate::product(&[rows, 8])?.max(1))
            .max(1);
        Ok(Self {
            ctx,
            rows,
            stride: 0,
            limit,
            data: ctx.allocate_lazily(crate::product(&[rows, limit])?)?,
            count: 0,
        })
    }

    pub(super) fn count(&self) -> usize {
        self.count
    }

    /// Row `a`, its filled entries.
    pub(super) fn row(&self, a: usize) -> &[f64] {
        &self.data.as_slice()[a * self.stride..a * self.stride + self.count]
    }

    /// `out[a] = sum_k L[a,k] pivot[k]` for every row (`pivot` is a filled row).
    pub(super) fn apply(&self, pivot: &[f64], out: &mut [f64]) {
        debug_assert_eq!(pivot.len(), self.count);
        if self.count == 0 {
            out.fill(0.);
            return;
        }
        let (stride, count) = (self.stride, self.count);
        let work = |(row, slot): (&[f64], &mut f64)| *slot = dot(&row[..count], pivot);
        let used = &self.data.as_slice()[..self.rows * stride];
        match self.ctx.pool() {
            Some(pool) => pool.install(|| {
                used.par_chunks(stride)
                    .zip(out.par_iter_mut())
                    .for_each(work)
            }),
            None => used.chunks(stride).zip(out.iter_mut()).for_each(work),
        }
    }

    /// Room for more entries per row: the stride grows by an eighth (at least
    /// [`FIRST_STRIDE`]), the rows slid up in place, last first; the bytes are charged first.
    fn grow(&mut self) -> Result<()> {
        if self.stride == self.limit {
            return Err(crate::Error::Budget {
                requested: self.rows * 8,
                available: 0,
            });
        }
        let next = (self.stride + (self.stride / 8).max(FIRST_STRIDE)).min(self.limit);
        self.data.charge(self.rows * (next - self.stride))?;
        let (old, count) = (self.stride, self.count);
        let data = self.data.as_mut_slice();
        for a in (1..self.rows).rev() {
            data.copy_within(a * old..a * old + count, a * next);
        }
        self.stride = next;
        Ok(())
    }

    /// Appends `values[a]` to every row.
    pub(super) fn push(&mut self, values: &[f64]) -> Result<()> {
        debug_assert_eq!(values.len(), self.rows);
        if self.count == self.stride {
            self.grow()?;
        }
        let (stride, count) = (self.stride, self.count);
        let data = self.data.as_mut_slice();
        for (a, v) in values.iter().enumerate() {
            data[a * stride + count] = *v;
        }
        self.count += 1;
        Ok(())
    }

    /// The pair-major matrix `rows x count` (row `a` at `a * count`, the layout of
    /// [`super::integrals::Eri::Cholesky`]), in the storage the rows grew in.
    pub(super) fn into_row_major(mut self) -> Result<Buffer<f64>> {
        let (stride, count) = (self.stride, self.count);
        let data = self.data.as_mut_slice();
        for a in 1..self.rows {
            data.copy_within(a * stride..a * stride + count, a * count);
        }
        self.data.finish(self.rows * count)
    }
}
