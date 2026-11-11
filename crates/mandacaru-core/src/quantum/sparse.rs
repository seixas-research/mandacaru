//! Compressed-row storage built directly from a Pauli sum on its basis.
use super::{amplitude::Amplitude, parallel::parallel_rows, Basis, Operator, C64};
use crate::{product, Buffer, Context, Error, Result};
use std::collections::BTreeMap;

/// Row `r` holds, for every flip mask that reaches it from a basis state, the column of that
/// source state and the exact coefficient sum of the mask's strings at it; applying the matrix
/// gathers each row from inputs it only reads, so the rows split across threads.
pub struct SparseMatrix {
    identity: C64,
    /// The building context: its worker pool runs every large action.
    context: Context,
    offsets: Buffer<usize>,
    columns: Buffer<usize>,
    values: Buffer<C64>,
}
impl SparseMatrix {
    pub fn new(ctx: &Context, basis: &Basis, op: &Operator) -> Result<Self> {
        if basis.n != op.n {
            return Err(Error::Invalid("operator basis mismatch".into()));
        }
        let d = basis.len();
        let _column = ctx.reserve(product(&[op.terms.len(), 128])?)?;
        let identity = op.identity();
        let mut flips = BTreeMap::<u64, Vec<_>>::new();
        for term in &op.terms {
            if term.x == 0 && term.z == 0 {
                continue;
            }
            flips
                .entry(term.x)
                .or_default()
                .push((term.z, term.amplitude(0)));
        }
        let mut capacity = 0usize;
        for &ket in basis.indices.as_slice() {
            for &flip in flips.keys() {
                if basis.index(ket ^ flip).is_some() {
                    capacity = capacity.checked_add(1).ok_or(Error::Overflow)?;
                }
            }
        }
        let mut offsets = ctx.allocate(d.checked_add(1).ok_or(Error::Overflow)?)?;
        let mut columns = ctx.allocate(capacity)?;
        let mut values = ctx.allocate(capacity)?;
        let mut cursor = 0;
        for (row, &ket) in basis.indices.as_slice().iter().enumerate() {
            for (&flip, terms) in &flips {
                let source = ket ^ flip;
                if let Some(column) = basis.index(source) {
                    let value: C64 = terms
                        .iter()
                        .map(|&(z, coefficient)| {
                            if (z & source).count_ones() % 2 == 0 {
                                coefficient
                            } else {
                                -coefficient
                            }
                        })
                        .sum();
                    if value.norm() <= 1e-14 {
                        continue;
                    }
                    columns.as_mut_slice()[cursor] = column;
                    values.as_mut_slice()[cursor] = value;
                    cursor += 1;
                }
            }
            offsets.as_mut_slice()[row + 1] = cursor;
        }
        Ok(Self {
            identity,
            context: ctx.clone(),
            offsets,
            columns,
            values,
        })
    }
    /// `out = H state`; the state is stored in either precision, the row is float64.
    pub fn apply_into<A: Amplitude>(&self, state: &[A], out: &mut [C64]) {
        out.fill(C64::default());
        self.add_into(state, C64::new(1., 0.), out);
        for (value, amplitude) in out.iter_mut().zip(state) {
            *value += self.identity * amplitude.widen();
        }
    }
    pub(super) fn apply_centered_into<A: Amplitude>(&self, state: &[A], out: &mut [C64]) {
        out.fill(C64::default());
        self.add_into(state, C64::new(1., 0.), out);
    }
    fn add_into<A: Amplitude>(&self, state: &[A], scale: C64, out: &mut [C64]) {
        let (offsets, columns, values) =
            (self.offsets.as_slice(), self.columns.as_slice(), self.values.as_slice());
        let rows = out.len();
        parallel_rows(&self.context, rows, values.len(), out, |start, block| {
            for (offset, value) in block.iter_mut().enumerate() {
                let row = start + offset;
                // Four independent partial sums in a fixed order: a row's entries no longer wait
                // on one another, and the result does not depend on the thread split.
                let mut partial = [C64::default(); 4];
                let (first, last) = (offsets[row], offsets[row + 1]);
                for (k, (&column, &entry)) in columns[first..last].iter().zip(&values[first..last]).enumerate() {
                    partial[k % 4] += entry * state[column].widen();
                }
                *value += scale * ((partial[0] + partial[1]) + (partial[2] + partial[3]));
            }
        });
    }
}
