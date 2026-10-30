//! Compressed-column storage built directly from a Pauli sum on its basis.
use super::{amplitude::Amplitude, Basis, Operator, C64};
use crate::{product, Buffer, Context, Error, Result};
use std::collections::BTreeMap;

pub struct SparseMatrix {
    identity: C64,
    offsets: Buffer<usize>,
    rows: Buffer<usize>,
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
        let mut rows = ctx.allocate(capacity)?;
        let mut values = ctx.allocate(capacity)?;
        let mut cursor = 0;
        for (col, &ket) in basis.indices.as_slice().iter().enumerate() {
            for (&flip, terms) in &flips {
                if let Some(row) = basis.index(ket ^ flip) {
                    let value: C64 = terms
                        .iter()
                        .map(|&(z, coefficient)| {
                            if (z & ket).count_ones() % 2 == 0 {
                                coefficient
                            } else {
                                -coefficient
                            }
                        })
                        .sum();
                    if value.norm() <= 1e-14 {
                        continue;
                    }
                    rows.as_mut_slice()[cursor] = row;
                    values.as_mut_slice()[cursor] = value;
                    cursor += 1;
                }
            }
            offsets.as_mut_slice()[col + 1] = cursor;
        }
        Ok(Self {
            identity,
            offsets,
            rows,
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
        for (col, amplitude) in state.iter().enumerate() {
            let weighted = scale * amplitude.widen();
            for k in self.offsets.as_slice()[col]..self.offsets.as_slice()[col + 1] {
                out[self.rows.as_slice()[k]] += self.values.as_slice()[k] * weighted;
            }
        }
    }
}
