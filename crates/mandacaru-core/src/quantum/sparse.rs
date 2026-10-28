//! Compressed-column storage built directly from a Pauli sum on its basis.
use super::{Basis, Operator, C64};
use crate::{product, Buffer, Context, Error, Result};
use std::collections::BTreeMap;

pub struct SparseMatrix {
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
        let capacity = product(&[d, op.terms.len().min(d)])?;
        let mut offsets = ctx.allocate(d.checked_add(1).ok_or(Error::Overflow)?)?;
        let mut rows = ctx.allocate(capacity)?;
        let mut values = ctx.allocate(capacity)?;
        let _column = ctx.reserve(product(&[op.terms.len(), 128])?)?;
        let mut cursor = 0;
        for (col, &ket) in basis.indices.as_slice().iter().enumerate() {
            let mut entries = BTreeMap::new();
            for t in &op.terms {
                if let Ok(row) = basis.indices.as_slice().binary_search(&(ket ^ t.x)) {
                    *entries.entry(row).or_insert(C64::default()) += t.amplitude(ket);
                }
            }
            for (row, value) in entries {
                if value.norm() > 1e-14 {
                    rows.as_mut_slice()[cursor] = row;
                    values.as_mut_slice()[cursor] = value;
                    cursor += 1;
                }
            }
            offsets.as_mut_slice()[col + 1] = cursor;
        }
        Ok(Self {
            offsets,
            rows,
            values,
        })
    }
    pub fn apply_into(&self, state: &[C64], out: &mut [C64]) {
        out.fill(C64::default());
        for (col, amplitude) in state.iter().enumerate() {
            for k in self.offsets.as_slice()[col]..self.offsets.as_slice()[col + 1] {
                out[self.rows.as_slice()[k]] += self.values.as_slice()[k] * amplitude;
            }
        }
    }
}
