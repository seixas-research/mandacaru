//! A complex compressed-row matrix and the few operations its interface view needs: the
//! conjugate, the transpose (columns ascending in every row), the dense form and the product with
//! a dense row-major operand. Each output entry is the sum of a row's products in column order.
use num_complex::Complex64;

#[derive(Clone, Debug)]
pub struct ComplexCsr {
    pub rows: usize,
    pub cols: usize,
    /// `rows + 1` row starts into `columns` and `values`.
    pub offsets: Vec<usize>,
    pub columns: Vec<usize>,
    pub values: Vec<Complex64>,
}

impl ComplexCsr {
    pub fn nnz(&self) -> usize {
        self.columns.len()
    }

    /// The elementwise complex conjugate.
    pub fn conj(&self) -> Self {
        Self {
            values: self.values.iter().map(|v| v.conj()).collect(),
            ..self.clone()
        }
    }

    /// The transpose, by counting sort: in every row of the result the columns ascend.
    pub fn transpose(&self) -> Self {
        let mut offsets = vec![0usize; self.cols + 1];
        for &c in &self.columns {
            if c < self.cols {
                offsets[c + 1] += 1;
            }
        }
        for c in 0..self.cols {
            offsets[c + 1] += offsets[c];
        }
        let mut head = offsets[..self.cols].to_vec();
        let total = offsets[self.cols];
        let mut columns = vec![0usize; total];
        let mut values = vec![Complex64::new(0.0, 0.0); total];
        for r in 0..self.rows {
            for k in self.offsets[r]..self.offsets[r + 1] {
                let c = self.columns[k];
                if c < self.cols {
                    columns[head[c]] = r;
                    values[head[c]] = self.values[k];
                    head[c] += 1;
                }
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            offsets,
            columns,
            values,
        }
    }

    /// The dense matrix, row-major.
    pub fn dense(&self) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); self.rows * self.cols];
        for r in 0..self.rows {
            for k in self.offsets[r]..self.offsets[r + 1] {
                if self.columns[k] < self.cols {
                    out[r * self.cols + self.columns[k]] = self.values[k];
                }
            }
        }
        out
    }

    /// `A X` for the dense row-major operand `X` of `width` columns (a vector for `width = 1`),
    /// each entry the products `A_rk * X_kj` (`mul`) of a row added in column order; columns of
    /// `A` beyond the operand's rows are skipped.
    pub fn apply<T: Copy>(
        &self,
        operand: &[T],
        width: usize,
        mul: impl Fn(Complex64, T) -> Complex64,
    ) -> Vec<Complex64> {
        let operand_rows = operand.len().checked_div(width).unwrap_or(0);
        let mut out = vec![Complex64::new(0.0, 0.0); self.rows * width];
        for r in 0..self.rows {
            for j in 0..width {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in self.offsets[r]..self.offsets[r + 1] {
                    let c = self.columns[k];
                    if c < operand_rows {
                        sum += mul(self.values[k], operand[c * width + j]);
                    }
                }
                out[r * width + j] = sum;
            }
        }
        out
    }
}
