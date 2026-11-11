//! A truncated minimum-norm solve of `A x = b` for a symmetric matrix known only up to noise.
//!
//! The matrix is a finite-difference Hessian: its entries carry a truncation error (the step) and
//! a roundoff error (the analytic gradients it differences), so its smallest eigenvalues are not
//! trustworthy. The noise is measured, not assumed: for the exact Hessian `A - A^T = 0`, so the
//! norm of the antisymmetric part is the size of the error. Eigen-directions whose eigenvalue is
//! within `safety` times that noise of zero (or below `relative` of the largest eigenvalue) are
//! dropped, which is the pseudo-inverse of the matrix: a flat or redundant direction carries no
//! correction. An eigenvalue below `-cutoff` is a real negative curvature and is refused.
use super::invalid;
use crate::Result;
use nalgebra::{DMatrix, DVector};

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// The cutoff is at least this many times the measured noise of the matrix.
    pub safety: f64,
    /// The cutoff is at least this fraction of the largest eigenvalue magnitude.
    pub relative: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            safety: 10.,
            relative: 1e-10,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub x: Vec<f64>,
    /// Eigenvalues of the symmetrized matrix, ascending.
    pub eigenvalues: Vec<f64>,
    /// The orthonormal eigenvectors, one column per eigenvalue in [`Self::eigenvalues`]' order.
    pub vectors: DMatrix<f64>,
    /// `|A - A^T|_F`: the measured size of the error of the matrix.
    pub noise: f64,
    /// Directions with `|eigenvalue| <= cutoff` are dropped.
    pub cutoff: f64,
    pub kept: usize,
    pub dropped: usize,
    /// The smallest eigenvalue that was kept (`None` if none was).
    pub smallest_kept: Option<f64>,
    /// The norm of the part of `b` along the dropped directions.
    pub dropped_rhs: f64,
}
impl Solution {
    /// `|b_dropped| / cutoff`: the part of `x` the dropped directions would add if their
    /// eigenvalues were exactly the cutoff. Their true eigenvalues are smaller, so the neglected
    /// part is at least this large; it measures what the truncation leaves out.
    pub fn neglected_scale(&self) -> f64 {
        if self.dropped_rhs == 0. {
            0.
        } else {
            self.dropped_rhs / self.cutoff
        }
    }
}

pub fn solve(matrix: &DMatrix<f64>, b: &[f64], options: &Options) -> Result<Solution> {
    let n = b.len();
    if matrix.nrows() != n
        || matrix.ncols() != n
        || n == 0
        || matrix.iter().chain(b).any(|v| !v.is_finite())
        || !(options.safety > 0.)
        || !(options.relative >= 0.)
    {
        return Err(invalid("the symmetric solve needs a finite square matrix, a matching right-hand side and positive tolerances"));
    }
    let noise = (matrix - matrix.transpose()).norm();
    let symmetric = (matrix + matrix.transpose()) * 0.5;
    // The verified solver: the unverified one can leave the reconstruction off by far more than
    // roundoff, and the truncation then drops or keeps the wrong directions.
    let (eigenvalues, vectors) = super::linalg::real_eigen(symmetric)?;
    let largest = eigenvalues.iter().fold(0f64, |m, v| m.max(v.abs()));
    let cutoff = (options.safety * noise).max(options.relative * largest);
    if eigenvalues[0] < -cutoff {
        return Err(invalid(format!(
            "negative curvature {:.3e} (cutoff {cutoff:.3e}): the matrix is indefinite",
            eigenvalues[0]
        )));
    }
    let rhs = DVector::from_column_slice(b);
    let mut x = DVector::zeros(n);
    let (mut kept, mut smallest_kept, mut dropped_rhs) = (0, None, 0f64);
    for (i, &value) in eigenvalues.iter().enumerate() {
        let vector = vectors.column(i);
        let component = vector.dot(&rhs);
        if value.abs() <= cutoff {
            dropped_rhs += component * component;
        } else {
            x += vector * (component / value);
            kept += 1;
            smallest_kept = Some(smallest_kept.map_or(value, |s: f64| s.min(value)));
        }
    }
    Ok(Solution {
        x: x.iter().copied().collect(),
        eigenvalues,
        vectors,
        noise,
        cutoff,
        kept,
        dropped: n - kept,
        smallest_kept,
        dropped_rhs: dropped_rhs.sqrt(),
    })
}
