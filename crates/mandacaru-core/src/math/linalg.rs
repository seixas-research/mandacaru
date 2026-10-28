//! Hermitian eigensolvers with deterministic ordering and phase conventions.
use super::invalid;
use crate::Result;
use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64 as C64;
type Matrix = DMatrix<C64>;
pub fn eigen(matrix: Matrix) -> Result<(Vec<f64>, Matrix)> {
    if matrix
        .iter()
        .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(invalid("nonfinite Hermitian matrix"));
    }
    let spectrum = SymmetricEigen::try_new(matrix, 1e-14, 10000)
        .ok_or_else(|| crate::Error::Kernel("Hermitian eigensolver did not converge".into()))?;
    let n = spectrum.eigenvalues.len();
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&a, &b| spectrum.eigenvalues[a].total_cmp(&spectrum.eigenvalues[b]));
    let energies = order.iter().map(|&j| spectrum.eigenvalues[j]).collect();
    let mut c = Matrix::from_fn(n, n, |i, j| spectrum.eigenvectors[(i, order[j])]);
    for j in 0..n {
        let pivot = (0..n)
            .max_by(|&a, &b| c[(a, j)].norm().total_cmp(&c[(b, j)].norm()))
            .unwrap();
        let phase = c[(pivot, j)].conj() / c[(pivot, j)].norm();
        for i in 0..n {
            c[(i, j)] *= phase;
        }
    }
    Ok((energies, c))
}
pub fn real_eigen(matrix: DMatrix<f64>) -> Result<(Vec<f64>, DMatrix<f64>)> {
    if matrix.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite symmetric matrix"));
    }
    let spectrum = SymmetricEigen::try_new(matrix, 1e-14, 10000)
        .ok_or_else(|| crate::Error::Kernel("symmetric eigensolver failed to converge".into()))?;
    let mut order: Vec<_> = (0..spectrum.eigenvalues.len()).collect();
    order.sort_by(|&a, &b| spectrum.eigenvalues[a].total_cmp(&spectrum.eigenvalues[b]));
    let values = order.iter().map(|&i| spectrum.eigenvalues[i]).collect();
    let mut vectors = DMatrix::from_fn(order.len(), order.len(), |i, j| {
        spectrum.eigenvectors[(i, order[j])]
    });
    for j in 0..vectors.ncols() {
        let pivot = (0..vectors.nrows())
            .max_by(|&a, &b| vectors[(a, j)].abs().total_cmp(&vectors[(b, j)].abs()))
            .unwrap();
        if vectors[(pivot, j)] < 0. {
            for i in 0..vectors.nrows() {
                vectors[(i, j)] *= -1.;
            }
        }
    }
    Ok((values, vectors))
}
