//! Hermitian eigensolvers with deterministic ordering and phase conventions.
use super::invalid;
use crate::Result;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
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
/// Largest residual of an eigendecomposition relative to the matrix scale.
fn decomposition_error(
    matrix: &DMatrix<f64>,
    values: &DVector<f64>,
    vectors: &DMatrix<f64>,
) -> f64 {
    let residual = matrix * vectors - vectors * DMatrix::from_diagonal(values);
    let orthogonality =
        vectors.transpose() * vectors - DMatrix::identity(values.len(), values.len());
    residual.norm().max(orthogonality.norm()) / matrix.norm().max(1.)
}
/// Cyclic Jacobi rotations: slower than the tridiagonal QR iteration but
/// unconditionally accurate for clustered and degenerate spectra.
fn jacobi(mut a: DMatrix<f64>) -> Option<(DVector<f64>, DMatrix<f64>)> {
    let n = a.nrows();
    let mut v = DMatrix::identity(n, n);
    let scale = a.norm().max(f64::MIN_POSITIVE);
    for _ in 0..100 {
        let off = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .map(|(p, q)| a[(p, q)].powi(2))
            .sum::<f64>()
            .sqrt();
        if off <= 1e-15 * scale {
            return Some((a.diagonal(), v));
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[(p, q)].abs() <= 1e-18 * scale {
                    continue;
                }
                let theta = (a[(q, q)] - a[(p, p)]) / (2. * a[(p, q)]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.).sqrt());
                let c = 1. / (t * t + 1.).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[(k, p)], a[(k, q)]);
                    a[(k, p)] = c * akp - s * akq;
                    a[(k, q)] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[(p, k)], a[(q, k)]);
                    a[(p, k)] = c * apk - s * aqk;
                    a[(q, k)] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[(k, p)], v[(k, q)]);
                    v[(k, p)] = c * vkp - s * vkq;
                    v[(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }
    None
}
/// The tridiagonal QR iteration can stop at an inaccurate decomposition of a
/// matrix with a degenerate or clustered spectrum (a symmetric molecular PAW
/// compensation block did, and its energy was wrong by two hartree), so its
/// result is accepted only after the residual and orthogonality are checked.
fn verified_symmetric(matrix: DMatrix<f64>) -> Result<(DVector<f64>, DMatrix<f64>)> {
    const ACCEPT: f64 = 1e-10;
    if let Some(spectrum) = SymmetricEigen::try_new(matrix.clone(), 1e-14, 10000) {
        if decomposition_error(&matrix, &spectrum.eigenvalues, &spectrum.eigenvectors) <= ACCEPT {
            return Ok((spectrum.eigenvalues, spectrum.eigenvectors));
        }
    }
    let (values, vectors) = jacobi(matrix.clone())
        .ok_or_else(|| crate::Error::Kernel("symmetric eigensolver failed to converge".into()))?;
    if decomposition_error(&matrix, &values, &vectors) > ACCEPT {
        return Err(crate::Error::Kernel(
            "symmetric eigensolver did not reach the requested accuracy".into(),
        ));
    }
    Ok((values, vectors))
}
pub fn real_eigen(matrix: DMatrix<f64>) -> Result<(Vec<f64>, DMatrix<f64>)> {
    if matrix.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite symmetric matrix"));
    }
    let (eigenvalues, eigenvectors) = verified_symmetric(matrix)?;
    let mut order: Vec<_> = (0..eigenvalues.len()).collect();
    order.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));
    let values = order.iter().map(|&i| eigenvalues[i]).collect();
    let mut vectors =
        DMatrix::from_fn(order.len(), order.len(), |i, j| eigenvectors[(i, order[j])]);
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
