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
    let (eigenvalues, eigenvectors) = verified_hermitian(matrix)?;
    let n = eigenvalues.len();
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));
    let energies = order.iter().map(|&j| eigenvalues[j]).collect();
    let mut c = Matrix::from_fn(n, n, |i, j| eigenvectors[(i, order[j])]);
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
    // The residual scales with the matrix; orthogonality is dimensionless.
    (residual.norm() / matrix.norm().max(f64::MIN_POSITIVE)).max(orthogonality.norm())
}
/// Largest relative error a decomposition may have and still be accepted. It
/// grows slowly with the dimension (rounding does); an inaccurate QR result is
/// orders of magnitude above it.
fn acceptance(n: usize) -> f64 {
    1e-13 * n.max(10) as f64
}
fn hermitian_error(matrix: &Matrix, values: &DVector<f64>, vectors: &Matrix) -> f64 {
    let n = values.len();
    let diagonal = Matrix::from_diagonal(&values.map(|v| C64::new(v, 0.)));
    let residual = matrix * vectors - vectors * diagonal;
    let orthogonality = vectors.adjoint() * vectors - Matrix::identity(n, n);
    // The residual scales with the matrix; orthogonality is dimensionless.
    (residual.norm() / matrix.norm().max(f64::MIN_POSITIVE)).max(orthogonality.norm())
}
/// Cyclic Jacobi rotations for a Hermitian matrix. The 2x2 unitary carries the
/// phase of `a_pq`, so a real symmetric input takes the real rotations.
fn hermitian_jacobi(mut a: Matrix) -> Option<(DVector<f64>, Matrix)> {
    let n = a.nrows();
    let mut v = Matrix::identity(n, n);
    let scale = a.norm().max(f64::MIN_POSITIVE);
    for _ in 0..100 {
        let off = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .map(|(p, q)| a[(p, q)].norm_sqr())
            .sum::<f64>()
            .sqrt();
        if off <= 1e-15 * scale {
            return Some((DVector::from_iterator(n, (0..n).map(|i| a[(i, i)].re)), v));
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[(p, q)];
                let r = apq.norm();
                if r <= 1e-18 * scale {
                    continue;
                }
                let phase = apq / r;
                let theta = (a[(q, q)].re - a[(p, p)].re) / (2. * r);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.).sqrt());
                let c = 1. / (t * t + 1.).sqrt();
                let s = t * c;
                let (sp, sm) = (phase * s, phase.conj() * s);
                for k in 0..n {
                    let (akp, akq) = (a[(k, p)], a[(k, q)]);
                    a[(k, p)] = akp * c - akq * sm;
                    a[(k, q)] = akp * sp + akq * c;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[(p, k)], a[(q, k)]);
                    a[(p, k)] = apk * c - aqk * sp;
                    a[(q, k)] = apk * sm + aqk * c;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[(k, p)], v[(k, q)]);
                    v[(k, p)] = vkp * c - vkq * sm;
                    v[(k, q)] = vkp * sp + vkq * c;
                }
            }
        }
    }
    None
}
/// The complex counterpart of [`verified_symmetric`]: the same tridiagonal QR
/// iteration, the same silent failure on degenerate or clustered spectra, the
/// same check and fallback. It diagonalizes every Fock matrix of an SCF.
fn verified_hermitian(matrix: Matrix) -> Result<(DVector<f64>, Matrix)> {
    let accept = acceptance(matrix.nrows());
    if let Some(spectrum) = SymmetricEigen::try_new(matrix.clone(), 1e-14, 10000) {
        if hermitian_error(&matrix, &spectrum.eigenvalues, &spectrum.eigenvectors) <= accept {
            return Ok((spectrum.eigenvalues, spectrum.eigenvectors));
        }
    }
    let (values, vectors) = hermitian_jacobi(matrix.clone())
        .ok_or_else(|| crate::Error::Kernel("Hermitian eigensolver failed to converge".into()))?;
    if hermitian_error(&matrix, &values, &vectors) > accept {
        return Err(crate::Error::Kernel(
            "Hermitian eigensolver did not reach the requested accuracy".into(),
        ));
    }
    Ok((values, vectors))
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
    let accept = acceptance(matrix.nrows());
    if let Some(spectrum) = SymmetricEigen::try_new(matrix.clone(), 1e-14, 10000) {
        if decomposition_error(&matrix, &spectrum.eigenvalues, &spectrum.eigenvectors) <= accept {
            return Ok((spectrum.eigenvalues, spectrum.eigenvectors));
        }
    }
    let (values, vectors) = jacobi(matrix.clone())
        .ok_or_else(|| crate::Error::Kernel("symmetric eigensolver failed to converge".into()))?;
    if decomposition_error(&matrix, &values, &vectors) > accept {
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
