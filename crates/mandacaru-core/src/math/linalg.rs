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
    let c = Matrix::from_fn(n, n, |i, j| eigenvectors[(i, order[j])]);
    Ok((energies, standardize_phases(c)))
}
pub fn standardize_phases(mut c: Matrix) -> Matrix {
    let n = c.nrows();
    let k = c.ncols();
    for j in 0..k {
        let pivot = (0..n)
            .max_by(|&a, &b| c[(a, j)].norm().total_cmp(&c[(b, j)].norm()))
            .unwrap_or(0);
        let norm = c[(pivot, j)].norm();
        if norm > 1e-15 {
            let phase = c[(pivot, j)].conj() / norm;
            for i in 0..n {
                c[(i, j)] *= phase;
            }
        }
    }
    c
}
/// Block Davidson iterative eigensolver for Hermitian matrices.
/// Computes eigenvalues and orthonormal eigenvectors starting from trial vectors `initial`.
pub fn davidson(
    matrix: &Matrix,
    initial: Option<&Matrix>,
    tol: f64,
    max_subspace_iter: usize,
) -> Result<(Vec<f64>, Matrix)> {
    let n = matrix.nrows();
    if n <= 2 || initial.is_none() {
        return eigen(matrix.clone());
    }
    let init = initial.unwrap();
    if init.nrows() != n || init.ncols() != n {
        return eigen(matrix.clone());
    }

    let diag: Vec<f64> = (0..n).map(|i| matrix[(i, i)].re).collect();
    let v = init.clone();
    let w = matrix * &v;
    let mut h_sub = v.adjoint() * &w;
    h_sub = (&h_sub + h_sub.adjoint()) * C64::new(0.5, 0.);
    let (mut energies, c_sub) = eigen(h_sub)?;
    let mut u = &v * &c_sub;
    let mut wu = &w * &c_sub;

    for _iter in 0..max_subspace_iter {
        let mut max_res = 0.0f64;
        let mut unconverged = Vec::new();
        for i in 0..n {
            let mut r_norm_sq = 0.0f64;
            for j in 0..n {
                let r_ji = wu[(j, i)] - C64::new(energies[i], 0.) * u[(j, i)];
                r_norm_sq += r_ji.norm_sqr();
            }
            let res = r_norm_sq.sqrt();
            if res > max_res {
                max_res = res;
            }
            if res > tol {
                unconverged.push((i, res));
            }
        }

        if max_res <= tol || unconverged.is_empty() {
            break;
        }

        unconverged.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut corrections: Vec<DVector<C64>> = Vec::new();
        for &(band, _) in unconverged.iter().take(4) {
            let theta = energies[band];
            let mut q = DVector::<C64>::zeros(n);
            for j in 0..n {
                let r_j = wu[(j, band)] - C64::new(theta, 0.) * u[(j, band)];
                let denom = theta - diag[j];
                let safe_denom = if denom.abs() < 1e-4 {
                    if denom >= 0.0 { 1e-4 } else { -1e-4 }
                } else {
                    denom
                };
                q[j] = r_j / safe_denom;
            }
            for col in 0..n {
                let u_col = u.column(col);
                let proj = (0..n).map(|k| u_col[k].conj() * q[k]).sum::<C64>();
                q -= u_col * proj;
            }
            for prev in &corrections {
                let proj = (0..n).map(|k| prev[k].conj() * q[k]).sum::<C64>();
                q -= prev * proj;
            }
            let norm = q.norm();
            if norm > 1e-8 {
                corrections.push(q / C64::new(norm, 0.));
            }
        }

        if corrections.is_empty() {
            break;
        }

        let num_corr = corrections.len();
        let mut v_exp = Matrix::zeros(n, n + num_corr);
        v_exp.view_mut((0, 0), (n, n)).copy_from(&u);
        for (idx, corr) in corrections.iter().enumerate() {
            v_exp.column_mut(n + idx).copy_from(corr);
        }

        let w_exp = matrix * &v_exp;
        let mut h_exp = v_exp.adjoint() * &w_exp;
        h_exp = (&h_exp + h_exp.adjoint()) * C64::new(0.5, 0.);
        let (e_exp, c_exp) = eigen(h_exp)?;

        let c_top = c_exp.view((0, 0), (n + num_corr, n));
        u = &v_exp * &c_top;
        wu = &w_exp * &c_top;
        energies = e_exp[..n].to_vec();
    }

    Ok((energies, standardize_phases(u)))
}

/// RMM-DIIS (Residual Minimization Method with DIIS) eigensolver for Hermitian matrices.
pub fn rmm_diis(
    matrix: &Matrix,
    initial: &Matrix,
    tol: f64,
    max_micro_iter: usize,
) -> Result<(Vec<f64>, Matrix)> {
    let n = matrix.nrows();
    if n <= 2 {
        return eigen(matrix.clone());
    }
    let diag: Vec<f64> = (0..n).map(|i| matrix[(i, i)].re).collect();
    let mut psi = initial.clone();

    for band in 0..n {
        let mut p_curr: DVector<C64> = psi.column(band).into_owned();
        let p_norm = p_curr.norm();
        if p_norm > 1e-14 {
            p_curr /= C64::new(p_norm, 0.);
        }

        let mut vectors: Vec<DVector<C64>> = Vec::with_capacity(max_micro_iter + 1);
        let mut residuals: Vec<DVector<C64>> = Vec::with_capacity(max_micro_iter + 1);

        for _micro in 0..max_micro_iter {
            let ap = matrix * &p_curr;
            let eps = (0..n).map(|k| p_curr[k].conj() * ap[k]).sum::<C64>().re;
            let r = &ap - &p_curr * C64::new(eps, 0.);
            let r_norm = r.norm();

            vectors.push(p_curr.clone());
            residuals.push(r.clone());

            if r_norm <= tol {
                break;
            }

            let mut step = DVector::<C64>::zeros(n);
            for j in 0..n {
                let denom = eps - diag[j];
                let safe_denom = if denom.abs() < 1e-4 {
                    if denom >= 0.0 { 1e-4 } else { -1e-4 }
                } else {
                    denom
                };
                step[j] = r[j] / safe_denom;
            }

            if vectors.len() == 1 {
                p_curr = &p_curr + step * C64::new(0.5, 0.);
                let norm = p_curr.norm();
                if norm > 1e-14 {
                    p_curr /= C64::new(norm, 0.);
                }
            } else {
                let m_diis = vectors.len();
                let mut b = DMatrix::<f64>::zeros(m_diis + 1, m_diis + 1);
                for i in 0..m_diis {
                    for j in 0..m_diis {
                        b[(i, j)] = (0..n).map(|k| residuals[i][k].conj() * residuals[j][k]).sum::<C64>().re;
                    }
                    b[(i, m_diis)] = -1.0;
                    b[(m_diis, i)] = -1.0;
                }
                let mut rhs = DVector::<f64>::zeros(m_diis + 1);
                rhs[m_diis] = -1.0;

                if let Some(alpha) = b.lu().solve(&rhs) {
                    if alpha.iter().all(|v| v.is_finite() && v.abs() < 100.0) {
                        let mut p_new = DVector::<C64>::zeros(n);
                        for i in 0..m_diis {
                            p_new += &vectors[i] * C64::new(alpha[i], 0.);
                        }
                        let norm = p_new.norm();
                        if norm > 1e-14 {
                            p_curr = p_new / C64::new(norm, 0.);
                            continue;
                        }
                    }
                }
                p_curr = &p_curr + step * C64::new(0.5, 0.);
                let norm = p_curr.norm();
                if norm > 1e-14 {
                    p_curr /= C64::new(norm, 0.);
                }
            }
        }
        psi.column_mut(band).copy_from(&p_curr);
    }

    let s = psi.adjoint() * &psi;
    let s_sym = (&s + s.adjoint()) * C64::new(0.5, 0.);
    let (s_evals, s_evecs) = eigen(s_sym)?;
    let mut inv_sqrt_s = Matrix::zeros(n, n);
    for i in 0..n {
        let val = if s_evals[i] > 1e-10 {
            1.0 / s_evals[i].sqrt()
        } else {
            1.0
        };
        inv_sqrt_s[(i, i)] = C64::new(val, 0.);
    }
    let lowdin = &s_evecs * inv_sqrt_s * s_evecs.adjoint();
    let psi_ortho = &psi * lowdin;

    let w_ortho = matrix * &psi_ortho;
    let mut h_sub = psi_ortho.adjoint() * &w_ortho;
    h_sub = (&h_sub + h_sub.adjoint()) * C64::new(0.5, 0.);
    let (energies, c_sub) = eigen(h_sub)?;
    let final_u = standardize_phases(&psi_ortho * c_sub);

    Ok((energies, final_u))
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
/// The eigen decomposition of a real symmetric matrix in the QR iteration's own order and
/// signs (unsorted, as `SymmetricEigen::new` returns it), accepted only after its residual and
/// orthogonality are checked; a decomposition that fails the check is redone by Jacobi
/// rotations. A matrix the QR iteration decomposes accurately gets its result unchanged.
pub fn checked_symmetric(matrix: DMatrix<f64>) -> Result<(DVector<f64>, DMatrix<f64>)> {
    if matrix.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite symmetric matrix"));
    }
    let accept = acceptance(matrix.nrows());
    let spectrum = SymmetricEigen::new(matrix.clone());
    if decomposition_error(&matrix, &spectrum.eigenvalues, &spectrum.eigenvectors) <= accept {
        return Ok((spectrum.eigenvalues, spectrum.eigenvectors));
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
