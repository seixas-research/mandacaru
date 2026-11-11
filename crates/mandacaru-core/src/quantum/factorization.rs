//! Quantum chemistry tensor factorizations: double factorization into squared one-body operators.

use nalgebra::{DMatrix, DVector};

/// The checked symmetric eigen decomposition (`math::linalg::checked_symmetric`): the bare QR
/// iteration can return orthonormal vectors with a wrong spectrum (a PAW leaf of H2O was off by
/// 1.1e-4), which would measure a different Hamiltonian.
struct Spectrum {
    eigenvalues: DVector<f64>,
    eigenvectors: DMatrix<f64>,
}
fn checked(matrix: DMatrix<f64>) -> Result<Spectrum, String> {
    let (eigenvalues, eigenvectors) =
        crate::math::linalg::checked_symmetric(matrix).map_err(|e| e.to_string())?;
    Ok(Spectrum { eigenvalues, eigenvectors })
}
use num_complex::Complex64;

#[derive(Debug, Clone)]
pub struct FactorizationLeafData {
    pub weight: f64,
    pub orbitals: DMatrix<f64>,
    pub coefficients: DVector<f64>,
}

#[derive(Debug, Clone)]
pub struct DoubleFactorizationData {
    pub n_modes: usize,
    pub constant: f64,
    pub one_body: DMatrix<Complex64>,
    pub one_body_orbitals: DMatrix<f64>,
    pub one_body_energies: DVector<f64>,
    pub leaves: Vec<FactorizationLeafData>,
    pub truncation_error: f64,
}

/// Compute the pair spectrum (weights, vectors) of the spatial pair matrix
/// W_{(pr),(qs)} = <pq|rs>.
pub fn pair_spectrum(
    eri_dense: Option<(&[f64], usize)>,
    eri_packed: Option<(&[f64], usize)>,
    factors: Option<(&[f64], usize, usize)>,
    coupling: Option<(&[f64], usize)>,
) -> Result<(DVector<f64>, DMatrix<f64>), String> {
    if let Some((f_slice, k, m)) = factors {
        let mut vectors = DMatrix::zeros(m * m, k);
        for i in 0..k {
            for p in 0..m {
                for r in 0..m {
                    vectors[(p * m + r, i)] = f_slice[i * m * m + p * m + r];
                }
            }
        }
        let mut k_mat = DMatrix::identity(k, k);
        if let Some((c_slice, c_dim)) = coupling {
            if c_dim != k {
                return Err(format!("coupling dim {} does not match k {}", c_dim, k));
            }
            for i in 0..k {
                for j in 0..k {
                    k_mat[(i, j)] = 0.5 * (c_slice[i * k + j] + c_slice[j * k + i]);
                }
            }
        }
        if k >= m * m {
            let w = &vectors * &k_mat * vectors.transpose();
            let w_sym = 0.5 * (&w + w.transpose());
            let eig = checked(w_sym)?;
            Ok((eig.eigenvalues, eig.eigenvectors))
        } else {
            let qr = vectors.qr();
            let q = qr.q();
            let r = qr.r();
            let small = &r * &k_mat * r.transpose();
            let small_sym = 0.5 * (&small + small.transpose());
            let eig = checked(small_sym)?;
            let v = eig.eigenvectors;
            let q_v = &q * &v;
            Ok((eig.eigenvalues, q_v))
        }
    } else if let Some((data, m)) = eri_packed {
        let p = m * (m + 1) / 2;
        let mut rows = Vec::with_capacity(p);
        let mut cols = Vec::with_capacity(p);
        for r in 0..m {
            for c in 0..=r {
                rows.push(r);
                cols.push(c);
            }
        }
        let root: Vec<f64> = rows
            .iter()
            .zip(cols.iter())
            .map(|(&r, &c)| if r == c { 1.0 } else { 2.0f64.sqrt() })
            .collect();

        let mut w = DMatrix::zeros(p, p);
        for u in 0..p {
            for v in 0..p {
                let idx = if u >= v {
                    u * (u + 1) / 2 + v
                } else {
                    v * (v + 1) / 2 + u
                };
                w[(u, v)] = data[idx] * root[u] * root[v];
            }
        }
        let w_sym = 0.5 * (&w + w.transpose());
        let eig = checked(w_sym)?;
        let weights = eig.eigenvalues;
        let y = eig.eigenvectors;

        let mut vectors = DMatrix::zeros(m * m, p);
        for l in 0..p {
            for u in 0..p {
                let y_val = y[(u, l)] / root[u];
                let r = rows[u];
                let c = cols[u];
                vectors[(r * m + c, l)] = y_val;
                vectors[(c * m + r, l)] = y_val;
            }
        }
        Ok((weights, vectors))
    } else if let Some((g, m)) = eri_dense {
        let m3 = m * m * m;
        let m2 = m * m;
        let mut w = DMatrix::zeros(m2, m2);
        for p in 0..m {
            for r in 0..m {
                let pr = p * m + r;
                for q in 0..m {
                    for s in 0..m {
                        let qs = q * m + s;
                        let val_prqs = g[p * m3 + q * m2 + r * m + s];
                        let val_qspr = g[q * m3 + p * m2 + s * m + r];
                        w[(pr, qs)] = 0.5 * (val_prqs + val_qspr);
                    }
                }
            }
        }
        let eig = checked(w)?;
        Ok((eig.eigenvalues, eig.eigenvectors))
    } else {
        Err("either dense eri, packed eri, or factors must be provided".to_string())
    }
}

/// Double factorization of a general spin-orbital Hamiltonian.
pub fn double_factorization(
    h_pq: &[f64],
    g_pqrs: &[f64],
    m: usize,
    tol: f64,
    coefficient_tol: f64,
    max_leaves: Option<usize>,
    constant: f64,
) -> Result<DoubleFactorizationData, String> {
    if h_pq.len() != m * m {
        return Err(format!("expected h_pq with {} elements, got {}", m * m, h_pq.len()));
    }
    if g_pqrs.len() != m * m * m * m {
        return Err(format!("expected g_pqrs with {} elements, got {}", m * m * m * m, g_pqrs.len()));
    }

    let m3 = m * m * m;
    let m2 = m * m;

    let mut one_body = DMatrix::zeros(m, m);
    for p in 0..m {
        for s in 0..m {
            let mut contracted = 0.0;
            for q in 0..m {
                contracted += g_pqrs[p * m3 + q * m2 + q * m + s];
            }
            let h_val = h_pq[p * m + s];
            one_body[(p, s)] = Complex64::new(h_val - 0.5 * contracted, 0.0);
        }
    }

    let mut w = DMatrix::zeros(m2, m2);
    for p in 0..m {
        for r in 0..m {
            let pr = p * m + r;
            for q in 0..m {
                for s in 0..m {
                    let qs = q * m + s;
                    let val_prqs = g_pqrs[p * m3 + q * m2 + r * m + s];
                    let val_qspr = g_pqrs[q * m3 + p * m2 + s * m + r];
                    w[(pr, qs)] = 0.5 * (val_prqs + val_qspr);
                }
            }
        }
    }

    let eig = checked(w)?;
    let weights = eig.eigenvalues;
    let vectors = eig.eigenvectors;

    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by(|&a, &b| {
        weights[b]
            .abs()
            .partial_cmp(&weights[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let largest = if !order.is_empty() {
        weights[order[0]].abs()
    } else {
        0.0
    };
    let cutoff = tol * largest.max(1.0);
    let max_keep = max_leaves.unwrap_or(usize::MAX);

    let mut keep = Vec::new();
    let mut dropped = Vec::new();
    for (rank, &idx) in order.iter().enumerate() {
        let w_val = weights[idx];
        if w_val.abs() > cutoff && rank < max_keep {
            keep.push(idx);
        } else {
            dropped.push(w_val);
        }
    }
    let truncation_error = dropped.iter().map(|w| w * w).sum::<f64>().sqrt();

    let mut leaves = Vec::with_capacity(keep.len());
    for &idx in &keep {
        let weight = weights[idx];
        let mut matrix = DMatrix::zeros(m, m);
        for p in 0..m {
            for r in 0..m {
                let v1 = vectors[(p * m + r, idx)];
                let v2 = vectors[(r * m + p, idx)];
                matrix[(p, r)] = 0.5 * (v1 + v2);
            }
        }
        let leaf_eig = checked(matrix)?;
        let mut values = leaf_eig.eigenvalues;
        let mut any_nonzero = false;
        for k in 0..m {
            if values[k].abs() <= coefficient_tol {
                values[k] = 0.0;
            } else {
                any_nonzero = true;
            }
        }
        if !any_nonzero {
            continue;
        }
        leaves.push(FactorizationLeafData {
            weight,
            orbitals: leaf_eig.eigenvectors,
            coefficients: values,
        });
    }

    let mut ob_sym = DMatrix::zeros(m, m);
    for p in 0..m {
        for s in 0..m {
            ob_sym[(p, s)] = 0.5 * (one_body[(p, s)].re + one_body[(s, p)].re);
        }
    }
    let ob_eig = checked(ob_sym)?;
    let one_body_energies = ob_eig.eigenvalues;
    let one_body_orbitals = ob_eig.eigenvectors;

    Ok(DoubleFactorizationData {
        n_modes: m,
        constant,
        one_body,
        one_body_orbitals,
        one_body_energies,
        leaves,
        truncation_error,
    })
}

/// Spatial double factorization of a spin-free Hamiltonian.
pub fn spatial_double_factorization(
    h: &[f64],
    m: usize,
    eri_dense: Option<(&[f64], usize)>,
    eri_packed: Option<(&[f64], usize)>,
    factors: Option<(&[f64], usize, usize)>,
    coupling: Option<(&[f64], usize)>,
    tol: f64,
    coefficient_tol: f64,
    max_leaves: Option<usize>,
    constant: f64,
) -> Result<DoubleFactorizationData, String> {
    if h.len() != m * m {
        return Err(format!("expected h with {} elements, got {}", m * m, h.len()));
    }

    let mut exchange_like = DMatrix::zeros(m, m);
    if let Some((f_slice, k, f_m)) = factors {
        if f_m != m {
            return Err(format!("factor orbital dim {} does not match {}", f_m, m));
        }
        let mut k_mat = DMatrix::identity(k, k);
        if let Some((c_slice, _)) = coupling {
            for i in 0..k {
                for j in 0..k {
                    k_mat[(i, j)] = 0.5 * (c_slice[i * k + j] + c_slice[j * k + i]);
                }
            }
        }
        let f_mats: Vec<DMatrix<f64>> = (0..k)
            .map(|i| DMatrix::from_fn(m, m, |r, c| f_slice[i * m * m + r * m + c]))
            .collect();

        for i in 0..k {
            let mut c_i = DMatrix::zeros(m, m);
            for j in 0..k {
                let kij = k_mat[(i, j)];
                if kij != 0.0 {
                    c_i += kij * &f_mats[j];
                }
            }
            exchange_like += &f_mats[i] * c_i;
        }
    } else if let Some((data, p_m)) = eri_packed {
        if p_m != m {
            return Err(format!("packed eri orbital dim {} does not match {}", p_m, m));
        }
        for a in 0..m {
            for c in 0..=a {
                let u = if a >= c {
                    a * (a + 1) / 2 + c
                } else {
                    c * (c + 1) / 2 + a
                };
                for q in 0..m {
                    let v_c = if c >= q {
                        c * (c + 1) / 2 + q
                    } else {
                        q * (q + 1) / 2 + c
                    };
                    let idx_c = if u >= v_c {
                        u * (u + 1) / 2 + v_c
                    } else {
                        v_c * (v_c + 1) / 2 + u
                    };
                    exchange_like[(a, q)] += data[idx_c];
                    if a != c {
                        let v_a = if a >= q {
                            a * (a + 1) / 2 + q
                        } else {
                            q * (q + 1) / 2 + a
                        };
                        let idx_a = if u >= v_a {
                            u * (u + 1) / 2 + v_a
                        } else {
                            v_a * (v_a + 1) / 2 + u
                        };
                        exchange_like[(c, q)] += data[idx_a];
                    }
                }
            }
        }
    } else if let Some((g, g_m)) = eri_dense {
        if g_m != m {
            return Err(format!("dense eri orbital dim {} does not match {}", g_m, m));
        }
        let m3 = m * m * m;
        let m2 = m * m;
        for p in 0..m {
            for s in 0..m {
                let mut ex = 0.0;
                for q in 0..m {
                    ex += g[p * m3 + q * m2 + q * m + s];
                }
                exchange_like[(p, s)] = ex;
            }
        }
    } else {
        return Err("either dense eri, packed eri, or factors must be provided".to_string());
    }

    let mut spatial_one = DMatrix::zeros(m, m);
    for p in 0..m {
        for s in 0..m {
            spatial_one[(p, s)] = h[p * m + s] - 0.5 * exchange_like[(p, s)];
        }
    }

    let (mut weights, vectors) = pair_spectrum(eri_dense, eri_packed, factors, coupling)?;
    weights *= 2.0;

    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by(|&a, &b| {
        weights[b]
            .abs()
            .partial_cmp(&weights[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let largest = if !order.is_empty() {
        weights[order[0]].abs()
    } else {
        0.0
    };
    let cutoff = tol * largest.max(1.0);
    let max_keep = max_leaves.unwrap_or(usize::MAX);

    let mut keep = Vec::new();
    let mut dropped = Vec::new();
    for (rank, &idx) in order.iter().enumerate() {
        let w_val = weights[idx];
        if w_val.abs() > cutoff && rank < max_keep {
            keep.push(idx);
        } else {
            dropped.push(w_val);
        }
    }
    let truncation_error = dropped.iter().map(|w| w * w).sum::<f64>().sqrt();

    let n = 2 * m;
    let root_half = 0.5f64.sqrt();
    let mut leaves = Vec::with_capacity(keep.len());

    for &idx in &keep {
        let weight = weights[idx];
        let mut matrix = DMatrix::zeros(m, m);
        for p in 0..m {
            for r in 0..m {
                let v1 = vectors[(p * m + r, idx)];
                let v2 = vectors[(r * m + p, idx)];
                matrix[(p, r)] = 0.5 * (v1 + v2);
            }
        }
        let leaf_eig = checked(matrix)?;
        let values_spatial = leaf_eig.eigenvalues;
        let orbitals_spatial = leaf_eig.eigenvectors;

        let mut values = DVector::zeros(n);
        let mut any_nonzero = false;
        for i in 0..m {
            let v = values_spatial[i] * root_half;
            let v_clamped = if v.abs() > coefficient_tol {
                any_nonzero = true;
                v
            } else {
                0.0
            };
            values[i] = v_clamped;
            values[i + m] = v_clamped;
        }
        if !any_nonzero {
            continue;
        }

        let mut orbitals = DMatrix::zeros(n, n);
        for i in 0..m {
            for j in 0..m {
                let val = orbitals_spatial[(i, j)];
                orbitals[(i, j)] = val;
                orbitals[(i + m, j + m)] = val;
            }
        }
        leaves.push(FactorizationLeafData {
            weight,
            orbitals,
            coefficients: values,
        });
    }

    let spatial_sym = 0.5 * (&spatial_one + spatial_one.transpose());
    let ob_eig = checked(spatial_sym)?;
    let energies_spatial = ob_eig.eigenvalues;
    let rotation_spatial = ob_eig.eigenvectors;

    let mut one_body_energies = DVector::zeros(n);
    for i in 0..m {
        one_body_energies[i] = energies_spatial[i];
        one_body_energies[i + m] = energies_spatial[i];
    }

    let mut one_body_orbitals = DMatrix::zeros(n, n);
    let mut one_body = DMatrix::zeros(n, n);
    for i in 0..m {
        for j in 0..m {
            let rot = rotation_spatial[(i, j)];
            one_body_orbitals[(i, j)] = rot;
            one_body_orbitals[(i + m, j + m)] = rot;

            let sp1 = Complex64::new(spatial_one[(i, j)], 0.0);
            one_body[(i, j)] = sp1;
            one_body[(i + m, j + m)] = sp1;
        }
    }

    Ok(DoubleFactorizationData {
        n_modes: n,
        constant,
        one_body,
        one_body_orbitals,
        one_body_energies,
        leaves,
        truncation_error,
    })
}

/// Exact energy reconstruction from RDMs.
pub fn factorized_energy_rdm(
    constant: f64,
    one_body: &[Complex64],
    leaves_weights: &[f64],
    leaves_orbitals: &[Vec<f64>],
    leaves_coeffs: &[Vec<f64>],
    n_modes: usize,
    one_rdm: &[f64],
    two_rdm: &[f64],
) -> f64 {
    let mut energy = constant;
    for p in 0..n_modes {
        for q in 0..n_modes {
            energy += one_body[p * n_modes + q].re * one_rdm[p * n_modes + q];
        }
    }

    let n2 = n_modes * n_modes;
    let n3 = n_modes * n2;

    for leaf_idx in 0..leaves_weights.len() {
        let weight = leaves_weights[leaf_idx];
        let orbitals = &leaves_orbitals[leaf_idx];
        let coeffs = &leaves_coeffs[leaf_idx];

        // M_{pr} = sum_k U_{pk} * c_k * U_{rk}
        let mut m_mat = vec![0.0f64; n2];
        for p in 0..n_modes {
            for r in 0..n_modes {
                let mut sum = 0.0;
                for k in 0..n_modes {
                    sum += orbitals[p * n_modes + k] * coeffs[k] * orbitals[r * n_modes + k];
                }
                m_mat[p * n_modes + r] = sum;
            }
        }

        // M^2
        let mut m2_mat = vec![0.0f64; n2];
        for p in 0..n_modes {
            for s in 0..n_modes {
                let mut sum = 0.0;
                for r in 0..n_modes {
                    sum += m_mat[p * n_modes + r] * m_mat[r * n_modes + s];
                }
                m2_mat[p * n_modes + s] = sum;
            }
        }

        // Tr(M^2 @ gamma)
        let mut tr_m2_gamma = 0.0;
        for p in 0..n_modes {
            for s in 0..n_modes {
                tr_m2_gamma += m2_mat[p * n_modes + s] * one_rdm[p * n_modes + s];
            }
        }

        // sum_{p,r,q,s} M_{pr} * M_{qs} * Gamma_{pqrs}
        let mut contraction = 0.0;
        for p in 0..n_modes {
            for r in 0..n_modes {
                let m_pr = m_mat[p * n_modes + r];
                if m_pr == 0.0 {
                    continue;
                }
                for q in 0..n_modes {
                    for s in 0..n_modes {
                        let m_qs = m_mat[q * n_modes + s];
                        if m_qs == 0.0 {
                            continue;
                        }
                        let gamma2 = two_rdm[p * n3 + q * n2 + r * n_modes + s];
                        contraction += m_pr * m_qs * gamma2;
                    }
                }
            }
        }

        energy += 0.5 * weight * (contraction + tr_m2_gamma);
    }

    energy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorization_simple() {
        let m = 2;
        let h = vec![1.0, 0.0, 0.0, 2.0];
        let mut g = vec![0.0; 16];
        // g[0, 0, 0, 0] = 0.5
        g[0] = 0.5;
        // g[1, 1, 1, 1] = 0.5
        g[15] = 0.5;

        let res = double_factorization(&h, &g, m, 1e-8, 1e-10, None, 0.0).unwrap();
        assert_eq!(res.n_modes, 2);
        assert!(!res.leaves.is_empty());
    }

    #[test]
    fn test_spatial_factorization_simple() {
        let m = 2;
        let h = vec![1.0, 0.0, 0.0, 2.0];
        let mut g = vec![0.0; 16];
        g[0] = 0.5;
        g[15] = 0.5;

        let res = spatial_double_factorization(
            &h,
            m,
            Some((&g, m)),
            None,
            None,
            None,
            1e-8,
            1e-10,
            None,
            0.0,
        )
        .unwrap();
        assert_eq!(res.n_modes, 4);
        assert!(!res.leaves.is_empty());
    }
}

