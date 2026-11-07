//! Python bindings for packed real two-electron integrals (PackedERI).
use crate::arrays::{array, error, layout, Array};
use mandacaru_core::molecular::integrals::pair;
use numpy::{
    Complex64, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyReadwriteArray1,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;

#[pyfunction]
pub fn packed_pair_indices<'py>(py: Python<'py>, n: usize) -> PyResult<Array<'py, i64>> {
    let mut indices = Vec::with_capacity(n * n);
    for a in 0..n {
        for c in 0..n {
            indices.push(pair(a, c) as i64);
        }
    }
    array(py, &[n, n], indices)
}

#[pyfunction]
pub fn packed_offsets<'py>(py: Python<'py>, n_pairs: usize) -> PyResult<Array<'py, i64>> {
    let mut offsets = Vec::with_capacity(n_pairs);
    for u in 0..n_pairs {
        offsets.push((u * (u + 1) / 2) as i64);
    }
    array(py, &[n_pairs], offsets)
}

#[pyfunction]
pub fn packed_from_dense<'py>(
    py: Python<'py>,
    eri: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    layout(&eri)?;
    let shape = eri.shape();
    if shape.len() != 4 || shape[0] != shape[1] || shape[0] != shape[2] || shape[0] != shape[3] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "eri must have shape (n, n, n, n)",
        ));
    }
    let n = shape[0];
    let p = n * (n + 1) / 2;
    let total_packed = p * (p + 1) / 2;
    let slice = eri.as_slice()?;

    let mut data = vec![0.0_f64; total_packed];

    for a in 0..n {
        for c in 0..=a {
            let u = pair(a, c);
            for b in 0..n {
                for d in 0..=b {
                    let v = pair(b, d);
                    if v <= u {
                        let idx = pair(u, v);
                        data[idx] = slice[((a * n + b) * n + c) * n + d];
                    }
                }
            }
        }
    }

    array(py, &[total_packed], data)
}

#[pyfunction]
pub fn packed_dense<'py>(
    py: Python<'py>,
    n: usize,
    data: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    layout(&data)?;
    let p = n * (n + 1) / 2;
    let total_packed = p * (p + 1) / 2;
    let slice = data.as_slice()?;
    if slice.len() != total_packed {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "packed data length ({}) does not match expected ({})",
            slice.len(),
            total_packed
        )));
    }

    let mut dense = vec![0.0_f64; n * n * n * n];
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                let u = pair(a, c);
                for d in 0..n {
                    let v = pair(b, d);
                    dense[((a * n + b) * n + c) * n + d] = slice[pair(u, v)];
                }
            }
        }
    }

    array(py, &[n, n, n, n], dense)
}

#[pyfunction]
pub fn packed_pair_matrix<'py>(
    py: Python<'py>,
    n: usize,
    data: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    layout(&data)?;
    let p = n * (n + 1) / 2;
    let slice = data.as_slice()?;
    let mut mat = vec![0.0_f64; p * p];
    for u in 0..p {
        for v in 0..p {
            mat[u * p + v] = slice[pair(u, v)];
        }
    }
    array(py, &[p, p], mat)
}

#[pyfunction]
pub fn packed_rows<'py>(
    py: Python<'py>,
    n_pairs: usize,
    data: PyReadonlyArray1<'py, f64>,
    index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Array<'py, f64>> {
    layout(&data)?;
    layout(&index)?;
    let idx_slice = index.as_slice()?;
    let data_slice = data.as_slice()?;
    let k = idx_slice.len();

    let mut rows = Vec::with_capacity(k * n_pairs);
    for &u_idx in idx_slice {
        let u = u_idx as usize;
        for v in 0..n_pairs {
            rows.push(data_slice[pair(u, v)]);
        }
    }
    array(py, &[k, n_pairs], rows)
}

#[pyfunction]
pub fn packed_coulomb<'py>(
    py: Python<'py>,
    n: usize,
    data: PyReadonlyArray1<'py, f64>,
    d_mat: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    layout(&data)?;
    let p = n * (n + 1) / 2;
    let data_slice = data.as_slice()?;

    if let Ok(d_arr) = d_mat.extract::<PyReadonlyArray2<'py, f64>>() {
        layout(&d_arr)?;
        let d_slice = d_arr.as_slice()?;

        // d_v = D[s, r] + D[r, s] (or D[r, r] on diagonal)
        let mut d_vec = vec![0.0_f64; p];
        for r in 0..n {
            for s in 0..=r {
                let v = pair(r, s);
                if r == s {
                    d_vec[v] = d_slice[r * n + r];
                } else {
                    d_vec[v] = d_slice[s * n + r] + d_slice[r * n + s];
                }
            }
        }

        // y = W * d_vec using packed symmetric matrix-vector product
        let mut y = vec![0.0_f64; p];
        for u in 0..p {
            let row_start = u * (u + 1) / 2;
            for v in 0..u {
                let w = data_slice[row_start + v];
                y[u] += w * d_vec[v];
                y[v] += w * d_vec[u];
            }
            y[u] += data_slice[row_start + u] * d_vec[u];
        }

        // J[p, q] = y[pair(p, q)]
        let mut j_mat = vec![0.0_f64; n * n];
        for r in 0..n {
            for s in 0..n {
                j_mat[r * n + s] = y[pair(r, s)];
            }
        }
        let out = array(py, &[n, n], j_mat)?;
        Ok(out.into_any())
    } else if let Ok(d_arr) = d_mat.extract::<PyReadonlyArray2<'py, Complex64>>() {
        layout(&d_arr)?;
        let d_slice = d_arr.as_slice()?;

        let mut d_re = vec![0.0_f64; p];
        let mut d_im = vec![0.0_f64; p];
        for r in 0..n {
            for s in 0..=r {
                let v = pair(r, s);
                if r == s {
                    d_re[v] = d_slice[r * n + r].re;
                    d_im[v] = d_slice[r * n + r].im;
                } else {
                    d_re[v] = d_slice[s * n + r].re + d_slice[r * n + s].re;
                    d_im[v] = d_slice[s * n + r].im + d_slice[r * n + s].im;
                }
            }
        }

        let mut y_re = vec![0.0_f64; p];
        let mut y_im = vec![0.0_f64; p];
        for u in 0..p {
            let row_start = u * (u + 1) / 2;
            for v in 0..u {
                let w = data_slice[row_start + v];
                y_re[u] += w * d_re[v];
                y_re[v] += w * d_re[u];
                y_im[u] += w * d_im[v];
                y_im[v] += w * d_im[u];
            }
            y_re[u] += data_slice[row_start + u] * d_re[u];
            y_im[u] += data_slice[row_start + u] * d_im[u];
        }

        let mut j_mat = vec![Complex64::new(0.0, 0.0); n * n];
        for r in 0..n {
            for s in 0..n {
                let v = pair(r, s);
                j_mat[r * n + s] = Complex64::new(y_re[v], y_im[v]);
            }
        }
        let out = array(py, &[n, n], j_mat)?;
        Ok(out.into_any())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            "D must be 2D float or complex array",
        ))
    }
}

#[pyfunction]
pub fn packed_exchange<'py>(
    py: Python<'py>,
    n: usize,
    data: PyReadonlyArray1<'py, f64>,
    d_mat: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    layout(&data)?;
    let data_slice = data.as_slice()?;

    if let Ok(d_arr) = d_mat.extract::<PyReadonlyArray2<'py, f64>>() {
        layout(&d_arr)?;
        let d_slice = d_arr.as_slice()?;
        let mut k_mat = vec![0.0_f64; n * n];

        for a in 0..n {
            for c in 0..=a {
                let u = pair(a, c);
                for r in 0..n {
                    let d_cr = d_slice[c * n + r];
                    let d_ar = d_slice[a * n + r];
                    for q in 0..n {
                        let v = pair(r, q);
                        let w = data_slice[pair(u, v)];
                        k_mat[a * n + q] += d_cr * w;
                        if a != c {
                            k_mat[c * n + q] += d_ar * w;
                        }
                    }
                }
            }
        }
        let out = array(py, &[n, n], k_mat)?;
        Ok(out.into_any())
    } else if let Ok(d_arr) = d_mat.extract::<PyReadonlyArray2<'py, Complex64>>() {
        layout(&d_arr)?;
        let d_slice = d_arr.as_slice()?;
        let mut k_re = vec![0.0_f64; n * n];
        let mut k_im = vec![0.0_f64; n * n];

        for a in 0..n {
            for c in 0..=a {
                let u = pair(a, c);
                for r in 0..n {
                    let d_cr = d_slice[c * n + r];
                    let d_ar = d_slice[a * n + r];
                    for q in 0..n {
                        let v = pair(r, q);
                        let w = data_slice[pair(u, v)];
                        k_re[a * n + q] += d_cr.re * w;
                        k_im[a * n + q] += d_cr.im * w;
                        if a != c {
                            k_re[c * n + q] += d_ar.re * w;
                            k_im[c * n + q] += d_ar.im * w;
                        }
                    }
                }
            }
        }
        let mut k_mat = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n * n {
            k_mat[i] = Complex64::new(k_re[i], k_im[i]);
        }
        let out = array(py, &[n, n], k_mat)?;
        Ok(out.into_any())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            "D must be 2D float or complex array",
        ))
    }
}

#[pyfunction]
pub fn packed_contract<'py>(
    py: Python<'py>,
    n: usize,
    data: PyReadonlyArray1<'py, f64>,
    g: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    layout(&data)?;
    let data_slice = data.as_slice()?;

    if let Ok(g_arr) = g.extract::<PyReadonlyArrayDyn<'py, f64>>() {
        layout(&g_arr)?;
        let g_slice = g_arr.as_slice()?;
        let mut sum = 0.0_f64;
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    let u = pair(p, r);
                    for s in 0..n {
                        let v = pair(q, s);
                        let w = data_slice[pair(u, v)];
                        sum += g_slice[((p * n + q) * n + r) * n + s] * w;
                    }
                }
            }
        }
        Ok(sum.into_pyobject(py)?.into_any())
    } else if let Ok(g_arr) = g.extract::<PyReadonlyArrayDyn<'py, Complex64>>() {
        layout(&g_arr)?;
        let g_slice = g_arr.as_slice()?;
        let mut sum_re = 0.0_f64;
        let mut sum_im = 0.0_f64;
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    let u = pair(p, r);
                    for s in 0..n {
                        let v = pair(q, s);
                        let w = data_slice[pair(u, v)];
                        let c = g_slice[((p * n + q) * n + r) * n + s];
                        sum_re += c.re * w;
                        sum_im += c.im * w;
                    }
                }
            }
        }
        Ok(pyo3::types::PyComplex::from_doubles(py, sum_re, sum_im).into_any())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            "G must be float or complex array",
        ))
    }
}

#[pyfunction]
pub fn packed_slab<'py>(
    py: Python<'py>,
    n: usize,
    data: PyReadonlyArray1<'py, f64>,
    p: usize,
) -> PyResult<Array<'py, f64>> {
    layout(&data)?;
    let data_slice = data.as_slice()?;
    let mut slab = vec![0.0_f64; n * n * n];
    for q in 0..n {
        for r in 0..n {
            let u = pair(p, r);
            for s in 0..n {
                let v = pair(q, s);
                slab[(q * n + r) * n + s] = data_slice[pair(u, v)];
            }
        }
    }
    array(py, &[n, n, n], slab)
}

#[pyfunction]
pub fn packed_subset<'py>(
    py: Python<'py>,
    _n: usize,
    data: PyReadonlyArray1<'py, f64>,
    index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Array<'py, f64>> {
    layout(&data)?;
    layout(&index)?;
    let idx_slice = index.as_slice()?;
    let data_slice = data.as_slice()?;
    let m = idx_slice.len();
    let p_m = m * (m + 1) / 2;
    let total_m = p_m * (p_m + 1) / 2;

    let mut out = vec![0.0_f64; total_m];

    for i in 0..m {
        let orig_i = idx_slice[i] as usize;
        for k in 0..=i {
            let orig_k = idx_slice[k] as usize;
            let u_m = pair(i, k);
            let u = pair(orig_i, orig_k);

            for j in 0..m {
                let orig_j = idx_slice[j] as usize;
                for l in 0..=j {
                    let orig_l = idx_slice[l] as usize;
                    let v_m = pair(j, l);
                    if v_m <= u_m {
                        let v = pair(orig_j, orig_l);
                        out[pair(u_m, v_m)] = data_slice[pair(u, v)];
                    }
                }
            }
        }
    }

    array(py, &[total_m], out)
}

#[pyfunction]
pub fn packed_transform<'py>(
    py: Python<'py>,
    n: usize,
    data: PyReadonlyArray1<'py, f64>,
    c: PyReadonlyArray2<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    layout(&data)?;
    layout(&c)?;
    let c_shape = c.shape();
    if c_shape[0] != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "C rows must equal n",
        ));
    }
    let m = c_shape[1];
    let p_n = n * (n + 1) / 2;
    let p_m = m * (m + 1) / 2;
    let total_m = p_m * (p_m + 1) / 2;

    let data_slice = data.as_slice()?;
    let c_slice = c.as_slice()?;

    // Step 1: H[(i, k), v] = (C^T A^{(v)} C)_{ik} for i >= k
    // H has shape (P_m, P_n)
    let mut h = vec![0.0_f64; p_m * p_n];
    let mut a_mat = vec![0.0_f64; n * n];
    let mut temp = vec![0.0_f64; n * m];

    for v in 0..p_n {
        // Unpack column v of W into (n, n) symmetric matrix
        for a in 0..n {
            for b in 0..=a {
                let val = data_slice[pair(pair(a, b), v)];
                a_mat[a * n + b] = val;
                a_mat[b * n + a] = val;
            }
        }

        // temp[a, k] = sum_b A[a, b] * C[b, k]
        for a in 0..n {
            for k in 0..m {
                let mut sum = 0.0_f64;
                for b in 0..n {
                    sum += a_mat[a * n + b] * c_slice[b * m + k];
                }
                temp[a * m + k] = sum;
            }
        }

        // H[u_m, v] = sum_a C[a, i] * temp[a, k] for i >= k
        for i in 0..m {
            for k in 0..=i {
                let u_m = pair(i, k);
                let mut sum = 0.0_f64;
                for a in 0..n {
                    sum += c_slice[a * m + i] * temp[a * m + k];
                }
                h[u_m * p_n + v] = sum;
            }
        }
    }

    // Step 2: out[pair(u_m, v_m)] = (C^T B^{(u_m)} C)_{jl} for v_m <= u_m
    let mut out = vec![0.0_f64; total_m];
    let mut b_mat = vec![0.0_f64; n * n];

    for u_m in 0..p_m {
        let h_row = &h[u_m * p_n..(u_m + 1) * p_n];
        for b in 0..n {
            for d in 0..=b {
                let val = h_row[pair(b, d)];
                b_mat[b * n + d] = val;
                b_mat[d * n + b] = val;
            }
        }

        // temp[b, l] = sum_d B[b, d] * C[d, l]
        for b in 0..n {
            for l in 0..m {
                let mut sum = 0.0_f64;
                for d in 0..n {
                    sum += b_mat[b * n + d] * c_slice[d * m + l];
                }
                temp[b * m + l] = sum;
            }
        }

        // out[pair(u_m, v_m)] = sum_b C[b, j] * temp[b, l] for v_m <= u_m
        for j in 0..m {
            for l in 0..=j {
                let v_m = pair(j, l);
                if v_m <= u_m {
                    let mut sum = 0.0_f64;
                    for b in 0..n {
                        sum += c_slice[b * m + j] * temp[b * m + l];
                    }
                    out[pair(u_m, v_m)] = sum;
                }
            }
        }
    }

    array(py, &[total_m], out)
}

#[pyfunction]
pub fn packed_block<'py>(
    py: Python<'py>,
    n: usize,
    data: PyReadonlyArray1<'py, f64>,
    c1: PyReadonlyArray2<'py, f64>,
    c2: PyReadonlyArray2<'py, f64>,
    c3: PyReadonlyArray2<'py, f64>,
    c4: PyReadonlyArray2<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    layout(&data)?;
    layout(&c1)?;
    layout(&c2)?;
    layout(&c3)?;
    layout(&c4)?;

    let n1 = c1.shape()[1];
    let n2 = c2.shape()[1];
    let n3 = c3.shape()[1];
    let n4 = c4.shape()[1];
    let p_n = n * (n + 1) / 2;

    let data_slice = data.as_slice()?;
    let c1_slice = c1.as_slice()?;
    let c2_slice = c2.as_slice()?;
    let c3_slice = c3.as_slice()?;
    let c4_slice = c4.as_slice()?;

    // Step 1: H[(i, k), v] = (C1^T A^{(v)} C3)_{ik}
    // H has shape (n1 * n3, P_n)
    let mut h = vec![0.0_f64; n1 * n3 * p_n];
    let mut a_mat = vec![0.0_f64; n * n];
    let mut temp1 = vec![0.0_f64; n * n3];

    for v in 0..p_n {
        for a in 0..n {
            for b in 0..=a {
                let val = data_slice[pair(pair(a, b), v)];
                a_mat[a * n + b] = val;
                a_mat[b * n + a] = val;
            }
        }
        for a in 0..n {
            for k in 0..n3 {
                let mut sum = 0.0_f64;
                for b in 0..n {
                    sum += a_mat[a * n + b] * c3_slice[b * n3 + k];
                }
                temp1[a * n3 + k] = sum;
            }
        }
        for i in 0..n1 {
            for k in 0..n3 {
                let mut sum = 0.0_f64;
                for a in 0..n {
                    sum += c1_slice[a * n1 + i] * temp1[a * n3 + k];
                }
                h[(i * n3 + k) * p_n + v] = sum;
            }
        }
    }

    // Step 2: out[i, j, k, l] = (C2^T B^{(i, k)} C4)_{jl}
    let mut out = vec![0.0_f64; n1 * n2 * n3 * n4];
    let mut b_mat = vec![0.0_f64; n * n];
    let mut temp2 = vec![0.0_f64; n * n4];

    for i in 0..n1 {
        for k in 0..n3 {
            let h_row = &h[(i * n3 + k) * p_n..(i * n3 + k + 1) * p_n];
            for b in 0..n {
                for d in 0..=b {
                    let val = h_row[pair(b, d)];
                    b_mat[b * n + d] = val;
                    b_mat[d * n + b] = val;
                }
            }
            for b in 0..n {
                for l in 0..n4 {
                    let mut sum = 0.0_f64;
                    for d in 0..n {
                        sum += b_mat[b * n + d] * c4_slice[d * n4 + l];
                    }
                    temp2[b * n4 + l] = sum;
                }
            }
            for j in 0..n2 {
                for l in 0..n4 {
                    let mut sum = 0.0_f64;
                    for b in 0..n {
                        sum += c2_slice[b * n2 + j] * temp2[b * n4 + l];
                    }
                    out[((i * n2 + j) * n3 + k) * n4 + l] = sum;
                }
            }
        }
    }

    array(py, &[n1, n2, n3, n4], out)
}

#[pyfunction]
pub fn packed_add_augmentation<'py>(
    n: usize,
    mut data: PyReadwriteArray1<'py, f64>,
    factors: PyReadonlyArrayDyn<'py, f64>,
    coupling: PyReadonlyArray2<'py, f64>,
    scale: f64,
) -> PyResult<()> {
    let p = n * (n + 1) / 2;
    let k = coupling.shape()[0];
    if coupling.shape() != [k, k] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coupling must be (k, k)",
        ));
    }
    if factors.shape() != [k, n, n] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "factors must be (k, n, n)",
        ));
    }

    let f_slice = factors.as_slice()?;
    let c_slice = coupling.as_slice()?;
    let data_slice = data.as_slice_mut()?;

    // f: (k, P)
    let mut f = vec![0.0_f64; k * p];
    for i in 0..k {
        let f_i = &f_slice[i * n * n..(i + 1) * n * n];
        for a in 0..n {
            for c in 0..=a {
                f[i * p + pair(a, c)] = f_i[a * n + c];
            }
        }
    }

    // right = scale * coupling @ f: (k, P)
    let mut right = vec![0.0_f64; k * p];
    for i in 0..k {
        for v in 0..p {
            let mut sum = 0.0_f64;
            for j in 0..k {
                sum += c_slice[i * k + j] * f[j * p + v];
            }
            right[i * p + v] = scale * sum;
        }
    }

    // Add block = f[:, u]^T @ right[:, v] to data[pair(u, v)]
    for u in 0..p {
        for v in 0..=u {
            let mut sum = 0.0_f64;
            for i in 0..k {
                sum += f[i * p + u] * right[i * p + v];
            }
            data_slice[pair(u, v)] += sum;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// XC Radial helpers
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn radial_cubic_spline_evaluate<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    query: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    layout(&r)?;
    layout(&values)?;
    layout(&query)?;

    let r_slice = r.as_slice()?;
    let val_slice = values.as_slice()?;
    let q_slice = query.as_slice()?;
    let q_shape = query.shape();

    if r_slice.len() < 4 || r_slice.len() != val_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r and values must have matching length >= 4",
        ));
    }

    let ctx = mandacaru_core::Context::new(None, 64 << 20).map_err(error)?;
    let spline = mandacaru_core::math::spline::Cubic::new(
        &ctx,
        ctx.copy_slice(r_slice).map_err(error)?,
        ctx.copy_slice(val_slice).map_err(error)?,
    )
    .map_err(error)?;

    let edge = r_slice[r_slice.len() - 1];
    let r0 = r_slice[0];

    let mut out = Vec::with_capacity(q_slice.len());
    for &rad in q_slice {
        if rad <= edge {
            let clamped = rad.max(r0);
            out.push(spline.value_slope(clamped).0);
        } else {
            out.push(0.0);
        }
    }

    array(py, q_shape, out)
}

#[pyfunction]
pub fn core_tau_radial_table<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    core: PyReadonlyArray1<'py, f64>,
    floor_frac: f64,
) -> PyResult<Array<'py, f64>> {
    layout(&r)?;
    layout(&core)?;

    let r_slice = r.as_slice()?;
    let core_slice = core.as_slice()?;
    let n = r_slice.len();
    if n < 4 || n != core_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r and core must have matching length >= 4",
        ));
    }

    let ctx = mandacaru_core::Context::new(None, 64 << 20).map_err(error)?;
    let spline = mandacaru_core::math::spline::Cubic::new(
        &ctx,
        ctx.copy_slice(r_slice).map_err(error)?,
        ctx.copy_slice(core_slice).map_err(error)?,
    )
    .map_err(error)?;

    let max_abs_core = core_slice.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let floor = floor_frac * max_abs_core;

    let mut tau = Vec::with_capacity(n);
    for i in 0..n {
        let c = core_slice[i];
        if c > floor {
            let slope = spline.value_slope(r_slice[i]).1;
            tau.push(slope * slope / (8.0 * c));
        } else {
            tau.push(0.0);
        }
    }

    array(py, &[n], tau)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(packed_pair_indices, m)?)?;
    m.add_function(wrap_pyfunction!(packed_offsets, m)?)?;
    m.add_function(wrap_pyfunction!(packed_from_dense, m)?)?;
    m.add_function(wrap_pyfunction!(packed_dense, m)?)?;
    m.add_function(wrap_pyfunction!(packed_pair_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(packed_rows, m)?)?;
    m.add_function(wrap_pyfunction!(packed_coulomb, m)?)?;
    m.add_function(wrap_pyfunction!(packed_exchange, m)?)?;
    m.add_function(wrap_pyfunction!(packed_contract, m)?)?;
    m.add_function(wrap_pyfunction!(packed_slab, m)?)?;
    m.add_function(wrap_pyfunction!(packed_subset, m)?)?;
    m.add_function(wrap_pyfunction!(packed_transform, m)?)?;
    m.add_function(wrap_pyfunction!(packed_block, m)?)?;
    m.add_function(wrap_pyfunction!(packed_add_augmentation, m)?)?;

    m.add_function(wrap_pyfunction!(radial_cubic_spline_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(core_tau_radial_table, m)?)?;
    Ok(())
}
