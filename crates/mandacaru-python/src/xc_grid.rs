//! Fast native kernels for 3D exchange-correlation grid contractions.
use crate::arrays::{array, error, layout, Array};
use numpy::{Complex64, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyfunction]
pub fn xc_product_rule_gradient<'py>(
    py: Python<'py>,
    orbitals: Bound<'py, PyAny>,
    orbital_gradients: Bound<'py, PyAny>,
    density: Bound<'py, PyAny>,
) -> PyResult<Array<'py, f64>> {
    if let (Ok(orb_r), Ok(grad_r), Ok(den_r)) = (
        orbitals.extract::<PyReadonlyArrayDyn<'py, f64>>(),
        orbital_gradients.extract::<PyReadonlyArrayDyn<'py, f64>>(),
        density.extract::<PyReadonlyArrayDyn<'py, f64>>(),
    ) {
        layout(&orb_r)?;
        layout(&grad_r)?;
        layout(&den_r)?;

        let orb_s = orb_r.as_slice()?;
        let grad_s = grad_r.as_slice()?;
        let den_s = den_r.as_slice()?;

        let orb_shape = orb_r.shape();
        let grad_shape = grad_r.shape();
        let den_shape = den_r.shape();

        if orb_shape.len() != 2 || den_shape.len() != 2 || grad_shape.len() != 3 {
            return Err(PyValueError::new_err("unexpected array dimensions"));
        }
        let m = orb_shape[0];
        let n = orb_shape[1];
        if den_shape[0] != m || den_shape[1] != m || grad_shape[0] != 3 || grad_shape[1] != m || grad_shape[2] != n {
            return Err(PyValueError::new_err("incompatible array shapes"));
        }

        // W = D^T @ phi: (m, n)
        // W[q, i] = sum_p D[p, q] * phi[p, i]
        let mut w = vec![0.0f64; m * n];
        for p in 0..m {
            for q in 0..m {
                let d_pq = den_s[p * m + q];
                if d_pq != 0.0 {
                    let phi_row = &orb_s[p * n..(p + 1) * n];
                    let w_row = &mut w[q * n..(q + 1) * n];
                    for i in 0..n {
                        w_row[i] += d_pq * phi_row[i];
                    }
                }
            }
        }

        // out[alpha, i] = 2.0 * sum_q W[q, i] * dphi[alpha, q, i]
        let mut out = vec![0.0f64; 3 * n];
        for alpha in 0..3 {
            let out_slice = &mut out[alpha * n..(alpha + 1) * n];
            let grad_alpha = &grad_s[alpha * m * n..(alpha + 1) * m * n];
            for q in 0..m {
                let w_row = &w[q * n..(q + 1) * n];
                let dphi_row = &grad_alpha[q * n..(q + 1) * n];
                for i in 0..n {
                    out_slice[i] += w_row[i] * dphi_row[i];
                }
            }
            for i in 0..n {
                out_slice[i] *= 2.0;
            }
        }

        Ok(array(py, &[3, n], out)?)
    } else {
        // Complex branch
        let orb_c = orbitals.extract::<PyReadonlyArrayDyn<'py, Complex64>>()?;
        let grad_c = orbital_gradients.extract::<PyReadonlyArrayDyn<'py, Complex64>>()?;
        let den_c = density.extract::<PyReadonlyArrayDyn<'py, Complex64>>()?;

        layout(&orb_c)?;
        layout(&grad_c)?;
        layout(&den_c)?;

        let orb_s = orb_c.as_slice()?;
        let grad_s = grad_c.as_slice()?;
        let den_s = den_c.as_slice()?;

        let m = orb_c.shape()[0];
        let n = orb_c.shape()[1];

        // W[q, i] = sum_p D[p, q] * phi[p, i]
        let mut w = vec![Complex64::new(0.0, 0.0); m * n];
        for p in 0..m {
            for q in 0..m {
                let d_pq = den_s[p * m + q];
                let phi_row = &orb_s[p * n..(p + 1) * n];
                let w_row = &mut w[q * n..(q + 1) * n];
                for i in 0..n {
                    w_row[i] += d_pq * phi_row[i];
                }
            }
        }

        // out[alpha, i] = 2.0 * Re(sum_q W[q, i] * conj(dphi[alpha, q, i]))
        let mut out = vec![0.0f64; 3 * n];
        for alpha in 0..3 {
            let out_slice = &mut out[alpha * n..(alpha + 1) * n];
            let grad_alpha = &grad_s[alpha * m * n..(alpha + 1) * m * n];
            for q in 0..m {
                let w_row = &w[q * n..(q + 1) * n];
                let dphi_row = &grad_alpha[q * n..(q + 1) * n];
                for i in 0..n {
                    let w_val = w_row[i];
                    let d_val = dphi_row[i];
                    out_slice[i] += 2.0 * (w_val.re * d_val.re + w_val.im * d_val.im);
                }
            }
        }

        Ok(array(py, &[3, n], out)?)
    }
}

#[pyfunction]
pub fn xc_gradient_term_matrix<'py>(
    py: Python<'py>,
    orbitals: Bound<'py, PyAny>,
    orbital_gradients: Bound<'py, PyAny>,
    field: PyReadonlyArrayDyn<'py, f64>,
    dv: f64,
) -> PyResult<Bound<'py, PyAny>> {
    layout(&field)?;
    let field_s = field.as_slice()?;

    if let (Ok(orb_r), Ok(grad_r)) = (
        orbitals.extract::<PyReadonlyArrayDyn<'py, f64>>(),
        orbital_gradients.extract::<PyReadonlyArrayDyn<'py, f64>>(),
    ) {
        layout(&orb_r)?;
        layout(&grad_r)?;
        let orb_s = orb_r.as_slice()?;
        let grad_s = grad_r.as_slice()?;

        let m = orb_r.shape()[0];
        let n = orb_r.shape()[1];

        // half[p, q] = sum_alpha sum_i orb[p, i] * field[alpha, i] * grad[alpha, q, i]
        let mut half = vec![0.0f64; m * m];
        for alpha in 0..3 {
            let field_alpha = &field_s[alpha * n..(alpha + 1) * n];
            let grad_alpha = &grad_s[alpha * m * n..(alpha + 1) * m * n];
            for p in 0..m {
                let orb_p = &orb_s[p * n..(p + 1) * n];
                for q in 0..m {
                    let grad_q = &grad_alpha[q * n..(q + 1) * n];
                    let mut sum = 0.0f64;
                    for i in 0..n {
                        sum += orb_p[i] * field_alpha[i] * grad_q[i];
                    }
                    half[p * m + q] += sum;
                }
            }
        }

        // G = (half + half^T) * dV
        let mut g = vec![0.0f64; m * m];
        for p in 0..m {
            for q in 0..m {
                g[p * m + q] = (half[p * m + q] + half[q * m + p]) * dv;
            }
        }
        Ok(array(py, &[m, m], g)?.into_any())
    } else {
        let orb_c = orbitals.extract::<PyReadonlyArrayDyn<'py, Complex64>>()?;
        let grad_c = orbital_gradients.extract::<PyReadonlyArrayDyn<'py, Complex64>>()?;
        layout(&orb_c)?;
        layout(&grad_c)?;
        let orb_s = orb_c.as_slice()?;
        let grad_s = grad_c.as_slice()?;

        let m = orb_c.shape()[0];
        let n = orb_c.shape()[1];

        let mut half = vec![Complex64::new(0.0, 0.0); m * m];
        for alpha in 0..3 {
            let field_alpha = &field_s[alpha * n..(alpha + 1) * n];
            let grad_alpha = &grad_s[alpha * m * n..(alpha + 1) * m * n];
            for p in 0..m {
                let orb_p = &orb_s[p * n..(p + 1) * n];
                for q in 0..m {
                    let grad_q = &grad_alpha[q * n..(q + 1) * n];
                    let mut sum = Complex64::new(0.0, 0.0);
                    for i in 0..n {
                        let w = field_alpha[i];
                        let op = orb_p[i];
                        let gq = grad_q[i];
                        // conj(orb_p) * w * grad_q
                        let conj_op = Complex64::new(op.re, -op.im);
                        sum += conj_op * w * gq;
                    }
                    half[p * m + q] += sum;
                }
            }
        }

        let mut g = vec![Complex64::new(0.0, 0.0); m * m];
        for p in 0..m {
            for q in 0..m {
                let h_pq = half[p * m + q];
                let h_qp = half[q * m + p];
                let conj_h_qp = Complex64::new(h_qp.re, -h_qp.im);
                g[p * m + q] = (h_pq + conj_h_qp) * dv;
            }
        }
        Ok(array(py, &[m, m], g)?.into_any())
    }
}

#[pyfunction]
pub fn xc_gradient_term_derivative<'py>(
    py: Python<'py>,
    orbitals: Bound<'py, PyAny>,
    orbital_gradients: Bound<'py, PyAny>,
    moved: Bound<'py, PyAny>,
    moved_gradients: Bound<'py, PyAny>,
    field: PyReadonlyArrayDyn<'py, f64>,
    dv: f64,
) -> PyResult<Bound<'py, PyAny>> {
    layout(&field)?;
    let field_s = field.as_slice()?;

    if let (Ok(orb_r), Ok(grad_r), Ok(mov_r), Ok(mgrad_r)) = (
        orbitals.extract::<PyReadonlyArrayDyn<'py, f64>>(),
        orbital_gradients.extract::<PyReadonlyArrayDyn<'py, f64>>(),
        moved.extract::<PyReadonlyArrayDyn<'py, f64>>(),
        moved_gradients.extract::<PyReadonlyArrayDyn<'py, f64>>(),
    ) {
        layout(&orb_r)?;
        layout(&grad_r)?;
        layout(&mov_r)?;
        layout(&mgrad_r)?;

        let orb_s = orb_r.as_slice()?;
        let grad_s = grad_r.as_slice()?;
        let mov_s = mov_r.as_slice()?;
        let mgrad_s = mgrad_r.as_slice()?;

        let m = orb_r.shape()[0];
        let n = orb_r.shape()[1];

        // T1 = sum_alpha (moved * field_alpha) @ dphi_alpha^T
        // T2 = sum_alpha (orbitals * field_alpha) @ dmoved_alpha^T
        // out = (T1 + T2 + (T1 + T2)^T) * dv
        let mut t = vec![0.0f64; m * m];
        for alpha in 0..3 {
            let field_alpha = &field_s[alpha * n..(alpha + 1) * n];
            let grad_alpha = &grad_s[alpha * m * n..(alpha + 1) * m * n];
            let mgrad_alpha = &mgrad_s[alpha * m * n..(alpha + 1) * m * n];

            for p in 0..m {
                let mov_p = &mov_s[p * n..(p + 1) * n];
                let orb_p = &orb_s[p * n..(p + 1) * n];

                for q in 0..m {
                    let grad_q = &grad_alpha[q * n..(q + 1) * n];
                    let mgrad_q = &mgrad_alpha[q * n..(q + 1) * n];

                    let mut sum1 = 0.0f64;
                    let mut sum2 = 0.0f64;
                    for i in 0..n {
                        let w = field_alpha[i];
                        sum1 += mov_p[i] * w * grad_q[i];
                        sum2 += orb_p[i] * w * mgrad_q[i];
                    }
                    t[p * m + q] += sum1 + sum2;
                }
            }
        }

        let mut out = vec![0.0f64; m * m];
        for p in 0..m {
            for q in 0..m {
                out[p * m + q] = (t[p * m + q] + t[q * m + p]) * dv;
            }
        }
        Ok(array(py, &[m, m], out)?.into_any())
    } else {
        let orb_c = orbitals.extract::<PyReadonlyArrayDyn<'py, Complex64>>()?;
        let grad_c = orbital_gradients.extract::<PyReadonlyArrayDyn<'py, Complex64>>()?;
        let mov_c = moved.extract::<PyReadonlyArrayDyn<'py, Complex64>>()?;
        let mgrad_c = moved_gradients.extract::<PyReadonlyArrayDyn<'py, Complex64>>()?;

        layout(&orb_c)?;
        layout(&grad_c)?;
        layout(&mov_c)?;
        layout(&mgrad_c)?;

        let orb_s = orb_c.as_slice()?;
        let grad_s = grad_c.as_slice()?;
        let mov_s = mov_c.as_slice()?;
        let mgrad_s = mgrad_c.as_slice()?;

        let m = orb_c.shape()[0];
        let n = orb_c.shape()[1];

        let mut t = vec![Complex64::new(0.0, 0.0); m * m];
        for alpha in 0..3 {
            let field_alpha = &field_s[alpha * n..(alpha + 1) * n];
            let grad_alpha = &grad_s[alpha * m * n..(alpha + 1) * m * n];
            let mgrad_alpha = &mgrad_s[alpha * m * n..(alpha + 1) * m * n];

            for p in 0..m {
                let mov_p = &mov_s[p * n..(p + 1) * n];
                let orb_p = &orb_s[p * n..(p + 1) * n];

                for q in 0..m {
                    let grad_q = &grad_alpha[q * n..(q + 1) * n];
                    let mgrad_q = &mgrad_alpha[q * n..(q + 1) * n];

                    let mut sum1 = Complex64::new(0.0, 0.0);
                    let mut sum2 = Complex64::new(0.0, 0.0);
                    for i in 0..n {
                        let w = field_alpha[i];
                        let cmov_p = Complex64::new(mov_p[i].re, -mov_p[i].im);
                        let corb_p = Complex64::new(orb_p[i].re, -orb_p[i].im);
                        sum1 += cmov_p * w * grad_q[i];
                        sum2 += corb_p * w * mgrad_q[i];
                    }
                    t[p * m + q] += sum1 + sum2;
                }
            }
        }

        let mut out = vec![Complex64::new(0.0, 0.0); m * m];
        for p in 0..m {
            for q in 0..m {
                let t_pq = t[p * m + q];
                let t_qp = t[q * m + p];
                let conj_t_qp = Complex64::new(t_qp.re, -t_qp.im);
                out[p * m + q] = (t_pq + conj_t_qp) * dv;
            }
        }
        Ok(array(py, &[m, m], out)?.into_any())
    }
}

#[pyfunction]
pub fn xc_place_radial_densities<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
    y: PyReadonlyArrayDyn<'py, f64>,
    z: PyReadonlyArrayDyn<'py, f64>,
    tables: Vec<(Vec<f64>, Vec<f64>, [f64; 3])>,
) -> PyResult<Array<'py, f64>> {
    layout(&x)?;
    layout(&y)?;
    layout(&z)?;

    let x_s = x.as_slice()?;
    let y_s = y.as_slice()?;
    let z_s = z.as_slice()?;
    let n = x_s.len();

    let ctx = mandacaru_core::Context::new(None, 64 << 20).map_err(error)?;
    let mut total = vec![0.0f64; n];

    for (r, vals, center) in tables {
        if r.len() < 4 || r.len() != vals.len() {
            continue;
        }
        let spline = mandacaru_core::math::spline::Cubic::new(
            &ctx,
            ctx.copy_slice(&r).map_err(error)?,
            ctx.copy_slice(&vals).map_err(error)?,
        )
        .map_err(error)?;

        let edge = r[r.len() - 1];
        let r0 = r[0];
        let cx = center[0];
        let cy = center[1];
        let cz = center[2];

        for i in 0..n {
            let dx = x_s[i] - cx;
            let dy = y_s[i] - cy;
            let dz = z_s[i] - cz;
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d <= edge {
                let clamped = d.max(r0);
                total[i] += spline.value_slope(clamped).0;
            }
        }
    }

    Ok(array(py, &[n], total)?)
}

#[pyfunction]
pub fn xc_valence_terms(
    r: PyReadonlyArray1<'_, f64>,
    valence: PyReadonlyArray1<'_, f64>,
    core: PyReadonlyArray1<'_, f64>,
    e_xc: PyReadonlyArray1<'_, f64>,
    v_xc: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    layout(&r)?;
    layout(&valence)?;
    layout(&core)?;
    layout(&e_xc)?;
    layout(&v_xc)?;

    let r_s = r.as_slice()?;
    let v_s = valence.as_slice()?;
    let c_s = core.as_slice()?;
    let e_s = e_xc.as_slice()?;
    let pot_s = v_xc.as_slice()?;

    let k = r_s.len();
    if k < 2 {
        return Ok(0.0);
    }

    let four_pi = 4.0 * std::f64::consts::PI;
    let mut trap_sum = 0.0f64;

    for i in 0..k - 1 {
        let r_i = r_s[i];
        let r_next = r_s[i + 1];
        let dr = r_next - r_i;

        let shell_i = four_pi * r_i * r_i;
        let shell_next = four_pi * r_next * r_next;

        let f_i = (v_s[i] + c_s[i]) * e_s[i] * shell_i - v_s[i] * pot_s[i] * shell_i;
        let f_next = (v_s[i + 1] + c_s[i + 1]) * e_s[i + 1] * shell_next
            - v_s[i + 1] * pot_s[i + 1] * shell_next;

        trap_sum += 0.5 * (f_i + f_next) * dr;
    }

    Ok(trap_sum)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(xc_product_rule_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(xc_gradient_term_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(xc_gradient_term_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(xc_place_radial_densities, m)?)?;
    m.add_function(wrap_pyfunction!(xc_valence_terms, m)?)?;
    Ok(())
}
