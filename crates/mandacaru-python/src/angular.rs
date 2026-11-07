//! Python bindings for spherical harmonics and angular coordinate machinery.
use crate::arrays::{array, layout, Array};
use numpy::{Complex64, PyReadonlyArray1};
use pyo3::prelude::*;
use std::f64::consts::PI;

pub const NODE_OFFSET: f64 = 1e-12;
const R_EPS: f64 = 1e-15;

#[pyfunction]
pub fn spherical_coords<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    center: [f64; 3],
) -> PyResult<(Array<'py, f64>, Array<'py, f64>, Array<'py, f64>)> {
    layout(&x)?;
    layout(&y)?;
    layout(&z)?;
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    let zs = z.as_slice()?;
    let n = xs.len();
    if ys.len() != n || zs.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x, y, z must have the same length",
        ));
    }
    let mut r_out = Vec::with_capacity(n);
    let mut theta_out = Vec::with_capacity(n);
    let mut phi_out = Vec::with_capacity(n);
    for i in 0..n {
        let xr = xs[i] - center[0];
        let yr = ys[i] - center[1];
        let zr = zs[i] - center[2];
        let r = (xr * xr + yr * yr + zr * zr).sqrt();
        let r_safe = if r < R_EPS { R_EPS } else { r };
        let z_ratio = (zr / r_safe).clamp(-1.0, 1.0);
        let theta = z_ratio.acos();
        let phi = yr.atan2(xr);
        r_out.push(r);
        theta_out.push(theta);
        phi_out.push(phi);
    }
    Ok((
        array(py, &[n], r_out)?,
        array(py, &[n], theta_out)?,
        array(py, &[n], phi_out)?,
    ))
}

#[pyfunction]
pub fn real_harmonic_transform<'py>(
    py: Python<'py>,
    l: usize,
) -> PyResult<Array<'py, Complex64>> {
    let dim = 2 * l + 1;
    let mut u = vec![Complex64::default(); dim * dim];
    let l_i = l as i32;
    u[l * dim + l] = Complex64::new(1.0, 0.0);
    let root = 1.0 / 2.0_f64.sqrt();
    for m in 1..=l as i32 {
        let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
        let row_p = (l_i + m) as usize;
        let row_m = (l_i - m) as usize;
        let col_p = (l_i + m) as usize;
        let col_m = (l_i - m) as usize;
        u[row_p * dim + col_p] = Complex64::new(sign * root, 0.0);
        u[row_p * dim + col_m] = Complex64::new(root, 0.0);
        u[row_m * dim + col_m] = Complex64::new(0.0, root);
        u[row_m * dim + col_p] = Complex64::new(0.0, -sign * root);
    }
    array(py, &[dim, dim], u)
}

fn evaluate_spherical_harmonic_scalar(
    l: usize,
    m: i32,
    theta: f64,
    phi: f64,
    real: bool,
) -> (f64, Complex64) {
    let order = m.unsigned_abs() as usize;
    if order > l {
        return (0.0, Complex64::default());
    }
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let mut legendre = 1.0 / (4.0 * PI).sqrt();
    for k in 1..=order {
        legendre *= -((2 * k + 1) as f64 / (2 * k) as f64).sqrt() * sin_t;
    }
    if l > order {
        let mut previous = legendre;
        legendre *= ((2 * order + 3) as f64).sqrt() * cos_t;
        for degree in order + 2..=l {
            let d = degree as f64;
            let o = order as f64;
            let a = ((4.0 * d * d - 1.0) / (d * d - o * o)).sqrt();
            let b = (((d - 1.0).powi(2) - o * o) / (4.0 * (d - 1.0).powi(2) - 1.0)).sqrt();
            let next = a * (cos_t * legendre - b * previous);
            previous = legendre;
            legendre = next;
        }
    }
    if real {
        let val = if m == 0 {
            legendre
        } else {
            let trig = if m > 0 {
                (order as f64 * phi).cos()
            } else {
                (order as f64 * phi).sin()
            };
            let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
            2.0_f64.sqrt() * sign * legendre * trig
        };
        (val, Complex64::default())
    } else {
        let exp_term = Complex64::from_polar(1.0, order as f64 * phi);
        let mut value = exp_term * legendre;
        if m < 0 {
            let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
            value = value.conj() * sign;
        }
        (0.0, value)
    }
}

#[pyfunction]
pub fn spherical_harmonic_evaluate<'py>(
    py: Python<'py>,
    l: usize,
    m: i32,
    theta: PyReadonlyArray1<'py, f64>,
    phi: PyReadonlyArray1<'py, f64>,
    real: bool,
) -> PyResult<Bound<'py, PyAny>> {
    layout(&theta)?;
    layout(&phi)?;
    let ts = theta.as_slice()?;
    let ps = phi.as_slice()?;
    let n = ts.len();
    if ps.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "theta and phi must have the same length",
        ));
    }
    if real {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let (v, _) = evaluate_spherical_harmonic_scalar(l, m, ts[i], ps[i], true);
            out.push(v);
        }
        Ok(array(py, &[n], out)?.into_any())
    } else {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let (_, v) = evaluate_spherical_harmonic_scalar(l, m, ts[i], ps[i], false);
            out.push(v);
        }
        Ok(array(py, &[n], out)?.into_any())
    }
}

fn evaluate_spherical_harmonic_gradient_scalar(
    l: usize,
    m: i32,
    x: f64,
    y: f64,
    z: f64,
    real: bool,
) -> (f64, [f64; 3], Complex64, [Complex64; 3]) {
    let order = m.unsigned_abs() as usize;
    let r = (x * x + y * y + z * z).sqrt();
    let u = z / r;
    let w = Complex64::new(x / r, y / r);

    let mut q0 = 1.0 / (4.0 * PI).sqrt();
    for k in 1..=order {
        q0 *= -((2 * k + 1) as f64 / (2 * k) as f64).sqrt();
    }
    let d0 = 0.0;
    let (q, dq) = if l == order {
        (q0, d0)
    } else {
        let mut q0_cur = q0;
        let mut d0_cur = d0;
        let mut q1 = ((2 * order + 3) as f64).sqrt() * u * q0;
        let mut d1 = ((2 * order + 3) as f64).sqrt() * q0;
        for degree in order + 2..=l {
            let d = degree as f64;
            let o = order as f64;
            let a = ((4.0 * d * d - 1.0) / (d * d - o * o)).sqrt();
            let b = (((d - 1.0).powi(2) - o * o) / (4.0 * (d - 1.0).powi(2) - 1.0)).sqrt();
            let next_q = a * (u * q1 - b * q0_cur);
            let next_d = a * (q1 + u * d1 - b * d0_cur);
            q0_cur = q1;
            d0_cur = d1;
            q1 = next_q;
            d1 = next_d;
        }
        (q1, d1)
    };

    let wm = w.powu(order as u32);
    let wm1 = if order > 0 {
        w.powu((order - 1) as u32)
    } else {
        Complex64::default()
    };
    let mut value = wm * q;

    let coords = [x, y, z];
    let e_axes = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, 0.0),
    ];
    let mut grad = [Complex64::default(); 3];
    for axis in 0..3 {
        let xa = coords[axis];
        let du = (((if axis == 2 { 1.0 } else { 0.0 }) - u * xa / r)) / r;
        let dw = (e_axes[axis] - w * (xa / r)) / r;
        grad[axis] = wm * (dq * du) + (wm1 * dw) * (q * order as f64);
    }

    if real {
        let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
        let scale = 2.0_f64.sqrt() * sign;
        if m == 0 {
            (
                value.re,
                [grad[0].re, grad[1].re, grad[2].re],
                Complex64::default(),
                [Complex64::default(); 3],
            )
        } else if m > 0 {
            (
                scale * value.re,
                [scale * grad[0].re, scale * grad[1].re, scale * grad[2].re],
                Complex64::default(),
                [Complex64::default(); 3],
            )
        } else {
            (
                scale * value.im,
                [scale * grad[0].im, scale * grad[1].im, scale * grad[2].im],
                Complex64::default(),
                [Complex64::default(); 3],
            )
        }
    } else {
        if m < 0 {
            let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
            value = value.conj() * sign;
            for g in grad.iter_mut() {
                *g = g.conj() * sign;
            }
        }
        (0.0, [0.0; 3], value, grad)
    }
}

#[pyfunction]
pub fn spherical_harmonic_gradient_evaluate<'py>(
    py: Python<'py>,
    l: usize,
    m: i32,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    real: bool,
) -> PyResult<(
    Bound<'py, PyAny>,
    (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>),
)> {
    layout(&x)?;
    layout(&y)?;
    layout(&z)?;
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    let zs = z.as_slice()?;
    let n = xs.len();
    if ys.len() != n || zs.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x, y, z must have the same length",
        ));
    }
    if real {
        let mut val_out = Vec::with_capacity(n);
        let mut gx = Vec::with_capacity(n);
        let mut gy = Vec::with_capacity(n);
        let mut gz = Vec::with_capacity(n);
        for i in 0..n {
            let (v, g, _, _) =
                evaluate_spherical_harmonic_gradient_scalar(l, m, xs[i], ys[i], zs[i], true);
            val_out.push(v);
            gx.push(g[0]);
            gy.push(g[1]);
            gz.push(g[2]);
        }
        Ok((
            array(py, &[n], val_out)?.into_any(),
            (
                array(py, &[n], gx)?.into_any(),
                array(py, &[n], gy)?.into_any(),
                array(py, &[n], gz)?.into_any(),
            ),
        ))
    } else {
        let mut val_out = Vec::with_capacity(n);
        let mut gx = Vec::with_capacity(n);
        let mut gy = Vec::with_capacity(n);
        let mut gz = Vec::with_capacity(n);
        for i in 0..n {
            let (_, _, v, g) =
                evaluate_spherical_harmonic_gradient_scalar(l, m, xs[i], ys[i], zs[i], false);
            val_out.push(v);
            gx.push(g[0]);
            gy.push(g[1]);
            gz.push(g[2]);
        }
        Ok((
            array(py, &[n], val_out)?.into_any(),
            (
                array(py, &[n], gx)?.into_any(),
                array(py, &[n], gy)?.into_any(),
                array(py, &[n], gz)?.into_any(),
            ),
        ))
    }
}

#[pyfunction]
pub fn angular_factor<'py>(
    py: Python<'py>,
    l: usize,
    m: i32,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    real: bool,
) -> PyResult<(
    Array<'py, f64>,
    (Array<'py, f64>, Array<'py, f64>, Array<'py, f64>),
    Bound<'py, PyAny>,
    (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>),
)> {
    layout(&x)?;
    layout(&y)?;
    layout(&z)?;
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    let zs = z.as_slice()?;
    let n = xs.len();
    if ys.len() != n || zs.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x, y, z must have the same length",
        ));
    }
    let mut r_out = Vec::with_capacity(n);
    let x_mod = xs.to_vec();
    let y_mod = ys.to_vec();
    let mut z_mod = zs.to_vec();

    for i in 0..n {
        let mut r = (x_mod[i] * x_mod[i] + y_mod[i] * y_mod[i] + z_mod[i] * z_mod[i]).sqrt();
        if r < NODE_OFFSET {
            z_mod[i] = NODE_OFFSET;
            r = NODE_OFFSET;
        }
        r_out.push(r);
    }

    if real {
        let mut val_out = Vec::with_capacity(n);
        let mut gx = Vec::with_capacity(n);
        let mut gy = Vec::with_capacity(n);
        let mut gz = Vec::with_capacity(n);
        for i in 0..n {
            let (v, g, _, _) = evaluate_spherical_harmonic_gradient_scalar(
                l, m, x_mod[i], y_mod[i], z_mod[i], true,
            );
            val_out.push(v);
            gx.push(g[0]);
            gy.push(g[1]);
            gz.push(g[2]);
        }
        let r_arr = array(py, &[n], r_out)?;
        let x_arr = array(py, &[n], x_mod)?;
        let y_arr = array(py, &[n], y_mod)?;
        let z_arr = array(py, &[n], z_mod)?;
        let val_any = array(py, &[n], val_out)?.into_any();
        let gx_any = array(py, &[n], gx)?.into_any();
        let gy_any = array(py, &[n], gy)?.into_any();
        let gz_any = array(py, &[n], gz)?.into_any();
        Ok((r_arr, (x_arr, y_arr, z_arr), val_any, (gx_any, gy_any, gz_any)))
    } else {
        let mut val_out = Vec::with_capacity(n);
        let mut gx = Vec::with_capacity(n);
        let mut gy = Vec::with_capacity(n);
        let mut gz = Vec::with_capacity(n);
        for i in 0..n {
            let (_, _, v, g) = evaluate_spherical_harmonic_gradient_scalar(
                l, m, x_mod[i], y_mod[i], z_mod[i], false,
            );
            val_out.push(v);
            gx.push(g[0]);
            gy.push(g[1]);
            gz.push(g[2]);
        }
        let r_arr = array(py, &[n], r_out)?;
        let x_arr = array(py, &[n], x_mod)?;
        let y_arr = array(py, &[n], y_mod)?;
        let z_arr = array(py, &[n], z_mod)?;
        let val_any = array(py, &[n], val_out)?.into_any();
        let gx_any = array(py, &[n], gx)?.into_any();
        let gy_any = array(py, &[n], gy)?.into_any();
        let gz_any = array(py, &[n], gz)?.into_any();
        Ok((r_arr, (x_arr, y_arr, z_arr), val_any, (gx_any, gy_any, gz_any)))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(spherical_coords, m)?)?;
    m.add_function(wrap_pyfunction!(real_harmonic_transform, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_harmonic_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_harmonic_gradient_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(angular_factor, m)?)?;
    Ok(())
}
