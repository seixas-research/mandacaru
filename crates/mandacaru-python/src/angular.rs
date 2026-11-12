//! Python bindings for spherical harmonics and angular coordinate machinery.
use crate::arrays::{array, layout, Array};
use numpy::{Complex64, PyReadonlyArray1};
use pyo3::prelude::*;
use mandacaru_core::math::harmonics::{harmonic, harmonic_gradient, polar, NODE_OFFSET};

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
        let (r, theta, phi) = polar(xs[i] - center[0], ys[i] - center[1], zs[i] - center[2]);
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
            let (v, _) = harmonic(l, m, ts[i], ps[i], true);
            out.push(v);
        }
        Ok(array(py, &[n], out)?.into_any())
    } else {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let (_, v) = harmonic(l, m, ts[i], ps[i], false);
            out.push(v);
        }
        Ok(array(py, &[n], out)?.into_any())
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
                harmonic_gradient(l, m, xs[i], ys[i], zs[i], true);
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
                harmonic_gradient(l, m, xs[i], ys[i], zs[i], false);
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
            let (v, g, _, _) = harmonic_gradient(
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
            let (_, _, v, g) = harmonic_gradient(
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
