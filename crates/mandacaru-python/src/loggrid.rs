//! Logarithmic grid solver for radial states.
use crate::arrays::{array, error, layout, Array};
use mandacaru_core::models::relativity::{Treatment, SCALAR_KAPPA, SPEED_OF_LIGHT};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

pub const MASS_FLOOR: f64 = 1e-3;
const _RESCALE_AT: f64 = 1e120;
const _NODE_FLOOR: f64 = 1e-10;

pub fn fourth_order_derivatives(f: &[f64], dx: f64) -> (Vec<f64>, Vec<f64>) {
    let n = f.len();
    let mut fx = vec![0.0; n];
    let mut fxx = vec![0.0; n];
    if n < 6 {
        // Fallback for short grids
        return (fx, fxx);
    }
    for i in 2..n - 2 {
        fx[i] = (-f[i + 2] + 8.0 * f[i + 1] - 8.0 * f[i - 1] + f[i - 2]) / (12.0 * dx);
        fxx[i] = (-f[i + 2] + 16.0 * f[i + 1] - 30.0 * f[i] + 16.0 * f[i - 1] - f[i - 2])
            / (12.0 * dx * dx);
    }
    fx[0] = (-25.0 * f[0] + 48.0 * f[1] - 36.0 * f[2] + 16.0 * f[3] - 3.0 * f[4]) / (12.0 * dx);
    fx[1] = (-3.0 * f[0] - 10.0 * f[1] + 18.0 * f[2] - 6.0 * f[3] + f[4]) / (12.0 * dx);
    fx[n - 1] =
        (25.0 * f[n - 1] - 48.0 * f[n - 2] + 36.0 * f[n - 3] - 16.0 * f[n - 4] + 3.0 * f[n - 5])
            / (12.0 * dx);
    fx[n - 2] =
        (3.0 * f[n - 1] + 10.0 * f[n - 2] - 18.0 * f[n - 3] + 6.0 * f[n - 4] - f[n - 5])
            / (12.0 * dx);

    let h2 = 12.0 * dx * dx;
    fxx[0] = (45.0 * f[0] - 154.0 * f[1] + 214.0 * f[2] - 156.0 * f[3] + 61.0 * f[4] - 10.0 * f[5])
        / h2;
    fxx[1] = (10.0 * f[0] - 15.0 * f[1] - 4.0 * f[2] + 14.0 * f[3] - 6.0 * f[4] + f[5]) / h2;
    fxx[n - 1] = (45.0 * f[n - 1]
        - 154.0 * f[n - 2]
        + 214.0 * f[n - 3]
        - 156.0 * f[n - 4]
        + 61.0 * f[n - 5]
        - 10.0 * f[n - 6])
        / h2;
    fxx[n - 2] = (10.0 * f[n - 1]
        - 15.0 * f[n - 2]
        - 4.0 * f[n - 3]
        + 14.0 * f[n - 4]
        - 6.0 * f[n - 5]
        + f[n - 6])
        / h2;
    (fx, fxx)
}

pub fn compute_radial_g(
    r: &[f64],
    potential: &[f64],
    l: usize,
    energy: f64,
    kappa: i32,
    relativistic: bool,
    c: f64,
    dx: f64,
    derivatives: Option<(&[f64], &[f64])>,
) -> (Vec<f64>, Option<Vec<f64>>) {
    let n = r.len();
    let ll = (l * (l + 1)) as f64;
    if !relativistic {
        let mut g = Vec::with_capacity(n);
        for i in 0..n {
            g.push(ll + 0.25 + 2.0 * r[i] * r[i] * (potential[i] - energy));
        }
        return (g, None);
    }
    let two_c2 = 2.0 * c * c;
    let mut m_vec = Vec::with_capacity(n);
    for i in 0..n {
        m_vec.push((1.0 + (energy - potential[i]) / two_c2).max(MASS_FLOOR));
    }
    let (vx, vxx) = match derivatives {
        Some((vx, vxx)) => (vx.to_vec(), vxx.to_vec()),
        None => fourth_order_derivatives(potential, dx),
    };
    let mut g = Vec::with_capacity(n);
    let k = kappa as f64;
    for i in 0..n {
        let mx = -vx[i] / two_c2;
        let mxx = -vxx[i] / two_c2;
        let m = m_vec[i];
        let w = mx / m;
        let wx = mxx / m - w * w;
        let val = ll + k * w + (1.0 + w).powi(2) / 4.0 - wx / 2.0
            - 2.0 * m * r[i] * r[i] * (energy - potential[i]);
        g.push(val);
    }
    (g, Some(m_vec))
}

fn numerov_outward(f: &[f64], dx: f64, stop: usize, slope: f64) -> Vec<f64> {
    let n = f.len();
    let mut y = vec![0.0; n];
    y[0] = 1.0;
    y[1] = (slope * dx).exp();
    for i in 1..=stop {
        if i + 1 >= n {
            break;
        }
        y[i + 1] = ((12.0 - 10.0 * f[i]) * y[i] - f[i - 1] * y[i - 1]) / f[i + 1];
        if y[i + 1].abs() > _RESCALE_AT {
            for v in y[..=i + 1].iter_mut() {
                *v /= _RESCALE_AT;
            }
        }
    }
    y
}

fn numerov_inward(f: &[f64], g: &[f64], dx: f64, stop: usize) -> Vec<f64> {
    let n = f.len();
    let mut y = vec![0.0; n];
    let n_last = n - 1;
    y[n_last] = 1.0;
    y[n_last - 1] = (g[n_last].max(0.0).sqrt() * dx).exp();
    for i in (stop..n_last).rev() {
        if i == 0 {
            break;
        }
        y[i - 1] = ((12.0 - 10.0 * f[i]) * y[i] - f[i + 1] * y[i + 1]) / f[i - 1];
        if y[i - 1].abs() > _RESCALE_AT {
            for v in y[i - 1..].iter_mut() {
                *v /= _RESCALE_AT;
            }
        }
    }
    y
}

fn match_state(
    r: &[f64],
    potential: &[f64],
    l: usize,
    energy: f64,
    kappa: i32,
    relativistic: bool,
    c: f64,
    dx: f64,
    derivatives: Option<(&[f64], &[f64])>,
) -> (Option<f64>, usize, Option<Vec<f64>>, Option<Vec<f64>>) {
    let (g, m) = compute_radial_g(
        r, potential, l, energy, kappa, relativistic, c, dx, derivatives,
    );
    let mut allowed_last = None;
    for (i, &val) in g.iter().enumerate() {
        if val < 0.0 {
            allowed_last = Some(i);
        }
    }
    let Some(last) = allowed_last else {
        return (None, 0, None, m);
    };
    let n = g.len();
    let m_idx = last.clamp(2, n - 3);

    let mut f = Vec::with_capacity(n);
    for &gi in &g {
        f.push(1.0 - dx * dx * gi / 12.0);
    }
    let slope = g[0].max(1e-12).sqrt();
    let out = numerov_outward(&f, dx, m_idx, slope);
    let inw = numerov_inward(&f, &g, dx, m_idx);
    if out[m_idx] == 0.0 || inw[m_idx] == 0.0 {
        return (None, 0, None, m);
    }

    let mut y = vec![0.0; n];
    y[..=m_idx].copy_from_slice(&out[..=m_idx]);
    let scale = out[m_idx] / inw[m_idx];
    for i in m_idx..n {
        y[i] = inw[i] * scale;
    }

    let head = &out[..=m_idx];
    let max_head = head.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let floor = _NODE_FLOOR * max_head;

    let mut nodes = 0;
    let mut prev_sign: Option<bool> = None;
    for &v in head {
        if v.abs() > floor {
            let sign = v.is_sign_positive();
            if let Some(prev) = prev_sign {
                if prev != sign {
                    nodes += 1;
                }
            }
            prev_sign = Some(sign);
        }
    }

    let residual = (f[m_idx - 1] * y[m_idx - 1] + f[m_idx + 1] * y[m_idx + 1]
        - (12.0 - 10.0 * f[m_idx]) * y[m_idx])
        / (dx * dx * y[m_idx]);
    (Some(residual), nodes, Some(y), m)
}

#[pyfunction]
#[pyo3(signature = (r, potential, l, n_nodes, kappa=None, treatment="none", atomic_number=0.0, c=None, energy_guess=None, max_iterations=200, tolerance=1e-13, derivatives=None))]
pub fn solve_radial_log<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    potential: PyReadonlyArray1<'py, f64>,
    l: usize,
    n_nodes: usize,
    kappa: Option<i32>,
    treatment: &str,
    atomic_number: f64,
    c: Option<f64>,
    energy_guess: Option<f64>,
    max_iterations: usize,
    tolerance: f64,
    derivatives: Option<(PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, f64>)>,
) -> PyResult<(Array<'py, f64>, f64)> {
    layout(&r)?;
    layout(&potential)?;
    let r_slice = r.as_slice()?;
    let pot_slice = potential.as_slice()?;
    let n = r_slice.len();
    if pot_slice.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r and potential must have matching length",
        ));
    }
    let c_speed = c.unwrap_or(SPEED_OF_LIGHT);
    let tr = Treatment::parse(treatment).map_err(error)?;
    let relativistic = tr != Treatment::None;
    let k = if tr == Treatment::Scalar {
        SCALAR_KAPPA
    } else {
        kappa.unwrap_or(SCALAR_KAPPA)
    };

    let dx = r_slice[1].ln() - r_slice[0].ln();

    let (vx_vec, vxx_vec);
    let deriv_slices = match &derivatives {
        Some((vx, vxx)) => {
            layout(vx)?;
            layout(vxx)?;
            vx_vec = vx.as_slice()?.to_vec();
            vxx_vec = vxx.as_slice()?.to_vec();
            Some((vx_vec.as_slice(), vxx_vec.as_slice()))
        }
        None => None,
    };

    let probe = |energy: f64| {
        match_state(
            r_slice,
            pot_slice,
            l,
            energy,
            k,
            relativistic,
            c_speed,
            dx,
            deriv_slices,
        )
    };

    let z = atomic_number.max(1.0);
    let mut lo = -(2.0 * z * z + 10.0);
    let mut hi = -1e-12;

    if let Some(guess) = energy_guess {
        let width = 0.25 * guess.abs().max(1.0);
        let near_lo = guess - width;
        let near_hi = (guess + width).min(hi);
        if lo < near_lo && near_lo < near_hi && near_hi < 0.0 {
            let (r_lo, n_lo, _, _) = probe(near_lo);
            let (r_hi, n_hi, _, _) = probe(near_hi);
            if r_lo.is_some() && r_hi.is_some() && n_lo <= n_nodes && n_nodes < n_hi {
                lo = near_lo;
                hi = near_hi;
            }
        }
    }

    let too_low = |energy: f64| -> bool {
        let (residual, nodes, _, _) = probe(energy);
        let Some(res) = residual else {
            return true;
        };
        if nodes < n_nodes {
            return true;
        }
        if nodes > n_nodes {
            return false;
        }
        res < 0.0
    };

    for _ in 0..max_iterations {
        let mid = 0.5 * (lo + hi);
        if too_low(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= tolerance * lo.abs().max(1.0) {
            break;
        }
    }
    let energy = 0.5 * (lo + hi);
    let (_residual, _nodes, y_opt, m_opt) = probe(energy);
    let Some(y) = y_opt else {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "no bound state with {} node(s) for l={} in this potential",
            n_nodes, l
        )));
    };

    let mut p = Vec::with_capacity(n);
    for i in 0..n {
        let x = r_slice[i].ln();
        let mut val = (x / 2.0).exp() * y[i];
        if relativistic {
            if let Some(m) = &m_opt {
                val *= m[i].sqrt();
            }
        }
        p.push(val);
    }

    // trapezoid rule: int P^2 r dx = sum_i 0.5 * (x[i] - x[i-1]) * (P[i-1]^2 r[i-1] + P[i]^2 r[i])
    let mut sum = 0.0;
    for i in 1..n {
        let f_prev = p[i - 1] * p[i - 1] * r_slice[i - 1];
        let f_cur = p[i] * p[i] * r_slice[i];
        sum += 0.5 * dx * (f_prev + f_cur);
    }
    let norm = sum.sqrt();
    for v in p.iter_mut() {
        *v /= norm;
    }
    if p[n / 4] < 0.0 {
        for v in p.iter_mut() {
            *v = -*v;
        }
    }
    Ok((array(py, &[n], p)?, energy))
}

#[pyfunction]
pub fn loggrid_fourth_order_derivatives<'py>(
    py: Python<'py>,
    f: PyReadonlyArray1<'py, f64>,
    dx: f64,
) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
    layout(&f)?;
    let (fx, fxx) = fourth_order_derivatives(f.as_slice()?, dx);
    let n = fx.len();
    Ok((array(py, &[n], fx)?, array(py, &[n], fxx)?))
}

#[pyfunction]
#[pyo3(signature = (r, potential, l, energy, kappa=None, relativistic=false, c=None, dx=None, derivatives=None))]
pub fn loggrid_radial_g<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    potential: PyReadonlyArray1<'py, f64>,
    l: usize,
    energy: f64,
    kappa: Option<i32>,
    relativistic: bool,
    c: Option<f64>,
    dx: Option<f64>,
    derivatives: Option<(PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, f64>)>,
) -> PyResult<(Array<'py, f64>, Option<Array<'py, f64>>)> {
    layout(&r)?;
    layout(&potential)?;
    let r_slice = r.as_slice()?;
    let pot_slice = potential.as_slice()?;
    let c_val = c.unwrap_or(SPEED_OF_LIGHT);
    let dx_val = dx.unwrap_or_else(|| r_slice[1].ln() - r_slice[0].ln());
    let (vx_vec, vxx_vec);
    let deriv_slices = match &derivatives {
        Some((vx, vxx)) => {
            layout(vx)?;
            layout(vxx)?;
            vx_vec = vx.as_slice()?.to_vec();
            vxx_vec = vxx.as_slice()?.to_vec();
            Some((vx_vec.as_slice(), vxx_vec.as_slice()))
        }
        None => None,
    };
    let k = kappa.unwrap_or(SCALAR_KAPPA);
    let (g, m_opt) = compute_radial_g(
        r_slice,
        pot_slice,
        l,
        energy,
        k,
        relativistic,
        c_val,
        dx_val,
        deriv_slices,
    );
    let n = g.len();
    let g_arr = array(py, &[n], g)?;
    let m_arr = match m_opt {
        Some(m) => Some(array(py, &[n], m)?),
        None => None,
    };
    Ok((g_arr, m_arr))
}

#[pyfunction]
#[pyo3(signature = (r_max, atomic_number=1.0, points=1200, r_min=None))]
pub fn loggrid_make_grid<'py>(
    py: Python<'py>,
    r_max: f64,
    atomic_number: f64,
    points: usize,
    r_min: Option<f64>,
) -> PyResult<(Array<'py, f64>, Array<'py, f64>, f64)> {
    if points < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "points must be at least 2",
        ));
    }
    let z = atomic_number.max(1.0);
    let r_min_val = r_min.unwrap_or(1e-8 / z);
    let x_start = r_min_val.ln();
    let x_end = r_max.ln();
    let dx = (x_end - x_start) / ((points - 1) as f64);

    let mut x_vec = Vec::with_capacity(points);
    let mut r_vec = Vec::with_capacity(points);
    for i in 0..points {
        let xi = x_start + (i as f64) * dx;
        x_vec.push(xi);
        r_vec.push(xi.exp());
    }
    if let Some(last_x) = x_vec.last_mut() {
        *last_x = x_end;
    }
    if let Some(last_r) = r_vec.last_mut() {
        *last_r = r_max;
    }

    let py_r = array(py, &[points], r_vec)?;
    let py_x = array(py, &[points], x_vec)?;
    Ok((py_r, py_x, dx))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(loggrid_make_grid, m)?)?;
    m.add_function(wrap_pyfunction!(solve_radial_log, m)?)?;
    m.add_function(wrap_pyfunction!(loggrid_fourth_order_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(loggrid_radial_g, m)?)?;
    Ok(())
}
