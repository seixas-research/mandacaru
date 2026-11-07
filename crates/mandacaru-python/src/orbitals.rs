//! Native implementations for orbital normalization, radial splines, and special functions.
use crate::arrays::{array, error, layout, Array};
use nalgebra::{DMatrix, DVector};
use numpy::{PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;

fn gamma_half_integer(l: usize) -> f64 {
    let mut g = std::f64::consts::PI.sqrt();
    for i in 0..=l {
        g *= (i as f64) + 0.5;
    }
    g
}

#[pyfunction]
pub fn gaussian_primitive_norm<'py>(
    py: Python<'py>,
    alpha: PyReadonlyArray1<'py, f64>,
    l: usize,
) -> PyResult<Array<'py, f64>> {
    layout(&alpha)?;
    let a_slice = alpha.as_slice()?;
    let g = gamma_half_integer(l);
    let p = l as f64 + 1.5;
    let out: Vec<f64> = a_slice
        .iter()
        .map(|&a| (2.0 * (2.0 * a).powf(p) / g).sqrt())
        .collect();
    let n = out.len();
    array(py, &[n], out)
}

#[pyfunction]
pub fn gaussian_contracted_self_overlap<'py>(
    exponents: PyReadonlyArray1<'py, f64>,
    d: PyReadonlyArray1<'py, f64>,
    l: usize,
) -> PyResult<f64> {
    layout(&exponents)?;
    layout(&d)?;
    let exp_slice = exponents.as_slice()?;
    let d_slice = d.as_slice()?;
    let n = exp_slice.len();
    if d_slice.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "exponents and d must have matching length",
        ));
    }
    let g = gamma_half_integer(l);
    let p = l as f64 + 1.5;
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            let a = exp_slice[i] + exp_slice[j];
            let integ = g / (2.0 * a.powf(p));
            sum += d_slice[i] * integ * d_slice[j];
        }
    }
    Ok(sum)
}

#[pyfunction]
pub fn tail_radius<'py>(
    r: PyReadonlyArray1<'py, f64>,
    u: PyReadonlyArray1<'py, f64>,
    tail_norm: f64,
) -> PyResult<f64> {
    layout(&r)?;
    layout(&u)?;
    let r_slice = r.as_slice()?;
    let u_slice = u.as_slice()?;
    let n = r_slice.len();
    if n == 0 {
        return Ok(0.0);
    }
    if u_slice.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r and u must have matching length",
        ));
    }
    let mut total = 0.0;
    for i in 1..n {
        let dr = r_slice[i] - r_slice[i - 1];
        total += 0.5 * dr * (u_slice[i - 1].powi(2) + u_slice[i].powi(2));
    }
    if total <= 0.0 {
        return Ok(r_slice[n - 1]);
    }
    let mut inner = 0.0;
    if (1.0 - inner / total) < tail_norm {
        return Ok(r_slice[0]);
    }
    for i in 1..n {
        let dr = r_slice[i] - r_slice[i - 1];
        inner += 0.5 * dr * (u_slice[i - 1].powi(2) + u_slice[i].powi(2));
        let tail = 1.0 - inner / total;
        if tail < tail_norm {
            return Ok(r_slice[i]);
        }
    }
    Ok(r_slice[n - 1])
}

fn eval_genlaguerre(k: usize, alpha: f64, x: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let mut l_prev = 1.0;
    let mut l_curr = 1.0 + alpha - x;
    if k == 1 {
        return l_curr;
    }
    for i in 1..k {
        let i_f = i as f64;
        let l_next =
            ((2.0 * i_f + 1.0 + alpha - x) * l_curr - (i_f + alpha) * l_prev) / (i_f + 1.0);
        l_prev = l_curr;
        l_curr = l_next;
    }
    l_curr
}

#[pyfunction]
pub fn hydrogenic_radial_norm(n: usize, l: usize, z: f64) -> f64 {
    let mut num = 1.0;
    if n > l + 1 {
        for i in 1..=(n - l - 1) {
            num *= i as f64;
        }
    }
    let mut den = (2 * n) as f64;
    for i in 1..=(n + l) {
        den *= i as f64;
    }
    let factor = 2.0 * z / (n as f64);
    (factor.powi(3) * num / den).sqrt()
}

#[pyfunction]
pub fn hydrogenic_laguerre<'py>(
    py: Python<'py>,
    k: usize,
    alpha: f64,
    rho: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    layout(&rho)?;
    let slice = rho.as_slice()?;
    let shape = rho.shape();
    let out: Vec<f64> = slice
        .iter()
        .map(|&x| eval_genlaguerre(k, alpha, x))
        .collect();
    array(py, shape, out)
}

#[pyfunction]
#[pyo3(signature = (r_grid, values, query, l=0, r_c=None, slope=false))]
pub fn radial_orbital_evaluate<'py>(
    py: Python<'py>,
    r_grid: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    query: PyReadonlyArrayDyn<'py, f64>,
    l: usize,
    r_c: Option<f64>,
    slope: bool,
) -> PyResult<Array<'py, f64>> {
    layout(&r_grid)?;
    layout(&values)?;
    layout(&query)?;
    let r_slice = r_grid.as_slice()?;
    let val_slice = values.as_slice()?;
    let q_slice = query.as_slice()?;
    let q_shape = query.shape();

    let n = r_slice.len();
    if n < 4 || n != val_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_grid and values must have matching length >= 4",
        ));
    }

    let ctx = mandacaru_core::Context::new(None, 64 << 20).map_err(error)?;
    let spline = mandacaru_core::math::spline::Cubic::new(
        &ctx,
        ctx.copy_slice(r_slice).map_err(error)?,
        ctx.copy_slice(val_slice).map_err(error)?,
    )
    .map_err(error)?;

    let r_max = r_c.unwrap_or(r_slice[n - 1]);
    let r0 = r_slice[0];

    let mut out = Vec::with_capacity(q_slice.len());
    if !slope {
        for &rad in q_slice {
            if rad >= r_max || rad.is_nan() {
                out.push(0.0);
            } else if rad < r0 {
                let base = spline.value_slope(r0).0;
                if r0 > 0.0 && l > 0 {
                    let ratio = (rad.max(0.0) / r0).powi(l as i32);
                    out.push(base * ratio);
                } else {
                    out.push(base);
                }
            } else {
                out.push(spline.value_slope(rad).0);
            }
        }
    } else {
        for &rad in q_slice {
            if rad >= r_max || rad.is_nan() {
                out.push(0.0);
            } else if rad < r0 {
                if r0 > 0.0 && l > 0 {
                    let base = spline.value_slope(r0).0;
                    let p = if l > 1 { (l - 1) as i32 } else { 0 };
                    let val = if l == 1 {
                        base * (l as f64) / r0
                    } else {
                        base * (l as f64) * rad.max(0.0).powi(p) / r0.powi(l as i32)
                    };
                    out.push(val);
                } else {
                    out.push(0.0);
                }
            } else {
                out.push(spline.value_slope(rad).1);
            }
        }
    }

    array(py, q_shape, out)
}

const GL64_X: [f64; 64] = [
    -9.99305041735772170e-01, -9.96340116771955220e-01, -9.91013371476744287e-01,
    -9.83336253884625977e-01, -9.73326827789910975e-01, -9.61008799652053658e-01,
    -9.46411374858402765e-01, -9.29569172131939570e-01, -9.10522137078502825e-01,
    -8.89315445995114140e-01, -8.65999398154092770e-01, -8.40629296252580316e-01,
    -8.13265315122797539e-01, -7.83972358943341385e-01, -7.52819907260531940e-01,
    -7.19881850171610771e-01, -6.85236313054233270e-01, -6.48965471254657311e-01,
    -6.11155355172393278e-01, -5.71895646202634000e-01, -5.31279464019894565e-01,
    -4.89403145707052956e-01, -4.46366017253464087e-01, -4.02270157963991570e-01,
    -3.57220158337668126e-01, -3.11322871990210970e-01, -2.64687162208767424e-01,
    -2.17423643740007083e-01, -1.69644420423992803e-01, -1.21462819296120558e-01,
    -7.29931217877990424e-02, -2.43502926634244360e-02, 2.43502926634244360e-02,
    7.29931217877990424e-02, 1.21462819296120558e-01, 1.69644420423992803e-01,
    2.17423643740007083e-01, 2.64687162208767424e-01, 3.11322871990210970e-01,
    3.57220158337668126e-01, 4.02270157963991570e-01, 4.46366017253464087e-01,
    4.89403145707052956e-01, 5.31279464019894565e-01, 5.71895646202634000e-01,
    6.11155355172393278e-01, 6.48965471254657311e-01, 6.85236313054233270e-01,
    7.19881850171610771e-01, 7.52819907260531940e-01, 7.83972358943341385e-01,
    8.13265315122797539e-01, 8.40629296252580316e-01, 8.65999398154092770e-01,
    8.89315445995114140e-01, 9.10522137078502825e-01, 9.29569172131939570e-01,
    9.46411374858402765e-01, 9.61008799652053658e-01, 9.73326827789910975e-01,
    9.83336253884625977e-01, 9.91013371476744287e-01, 9.96340116771955220e-01,
    9.99305041735772170e-01,
];

const GL64_W: [f64; 64] = [
    1.78328072169426418e-03, 4.14703326056464847e-03, 6.50445796897843737e-03,
    8.84675982636336060e-03, 1.11681394601310085e-02, 1.34630478967191197e-02,
    1.57260304760251102e-02, 1.79517157756972877e-02, 2.01348231535300390e-02,
    2.22701738083829447e-02, 2.43527025687110855e-02, 2.63774697150548944e-02,
    2.83396726142594244e-02, 3.02346570724024884e-02, 3.20579283548514116e-02,
    3.38051618371418769e-02, 3.54722132568822263e-02, 3.70551285402400191e-02,
    3.85501531786156051e-02, 3.99537411327204536e-02, 4.12625632426234859e-02,
    4.24735151236535421e-02, 4.35837245293234712e-02, 4.45905581637566009e-02,
    4.54916279274181004e-02, 4.62847965813143747e-02, 4.69681828162100065e-02,
    4.75401657148303430e-02, 4.79993885964583034e-02, 4.83447622348029335e-02,
    4.85754674415034282e-02, 4.86909570091396890e-02, 4.86909570091396890e-02,
    4.85754674415034282e-02, 4.83447622348029335e-02, 4.79993885964583034e-02,
    4.75401657148303430e-02, 4.69681828162100065e-02, 4.62847965813143747e-02,
    4.54916279274181004e-02, 4.45905581637566009e-02, 4.35837245293234712e-02,
    4.24735151236535421e-02, 4.12625632426234859e-02, 3.99537411327204536e-02,
    3.85501531786156051e-02, 3.70551285402400191e-02, 3.54722132568822263e-02,
    3.38051618371418769e-02, 3.20579283548514116e-02, 3.02346570724024884e-02,
    2.83396726142594244e-02, 2.63774697150548944e-02, 2.43527025687110855e-02,
    2.22701738083829447e-02, 2.01348231535300390e-02, 1.79517157756972877e-02,
    1.57260304760251102e-02, 1.34630478967191197e-02, 1.11681394601310085e-02,
    8.84675982636336060e-03, 6.50445796897843737e-03, 4.14703326056464847e-03,
    1.78328072169426418e-03,
];

pub fn slater_moment_single(alpha: f64, q: usize) -> f64 {
    if !alpha.is_finite() || alpha <= 0.0 {
        return f64::NAN;
    }
    if alpha >= 0.2 {
        let sa = alpha.sqrt();
        let mut integ = 0.0;
        for i in 0..64 {
            let u = 3.5 * (GL64_X[i] + 1.0);
            let w = 3.5 * GL64_W[i];
            integ += w * u.powi(q as i32) * (-u * u - u / sa).exp();
        }
        integ * alpha.powf(-((q as f64) + 1.0) / 2.0)
    } else {
        let mut integ = 0.0;
        for i in 0..64 {
            let r = 22.5 * (GL64_X[i] + 1.0);
            let w = 22.5 * GL64_W[i];
            integ += w * r.powi(q as i32) * (-alpha * r * r - r).exp();
        }
        integ
    }
}

pub fn slater_projection_single(alpha: f64, n: usize, l: usize) -> f64 {
    let p = l as f64 + 1.5;
    let g = gamma_half_integer(l);
    let norm = (2.0 * (2.0 * alpha).powf(p) / g).sqrt();
    let mut factorial = 1.0;
    for k in 1..=(2 * n) {
        factorial *= k as f64;
    }
    let sto_norm = ((2.0f64).powi((2 * n + 1) as i32) / factorial).sqrt();
    let integ = slater_moment_single(alpha, n + l + 1);
    norm * sto_norm * integ
}

#[pyfunction]
pub fn slater_moment<'py>(
    py: Python<'py>,
    alpha: PyReadonlyArray1<'py, f64>,
    q: usize,
) -> PyResult<Array<'py, f64>> {
    layout(&alpha)?;
    let slice = alpha.as_slice()?;
    let out: Vec<f64> = slice.iter().map(|&a| slater_moment_single(a, q)).collect();
    let n = out.len();
    array(py, &[n], out)
}

#[pyfunction]
pub fn slater_projection<'py>(
    py: Python<'py>,
    alpha: PyReadonlyArray1<'py, f64>,
    n: usize,
    l: usize,
) -> PyResult<Array<'py, f64>> {
    layout(&alpha)?;
    let slice = alpha.as_slice()?;
    let out: Vec<f64> = slice.iter().map(|&a| slater_projection_single(a, n, l)).collect();
    let count = out.len();
    array(py, &[count], out)
}

#[pyfunction]
pub fn fit_sto_reference(
    n: usize,
    l: usize,
    n_gaussians: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, f64)> {
    if n_gaussians < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_gaussians must be >= 1",
        ));
    }
    let p = l as f64 + 1.5;
    let center = 1.0 / ((n.saturating_sub(1).max(1)) as f64).powi(2);
    let k = n_gaussians;

    let projected_overlap = |alpha: &[f64]| -> Option<(f64, Vec<f64>)> {
        if alpha.iter().any(|&a| !a.is_finite() || a <= 0.0) {
            return None;
        }
        let mut mat_s = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                let ai = alpha[i];
                let aj = alpha[j];
                let val = (2.0 * (ai * aj).sqrt() / (ai + aj)).powf(p);
                mat_s[(i, j)] = val;
            }
        }
        let mut vec_b = DVector::zeros(k);
        for i in 0..k {
            let bi = slater_projection_single(alpha[i], n, l);
            if !bi.is_finite() {
                return None;
            }
            vec_b[i] = bi;
        }
        let decomp = mat_s.lu();
        let c = decomp.solve(&vec_b)?;
        let val = vec_b.dot(&c);
        if !val.is_finite() || val > 1.0 + 1e-9 || val < 0.0 {
            return None;
        }
        Some((val, c.as_slice().to_vec()))
    };

    let neg_overlap_sq = |log_alpha: &[f64]| -> f64 {
        for &la in log_alpha {
            if la < -12.0 || la > 12.0 || !la.is_finite() {
                return 1.0;
            }
        }
        let alpha: Vec<f64> = log_alpha.iter().map(|&la| la.exp()).collect();
        match projected_overlap(&alpha) {
            Some((val, _)) => -val,
            None => 1.0,
        }
    };

    let ctx = mandacaru_core::Context::new(None, 64 * 1024 * 1024)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let mut best_val = 1.0;
    let mut best_x = Vec::new();

    for &spread in &[1.6f64, 2.2, 3.0] {
        let mut t0 = Vec::with_capacity(k);
        for idx in 0..k {
            let offset = (idx as f64) - ((k as f64) - 1.0) / 2.0;
            t0.push((center * spread.powf(offset)).ln());
        }
        let options = mandacaru_core::quantum::optimizer::Options {
            method: mandacaru_core::quantum::optimizer::Method::NelderMead,
            tolerance: Some(1e-12),
            max_iterations: 20000,
            max_evaluations: 20000,
            seed: 0,
            track_best: true,
            patience: 5,
        };
        if let Ok(outcome) = mandacaru_core::quantum::optimizer::minimize(
            &ctx,
            &t0,
            &options,
            |x| Ok((neg_overlap_sq(x), vec![0.0; x.len()])),
        ) {
            if outcome.energy < best_val {
                best_val = outcome.energy;
                best_x = outcome.parameters;
            }
        }
    }

    if best_x.is_empty() || best_val >= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "the STO-{}G fit of the ({}, {}) Slater orbital did not converge to a well-conditioned contraction; use fewer primitives",
            n_gaussians, n, l
        )));
    }

    let alpha: Vec<f64> = best_x.iter().map(|&la| la.exp()).collect();
    let (ov, c) = projected_overlap(&alpha).ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("infeasible best fit parameters")
    })?;

    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_by(|&a, &b| alpha[b].total_cmp(&alpha[a]));

    let sorted_alpha: Vec<f64> = indices.iter().map(|&i| alpha[i]).collect();
    let sorted_c: Vec<f64> = indices.iter().map(|&i| c[i]).collect();

    Ok((sorted_alpha, sorted_c, ov))
}

#[pyfunction]
#[pyo3(signature = (r, r_onset, width, scale=1.0, cap=1.0e5))]
pub fn confinement_potential<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    r_onset: f64,
    width: f64,
    scale: f64,
    cap: f64,
) -> PyResult<Array<'py, f64>> {
    layout(&r)?;
    if width <= 0.0 || r_onset < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "require width > 0 and r_onset >= 0",
        ));
    }
    let r_slice = r.as_slice()?;
    let n = r_slice.len();
    let mut v = Vec::with_capacity(n);
    let r0 = r_onset;
    let w = width;
    let r_end = r0 + w;
    for &x in r_slice {
        if x <= r0 {
            v.push(0.0);
        } else if x < r_end {
            let val = scale * (-w / (x - r0)).exp() / (r_end - x).powi(2);
            v.push(val.min(cap));
        } else {
            v.push(cap);
        }
    }
    array(py, &[n], v)
}

#[pyfunction]
pub fn radial_mean_radius(
    r: PyReadonlyArray1<'_, f64>,
    u: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    layout(&r)?;
    layout(&u)?;
    let r_s = r.as_slice()?;
    let u_s = u.as_slice()?;
    let n = r_s.len();
    if u_s.len() != n || n < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r and u must have matching length >= 2",
        ));
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..(n - 1) {
        let dr = r_s[i + 1] - r_s[i];
        let y_num_0 = u_s[i] * u_s[i] * r_s[i];
        let y_num_1 = u_s[i + 1] * u_s[i + 1] * r_s[i + 1];
        num += 0.5 * (y_num_0 + y_num_1) * dr;
        let y_den_0 = u_s[i] * u_s[i];
        let y_den_1 = u_s[i + 1] * u_s[i + 1];
        den += 0.5 * (y_den_0 + y_den_1) * dr;
    }
    if den == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "norm of u is zero in mean_radius",
        ));
    }
    Ok(num / den)
}

#[pyfunction]
#[pyo3(signature = (r, u, threshold=1e-6))]
pub fn radial_outer_radius(
    r: PyReadonlyArray1<'_, f64>,
    u: PyReadonlyArray1<'_, f64>,
    threshold: f64,
) -> PyResult<f64> {
    layout(&r)?;
    layout(&u)?;
    let r_s = r.as_slice()?;
    let u_s = u.as_slice()?;
    let n = r_s.len().min(u_s.len());
    for i in (0..n).rev() {
        if u_s[i].abs() > threshold {
            return Ok(r_s[i]);
        }
    }
    Ok(0.0)
}

fn trap_prod(a: &[f64], b: &[f64], r: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..(a.len() - 1) {
        sum += 0.5 * (a[i] * b[i] + a[i + 1] * b[i + 1]) * (r[i + 1] - r[i]);
    }
    sum
}

#[pyfunction]
#[pyo3(signature = (r, u_list, l_list, is_minimal_list, labels, tol=1e-4))]
pub fn radial_orthonormalize<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    u_list: Vec<PyReadonlyArray1<'py, f64>>,
    l_list: Vec<usize>,
    is_minimal_list: Vec<bool>,
    labels: Vec<String>,
    tol: f64,
) -> PyResult<(Vec<usize>, Vec<Array<'py, f64>>)> {
    layout(&r)?;
    let r_s = r.as_slice()?;
    let n = r_s.len();
    let m = u_list.len();
    if l_list.len() != m || is_minimal_list.len() != m || labels.len() != m {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "u_list, l_list, is_minimal_list, and labels must have matching length",
        ));
    }
    for u in &u_list {
        layout(u)?;
        if u.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "all u arrays must match length of r",
            ));
        }
    }

    let mut accepted_idx = Vec::new();
    let mut accepted_l = Vec::new();
    let mut accepted_u: Vec<Vec<f64>> = Vec::new();

    for idx in 0..m {
        let mut u = u_list[idx].as_slice()?.to_vec();
        let l = l_list[idx];
        let is_min = is_minimal_list[idx];
        let label = &labels[idx];
        let mut dropped = false;

        for pass in 0..3 {
            for (g, &g_l) in accepted_u.iter().zip(&accepted_l) {
                if g_l == l {
                    let overlap = trap_prod(g, &u, r_s);
                    for k in 0..n {
                        u[k] -= overlap * g[k];
                    }
                }
            }
            let norm2 = trap_prod(&u, &u, r_s);
            if pass == 0 && norm2 < tol {
                if is_min {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "minimal function {} is linearly dependent",
                        label
                    )));
                }
                dropped = true;
                break;
            }
            if !norm2.is_finite() || norm2 <= 0.0 {
                dropped = true;
                break;
            }
            let inv_norm = 1.0 / norm2.sqrt();
            for k in 0..n {
                u[k] *= inv_norm;
            }
        }

        if !dropped {
            let mut max_abs = -1.0;
            let mut pivot = 0;
            for (k, &val) in u.iter().enumerate() {
                let a = val.abs();
                if a > max_abs {
                    max_abs = a;
                    pivot = k;
                }
            }
            if u[pivot] < 0.0 {
                for val in u.iter_mut() {
                    *val = -*val;
                }
            }
            accepted_idx.push(idx);
            accepted_l.push(l);
            accepted_u.push(u);
        }
    }

    let mut out_u = Vec::with_capacity(accepted_u.len());
    for u in accepted_u {
        out_u.push(array(py, &[n], u)?);
    }
    Ok((accepted_idx, out_u))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gaussian_primitive_norm, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_contracted_self_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(tail_radius, m)?)?;
    m.add_function(wrap_pyfunction!(hydrogenic_radial_norm, m)?)?;
    m.add_function(wrap_pyfunction!(hydrogenic_laguerre, m)?)?;
    m.add_function(wrap_pyfunction!(radial_orbital_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(slater_moment, m)?)?;
    m.add_function(wrap_pyfunction!(slater_projection, m)?)?;
    m.add_function(wrap_pyfunction!(fit_sto_reference, m)?)?;
    m.add_function(wrap_pyfunction!(confinement_potential, m)?)?;
    m.add_function(wrap_pyfunction!(radial_mean_radius, m)?)?;
    m.add_function(wrap_pyfunction!(radial_outer_radius, m)?)?;
    m.add_function(wrap_pyfunction!(radial_orthonormalize, m)?)?;
    Ok(())
}
