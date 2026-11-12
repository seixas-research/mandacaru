//! Typed boundaries for scientific models shared by multiple solver families.
use super::{array, copy, error, layout, Array, PyContext};
use mandacaru_core::models::{
    radial_xc,
    configuration,
    relativity::{self, PotentialTerms, Treatment, SCALAR_KAPPA, SPEED_OF_LIGHT},
    spherical,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub const MASS_FLOOR: f64 = 1e-3;

#[pyfunction]
pub fn radial_hartree<'py>(
    context: &PyContext,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    density: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    let (r, _r) = copy(&context.inner, &r)?;
    let (density, _d) = copy(&context.inner, &density)?;
    let out = py
        .detach(|| spherical::hartree(&context.inner, &r, &density))
        .map_err(error)?;
    array(py, &[r.len()], out.as_slice().to_vec())
}

#[pyfunction]
#[pyo3(signature = (context, r, rho, functional="lda", relativistic=false))]
pub fn radial_xc_potential<'py>(
    context: &PyContext,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    rho: PyReadonlyArray1<'py, f64>,
    functional: &str,
    relativistic: bool,
) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
    layout(&r)?;
    layout(&rho)?;
    let r_slice = r.as_slice()?;
    let rho_slice = rho.as_slice()?;
    let (energy, potential) = py
        .detach(|| radial_xc::xc_potential(&context.inner, r_slice, rho_slice, functional, relativistic))
        .map_err(error)?;
    let n = energy.len();
    Ok((array(py, &[n], energy)?, array(py, &[n], potential)?))
}

#[pyfunction]
pub fn radial_derivative_nodes<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
) -> PyResult<Option<Array<'py, i64>>> {
    layout(&r)?;
    let r_slice = r.as_slice()?;
    match radial_xc::derivative_nodes(r_slice) {
        Some(nodes) => {
            let i_nodes: Vec<i64> = nodes.into_iter().map(|i| i as i64).collect();
            let n = i_nodes.len();
            Ok(Some(array(py, &[n], i_nodes)?))
        }
        None => Ok(None),
    }
}

#[pyfunction]
#[pyo3(signature = (r, potential, atomic_number=0.0))]
pub fn relativity_potential_derivatives<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    potential: PyReadonlyArray1<'py, f64>,
    atomic_number: f64,
) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
    layout(&r)?;
    layout(&potential)?;
    let r_slice = r.as_slice()?;
    let pot_slice = potential.as_slice()?;
    let terms = PotentialTerms::new(r_slice, pot_slice, atomic_number).map_err(error)?;
    let n = r_slice.len();
    let d2v: Vec<f64> = (0..n)
        .map(|i| -2.0 * atomic_number / (r_slice[i].powi(3)) + terms.d2_smooth[i])
        .collect();
    Ok((array(py, &[n], terms.dv)?, array(py, &[n], d2v)?))
}

#[pyfunction]
#[pyo3(signature = (r, potential, kappa, m, atomic_number=0.0, c=SPEED_OF_LIGHT))]
pub fn relativity_darwin_spin_orbit<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    potential: PyReadonlyArray1<'py, f64>,
    kappa: i32,
    m: PyReadonlyArray1<'py, f64>,
    atomic_number: f64,
    c: f64,
) -> PyResult<Array<'py, f64>> {
    layout(&r)?;
    layout(&potential)?;
    layout(&m)?;
    let r_slice = r.as_slice()?;
    let pot_slice = potential.as_slice()?;
    let m_slice = m.as_slice()?;
    let terms = PotentialTerms::new(r_slice, pot_slice, atomic_number).map_err(error)?;
    let k = kappa as f64;
    let n = r_slice.len();
    let out: Vec<f64> = (0..n)
        .map(|i| {
            let bracket = -(1.0 + k) * atomic_number / r_slice[i].powi(3)
                + 0.5 * terms.d2_smooth[i]
                - k * terms.d_smooth[i] / r_slice[i];
            bracket / (2.0 * c * c * m_slice[i])
        })
        .collect();
    array(py, &[n], out)
}

#[pyfunction]
#[pyo3(signature = (atomic_number, n, kappa, c=SPEED_OF_LIGHT))]
pub fn relativity_hydrogenic_dirac_energy(
    atomic_number: f64,
    n: usize,
    kappa: i32,
    c: f64,
) -> PyResult<f64> {
    let za = atomic_number / c;
    let k = kappa;
    if k.unsigned_abs() as usize > n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "|kappa| = {} exceeds n = {}",
            k.abs(),
            n
        )));
    }
    if za >= k.abs() as f64 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Z/c = {:.4} reaches |kappa| = {}: the point-nucleus Dirac equation has no bound state there",
            za,
            k.abs()
        )));
    }
    let gamma = ((k * k) as f64 - za * za).sqrt();
    let denom = n as f64 - k.abs() as f64 + gamma;
    let x = za * za / (denom * denom);
    let exp_term = (-0.5 * x.ln_1p()).exp_m1();
    Ok(c * c * exp_term)
}

#[pyfunction]
#[pyo3(signature = (r, potential, atomic_number=0.0, c=SPEED_OF_LIGHT, mass_corrected=true, energy=0.0))]
pub fn relativity_spin_orbit_radial<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    potential: PyReadonlyArray1<'py, f64>,
    atomic_number: f64,
    c: f64,
    mass_corrected: bool,
    energy: f64,
) -> PyResult<Array<'py, f64>> {
    layout(&r)?;
    layout(&potential)?;
    let r_slice = r.as_slice()?;
    let pot_slice = potential.as_slice()?;
    let terms = PotentialTerms::new(r_slice, pot_slice, atomic_number).map_err(error)?;
    let n = r_slice.len();
    let two_c2 = 2.0 * c * c;
    let mut xi = Vec::with_capacity(n);
    for i in 0..n {
        let mut x = terms.dv[i] / (two_c2 * r_slice[i]);
        if mass_corrected {
            let m = (1.0 + (energy - pot_slice[i]) / two_c2).max(MASS_FLOOR);
            x /= m * m;
        }
        xi.push(x);
    }
    array(py, &[n], xi)
}

#[pyfunction]
#[pyo3(signature = (r, potential, energy, atomic_number=0.0, c=SPEED_OF_LIGHT))]
pub fn relativity_mass_factor<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    potential: PyReadonlyArray1<'py, f64>,
    energy: f64,
    atomic_number: f64,
    c: f64,
) -> PyResult<(Array<'py, f64>, Array<'py, f64>, Array<'py, f64>)> {
    layout(&r)?;
    layout(&potential)?;
    let r_slice = r.as_slice()?;
    let pot_slice = potential.as_slice()?;
    let terms = PotentialTerms::new(r_slice, pot_slice, atomic_number).map_err(error)?;
    let n = r_slice.len();
    let two_c2 = 2.0 * c * c;
    let mut m = Vec::with_capacity(n);
    let mut dm = Vec::with_capacity(n);
    let mut d2m = Vec::with_capacity(n);
    for i in 0..n {
        let val = (1.0 + (energy - pot_slice[i]) / two_c2).max(MASS_FLOOR);
        let d2v = -2.0 * atomic_number / (r_slice[i].powi(3)) + terms.d2_smooth[i];
        m.push(val);
        dm.push(-terms.dv[i] / two_c2);
        d2m.push(-d2v / two_c2);
    }
    Ok((array(py, &[n], m)?, array(py, &[n], dm)?, array(py, &[n], d2m)?))
}

#[pyfunction]
#[pyo3(signature = (r, potential, l, kappa, energy, atomic_number=0.0, c=SPEED_OF_LIGHT, treatment="scalar"))]
pub fn relativity_f_evaluate<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    potential: PyReadonlyArray1<'py, f64>,
    l: usize,
    kappa: i32,
    energy: f64,
    atomic_number: f64,
    c: f64,
    treatment: &str,
) -> PyResult<Array<'py, f64>> {
    layout(&r)?;
    layout(&potential)?;
    let r_slice = r.as_slice()?;
    let pot_slice = potential.as_slice()?;
    let tr = Treatment::parse(treatment).map_err(error)?;
    let ll = (l * (l + 1)) as f64;
    let n = r_slice.len();
    if tr == Treatment::None {
        let f: Vec<f64> = (0..n)
            .map(|i| ll / (r_slice[i] * r_slice[i]) + 2.0 * (pot_slice[i] - energy))
            .collect();
        return array(py, &[n], f);
    }
    let k = if tr == Treatment::Scalar {
        SCALAR_KAPPA
    } else {
        kappa
    };
    let terms = PotentialTerms::new(r_slice, pot_slice, atomic_number).map_err(error)?;
    let two_c2 = 2.0 * c * c;
    let out: Vec<f64> = (0..n)
        .map(|i| {
            let m = (1.0 + (energy - pot_slice[i]) / two_c2).max(MASS_FLOOR);
            let dm = -terms.dv[i] / two_c2;
            let bracket = -(1.0 + k as f64) * atomic_number / r_slice[i].powi(3)
                + 0.5 * terms.d2_smooth[i]
                - (k as f64) * terms.d_smooth[i] / r_slice[i];
            let darwin = bracket / (two_c2 * m);
            ll / (r_slice[i] * r_slice[i])
                + darwin
                + 3.0 * dm * dm / (4.0 * m * m)
                + 2.0 * m * (pot_slice[i] - energy)
        })
        .collect();
    array(py, &[n], out)
}

#[pyfunction]
#[pyo3(signature = (points, r_max))]
pub fn uniform_grid<'py>(
    py: Python<'py>,
    points: usize,
    r_max: f64,
) -> PyResult<Array<'py, f64>> {
    let step = r_max / (points as f64 + 1.0);
    let mut out = Vec::with_capacity(points);
    for i in 1..=points {
        out.push(i as f64 * step);
    }
    array(py, &[points], out)
}

/// `kappa` of the `(l, j)` subshell (`models::relativity::kappa_of`).
#[pyfunction]
pub fn relativity_kappa_of(l: usize, j: f64) -> PyResult<i32> {
    relativity::kappa_of(l, j).map_err(error)
}

/// The orbital angular momentum of a Dirac `kappa`.
#[pyfunction]
pub fn relativity_l_of(kappa: i32) -> PyResult<usize> {
    relativity::l_of(kappa).map_err(error)
}

/// `j = |kappa| - 1/2`.
#[pyfunction]
pub fn relativity_j_of(kappa: i32) -> f64 {
    relativity::j_of(kappa)
}

/// `2j + 1`.
#[pyfunction]
pub fn relativity_degeneracy(kappa: i32) -> usize {
    relativity::degeneracy(kappa)
}

/// The `kappa` values of a subshell, by increasing `j`.
#[pyfunction]
pub fn relativity_kappa_values(l: usize) -> Vec<i32> {
    relativity::kappa_values(l)
}

/// `2 / (2l + 1) (eps_{l+1/2} - eps_{l-1/2})` of `{kappa: eps}`; zero for an s shell.
#[pyfunction]
pub fn relativity_spin_orbit_difference(values: &Bound<'_, PyDict>, l: usize) -> PyResult<f64> {
    if l == 0 {
        return Ok(0.);
    }
    let get = |kappa: i64| -> PyResult<f64> {
        values
            .get_item(kappa)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(kappa))?
            .extract::<f64>()
    };
    Ok(relativity::spin_orbit_difference(l, get(-(l as i64 + 1))?, get(l as i64)?))
}

/// The valence subshells `(n, l)` of a configuration `[(n, l, occupation)]`.
#[pyfunction]
pub fn valence_subshells(configuration: Vec<(usize, usize, f64)>) -> Vec<(usize, usize)> {
    configuration::valence_subshells(&configuration)
}

#[pyfunction]
pub fn relativity_j_average(values: &Bound<'_, PyDict>, l: usize) -> PyResult<f64> {
    if l == 0 {
        let minus_one = (-1i64).into_pyobject(values.py())?;
        let v = values.get_item(minus_one)?.ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err("missing kappa=-1 for l=0 in j_average")
        })?;
        return v.extract::<f64>();
    }
    let k_down = (l as i64).into_pyobject(values.py())?;
    let k_up = (-(l as i64 + 1)).into_pyobject(values.py())?;
    let down_val = values.get_item(k_down)?.ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!("missing kappa={} for l={} in j_average", l, l))
    })?.extract::<f64>()?;
    let up_val = values.get_item(k_up)?.ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!("missing kappa=-{} for l={} in j_average", l + 1, l))
    })?.extract::<f64>()?;
    let l_f = l as f64;
    Ok((l_f * down_val + (l_f + 1.0) * up_val) / (2.0 * l_f + 1.0))
}

#[pyfunction]
pub fn relativity_large_component<'py>(
    py: Python<'py>,
    w: PyReadonlyArray1<'py, f64>,
    m: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    layout(&w)?;
    layout(&m)?;
    let w_s = w.as_slice()?;
    let m_s = m.as_slice()?;
    if w_s.len() != m_s.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "w and M must have the same length",
        ));
    }
    let n = w_s.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(w_s[i] * m_s[i].sqrt());
    }
    array(py, &[n], out)
}

#[pyfunction]
#[pyo3(signature = (r, p, kappa, m, c=SPEED_OF_LIGHT))]
pub fn relativity_small_component<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    kappa: i32,
    m: PyReadonlyArray1<'py, f64>,
    c: f64,
) -> PyResult<Array<'py, f64>> {
    layout(&r)?;
    layout(&p)?;
    layout(&m)?;
    let r_s = r.as_slice()?;
    let p_s = p.as_slice()?;
    let m_s = m.as_slice()?;
    let n = r_s.len();
    if p_s.len() != n || m_s.len() != n || n < 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r, P, and M must have matching length >= 3",
        ));
    }
    let mut dp = vec![0.0; n];
    let h1_0 = r_s[1] - r_s[0];
    let h2_0 = r_s[2] - r_s[1];
    dp[0] = -(2.0 * h1_0 + h2_0) / (h1_0 * (h1_0 + h2_0)) * p_s[0]
        + (h1_0 + h2_0) / (h1_0 * h2_0) * p_s[1]
        - h1_0 / (h2_0 * (h1_0 + h2_0)) * p_s[2];
    for i in 1..(n - 1) {
        let h1 = r_s[i] - r_s[i - 1];
        let h2 = r_s[i + 1] - r_s[i];
        dp[i] = (h1 * h1 * p_s[i + 1] + (h2 * h2 - h1 * h1) * p_s[i] - h2 * h2 * p_s[i - 1])
            / (h1 * h2 * (h1 + h2));
    }
    let h1_n = r_s[n - 1] - r_s[n - 2];
    let h2_n = r_s[n - 2] - r_s[n - 3];
    dp[n - 1] = (2.0 * h1_n + h2_n) / (h1_n * (h1_n + h2_n)) * p_s[n - 1]
        - (h1_n + h2_n) / (h1_n * h2_n) * p_s[n - 2]
        + h1_n / (h2_n * (h1_n + h2_n)) * p_s[n - 3];

    let two_c = 2.0 * c;
    let k = kappa as f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let f = (dp[i] + k * p_s[i] / r_s[i]) / (two_c * m_s[i]);
        out.push(f);
    }
    array(py, &[n], out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(radial_hartree, m)?)?;
    m.add_function(wrap_pyfunction!(radial_xc_potential, m)?)?;
    m.add_function(wrap_pyfunction!(radial_derivative_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_potential_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_darwin_spin_orbit, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_hydrogenic_dirac_energy, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_spin_orbit_radial, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_mass_factor, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_f_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_grid, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_j_average, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_kappa_of, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_l_of, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_j_of, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_degeneracy, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_kappa_values, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_spin_orbit_difference, m)?)?;
    m.add_function(wrap_pyfunction!(valence_subshells, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_large_component, m)?)?;
    m.add_function(wrap_pyfunction!(relativity_small_component, m)?)?;
    Ok(())
}
