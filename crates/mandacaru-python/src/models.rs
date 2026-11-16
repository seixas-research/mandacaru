//! Typed boundaries for scientific models shared by multiple solver families.
use super::{array, copy, error, layout, Array, PyContext};
use mandacaru_core::models::{
    configuration,
    relativity::{self, PotentialTerms, Treatment, SPEED_OF_LIGHT},
    spherical,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
#[pyo3(signature = (r, potential, atomic_number=0.0))]
pub fn relativity_potential_derivatives<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    potential: PyReadonlyArray1<'py, f64>,
    atomic_number: f64,
) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
    layout(&r)?;
    layout(&potential)?;
    let r = r.as_slice()?;
    let terms = PotentialTerms::new(r, potential.as_slice()?, atomic_number).map_err(error)?;
    let d2v = terms.second_derivative(r);
    Ok((array(py, &[r.len()], terms.dv)?, array(py, &[r.len()], d2v)?))
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
    let r = r.as_slice()?;
    let terms = PotentialTerms::new(r, potential.as_slice()?, atomic_number).map_err(error)?;
    array(py, &[r.len()], terms.darwin_spin_orbit_with(r, kappa, m.as_slice()?, c))
}

#[pyfunction]
#[pyo3(signature = (atomic_number, n, kappa, c=SPEED_OF_LIGHT))]
pub fn relativity_hydrogenic_dirac_energy(
    atomic_number: f64,
    n: usize,
    kappa: i32,
    c: f64,
) -> PyResult<f64> {
    relativity::hydrogenic_dirac_energy(atomic_number, n, kappa, c).map_err(error)
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
    let (r, potential) = (r.as_slice()?, potential.as_slice()?);
    let terms = PotentialTerms::new(r, potential, atomic_number).map_err(error)?;
    array(
        py,
        &[r.len()],
        relativity::spin_orbit_radial(r, potential, &terms, c, mass_corrected, energy),
    )
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
    let (r, potential) = (r.as_slice()?, potential.as_slice()?);
    let terms = PotentialTerms::new(r, potential, atomic_number).map_err(error)?;
    let (m, dm, d2m) = relativity::mass_terms(r, potential, energy, &terms, c);
    let n = r.len();
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
    let (r, potential) = (r.as_slice()?, potential.as_slice()?);
    let treatment = Treatment::parse(treatment).map_err(error)?;
    // The nonrelativistic coefficient reads no derivative of the potential.
    let terms = if treatment == Treatment::None {
        PotentialTerms {
            z: atomic_number,
            dv: vec![],
            d_smooth: vec![],
            d2_smooth: vec![],
        }
    } else {
        PotentialTerms::new(r, potential, atomic_number).map_err(error)?
    };
    array(
        py,
        &[r.len()],
        relativity::relativistic_f_with(r, potential, l, kappa, energy, &terms, treatment, c),
    )
}

#[pyfunction]
#[pyo3(signature = (points, r_max))]
pub fn uniform_grid<'py>(
    py: Python<'py>,
    points: usize,
    r_max: f64,
) -> PyResult<Array<'py, f64>> {
    array(py, &[points], mandacaru_core::math::radial::uniform_grid(points, r_max))
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
    let mut found = Vec::new();
    for kappa in relativity::kappa_values(l) {
        let value = values.get_item(kappa as i64)?.ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "missing kappa={kappa} for l={l} in j_average"
            ))
        })?;
        found.push((kappa, value.extract::<f64>()?));
    }
    Ok(relativity::j_average(l, |kappa| {
        found
            .iter()
            .find(|(k, _)| *k == kappa)
            .map_or(f64::NAN, |(_, v)| *v)
    }))
}

#[pyfunction]
pub fn relativity_large_component<'py>(
    py: Python<'py>,
    w: PyReadonlyArray1<'py, f64>,
    m: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    layout(&w)?;
    layout(&m)?;
    let out = relativity::large_component(w.as_slice()?, m.as_slice()?).map_err(error)?;
    array(py, &[out.len()], out)
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
    let out = relativity::small_component(r.as_slice()?, p.as_slice()?, kappa, m.as_slice()?, c)
        .map_err(error)?;
    array(py, &[out.len()], out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(radial_hartree, m)?)?;
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
