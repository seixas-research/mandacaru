//! Typed reference-atom boundaries; the SCF, the radial equations and the configuration
//! selection all execute in `mandacaru_core::models`.
use super::{array, copy, error, Array, PyContext};
use mandacaru_core::models::{
    atom::{self, AtomModel, AtomOptions},
    relativity::{self, Treatment},
    selection, xc,
};
use numpy::PyReadonlyArray1;
use pyo3::{prelude::*, types::PyDict};

pub fn solve<'py>(
    context: &PyContext,
    py: Python<'py>,
    atomic_number: u8,
    options: AtomOptions,
    model: AtomModel,
    configuration: Option<Vec<(usize, usize, f64)>>,
    confinement: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let wall = confinement
        .as_ref()
        .map(|v| copy(&context.inner, v))
        .transpose()?;
    let points = options.points;
    let atom = py
        .detach(|| {
            atom::solve_model(
                &context.inner,
                atomic_number,
                &options,
                &model,
                configuration.as_deref(),
                wall.as_ref().map(|v| v.0.as_slice()),
            )
        })
        .map_err(error)?;
    let out = PyDict::new(py);
    out.set_item("atomic_number", atomic_number)?;
    for (name, values) in [
        ("r", &atom.r),
        ("density", &atom.density),
        ("v_effective", &atom.effective),
        ("v_hartree", &atom.hartree),
        ("v_xc", &atom.xc),
    ] {
        out.set_item(name, array(py, &[points], values.as_slice().to_vec())?)?;
    }
    let orbitals = PyDict::new(py);
    let eigenvalues = PyDict::new(py);
    let occupations = PyDict::new(py);
    for (i, &(n, l, q)) in atom.occupations.iter().enumerate() {
        orbitals.set_item(
            (n, l),
            array(py, &[points], atom.orbitals[i].as_slice().to_vec())?,
        )?;
        eigenvalues.set_item((n, l), atom.eigenvalues[i])?;
        occupations.set_item((n, l), q)?;
    }
    let orbitals_j = PyDict::new(py);
    let eigenvalues_j = PyDict::new(py);
    let occupations_j = PyDict::new(py);
    for level in &atom.j_states {
        let key = (level.n, level.l, level.kappa);
        orbitals_j.set_item(
            key,
            array(py, &[points], level.orbital.as_slice().to_vec())?,
        )?;
        eigenvalues_j.set_item(key, level.eigenvalue)?;
        occupations_j.set_item(key, level.occupation)?;
    }
    out.set_item("orbitals", orbitals)?;
    out.set_item("eigenvalues", eigenvalues)?;
    out.set_item("occupations", occupations)?;
    out.set_item("orbitals_j", orbitals_j)?;
    out.set_item("eigenvalues_j", eigenvalues_j)?;
    out.set_item("occupations_j", occupations_j)?;
    out.set_item("total_energy", atom.energy)?;
    out.set_item("converged", atom.converged)?;
    out.set_item("iterations", atom.iterations)?;
    out.set_item("functional", atom.functional.as_str())?;
    out.set_item("relativity", atom.relativity.name())?;
    Ok(out)
}

pub fn relaxed_configuration(
    context: &PyContext,
    py: Python<'_>,
    atomic_number: u8,
    functional: &str,
    relativity: &str,
    r_max: f64,
    points: Option<usize>,
) -> PyResult<Vec<(usize, usize, f64)>> {
    let treatment = Treatment::parse(relativity).map_err(error)?;
    py.detach(|| {
        selection::relaxed_configuration(
            &context.inner,
            atomic_number,
            functional,
            treatment,
            r_max,
            points,
        )
    })
    .map_err(error)
}

pub fn solve_radial<'py>(
    context: &PyContext,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    potential: PyReadonlyArray1<'py, f64>,
    l: usize,
    n_nodes: usize,
    relativity: &str,
    kappa: Option<i32>,
    atomic_number: f64,
    energy_guess: Option<f64>,
) -> PyResult<(Array<'py, f64>, f64)> {
    let treatment = Treatment::parse(relativity).map_err(error)?;
    let (r, _r) = copy(&context.inner, &r)?;
    let (potential, _v) = copy(&context.inner, &potential)?;
    let (u, energy) = py
        .detach(|| {
            relativity::solve_radial(
                &context.inner,
                &r,
                &potential,
                l,
                n_nodes,
                treatment,
                kappa,
                atomic_number,
                energy_guess,
                None,
            )
        })
        .map_err(error)?;
    Ok((array(py, &[u.len()], u)?, energy))
}

pub fn lda_terms<'py>(
    context: &PyContext,
    py: Python<'py>,
    density: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    let (density, _d) = copy(&context.inner, &density)?;
    let n = density.len();
    let mut out = vec![0.; 4 * n];
    py.detach(|| {
        for (i, rho) in density.iter().enumerate() {
            let (ex, vx, ec, vc) = xc::lda_terms(*rho);
            out[i] = ex;
            out[n + i] = vx;
            out[2 * n + i] = ec;
            out[3 * n + i] = vc;
        }
    });
    array(py, &[4, n], out)
}
