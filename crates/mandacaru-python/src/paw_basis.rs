//! The PAW-LCAO basis of one element (`paw::basis_recipe`), for the molecular route.
//!
//! The crystal setup calls the same recipe in Rust; a molecule hands its dataset record and its
//! recipe over here and receives the radial shells and what the recipe resolved (for the run log).
use super::{array, error, paw_record, PyContext};
use mandacaru_core::paw::basis_recipe::{self, Recipe};
use pyo3::{exceptions::PyValueError, prelude::*, types::{PyDict, PyList}};

fn recipe(text: &str) -> PyResult<Recipe> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Recipe::from_json(&v, None).map_err(error)
}

/// `{"shells": [(n, l, zeta, polarization, diffuse, r, values)], "confined": [...],
/// "polarization": {...} | None, "diffuse": [...]}`.
#[pyfunction]
fn paw_basis<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    payload: &str,
    tables: &Bound<'py, PyDict>,
    recipe_json: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = &context.inner;
    let dataset = paw_record::import(ctx, payload, tables)?;
    let recipe = recipe(recipe_json)?;
    let (shells, report) = py
        .detach(|| basis_recipe::species(ctx, &dataset, &recipe))
        .map_err(error)?;
    let out = PyDict::new(py);
    let list = PyList::empty(py);
    for shell in shells {
        let r = array(py, &[shell.r.len()], shell.r)?;
        let values = array(py, &[shell.values.len()], shell.values)?;
        list.append((shell.n, shell.l, shell.zeta, shell.polarization, shell.diffuse, r, values))?;
    }
    out.set_item("shells", list)?;
    let confined = PyList::empty(py);
    for c in report.confined {
        let entry = PyDict::new(py);
        entry.set_item("n", c.n)?;
        entry.set_item("l", c.l)?;
        entry.set_item("state", c.state)?;
        entry.set_item("r_c", c.orbital.r_c)?;
        entry.set_item("energy", c.orbital.energy)?;
        entry.set_item("free_energy", c.orbital.free_energy)?;
        // The realized shift (eV): the requested one, to the root search's tolerance.
        entry.set_item("achieved_shift", (c.orbital.energy - c.orbital.free_energy) * mandacaru_core::units::HARTREE_TO_EV)?;
        confined.append(entry)?;
    }
    out.set_item("confined", confined)?;
    let diffuse = PyList::empty(py);
    for w in report.diffuse {
        let entry = PyDict::new(py);
        for (key, value) in [("reference_rms", w.reference_rms), ("rms", w.rms), ("alpha", w.alpha),
                             ("r_cut", w.r_cut), ("residual", w.residual), ("r_max", w.r_max)] {
            entry.set_item(key, value)?;
        }
        for (key, value) in [("l", w.l), ("index", w.index), ("n", w.n), ("reference_l", w.reference_l)] {
            entry.set_item(key, value)?;
        }
        entry.set_item("accepted", w.accepted)?;
        diffuse.append(entry)?;
    }
    out.set_item("diffuse", diffuse)?;
    match report.polarization {
        None => out.set_item("polarization", py.None())?,
        Some(p) => {
            let entry = PyDict::new(py);
            entry.set_item("l", p.l)?;
            entry.set_item("base_l", p.base_l)?;
            entry.set_item("r_cut", p.r_cut)?;
            entry.set_item("r_char", p.r_char)?;
            entry.set_item("reference_radius", p.reference_radius)?;
            out.set_item("polarization", entry)?;
        }
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(paw_basis, m)?)?;
    Ok(())
}
