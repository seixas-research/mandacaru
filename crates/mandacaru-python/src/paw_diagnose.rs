//! Bindings of the PAW dataset diagnostics. A record comes in the layout of a dataset file
//! ([`super::paw_record`]); every check runs in `mandacaru_core::paw::diagnose`. A check that
//! cannot be evaluated (a level or phase that is not finite) is returned as data
//! (`unevaluated`), never as a pass.
use super::{array, atom, copy, error, paw_record, Array, PyContext};
use mandacaru_core::paw::diagnose::{
    self,
    reconstruct::reconstruct_ae,
    reference::AeLogs,
    scattering::log_derivative,
    spectrum::{eigenvalues, extent, prepared},
    view::{Key, Smooth},
    Diagnosis,
};
use numpy::PyReadonlyArray1;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

fn label(key: &Key) -> String {
    match key.kappa {
        None => key.l.to_string(),
        Some(kappa) => format!("{}:{kappa}", key.l),
    }
}

fn diagnosis_dict<'py>(py: Python<'py>, d: &Diagnosis) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("ghosts", d.ghosts.clone())?;
    let scalar = PyDict::new(py);
    let branch = PyDict::new(py);
    for (key, phase) in &d.phases {
        match key.kappa {
            None => scalar.set_item(key.l, phase.to_vec())?,
            Some(_) => branch.set_item(label(key), phase.to_vec())?,
        }
    }
    out.set_item("phases", scalar)?;
    out.set_item("branch_phases", branch)?;
    out.set_item("unprojected", d.unprojected.clone())?;
    out.set_item("s_miss", d.s_miss)?;
    out.set_item("missing", d.missing.clone())?;
    let ghosts = PyDict::new(py);
    for (key, depth) in &d.branch_ghosts {
        ghosts.set_item(label(key), depth)?;
    }
    out.set_item("branch_ghosts", ghosts)?;
    out.set_item(
        "wrong_phases",
        d.wrong_phases()
            .into_iter()
            .map(|(l, p)| (l, p.to_vec()))
            .collect::<std::collections::BTreeMap<_, _>>(),
    )?;
    let defects = d.defects();
    let record = PyDict::new(py);
    record.set_item("ghosts", defects.ghosts)?;
    record.set_item(
        "phases",
        defects
            .phases
            .into_iter()
            .map(|(l, p)| (l, p.to_vec()))
            .collect::<std::collections::BTreeMap<_, _>>(),
    )?;
    record.set_item("s_miss", defects.s_miss)?;
    record.set_item("unprojected", defects.unprojected)?;
    record.set_item("missing", defects.missing)?;
    record.set_item("branch_ghosts", defects.branch_ghosts)?;
    out.set_item("defects", record)?;
    Ok(out)
}

/// Every check of a dataset against the all-electron atom solved again from the record: ghosts,
/// scattering phases, unprojected channels, the intruding-1s miss, missing states and the
/// ghosts of the `j` branches. With `levels`, the levels the ghost test reads for each channel.
#[pyfunction]
#[pyo3(signature = (context, payload, tables, *, levels = false))]
fn paw_diagnose<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    payload: &str,
    tables: &Bound<'py, PyDict>,
    levels: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = &context.inner;
    let dataset = paw_record::import(ctx, payload, tables)?;
    let found = py.detach(|| diagnose::diagnose_record(ctx, &dataset, None, &mut AeLogs::new()));
    let out = match found {
        Ok(diagnosis) => diagnosis_dict(py, &diagnosis)?,
        Err(diagnose::Error::Unevaluated(text)) => {
            let out = PyDict::new(py);
            out.set_item("unevaluated", text)?;
            return Ok(out);
        }
        Err(diagnose::Error::Failed(failure)) => return Err(error(failure)),
    };
    if levels {
        out.set_item("levels", ghost_levels(py, ctx, &dataset)?)?;
    }
    Ok(out)
}

/// The levels the ghost test reads for each channel (two, three when a channel holds two occupied
/// references), `{l: [eps_0, eps_1, ...]}`.
fn ghost_levels<'py>(
    py: Python<'py>,
    ctx: &mandacaru_core::Context,
    dataset: &mandacaru_core::paw::record::PawDataset,
) -> PyResult<Bound<'py, PyDict>> {
    let all = py
        .detach(|| -> mandacaru_core::Result<Vec<(usize, Vec<f64>)>> {
            let smooth = Smooth::scalar(ctx, dataset)?;
            smooth
                .angular_momenta()
                .into_iter()
                .map(|l| {
                    let channel = smooth.channel(l).expect("listed channel");
                    Ok((l, diagnose::spectrum::ghost_levels(ctx, &smooth, channel)?))
                })
                .collect()
        })
        .map_err(error)?;
    let spectra = PyDict::new(py);
    for (l, values) in all {
        spectra.set_item(l, values)?;
    }
    Ok(spectra)
}

/// The ghost-test levels of every channel of a record (no all-electron atom needed).
#[pyfunction]
fn paw_ghost_levels<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    payload: &str,
    tables: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = &context.inner;
    ghost_levels(py, ctx, &paw_record::import(ctx, payload, tables)?)
}

fn smooth_of(
    ctx: &mandacaru_core::Context,
    payload: &str,
    tables: &Bound<'_, PyDict>,
) -> PyResult<Smooth> {
    let dataset = paw_record::import(ctx, payload, tables)?;
    Smooth::scalar(ctx, &dataset).map_err(error)
}

fn channel_of(smooth: &Smooth, l: usize) -> PyResult<&diagnose::view::Channel> {
    smooth
        .channel(l)
        .ok_or_else(|| PyValueError::new_err(format!("the dataset has no channel l={l}")))
}

/// The lowest `states` eigenvalues of the generalized radial problem of channel `l`
/// (Richardson-extrapolated), on the box the channel's occupied waves ask for or `r_max`.
#[pyfunction]
#[pyo3(signature = (context, payload, tables, l, states, *, r_max = None))]
fn paw_levels(
    context: PyRef<'_, PyContext>,
    py: Python<'_>,
    payload: &str,
    tables: &Bound<'_, PyDict>,
    l: usize,
    states: usize,
    r_max: Option<f64>,
) -> PyResult<Vec<f64>> {
    let ctx = &context.inner;
    let smooth = smooth_of(ctx, payload, tables)?;
    let channel = channel_of(&smooth, l)?;
    py.detach(|| {
        let operator = prepared(ctx, &smooth, channel)?;
        let r_max = match r_max {
            Some(r) => r,
            None => extent(&smooth, channel)?,
        };
        eigenvalues(ctx, &operator, r_max, states)
    })
    .map_err(error)
}

/// How much of a hydrogen 1s entering the sphere the s projectors miss (a fraction of its norm),
/// `None` for a record without an s channel; needs no all-electron atom.
#[pyfunction]
fn paw_intruding_s_miss(
    context: PyRef<'_, PyContext>,
    payload: &str,
    tables: &Bound<'_, PyDict>,
) -> PyResult<Option<f64>> {
    let smooth = smooth_of(&context.inner, payload, tables)?;
    Ok(diagnose::intruding::intruding_s_miss(&smooth))
}

/// The all-electron atom a dataset was built on, solved again from its record (element,
/// functional, treatment, the recipe's grid and tolerances), in the dict of `Context.atom`.
#[pyfunction]
fn paw_reference_atom<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    payload: &str,
    tables: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = &context.inner;
    let dataset = paw_record::import(ctx, payload, tables)?;
    let solved = py
        .detach(|| diagnose::reference::reference_atom(ctx, &dataset))
        .map_err(error)?;
    atom::describe(py, dataset.atomic_number as u8, &solved)
}

/// The radius of the box the spectrum of channel `l` is solved in: where the outermost occupied
/// smooth wave has decayed to its floor.
#[pyfunction]
fn paw_spectrum_box(
    context: PyRef<'_, PyContext>,
    payload: &str,
    tables: &Bound<'_, PyDict>,
    l: usize,
) -> PyResult<f64> {
    let smooth = smooth_of(&context.inner, payload, tables)?;
    extent(&smooth, channel_of(&smooth, l)?).map_err(error)
}

/// The smooth logarithmic derivative of channel `l` at each of `energies`; a channel without
/// projectors sees the local potential alone and needs `r_cut`.
#[pyfunction]
#[pyo3(signature = (context, payload, tables, l, energies, *, r_cut = None))]
fn paw_log_derivatives<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    payload: &str,
    tables: &Bound<'py, PyDict>,
    l: usize,
    energies: Vec<f64>,
    r_cut: Option<f64>,
) -> PyResult<Array<'py, f64>> {
    let ctx = &context.inner;
    let smooth = smooth_of(ctx, payload, tables)?;
    let channel = smooth.channel(l);
    let values = py
        .detach(|| {
            energies
                .iter()
                .map(|&energy| log_derivative(ctx, &smooth, l, channel, energy, r_cut))
                .collect::<mandacaru_core::Result<Vec<f64>>>()
        })
        .map_err(error)?;
    array(py, &[values.len()], values)
}

/// Bloechl's transformation of the smooth radial function `u` on `r` back to all-electron.
#[pyfunction]
fn paw_reconstruct<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    payload: &str,
    tables: &Bound<'py, PyDict>,
    l: usize,
    r: PyReadonlyArray1<'py, f64>,
    u: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    let ctx = &context.inner;
    let (r, _r) = copy(ctx, &r)?;
    let (u, _u) = copy(ctx, &u)?;
    let smooth = smooth_of(ctx, payload, tables)?;
    let channel = channel_of(&smooth, l)?;
    let out = py
        .detach(|| reconstruct_ae(ctx, &smooth, channel, &r, &u))
        .map_err(error)?;
    array(py, &[out.len()], out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(paw_diagnose, m)?)?;
    m.add_function(wrap_pyfunction!(paw_levels, m)?)?;
    m.add_function(wrap_pyfunction!(paw_ghost_levels, m)?)?;
    m.add_function(wrap_pyfunction!(paw_intruding_s_miss, m)?)?;
    m.add_function(wrap_pyfunction!(paw_spectrum_box, m)?)?;
    m.add_function(wrap_pyfunction!(paw_reference_atom, m)?)?;
    m.add_function(wrap_pyfunction!(paw_log_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(paw_reconstruct, m)?)?;
    Ok(())
}
