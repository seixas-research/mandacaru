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

pub(crate) fn diagnosis_dict<'py>(py: Python<'py>, d: &Diagnosis) -> PyResult<Bound<'py, PyDict>> {
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
/// scattering phases, unprojected channels, the intruding-1s miss, missing states, the
/// ghosts of the `j` branches and the bound levels a channel without projectors misses. With `levels`, the levels the ghost test reads for each channel.
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
    let found =
        py.detach(|| diagnose::diagnose_record_with_unprojected(ctx, &dataset, &mut AeLogs::new()));
    let out = match found {
        Ok((diagnosis, unprojected_missing)) => {
            let out = diagnosis_dict(py, &diagnosis)?;
            out.set_item("unprojected_missing", unprojected_missing)?;
            out
        }
        Err(diagnose::Error::Unevaluated(text)) => {
            let out = PyDict::new(py);
            out.set_item("unevaluated", text)?;
            return Ok(out);
        }
        Err(diagnose::Error::Failed(failure)) => return Err(error(failure)),
    };
    if levels {
        let (spectra, errors) = ghost_levels_and_errors(py, ctx, &dataset)?;
        out.set_item("levels", spectra)?;
        out.set_item("level_errors", errors)?;
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
    Ok(ghost_levels_and_errors(py, ctx, dataset)?.0)
}

/// `({l: levels}, {l: eps_0 - reference energy 0})`: the levels the ghost test reads and how far
/// the lowest one lies from the channel's first reference energy.
fn ghost_levels_and_errors<'py>(
    py: Python<'py>,
    ctx: &mandacaru_core::Context,
    dataset: &mandacaru_core::paw::record::PawDataset,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let all = py
        .detach(|| -> mandacaru_core::Result<Vec<(usize, Vec<f64>, f64)>> {
            let smooth = Smooth::scalar(ctx, dataset)?;
            smooth
                .angular_momenta()
                .into_iter()
                .map(|l| {
                    let channel = smooth.channel(l).expect("listed channel");
                    let levels = diagnose::spectrum::ghost_levels(ctx, &smooth, channel)?;
                    let error = levels[0] - channel.energies[0];
                    Ok((l, levels, error))
                })
                .collect()
        })
        .map_err(error)?;
    let (spectra, errors) = (PyDict::new(py), PyDict::new(py));
    for (l, values, difference) in all {
        spectra.set_item(l, values)?;
        errors.set_item(l, difference)?;
    }
    Ok((spectra, errors))
}

/// The radius beyond which each channel's projectors vanish and its smooth wave obeys the
/// all-electron equation again, `{l: radius}`: the channel's own cutoff or the local potential's
/// when that is larger (`view::Channel::projector_radius`).
#[pyfunction]
fn paw_projector_radii<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    payload: &str,
    tables: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let smooth = smooth_of(&context.inner, payload, tables)?;
    let radii = PyDict::new(py);
    for l in smooth.angular_momenta() {
        let channel = channel_of(&smooth, l)?;
        radii.set_item(l, channel.projector_radius(smooth.r_cut_local))?;
    }
    Ok(radii)
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

/// The verification gates of `diagnose::quality` that need no all-electron atom: the residual
/// kinetic energy of each channel above `q_cut` and the hard channels, the grid residual at each
/// gate spacing and the working spacing, the widest sphere against `bond` (Bohr; by default the
/// element's own bond to hydrogen, `quality::shortest_bond`), and the
/// partition against a `twin` record of the same element (its `(payload, tables)`).
#[pyfunction]
#[pyo3(signature = (context, payload, tables, *, bond = None, twin = None))]
fn paw_quality<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    payload: &str,
    tables: &Bound<'py, PyDict>,
    bond: Option<f64>,
    twin: Option<(String, Bound<'py, PyDict>)>,
) -> PyResult<Bound<'py, PyDict>> {
    use mandacaru_core::paw::diagnose::quality;
    let ctx = &context.inner;
    let dataset = paw_record::import(ctx, payload, tables)?;
    let other = match &twin {
        Some((p, t)) => Some(paw_record::import(ctx, p, t)?),
        None => None,
    };
    let (residuals, grids, working) = py
        .detach(|| -> mandacaru_core::Result<_> {
            let grids = quality::GATE_SPACINGS
                .iter()
                .map(|&h| Ok((h, quality::grid_residual(ctx, &dataset, h)?)))
                .collect::<mandacaru_core::Result<Vec<_>>>()?;
            Ok((
                quality::residual_kinetic(&dataset),
                grids,
                quality::working_spacing(ctx, &dataset)?,
            ))
        })
        .map_err(error)?;
    let out = PyDict::new(py);
    out.set_item("residual_kinetic", residuals)?;
    out.set_item("hard", quality::hardness(&dataset))?;
    let grid = PyDict::new(py);
    for (h, values) in grids {
        grid.set_item(h, values)?;
    }
    out.set_item("grid_residual", grid)?;
    out.set_item("working_spacing", working)?;
    let sphere = match bond {
        Some(b) => quality::sphere_reach(&dataset, b),
        None => quality::sphere_against_bond(&dataset).map(|(widest, _)| widest),
    };
    out.set_item("sphere", sphere)?;
    out.set_item(
        "partition",
        other
            .as_ref()
            .and_then(|o| quality::partition_mismatch(&dataset, o)),
    )?;
    Ok(out)
}

/// The `j` splitting of every bound occupied reference of every spin-orbit channel of a Dirac
/// record through the engine's isolated-atom SCF (`paw::engine_spin_orbit`) next to the all-electron
/// Dirac one of the record: a list of `{"l", "state", "engine", "dirac", "width", "passes"}`
/// (Hartree; `width` (eV, optional) restricts the SCF to that one Fermi-Dirac width instead of the
/// ladder; `engine` is `None` when the atom shows no pair of multiplets for the shell, `width`
/// the Fermi-Dirac width in eV of the SCF). `passes` is gate G13 (2 percent or 10 meV).
#[pyfunction]
#[pyo3(signature = (context, payload, tables, *, width = None))]
fn paw_spin_orbit_splittings<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    payload: &str,
    tables: &Bound<'py, PyDict>,
    width: Option<f64>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let ctx = &context.inner;
    let dataset = paw_record::import(ctx, payload, tables)?;
    let found = py
        .detach(|| mandacaru_core::paw::engine_spin_orbit::splittings_from(ctx, &dataset, width))
        .map_err(error)?;
    found
        .into_iter()
        .map(|s| {
            let row = PyDict::new(py);
            row.set_item("l", s.l)?;
            row.set_item("state", s.state)?;
            row.set_item("engine", s.engine)?;
            row.set_item("dirac", s.dirac)?;
            row.set_item("width", s.width)?;
            row.set_item("passes", s.passes())?;
            Ok(row)
        })
        .collect()
}

/// The levels of the isolated atom of a record through the engine's own SCF (`paw::atom_scf`):
/// `{"levels": [(energy in Hartree, occupation)] ascending, "width": the Fermi-Dirac width in eV of
/// the converged SCF, "energy": the total energy of the cell in Hartree}`, in a cubic box of
/// `box_edge` Angstrom. What gate G13 groups into `j` multiplets; two box edges give the energy per
/// volume the dataset's constants put into a crystal.
#[pyfunction]
#[pyo3(signature = (context, payload, tables, *, box_edge = 10.0))]
fn paw_atom_levels<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    payload: &str,
    tables: &Bound<'py, PyDict>,
    box_edge: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = &context.inner;
    let dataset = paw_record::import(ctx, payload, tables)?;
    let controls = mandacaru_core::paw::atom_scf::Controls {
        box_edge,
        ..mandacaru_core::paw::atom_scf::Controls::default()
    };
    let atom = py
        .detach(|| mandacaru_core::paw::atom_scf::levels(ctx, &dataset, &controls))
        .map_err(error)?;
    let out = PyDict::new(py);
    out.set_item(
        "levels",
        atom.levels
            .iter()
            .map(|l| (l.energy, l.occupation))
            .collect::<Vec<_>>(),
    )?;
    out.set_item("width", atom.width)?;
    out.set_item("energy", atom.energy)?;
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(paw_atom_levels, m)?)?;
    m.add_function(wrap_pyfunction!(paw_quality, m)?)?;
    m.add_function(wrap_pyfunction!(paw_diagnose, m)?)?;
    m.add_function(wrap_pyfunction!(paw_levels, m)?)?;
    m.add_function(wrap_pyfunction!(paw_ghost_levels, m)?)?;
    m.add_function(wrap_pyfunction!(paw_projector_radii, m)?)?;
    m.add_function(wrap_pyfunction!(paw_spin_orbit_splittings, m)?)?;
    m.add_function(wrap_pyfunction!(paw_intruding_s_miss, m)?)?;
    m.add_function(wrap_pyfunction!(paw_spectrum_box, m)?)?;
    m.add_function(wrap_pyfunction!(paw_reference_atom, m)?)?;
    m.add_function(wrap_pyfunction!(paw_log_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(paw_reconstruct, m)?)?;
    Ok(())
}
