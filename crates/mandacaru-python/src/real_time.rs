//! Real-time propagation of a converged molecular reference after a delta kick, and the
//! time-to-frequency transforms of its trace: the request is copied in, Python is detached for
//! the run, and owned arrays come back. No numerics here; `mandacaru_core::molecular::realtime`
//! and `mandacaru_core::models::time_spectrum` compute every value (atomic units).
use super::{array, error, molecular::PyMeanField, periodic::PyPeriodicResult, Array};
use mandacaru_core::{
    math::propagator::Scheme,
    models::time_spectrum::{self as spectra, Kind, Window},
    molecular::realtime::{self, Gauge, Kick, Request},
    periodic::{bloch_mesh::Sampler, optics::realtime as crystal},
    Context,
};
use numpy::{Complex64, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyDict};

fn vectors<'py>(py: Python<'py>, rows: &[[f64; 3]]) -> PyResult<Array<'py, f64>> {
    array(
        py,
        &[rows.len(), 3],
        rows.iter().flatten().copied().collect(),
    )
}

/// Propagates `mean_field` after a kick of `strength` along `direction` for `duration` in steps
/// of `time_step`, in user units (1/Angstrom and femtoseconds; atomic with `atomic_units`).
/// Returns, in atomic units, `times_au`, `dipole_au` `(nt, 3)` (the
/// induced dipole, electrons only; the spin dipole for `kick="spin"`), `momentum_au` `(nt, 3)`
/// (the electrons' momentum `dE/dA`: the kinetic one plus a meta-GGA's exchange-correlation
/// current with a partial core), `energy_ha`, `norm_error` `(nt,)`, `ground_energy_ha`,
/// `kick_energy_ha`, `energy_drift` (kick energies), `width_ha` (the basis width `W` of the
/// reference Fock matrix), `largest_width_ha`, `step_width_bound` (the scheme's largest `dt W`),
/// `aliased_weight` (the bare weight above `pi / dt`), `builds`, `corrector_iterations`, and
/// the request.
#[pyfunction]
#[pyo3(signature=(mean_field, *, direction, strength, time_step, duration, atomic_units=false,
                  kick="density", gauge=None, propagator="etrs", corrector_tolerance=1e-6,
                  max_corrector_iterations=8, max_energy_drift=Some(0.05),
                  kernel="adiabatic", kpts=None))]
#[allow(clippy::too_many_arguments)]
pub fn molecular_real_time<'py>(
    py: Python<'py>,
    mean_field: &PyMeanField,
    direction: &Bound<'py, PyAny>,
    strength: f64,
    time_step: f64,
    duration: f64,
    atomic_units: bool,
    kick: &str,
    gauge: Option<&str>,
    propagator: &str,
    corrector_tolerance: f64,
    max_corrector_iterations: usize,
    max_energy_drift: Option<f64>,
    kernel: &str,
    kpts: Option<[usize; 3]>,
) -> PyResult<Bound<'py, PyDict>> {
    realtime::check_molecular_options(kernel, kpts).map_err(error)?;
    let direction = directions(direction)?.ok_or_else(|| {
        pyo3::exceptions::PyNotImplementedError::new_err(
            "direction='all' is offered for a crystal; a molecule is kicked along one direction \
             per run ('x', 'y', 'z' or a 3-vector)",
        )
    })?;
    let gauge = gauge.unwrap_or("length");
    let mut request =
        Request::from_user_units(direction, strength, time_step, duration, atomic_units)
            .map_err(error)?;
    request.kick = Kick::parse(kick).map_err(error)?;
    request.gauge = Gauge::parse(gauge).map_err(error)?;
    request.scheme = Scheme::parse(propagator).map_err(error)?;
    request.corrector_tolerance = corrector_tolerance;
    request.max_corrector_iterations = max_corrector_iterations;
    request.max_energy_drift = max_energy_drift;
    let (integrals, reference) = (mean_field.integrals.clone(), mean_field.inner.clone());
    let run = py
        .detach(|| realtime::propagate(&integrals, &reference, &request))
        .map_err(error)?;
    let nt = run.times.len();
    let out = PyDict::new(py);
    out.set_item("times_au", array(py, &[nt], run.times.clone())?)?;
    out.set_item("dipole_au", vectors(py, &run.dipole)?)?;
    out.set_item("momentum_au", vectors(py, &run.momentum)?)?;
    out.set_item("along_au", array(py, &[nt], run.along.clone())?)?;
    out.set_item("energy_ha", array(py, &[nt], run.energy.clone())?)?;
    out.set_item("norm_error", array(py, &[nt], run.norm_error.clone())?)?;
    out.set_item("ground_energy_ha", run.ground_energy)?;
    out.set_item("kick_energy_ha", run.kick_energy)?;
    out.set_item("energy_drift", run.energy_drift)?;
    out.set_item("width_ha", run.width)?;
    out.set_item("largest_width_ha", run.largest_width)?;
    out.set_item("step_width", run.step_width)?;
    out.set_item("worst_norm_error", run.worst_norm_error)?;
    out.set_item("step_width_bound", request.scheme.step_width_bound())?;
    out.set_item("aliased_weight", run.aliased_weight)?;
    out.set_item("builds", run.builds)?;
    out.set_item("corrector_iterations", run.corrector_iterations)?;
    out.set_item("functional", mean_field.inner.functional.clone())?;
    out.set_item("propagator", request.scheme.name())?;
    out.set_item("kick", request.kick.name())?;
    out.set_item("gauge", request.gauge.name())?;
    out.set_item("strength_au", request.strength)?;
    out.set_item("direction", request.direction.to_vec())?;
    out.set_item("time_step_au", request.time_step)?;
    out.set_item("steps", request.steps)?;
    out.set_item("corrector_tolerance", request.corrector_tolerance)?;
    out.set_item("effective_corrector_tolerance", run.corrector_tolerance)?;
    out.set_item("max_energy_drift", request.max_energy_drift)?;
    Ok(out)
}

/// The kick directions of a request: one 3-vector, or the three axes for `"all"`.
/// A direction as the calculator takes it: `"x"`, `"y"`, `"z"`, a 3-vector, or `"all"` (`None`:
/// the three axes).
fn directions(direction: &Bound<'_, PyAny>) -> PyResult<Option<[f64; 3]>> {
    if let Ok(name) = direction.extract::<String>() {
        return match name.as_str() {
            "x" => Ok(Some([1., 0., 0.])),
            "y" => Ok(Some([0., 1., 0.])),
            "z" => Ok(Some([0., 0., 1.])),
            "all" => Ok(None),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "direction must be 'x', 'y', 'z', 'all' or a 3-vector, not {name:?}"
            ))),
        };
    }
    direction.extract::<[f64; 3]>().map(Some).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "direction must be 'x', 'y', 'z', 'all' or a 3-vector of numbers",
        )
    })
}

/// Propagates the converged crystal `result` after a vector-potential kick `A_0 = strength`
/// along `direction` (velocity gauge; `"all"`: one run along each axis) for `duration` in steps
/// of `time_step`, in user units (1/Angstrom and femtoseconds; atomic with `atomic_units`), on
/// the whole Gamma-anchored mesh `kpts` (`None`: the SCF's). Returns, in atomic units,
/// `times_au`, `current_au` `(nt, 3)` (the macroscopic current density), `along_au` and
/// `corrected_au` (along the field; the second without the static deficit
/// `A (N - N_dia) / Omega`), `energy_ha`, `norm_error`; with `"all"` the run along axis `b` is
/// column `b`: `current_au` and `corrected_au` `(nt, 3, 3)` (`[t, a, b]`, the tensor),
/// `along_au`, `energy_ha` and `norm_error` `(nt, 3)`, and the scalar checks are the worst of
/// the three. Also `ground_energy_ha`, `kick_energy_ha`, `energy_drift`, `width_ha`,
/// `largest_width_ha`, `step_width`, `step_width_bound`, `aliased_weight`, `builds`,
/// `corrector_iterations`, `electrons`, `diamagnetic` and `deficit` (one entry per run),
/// `diamagnetic_source` (`"band sum"` or `"static field"`: DFT+U and meta-GGAs, by
/// self-consistency at a fixed field), `momentum_residual`,
/// `volume_bohr3`,
/// `mesh`, `kpoints`, `initial_density_change`, and the request.
#[pyfunction]
#[pyo3(signature=(result, *, direction, strength, time_step, duration, atomic_units=false,
                  kick="density", gauge=None, kernel="adiabatic", propagator="etrs",
                  corrector_tolerance=1e-6, max_corrector_iterations=8,
                  max_energy_drift=Some(0.05), kpts=None, formulation="fixed"))]
#[allow(clippy::too_many_arguments)]
pub fn crystal_real_time<'py>(
    py: Python<'py>,
    result: &PyPeriodicResult,
    direction: &Bound<'py, PyAny>,
    strength: f64,
    time_step: f64,
    duration: f64,
    atomic_units: bool,
    kick: &str,
    gauge: Option<&str>,
    kernel: &str,
    propagator: &str,
    corrector_tolerance: f64,
    max_corrector_iterations: usize,
    max_energy_drift: Option<f64>,
    kpts: Option<[usize; 3]>,
    formulation: &str,
) -> PyResult<Bound<'py, PyDict>> {
    crystal::check_options(kick, gauge).map_err(error)?;
    let single = directions(direction)?;
    let setup = result.resolve_setup(py)?;
    let mut request = crystal::Request::from_user_units(
        single.unwrap_or([1., 0., 0.]),
        strength,
        time_step,
        duration,
        atomic_units,
    )
    .map_err(error)?;
    request.kernel = crystal::Kernel::parse(kernel).map_err(error)?;
    request.scheme = Scheme::parse(propagator).map_err(error)?;
    request.corrector_tolerance = corrector_tolerance;
    request.max_corrector_iterations = max_corrector_iterations;
    request.max_energy_drift = max_energy_drift;
    request.mesh = kpts;
    request.formulation = crystal::Formulation::parse(formulation).map_err(error)?;
    let (state, periodic) = (&*result.inner, &*result.crystal);
    let runs = py
        .detach(|| {
            let sampler = Sampler::new(periodic, state, &setup)?;
            match single {
                Some(_) => Ok(vec![crystal::propagate(&sampler, &request)?]),
                None => crystal::propagate_axes(&sampler, &request),
            }
        })
        .map_err(error)?;
    let runs = runs.as_slice();
    let run = &runs[0];
    let (nt, columns) = (run.times.len(), runs.len());
    let tensor = single.is_none();
    // Row-major `(nt, columns)` of one value per run, and `(nt, 3, columns)` of a vector.
    let table = |value: &dyn Fn(&crystal::CrystalRealTime, usize) -> f64| -> Vec<f64> {
        (0..nt)
            .flat_map(|t| runs.iter().map(move |r| value(r, t)))
            .collect()
    };
    let vectors_of = |value: &dyn Fn(&crystal::CrystalRealTime, usize) -> [f64; 3]| -> Vec<f64> {
        (0..nt)
            .flat_map(|t| (0..3).flat_map(move |a| runs.iter().map(move |r| value(r, t)[a])))
            .collect()
    };
    let shape = |inner: &[usize]| -> Vec<usize> {
        let mut out = vec![nt];
        out.extend_from_slice(inner);
        if tensor {
            out.push(columns);
        }
        out
    };
    let worst = |value: &dyn Fn(&crystal::CrystalRealTime) -> f64| {
        runs.iter().map(value).fold(f64::NEG_INFINITY, f64::max)
    };
    let out = PyDict::new(py);
    out.set_item("times_au", array(py, &[nt], run.times.clone())?)?;
    out.set_item(
        "current_au",
        array(py, &shape(&[3]), vectors_of(&|r, t| r.current[t]))?,
    )?;
    out.set_item(
        "along_au",
        array(py, &shape(&[]), table(&|r, t| r.along[t]))?,
    )?;
    let corrected = if tensor {
        array(py, &shape(&[3]), vectors_of(&|r, t| r.corrected_current[t]))?
    } else {
        array(py, &[nt], run.corrected.clone())?
    };
    out.set_item("corrected_au", corrected)?;
    out.set_item(
        "energy_ha",
        array(py, &shape(&[]), table(&|r, t| r.energy[t]))?,
    )?;
    out.set_item(
        "norm_error",
        array(py, &shape(&[]), table(&|r, t| r.norm_error[t]))?,
    )?;
    out.set_item("ground_energy_ha", run.ground_energy)?;
    out.set_item("kick_energy_ha", worst(&|r| r.kick_energy))?;
    out.set_item("energy_drift", worst(&|r| r.energy_drift))?;
    out.set_item("width_ha", worst(&|r| r.width))?;
    out.set_item("largest_width_ha", worst(&|r| r.largest_width))?;
    out.set_item("step_width", worst(&|r| r.step_width))?;
    out.set_item("worst_norm_error", worst(&|r| r.worst_norm_error))?;
    out.set_item("step_width_bound", request.scheme.step_width_bound())?;
    out.set_item("aliased_weight", worst(&|r| r.aliased_weight))?;
    out.set_item("builds", runs.iter().map(|r| r.builds).sum::<usize>())?;
    out.set_item(
        "corrector_iterations",
        runs.iter().map(|r| r.corrector_iterations).max(),
    )?;
    out.set_item("electrons", run.electrons)?;
    out.set_item(
        "diamagnetic",
        runs.iter().map(|r| r.diamagnetic).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "deficit",
        runs.iter().map(|r| r.deficit).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "diamagnetic_source",
        if run.self_consistent {
            "static field"
        } else {
            "band sum"
        },
    )?;
    out.set_item("momentum_residual", worst(&|r| r.momentum_residual))?;
    out.set_item("volume_bohr3", run.volume)?;
    out.set_item("mesh", (run.mesh[0], run.mesh[1], run.mesh[2]))?;
    out.set_item("kpoints", run.kpoints)?;
    out.set_item(
        "initial_density_change",
        worst(&|r| r.initial_density_change),
    )?;
    out.set_item("formulation", run.formulation.name())?;
    out.set_item(
        "static_current_au",
        runs.iter().map(|r| r.static_current).collect::<Vec<_>>(),
    )?;
    out.set_item("transfer_defect", worst(&|r| r.transfer_defect))?;
    out.set_item("functional", periodic.functional.clone())?;
    out.set_item("propagator", request.scheme.name())?;
    out.set_item("kernel", request.kernel.name())?;
    out.set_item("kick", "density")?;
    out.set_item("gauge", "velocity")?;
    out.set_item("strength_au", request.strength)?;
    let direction: Vec<f64> = match single {
        Some(_) => request.direction.to_vec(),
        None => (0..9).map(|i| if i % 4 == 0 { 1. } else { 0. }).collect(),
    };
    out.set_item("direction", direction)?;
    out.set_item("time_step_au", request.time_step)?;
    out.set_item("steps", request.steps)?;
    out.set_item("corrector_tolerance", request.corrector_tolerance)?;
    out.set_item("effective_corrector_tolerance", run.corrector_tolerance)?;
    out.set_item("max_energy_drift", request.max_energy_drift)?;
    Ok(out)
}

/// The response function `kind` (`"strength"`, `"polarizability"` or `"dielectric"`) of a trace
/// sampled in atomic units at `n time_step` after a kick of `strength` (atomic units), on
/// `frequencies` with `damping` in the user's energy unit (eV; Hartree with `atomic_units`),
/// in user units (`models::time_spectrum::spectrum_in_user_units`): the real `S` (1/eV), the
/// complex `alpha` (Angstrom^3) or `epsilon` (exponential window only); `current` says the
/// trace is a crystal's current (only `epsilon` takes it). Returns `(values, tail)`, `tail` the
/// window's weight on the last sample.
#[pyfunction]
#[pyo3(signature=(context, samples, frequencies, *, time_step, strength, damping,
                  window="exponential", kind="strength", atomic_units=false, current=false))]
#[allow(clippy::too_many_arguments)]
pub fn time_spectrum<'py>(
    py: Python<'py>,
    context: &super::PyContext,
    samples: PyReadonlyArrayDyn<'py, f64>,
    frequencies: PyReadonlyArray1<'py, f64>,
    time_step: f64,
    strength: f64,
    damping: f64,
    window: &str,
    kind: &str,
    atomic_units: bool,
    current: bool,
) -> PyResult<(Bound<'py, PyAny>, f64)> {
    let window = Window::parse(window).map_err(error)?;
    let kind = Kind::parse(kind).map_err(error)?;
    spectra::check_source(kind, current).map_err(error)?;
    let trailing: Vec<usize> = samples.shape().iter().skip(1).copied().collect();
    let columns = trailing.iter().product::<usize>();
    let (samples, frequencies) = (
        samples.as_array().iter().copied().collect::<Vec<f64>>(),
        frequencies.as_slice()?.to_vec(),
    );
    let ctx: Context = context.inner.clone();
    let mut shape = vec![frequencies.len()];
    shape.extend_from_slice(&trailing);
    let (values, tail) = py
        .detach(|| {
            spectra::columns_in_user_units(
                &ctx,
                kind,
                &samples,
                columns,
                time_step,
                strength,
                &frequencies,
                damping,
                window,
                atomic_units,
            )
        })
        .map_err(error)?;
    let out = if kind == Kind::Strength {
        array(py, &shape, values.iter().map(|v| v.re).collect())?.into_any()
    } else {
        array(
            py,
            &shape,
            values.iter().map(|v| Complex64::new(v.re, v.im)).collect(),
        )?
        .into_any()
    };
    Ok((out, tail))
}

/// The macroscopic dielectric function of the converged insulating crystal `result` at
/// `q -> 0` (`periodic::optics::dielectric`, T5a) at `energies + i damping`, every value in user
/// units (eV, 1/Angstrom; Hartree and 1/Bohr with `atomic_units`): `energies` (`None`: 0 to
/// 20 eV in 0.05 eV steps), `kernel` `"independent"` or `"hartree"` (the RPA with local fields
/// on `|G| <= cutoff`), `bands` `(valence, conduction)` per channel next to the gap (`None`:
/// the smallest window that holds every transition up to the highest energy and leaves out at
/// most `STATIC_TOLERANCE` of the independent-particle `eps(0) - 1`), `scissor` added to every
/// transition, `direction` a 3-vector (`None`: the
/// isotropic average), on the whole Gamma-anchored mesh `kpts` (`None`: the SCF's). Returns
/// `energies`, `tensor` `(n, 3, 3)` complex, `epsilon` `(n,)` complex along the direction,
/// `loss` (`-Im 1 / epsilon`), `absorption` (`2 omega Im sqrt(epsilon) / c`, 1/Angstrom),
/// `static_tensor` `(3, 3)` (`eps(0)` without damping), `sum_rule` `(3, 3)` (`N^{ab}` of the
/// first moment, the diamagnetic band sum without a scissor), and the record.
#[pyfunction]
#[pyo3(signature=(result, *, energies=None, damping=0.1, atomic_units=false, kernel="hartree",
                  bands=None, scissor=0.0, cutoff=None, direction=None, kpts=None, alpha=None))]
#[allow(clippy::too_many_arguments)]
pub fn crystal_dielectric_function<'py>(
    py: Python<'py>,
    result: &PyPeriodicResult,
    energies: Option<Vec<f64>>,
    damping: f64,
    atomic_units: bool,
    kernel: &str,
    bands: Option<(usize, usize)>,
    scissor: f64,
    cutoff: Option<f64>,
    direction: Option<&Bound<'py, PyAny>>,
    kpts: Option<[usize; 3]>,
    alpha: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    use mandacaru_core::{
        periodic::{bloch_mesh::MeshIndex, optics::dielectric as optics},
        units::{from_user, to_user, Quantity},
    };
    let direction = match direction.map(directions).transpose()? {
        Some(None) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "direction='all' is not an option of the dielectric function: the view always \
                 holds the whole tensor (direction=None gives the isotropic average)",
            ))
        }
        Some(single) => single,
        None => None,
    };
    let energy = |v: f64| from_user(v, Quantity::Energy, atomic_units);
    let frequencies = match energies {
        Some(values) => values.iter().map(|v| energy(*v)).collect(),
        None => optics::default_frequencies(),
    };
    let mut request = optics::Request::new(
        frequencies,
        energy(damping),
        optics::Kernel::parse(kernel).map_err(error)?,
    );
    request.window = bands;
    request.scissor = energy(scissor);
    request.alpha = alpha;
    if let Some(c) = cutoff {
        request.cutoff = from_user(c, Quantity::Wavenumber, atomic_units);
    }
    let setup = result.resolve_setup(py)?;
    let (state, periodic) = (&*result.inner, &*result.crystal);
    let started = std::time::Instant::now();
    let out = py
        .detach(|| {
            let size = match kpts {
                Some(size) => size,
                None => crystal::scf_mesh(&setup)?,
            };
            let sampler = Sampler::new(periodic, state, &setup)?;
            let mesh = MeshIndex::new(size, [0.; 3])?;
            // `bands=None`: the smallest window holding every transition up to the highest
            // requested energy and the independent-particle `eps(0)` to `STATIC_TOLERANCE`.
            let mut request = request.clone();
            if request.window.is_none() {
                let highest = request
                    .frequencies
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                request.window = Some(optics::default_window(
                    &sampler,
                    &mesh,
                    highest,
                    request.scissor,
                    optics::STATIC_TOLERANCE,
                )?);
            }
            optics::dielectric(&sampler, &mesh, &request)
        })
        .map_err(error)?;
    let along = out.along(direction).map_err(error)?;
    let n = out.frequencies.len();
    let user = |values: &[f64], quantity| to_user(values, quantity, atomic_units, 1).map_err(error);
    let complex = |values: Vec<mandacaru_core::quantum::C64>| -> Vec<Complex64> {
        values.iter().map(|v| Complex64::new(v.re, v.im)).collect()
    };
    let flat = |rows: &[[f64; 3]; 3]| rows.iter().flatten().copied().collect::<Vec<f64>>();
    let dict = PyDict::new(py);
    dict.set_item(
        "energies",
        array(py, &[n], user(&out.frequencies, Quantity::Energy)?)?,
    )?;
    dict.set_item(
        "tensor",
        array(
            py,
            &[n, 3, 3],
            complex(out.tensor.iter().flatten().flatten().copied().collect()),
        )?,
    )?;
    dict.set_item("epsilon", array(py, &[n], complex(along.clone()))?)?;
    dict.set_item("loss", array(py, &[n], optics::loss(&along))?)?;
    dict.set_item(
        "absorption",
        array(
            py,
            &[n],
            user(
                &optics::absorption(&out.frequencies, &along),
                Quantity::Wavenumber,
            )?,
        )?,
    )?;
    dict.set_item(
        "static_tensor",
        array(py, &[3, 3], flat(&out.static_tensor))?,
    )?;
    dict.set_item("sum_rule", array(py, &[3, 3], flat(&out.sum_rule))?)?;
    dict.set_item("damping", user(&[out.damping], Quantity::Energy)?[0])?;
    dict.set_item("scissor", user(&[out.scissor], Quantity::Energy)?[0])?;
    dict.set_item("onset", user(&[out.onset], Quantity::Energy)?[0])?;
    dict.set_item(
        "cutoff",
        (out.kernel != optics::Kernel::Independent)
            .then(|| user(&[out.cutoff], Quantity::Wavenumber).map(|v| v[0]))
            .transpose()?,
    )?;
    dict.set_item("kernel", out.kernel.name())?;
    dict.set_item("alpha", out.alpha)?;
    dict.set_item("direction", direction.map(|d| d.to_vec()))?;
    dict.set_item("mesh", (out.mesh[0], out.mesh[1], out.mesh[2]))?;
    dict.set_item("kpoints", out.mesh.iter().product::<usize>())?;
    dict.set_item("bands", out.window.clone())?;
    dict.set_item("transitions", out.transitions)?;
    dict.set_item("vectors", out.vectors)?;
    dict.set_item("electrons", out.electrons)?;
    dict.set_item("momentum_residual", out.residual)?;
    dict.set_item("functional", periodic.functional.clone())?;
    let unit = if atomic_units { "Ha" } else { "eV" };
    dict.set_item("energy_unit", unit)?;
    let wavenumber = if atomic_units { "bohr" } else { "angstrom" };
    let diagonal = |t: &[[f64; 3]; 3]| format!("{:.6} {:.6} {:.6}", t[0][0], t[1][1], t[2][2]);
    let mut record: Vec<(String, String)> = vec![
        ("kernel".into(), out.kernel.name().into()),
        (
            "reference".into(),
            format!("kohn-sham {}", periodic.functional),
        ),
        (
            "mesh".into(),
            format!("{} {} {}", out.mesh[0], out.mesh[1], out.mesh[2]),
        ),
        (
            "kpoints".into(),
            out.mesh.iter().product::<usize>().to_string(),
        ),
        ("symmetry".into(), "whole mesh".into()),
        (
            "bands_valence_conduction".into(),
            out.window
                .iter()
                .map(|(v, c)| format!("{v},{c}"))
                .collect::<Vec<_>>()
                .join(" "),
        ),
        ("transitions".into(), out.transitions.to_string()),
        ("local_field_vectors".into(), out.vectors.to_string()),
    ];
    if let Some(alpha) = out.alpha {
        record.push(("long_range_alpha".into(), format!("{alpha:.6}")));
    }
    if out.kernel != optics::Kernel::Independent {
        let c = user(&[out.cutoff], Quantity::Wavenumber)?[0];
        record.push((
            format!("local_field_cutoff_per_{wavenumber}"),
            format!("{c:.6}"),
        ));
    }
    let scalar =
        |v: f64| -> PyResult<String> { Ok(format!("{:.6}", user(&[v], Quantity::Energy)?[0])) };
    record.extend([
        (format!("damping_{unit}"), scalar(out.damping)?),
        (format!("scissor_{unit}"), scalar(out.scissor)?),
        (format!("onset_{unit}"), scalar(out.onset)?),
        ("frequencies".into(), n.to_string()),
        (
            "static_dielectric_diagonal".into(),
            diagonal(&out.static_tensor),
        ),
        ("sum_rule_diagonal".into(), diagonal(&out.sum_rule)),
        ("electrons".into(), format!("{:.6}", out.electrons)),
        ("momentum_residual".into(), format!("{:.3e}", out.residual)),
    ]);
    record.push((
        "wall_time_s".into(),
        format!("{:.2}", started.elapsed().as_secs_f64()),
    ));
    dict.set_item("record", record)?;
    Ok(dict)
}

/// The `num_states` lowest Bethe-Salpeter excitons of the converged insulating crystal `result`
/// (`periodic::optics::bse`, T5b) in the Tamm-Dancoff approximation (`approximation="tda"`) or
/// the coupling form (`"casida"`, dense: the resonant amplitudes `X` and the antiresonant `Y`,
/// the line weights from both), on the whole Gamma-anchored mesh `kpts` (`None`: the SCF's): `kernel`
/// `None` or `"bse"` (exchange and the statically screened direct term), `"hartree"` (exchange
/// only) or `"independent"`; `spin` `None` or `"singlet"`, or `"triplet"` (no exchange); `bands`
/// `(valence, conduction)` next to the gap (`None`: every band), `scissor` added to every
/// transition and `cutoff` the largest `|q + G|` of the exchange and of `W` (user units; default
/// 3 / Bohr). Davidson runs above the dense limit with residuals below `tolerance` in at most
/// `max_iterations` steps. Returns the energies, bindings against the continuum's onset, line
/// weights in `Im epsilon` per axis, participation ratios, k-resolved weights and dominant
/// transitions of the excitons, with the operator's record.
#[pyfunction]
#[pyo3(signature=(result, *, num_states=5, approximation="tda", spin=None, kernel=None,
                  tolerance=1e-5, max_iterations=200, atomic_units=false, bands=None,
                  scissor=0.0, cutoff=None, kpts=None))]
#[allow(clippy::too_many_arguments)]
pub fn crystal_excitations<'py>(
    py: Python<'py>,
    result: &PyPeriodicResult,
    num_states: usize,
    approximation: &str,
    spin: Option<&str>,
    kernel: Option<&str>,
    tolerance: f64,
    max_iterations: usize,
    atomic_units: bool,
    bands: Option<(usize, usize)>,
    scissor: f64,
    cutoff: Option<f64>,
    kpts: Option<[usize; 3]>,
) -> PyResult<Bound<'py, PyDict>> {
    use mandacaru_core::{
        periodic::{
            bloch_mesh::MeshIndex,
            optics::{bse, dielectric::Kernel},
        },
        units::{from_user, to_user, Quantity},
    };
    let coupling = match approximation {
        "tda" => false,
        "casida" => true,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown approximation {other:?}; expected 'tda' or 'casida'"
            )))
        }
    };
    let kernel = match kernel.unwrap_or("bse") {
        "adiabatic" => return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "kernel='adiabatic' on a crystal (the exchange-correlation kernel in the crystal's \
                 response) is not built: crystal excitons take kernel='bse' (the default), \
                 'hartree' or 'independent'",
        )),
        name => Kernel::parse(name).map_err(error)?,
    };
    let spin = bse::Spin::parse(spin.unwrap_or("singlet")).map_err(error)?;
    if !(tolerance.is_finite() && tolerance > 0.) || max_iterations == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tolerance must be positive and max_iterations at least 1",
        ));
    }
    let energy = |v: f64| from_user(v, Quantity::Energy, atomic_units);
    let mut request = bse::Request::new(kernel, spin);
    request.coupling = coupling;
    request.window = bands;
    request.scissor = energy(scissor);
    if let Some(c) = cutoff {
        request.cutoff = from_user(c, Quantity::Wavenumber, atomic_units);
    }
    let setup = result.resolve_setup(py)?;
    let (state, periodic) = (&*result.inner, &*result.crystal);
    let started = std::time::Instant::now();
    let (op, excitons) = py
        .detach(|| {
            let size = match kpts {
                Some(size) => size,
                None => crystal::scf_mesh(&setup)?,
            };
            let sampler = Sampler::new(periodic, state, &setup)?;
            let op = bse::operator(&sampler, &MeshIndex::new(size, [0.; 3])?, &request)?;
            let excitons = op.lowest(num_states, tolerance, max_iterations)?;
            Ok::<_, mandacaru_core::Error>((op, excitons))
        })
        .map_err(error)?;
    let roots = excitons.energies.len();
    let nk = op.mesh.iter().product::<usize>();
    let user = |values: &[f64]| to_user(values, Quantity::Energy, atomic_units, 1).map_err(error);
    let summary = op.analyze(&excitons);
    let flat = |rows: &[[f64; 3]; 3]| rows.iter().flatten().copied().collect::<Vec<f64>>();
    let dict = PyDict::new(py);
    dict.set_item("energies", array(py, &[roots], user(&excitons.energies)?)?)?;
    dict.set_item("binding", array(py, &[roots], user(&summary.binding)?)?)?;
    dict.set_item("onset", user(&[op.onset])?[0])?;
    dict.set_item("weights", array(py, &[roots, 3], user(&summary.areas)?)?)?;
    dict.set_item("participation", array(py, &[roots], summary.participation)?)?;
    dict.set_item(
        "kpoint_weights",
        array(py, &[roots, nk], summary.kpoint_weights)?,
    )?;
    dict.set_item("dominant", summary.dominant)?;
    dict.set_item("dominant_channels", summary.dominant_channels)?;
    dict.set_item("spin", spin.name())?;
    dict.set_item("kernel", kernel.name())?;
    dict.set_item("approximation", approximation)?;
    dict.set_item(
        "static_tensor",
        array(py, &[3, 3], flat(&op.static_tensor))?,
    )?;
    dict.set_item("screened_head", op.head)?;
    dict.set_item("mesh", (op.mesh[0], op.mesh[1], op.mesh[2]))?;
    dict.set_item("kpoints", nk)?;
    dict.set_item("bands", op.window)?;
    dict.set_item("transitions", op.energies.len())?;
    dict.set_item("hermiticity", op.hermiticity)?;
    dict.set_item("residuals", excitons.residuals.clone())?;
    dict.set_item("converged", excitons.converged)?;
    dict.set_item("solver", excitons.solver)?;
    dict.set_item("scissor", user(&[op.scissor])?[0])?;
    dict.set_item(
        "cutoff",
        to_user(&[op.cutoff], Quantity::Wavenumber, atomic_units, 1).map_err(error)?[0],
    )?;
    dict.set_item("functional", periodic.functional.clone())?;
    let unit = if atomic_units { "Ha" } else { "eV" };
    dict.set_item("energy_unit", unit)?;
    let list = |values: &[f64]| -> PyResult<String> {
        Ok(user(values)?
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(" "))
    };
    let record: Vec<(String, String)> = vec![
        ("method".into(), "bethe-salpeter".into()),
        ("approximation".into(), approximation.into()),
        ("kernel".into(), kernel.name().into()),
        ("spin".into(), spin.name().into()),
        (
            "reference".into(),
            format!("kohn-sham {}", periodic.functional),
        ),
        (
            "mesh".into(),
            format!("{} {} {}", op.mesh[0], op.mesh[1], op.mesh[2]),
        ),
        ("kpoints".into(), nk.to_string()),
        ("symmetry".into(), "whole mesh".into()),
        (
            "bands_valence_conduction".into(),
            format!("{},{}", op.window.0, op.window.1),
        ),
        ("transitions".into(), op.energies.len().to_string()),
        ("exchange_vectors".into(), op.vectors.to_string()),
        (format!("scissor_{unit}"), list(&[op.scissor])?),
        (format!("onset_{unit}"), list(&[op.onset])?),
        (format!("energies_{unit}"), list(&excitons.energies)?),
        (format!("binding_{unit}"), list(&summary.binding)?),
        (
            "static_dielectric_diagonal".into(),
            format!(
                "{:.6} {:.6} {:.6}",
                op.static_tensor[0][0], op.static_tensor[1][1], op.static_tensor[2][2]
            ),
        ),
        ("screened_head_ha_bohr3".into(), format!("{:.6}", op.head)),
        ("solver".into(), excitons.solver.into()),
        ("converged".into(), excitons.converged.to_string()),
        ("hermiticity".into(), format!("{:.3e}", op.hermiticity)),
        ("momentum_residual".into(), format!("{:.3e}", op.residual)),
        (
            "wall_time_s".into(),
            format!("{:.2}", started.elapsed().as_secs_f64()),
        ),
    ];
    dict.set_item("record", record)?;
    Ok(dict)
}
