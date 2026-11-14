//! Real-time propagation of a converged molecular reference after a delta kick, and the
//! time-to-frequency transforms of its trace: the request is copied in, Python is detached for
//! the run, and owned arrays come back. No numerics here; `mandacaru_core::molecular::realtime`
//! and `mandacaru_core::models::time_spectrum` compute every value (atomic units).
use super::{array, error, molecular::PyMeanField, Array};
use mandacaru_core::{
    math::propagator::Scheme,
    models::time_spectrum::{self as spectra, Kind, Window},
    molecular::realtime::{self, Gauge, Kick, Request},
    Context,
};
use numpy::{Complex64, PyReadonlyArray1};
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
/// (the electrons' kinetic momentum), `energy_ha`, `norm_error` `(nt,)`, `ground_energy_ha`,
/// `kick_energy_ha`, `energy_drift` (kick energies), `width_ha` (the basis width `W` of the
/// reference Fock matrix), `largest_width_ha`, `step_width_bound` (the scheme's largest `dt W`),
/// `aliased_weight` (the bare weight above `pi / dt`), `builds`, `corrector_iterations`, and
/// the request.
#[pyfunction]
#[pyo3(signature=(mean_field, *, direction, strength, time_step, duration, atomic_units=false,
                  kick="density", gauge="length", propagator="etrs", corrector_tolerance=1e-6,
                  max_corrector_iterations=8, max_energy_drift=Some(0.05)))]
#[allow(clippy::too_many_arguments)]
pub fn molecular_real_time<'py>(
    py: Python<'py>,
    mean_field: &PyMeanField,
    direction: [f64; 3],
    strength: f64,
    time_step: f64,
    duration: f64,
    atomic_units: bool,
    kick: &str,
    gauge: &str,
    propagator: &str,
    corrector_tolerance: f64,
    max_corrector_iterations: usize,
    max_energy_drift: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
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

/// The response function `kind` (`"strength"`, `"polarizability"` or `"dielectric"`) of a trace
/// sampled in atomic units at `n time_step` after a kick of `strength` (atomic units), on
/// `frequencies` with `damping` in the user's energy unit (eV; Hartree with `atomic_units`),
/// in user units (`models::time_spectrum::spectrum_in_user_units`): the real `S` (1/eV), the
/// complex `alpha` (Angstrom^3) or `epsilon` (exponential window only). Returns
/// `(values, tail)`, `tail` the window's weight on the last sample.
#[pyfunction]
#[pyo3(signature=(context, samples, frequencies, *, time_step, strength, damping,
                  window="exponential", kind="strength", atomic_units=false))]
#[allow(clippy::too_many_arguments)]
pub fn time_spectrum<'py>(
    py: Python<'py>,
    context: &super::PyContext,
    samples: PyReadonlyArray1<'py, f64>,
    frequencies: PyReadonlyArray1<'py, f64>,
    time_step: f64,
    strength: f64,
    damping: f64,
    window: &str,
    kind: &str,
    atomic_units: bool,
) -> PyResult<(Bound<'py, PyAny>, f64)> {
    let window = Window::parse(window).map_err(error)?;
    let kind = Kind::parse(kind).map_err(error)?;
    let (samples, frequencies) = (
        samples.as_slice()?.to_vec(),
        frequencies.as_slice()?.to_vec(),
    );
    let ctx: Context = context.inner.clone();
    let n = frequencies.len();
    let (values, tail) = py
        .detach(|| {
            spectra::spectrum_in_user_units(
                &ctx,
                kind,
                &samples,
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
        array(py, &[n], values.iter().map(|v| v.re).collect())?.into_any()
    } else {
        array(
            py,
            &[n],
            values.iter().map(|v| Complex64::new(v.re, v.im)).collect(),
        )?
        .into_any()
    };
    Ok((out, tail))
}
