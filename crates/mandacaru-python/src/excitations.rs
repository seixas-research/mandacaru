//! Linear-response excitations of a converged molecular reference: the request is copied in,
//! Python is detached for the solve, and owned arrays come back. No numerics here;
//! `mandacaru_core::molecular::excitations` computes every value (atomic units).
use super::{array, error, molecular::PyMeanField, Array};
use mandacaru_core::{
    models::spectrum::{line_spectrum as broaden, LineShape},
    molecular::{
        excitations::{
            self, Approximation, Coupling, Excitations, ExcitedState, Followed, Picked, Request,
            Spin, Track,
        },
        MeanField, MolecularIntegrals,
    },
    quantum::C64,
    units, Context,
};
use nalgebra::DMatrix;
use numpy::PyReadonlyArray1;
use pyo3::{
    exceptions::{PyNotImplementedError, PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use std::sync::{Arc, Mutex};

fn rows<'py>(py: Python<'py>, m: &DMatrix<f64>) -> PyResult<Array<'py, f64>> {
    array(
        py,
        &[m.nrows(), m.ncols()],
        (0..m.nrows())
            .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
            .collect(),
    )
}
fn stack<'py, T: numpy::Element + nalgebra::Scalar>(
    py: Python<'py>,
    ms: &[&DMatrix<T>],
) -> PyResult<Array<'py, T>> {
    let (r, c) = ms.first().map_or((0, 0), |m| m.shape());
    array(
        py,
        &[ms.len(), r, c],
        ms.iter()
            .flat_map(|m| (0..r).flat_map(move |i| (0..c).map(move |j| m[(i, j)].clone())))
            .collect(),
    )
}

/// The solved roots of one request, kept for the methods that read their amplitudes.
#[pyclass(name = "MolecularExcitations", module = "mandacaru._native", frozen)]
pub struct PyExcitations {
    inner: Excitations,
}

#[pymethods]
impl PyExcitations {
    /// Every value of the roots, atomic units: `energies_ha`, `wavelengths_nm`,
    /// `oscillator_strengths`, `transition_dipoles_bohr` `(n, 3)`, `plus` and `minus`
    /// `(n, occupied, virtual)`, `transitions` (per root, `(occupied, virtual, weight)` with
    /// orbital indices of the reference), `nto_weights`, `one_center_hartree_ha` (or `None`),
    /// `orbital_energies_ha`, `occupied`, `virtual` (per spin channel: one for a closed shell,
    /// alpha and beta otherwise), `spin_squared` and `reference_spin_squared` (unrestricted
    /// references; else `None`), the request and the solver's counters. `plus` and `minus` are a
    /// list over the channels of `(n, occupied, virtual)` arrays, and each transition is
    /// `(channel, occupied, virtual, weight)`. A spinor reference has one channel of spinors and
    /// complex moments and amplitudes (`Excitations::spinor`).
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let e = &self.inner;
        let n = e.energies.len();
        let out = PyDict::new(py);
        out.set_item("approximation", e.approximation.name())?;
        out.set_item("spin", e.spin.name())?;
        out.set_item("kernel", e.coupling.name())?;
        out.set_item("functional", e.functional.clone())?;
        out.set_item("energies_ha", array(py, &[n], e.energies.clone())?)?;
        let wavelengths = e
            .energies
            .iter()
            .map(|&w| units::wavelength_nm(w))
            .collect::<mandacaru_core::Result<Vec<f64>>>()
            .map_err(error)?;
        out.set_item("wavelengths_nm", array(py, &[n], wavelengths)?)?;
        out.set_item(
            "oscillator_strengths",
            array(py, &[n], e.oscillator_strengths.clone())?,
        )?;
        let moments = |real: &[[f64; 3]], complex: Option<&Vec<[C64; 3]>>| match complex {
            Some(c) => {
                array(py, &[n, 3], c.iter().flatten().copied().collect()).map(|a| a.into_any())
            }
            None => {
                array(py, &[n, 3], real.iter().flatten().copied().collect()).map(|a| a.into_any())
            }
        };
        let spinor = e.spinor.as_ref();
        out.set_item(
            "transition_dipoles_bohr",
            moments(&e.transition_dipoles, spinor.map(|s| &s.transition_dipoles))?,
        )?;
        out.set_item(
            "velocity_oscillator_strengths",
            array(py, &[n], e.velocity_oscillator_strengths.clone())?,
        )?;
        out.set_item(
            "transition_velocities_bohr_inverse",
            moments(
                &e.transition_velocities,
                spinor.map(|s| &s.transition_velocities),
            )?,
        )?;
        let channels = e.channels.len();
        let per_channel = |all: &[Vec<DMatrix<f64>>]| -> PyResult<Vec<Array<'py, f64>>> {
            (0..channels)
                .map(|c| stack(py, &all.iter().map(|root| &root[c]).collect::<Vec<_>>()))
                .collect()
        };
        if let Some(s) = spinor {
            out.set_item("plus", vec![stack(py, &s.plus.iter().collect::<Vec<_>>())?])?;
            out.set_item(
                "minus",
                vec![stack(py, &s.minus.iter().collect::<Vec<_>>())?],
            )?;
        } else {
            out.set_item("plus", per_channel(&e.plus)?)?;
            out.set_item("minus", per_channel(&e.minus)?)?;
        }
        let transitions: Vec<Vec<(usize, usize, usize, f64)>> = e
            .transitions
            .iter()
            .map(|root| {
                root.iter()
                    .map(|t| (t.channel, t.occupied, t.virtual_orbital, t.weight))
                    .collect()
            })
            .collect();
        out.set_item("transitions", transitions)?;
        out.set_item("nto_weights", e.nto_weights.clone())?;
        out.set_item("one_center_hartree_ha", e.one_center_hartree.clone())?;
        out.set_item(
            "occupied",
            e.channels.iter().map(|c| c.occupied).collect::<Vec<_>>(),
        )?;
        out.set_item(
            "virtual",
            e.channels
                .iter()
                .map(|c| c.virtual_orbitals)
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "orbital_energies_ha",
            e.channels
                .iter()
                .map(|c| c.orbital_energies.clone())
                .collect::<Vec<_>>(),
        )?;
        out.set_item("spin_squared", e.spin_squared.clone())?;
        out.set_item("reference_spin_squared", e.reference_spin_squared)?;
        out.set_item("requested", e.requested)?;
        out.set_item("added", e.added)?;
        out.set_item("degenerate_tolerance_ha", excitations::DEGENERATE_ROOTS)?;
        out.set_item("residuals", e.residuals.clone())?;
        out.set_item("iterations", e.iterations)?;
        out.set_item("applications", e.applications)?;
        Ok(out)
    }
    /// `(weights, occupied, empty)`: the natural transition orbitals of `root` in spin channel
    /// `channel` (0 for a closed shell; 0 alpha, 1 beta), columns in the nonorthogonal atomic
    /// basis, descending weight.
    #[pyo3(signature=(root, channel=0))]
    fn natural_transition_orbitals<'py>(
        &self,
        py: Python<'py>,
        root: usize,
        channel: usize,
    ) -> PyResult<(Vec<f64>, Array<'py, f64>, Array<'py, f64>)> {
        let (weights, occupied, empty) = self
            .inner
            .natural_transition_orbitals(root, channel)
            .map_err(error)?;
        Ok((weights, rows(py, &occupied)?, rows(py, &empty)?))
    }
    /// The symmetrized transition density matrix of `root` in spin channel `channel` in the
    /// nonorthogonal atomic basis.
    #[pyo3(signature=(root, channel=0))]
    fn transition_density<'py>(
        &self,
        py: Python<'py>,
        root: usize,
        channel: usize,
    ) -> PyResult<Array<'py, f64>> {
        rows(
            py,
            &self
                .inner
                .transition_density(root, channel)
                .map_err(error)?,
        )
    }
}

/// The lowest `num_states` excitations of a converged closed-shell molecular reference
/// (`molecular::excitations::excitations`), completed to whole multiplets.
#[pyfunction]
#[pyo3(signature=(mean_field, *, num_states, approximation="tda", spin=None, kernel="adiabatic", tolerance=1e-5, max_iterations=200))]
#[allow(clippy::too_many_arguments)]
pub fn molecular_excitations(
    py: Python<'_>,
    mean_field: &PyMeanField,
    num_states: usize,
    approximation: &str,
    spin: Option<&str>,
    kernel: &str,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<PyExcitations> {
    let request = Request {
        num_states,
        approximation: Approximation::parse(approximation).map_err(error)?,
        spin: spin.map(Spin::parse).transpose().map_err(error)?,
        coupling: Coupling::parse(kernel).map_err(error)?,
        tolerance,
        max_iterations,
    };
    let (integrals, reference) = (mean_field.integrals.clone(), mean_field.inner.clone());
    let inner = py
        .detach(|| excitations::excitations(&integrals, &reference, &request))
        .map_err(error)?;
    Ok(PyExcitations { inner })
}

/// The calculator's `excited_state=` request (`Mandacaru(method="dft", "rhf" or "uhf",
/// excited_state={...})`, a spinor Kohn-Sham reference included),
/// parsed here, and the state it followed at the previous geometry of a scan.
#[pyclass(name = "ExcitedStateTarget", module = "mandacaru._native", frozen)]
pub struct PyExcitedStateTarget {
    request: Request,
    root: usize,
    track: Track,
    previous: Mutex<Option<Followed>>,
}

/// One excited state of a Kohn-Sham or Hartree-Fock reference, how its root was picked, and its gradient once
/// asked for (an energy alone needs only the excitation solve).
pub struct Excited {
    picked: Picked,
    approximation: Approximation,
    track: Track,
    /// The root the previous geometry followed (`Track::Overlap`), if any.
    from: Option<usize>,
    gradient: Mutex<Option<Arc<ExcitedState>>>,
}

impl PyExcitedStateTarget {
    /// The excited state of `reference` this request picks: the requested root at the first
    /// geometry and with `track="index"`, the root of largest transition-density overlap with
    /// the previous geometry's with `track="overlap"`. It becomes the state followed next.
    pub(crate) fn solve(
        &self,
        integrals: &MolecularIntegrals,
        reference: &MeanField,
    ) -> mandacaru_core::Result<Excited> {
        let mut previous = self.previous.lock().unwrap_or_else(|p| p.into_inner());
        let followed = previous.as_ref().filter(|_| self.track == Track::Overlap);
        let picked = excitations::pick(integrals, reference, &self.request, self.root, followed)?;
        let from = followed.map(|f| f.root);
        *previous = Some(picked.followed.clone());
        Ok(Excited {
            picked,
            approximation: self.request.approximation,
            track: self.track,
            from,
            gradient: Mutex::new(None),
        })
    }
}

/// The SCF keywords of the calculator's `excited_state=` option: none for `None`, else
/// `excited_state` with the parsed request: `root` (0-based among the roots of `spin`, default
/// 0), `spin` (`"singlet"` or `"triplet"`; unset, a closed shell's singlets and a spin-polarized
/// reference's spin-conserving roots), `approximation` (`"tda"` or `"casida"`) and `track`
/// (`"index"` or `"overlap"`, how a scan picks the root at each new geometry).
#[pyfunction]
#[pyo3(signature=(value))]
pub fn excited_state_options<'py>(
    py: Python<'py>,
    value: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let Some(value) = value else {
        return Ok(out);
    };
    let given = value.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err(
            "excited_state must be a dict with the keys 'root', 'spin', 'approximation' and 'track', or None",
        )
    })?;
    for key in given.keys() {
        let key: String = key.extract()?;
        if !["root", "spin", "approximation", "track"].contains(&key.as_str()) {
            return Err(PyValueError::new_err(format!(
                "unknown excited_state key {key:?}; expected 'root', 'spin', 'approximation' or 'track'"
            )));
        }
    }
    let text = |key: &str, default: &str| -> PyResult<String> {
        given
            .get_item(key)?
            .map_or(Ok(default.into()), |v| v.extract())
    };
    let root: usize = match given.get_item("root")? {
        Some(v) if !v.is_instance_of::<pyo3::types::PyBool>() => v.extract().ok(),
        Some(_) => None,
        None => Some(0),
    }
    .ok_or_else(|| {
        PyValueError::new_err("excited_state root must be an integer of at least 0 (0-based)")
    })?;
    let request = Request {
        num_states: root + 1,
        approximation: Approximation::parse(&text("approximation", "tda")?).map_err(error)?,
        // Unset is a closed shell's singlet and an unrestricted reference's spin-conserving roots.
        spin: given
            .get_item("spin")?
            .map(|v| v.extract::<String>())
            .transpose()?
            .map(|name| Spin::parse(&name))
            .transpose()
            .map_err(error)?,
        coupling: Coupling::Adiabatic,
        tolerance: 1e-9,
        max_iterations: 400,
    };
    let target = PyExcitedStateTarget {
        request,
        root,
        track: Track::parse(&text("track", "index")?).map_err(error)?,
        previous: Mutex::new(None),
    };
    out.set_item("excited_state", Bound::new(py, target)?)?;
    Ok(out)
}

/// `NotImplementedError` for an `excited_state=` request on a reference without one.
pub(crate) fn refuse_excited_state(
    target: &Option<Bound<'_, PyExcitedStateTarget>>,
) -> PyResult<()> {
    match target {
        Some(_) => Err(PyNotImplementedError::new_err(
            "excited_state follows one root of a Kohn-Sham or Hartree-Fock reference; \
             two-component Hartree-Fock (method='ghf' without a functional) has no response yet",
        )),
        None => Ok(()),
    }
}

fn energy_text(value: f64) -> String {
    format!("{value:.10} Ha ({:.6} eV)", value * units::HARTREE_TO_EV)
}

/// The excited state's lines of a molecular snapshot: `excitation` (Hartree), the
/// `free_energy` of the excited state and the `excited_state` facts of `[SCF SUMMARY]`.
pub(crate) fn snapshot_excited(out: &Bound<'_, PyDict>, excited: &Excited) -> PyResult<()> {
    let state = &excited.picked;
    out.set_item("excitation", state.excitation)?;
    out.set_item("free_energy", state.energy)?;
    let facts = PyDict::new(out.py());
    let picked = match (state.overlap, excited.from) {
        (Some(overlap), Some(from)) => {
            format!("followed by transition-density overlap {overlap:.6} from root {from} at the previous geometry")
        }
        _ if excited.track == Track::Overlap => {
            "picked by index at the first geometry (track=\"overlap\" follows it from the next)"
                .to_string()
        }
        _ => "picked by index (track=\"index\")".to_string(),
    };
    facts.set_item(
        "excited_state",
        format!(
            "root {} of the {} roots ({}, adiabatic kernel), {picked}",
            state.root,
            state.spin.name(),
            excited.approximation.name()
        ),
    )?;
    facts.set_item("excitation_energy", energy_text(state.excitation))?;
    facts.set_item(
        "ground_state_energy",
        format!(
            "{}; the energy above is it plus the excitation",
            energy_text(state.energy - state.excitation)
        ),
    )?;
    facts.set_item("nearest_root_separation", energy_text(state.separation))?;
    facts.set_item("root_residual", format!("{:.1e}", state.root_residual))?;
    out.set_item("excited_state", facts)?;
    Ok(())
}

/// The analytic gradient of the excited state (`MeanField.gradient()` of an `excited_state=`
/// reference, computed on the first call): `gradient` (atoms x 3, Hartree per Bohr) and the
/// details the forces block logs.
pub(crate) fn excited_gradient<'py>(
    py: Python<'py>,
    excited: &Excited,
    integrals: &MolecularIntegrals,
    reference: &MeanField,
) -> PyResult<Bound<'py, PyDict>> {
    let state = py
        .detach(|| -> mandacaru_core::Result<Arc<ExcitedState>> {
            let mut held = excited.gradient.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(state) = held.as_ref() {
                return Ok(state.clone());
            }
            let state = Arc::new(excited.picked.gradient(integrals, reference)?);
            *held = Some(state.clone());
            Ok(state)
        })
        .map_err(error)?;
    let gradient = state.gradient.total.as_slice().to_vec();
    let out = PyDict::new(py);
    out.set_item("gradient", array(py, &[gradient.len() / 3, 3], gradient)?)?;
    out.set_item(
        "scheme",
        format!(
            "analytic excited-state Lagrangian (relaxed Z-vector) gradient of root {} ({}, {})",
            state.root,
            state.spin.name(),
            excited.approximation.name()
        ),
    )?;
    out.set_item("excitation_ha", state.excitation)?;
    out.set_item("z_vector_residual", state.z_residual)?;
    out.set_item("z_vector_iterations", state.z_iterations)?;
    out.set_item("nearest_root_separation_ha", state.separation)?;
    Ok(out)
}

/// `sum_n s_n g(x - x_n)` on `axis`, unit-area Gaussian or Lorentzian lines of `width`
/// (`models::spectrum::line_spectrum`); units are the caller's.
#[pyfunction]
#[pyo3(signature=(context, positions, strengths, axis, *, width, shape="gaussian"))]
pub fn line_spectrum<'py>(
    py: Python<'py>,
    context: &super::PyContext,
    positions: PyReadonlyArray1<'py, f64>,
    strengths: PyReadonlyArray1<'py, f64>,
    axis: PyReadonlyArray1<'py, f64>,
    width: f64,
    shape: &str,
) -> PyResult<Array<'py, f64>> {
    let shape = LineShape::parse(shape).map_err(error)?;
    let (positions, strengths, axis) = (
        positions.as_slice()?.to_vec(),
        strengths.as_slice()?.to_vec(),
        axis.as_slice()?.to_vec(),
    );
    let ctx: Context = context.inner.clone();
    let values = py
        .detach(|| broaden(&ctx, &positions, &strengths, &axis, width, shape))
        .map_err(error)?;
    let n = values.as_slice().len();
    array(py, &[n], values.as_slice().to_vec())
}
