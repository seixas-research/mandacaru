//! Linear-response excitations of a converged molecular reference: the request is copied in,
//! Python is detached for the solve, and owned arrays come back. No numerics here;
//! `mandacaru_core::molecular::excitations` computes every value (atomic units).
use super::{array, error, molecular::PyMeanField, Array};
use mandacaru_core::{
    models::spectrum::{line_spectrum as broaden, LineShape},
    molecular::excitations::{self, Approximation, Coupling, Excitations, Request, Spin},
    units, Context,
};
use nalgebra::DMatrix;
use numpy::PyReadonlyArray1;
use pyo3::{prelude::*, types::PyDict};

fn rows<'py>(py: Python<'py>, m: &DMatrix<f64>) -> PyResult<Array<'py, f64>> {
    array(
        py,
        &[m.nrows(), m.ncols()],
        (0..m.nrows())
            .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
            .collect(),
    )
}
fn stack<'py>(py: Python<'py>, ms: &[&DMatrix<f64>]) -> PyResult<Array<'py, f64>> {
    let (r, c) = ms.first().map_or((0, 0), |m| m.shape());
    array(
        py,
        &[ms.len(), r, c],
        ms.iter()
            .flat_map(|m| (0..r).flat_map(move |i| (0..c).map(move |j| m[(i, j)])))
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
    /// `(channel, occupied, virtual, weight)`.
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
        out.set_item(
            "transition_dipoles_bohr",
            array(
                py,
                &[n, 3],
                e.transition_dipoles.iter().flatten().copied().collect(),
            )?,
        )?;
        out.set_item(
            "velocity_oscillator_strengths",
            array(py, &[n], e.velocity_oscillator_strengths.clone())?,
        )?;
        out.set_item(
            "transition_velocities_bohr_inverse",
            array(py, &[n, 3], e.transition_velocities.iter().flatten().copied().collect())?,
        )?;
        let channels = e.channels.len();
        let per_channel = |all: &[Vec<DMatrix<f64>>]| -> PyResult<Vec<Array<'py, f64>>> {
            (0..channels)
                .map(|c| stack(py, &all.iter().map(|root| &root[c]).collect::<Vec<_>>()))
                .collect()
        };
        out.set_item("plus", per_channel(&e.plus)?)?;
        out.set_item("minus", per_channel(&e.minus)?)?;
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
            e.channels.iter().map(|c| c.virtual_orbitals).collect::<Vec<_>>(),
        )?;
        out.set_item(
            "orbital_energies_ha",
            e.channels.iter().map(|c| c.orbital_energies.clone()).collect::<Vec<_>>(),
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
        rows(py, &self.inner.transition_density(root, channel).map_err(error)?)
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
