//! Owned periodic inputs and results. Numerical loops run detached in Rust.
use super::{array, copy, error, Array, PyContext};
use mandacaru_core::{
    math::storage::MatrixSource,
    periodic::{
        spectrum::{self, Smearing},
        Crystal, KPoint, Options, PeriodicResult,
    },
    product,
    quantum::C64,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyDict};
use std::sync::Arc;
fn corep_label(
    corep: mandacaru_core::periodic::space_group::double_group::CorepKind,
) -> &'static str {
    use mandacaru_core::periodic::space_group::double_group::CorepKind;
    match corep {
        CorepKind::UnitaryOnly => "unitary",
        CorepKind::TypeA => "a",
        CorepKind::TypeB => "b",
        CorepKind::TypeC { .. } => "c",
    }
}

fn crystal_band_irreps_to_py<'py>(
    py: Python<'py>,
    out_data: &mandacaru_core::periodic::irreps::CrystalBandIrreps,
) -> PyResult<Bound<'py, PyDict>> {
    use pyo3::types::PyList;
    let out = PyDict::new(py);
    out.set_item("k_fractional", out_data.double_group.k_frac.to_vec())?;
    out.set_item("n_unitary", out_data.double_group.unitary.len())?;
    out.set_item("bar_e", out_data.double_group.bar_e)?;
    out.set_item("max_unitarity_error", out_data.irreps.max_unitarity_error)?;
    out.set_item("overlap_residual", out_data.overlap_residual)?;

    let irreps_list = PyList::empty(py);
    for irrep in &out_data.double_group.irreps {
        let d = PyDict::new(py);
        d.set_item("index", irrep.index)?;
        d.set_item("dimension", irrep.dimension)?;
        d.set_item("spinor", irrep.spinor)?;
        d.set_item("corep", corep_label(irrep.corep))?;
        d.set_item(
            "characters",
            irrep
                .characters
                .iter()
                .map(|c| (c.re, c.im))
                .collect::<Vec<_>>(),
        )?;
        irreps_list.append(d)?;
    }
    out.set_item("irreps", irreps_list)?;

    let clusters_list = PyList::empty(py);
    for cl in &out_data.irreps.clusters {
        let d = PyDict::new(py);
        d.set_item("lo", cl.lo)?;
        d.set_item("hi", cl.hi)?;
        d.set_item("energy", cl.energy)?;
        d.set_item("dominant_irrep", cl.dominant_irrep)?;
        d.set_item(
            "spinor",
            out_data.double_group.irreps[cl.dominant_irrep].spinor,
        )?;
        d.set_item("corep", corep_label(cl.corep))?;
        d.set_item("irreps", cl.irreps.clone())?;
        clusters_list.append(d)?;
    }
    out.set_item("clusters", clusters_list)?;
    Ok(out)
}

fn crystal_band_irreps_from_parts<'py>(
    py: Python<'py>,
    dg: &mandacaru_core::periodic::space_group::double_group::DoubleGroup,
    irreps: &mandacaru_core::periodic::irreps::BandIrreps,
    overlap_residual: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let wrapped = mandacaru_core::periodic::irreps::CrystalBandIrreps {
        double_group: dg.clone(),
        irreps: irreps.clone(),
        overlap_residual,
    };
    crystal_band_irreps_to_py(py, &wrapped)
}

fn dipole_selection_to_py<'py>(
    py: Python<'py>,
    sel: &mandacaru_core::periodic::irreps::DipoleSelection,
) -> PyResult<Bound<'py, PyDict>> {
    let out =
        crystal_band_irreps_from_parts(py, &sel.double_group, &sel.irreps, sel.overlap_residual)?;
    out.set_item("n_valence", sel.n_valence)?;
    out.set_item("n_conduction", sel.n_conduction)?;
    out.set_item("allowed", sel.allowed.clone())?;
    out.set_item(
        "allowed_axes",
        (
            sel.allowed_axes[0].clone(),
            sel.allowed_axes[1].clone(),
            sel.allowed_axes[2].clone(),
        ),
    )?;
    out.set_item("weights", sel.weights.clone())?;
    Ok(out)
}

pub(super) fn input_arrays(
    ctx: &mandacaru_core::Context,
    values: &Bound<'_, PyDict>,
    metadata: &str,
    precision: Option<mandacaru_core::math::storage::Precision>,
    consume: bool,
) -> PyResult<mandacaru_core::periodic::input::InputArrays> {
    use mandacaru_core::math::storage::{ComplexSamples, Precision};
    use mandacaru_core::periodic::input::{InputArray, InputArrays};
    let _metadata = ctx
        .reserve(product(&[metadata.len(), 32]).map_err(error)?)
        .map_err(error)?;
    let metadata: serde_json::Value = serde_json::from_str(metadata)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let precision = match precision {
        Some(p) => p,
        None => {
            Precision::parse(metadata["precision"].as_str().unwrap_or("float64")).map_err(error)?
        }
    };
    let raw = metadata
        .as_array()
        .or_else(|| metadata["kpoints"].as_array());
    let sample_keys: std::collections::BTreeSet<_> = raw
        .into_iter()
        .flatten()
        .filter_map(|row| row["psi"].as_str())
        .collect();
    let mut out = InputArrays::default();
    let mut sample_bytes = 0usize;
    for key in &sample_keys {
        if let Some(value) = values.get_item(*key)? {
            let length = if let Ok(data) = value.extract::<PyReadonlyArrayDyn<'_, C64>>() {
                data.len()
            } else {
                value
                    .extract::<PyReadonlyArrayDyn<'_, numpy::Complex32>>()?
                    .len()
            };
            let bytes = product(&[
                length,
                if precision == Precision::Float32 {
                    8
                } else {
                    16
                },
            ])
            .map_err(error)?;
            sample_bytes = sample_bytes
                .checked_add(bytes)
                .ok_or_else(|| error(mandacaru_core::Error::Overflow))?;
        }
    }
    let archive = ctx.offloads(sample_bytes);
    // A private, disposable handoff dictionary can release each Python array
    // as soon as Rust has copied or archived it. Public callers keep ownership.
    for py_key in values.keys().iter() {
        let value = values.get_item(&py_key)?.ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err("input array disappeared during transfer")
        })?;
        let key: String = py_key.extract()?;
        let owned = if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, f64>>() {
            super::layout(&array)?;
            InputArray::Real(ctx.copy_slice(array.as_slice()?).map_err(error)?)
        } else if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, C64>>() {
            super::layout(&array)?;
            if sample_keys.contains(key.as_str()) || metadata["hats"].as_str() == Some(key.as_str())
            {
                let is_hat = metadata["hats"].as_str() == Some(key.as_str());
                let precision = if is_hat {
                    Precision::Float64
                } else {
                    precision
                };
                let archive = if is_hat {
                    ctx.offloads(product(&[array.len(), 16]).map_err(error)?)
                } else {
                    archive
                };
                InputArray::Samples(
                    if archive {
                        ComplexSamples::archived_copy(ctx, array.as_slice()?, precision)
                    } else {
                        ComplexSamples::copy(ctx, array.as_slice()?, precision)
                    }
                    .map_err(error)?,
                )
            } else {
                InputArray::Complex(ctx.copy_slice(array.as_slice()?).map_err(error)?)
            }
        } else if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, numpy::Complex32>>() {
            super::layout(&array)?;
            if !sample_keys.contains(key.as_str()) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "complex64 inputs are reserved for Bloch samples",
                ));
            }
            InputArray::Samples(
                ComplexSamples::copy_float32(ctx, array.as_slice()?, precision, archive)
                    .map_err(error)?,
            )
        } else if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, u64>>() {
            super::layout(&array)?;
            InputArray::Indices(ctx.copy_slice(array.as_slice()?).map_err(error)?)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "array {key} must have float64, complex128, complex64 Bloch samples or uint64 dtype"
            )));
        };
        out.0.insert(key, owned);
        if consume {
            values.del_item(&py_key)?;
        }
    }
    Ok(out)
}
fn matrix_values(a: &nalgebra::DMatrix<C64>) -> Vec<C64> {
    (0..a.nrows())
        .flat_map(|i| (0..a.ncols()).map(move |j| a[(i, j)]))
        .collect()
}
#[pyclass(name = "PeriodicSymmetry", module = "mandacaru._native", frozen)]
pub struct PyPeriodicSymmetry {
    pub n_operations: usize,
}

#[pymethods]
impl PyPeriodicSymmetry {
    #[getter]
    fn n_operations(&self) -> usize {
        self.n_operations
    }
}

#[pyclass(name = "PeriodicProblem", module = "mandacaru._native", frozen)]
pub struct PyCrystal {
    pub inner: Arc<Crystal>,
    pub setup: Option<Arc<mandacaru_core::periodic::setup::PeriodicSetup>>,
}

impl Drop for PyCrystal {
    fn drop(&mut self) {
        let _ = mandacaru_core::release_freed_memory();
    }
}
#[pymethods]
impl PyCrystal {
    #[getter]
    fn context(&self) -> PyContext {
        PyContext {
            inner: self.inner.context.clone(),
        }
    }
    #[getter]
    fn functional(&self) -> &str {
        &self.inner.functional
    }
    #[getter]
    fn precision(&self) -> &'static str {
        self.inner.precision.name()
    }
    #[getter]
    fn spins(&self) -> usize {
        self.inner.spins
    }
    #[getter]
    fn spinor(&self) -> bool {
        self.inner.spinor
    }
    #[getter]
    fn time_reversal(&self) -> bool {
        self.inner.time_reversal
    }
    #[getter]
    fn kpoints(&self) -> Option<Vec<[f64; 3]>> {
        self.setup.as_ref().map(|s| s.kpoints_cart.clone())
    }
    #[getter]
    fn kpoints_fractional(&self) -> Option<Vec<[f64; 3]>> {
        self.setup.as_ref().map(|s| s.kpoints_frac.clone())
    }
    #[getter]
    fn weights(&self) -> Option<Vec<f64>> {
        self.setup.as_ref().map(|s| s.weights.clone())
    }
    #[getter]
    fn symmetry(&self) -> Option<PyPeriodicSymmetry> {
        self.setup.as_ref().and_then(|s| {
            s.symmetry.as_ref().map(|sym| PyPeriodicSymmetry {
                n_operations: sym.len(),
            })
        })
    }
    /// The DFT+U shells the crystal carries, `(atom, symbol, l, u_ha, j_ha)`; `None` without.
    #[getter]
    fn hubbard(&self) -> Option<Vec<(usize, String, usize, f64, f64)>> {
        super::hubbard::definition(&self.inner)
    }
    /// Raises unless the `hubbard_u` option (JSON text) agrees with the DFT+U table this crystal
    /// carries; the file owns the correction.
    fn check_hubbard_u(&self, spec: &str) -> PyResult<()> {
        super::hubbard::check_loaded(spec, &self.inner)
    }
    #[getter]
    fn n_orbitals(&self) -> usize {
        self.inner.m
    }
    /// The real-space grid of the cell (periodic).
    #[getter]
    fn grid(&self) -> super::geometry::PySpatialGrid {
        super::geometry::PySpatialGrid {
            inner: self.inner.grid.clone(),
        }
    }
    /// Valence electrons per cell, the charge of the cell included.
    #[getter]
    fn electrons(&self) -> f64 {
        self.inner.electrons
    }
    #[getter]
    fn orbital_groups(&self) -> PyResult<Vec<(usize, usize)>> {
        if let Some(desc) = &self.inner.description {
            let mut groups = Vec::with_capacity(desc.function_atom.len());
            for (&atom, &table_idx) in desc.function_atom.iter().zip(&desc.function_table) {
                let l = desc.tables.get(table_idx).map(|t| t.l).unwrap_or(0);
                groups.push((atom, l));
            }
            if self.inner.spinor {
                let len = groups.len();
                for i in 0..len {
                    groups.push(groups[i]);
                }
            }
            Ok(groups)
        } else {
            Ok(Vec::new())
        }
    }
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.inner.save(path)).map_err(error)
    }
    #[pyo3(signature=(*,max_iterations=300,tolerance=1e-7,density_tolerance=1e-5,smearing="fermi-dirac",width_ha=0.003674932217565499,mixing=0.25,history=7,exxdiv_treatment="none",algorithm="diis",spin_mixing=None,eigensolver="davidson",switch_tolerance=1e-2))]
    fn solve(
        &self,
        py: Python<'_>,
        max_iterations: usize,
        tolerance: f64,
        density_tolerance: f64,
        smearing: &str,
        width_ha: f64,
        mixing: f64,
        history: usize,
        exxdiv_treatment: &str,
        algorithm: &str,
        spin_mixing: Option<f64>,
        eigensolver: &str,
        switch_tolerance: f64,
    ) -> PyResult<PyPeriodicResult> {
        let exxdiv =
            mandacaru_core::periodic::screened_exchange::ExxDivTreatment::parse(exxdiv_treatment)
                .map_err(error)?;
        let scf_algo =
            mandacaru_core::periodic::solver::ScfAlgorithm::parse(algorithm).map_err(error)?;
        let eigen_algo =
            mandacaru_core::periodic::solver::Eigensolver::parse(eigensolver).map_err(error)?;
        let options = Options {
            max_iterations,
            tolerance,
            density_tolerance,
            smearing: Smearing::parse(smearing).map_err(error)?,
            width: width_ha,
            mixing,
            history,
            exxdiv_treatment: exxdiv,
            algorithm: scf_algo,
            spin_mixing,
            eigensolver: eigen_algo,
            switch_tolerance,
        };
        let inner = py.detach(|| self.inner.solve(&options)).map_err(error)?;
        Ok(PyPeriodicResult {
            inner: Arc::new(inner),
            crystal: self.inner.clone(),
            setup: self.setup.clone(),
            rebuilt_setup: std::sync::OnceLock::new(),
            response: std::sync::OnceLock::new(),
        })
    }
    #[pyo3(signature=(result, *, alphas=None, self_consistent=false, damping=0.5, max_iterations=10, tolerance_ha=None, kernel="spin-summed", spread_tolerance_ha=None, energy_scale=1.0, unit="Ha"))]
    #[allow(clippy::too_many_arguments)]
    fn hubbard_response(
        &self,
        py: Python<'_>,
        result: &PyPeriodicResult,
        alphas: Option<Vec<f64>>,
        self_consistent: bool,
        damping: f64,
        max_iterations: usize,
        tolerance_ha: Option<f64>,
        kernel: &str,
        spread_tolerance_ha: Option<f64>,
        energy_scale: f64,
        unit: &str,
    ) -> PyResult<super::hubbard::PyHubbardResponse> {
        result.hubbard_response(
            py,
            alphas,
            self_consistent,
            damping,
            max_iterations,
            tolerance_ha,
            kernel,
            spread_tolerance_ha,
            energy_scale,
            unit,
        )
    }
}
#[pyclass(name = "PeriodicResult", module = "mandacaru._native", frozen)]
pub struct PyPeriodicResult {
    pub inner: Arc<PeriodicResult>,
    pub crystal: Arc<Crystal>,
    pub setup: Option<Arc<mandacaru_core::periodic::setup::PeriodicSetup>>,
    rebuilt_setup: std::sync::OnceLock<Arc<mandacaru_core::periodic::setup::PeriodicSetup>>,
    /// The geometry, quadrature and frozen state a force and a stress of this result share, formed
    /// by the first of them.
    response: std::sync::OnceLock<mandacaru_core::periodic::response::Response>,
}

impl PyPeriodicResult {
    pub(crate) fn resolve_setup(
        &self,
        py: Python<'_>,
    ) -> PyResult<Arc<mandacaru_core::periodic::setup::PeriodicSetup>> {
        if let Some(setup) = &self.setup {
            return Ok(setup.clone());
        }
        if let Some(setup) = self.rebuilt_setup.get() {
            return Ok(setup.clone());
        }
        let rebuilt = py
            .detach(|| mandacaru_core::periodic::setup::PeriodicSetup::from_crystal(&self.crystal))
            .map(Arc::new)
            .map_err(error)?;
        Ok(self.rebuilt_setup.get_or_init(|| rebuilt).clone())
    }

    fn response(&self) -> mandacaru_core::Result<&mandacaru_core::periodic::response::Response> {
        if let Some(response) = self.response.get() {
            return Ok(response);
        }
        let response = mandacaru_core::periodic::response::Response::new(&self.crystal)?;
        Ok(self.response.get_or_init(|| response))
    }
}

impl Drop for PyPeriodicResult {
    fn drop(&mut self) {
        let _ = mandacaru_core::release_freed_memory();
    }
}
#[pymethods]
impl PyPeriodicResult {
    /// A crystal has no dipole moment (`MeanField.dipole_moment` is a molecule's): raises
    /// `NotImplementedError` naming `get_polarization`.
    #[pyo3(signature=(*, atomic_units=false))]
    fn dipole_moment(&self, atomic_units: bool) -> PyResult<()> {
        let _ = atomic_units;
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "a crystal has no dipole moment: its polarization is get_polarization (the Berry phase \
             of the occupied bands)",
        ))
    }
    /// Evaluates the linear-response Hubbard U parameters for this converged crystal.
    /// Returns a native `HubbardResponse` view scaled by `energy_scale`.
    #[pyo3(signature=(*, alphas=None, self_consistent=false, damping=0.5, max_iterations=10, tolerance_ha=None, kernel="spin-summed", spread_tolerance_ha=None, energy_scale=1.0, unit="Ha"))]
    #[allow(clippy::too_many_arguments)]
    fn hubbard_response(
        &self,
        py: Python<'_>,
        alphas: Option<Vec<f64>>,
        self_consistent: bool,
        damping: f64,
        max_iterations: usize,
        tolerance_ha: Option<f64>,
        kernel: &str,
        spread_tolerance_ha: Option<f64>,
        energy_scale: f64,
        unit: &str,
    ) -> PyResult<super::hubbard::PyHubbardResponse> {
        let alphas = alphas.unwrap_or_else(|| vec![0.001837466, 0.003674932]); // ~0.05, 0.1 eV in Ha
        if self_consistent {
            use mandacaru_core::models::hubbard::Form;
            use mandacaru_core::periodic::response_scf::{converge_hubbard_u, Options};
            let kern = super::hubbard::parse_kernel(kernel)?;
            let form = self
                .crystal
                .hubbard
                .as_ref()
                .map_or(Form::Dudarev, |h| h.form);
            let mut opts = Options {
                max_iterations,
                damping,
                kernel: kern,
                form,
                alphas,
                ..Options::default()
            };
            if let Some(t) = tolerance_ha {
                opts.tolerance = t;
            }
            if let Some(st) = spread_tolerance_ha {
                opts.spread_tolerance = st;
            }
            let out = py
                .detach(|| converge_hubbard_u(&self.crystal, &self.inner, &opts))
                .map_err(error)?;
            return Ok(super::hubbard::PyHubbardResponse::from_periodic(
                out.response,
                energy_scale,
                unit.to_string(),
            ));
        }
        let res = py
            .detach(|| {
                mandacaru_core::periodic::hubbard_response(&self.crystal, &self.inner, &alphas)
            })
            .map_err(error)?;
        Ok(super::hubbard::PyHubbardResponse::from_periodic(
            res,
            energy_scale,
            unit.to_string(),
        ))
    }
    /// Maximally localized Wannier functions of the converged crystal, as `request_json` asks
    /// (`size`, `trials`, `n_functions`, `windows` in Hartree, `bands`). The mesh frames, the
    /// PAW geometry and the localization are native; a crystal loaded from a snapshot (no setup)
    /// raises `NotImplementedError`.
    #[pyo3(signature=(request_json, *, max_iterations=2000, tolerance=1e-10, step=0.5))]
    fn wannier(
        &self,
        py: Python<'_>,
        request_json: &str,
        max_iterations: usize,
        tolerance: f64,
        step: f64,
    ) -> PyResult<super::wannier::PyWannierResult> {
        let setup = self.resolve_setup(py)?;
        let options = mandacaru_core::periodic::wannier::Options {
            max_iterations,
            tolerance,
            step,
            ..Default::default()
        };
        let (crystal, state) = (self.crystal.clone(), self.inner.clone());
        let inner = py
            .detach(|| {
                mandacaru_core::periodic::wannier::setup::Wannier::build(
                    crystal,
                    state,
                    setup,
                    request_json,
                    &options,
                )
            })
            .map_err(error)?;
        Ok(super::wannier::PyWannierResult {
            inner: Arc::new(inner),
        })
    }
    /// The analytic force `-dF/dtau` per atom, Hartree/Bohr, shape `(n_atoms, 3)`, of the
    /// converged crystal, rebuilt in Rust from the crystal's own setup description. Raises
    /// `NotImplementedError` naming the cause for everything the engine does not differentiate.
    fn forces<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, f64>> {
        let forces = py
            .detach(|| self.response()?.forces(&self.crystal, &self.inner))
            .map_err(error)?;
        array(
            py,
            &[forces.len(), 3],
            forces.into_iter().flatten().collect(),
        )
    }
    /// The analytic stress at the pins of the crystal, Hartree/Bohr^3: `sigma = (g + g^T) / (2 Omega)`
    /// with `g_ab = dF/d eps_ab` (`gradient`, Hartree, symmetrized and averaged over the point group
    /// of the mesh) and the cell `volume` (Bohr^3). The 3x3 arrays are row-major `[a][b]`; the
    /// Voigt order and units are the Python caller's.
    fn stress<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stress = py
            .detach(|| self.response()?.stress(&self.crystal, &self.inner))
            .map_err(error)?;
        let out = PyDict::new(py);
        for (name, tensor) in [("sigma", stress.sigma), ("gradient", stress.gradient)] {
            out.set_item(
                name,
                array(py, &[3, 3], tensor.into_iter().flatten().collect())?,
            )?;
        }
        out.set_item("volume", stress.volume)?;
        Ok(out)
    }
    /// Berry phases and Wilson loops of the strings along `G_direction` of the mesh `size`
    /// (`periodic::berry::strings`); `bands` is `None` (an insulator's occupied bands) or a
    /// half-open `(lo, hi)`. Radians, fractional centers, Bohr.
    #[pyo3(signature=(direction, size, *, bands=None))]
    fn berry_phase<'py>(
        &self,
        py: Python<'py>,
        direction: usize,
        size: [usize; 3],
        bands: Option<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        super::berry::berry_phase(self, py, direction, size, bands)
    }
    /// The Berry curvature and Chern number of the plane normal to `G_normal`
    /// (`periodic::berry::curvature`) on the in-plane entries of the mesh `size` (the entry along
    /// the normal is not used), at the fractional `offset` along the normal. Radians, Bohr^2.
    #[pyo3(signature=(normal, size, *, offset=0.0, bands=None))]
    fn berry_curvature<'py>(
        &self,
        py: Python<'py>,
        normal: usize,
        size: [usize; 3],
        offset: f64,
        bands: Option<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        super::berry::berry_curvature(self, py, normal, size, offset, bands)
    }
    /// The Wannier-center flow of the plane normal to `G_normal` at the fractional `offset`, with
    /// strings along `G_direction` of the mesh `size` (`periodic::berry::flow`).
    #[pyo3(signature=(normal, direction, size, *, offset=0.0, bands=None))]
    fn wannier_flow<'py>(
        &self,
        py: Python<'py>,
        normal: usize,
        direction: usize,
        size: [usize; 3],
        offset: f64,
        bands: Option<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        super::berry::wannier_flow(self, py, normal, direction, size, offset, bands)
    }
    /// The Z2 indices `(nu_0; nu_1 nu_2 nu_3)` of the occupied bands, from the Wannier-center flows
    /// of the six time-reversal-invariant planes (`method="flow"`, `periodic::berry::flow`) or the
    /// Fu-Kane parity product (`method="parity"`, `periodic::berry::parity`).
    #[pyo3(signature=(size, *, method="flow"))]
    fn z2_invariant<'py>(
        &self,
        py: Python<'py>,
        size: [usize; 3],
        method: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        super::berry::z2_invariant(self, py, size, method)
    }
    /// The Berry-phase polarization of an insulator from the string mesh `size`
    /// (`periodic::berry::polarization`), e/Bohr^2.
    fn polarization<'py>(&self, py: Python<'py>, size: [usize; 3]) -> PyResult<Bound<'py, PyDict>> {
        super::berry::polarization(self, py, size)
    }
    /// The Chern numbers of the three plane families of the mesh `size` and the Hall
    /// conductivity of an insulator (`periodic::berry::hall`), e^2/(hbar Bohr).
    fn anomalous_hall<'py>(
        &self,
        py: Python<'py>,
        size: [usize; 3],
    ) -> PyResult<Bound<'py, PyDict>> {
        super::berry::anomalous_hall(self, py, size)
    }
    /// Born effective charges (`periodic::berry::perturbation::born_effective_charges`), e.
    #[pyo3(signature=(size, displacement=0.01, richardson=false, impose_rule=false, atomic_units=false))]
    fn born_charges<'py>(
        &self,
        py: Python<'py>,
        size: [usize; 3],
        displacement: f64,
        richardson: bool,
        impose_rule: bool,
        atomic_units: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        super::berry::born_charges(
            self,
            py,
            size,
            displacement,
            richardson,
            impose_rule,
            atomic_units,
        )
    }
    /// Clamped- or relaxed-ion proper piezoelectric tensor (`periodic::berry::perturbation::piezoelectric_tensor`), e/Bohr^2.
    #[pyo3(signature=(size, strain=0.005, displacement=0.01, richardson=false, impose_rule=false, relaxed_ions=false, atomic_units=false, dispersion_xc=None))]
    #[allow(clippy::too_many_arguments)]
    fn piezoelectric<'py>(
        &self,
        py: Python<'py>,
        size: [usize; 3],
        strain: f64,
        displacement: f64,
        richardson: bool,
        impose_rule: bool,
        relaxed_ions: bool,
        atomic_units: bool,
        dispersion_xc: Option<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        super::berry::piezoelectric(
            self,
            py,
            size,
            strain,
            displacement,
            richardson,
            impose_rule,
            relaxed_ions,
            atomic_units,
            dispersion_xc,
        )
    }
    #[getter]
    fn spin_orbit_energy(&self) -> Option<f64> {
        self.inner.spin_orbit_energy
    }
    #[getter]
    fn hubbard_energy(&self) -> Option<f64> {
        self.inner.hubbard_energy
    }
    #[getter]
    fn n_iterations(&self) -> usize {
        self.inner.history.len()
    }
    #[getter]
    fn exxdiv_treatment(&self) -> &'static str {
        self.inner.options.exxdiv_treatment.as_str()
    }
    fn projectability<'py>(
        &self,
        py: Python<'py>,
        trials: PyReadonlyArray2<'py, C64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let targets = trials.shape()[1];
        if trials.shape()[0] != self.crystal.bands() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "trials need shape (n_basis,n_targets)",
            ));
        }
        let (values, _input) = copy(&self.crystal.context, &trials)?;
        let out = py
            .detach(|| {
                mandacaru_core::periodic::analysis::projectability(
                    &self.crystal,
                    &self.inner,
                    &values,
                    targets,
                )
            })
            .map_err(error)?;
        let c = &self.crystal;
        let result = PyDict::new(py);
        result.set_item(
            "weights",
            array(
                py,
                &[c.spins, c.points.len(), c.bands()],
                out.weights.as_slice().to_vec(),
            )?,
        )?;
        result.set_item(
            "hamiltonians",
            array(
                py,
                &[c.spins, c.points.len(), targets, targets],
                out.hamiltonians.as_slice().to_vec(),
            )?,
        )?;
        Ok(result)
    }
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }
    #[pyo3(signature=(*,include_matrices=true))]
    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        include_matrices: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let c = &self.crystal;
        let r = &self.inner;
        let nk = c.points.len();
        let m = c.bands();
        let s = c.spins;
        let n = c.grid.index.points();
        let out = PyDict::new(py);
        for (name, value) in [
            ("energy", r.energy),
            ("free_energy", r.free_energy),
            ("extrapolated_energy", r.extrapolated_energy),
            ("fermi_level", r.fermi),
            ("magnetic_moment", r.magnetization),
        ] {
            out.set_item(name, value)?;
        }
        out.set_item("functional", &c.functional)?;
        out.set_item("n_spins", s)?;
        out.set_item("spinor", c.spinor)?;
        out.set_item("time_reversal", c.time_reversal)?;
        out.set_item("spin_orbit_energy", r.spin_orbit_energy)?;
        if let Some(view) = super::hubbard::view(py, c, r)? {
            out.set_item("hubbard_energy", r.hubbard_energy)?;
            out.set_item("hubbard", view)?;
        }
        out.set_item("converged", r.converged)?;
        out.set_item("n_iterations", r.history.len())?;
        out.set_item("precision", c.precision.name())?;
        out.set_item("exxdiv_treatment", r.options.exxdiv_treatment.as_str())?;
        if let Some(tau) = &r.kinetic_density {
            out.set_item(
                "kinetic_density",
                array(py, &[c.field_channels(), n], tau.as_slice().to_vec())?,
            )?;
            out.set_item(
                "kinetic_residuals",
                r.history
                    .iter()
                    .map(|it| it.kinetic_residual)
                    .collect::<Vec<_>>(),
            )?;
        }
        if c.screened_exchange().map_err(error)?.is_some() {
            out.set_item(
                "exchange_residuals",
                r.history
                    .iter()
                    .filter_map(|it| it.exchange_residual)
                    .collect::<Vec<_>>(),
            )?;
        }
        out.set_item("smearing", (r.options.smearing.name(), r.options.width))?;
        out.set_item(
            "eigenvalues",
            array(
                py,
                &[s, nk, m],
                r.eigenvalues.iter().flatten().copied().collect(),
            )?,
        )?;
        out.set_item(
            "occupations",
            array(
                py,
                &[s, nk, m],
                r.occupations.iter().flatten().copied().collect(),
            )?,
        )?;
        if include_matrices {
            let coefficients = r
                .coefficients
                .iter()
                .map(|v| v.matrix(&c.context).map(|v| matrix_values(&v)))
                .collect::<mandacaru_core::Result<Vec<_>>>()
                .map_err(error)?;
            out.set_item(
                "coefficients",
                array(
                    py,
                    &[s, nk, m, m],
                    coefficients.into_iter().flatten().collect(),
                )?,
            )?;
            let matrices = r
                .matrices
                .iter()
                .map(|v| v.matrix(&c.context).map(|v| matrix_values(&v)))
                .collect::<mandacaru_core::Result<Vec<_>>>()
                .map_err(error)?;
            out.set_item(
                "density_matrices",
                array(py, &[s, nk, m, m], matrices.into_iter().flatten().collect())?,
            )?;
        }
        out.set_item(
            "density",
            array(py, &[c.field_channels(), n], r.density.as_slice().to_vec())?,
        )?;
        out.set_item(
            "moments",
            array(py, &[c.channels], r.moments.as_slice().to_vec())?,
        )?;
        out.set_item(
            "kpoints",
            array(py, &[nk, 3], c.points.iter().flat_map(|p| p.k).collect())?,
        )?;
        out.set_item(
            "fractional_kpoints",
            array(
                py,
                &[nk, 3],
                c.points.iter().flat_map(|p| p.fractional).collect(),
            )?,
        )?;
        out.set_item(
            "weights",
            c.points.iter().map(|p| p.weight).collect::<Vec<_>>(),
        )?;
        let terms = PyDict::new(py);
        for (name, value) in ["band_fixed", "electrostatic", "xc", "constant", "entropy"]
            .iter()
            .zip(r.terms)
        {
            terms.set_item(*name, value)?;
        }
        if let Some(energy) = r.hubbard_energy {
            terms.set_item("hubbard", energy)?;
        }
        out.set_item("terms", terms)?;
        out.set_item(
            "history",
            r.history
                .iter()
                .map(|i| {
                    (
                        i.iteration,
                        i.free_energy,
                        i.delta,
                        i.residual,
                        i.moment_residual,
                        i.magnetization,
                    )
                })
                .collect::<Vec<_>>(),
        )?;
        Ok(out)
    }
    #[pyo3(signature=(energies_ha,*,width_ha,projected=false))]
    fn dos<'py>(
        &self,
        py: Python<'py>,
        energies_ha: PyReadonlyArray1<'py, f64>,
        width_ha: f64,
        projected: bool,
    ) -> PyResult<Array<'py, f64>> {
        let c = &self.crystal;
        let (energies, _memory) = copy(&c.context, &energies_ha)?;
        let out = py
            .detach(|| {
                let weights = (0..c.spins)
                    .flat_map(|_| c.points.iter().map(|p| p.weight))
                    .collect::<Vec<_>>();
                if projected {
                    return mandacaru_core::periodic::analysis::projected_dos(
                        c,
                        &self.inner,
                        &energies,
                        width_ha,
                    );
                }
                spectrum::broaden(
                    &c.context,
                    &energies,
                    &self.inner.eigenvalues,
                    &weights,
                    width_ha,
                    c.degeneracy(),
                    None,
                )
            })
            .map_err(error)?;
        array(
            py,
            &[energies.len(), if projected { c.bands() } else { 1 }],
            out.as_slice().to_vec(),
        )
    }
    /// Extra density-independent k-point matrices supplied by the setup adapter.
    fn bands<'py>(
        &self,
        py: Python<'py>,
        data_json: &str,
        arrays: &Bound<'_, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let c = &self.crystal;
        let mut arrays = input_arrays(&c.context, arrays, data_json, Some(c.precision), false)?;
        let _parse = c
            .context
            .reserve(product(&[data_json.len(), 32]).map_err(error)?)
            .map_err(error)?;
        let owned = data_json.to_owned();
        let (eps, projections) = py
            .detach(|| {
                let v: serde_json::Value = serde_json::from_str(&owned)
                    .map_err(|e| mandacaru_core::Error::Invalid(e.to_string()))?;
                let raw = v.as_array().filter(|a| !a.is_empty()).ok_or_else(|| {
                    mandacaru_core::Error::Invalid("bands need a nonempty k-point list".into())
                })?;
                let points = raw
                    .iter()
                    .map(|v| {
                        mandacaru_core::periodic::input::kpoint_arrays(
                            &c.context,
                            v,
                            c.m,
                            c.p,
                            c.grid.index.points(),
                            &mut arrays,
                            c.precision,
                        )
                    })
                    .collect::<mandacaru_core::Result<Vec<KPoint>>>()?;
                if !arrays.0.is_empty() {
                    return Err(mandacaru_core::Error::Invalid(
                        "unreferenced band input arrays".into(),
                    ));
                }
                let bands =
                    mandacaru_core::periodic::analysis::bands_of_points(c, &self.inner, &points)?;
                Ok::<_, mandacaru_core::Error>((bands.eigenvalues, bands.projections))
            })
            .map_err(error)?;
        let nk = eps.len() / (c.spins * c.bands());
        let out = PyDict::new(py);
        out.set_item("eigenvalues", array(py, &[c.spins, nk, c.bands()], eps)?)?;
        out.set_item(
            "projections",
            array(py, &[c.spins, nk, c.bands(), c.bands()], projections)?,
        )?;
        Ok(out)
    }

    /// Native off-mesh band structure computed directly from the crystal's setup recipe.
    #[pyo3(signature=(fractional_k))]
    fn bands_at<'py>(
        &self,
        py: Python<'py>,
        fractional_k: Vec<[f64; 3]>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let setup = self.resolve_setup(py)?;
        let c = &self.crystal;
        let bands = py
            .detach(|| {
                mandacaru_core::periodic::analysis::bands_at(c, &setup, &self.inner, &fractional_k)
            })
            .map_err(error)?;
        let nk = fractional_k.len();
        let out = PyDict::new(py);
        out.set_item(
            "eigenvalues",
            array(py, &[c.spins, nk, c.bands()], bands.eigenvalues)?,
        )?;
        out.set_item(
            "projections",
            array(py, &[c.spins, nk, c.bands(), c.bands()], bands.projections)?,
        )?;
        Ok(out)
    }

    /// Double-group irreducible representation classification of converged Bloch bands at mesh
    /// index `k_index`.
    #[pyo3(signature=(k_index=0, *, spin=0, degeneracy_tol=1e-5))]
    fn band_irreps<'py>(
        &self,
        py: Python<'py>,
        k_index: usize,
        spin: usize,
        degeneracy_tol: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let classified = py
            .detach(|| {
                mandacaru_core::periodic::irreps::classify_crystal_kpoint(
                    &self.crystal,
                    &self.inner,
                    spin,
                    k_index,
                    degeneracy_tol,
                )
            })
            .map_err(error)?;
        crystal_band_irreps_to_py(py, &classified)
    }

    /// Double-group irreducible representation classification at arbitrary `fractional_k` using
    /// the native periodic setup.
    #[pyo3(signature=(fractional_k, *, spin=0, degeneracy_tol=1e-5))]
    fn band_irreps_at<'py>(
        &self,
        py: Python<'py>,
        fractional_k: [f64; 3],
        spin: usize,
        degeneracy_tol: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let setup = self.resolve_setup(py)?;
        let (classified, _, _) = py
            .detach(|| {
                mandacaru_core::periodic::irreps::classify_crystal_at(
                    &self.crystal,
                    &setup,
                    &self.inner,
                    spin,
                    fractional_k,
                    degeneracy_tol,
                )
            })
            .map_err(error)?;
        crystal_band_irreps_to_py(py, &classified)
    }

    /// Double-group Cartesian electric-dipole selection rules (`x`, `y`, `z`, combined) for
    /// occupied -> virtual transitions at mesh `k_index`.
    #[pyo3(signature=(k_index=0, *, spin=0, degeneracy_tol=1e-5))]
    fn dipole_selection<'py>(
        &self,
        py: Python<'py>,
        k_index: usize,
        spin: usize,
        degeneracy_tol: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let sel = py
            .detach(|| {
                mandacaru_core::periodic::irreps::crystal_dipole_selection(
                    &self.crystal,
                    &self.inner,
                    spin,
                    k_index,
                    degeneracy_tol,
                )
            })
            .map_err(error)?;
        dipole_selection_to_py(py, &sel)
    }

    /// Double-group Cartesian electric-dipole selection rules at arbitrary `fractional_k` using
    /// the native periodic setup.
    #[pyo3(signature=(fractional_k, *, spin=0, degeneracy_tol=1e-5))]
    fn dipole_selection_at<'py>(
        &self,
        py: Python<'py>,
        fractional_k: [f64; 3],
        spin: usize,
        degeneracy_tol: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let setup = self.resolve_setup(py)?;
        let sel = py
            .detach(|| {
                mandacaru_core::periodic::irreps::crystal_dipole_selection_at(
                    &self.crystal,
                    &setup,
                    &self.inner,
                    spin,
                    fractional_k,
                    degeneracy_tol,
                )
            })
            .map_err(error)?;
        dipole_selection_to_py(py, &sel)
    }

    /// Double-group irrep classification and step-by-step band connectivity along a path of
    /// `fractional_k` coordinates using the native periodic setup.
    #[pyo3(signature=(fractional_k, *, spin=0, degeneracy_tol=1e-5))]
    fn band_path_irreps<'py>(
        &self,
        py: Python<'py>,
        fractional_k: Vec<[f64; 3]>,
        spin: usize,
        degeneracy_tol: f64,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let setup = self.resolve_setup(py)?;
        let steps = py
            .detach(|| {
                mandacaru_core::periodic::irreps::classify_kpath(
                    &self.crystal,
                    &setup,
                    &self.inner,
                    spin,
                    &fractional_k,
                    degeneracy_tol,
                )
            })
            .map_err(error)?;
        let mut out = Vec::with_capacity(steps.len());
        for step in steps {
            let d = crystal_band_irreps_to_py(py, &step.classification)?;
            d.set_item("energies", step.energies)?;
            d.set_item("links_from_prev", step.links_from_prev)?;
            out.push(d);
        }
        Ok(out)
    }

    #[pyo3(signature=(method="bader", *, numbers=None, positions=None, valence_charges=None, interpolate=true))]
    fn partial_charges<'py>(
        &self,
        py: Python<'py>,
        method: &str,
        numbers: Option<Vec<i64>>,
        positions: Option<Vec<[f64; 3]>>,
        valence_charges: Option<Vec<f64>>,
        interpolate: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let resolved_setup =
            if numbers.is_none() || valence_charges.is_none() || positions.is_none() {
                self.resolve_setup(py).ok()
            } else {
                None
            };
        let numbers_u8: Vec<u8> = match numbers {
            Some(nums) => nums
                .into_iter()
                .map(|z| {
                    u8::try_from(z).map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err("atomic number out of range")
                    })
                })
                .collect::<PyResult<_>>()?,
            None => {
                let setup = resolved_setup.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "periodic partial charges require atomic numbers without a native setup",
                    )
                })?;
                setup
                    .atoms_species
                    .iter()
                    .map(|&sp| {
                        u8::try_from(setup.datasets[sp].atomic_number).map_err(|_| {
                            pyo3::exceptions::PyValueError::new_err("atomic number out of range")
                        })
                    })
                    .collect::<PyResult<_>>()?
            }
        };
        let valence_charges = valence_charges.or_else(|| {
            resolved_setup.as_ref().map(|setup| {
                setup
                    .atoms_species
                    .iter()
                    .map(|&sp| setup.datasets[sp].valence_charge)
                    .collect()
            })
        });
        let positions =
            positions.or_else(|| resolved_setup.as_ref().map(|setup| setup.centers.clone()));
        let out = py
            .detach(|| {
                mandacaru_core::periodic::analysis::partial_charges(
                    &self.crystal,
                    &self.inner,
                    method,
                    &numbers_u8,
                    positions.as_deref(),
                    valence_charges.as_deref(),
                    interpolate,
                )
            })
            .map_err(error)?;
        crate::arrays::partial_charges_dict(py, out)
    }
}
