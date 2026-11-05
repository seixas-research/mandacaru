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
#[pyclass(name = "PeriodicProblem", module = "mandacaru._native", frozen)]
pub struct PyCrystal {
    pub inner: Arc<Crystal>,
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
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.inner.save(path)).map_err(error)
    }
    #[pyo3(signature=(*,max_iterations=300,tolerance=1e-7,density_tolerance=1e-5,smearing="fermi-dirac",width_ha=0.003674932217565499,mixing=0.25,history=7,exxdiv_treatment="none"))]
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
    ) -> PyResult<PyPeriodicResult> {
        let exxdiv = mandacaru_core::periodic::screened_exchange::ExxDivTreatment::parse(exxdiv_treatment)
            .map_err(error)?;
        let options = Options {
            max_iterations,
            tolerance,
            density_tolerance,
            smearing: Smearing::parse(smearing).map_err(error)?,
            width: width_ha,
            mixing,
            history,
            exxdiv_treatment: exxdiv,
        };
        let inner = py.detach(|| self.inner.solve(&options)).map_err(error)?;
        Ok(PyPeriodicResult {
            inner: Arc::new(inner),
            crystal: self.inner.clone(),
        })
    }
    #[pyo3(signature=(result, *,alphas=None))]
    fn hubbard_response<'py>(
        &self,
        py: Python<'py>,
        result: &PyPeriodicResult,
        alphas: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        result.hubbard_response(py, alphas)
    }
    #[doc(hidden)]
    fn _setup_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let ctx = &self.inner.context;
        for (i, p) in self.inner.points.iter().enumerate() {
            let read_psi = p.psi.read(ctx).map_err(error)?;
            dict.set_item(format!("k{i}.psi"), array(py, &[read_psi.len()], read_psi.to_vec())?)?;
            dict.set_item(format!("k{i}.overlap"), array(py, &[self.inner.m, self.inner.m], matrix_values(&p.overlap))?)?;
            dict.set_item(format!("k{i}.fixed"), array(py, &[self.inner.m, self.inner.m], matrix_values(&p.fixed))?)?;
            dict.set_item(format!("k{i}.projections"), array(py, &[self.inner.m, self.inner.p], matrix_values(&p.projections))?)?;
            if let Some(ref so) = p.spin_orbit {
                dict.set_item(format!("k{i}.spin_orbit"), array(py, &[so.nrows(), so.ncols()], matrix_values(so))?)?;
            }
        }
        dict.set_item("blocks", array(py, &[self.inner.blocks.as_slice().len()], self.inner.blocks.as_slice().to_vec())?)?;
        dict.set_item("core", array(py, &[self.inner.core.as_slice().len()], self.inner.core.as_slice().to_vec())?)?;
        if let Some(ref tau) = self.inner.core_tau {
            dict.set_item("core_tau", array(py, &[tau.as_slice().len()], tau.as_slice().to_vec())?)?;
        }
        dict.set_item("initial", array(py, &[self.inner.initial.as_slice().len()], self.inner.initial.as_slice().to_vec())?)?;
        dict.set_item("initial_m", array(py, &[self.inner.initial_m.as_slice().len()], self.inner.initial_m.as_slice().to_vec())?)?;
        dict.set_item("initial_q", array(py, &[self.inner.initial_q.as_slice().len()], self.inner.initial_q.as_slice().to_vec())?)?;
        dict.set_item("kernel", array(py, &[self.inner.kernel.as_slice().len()], self.inner.kernel.as_slice().to_vec())?)?;
        dict.set_item("xc_wave", array(py, &[self.inner.xc_wave.as_slice().len()], self.inner.xc_wave.as_slice().to_vec())?)?;
        dict.set_item("phase", array(py, &[self.inner.phase.as_slice().len()], self.inner.phase.as_slice().to_vec())?)?;
        dict.set_item("kerker", array(py, &[self.inner.kerker.as_slice().len()], self.inner.kerker.as_slice().to_vec())?)?;
        dict.set_item("ion", array(py, &[self.inner.ion.as_slice().len()], self.inner.ion.as_slice().to_vec())?)?;
        let read_hats = self.inner.hats.read(ctx).map_err(error)?;
        dict.set_item("hats", array(py, &[read_hats.len()], read_hats.to_vec())?)?;
        dict.set_item("u", array(py, &[self.inner.u.as_slice().len()], self.inner.u.as_slice().to_vec())?)?;
        dict.set_item("uion", array(py, &[self.inner.uion.as_slice().len()], self.inner.uion.as_slice().to_vec())?)?;
        dict.set_item("linear", array(py, &[self.inner.linear.as_slice().len()], self.inner.linear.as_slice().to_vec())?)?;
        let flat_perms: Vec<u64> = self.inner.permutations.iter().flat_map(|p| p.iter().map(|&x| x as u64)).collect();
        dict.set_item("permutations", array(py, &[flat_perms.len()], flat_perms)?)?;
        dict.set_item("qsym", array(py, &[self.inner.qsym.as_slice().len()], self.inner.qsym.as_slice().to_vec())?)?;
        Ok(dict)
    }
    #[doc(hidden)]
    fn _setup_metadata(&self) -> PyResult<String> {
        let meta = serde_json::json!({
            "schema_version": 1,
            "representation": "periodic-paw-snapshot",
            "energy_unit": "Ha",
            "length_unit": "bohr",
            "m": self.inner.m,
            "p": self.inner.p,
            "channels": self.inner.channels,
            "eion": self.inner.eion,
            "ion_constant": self.inner.ion_constant,
            "constant": self.inner.constant,
            "reference": self.inner.reference,
            "electrons": self.inner.electrons,
            "spins": self.inner.spins,
            "spinor": self.inner.spinor,
            "time_reversal": self.inner.time_reversal,
            "functional": self.inner.functional,
            "precision": self.inner.precision.name(),
            "relativistic": self.inner.relativistic,
            "datasets": self.inner.description.as_ref().map(|d| &d.datasets),
            "permutation_count": self.inner.permutations.len(),
        });
        serde_json::to_string(&meta)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
    #[doc(hidden)]
    fn _build_stages(&self) -> PyResult<Vec<(String, f64, usize)>> {
        Ok(vec![("crystal_build".to_string(), 0.0, self.inner.context.memory_peak())])
    }
}
#[pyclass(name = "PeriodicResult", module = "mandacaru._native", frozen)]
pub struct PyPeriodicResult {
    pub inner: Arc<PeriodicResult>,
    pub crystal: Arc<Crystal>,
}
#[pymethods]
impl PyPeriodicResult {
    /// Evaluates the linear-response Hubbard U parameters for this converged crystal.
    /// Returns dictionary with `u_matrix`, `u_charge`, `chi0`, `chi`, `alphas`, and optionally `j_response`.
    #[pyo3(signature=(*,alphas=None))]
    fn hubbard_response<'py>(
        &self,
        py: Python<'py>,
        alphas: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let alphas = alphas.unwrap_or_else(|| vec![0.001837466, 0.003674932]); // ~0.05, 0.1 eV in Ha
        let res = py
            .detach(|| {
                mandacaru_core::periodic::hubbard_response(
                    &self.crystal,
                    &self.inner,
                    &alphas,
                )
            })
            .map_err(error)?;

        let out = PyDict::new(py);
        let dmatrix_to_py = |m: &nalgebra::DMatrix<f64>| -> PyResult<Array<'py, f64>> {
            let rows = m.nrows();
            let cols = m.ncols();
            let data: Vec<f64> = (0..rows)
                .flat_map(|r| (0..cols).map(move |c| m[(r, c)]))
                .collect();
            array(py, &[rows, cols], data)
        };
        out.set_item("u_matrix", dmatrix_to_py(&res.u_matrix)?)?;
        out.set_item("u_charge", dmatrix_to_py(&res.u_charge)?)?;
        out.set_item("chi0", dmatrix_to_py(&res.chi0)?)?;
        out.set_item("chi", dmatrix_to_py(&res.chi)?)?;
        out.set_item("alphas", res.alphas)?;
        if let Some(j) = res.j_response {
            out.set_item("j_response", j)?;
        }
        Ok(out)
    }
    fn wannier_setup(
        &self,
        py: Python<'_>,
        metadata_json: &str,
    ) -> PyResult<super::wannier::PyWannierSetup> {
        let inner = py
            .detach(|| {
                mandacaru_core::periodic::wannier::setup::Setup::from_json(
                    self.crystal.clone(),
                    self.inner.clone(),
                    metadata_json,
                )
            })
            .map_err(error)?;
        Ok(super::wannier::PyWannierSetup { inner: Some(inner) })
    }
    /// The analytic force `-dF/dtau` per atom, Hartree/Bohr, shape `(n_atoms, 3)`, of the
    /// converged crystal, rebuilt in Rust from the crystal's own setup description. Raises
    /// `NotImplementedError` naming the cause for everything the engine does not differentiate.
    fn forces<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, f64>> {
        let forces = py
            .detach(|| mandacaru_core::periodic::response::forces(&self.crystal, &self.inner))
            .map_err(error)?;
        array(
            py,
            &[forces.len(), 3],
            forces.into_iter().flatten().collect(),
        )
    }
    /// The analytic stress at the pins of the crystal, Hartree/Bohr^3: `sigma = (g + g^T) / (2 Omega)`
    /// with `g_ab = dF/d eps_ab` (`gradient`, Hartree, unsymmetrized), the antisymmetric part
    /// `(g - g^T) / 2` (Hartree, a diagnostic) and the cell `volume` (Bohr^3). The 3x3 arrays are
    /// row-major `[a][b]`; the Voigt order and units are the Python caller's.
    fn stress<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stress = py
            .detach(|| mandacaru_core::periodic::response::stress(&self.crystal, &self.inner))
            .map_err(error)?;
        let out = PyDict::new(py);
        for (name, tensor) in [
            ("sigma", stress.sigma),
            ("gradient", stress.gradient),
            ("antisymmetric", stress.antisymmetric),
        ] {
            out.set_item(
                name,
                array(py, &[3, 3], tensor.into_iter().flatten().collect())?,
            )?;
        }
        out.set_item("volume", stress.volume)?;
        Ok(out)
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
                array(py, &[s, n], tau.as_slice().to_vec())?,
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
        let (eps, projections, _memory) = py
            .detach(|| {
                let v: serde_json::Value = serde_json::from_str(&owned)
                    .map_err(|e| mandacaru_core::Error::Invalid(e.to_string()))?;
                let raw = v.as_array().filter(|a| !a.is_empty()).ok_or_else(|| {
                    mandacaru_core::Error::Invalid("bands need a nonempty k-point list".into())
                })?;
                let memory = c.context.reserve(product(&[
                    raw.len(),
                    c.spins,
                    c.bands(),
                    c.bands()
                        .checked_add(c.p)
                        .ok_or(mandacaru_core::Error::Overflow)?,
                    16,
                    16,
                ])?)?;
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
                let mut eps = vec![];
                let mut projections = vec![];
                let n = c.grid.index.points();
                let is_hybrid = c.screened_exchange()?.is_some();
                for s in 0..c.spins {
                    for p in &points {
                        let exchange = if is_hybrid {
                            Some(c.hybrid_exchange_at_k(
                                &self.inner.matrices,
                                p,
                                s,
                                self.inner.options.exxdiv_treatment,
                            )?)
                        } else {
                            None
                        };
                        let (e, _, y) = mandacaru_core::periodic::solver::diagonalize_with_exchange(
                            c,
                            p,
                            if c.spinor {
                                self.inner.potential.as_slice()
                            } else {
                                &self.inner.potential.as_slice()[s * n..(s + 1) * n]
                            },
                            self.inner.moment_potential.as_slice(),
                            self.inner
                                .kinetic_potential
                                .as_ref()
                                .map(|v| &v.as_slice()[s * n..(s + 1) * n]),
                            // The converged DFT+U operator of this spin channel.
                            self.inner.hubbard.as_ref().map(|h| &h.operators[s]),
                            exchange.as_ref(),
                        )?;
                        eps.extend(e.iter().map(|e| e + c.reference));
                        for a in 0..c.bands() {
                            for b in 0..c.bands() {
                                projections.push(y[(a, b)].norm_sqr());
                            }
                        }
                    }
                }
                Ok::<_, mandacaru_core::Error>((eps, projections, memory))
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
}
