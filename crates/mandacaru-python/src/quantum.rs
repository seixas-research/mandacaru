use super::{array, copy, error, Array};
use mandacaru_core::{
    math::storage::Precision,
    quantum::{
        optimizer, Amplitude, Basis, CachedProblem, Encoding, Operator, SolveOptions, Space, State,
        C64,
    },
    Context, Reservation,
};
use numpy::{Complex32, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};

/// One executor call: hand `requests` to the Python executor and copy its answer (states,
/// expectations or a stop) into Rust. The executor holds the GIL only for the call.
fn call_executor(
    context: &Context,
    executor: &Py<PyAny>,
    requests: &[mandacaru_core::quantum::execution::Request],
) -> PyResult<mandacaru_core::quantum::execution::Response> {
    use mandacaru_core::quantum::execution::Response;
    Python::attach(|py| {
        let batch: Vec<_> = requests
            .iter()
            .map(|r| (r.labels.clone(), r.angles.clone(), r.observables.clone(), r.priced))
            .collect();
        let response = executor.bind(py).call1((batch,))?;
        if let Ok(stop) = response.get_item("stop") {
            return Ok(Response::Stop(stop.extract()?));
        }
        if let Ok(states) = response.get_item("states") {
            let states: Vec<PyReadonlyArray1<'_, C64>> = states.extract()?;
            let mut owned = Vec::with_capacity(states.len());
            for state in &states {
                let mut copy = context.allocate(state.len()).map_err(error)?;
                for (out, value) in copy.as_mut_slice().iter_mut().zip(state.as_array()) {
                    *out = *value;
                }
                owned.push(copy);
            }
            Ok(Response::States(owned))
        } else {
            Ok(Response::Expectations(response.get_item("expectations")?.extract()?))
        }
    })
}

/// A validated state copied in from NumPy, in the dtype of its precision.
/// The reservation keeps the copy charged to the budget while it is alive.
#[allow(dead_code)]
pub(crate) enum Input {
    Float64(Vec<C64>, Reservation),
    Float32(Vec<Complex32>, Reservation),
}
impl Input {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Float64(v, _) => v.len(),
            Self::Float32(v, _) => v.len(),
        }
    }
}
/// Run `$body` with `$values` bound to the typed slice of an [`Input`].
macro_rules! typed {
    ($input:expr, $values:ident => $body:expr) => {
        match $input {
            $crate::quantum::Input::Float64(ref $values, _) => $body,
            $crate::quantum::Input::Float32(ref $values, _) => $body,
        }
    };
}
pub(crate) use typed;

/// The space a problem is built in: the sector or the whole register, or, `register_free` (a
/// shot-based run), no register at all, where `sector` asks for the algebraic number-conservation
/// check.
pub(crate) fn space(register_free: bool, sector: bool) -> Space {
    if register_free {
        Space::Pauli {
            number_conserving: sector,
        }
    } else if sector {
        Space::Sector
    } else {
        Space::Register
    }
}
/// Build the sparse Hamiltonian matrix of `problem` (refused when it is register-free).
pub(crate) fn attach_sparse(py: Python<'_>, problem: &mut CachedProblem) -> PyResult<()> {
    let matrix = py
        .detach(|| {
            mandacaru_core::quantum::sparse::SparseMatrix::new(
                &problem.context,
                problem.register("a sparse Hamiltonian matrix")?,
                &problem.hamiltonian,
            )
        })
        .map_err(error)?;
    problem.sparse = Some(matrix);
    Ok(())
}
pub(crate) fn parse_precision(name: &str) -> PyResult<Precision> {
    Precision::parse(name).map_err(error)
}
fn dtype_name(precision: Precision) -> &'static str {
    match precision {
        Precision::Float64 => "complex128",
        Precision::Float32 => "complex64",
    }
}
/// Copy a state array in. Its dtype must be the one of `precision`; a state
/// of the other dtype raises `TypeError` instead of being cast.
pub(crate) fn input_state(
    ctx: &Context,
    precision: Precision,
    state: &Bound<'_, PyAny>,
) -> PyResult<Input> {
    let mismatch = || -> PyErr {
        let got = state
            .getattr("dtype")
            .and_then(|d| d.str().map(|s| s.to_string()))
            .unwrap_or_else(|_| "a non-array".into());
        PyTypeError::new_err(format!(
            "state has dtype {got}, but precision='{}' requires {}; convert it explicitly",
            precision.name(),
            dtype_name(precision)
        ))
    };
    match precision {
        Precision::Float64 => {
            let array: PyReadonlyArray1<'_, C64> = state.extract().map_err(|_| mismatch())?;
            let (values, memory) = copy(ctx, &array)?;
            Ok(Input::Float64(values, memory))
        }
        Precision::Float32 => {
            let array: PyReadonlyArray1<'_, Complex32> = state.extract().map_err(|_| mismatch())?;
            let (values, memory) = copy(ctx, &array)?;
            Ok(Input::Float32(values, memory))
        }
    }
}
/// Hand a state to NumPy as an owned array of its own dtype (moved, not copied).
pub(crate) fn state_array<'py>(
    py: Python<'py>,
    shape: &[usize],
    state: State,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(match state {
        State::Float64(buffer) => array(py, shape, buffer.into_parts().0)?.into_any(),
        State::Float32(buffer) => array(py, shape, buffer.into_parts().0)?.into_any(),
    })
}

/// Run `$body` with `$values` bound to the slice of a complex64 or complex128
/// state array; any other array raises `TypeError`. The array is only read,
/// under the GIL and its read-only borrow, so nothing is copied or detached.
macro_rules! read_state {
    ($state:expr, $values:ident => $body:expr) => {{
        let state = $state;
        if let Ok(array) = state.extract::<PyReadonlyArray1<'_, C64>>() {
            let $values = array
                .as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            $body
        } else if let Ok(array) = state.extract::<PyReadonlyArray1<'_, Complex32>>() {
            let $values = array
                .as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            $body
        } else {
            return Err(PyTypeError::new_err(
                "a state must be a one-dimensional complex64 or complex128 array",
            ));
        }
    }};
}

/// `<psi|psi>` of a stored state, summed in float64 whatever its dtype.
#[pyfunction]
pub fn state_norm_squared(state: &Bound<'_, PyAny>) -> PyResult<f64> {
    read_state!(state, values => Ok(mandacaru_core::quantum::amplitude::norm_squared(values)))
}

/// `sqrt(<psi|psi>)` of a stored state, summed in float64 whatever its dtype.
#[pyfunction]
pub fn state_norm(state: &Bound<'_, PyAny>) -> PyResult<f64> {
    read_state!(state, values => Ok(mandacaru_core::quantum::amplitude::norm(values)))
}

/// Raise `ValueError` unless `|<psi|psi> - 1|` is within the dtype's input
/// tolerance (1e-10 for complex128, 1e-5 for complex64), summed in float64.
#[pyfunction]
pub fn require_normalized_state(state: &Bound<'_, PyAny>) -> PyResult<()> {
    let off = |norm: f64, tolerance: f64| {
        if (norm - 1.).abs() <= tolerance {
            Ok(())
        } else {
            Err(PyValueError::new_err("initial_state must be normalized"))
        }
    };
    if let Ok(array) = state.extract::<PyReadonlyArray1<'_, C64>>() {
        let values = array
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        off(
            mandacaru_core::quantum::amplitude::norm_squared(values),
            <C64 as Amplitude>::INPUT_NORM_TOLERANCE,
        )
    } else if let Ok(array) = state.extract::<PyReadonlyArray1<'_, Complex32>>() {
        let values = array
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        off(
            mandacaru_core::quantum::amplitude::norm_squared(values),
            <Complex32 as Amplitude>::INPUT_NORM_TOLERANCE,
        )
    } else {
        Err(PyTypeError::new_err(
            "a state must be a one-dimensional complex64 or complex128 array",
        ))
    }
}

/// `sum conj(a) b` of two states of the same dtype, summed in float64.
/// Returns `(real, imaginary)`.
#[pyfunction]
pub fn state_inner(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    let mismatch = || {
        PyTypeError::new_err("both states must be one-dimensional arrays of the same complex dtype")
    };
    if let (Ok(x), Ok(y)) = (
        a.extract::<PyReadonlyArray1<'_, C64>>(),
        b.extract::<PyReadonlyArray1<'_, C64>>(),
    ) {
        let (x, y) = (
            x.as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            y.as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
        if x.len() != y.len() {
            return Err(PyValueError::new_err("states must have the same length"));
        }
        let v = mandacaru_core::quantum::amplitude::inner(x, y);
        Ok((v.re, v.im))
    } else if let (Ok(x), Ok(y)) = (
        a.extract::<PyReadonlyArray1<'_, Complex32>>(),
        b.extract::<PyReadonlyArray1<'_, Complex32>>(),
    ) {
        let (x, y) = (
            x.as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            y.as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
        if x.len() != y.len() {
            return Err(PyValueError::new_err("states must have the same length"));
        }
        let v = mandacaru_core::quantum::amplitude::inner(x, y);
        Ok((v.re, v.im))
    } else {
        Err(mismatch())
    }
}

#[pyclass(name = "PauliOperator", module = "mandacaru._native", frozen)]
pub struct PyOperator {
    pub context: Context,
    pub inner: Operator,
}
#[pymethods]
impl PyOperator {
    #[pyo3(signature=(state,evaluation_qubits,window,*,keep_amplitudes=true,precision="float64"))]
    fn qpe<'py>(
        &self,
        py: Python<'py>,
        state: &Bound<'py, PyAny>,
        evaluation_qubits: usize,
        window: [f64; 2],
        keep_amplitudes: bool,
        precision: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        super::advanced::qpe(
            py,
            self,
            state,
            evaluation_qubits,
            window,
            keep_amplitudes,
            parse_precision(precision)?,
        )
    }
    /// `(lo, hi)`: `window`, or the Gershgorin bound widened by `1 + 2/2^t` (`dynamics::qpe_window`).
    #[pyo3(signature=(evaluation_qubits, window=None))]
    fn qpe_window(&self, evaluation_qubits: usize, window: Option<[f64; 2]>) -> PyResult<(f64, f64)> {
        let [lo, hi] = mandacaru_core::quantum::dynamics::qpe_window(&self.inner, evaluation_qubits, window)
            .map_err(error)?;
        Ok((lo, hi))
    }
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.n
    }
    #[getter]
    fn terms(&self) -> Vec<(String, f64, f64)> {
        self.inner
            .labels()
            .into_iter()
            .map(|(l, c)| (l, c.re, c.im))
            .collect()
    }
    fn dagger(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.dagger(&self.context))
            .map_err(error)?;
        Ok(Self {
            context: self.context.clone(),
            inner,
        })
    }
    #[pyo3(signature=(other,real=1.0,imaginary=0.0))]
    fn add(&self, py: Python<'_>, other: &Self, real: f64, imaginary: f64) -> PyResult<Self> {
        let inner = py
            .detach(|| {
                self.inner
                    .add(&self.context, &other.inner, C64::new(real, imaginary))
            })
            .map_err(error)?;
        Ok(Self {
            context: self.context.clone(),
            inner,
        })
    }
    fn compose(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.compose(&self.context, &other.inner))
            .map_err(error)?;
        Ok(Self {
            context: self.context.clone(),
            inner,
        })
    }
    fn commutator(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.commutator(&self.context, &other.inner))
            .map_err(error)?;
        Ok(Self {
            context: self.context.clone(),
            inner,
        })
    }
    fn map_to(&self, py: Python<'_>, mapping: &str, num_particles: [usize; 2]) -> PyResult<Self> {
        let inner = py
            .detach(|| {
                Encoding::new(mapping, self.inner.n, num_particles)?
                    .map_operator(&self.context, &self.inner)
            })
            .map_err(error)?;
        Ok(Self {
            context: self.context.clone(),
            inner,
        })
    }
    /// `(A + A^dagger) / 2`: the real parts of the coefficients.
    fn hermitian_part(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.hermitian_part(&self.context))
            .map_err(error)?;
        Ok(Self {
            context: self.context.clone(),
            inner,
        })
    }
    /// The matrix on the whole register as a compressed-row sparse matrix.
    fn sparse_matrix(&self, py: Python<'_>) -> PyResult<crate::sparse::PyCSRMatrix> {
        let (d, csr) = py
            .detach(|| {
                let basis = Basis::register(&self.context, self.inner.n)?;
                let csr = self.inner.csr(&self.context, &basis)?;
                Ok::<_, mandacaru_core::Error>((basis.len(), csr))
            })
            .map_err(error)?;
        let nnz = csr.nnz;
        Ok(crate::sparse::PyCSRMatrix {
            shape: (d, d),
            rust_indices: csr.columns.as_slice()[..nnz].to_vec(),
            rust_indptr: csr.offsets.as_slice().to_vec(),
            is_complex: true,
            real_data: Vec::new(),
            complex_data: csr.values.as_slice()[..nnz].to_vec(),
            has_sorted_indices: true,
            use_int32: d < i32::MAX as usize && nnz < i32::MAX as usize,
        })
    }
    fn matrix<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, C64>> {
        let (d, out) = py
            .detach(|| {
                let basis = Basis::register(&self.context, self.inner.n)?;
                let out = self.inner.dense(&self.context, &basis)?;
                Ok::<_, mandacaru_core::Error>((basis.len(), out))
            })
            .map_err(error)?;
        array(py, &[d, d], out.as_slice().to_vec())
    }
    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, C64>,
    ) -> PyResult<Array<'py, C64>> {
        let (input, _memory) = copy(&self.context, &state)?;
        let out = py
            .detach(|| {
                let basis = Basis::register(&self.context, self.inner.n)?;
                self.inner.apply(&self.context, &basis, &input)
            })
            .map_err(error)?;
        array(py, &[input.len()], out.as_slice().to_vec())
    }
}

/// Most energy evaluations the named optimizer can request, or `None` when
/// an objective call is not one energy evaluation (the gradient methods).
#[pyfunction]
#[pyo3(signature=(method,max_iterations,max_evaluations,track_best=true))]
pub fn optimizer_evaluation_bound(
    method: &str,
    max_iterations: usize,
    max_evaluations: usize,
    track_best: bool,
) -> PyResult<Option<usize>> {
    Ok(optimizer::Options {
        method: optimizer::Method::parse(method).map_err(error)?,
        max_iterations,
        max_evaluations,
        track_best,
        ..Default::default()
    }
    .evaluation_bound())
}

#[pyclass(name = "QuantumProblem", module = "mandacaru._native", frozen)]
pub struct PyProblem {
    pub inner: CachedProblem,
}
#[pymethods]
impl PyProblem {
    #[pyo3(signature=(*,max_steps=200,max_length=20,move_weights=[1.0;3],temperatures=[0.003674932217565499,0.00003674932217565499],length_penalty=0.0,proposal_temperature=0.1,proposal="gradient",warm_start=true,replace_start="zero",screen_insertions=0,record=false,seed=0,optimizer="BFGS",max_iterations=300,max_evaluations=5000,tolerance=Some(1e-9),track_best=true,initial_indices=None,initial_parameters=None,proposal_model=None,transfer_steps=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        max_steps: usize,
        max_length: usize,
        move_weights: [f64; 3],
        temperatures: [f64; 2],
        length_penalty: f64,
        proposal_temperature: f64,
        proposal: &str,
        warm_start: bool,
        replace_start: &str,
        screen_insertions: usize,
        record: bool,
        seed: u64,
        optimizer: &str,
        max_iterations: usize,
        max_evaluations: usize,
        tolerance: Option<f64>,
        track_best: bool,
        initial_indices: Option<Vec<usize>>,
        initial_parameters: Option<Vec<f64>>,
        proposal_model: Option<&str>,
        transfer_steps: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        use mandacaru_core::quantum::search::driver::Options;
        if self.inner.taper.is_some() {
            return Err(PyValueError::new_err(
                "architecture search builds its reference from the untapered register and cannot run on a tapered one",
            ));
        }
        if !["gradient", "uniform"].contains(&proposal)
            || !["zero", "inherit"].contains(&replace_start)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "search proposals are gradient or uniform; replacement starts are zero or inherit",
            ));
        }
        let options = Options {
            max_steps,
            transfer_steps,
            max_length,
            move_weights,
            temperatures,
            length_penalty,
            proposal_temperature,
            uniform: proposal == "uniform",
            warm_start,
            replace_inherit: replace_start == "inherit",
            screen_insertions,
            record,
            seed,
            optimizer: optimizer::Options {
                method: optimizer::Method::parse(optimizer).map_err(error)?,
                max_iterations,
                max_evaluations,
                tolerance,
                seed,
                track_best,
                ..Default::default()
            },
        };
        let mut result = py
            .detach(|| {
                let model = proposal_model
                    .map(|s| {
                        mandacaru_core::quantum::search::learning::training::Model::from_json(
                            &self.inner.context,
                            s,
                        )
                    })
                    .transpose()?;
                self.inner.search_with_model(
                    &options,
                    &initial_indices.unwrap_or_default(),
                    &initial_parameters.unwrap_or_default(),
                    model.as_ref(),
                )
            })
            .map_err(error)?;
        let mut details = serde_json::json!({"acceptance_by_move":result.acceptance,"optimal_cost_ha":result.optimal_cost,
            "best_energy_ha":result.best_energy,"best_energy_indices":result.best_energy_indices,"best_energy_parameters":result.best_energy_parameters,
            "final_energy_ha":result.final_energy,"final_indices":result.final_indices,
            "final_parameters":result.final_parameters,"num_architectures":result.num_architectures,"num_screenings":result.num_screenings,
            "optimizer_failures":result.failures,"seed":seed,"completed":true});
        details["steps"] = serde_json::Value::Array(std::mem::take(&mut result.steps));
        details["records"] = serde_json::Value::Array(std::mem::take(&mut result.records));
        let details = serde_json::to_string(&details)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let out = solution_dict(py, result.solution)?;
        out.set_item("search_json", details)?;
        Ok(out)
    }
    #[pyo3(signature=(indices,*,pairs=3000,seed=0,product_form=true))]
    fn sample_fidelities<'py>(
        &self,
        py: Python<'py>,
        indices: Vec<usize>,
        pairs: usize,
        seed: u64,
        product_form: bool,
    ) -> PyResult<Array<'py, f64>> {
        let samples = py
            .detach(|| {
                self.inner
                    .sample_fidelities(&indices, pairs, seed, product_form)
            })
            .map_err(error)?;
        array(py, &[pairs], samples.as_slice().to_vec())
    }
    #[pyo3(signature=(*,layers=2,evolution="exact",order=2,steps=1,seed_angle=0.1))]
    fn hva(
        &self,
        py: Python<'_>,
        layers: usize,
        evolution: &str,
        order: usize,
        steps: usize,
        seed_angle: f64,
    ) -> PyResult<Self> {
        if self.inner.taper.is_some() {
            return Err(PyValueError::new_err(
                "the Hamiltonian-variational ansatz is built from the untapered Hamiltonian groups and cannot be tapered",
            ));
        }
        let inner = py
            .detach(|| {
                let encoding = Encoding::new(
                    &self.inner.encoding.mapping,
                    self.inner.encoding.modes,
                    self.inner.encoding.particles,
                )?;
                let h = self.inner.hamiltonian.copy_into(&self.inner.context)?;
                let mut problem = CachedProblem::new_in(
                    &self.inner.context,
                    encoding,
                    h,
                    if self.inner.is_register_free() {
                        Space::Pauli {
                            number_conserving: false,
                        }
                    } else {
                        Space::Register
                    },
                    self.inner.dense.is_some(),
                    true,
                )?;
                problem.precision = self.inner.precision;
                problem.use_hva(layers, evolution, order, steps, seed_angle)?;
                Ok::<_, mandacaru_core::Error>(problem)
            })
            .map_err(error)?;
        let mut inner = inner;
        if self.inner.sparse.is_some() {
            attach_sparse(py, &mut inner)?;
        }
        Ok(Self { inner })
    }
    #[getter]
    fn exact_hva(&self) -> bool {
        self.inner.hva.as_ref().is_some_and(|h| !h.trotter)
    }
    #[pyo3(signature=(*,num_states=2,restarts=4,beta=None,seed=0,trotter=false,optimizer="BFGS",max_iterations=300,max_evaluations=5000,tolerance=1e-9))]
    fn energy_levels<'py>(
        &self,
        py: Python<'py>,
        num_states: usize,
        restarts: usize,
        beta: Option<f64>,
        seed: u64,
        trotter: bool,
        optimizer: &str,
        max_iterations: usize,
        max_evaluations: usize,
        tolerance: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let options = SolveOptions {
            trotter,
            optimizer: optimizer::Options {
                method: optimizer::Method::parse(optimizer).map_err(error)?,
                max_iterations,
                max_evaluations,
                tolerance: Some(tolerance),
                seed,
                track_best: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let dimension = self.dimension()?;
        let result = py
            .detach(|| {
                self.inner
                    .energy_levels(&options, num_states, restarts, beta, seed)
            })
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item("energies_ha", array(py, &[num_states], result.energies)?)?;
        out.set_item(
            "states",
            state_array(py, &[num_states, dimension], result.states)?,
        )?;
        out.set_item("parameters", result.parameters)?;
        out.set_item("success", result.success)?;
        out.set_item("messages", result.messages)?;
        out.set_item("num_evaluations", result.evaluations)?;
        out.set_item("reference_ha", result.reference)?;
        out.set_item("elapsed_seconds", result.elapsed_seconds)?;
        Ok(out)
    }
    fn expand_state<'py>(
        &self,
        py: Python<'py>,
        state: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape: Vec<usize> = state.getattr("shape")?.extract()?;
        let basis = self.register("expand_state")?;
        let dimension = basis.len();
        if !(1..=2).contains(&shape.len()) || shape[shape.len() - 1] != dimension {
            return Err(PyValueError::new_err(
                "states need shape (sector_dimension,) or (states,sector_dimension)",
            ));
        }
        let rows = if shape.len() == 1 { 1 } else { shape[0] };
        let n = 1usize
            .checked_shl(basis.n as u32)
            .ok_or_else(|| error(mandacaru_core::Error::Overflow))?;
        let flat = state.call_method1("reshape", (rows * dimension,))?;
        let input = input_state(&self.inner.context, self.inner.precision, &flat)?;
        let out = py
            .detach(|| {
                typed!(input, values => {
                    self.expand_rows(values, rows, n)
                })
            })
            .map_err(error)?;
        let output_shape = if shape.len() == 1 {
            vec![n]
        } else {
            vec![rows, n]
        };
        state_array(py, &output_shape, out)
    }
    #[pyo3(signature=(parameters,*,trotter=false,indices=None))]
    fn state<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        trotter: bool,
        indices: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let indices = indices.unwrap_or_else(|| {
            (0..self.inner.pool.len())
                .filter(|&i| self.inner.pool_screenable[i])
                .collect()
        });
        let out = py
            .detach(|| {
                let state = self.inner.prepare_state(&indices, &parameters, trotter)?;
                let ctx = &self.inner.context;
                let basis = self.inner.register("state")?;
                Ok::<_, mandacaru_core::Error>(match &state {
                    State::Float64(v) => State::Float64(basis.expand_state(ctx, v.as_slice())?),
                    State::Float32(v) => State::Float32(basis.expand_state(ctx, v.as_slice())?),
                })
            })
            .map_err(error)?;
        let length = out.len();
        state_array(py, &[length], out)
    }
    #[pyo3(signature=(indices,parameters,*,product_form=true,energy_ha,status_json="{}",method="adapt-vqe",hessian=None,hessian_optimizer=None))]
    #[allow(clippy::too_many_arguments)]
    fn checkpoint(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
        parameters: Vec<f64>,
        product_form: bool,
        energy_ha: f64,
        status_json: &str,
        method: &str,
        hessian: Option<PyReadonlyArray2<'_, f64>>,
        hessian_optimizer: Option<&str>,
    ) -> PyResult<super::persistence::PySnapshot> {
        let hessian: Option<Vec<f64>> = hessian.map(|h| h.as_array().iter().copied().collect());
        if hessian.is_some() && hessian_optimizer.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a checkpoint Hessian needs the optimizer that kept it (hessian_optimizer)",
            ));
        }
        let inner = py
            .detach(|| {
                if status_json.len() > mandacaru_core::persistence::MAX_JSON_BYTES {
                    return Err(mandacaru_core::Error::Invalid(
                        "checkpoint status exceeds JSON limit".into(),
                    ));
                }
                let _status = self
                    .inner
                    .context
                    .reserve(mandacaru_core::product(&[status_json.len() + 1, 32])?)?;
                let status = serde_json::from_str(status_json)
                    .map_err(|e| mandacaru_core::Error::Invalid(e.to_string()))?;
                let status = mandacaru_core::persistence::status_with_hessian(
                    status,
                    hessian_optimizer.unwrap_or(""),
                    hessian.as_deref(),
                )?;
                let doc = mandacaru_core::persistence::checkpoint(
                    &self.inner.context,
                    &self.inner,
                    &indices,
                    &parameters,
                    product_form,
                    method,
                    status,
                    energy_ha,
                )?;
                Ok::<_, mandacaru_core::Error>(doc)
            })
            .map_err(error)?;
        Ok(super::persistence::PySnapshot {
            context: self.inner.context.clone(),
            inner,
        })
    }
    /// A subspace checkpoint: the shared ansatz, the state weights and the sorted energies.
    #[pyo3(signature=(indices, parameters, *, adapt, product_form, weights, energies_ha, status_json, hessian=None, hessian_optimizer=None))]
    #[allow(clippy::too_many_arguments)]
    fn subspace_checkpoint(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
        parameters: Vec<f64>,
        adapt: bool,
        product_form: bool,
        weights: Vec<f64>,
        energies_ha: Vec<f64>,
        status_json: &str,
        hessian: Option<PyReadonlyArray2<'_, f64>>,
        hessian_optimizer: Option<&str>,
    ) -> PyResult<super::persistence::PySnapshot> {
        let hessian: Option<Vec<f64>> = hessian.map(|h| h.as_array().iter().copied().collect());
        if hessian.is_some() && hessian_optimizer.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a checkpoint Hessian needs the optimizer that kept it (hessian_optimizer)",
            ));
        }
        let inner = py
            .detach(|| {
                if status_json.len() > mandacaru_core::persistence::MAX_JSON_BYTES {
                    return Err(mandacaru_core::Error::Invalid(
                        "checkpoint status exceeds JSON limit".into(),
                    ));
                }
                let status = serde_json::from_str(status_json)
                    .map_err(|e| mandacaru_core::Error::Invalid(e.to_string()))?;
                let status = mandacaru_core::persistence::status_with_hessian(
                    status,
                    hessian_optimizer.unwrap_or(""),
                    hessian.as_deref(),
                )?;
                mandacaru_core::persistence::subspace_checkpoint(
                    &self.inner.context,
                    &self.inner,
                    &indices,
                    &parameters,
                    adapt,
                    product_form,
                    &weights,
                    &energies_ha,
                    status,
                )
            })
            .map_err(error)?;
        Ok(super::persistence::PySnapshot {
            context: self.inner.context.clone(),
            inner,
        })
    }
    #[pyo3(signature=(weights,*,adapt=false,trotter=false,optimizer="BFGS",max_iterations=300,max_evaluations=5000,tolerance=Some(1e-9),track_best=true,max_growth=50,gradient_tolerance=Some(1e-7),energy_tolerance=None,initial_parameters=None,resume=None))]
    fn solve_subspace<'py>(
        &self,
        py: Python<'py>,
        weights: Vec<f64>,
        adapt: bool,
        trotter: bool,
        optimizer: &str,
        max_iterations: usize,
        max_evaluations: usize,
        tolerance: Option<f64>,
        track_best: bool,
        max_growth: usize,
        gradient_tolerance: Option<f64>,
        energy_tolerance: Option<f64>,
        initial_parameters: Option<Vec<f64>>,
        resume: Option<&super::persistence::PySnapshot>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let method = optimizer::Method::parse(optimizer).map_err(error)?;
        if resume.is_some() && initial_parameters.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "choose initial parameters or a resume checkpoint",
            ));
        }
        // The subspace methods always re-optimize every parameter and grow one operator per
        // step (`quenching=False`, `tetris` and `prune` are refused by the calculator).
        let options = SolveOptions {
            adapt,
            trotter,
            quenching: true,
            tetris: false,
            prune: false,
            checkpoint: None,
            max_growth,
            gradient_tolerance,
            energy_tolerance,
            optimizer: optimizer::Options {
                method,
                max_iterations,
                max_evaluations,
                tolerance,
                seed: 0,
                track_best,
                ..Default::default()
            },
            hessian: None,
            continuation: false,
        };
        let dimension = self.dimension()?;
        let result = py
            .detach(|| match resume {
                Some(snapshot) => snapshot.inner.resume_subspace(&self.inner, &options, &weights),
                None => self
                    .inner
                    .solve_subspace(&options, &weights, initial_parameters.as_deref()),
            })
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item("energies_ha", result.energies)?;
        out.set_item(
            "parameters",
            array(py, &[result.parameters.len()], result.parameters)?,
        )?;
        out.set_item(
            "states",
            state_array(py, &[weights.len(), dimension], result.states)?,
        )?;
        out.set_item("reference_ha", result.reference)?;
        out.set_item(
            "operators",
            result
                .indices
                .iter()
                .map(|&i| self.inner.pool[i].label.clone())
                .collect::<Vec<_>>(),
        )?;
        let operators = result.indices.len();
        out.set_item("indices", result.indices)?;
        out.set_item(
            "hessian",
            match result.hessian {
                Some(h) => Some(array(py, &[operators, operators], h)?),
                None => None,
            },
        )?;
        out.set_item("notes", result.notes)?;
        out.set_item("reference_indices", result.reference_indices)?;
        out.set_item("history_ha", result.history)?;
        out.set_item("optimizer_failures", result.optimizer_failures)?;
        out.set_item("success", result.success)?;
        out.set_item("message", result.message)?;
        out.set_item("final_max_gradient", result.final_gradient)?;
        out.set_item("elapsed_seconds", result.elapsed_seconds)?;
        Ok(out)
    }
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.n_qubits()
    }
    /// What Z2 tapering removed, or `None` for an untapered register: the register widths, the
    /// anchor qubits, the sector signs, the symmetry generators and the pool generators dropped as
    /// symmetry-changing.
    #[getter]
    fn taper<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(taper) = &self.inner.taper else {
            return Ok(None);
        };
        let out = PyDict::new(py);
        out.set_item("qubits_before", taper.n)?;
        out.set_item("qubits", taper.width())?;
        out.set_item("removed", taper.anchors.len())?;
        out.set_item("anchors", taper.anchors.clone())?;
        out.set_item("signs", taper.signs.clone())?;
        out.set_item("generators", taper.labels())?;
        out.set_item(
            "dropped",
            self.inner.dropped.iter().map(|(_, label)| label.clone()).collect::<Vec<_>>(),
        )?;
        Ok(Some(out))
    }
    /// The factorized energy measurement this problem was built with, or `None` for the
    /// qubit-wise commuting groups: the number of measurement bases (`L + 1`, each a single
    /// commuting group), the largest number of Givens rotations a basis change needs, and the
    /// Frobenius norm of the two-body tensor the leaf cutoff discarded.
    #[getter]
    fn factorized_measurement<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(plan) = &self.inner.factorized else {
            return Ok(None);
        };
        let out = PyDict::new(py);
        out.set_item("bases", plan.measurement_bases())?;
        out.set_item(
            "rotations",
            plan.bases.iter().map(|b| b.rotation.len() / 2).max().unwrap_or(0),
        )?;
        out.set_item("truncation_error", plan.truncation_error)?;
        Ok(Some(out))
    }
    #[getter]
    fn n_spatial_orbitals(&self) -> usize {
        self.inner.encoding.modes / 2
    }
    #[getter]
    fn num_particles(&self) -> [usize; 2] {
        self.inner.encoding.particles
    }
    #[getter]
    fn reference_qubits(&self) -> Vec<usize> {
        let n = self.inner.n_qubits();
        let reference = self.inner.register_reference();
        (0..n)
            .filter(|&q| reference & (1u64 << (n - 1 - q)) != 0)
            .collect()
    }
    #[getter]
    fn mapping(&self) -> String {
        self.inner.encoding.mapping.clone()
    }
    #[getter]
    fn dimension(&self) -> PyResult<usize> {
        Ok(self.register("dimension")?.len())
    }
    #[getter]
    fn pool_labels(&self) -> Vec<String> {
        self.inner.pool.iter().map(|g| g.label.clone()).collect()
    }
    #[getter]
    fn pool_screenable(&self) -> Vec<bool> {
        self.inner.pool_screenable.clone()
    }
    /// Every pool commutator `[H, A_k]` as its Pauli terms, all at once: for the independent
    /// checks of small problems. A measured run reads `pool_paulis`, which never holds them all.
    #[getter]
    fn pool_observables(&self, py: Python<'_>) -> PyResult<Vec<Vec<(String, f64, f64)>>> {
        py.detach(|| {
            self.inner
                .pool
                .iter()
                .map(|g| {
                    let operator = self
                        .inner
                        .hamiltonian
                        .commutator(&self.inner.context, &g.operator)?;
                    Ok(operator
                        .labels()
                        .into_iter()
                        .map(|(l, c)| (l, c.re, c.im))
                        .collect())
                })
                .collect::<mandacaru_core::Result<Vec<_>>>()
        })
        .map_err(error)
    }
    /// The distinct Pauli strings of every pool commutator `[H, A_k]` and how many of the
    /// commutators are not zero, found in bounded batches (`quantum::pool_measurement`). With a
    /// `limit` the walk stops as soon as the distinct strings pass it: `labels` is then empty
    /// and `over` holds the strings and generators seen (a lower bound of what the pool needs).
    #[pyo3(signature = (limit=None))]
    fn pool_paulis<'py>(
        &self,
        py: Python<'py>,
        limit: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        use mandacaru_core::quantum::pool_measurement::{batch_bytes, PoolWalk};
        let walk = py
            .detach(|| {
                self.inner
                    .pool_paulis_within(batch_bytes(&self.inner.context), limit)
            })
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item("pool_size", self.inner.pool.len())?;
        match walk {
            PoolWalk::Planned(paulis) => {
                out.set_item("labels", paulis.set.labels())?;
                out.set_item("measured_operators", paulis.measured)?;
                out.set_item("over", py.None())?;
            }
            PoolWalk::Over {
                distinct,
                generators,
            } => {
                out.set_item("labels", Vec::<String>::new())?;
                out.set_item("measured_operators", 0)?;
                let over = PyDict::new(py);
                over.set_item("strings", distinct)?;
                over.set_item("generators", generators)?;
                out.set_item("over", over)?;
            }
        }
        Ok(out)
    }
    #[getter]
    fn pool_terms(&self) -> Vec<Vec<(String, f64, f64)>> {
        self.inner
            .pool
            .iter()
            .map(|g| {
                g.operator
                    .labels()
                    .into_iter()
                    .map(|(l, c)| (l, c.re, c.im))
                    .collect()
            })
            .collect()
    }
    #[getter]
    fn basis_indices<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, u64>> {
        let basis = self.register("basis_indices")?;
        array(py, &[basis.len()], basis.indices.as_slice().to_vec())
    }
    #[getter]
    fn register_free(&self) -> bool {
        self.inner.is_register_free()
    }
    #[getter]
    fn hamiltonian_terms(&self) -> Vec<(String, f64, f64)> {
        self.inner
            .hamiltonian
            .labels()
            .into_iter()
            .map(|(l, c)| (l, c.re, c.im))
            .collect()
    }
    /// The plan that measures the spatial RDMs on a circuit device (`quantum::rdm_measurement`).
    #[pyo3(signature=(limit=None))]
    fn rdm_measurement(&self, py: Python<'_>, limit: Option<usize>) -> PyResult<PyRdmMeasurement> {
        let inner = py
            .detach(|| {
                mandacaru_core::quantum::rdm_measurement::RdmMeasurement::of_problem(
                    &self.inner,
                    limit,
                )
            })
            .map_err(error)?;
        Ok(PyRdmMeasurement {
            context: self.inner.context.clone(),
            inner,
        })
    }
    /// The (labels, angles) of the ansatz circuit at `indices` and `parameters`
    /// (`quantum::execution::circuit_rotations`).
    fn circuit(&self, indices: Vec<usize>, parameters: Vec<f64>) -> PyResult<(Vec<String>, Vec<f64>)> {
        mandacaru_core::quantum::execution::circuit_rotations(&self.inner, &indices, &parameters)
            .map_err(error)
    }
    /// The cost of the device Hessian before any job: `(rotations, energy evaluations,
    /// circuits)`, the circuits of a `sampled` device counted in the bases of the energy measured
    /// with its error (`quantum::execution::device_hessian_plan`).
    #[pyo3(signature=(indices,parameters,*,sampled=false))]
    fn device_hessian_plan(&self, indices: Vec<usize>, parameters: Vec<f64>, sampled: bool) -> PyResult<(usize, usize, usize)> {
        mandacaru_core::quantum::execution::device_hessian_plan(&self.inner, &indices, &parameters, sampled)
            .map_err(error)
    }
    /// The energy, gradient and Hessian measured on the device through `executor` (one call),
    /// and the Z vector of the native truncated solve (`None` under the gradient floor). On a
    /// sampled executor (`executor.shots > 0`) every energy carries its standard error, the
    /// Hessian's eigenvalues are checked against theirs and `z_covariance` is `Cov(z)`
    /// (`quantum::lagrangian::measured_z`).
    #[pyo3(signature=(indices,parameters,executor,*,gradient_floor=None))]
    fn device_hessian<'py>(
        &self,
        py: Python<'py>,
        indices: Vec<usize>,
        parameters: Vec<f64>,
        executor: Py<PyAny>,
        gradient_floor: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        use mandacaru_core::quantum::execution::{CircuitExecution, Request};
        let mut execution_error = None;
        let mut options = mandacaru_core::quantum::lagrangian::Options::default();
        if let Some(floor) = gradient_floor {
            options.floor = floor;
        }
        // The executor's shots per basis (0 for an exact device): read, never assumed, so an
        // executor without them refuses instead of measuring an exact-device Hessian.
        let shots: u64 = executor.bind(py).getattr("shots")?.extract()?;
        let (measured, solved) = py
            .detach(|| {
                let mut invoke = |requests: &[Request]| {
                    call_executor(&self.inner.context, &executor, requests).map_err(|error| {
                        execution_error = Some(error);
                        mandacaru_core::Error::Kernel("circuit execution failed".into())
                    })
                };
                let mut execution = CircuitExecution::new(&mut invoke).with_shots(shots);
                let measured = execution.gradient_and_hessian(&self.inner, &indices, &parameters)?;
                let solved = mandacaru_core::quantum::lagrangian::measured_z(
                    &measured.gradient,
                    &measured.hessian,
                    &measured.energies,
                    &options,
                )?;
                Ok((measured, solved))
            })
            .map_err(|e| execution_error.take().unwrap_or_else(|| error(e)))?;
        let out = PyDict::new(py);
        let n = measured.gradient.len();
        out.set_item("energy", measured.energy)?;
        out.set_item("gradient", array(py, &[n], measured.gradient.clone())?)?;
        out.set_item("hessian", array(py, &[n, n], measured.hessian.transpose().iter().copied().collect())?)?;
        out.set_item("rotations", measured.rotations)?;
        out.set_item("energy_evaluations", measured.energy_evaluations)?;
        out.set_item("circuits", measured.circuits)?;
        out.set_item("hessian_error_unavailable", measured.error_unavailable.clone())?;
        out.set_item("gradient_norm", solved.gradient_norm)?;
        match solved.solution {
            None => out.set_item("z", py.None())?,
            Some(solution) => {
                out.set_item("z", array(py, &[n], solution.x.clone())?)?;
                out.set_item("z_vector_norm", solved.z_norm)?;
                out.set_item("hessian_cutoff", solution.cutoff)?;
                out.set_item("hessian_lowest_curvature", solution.eigenvalues[0])?;
                out.set_item("hessian_highest_curvature", solution.eigenvalues[n - 1])?;
                out.set_item("z_vector_dropped", solution.dropped)?;
                out.set_item("z_vector_dropped_gradient_norm", solution.dropped_rhs)?;
                out.set_item("hessian_lowest_curvature_error", solved.lowest_curvature_error)?;
                out.set_item("hessian_curvature_significance", solved.curvature_significance)?;
                match &solved.z_covariance {
                    None => out.set_item("z_covariance", py.None())?,
                    Some(cov) => out.set_item("z_covariance", array(py, &[n, n], cov.transpose().iter().copied().collect())?)?,
                }
            }
        }
        Ok(out)
    }
    /// The `2 R` shifted circuits `(labels, angles, parameter, coefficient)` whose energies give
    /// `z . g = sum z_parameter coefficient E`.
    fn shifted_circuits(
        &self,
        indices: Vec<usize>,
        parameters: Vec<f64>,
        z: Vec<f64>,
    ) -> PyResult<Vec<(Vec<String>, Vec<f64>, usize, f64)>> {
        mandacaru_core::quantum::execution::shifted_circuits(&self.inner, &indices, &parameters, &z).map_err(error)
    }
    #[pyo3(signature=(state,*,two_body=true))]
    fn rdms<'py>(
        &self,
        py: Python<'py>,
        state: &Bound<'py, PyAny>,
        two_body: bool,
    ) -> PyResult<(Array<'py, C64>, Option<Array<'py, C64>>)> {
        let input = input_state(&self.inner.context, self.inner.precision, state)?;
        let rdms = py
            .detach(|| typed!(input, values => self.rdms_of(values, two_body)))
            .map_err(error)?;
        let n = self.inner.encoding.modes;
        Ok((
            array(py, &[n, n], rdms.one.as_slice().to_vec())?,
            rdms.two
                .map(|two| array(py, &[n, n, n, n], two.as_slice().to_vec()))
                .transpose()?,
        ))
    }
    #[getter]
    fn precision(&self) -> &'static str {
        self.inner.precision.name()
    }
    #[pyo3(signature=(parameters,*,trotter=false,indices=None))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        trotter: bool,
        indices: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let indices = indices.unwrap_or_else(|| {
            (0..self.inner.pool.len())
                .filter(|&i| self.inner.pool_screenable[i])
                .collect()
        });
        let value = py
            .detach(|| self.inner.evaluate(&indices, &parameters, trotter))
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item("energy_ha", value.energy)?;
        out.set_item(
            "gradient",
            array(py, &[value.gradient.len()], value.gradient)?,
        )?;
        let length = value.state.len();
        out.set_item("state", state_array(py, &[length], value.state)?)?;
        Ok(out)
    }
    #[pyo3(signature=(*,adapt=false,trotter=false,quenching=true,tetris=false,prune=false,max_growth=50,gradient_tolerance=Some(1e-7),energy_tolerance=None,max_iterations=300,max_evaluations=5000,tolerance=Some(1e-9),track_best=true,initial_parameters=None,initial_indices=None,initial_hessian=None,optimizer="BFGS",resume=None,callback=None,executor=None,checkpoint_path=None,checkpoint_every=0))]
    #[allow(clippy::too_many_arguments)]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        adapt: bool,
        trotter: bool,
        quenching: bool,
        tetris: bool,
        prune: bool,
        max_growth: usize,
        gradient_tolerance: Option<f64>,
        energy_tolerance: Option<f64>,
        max_iterations: usize,
        max_evaluations: usize,
        tolerance: Option<f64>,
        track_best: bool,
        initial_parameters: Option<Vec<f64>>,
        initial_indices: Option<Vec<usize>>,
        initial_hessian: Option<PyReadonlyArray2<'_, f64>>,
        optimizer: &str,
        resume: Option<&super::persistence::PySnapshot>,
        callback: Option<Py<PyAny>>,
        executor: Option<Py<PyAny>>,
        checkpoint_path: Option<String>,
        checkpoint_every: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let method = optimizer::Method::parse(optimizer).map_err(error)?;
        if checkpoint_path.is_some() != (checkpoint_every > 0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a mid-run checkpoint needs both a path and a positive cadence",
            ));
        }
        let checkpoint = checkpoint_path.map(|path| mandacaru_core::quantum::solver::Checkpointing {
            path: path.into(),
            every: checkpoint_every,
            method: if adapt { "adapt-vqe" } else { "vqe" }.into(),
        });
        let options = SolveOptions {
            adapt,
            trotter,
            quenching,
            tetris,
            prune,
            checkpoint,
            max_growth,
            gradient_tolerance,
            energy_tolerance,
            optimizer: optimizer::Options {
                method,
                max_iterations,
                max_evaluations,
                tolerance,
                seed: 0,
                track_best,
                ..Default::default()
            },
            hessian: initial_hessian.map(|h| h.as_array().iter().copied().collect()),
            continuation: false,
        };
        if resume.is_some() && (initial_parameters.is_some() || initial_indices.is_some()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "choose initial parameters or a resume checkpoint",
            ));
        }
        if executor.is_some() && initial_indices.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "transferred sequences currently require local native execution",
            ));
        }
        let mut callback_error = None;
        let mut execution_error = None;
        let mut energy_errors: Option<(Option<f64>, Vec<Option<f64>>)> = None;
        let mut energy_error_reason: Option<String> = None;
        // Shots per basis of the executor's device (0 for an exact one), for the energy errors.
        let shots: u64 = match executor.as_ref() {
            Some(executor) => executor.bind(py).getattr("shots").and_then(|s| s.extract()).unwrap_or(0),
            None => 0,
        };
        if executor.is_some() && (resume.is_some() || (!adapt && !trotter)) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "circuit execution requires a product ansatz and a fresh solve",
            ));
        }
        let result = py
            .detach(|| {
                let mut observer = |indices: &[usize], parameters: &[f64], energy: f64| {
                    Python::attach(|py| {
                        let invoke = || -> PyResult<()> {
                            let info = PyDict::new(py);
                            info.set_item("indices", indices.to_vec())?;
                            info.set_item("parameters", parameters.to_vec())?;
                            info.set_item("energy_ha", energy)?;
                            info.set_item("num_operators", indices.len())?;
                            if let Some(callback) = callback.as_ref() {
                                callback.bind(py).call1((info,))?;
                            }
                            Ok(())
                        };
                        invoke().map_err(|error| {
                            callback_error = Some(error);
                            mandacaru_core::Error::Kernel("growth observer failed".into())
                        })
                    })
                };
                if let Some(executor) = executor.as_ref() {
                    use mandacaru_core::quantum::execution::{CircuitExecution, Request};
                    let mut invoke = |requests: &[Request]| {
                        call_executor(&self.inner.context, executor, requests).map_err(|error| {
                            execution_error = Some(error);
                            mandacaru_core::Error::Kernel("circuit execution failed".into())
                        })
                    };
                    let mut execution = CircuitExecution::new(&mut invoke).with_shots(shots);
                    let solved = self.inner.solve_executed(
                        &options,
                        initial_parameters.as_deref(),
                        if callback.is_some() {
                            Some(&mut observer)
                        } else {
                            None
                        },
                        &mut execution,
                    );
                    if let Ok(result) = &solved {
                        energy_error_reason = execution.energy_error_reason().map(str::to_owned);
                        energy_errors = Some((
                            execution.energy_error(result.energy),
                            result
                                .iterations
                                .iter()
                                .map(|it| execution.energy_error(it.energy))
                                .collect::<Vec<_>>(),
                        ));
                    }
                    return solved;
                }
                if callback.is_some() {
                    return if let Some(resume) = resume {
                        resume
                            .inner
                            .resume_observed(&self.inner, &options, Some(&mut observer))
                    } else {
                        self.inner.solve_observed(
                            &options,
                            initial_parameters.as_deref(),
                            initial_indices.as_deref(),
                            &mut observer,
                        )
                    };
                }
                if let Some(resume) = resume {
                    resume.inner.resume(&self.inner, &options)
                } else {
                    self.inner.solve_sequence(
                        &options,
                        initial_parameters.as_deref(),
                        initial_indices.as_deref(),
                    )
                }
            })
            .map_err(|e| {
                callback_error
                    .or(execution_error)
                    .unwrap_or_else(|| error(e))
            })?;
        let out = solution_dict(py, result)?;
        // A sampled run's standard errors (Hartree): of the final energy and of each iteration's.
        if let Some((energy, iterations)) = energy_errors {
            out.set_item("energy_error_ha", energy)?;
            out.set_item("iteration_energy_errors_ha", iterations)?;
            out.set_item("energy_error_reason", energy_error_reason)?;
        }
        Ok(out)
    }
}

fn solution_dict<'py>(
    py: Python<'py>,
    result: mandacaru_core::quantum::SolveResult,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("energy_ha", result.energy)?;
    out.set_item("reference_ha", result.reference)?;
    out.set_item(
        "parameters",
        array(py, &[result.parameters.len()], result.parameters)?,
    )?;
    // A register-free (shot-based) result carries no state vector, so the key is absent.
    let length = result.state.len();
    if length > 0 {
        out.set_item("state", state_array(py, &[length], result.state)?)?;
    }
    out.set_item("success", result.success)?;
    out.set_item("message", result.message)?;
    out.set_item("budget_stopped", result.budget_stopped)?;
    out.set_item("pruned", result.pruned.clone())?;
    out.set_item("operators", result.labels)?;
    let operators = result.indices.len();
    out.set_item("indices", result.indices)?;
    // The final ansatz's Hessian approximation, `(n, n)`, for the next growth, resume or handover.
    out.set_item(
        "hessian",
        match result.hessian {
            Some(h) => Some(array(py, &[operators, operators], h)?),
            None => None,
        },
    )?;
    out.set_item("notes", result.notes)?;
    out.set_item("history_ha", result.history)?;
    out.set_item("optimizer_steps", result.optimizer_steps)?;
    out.set_item("final_max_gradient", result.final_gradient)?;
    out.set_item("elapsed_seconds", result.elapsed_seconds)?;
    if let Some(seed) = result.seed {
        let details = PyDict::new(py);
        details.set_item("kept", seed.kept)?;
        details.set_item("energy_ha", seed.energy)?;
        details.set_item("success", seed.success)?;
        details.set_item("message", seed.message)?;
        details.set_item("num_evaluations", seed.evaluations)?;
        details.set_item("optimizer_steps", seed.steps)?;
        details.set_item("warm_hessian", seed.warm)?;
        out.set_item("seed_relaxation", details)?;
    }
    let iterations: Vec<_> = result
        .iterations
        .into_iter()
        .map(|it| {
            (
                it.pool_index,
                it.energy,
                it.max_gradient,
                it.optimizer_steps,
                it.num_evaluations,
                it.success,
                it.message,
                it.num_parameters,
            )
        })
        .collect();
    out.set_item("iterations", iterations)?;
    Ok(out)
}

pub fn operator(
    context: &Context,
    n: usize,
    terms: Vec<(String, f64, f64)>,
) -> PyResult<PyOperator> {
    let terms: Vec<_> = terms
        .into_iter()
        .map(|(s, re, im)| (s, C64::new(re, im)))
        .collect();
    Ok(PyOperator {
        context: context.clone(),
        inner: Operator::new(context, n, &terms).map_err(error)?,
    })
}
pub type FermionTerms = Vec<(Vec<(usize, bool)>, f64, f64)>;
/// A fermionic operator mapped to qubits. `particles` fixes the spin-blocked sector (required by
/// `parity_reduced`); without it any mode count is taken and the reduced mapping is refused.
pub fn fermion(
    context: &Context,
    modes: usize,
    particles: Option<[usize; 2]>,
    mapping: &str,
    terms: FermionTerms,
) -> PyResult<PyOperator> {
    let encoding = match particles {
        Some(particles) => Encoding::new(mapping, modes, particles),
        None => Encoding::plain(mapping, modes),
    }
    .map_err(error)?;
    let terms: Vec<_> = terms
        .into_iter()
        .map(|(ladders, re, im)| (ladders, C64::new(re, im)))
        .collect();
    Ok(PyOperator {
        context: context.clone(),
        inner: encoding.map_fermion(context, &terms).map_err(error)?,
    })
}
/// A real or complex array as complex values, with its shape; the reservation keeps the copy
/// charged to the budget.
fn complex_values(
    context: &Context,
    values: &Bound<'_, PyAny>,
) -> PyResult<(Vec<usize>, Vec<C64>, Reservation)> {
    if let Ok(array) = values.extract::<PyReadonlyArrayDyn<'_, C64>>() {
        let (data, reservation) = copy(context, &array)?;
        return Ok((array.shape().to_vec(), data, reservation));
    }
    if let Ok(array) = values.extract::<PyReadonlyArrayDyn<'_, f64>>() {
        let (data, reservation) = copy(context, &array)?;
        let data = data.into_iter().map(|v| C64::new(v, 0.)).collect();
        return Ok((array.shape().to_vec(), data, reservation));
    }
    Err(PyTypeError::new_err(
        "integrals must be float64 or complex128 arrays",
    ))
}
fn fermion_term_list(terms: Vec<mandacaru_core::quantum::fermion_terms::FermionTerm>) -> FermionTerms {
    terms
        .into_iter()
        .map(|(ladders, c)| (ladders, c.re, c.im))
        .collect()
}
/// The terms of `H = sum h_pq a+_p a_q + 1/2 sum <pq|rs> a+_p a+_q a_s a_r` from spin-orbital
/// integrals, entries at or below `tol` skipped.
pub fn fermion_integral_terms(
    py: Python<'_>,
    context: &Context,
    one: &Bound<'_, PyAny>,
    two: Option<&Bound<'_, PyAny>>,
    tol: f64,
) -> PyResult<FermionTerms> {
    let (shape, one, _one) = complex_values(context, one)?;
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err(format!(
            "one-body integrals must be square, got {shape:?}"
        )));
    }
    let n = shape[0];
    let two = two.map(|g| complex_values(context, g)).transpose()?;
    if let Some((shape, _, _)) = &two {
        if shape != &[n; 4] {
            return Err(PyValueError::new_err(format!(
                "two-body integrals must be ({n}, {n}, {n}, {n}), got {shape:?}"
            )));
        }
    }
    let terms = py
        .detach(|| {
            mandacaru_core::quantum::fermion_terms::from_spin_orbital(
                context,
                n,
                &one,
                two.as_ref().map(|(_, g, _)| g.as_slice()),
                tol,
            )
        })
        .map_err(error)?;
    Ok(fermion_term_list(terms))
}
/// The terms of spatial integrals over `M` orbitals in the spin-orbital order `P = p + sigma M`,
/// the `(2M)^4` tensor never formed; `(2, M, M)` and `(2, 2, M, M, M, M)` are spin-resolved.
pub fn fermion_spatial_terms(
    py: Python<'_>,
    context: &Context,
    one: &Bound<'_, PyAny>,
    two: Option<&Bound<'_, PyAny>>,
    spin: Option<&Bound<'_, PyAny>>,
    tol: f64,
) -> PyResult<(usize, FermionTerms)> {
    let (shape, one, _one) = complex_values(context, one)?;
    let resolved = shape.len() == 3;
    if resolved && shape[0] != 2 {
        return Err(PyValueError::new_err(format!(
            "spin-resolved one-body integrals are (2, M, M), got {shape:?}"
        )));
    }
    if !(shape.len() == 2 || resolved) || shape[shape.len() - 1] != shape[shape.len() - 2] {
        return Err(PyValueError::new_err(format!(
            "one-body integrals are (M, M) or (2, M, M), got {shape:?}"
        )));
    }
    let m = shape[shape.len() - 1];
    let n = 2 * m;
    let spin = spin.map(|h| complex_values(context, h)).transpose()?;
    if let Some((shape, _, _)) = &spin {
        if shape != &[n, n] {
            return Err(PyValueError::new_err(format!(
                "h_spin must be ({n}, {n}) for {m} spatial orbitals, got {shape:?}"
            )));
        }
    }
    let two = two.map(|g| complex_values(context, g)).transpose()?;
    if let Some((shape, _, _)) = &two {
        let expected: Vec<usize> = if resolved {
            vec![2, 2, m, m, m, m]
        } else {
            vec![m; 4]
        };
        if shape != &expected {
            return Err(PyValueError::new_err(if resolved {
                format!(
                    "spin-resolved two-body integrals are (2, 2, {m}, {m}, {m}, {m}), got {shape:?}"
                )
            } else {
                format!("two-body integrals are ({m}, {m}, {m}, {m}), got {shape:?}")
            }));
        }
    }
    let terms = py
        .detach(|| {
            mandacaru_core::quantum::fermion_terms::from_spatial(
                context,
                m,
                resolved,
                &one,
                spin.as_ref().map(|(_, h, _)| h.as_slice()),
                two.as_ref().map(|(_, g, _)| g.as_slice()),
                tol,
            )
        })
        .map_err(error)?;
    Ok((n, fermion_term_list(terms)))
}

impl PyProblem {
    /// The state-vector register, refusing by name when the problem is register-free.
    fn register(&self, what: &str) -> PyResult<&Basis> {
        self.inner.register(what).map_err(error)
    }
    fn expand_rows<A: Amplitude>(
        &self,
        values: &[A],
        rows: usize,
        n: usize,
    ) -> mandacaru_core::Result<State> {
        let ctx = &self.inner.context;
        let basis = self.inner.register("expand_state")?;
        let dimension = basis.len();
        let mut out = ctx.allocate::<A>(mandacaru_core::product(&[rows, n])?)?;
        for row in 0..rows {
            let expanded =
                basis.expand_state(ctx, &values[row * dimension..(row + 1) * dimension])?;
            out.as_mut_slice()[row * n..(row + 1) * n].copy_from_slice(expanded.as_slice());
        }
        Ok(A::wrap(out))
    }
}
/// The Pauli strings of the spatial RDMs and their assembly (`quantum::rdm_measurement`).
#[pyclass(name = "RdmMeasurement", module = "mandacaru._native", frozen)]
pub struct PyRdmMeasurement {
    context: mandacaru_core::Context,
    inner: mandacaru_core::quantum::rdm_measurement::RdmMeasurement,
}
#[pymethods]
impl PyRdmMeasurement {
    #[getter]
    fn n_spatial_orbitals(&self) -> usize {
        self.inner.m
    }
    #[getter]
    fn observables(&self) -> Vec<String> {
        self.inner.observables()
    }
    #[getter]
    fn bases(&self) -> usize {
        self.inner.bases()
    }
    /// The number of values `assemble` takes (the strings, then the covariance products).
    #[getter]
    fn measured(&self) -> usize {
        self.inner.measured()
    }
    /// One `(rotations, observables)` per measurement basis (`RdmMeasurement::requests`).
    #[allow(clippy::type_complexity)]
    fn requests(&self, py: Python<'_>) -> Vec<(Vec<(String, f64)>, Vec<(String, u32)>)> {
        py.detach(|| self.inner.requests())
    }
    /// `(one, two, one_error, two_error)`: real `(m, m)` and `(m, m, m, m)` arrays, the errors
    /// `None` without `errors`.
    #[pyo3(signature=(values, errors=None, shots=None))]
    #[allow(clippy::type_complexity)]
    fn assemble<'py>(
        &self,
        py: Python<'py>,
        values: Vec<f64>,
        errors: Option<Vec<f64>>,
        shots: Option<u64>,
    ) -> PyResult<(
        Array<'py, f64>,
        Array<'py, f64>,
        Option<Array<'py, f64>>,
        Option<Array<'py, f64>>,
    )> {
        let (rdms, spread) = py
            .detach(|| self.inner.assemble(&self.context, &values, errors.as_deref(), shots))
            .map_err(error)?;
        let m = rdms.m;
        let real = |v: &[mandacaru_core::quantum::C64]| v.iter().map(|c| c.re).collect::<Vec<_>>();
        let (one_error, two_error) = match spread {
            Some((a, b)) => (Some(array(py, &[m, m], a)?), Some(array(py, &[m, m, m, m], b)?)),
            None => (None, None),
        };
        Ok((
            array(py, &[m, m], real(rdms.one.as_slice()))?,
            array(py, &[m, m, m, m], real(rdms.two.as_slice()))?,
            one_error,
            two_error,
        ))
    }
}
impl PyProblem {
    fn rdms_of<A: Amplitude>(
        &self,
        state: &[A],
        two_body: bool,
    ) -> mandacaru_core::Result<mandacaru_core::quantum::properties::Rdms> {
        self.inner.rdms(state, two_body)
    }
}

#[pyfunction]
pub fn periodic_quantum_forces_fd<'py>(
    py: Python<'py>,
    energies_plus: Vec<[f64; 3]>,
    energies_minus: Vec<[f64; 3]>,
    h: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let result = py
        .detach(|| {
            mandacaru_core::quantum::periodic_response::periodic_quantum_forces_fd(
                &energies_plus,
                &energies_minus,
                h,
            )
        })
        .map_err(error)?;
    let out = PyDict::new(py);
    let n = result.forces.len();
    out.set_item(
        "forces",
        array(py, &[n, 3], result.forces.into_iter().flatten().collect())?,
    )?;
    out.set_item("scheme", result.scheme)?;
    Ok(out)
}

#[pyfunction]
pub fn periodic_quantum_stress_fd<'py>(
    py: Python<'py>,
    strained_plus: [f64; 6],
    strained_minus: [f64; 6],
    eps: f64,
    volume: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let result = py
        .detach(|| {
            mandacaru_core::quantum::periodic_response::periodic_quantum_stress_fd(
                &strained_plus,
                &strained_minus,
                eps,
                volume,
            )
        })
        .map_err(error)?;
    let out = PyDict::new(py);
    out.set_item(
        "sigma",
        array(py, &[3, 3], result.sigma.into_iter().flatten().collect())?,
    )?;
    out.set_item(
        "voigt",
        array(py, &[6], result.voigt.to_vec())?,
    )?;
    out.set_item("volume", result.volume)?;
    out.set_item("scheme", result.scheme)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (n_atoms, m, rdm1, rdm2=None, dh_atomic=Vec::new(), dg_atomic=None, ds_atomic=None, w_matrix=None, ewald_forces=Vec::new()))]
pub fn periodic_quantum_forces_analytic<'py>(
    py: Python<'py>,
    n_atoms: usize,
    m: usize,
    rdm1: PyReadonlyArrayDyn<'_, C64>,
    rdm2: Option<PyReadonlyArrayDyn<'_, C64>>,
    dh_atomic: Vec<Vec<[f64; 3]>>,
    dg_atomic: Option<Vec<Vec<[f64; 3]>>>,
    ds_atomic: Option<Vec<Vec<[f64; 3]>>>,
    w_matrix: Option<PyReadonlyArrayDyn<'_, C64>>,
    ewald_forces: Vec<[f64; 3]>,
) -> PyResult<Bound<'py, PyDict>> {
    let rdm1_slice = rdm1.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rdm2_slice = match &rdm2 {
        Some(a) => Some(a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?),
        None => None,
    };
    let w_slice = match &w_matrix {
        Some(w) => Some(w.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?),
        None => None,
    };
    let dg_ref = dg_atomic.as_deref();
    let ds_ref = ds_atomic.as_deref();

    let result = py
        .detach(|| {
            mandacaru_core::quantum::periodic_response::periodic_quantum_forces_analytic(
                n_atoms,
                m,
                rdm1_slice,
                rdm2_slice,
                &dh_atomic,
                dg_ref,
                ds_ref,
                w_slice,
                &ewald_forces,
            )
        })
        .map_err(error)?;
    let out = PyDict::new(py);
    let n = result.forces.len();
    out.set_item(
        "forces",
        array(py, &[n, 3], result.forces.into_iter().flatten().collect())?,
    )?;
    out.set_item("scheme", result.scheme)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (m, volume, rdm1, rdm2=None, electronic_energy=0.0, dh_strain=[[Vec::new(), Vec::new(), Vec::new()], [Vec::new(), Vec::new(), Vec::new()], [Vec::new(), Vec::new(), Vec::new()]], dg_strain=None, ds_strain=None, w_matrix=None, ewald_stress=[[0.0; 3]; 3]))]
pub fn periodic_quantum_stress_analytic<'py>(
    py: Python<'py>,
    m: usize,
    volume: f64,
    rdm1: PyReadonlyArrayDyn<'_, C64>,
    rdm2: Option<PyReadonlyArrayDyn<'_, C64>>,
    electronic_energy: f64,
    dh_strain: [[Vec<f64>; 3]; 3],
    dg_strain: Option<[[Vec<f64>; 3]; 3]>,
    ds_strain: Option<[[Vec<f64>; 3]; 3]>,
    w_matrix: Option<PyReadonlyArrayDyn<'_, C64>>,
    ewald_stress: [[f64; 3]; 3],
) -> PyResult<Bound<'py, PyDict>> {
    let rdm1_slice = rdm1.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rdm2_slice = match &rdm2 {
        Some(a) => Some(a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?),
        None => None,
    };
    let w_slice = match &w_matrix {
        Some(w) => Some(w.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?),
        None => None,
    };
    let dg_ref = dg_strain.as_ref();
    let ds_ref = ds_strain.as_ref();

    let result = py
        .detach(|| {
            mandacaru_core::quantum::periodic_response::periodic_quantum_stress_analytic(
                m,
                volume,
                rdm1_slice,
                rdm2_slice,
                electronic_energy,
                &dh_strain,
                dg_ref,
                ds_ref,
                w_slice,
                &ewald_stress,
            )
        })
        .map_err(error)?;
    let out = PyDict::new(py);
    out.set_item(
        "sigma",
        array(py, &[3, 3], result.sigma.into_iter().flatten().collect())?,
    )?;
    out.set_item(
        "voigt",
        array(py, &[6], result.voigt.to_vec())?,
    )?;
    out.set_item("volume", result.volume)?;
    out.set_item("scheme", result.scheme)?;
    Ok(out)
}

