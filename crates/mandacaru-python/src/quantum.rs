use super::{array, copy, error, Array};
use mandacaru_core::{
    math::storage::Precision,
    quantum::{
        optimizer, Amplitude, Basis, CachedProblem, Encoding, Operator, SolveOptions, State, C64,
    },
    Context, Reservation,
};
use numpy::{Complex32, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};

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
    #[pyo3(signature=(*,max_steps=200,max_length=20,move_weights=[1.0;4],temperatures=[0.003674932217565499,0.00003674932217565499],length_penalty=0.0,proposal_temperature=0.2,proposal="gradient",warm_start=true,replace_start="zero",screen_insertions=0,record=false,seed=0,optimizer="BFGS",max_iterations=300,max_evaluations=5000,tolerance=Some(1e-9),track_best=true,initial_indices=None,initial_parameters=None,proposal_model=None,transfer_steps=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        max_steps: usize,
        max_length: usize,
        move_weights: [f64; 4],
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
        let inner = py
            .detach(|| {
                let encoding = Encoding::new(
                    &self.inner.encoding.mapping,
                    self.inner.encoding.modes,
                    self.inner.encoding.particles,
                )?;
                let h = self.inner.hamiltonian.copy_into(&self.inner.context)?;
                let mut problem = CachedProblem::new(
                    &self.inner.context,
                    encoding,
                    h,
                    false,
                    self.inner.dense.is_some(),
                    true,
                )?;
                problem.precision = self.inner.precision;
                problem.use_hva(layers, evolution, order, steps, seed_angle)?;
                if self.inner.sparse.is_some() {
                    problem.sparse = Some(mandacaru_core::quantum::sparse::SparseMatrix::new(
                        &problem.context,
                        &problem.basis,
                        &problem.hamiltonian,
                    )?);
                }
                Ok::<_, mandacaru_core::Error>(problem)
            })
            .map_err(error)?;
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
            },
            ..Default::default()
        };
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
            state_array(py, &[num_states, self.inner.basis.len()], result.states)?,
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
        let dimension = self.inner.basis.len();
        if !(1..=2).contains(&shape.len()) || shape[shape.len() - 1] != dimension {
            return Err(PyValueError::new_err(
                "states need shape (sector_dimension,) or (states,sector_dimension)",
            ));
        }
        let rows = if shape.len() == 1 { 1 } else { shape[0] };
        let n = 1usize
            .checked_shl(self.inner.basis.n as u32)
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
                Ok::<_, mandacaru_core::Error>(match &state {
                    State::Float64(v) => {
                        State::Float64(self.inner.basis.expand_state(ctx, v.as_slice())?)
                    }
                    State::Float32(v) => {
                        State::Float32(self.inner.basis.expand_state(ctx, v.as_slice())?)
                    }
                })
            })
            .map_err(error)?;
        let length = out.len();
        state_array(py, &[length], out)
    }
    #[pyo3(signature=(indices,parameters,*,product_form=true,energy_ha,status_json="{}",method="adapt-vqe"))]
    fn checkpoint(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
        parameters: Vec<f64>,
        product_form: bool,
        energy_ha: f64,
        status_json: &str,
        method: &str,
    ) -> PyResult<super::persistence::PySnapshot> {
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
    #[pyo3(signature=(weights,*,adapt=false,trotter=false,optimizer="BFGS",max_iterations=300,max_evaluations=5000,tolerance=Some(1e-9),track_best=true,max_growth=50,gradient_tolerance=Some(1e-7),energy_tolerance=None,initial_parameters=None))]
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
    ) -> PyResult<Bound<'py, PyDict>> {
        let method = optimizer::Method::parse(optimizer).map_err(error)?;
        let options = SolveOptions {
            adapt,
            trotter,
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
            },
        };
        let result = py
            .detach(|| {
                self.inner
                    .solve_subspace(&options, &weights, initial_parameters.as_deref())
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
            state_array(py, &[weights.len(), self.inner.basis.len()], result.states)?,
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
        out.set_item("indices", result.indices)?;
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
        self.inner.encoding.n_qubits()
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
        let n = self.inner.encoding.n_qubits();
        let reference = self.inner.encoding.reference();
        (0..n)
            .filter(|&q| reference & (1u64 << (n - 1 - q)) != 0)
            .collect()
    }
    #[getter]
    fn mapping(&self) -> String {
        self.inner.encoding.mapping.clone()
    }
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.basis.len()
    }
    #[getter]
    fn pool_labels(&self) -> Vec<String> {
        self.inner.pool.iter().map(|g| g.label.clone()).collect()
    }
    #[getter]
    fn pool_screenable(&self) -> Vec<bool> {
        self.inner.pool_screenable.clone()
    }
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
        array(
            py,
            &[self.inner.basis.len()],
            self.inner.basis.indices.as_slice().to_vec(),
        )
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
    #[pyo3(signature=(*,adapt=false,trotter=false,max_growth=50,gradient_tolerance=Some(1e-7),energy_tolerance=None,max_iterations=300,max_evaluations=5000,tolerance=Some(1e-9),track_best=true,initial_parameters=None,initial_indices=None,optimizer="BFGS",resume=None,callback=None,executor=None))]
    #[allow(clippy::too_many_arguments)]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        adapt: bool,
        trotter: bool,
        max_growth: usize,
        gradient_tolerance: Option<f64>,
        energy_tolerance: Option<f64>,
        max_iterations: usize,
        max_evaluations: usize,
        tolerance: Option<f64>,
        track_best: bool,
        initial_parameters: Option<Vec<f64>>,
        initial_indices: Option<Vec<usize>>,
        optimizer: &str,
        resume: Option<&super::persistence::PySnapshot>,
        callback: Option<Py<PyAny>>,
        executor: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let method = optimizer::Method::parse(optimizer).map_err(error)?;
        let options = SolveOptions {
            adapt,
            trotter,
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
            },
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
                    use mandacaru_core::quantum::execution::{CircuitExecution, Response};
                    let mut invoke = |labels: &[String], angles: &[f64], observables: &[String]| {
                        Python::attach(|py| {
                            let invoke = || -> PyResult<Response> {
                                let response = executor.bind(py).call1((
                                    labels.to_vec(),
                                    angles.to_vec(),
                                    observables.to_vec(),
                                ))?;
                                if let Ok(stop) = response.get_item("stop") {
                                    return Ok(Response::Stop(stop.extract()?));
                                }
                                if let Ok(state) = response.get_item("state") {
                                    let state: PyReadonlyArray1<'_, C64> = state.extract()?;
                                    let mut owned =
                                        self.inner.context.allocate(state.len()).map_err(error)?;
                                    for (out, value) in
                                        owned.as_mut_slice().iter_mut().zip(state.as_array())
                                    {
                                        *out = *value;
                                    }
                                    Ok(Response::State(owned))
                                } else {
                                    Ok(Response::Expectations(
                                        response.get_item("expectations")?.extract()?,
                                    ))
                                }
                            };
                            invoke().map_err(|error| {
                                execution_error = Some(error);
                                mandacaru_core::Error::Kernel("circuit execution failed".into())
                            })
                        })
                    };
                    let mut execution = CircuitExecution::new(&mut invoke);
                    return self.inner.solve_executed(
                        &options,
                        initial_parameters.as_deref(),
                        if callback.is_some() {
                            Some(&mut observer)
                        } else {
                            None
                        },
                        &mut execution,
                    );
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
        solution_dict(py, result)
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
    let length = result.state.len();
    out.set_item("state", state_array(py, &[length], result.state)?)?;
    out.set_item("success", result.success)?;
    out.set_item("message", result.message)?;
    out.set_item("budget_stopped", result.budget_stopped)?;
    out.set_item("operators", result.labels)?;
    out.set_item("indices", result.indices)?;
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
pub fn fermion(
    context: &Context,
    modes: usize,
    particles: [usize; 2],
    mapping: &str,
    terms: FermionTerms,
) -> PyResult<PyOperator> {
    let encoding = Encoding::new(mapping, modes, particles).map_err(error)?;
    let mut inner = Operator::new(context, encoding.n_qubits(), &[]).map_err(error)?;
    for (ladders, re, im) in terms {
        let term = encoding.monomial(context, &ladders).map_err(error)?;
        inner = inner.add(context, &term, C64::new(re, im)).map_err(error)?;
    }
    Ok(PyOperator {
        context: context.clone(),
        inner,
    })
}

impl PyProblem {
    fn expand_rows<A: Amplitude>(
        &self,
        values: &[A],
        rows: usize,
        n: usize,
    ) -> mandacaru_core::Result<State> {
        let ctx = &self.inner.context;
        let dimension = self.inner.basis.len();
        let mut out = ctx.allocate::<A>(mandacaru_core::product(&[rows, n])?)?;
        for row in 0..rows {
            let expanded = self
                .inner
                .basis
                .expand_state(ctx, &values[row * dimension..(row + 1) * dimension])?;
            out.as_mut_slice()[row * n..(row + 1) * n].copy_from_slice(expanded.as_slice());
        }
        Ok(A::wrap(out))
    }
}
impl PyProblem {
    fn rdms_of<A: Amplitude>(
        &self,
        state: &[A],
        two_body: bool,
    ) -> mandacaru_core::Result<mandacaru_core::quantum::properties::Rdms> {
        let compressed = if state.len() != self.inner.basis.len() {
            Some(
                self.inner
                    .basis
                    .compress_state(&self.inner.context, state)?,
            )
        } else {
            None
        };
        mandacaru_core::quantum::properties::rdms(
            &self.inner.context,
            &self.inner.encoding,
            &self.inner.basis,
            compressed.as_ref().map_or(state, |v| v.as_slice()),
            two_body,
        )
    }
}
