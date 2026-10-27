use super::{array, copy, error, Array};
use mandacaru_core::{
    quantum::{optimizer, Basis, CachedProblem, Encoding, Operator, SolveOptions, C64},
    Context,
};
use numpy::PyReadonlyArray1;
use pyo3::{prelude::*, types::PyDict};

#[pyclass(name = "PauliOperator", module = "mandacaru._native", frozen)]
pub struct PyOperator {
    pub context: Context,
    pub inner: Operator,
}
#[pymethods]
impl PyOperator {
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

#[pyclass(name = "QuantumProblem", module = "mandacaru._native", frozen)]
pub struct PyProblem {
    pub inner: CachedProblem,
}
#[pymethods]
impl PyProblem {
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
        state: PyReadonlyArray1<'py, C64>,
        two_body: bool,
    ) -> PyResult<(Array<'py, C64>, Option<Array<'py, C64>>)> {
        let (state, _memory) = copy(&self.inner.context, &state)?;
        let rdms = py
            .detach(|| {
                mandacaru_core::quantum::properties::rdms(
                    &self.inner.context,
                    &self.inner.encoding,
                    &self.inner.basis,
                    &state,
                    two_body,
                )
            })
            .map_err(error)?;
        let n = self.inner.encoding.modes;
        Ok((
            array(py, &[n, n], rdms.one.as_slice().to_vec())?,
            rdms.two
                .map(|two| array(py, &[n, n, n, n], two.as_slice().to_vec()))
                .transpose()?,
        ))
    }
    #[pyo3(signature=(parameters,*,trotter=false,indices=None))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        trotter: bool,
        indices: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let indices = indices.unwrap_or_else(|| (0..self.inner.pool.len()).collect());
        let value = py
            .detach(|| self.inner.evaluate(&indices, &parameters, trotter))
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item("energy_ha", value.energy)?;
        out.set_item(
            "gradient",
            array(py, &[value.gradient.len()], value.gradient)?,
        )?;
        out.set_item(
            "state",
            array(
                py,
                &[value.state.as_slice().len()],
                value.state.as_slice().to_vec(),
            )?,
        )?;
        Ok(out)
    }
    #[pyo3(signature=(*,adapt=false,trotter=false,max_growth=50,gradient_tolerance=Some(1e-7),energy_tolerance=None,max_iterations=300,max_evaluations=5000,tolerance=1e-9,initial_parameters=None,optimizer="BFGS"))]
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
        tolerance: f64,
        initial_parameters: Option<Vec<f64>>,
        optimizer: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        if optimizer != "BFGS" && optimizer != "L-BFGS" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "optimizer must be BFGS or L-BFGS",
            ));
        }
        let options = SolveOptions {
            adapt,
            trotter,
            max_growth,
            gradient_tolerance,
            energy_tolerance,
            optimizer: optimizer::Options {
                limited_memory: optimizer == "L-BFGS",
                max_iterations,
                max_evaluations,
                tolerance,
            },
        };
        let result = py
            .detach(|| self.inner.solve(&options, initial_parameters.as_deref()))
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item("energy_ha", result.energy)?;
        out.set_item("reference_ha", result.reference)?;
        out.set_item(
            "parameters",
            array(py, &[result.parameters.len()], result.parameters)?,
        )?;
        out.set_item(
            "state",
            array(
                py,
                &[result.state.as_slice().len()],
                result.state.as_slice().to_vec(),
            )?,
        )?;
        out.set_item("success", result.success)?;
        out.set_item("message", result.message)?;
        out.set_item("operators", result.labels)?;
        out.set_item("indices", result.indices)?;
        out.set_item("history_ha", result.history)?;
        out.set_item("optimizer_steps", result.optimizer_steps)?;
        out.set_item("final_max_gradient", result.final_gradient)?;
        out.set_item("elapsed_seconds", result.elapsed_seconds)?;
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
                )
            })
            .collect();
        out.set_item("iterations", iterations)?;
        Ok(out)
    }
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
