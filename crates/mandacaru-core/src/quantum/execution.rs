//! Circuit execution with native Pauli algebra and exact parameter-shift derivatives.
use super::{
    algebra::Term,
    evolution,
    pool_measurement::{self, PauliSet},
    solver::{register_free_refusal, Execution},
    Basis, CachedProblem, Operator, C64,
};
use crate::{Buffer, Error, Result};

/// One circuit of a batch: the Pauli rotations that prepare the state to measure, and the Pauli
/// strings the executor reads off it.
pub struct Request {
    pub labels: Vec<String>,
    pub angles: Vec<f64>,
    pub observables: Vec<String>,
}
/// The executor's answer to a batch of [`Request`]s, one entry per request in the batch's order.
pub enum Response {
    /// The state vector each circuit prepares (an exact simulator).
    States(Vec<Buffer<C64>>),
    /// The expectation of each request's `observables`, in their order (a device).
    Expectations(Vec<Vec<f64>>),
    /// The executor refuses further circuits (its run budget is exhausted).
    Stop(String),
}
/// The executor: one call per measurement, however many circuits the measurement needs, so the
/// energy of a factorized Hamiltonian (one circuit per basis) is one submission.
pub type CircuitCall<'a> = dyn FnMut(&[Request]) -> Result<Response> + 'a;
pub struct CircuitExecution<'a> {
    call: &'a mut CircuitCall<'a>,
    stopped: Option<String>,
    batch_bytes: Option<usize>,
}
/// The answers to a batch, with the register they live on when they are state vectors.
struct Readings {
    readings: Vec<Reading>,
    basis: Option<Basis>,
}
enum Reading {
    State(Buffer<C64>),
    /// Expectations, in the order of the request's observables.
    Values(Vec<f64>),
}
struct Circuit {
    labels: Vec<String>,
    angles: Vec<f64>,
    links: Vec<(usize, f64)>,
}
fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
impl Circuit {
    /// The request that measures `observables` after this circuit and then `tail`, a basis
    /// rotation.
    fn request(&self, tail: &[(String, f64)], observables: Vec<String>) -> Request {
        let mut labels = self.labels.clone();
        let mut angles = self.angles.clone();
        for (label, angle) in tail {
            labels.push(label.clone());
            angles.push(*angle);
        }
        Request {
            labels,
            angles,
            observables,
        }
    }
}
impl<'a> CircuitExecution<'a> {
    pub fn new(call: &'a mut CircuitCall<'a>) -> Self {
        Self {
            call,
            stopped: None,
            batch_bytes: None,
        }
    }
    /// Form the pool commutators of a gradient measurement in batches of about `bytes` accounted
    /// bytes, instead of the context's default ([`pool_measurement::batch_bytes`]).
    pub fn with_batch_bytes(mut self, bytes: usize) -> Self {
        self.batch_bytes = Some(bytes);
        self
    }
}
impl CircuitExecution<'_> {
    fn circuit(problem: &CachedProblem, indices: &[usize], parameters: &[f64]) -> Result<Circuit> {
        if problem.hva.as_ref().is_some_and(|h| !h.trotter) {
            return Err(invalid(
                "exact Hamiltonian-group evolution cannot be serialized as Pauli rotations",
            ));
        }
        let expanded = problem.sequence(indices, parameters)?;
        let mut circuit = Circuit {
            labels: vec![],
            angles: vec![],
            links: vec![],
        };
        for ((operator, theta), (index, scale)) in expanded
            .operators
            .iter()
            .zip(expanded.parameters)
            .zip(expanded.links)
        {
            if !operator.is_antihermitian() {
                return Err(invalid("circuit generators must be anti-Hermitian"));
            }
            for first in &operator.terms {
                for second in &operator.terms {
                    if ((first.x & second.z).count_ones() + (first.z & second.x).count_ones()) % 2
                        != 0
                    {
                        return Err(invalid("a circuit generator has noncommuting terms; supply an explicit product formula"));
                    }
                }
            }
            for (label, coefficient) in operator.labels() {
                if label.chars().all(|c| c == 'I') {
                    continue;
                }
                circuit.labels.push(label);
                circuit.angles.push(theta * coefficient.im);
                circuit.links.push((index, scale * coefficient.im));
            }
        }
        Ok(circuit)
    }
    /// Send `requests` to the executor in one call and check its answer. After a stop the
    /// executor is never called again.
    fn call(&mut self, problem: &CachedProblem, requests: Vec<Request>) -> Result<Readings> {
        if let Some(message) = &self.stopped {
            return Err(Error::Interrupted(message.clone()));
        }
        match (self.call)(&requests)? {
            Response::Stop(message) => {
                self.stopped = Some(message.clone());
                Err(Error::Interrupted(message))
            }
            Response::States(states) => {
                if problem.is_register_free() {
                    return Err(register_free_refusal("a state-vector response"));
                }
                let basis = Basis::register(&problem.context, problem.hamiltonian.n)?;
                if states.len() != requests.len()
                    || states.iter().any(|state| {
                        let state = state.as_slice();
                        state.len() != basis.len()
                            || state.iter().any(|c| !c.re.is_finite() || !c.im.is_finite())
                            || (evolution::inner(state, state).re - 1.).abs() > 1e-7
                    })
                {
                    return Err(invalid("circuit executor returned an invalid state vector"));
                }
                Ok(Readings {
                    readings: states.into_iter().map(Reading::State).collect(),
                    basis: Some(basis),
                })
            }
            Response::Expectations(values) => {
                if values.len() != requests.len()
                    || values.iter().zip(&requests).any(|(values, request)| {
                        values.len() != request.observables.len()
                            || values.iter().any(|v| !v.is_finite())
                    })
                {
                    return Err(invalid(
                        "circuit executor returned invalid Pauli expectations",
                    ));
                }
                Ok(Readings {
                    readings: values.into_iter().map(Reading::Values).collect(),
                    basis: None,
                })
            }
        }
    }
}
impl Readings {
    /// `<operator>` in the state the `request`-th circuit prepared. A state is applied; Pauli
    /// expectations are summed with the operator's coefficients, `position` finding a term's
    /// string among the request's observables.
    fn read(
        &self,
        problem: &CachedProblem,
        request: usize,
        operator: &Operator,
        position: &dyn Fn(&Term) -> Option<usize>,
    ) -> Result<f64> {
        match (&self.readings[request], &self.basis) {
            (Reading::State(state), Some(basis)) => {
                let state = state.as_slice();
                let action = operator.apply(&problem.context, basis, state)?;
                Ok(evolution::inner(state, action.as_slice()).re)
            }
            (Reading::Values(values), _) => {
                if !operator.is_hermitian() {
                    return Err(invalid("measured observables must be Hermitian"));
                }
                operator
                    .terms
                    .iter()
                    .map(|t| {
                        if (t.x, t.z) == (0, 0) {
                            return Ok(t.coefficient.re);
                        }
                        let k = position(t).ok_or_else(|| {
                            invalid("an observable was not among the requested strings")
                        })?;
                        Ok(t.coefficient.re * values[k])
                    })
                    .sum()
            }
            (Reading::State(_), None) => Err(invalid("a state reading has no register")),
        }
    }
}
impl CircuitExecution<'_> {
    /// `<H>` of the state `circuit` prepares: the Hamiltonian's Pauli expectations, or, when the
    /// problem carries a factorized measurement plan, the sum of its constant and one diagonal
    /// observable per basis, each measured after that basis's Givens network. The objective and
    /// every parameter-shift evaluation go through here, so energy and gradient always use the
    /// same measurement.
    fn energy(&mut self, problem: &CachedProblem, circuit: &Circuit) -> Result<f64> {
        let no_rotation: &[(String, f64)] = &[];
        let bases: Vec<(&[(String, f64)], &Operator)> = match &problem.factorized {
            None => vec![(no_rotation, &problem.hamiltonian)],
            Some(plan) => plan
                .bases
                .iter()
                .map(|basis| (basis.rotation.as_slice(), &basis.observable))
                .collect(),
        };
        let mut sets = Vec::with_capacity(bases.len());
        let mut requests = Vec::with_capacity(bases.len());
        for (rotation, observable) in &bases {
            let mut set = PauliSet::new(&problem.context, observable.n)?;
            set.insert(&problem.context, [*observable])?;
            requests.push(circuit.request(rotation, set.labels()));
            sets.push(set);
        }
        let readings = self.call(problem, requests)?;
        let values = bases
            .iter()
            .zip(&sets)
            .enumerate()
            .map(|(k, ((_, observable), set))| {
                readings.read(problem, k, observable, &|t| set.position(t))
            })
            .collect::<Result<Vec<_>>>()?;
        match &problem.factorized {
            None => Ok(values[0]),
            Some(plan) => plan.energy(&values),
        }
    }
}
impl Execution for CircuitExecution<'_> {
    fn evaluate(
        &mut self,
        problem: &CachedProblem,
        indices: &[usize],
        parameters: &[f64],
        gradients: bool,
    ) -> Result<evolution::Evaluation> {
        let mut circuit = Self::circuit(problem, indices, parameters)?;
        let energy = self.energy(problem, &circuit)?;
        let mut gradient = vec![0.; parameters.len()];
        if gradients {
            for k in 0..circuit.angles.len() {
                let original = circuit.angles[k];
                circuit.angles[k] = original + std::f64::consts::FRAC_PI_4;
                let plus = self.energy(problem, &circuit)?;
                circuit.angles[k] = original - std::f64::consts::FRAC_PI_4;
                let minus = self.energy(problem, &circuit)?;
                circuit.angles[k] = original;
                let (index, scale) = circuit.links[k];
                gradient[index] += scale * (plus - minus);
            }
        }
        Ok(evolution::Evaluation {
            energy,
            gradient,
            state: problem.result_state(indices, parameters, true)?,
        })
    }
    /// `<[H, A_k]>` for every pool generator, measured on the circuit in one call. The
    /// commutators are formed in bounded batches, twice: first for their distinct Pauli strings
    /// (what the device is asked), then again to sum each one's coefficients against the
    /// answers, so the pool's commutators are never all alive at once.
    fn pool_gradients(
        &mut self,
        problem: &CachedProblem,
        indices: &[usize],
        parameters: &[f64],
    ) -> Result<Vec<f64>> {
        let circuit = Self::circuit(problem, indices, parameters)?;
        let bound = self
            .batch_bytes
            .unwrap_or_else(|| pool_measurement::batch_bytes(&problem.context));
        let paulis = problem.pool_paulis(bound)?;
        let readings = {
            let _request = problem.context.reserve(paulis.set.request_bytes())?;
            self.call(problem, vec![circuit.request(&[], paulis.set.labels())])?
        };
        let mut gradients = Vec::with_capacity(problem.pool.len());
        problem.for_each_pool_commutator_batch(bound, |batch| {
            for operator in batch {
                gradients.push(readings.read(problem, 0, operator, &|t| paulis.set.position(t))?);
            }
            Ok(())
        })?;
        Ok(gradients)
    }
}
