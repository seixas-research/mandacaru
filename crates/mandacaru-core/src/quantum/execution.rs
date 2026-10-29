//! Circuit execution with native Pauli algebra and exact parameter-shift derivatives.
use super::{evolution, solver::Execution, Basis, CachedProblem, Operator, C64};
use crate::{Buffer, Error, Result};
use std::collections::{BTreeMap, BTreeSet};

pub enum Response {
    State(Buffer<C64>),
    Expectations(Vec<f64>),
}
pub type CircuitCall<'a> = dyn FnMut(&[String], &[f64], &[String]) -> Result<Response> + 'a;
pub struct CircuitExecution<'a> {
    pub call: &'a mut CircuitCall<'a>,
}
struct Circuit {
    labels: Vec<String>,
    angles: Vec<f64>,
    links: Vec<(usize, f64)>,
}
fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
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
    fn measure(
        &mut self,
        problem: &CachedProblem,
        circuit: &Circuit,
        observables: &[&Operator],
    ) -> Result<Vec<f64>> {
        let labels: Vec<_> = observables
            .iter()
            .flat_map(|o| o.labels())
            .map(|(l, _)| l)
            .filter(|l| !l.chars().all(|c| c == 'I'))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let response = (self.call)(&circuit.labels, &circuit.angles, &labels)?;
        match response {
            Response::State(state) => {
                let state = state.as_slice();
                let basis = Basis::register(&problem.context, problem.hamiltonian.n)?;
                if state.len() != basis.len()
                    || state.iter().any(|c| !c.re.is_finite() || !c.im.is_finite())
                    || (evolution::inner(state, state).re - 1.).abs() > 1e-7
                {
                    return Err(invalid("circuit executor returned an invalid state vector"));
                }
                observables
                    .iter()
                    .map(|operator| {
                        let action = operator.apply(&problem.context, &basis, state)?;
                        Ok(evolution::inner(state, action.as_slice()).re)
                    })
                    .collect()
            }
            Response::Expectations(values) => {
                if values.len() != labels.len() || values.iter().any(|v| !v.is_finite()) {
                    return Err(invalid(
                        "circuit executor returned invalid Pauli expectations",
                    ));
                }
                let values: BTreeMap<_, _> = labels.into_iter().zip(values).collect();
                observables
                    .iter()
                    .map(|operator| {
                        if !operator.is_hermitian() {
                            return Err(invalid("measured observables must be Hermitian"));
                        }
                        Ok(operator
                            .labels()
                            .iter()
                            .map(|(label, c)| {
                                c.re * if label.chars().all(|c| c == 'I') {
                                    1.
                                } else {
                                    values[label]
                                }
                            })
                            .sum())
                    })
                    .collect()
            }
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
        let energy = self.measure(problem, &circuit, &[&problem.hamiltonian])?[0];
        let mut gradient = vec![0.; parameters.len()];
        if gradients {
            for k in 0..circuit.angles.len() {
                let original = circuit.angles[k];
                circuit.angles[k] = original + std::f64::consts::FRAC_PI_4;
                let plus = self.measure(problem, &circuit, &[&problem.hamiltonian])?[0];
                circuit.angles[k] = original - std::f64::consts::FRAC_PI_4;
                let minus = self.measure(problem, &circuit, &[&problem.hamiltonian])?[0];
                circuit.angles[k] = original;
                let (index, scale) = circuit.links[k];
                gradient[index] += scale * (plus - minus);
            }
        }
        Ok(evolution::Evaluation {
            energy,
            gradient,
            state: problem.prepare_state(indices, parameters, true)?,
        })
    }
    fn pool_gradients(
        &mut self,
        problem: &CachedProblem,
        indices: &[usize],
        parameters: &[f64],
    ) -> Result<Vec<f64>> {
        let circuit = Self::circuit(problem, indices, parameters)?;
        let observables = problem
            .pool
            .iter()
            .map(|g| {
                problem
                    .hamiltonian
                    .commutator(&problem.context, &g.operator)
            })
            .collect::<Result<Vec<_>>>()?;
        self.measure(problem, &circuit, &observables.iter().collect::<Vec<_>>())
    }
}
