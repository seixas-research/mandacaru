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
    /// How many leading `observables` a measurement budget counts: the rest are covariance
    /// product strings, read on the same shots of the same basis at no cost.
    pub priced: usize,
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
    /// Shots per basis of a sampled device: a register-free energy is then measured basis by
    /// basis with its standard error ([`super::energy_error`]).
    shots: Option<u64>,
    error_plan: Option<(
        super::energy_error::Plan,
        Vec<(Vec<String>, Vec<usize>, Vec<usize>)>,
    )>,
    /// The standard error of every energy measured, keyed by the energy's bits.
    errors: std::collections::HashMap<u64, f64>,
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
        let priced = observables.len();
        Request {
            labels,
            angles,
            observables,
            priced,
        }
    }
}
impl<'a> CircuitExecution<'a> {
    pub fn new(call: &'a mut CircuitCall<'a>) -> Self {
        Self {
            call,
            stopped: None,
            batch_bytes: None,
            shots: None,
            error_plan: None,
            errors: std::collections::HashMap::new(),
        }
    }
    /// A sampled device taking `shots` per basis: register-free energies report their error.
    pub fn with_shots(mut self, shots: u64) -> Self {
        self.shots = (shots > 0).then_some(shots);
        self
    }
    /// The standard error of the measured energy `energy` (Hartree), when this execution
    /// measured it on a sampled device.
    pub fn energy_error(&self, energy: f64) -> Option<f64> {
        self.errors.get(&energy.to_bits()).copied()
    }
    /// Why the sampled energies carry no standard error, when they do not.
    pub fn energy_error_reason(&self) -> Option<&str> {
        self.error_plan.as_ref().and_then(|(plan, _)| plan.unavailable.as_deref())
    }
    /// Form the pool commutators of a gradient measurement in batches of about `bytes` accounted
    /// bytes, instead of the context's default ([`pool_measurement::batch_bytes`]).
    pub fn with_batch_bytes(mut self, bytes: usize) -> Self {
        self.batch_bytes = Some(bytes);
        self
    }
}
/// The Pauli rotations (labels, angles) of the circuit that prepares the ansatz state of
/// `problem` at `indices` and `parameters`, in circuit order, as the executor receives them.
pub fn circuit_rotations(
    problem: &CachedProblem,
    indices: &[usize],
    parameters: &[f64],
) -> Result<(Vec<String>, Vec<f64>)> {
    let circuit = CircuitExecution::circuit(problem, indices, parameters)?;
    Ok((circuit.labels, circuit.angles))
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
        if let (true, Some(shots)) = (problem.is_register_free(), self.shots) {
            return self.energy_with_error(problem, circuit, shots);
        }
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
impl CircuitExecution<'_> {
    /// `<H>` on a sampled device, one rotated basis per request with the product strings its
    /// covariance needs, and its standard error recorded (see [`super::energy_error`]).
    fn energy_with_error(
        &mut self,
        problem: &CachedProblem,
        circuit: &Circuit,
        shots: u64,
    ) -> Result<f64> {
        self.prepare_error_plan(problem)?;
        let requests = {
            let (plan, positions) = self.error_plan.as_ref().unwrap();
            plan.bases
                .iter()
                .zip(positions)
                .map(|(basis, (labels, strings, _))| {
                    let mut request = circuit.request(&basis.tail, labels.clone());
                    // The basis's strings come first; its products follow and are not priced.
                    request.priced = strings.len();
                    request
                })
                .collect()
        };
        let readings = self.call(problem, requests)?;
        let values = readings
            .readings
            .iter()
            .map(|r| match r {
                Reading::Values(v) => Ok(v.clone()),
                Reading::State(_) => Err(invalid("a sampled energy needs Pauli expectations")),
            })
            .collect::<Result<Vec<_>>>()?;
        let (plan, positions) = self.error_plan.as_ref().unwrap();
        let (energy, error) = plan.energy(positions, &values, shots)?;
        if let Some(error) = error {
            self.errors.insert(energy.to_bits(), error);
        }
        Ok(energy)
    }
}
impl CircuitExecution<'_> {
    /// Build the sampled energy's measurement plan once per execution.
    fn prepare_error_plan(&mut self, problem: &CachedProblem) -> Result<()> {
        if self.error_plan.is_none() {
            let plan = error_plan(problem)?;
            let positions = plan.observables();
            self.error_plan = Some((plan, positions));
        }
        Ok(())
    }
}
/// The measurement of a sampled energy with its error: the Hamiltonian's commuting bases, or the
/// factorized plan's.
fn error_plan(problem: &CachedProblem) -> Result<super::energy_error::Plan> {
    match &problem.factorized {
        None => super::energy_error::Plan::commuting(&problem.context, &problem.hamiltonian),
        Some(plan) => super::energy_error::Plan::factorized(plan),
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

/// The parameter gradient and Hessian of an ansatz measured on the device by the shift rule of
/// its Pauli rotations (see [`CircuitExecution::gradient_and_hessian`]).
pub struct DeviceHessian {
    pub energy: f64,
    pub gradient: Vec<f64>,
    pub hessian: nalgebra::DMatrix<f64>,
    /// The Pauli rotations of the circuit (`R`).
    pub rotations: usize,
    /// Energy evaluations spent: `1 + 2 R^2`.
    pub energy_evaluations: usize,
    /// Circuits submitted: the evaluations times the energy's measurement bases.
    pub circuits: usize,
    /// How each of the `1 + 2 R^2` energies enters `gradient` and `hessian`, with its standard
    /// error (zero on an exact device), for [`super::lagrangian::measured_z`].
    pub energies: Vec<super::lagrangian::MeasuredEnergy>,
    /// On a sampled device whose energies carry no error (`energy_error::Plan::unavailable`), the
    /// reason; `energies` then hold zero errors, which `measured_z` takes as no error at all (no
    /// curvature check, no covariance), and the device force reports no error.
    pub error_unavailable: Option<String>,
}

/// The cost of [`CircuitExecution::gradient_and_hessian`] before any job: the rotations `R`, the
/// energy evaluations `1 + 2 R^2` and the circuits (evaluations times the energy's bases: on a
/// `sampled` device the bases of the energy measured with its error, [`super::energy_error`]).
pub fn device_hessian_plan(
    problem: &CachedProblem,
    indices: &[usize],
    parameters: &[f64],
    sampled: bool,
) -> Result<(usize, usize, usize)> {
    let circuit = CircuitExecution::circuit(problem, indices, parameters)?;
    let r = circuit.angles.len();
    let evaluations = r
        .checked_mul(r)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_add(1))
        .ok_or(Error::Overflow)?;
    let bases = if sampled {
        error_plan(problem)?.bases.len()
    } else {
        problem.factorized.as_ref().map_or(1, |plan| plan.bases.len())
    };
    Ok((
        r,
        evaluations,
        evaluations.checked_mul(bases).ok_or(Error::Overflow)?,
    ))
}

/// The `2 R` shifted circuits whose energies give `z . g = sum_a z_k c_a E(a)` at any geometry,
/// rotation `a` of parameter `k` (angle `s_a theta_k`) shifted by `+-pi/4` with the coefficient
/// `c_a = +-s_a`: `(labels, angles, k, c_a)`. The shift rule is exact for every Hamiltonian, so
/// the RDMs of these circuits turn `z . g(R)` into a density; the weight of a circuit is
/// `z_k c_a`, and `k` lets the force carry the error of `z` (`response::device_gradient`).
pub fn shifted_circuits(
    problem: &CachedProblem,
    indices: &[usize],
    parameters: &[f64],
    z: &[f64],
) -> Result<Vec<(Vec<String>, Vec<f64>, usize, f64)>> {
    if z.len() != parameters.len() || z.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "the Z vector must be finite with one entry per parameter",
        ));
    }
    let circuit = CircuitExecution::circuit(problem, indices, parameters)?;
    let mut out = Vec::with_capacity(2 * circuit.angles.len());
    for (a, &(index, scale)) in circuit.links.iter().enumerate() {
        for (sign, shift) in [
            (1., std::f64::consts::FRAC_PI_4),
            (-1., -std::f64::consts::FRAC_PI_4),
        ] {
            let mut angles = circuit.angles.clone();
            angles[a] += shift;
            out.push((circuit.labels.clone(), angles, index, sign * scale));
        }
    }
    Ok(out)
}

impl CircuitExecution<'_> {
    /// The energy, parameter gradient and Hessian of the ansatz at `parameters`, measured in one
    /// batch. Each Pauli rotation `exp(i phi_a P_a)` (angle `phi_a = s_a theta_k`) makes the
    /// energy `c0 + c1 cos 2 phi_a + c2 sin 2 phi_a` in `phi_a`, so with shifts of `pi/4`:
    /// `dE/dphi_a = E(+a) - E(-a)`, `d2E/dphi_a^2 = 2 [E(+a) + E(-a)] - 4 E`, and
    /// `d2E/dphi_a dphi_b = E(++) - E(+-) - E(-+) + E(--)` for `a != b`, all exact; the parameter
    /// Hessian is `sum_{a in k, b in l} s_a s_b d2E/dphi_a dphi_b`. `1 + 2 R^2` energies.
    pub fn gradient_and_hessian(
        &mut self,
        problem: &CachedProblem,
        indices: &[usize],
        parameters: &[f64],
    ) -> Result<DeviceHessian> {
        use std::f64::consts::FRAC_PI_4;
        let circuit = Self::circuit(problem, indices, parameters)?;
        let r = circuit.angles.len();
        let sampled = self.shots.is_some();
        let (_, evaluations, circuits) = device_hessian_plan(problem, indices, parameters, sampled)?;
        // The shifted angle sets, in the order the energies are read back.
        let mut variants: Vec<Vec<f64>> = Vec::with_capacity(evaluations);
        variants.push(circuit.angles.clone());
        for a in 0..r {
            for shift in [FRAC_PI_4, -FRAC_PI_4] {
                let mut angles = circuit.angles.clone();
                angles[a] += shift;
                variants.push(angles);
            }
        }
        for a in 0..r {
            for b in a + 1..r {
                for (sa, sb) in [(1., 1.), (1., -1.), (-1., 1.), (-1., -1.)] {
                    let mut angles = circuit.angles.clone();
                    angles[a] += sa * FRAC_PI_4;
                    angles[b] += sb * FRAC_PI_4;
                    variants.push(angles);
                }
            }
        }
        debug_assert_eq!(variants.len(), evaluations);
        let shifted = |angles: &Vec<f64>| Circuit {
            labels: circuit.labels.clone(),
            angles: angles.clone(),
            links: vec![],
        };
        // `(E, sigma_E)` of every variant, all in one call.
        let mut error_unavailable = None;
        let measured: Vec<(f64, f64)> = if let Some(shots) = self.shots {
            // A sampled device: each energy basis by basis with the product strings of its
            // covariance (`energy_error`), so every energy carries its standard error.
            self.prepare_error_plan(problem)?;
            let requests = {
                let (plan, positions) = self.error_plan.as_ref().unwrap();
                let mut requests = Vec::with_capacity(circuits);
                for angles in &variants {
                    let variant = shifted(angles);
                    for (basis, (labels, strings, _)) in plan.bases.iter().zip(positions) {
                        let mut request = variant.request(&basis.tail, labels.clone());
                        // The basis's strings come first; its products follow and are not priced.
                        request.priced = strings.len();
                        requests.push(request);
                    }
                }
                requests
            };
            let readings = self.call(problem, requests)?;
            let (plan, positions) = self.error_plan.as_ref().unwrap();
            let nb = plan.bases.len();
            let values = readings
                .readings
                .iter()
                .map(|r| match r {
                    Reading::Values(v) => Ok(v.clone()),
                    Reading::State(_) => Err(invalid("a sampled energy needs Pauli expectations")),
                })
                .collect::<Result<Vec<_>>>()?;
            // A basis with too many covariance products leaves the energies without an error
            // (owner, "Exempt products from budget"; for the device force "Run, error None"): the
            // force is still measured and its error is not reported, with this reason.
            error_unavailable = plan.unavailable.clone();
            (0..evaluations)
                .map(|v| {
                    plan.energy(positions, &values[v * nb..(v + 1) * nb], shots)
                        .map(|(energy, error)| (energy, error.unwrap_or(0.)))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
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
            for (_, observable) in &bases {
                let mut set = PauliSet::new(&problem.context, observable.n)?;
                set.insert(&problem.context, [*observable])?;
                sets.push(set);
            }
            let mut requests = Vec::with_capacity(circuits);
            for angles in &variants {
                let variant = shifted(angles);
                for ((rotation, _), set) in bases.iter().zip(&sets) {
                    requests.push(variant.request(rotation, set.labels()));
                }
            }
            let readings = self.call(problem, requests)?;
            let nb = bases.len();
            let mut energies = Vec::with_capacity(evaluations);
            for v in 0..evaluations {
                let values = bases
                    .iter()
                    .zip(&sets)
                    .enumerate()
                    .map(|(k, ((_, observable), set))| {
                        readings.read(problem, v * nb + k, observable, &|t| set.position(t))
                    })
                    .collect::<Result<Vec<_>>>()?;
                energies.push((
                    match &problem.factorized {
                        None => values[0],
                        Some(plan) => plan.energy(&values)?,
                    },
                    0.,
                ));
            }
            energies
        };
        let energies: Vec<f64> = measured.iter().map(|&(e, _)| e).collect();
        let dependences = energy_dependences(&circuit.links, &measured);
        let base = energies[0];
        let plus = |a: usize| energies[1 + 2 * a];
        let minus = |a: usize| energies[2 + 2 * a];
        let mut angle_hessian = nalgebra::DMatrix::<f64>::zeros(r, r);
        let mut cursor = 1 + 2 * r;
        for a in 0..r {
            angle_hessian[(a, a)] = 2. * (plus(a) + minus(a)) - 4. * base;
            for b in a + 1..r {
                let e = &energies[cursor..cursor + 4];
                let value = e[0] - e[1] - e[2] + e[3];
                angle_hessian[(a, b)] = value;
                angle_hessian[(b, a)] = value;
                cursor += 4;
            }
        }
        let n = parameters.len();
        let mut gradient = vec![0.; n];
        let mut hessian = nalgebra::DMatrix::<f64>::zeros(n, n);
        for (a, &(i, si)) in circuit.links.iter().enumerate() {
            gradient[i] += si * (plus(a) - minus(a));
            for (b, &(j, sj)) in circuit.links.iter().enumerate() {
                hessian[(i, j)] += si * sj * angle_hessian[(a, b)];
            }
        }
        if gradient
            .iter()
            .chain(hessian.iter())
            .any(|v| !v.is_finite())
        {
            return Err(Error::Kernel("nonfinite measured parameter Hessian".into()));
        }
        Ok(DeviceHessian {
            energy: base,
            gradient,
            hessian,
            rotations: r,
            energy_evaluations: evaluations,
            circuits,
            energies: dependences,
            error_unavailable,
        })
    }
}

/// How each energy of [`CircuitExecution::gradient_and_hessian`] (in its batch order) enters the
/// parameter gradient and Hessian, with its error: the base `-4 s_a^2` on every diagonal; `E(+-a)`
/// `+-s_a` in the gradient and `2 s_a^2` on the diagonal; the four energies of a pair
/// `(+-1) s_a s_b` on `(k_a, k_b)` and `(k_b, k_a)`.
fn energy_dependences(
    links: &[(usize, f64)],
    measured: &[(f64, f64)],
) -> Vec<super::lagrangian::MeasuredEnergy> {
    use super::lagrangian::MeasuredEnergy;
    let r = links.len();
    let mut out = Vec::with_capacity(measured.len());
    out.push(MeasuredEnergy {
        error: measured[0].1,
        gradient: vec![],
        hessian: links.iter().map(|&(i, s)| (i, i, -4. * s * s)).collect(),
    });
    for &(i, s) in links {
        for sign in [1., -1.] {
            out.push(MeasuredEnergy {
                error: measured[out.len()].1,
                gradient: vec![(i, sign * s)],
                hessian: vec![(i, i, 2. * s * s)],
            });
        }
    }
    for a in 0..r {
        for b in a + 1..r {
            let ((i, si), (j, sj)) = (links[a], links[b]);
            for sign in [1., -1., -1., 1.] {
                let c = sign * si * sj;
                out.push(MeasuredEnergy {
                    error: measured[out.len()].1,
                    gradient: vec![],
                    hessian: vec![(i, j, c), (j, i, c)],
                });
            }
        }
    }
    debug_assert_eq!(out.len(), measured.len());
    out
}
