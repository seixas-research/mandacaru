use super::amplitude::{self, inner_wide, Amplitude, State};
use super::{algebra::Generator, evolution, optimizer, Basis, Encoding, Operator, C64};
use crate::{math::storage::Precision, product, Buffer, Context, Error, Reservation, Result};
use num_complex::Complex32;
use std::{path::Path, time::Instant};

/// Run `body` once with `A` bound to the stored amplitude of `precision`.
/// The one dispatch point of every quantum entry: below it the code is
/// generic, so there is a single numerical definition per precision.
macro_rules! by_precision {
    ($precision:expr, $a:ident => $body:expr) => {
        match $precision {
            $crate::math::storage::Precision::Float64 => {
                type $a = $crate::quantum::C64;
                $body
            }
            $crate::math::storage::Precision::Float32 => {
                type $a = num_complex::Complex32;
                $body
            }
        }
    };
}
pub(crate) use by_precision;

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
pub type GrowthObserver<'a> = dyn FnMut(&[usize], &[f64], f64) -> Result<()> + 'a;

/// External circuit execution. Optimization and operator construction stay native.
pub trait Execution {
    fn evaluate(
        &mut self,
        problem: &CachedProblem,
        indices: &[usize],
        parameters: &[f64],
        gradients: bool,
    ) -> Result<evolution::Evaluation>;
    fn pool_gradients(
        &mut self,
        problem: &CachedProblem,
        indices: &[usize],
        parameters: &[f64],
    ) -> Result<Vec<f64>>;
}

pub struct CachedProblem {
    pub context: Context,
    pub encoding: Encoding,
    pub hamiltonian: Operator,
    pub basis: Basis,
    pub pool: Vec<Generator>,
    pub dense: Option<Buffer<C64>>,
    /// Storage precision of every state this problem prepares. The
    /// Hamiltonian, the pool, the parameters and all reductions are float64.
    pub precision: Precision,
    pub sparse: Option<super::sparse::SparseMatrix>,
    pub pool_screenable: Vec<bool>,
    pub growth_members: Vec<Vec<usize>>,
    pub hva: Option<super::hva::HVA>,
    pub(super) _pool_metadata: Reservation,
}
impl CachedProblem {
    /// Read a bounded schema-v1 JSON cache; integrals and quantum SDKs are absent.
    pub fn load(
        ctx: &Context,
        path: &Path,
        sector: bool,
        dense: bool,
        singles: bool,
    ) -> Result<Self> {
        crate::persistence::Document::read(ctx, path, crate::persistence::HAMILTONIAN)?
            .problem(ctx, sector, dense, singles)
    }

    pub fn new(
        ctx: &Context,
        encoding: Encoding,
        hamiltonian: Operator,
        sector: bool,
        dense: bool,
        singles: bool,
    ) -> Result<Self> {
        if hamiltonian.n != encoding.n_qubits() || !hamiltonian.is_hermitian() {
            return Err(invalid(
                "Hamiltonian must be Hermitian and match its encoding",
            ));
        }
        let basis = Basis::new(ctx, &encoding, sector)?;
        basis.check_invariant(&hamiltonian)?;
        let built = super::pools::build(ctx, &encoding, "fermionic", singles)?;
        let pool = built.operators;
        for g in &pool {
            basis.check_invariant(&g.operator)?;
        }
        let dense = if dense {
            Some(hamiltonian.dense(ctx, &basis)?)
        } else {
            None
        };
        Ok(Self {
            context: ctx.clone(),
            encoding,
            hamiltonian,
            basis,
            pool,
            dense,
            precision: Precision::Float64,
            sparse: None,
            pool_screenable: built.screenable,
            growth_members: built.growth_members,
            hva: None,
            _pool_metadata: built.memory,
        })
    }
    /// State storage: complex128 (`Float64`, the default) or complex64.
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }
    pub fn use_pool(&mut self, name: &str, singles: bool) -> Result<()> {
        let pool = super::pools::build(&self.context, &self.encoding, name, singles)?;
        for generator in &pool.operators {
            self.basis.check_invariant(&generator.operator)?;
        }
        self.pool = pool.operators;
        self.pool_screenable = pool.screenable;
        self.growth_members = pool.growth_members;
        self._pool_metadata = pool.memory;
        self.hva = None;
        Ok(())
    }
    fn growth_steps(&self, indices: &[usize]) -> Result<usize> {
        let mut count = 0;
        let mut i = 0;
        while i < indices.len() {
            let index = indices[i];
            if index >= self.pool.len() {
                return Err(invalid("generator index exceeds pool"));
            }
            if self.pool_screenable[index] {
                i += 1;
            } else {
                if i + 1 >= indices.len()
                    || !self
                        .growth_members
                        .iter()
                        .any(|m| m.as_slice() == &indices[i..i + 2])
                {
                    return Err(invalid("checkpoint splits a coupled-exchange growth step"));
                }
                i += 2;
            }
            count += 1;
        }
        Ok(count)
    }
    pub fn evaluate(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
    ) -> Result<evolution::Evaluation> {
        by_precision!(self.precision, A => self.evaluate_as::<A>(indices, parameters, trotter))
    }
    fn evaluate_as<A: Amplitude>(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
    ) -> Result<evolution::Evaluation> {
        if let Some(hva) = &self.hva {
            let expanded = hva.expand(&self.pool, indices, parameters)?;
            let mut out = evolution::evaluate::<A>(
                &self.context,
                &self.basis,
                &self.hamiltonian,
                self.dense.as_ref().map(|d| d.as_slice()),
                self.sparse.as_ref(),
                &expanded.operators,
                &expanded.parameters,
                true,
            )?;
            let mut gradient = vec![0.; parameters.len()];
            for (g, (i, w)) in out.gradient.iter().zip(expanded.links) {
                gradient[i] += w * g;
            }
            out.gradient = gradient;
            return Ok(out);
        }
        let ops: Result<Vec<_>> = indices
            .iter()
            .map(|&i| {
                self.pool
                    .get(i)
                    .map(|g| &g.operator)
                    .ok_or_else(|| invalid("generator index exceeds pool"))
            })
            .collect();
        evolution::evaluate::<A>(
            &self.context,
            &self.basis,
            &self.hamiltonian,
            self.dense.as_ref().map(|d| d.as_slice()),
            self.sparse.as_ref(),
            &ops?,
            parameters,
            trotter,
        )
    }
    pub fn prepare_state(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
    ) -> Result<State> {
        by_precision!(self.precision, A => self.prepare_state_as::<A>(indices, parameters, trotter))
    }
    fn prepare_state_as<A: Amplitude>(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
    ) -> Result<State> {
        Ok(A::wrap(
            self.prepare_buffer::<A>(indices, parameters, trotter)?,
        ))
    }
    pub(super) fn prepare_buffer<A: Amplitude>(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
    ) -> Result<Buffer<A>> {
        let expanded = self.sequence(indices, parameters)?;
        evolution::state_from::<A>(
            &self.context,
            &self.basis,
            &expanded.operators,
            &expanded.parameters,
            trotter || self.hva.is_some(),
            false,
            None,
        )
    }
    /// `2 Re <H psi | G_i psi>` for every pool generator (the Hamiltonian
    /// without its identity term when `centered`). `H psi` is stored in the
    /// state precision, `G_i psi` is a float64 row, the sums are float64.
    pub(super) fn pool_gradients(&self, state: &State, centered: bool) -> Result<Vec<f64>> {
        match state {
            State::Float64(buffer) => self.pool_gradients_of::<C64>(buffer.as_slice(), centered),
            State::Float32(buffer) => {
                self.pool_gradients_of::<Complex32>(buffer.as_slice(), centered)
            }
        }
    }
    fn pool_gradients_of<A: Amplitude>(&self, state: &[A], centered: bool) -> Result<Vec<f64>> {
        let ctx = &self.context;
        let d = self.basis.len();
        if state.len() != d || !amplitude::finite(state) {
            return Err(invalid("state must be finite and match the operator basis"));
        }
        let mut hpsi = ctx.allocate::<A>(d)?;
        {
            let mut acc = ctx.allocate::<C64>(A::accumulator_len(d))?;
            A::with_wide(hpsi.as_mut_slice(), acc.as_mut_slice(), |wide| {
                if centered {
                    self.hamiltonian
                        .add_centered_into(&self.basis, state, C64::new(1., 0.), wide)
                } else {
                    self.hamiltonian
                        .add_into(&self.basis, state, C64::new(1., 0.), wide)
                }
            });
        }
        let mut action = ctx.allocate::<C64>(d)?;
        let mut out = Vec::with_capacity(self.pool.len());
        for g in &self.pool {
            g.operator
                .apply_into(&self.basis, state, action.as_mut_slice());
            out.push(2. * inner_wide(hpsi.as_slice(), action.as_slice()).re);
        }
        Ok(out)
    }
    pub fn sequence<'a>(
        &'a self,
        indices: &[usize],
        parameters: &[f64],
    ) -> Result<super::hva::Expanded<'a>> {
        if let Some(hva) = &self.hva {
            return hva.expand(&self.pool, indices, parameters);
        }
        if indices.len() != parameters.len() {
            return Err(invalid("generator indices must match parameters"));
        }
        let operators = indices
            .iter()
            .map(|&i| {
                self.pool
                    .get(i)
                    .map(|g| &g.operator)
                    .ok_or_else(|| invalid("generator index exceeds pool"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(super::hva::Expanded {
            operators,
            parameters: parameters.to_vec(),
            links: (0..parameters.len()).map(|i| (i, 1.)).collect(),
        })
    }
    pub fn solve(&self, options: &SolveOptions, initial: Option<&[f64]>) -> Result<SolveResult> {
        self.solve_sequence(options, initial, None)
    }
    pub fn solve_sequence(
        &self,
        options: &SolveOptions,
        initial: Option<&[f64]>,
        sequence: Option<&[usize]>,
    ) -> Result<SolveResult> {
        self.context
            .run(|| self.solve_inner(options, initial, sequence, None, None))
    }
    pub fn solve_observed(
        &self,
        options: &SolveOptions,
        initial: Option<&[f64]>,
        sequence: Option<&[usize]>,
        observer: &mut GrowthObserver<'_>,
    ) -> Result<SolveResult> {
        self.context
            .run(|| self.solve_inner(options, initial, sequence, Some(observer), None))
    }
    pub fn solve_executed(
        &self,
        options: &SolveOptions,
        initial: Option<&[f64]>,
        observer: Option<&mut GrowthObserver<'_>>,
        execution: &mut dyn Execution,
    ) -> Result<SolveResult> {
        self.context
            .run(|| self.solve_inner(options, initial, None, observer, Some(execution)))
    }
    fn solve_inner(
        &self,
        options: &SolveOptions,
        initial: Option<&[f64]>,
        sequence: Option<&[usize]>,
        mut observer: Option<&mut GrowthObserver<'_>>,
        mut execution: Option<&mut dyn Execution>,
    ) -> Result<SolveResult> {
        // A circuit executor measures energies: the state floors do not apply.
        options.validate(if execution.is_none() {
            self.precision
        } else {
            Precision::Float64
        })?;
        if options.adapt && self.hva.is_some() {
            return Err(invalid("HVA is a fixed ansatz"));
        }
        if sequence.is_some_and(|s| {
            !options.adapt
                || self.growth_steps(s).is_err()
                || self.growth_steps(s).is_ok_and(|n| n > options.max_growth)
                || s.iter().any(|&i| i >= self.pool.len())
                || initial.is_none_or(|p| p.len() != s.len())
        }) {
            return Err(invalid(
                "resumed ADAPT sequence must fit the pool, parameters and total growth budget",
            ));
        }
        let start = Instant::now();
        let ctx = &self.context;
        let max_parameters = if options.adapt {
            product(&[
                options.max_growth,
                self.growth_members
                    .iter()
                    .map(Vec::len)
                    .max()
                    .unwrap_or(1)
                    .max(1),
            ])?
        } else {
            self.pool.len()
        };
        // Hold output/history reservations for the entire lifetime of the result.
        let _memory = ctx.reserve(
            product(&[
                options.optimizer.max_evaluations,
                if options.adapt {
                    options
                        .max_growth
                        .checked_add(usize::from(sequence.is_some_and(|s| !s.is_empty())))
                        .ok_or(Error::Overflow)?
                        .max(1)
                } else {
                    1
                },
                8,
            ])?
            .checked_add(product(&[max_parameters, 256])?)
            .ok_or(Error::Overflow)?,
        )?;
        let reference = self.evaluate(&[], &[], true)?.energy;
        let derivatives = !matches!(
            options.optimizer.method,
            optimizer::Method::Cobyla | optimizer::Method::NelderMead | optimizer::Method::Spsa
        );
        let mut evaluate = |indices: &[usize],
                            parameters: &[f64],
                            trotter: bool,
                            gradients: bool| match execution.as_mut() {
            Some(device) => device.evaluate(self, indices, parameters, gradients),
            None => self.evaluate(indices, parameters, trotter),
        };
        let mut indices = Vec::new();
        let mut parameters = Vec::new();
        let mut iterations = Vec::new();
        let mut history = Vec::new();
        let mut steps = 0;
        let mut success = false;
        let mut message = "growth budget exhausted".to_owned();
        let mut final_gradient = 0.;
        let mut seed = None;
        // A run budget exhausted mid-solve (the executor answered with a stop)
        // ends the solve at the best point reached. `measured` holds the last
        // measured energy of the accepted ansatz; without one there is no
        // point to keep and the interruption propagates.
        let mut stopped: Option<String> = None;
        let mut measured: Option<f64> = None;
        if !options.adapt {
            indices = (0..self.pool.len())
                .filter(|&i| self.pool_screenable[i])
                .collect();
            parameters = initial.map(|x| x.to_vec()).unwrap_or_else(|| {
                vec![self.hva.as_ref().map_or(0., |h| h.seed_angle); indices.len()]
            });
            if parameters.len() != indices.len() {
                return Err(invalid(
                    "initial parameter count differs from the UCCSD pool",
                ));
            }
            let outcome =
                optimizer::minimize_interruptible(ctx, &parameters, &options.optimizer, |t| {
                    let v = evaluate(&indices, t, options.trotter, derivatives)?;
                    Ok((v.energy, v.gradient))
                })?;
            parameters = outcome.parameters;
            history = outcome.history;
            steps = outcome.steps;
            success = outcome.success;
            message = outcome.message;
            measured = Some(outcome.energy);
            stopped = outcome.interrupted;
            final_gradient = f64::NAN;
            if stopped.is_none() {
                match evaluate(&indices, &parameters, options.trotter, true) {
                    Ok(v) => {
                        final_gradient = v.gradient.iter().fold(0f64, |a, g| a.max(g.abs()));
                    }
                    Err(Error::Interrupted(m)) => stopped = Some(m),
                    Err(e) => return Err(e),
                }
            }
        } else {
            if sequence.is_none() && initial.is_some_and(|p| !p.is_empty()) {
                return Err(invalid("basic ADAPT starts with an empty ansatz"));
            }
            if let Some(sequence) = sequence {
                indices = sequence.to_vec();
                parameters = initial.unwrap().to_vec();
                if !indices.is_empty() {
                    let outcome = optimizer::minimize(ctx, &parameters, &options.optimizer, |t| {
                        let value = evaluate(&indices, t, true, derivatives)?;
                        Ok((value.energy, value.gradient))
                    })?;
                    let kept = outcome.energy < reference;
                    seed = Some(SeedRelaxation {
                        kept,
                        energy: outcome.energy,
                        success: outcome.success,
                        message: outcome.message.clone(),
                        evaluations: outcome.history.len(),
                        steps: outcome.steps,
                    });
                    steps += outcome.steps;
                    history.extend(outcome.history);
                    parameters = outcome.parameters;
                    if !kept {
                        indices.clear();
                        parameters.clear();
                    }
                }
            }
            let mut delta = None;
            for growth in self.growth_steps(&indices)?..=options.max_growth {
                // Release the objective closure before measured pool screening.
                let evaluation = match execution.as_mut() {
                    Some(device) => match device.evaluate(self, &indices, &parameters, false) {
                        Err(Error::Interrupted(m)) if measured.is_some() => {
                            stopped = Some(m);
                            break;
                        }
                        other => other?,
                    },
                    None => self.evaluate(&indices, &parameters, true)?,
                };
                let energy = evaluation.energy;
                measured = Some(energy);
                let mut selected = 0;
                final_gradient = 0.;
                let mut gradients = vec![0.; self.pool.len()];
                for (i, signed) in self
                    .pool_gradients(&evaluation.state, true)?
                    .into_iter()
                    .enumerate()
                {
                    let magnitude = signed.abs();
                    gradients[i] = magnitude;
                    if self.pool_screenable[i] && magnitude > final_gradient {
                        final_gradient = magnitude;
                        selected = i;
                    }
                }
                if !indices.is_empty() {
                    if let Some(device) = execution.as_mut() {
                        gradients = match device.pool_gradients(self, &indices, &parameters) {
                            Err(Error::Interrupted(m)) => {
                                stopped = Some(m);
                                break;
                            }
                            other => other?,
                        };
                        if gradients.len() != self.pool.len()
                            || gradients.iter().any(|g| !g.is_finite())
                        {
                            return Err(invalid(
                                "external pool gradients have invalid size or values",
                            ));
                        }
                        final_gradient = 0.;
                        for (i, gradient) in gradients.iter_mut().enumerate() {
                            *gradient = gradient.abs();
                            if self.pool_screenable[i] && *gradient > final_gradient {
                                final_gradient = *gradient;
                                selected = i;
                            }
                        }
                    }
                }
                let gradient_ok = options
                    .gradient_tolerance
                    .is_none_or(|t| final_gradient < t);
                let energy_ok = options
                    .energy_tolerance
                    .is_none_or(|t| delta.is_some_and(|d: f64| d.abs() < t));
                if final_gradient < 1e-12 || (gradient_ok && energy_ok) {
                    success = true;
                    message = "pool gradients converged".into();
                    break;
                }
                if growth == options.max_growth {
                    break;
                }
                let members: Vec<_> = self.growth_members[selected]
                    .iter()
                    .copied()
                    .filter(|&i| gradients[i] > 1e-12)
                    .collect();
                let grown = if members.len() >= 2 {
                    members
                } else {
                    vec![selected]
                };
                let accepted = indices.len();
                for index in grown {
                    indices.push(index);
                    parameters.push(0.);
                }

                let outcome = match optimizer::minimize_interruptible(
                    ctx,
                    &parameters,
                    &options.optimizer,
                    |t| {
                        let v = match execution.as_mut() {
                            Some(device) => device.evaluate(self, &indices, t, derivatives)?,
                            None => self.evaluate(&indices, t, true)?,
                        };
                        Ok((v.energy, v.gradient))
                    },
                ) {
                    // Stopped before the grown ansatz was measured once:
                    // growth ends at the last accepted ansatz.
                    Err(Error::Interrupted(m)) if measured.is_some() => {
                        indices.truncate(accepted);
                        parameters.truncate(accepted);
                        stopped = Some(m);
                        break;
                    }
                    other => other?,
                };
                delta = Some(outcome.energy - energy);
                measured = Some(outcome.energy);
                parameters = outcome.parameters;
                steps += outcome.steps;
                iterations.push(Iteration {
                    pool_index: selected,
                    energy: outcome.energy,
                    max_gradient: final_gradient,
                    optimizer_steps: outcome.steps,
                    num_evaluations: outcome.history.len(),
                    success: outcome.success,
                    message: outcome.message.clone(),
                    num_parameters: parameters.len(),
                });
                history.extend(outcome.history);
                if let Some(observer) = observer.as_deref_mut() {
                    observer(&indices, &parameters, outcome.energy)?;
                }
                if outcome.interrupted.is_some() {
                    // The partly optimized ansatz at its best point is kept.
                    stopped = outcome.interrupted;
                    break;
                }
                // Preserve an inner optimizer's best point and failure status.
                // Growth continues; only the outer pool criteria certify ADAPT
                // convergence, even when the inner tolerance hits roundoff.
            }
        }
        let trotter = options.adapt || options.trotter;
        let final_evaluation = match (execution.as_mut(), stopped.is_some()) {
            (_, true) => None,
            (Some(device), false) => match device.evaluate(self, &indices, &parameters, false) {
                Err(Error::Interrupted(m)) if measured.is_some() => {
                    stopped = Some(m);
                    None
                }
                other => Some(other?),
            },
            (None, false) => Some(self.evaluate(&indices, &parameters, trotter)?),
        };
        // After a stop no circuit is submitted: the state comes from the
        // native simulator and the energy is the last one measured.
        let (energy, state) = match final_evaluation {
            Some(evaluation) => (evaluation.energy, evaluation.state),
            None => (
                measured.expect("a stop keeps a measured point"),
                self.prepare_state(&indices, &parameters, trotter)?,
            ),
        };
        if let Some(reason) = &stopped {
            success = false;
            message = format!("run budget exhausted: {reason}");
        }
        if options.adapt && execution.is_none() {
            final_gradient = 0.;
            for (i, signed) in self.pool_gradients(&state, true)?.into_iter().enumerate() {
                if self.pool_screenable[i] {
                    final_gradient = final_gradient.max(signed.abs());
                }
            }
        }
        let labels = indices
            .iter()
            .map(|&i| self.pool[i].label.clone())
            .collect();
        Ok(SolveResult {
            seed,
            energy,
            reference,
            parameters,
            state,
            indices,
            labels,
            iterations,
            history,
            optimizer_steps: steps,
            success,
            message,
            final_gradient,
            budget_stopped: stopped.is_some(),
            elapsed_seconds: start.elapsed().as_secs_f64(),
            _memory,
        })
    }
}
#[derive(Clone, Debug)]
pub struct SolveOptions {
    pub adapt: bool,
    pub trotter: bool,
    pub max_growth: usize,
    pub gradient_tolerance: Option<f64>,
    pub energy_tolerance: Option<f64>,
    pub optimizer: optimizer::Options,
}
impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            adapt: false,
            trotter: false,
            max_growth: 50,
            gradient_tolerance: Some(1e-7),
            energy_tolerance: None,
            optimizer: optimizer::Options::default(),
        }
    }
}
impl SolveOptions {
    /// Reject thresholds the state precision cannot resolve. The float32
    /// floors are the measured values of `amplitude::FLOAT32_FLOORS`.
    pub fn validate(&self, precision: Precision) -> Result<()> {
        self.optimizer.validate()?;
        if self
            .gradient_tolerance
            .into_iter()
            .chain(self.energy_tolerance)
            .any(|t| !t.is_finite() || t <= 0.)
            || (self.adapt && self.gradient_tolerance.is_none() && self.energy_tolerance.is_none())
        {
            return Err(invalid(
                "ADAPT needs positive finite convergence thresholds and at least one criterion",
            ));
        }
        if let Some(floors) = amplitude::floors(precision) {
            let name = precision.name();
            // `None` is the optimizer default (1e-9), below the floor; SPSA
            // without a tolerance runs its whole budget and tests nothing.
            let tolerance = self
                .optimizer
                .tolerance
                .or((self.optimizer.method != optimizer::Method::Spsa)
                    .then_some(optimizer::DEFAULT_TOLERANCE));
            if tolerance.is_some_and(|t| t < floors.energy) {
                return Err(invalid(&format!(
                    "optimizer tolerance {:e} Ha is below the {name} state floor {:e} Ha; \
                     raise it or use precision='float64'",
                    tolerance.unwrap(),
                    floors.energy
                )));
            }
            if self.adapt {
                if let Some(t) = self.energy_tolerance.filter(|&t| t < floors.energy) {
                    return Err(invalid(&format!(
                        "ADAPT energy threshold {t:e} Ha is below the {name} state floor {:e} Ha; \
                         raise it or use precision='float64'",
                        floors.energy
                    )));
                }
                if let Some(t) = self.gradient_tolerance.filter(|&t| t < floors.gradient) {
                    return Err(invalid(&format!(
                        "ADAPT gradient threshold {t:e} Ha/rad is below the {name} state floor {:e} Ha/rad; \
                         raise it or use precision='float64'",
                        floors.gradient
                    )));
                }
            }
        }
        Ok(())
    }
}
pub struct Iteration {
    pub pool_index: usize,
    pub energy: f64,
    pub max_gradient: f64,
    pub optimizer_steps: usize,
    pub num_evaluations: usize,
    pub success: bool,
    pub message: String,
    pub num_parameters: usize,
}
pub struct SolveResult {
    pub seed: Option<SeedRelaxation>,
    pub energy: f64,
    pub reference: f64,
    pub parameters: Vec<f64>,
    pub state: State,
    pub indices: Vec<usize>,
    pub labels: Vec<String>,
    pub iterations: Vec<Iteration>,
    pub history: Vec<f64>,
    pub optimizer_steps: usize,
    pub success: bool,
    pub message: String,
    pub final_gradient: f64,
    /// The executor refused further circuits (its run budget is exhausted)
    /// and the result is the best point reached, `success == false`.
    pub budget_stopped: bool,
    pub elapsed_seconds: f64,
    pub(super) _memory: Reservation,
}
pub struct SeedRelaxation {
    pub kept: bool,
    pub energy: f64,
    pub success: bool,
    pub message: String,
    pub evaluations: usize,
    pub steps: usize,
}
