use super::{
    hessian::{drop_parameter, drop_parameter_of_inverse, extend_hessian, write_checkpoint},
    invalid, register_free_refusal, CapturedPoint, CapturedScreen, Execution, GrowthObserver,
    Iteration, QuantumProblem, SeedRelaxation, SolveOptions, SolveResult,
};
use crate::{math::storage::Precision, product, quantum::optimizer, Error, Result};
use std::time::Instant;

impl QuantumProblem {
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
        if self.is_register_free() && execution.is_none() {
            return Err(register_free_refusal("a native simulation"));
        }
        // A circuit executor measures energies: the state floors do not apply.
        options.validate(if execution.is_none() {
            self.precision
        } else {
            Precision::Float64
        })?;
        if options.adapt && self.hva.is_some() {
            return Err(invalid("HVA is a fixed ansatz"));
        }
        if options
            .checkpoint
            .as_ref()
            .is_some_and(|c| c.every == 0 || !["vqe", "adapt-vqe"].contains(&c.method.as_str()))
        {
            return Err(invalid(
                "a mid-run checkpoint needs a positive cadence and the vqe or adapt-vqe method",
            ));
        }
        if (options.tetris || options.prune) && (!options.adapt || sequence.is_some()) {
            return Err(invalid(
                "tetris and prune shape the growth of a fresh ADAPT run; a fixed ansatz, a resumed \
                 or a transferred sequence has none",
            ));
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
        if options.continuation && options.hessian.is_none() {
            return Err(invalid(
                "a continued growth needs the Hessian it stopped with",
            ));
        }
        if let Some(hessian) = &options.hessian {
            let n = sequence.map_or(0, <[usize]>::len);
            if !options.optimizer.method.carries_curvature()
                || sequence.is_none()
                || hessian.len() != n * n
            {
                return Err(invalid(
                    "a carried Hessian needs a carried SLSQP or BFGS ADAPT sequence of as many parameters",
                ));
            }
        }
        let start = Instant::now();
        let ctx = &self.context;
        // SLSQP keeps a Hessian approximation of the parameters, BFGS an inverse Hessian
        // approximation; a growth step that re-optimizes all of them starts from the previous
        // step's, extended by the identity for the appended ones (the identity is the cold start
        // of either), instead of relearning the curvature from the identity.
        let carries_curvature = options.carry_hessian
            && options.adapt
            && options.quenching
            && options.optimizer.method.carries_curvature();
        let mut curvature: Option<Vec<f64>> = None;
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
        // The energy of the reference determinant. A register-free problem reads it off the
        // Hamiltonian's diagonal Pauli terms (exact, no state); otherwise it is the native
        // evaluation of the reference state.
        let reference = if self.is_register_free() {
            self.hamiltonian
                .expectation_at(self.register_reference())
                .re
        } else {
            self.probe_with_mode(&[], &[], true, options.sparse_kernels)?
                .energy
        };
        let derivatives = !matches!(
            options.optimizer.method,
            optimizer::Method::Cobyla | optimizer::Method::NelderMead | optimizer::Method::Spsa
        );
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
        let mut pruned: Vec<(usize, String)> = Vec::new();
        let mut reused_eval: Option<CapturedPoint> = None;
        let mut screened_final = false;
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
            // The best point seen, written every `every` evaluations when checkpointing: the
            // optimizer's own iterate may be a trial step, the best is not.
            let mut best: Option<(Vec<f64>, f64)> = None;
            let mut evaluations = 0usize;
            let capture: Option<CapturedScreen> = (execution.is_none()
                && options.reuse_evaluations)
                .then(|| std::rc::Rc::new(std::cell::RefCell::new(None)));
            let mut objective = |t: &[f64]| -> Result<optimizer::Probe<'_>> {
                let v = probe_of(
                    self,
                    execution.as_deref_mut(),
                    &indices,
                    t,
                    options.trotter,
                    derivatives,
                    capture.as_ref().cloned(),
                    options.sparse_kernels,
                )?;
                if let Some(plan) = &options.checkpoint {
                    evaluations += 1;
                    if best.as_ref().is_none_or(|(_, e)| v.energy < *e) {
                        best = Some((t.to_vec(), v.energy));
                    }
                    if evaluations % plan.every == 0 {
                        let (x, e) = best.as_ref().expect("an evaluation was made");
                        write_checkpoint(
                            self,
                            plan,
                            &indices,
                            x,
                            options.trotter,
                            *e,
                            false,
                            evaluations,
                            options.optimizer.method.name(),
                            None,
                        )?;
                    }
                }
                Ok(v)
            };
            let outcome = if options.quenching {
                optimizer::minimize_interruptible_deferred_from(
                    ctx,
                    &parameters,
                    &options.optimizer,
                    None,
                    &mut objective,
                )?
            } else {
                sweep(ctx, &parameters, &options.optimizer, &mut objective)?
            };
            parameters = outcome.parameters;
            history = outcome.history;
            steps = outcome.steps;
            success = outcome.success;
            message = outcome.message;
            measured = Some(outcome.energy);
            stopped = outcome.interrupted;
            reused_eval = capture
                .as_ref()
                .and_then(|c| c.borrow_mut().take())
                .filter(|p| p.parameters == parameters);
            final_gradient = f64::NAN;
            // A measured gradient is two device evaluations per rotation: only a gradient-based
            // optimizer, which already pays for them, reports one. A derivative-free shot-based
            // run (register-free) leaves it NaN rather than spending thousands of jobs on a
            // diagnostic; a simulated one is free and keeps it.
            if stopped.is_none() && (!self.is_register_free() || derivatives) {
                if let Some(point) = reused_eval.as_ref() {
                    final_gradient = point.gradient.iter().fold(0f64, |a, g| a.max(g.abs()));
                } else {
                    let post_capture: Option<CapturedScreen> = (execution.is_none()
                        && options.reuse_evaluations)
                        .then(|| std::rc::Rc::new(std::cell::RefCell::new(None)));
                    let measured_gradient = probe_of(
                        self,
                        execution.as_deref_mut(),
                        &indices,
                        &parameters,
                        options.trotter,
                        true,
                        post_capture.as_ref().cloned(),
                        options.sparse_kernels,
                    )
                    .and_then(optimizer::Probe::into_gradient);
                    match measured_gradient {
                        Ok(gradient) => {
                            final_gradient = gradient.iter().fold(0f64, |a, g| a.max(g.abs()));
                            reused_eval = post_capture
                                .as_ref()
                                .and_then(|c| c.borrow_mut().take())
                                .filter(|p| p.parameters == parameters);
                        }
                        Err(Error::Interrupted(m)) => stopped = Some(m),
                        Err(e) => return Err(e),
                    }
                }
            }
        } else {
            if sequence.is_none() && initial.is_some_and(|p| !p.is_empty()) {
                return Err(invalid("basic ADAPT starts with an empty ansatz"));
            }
            if let Some(sequence) = sequence {
                indices = sequence.to_vec();
                parameters = initial
                    .ok_or_else(|| invalid("seeded ansatz sequence requires initial parameters"))?
                    .to_vec();
                if options.continuation {
                    curvature = options.hessian.clone().filter(|_| carries_curvature);
                } else if !indices.is_empty() {
                    let warm = options.hessian.is_some();
                    let seed_capture: Option<CapturedScreen> = (execution.is_none()
                        && options.reuse_evaluations)
                        .then(|| std::rc::Rc::new(std::cell::RefCell::new(None)));
                    let mut outcome = optimizer::minimize_deferred_from(
                        ctx,
                        &parameters,
                        &options.optimizer,
                        options.hessian.as_deref(),
                        |t| {
                            probe_of(
                                self,
                                execution.as_deref_mut(),
                                &indices,
                                t,
                                true,
                                derivatives,
                                seed_capture.as_ref().cloned(),
                                options.sparse_kernels,
                            )
                        },
                    )?;
                    let kept = outcome.energy < reference;
                    if kept && carries_curvature {
                        curvature = outcome.hessian.take();
                    }
                    seed = Some(SeedRelaxation {
                        kept,
                        energy: outcome.energy,
                        success: outcome.success,
                        message: outcome.message.clone(),
                        evaluations: outcome.history.len(),
                        steps: outcome.steps,
                        warm,
                    });
                    steps += outcome.steps;
                    history.extend(outcome.history);
                    parameters = outcome.parameters;
                    if kept {
                        reused_eval = seed_capture
                            .as_ref()
                            .and_then(|c| c.borrow_mut().take())
                            .filter(|p| p.parameters == parameters);
                    } else {
                        indices.clear();
                        parameters.clear();
                    }
                }
            }
            let mut delta = None;
            for growth in self.growth_steps(&indices)?..=options.max_growth {
                // Release the objective closure before measured pool screening.
                let (energy, reused_state, reused_hpsi) = match execution.as_mut() {
                    Some(device) => {
                        reused_eval = None;
                        match device.evaluate(self, &indices, &parameters, false) {
                            Err(Error::Interrupted(m)) if measured.is_some() => {
                                stopped = Some(m);
                                break;
                            }
                            other => {
                                let eval = other?;
                                (eval.energy, Some(eval.state), None)
                            }
                        }
                    }
                    None => {
                        if let Some(point) = reused_eval.take() {
                            (point.energy, Some(point.state), point.centered_hpsi)
                        } else {
                            let eval = self.evaluate_with_mode(
                                &indices,
                                &parameters,
                                true,
                                options.sparse_kernels,
                            )?;
                            (eval.energy, Some(eval.state), None)
                        }
                    }
                };
                measured = Some(energy);
                let mut selected = 0;
                final_gradient = 0.;
                let mut gradients = vec![0.; self.pool.len()];
                // Local screening is the first step only (the reference determinant, where the
                // gradients are exact); from the second step on the device measures them all.
                // A register-free problem reads the first step off Pauli terms, never a state.
                if execution.is_none() || indices.is_empty() {
                    let signed = if self.is_register_free() {
                        self.reference_gradients()?
                    } else {
                        let state = reused_state
                            .as_ref()
                            .expect("native screening holds a prepared state");
                        match reused_hpsi.as_ref() {
                            Some(hpsi) => {
                                self.pool_gradients_with_centered_hpsi(state, hpsi.as_slice())?
                            }
                            None => self.pool_gradients(state, true)?,
                        }
                    };
                    for (i, signed) in signed.into_iter().enumerate() {
                        let magnitude = signed.abs();
                        gradients[i] = magnitude;
                        if self.pool_screenable[i] && magnitude > final_gradient {
                            final_gradient = magnitude;
                            selected = i;
                        }
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
                    screened_final = execution.is_none();
                    if execution.is_none() && options.reuse_evaluations {
                        if let Some(state) = reused_state {
                            reused_eval = Some(CapturedPoint {
                                parameters: parameters.clone(),
                                energy,
                                gradient: Vec::new(),
                                state,
                                centered_hpsi: reused_hpsi,
                            });
                        }
                    }
                    success = true;
                    message = "pool gradients converged".into();
                    break;
                }
                if growth == options.max_growth {
                    screened_final = execution.is_none();
                    if execution.is_none() && options.reuse_evaluations {
                        if let Some(state) = reused_state {
                            reused_eval = Some(CapturedPoint {
                                parameters: parameters.clone(),
                                energy,
                                gradient: Vec::new(),
                                state,
                                centered_hpsi: reused_hpsi,
                            });
                        }
                    }
                    break;
                }
                let picks = if options.tetris {
                    disjoint_selection(
                        &self.pool,
                        &self.pool_screenable,
                        &gradients,
                        self.n_qubits(),
                    )
                } else {
                    vec![selected]
                };
                let accepted = indices.len();
                for pick in picks {
                    let members: Vec<_> = self.growth_members[pick]
                        .iter()
                        .copied()
                        .filter(|&i| gradients[i] > 1e-12)
                        .collect();
                    let grown = if members.len() >= 2 {
                        members
                    } else {
                        vec![pick]
                    };
                    for index in grown {
                        indices.push(index);
                        parameters.push(0.);
                    }
                }

                let step_capture: Option<CapturedScreen> = (execution.is_none()
                    && options.reuse_evaluations)
                    .then(|| std::rc::Rc::new(std::cell::RefCell::new(None)));
                let objective = |t: &[f64]| {
                    probe_of(
                        self,
                        execution.as_deref_mut(),
                        &indices,
                        t,
                        true,
                        derivatives,
                        step_capture.as_ref().cloned(),
                        options.sparse_kernels,
                    )
                };
                let start_curvature = curvature
                    .as_deref()
                    .filter(|h| carries_curvature && h.len() == accepted * accepted)
                    .map(|h| extend_hessian(h, accepted, parameters.len()));
                let optimized = if options.quenching {
                    optimizer::minimize_interruptible_deferred_from(
                        ctx,
                        &parameters,
                        &options.optimizer,
                        start_curvature.as_deref(),
                        objective,
                    )
                } else {
                    // Only the parameters this step appended; the earlier ones stay at their optimum.
                    minimize_block(
                        ctx,
                        &parameters,
                        accepted..parameters.len(),
                        &options.optimizer,
                        objective,
                    )
                };
                let mut outcome = match optimized {
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
                let mut step_energy = outcome.energy;
                parameters = std::mem::take(&mut outcome.parameters);
                curvature = if carries_curvature {
                    outcome.hessian.take()
                } else {
                    None
                };
                reused_eval = if outcome.interrupted.is_none() {
                    step_capture
                        .as_ref()
                        .and_then(|c| c.borrow_mut().take())
                        .filter(|p| p.parameters == parameters)
                } else {
                    None
                };
                if options.prune && outcome.interrupted.is_none() {
                    if let Some(at) = prune_candidate(&parameters) {
                        reused_eval = None;
                        let (mut shorter, mut angles) = (indices.clone(), parameters.clone());
                        shorter.remove(at);
                        angles.remove(at);
                        curvature = curvature.map(|h| {
                            if options.optimizer.method == optimizer::Method::Bfgs {
                                drop_parameter_of_inverse(&h, parameters.len(), at)
                            } else {
                                drop_parameter(&h, parameters.len(), at)
                            }
                        });
                        // The energy of the shorter ansatz at the same angles: not re-optimized.
                        // A stop before it is measured keeps the unpruned ansatz and its energy.
                        let evaluated = match execution.as_mut() {
                            Some(device) => device
                                .evaluate(self, &shorter, &angles, false)
                                .map(|eval| eval.energy),
                            None if options.reuse_evaluations => self
                                .probe_with_mode(&shorter, &angles, true, options.sparse_kernels)
                                .map(|probe| probe.energy),
                            None => self
                                .evaluate_with_mode(&shorter, &angles, true, options.sparse_kernels)
                                .map(|eval| eval.energy),
                        };
                        match evaluated {
                            Err(Error::Interrupted(m)) => {
                                measured = Some(step_energy);
                                stopped = Some(m);
                                break;
                            }
                            other => step_energy = other?,
                        }
                        pruned.push((iterations.len() + 1, self.pool[indices[at]].label.clone()));
                        indices = shorter;
                        parameters = angles;
                    }
                }
                delta = Some(step_energy - energy);
                measured = Some(step_energy);
                steps += outcome.steps;
                iterations.push(Iteration {
                    pool_index: selected,
                    energy: step_energy,
                    max_gradient: final_gradient,
                    optimizer_steps: outcome.steps,
                    num_evaluations: outcome.history.len(),
                    success: outcome.success,
                    message: outcome.message.clone(),
                    num_parameters: parameters.len(),
                });
                history.extend(outcome.history);
                if let Some(plan) = &options.checkpoint {
                    if iterations.len() % plan.every == 0 {
                        write_checkpoint(
                            self,
                            plan,
                            &indices,
                            &parameters,
                            true,
                            step_energy,
                            true,
                            iterations.len(),
                            options.optimizer.method.name(),
                            curvature.as_deref(),
                        )?;
                    }
                }
                if let Some(observer) = observer.as_deref_mut() {
                    observer(&indices, &parameters, step_energy)?;
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
                other => {
                    let eval = other?;
                    Some((eval.energy, eval.state, None))
                }
            },
            (None, false) => {
                if let Some(point) = reused_eval.take() {
                    Some((point.energy, point.state, point.centered_hpsi))
                } else {
                    let eval = self.evaluate_with_mode(
                        &indices,
                        &parameters,
                        trotter,
                        options.sparse_kernels,
                    )?;
                    Some((eval.energy, eval.state, None))
                }
            }
        };
        // After a stop no circuit is submitted: the state comes from the
        // native simulator and the energy is the last one measured.
        let (energy, state, final_hpsi) = match final_evaluation {
            Some(evaluation) => evaluation,
            None => (
                measured.expect("a stop keeps a measured point"),
                self.result_state(&indices, &parameters, trotter)?,
                None,
            ),
        };
        if let Some(reason) = &stopped {
            success = false;
            message = format!("run budget exhausted: {reason}");
        }
        if options.adapt
            && execution.is_none()
            && (!options.reuse_evaluations || !screened_final || stopped.is_some())
        {
            final_gradient = 0.;
            let signed = match final_hpsi.as_ref() {
                Some(hpsi) if options.reuse_evaluations => {
                    self.pool_gradients_with_centered_hpsi(&state, hpsi.as_slice())?
                }
                _ => self.pool_gradients(&state, true)?,
            };
            for (i, signed) in signed.into_iter().enumerate() {
                if self.pool_screenable[i] {
                    final_gradient = final_gradient.max(signed.abs());
                }
            }
        }
        let labels = indices
            .iter()
            .map(|&i| self.pool[i].label.clone())
            .collect();
        let indices_len = indices.len();
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
            pruned,
            hessian: curvature.filter(|h| h.len() == indices_len * indices_len),
            notes: Vec::new(),
            _memory,
        })
    }
}

/// Pruning's position weight `alpha` in `exp(-alpha x)`.
pub const PRUNE_ALPHA: f64 = 10.;
/// Fraction of the recent mean amplitude below which the top-ranked operator is removed.
pub const PRUNE_FRACTION: f64 = 0.1;
/// How many of the most recently added operators define that mean.
pub const PRUNE_RECENT: usize = 4;
/// No pruning at or below this many parameters: the ranking needs a position scale.
pub const PRUNE_MIN_OPERATORS: usize = 2;

/// The qubits a generator acts on, as a bit mask over its Pauli terms.
fn support(operator: &crate::quantum::Operator) -> u64 {
    operator
        .terms
        .iter()
        .filter(|t| t.coefficient.norm() > 0.)
        .fold(0, |mask, t| mask | t.x | t.z)
}

/// TETRIS selection: the screenable pool indices of one growth step, the largest gradient
/// first, then each next largest whose support touches no qubit taken so far, until the
/// register is covered or nothing disjoint with a gradient above 1e-12 is left.
fn disjoint_selection(
    pool: &[crate::quantum::algebra::Generator],
    screenable: &[bool],
    gradients: &[f64],
    n_qubits: usize,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..pool.len()).filter(|&i| screenable[i]).collect();
    order.sort_by(|&a, &b| gradients[b].total_cmp(&gradients[a]));
    let full = if n_qubits >= 64 {
        u64::MAX
    } else {
        (1u64 << n_qubits) - 1
    };
    let (mut chosen, mut covered) = (Vec::new(), 0u64);
    for i in order {
        if gradients[i] <= 1e-12 {
            break;
        }
        let mask = support(&pool[i].operator);
        if !chosen.is_empty() && mask & covered != 0 {
            continue;
        }
        chosen.push(i);
        covered |= mask;
        if covered & full == full {
            break;
        }
    }
    chosen
}

/// Pruning's candidate among optimized `parameters` (ansatz order): the position of the one to
/// remove, or `None`.
pub fn prune_candidate(parameters: &[f64]) -> Option<usize> {
    let n = parameters.len();
    if n <= PRUNE_MIN_OPERATORS {
        return None;
    }
    let recent = PRUNE_RECENT.min(n);
    let threshold = PRUNE_FRACTION
        * parameters[n - recent..]
            .iter()
            .map(|t| t.abs())
            .sum::<f64>()
        / recent as f64;
    // The newest operator is never a candidate: it was just selected on the largest gradient.
    let (candidate, _) = parameters[..n - 1]
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let x = (i + 1) as f64 / n as f64;
            (i, (-PRUNE_ALPHA * x).exp() / (t * t))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    (parameters[candidate].abs() < threshold).then_some(candidate)
}

/// One objective evaluation of a solve: a circuit executor's measurement when there is one (its
/// gradient is measured with the energy, as the executor decides, so it is ready), else the
/// native evaluation with its gradient deferred ([`QuantumProblem::probe_with_mode`]).
#[allow(clippy::too_many_arguments)]
fn probe_of<'p>(
    problem: &'p QuantumProblem,
    execution: Option<&mut (dyn Execution + '_)>,
    indices: &[usize],
    parameters: &[f64],
    trotter: bool,
    gradients: bool,
    capture: Option<CapturedScreen>,
    sparse_kernels: bool,
) -> Result<optimizer::Probe<'p>> {
    match execution {
        Some(device) => {
            let value = device.evaluate(problem, indices, parameters, gradients)?;
            Ok(optimizer::Probe::ready(value.energy, value.gradient))
        }
        None => match capture {
            Some(slot) => {
                problem.probe_capturing(indices, parameters, trotter, slot, sparse_kernels)
            }
            None => problem.probe_with_mode(indices, parameters, trotter, sparse_kernels),
        },
    }
}

/// Minimize `objective` over the coordinates `free` of `full`, the others held: the objective and
/// the outcome see the full vector, the optimizer only the free block (and its gradient slice).
fn minimize_block<'p>(
    ctx: &crate::Context,
    full: &[f64],
    free: std::ops::Range<usize>,
    options: &optimizer::Options,
    mut objective: impl FnMut(&[f64]) -> Result<optimizer::Probe<'p>>,
) -> Result<optimizer::Outcome> {
    let mut point = full.to_vec();
    let start = full[free.clone()].to_vec();
    let mut outcome =
        optimizer::minimize_interruptible_deferred_from(ctx, &start, options, None, |block| {
            point[free.clone()].copy_from_slice(block);
            let probe = objective(&point)?;
            let free = free.clone();
            let length = point.len();
            Ok(probe.map_gradient(move |gradient| {
                if gradient.len() == length {
                    gradient[free].to_vec()
                } else {
                    gradient
                }
            }))
        })?;
    let mut parameters = full.to_vec();
    parameters[free].copy_from_slice(&outcome.parameters);
    outcome.parameters = parameters;
    Ok(outcome)
}

/// One sequential sweep (`quenching = false` on a fixed ansatz): parameter `k` alone, `0..k` at
/// their optimized values and `k+1..` at their starting values. The outcome's steps and history
/// are the sums over the one-parameter searches; a stop ends the sweep at the best point.
fn sweep<'p>(
    ctx: &crate::Context,
    initial: &[f64],
    options: &optimizer::Options,
    mut objective: impl FnMut(&[f64]) -> Result<optimizer::Probe<'p>>,
) -> Result<optimizer::Outcome> {
    if initial.len() <= 1 {
        return optimizer::minimize_interruptible_deferred_from(
            ctx, initial, options, None, objective,
        );
    }
    let mut parameters = initial.to_vec();
    let (mut history, mut steps, mut success) = (Vec::new(), 0, true);
    let mut energy = f64::NAN;
    for k in 0..parameters.len() {
        let step = minimize_block(ctx, &parameters, k..k + 1, options, &mut objective)?;
        parameters = step.parameters;
        energy = step.energy;
        history.extend(step.history);
        steps += step.steps;
        success &= step.success;
        if step.interrupted.is_some() {
            return Ok(optimizer::Outcome {
                parameters,
                energy,
                history,
                steps,
                success: false,
                message: step.message,
                hessian: None,
                interrupted: step.interrupted,
            });
        }
    }
    Ok(optimizer::Outcome {
        parameters,
        energy,
        history,
        steps,
        success,
        message: "sequential sweep (quenching=False)".into(),
        hessian: None,
        interrupted: None,
    })
}
