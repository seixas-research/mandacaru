use super::{invalid, SearchOptions};
use crate::quantum::{
    amplitude::State,
    evolution, optimizer,
    search::{
        kernel::{self, Action, Move},
        learning::{
            graph::{BipartiteGraph, Candidates},
            training::{self, Model},
        },
    },
    solver::{self, Execution},
    subspace, Operator, QuantumProblem, C64,
};
use crate::{Buffer, Result};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    sync::Arc,
};

/// The pool screen of a state and the operator proposal drawn from it. `gradients` is empty when
/// nothing but the proposal reads them (a uniform chain that records nothing).
#[derive(Clone)]
pub(super) struct Screen {
    pub(super) gradients: Vec<f64>,
    pub(super) probabilities: Vec<f64>,
    pub(super) curvatures: Option<Vec<f64>>,
}
/// The position weights of a state's deletions and replacements.
#[derive(Clone)]
pub(super) struct Weights {
    /// Each operator's importance: the |gradient| it would have if re-inserted at its own position
    /// (its angle zero, the other angles of this state).
    pub(super) importances: Vec<f64>,
    /// The deletion and replacement position weights of [`kernel::removal`].
    pub(super) removal: Vec<f64>,
}
/// A relaxed state and, once something needs them, its two analyses. The relaxation is what the
/// Metropolis test reads (the cost); the screen and the weights are the proposal distribution
/// drawn from the state. A proposal's reverse probability reads only the half of the distribution
/// that its move kind uses, and a rejected proposal's state is never drawn from, so each half is
/// computed when something first needs it ([`Evaluator::screen`], [`Evaluator::weigh`]) and a state
/// becomes the chain's current state only complete ([`Evaluator::complete`]).
#[derive(Clone)]
pub(super) struct Evaluated {
    pub(super) sequence: Vec<usize>,
    pub(super) parameters: Vec<f64>,
    pub(super) energy: f64,
    pub(super) cost: f64,
    pub(super) screen: Option<Arc<Screen>>,
    pub(super) weights: Option<Arc<Weights>>,
    pub(super) hessian: Option<Vec<f64>>,
    pub(super) success: bool,
    pub(super) message: String,
}
impl Evaluated {
    pub(super) fn screened(&self) -> Result<&Screen> {
        self.screen
            .as_deref()
            .ok_or_else(|| invalid("a search state was used before its pool screen"))
    }
    pub(super) fn weighed(&self) -> Result<&Weights> {
        self.weights
            .as_deref()
            .ok_or_else(|| invalid("a search state was used before its removal weights"))
    }
}

#[derive(Clone, Copy)]
pub(super) struct Ensemble<'a> {
    pub(super) weights: &'a [f64],
    pub(super) ref_rows: &'a [usize],
}

pub(super) struct ScreenCache {
    pub(super) states: Vec<State>,
    pub(super) hpsis: Vec<Option<Buffer<C64>>>,
}

pub(super) fn parameter_bits(parameters: &[f64]) -> Vec<u64> {
    parameters.iter().map(|&p| p.to_bits()).collect()
}

pub(super) fn adapt_hessian(
    action: Action,
    hessian: &[f64],
    old_len: usize,
    method: optimizer::Method,
    replace_inherit: bool,
) -> Option<Vec<f64>> {
    if hessian.len() != old_len * old_len {
        return None;
    }
    match action.kind {
        Move::Insert => Some(solver::insert_hessian(hessian, old_len, action.position)),
        Move::Delete => {
            if old_len <= 1 {
                Some(vec![])
            } else if method == optimizer::Method::Bfgs {
                Some(solver::drop_parameter_of_inverse(
                    hessian,
                    old_len,
                    action.position,
                ))
            } else {
                Some(solver::drop_parameter(hessian, old_len, action.position))
            }
        }
        Move::Replace => {
            if old_len == 0 {
                None
            } else if replace_inherit {
                Some(hessian.to_vec())
            } else {
                Some(solver::reset_hessian_parameter(
                    hessian,
                    old_len,
                    action.position,
                ))
            }
        }
    }
}

pub(super) struct WorkerDelta {
    pub(super) cache: BTreeMap<Vec<usize>, Evaluated>,
    pub(super) warm_cache: BTreeMap<(Vec<usize>, Vec<u64>), Evaluated>,
    pub(super) seen: BTreeSet<Vec<usize>>,
    pub(super) history: Vec<f64>,
    pub(super) optimizer_steps: usize,
    pub(super) screenings: usize,
    pub(super) importance_evaluations: usize,
    pub(super) importance_work: evolution::Work,
    pub(super) gradient_evaluations: usize,
    pub(super) failures: Vec<(usize, String)>,
}

pub(super) struct Evaluator<'a> {
    pub(super) problem: &'a QuantumProblem,
    pub(super) options: &'a SearchOptions,
    pub(super) learned: Option<&'a Model>,
    pub(super) graph: Option<&'a BipartiteGraph>,
    pub(super) screen_hamiltonian: Option<&'a Operator>,
    pub(super) ensemble: Option<Ensemble<'a>>,
    pub(super) prior_cache: Option<&'a BTreeMap<Vec<usize>, Evaluated>>,
    pub(super) prior_warm: Option<&'a BTreeMap<(Vec<usize>, Vec<u64>), Evaluated>>,
    pub(super) cache: BTreeMap<Vec<usize>, Evaluated>,
    pub(super) warm_cache: BTreeMap<(Vec<usize>, Vec<u64>), Evaluated>,
    pub(super) seen: BTreeSet<Vec<usize>>,
    pub(super) history: Vec<f64>,
    pub(super) optimizer_steps: usize,
    pub(super) screenings: usize,
    pub(super) importance_evaluations: usize,
    pub(super) importance_work: evolution::Work,
    pub(super) gradient_evaluations: usize,
    pub(super) failures: Vec<(usize, String)>,
}
impl<'a> Evaluator<'a> {
    pub(super) fn new(
        problem: &'a QuantumProblem,
        options: &'a SearchOptions,
        learned: Option<&'a Model>,
        graph: Option<&'a BipartiteGraph>,
        screen_hamiltonian: Option<&'a Operator>,
        ensemble: Option<Ensemble<'a>>,
    ) -> Self {
        Self {
            problem,
            options,
            learned,
            graph,
            screen_hamiltonian,
            ensemble,
            prior_cache: None,
            prior_warm: None,
            cache: BTreeMap::new(),
            warm_cache: BTreeMap::new(),
            seen: BTreeSet::new(),
            history: vec![],
            optimizer_steps: 0,
            screenings: 0,
            importance_evaluations: 0,
            importance_work: evolution::Work::default(),
            gradient_evaluations: 0,
            failures: vec![],
        }
    }
    pub(super) fn fork_readonly(&self) -> Evaluator<'_> {
        Evaluator {
            problem: self.problem,
            options: self.options,
            learned: self.learned,
            graph: self.graph,
            screen_hamiltonian: self.screen_hamiltonian,
            ensemble: self.ensemble,
            prior_cache: Some(&self.cache),
            prior_warm: Some(&self.warm_cache),
            cache: BTreeMap::new(),
            warm_cache: BTreeMap::new(),
            seen: BTreeSet::new(),
            history: vec![],
            optimizer_steps: 0,
            screenings: 0,
            importance_evaluations: 0,
            importance_work: evolution::Work::default(),
            gradient_evaluations: 0,
            failures: vec![],
        }
    }
    pub(super) fn into_delta(self) -> WorkerDelta {
        WorkerDelta {
            cache: self.cache,
            warm_cache: self.warm_cache,
            seen: self.seen,
            history: self.history,
            optimizer_steps: self.optimizer_steps,
            screenings: self.screenings,
            importance_evaluations: self.importance_evaluations,
            importance_work: self.importance_work,
            gradient_evaluations: self.gradient_evaluations,
            failures: self.failures,
        }
    }
    pub(super) fn merge_delta(&mut self, worker: WorkerDelta) {
        for (k, v) in worker.cache {
            self.cache.entry(k).or_insert(v);
        }
        for (k, v) in worker.warm_cache {
            self.warm_cache.entry(k).or_insert(v);
        }
        self.seen.extend(worker.seen);
        self.history.extend(worker.history);
        self.optimizer_steps += worker.optimizer_steps;
        self.screenings += worker.screenings;
        self.importance_evaluations += worker.importance_evaluations;
        self.importance_work += worker.importance_work;
        self.gradient_evaluations += worker.gradient_evaluations;
        self.failures.extend(worker.failures);
    }
    /// The relaxed state of `sequence`, without its analyses unless the memo holds them, plus
    /// the converged `(state, centered_hpsi)` pair when available for direct screening reuse.
    pub(super) fn evaluate(
        &mut self,
        sequence: Vec<usize>,
        initial: Vec<f64>,
        start_hessian: Option<&[f64]>,
        step: usize,
        execution: Option<&mut (dyn Execution + '_)>,
    ) -> Result<(Evaluated, Option<ScreenCache>)> {
        if !self.seen.contains(&sequence) {
            self.seen.insert(sequence.clone());
        }
        if !self.options.warm_start {
            if let Some(cached) = self
                .cache
                .get(&sequence)
                .or_else(|| self.prior_cache.and_then(|c| c.get(&sequence)))
            {
                return Ok((cached.clone(), None));
            }
        } else if self.options.memoize_warm && start_hessian.is_none() {
            let key = (sequence.clone(), parameter_bits(&initial));
            if let Some(cached) = self
                .warm_cache
                .get(&key)
                .or_else(|| self.prior_warm.and_then(|c| c.get(&key)))
            {
                return Ok((cached.clone(), None));
            }
            let (result, screen_cache) =
                self.relax(sequence, initial, start_hessian, step, execution)?;
            self.warm_cache.insert(key, result.clone());
            return Ok((result, screen_cache));
        }
        let (result, screen_cache) =
            self.relax(sequence, initial, start_hessian, step, execution)?;
        if !self.options.warm_start {
            self.cache.insert(result.sequence.clone(), result.clone());
        }
        Ok((result, screen_cache))
    }
    /// Write the analyses of `state` into the memo entry of its sequence, when that entry is this
    /// very state (an uncached transferred start shares the sequence, not the angles).
    fn store(&mut self, state: &Evaluated, initial_bits: Option<&[u64]>) {
        if !self.options.warm_start {
            if let Some(entry) = self.cache.get_mut(&state.sequence) {
                if entry.parameters == state.parameters {
                    entry.screen = state.screen.clone();
                    entry.weights = state.weights.clone();
                }
            }
        } else if self.options.memoize_warm {
            if let Some(bits) = initial_bits {
                let key = (state.sequence.clone(), bits.to_vec());
                if let Some(entry) = self.warm_cache.get_mut(&key) {
                    if entry.parameters == state.parameters {
                        entry.screen = state.screen.clone();
                        entry.weights = state.weights.clone();
                    }
                }
            }
        }
    }
    /// Relax `sequence` from `initial` without reading or writing the cache: a start that is not
    /// the zero start is not a function of the sequence, so it must never be stored as one.
    pub(super) fn evaluate_uncached(
        &mut self,
        sequence: Vec<usize>,
        initial: Vec<f64>,
        step: usize,
        execution: Option<&mut (dyn Execution + '_)>,
    ) -> Result<(Evaluated, Option<ScreenCache>)> {
        self.seen.insert(sequence.clone());
        self.relax(sequence, initial, None, step, execution)
    }
    fn relax(
        &mut self,
        sequence: Vec<usize>,
        initial: Vec<f64>,
        start_hessian: Option<&[f64]>,
        step: usize,
        execution: Option<&mut (dyn Execution + '_)>,
    ) -> Result<(Evaluated, Option<ScreenCache>)> {
        let mut screen_cache = None;
        let (parameters, energy, hessian, success, message) = if sequence.is_empty() {
            let energy = match (execution, &self.ensemble) {
                (Some(exec), None) => exec.evaluate(self.problem, &[], &[], false)?.energy,
                (Some(_), Some(_)) => {
                    return Err(invalid(
                        "subspace search uses local state-vector simulation",
                    ));
                }
                (None, Some(ens)) => {
                    self.problem
                        .subspace_probe_capturing(
                            &[],
                            &[],
                            true,
                            ens.weights,
                            ens.ref_rows,
                            None,
                            self.options.sparse_kernels,
                        )?
                        .energy
                }
                (None, None) => {
                    self.problem
                        .probe_with_mode(&[], &[], true, self.options.sparse_kernels)?
                        .energy
                }
            };
            (vec![], energy, None, true, "empty reference".into())
        } else if let Some(exec) = execution {
            let carries = self.options.carry_hessian
                && self.options.warm_start
                && self.options.optimizer.method.carries_curvature();
            let mut gradient_count = 0usize;
            let result = optimizer::minimize_deferred_from(
                &self.problem.context,
                &initial,
                &self.options.optimizer,
                start_hessian.filter(|_| carries),
                |p| {
                    let eval = exec.evaluate(self.problem, &sequence, p, true)?;
                    gradient_count += 1;
                    Ok(optimizer::Probe::ready(eval.energy, eval.gradient))
                },
            )?;
            self.gradient_evaluations += gradient_count;
            self.history.extend(result.history);
            self.optimizer_steps += result.steps;
            if !result.success {
                self.failures.push((step, result.message.clone()));
            }
            (
                result.parameters,
                result.energy,
                result.hessian.filter(|_| carries),
                result.success,
                result.message,
            )
        } else {
            let gradients = std::cell::Cell::new(0);
            let single_capture: solver::CapturedScreen = Rc::new(RefCell::new(None));
            let subspace_capture: subspace::CapturedSubspace = Rc::new(RefCell::new(None));
            let carries = self.options.carry_hessian
                && self.options.warm_start
                && self.options.optimizer.method.carries_curvature();
            let sparse_kernels = self.options.sparse_kernels;
            let result = optimizer::minimize_deferred_from(
                &self.problem.context,
                &initial,
                &self.options.optimizer,
                start_hessian.filter(|_| carries),
                |p| {
                    // A rejected trial of the line search never asks for its gradient.
                    let probe = match &self.ensemble {
                        Some(ens) => self.problem.subspace_probe_capturing(
                            &sequence,
                            p,
                            true,
                            ens.weights,
                            ens.ref_rows,
                            Some(Rc::clone(&subspace_capture)),
                            sparse_kernels,
                        )?,
                        None => self.problem.probe_capturing(
                            &sequence,
                            p,
                            true,
                            Rc::clone(&single_capture),
                            sparse_kernels,
                        )?,
                    };
                    let counter = &gradients;
                    Ok(optimizer::Probe::deferred(probe.energy, move || {
                        counter.set(counter.get() + 1);
                        probe.into_gradient()
                    }))
                },
            )?;
            self.gradient_evaluations += gradients.get();
            self.history.extend(result.history);
            self.optimizer_steps += result.steps;
            if !result.success {
                self.failures.push((step, result.message.clone()));
            }
            if self.ensemble.is_some() {
                if let Some(point) = subspace_capture.borrow_mut().take() {
                    if point.parameters == result.parameters {
                        screen_cache = Some(ScreenCache {
                            states: point.states,
                            hpsis: point.hpsis,
                        });
                    }
                }
            } else if let Some(point) = single_capture.borrow_mut().take() {
                if point.parameters == result.parameters {
                    if let Some(hpsi) = point.centered_hpsi {
                        screen_cache = Some(ScreenCache {
                            states: vec![point.state],
                            hpsis: vec![Some(hpsi)],
                        });
                    }
                }
            }
            (
                result.parameters,
                result.energy,
                result.hessian.filter(|_| carries),
                result.success,
                result.message,
            )
        };
        Ok((
            Evaluated {
                cost: energy + self.options.length_penalty * sequence.len() as f64,
                sequence,
                parameters,
                energy,
                screen: None,
                weights: None,
                hessian,
                success,
                message,
            },
            screen_cache,
        ))
    }
    /// The pool screen and the operator proposal of `state` (a no-op once computed), optionally
    /// reusing the `(state, centered_hpsi)` captured from the relaxation's final accepted step.
    pub(super) fn screen(
        &mut self,
        state: &mut Evaluated,
        precomputed: Option<&ScreenCache>,
        initial_bits: Option<&[u64]>,
        execution: Option<&mut (dyn Execution + '_)>,
    ) -> Result<()> {
        if state.screen.is_some() {
            return Ok(());
        }
        // The pool screen feeds the gradient proposal, a ready learned model, the recorded rows and
        // the final gradient (computed once at the end when no state screens it): a uniform chain
        // that records nothing needs none of it per proposal.
        let screen = !self.options.uniform
            || self.options.record
            || (self.learned.is_some_and(|m| m.ready) && self.graph.is_some());
        let wants_curvatures = self.ensemble.is_none()
            && (self.options.record_curvatures
                || self.learned.is_some_and(|m| m.ready && m.uses_curvatures()));
        let is_local = execution.is_none();
        let gradients = if screen {
            self.screenings += 1;
            if let Some(exec) = execution {
                exec.pool_gradients(self.problem, &state.sequence, &state.parameters)?
            } else {
                match (&self.ensemble, self.screen_hamiltonian) {
                    (Some(ens), Some(h_surr)) => {
                        let mut acc = vec![0.; self.problem.pool.len()];
                        for (&w, &ref_row) in ens.weights.iter().zip(ens.ref_rows) {
                            let prepared = self.problem.subspace_prepare_state(
                                &state.sequence,
                                &state.parameters,
                                true,
                                ref_row,
                                self.options.sparse_kernels,
                            )?;
                            for (slot, g) in acc
                                .iter_mut()
                                .zip(self.problem.pool_gradients_for(&prepared, true, h_surr)?)
                            {
                                *slot += w * g;
                            }
                        }
                        acc
                    }
                    (Some(ens), None) => {
                        let mut acc = vec![0.; self.problem.pool.len()];
                        for (j, (&w, &ref_row)) in ens.weights.iter().zip(ens.ref_rows).enumerate()
                        {
                            let cached_pair = precomputed
                                .and_then(|c| Some((c.states.get(j)?, c.hpsis.get(j)?.as_ref())));
                            let g_vec = match cached_pair {
                                Some((st, Some(hpsi))) => self
                                    .problem
                                    .pool_gradients_with_centered_hpsi(st, hpsi.as_slice())?,
                                Some((st, None)) => self.problem.pool_gradients(st, true)?,
                                None => {
                                    let prepared = self.problem.subspace_prepare_state(
                                        &state.sequence,
                                        &state.parameters,
                                        true,
                                        ref_row,
                                        self.options.sparse_kernels,
                                    )?;
                                    self.problem.pool_gradients(&prepared, true)?
                                }
                            };
                            for (slot, g) in acc.iter_mut().zip(g_vec) {
                                *slot += w * g;
                            }
                        }
                        acc
                    }
                    (None, Some(h_surr)) => {
                        let prepared =
                            self.problem
                                .prepare_state(&state.sequence, &state.parameters, true)?;
                        self.problem.pool_gradients_for(&prepared, true, h_surr)?
                    }
                    (None, None) => {
                        let cached = precomputed
                            .and_then(|c| Some((c.states.first()?, c.hpsis.first()?.as_ref()?)));
                        if let Some((cached_state, cached_hpsi)) = cached {
                            self.problem.pool_gradients_with_centered_hpsi(
                                cached_state,
                                cached_hpsi.as_slice(),
                            )?
                        } else {
                            let prepared = self.problem.prepare_state(
                                &state.sequence,
                                &state.parameters,
                                true,
                            )?;
                            self.problem.pool_gradients(&prepared, true)?
                        }
                    }
                }
            }
        } else {
            vec![]
        };
        let curvatures = if wants_curvatures && is_local {
            let prepared = self
                .problem
                .prepare_state(&state.sequence, &state.parameters, true)?;
            Some(self.problem.pool_curvatures(&prepared)?)
        } else {
            None
        };
        let mut probabilities = if self.options.uniform {
            let pool = self.problem.pool.len();
            vec![1. / pool as f64; pool]
        } else {
            kernel::softmax(&gradients, self.options.proposal_temperature)?
        };
        if let (Some(model), Some(graph)) = (self.learned, &self.graph) {
            if model.ready && !gradients.is_empty() {
                let candidates: Vec<_> = (0..gradients.len()).collect();
                let rows = Candidates::new(
                    &self.problem.context,
                    graph,
                    &state.sequence,
                    &candidates,
                    &gradients,
                    state.energy - graph.reference_energy,
                )?;
                let scores = model.scores_with_curvatures(
                    &self.problem.context,
                    graph,
                    &rows,
                    curvatures.as_deref(),
                )?;
                let scores: Vec<_> = scores.iter().map(|v| v / model.temperature).collect();
                if scores.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("learned proposal temperature overflows scores"));
                }
                // A ready model draws from its own proposal alone: the gradient enters only the
                // first stage of its training (owner, 2026-11-09).
                probabilities = training::softmax(&scores);
                // Every operator must keep a positive probability, or an insertion loses its reverse
                // deletion and the Metropolis-Hastings ratio is no longer exact.
                if probabilities.iter().any(|p| *p <= 0.) {
                    return Err(invalid(
                        "the learned proposal gives an operator zero probability (its scores span more \
                         than the float range at this temperature): raise the model's temperature",
                    ));
                }
            }
        }
        state.screen = Some(Arc::new(Screen {
            gradients,
            probabilities,
            curvatures,
        }));
        self.store(state, initial_bits);
        Ok(())
    }
    /// The importances and removal weights of `state` (a no-op once computed).
    pub(super) fn weigh(
        &mut self,
        state: &mut Evaluated,
        initial_bits: Option<&[u64]>,
        mut execution: Option<&mut (dyn Execution + '_)>,
    ) -> Result<()> {
        if state.weights.is_some() {
            return Ok(());
        }
        if self.learned.is_some_and(|m| m.ready && m.uses_delete())
            && state.screen.is_none()
            && !state.sequence.is_empty()
        {
            self.screen(state, None, initial_bits, execution.as_deref_mut())?;
        }
        let (importances, work, count) = if let Some(exec) = execution {
            let mut importances = vec![0.; state.sequence.len()];
            for i in 0..state.sequence.len() {
                let mut zeroed = state.parameters.clone();
                zeroed[i] = 0.;
                let eval = exec.evaluate(self.problem, &state.sequence, &zeroed, true)?;
                importances[i] = eval.gradient.get(i).copied().unwrap_or(0.).abs();
            }
            (
                importances,
                evolution::Work::default(),
                state.sequence.len(),
            )
        } else {
            match &self.ensemble {
                Some(ens) => {
                    let mut acc = vec![0.; state.sequence.len()];
                    let mut total_work = evolution::Work::default();
                    for (&w, &ref_row) in ens.weights.iter().zip(ens.ref_rows) {
                        let measured = self.problem.importances_from_row(
                            &state.sequence,
                            &state.parameters,
                            ref_row,
                            self.options.sparse_kernels,
                        )?;
                        for (slot, v) in acc.iter_mut().zip(measured.values) {
                            *slot += w * v;
                        }
                        total_work += measured.work;
                    }
                    (acc, total_work, ens.weights.len() * state.sequence.len())
                }
                None => {
                    let measured = self.problem.importances_with_mode(
                        &state.sequence,
                        &state.parameters,
                        self.options.sparse_kernels,
                    )?;
                    (measured.values, measured.work, state.sequence.len())
                }
            }
        };
        self.importance_evaluations += count;
        self.importance_work += work;
        let mut removal = kernel::removal(&importances, self.options.proposal_temperature)?;
        if let (Some(model), Some(graph), Some(screen)) =
            (self.learned, &self.graph, state.screen.as_ref())
        {
            if model.ready && !state.sequence.is_empty() && !screen.gradients.is_empty() {
                if let Some(rem_scores) = model.removal_scores(
                    &self.problem.context,
                    graph,
                    &state.sequence,
                    &screen.gradients,
                    screen.curvatures.as_deref(),
                    state.energy - graph.reference_energy,
                )? {
                    let scaled: Vec<_> = rem_scores.iter().map(|v| v / model.temperature).collect();
                    if scaled.iter().any(|v| !v.is_finite()) {
                        return Err(invalid("learned removal temperature overflows scores"));
                    }
                    let learned_removal = training::softmax(&scaled);
                    if learned_removal.iter().any(|p| *p <= 0.) {
                        return Err(invalid(
                            "the learned deletion proposal gives a position zero probability: raise the model's temperature",
                        ));
                    }
                    removal = learned_removal;
                }
            }
        }
        state.weights = Some(Arc::new(Weights {
            importances,
            removal,
        }));
        self.store(state, initial_bits);
        Ok(())
    }
    /// Both analyses: what a state needs to be drawn from, so to become the current state.
    pub(super) fn complete(
        &mut self,
        state: &mut Evaluated,
        precomputed: Option<&ScreenCache>,
        initial_bits: Option<&[u64]>,
        mut execution: Option<&mut (dyn Execution + '_)>,
    ) -> Result<()> {
        self.screen(state, precomputed, initial_bits, execution.as_deref_mut())?;
        self.weigh(state, initial_bits, execution)
    }
}
