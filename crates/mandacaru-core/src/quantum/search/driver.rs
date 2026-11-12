//! Quenched Metropolis–Hastings search over variational operator sequences.
use super::kernel::{self, Action, Kernel, Move};
use crate::quantum::evolution;
use super::learning::{
    graph::{Candidates, Graph},
    training::{self, Model},
};
use crate::{
    math::random::Rng,
    product,
    quantum::{optimizer, CachedProblem, SolveResult},
    Context, Error, Reservation, Result,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};
fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
#[derive(Clone, Debug)]
pub struct Options {
    pub max_steps: usize,
    pub transfer_steps: Option<usize>,
    pub max_length: usize,
    pub move_weights: [f64; 3],
    pub temperatures: [f64; 2],
    pub length_penalty: f64,
    pub proposal_temperature: f64,
    pub uniform: bool,
    pub warm_start: bool,
    pub replace_inherit: bool,
    pub screen_insertions: usize,
    pub record: bool,
    pub seed: u64,
    pub optimizer: optimizer::Options,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_steps: 200,
            transfer_steps: None,
            max_length: 20,
            move_weights: [1.; 3],
            temperatures: [0.1 / 27.211386245988, 0.001 / 27.211386245988],
            length_penalty: 0.,
            proposal_temperature: 0.1,
            uniform: false,
            warm_start: true,
            replace_inherit: false,
            screen_insertions: 0,
            record: false,
            seed: 0,
            optimizer: optimizer::Options::default(),
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        self.optimizer.validate()?;
        Kernel::new(2, self.max_length, self.move_weights)?;
        if self.temperatures.iter().any(|v| !v.is_finite() || *v < 0.)
            || (self.temperatures[0] != self.temperatures[1] && self.temperatures.contains(&0.))
            || !self.length_penalty.is_finite()
            || self.length_penalty < 0.
            || !self.proposal_temperature.is_finite()
            || self.proposal_temperature <= 0.
            || (self.screen_insertions > 0 && !self.record)
        {
            return Err(invalid("search needs finite nonnegative temperatures and length penalty, positive proposal temperature, and recording for screened insertions"));
        }
        Ok(())
    }
    fn temperature(&self, step: usize, steps: usize) -> f64 {
        if steps <= 1 || self.temperatures[0] == self.temperatures[1] {
            return self.temperatures[0];
        }
        self.temperatures[0]
            * (self.temperatures[1] / self.temperatures[0]).powf(step as f64 / (steps - 1) as f64)
    }
}
/// The pool screen of a state and the operator proposal drawn from it. `gradients` is empty when
/// nothing but the proposal reads them (a uniform chain that records nothing).
#[derive(Clone)]
struct Screen {
    gradients: Vec<f64>,
    probabilities: Vec<f64>,
}
/// The position weights of a state's deletions and replacements.
#[derive(Clone)]
struct Weights {
    /// Each operator's importance: the |gradient| it would have if re-inserted at its own position
    /// (its angle zero, the other angles of this state).
    importances: Vec<f64>,
    /// The deletion and replacement position weights of [`kernel::removal`].
    removal: Vec<f64>,
}
/// A relaxed state and, once something needs them, its two analyses. The relaxation is what the
/// Metropolis test reads (the cost); the screen and the weights are the proposal distribution
/// drawn from the state. A proposal's reverse probability reads only the half of the distribution
/// that its move kind uses, and a rejected proposal's state is never drawn from, so each half is
/// computed when something first needs it ([`Evaluator::screen`], [`Evaluator::weigh`]) and a state
/// becomes the chain's current state only complete ([`Evaluator::complete`]).
#[derive(Clone)]
struct Evaluated {
    sequence: Vec<usize>,
    parameters: Vec<f64>,
    energy: f64,
    cost: f64,
    screen: Option<Screen>,
    weights: Option<Weights>,
    success: bool,
    message: String,
}
impl Evaluated {
    fn screened(&self) -> Result<&Screen> {
        self.screen
            .as_ref()
            .ok_or_else(|| invalid("a search state was used before its pool screen"))
    }
    fn weighed(&self) -> Result<&Weights> {
        self.weights
            .as_ref()
            .ok_or_else(|| invalid("a search state was used before its removal weights"))
    }
}
pub struct SearchResult {
    pub solution: SolveResult,
    pub records: Vec<Value>,
    pub steps: Vec<Value>,
    pub acceptance: [(usize, usize); 3],
    pub optimal_cost: f64,
    pub best_energy: f64,
    pub best_energy_indices: Vec<usize>,
    pub best_energy_parameters: Vec<f64>,
    pub final_energy: f64,
    pub final_indices: Vec<usize>,
    pub final_parameters: Vec<f64>,
    pub num_architectures: usize,
    pub num_screenings: usize,
    /// Operator importances computed (one per position of a weighed state), not in `history`.
    /// Each is the gradient entry of one evaluation with that angle zeroed, taken from the
    /// shared prefix and the truncated sweep of [`evolution::importances`], not from a full
    /// evaluation.
    pub num_importance_evaluations: usize,
    /// What the importances did in whole-state actions, against
    /// [`evolution::Work::separate_evaluations`] per weighed state (one full evaluation each).
    pub importance_work: evolution::Work,
    /// Gradients the optimizer asked for: the others of the `history` entries (one energy per
    /// objective call) were trial points its line search rejected on the energy alone.
    pub num_gradient_evaluations: usize,
    pub failures: Vec<(usize, String)>,
    _memory: Reservation,
}
struct Evaluator<'a> {
    problem: &'a CachedProblem,
    options: &'a Options,
    learned: Option<&'a Model>,
    graph: Option<Graph>,
    cache: BTreeMap<Vec<usize>, Evaluated>,
    seen: BTreeSet<Vec<usize>>,
    history: Vec<f64>,
    optimizer_steps: usize,
    screenings: usize,
    importance_evaluations: usize,
    importance_work: evolution::Work,
    gradient_evaluations: usize,
    failures: Vec<(usize, String)>,
}
impl Evaluator<'_> {
    /// The relaxed state of `sequence`, without its analyses unless the memo holds them. With
    /// `warm_start=false` the relaxation is a function of the sequence alone (it starts from
    /// zeros), so the state is memoized, and every analysis computed later is stored with it.
    fn evaluate(
        &mut self,
        sequence: Vec<usize>,
        initial: Vec<f64>,
        step: usize,
    ) -> Result<Evaluated> {
        self.seen.insert(sequence.clone());
        if !self.options.warm_start {
            if let Some(cached) = self.cache.get(&sequence) {
                return Ok(cached.clone());
            }
        }
        let result = self.relax(sequence, initial, step)?;
        if !self.options.warm_start {
            self.cache.insert(result.sequence.clone(), result.clone());
        }
        Ok(result)
    }
    /// Write the analyses of `state` into the memo entry of its sequence, when that entry is this
    /// very state (an uncached transferred start shares the sequence, not the angles).
    fn store(&mut self, state: &Evaluated) {
        if self.options.warm_start {
            return;
        }
        if let Some(entry) = self.cache.get_mut(&state.sequence) {
            if entry.parameters == state.parameters {
                entry.screen = state.screen.clone();
                entry.weights = state.weights.clone();
            }
        }
    }
    /// Relax `sequence` from `initial` without reading or writing the cache: a start that is not
    /// the zero start is not a function of the sequence, so it must never be stored as one.
    fn evaluate_uncached(
        &mut self,
        sequence: Vec<usize>,
        initial: Vec<f64>,
        step: usize,
    ) -> Result<Evaluated> {
        self.seen.insert(sequence.clone());
        self.relax(sequence, initial, step)
    }
    fn relax(
        &mut self,
        sequence: Vec<usize>,
        initial: Vec<f64>,
        step: usize,
    ) -> Result<Evaluated> {
        let (parameters, energy, success, message) = if sequence.is_empty() {
            (
                vec![],
                self.problem.evaluate(&[], &[], true)?.energy,
                true,
                "empty reference".into(),
            )
        } else {
            let gradients = std::cell::Cell::new(0);
            let result = optimizer::minimize_deferred_from(
                &self.problem.context,
                &initial,
                &self.options.optimizer,
                None,
                |p| {
                    // A rejected trial of the line search never asks for its gradient.
                    let probe = self.problem.probe(&sequence, p, true)?;
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
            (
                result.parameters,
                result.energy,
                result.success,
                result.message,
            )
        };
        Ok(Evaluated {
            cost: energy + self.options.length_penalty * sequence.len() as f64,
            sequence,
            parameters,
            energy,
            screen: None,
            weights: None,
            success,
            message,
        })
    }
    /// The pool screen and the operator proposal of `state` (a no-op once computed).
    fn screen(&mut self, state: &mut Evaluated) -> Result<()> {
        if state.screen.is_some() {
            return Ok(());
        }
        // The pool screen feeds the gradient proposal, a ready learned model, the recorded rows and
        // the final gradient (computed once at the end when no state screens it): a uniform chain
        // that records nothing needs none of it per proposal.
        let screen = !self.options.uniform
            || self.options.record
            || (self.learned.is_some_and(|m| m.ready) && self.graph.is_some());
        let gradients = if screen {
            let prepared = self
                .problem
                .prepare_state(&state.sequence, &state.parameters, true)?;
            self.screenings += 1;
            self.problem.pool_gradients(&prepared, true)?
        } else {
            vec![]
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
                let scores = model.scores(&self.problem.context, graph, &rows)?;
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
        state.screen = Some(Screen {
            gradients,
            probabilities,
        });
        self.store(state);
        Ok(())
    }
    /// The importances and removal weights of `state` (a no-op once computed).
    fn weigh(&mut self, state: &mut Evaluated) -> Result<()> {
        if state.weights.is_some() {
            return Ok(());
        }
        let measured = self.problem.importances(&state.sequence, &state.parameters)?;
        self.importance_evaluations += state.sequence.len();
        self.importance_work += measured.work;
        let importances = measured.values;
        let removal = kernel::removal(&importances, self.options.proposal_temperature)?;
        state.weights = Some(Weights {
            importances,
            removal,
        });
        self.store(state);
        Ok(())
    }
    /// Both analyses: what a state needs to be drawn from, so to become the current state.
    fn complete(&mut self, state: &mut Evaluated) -> Result<()> {
        self.screen(state)?;
        self.weigh(state)
    }
}
fn record(
    step: usize,
    action: Action,
    source: &Evaluated,
    proposed: &Evaluated,
    temperature: f64,
    outcome: Option<(f64, f64, f64, bool)>,
    vectors: bool,
) -> Result<Value> {
    let mut row = json!({"step":step,"move":action.kind.name(),"position":action.position,
        "operator":if matches!(action.kind,Move::Insert|Move::Replace) {Some(action.operator)} else {None},
        "source":source.sequence,"proposed":proposed.sequence,"source_parameters":source.parameters,"parameters":proposed.parameters,
        "source_energy":source.energy,"energy":proposed.energy,"source_cost":source.cost,"cost":proposed.cost,
        "delta_energy":proposed.energy-source.energy,"delta_cost":proposed.cost-source.cost,
        "temperature":temperature,"log_q_forward":outcome.map(|v|v.0),"log_q_reverse":outcome.map(|v|v.1),
        "log_acceptance":outcome.map(|v|v.2),"accepted":outcome.map(|v|v.3),"screened":outcome.is_none(),
        "optimizer_success":proposed.success,"optimizer_message":proposed.message});
    // The four source vectors (pool and sequence long) are the recorded edit's training data: a
    // step row of a chain that records nothing does not copy them.
    if vectors {
        let (screen, weights) = (source.screened()?, source.weighed()?);
        row["source_gradients"] = json!(screen.gradients);
        row["source_probabilities"] = json!(screen.probabilities);
        row["source_importances"] = json!(weights.importances);
        row["source_removal"] = json!(weights.removal);
    }
    Ok(row)
}
impl CachedProblem {
    pub fn search(
        &self,
        options: &Options,
        initial_indices: &[usize],
        initial_parameters: &[f64],
    ) -> Result<SearchResult> {
        self.search_with_model(options, initial_indices, initial_parameters, None)
    }
    pub fn search_with_model(
        &self,
        options: &Options,
        initial_indices: &[usize],
        initial_parameters: &[f64],
        learned: Option<&Model>,
    ) -> Result<SearchResult> {
        options.validate()?;
        if self.hva.is_some()
            || initial_indices.len() != initial_parameters.len()
            || initial_indices.len() > options.max_length
            || initial_indices.iter().any(|i| *i >= self.pool.len())
            || initial_parameters.iter().any(|p| !p.is_finite())
        {
            return Err(invalid(
                "search needs a valid excitation sequence and finite initial parameters",
            ));
        }
        self.context.run(|| {
            let started = Instant::now();
            let ctx: &Context = &self.context;
            let evaluations = options.max_steps.max(options.transfer_steps.unwrap_or(0)).checked_mul(options.screen_insertions.min(self.pool.len())+1).and_then(|v|v.checked_add(3)).ok_or(Error::Overflow)?;
            let memory = ctx.reserve(product(&[evaluations,self.pool.len()+options.max_length+32,256])?)?;
            let result_memory = ctx.reserve(product(&[evaluations,options.optimizer.max_evaluations,8])?)?;
            let graph = if learned.is_some_and(|m| m.ready) {Some(Graph::from_problem(self)?)} else {None};
            let mut evaluator = Evaluator {problem:self,options,learned,graph,cache:BTreeMap::new(),seen:BTreeSet::new(),history:vec![],optimizer_steps:0,screenings:0,importance_evaluations:0,importance_work:evolution::Work::default(),gradient_evaluations:0,failures:vec![]};
            let mut reference = evaluator.evaluate(vec![],vec![],0)?;
            evaluator.complete(&mut reference)?;
            let reference_energy = reference.energy;
            let mut current = reference;
            let mut seed=None;
            let mut transferred_start=None;
            if !initial_indices.is_empty() {
                let before=evaluator.history.len();let before_steps=evaluator.optimizer_steps;
                // With `warm_start=false` every state's cost is the optimum reached from zeros. The
                // transferred start is another start: it is evaluated apart (never cached under the
                // sequence) and only decides `kept` and the returned candidate, while the chain starts
                // from the zero-start state of the same sequence.
                let mut transferred = if options.warm_start {evaluator.evaluate(initial_indices.to_vec(),initial_parameters.to_vec(),0)?}
                    else {evaluator.evaluate_uncached(initial_indices.to_vec(),initial_parameters.to_vec(),0)?};
                let kept=transferred.cost<current.cost;
                seed=Some(crate::quantum::solver::SeedRelaxation {kept,energy:transferred.energy,success:transferred.success,
                    message:transferred.message.clone(),evaluations:evaluator.history.len()-before,steps:evaluator.optimizer_steps-before_steps,warm:false});
                if kept {
                    if options.warm_start {
                        evaluator.complete(&mut transferred)?;
                        current=transferred;
                    } else {
                        current=evaluator.evaluate(initial_indices.to_vec(),vec![0.;initial_indices.len()],0)?;
                        evaluator.complete(&mut current)?;
                        transferred_start=Some(transferred);
                    }
                }
            }
            let chain_steps=if seed.as_ref().is_some_and(|s| s.kept) {options.transfer_steps.unwrap_or(options.max_steps)} else {options.max_steps};
            let mut best = current.clone();
            let mut best_energy = current.clone();
            if let Some(start) = transferred_start {
                if start.cost<best.cost {best=start.clone();}
                if start.energy<best_energy.energy {best_energy=start;}
            }
            let kernel = if self.pool.is_empty() {None} else {Some(Kernel::new(self.pool.len(),options.max_length,options.move_weights)?)};
            let mut random = Rng(options.seed);
            let mut screening_random = Rng(options.seed ^ 0xd1b54a32d192ed03);
            let mut records = Vec::new();
            let mut steps = Vec::new();
            let mut acceptance = [(0,0);3];
            let mut screened = BTreeSet::new();
            for step in 0..if kernel.is_some() {chain_steps} else {0} {
                let kernel = kernel.as_ref().unwrap();
                let temperature = options.temperature(step,chain_steps);
                if options.screen_insertions>0 && current.sequence.len()<options.max_length && screened.insert(current.sequence.clone()) {
                    let mut remaining: Vec<_> = (0..self.pool.len()).collect();
                    let probabilities = &current.screened()?.probabilities;
                    for _ in 0..options.screen_insertions.min(remaining.len()) {
                        let mut draw = screening_random.uniform()*remaining.iter().map(|&i|probabilities[i]).sum::<f64>();
                        let mut selected = remaining.len()-1;
                        for (j,&i) in remaining.iter().enumerate() {draw-=probabilities[i];if draw<0. {selected=j;break;}}
                        let operator = remaining.remove(selected);
                        let action = Action {kind:Move::Insert,position:screening_random.index(current.sequence.len()+1),operator};
                        let sequence = action.apply(&current.sequence);
                        let initial = if options.warm_start {action.transfer(&current.parameters,options.replace_inherit)} else {vec![0.;sequence.len()]};
                        let proposed = evaluator.evaluate(sequence,initial,step+1)?;
                        // A screened insertion is only recorded: its relaxation is all the row reads.
                        records.push(record(step+1,action,&current,&proposed,temperature,None,true)?);
                    }
                }
                let (screen,weights) = (current.screened()?,current.weighed()?);
                let action = kernel.draw(ctx,&current.sequence,&screen.probabilities,&weights.removal,&mut random)?;
                let sequence = action.apply(&current.sequence);
                let initial = if options.warm_start {action.transfer(&current.parameters,options.replace_inherit)} else {vec![0.;sequence.len()]};
                let mut proposed = evaluator.evaluate(sequence,initial,step+1)?;
                let forward = kernel.log_q(ctx,&proposed.sequence,&current.sequence,&screen.probabilities,&weights.removal)?;
                // The reverse edit reads one half of the proposed state's distribution: an insertion
                // is undone by a deletion (the removal weights), a deletion by an insertion (the
                // operator probabilities), a replacement by a replacement (both).
                if matches!(action.kind,Move::Insert|Move::Replace) {evaluator.weigh(&mut proposed)?;}
                if matches!(action.kind,Move::Delete|Move::Replace) {evaluator.screen(&mut proposed)?;}
                let reverse = kernel.log_q_matched(&current.sequence,&proposed.sequence,proposed.screen.as_ref().map(|s|&s.probabilities[..]),proposed.weights.as_ref().map(|w|&w.removal[..]))?;
                let ratio = kernel::log_acceptance(temperature,current.cost,proposed.cost,forward,reverse);
                let accepted = ratio>=0. || random.uniform().ln()<ratio;
                // A rejected state is never drawn from; an accepted one becomes `current` complete.
                if accepted {evaluator.complete(&mut proposed)?;}
                let row = record(step+1,action,&current,&proposed,temperature,Some((forward,reverse,ratio,accepted)),options.record)?;
                if options.record {records.push(row.clone());}
                steps.push(row);
                let move_index = match action.kind {Move::Insert=>0,Move::Delete=>1,Move::Replace=>2};
                acceptance[move_index].1+=1;
                if accepted {acceptance[move_index].0+=1;}
                if proposed.energy<best_energy.energy {best_energy=proposed.clone();}
                if proposed.cost<best.cost {best=proposed.clone();}
                if accepted {current=proposed;}
            }
            let final_state = self.prepare_state(&best.sequence,&best.parameters,true)?;
            // A state that was not screened (a uniform chain that records nothing) is screened once.
            // (`best` may be a rejected state whose screen was skipped, or a transferred start.)
            let final_gradients = match best.screen.as_ref().filter(|s|s.gradients.len()==self.pool.len()) {
                Some(s) => s.gradients.clone(),
                None => {evaluator.screenings+=1;self.pool_gradients(&final_state,true)?}
            };
            Ok(SearchResult {
                solution: SolveResult {seed,energy:best.energy,reference:reference_energy,parameters:best.parameters,
                    state:final_state,labels:best.sequence.iter().map(|&i|self.pool[i].label.clone()).collect(),indices:best.sequence,
                    iterations:vec![],history:evaluator.history,optimizer_steps:evaluator.optimizer_steps,success:false,
                    message:"architecture search budget completed; no eigenstate convergence certificate".into(),
                    final_gradient:final_gradients.iter().map(|g|g.abs()).fold(0.,f64::max),budget_stopped:false,elapsed_seconds:started.elapsed().as_secs_f64(),pruned:vec![],hessian:None,notes:vec![],_memory:result_memory},
                records,steps,acceptance,optimal_cost:best.cost,best_energy:best_energy.energy,
                best_energy_indices:best_energy.sequence,best_energy_parameters:best_energy.parameters,final_energy:current.energy,
                final_indices:current.sequence,final_parameters:current.parameters,num_architectures:evaluator.seen.len(),
                num_screenings:evaluator.screenings,num_importance_evaluations:evaluator.importance_evaluations,importance_work:evaluator.importance_work,num_gradient_evaluations:evaluator.gradient_evaluations,failures:evaluator.failures,_memory:memory,
            })
        })
    }
}
