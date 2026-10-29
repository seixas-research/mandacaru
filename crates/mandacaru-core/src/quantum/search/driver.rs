//! Quenched Metropolis–Hastings search over variational operator sequences.
use super::kernel::{self, Action, Kernel, Move};
use super::learning::{
    graph::{Candidates, Graph},
    training::{self, Model},
};
use crate::{
    math::random::Rng,
    product,
    quantum::{evolution, optimizer, CachedProblem, SolveResult, C64},
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
    pub move_weights: [f64; 4],
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
            move_weights: [1.; 4],
            temperatures: [0.1 / 27.211386245988, 0.001 / 27.211386245988],
            length_penalty: 0.,
            proposal_temperature: 0.2,
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
#[derive(Clone)]
struct Evaluated {
    sequence: Vec<usize>,
    parameters: Vec<f64>,
    energy: f64,
    cost: f64,
    gradients: Vec<f64>,
    probabilities: Vec<f64>,
    success: bool,
    message: String,
}
pub struct SearchResult {
    pub solution: SolveResult,
    pub records: Vec<Value>,
    pub steps: Vec<Value>,
    pub acceptance: [(usize, usize); 4],
    pub optimal_cost: f64,
    pub best_energy: f64,
    pub best_energy_indices: Vec<usize>,
    pub best_energy_parameters: Vec<f64>,
    pub final_energy: f64,
    pub final_indices: Vec<usize>,
    pub final_parameters: Vec<f64>,
    pub num_architectures: usize,
    pub num_screenings: usize,
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
    failures: Vec<(usize, String)>,
}
impl Evaluator<'_> {
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
        let (parameters, energy, success, message) = if sequence.is_empty() {
            (
                vec![],
                self.problem.evaluate(&[], &[], true)?.energy,
                true,
                "empty reference".into(),
            )
        } else {
            let result = optimizer::minimize(
                &self.problem.context,
                &initial,
                &self.options.optimizer,
                |p| {
                    let value = self.problem.evaluate(&sequence, p, true)?;
                    Ok((value.energy, value.gradient))
                },
            )?;
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
        let state = self.problem.prepare_state(&sequence, &parameters, true)?;
        let hpsi = self.problem.hamiltonian.apply_centered(
            &self.problem.context,
            &self.problem.basis,
            state.as_slice(),
        )?;
        let mut action = self
            .problem
            .context
            .allocate::<C64>(self.problem.basis.len())?;
        let gradients: Vec<_> = self
            .problem
            .pool
            .iter()
            .map(|g| {
                g.operator
                    .apply_into(&self.problem.basis, state.as_slice(), action.as_mut_slice());
                2. * evolution::inner(hpsi.as_slice(), action.as_slice()).re
            })
            .collect();
        self.screenings += 1;
        let mut probabilities = if self.options.uniform {
            vec![1. / gradients.len() as f64; gradients.len()]
        } else {
            kernel::softmax(&gradients, self.options.proposal_temperature)?
        };
        if let (Some(model), Some(graph)) = (self.learned, &self.graph) {
            if model.ready && !gradients.is_empty() {
                let candidates: Vec<_> = (0..gradients.len()).collect();
                let rows = Candidates::new(
                    &self.problem.context,
                    graph,
                    &sequence,
                    &candidates,
                    &gradients,
                    energy - graph.reference_energy,
                )?;
                let scores = model.scores(&self.problem.context, graph, &rows)?;
                let scores: Vec<_> = scores.iter().map(|v| v / model.temperature).collect();
                if scores.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("learned proposal temperature overflows scores"));
                }
                let learned = training::softmax(&scores);
                for (p, learned) in probabilities.iter_mut().zip(learned) {
                    *p = model.mixing * *p + (1. - model.mixing) * learned;
                }
            }
        }
        let result = Evaluated {
            cost: energy + self.options.length_penalty * sequence.len() as f64,
            sequence,
            parameters,
            energy,
            gradients,
            probabilities,
            success,
            message,
        };
        if !self.options.warm_start {
            self.cache.insert(result.sequence.clone(), result.clone());
        }
        Ok(result)
    }
}
fn record(
    step: usize,
    action: Action,
    source: &Evaluated,
    proposed: &Evaluated,
    temperature: f64,
    outcome: Option<(f64, f64, f64, bool)>,
) -> Value {
    json!({"step":step,"move":action.kind.name(),"position":action.position,
        "operator":if matches!(action.kind,Move::Insert|Move::Replace) {Some(action.operator)} else {None},
        "other":if action.kind==Move::Swap {Some(action.other)} else {None},
        "source":source.sequence,"proposed":proposed.sequence,"source_parameters":source.parameters,"parameters":proposed.parameters,
        "source_energy":source.energy,"energy":proposed.energy,"source_cost":source.cost,"cost":proposed.cost,
        "delta_energy":proposed.energy-source.energy,"delta_cost":proposed.cost-source.cost,
        "source_gradients":source.gradients,"source_probabilities":source.probabilities,
        "temperature":temperature,"log_q_forward":outcome.map(|v|v.0),"log_q_reverse":outcome.map(|v|v.1),
        "log_acceptance":outcome.map(|v|v.2),"accepted":outcome.map(|v|v.3),"screened":outcome.is_none(),
        "optimizer_success":proposed.success,"optimizer_message":proposed.message})
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
            let evaluations = options.max_steps.max(options.transfer_steps.unwrap_or(0)).checked_mul(options.screen_insertions.min(self.pool.len())+1).and_then(|v|v.checked_add(2)).ok_or(Error::Overflow)?;
            let memory = ctx.reserve(product(&[evaluations,self.pool.len()+options.max_length+32,256])?)?;
            let result_memory = ctx.reserve(product(&[evaluations,options.optimizer.max_evaluations,8])?)?;
            let graph = if learned.is_some_and(|m| m.ready) {Some(Graph::from_problem(self)?)} else {None};
            let mut evaluator = Evaluator {problem:self,options,learned,graph,cache:BTreeMap::new(),seen:BTreeSet::new(),history:vec![],optimizer_steps:0,screenings:0,failures:vec![]};
            let reference = evaluator.evaluate(vec![],vec![],0)?;
            let reference_energy = reference.energy;
            let mut current = reference;
            let mut seed=None;
            if !initial_indices.is_empty() {
                let before=evaluator.history.len();let before_steps=evaluator.optimizer_steps;
                let transferred = evaluator.evaluate(initial_indices.to_vec(),initial_parameters.to_vec(),0)?;
                let kept=transferred.cost<current.cost;
                seed=Some(crate::quantum::solver::SeedRelaxation {kept,energy:transferred.energy,success:transferred.success,
                    message:transferred.message.clone(),evaluations:evaluator.history.len()-before,steps:evaluator.optimizer_steps-before_steps});
                if kept {current=transferred;}
            }
            let chain_steps=if seed.as_ref().is_some_and(|s| s.kept) {options.transfer_steps.unwrap_or(options.max_steps)} else {options.max_steps};
            let mut best = current.clone();
            let mut best_energy = current.clone();
            let kernel = if self.pool.is_empty() {None} else {Some(Kernel::new(self.pool.len(),options.max_length,options.move_weights)?)};
            let mut random = Rng(options.seed);
            let mut screening_random = Rng(options.seed ^ 0xd1b54a32d192ed03);
            let mut records = Vec::new();
            let mut steps = Vec::new();
            let mut acceptance = [(0,0);4];
            let mut screened = BTreeSet::new();
            for step in 0..if kernel.is_some() {chain_steps} else {0} {
                let kernel = kernel.as_ref().unwrap();
                let temperature = options.temperature(step,chain_steps);
                if options.screen_insertions>0 && current.sequence.len()<options.max_length && screened.insert(current.sequence.clone()) {
                    let mut remaining: Vec<_> = (0..self.pool.len()).collect();
                    for _ in 0..options.screen_insertions.min(remaining.len()) {
                        let mut draw = screening_random.uniform()*remaining.iter().map(|&i|current.probabilities[i]).sum::<f64>();
                        let mut selected = remaining.len()-1;
                        for (j,&i) in remaining.iter().enumerate() {draw-=current.probabilities[i];if draw<0. {selected=j;break;}}
                        let operator = remaining.remove(selected);
                        let action = Action {kind:Move::Insert,position:screening_random.index(current.sequence.len()+1),operator,other:0};
                        let sequence = action.apply(&current.sequence);
                        let initial = if options.warm_start {action.transfer(&current.parameters,options.replace_inherit)} else {vec![0.;sequence.len()]};
                        let proposed = evaluator.evaluate(sequence,initial,step+1)?;
                        records.push(record(step+1,action,&current,&proposed,temperature,None));
                    }
                }
                let action = kernel.draw(ctx,&current.sequence,&current.probabilities,&mut random)?;
                let sequence = action.apply(&current.sequence);
                let initial = if options.warm_start {action.transfer(&current.parameters,options.replace_inherit)} else {vec![0.;sequence.len()]};
                let proposed = evaluator.evaluate(sequence,initial,step+1)?;
                let forward = kernel.log_q(ctx,&proposed.sequence,&current.sequence,&current.probabilities)?;
                let reverse = kernel.log_q(ctx,&current.sequence,&proposed.sequence,&proposed.probabilities)?;
                let ratio = kernel::log_acceptance(temperature,current.cost,proposed.cost,forward,reverse);
                let accepted = ratio>=0. || random.uniform().ln()<ratio;
                let row = record(step+1,action,&current,&proposed,temperature,Some((forward,reverse,ratio,accepted)));
                if options.record {records.push(row.clone());}
                steps.push(row);
                let move_index = match action.kind {Move::Insert=>0,Move::Delete=>1,Move::Replace=>2,Move::Swap=>3};
                acceptance[move_index].1+=1;
                if accepted {acceptance[move_index].0+=1;}
                if proposed.energy<best_energy.energy {best_energy=proposed.clone();}
                if proposed.cost<best.cost {best=proposed.clone();}
                if accepted {current=proposed;}
            }
            let final_state = self.prepare_state(&best.sequence,&best.parameters,true)?;
            Ok(SearchResult {
                solution: SolveResult {seed,energy:best.energy,reference:reference_energy,parameters:best.parameters,
                    state:final_state,labels:best.sequence.iter().map(|&i|self.pool[i].label.clone()).collect(),indices:best.sequence,
                    iterations:vec![],history:evaluator.history,optimizer_steps:evaluator.optimizer_steps,success:false,
                    message:"architecture search budget completed; no eigenstate convergence certificate".into(),
                    final_gradient:best.gradients.iter().map(|g|g.abs()).fold(0.,f64::max),budget_stopped:false,elapsed_seconds:started.elapsed().as_secs_f64(),_memory:result_memory},
                records,steps,acceptance,optimal_cost:best.cost,best_energy:best_energy.energy,
                best_energy_indices:best_energy.sequence,best_energy_parameters:best_energy.parameters,final_energy:current.energy,
                final_indices:current.sequence,final_parameters:current.parameters,num_architectures:evaluator.seen.len(),
                num_screenings:evaluator.screenings,failures:evaluator.failures,_memory:memory,
            })
        })
    }
}
