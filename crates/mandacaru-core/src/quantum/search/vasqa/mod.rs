//! Quenched Metropolis–Hastings search over variational operator sequences (VASQA).
//!
//! Each Markov chain step proposes an architecture edit (`Insert`, `Delete`, or `Replace`) over a
//! bounded operator sequence $C = (\mu_1, \dots, \mu_L)$ with $0 \le L \le L_{\max}$, relaxes the
//! continuous parameters $\theta$ to obtain $E_{\mathrm{opt}}(C)$, forms the regularized cost
//! $F(C) = E_{\mathrm{opt}}(C) + \lambda L$, and accepts the edit with Metropolis–Hastings
//! probability:
//!
//! $$\alpha(C \to C') = \min\!\left(1, \exp\!\left(-\frac{F(C') - F(C)}{T}\right) \frac{q(C \mid C')}{q(C' \mid C)}\right)$$
mod checkpoint;
mod evaluator;
mod options;
mod replica;

pub use checkpoint::{ReplicaState, SearchCheckpoint, SearchResult, SubspaceSearchResult};
pub use options::SearchOptions;

use crate::quantum::{
    amplitude::State,
    parallel,
    search::{
        kernel::{Action, Kernel, Move},
        learning::{graph::BipartiteGraph, training::Model},
    },
    solver::{Execution, SolveResult},
    subspace, QuantumProblem,
};
use crate::{math::random::Rng, product, Context, Error, Result};
use evaluator::{parameter_bits, Ensemble, Evaluated, Evaluator};
use rayon::prelude::*;
use replica::{record, replica_stream, step_replica, strip_zeros_from, Replica, ReplicaStepData};
use serde_json::json;
use std::{collections::BTreeSet, time::Instant};

pub(super) fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

impl QuantumProblem {
    pub fn search(
        &self,
        options: &SearchOptions,
        initial_indices: &[usize],
        initial_parameters: &[f64],
    ) -> Result<SearchResult> {
        self.search_with_model(options, initial_indices, initial_parameters, None)
    }
    pub fn search_with_model(
        &self,
        options: &SearchOptions,
        initial_indices: &[usize],
        initial_parameters: &[f64],
        learned: Option<&Model>,
    ) -> Result<SearchResult> {
        self.search_with_execution(options, initial_indices, initial_parameters, learned, None)
    }
    /// Run VASQA/VALSQA using an optional external [`Execution`] circuit/shot backend.
    pub fn search_with_execution(
        &self,
        options: &SearchOptions,
        initial_indices: &[usize],
        initial_parameters: &[f64],
        learned: Option<&Model>,
        execution: Option<&mut dyn Execution>,
    ) -> Result<SearchResult> {
        let (result, _) = self.search_inner(
            options,
            initial_indices,
            initial_parameters,
            learned,
            None,
            execution,
            None,
        )?;
        Ok(result)
    }
    /// Resume a previously checkpointed VASQA/VALSQA chain up to `options.max_steps`.
    pub fn resume_search(
        &self,
        options: &SearchOptions,
        checkpoint: &SearchCheckpoint,
        learned: Option<&Model>,
        execution: Option<&mut dyn Execution>,
    ) -> Result<SearchResult> {
        let (result, _) = self.search_inner(
            options,
            &[],
            &[],
            learned,
            None,
            execution,
            Some(checkpoint),
        )?;
        Ok(result)
    }
    /// Subspace-VASQA: Markov-chain architecture search minimizing weighted ensemble energy
    /// followed by Rayleigh–Ritz subspace diagonalization for ground and excited eigenstates.
    pub fn search_subspace(
        &self,
        options: &SearchOptions,
        weights: &[f64],
        initial_indices: &[usize],
        initial_parameters: &[f64],
    ) -> Result<SubspaceSearchResult> {
        self.search_subspace_with_model(options, weights, initial_indices, initial_parameters, None)
    }
    /// Like [`Self::search_subspace`], accepting an optional learned graph proposal [`Model`].
    pub fn search_subspace_with_model(
        &self,
        options: &SearchOptions,
        weights: &[f64],
        initial_indices: &[usize],
        initial_parameters: &[f64],
        learned: Option<&Model>,
    ) -> Result<SubspaceSearchResult> {
        self.search_subspace_inner(
            options,
            weights,
            initial_indices,
            initial_parameters,
            learned,
            None,
        )
    }
    /// Resume a previously checkpointed Subspace-VASQA/VALSQA chain up to `options.max_steps`.
    pub fn resume_search_subspace(
        &self,
        options: &SearchOptions,
        weights: &[f64],
        checkpoint: &SearchCheckpoint,
        learned: Option<&Model>,
    ) -> Result<SubspaceSearchResult> {
        self.search_subspace_inner(options, weights, &[], &[], learned, Some(checkpoint))
    }
    fn search_subspace_inner(
        &self,
        options: &SearchOptions,
        weights: &[f64],
        initial_indices: &[usize],
        initial_parameters: &[f64],
        learned: Option<&Model>,
        checkpoint: Option<&SearchCheckpoint>,
    ) -> Result<SubspaceSearchResult> {
        subspace::validate_subspace_weights(weights)?;
        let ref_rows = self.subspace_ref_rows(weights.len())?;
        let ensemble = Ensemble {
            weights,
            ref_rows: &ref_rows,
        };
        let (core, diag) = self.search_inner(
            options,
            initial_indices,
            initial_parameters,
            learned,
            Some(ensemble),
            None,
            checkpoint,
        )?;
        let (ensemble_energies, energies, states) = diag.ok_or_else(|| {
            Error::Kernel("ensemble search did not produce subspace eigenstates".into())
        })?;
        let mut reference_indices: Vec<usize> = (0..ensemble_energies.len()).collect();
        reference_indices.sort_by(|&a, &b| ensemble_energies[a].total_cmp(&ensemble_energies[b]));
        let SearchResult {
            solution,
            records,
            steps,
            acceptance,
            optimal_cost,
            best_energy,
            best_energy_indices,
            best_energy_parameters,
            final_energy,
            final_indices,
            final_parameters,
            num_architectures,
            num_screenings,
            num_importance_evaluations,
            importance_work,
            num_gradient_evaluations,
            failures,
            swaps,
            swap_acceptance,
            checkpoint,
            _memory,
        } = core;
        Ok(SubspaceSearchResult {
            solution,
            energies,
            ensemble_energies,
            states,
            reference_indices,
            records,
            steps,
            acceptance,
            optimal_cost,
            best_energy,
            best_energy_indices,
            best_energy_parameters,
            final_energy,
            final_indices,
            final_parameters,
            num_architectures,
            num_screenings,
            num_importance_evaluations,
            importance_work,
            num_gradient_evaluations,
            failures,
            swaps,
            swap_acceptance,
            checkpoint,
            _memory,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn search_inner(
        &self,
        options: &SearchOptions,
        initial_indices: &[usize],
        initial_parameters: &[f64],
        learned: Option<&Model>,
        ensemble: Option<Ensemble<'_>>,
        execution: Option<&mut dyn Execution>,
        resume: Option<&SearchCheckpoint>,
    ) -> Result<(SearchResult, Option<(Vec<f64>, Vec<f64>, State)>)> {
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
        if ensemble.is_some() && execution.is_some() {
            return Err(invalid(
                "subspace search uses local state-vector simulation",
            ));
        }
        let mut execution = execution;
        self.context.run(|| {
            let started = Instant::now();
            let ctx: &Context = &self.context;
            let replicas = options.replicas;
            // Every replica proposes once per step.
            let evaluations = options.max_steps.max(options.transfer_steps.unwrap_or(0)).checked_mul(options.screen_insertions.min(self.pool.len())+1).and_then(|v|v.checked_add(3)).and_then(|v|v.checked_mul(replicas)).ok_or(Error::Overflow)?;
            let memory = ctx.reserve(product(&[evaluations,self.pool.len()+options.max_length+32,256])?)?;
            let result_memory = ctx.reserve(product(&[evaluations,options.optimizer.max_evaluations,8])?)?;
            let graph = if learned.is_some_and(|m| m.ready) {Some(BipartiteGraph::from_problem(self)?)} else {None};
            let screen_hamiltonian = options
                .screen_hamiltonian_terms
                .filter(|&k| k < self.hamiltonian.terms.len())
                .map(|k| self.hamiltonian.top_terms(ctx, k))
                .transpose()?;
            let mut evaluator = Evaluator::new(
                self,
                options,
                learned,
                graph.as_ref(),
                screen_hamiltonian.as_ref(),
                ensemble,
            );
            let (mut reference, ref_cache) =
                evaluator.evaluate(vec![], vec![], None, 0, execution.as_deref_mut())?;
            evaluator.complete(
                &mut reference,
                ref_cache.as_ref(),
                None,
                execution.as_deref_mut(),
            )?;
            let reference_energy = reference.energy;
            let mut seed = None;
            let (start_step, seeded_kept, chain_steps, mut best, mut best_energy, mut chains, mut swap_random, mut screening_random, mut swap_acceptance, mut acceptance) = if let Some(ckpt) = resume {
                if ckpt.replicas.len() != replicas {
                    return Err(invalid("search checkpoint replica count disagrees with options.replicas"));
                }
                let chain_steps = if ckpt.seeded_kept {
                    options.transfer_steps.unwrap_or(options.max_steps)
                } else {
                    options.max_steps
                };
                if ckpt.step > chain_steps {
                    return Err(invalid("search checkpoint step already exceeds max_steps"));
                }
                let probe_state = |seq: &[usize], params: &[f64], exec: Option<&mut (dyn Execution + '_)>| -> Result<Evaluated> {
                    let energy = if let Some(e) = exec {
                        e.evaluate(self, seq, params, false)?.energy
                    } else if let Some(ens) = &ensemble {
                        self.subspace_probe_capturing(seq, params, true, ens.weights, ens.ref_rows, None, options.sparse_kernels)?.energy
                    } else {
                        self.probe_with_mode(seq, params, true, options.sparse_kernels)?.energy
                    };
                    Ok(Evaluated {
                        cost: energy + options.length_penalty * seq.len() as f64,
                        sequence: seq.to_vec(),
                        parameters: params.to_vec(),
                        energy,
                        screen: None,
                        weights: None,
                        hessian: None,
                        success: true,
                        message: "resumed state".into(),
                    })
                };
                let best = probe_state(&ckpt.best_sequence, &ckpt.best_parameters, execution.as_deref_mut())?;
                let best_energy = probe_state(&ckpt.best_energy_sequence, &ckpt.best_energy_parameters, execution.as_deref_mut())?;
                let mut chains = Vec::with_capacity(replicas);
                for r in &ckpt.replicas {
                    let mut cur = probe_state(&r.sequence, &r.parameters, execution.as_deref_mut())?;
                    evaluator.seen.insert(cur.sequence.clone());
                    evaluator.complete(&mut cur, None, None, execution.as_deref_mut())?;
                    chains.push(Replica { current: cur, random: Rng(r.rng) });
                }
                (
                    ckpt.step,
                    ckpt.seeded_kept,
                    chain_steps,
                    best,
                    best_energy,
                    chains,
                    Rng(ckpt.swap_rng),
                    Rng(ckpt.screening_rng),
                    ckpt.swap_acceptance,
                    ckpt.acceptance,
                )
            } else {
                let mut current = reference;
                let mut transferred_start=None;
                if !initial_indices.is_empty() {
                    let before=evaluator.history.len();let before_steps=evaluator.optimizer_steps;
                    let init_bits = parameter_bits(initial_parameters);
                    let (mut transferred, tr_cache) = if options.warm_start {
                        evaluator.evaluate(
                            initial_indices.to_vec(),
                            initial_parameters.to_vec(),
                            None,
                            0,
                            execution.as_deref_mut(),
                        )?
                    } else {
                        evaluator.evaluate_uncached(
                            initial_indices.to_vec(),
                            initial_parameters.to_vec(),
                            0,
                            execution.as_deref_mut(),
                        )?
                    };
                    let kept=transferred.cost<current.cost;
                    seed=Some(crate::quantum::solver::SeedRelaxation {kept,energy:transferred.energy,success:transferred.success,
                        message:transferred.message.clone(),evaluations:evaluator.history.len()-before,steps:evaluator.optimizer_steps-before_steps,warm:false});
                    if kept {
                        if options.warm_start {
                            evaluator.complete(
                                &mut transferred,
                                tr_cache.as_ref(),
                                Some(&init_bits),
                                execution.as_deref_mut(),
                            )?;
                            current=transferred;
                        } else {
                            let (mut zero_cur, zero_cache) = evaluator.evaluate(
                                initial_indices.to_vec(),
                                vec![0.; initial_indices.len()],
                                None,
                                0,
                                execution.as_deref_mut(),
                            )?;
                            evaluator.complete(
                                &mut zero_cur,
                                zero_cache.as_ref(),
                                None,
                                execution.as_deref_mut(),
                            )?;
                            current=zero_cur;
                            transferred_start=Some(transferred);
                        }
                    }
                }
                let seeded_kept = seed.as_ref().is_some_and(|s| s.kept);
                let chain_steps=if seeded_kept {options.transfer_steps.unwrap_or(options.max_steps)} else {options.max_steps};
                let mut best = current.clone();
                let mut best_energy = current.clone();
                if let Some(start) = transferred_start {
                    if start.cost<best.cost {best=start.clone();}
                    if start.energy<best_energy.energy {best_energy=start;}
                }
                // Replica 0 is the single chain: the same start, the same stream, the same draws.
                let mut chains: Vec<Replica> = (1..replicas).map(|r| Replica {current: current.clone(), random: replica_stream(options.seed, r)}).collect();
                chains.insert(0, Replica {current, random: replica_stream(options.seed, 0)});
                (
                    0,
                    seeded_kept,
                    chain_steps,
                    best,
                    best_energy,
                    chains,
                    Rng(options.seed ^ 0x5851f42d4c957f2d),
                    Rng(options.seed ^ 0xd1b54a32d192ed03),
                    (0, 0),
                    [(0,0);3],
                )
            };
            let schedule = options.active_temperatures(seeded_kept);
            let kernel = if self.pool.is_empty() {None} else {Some(Kernel::new(self.pool.len(),options.max_length,options.move_weights)?)};
            let mut records = Vec::new();
            let mut steps = Vec::new();
            let mut swaps = Vec::new();
            let mut screened = BTreeSet::new();
            let parallel_replicas = replicas > 1
                && execution.is_none()
                && ctx.width() > 1
                && ctx.pool().is_some()
                && !parallel::is_inside();
            for step in start_step..if kernel.is_some() {chain_steps} else {start_step} {
                let Some(kernel) = kernel.as_ref() else { break; };
                if parallel_replicas {
                    let Some(pool) = ctx.pool() else {
                        return Err(invalid("parallel replicas require a thread pool"));
                    };
                    let mut workers: Vec<_> = (0..replicas).map(|_| evaluator.fork_readonly()).collect();
                    let outcomes: Vec<Result<ReplicaStepData>> = pool.install(|| {
                        chains
                            .par_iter_mut()
                            .zip(workers.par_iter_mut())
                            .enumerate()
                            .map(|(replica, (chain, worker))| {
                                let temperature = options.replica_temperature_on(schedule, step, chain_steps, replica);
                                parallel::inside(|| {
                                    step_replica(worker, chain, kernel, step, replica, replicas, temperature, None)
                                })
                            })
                            .collect()
                    });
                    let deltas: Vec<_> = workers.into_iter().map(Evaluator::into_delta).collect();
                    for (delta, outcome) in deltas.into_iter().zip(outcomes) {
                        evaluator.merge_delta(delta);
                        let data = outcome?;
                        if options.record {records.push(data.row.clone());}
                        steps.push(data.row);
                        acceptance[data.move_index].1 += 1;
                        if data.accepted {acceptance[data.move_index].0 += 1;}
                        if data.proposed.energy < best_energy.energy {best_energy = data.proposed.clone();}
                        if data.proposed.cost < best.cost {best = data.proposed;}
                    }
                } else {
                    for (replica, chain) in chains.iter_mut().enumerate() {
                        let temperature = options.replica_temperature_on(schedule, step, chain_steps, replica);
                        if options.screen_insertions > 0 && chain.current.sequence.len() < options.max_length && screened.insert(chain.current.sequence.clone()) {
                            let mut remaining: Vec<_> = (0..self.pool.len()).collect();
                            let probabilities = chain.current.screened()?.probabilities.clone();
                            for _ in 0..options.screen_insertions.min(remaining.len()) {
                                let mut draw = screening_random.uniform()*remaining.iter().map(|&i|probabilities[i]).sum::<f64>();
                                let mut selected = remaining.len()-1;
                                for (j,&i) in remaining.iter().enumerate() {draw-=probabilities[i];if draw<0. {selected=j;break;}}
                                let operator = remaining.remove(selected);
                                let action = Action {kind:Move::Insert,position:screening_random.index(chain.current.sequence.len()+1),operator};
                                let sequence = action.apply(&chain.current.sequence);
                                let initial = if options.warm_start {action.transfer(&chain.current.parameters,options.replace_inherit)} else {vec![0.;sequence.len()]};
                                let (proposed, _) = evaluator.evaluate(
                                    sequence,
                                    initial,
                                    None,
                                    step + 1,
                                    execution.as_deref_mut(),
                                )?;
                                // A screened insertion is only recorded: its relaxation is all the row reads.
                                records.push(record(step+1,action,&chain.current,&proposed,temperature,None,true)?);
                            }
                        }
                        let data = step_replica(
                            &mut evaluator,
                            chain,
                            kernel,
                            step,
                            replica,
                            replicas,
                            temperature,
                            execution.as_deref_mut(),
                        )?;
                        if options.record {records.push(data.row.clone());}
                        steps.push(data.row);
                        acceptance[data.move_index].1 += 1;
                        if data.accepted {acceptance[data.move_index].0 += 1;}
                        if data.proposed.energy < best_energy.energy {best_energy = data.proposed.clone();}
                        if data.proposed.cost < best.cost {best = data.proposed;}
                    }
                }
                if replicas > 1 {
                    // One adjacent pair exchanges its states; the streams stay with their replicas.
                    let pair = swap_random.index(replicas - 1);
                    let (cold, hot) = (
                        options.replica_temperature_on(schedule, step, chain_steps, pair),
                        options.replica_temperature_on(schedule, step, chain_steps, pair + 1),
                    );
                    let (cold_cost, hot_cost) = (chains[pair].current.cost, chains[pair+1].current.cost);
                    let sequences = json!([chains[pair].current.sequence, chains[pair+1].current.sequence]);
                    let ratio = (1./cold - 1./hot)*(cold_cost - hot_cost);
                    let accepted = ratio>=0. || swap_random.uniform().ln()<ratio;
                    swap_acceptance.1 += 1;
                    if accepted {
                        swap_acceptance.0 += 1;
                        let (lower, upper) = chains.split_at_mut(pair+1);
                        std::mem::swap(&mut lower[pair].current, &mut upper[0].current);
                    }
                    swaps.push(json!({"step":step+1,"replicas":[pair,pair+1],"temperatures":[cold,hot],"costs":[cold_cost,hot_cost],"sequences":sequences,"log_acceptance":ratio,"accepted":accepted}));
                }
            }
            let checkpoint = SearchCheckpoint {
                step: if kernel.is_some() { chain_steps } else { start_step },
                total_steps: chain_steps,
                seeded_kept,
                replicas: chains
                    .iter()
                    .map(|c| ReplicaState {
                        sequence: c.current.sequence.clone(),
                        parameters: c.current.parameters.clone(),
                        rng: c.random.0,
                    })
                    .collect(),
                best_sequence: best.sequence.clone(),
                best_parameters: best.parameters.clone(),
                best_energy_sequence: best_energy.sequence.clone(),
                best_energy_parameters: best_energy.parameters.clone(),
                swap_rng: swap_random.0,
                screening_rng: screening_random.0,
                acceptance,
                swap_acceptance,
            };
            let mut current = chains.swap_remove(0).current;
            let Evaluator {
                mut history,
                mut optimizer_steps,
                mut screenings,
                mut importance_evaluations,
                mut importance_work,
                mut gradient_evaluations,
                mut failures,
                seen,
                ..
            } = evaluator;
            if options.strip_zero_operators {
                strip_zeros_from(self, &mut best, options.length_penalty, options.sparse_kernels, ensemble.as_ref(), execution.as_deref_mut())?;
                strip_zeros_from(self, &mut best_energy, options.length_penalty, options.sparse_kernels, ensemble.as_ref(), execution.as_deref_mut())?;
                strip_zeros_from(self, &mut current, options.length_penalty, options.sparse_kernels, ensemble.as_ref(), execution.as_deref_mut())?;
            }
            let exact_screen_available = screen_hamiltonian.is_none();
            let (solution_energy, final_state, final_gradients, diag) = if let Some(ens) = ensemble {
                let diag = self.subspace_prepare_and_diagonalize(
                    &best.sequence,
                    &best.parameters,
                    true,
                    ens.ref_rows,
                    options.sparse_kernels,
                )?;
                let final_gradients = match best.screen.as_ref().filter(|s| exact_screen_available && s.gradients.len() == self.pool.len()) {
                    Some(s) => s.gradients.clone(),
                    None => {
                        screenings += 1;
                        let mut acc = vec![0.; self.pool.len()];
                        for (&w, &ref_row) in ens.weights.iter().zip(ens.ref_rows) {
                            let st = self.subspace_prepare_state(
                                &best.sequence,
                                &best.parameters,
                                true,
                                ref_row,
                                options.sparse_kernels,
                            )?;
                            for (slot, g) in acc.iter_mut().zip(self.pool_gradients(&st, true)?) {
                                *slot += w * g;
                            }
                        }
                        acc
                    }
                };
                (
                    diag.energies[0],
                    diag.ground_state,
                    final_gradients,
                    Some((diag.ensemble_energies, diag.energies, diag.states)),
                )
            } else {
                let final_state = self.result_state(&best.sequence, &best.parameters, true)?;
                // A state that was not screened (a uniform chain that records nothing, or screened on a
                // surrogate Hamiltonian) is screened once against the exact Hamiltonian at completion.
                let final_gradients = match best.screen.as_ref().filter(|s| exact_screen_available && s.gradients.len() == self.pool.len()) {
                    Some(s) => s.gradients.clone(),
                    None => {
                        screenings += 1;
                        if let Some(exec) = execution.as_deref_mut() {
                            exec.pool_gradients(self, &best.sequence, &best.parameters)?
                        } else {
                            self.pool_gradients(&final_state, true)?
                        }
                    }
                };
                (best.energy, final_state, final_gradients, None)
            };
            let _ = (&mut history, &mut optimizer_steps, &mut importance_evaluations, &mut importance_work, &mut gradient_evaluations, &mut failures);
            Ok((SearchResult {
                solution: SolveResult {seed,energy:solution_energy,reference:reference_energy,parameters:best.parameters,
                    state:final_state,labels:best.sequence.iter().map(|&i|self.pool[i].label.clone()).collect(),indices:best.sequence,
                    iterations:vec![],history,optimizer_steps,success:false,
                    message:"architecture search budget completed; no eigenstate convergence certificate".into(),
                    final_gradient:final_gradients.iter().map(|g|g.abs()).fold(0.,f64::max),budget_stopped:false,elapsed_seconds:started.elapsed().as_secs_f64(),pruned:vec![],hessian:best.hessian,notes:vec![],_memory:result_memory},
                records,steps,acceptance,optimal_cost:best.cost,best_energy:best_energy.energy,
                best_energy_indices:best_energy.sequence,best_energy_parameters:best_energy.parameters,final_energy:current.energy,
                final_indices:current.sequence,final_parameters:current.parameters,num_architectures:seen.len(),
                num_screenings:screenings,num_importance_evaluations:importance_evaluations,importance_work,num_gradient_evaluations:gradient_evaluations,failures,swaps,swap_acceptance,checkpoint,_memory:memory,
            }, diag))
        })
    }
}
