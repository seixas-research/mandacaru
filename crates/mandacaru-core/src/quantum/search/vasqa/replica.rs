use super::evaluator::{adapt_hessian, parameter_bits, Ensemble, Evaluated, Evaluator};
use crate::math::random::Rng;
use crate::quantum::{
    search::kernel::{self, Action, Kernel, Move},
    solver::Execution,
    QuantumProblem,
};
use crate::Result;
use serde_json::{json, Value};

/// One replica: its current state and its own random stream.
pub(super) struct Replica {
    pub(super) current: Evaluated,
    pub(super) random: Rng,
}
/// Replica `replica`'s stream: replica 0 draws the stream of a single chain at `seed`.
pub(super) fn replica_stream(seed: u64, replica: usize) -> Rng {
    if replica == 0 {
        Rng(seed)
    } else {
        let mut mixer = Rng(seed ^ (replica as u64).wrapping_mul(0xa0761d6478bd642f));
        Rng(mixer.next_u64())
    }
}
type StepOutcome = (f64, Option<(f64, f64)>, bool);

pub(super) fn record(
    step: usize,
    action: Action,
    source: &Evaluated,
    proposed: &Evaluated,
    temperature: f64,
    outcome: Option<StepOutcome>,
    vectors: bool,
) -> Result<Value> {
    let mut row = json!({"step":step,"move":action.kind.name(),"position":action.position,
        "operator":if matches!(action.kind,Move::Insert|Move::Replace) {Some(action.operator)} else {None},
        "source":source.sequence,"proposed":proposed.sequence,"source_parameters":source.parameters,"parameters":proposed.parameters,
        "source_energy":source.energy,"energy":proposed.energy,"source_cost":source.cost,"cost":proposed.cost,
        "delta_energy":proposed.energy-source.energy,"delta_cost":proposed.cost-source.cost,
        "temperature":temperature,"log_q_forward":outcome.as_ref().map(|v|v.0),
        "log_q_reverse":outcome.as_ref().and_then(|v|v.1.map(|r|r.0)),
        "log_acceptance":outcome.as_ref().and_then(|v|v.1.map(|r|r.1)),
        "accepted":outcome.as_ref().map(|v|v.2),"screened":outcome.is_none(),
        "optimizer_success":proposed.success,"optimizer_message":proposed.message});
    // The four source vectors (pool and sequence long) are the recorded edit's training data: a
    // step row of a chain that records nothing does not copy them.
    if vectors {
        let (screen, weights) = (source.screened()?, source.weighed()?);
        row["source_gradients"] = json!(screen.gradients);
        row["source_probabilities"] = json!(screen.probabilities);
        row["source_importances"] = json!(weights.importances);
        row["source_removal"] = json!(weights.removal);
        if let Some(curvs) = &screen.curvatures {
            row["source_curvatures"] = json!(curvs);
        }
    }
    Ok(row)
}
pub(super) fn strip_zeros_from(
    problem: &QuantumProblem,
    state: &mut Evaluated,
    length_penalty: f64,
    sparse_kernels: bool,
    ensemble: Option<&Ensemble<'_>>,
    execution: Option<&mut (dyn Execution + '_)>,
) -> Result<()> {
    let keep: Vec<usize> = state
        .parameters
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p.abs() > 1e-14)
        .map(|(i, _)| i)
        .collect();
    if keep.len() == state.sequence.len() {
        return Ok(());
    }
    state.sequence = keep.iter().map(|&i| state.sequence[i]).collect();
    state.parameters = keep.iter().map(|&i| state.parameters[i]).collect();
    let energy = if let Some(exec) = execution {
        exec.evaluate(problem, &state.sequence, &state.parameters, false)?
            .energy
    } else {
        match ensemble {
            Some(ens) => {
                problem
                    .subspace_probe_capturing(
                        &state.sequence,
                        &state.parameters,
                        true,
                        ens.weights,
                        ens.ref_rows,
                        None,
                        sparse_kernels,
                    )?
                    .energy
            }
            None => {
                problem
                    .evaluate_with_mode(&state.sequence, &state.parameters, true, sparse_kernels)?
                    .energy
            }
        }
    };
    state.energy = energy;
    state.cost = energy + length_penalty * state.sequence.len() as f64;
    state.screen = None;
    state.weights = None;
    state.hessian = None;
    Ok(())
}
pub(super) struct ReplicaStepData {
    pub(super) row: Value,
    pub(super) move_index: usize,
    pub(super) accepted: bool,
    pub(super) proposed: Evaluated,
}
pub(super) fn step_replica(
    evaluator: &mut Evaluator<'_>,
    chain: &mut Replica,
    kernel: &Kernel,
    step: usize,
    replica: usize,
    replicas: usize,
    temperature: f64,
    mut execution: Option<&mut (dyn Execution + '_)>,
) -> Result<ReplicaStepData> {
    let ctx = &evaluator.problem.context;
    let options = evaluator.options;
    let current = &mut chain.current;
    let random = &mut chain.random;
    let (screen, weights) = (current.screened()?, current.weighed()?);
    let action = kernel.draw(
        ctx,
        &current.sequence,
        &screen.probabilities,
        &weights.removal,
        random,
    )?;
    let sequence = action.apply(&current.sequence);
    let initial = if options.warm_start {
        action.transfer(&current.parameters, options.replace_inherit)
    } else {
        vec![0.; sequence.len()]
    };
    let init_bits = (options.warm_start && options.memoize_warm).then(|| parameter_bits(&initial));
    let start_hessian = if options.carry_hessian && options.warm_start {
        current.hessian.as_deref().and_then(|h| {
            adapt_hessian(
                action,
                h,
                current.sequence.len(),
                options.optimizer.method,
                options.replace_inherit,
            )
        })
    } else {
        None
    };
    let (mut proposed, proposed_cache) = evaluator.evaluate(
        sequence,
        initial,
        start_hessian.as_deref(),
        step + 1,
        execution.as_deref_mut(),
    )?;
    let forward = kernel.log_q(
        ctx,
        &proposed.sequence,
        &current.sequence,
        &screen.probabilities,
        &weights.removal,
    )?;
    let mut drawn_log_u = None;
    let mut early_rejected = false;
    if options.early_reject {
        let bound =
            kernel::log_acceptance(temperature, current.cost, proposed.cost, forward, 1e-12);
        if bound < 0. {
            let log_u = random.uniform().ln();
            if log_u >= bound {
                early_rejected = true;
            } else {
                drawn_log_u = Some(log_u);
            }
        }
    }
    let (record_outcome, accepted) = if early_rejected {
        (Some((forward, None, false)), false)
    } else {
        if matches!(action.kind, Move::Insert | Move::Replace) {
            evaluator.weigh(
                &mut proposed,
                init_bits.as_deref(),
                execution.as_deref_mut(),
            )?;
        }
        if matches!(action.kind, Move::Delete | Move::Replace) {
            evaluator.screen(
                &mut proposed,
                proposed_cache.as_ref(),
                init_bits.as_deref(),
                execution.as_deref_mut(),
            )?;
        }
        let reverse = kernel.log_q_matched(
            &current.sequence,
            &proposed.sequence,
            proposed.screen.as_ref().map(|s| &s.probabilities[..]),
            proposed.weights.as_ref().map(|w| &w.removal[..]),
        )?;
        let ratio =
            kernel::log_acceptance(temperature, current.cost, proposed.cost, forward, reverse);
        let accepted = if ratio >= 0. {
            true
        } else {
            let log_u = drawn_log_u.unwrap_or_else(|| random.uniform().ln());
            log_u < ratio
        };
        (Some((forward, Some((reverse, ratio)), accepted)), accepted)
    };
    if accepted {
        evaluator.complete(
            &mut proposed,
            proposed_cache.as_ref(),
            init_bits.as_deref(),
            execution,
        )?;
    }
    let mut row = record(
        step + 1,
        action,
        current,
        &proposed,
        temperature,
        record_outcome,
        options.record,
    )?;
    if replicas > 1 {
        row["replica"] = json!(replica);
    }
    let move_index = match action.kind {
        Move::Insert => 0,
        Move::Delete => 1,
        Move::Replace => 2,
    };
    let recorded_proposed = if accepted {
        let copy = proposed.clone();
        *current = proposed;
        copy
    } else {
        proposed
    };
    Ok(ReplicaStepData {
        row,
        move_index,
        accepted,
        proposed: recorded_proposed,
    })
}
