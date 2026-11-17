//! Detailed balance of the architecture search, checked algebraically on every state.
//!
//! A small model (six qubits, one alpha and one beta electron in three orbitals, the eight
//! operators of the fermionic pool with singles, `max_length = 2`) has exactly 73 sequences. With
//! `warm_start = false` the driver's cost, gradient proposal and removal weights are functions of the
//! sequence alone, and a long recorded chain at a fixed scalar temperature visits every sequence as a
//! source. The test collects those per-state data from the driver's own records, builds the
//! transition matrix of the whole chain from the kernel's own `log_q` and `log_acceptance` (the
//! functions the driver calls), and asserts `pi_i P_ij = pi_j P_ji` and `pi P = pi` for
//! `pi_i ~ exp(-F_i / T)` to rounding. No frequency statistic is involved: a statistic on a finite
//! chain cannot tell an exact kernel from a biased one that is close, the algebra can.
//!
//! Two proposals are covered, the gradient softmax with move weights 1/2/3 and the uniform
//! proposal with 2/1/1. Dropping the proposal-ratio term from the acceptance must break the
//! balance, which proves the check sees the term it is there to guard.
//!
//! With two replicas (parallel tempering) at the fixed temperatures `T` and `2 T`, each replica's
//! step is the single chain's kernel at its own temperature, and the swap of the two states is
//! checked to balance the product target `pi_T x pi_2T` on all 73 x 73 pairs, with every recorded
//! swap's log acceptance reproduced from the per-sequence costs.
//!
//! Not a test, on purpose: `warm_start = true` (the default). Its relaxation starts from the
//! current angles, so a sequence's cost, probabilities and importances depend on the path that
//! reached it, there is no fixed target, and the balance does not hold (on H2 with a flat double
//! excitation manifold the visit frequency of one sequence was 60 percent above `pi`). The default
//! chain is an optimizer, not a sampler; the tutorial says so.
mod search_support;
use mandacaru_core::{
    quantum::{
        optimizer,
        search::{
            kernel::{self, Kernel},
            vasqa::SearchOptions,
        },
    },
    Context,
};
use search_support::architectures;
use std::collections::BTreeMap;

const TEMPERATURE: f64 = 0.5;
const LENGTH_PENALTY: f64 = 0.02;

/// What the driver evaluated for one sequence.
struct State {
    cost: f64,
    probabilities: Vec<f64>,
    removal: Vec<f64>,
}

fn indices(value: &serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

fn floats(value: &serde_json::Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

/// The transition matrix of the chain over `states`, from the kernel's `log_q` and the driver's
/// acceptance rule. `with_ratio = false` drops the proposal-ratio term (reverse set to forward).
fn transition(
    context: &Context,
    kernel: &Kernel,
    states: &[Vec<usize>],
    data: &BTreeMap<Vec<usize>, State>,
    with_ratio: bool,
    temperature: f64,
) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.; states.len()]; states.len()];
    for (i, source) in states.iter().enumerate() {
        let from = &data[source];
        let mut moved = 0.;
        for (j, target) in states.iter().enumerate() {
            if i == j {
                continue;
            }
            let forward = kernel
                .log_q(context, target, source, &from.probabilities, &from.removal)
                .unwrap();
            if forward == f64::NEG_INFINITY {
                continue;
            }
            let to = &data[target];
            let reverse = if with_ratio {
                kernel
                    .log_q(context, source, target, &to.probabilities, &to.removal)
                    .unwrap()
            } else {
                forward
            };
            let ratio = kernel::log_acceptance(temperature, from.cost, to.cost, forward, reverse);
            matrix[i][j] = forward.exp() * ratio.min(0.).exp();
            moved += matrix[i][j];
        }
        matrix[i][i] = 1. - moved;
    }
    matrix
}

fn boltzmann(
    states: &[Vec<usize>],
    data: &BTreeMap<Vec<usize>, State>,
    temperature: f64,
) -> Vec<f64> {
    let lowest = states
        .iter()
        .map(|s| data[s].cost)
        .fold(f64::INFINITY, f64::min);
    let weights: Vec<f64> = states
        .iter()
        .map(|s| (-(data[s].cost - lowest) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    weights.into_iter().map(|w| w / total).collect()
}

fn worst_imbalance(matrix: &[Vec<f64>], pi: &[f64]) -> f64 {
    let mut worst = 0_f64;
    for i in 0..pi.len() {
        for j in 0..pi.len() {
            worst = worst.max((pi[i] * matrix[i][j] - pi[j] * matrix[j][i]).abs());
        }
    }
    worst
}

fn check(uniform: bool, move_weights: [f64; 3], seed: u64) {
    check_with(uniform, move_weights, seed, None);
}

fn check_with(
    uniform: bool,
    move_weights: [f64; 3],
    seed: u64,
    learned: Option<&mandacaru_core::quantum::search::learning::training::Model>,
) {
    let context = Context::new(Some(1), 512 << 20).unwrap();
    let problem = search_support::problem(&context);
    let pool = problem.pool.len();
    let options = SearchOptions {
        // Long enough that the gradient chain (seed 11) sources all 73 sequences; 4000 missed [6, 4].
        max_steps: 20000,
        max_length: 2,
        move_weights,
        temperatures: [TEMPERATURE; 2],
        length_penalty: LENGTH_PENALTY,
        proposal_temperature: 0.3,
        uniform,
        warm_start: false,
        record: true,
        seed,
        optimizer: optimizer::Options {
            max_evaluations: 300,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = problem
        .search_with_model(&options, &[], &[], learned)
        .unwrap();
    let kernel = Kernel::new(pool, options.max_length, options.move_weights).unwrap();

    // The per-sequence data, from the first time each sequence is a source; every later row, and
    // every row's proposed sequence, must agree with it (the cost is a function of the sequence).
    let mut data = BTreeMap::<Vec<usize>, State>::new();
    for row in &result.records {
        data.entry(indices(&row["source"]))
            .or_insert_with(|| State {
                cost: row["source_cost"].as_f64().unwrap(),
                probabilities: floats(&row["source_probabilities"]),
                removal: floats(&row["source_removal"]),
            });
    }
    let states = architectures(pool, options.max_length);
    assert_eq!(states.len(), 73);
    for state in &states {
        assert!(
            data.contains_key(state),
            "the chain never sourced {state:?}: lengthen it"
        );
    }
    for row in &result.records {
        let source = &data[&indices(&row["source"])];
        assert_eq!(row["source_cost"].as_f64().unwrap(), source.cost);
        assert_eq!(floats(&row["source_probabilities"]), source.probabilities);
        assert_eq!(floats(&row["source_removal"]), source.removal);
        assert_eq!(
            row["cost"].as_f64().unwrap(),
            data[&indices(&row["proposed"])].cost
        );
        assert_eq!(row["temperature"].as_f64().unwrap(), TEMPERATURE);
    }

    // The recorded transitions are the ones the matrix is built from.
    for row in &result.records {
        let (source, target) = (indices(&row["source"]), indices(&row["proposed"]));
        let (from, to) = (&data[&source], &data[&target]);
        let forward = kernel
            .log_q(
                &context,
                &target,
                &source,
                &from.probabilities,
                &from.removal,
            )
            .unwrap();
        let reverse = kernel
            .log_q(&context, &source, &target, &to.probabilities, &to.removal)
            .unwrap();
        let ratio = kernel::log_acceptance(TEMPERATURE, from.cost, to.cost, forward, reverse);
        assert!((row["log_q_forward"].as_f64().unwrap() - forward).abs() < 1e-12);
        assert!((row["log_q_reverse"].as_f64().unwrap() - reverse).abs() < 1e-12);
        assert!((row["log_acceptance"].as_f64().unwrap() - ratio).abs() < 1e-12);
    }

    let pi = boltzmann(&states, &data, TEMPERATURE);
    let matrix = transition(&context, &kernel, &states, &data, true, TEMPERATURE);
    for (i, row) in matrix.iter().enumerate() {
        assert!(
            (row.iter().sum::<f64>() - 1.).abs() < 1e-12,
            "row {i} is not a distribution"
        );
        assert!(
            row.iter().all(|p| *p >= -1e-12),
            "row {i} has a negative entry"
        );
    }
    let imbalance = worst_imbalance(&matrix, &pi);
    assert!(imbalance < 1e-12, "detailed balance fails by {imbalance:e}");
    for j in 0..states.len() {
        let stationary: f64 = (0..states.len()).map(|i| pi[i] * matrix[i][j]).sum();
        assert!(
            (stationary - pi[j]).abs() < 1e-12,
            "pi P differs from pi at {:?}",
            states[j]
        );
    }

    let biased = transition(&context, &kernel, &states, &data, false, TEMPERATURE);
    let broken = worst_imbalance(&biased, &pi);
    assert!(
        broken > 1e-4,
        "without the proposal ratio the balance should break (imbalance {broken:e})"
    );
    drop(result);
    drop(problem);
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn the_gradient_proposal_chain_balances_on_all_73_sequences() {
    check(false, [1., 2., 3.], 11);
}

#[test]
fn the_uniform_proposal_chain_balances_on_all_73_sequences() {
    check(true, [2., 1., 1.], 5);
}

#[test]
fn the_multimove_edit_proposal_chain_balances_on_all_73_sequences() {
    use mandacaru_core::quantum::search::learning::{
        ranker::PairwiseRanker,
        training::{EditModel, Model, EDIT_FEATURES},
    };

    let make_head = |grad_w: f64, ov_w: f64, curv_w: f64, pos_w: f64| PairwiseRanker {
        weights: [0., 0., 0., 0., 0., grad_w, 0., 0., 0., ov_w, curv_w, pos_w],
        mean: [0.; EDIT_FEATURES],
        std: [1.; EDIT_FEATURES],
        pairs: 10,
    };
    let edit = EditModel {
        insert: Some(make_head(0.25, 0.15, -0.05, 0.05)),
        delete: Some(make_head(-0.2, 0.1, 0.05, -0.15)),
        replace: None,
        l2: 1e-2,
    };
    let model = Model::from_edit(edit, 1.0).unwrap();
    assert!(model.uses_curvatures());
    check_with(false, [1., 2., 3.], 11, Some(&model));
}

#[test]
fn two_replicas_keep_their_targets_and_the_swap_balances_the_product() {
    let context = Context::new(Some(1), 512 << 20).unwrap();
    let problem = search_support::problem(&context);
    let pool = problem.pool.len();
    let spacing = 2.;
    let temperatures = [TEMPERATURE, TEMPERATURE * spacing];
    let options = SearchOptions {
        max_steps: 20000,
        max_length: 2,
        move_weights: [1., 2., 3.],
        temperatures: [TEMPERATURE; 2],
        length_penalty: LENGTH_PENALTY,
        proposal_temperature: 0.3,
        warm_start: false,
        record: true,
        seed: 11,
        replicas: 2,
        replica_spacing: spacing,
        optimizer: optimizer::Options {
            max_evaluations: 300,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = problem.search(&options, &[], &[]).unwrap();
    let kernel = Kernel::new(pool, options.max_length, options.move_weights).unwrap();
    assert_eq!(result.steps.len(), 2 * options.max_steps);
    assert_eq!(result.swaps.len(), options.max_steps);

    // One data set for both replicas: with `warm_start = false` a sequence's cost and proposal are
    // the same whichever replica evaluates it.
    let mut data = BTreeMap::<Vec<usize>, State>::new();
    for row in &result.records {
        data.entry(indices(&row["source"]))
            .or_insert_with(|| State {
                cost: row["source_cost"].as_f64().unwrap(),
                probabilities: floats(&row["source_probabilities"]),
                removal: floats(&row["source_removal"]),
            });
    }
    let states = architectures(pool, options.max_length);
    for state in &states {
        assert!(
            data.contains_key(state),
            "no replica sourced {state:?}: lengthen the chain"
        );
    }
    for row in &result.records {
        let replica = row["replica"].as_u64().unwrap() as usize;
        let temperature = temperatures[replica];
        assert_eq!(row["temperature"].as_f64().unwrap(), temperature);
        let (source, target) = (indices(&row["source"]), indices(&row["proposed"]));
        let (from, to) = (&data[&source], &data[&target]);
        assert_eq!(row["source_cost"].as_f64().unwrap(), from.cost);
        assert_eq!(floats(&row["source_probabilities"]), from.probabilities);
        assert_eq!(floats(&row["source_removal"]), from.removal);
        let forward = kernel
            .log_q(
                &context,
                &target,
                &source,
                &from.probabilities,
                &from.removal,
            )
            .unwrap();
        let reverse = kernel
            .log_q(&context, &source, &target, &to.probabilities, &to.removal)
            .unwrap();
        let ratio = kernel::log_acceptance(temperature, from.cost, to.cost, forward, reverse);
        assert!((row["log_acceptance"].as_f64().unwrap() - ratio).abs() < 1e-12);
    }

    // Each replica's own step balances its own target.
    let pis: Vec<Vec<f64>> = temperatures
        .iter()
        .map(|&t| {
            let pi = boltzmann(&states, &data, t);
            let matrix = transition(&context, &kernel, &states, &data, true, t);
            let imbalance = worst_imbalance(&matrix, &pi);
            assert!(
                imbalance < 1e-12,
                "replica at T = {t} fails detailed balance by {imbalance:e}"
            );
            pi
        })
        .collect();

    // The swap balances the product target on every pair of states.
    let swap = |a: usize, b: usize| {
        let log = (1. / temperatures[0] - 1. / temperatures[1])
            * (data[&states[a]].cost - data[&states[b]].cost);
        log.min(0.).exp()
    };
    for a in 0..states.len() {
        for b in 0..states.len() {
            let forward = pis[0][a] * pis[1][b] * swap(a, b);
            let backward = pis[0][b] * pis[1][a] * swap(b, a);
            assert!(
                (forward - backward).abs() <= 1e-12 * forward.max(backward),
                "the swap of {:?} and {:?} is not balanced",
                states[a],
                states[b]
            );
        }
    }

    // Every recorded swap reads the replicas' current states, and an accepted one exchanges them:
    // each replica's next row starts where the swap left it.
    let mut accepted = 0;
    let rows: Vec<_> = result.records.iter().collect();
    for (step, row) in result.swaps.iter().enumerate() {
        let sequences: Vec<Vec<usize>> = row["sequences"]
            .as_array()
            .unwrap()
            .iter()
            .map(indices)
            .collect();
        let costs = floats(&row["costs"]);
        assert_eq!(
            costs,
            vec![data[&sequences[0]].cost, data[&sequences[1]].cost]
        );
        let expected = (1. / temperatures[0] - 1. / temperatures[1]) * (costs[0] - costs[1]);
        assert!((row["log_acceptance"].as_f64().unwrap() - expected).abs() < 1e-12);
        let mut after: Vec<Vec<usize>> = (0..2)
            .map(|r| {
                let moved = rows[2 * step + r];
                assert_eq!(moved["replica"].as_u64().unwrap() as usize, r);
                if moved["accepted"] == true {
                    indices(&moved["proposed"])
                } else {
                    indices(&moved["source"])
                }
            })
            .collect();
        assert_eq!(after, sequences);
        if row["accepted"] == true {
            accepted += 1;
            after.swap(0, 1);
        }
        if step + 1 < result.swaps.len() {
            for (r, sequence) in after.iter().enumerate() {
                assert_eq!(&indices(&rows[2 * (step + 1) + r]["source"]), sequence);
            }
        }
    }
    assert!(
        accepted > 0 && accepted < result.swaps.len(),
        "{accepted} of {} swaps",
        result.swaps.len()
    );
    assert_eq!(result.swap_acceptance, (accepted, result.swaps.len()));
    drop(result);
    drop(problem);
    assert_eq!(context.memory_used(), 0);
}
