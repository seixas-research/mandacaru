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
//! Not a test, on purpose: `warm_start = true` (the default). Its relaxation starts from the
//! current angles, so a sequence's cost, probabilities and importances depend on the path that
//! reached it, there is no fixed target, and the balance does not hold (on H2 with a flat double
//! excitation manifold the visit frequency of one sequence was 60 percent above `pi`). The default
//! chain is an optimizer, not a sampler; the tutorial says so.
use mandacaru_core::{
    quantum::{
        optimizer,
        search::{
            driver::Options,
            kernel::{self, Kernel},
        },
        CachedProblem, Encoding, Operator, C64,
    },
    Context,
};
use std::collections::BTreeMap;

const TEMPERATURE: f64 = 0.5;
const LENGTH_PENALTY: f64 = 0.02;
const MODES: usize = 6;

/// A number-conserving Hermitian Hamiltonian on three spatial orbitals (modes 0-2 alpha, 3-5 beta)
/// with orbital energies, hopping, an on-site and an inter-orbital density interaction.
fn hamiltonian(context: &Context) -> Operator {
    let mut terms = BTreeMap::<String, f64>::new();
    let string = |paulis: &[(usize, char)]| -> String {
        let mut chars = vec!['I'; MODES];
        for &(mode, pauli) in paulis {
            chars[mode] = pauli;
        }
        chars.into_iter().collect()
    };
    let mut add = |key: String, value: f64| *terms.entry(key).or_default() += value;
    // n_p = (1 - Z_p) / 2.
    for offset in [0, 3] {
        for (orbital, energy) in [0.0, 0.5, 1.1].into_iter().enumerate() {
            add(string(&[]), energy / 2.);
            add(string(&[(offset + orbital, 'Z')]), -energy / 2.);
        }
        // Hopping between orbitals p < q of one spin: (X Z..Z X + Y Z..Z Y) / 2.
        for (p, q, t) in [(0, 1, -0.35), (1, 2, -0.2), (0, 2, 0.15)] {
            for pauli in ['X', 'Y'] {
                let mut paulis = vec![(offset + p, pauli), (offset + q, pauli)];
                paulis.extend((offset + p + 1..offset + q).map(|mode| (mode, 'Z')));
                add(string(&paulis), t / 2.);
            }
        }
    }
    // Density-density terms n_a n_b = (1 - Z_a - Z_b + Z_a Z_b) / 4.
    let mut density = |a: usize, b: usize, u: f64| {
        add(string(&[]), u / 4.);
        add(string(&[(a, 'Z')]), -u / 4.);
        add(string(&[(b, 'Z')]), -u / 4.);
        add(string(&[(a, 'Z'), (b, 'Z')]), u / 4.);
    };
    for p in 0..3 {
        density(p, 3 + p, 0.6);
    }
    density(0, 4, 0.25);
    density(1, 5, 0.15);
    density(2, 3, 0.3);
    let terms: Vec<_> = terms
        .into_iter()
        .map(|(key, value)| (key, C64::new(value, 0.)))
        .collect();
    Operator::new(context, MODES, &terms).unwrap()
}

fn architectures(pool: usize, maximum: usize) -> Vec<Vec<usize>> {
    let mut all = vec![vec![]];
    let mut last = vec![vec![]];
    for _ in 0..maximum {
        let next: Vec<Vec<usize>> = last
            .iter()
            .flat_map(|source: &Vec<usize>| {
                (0..pool).map(move |op| {
                    let mut next = source.clone();
                    next.push(op);
                    next
                })
            })
            .collect();
        all.extend(next.clone());
        last = next;
    }
    all
}

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
            let ratio = kernel::log_acceptance(TEMPERATURE, from.cost, to.cost, forward, reverse);
            matrix[i][j] = forward.exp() * ratio.min(0.).exp();
            moved += matrix[i][j];
        }
        matrix[i][i] = 1. - moved;
    }
    matrix
}

fn boltzmann(states: &[Vec<usize>], data: &BTreeMap<Vec<usize>, State>) -> Vec<f64> {
    let lowest = states
        .iter()
        .map(|s| data[s].cost)
        .fold(f64::INFINITY, f64::min);
    let weights: Vec<f64> = states
        .iter()
        .map(|s| (-(data[s].cost - lowest) / TEMPERATURE).exp())
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
    let context = Context::new(Some(1), 512 << 20).unwrap();
    let problem = CachedProblem::new(
        &context,
        Encoding::new("jordan_wigner", MODES, [1, 1]).unwrap(),
        hamiltonian(&context),
        true,
        false,
        true,
    )
    .unwrap();
    let pool = problem.pool.len();
    assert_eq!(pool, 8, "the model's pool is the eight singles and doubles");
    let options = Options {
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
    let result = problem.search(&options, &[], &[]).unwrap();
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

    let pi = boltzmann(&states, &data);
    let matrix = transition(&context, &kernel, &states, &data, true);
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

    let biased = transition(&context, &kernel, &states, &data, false);
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
