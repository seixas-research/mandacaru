//! The pool gradients of a measured ADAPT step: one executor call whatever the batch size, the
//! same values as forming every commutator at once, and a memory bound that does not grow with
//! the pool.
use mandacaru_core::{
    quantum::{
        execution::{CircuitExecution, Request, Response},
        pool_measurement::PoolWalk,
        solver::Execution,
        CachedProblem, Encoding, Operator, Space, C64,
    },
    Context,
};
use std::{cell::Cell, collections::BTreeSet};

const N: usize = 50;

/// Spin-block-preserving hops and on-site energies on 25 spatial orbitals, `N` qubits (JW).
fn hamiltonian(ctx: &Context) -> Operator {
    let mut terms = Vec::new();
    for j in 0..N {
        let label: String = (0..N).map(|k| if k == j { 'Z' } else { 'I' }).collect();
        terms.push((label, C64::new(0.1 * (j + 1) as f64, 0.)));
    }
    for j in (0..N - 1).filter(|&j| j != N / 2 - 1) {
        for letter in ['X', 'Y'] {
            let label: String = (0..N)
                .map(|k| if k == j || k == j + 1 { letter } else { 'I' })
                .collect();
            terms.push((label, C64::new(0.3, 0.)));
        }
    }
    Operator::new(ctx, N, &terms).unwrap()
}
fn problem(ctx: &Context, particles: [usize; 2]) -> CachedProblem {
    CachedProblem::new_in(
        ctx,
        Encoding::new("jordan_wigner", N, particles).unwrap(),
        hamiltonian(ctx),
        Space::Pauli {
            number_conserving: true,
        },
        false,
        true,
    )
    .unwrap()
}

/// What a device would answer for a string: smooth in the rotation angles, and a function of the
/// string alone, so a different order or batching of the commutators cannot change it.
fn expectation(label: &str, total: f64) -> f64 {
    let weight: f64 = label
        .bytes()
        .enumerate()
        .map(|(k, b)| (k as f64 + 1.) * f64::from(b))
        .sum();
    (1e-3 * weight + total).cos()
}
fn answer(requests: &[Request]) -> Response {
    Response::Expectations(
        requests
            .iter()
            .map(|r| {
                let total: f64 = r.angles.iter().sum();
                r.observables
                    .iter()
                    .map(|l| expectation(l, total))
                    .collect()
            })
            .collect(),
    )
}

/// `sum_t c_t <P_t>` of every `[H, A_k]` formed whole, the way the executor did before it
/// streamed them.
fn all_at_once(problem: &CachedProblem, total: f64) -> Vec<f64> {
    problem
        .pool
        .iter()
        .map(|g| {
            problem
                .hamiltonian
                .commutator(&problem.context, &g.operator)
                .unwrap()
                .labels()
                .iter()
                .map(|(label, c)| {
                    c.re * if label.chars().all(|ch| ch == 'I') {
                        1.
                    } else {
                        expectation(label, total)
                    }
                })
                .sum()
        })
        .collect()
}

#[test]
fn the_gradients_are_the_same_at_every_batch_size_in_one_executor_call() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let problem = problem(&ctx, [1, 1]);
    let (indices, parameters) = ([0usize], [0.3]);
    let expected = all_at_once(&problem, 0.3 * problem_angle(&problem, &indices));
    assert!(expected.len() > 100 && expected.iter().any(|g| g.abs() > 1e-6));
    for bytes in [1, 4 << 10, 1 << 20, usize::MAX / 2] {
        let calls = Cell::new(0);
        let asked = Cell::new(0);
        let mut call = |requests: &[Request]| {
            calls.set(calls.get() + 1);
            asked.set(asked.get() + requests.len());
            Ok(answer(requests))
        };
        let mut execution = CircuitExecution::new(&mut call).with_batch_bytes(bytes);
        let gradients = execution
            .pool_gradients(&problem, &indices, &parameters)
            .unwrap();
        assert_eq!((calls.get(), asked.get()), (1, 1), "batch of {bytes} bytes");
        assert_eq!(gradients.len(), expected.len());
        for (k, (a, b)) in gradients.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() < 1e-12, "batch {bytes}, generator {k}: {a} {b}");
        }
    }
}

/// The sum of the circuit's angles after the first generator of the ansatz is placed: the device
/// above answers from it, so the oracle needs the same number.
fn problem_angle(problem: &CachedProblem, indices: &[usize]) -> f64 {
    indices
        .iter()
        .map(|&i| {
            problem.pool[i]
                .operator
                .labels()
                .iter()
                .map(|(_, c)| c.im)
                .sum::<f64>()
        })
        .sum()
}

#[test]
fn the_labels_asked_are_the_distinct_strings_of_every_commutator() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let problem = problem(&ctx, [1, 1]);
    let mut distinct = BTreeSet::new();
    let mut nonzero = 0;
    for g in &problem.pool {
        let commutator = problem
            .hamiltonian
            .commutator(&ctx, &g.operator)
            .unwrap();
        nonzero += usize::from(!commutator.terms.is_empty());
        for (label, _) in commutator.labels() {
            if !label.chars().all(|c| c == 'I') {
                distinct.insert(label);
            }
        }
    }
    for bytes in [1, 1 << 20] {
        let paulis = problem.pool_paulis(bytes).unwrap();
        assert_eq!(paulis.measured, nonzero);
        let labels: BTreeSet<_> = paulis.set.labels().into_iter().collect();
        assert_eq!(labels, distinct, "batch of {bytes} bytes");
        assert_eq!(paulis.set.len(), distinct.len());
    }
}

#[test]
fn a_ceiling_stops_the_walk_on_a_proven_excess_and_never_on_a_fit() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let problem = problem(&ctx, [1, 1]);
    for bytes in [1, 1 << 20] {
        let all = problem.pool_paulis(bytes).unwrap();
        let total = all.set.len();
        // A ceiling at the exact count, and above it, is not an excess: the plan is identical.
        for limit in [total, total + 1, usize::MAX] {
            match problem.pool_paulis_within(bytes, Some(limit)).unwrap() {
                PoolWalk::Planned(plan) => {
                    assert_eq!(plan.set.labels(), all.set.labels());
                    assert_eq!(plan.measured, all.measured);
                }
                PoolWalk::Over { .. } => panic!("a ceiling of {limit} refused {total} strings"),
            }
        }
        // One below the count is an excess; what the walk reports is a count it really saw,
        // so it is over the ceiling and never over the total.
        match problem.pool_paulis_within(bytes, Some(total - 1)).unwrap() {
            PoolWalk::Over {
                distinct,
                generators,
            } => {
                assert!(distinct > total - 1 && distinct <= total);
                assert!(generators >= 1 && generators <= problem.pool.len());
            }
            PoolWalk::Planned(_) => panic!("{total} strings fit a ceiling of {}", total - 1),
        }
    }
    // With one commutator per batch, a tiny ceiling ends the walk well before the pool's end.
    match problem.pool_paulis_within(1, Some(20)).unwrap() {
        PoolWalk::Over { generators, .. } => assert!(generators < problem.pool.len() / 4),
        PoolWalk::Planned(_) => panic!("20 strings should be exceeded"),
    }
}

/// The peak the context saw while `pool_gradients` ran on a fresh problem, in bytes, and the
/// gradients it returned.
fn peak_of(bytes: usize, particles: [usize; 2]) -> (usize, Vec<f64>) {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let problem = problem(&ctx, particles);
    let built = ctx.memory_peak();
    let mut call = |requests: &[Request]| Ok(answer(requests));
    let mut execution = CircuitExecution::new(&mut call).with_batch_bytes(bytes);
    let gradients = execution.pool_gradients(&problem, &[0], &[0.3]).unwrap();
    let during = ctx.memory_peak();
    eprintln!(
        "particles {particles:?}, pool {}, batch {bytes} bytes: built {} B, with the step {} B",
        problem.pool.len(),
        built,
        during
    );
    (during, gradients)
}

#[test]
fn the_memory_of_a_step_is_bounded_by_the_batch_not_by_the_pool() {
    let (streamed, a) = peak_of(1 << 20, [2, 2]);
    let (whole, b) = peak_of(usize::MAX / 2, [2, 2]);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() < 1e-12);
    }
    assert!(
        streamed * 2 < whole,
        "streamed {streamed} B against whole {whole} B"
    );
}
