//! A circuit executor that answers `Stop` ends the solve at its best point.
use mandacaru_core::{
    quantum::{
        execution::{CircuitExecution, Request, Response},
        optimizer::{Method, Options},
        CachedProblem, SolveOptions,
    },
    Context, Error,
};
use std::cell::{Cell, RefCell};

fn context() -> Context {
    Context::new(Some(1), 64 << 20).unwrap()
}
fn h2(ctx: &Context) -> CachedProblem {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference/h2_hamiltonian.json");
    CachedProblem::load(ctx, &fixture, true, false, true).unwrap()
}

/// A synthetic landscape: each Pauli string has a smooth expectation in the
/// rotation angles. Records every energy it hands out and refuses to answer
/// once `stop_at` calls have been made (1-based; 0 never stops).
struct Executor<'p> {
    problem: &'p CachedProblem,
    stop_at: Cell<usize>,
    calls: Cell<usize>,
    after_stop: Cell<usize>,
    /// Energy of every call that carried the Hamiltonian observables.
    energies: RefCell<Vec<f64>>,
    /// Index of the call that was refused.
    stopped_on: Cell<usize>,
    measured_labels: usize,
}
impl<'p> Executor<'p> {
    fn new(problem: &'p CachedProblem, stop_at: usize) -> Self {
        let labels = problem
            .hamiltonian
            .labels()
            .into_iter()
            .filter(|(l, _)| !l.chars().all(|c| c == 'I'))
            .count();
        Self {
            problem,
            stop_at: Cell::new(stop_at),
            calls: Cell::new(0),
            after_stop: Cell::new(0),
            energies: RefCell::new(vec![]),
            stopped_on: Cell::new(0),
            measured_labels: labels,
        }
    }
    fn answer(&self, angles: &[f64], observables: &[String]) -> Vec<f64> {
        // Rotation angles of one generator can cancel in a plain sum, so the
        // weights differ from one rotation to the next.
        let total: f64 = angles
            .iter()
            .enumerate()
            .map(|(j, a)| (1. + 0.37 * j as f64) * a)
            .sum();
        let values: Vec<f64> = observables
            .iter()
            .enumerate()
            .map(|(k, _)| (0.7 * total + 0.3 + 0.1 * k as f64).cos())
            .collect();
        if observables.len() == self.measured_labels {
            let by_label: std::collections::BTreeMap<_, _> = observables
                .iter()
                .cloned()
                .zip(values.iter().copied())
                .collect();
            let energy: f64 = self
                .problem
                .hamiltonian
                .labels()
                .iter()
                .map(|(label, c)| {
                    c.re * if label.chars().all(|c| c == 'I') {
                        1.
                    } else {
                        by_label[label]
                    }
                })
                .sum();
            self.energies.borrow_mut().push(energy);
        }
        values
    }
    /// One call carries a batch of circuits (the factorized energy sends one per basis); each
    /// is answered in turn.
    fn call(&self, requests: &[Request]) -> Response {
        if self.stopped_on.get() != 0 {
            self.after_stop.set(self.after_stop.get() + 1);
        }
        self.calls.set(self.calls.get() + 1);
        if self.stop_at.get() != 0 && self.calls.get() == self.stop_at.get() {
            self.stopped_on.set(self.calls.get());
            return Response::Stop("test budget".into());
        }
        Response::Expectations(
            requests
                .iter()
                .map(|r| self.answer(&r.angles, &r.observables))
                .collect(),
        )
    }
}

fn options(adapt: bool, method: Method, track_best: bool) -> SolveOptions {
    SolveOptions {
        adapt,
        trotter: true,
        max_growth: 4,
        optimizer: Options {
            method,
            max_iterations: 60,
            tolerance: if method == Method::Spsa {
                None
            } else {
                Some(1e-9)
            },
            track_best,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn fixed_ansatz_stop_keeps_the_best_completed_point_for_every_derivative_free_method() {
    let ctx = context();
    let problem = h2(&ctx);
    for (method, track_best) in [
        (Method::NelderMead, true),
        (Method::Cobyla, true),
        (Method::Spsa, true),
        (Method::Spsa, false),
    ] {
        for stop_at in [2, 6, 11] {
            let executor = Executor::new(&problem, stop_at);
            let mut call = |requests: &[Request]| Ok(executor.call(requests));
            let mut execution = CircuitExecution::new(&mut call);
            let result = problem
                .solve_executed(
                    &options(false, method, track_best),
                    None,
                    None,
                    &mut execution,
                )
                .unwrap();
            let label = format!("{method:?} track_best={track_best} stop_at={stop_at}");
            assert_eq!(executor.calls.get(), stop_at, "{label}: calls requested");
            assert_eq!(executor.after_stop.get(), 0, "{label}: call after the stop");
            assert!(result.budget_stopped && !result.success, "{label}");
            assert_eq!(
                result.message, "run budget exhausted: test budget",
                "{label}"
            );
            let energies = executor.energies.borrow();
            assert_eq!(energies.len(), stop_at - 1, "{label}");
            assert_eq!(result.history.len(), stop_at - 1, "{label}");
            let lowest = energies.iter().copied().fold(f64::INFINITY, f64::min);
            assert!((result.energy - lowest).abs() < 1e-12, "{label}");
            assert!(result.state.len() == problem.register("test").unwrap().len(), "{label}");
        }
    }
    drop(problem);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_stop_before_any_energy_is_measured_is_an_interruption_error() {
    let ctx = context();
    let problem = h2(&ctx);
    for adapt in [false, true] {
        let executor = Executor::new(&problem, 1);
        let mut call = |requests: &[Request]| Ok(executor.call(requests));
        let mut execution = CircuitExecution::new(&mut call);
        let outcome = problem.solve_executed(
            &options(adapt, Method::NelderMead, true),
            None,
            None,
            &mut execution,
        );
        assert!(
            matches!(outcome, Err(Error::Interrupted(ref m)) if m == "test budget"),
            "adapt={adapt}"
        );
        assert_eq!(executor.calls.get(), 1);
    }
    drop(problem);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn gradient_methods_stopped_inside_a_parameter_shift_keep_the_last_complete_point() {
    let ctx = context();
    let problem = h2(&ctx);
    // One evaluation with gradients is 1 + 2 * (rotations) calls. Stop at the
    // third call of the second evaluation, inside a parameter shift, so only
    // the first evaluation completed.
    let executor = Executor::new(&problem, 0);
    let mut call = |requests: &[Request]| {
        if executor.calls.get() == 0 {
            executor.stop_at.set(1 + 2 * requests[0].labels.len() + 3);
        }
        Ok(executor.call(requests))
    };
    let mut execution = CircuitExecution::new(&mut call);
    let result = problem
        .solve_executed(
            &options(false, Method::Bfgs, true),
            None,
            None,
            &mut execution,
        )
        .unwrap();
    assert_eq!(executor.calls.get(), executor.stop_at.get());
    assert_eq!(executor.after_stop.get(), 0);
    assert!(result.budget_stopped && !result.success);
    assert_eq!(result.history.len(), 1);
    assert!((result.energy - executor.energies.borrow()[0]).abs() < 1e-12);
}

#[test]
fn adapt_stop_ends_growth_at_the_last_accepted_ansatz() {
    let ctx = context();
    let problem = h2(&ctx);
    // Call 1 measures the reference. Calls 2.. are the first operator's
    // optimization. A stop at call 2 leaves the empty reference ansatz.
    let executor = Executor::new(&problem, 2);
    let mut call = |requests: &[Request]| Ok(executor.call(requests));
    let mut execution = CircuitExecution::new(&mut call);
    let result = problem
        .solve_executed(
            &options(true, Method::NelderMead, true),
            None,
            None,
            &mut execution,
        )
        .unwrap();
    assert!(result.budget_stopped && !result.success);
    assert!(result.indices.is_empty() && result.parameters.is_empty());
    assert_eq!(executor.calls.get(), 2);
    assert_eq!(result.energy, executor.energies.borrow()[0]);

    // A stop in the middle of the first operator's optimization keeps it at
    // its best point.
    let executor = Executor::new(&problem, 6);
    let mut call = |requests: &[Request]| Ok(executor.call(requests));
    let mut execution = CircuitExecution::new(&mut call);
    let result = problem
        .solve_executed(
            &options(true, Method::NelderMead, true),
            None,
            None,
            &mut execution,
        )
        .unwrap();
    assert!(result.budget_stopped && !result.success);
    assert_eq!(result.indices.len(), 1);
    assert_eq!(executor.calls.get(), 6);
    assert_eq!(executor.after_stop.get(), 0);
    let measured = executor.energies.borrow();
    let lowest = measured[1..].iter().copied().fold(f64::INFINITY, f64::min);
    assert!((result.energy - lowest).abs() < 1e-12);
    assert!(result.message.starts_with("run budget exhausted: "));
}

#[test]
fn adapt_stop_during_pool_gradient_measurement_keeps_the_accepted_ansatz() {
    let ctx = context();
    let problem = h2(&ctx);
    // Pool gradients are the only calls whose observable list differs from
    // the Hamiltonian's. Stop on the first of them.
    struct PoolStop<'p>(Executor<'p>);
    let inner = Executor::new(&problem, 0);
    let executor = PoolStop(inner);
    let stopped = Cell::new(false);
    let after = Cell::new(0);
    let mut call = |requests: &[Request]| {
        if stopped.get() {
            after.set(after.get() + 1);
        }
        if requests[0].observables.len() != executor.0.measured_labels {
            stopped.set(true);
            return Ok(Response::Stop("test budget".into()));
        }
        Ok(Response::Expectations(vec![executor
            .0
            .answer(&requests[0].angles, &requests[0].observables)]))
    };
    let mut execution = CircuitExecution::new(&mut call);
    let result = problem
        .solve_executed(
            &options(true, Method::NelderMead, true),
            None,
            None,
            &mut execution,
        )
        .unwrap();
    assert!(stopped.get());
    assert_eq!(after.get(), 0);
    assert!(result.budget_stopped && !result.success);
    assert_eq!(result.indices.len(), 1);
    assert_eq!(result.iterations.len(), 1);
    // The reported energy is the re-measurement of the accepted ansatz that
    // preceded the refused gradient measurement.
    assert_eq!(result.energy, *executor.0.energies.borrow().last().unwrap());
    assert!(result.message.starts_with("run budget exhausted: "));
}

#[test]
fn an_uninterrupted_executed_run_is_not_flagged() {
    let ctx = context();
    let problem = h2(&ctx);
    let executor = Executor::new(&problem, 0);
    let mut call = |requests: &[Request]| Ok(executor.call(requests));
    let mut execution = CircuitExecution::new(&mut call);
    let result = problem
        .solve_executed(
            &options(false, Method::Cobyla, true),
            None,
            None,
            &mut execution,
        )
        .unwrap();
    assert!(!result.budget_stopped);
    assert_ne!(
        result.message, "run budget exhausted: test budget",
        "an unstopped run keeps its optimizer message"
    );
}

#[test]
fn adapt_stop_when_the_next_step_is_first_measured_reports_the_optimized_energy() {
    let ctx = context();
    let problem = h2(&ctx);
    // Dry run: find the first pool-gradient call. The call before it is the
    // re-measurement of the first optimized ansatz.
    let executor = Executor::new(&problem, 0);
    let pool_call = Cell::new(0);
    let mut call = |requests: &[Request]| {
        if requests[0].observables.len() != executor.measured_labels && pool_call.get() == 0 {
            pool_call.set(executor.calls.get() + 1);
        }
        Ok(executor.call(requests))
    };
    let mut execution = CircuitExecution::new(&mut call);
    problem
        .solve_executed(
            &options(true, Method::NelderMead, true),
            None,
            None,
            &mut execution,
        )
        .unwrap();
    assert!(pool_call.get() > 3);

    let stop_at = pool_call.get() - 1;
    let executor = Executor::new(&problem, stop_at);
    let mut call = |requests: &[Request]| Ok(executor.call(requests));
    let mut execution = CircuitExecution::new(&mut call);
    let result = problem
        .solve_executed(
            &options(true, Method::NelderMead, true),
            None,
            None,
            &mut execution,
        )
        .unwrap();
    assert!(result.budget_stopped && !result.success);
    assert_eq!(executor.calls.get(), stop_at);
    assert_eq!(executor.after_stop.get(), 0);
    assert_eq!(result.indices.len(), 1);
    // The energy is the first operator's optimized one, not the reference.
    let measured = executor.energies.borrow();
    let optimized = measured[1..].iter().copied().fold(f64::INFINITY, f64::min);
    assert!((result.energy - optimized).abs() < 1e-12);
    assert!(result.energy < measured[0]);
}

#[test]
fn measurement_cost_counts_jobs_and_shots_as_the_providers_submit_them() {
    use mandacaru_core::execution::measurement_cost;
    // One Estimator job carries every basis; split at 2 bases a job, 5 bases are 3 jobs.
    assert_eq!(measurement_cost(5, None, false, 1000, 1, 1, 0).unwrap(), (1, 5000));
    assert_eq!(measurement_cost(5, Some(2), false, 1000, 1, 1, 0).unwrap(), (3, 5000));
    // Braket submits one task per basis; no basis is no job.
    assert_eq!(measurement_cost(4, None, true, 100, 1, 1, 0).unwrap(), (4, 400));
    assert_eq!(measurement_cost(0, None, false, 100, 1, 1, 0).unwrap(), (0, 0));
    // Three ZNE factors, and twirled instances sharing the shots: 32 x 64 per basis.
    assert_eq!(measurement_cost(2, None, false, 2000, 3, 32, 64).unwrap(), (1, 2 * 3 * 32 * 64));
    assert!(measurement_cost(1, Some(0), false, 1, 1, 1, 0).is_err());
    assert!(measurement_cost(u64::MAX, None, false, 2, 1, 1, 0).is_err());
}

#[test]
fn count_statistics_give_each_string_its_mean_and_standard_error() {
    use mandacaru_core::execution::count_statistics;
    let ctx = context();
    let labels: Vec<String> = ["ZZ", "ZI", "II"].iter().map(|s| s.to_string()).collect();
    let counts = vec![("00".to_string(), 3.), ("11".to_string(), 1.)];
    let (values, errors) = count_statistics(&ctx, &labels, &counts).unwrap();
    // ZZ is +1 on both outcomes; ZI is (3 - 1) / 4; the identity is exact.
    assert_eq!(values, vec![1., 0.5, 1.]);
    assert_eq!(errors[0], 0.);
    assert!((errors[1] - (0.75f64 / 4.).sqrt()).abs() < 1e-15);
    assert_eq!(errors[2], 0.);
}

#[test]
fn a_hamiltonian_is_grouped_with_its_identity_counted_once() {
    use mandacaru_core::execution::qwc_hamiltonian;
    let ctx = context();
    let terms = vec![("I".to_string(), 2., 0.), ("Z".to_string(), 1., 0.), ("X".to_string(), 1e-14, 0.)];
    let dropped = qwc_hamiltonian(&ctx, &terms, true, 1e-12).unwrap();
    assert_eq!(dropped.identity, 2.);
    assert_eq!(dropped.groups, vec![("Z".to_string(), vec![("Z".to_string(), 1., 0.)])]);
    let kept = qwc_hamiltonian(&ctx, &terms, false, 1e-12).unwrap();
    assert_eq!(kept.identity, 0.);
    let measured: f64 = kept.groups.iter().flat_map(|(_, members)| members)
        .filter(|(label, _, _)| label == "I").map(|(_, re, _)| re).sum();
    assert_eq!(measured, 2.);
}

#[test]
fn the_shot_noise_bound_is_the_coefficient_one_norm_over_the_root_shots() {
    use mandacaru_core::execution::shot_noise;
    let terms = vec![("II".to_string(), -1., 0.), ("ZI".to_string(), 0.3, 0.), ("XX".to_string(), 0., 0.4)];
    assert!((shot_noise(&terms, 100, 1e-12).unwrap() - 0.07).abs() < 1e-15);
    let ratio = shot_noise(&terms, 100, 1e-12).unwrap() / shot_noise(&terms, 400, 1e-12).unwrap();
    assert!((ratio - 2.).abs() < 1e-12);
    assert_eq!(shot_noise(&terms, 0, 1e-12).unwrap(), shot_noise(&terms, 1, 1e-12).unwrap());
}

#[test]
fn twirled_instances_share_the_shots_by_the_estimator_rule() {
    use mandacaru_core::execution::twirled_instances;
    assert_eq!(twirled_instances(1000, None, None).unwrap(), (16, 64));
    assert_eq!(twirled_instances(4096, None, None).unwrap(), (32, 128));
    assert_eq!(twirled_instances(1000, None, Some(100)).unwrap(), (10, 100));
    assert_eq!(twirled_instances(1000, Some(8), None).unwrap(), (8, 125));
    assert_eq!(twirled_instances(0, None, None).unwrap(), (1, 64));
    assert!(twirled_instances(10, Some(0), None).is_err());
}

#[test]
fn basis_chunks_keep_whole_bases_and_batches_count_strings_and_bases() {
    use mandacaru_core::execution::{basis_chunks, batch_size};
    let ctx = context();
    let labels: Vec<String> = ["ZI", "XI", "IZ", "IX"].iter().map(|s| s.to_string()).collect();
    // Two bases: ZZ (ZI, IZ) and XX (XI, IX).
    let chunks = basis_chunks(&ctx, &labels, None, Some(1)).unwrap();
    assert_eq!(chunks.len(), 2);
    let mut sorted: Vec<Vec<usize>> = chunks.into_iter().map(|mut c| { c.sort(); c }).collect();
    sorted.sort();
    assert_eq!(sorted, vec![vec![0, 2], vec![1, 3]]);
    assert_eq!(basis_chunks(&ctx, &labels, None, Some(2)).unwrap(), vec![vec![0, 1, 2, 3]]);
    assert_eq!(basis_chunks(&ctx, &labels, None, None).unwrap(), vec![vec![0, 1, 2, 3]]);
    assert!(basis_chunks(&ctx, &[], None, Some(1)).unwrap().is_empty());
    let requests = vec![vec!["ZI".to_string(), "IZ".to_string()], vec!["XX".to_string()], vec![]];
    assert_eq!(batch_size(&ctx, &requests).unwrap(), (3, 2));
}

#[test]
fn the_budget_check_reports_every_limit_a_submission_passes() {
    use mandacaru_core::execution::{budget_check, estimator_precision, mps_job_gigabytes};
    let (after, exceeded) = budget_check((1, 100), (2, 50), 3, (Some(6), None)).unwrap();
    assert_eq!(after, (7, 250));
    assert_eq!(exceeded, vec![("jobs", 7, 6)]);
    let (_, both) = budget_check((0, 0), (1, 1000), 2, (Some(1), Some(1500))).unwrap();
    assert_eq!(both, vec![("jobs", 2, 1), ("total_shots", 2000, 1500)]);
    assert!(budget_check((u64::MAX, 0), (1, 0), 1, (None, None)).is_err());
    assert_eq!(estimator_precision(0), 0.);
    assert_eq!(estimator_precision(100), 0.1);
    assert!((mps_job_gigabytes(1000) - 0.67).abs() < 1e-12);
}

#[test]
fn state_expectations_read_the_exact_value_of_each_string() {
    use mandacaru_core::{execution::state_expectations, quantum::C64};
    let ctx = context();
    // |01>: qubit zero (the most significant bit) empty, qubit one occupied.
    let mut state = vec![C64::default(); 4];
    state[1] = C64::new(1., 0.);
    let labels: Vec<String> = ["ZI", "IZ", "XI", "ZZ", "II"].iter().map(|s| s.to_string()).collect();
    let values = state_expectations(&ctx, &labels, &state).unwrap();
    assert_eq!(values, vec![1., -1., 0., -1., 1.]);
    // |+0>: <XI> = 1.
    let plus = vec![C64::new(0.5f64.sqrt(), 0.), C64::default(), C64::new(0.5f64.sqrt(), 0.), C64::default()];
    let values = state_expectations(&ctx, &["XI".to_string()], &plus).unwrap();
    assert!((values[0] - 1.).abs() < 1e-15);
    assert!(state_expectations(&ctx, &labels, &state[..2]).is_err());
}
