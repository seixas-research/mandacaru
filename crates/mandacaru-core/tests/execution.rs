//! A circuit executor that answers `Stop` ends the solve at its best point.
use mandacaru_core::{
    quantum::{
        execution::{CircuitExecution, Response},
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
    fn answer(&self, angles: &[f64], observables: &[String]) -> Response {
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
        Response::Expectations(values)
    }
    fn call(&self, angles: &[f64], observables: &[String]) -> Response {
        if self.stopped_on.get() != 0 {
            self.after_stop.set(self.after_stop.get() + 1);
        }
        self.calls.set(self.calls.get() + 1);
        if self.stop_at.get() != 0 && self.calls.get() == self.stop_at.get() {
            self.stopped_on.set(self.calls.get());
            return Response::Stop("test budget".into());
        }
        self.answer(angles, observables)
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
            let mut call = |_: &[String], angles: &[f64], observables: &[String]| {
                Ok(executor.call(angles, observables))
            };
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
            assert!(result.state.len() == problem.basis.len(), "{label}");
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
        let mut call = |_: &[String], angles: &[f64], observables: &[String]| {
            Ok(executor.call(angles, observables))
        };
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
    let mut call = |labels: &[String], angles: &[f64], observables: &[String]| {
        if executor.calls.get() == 0 {
            executor.stop_at.set(1 + 2 * labels.len() + 3);
        }
        Ok(executor.call(angles, observables))
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
    let mut call = |_: &[String], angles: &[f64], observables: &[String]| {
        Ok(executor.call(angles, observables))
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
    assert!(result.budget_stopped && !result.success);
    assert!(result.indices.is_empty() && result.parameters.is_empty());
    assert_eq!(executor.calls.get(), 2);
    assert_eq!(result.energy, executor.energies.borrow()[0]);

    // A stop in the middle of the first operator's optimization keeps it at
    // its best point.
    let executor = Executor::new(&problem, 6);
    let mut call = |_: &[String], angles: &[f64], observables: &[String]| {
        Ok(executor.call(angles, observables))
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
    let mut call = |_: &[String], angles: &[f64], observables: &[String]| {
        if stopped.get() {
            after.set(after.get() + 1);
        }
        if observables.len() != executor.0.measured_labels {
            stopped.set(true);
            return Ok(Response::Stop("test budget".into()));
        }
        Ok(executor.0.answer(angles, observables))
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
    let mut call = |_: &[String], angles: &[f64], observables: &[String]| {
        Ok(executor.call(angles, observables))
    };
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
    let mut call = |_: &[String], angles: &[f64], observables: &[String]| {
        if observables.len() != executor.measured_labels && pool_call.get() == 0 {
            pool_call.set(executor.calls.get() + 1);
        }
        Ok(executor.call(angles, observables))
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
    let mut call = |_: &[String], angles: &[f64], observables: &[String]| {
        Ok(executor.call(angles, observables))
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
