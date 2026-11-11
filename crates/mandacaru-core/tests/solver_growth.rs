//! ADAPT growth strategies of the native solver: TETRIS (`SolveOptions::tetris`, several
//! operators with disjoint qubit supports per growth step) and pruning (`SolveOptions::prune`,
//! one irrelevant operator removed after a step).
mod solver_support;

use mandacaru_core::{
    quantum::{solver::prune_candidate, CachedProblem, SolveOptions},
    Context,
};

fn adapt(tetris: bool, prune: bool, max_growth: usize) -> SolveOptions {
    SolveOptions {
        adapt: true,
        tetris,
        prune,
        max_growth,
        gradient_tolerance: Some(1e-6),
        ..Default::default()
    }
}

fn support(problem: &CachedProblem, index: usize) -> u64 {
    problem.pool[index]
        .operator
        .terms
        .iter()
        .fold(0, |mask, t| mask | t.x | t.z)
}

#[test]
fn a_tetris_step_appends_operators_on_disjoint_qubits() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    let problem = solver_support::problem(&ctx, "fermionic");
    let mut steps: Vec<Vec<usize>> = Vec::new();
    let mut observer = |indices: &[usize], _: &[f64], _: f64| {
        steps.push(indices.to_vec());
        Ok(())
    };
    let tetris = problem
        .solve_observed(&adapt(true, false, 30), None, None, &mut observer)
        .unwrap();
    let single = problem.solve(&adapt(false, false, 30), None).unwrap();
    let mut layers = 0;
    let mut previous = 0;
    for (step, indices) in steps.iter().enumerate() {
        let appended = &indices[previous..];
        // The first appended operator is the one the step selected on the largest gradient.
        assert_eq!(appended[0], tetris.iterations[step].pool_index);
        let mut covered = 0u64;
        for &index in appended {
            let mask = support(&problem, index);
            assert_eq!(mask & covered, 0, "step {step}: overlapping supports");
            covered |= mask;
        }
        layers += usize::from(appended.len() > 1);
        previous = indices.len();
    }
    eprintln!(
        "TETRIS {} steps, {} operators, {:.12}; single {} steps, {:.12}",
        tetris.iterations.len(),
        tetris.indices.len(),
        tetris.energy,
        single.iterations.len(),
        single.energy
    );
    assert!(
        layers > 0,
        "some step appended a layer of several operators"
    );
    assert!(tetris.success && single.success);
    // Both converge on the same ground state; TETRIS in fewer growth steps.
    assert!((tetris.energy - single.energy).abs() < 1e-6);
    assert!(tetris.iterations.len() < single.iterations.len());
    drop((tetris, single, problem));
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_pruning_rule_takes_the_early_small_amplitude_and_never_the_newest() {
    // The ranking exp(-10 x) / theta^2 puts the tiny second amplitude first, and it is under
    // a tenth of the mean of the last four (0.18775).
    assert_eq!(prune_candidate(&[0.5, 0.001, 0.3, 0.2, 0.25]), Some(1));
    // Every amplitude above the threshold: nothing is removed.
    assert_eq!(prune_candidate(&[0.5, 0.2, 0.3, 0.25]), None);
    // The newest is never a candidate, even at zero.
    assert_eq!(prune_candidate(&[0.5, 0.3, 0.0]), None);
    // Too short to rank.
    assert_eq!(prune_candidate(&[0.0, 0.4]), None);
    // A zero amplitude among the earlier operators ranks first.
    assert_eq!(prune_candidate(&[0.4, 0.3, 0.0, 0.5, 0.2]), Some(2));
}

/// Pruning removes what an under-converged inner optimization leaves behind (two optimizer
/// iterations per growth step here): the ansatz is compacted at the plain run's energy, and every
/// recorded energy is that of the ansatz the step kept.
#[test]
fn a_pruned_run_compacts_a_loosely_optimized_ansatz_at_the_same_energy() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    let problem = solver_support::problem(&ctx, "fermionic");
    let loose = |prune| {
        let mut options = adapt(false, prune, 30);
        options.optimizer.max_iterations = 2;
        options.optimizer.tolerance = Some(1e-5);
        options
    };
    let plain = problem.solve(&loose(false), None).unwrap();
    let mut steps: Vec<(Vec<usize>, Vec<f64>, f64)> = Vec::new();
    let mut observer = |indices: &[usize], parameters: &[f64], energy: f64| {
        steps.push((indices.to_vec(), parameters.to_vec(), energy));
        Ok(())
    };
    let pruned = problem
        .solve_observed(&loose(true), None, None, &mut observer)
        .unwrap();
    eprintln!(
        "plain {} operators {:.12}; pruned {} operators {:.12}, removed {:?}",
        plain.indices.len(),
        plain.energy,
        pruned.indices.len(),
        pruned.energy,
        pruned.pruned
    );
    assert!(!pruned.pruned.is_empty());
    assert!(pruned.indices.len() < plain.indices.len());
    assert!((pruned.energy - plain.energy).abs() < 1e-6);
    for (indices, parameters, energy) in &steps {
        let value = problem.evaluate(indices, parameters, true).unwrap().energy;
        assert!((value - energy).abs() < 1e-12);
    }
    // One operator appended per step: the kept ones are those minus the removed.
    assert_eq!(
        pruned.indices.len() + pruned.pruned.len(),
        pruned.iterations.len()
    );
    assert!(pruned
        .pruned
        .iter()
        .all(|(step, _)| *step >= 1 && *step <= pruned.iterations.len()));
    drop((plain, pruned, problem));
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn growth_strategies_need_a_fresh_adapt_run() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    let problem = solver_support::problem(&ctx, "fermionic");
    let fixed = SolveOptions {
        tetris: true,
        ..Default::default()
    };
    assert!(problem.solve(&fixed, None).is_err());
    let resumed = problem.solve_sequence(&adapt(false, true, 5), Some(&[0.1]), Some(&[0]));
    assert!(resumed.is_err());
    drop(problem);
    assert_eq!(ctx.memory_used(), 0);
}
