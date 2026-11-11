//! `SolveOptions::quenching`: `true` re-optimizes every parameter after each change (standard
//! ADAPT-VQE and VQE); `false` optimizes only what changed, ADAPT's newly appended parameters
//! with the earlier ones frozen, and a fixed ansatz in one sequential one-parameter sweep.
mod solver_support;

use mandacaru_core::{quantum::SolveOptions, Context};

fn options(adapt: bool, quenching: bool) -> SolveOptions {
    SolveOptions {
        adapt,
        quenching,
        max_growth: 6,
        gradient_tolerance: Some(1e-6),
        ..Default::default()
    }
}

#[test]
fn frozen_adapt_varies_only_the_appended_parameters() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    let problem = solver_support::problem(&ctx, "fermionic");
    for quenching in [true, false] {
        let mut steps: Vec<Vec<f64>> = Vec::new();
        let mut observer = |_: &[usize], parameters: &[f64], _: f64| {
            steps.push(parameters.to_vec());
            Ok(())
        };
        let result = problem
            .solve_observed(&options(true, quenching), None, None, &mut observer)
            .unwrap();
        assert!(
            steps.len() >= 3,
            "a multi-step ADAPT run ({} steps)",
            steps.len()
        );
        let moved = steps
            .windows(2)
            .map(|w| {
                w[0].iter()
                    .zip(&w[1])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f64, f64::max)
            })
            .fold(0f64, f64::max);
        let gradient = problem
            .evaluate(&result.indices, &result.parameters, true)
            .unwrap()
            .gradient;
        let earlier = gradient[..gradient.len() - 1]
            .iter()
            .fold(0f64, |a, g| a.max(g.abs()));
        let last = gradient.last().unwrap().abs();
        eprintln!(
            "quenching {quenching}: {} steps, energy {:.12}, earliest-angle drift {moved:.2e}, |dE/d last| {last:.2e}, max |dE/d earlier| {earlier:.2e}",
            steps.len(),
            result.energy
        );
        if quenching {
            // Joint optimization: earlier angles move, and the point is stationary in all.
            assert!(moved > 1e-6);
            assert!(earlier < 1e-5 && last < 1e-5);
        } else {
            // Frozen: every earlier angle keeps the value it had after its own step, exactly.
            assert_eq!(moved, 0.);
            // Stationary in the free coordinate only.
            assert!(last < 1e-5);
            assert!(
                earlier > 1e-4,
                "the frozen angles are not at a joint optimum"
            );
        }
    }
    drop(problem);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn one_sweep_bounds_the_joint_vqe_from_above() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    let problem = solver_support::problem(&ctx, "fermionic");
    let joint = problem.solve(&options(false, true), None).unwrap();
    let swept = problem.solve(&options(false, false), None).unwrap();
    let gradient = problem
        .evaluate(&swept.indices, &swept.parameters, false)
        .unwrap()
        .gradient;
    eprintln!(
        "VQE joint {:.12}, one sweep {:.12} ({} parameters), |dE/d last| {:.2e}: {}",
        joint.energy,
        swept.energy,
        swept.parameters.len(),
        gradient.last().unwrap().abs(),
        swept.message
    );
    assert!(swept.parameters.len() > 1);
    assert!(swept.energy >= joint.energy - 1e-10);
    assert!(
        swept.energy < swept.reference,
        "the sweep lowers the reference energy"
    );
    // The last coordinate was optimized last, with every other one fixed: it is stationary.
    assert!(gradient.last().unwrap().abs() < 1e-5);
    assert!(swept.message.contains("sequential sweep"));
    drop((joint, swept, problem));
    assert_eq!(ctx.memory_used(), 0);
}
