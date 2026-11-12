//! The operator importances of a product ansatz (`CachedProblem::importances`) give the bits of the
//! evaluations they replace.
//!
//! The importance of position `i` is `|dE/dtheta_i|` with `theta_i` set to zero: formerly one full
//! evaluation per position. The new routine advances one prefix state, applies only the generators
//! after `i`, and stops the adjoint sweep at `i`. These tests compare every importance, as bits,
//! with the zero-and-evaluate loop, on a state above the parallel threshold (so the gathered
//! rotations and the fused adjoint steps run) on one thread and on eight, in both storage
//! precisions, and check the work the routine reports.
mod product_ansatz_support;

use mandacaru_core::{
    math::storage::Precision,
    quantum::{evolution::Work, CachedProblem},
};
use product_ansatz_support::problem;

/// The importances as the zero-and-evaluate loop computes them.
fn separate(problem: &CachedProblem, indices: &[usize], theta: &[f64]) -> Vec<u64> {
    (0..indices.len())
        .map(|i| {
            let mut zeroed = theta.to_vec();
            zeroed[i] = 0.;
            problem.evaluate(indices, &zeroed, true).unwrap().gradient[i]
                .abs()
                .to_bits()
        })
        .collect()
}

fn importances(problem: &CachedProblem, indices: &[usize], theta: &[f64]) -> (Vec<u64>, Work) {
    let out = problem.importances(indices, theta).unwrap();
    (out.values.iter().map(|v| v.to_bits()).collect(), out.work)
}

fn check(
    modes: usize,
    electrons: [usize; 2],
    operators: usize,
    precision: Precision,
    threads: &[usize],
) {
    let mut first: Option<Vec<u64>> = None;
    for &count in threads {
        let (problem, indices, theta) = problem(count, modes, electrons, operators, precision);
        let (shared, _) = importances(&problem, &indices, &theta);
        assert_eq!(
            shared,
            separate(&problem, &indices, &theta),
            "{count} threads: an importance differs from its evaluation"
        );
        match &first {
            None => first = Some(shared),
            Some(reference) => assert_eq!(
                &shared, reference,
                "{count} threads: the importances depend on the thread count"
            ),
        }
    }
}

#[test]
fn importances_are_the_zeroed_evaluations_on_a_small_sector() {
    // 12 modes, 3 + 3 electrons: 400 rows, every action stays in place.
    check(12, [3, 3], 6, Precision::Float64, &[1]);
    check(12, [3, 3], 6, Precision::Float32, &[1]);
}

#[test]
fn importances_are_the_zeroed_evaluations_above_the_parallel_threshold() {
    // 20 modes, 4 + 4 electrons: the 44100-row sector of the H2O active space. The eight-string
    // members run the gathered rotation and the fused adjoint step on eight threads and the
    // in-place kernels on one; the four-string combinations stay in place on both.
    check(20, [4, 4], 10, Precision::Float64, &[1, 8]);
}

#[test]
fn float32_importances_are_the_zeroed_evaluations_above_the_parallel_threshold() {
    check(20, [4, 4], 10, Precision::Float32, &[1, 8]);
}

#[test]
fn the_importances_of_a_short_and_an_empty_ansatz_are_the_evaluations() {
    let (problem, indices, theta) = problem(1, 12, [3, 3], 4, Precision::Float64);
    for length in 0..=2 {
        let (shared, _) = importances(&problem, &indices[..length], &theta[..length]);
        assert_eq!(
            shared,
            separate(&problem, &indices[..length], &theta[..length]),
            "length {length}"
        );
    }
}

#[test]
fn the_importances_report_the_work_they_did() {
    let (problem, indices, theta) = problem(1, 12, [3, 3], 8, Precision::Float64);
    for length in 1..=8 {
        let out = problem
            .importances(&indices[..length], &theta[..length])
            .unwrap();
        let l = length;
        // The running prefix: `l - 1` rotations; position `i`: `l - i - 1` rotations of the
        // suffix, `l - i` adjoint steps and one `H psi`.
        assert_eq!(out.work.rotations, (l - 1) + l * (l - 1) / 2, "length {l}");
        assert_eq!(out.work.adjoint_steps, l * (l + 1) / 2, "length {l}");
        assert_eq!(out.work.expectations, l, "length {l}");
        // Against one forward pass and one sweep per position: at most the same, and about half
        // as many adjoint steps and (for long ansatzes) rotations.
        let separate = Work::separate_evaluations(l);
        assert!(
            out.work.rotations <= separate.rotations
                && out.work.adjoint_steps <= separate.adjoint_steps,
            "{l}"
        );
        assert_eq!(
            2 * out.work.adjoint_steps,
            separate.adjoint_steps + l,
            "{l}"
        );
        assert_eq!(out.work.expectations, separate.expectations, "{l}");
    }
}
