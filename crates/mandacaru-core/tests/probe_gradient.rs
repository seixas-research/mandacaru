//! The deferred gradient of an evaluation (`CachedProblem::probe`) gives the bits of the evaluation
//! it replaces: the energy at once and the gradient when the optimizer asks for it.
mod product_ansatz_support;

use mandacaru_core::math::storage::Precision;
use product_ansatz_support::problem;

/// The deferred gradient of a probe, forced after the energy was read, is the evaluation's: the
/// same energy and gradient bits for a product ansatz, and for a summed one whose derivative rows
/// stay in the forward pass.
#[test]
fn a_probe_gives_the_energy_and_gradient_of_an_evaluation() {
    for (threads, modes, electrons, operators, precision) in [
        (1, 12, [3, 3], 6, Precision::Float64),
        (1, 12, [3, 3], 6, Precision::Float32),
        (8, 20, [4, 4], 8, Precision::Float64),
        (8, 20, [4, 4], 8, Precision::Float32),
    ] {
        let (problem, indices, theta) = problem(threads, modes, electrons, operators, precision);
        for trotter in [true, false] {
            if !trotter && modes > 12 {
                continue;
            }
            let full = problem.evaluate(&indices, &theta, trotter).unwrap();
            let probe = problem.probe(&indices, &theta, trotter).unwrap();
            assert_eq!(
                probe.energy.to_bits(),
                full.energy.to_bits(),
                "{threads} {modes} {trotter}: energy"
            );
            let gradient = probe.into_gradient().unwrap();
            assert_eq!(
                gradient.iter().map(|g| g.to_bits()).collect::<Vec<_>>(),
                full.gradient
                    .iter()
                    .map(|g| g.to_bits())
                    .collect::<Vec<_>>(),
                "{threads} {modes} {trotter}: gradient"
            );
        }
    }
}
