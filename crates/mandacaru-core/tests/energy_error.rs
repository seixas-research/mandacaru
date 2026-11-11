//! The standard error of a sampled energy (`quantum::energy_error`, measured by
//! `quantum::execution::CircuitExecution::with_shots`).
//!
//! The device here samples real shots: each request's circuit is applied to the exact reference
//! state, and `N` bit strings are drawn from its probabilities. Two independent references:
//! the exact `sigma` of the state (`sum_b Var(H_b) / N` over the commuting bases, from operator
//! algebra in the original frame, no rotation and no product string), and the scatter of the
//! sampled energy over repeated runs.
use mandacaru_core::{
    execution::qwc,
    persistence::{Document, HAMILTONIAN},
    quantum::{
        execution::{CircuitExecution, Request, Response},
        solver::Execution,
        Basis, Operator, Space, C64,
    },
    Context,
};
use std::cell::Cell;

fn rotate(
    ctx: &Context,
    basis: &Basis,
    n: usize,
    state: &[C64],
    label: &str,
    angle: f64,
) -> Vec<C64> {
    let p = Operator::new(ctx, n, &[(label.to_string(), C64::new(1., 0.))]).unwrap();
    let action = p.apply(ctx, basis, state).unwrap();
    state
        .iter()
        .zip(action.as_slice())
        .map(|(a, b)| a * angle.cos() + C64::new(0., angle.sin()) * b)
        .collect()
}

/// The state each request prepares, on the full register.
fn prepared(ctx: &Context, basis: &Basis, n: usize, reference: u64, request: &Request) -> Vec<C64> {
    let mut state = vec![C64::default(); basis.len()];
    state[basis.index(reference).unwrap()] = C64::new(1., 0.);
    for (label, angle) in request.labels.iter().zip(&request.angles) {
        state = rotate(ctx, basis, n, &state, label, *angle);
    }
    state
}

#[test]
fn the_sampled_energy_reports_its_shot_noise() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference/h2_hamiltonian.json");
    let document = Document::read(&ctx, &fixture, HAMILTONIAN).unwrap();
    let problem = document
        .problem_in(
            &ctx,
            Space::Pauli {
                number_conserving: true,
            },
            false,
            true,
        )
        .unwrap();
    assert!(problem.is_register_free());
    let n = problem.hamiltonian.n;
    let register = Basis::register(&ctx, n).unwrap();
    let reference = problem.register_reference();
    let shots = 2000u64;
    let seed = Cell::new(1u64);
    let mut sample = |requests: &[Request]| -> mandacaru_core::Result<Response> {
        let mut x = seed.get();
        let mut uniform = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut answers = Vec::with_capacity(requests.len());
        for request in requests {
            let state = prepared(&ctx, &register, n, reference, request);
            let probabilities: Vec<f64> = state.iter().map(|a| a.norm_sqr()).collect();
            let masks: Vec<u64> = request
                .observables
                .iter()
                .map(|label| {
                    assert!(label.chars().all(|c| c == 'I' || c == 'Z'), "{label}");
                    label
                        .bytes()
                        .enumerate()
                        .filter(|(_, c)| *c == b'Z')
                        .map(|(j, _)| 1u64 << (n - 1 - j))
                        .sum()
                })
                .collect();
            let mut sums = vec![0f64; masks.len()];
            for _ in 0..shots {
                let (mut u, mut k) = (uniform(), 0);
                while k + 1 < probabilities.len() && u >= probabilities[k] {
                    u -= probabilities[k];
                    k += 1;
                }
                let ket = register.indices.as_slice()[k];
                for (sum, mask) in sums.iter_mut().zip(&masks) {
                    *sum += if (ket & mask).count_ones() % 2 == 0 {
                        1.
                    } else {
                        -1.
                    };
                }
            }
            answers.push(sums.into_iter().map(|s| s / shots as f64).collect());
        }
        seed.set(x);
        Ok(Response::Expectations(answers))
    };
    let (indices, parameters) = (vec![0usize], vec![0.4f64]);
    // The exact sigma, from the state the circuit prepares and the commuting bases of H.
    let circuit =
        mandacaru_core::quantum::execution::circuit_rotations(&problem, &indices, &parameters)
            .unwrap();
    let state = prepared(
        &ctx,
        &register,
        n,
        reference,
        &Request {
            labels: circuit.0,
            angles: circuit.1,
            observables: vec![],
            priced: 0,
        },
    );
    let labels: Vec<String> = problem
        .hamiltonian
        .terms
        .iter()
        .filter(|t| (t.x, t.z) != (0, 0))
        .map(|t| t.label(n))
        .collect();
    let coefficients: std::collections::HashMap<String, C64> =
        problem.hamiltonian.labels().into_iter().collect();
    let mut variance = 0.;
    for (_, members) in qwc(&ctx, &labels, None).unwrap().groups {
        let terms: Vec<_> = members
            .iter()
            .map(|&i| (labels[i].clone(), coefficients[&labels[i]]))
            .collect();
        let part = Operator::new(&ctx, n, &terms).unwrap();
        let square = part.compose(&ctx, &part).unwrap();
        let mean = |op: &Operator| {
            let action = op.apply(&ctx, &register, &state).unwrap();
            state
                .iter()
                .zip(action.as_slice())
                .map(|(a, b)| (a.conj() * b).re)
                .sum::<f64>()
        };
        variance += (mean(&square) - mean(&part).powi(2)) / shots as f64;
    }
    let exact = variance.sqrt();
    let (mut energies, mut errors) = (Vec::new(), Vec::new());
    let mut execution = CircuitExecution::new(&mut sample).with_shots(shots);
    for _ in 0..300 {
        let evaluation = execution
            .evaluate(&problem, &indices, &parameters, false)
            .unwrap();
        energies.push(evaluation.energy);
        errors.push(
            execution
                .energy_error(evaluation.energy)
                .expect("a sampled energy has an error"),
        );
    }
    let mean = energies.iter().sum::<f64>() / energies.len() as f64;
    let observed = (energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>()
        / (energies.len() - 1) as f64)
        .sqrt();
    let reported = errors.iter().sum::<f64>() / errors.len() as f64;
    eprintln!(
        "sigma: exact {exact:.6e}, reported {reported:.6e}, observed over 300 runs {observed:.6e}"
    );
    // The estimate from 2000 shots is within a few percent of the exact sigma.
    assert!((reported / exact - 1.).abs() < 0.05, "{reported} {exact}");
    // 300 runs estimate a standard deviation to about 4 percent.
    assert!((observed / exact - 1.).abs() < 0.15, "{observed} {exact}");
}

#[test]
fn a_basis_past_the_product_limit_leaves_the_energy_without_an_error() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    // 40 qubits, every Z_i and Z_i Z_j: one diagonal basis of 820 strings, whose products
    // (strings of up to four Z) number far more than the limit.
    let n = 40;
    let mut terms = Vec::new();
    for i in 0..n {
        for j in i..n {
            let label: String = (0..n)
                .map(|k| if k == i || k == j { 'Z' } else { 'I' })
                .collect();
            terms.push((label, C64::new(0.01 * (i + j + 1) as f64, 0.)));
        }
    }
    let hamiltonian = Operator::new(&ctx, n, &terms).unwrap();
    let plan = mandacaru_core::quantum::energy_error::Plan::commuting(&ctx, &hamiltonian).unwrap();
    let reason = plan.unavailable.clone().expect("past the limit: no error");
    assert!(
        reason.contains("not reported") && reason.contains("20000"),
        "{reason}"
    );
    // No products are measured, so every request reads exactly the strings it did before.
    assert!(plan.bases.iter().all(|b| b.products.is_empty()));
    let positions = plan.observables();
    let values: Vec<Vec<f64>> = positions
        .iter()
        .map(|(labels, _, _)| vec![0.5; labels.len()])
        .collect();
    let (energy, error) = plan.energy(&positions, &values, 1000).unwrap();
    assert!(energy.is_finite() && error.is_none());
}
