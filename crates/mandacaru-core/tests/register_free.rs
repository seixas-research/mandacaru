//! A register-free problem (the shot-based mode) holds Pauli operators only: no `Basis`, no
//! state vector, no matrix. A 256 MB context is the proof, since one `2^25` amplitude array is
//! 512 MB (and the sector of 25 orbitals with 5 + 5 particles holds 2.8 billion amplitudes).
use mandacaru_core::{
    persistence::{Document, HAMILTONIAN},
    quantum::{
        execution::{CircuitExecution, Request, Response},
        optimizer::{Method, Options},
        CachedProblem, Encoding, Operator, SolveOptions, Space, C64,
    },
    Context,
};
use std::cell::Cell;

const N: usize = 50;

fn context() -> Context {
    Context::new(Some(1), 256 << 20).unwrap()
}
/// Spin-block-preserving hops and on-site energies on 25 spatial orbitals, `n` qubits (JW).
fn hamiltonian(ctx: &Context, with_leak: bool) -> Operator {
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
    if with_leak {
        let label: String = (0..N).map(|k| if k == 3 { 'X' } else { 'I' }).collect();
        terms.push((label, C64::new(0.2, 0.)));
    }
    Operator::new(ctx, N, &terms).unwrap()
}
fn problem(ctx: &Context, number_conserving: bool) -> CachedProblem {
    CachedProblem::new_in(
        ctx,
        Encoding::new("jordan_wigner", N, [1, 1]).unwrap(),
        hamiltonian(ctx, false),
        Space::Pauli { number_conserving },
        false,
        true,
    )
    .unwrap()
}
fn refusal<T>(outcome: mandacaru_core::Result<T>, what: &str) {
    let message = match outcome {
        Ok(_) => panic!("{what} must be refused on a register-free problem"),
        Err(e) => e.to_string(),
    };
    assert!(
        message.contains("register-free") && message.contains(what),
        "{what}: {message}"
    );
}

#[test]
fn fifty_qubits_build_with_no_register_inside_a_small_budget() {
    let ctx = context();
    let problem = problem(&ctx, true);
    assert!(problem.is_register_free() && problem.basis.is_none());
    assert!(!problem.pool.is_empty() && problem.n_qubits() == N);
    assert!(
        ctx.memory_used() < 32 << 20,
        "{} bytes in use",
        ctx.memory_used()
    );
}

#[test]
fn every_state_vector_method_refuses_by_name() {
    let ctx = context();
    let problem = problem(&ctx, false);
    refusal(problem.register("the test"), "the test");
    refusal(problem.evaluate(&[], &[], true), "energy evaluation");
    refusal(problem.prepare_state(&[], &[], true), "prepare_state");
    refusal(problem.rdms::<C64>(&[], false), "density matrices");
    refusal(
        problem.solve(&SolveOptions::default(), None),
        "a native simulation",
    );
    refusal(
        problem.energy_levels(&SolveOptions::default(), 2, 1, None, 0),
        "sequential deflation",
    );
    refusal(
        problem.solve_subspace(&SolveOptions::default(), &[2., 1.], None),
        "subspace",
    );
    let dense = CachedProblem::new_in(
        &ctx,
        Encoding::new("jordan_wigner", N, [1, 1]).unwrap(),
        hamiltonian(&ctx, false),
        Space::Pauli {
            number_conserving: false,
        },
        true,
        true,
    );
    refusal(dense, "dense Hamiltonian matrix");
}

#[test]
fn the_particle_number_check_is_algebraic() {
    let ctx = context();
    let leaking = CachedProblem::new_in(
        &ctx,
        Encoding::new("jordan_wigner", N, [1, 1]).unwrap(),
        hamiltonian(&ctx, true),
        Space::Pauli {
            number_conserving: true,
        },
        false,
        true,
    );
    let message = leaking
        .err()
        .expect("an X term changes the particle number");
    assert!(message.to_string().contains("particle sector"), "{message}");
    // Without the request nothing is checked (a register-free problem has nothing to check with).
    CachedProblem::new_in(
        &ctx,
        Encoding::new("jordan_wigner", N, [1, 1]).unwrap(),
        hamiltonian(&ctx, true),
        Space::Pauli {
            number_conserving: false,
        },
        false,
        true,
    )
    .unwrap();
}

#[test]
fn tapering_keeps_the_problem_register_free() {
    let ctx = context();
    let mut problem = problem(&ctx, false);
    let reference = problem
        .hamiltonian
        .expectation_at(problem.register_reference());
    problem.taper().unwrap();
    let taper = problem
        .taper
        .as_ref()
        .expect("the hops leave Z2 symmetries");
    assert!(problem.n_qubits() == taper.width() && problem.n_qubits() < N);
    assert!(problem.is_register_free());
    assert!(
        (problem
            .hamiltonian
            .expectation_at(problem.register_reference())
            - reference)
            .norm()
            < 1e-12
    );
    assert!(ctx.memory_used() < 32 << 20);
}

/// Smooth synthetic landscape, the same for any problem that asks the same questions.
fn landscape(requests: &[Request]) -> Response {
    Response::Expectations(
        requests
            .iter()
            .map(|request| {
                let total: f64 = request
                    .angles
                    .iter()
                    .enumerate()
                    .map(|(j, a)| (1. + 0.37 * j as f64) * a)
                    .sum();
                (0..request.observables.len())
                    .map(|k| (0.7 * total + 0.3 + 0.1 * k as f64).cos())
                    .collect()
            })
            .collect(),
    )
}

#[test]
fn a_register_free_solve_asks_the_device_what_the_register_solve_asks() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference/h2_hamiltonian.json");
    let document = Document::read(&ctx, &fixture, HAMILTONIAN).unwrap();
    let with = document
        .problem_in(&ctx, Space::Sector, false, true)
        .unwrap();
    let without = document
        .problem_in(
            &ctx,
            Space::Pauli {
                number_conserving: true,
            },
            false,
            true,
        )
        .unwrap();
    assert!(without.is_register_free() && !with.is_register_free());
    let options = SolveOptions {
        adapt: true,
        trotter: true,
        max_growth: 3,
        optimizer: Options {
            method: Method::NelderMead,
            max_iterations: 30,
            tolerance: Some(1e-9),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut solved = Vec::new();
    for p in [&with, &without] {
        let calls = Cell::new(0);
        let mut call = |requests: &[Request]| {
            calls.set(calls.get() + 1);
            Ok(landscape(requests))
        };
        let mut execution = CircuitExecution::new(&mut call);
        let result = p
            .solve_executed(&options, None, None, &mut execution)
            .unwrap();
        solved.push((result, calls.get()));
    }
    let (a, b) = (&solved[0], &solved[1]);
    assert_eq!(a.1, b.1, "device calls");
    assert_eq!(a.0.indices, b.0.indices);
    assert_eq!(a.0.history, b.0.history);
    assert_eq!(a.0.energy, b.0.energy);
    assert!((a.0.reference - b.0.reference).abs() < 1e-12);
    assert!(a.0.state.len() > 0 && b.0.state.is_empty());
}

#[test]
fn a_fifty_qubit_vqe_and_adapt_run_on_pauli_observables_alone() {
    let ctx = context();
    let problem = problem(&ctx, true);
    for adapt in [false, true] {
        let calls = Cell::new(0);
        let mut call = |requests: &[Request]| {
            calls.set(calls.get() + 1);
            Ok(landscape(requests))
        };
        let mut execution = CircuitExecution::new(&mut call);
        let options = SolveOptions {
            adapt,
            trotter: true,
            max_growth: 1,
            optimizer: Options {
                method: Method::Spsa,
                max_iterations: 2,
                max_evaluations: 8,
                tolerance: None,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = problem.solve_executed(&options, None, None, &mut execution);
        let result = result.unwrap();
        assert!(result.state.is_empty() && calls.get() > 0, "adapt={adapt}");
        assert!(result.energy.is_finite() && result.reference.is_finite());
        assert!(
            ctx.memory_used() < 48 << 20,
            "adapt={adapt}: {} bytes",
            ctx.memory_used()
        );
    }
}
