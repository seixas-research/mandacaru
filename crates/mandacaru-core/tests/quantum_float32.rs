//! Complex64 state vectors: the float32 instance against the float64 one on
//! the same Hamiltonian, the refusals of settings float32 cannot resolve, and
//! the state-memory saving. Hamiltonian, parameters and reductions are float64
//! in both; only the stored state differs.
use mandacaru_core::{
    math::storage::Precision,
    quantum::{
        dynamics::{qpe, Propagator},
        evolution, optimizer,
        sparse::SparseMatrix,
        Basis, CachedProblem, Encoding, Operator, SolveOptions, C64,
    },
    Context, Error,
};

fn context(megabytes: usize) -> Context {
    Context::new(Some(1), megabytes << 20).unwrap()
}
/// Half-filled Hubbard chain (t = 1, U = 2), `n` spatial orbitals, Jordan-Wigner.
fn hubbard(ctx: &Context, n: usize, sector: bool, precision: Precision) -> CachedProblem {
    let encoding = Encoding::new("jordan_wigner", 2 * n, [n / 2, n / 2]).unwrap();
    let mut h = Operator::new(ctx, 2 * n, &[]).unwrap();
    let mut add = |ladders: &[(usize, bool)], value: f64| {
        let term = encoding.monomial(ctx, ladders).unwrap();
        h = h.add(ctx, &term, C64::new(value, 0.)).unwrap();
    };
    for spin in 0..2 {
        for i in 0..n - 1 {
            let (a, b) = (i + spin * n, i + 1 + spin * n);
            add(&[(a, true), (b, false)], -1.);
            add(&[(b, true), (a, false)], -1.);
        }
    }
    for i in 0..n {
        add(&[(i, true), (i, false), (i + n, true), (i + n, false)], 2.);
    }
    CachedProblem::new(
        ctx,
        Encoding::new("jordan_wigner", 2 * n, [n / 2, n / 2]).unwrap(),
        h,
        sector,
        false,
        true,
    )
    .unwrap()
    .with_precision(precision)
}
fn parameters(count: usize) -> Vec<f64> {
    (0..count)
        .map(|k| 0.3 * ((k * 7 + 3) % 11) as f64 / 11. - 0.1)
        .collect()
}

#[test]
fn a_float32_problem_stores_complex64_and_refuses_to_widen_silently() {
    let ctx = context(256);
    {
        let problem = hubbard(&ctx, 4, true, Precision::Float32);
        let indices: Vec<usize> = (0..4).collect();
        let value = problem.evaluate(&indices, &[0.1; 4], true).unwrap();
        assert_eq!(value.state.precision(), Precision::Float32);
        assert!(value.state.complex64().is_ok());
        assert!(value.state.complex128().is_err());
        assert!((value.state.norm_squared() - 1.).abs() < 1e-5);
        let reference = hubbard(&ctx, 4, true, Precision::Float64);
        let exact = reference.evaluate(&indices, &[0.1; 4], true).unwrap();
        assert_eq!(exact.state.precision(), Precision::Float64);
        assert!(exact.state.complex64().is_err());
        // The widened copy is explicit, charged to the budget and equal to the
        // stored amplitudes.
        let widened = value.state.widen(&ctx).unwrap();
        for (i, w) in widened.as_slice().iter().enumerate() {
            assert_eq!(*w, value.state.amplitude(i));
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn energy_and_gradient_agree_to_float32_rounding_in_both_ansatz_forms() {
    let ctx = context(256);
    {
        for n in [4, 5] {
            for sector in [true, false] {
                let high = hubbard(&ctx, n, sector, Precision::Float64);
                let low = hubbard(&ctx, n, sector, Precision::Float32);
                let count = 6;
                let indices: Vec<usize> = (0..count).collect();
                let theta = parameters(count);
                for trotter in [true, false] {
                    let a = high.evaluate(&indices, &theta, trotter).unwrap();
                    let b = low.evaluate(&indices, &theta, trotter).unwrap();
                    assert!((a.energy - b.energy).abs() < 2e-6, "{n} {sector} {trotter}");
                    for (x, y) in a.gradient.iter().zip(&b.gradient) {
                        assert!((x - y).abs() < 2e-6, "{n} {sector} {trotter}");
                    }
                }
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn sparse_dense_and_matrix_free_hamiltonians_agree_in_float32() {
    let ctx = context(256);
    {
        let mut energies = Vec::new();
        for storage in 0..3 {
            let mut problem = hubbard(&ctx, 4, true, Precision::Float32);
            if storage == 1 {
                problem.dense = Some(problem.hamiltonian.dense(&ctx, &problem.basis).unwrap());
            }
            if storage == 2 {
                problem.sparse =
                    Some(SparseMatrix::new(&ctx, &problem.basis, &problem.hamiltonian).unwrap());
            }
            let value = problem
                .evaluate(&[0, 1, 2], &[0.2, -0.1, 0.15], false)
                .unwrap();
            energies.push((value.energy, value.gradient));
        }
        for other in &energies[1..] {
            assert!((other.0 - energies[0].0).abs() < 2e-6);
            for (x, y) in other.1.iter().zip(&energies[0].1) {
                assert!((x - y).abs() < 2e-6);
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_14_qubit_solve_in_float32_stays_within_the_quantum_gate() {
    // 7 spatial orbitals: 14 qubits. The gate is |dE| <= 5e-4 Ha.
    let ctx = context(512);
    {
        let options = SolveOptions {
            adapt: true,
            trotter: true,
            max_growth: 4,
            gradient_tolerance: Some(1e-4),
            optimizer: optimizer::Options {
                tolerance: Some(1e-5),
                max_iterations: 60,
                ..Default::default()
            },
            ..Default::default()
        };
        let high = hubbard(&ctx, 7, true, Precision::Float64)
            .solve(&options, None)
            .unwrap();
        let low = hubbard(&ctx, 7, true, Precision::Float32)
            .solve(&options, None)
            .unwrap();
        assert_eq!(low.state.precision(), Precision::Float32);
        assert!((high.energy - low.energy).abs() < 5e-4);
        assert!((high.reference - low.reference).abs() < 5e-6);
        assert_eq!(high.indices, low.indices);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn settings_float32_cannot_resolve_raise_naming_the_floor() {
    let ctx = context(64);
    let message = |error: Error| match error {
        Error::Invalid(text) => text,
        other => panic!("expected an invalid-input error, got {other:?}"),
    };
    {
        let low = hubbard(&ctx, 3, true, Precision::Float32);
        let high = hubbard(&ctx, 3, true, Precision::Float64);
        let tight_tolerance = SolveOptions::default(); // optimizer tolerance 1e-9
        let text = message(low.solve(&tight_tolerance, None).err().unwrap());
        assert!(text.contains("float32") && text.contains("floor"), "{text}");
        assert!(high.solve(&tight_tolerance, None).is_ok());
        let tight_gradient = SolveOptions {
            adapt: true,
            gradient_tolerance: Some(1e-7),
            optimizer: optimizer::Options {
                tolerance: Some(1e-5),
                ..Default::default()
            },
            ..Default::default()
        };
        let text = message(low.solve(&tight_gradient, None).err().unwrap());
        assert!(
            text.contains("gradient") && text.contains("float32"),
            "{text}"
        );
        let tight_energy = SolveOptions {
            adapt: true,
            gradient_tolerance: Some(1e-3),
            energy_tolerance: Some(1e-9),
            optimizer: optimizer::Options {
                tolerance: Some(1e-5),
                ..Default::default()
            },
            ..Default::default()
        };
        let text = message(low.solve(&tight_energy, None).err().unwrap());
        assert!(
            text.contains("energy") && text.contains("float32"),
            "{text}"
        );
        let text = message(
            low.energy_levels(&tight_tolerance, 2, 1, None, 0)
                .err()
                .unwrap(),
        );
        assert!(text.contains("float32"), "{text}");
        let text = message(
            low.solve_subspace(&tight_tolerance, &[2., 1.], None)
                .err()
                .unwrap(),
        );
        assert!(text.contains("float32"), "{text}");
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn propagation_and_phase_estimation_agree_in_float32() {
    let ctx = context(64);
    {
        let terms: Vec<(String, C64)> = vec![
            ("XXI".into(), C64::new(0.7, 0.)),
            ("IYY".into(), C64::new(-0.4, 0.)),
            ("ZIZ".into(), C64::new(0.3, 0.)),
            ("III".into(), C64::new(0.2, 0.)),
        ];
        let propagator = Propagator::new(&ctx, 3, &terms, 2).unwrap();
        let mut state = vec![C64::default(); 8];
        state[1] = C64::new(0.6, 0.);
        state[6] = C64::new(0., 0.8);
        let narrow: Vec<_> = state
            .iter()
            .map(|c| num_complex::Complex32::new(c.re as f32, c.im as f32))
            .collect();
        let exact = propagator
            .evolve(&ctx, &state, 0.9, 6, false, true)
            .unwrap();
        let rounded = propagator
            .evolve(&ctx, &narrow, 0.9, 6, false, true)
            .unwrap();
        for (a, b) in exact.as_slice().iter().zip(rounded.as_slice()) {
            assert!((a - C64::new(f64::from(b.re), f64::from(b.im))).norm() < 5e-6);
        }
        let h = Operator::new(&ctx, 3, &terms).unwrap();
        let reference = qpe(&ctx, &h, &state, 4, [-1.5, 1.5], true).unwrap();
        let low = qpe(&ctx, &h, &narrow, 4, [-1.5, 1.5], true).unwrap();
        for (a, b) in reference
            .probabilities
            .as_slice()
            .iter()
            .zip(low.probabilities.as_slice())
        {
            assert!((a - b).abs() < 5e-6);
        }
        assert!(low.amplitudes.unwrap().complex64().is_ok());
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_exponential_action_keeps_one_apply_definition_for_both_precisions() {
    let ctx = context(64);
    {
        let problem = hubbard(&ctx, 4, true, Precision::Float64);
        let basis: &Basis = &problem.basis;
        let state: Vec<C64> = (0..basis.len())
            .map(|k| C64::new(((k * 5 + 1) % 7) as f64 - 3., ((k * 3) % 5) as f64 - 2.))
            .collect();
        let norm = evolution::inner(&state, &state).re.sqrt();
        let state: Vec<C64> = state.iter().map(|c| c / norm).collect();
        let narrow: Vec<_> = state
            .iter()
            .map(|c| num_complex::Complex32::new(c.re as f32, c.im as f32))
            .collect();
        let mut wide = vec![C64::default(); basis.len()];
        let mut from_narrow = vec![C64::default(); basis.len()];
        problem
            .hamiltonian
            .apply_into(basis, &state, wide.as_mut_slice());
        problem
            .hamiltonian
            .apply_into(basis, &narrow, from_narrow.as_mut_slice());
        let error = wide
            .iter()
            .zip(&from_narrow)
            .map(|(a, b)| (a - b).norm())
            .fold(0., f64::max);
        assert!(error < 5e-6, "{error}");
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn float32_states_lower_the_context_peak_of_a_derivative_row_evaluation() {
    // 16 qubits, full register, four parameters, no Trotter product: the three
    // row stacks dominate. The measured 22-qubit ratios are in the team log.
    let mut peaks = Vec::new();
    for precision in [Precision::Float64, Precision::Float32] {
        let ctx = context(512);
        {
            let problem = hubbard(&ctx, 8, false, precision);
            let before = ctx.memory_peak();
            let indices: Vec<usize> = (0..4).collect();
            let _ = problem.evaluate(&indices, &[0.1; 4], false).unwrap();
            peaks.push(ctx.memory_peak().max(before));
        }
        assert_eq!(ctx.memory_used(), 0);
    }
    assert!(
        (peaks[1] as f64) < 0.7 * peaks[0] as f64,
        "float32 {} float64 {}",
        peaks[1],
        peaks[0]
    );
}
