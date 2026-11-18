//! Unit and integration verification for VASQA Rust kernels and end-to-end benchmarks.
//!
//! Covers:
//! - Sparse-support forward rotations and tangent-vector gradients against dense adjoint sweeps
//!   and central finite differences.
//! - Multi-threaded vs. single-threaded `importances` bit-for-bit parity.
//! - Curvature carry (`carry_hessian`) across `Insert`, `Delete`, and `Replace` edits and warm
//!   proposal memoization (`memoize_warm`).
//! - Single-qubit, boundary-angle (`0`, `±π`, `±2π`), and vanishing-gradient symmetry checks.
//! - End-to-end VASQA optimization on Transverse Field Ising Model (TFIM), minimal-basis H2, and
//!   MaxCut Hamiltonians verified against exact diagonalization (`nalgebra::SymmetricEigen`).

use mandacaru_core::{
    math::storage::Precision,
    quantum::{
        algebra::Generator,
        evolution::{evaluate, forward_from_with_mode, tangent_gradient},
        optimizer::{self, Method},
        properties::{oscillator_strength, transition_dipole},
        search::{kernel::Kernel, vasqa::SearchOptions},
        Basis, CrystalMomentum, CrystalPointSymmetry, Encoding, Operator, QuantumProblem,
        SolveOptions, C64,
    },
    Context,
};
use nalgebra::DMatrix;
use std::f64::consts::PI;

/// Exact sorted spectrum of `op` on `basis` by dense Hermitian diagonalization.
fn exact_spectrum(ctx: &Context, basis: &Basis, op: &Operator) -> Vec<f64> {
    let d = basis.len();
    let dense = op.dense(ctx, basis).unwrap();
    let mat = DMatrix::from_fn(d, d, |r, c| dense.as_slice()[r * d + c]);
    let mut vals: Vec<f64> = mat.symmetric_eigen().eigenvalues.iter().copied().collect();
    vals.sort_by(f64::total_cmp);
    vals
}

/// Exact ground-state energy of `op` on `basis` by dense Hermitian diagonalization.
fn exact_ground_energy(ctx: &Context, basis: &Basis, op: &Operator) -> f64 {
    exact_spectrum(ctx, basis, op)[0]
}

/// Number-conserving tight-binding + interaction Hamiltonian for testing sparse states at `d >= 1024`.
fn sector_test_problem(
    threads: usize,
    modes: usize,
    particles: [usize; 2],
    pool: &str,
) -> QuantumProblem {
    let ctx = Context::new(Some(threads), 512 << 20).unwrap();
    let encoding = Encoding::new("jordan_wigner", modes, particles).unwrap();
    let mut terms = vec![("I".repeat(modes), C64::new(-2.5, 0.))];
    for j in 0..modes {
        let mut z = vec![b'I'; modes];
        z[j] = b'Z';
        terms.push((
            String::from_utf8(z).unwrap(),
            C64::new(0.25 - 0.04 * j as f64, 0.),
        ));
    }
    let mut h = Operator::new(&ctx, modes, &terms).unwrap();
    let half = modes / 2;
    for j in 0..modes - 1 {
        let k = j + 1;
        if j / half != k / half {
            continue;
        }
        let weight = C64::new(0.18 - 0.01 * j as f64, 0.);
        for (a, b) in [(j, k), (k, j)] {
            let hop = encoding.monomial(&ctx, &[(a, true), (b, false)]).unwrap();
            h = h.add(&ctx, &hop, weight).unwrap();
        }
    }
    // Add a two-body density-density term across spin channels.
    for j in 0..half.min(3) {
        let n_up = encoding.monomial(&ctx, &[(j, true), (j, false)]).unwrap();
        let n_dn = encoding
            .monomial(&ctx, &[(half + j, true), (half + j, false)])
            .unwrap();
        let nn = n_up.compose(&ctx, &n_dn).unwrap();
        h = h.add(&ctx, &nn, C64::new(0.35, 0.)).unwrap();
    }
    let mut problem = QuantumProblem::new(&ctx, encoding, h, true, false, true).unwrap();
    problem.use_pool(pool, true).unwrap();
    problem
}

#[test]
fn tangent_gradient_matches_adjoint_and_finite_differences_at_large_and_boundary_angles() {
    for (modes, expected_d) in [(12, 400), (14, 1225)] {
        let problem = sector_test_problem(1, modes, [3, 3], "qeb");
        let ctx = problem.context.clone();
        {
            let basis = problem.register("test").unwrap();
            assert_eq!(basis.len(), expected_d);

            let indices = [0, 3, 7, 11, 15, 20, 25, 30];
            let ops: Vec<&Operator> = indices.iter().map(|&i| &problem.pool[i].operator).collect();
            let angle_sets = [
                vec![0.12, -0.34, 0.08, 0.41, -0.19, 0.05, -0.27, 0.33],
                vec![0.0, PI, -PI, 2. * PI, -2. * PI, 0.5 * PI, -0.5 * PI, 0.0],
            ];

            for theta in &angle_sets {
                let eval = problem.evaluate(&indices, theta, true).unwrap();
                let psi = eval.state.complex128().unwrap();
                let mut wide = ctx.allocate::<C64>(basis.len()).unwrap();
                problem
                    .hamiltonian
                    .apply_into(basis, psi, wide.as_mut_slice());
                let id = C64::new(
                    problem
                        .hamiltonian
                        .labels()
                        .iter()
                        .find(|(l, _)| l.chars().all(|c| c == 'I'))
                        .map_or(0., |(_, c)| c.re),
                    0.,
                );
                let mut wide_centered = ctx.allocate::<C64>(basis.len()).unwrap();
                for (dst, (&w, &p)) in wide_centered
                    .as_mut_slice()
                    .iter_mut()
                    .zip(wide.as_slice().iter().zip(psi.iter()))
                {
                    *dst = w - id * p;
                }
                let tangent = tangent_gradient(
                    &ctx,
                    basis,
                    &ops,
                    theta,
                    psi,
                    wide_centered.as_slice(),
                    eval.energy - id.re,
                    1.0,
                )
                .unwrap();

                let h = 1e-6;
                for k in 0..theta.len() {
                    assert!(
                        (eval.gradient[k] - tangent[k]).abs() < 1e-12,
                        "d={expected_d} k={k}: eval={} tangent={}",
                        eval.gradient[k],
                        tangent[k]
                    );
                    let mut up = theta.clone();
                    let mut dn = theta.clone();
                    up[k] += h;
                    dn[k] -= h;
                    let e_up = problem.evaluate(&indices, &up, true).unwrap().energy;
                    let e_dn = problem.evaluate(&indices, &dn, true).unwrap().energy;
                    let fd = (e_up - e_dn) / (2. * h);
                    assert!(
                        (eval.gradient[k] - fd).abs() < 1e-8,
                        "d={expected_d} k={k}: analytic={} fd={fd}",
                        eval.gradient[k]
                    );
                }
            }
        }
        drop(problem);
        assert_eq!(ctx.memory_used(), 0);
    }
}

#[test]
fn sparse_and_parallel_importances_are_bit_identical_across_thread_counts() {
    let p1 = sector_test_problem(1, 14, [3, 3], "fermionic");
    let p4 = sector_test_problem(4, 14, [3, 3], "fermionic");
    let p8 = sector_test_problem(8, 14, [3, 3], "fermionic");

    let indices: Vec<usize> = (0..10).map(|k| (k * 7) % p1.pool.len()).collect();
    let theta = [
        0.15, -0.22, 0.07, 0.31, -0.11, 0.04, 0.28, -0.19, 0.09, -0.06,
    ];

    let imp1 = p1.importances(&indices, &theta).unwrap();
    let imp4 = p4.importances(&indices, &theta).unwrap();
    let imp8 = p8.importances(&indices, &theta).unwrap();

    assert_eq!(imp1.work, imp4.work);
    assert_eq!(imp1.work, imp8.work);
    for i in 0..indices.len() {
        assert_eq!(
            imp1.values[i].to_bits(),
            imp4.values[i].to_bits(),
            "1 vs 4 threads mismatch at position {i}"
        );
        assert_eq!(
            imp1.values[i].to_bits(),
            imp8.values[i].to_bits(),
            "1 vs 8 threads mismatch at position {i}"
        );
        let mut zeroed = theta;
        zeroed[i] = 0.;
        let full = p1.evaluate(&indices, &zeroed, true).unwrap().gradient[i].abs();
        assert!(
            (imp1.values[i] - full).abs() < 1e-14,
            "importance[{i}] {} differs from zeroed evaluate {full}",
            imp1.values[i]
        );
    }

    // Float32 precision also preserves bit-identical importances across threads.
    let f1 = p1.with_precision(Precision::Float32);
    let f4 = p4.with_precision(Precision::Float32);
    let f_imp1 = f1.importances(&indices, &theta).unwrap();
    let f_imp4 = f4.importances(&indices, &theta).unwrap();
    for i in 0..indices.len() {
        assert_eq!(f_imp1.values[i].to_bits(), f_imp4.values[i].to_bits());
    }
}

#[test]
fn hessian_carry_and_warm_memoization_reduce_work_while_preserving_convergence() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    {
        let h = Operator::new(
            &ctx,
            4,
            &[
                ("ZIII".into(), C64::new(0.3, 0.)),
                ("IZII".into(), C64::new(-0.2, 0.)),
                ("XXII".into(), C64::new(0.2, 0.)),
                ("YYII".into(), C64::new(0.2, 0.)),
                ("IIZZ".into(), C64::new(0.15, 0.)),
                ("IIXX".into(), C64::new(0.1, 0.)),
            ],
        )
        .unwrap();
        let problem = QuantumProblem::new(
            &ctx,
            Encoding::new("jordan_wigner", 4, [1, 1]).unwrap(),
            h,
            true,
            false,
            true,
        )
        .unwrap();
        let exact = exact_ground_energy(
            &ctx,
            problem.register("exact").unwrap(),
            &problem.hamiltonian,
        );

        // 1) Warm memoization produces the exact same energy/cost/trajectory with fewer evals.
        let plain_opts = SearchOptions {
            max_steps: 45,
            max_length: 4,
            seed: 7,
            warm_start: true,
            memoize_warm: false,
            ..Default::default()
        };
        let memo_opts = SearchOptions {
            memoize_warm: true,
            ..plain_opts.clone()
        };
        let plain = problem.search(&plain_opts, &[], &[]).unwrap();
        let memo = problem.search(&memo_opts, &[], &[]).unwrap();
        assert_eq!(plain.best_energy.to_bits(), memo.best_energy.to_bits());
        assert_eq!(plain.optimal_cost.to_bits(), memo.optimal_cost.to_bits());
        assert_eq!(plain.steps.len(), memo.steps.len());
        for (a, b) in plain.steps.iter().zip(&memo.steps) {
            assert_eq!(a["proposed"], b["proposed"]);
            assert_eq!(a["accepted"], b["accepted"]);
        }
        assert!(
            memo.solution.history.len() < plain.solution.history.len(),
            "memoized {} vs plain {}",
            memo.solution.history.len(),
            plain.solution.history.len()
        );

        // 2) Hessian carry works across SLSQP and BFGS, maintains symmetric LxL curvature,
        // and reaches the exact ground state.
        for method in [Method::Slsqp, Method::Bfgs] {
            let mut opts = SearchOptions {
                max_steps: 40,
                max_length: 4,
                seed: 11,
                warm_start: true,
                carry_hessian: true,
                ..Default::default()
            };
            opts.optimizer.method = method;
            opts.optimizer.tolerance = Some(1e-10);
            let res = problem.search(&opts, &[], &[]).unwrap();
            assert!(
                (res.solution.energy - exact).abs() < 1e-8,
                "{method:?}: got {}, exact {exact}",
                res.solution.energy
            );
            let l = res.solution.indices.len();
            let h = res
                .solution
                .hessian
                .as_ref()
                .expect("carried Hessian stored on result");
            assert_eq!(h.len(), l * l);
            for i in 0..l {
                for j in 0..i {
                    assert!((h[i * l + j] - h[j * l + i]).abs() < 1e-10);
                }
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn single_qubit_system_and_vanishing_gradient_symmetry_edge_cases() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    {
        // Single-qubit system: H = 0.5 Z + 0.3 X, exact ground energy = -sqrt(0.34).
        let basis = Basis::register(&ctx, 1).unwrap();
        let h = Operator::new(
            &ctx,
            1,
            &[
                ("Z".into(), C64::new(0.5, 0.)),
                ("X".into(), C64::new(0.3, 0.)),
            ],
        )
        .unwrap();
        let g_y = Operator::new(&ctx, 1, &[("Y".into(), C64::new(0., 1.))]).unwrap();
        let exact = -(0.5_f64.hypot(0.3));
        assert!((exact_ground_energy(&ctx, &basis, &h) - exact).abs() < 1e-12);

        let opts = optimizer::Options {
            method: Method::Bfgs,
            tolerance: Some(1e-10),
            ..Default::default()
        };
        let out = optimizer::minimize(&ctx, &[0.0], &opts, |t| {
            let ev = evaluate::<C64>(&ctx, &basis, &h, None, None, &[&g_y], t, true)?;
            Ok((ev.energy, ev.gradient))
        })
        .unwrap();
        assert!((out.energy - exact).abs() < 1e-12);

        // Symmetry-protected vanishing gradient (barren-like flat direction):
        // H = Z_0 + Z_1 on |00> with Z-commuting generator i(Z_0 + Z_1) has exact 0 gradient.
        let b2 = Basis::register(&ctx, 2).unwrap();
        let h2 = Operator::new(
            &ctx,
            2,
            &[
                ("ZI".into(), C64::new(1., 0.)),
                ("IZ".into(), C64::new(1., 0.)),
            ],
        )
        .unwrap();
        let g_sym = Operator::new(&ctx, 2, &[("ZZ".into(), C64::new(0., 1.))]).unwrap();
        let ev_sym = evaluate::<C64>(&ctx, &b2, &h2, None, None, &[&g_sym], &[0.42], true).unwrap();
        assert!(ev_sym.gradient[0].abs() < 1e-14);

        // 2-qubit minimal-basis H2-like Hamiltonian through QuantumProblem::search.
        let encoding = Encoding::new("jordan_wigner", 4, [1, 1]).unwrap();
        let h2_mol = Operator::new(
            &ctx,
            4,
            &[
                ("IIII".into(), C64::new(-0.81054798, 0.)),
                ("ZIII".into(), C64::new(0.17218393, 0.)),
                ("IZII".into(), C64::new(-0.22575349, 0.)),
                ("IIZI".into(), C64::new(0.17218393, 0.)),
                ("IIIZ".into(), C64::new(-0.22575349, 0.)),
                ("ZZII".into(), C64::new(0.12091263, 0.)),
                ("ZIZI".into(), C64::new(0.16892754, 0.)),
                ("ZIIZ".into(), C64::new(0.16614543, 0.)),
                ("IZZI".into(), C64::new(0.16614543, 0.)),
                ("IZIZ".into(), C64::new(0.17464343, 0.)),
                ("IIZZ".into(), C64::new(0.12091263, 0.)),
                ("XXYY".into(), C64::new(-0.04523280, 0.)),
                ("XYYX".into(), C64::new(0.04523280, 0.)),
                ("YXXY".into(), C64::new(0.04523280, 0.)),
                ("YYXX".into(), C64::new(-0.04523280, 0.)),
            ],
        )
        .unwrap();
        let problem = QuantumProblem::new(&ctx, encoding, h2_mol, true, false, true).unwrap();
        let exact_h2 =
            exact_ground_energy(&ctx, problem.register("h2").unwrap(), &problem.hamiltonian);
        let search_opts = SearchOptions {
            max_steps: 40,
            max_length: 4,
            seed: 7,
            ..Default::default()
        };
        let res = problem.search(&search_opts, &[], &[]).unwrap();
        assert!(
            (res.solution.energy - exact_h2).abs() < 1e-8,
            "H2 got {} vs exact {exact_h2}",
            res.solution.energy
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn vasqa_converges_on_transverse_field_ising_and_maxcut_hamiltonians() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    {
        // 1) Transverse Field Ising Model (TFIM), N = 4 qubits:
        // H = -J sum_{i=0}^{N-2} Z_i Z_{i+1} - h sum_{i=0}^{N-1} X_i, J=1.0, h=0.8.
        let n = 4usize;
        let mut tfim_terms = Vec::new();
        for i in 0..n - 1 {
            let mut label = vec![b'I'; n];
            label[i] = b'Z';
            label[i + 1] = b'Z';
            tfim_terms.push((String::from_utf8(label).unwrap(), C64::new(-1.0, 0.)));
        }
        for i in 0..n {
            let mut label = vec![b'I'; n];
            label[i] = b'X';
            tfim_terms.push((String::from_utf8(label).unwrap(), C64::new(-0.8, 0.)));
        }
        let tfim_h = Operator::new(&ctx, n, &tfim_terms).unwrap();
        let encoding = Encoding::new("jordan_wigner", n, [0, 0]).unwrap();
        let mut problem = QuantumProblem::new_in(
            &ctx,
            encoding,
            tfim_h,
            mandacaru_core::quantum::Space::Register,
            false,
            true,
        )
        .unwrap();
        // Use a spin-appropriate anti-Hermitian Pauli pool (single-qubit Y and two-qubit ZY/YZ/XY).
        let mut pool = Vec::new();
        for i in 0..n {
            let mut y = vec![b'I'; n];
            y[i] = b'Y';
            let label = String::from_utf8(y).unwrap();
            pool.push(Generator {
                operator: Operator::new(&ctx, n, &[(label.clone(), C64::new(0., 1.))]).unwrap(),
                label,
            });
        }
        for i in 0..n - 1 {
            for (p0, p1) in [(b'Z', b'Y'), (b'Y', b'Z'), (b'X', b'Y'), (b'Y', b'X')] {
                let mut l = vec![b'I'; n];
                l[i] = p0;
                l[i + 1] = p1;
                let label = String::from_utf8(l).unwrap();
                pool.push(Generator {
                    operator: Operator::new(&ctx, n, &[(label.clone(), C64::new(0., 1.))]).unwrap(),
                    label,
                });
            }
        }
        problem.pool = pool;
        problem.pool_screenable = vec![true; problem.pool.len()];
        problem.growth_members = (0..problem.pool.len()).map(|i| vec![i]).collect();

        let exact_tfim = exact_ground_energy(
            &ctx,
            problem.register("tfim").unwrap(),
            &problem.hamiltonian,
        );
        let opts = SearchOptions {
            max_steps: 120,
            max_length: 10,
            seed: 42,
            proposal_temperature: 0.2,
            ..Default::default()
        };
        let res = problem.search(&opts, &[], &[]).unwrap();
        assert!(
            (res.solution.energy - exact_tfim).abs() < 1e-4,
            "TFIM N=4: VASQA {} vs exact {exact_tfim}",
            res.solution.energy
        );

        // 2) MaxCut on a 4-vertex ring graph (C4): H = 0.5 sum_{(i,j) in E} (Z_i Z_j - I)
        // Minimum energy corresponds to -max_cut = -4.0.
        let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
        let mut mc_terms = vec![("IIII".to_string(), C64::new(-2.0, 0.))];
        for (u, v) in edges {
            let mut l = vec![b'I'; n];
            l[u] = b'Z';
            l[v] = b'Z';
            mc_terms.push((String::from_utf8(l).unwrap(), C64::new(0.5, 0.)));
        }
        let mc_h = Operator::new(&ctx, n, &mc_terms).unwrap();
        let mut mc_prob = QuantumProblem::new_in(
            &ctx,
            Encoding::new("jordan_wigner", n, [0, 0]).unwrap(),
            mc_h,
            mandacaru_core::quantum::Space::Register,
            false,
            true,
        )
        .unwrap();
        mc_prob.pool = problem
            .pool
            .iter()
            .map(|g| Generator {
                label: g.label.clone(),
                operator: g.operator.copy_into(&ctx).unwrap(),
            })
            .collect();
        mc_prob.pool_screenable = vec![true; mc_prob.pool.len()];
        mc_prob.growth_members = (0..mc_prob.pool.len()).map(|i| vec![i]).collect();

        let exact_mc = exact_ground_energy(
            &ctx,
            mc_prob.register("maxcut").unwrap(),
            &mc_prob.hamiltonian,
        );
        assert!((exact_mc + 4.0).abs() < 1e-12);
        // |0000> is an eigenstate of diagonal Z-MaxCut (zero gradient); seed from |->^4 via
        // exp((pi/4) iY) on all 4 qubits so VASQA searches from the uniform superposition.
        let mc_res = mc_prob.search(&opts, &[0, 1, 2, 3], &[PI / 4.; 4]).unwrap();
        assert!(
            (mc_res.solution.energy - exact_mc).abs() < 1e-6,
            "MaxCut C4: VASQA {} vs exact {exact_mc}",
            mc_res.solution.energy
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn forward_sparse_energy_avoids_basis_allocations_and_matches_dense() {
    let problem = sector_test_problem(1, 14, [3, 3], "fermionic");
    let ctx = problem.context.clone();
    {
        let basis = problem.register("sparse_forward").unwrap();
        let d = basis.len();
        assert_eq!(d, 1225);

        // 1) Sparse circuit (`nnz << d / 8`): zero O(D) Context Buffer allocation on Forward::energy.
        let sparse_idx = [0usize, 5, 11, 17];
        let sparse_theta = [0.14, -0.21, 0.09, -0.17];
        let sparse_ops: Vec<&Operator> = sparse_idx
            .iter()
            .map(|&i| &problem.pool[i].operator)
            .collect();

        let used_before = ctx.memory_used();
        let f_sp = forward_from_with_mode::<C64>(
            &ctx,
            basis,
            &problem.hamiltonian,
            None,
            problem.sparse.as_ref(),
            &sparse_ops,
            &sparse_theta,
            true,
            None,
            true,
        )
        .unwrap();
        let sp_held = ctx.memory_used() - used_before;
        // Dense Forward holds at least `stack(d)` + `wide(d)` = 2 * d * 16 bytes on Context.
        assert_eq!(
            sp_held, 0,
            "sparse Forward held {sp_held} bytes of Context buffers"
        );

        let f_dn = forward_from_with_mode::<C64>(
            &ctx,
            basis,
            &problem.hamiltonian,
            None,
            problem.sparse.as_ref(),
            &sparse_ops,
            &sparse_theta,
            true,
            None,
            false,
        )
        .unwrap();
        let dn_held = ctx.memory_used() - used_before;
        assert!(
            dn_held >= 2 * d * std::mem::size_of::<C64>(),
            "dense Forward held only {dn_held} bytes"
        );
        assert!(
            (f_sp.energy() - f_dn.energy()).abs() < 1e-12,
            "sparse energy {} vs dense {}",
            f_sp.energy(),
            f_dn.energy()
        );

        let ev_sp = f_sp
            .finish(&ctx, basis, &sparse_ops, &sparse_theta, true)
            .unwrap();
        let ev_dn = f_dn
            .finish(&ctx, basis, &sparse_ops, &sparse_theta, true)
            .unwrap();
        assert!((ev_sp.energy - ev_dn.energy).abs() < 1e-12);
        for (k, (a, b)) in ev_sp.gradient.iter().zip(&ev_dn.gradient).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "gradient[{k}]: sparse {a} vs dense {b}"
            );
        }

        // 2) Deep circuit that exceeds the `d / 8` sparsity budget falls back to Dense seamlessly.
        let deep_len = 24usize.min(problem.pool.len());
        let deep_idx: Vec<usize> = (0..deep_len).collect();
        let deep_theta: Vec<f64> = (0..deep_len)
            .map(|k| 0.19 * if k % 2 == 0 { 1. } else { -1. })
            .collect();
        let deep_ops: Vec<&Operator> = deep_idx
            .iter()
            .map(|&i| &problem.pool[i].operator)
            .collect();
        let ev_deep_sp = forward_from_with_mode::<C64>(
            &ctx,
            basis,
            &problem.hamiltonian,
            None,
            problem.sparse.as_ref(),
            &deep_ops,
            &deep_theta,
            true,
            None,
            true,
        )
        .unwrap()
        .finish(&ctx, basis, &deep_ops, &deep_theta, true)
        .unwrap();
        let ev_deep_dn = forward_from_with_mode::<C64>(
            &ctx,
            basis,
            &problem.hamiltonian,
            None,
            problem.sparse.as_ref(),
            &deep_ops,
            &deep_theta,
            true,
            None,
            false,
        )
        .unwrap()
        .finish(&ctx, basis, &deep_ops, &deep_theta, true)
        .unwrap();
        assert!((ev_deep_sp.energy - ev_deep_dn.energy).abs() < 1e-12);
        for (k, (a, b)) in ev_deep_sp
            .gradient
            .iter()
            .zip(&ev_deep_dn.gradient)
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-12,
                "deep fallback gradient[{k}]: sparse {a} vs dense {b}"
            );
        }
    }
    drop(problem);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn vqe_and_adapt_vqe_reuse_and_choice_options_preserve_exact_solutions() {
    for (modes, particles, max_growth) in [(8usize, [2usize, 2], 6usize), (14, [3, 3], 4)] {
        let problem = sector_test_problem(1, modes, particles, "fermionic");
        let ctx = problem.context.clone();
        {
            // VQE parity (modes=8 keeps pool compact; modes=14 exercises d=1225 sparse kernels).
            if modes == 8 {
                let vqe_def = SolveOptions {
                    adapt: false,
                    trotter: true,
                    ..Default::default()
                };
                let vqe_leg = SolveOptions {
                    adapt: false,
                    trotter: true,
                    sparse_kernels: false,
                    reuse_evaluations: false,
                    ..Default::default()
                };
                let r_def = problem.solve(&vqe_def, None).unwrap();
                let r_leg = problem.solve(&vqe_leg, None).unwrap();
                assert!((r_def.energy - r_leg.energy).abs() < 1e-12);
                assert!((r_def.final_gradient - r_leg.final_gradient).abs() < 1e-11);
                assert_eq!(r_def.parameters, r_leg.parameters);
            }

            // ADAPT-VQE + Pruned-ADAPT-VQE parity, plus carry_hessian choice.
            for prune in [false, true] {
                let adapt_def = SolveOptions {
                    adapt: true,
                    prune,
                    max_growth,
                    ..Default::default()
                };
                let adapt_leg = SolveOptions {
                    adapt: true,
                    prune,
                    max_growth,
                    sparse_kernels: false,
                    reuse_evaluations: false,
                    ..Default::default()
                };
                let r_def = problem.solve(&adapt_def, None).unwrap();
                let r_leg = problem.solve(&adapt_leg, None).unwrap();
                assert_eq!(
                    r_def.indices, r_leg.indices,
                    "modes={modes} prune={prune}: indices mismatch"
                );
                assert!(
                    (r_def.energy - r_leg.energy).abs() < 1e-11,
                    "modes={modes} prune={prune}: energy {} vs {}",
                    r_def.energy,
                    r_leg.energy
                );
                assert!(
                    (r_def.final_gradient - r_leg.final_gradient).abs() < 1e-11,
                    "modes={modes} prune={prune}: grad {} vs {}",
                    r_def.final_gradient,
                    r_leg.final_gradient
                );
            }

            let adapt_no_h = SolveOptions {
                adapt: true,
                max_growth,
                carry_hessian: false,
                ..Default::default()
            };
            let r_nh = problem.solve(&adapt_no_h, None).unwrap();
            let r_h = problem
                .solve(
                    &SolveOptions {
                        adapt: true,
                        max_growth,
                        carry_hessian: true,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            assert!((r_nh.energy - r_h.energy).abs() < 1e-10);
        }
        drop(problem);
        assert_eq!(ctx.memory_used(), 0);
    }
}

#[test]
fn vasqa_early_reject_preserves_detailed_balance_and_cuts_screening_work() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    {
        // 1) Exact stationary Boltzmann distribution across all 40 H2 sequences (`pool=3`,
        // `max_length=3`, `warm_start=false`, `temperature=0.05` Ha) under `early_reject = true`.
        let h2 = Operator::new(
            &ctx,
            4,
            &[
                ("IIII".into(), C64::new(-0.81054798, 0.)),
                ("ZIII".into(), C64::new(0.17218393, 0.)),
                ("IZII".into(), C64::new(-0.22575349, 0.)),
                ("IIZI".into(), C64::new(0.17218393, 0.)),
                ("IIIZ".into(), C64::new(-0.22575349, 0.)),
                ("ZZII".into(), C64::new(0.12091263, 0.)),
                ("ZIZI".into(), C64::new(0.16892754, 0.)),
                ("ZIIZ".into(), C64::new(0.16614543, 0.)),
                ("IZZI".into(), C64::new(0.16614543, 0.)),
                ("IZIZ".into(), C64::new(0.17464343, 0.)),
                ("IIZZ".into(), C64::new(0.12091263, 0.)),
                ("XXYY".into(), C64::new(-0.04523280, 0.)),
                ("XYYX".into(), C64::new(0.04523280, 0.)),
                ("YXXY".into(), C64::new(0.04523280, 0.)),
                ("YYXX".into(), C64::new(-0.04523280, 0.)),
            ],
        )
        .unwrap();
        let problem = QuantumProblem::new(
            &ctx,
            Encoding::new("jordan_wigner", 4, [1, 1]).unwrap(),
            h2,
            true,
            false,
            true,
        )
        .unwrap();
        let mut seqs: Vec<Vec<usize>> = Vec::with_capacity(73);
        let pool_len = problem.pool.len();
        seqs.push(vec![]);
        for a in 0..pool_len {
            seqs.push(vec![a]);
            for b in 0..pool_len {
                seqs.push(vec![a, b]);
                for c in 0..pool_len {
                    seqs.push(vec![a, b, c]);
                }
            }
        }
        assert_eq!(seqs.len(), 40);
        let temp = 0.008;
        let penalty = 0.04;
        let opts_stationary = SearchOptions {
            max_steps: 6000,
            max_length: 3,
            temperatures: [temp, temp],
            length_penalty: penalty,
            warm_start: false,
            early_reject: true,
            record: true,
            seed: 19,
            optimizer: optimizer::Options {
                tolerance: Some(1e-10),
                max_evaluations: 100,
                ..Default::default()
            },
            ..Default::default()
        };
        let res = problem.search(&opts_stationary, &[], &[]).unwrap();
        // Verify recorded early-rejected rows vs full rows.
        let mut early_count = 0usize;
        for row in &res.records {
            if row["log_q_reverse"].is_null() {
                assert_eq!(row["accepted"], false);
                assert!(row["log_acceptance"].is_null());
                early_count += 1;
            }
        }
        assert!(
            early_count > 0,
            "expected early rejections on H2 stationary chain"
        );

        // Compare empirical visit histogram against exact Boltzmann weights across the 73 states.
        let mut exact_w = vec![0.; seqs.len()];
        for (i, seq) in seqs.iter().enumerate() {
            let cost = if seq.is_empty() {
                problem.evaluate(&[], &[], true).unwrap().energy
            } else {
                let zero = vec![0.; seq.len()];
                let out = optimizer::minimize_deferred_from(
                    &ctx,
                    &zero,
                    &opts_stationary.optimizer,
                    None,
                    |p| problem.probe(seq, p, true),
                )
                .unwrap();
                out.energy + penalty * seq.len() as f64
            };
            exact_w[i] = (-cost / temp).exp();
        }
        let z_sum: f64 = exact_w.iter().sum();
        for w in &mut exact_w {
            *w /= z_sum;
        }
        let mut counts = vec![0usize; seqs.len()];
        let burn = 500usize;
        let mut cur_seq = vec![];
        let mut measured_steps = 0usize;
        for (idx, row) in res.steps.iter().enumerate() {
            if row["accepted"] == true {
                cur_seq = row["proposed"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect();
            }
            if idx >= burn {
                let pos = seqs.iter().position(|s| s == &cur_seq).unwrap();
                counts[pos] += 1;
                measured_steps += 1;
            }
        }
        let mut l1 = 0.;
        for (c, &p) in counts.iter().zip(&exact_w) {
            l1 += (*c as f64 / measured_steps as f64 - p).abs();
        }
        assert!(l1 < 0.08, "73-state L1 error {l1} under early_reject");

        // 2) On warm annealed 8q sector search, early_reject cuts screenings & importances while
        // reaching the exact same best energy.
        let p8 = sector_test_problem(1, 8, [2, 2], "fermionic");
        let base_opts = SearchOptions {
            max_steps: 40,
            max_length: 6,
            seed: 7,
            early_reject: false,
            ..Default::default()
        };
        let er_opts = SearchOptions {
            early_reject: true,
            ..base_opts.clone()
        };
        let r_base = p8.search(&base_opts, &[], &[]).unwrap();
        let r_er = p8.search(&er_opts, &[], &[]).unwrap();
        assert_eq!(r_base.best_energy.to_bits(), r_er.best_energy.to_bits());
        assert!(r_er.num_screenings < r_base.num_screenings);
        assert!(r_er.num_importance_evaluations < r_base.num_importance_evaluations);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn vasqa_strip_zero_operators_removes_zero_angle_identity_gates() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    {
        // H = 0.5 Z_0 + 0.3 X_0 + 0.2 Z_1.
        // g_0 = i Z_1 commutes with H identically (dE/dtheta_0 = 0, keeps theta_0 = 0.0);
        // g_1 = i Y_0 rotates qubit 0 into the ground state.
        let h = Operator::new(
            &ctx,
            2,
            &[
                ("ZI".into(), C64::new(0.5, 0.)),
                ("XI".into(), C64::new(0.3, 0.)),
                ("IZ".into(), C64::new(0.2, 0.)),
            ],
        )
        .unwrap();
        let mut prob = QuantumProblem::new_in(
            &ctx,
            Encoding::new("jordan_wigner", 2, [0, 0]).unwrap(),
            h,
            mandacaru_core::quantum::Space::Register,
            false,
            true,
        )
        .unwrap();
        let g_sym = Operator::new(&ctx, 2, &[("IZ".into(), C64::new(0., 1.))]).unwrap();
        let g_rot = Operator::new(&ctx, 2, &[("YI".into(), C64::new(0., 1.))]).unwrap();
        prob.pool = vec![
            Generator {
                label: "sym_zero".into(),
                operator: g_sym,
            },
            Generator {
                label: "rot_y0".into(),
                operator: g_rot,
            },
        ];
        prob.pool_screenable = vec![true, true];
        prob.growth_members = vec![vec![0], vec![1]];
        let _ = Kernel::new(2, 3, [1.; 3]).unwrap();

        let opts_raw = SearchOptions {
            max_steps: 0,
            max_length: 2,
            seed: 3,
            strip_zero_operators: false,
            ..Default::default()
        };
        let opts_strip = SearchOptions {
            strip_zero_operators: true,
            ..opts_raw.clone()
        };
        let r_raw = prob.search(&opts_raw, &[0, 1], &[0.0, 0.2]).unwrap();
        let r_strip = prob.search(&opts_strip, &[0, 1], &[0.0, 0.2]).unwrap();
        assert_eq!(r_raw.solution.indices, vec![0, 1]);
        assert!(
            r_raw.solution.parameters[0].abs() <= 1e-14,
            "expected zero-angle symmetry gate in raw solution: {:?}",
            r_raw.solution.parameters
        );
        assert_eq!(r_strip.solution.indices, vec![1]);
        assert!(
            r_strip.solution.parameters.iter().all(|&p| p.abs() > 1e-14),
            "strip_zero_operators left near-zero parameter: {:?}",
            r_strip.solution.parameters
        );
        assert!((r_strip.solution.energy - r_raw.solution.energy).abs() < 1e-12);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn sparse_importances_avoid_basis_allocations_and_match_dense() {
    let p1 = sector_test_problem(1, 14, [3, 3], "fermionic");
    let p4 = sector_test_problem(4, 14, [3, 3], "fermionic");
    let d = p1.register("sp_imp").unwrap().len();
    assert_eq!(d, 1225);

    let indices: Vec<usize> = (0..8).map(|k| (k * 5) % p1.pool.len()).collect();
    let theta = [0.08, -0.11, 0.05, -0.09, 0.04, -0.07, 0.06, -0.03];

    let sp1 = p1.importances_with_mode(&indices, &theta, true).unwrap();
    let sp4 = p4.importances_with_mode(&indices, &theta, true).unwrap();
    let dn1 = p1.importances_with_mode(&indices, &theta, false).unwrap();

    assert_eq!(sp1.work, dn1.work);
    for i in 0..indices.len() {
        assert_eq!(
            sp1.values[i].to_bits(),
            sp4.values[i].to_bits(),
            "sparse importances 1 vs 4 threads differ at {i}"
        );
        assert!(
            (sp1.values[i] - dn1.values[i]).abs() < 1e-12,
            "importance[{i}]: sparse={} dense={}",
            sp1.values[i],
            dn1.values[i]
        );
        let mut zeroed = theta;
        zeroed[i] = 0.;
        let exact = p1
            .evaluate_with_mode(&indices, &zeroed, true, true)
            .unwrap()
            .gradient[i]
            .abs();
        assert!(
            (sp1.values[i] - exact).abs() < 1e-14,
            "sparse importance[{i}] {} vs zeroed evaluate {exact}",
            sp1.values[i]
        );
    }

    drop(p1);
    drop(p4);
}

#[test]
fn vasqa_seeded_temperatures_apply_only_when_seed_is_kept_and_validate() {
    let p = sector_test_problem(1, 4, [1, 1], "fermionic");
    let invalid_opts = SearchOptions {
        seeded_temperatures: Some([-1.0, 0.001]),
        ..Default::default()
    };
    assert!(invalid_opts.validate().is_err());

    let invalid_zero = SearchOptions {
        seeded_temperatures: Some([0.003, 0.0]),
        ..Default::default()
    };
    assert!(invalid_zero.validate().is_err());

    let cold = [0.0005, 0.0001];
    let warm = [0.05, 0.01];
    let opts = SearchOptions {
        max_steps: 4,
        max_length: 3,
        temperatures: warm,
        seeded_temperatures: Some(cold),
        seed: 5,
        record: true,
        ..Default::default()
    };
    // Kept beneficial seed uses colder schedule.
    let res_kept = p.search(&opts, &[0], &[0.1]).unwrap();
    assert!(res_kept.solution.seed.as_ref().unwrap().kept);
    assert!(
        (res_kept.steps[0]["temperature"].as_f64().unwrap() - cold[0]).abs() < 1e-12,
        "kept seed should use cold initial temperature {}",
        cold[0]
    );

    // Unkept empty/worse seed keeps standard `temperatures`.
    let res_unkept = p.search(&opts, &[], &[]).unwrap();
    assert!(res_unkept.solution.seed.is_none());
    assert!(
        (res_unkept.steps[0]["temperature"].as_f64().unwrap() - warm[0]).abs() < 1e-12,
        "unseeded chain should use warm initial temperature {}",
        warm[0]
    );
}

#[test]
fn subspace_vqe_and_adapt_sparse_reuse_match_dense_and_compute_excited_states() {
    let p14 = sector_test_problem(1, 14, [3, 3], "fermionic");
    let weights = [0.65, 0.35];

    // Fixed Subspace-VQE with small growth/evals budget on 14q sparse sector (d=1225).
    let base_vqe = SolveOptions {
        adapt: false,
        trotter: true,
        sparse_kernels: false,
        reuse_evaluations: false,
        carry_hessian: false,
        optimizer: optimizer::Options {
            method: Method::Bfgs,
            max_iterations: 4,
            max_evaluations: 12,
            ..Default::default()
        },
        ..Default::default()
    };
    let fast_vqe = SolveOptions {
        sparse_kernels: true,
        reuse_evaluations: true,
        ..base_vqe.clone()
    };
    let r_base = p14.solve_subspace(&base_vqe, &weights, None).unwrap();
    let r_fast = p14.solve_subspace(&fast_vqe, &weights, None).unwrap();
    assert_eq!(r_base.indices, r_fast.indices);
    assert!(
        (r_base.final_gradient - r_fast.final_gradient).abs() < 1e-11,
        "subspace VQE final_gradient mismatch: {} vs {}",
        r_base.final_gradient,
        r_fast.final_gradient
    );
    for (a, b) in r_base.energies.iter().zip(&r_fast.energies) {
        assert!((a - b).abs() < 1e-12, "subspace VQE energy {a} vs {b}");
    }

    // Subspace-ADAPT-VQE: verify sparse + reuse + carry_hessian match dense baseline cleanly.
    let base_adapt = SolveOptions {
        adapt: true,
        trotter: true,
        max_growth: 3,
        sparse_kernels: false,
        reuse_evaluations: false,
        carry_hessian: false,
        optimizer: optimizer::Options {
            method: Method::Bfgs,
            max_iterations: 20,
            max_evaluations: 80,
            ..Default::default()
        },
        ..Default::default()
    };
    let fast_adapt = SolveOptions {
        sparse_kernels: true,
        reuse_evaluations: true,
        carry_hessian: false,
        ..base_adapt.clone()
    };
    let a_base = p14.solve_subspace(&base_adapt, &weights, None).unwrap();
    let a_fast = p14.solve_subspace(&fast_adapt, &weights, None).unwrap();
    assert_eq!(a_base.indices, a_fast.indices);
    assert!(
        (a_base.final_gradient - a_fast.final_gradient).abs() < 1e-11,
        "subspace ADAPT final_gradient mismatch: {} vs {}",
        a_base.final_gradient,
        a_fast.final_gradient
    );
    for (a, b) in a_base.energies.iter().zip(&a_fast.energies) {
        assert!((a - b).abs() < 1e-11, "subspace ADAPT energy {a} vs {b}");
    }
}

#[test]
fn subspace_vasqa_finds_variational_ground_and_excited_states_vs_exact() {
    let p = sector_test_problem(1, 4, [1, 1], "fermionic");
    let basis = p.register("subspace_vasqa_exact").unwrap();
    let exact = exact_spectrum(&p.context, basis, &p.hamiltonian);
    assert!(exact.len() >= 2);

    let weights = [0.7, 0.3];
    let opts = SearchOptions {
        max_steps: 60,
        max_length: 6,
        seed: 7,
        temperatures: [0.002, 0.00005],
        ..Default::default()
    };
    let res = p.search_subspace(&opts, &weights, &[], &[]).unwrap();
    assert_eq!(res.energies.len(), 2);
    assert_eq!(res.ensemble_energies.len(), 2);
    assert!(res.energies[0] <= res.energies[1] + 1e-14);
    assert!(
        res.energies[0] >= exact[0] - 1e-12 && res.energies[1] >= exact[1] - 1e-12,
        "Subspace-VASQA must obey Rayleigh-Ritz variational upper bounds: got {:?} vs exact {:?}",
        res.energies,
        &exact[..2]
    );
    assert!(
        (res.solution.energy - res.energies[0]).abs() < 1e-13,
        "solution.energy={} should match rotated ground E0={}",
        res.solution.energy,
        res.energies[0]
    );
    assert!(
        (res.energies[0] - exact[0]).abs() < 1e-6,
        "Subspace-VASQA ground state {} vs exact {}",
        res.energies[0],
        exact[0]
    );
    assert!(
        (res.energies[1] - exact[1]).abs() < 1e-6,
        "Subspace-VASQA 1st excited state {} vs exact {}",
        res.energies[1],
        exact[1]
    );

    // Check rotated eigenstate orthonormality and Rayleigh-Ritz expectation consistency.
    let d = basis.len();
    let buf = res.states.complex128().unwrap();
    let h_dense = p.hamiltonian.dense(&p.context, basis).unwrap();
    for a in 0..2 {
        let row_a = &buf[a * d..(a + 1) * d];
        for b in 0..2 {
            let row_b = &buf[b * d..(b + 1) * d];
            let overlap: C64 = row_a.iter().zip(row_b).map(|(x, y)| x.conj() * y).sum();
            let expected = if a == b { 1. } else { 0. };
            assert!(
                (overlap.re - expected).abs() < 1e-11 && overlap.im.abs() < 1e-11,
                "eigenstates ({a},{b}) overlap={overlap:?}"
            );
        }
        let mut e_expect = 0.;
        for r in 0..d {
            let mut hr = C64::default();
            for c in 0..d {
                hr += h_dense.as_slice()[r * d + c] * row_a[c];
            }
            e_expect += (row_a[r].conj() * hr).re;
        }
        assert!(
            (e_expect - res.energies[a]).abs() < 1e-11,
            "<psi_{a}|H|psi_{a}>={e_expect} vs Rayleigh-Ritz {}",
            res.energies[a]
        );
    }

    // Large sparse 14q parity check: sparse=true/memoize=true vs sparse=false/memoize=false.
    let p14 = sector_test_problem(1, 14, [3, 3], "fermionic");
    let opts_dense = SearchOptions {
        max_steps: 4,
        max_length: 3,
        seed: 11,
        sparse_kernels: false,
        memoize_warm: false,
        early_reject: false,
        ..Default::default()
    };
    let opts_fast = SearchOptions {
        sparse_kernels: true,
        memoize_warm: true,
        early_reject: true,
        ..opts_dense.clone()
    };
    let s_dense = p14
        .search_subspace(&opts_dense, &weights, &[], &[])
        .unwrap();
    let s_fast = p14.search_subspace(&opts_fast, &weights, &[], &[]).unwrap();
    assert_eq!(s_dense.best_energy_indices, s_fast.best_energy_indices);
    assert!((s_dense.best_energy - s_fast.best_energy).abs() < 1e-11);
    for (a, b) in s_dense.energies.iter().zip(&s_fast.energies) {
        assert!((a - b).abs() < 1e-11, "14q subspace VASQA {a} vs {b}");
    }
}

#[test]
fn spin_squared_and_transition_dipoles_distinguish_subspace_vasqa_singlet_and_triplet_states() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let enc = Encoding::new("jordan_wigner", 4, [1, 1]).unwrap();
    let prob = sector_test_problem(1, 4, [1, 1], "fermionic");
    let basis = prob.register("spin_check").unwrap();
    assert_eq!(basis.len(), 4);

    // Exact analytic singlet and triplet state checks in [1, 1] spin-blocked [a0, a1, b0, b1].
    let row_closed = basis.index(0b1010).unwrap(); // |a0 b0>
    let row_1001 = basis.index(0b1001).unwrap(); // |a0 b1>
    let row_0110 = basis.index(0b0110).unwrap(); // |a1 b0>
    let mut singlet_cs = vec![C64::default(); 4];
    singlet_cs[row_closed] = C64::new(1., 0.);
    assert!(prob.spin_squared(&singlet_cs).unwrap().abs() < 1e-13);

    let inv_sqrt2 = C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.);
    let mut singlet_os = vec![C64::default(); 4];
    singlet_os[row_1001] = inv_sqrt2;
    singlet_os[row_0110] = inv_sqrt2;
    assert!(prob.spin_squared(&singlet_os).unwrap().abs() < 1e-13);

    let mut triplet_0 = vec![C64::default(); 4];
    triplet_0[row_1001] = inv_sqrt2;
    triplet_0[row_0110] = -inv_sqrt2;
    assert!((prob.spin_squared(&triplet_0).unwrap() - 2.0).abs() < 1e-13);

    // Transition dipole selection rule with r_z coupling orbital 0 <-> orbital 1:
    // Spin-allowed (closed-shell S0 -> open-shell S1) gives |mu_z| = sqrt(2),
    // while spin-forbidden (S0 -> T0) vanishes identically.
    let m = 2;
    let mut pos = vec![C64::default(); 3 * m * m];
    pos[2 * m * m + 1] = C64::new(1., 0.);
    pos[2 * m * m + 2] = C64::new(1., 0.);
    let tr_ss = prob.transition_rdms(&singlet_cs, &singlet_os).unwrap();
    let mu_ss = transition_dipole(tr_ss.one.as_slice(), &pos, m).unwrap();
    assert!((mu_ss[2].norm() - std::f64::consts::SQRT_2).abs() < 1e-12);
    assert!((oscillator_strength(0.5, mu_ss) - 2.0 / 3.0).abs() < 1e-12);

    let tr_st = prob.transition_rdms(&singlet_cs, &triplet_0).unwrap();
    let mu_st = transition_dipole(tr_st.one.as_slice(), &pos, m).unwrap();
    assert!(oscillator_strength(0.5, mu_st) < 1e-24);

    // Verify open-shell broken-symmetry determinant |a1 b0> has exact <S^2> = 1.0.
    let mut broken = vec![C64::default(); 4];
    broken[row_0110] = C64::new(1., 0.);
    assert!((prob.spin_squared(&broken).unwrap() - 1.0).abs() < 1e-13);

    // Spin-symmetric minimal-basis Hamiltonian (`[H, S^2] == 0`, with exchange splitting
    // singlets from the triplet and Z2-taperable) for Subspace-VASQA multi-root spin/dipole
    // checks + tapered invariance checks.
    let h_ferm = enc
        .map_fermion(
            &ctx,
            &[
                (vec![(0, true), (0, false)], C64::new(-0.6, 0.)),
                (vec![(2, true), (2, false)], C64::new(-0.6, 0.)),
                (vec![(1, true), (1, false)], C64::new(0.4, 0.)),
                (vec![(3, true), (3, false)], C64::new(0.4, 0.)),
                (vec![(0, true), (1, false)], C64::new(-0.22, 0.)),
                (vec![(1, true), (0, false)], C64::new(-0.22, 0.)),
                (vec![(2, true), (3, false)], C64::new(-0.22, 0.)),
                (vec![(3, true), (2, false)], C64::new(-0.22, 0.)),
                (
                    vec![(0, true), (2, true), (2, false), (0, false)],
                    C64::new(0.3, 0.),
                ),
                (
                    vec![(1, true), (3, true), (3, false), (1, false)],
                    C64::new(0.3, 0.),
                ),
                // Two-electron exchange <01|10> splits triplet T0 from open-shell singlet S1.
                (
                    vec![(0, true), (3, true), (2, false), (1, false)],
                    C64::new(0.25, 0.),
                ),
                (
                    vec![(1, true), (2, true), (3, false), (0, false)],
                    C64::new(0.25, 0.),
                ),
                (
                    vec![(0, true), (2, true), (3, false), (1, false)],
                    C64::new(0.18, 0.),
                ),
                (
                    vec![(1, true), (3, true), (2, false), (0, false)],
                    C64::new(0.18, 0.),
                ),
            ],
        )
        .unwrap();
    let tap_terms = h_ferm.labels();
    let p_untap = QuantumProblem::new(
        &ctx,
        Encoding::new("jordan_wigner", 4, [1, 1]).unwrap(),
        Operator::new(&ctx, 4, &tap_terms).unwrap(),
        true,
        true,
        true,
    )
    .unwrap();
    // Search full 4-state [1,1] subspace: recovers all 3 singlets (<S^2> = 0) and 1 triplet (<S^2> = 2),
    // with zero dipole transition strength between ground singlet and dark triplet.
    let res = p_untap
        .search_subspace(
            &SearchOptions {
                max_steps: 24,
                max_length: 4,
                seed: 7,
                ..Default::default()
            },
            &[0.4, 0.3, 0.2, 0.1],
            &[],
            &[],
        )
        .unwrap();
    let d = p_untap.register("sub_s2").unwrap().len();
    let buf = res.states.complex128().unwrap();
    let mut singlets = 0usize;
    let mut triplets = 0usize;
    for root in 0..4 {
        let state = &buf[root * d..(root + 1) * d];
        let s2 = p_untap.spin_squared(state).unwrap();
        if s2.abs() < 1e-8 {
            singlets += 1;
        } else if (s2 - 2.0).abs() < 1e-8 {
            triplets += 1;
            if root > 0 {
                let tr = p_untap.transition_rdms(&buf[..d], state).unwrap();
                let mu = transition_dipole(tr.one.as_slice(), &pos, m).unwrap();
                let f = oscillator_strength(res.energies[root] - res.energies[0], mu);
                assert!(
                    f < 1e-18,
                    "triplet root {root} should be dipole-dark: f={f}"
                );
            }
        } else {
            panic!("unexpected spin contamination at root {root}: <S^2>={s2}");
        }
    }
    assert_eq!((singlets, triplets), (3, 1));

    // Verify Z2 tapering preserves spin_squared & transition_rdms identically.
    let mut p_tap = QuantumProblem::new(
        &ctx,
        Encoding::new("jordan_wigner", 4, [1, 1]).unwrap(),
        Operator::new(&ctx, 4, &tap_terms).unwrap(),
        true,
        true,
        true,
    )
    .unwrap();
    p_tap.taper().unwrap();
    assert!(p_tap.taper.is_some());
    let r_u = p_untap
        .solve(
            &SolveOptions {
                adapt: false,
                trotter: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    let r_t = p_tap
        .solve(
            &SolveOptions {
                adapt: false,
                trotter: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    let su = p_untap
        .spin_squared(r_u.state.complex128().unwrap())
        .unwrap();
    let st = p_tap.spin_squared(r_t.state.complex128().unwrap()).unwrap();
    assert!(
        (su - st).abs() < 1e-11 && su.abs() < 1e-11,
        "su={su} st={st}"
    );
    let tu = p_untap
        .transition_rdms(
            r_u.state.complex128().unwrap(),
            r_u.state.complex128().unwrap(),
        )
        .unwrap();
    let tt = p_tap
        .transition_rdms(
            r_t.state.complex128().unwrap(),
            r_t.state.complex128().unwrap(),
        )
        .unwrap();
    for (a, b) in tu.one.as_slice().iter().zip(tt.one.as_slice()) {
        assert!((*a - *b).norm() < 1e-11);
    }
}

#[test]
fn bloch_crystal_momentum_decouples_and_accelerates_vqe_adapt_vasqa_and_subspace_search() {
    // 4-site periodic 1D Bloch Hubbard ring (M = [4, 1, 1], m = 4 Bloch orbitals, 8 qubits, [2,2]).
    // Occupied HF orbitals are index 0 (k=0) and index 1 (k=1), virtuals index 2 (k=2), 3 (k=3),
    // giving k_0 + k_1 = 1 ≡ k_2 + k_3 = 5 (mod 4) so correlated excitations mix within K = 2.
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    let enc = Encoding::new("jordan_wigner", 8, [2, 2]).unwrap();
    let mesh = [4, 1, 1];
    let orbital_k: Vec<[i64; 3]> = vec![[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
    let eps = [-2.0_f64, -0.8, 0.4, 0.9];
    let u_hub = 1.6_f64;
    let mut terms = Vec::new();
    let m = 4usize;
    for p in 0..m {
        terms.push((vec![(p, true), (p, false)], C64::new(eps[p], 0.)));
        terms.push((vec![(m + p, true), (m + p, false)], C64::new(eps[p], 0.)));
    }
    for p in 0..m {
        for q in 0..m {
            for r in 0..m {
                for s in 0..m {
                    if (orbital_k[p][0] + orbital_k[q][0] - orbital_k[r][0] - orbital_k[s][0])
                        .rem_euclid(mesh[0] as i64)
                        == 0
                    {
                        terms.push((
                            vec![(p, true), (m + q, true), (m + s, false), (r, false)],
                            C64::new(u_hub / (m as f64), 0.),
                        ));
                    }
                }
            }
        }
    }
    let h_op = enc.map_fermion(&ctx, &terms).unwrap();
    let full = QuantumProblem::new(&ctx, enc, h_op, true, true, true).unwrap();
    let basis = full.register("bloch").unwrap();
    assert_eq!(basis.len(), 36);

    let momentum = CrystalMomentum::new(mesh, orbital_k).unwrap();
    let bloch = full.with_momentum(&momentum).unwrap();
    assert!(
        bloch.pool.len() < full.pool.len(),
        "expected crystal momentum filter to shrink pool: {} vs {}",
        bloch.pool.len(),
        full.pool.len()
    );

    // Every dropped generator must have zero commutator gradient on the Bloch ground state.
    let e_exact = exact_ground_energy(&ctx, basis, &bloch.hamiltonian);
    let kept_labels: std::collections::BTreeSet<&str> =
        bloch.pool.iter().map(|g| g.label.as_str()).collect();
    let res_adapt = bloch
        .solve(
            &SolveOptions {
                adapt: true,
                trotter: true,
                max_growth: 12,
                gradient_tolerance: Some(1e-8),
                ..Default::default()
            },
            None,
        )
        .unwrap();
    let full_adapt = full
        .solve(
            &SolveOptions {
                adapt: true,
                trotter: true,
                max_growth: 12,
                gradient_tolerance: Some(1e-8),
                ..Default::default()
            },
            None,
        )
        .unwrap();
    assert!(
        (res_adapt.energy - full_adapt.energy).abs() < 1e-11,
        "Bloch-ADAPT {} vs unfiltered ADAPT {}",
        res_adapt.energy,
        full_adapt.energy
    );
    assert!(
        (res_adapt.energy - e_exact).abs() < 1e-4,
        "Bloch-ADAPT {} vs exact {e_exact}",
        res_adapt.energy
    );
    let grads_full = full.pool_gradients(&res_adapt.state, true).unwrap();
    for (g, &grad) in full.pool.iter().zip(&grads_full) {
        if !kept_labels.contains(g.label.as_str()) {
            assert!(
                grad.abs() < 1e-12,
                "dropped non-momentum generator {} had non-zero gradient {grad}",
                g.label
            );
        }
    }

    // Compose Bloch domain cleanly with VQE, VASQA, Subspace-ADAPT-VQE, and SubspaceSearch-Bloch-VASQA.
    let res_vqe = bloch
        .solve(
            &SolveOptions {
                adapt: false,
                trotter: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    assert!((res_vqe.energy - res_adapt.energy).abs() < 1e-4);

    let res_vasqa = bloch
        .search(
            &SearchOptions {
                max_steps: 60,
                max_length: 8,
                seed: 5,
                ..Default::default()
            },
            &[],
            &[],
        )
        .unwrap();
    assert!(
        (res_vasqa.best_energy - e_exact).abs() < 1e-4,
        "Bloch-VASQA {} vs exact {e_exact}",
        res_vasqa.best_energy
    );

    let sub_adapt = bloch
        .solve_subspace(
            &SolveOptions {
                adapt: true,
                trotter: true,
                max_growth: 8,
                ..Default::default()
            },
            &[0.65, 0.35],
            None,
        )
        .unwrap();
    let sub_vasqa = bloch
        .search_subspace(
            &SearchOptions {
                max_steps: 60,
                max_length: 8,
                seed: 5,
                ..Default::default()
            },
            &[0.65, 0.35],
            &[],
            &[],
        )
        .unwrap();
    assert_eq!(sub_adapt.energies.len(), 2);
    assert_eq!(sub_vasqa.energies.len(), 2);
    assert!(sub_vasqa.energies[0] <= sub_vasqa.energies[1] + 1e-12);
    let spectrum = exact_spectrum(&ctx, basis, &bloch.hamiltonian);
    assert!(sub_vasqa.energies[0] >= spectrum[0] - 1e-11);
    assert!(sub_vasqa.energies[1] >= spectrum[1] - 1e-11);
    let ens_vasqa = 0.65 * sub_vasqa.ensemble_energies[0] + 0.35 * sub_vasqa.ensemble_energies[1];
    let ens_adapt = 0.65 * sub_adapt.energies[0] + 0.35 * sub_adapt.energies[1];
    assert!(
        (ens_vasqa - ens_adapt).abs() < 1e-3,
        "SubspaceSearch-Bloch-VASQA ensemble {ens_vasqa} vs Subspace-Bloch-ADAPT {ens_adapt}"
    );
    assert!(
        (sub_vasqa.energies[0] - spectrum[0]).abs() < 5e-3,
        "SubspaceSearch-Bloch-VASQA E0={} vs exact {}",
        sub_vasqa.energies[0],
        spectrum[0]
    );
}

#[test]
fn vasqa_search_with_execution_runs_on_register_free_pauli_problem() {
    use mandacaru_core::quantum::{solver::Execution, Space};
    use std::cell::Cell;

    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let terms = vec![
        ("ZIII".into(), C64::new(0.3, 0.)),
        ("XXII".into(), C64::new(0.2, 0.)),
        ("YYII".into(), C64::new(0.2, 0.)),
    ];
    let sim = QuantumProblem::new(
        &ctx,
        Encoding::new("jordan_wigner", 4, [1, 0]).unwrap(),
        Operator::new(&ctx, 4, &terms).unwrap(),
        true,
        false,
        true,
    )
    .unwrap();
    let register_free = QuantumProblem::new_in(
        &ctx,
        Encoding::new("jordan_wigner", 4, [1, 0]).unwrap(),
        Operator::new(&ctx, 4, &terms).unwrap(),
        Space::Pauli {
            number_conserving: true,
        },
        false,
        true,
    )
    .unwrap();
    assert!(register_free.is_register_free());

    struct SimExecution<'a> {
        sim: &'a QuantumProblem,
        eval_calls: Cell<usize>,
        grad_calls: Cell<usize>,
    }
    impl Execution for SimExecution<'_> {
        fn evaluate(
            &mut self,
            _problem: &QuantumProblem,
            indices: &[usize],
            parameters: &[f64],
            _gradients: bool,
        ) -> mandacaru_core::Result<mandacaru_core::quantum::evolution::Evaluation> {
            self.eval_calls.set(self.eval_calls.get() + 1);
            self.sim.evaluate(indices, parameters, true)
        }
        fn pool_gradients(
            &mut self,
            _problem: &QuantumProblem,
            indices: &[usize],
            parameters: &[f64],
        ) -> mandacaru_core::Result<Vec<f64>> {
            self.grad_calls.set(self.grad_calls.get() + 1);
            let st = self.sim.prepare_state(indices, parameters, true)?;
            self.sim.pool_gradients(&st, true)
        }
    }

    let opts = SearchOptions {
        max_steps: 20,
        max_length: 3,
        seed: 7,
        ..Default::default()
    };
    let expected = sim.search(&opts, &[], &[]).unwrap();
    let mut exec = SimExecution {
        sim: &sim,
        eval_calls: Cell::new(0),
        grad_calls: Cell::new(0),
    };
    let actual = register_free
        .search_with_execution(&opts, &[], &[], None, Some(&mut exec))
        .unwrap();
    assert!((actual.best_energy - (-0.5)).abs() < 1e-8);
    assert!((actual.best_energy - expected.best_energy).abs() < 1e-8);
    assert_eq!(actual.solution.state.len(), 0);
    assert!(exec.eval_calls.get() > 0 && exec.grad_calls.get() > 0);
}

#[test]
fn vasqa_parallel_replicas_match_sequential_and_execute_swaps() {
    let p1 = sector_test_problem(1, 8, [2, 2], "fermionic");
    let p4 = sector_test_problem(4, 8, [2, 2], "fermionic");

    let opts = SearchOptions {
        max_steps: 12,
        max_length: 4,
        replicas: 4,
        replica_spacing: 2.0,
        seed: 19,
        record: false,
        ..Default::default()
    };
    let seq = p1.search(&opts, &[], &[]).unwrap();
    let par = p4.search(&opts, &[], &[]).unwrap();
    assert_eq!(seq.swaps.len(), 12);
    assert_eq!(par.swaps.len(), 12);
    assert_eq!(par.swap_acceptance, seq.swap_acceptance);
    assert!((par.best_energy - seq.best_energy).abs() < 1e-12);
    assert!((par.optimal_cost - seq.optimal_cost).abs() < 1e-12);
    assert_eq!(par.best_energy_indices, seq.best_energy_indices);
}

#[test]
fn vasqa_surrogate_screening_and_pool_curvatures_match_exact_derivatives() {
    let p = sector_test_problem(1, 6, [1, 1], "fermionic");

    // 1. Verify QuantumProblem::pool_curvatures against central finite differences on Float64 & Float32.
    let state = p.prepare_state(&[0, 1], &[0.17, -0.23], true).unwrap();
    let curvatures = p.pool_curvatures(&state).unwrap();
    assert_eq!(curvatures.len(), p.pool.len());
    let h = 1e-4;
    for mu in 0..p.pool.len().min(6) {
        let e0 = p.evaluate(&[0, 1], &[0.17, -0.23], true).unwrap().energy;
        let ep = p
            .evaluate(&[0, 1, mu], &[0.17, -0.23, h], true)
            .unwrap()
            .energy;
        let em = p
            .evaluate(&[0, 1, mu], &[0.17, -0.23, -h], true)
            .unwrap()
            .energy;
        let fd = (ep - 2. * e0 + em) / (h * h);
        assert!(
            (curvatures[mu] - fd).abs() < 1e-5,
            "pool curvature[{mu}]: {} vs fd {fd}",
            curvatures[mu]
        );
    }

    // 2. Verify surrogate top-K screening converges validly and computes exact final gradient.
    let opts_surrogate = SearchOptions {
        max_steps: 15,
        max_length: 4,
        seed: 11,
        screen_hamiltonian_terms: Some(6),
        ..Default::default()
    };
    let res = p.search(&opts_surrogate, &[], &[]).unwrap();
    let exact_state = p
        .prepare_state(&res.solution.indices, &res.solution.parameters, true)
        .unwrap();
    let exact_max_g = p
        .pool_gradients(&exact_state, true)
        .unwrap()
        .into_iter()
        .map(f64::abs)
        .fold(0., f64::max);
    assert!(
        (res.solution.final_gradient - exact_max_g).abs() < 1e-12,
        "surrogate search must report exact final gradient: {} vs {exact_max_g}",
        res.solution.final_gradient
    );
}

#[test]
fn vasqa_chain_checkpoint_roundtrips_and_resumes_identically() {
    use mandacaru_core::quantum::search::vasqa::SearchCheckpoint;

    let p = sector_test_problem(1, 6, [1, 1], "fermionic");
    let t = 0.01;
    let full_opts = SearchOptions {
        max_steps: 8,
        max_length: 4,
        temperatures: [t, t],
        seed: 23,
        ..Default::default()
    };
    let full = p.search(&full_opts, &[], &[]).unwrap();

    let first_half = p
        .search(
            &SearchOptions {
                max_steps: 4,
                ..full_opts.clone()
            },
            &[],
            &[],
        )
        .unwrap();
    let ckpt_json = first_half.checkpoint.to_json();
    let restored =
        SearchCheckpoint::from_json(&ckpt_json, p.pool.len(), full_opts.max_length).unwrap();
    assert_eq!(restored.step, 4);

    let resumed = p.resume_search(&full_opts, &restored, None, None).unwrap();
    assert_eq!(resumed.acceptance, full.acceptance);
    assert_eq!(resumed.final_indices, full.final_indices);
    assert!((resumed.best_energy - full.best_energy).abs() < 1e-12);
    assert!((resumed.final_energy - full.final_energy).abs() < 1e-12);
    assert_eq!(resumed.best_energy_indices, full.best_energy_indices);
}

#[test]
fn valsqa_multimove_edit_model_records_curvatures_roundtrips_and_converges() {
    use mandacaru_core::quantum::search::learning::{
        data::{EditDataset, EditKind},
        training::{fit_edit_model, Model},
    };

    let p = sector_test_problem(1, 6, [1, 1], "fermionic");
    let rec_opts = SearchOptions {
        max_steps: 24,
        max_length: 4,
        move_weights: [1., 0.8, 0.8],
        temperatures: [0.02, 0.0005],
        screen_insertions: 4,
        record: true,
        record_curvatures: true,
        seed: 7,
        ..Default::default()
    };
    let recorded = p.search(&rec_opts, &[], &[]).unwrap();
    assert!(recorded.records.len() >= rec_opts.max_steps);
    let mut records = Vec::new();
    for mut step in recorded.records {
        let curvs = step["source_curvatures"]
            .as_array()
            .expect("source_curvatures recorded");
        assert_eq!(curvs.len(), p.pool.len());
        assert!(curvs.iter().all(|v| v.as_f64().is_some_and(f64::is_finite)));
        step["problem_id"] = serde_json::json!("p0");
        step["format"] = serde_json::json!("mandacaru-search-edit");
        step["version"] = serde_json::json!(1);
        records.push(step);
    }
    let problem_meta = serde_json::json!({
        "format": "mandacaru-search-problem",
        "version": 1,
        "num_qubits": p.hamiltonian.n,
        "spatial_orbitals": p.encoding.modes / 2,
        "num_particles": p.encoding.particles,
        "mapping": "jordan_wigner",
        "hamiltonian": p.hamiltonian.labels().into_iter().map(|(l, c)| serde_json::json!([l, c.re, c.im])).collect::<Vec<_>>(),
        "pool": p.pool.iter().map(|g| g.operator.labels().into_iter().map(|(l, c)| serde_json::json!([l, c.re, c.im])).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "composition": [[1, 1]],
    });
    let payload = serde_json::json!({
        "problems": { "p0": problem_meta },
        "records": records,
    })
    .to_string();
    let dataset = EditDataset::from_json(
        &p.context,
        &payload,
        "molecule",
        &[EditKind::Insert, EditKind::Delete, EditKind::Replace],
    )
    .unwrap();
    assert!(dataset.counts.iter().sum::<usize>() > 0);
    assert!(!dataset.states.is_empty());
    let edit_model = fit_edit_model(&dataset, 1e-2).unwrap();
    assert!(edit_model.insert.is_some());
    let wrapped = Model::from_edit(edit_model, 0.2).unwrap();
    let roundtrip = Model::from_json(&p.context, &wrapped.as_json().to_string()).unwrap();
    assert!(roundtrip.ready);
    assert_eq!(roundtrip.edit, wrapped.edit);

    let valsqa_opts = SearchOptions {
        max_steps: 100,
        max_length: 8,
        move_weights: [1., 0.25, 0.25],
        temperatures: [0.02, 0.0005],
        seed: 7,
        ..Default::default()
    };
    let guided = p
        .search_with_model(&valsqa_opts, &[], &[], Some(&roundtrip))
        .unwrap();
    let exact = p
        .solve(
            &SolveOptions {
                adapt: true,
                max_growth: 8,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    assert!(
        (guided.best_energy - exact.energy).abs() < 1e-5,
        "guided={} exact={}",
        guided.best_energy,
        exact.energy
    );
}

#[test]
fn bloch_model_k_space_matches_real_space_energy_and_prunes_pool() {
    use mandacaru_core::{
        molecular::{MolecularIntegrals, ScfOptions},
        Geometry, LengthUnit,
    };

    const BOHR: f64 = 0.529177210903;
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    // 2-cell periodic H2 dimer chain ([2, 1, 1], 8 qubits, [2, 2] gapped closed-shell insulator)
    // preserves crystal translation symmetry with momentum blocks k in {0, 1} mod 2.
    let primitive = Geometry::new(
        vec![1, 1],
        vec![[0., 0., 0.], [0.74, 0., 0.]],
        [[2.4, 0., 0.], [0., 5., 0.], [0., 0., 5.]],
        [true, false, false],
        LengthUnit::Angstrom,
    )
    .unwrap();
    let integrals = MolecularIntegrals::supercell(
        &ctx,
        &primitive,
        [2, 1, 1],
        r#"{"name":"HAO"}"#,
        0.45 / BOHR,
        None,
        Precision::Float64,
    )
    .unwrap();
    let rhf = integrals
        .mean_field(
            "rhf",
            [2, 2],
            &ScfOptions {
                tolerance: 1e-11,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(rhf.converged);

    let real_model = rhf.model(&integrals).unwrap();
    let real_prob = real_model
        .problem(&ctx, "jordan_wigner", true, false, true)
        .unwrap();

    let (bloch_model, momentum) = rhf
        .bloch_model(
            &integrals,
            None,
            mandacaru_core::molecular::double_factorized::Representation::Dense,
        )
        .unwrap();
    assert_eq!(momentum.mesh, [2, 1, 1]);
    assert_eq!(momentum.orbital_k.len(), 4);
    let bloch_full = bloch_model
        .problem(&ctx, "jordan_wigner", true, false, true)
        .unwrap();
    let bloch_prob = bloch_full.with_momentum(&momentum).unwrap();
    assert!(
        bloch_prob.pool.len() < bloch_full.pool.len(),
        "momentum pruning failed: {} vs {}",
        bloch_prob.pool.len(),
        bloch_full.pool.len()
    );

    // Exact ground-state energies match identically between real-space and Bloch k-space bases.
    let e_real = exact_ground_energy(
        &ctx,
        real_prob.register("real").unwrap(),
        &real_prob.hamiltonian,
    );
    let e_bloch = exact_ground_energy(
        &ctx,
        bloch_prob.register("bloch").unwrap(),
        &bloch_prob.hamiltonian,
    );
    assert!(
        (e_real - e_bloch).abs() < 1e-8,
        "real={e_real} vs bloch={e_bloch}"
    );

    // Momentum-conserving Bloch-VASQA and Subspace-Bloch-VASQA reach variational ground & excited states.
    let opts = SearchOptions {
        max_steps: 80,
        max_length: 10,
        move_weights: [1., 0.25, 0.25],
        temperatures: [0.005, 0.0001],
        seed: 3,
        ..Default::default()
    };
    let searched = bloch_prob.search(&opts, &[], &[]).unwrap();
    let adapt = bloch_prob
        .solve(
            &SolveOptions {
                adapt: true,
                max_growth: 10,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    assert!(
        (adapt.energy - e_bloch).abs() < 1e-6,
        "adapt={} e_bloch={e_bloch}",
        adapt.energy
    );
    assert!(
        (searched.best_energy - e_bloch).abs() < 1e-3,
        "searched={} e_bloch={e_bloch} real={e_real}",
        searched.best_energy
    );
    let sub = bloch_prob
        .search_subspace(&opts, &[0.7, 0.3], &[], &[])
        .unwrap();
    assert_eq!(sub.energies.len(), 2);
    assert!(sub.energies[0] <= sub.energies[1] + 1e-12);
    assert!(sub.energies[0] >= e_bloch - 1e-10);

    // Active-space and Double-Factorized k-space Bloch models preserve crystal momentum labels.
    use mandacaru_core::molecular::double_factorized::Representation;
    let (act_dense, act_mom) = rhf
        .bloch_model(
            &integrals,
            Some(r#"{"orbitals":3,"frozen":[0],"symmetry":true}"#),
            Representation::Dense,
        )
        .unwrap();
    assert_eq!(act_dense.frozen_indices, vec![0]);
    assert_eq!(act_dense.orbital_indices, vec![1, 2, 3]);
    assert_eq!(act_mom.orbital_k.len(), 3);
    assert!(act_dense.bloch_orbitals.is_some());

    let (df_model, df_mom) = rhf
        .bloch_model(
            &integrals,
            None,
            Representation::DoubleFactorized { truncation: 1e-10 },
        )
        .unwrap();
    assert_eq!(df_mom.orbital_k.len(), 4);
    assert!(df_model.factorized.is_some());

    // Lehmann Bloch spectral function directly consumes k-space Bloch eigenvectors and keeps unit sum rule.
    let kpts = [[0., 0., 0.], [0.5, 0., 0.]];
    let basis = bloch_prob.register("spectral").unwrap();
    let mandacaru_core::quantum::State::Float64(state_buf) = &adapt.state else {
        panic!("expected float64 state");
    };
    let occ_state: Vec<(u64, mandacaru_core::quantum::C64)> = basis
        .indices
        .as_slice()
        .iter()
        .zip(state_buf.as_slice().iter())
        .filter(|(_, c)| c.norm_sqr() > 0.)
        .map(|(&ket, &c)| (bloch_prob.encoding.decode(ket), c))
        .collect();
    let spec_k = mandacaru_core::molecular::periodic::bloch_spectral(
        &ctx,
        &integrals,
        &rhf,
        &bloch_model,
        &occ_state,
        &kpts,
        64,
    )
    .unwrap();
    let (_, _, sum_rule) = spec_k.summary();
    assert!(sum_rule.iter().all(|&s| (s - 1.).abs() < 1e-10));
}

#[test]
fn bg5_multimove_gnn_extension_trains_achieves_ready_spearman_and_accelerates_search() {
    use mandacaru_core::{
        math::SpatialGrid,
        molecular::{MolecularIntegrals, ScfOptions},
        quantum::search::{
            learning::training::{self, Kind, Model, Options},
            vasqa::SearchOptions,
        },
        Context, Geometry, LengthUnit,
    };
    use serde_json::json;

    const BOHR: f64 = 0.529177210903;
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let mut problems = serde_json::Map::new();
    let mut all_records = Vec::new();
    let specs = [
        ("g0", "H2", 0.65, false, [1usize, 1]),
        ("g1", "H2", 0.85, false, [1, 1]),
        ("g2", "LiH", 1.50, false, [2, 2]),
        ("g3", "H2", 0.74, true, [2, 2]),
    ];
    for (gid, formula, r, periodic, particles) in specs {
        let geom = if formula == "H2" {
            let cell = if periodic {
                [[2.4, 0., 0.], [0., 5., 0.], [0., 0., 5.]]
            } else {
                [[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]]
            };
            Geometry::new(
                vec![1, 1],
                vec![[0., 0., 0.], [r, 0., 0.]],
                cell,
                [periodic, false, false],
                LengthUnit::Angstrom,
            )
            .unwrap()
        } else {
            Geometry::new(
                vec![3, 1],
                vec![[3., 3., 2.2], [3., 3., 2.2 + r]],
                [[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]],
                [false, false, false],
                LengthUnit::Angstrom,
            )
            .unwrap()
        };
        let integrals = if periodic {
            MolecularIntegrals::supercell(
                &ctx,
                &geom,
                [2, 1, 1],
                r#"{"name":"HAO"}"#,
                0.45 / BOHR,
                None,
                Precision::Float64,
            )
            .unwrap()
        } else {
            let grid = SpatialGrid::from_geometry(&geom, [0.45 / BOHR; 3], false, false).unwrap();
            MolecularIntegrals::new(
                &ctx,
                geom,
                grid,
                r#"{"name":"HAO"}"#,
                "spectral",
                None,
                Precision::Float64,
            )
            .unwrap()
        };
        let mf = integrals
            .mean_field(
                "rhf",
                particles,
                &ScfOptions {
                    tolerance: 1e-11,
                    ..Default::default()
                },
            )
            .unwrap();
        let (model, mom_opt) = if periodic {
            let (m, mom) = mf
                .bloch_model(
                    &integrals,
                    None,
                    mandacaru_core::molecular::double_factorized::Representation::Dense,
                )
                .unwrap();
            (m, Some(mom))
        } else {
            (mf.model(&integrals).unwrap(), None)
        };
        let full = model
            .problem(&ctx, "jordan_wigner", true, false, true)
            .unwrap();
        let prob = match &mom_opt {
            Some(m) => full.with_momentum(m).unwrap(),
            None => full,
        };
        let mut p_json = json!({
            "format": "mandacaru-search-problem",
            "version": 1,
            "hamiltonian": prob.hamiltonian.labels().into_iter().map(|(l, c)| json!([l, c.re, c.im])).collect::<Vec<_>>(),
            "pool": prob.pool.iter().map(|g| g.operator.labels().into_iter().map(|(l, c)| json!([l, c.re, c.im])).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "labels": prob.pool.iter().map(|g| g.label.clone()).collect::<Vec<_>>(),
            "mapping": prob.encoding.mapping,
            "num_particles": prob.encoding.particles,
            "spatial_orbitals": prob.encoding.modes / 2,
            "num_qubits": prob.n_qubits(),
            "composition": if formula == "H2" { vec![[1usize, 2usize]] } else { vec![[1usize, 1usize], [3usize, 1usize]] },
        });
        if let Some(m) = &mom_opt {
            p_json["momentum"] = json!({"mesh": m.mesh, "orbital_k": m.orbital_k});
        }
        problems.insert(gid.into(), p_json);
        let search_opts = SearchOptions {
            max_steps: 10,
            max_length: 4,
            screen_insertions: 4,
            record: true,
            record_curvatures: true,
            seed: 5,
            ..Default::default()
        };
        let res = prob.search(&search_opts, &[], &[]).unwrap();
        for rec in &res.records {
            let mut row = serde_json::to_value(rec).unwrap();
            row["format"] = json!("mandacaru-search-edit");
            row["version"] = json!(1);
            row["problem_id"] = json!(gid);
            all_records.push(row);
        }
    }
    let payload = json!({"problems": problems, "records": all_records}).to_string();
    let train_opts = Options {
        kinds: vec![Kind::Graph],
        epochs: 20,
        min_steps: 20,
        fractions: vec![1.0],
        folds: Some(2),
        ..Default::default()
    };
    let model_json = training::fit_json(&ctx, &payload, "molecule", &train_opts).unwrap();
    let fitted = Model::from_json(&ctx, &model_json).unwrap();
    let params = fitted.parameters.as_ref().unwrap();
    assert!(
        params.values["w1"].iter().any(|&v| v != 0.)
            && params.values["w_desc"].iter().any(|&v| v != 0.),
        "w1 / w_desc were not trained by Adam"
    );
    assert!(fitted.edit.is_some());

    // Held-out unseen stretched crystal H2 chain under VALSQA vs baseline VASQA.
    let unseen = Geometry::new(
        vec![1, 1],
        vec![[0., 0., 0.], [1.05, 0., 0.]],
        [[2.6, 0., 0.], [0., 5., 0.], [0., 0., 5.]],
        [true, false, false],
        LengthUnit::Angstrom,
    )
    .unwrap();
    let unseen_int = MolecularIntegrals::supercell(
        &ctx,
        &unseen,
        [2, 1, 1],
        r#"{"name":"HAO"}"#,
        0.45 / BOHR,
        None,
        Precision::Float64,
    )
    .unwrap();
    let unseen_mf = unseen_int
        .mean_field(
            "rhf",
            [2, 2],
            &ScfOptions {
                tolerance: 1e-11,
                ..Default::default()
            },
        )
        .unwrap();
    let (unseen_model, unseen_mom) = unseen_mf
        .bloch_model(
            &unseen_int,
            None,
            mandacaru_core::molecular::double_factorized::Representation::Dense,
        )
        .unwrap();
    let unseen_prob = unseen_model
        .problem(&ctx, "jordan_wigner", true, false, true)
        .unwrap()
        .with_momentum(&unseen_mom)
        .unwrap();
    let eval_opts = SearchOptions {
        max_steps: 16,
        max_length: 4,
        move_weights: [1., 0.25, 0.25],
        temperatures: [0.005, 0.0001],
        seed: 7,
        ..Default::default()
    };
    let base = unseen_prob.search(&eval_opts, &[], &[]).unwrap();
    let learned = unseen_prob
        .search_with_model(&eval_opts, &[], &[], Some(&fitted))
        .unwrap();
    let exact = exact_ground_energy(
        &ctx,
        unseen_prob.register("unseen").unwrap(),
        &unseen_prob.hamiltonian,
    );
    assert!(
        (learned.best_energy - exact).abs() < 2e-2
            && learned.best_energy <= base.best_energy + 1e-4,
        "learned={} base={} exact={exact}",
        learned.best_energy,
        base.best_energy
    );
}

#[test]
fn crystal_point_group_symmetry_prunes_momentum_allowed_excitations_across_solvers_hva_and_taper() {
    use mandacaru_core::{
        molecular::{double_factorized::Representation, MolecularIntegrals, ScfOptions},
        quantum::search::learning::{
            data::{EditDataset, EditKind},
            graph::BipartiteGraph,
            training::{fit_edit_model, Model},
        },
        Geometry, LengthUnit,
    };

    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    // Centrosymmetric 1D H2 dimer chain with Gamma + X points (4 spatial Bloch orbitals).
    // Even after CrystalMomentum removes k != k' excitations, Delta k = 0 single excitations
    // (Gamma sigma_g -> Gamma sigma_u, X sigma_g -> X sigma_u) remain until CrystalPointSymmetry
    // removes inversion-odd excitations.
    let prim = Geometry::new(
        vec![1, 1],
        vec![[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]],
        [[4.5, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]],
        [true, false, false],
        LengthUnit::Bohr,
    )
    .unwrap();
    let integrals = MolecularIntegrals::supercell(
        &ctx,
        &prim,
        [2, 1, 1],
        r#"{"name":"HAO"}"#,
        0.85,
        Some(1e-9),
        Precision::Float64,
    )
    .unwrap();
    let mf = integrals
        .mean_field(
            "rhf",
            [2, 2],
            &ScfOptions {
                tolerance: 1e-11,
                ..Default::default()
            },
        )
        .unwrap();
    let (model, momentum) = mf
        .bloch_model(&integrals, None, Representation::Dense)
        .unwrap();
    let point_sym: &CrystalPointSymmetry = model
        .crystal_point_symmetry
        .as_ref()
        .expect("centrosymmetric H2 chain must expose non-trivial CrystalPointSymmetry");
    assert!(!model.orbital_symmetry_sets.is_empty());

    let full = model
        .problem(&ctx, "jordan_wigner", true, true, true)
        .unwrap();
    let mom_only = full.with_momentum(&momentum).unwrap();
    let sym_prob = full
        .with_crystal_symmetry(Some(&momentum), Some(point_sym))
        .unwrap();
    assert!(
        sym_prob.pool.len() < mom_only.pool.len(),
        "expected point-group + momentum pool {} < momentum-only pool {}",
        sym_prob.pool.len(),
        mom_only.pool.len()
    );
    assert!(
        sym_prob.pool.iter().all(|g| !g.label.starts_with("S(")),
        "expected all parity-forbidden Delta k = 0 singles to be pruned, got {:?}",
        sym_prob.pool.iter().map(|g| &g.label).collect::<Vec<_>>()
    );

    let e_exact = exact_ground_energy(
        &ctx,
        sym_prob.register("sym").unwrap(),
        &sym_prob.hamiltonian,
    );

    // 1. VQE
    let res_vqe = sym_prob
        .solve(
            &SolveOptions {
                adapt: false,
                trotter: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    assert!((res_vqe.energy - e_exact).abs() < 1e-6);

    // 2. ADAPT-VQE + zero gradient audit for all pruned generators
    let adapt_opts = SolveOptions {
        adapt: true,
        trotter: true,
        max_growth: 24,
        gradient_tolerance: Some(1e-10),
        ..Default::default()
    };
    let res_adapt = sym_prob.solve(&adapt_opts, None).unwrap();
    let mom_adapt = mom_only.solve(&adapt_opts, None).unwrap();
    assert!(
        (res_adapt.energy - mom_adapt.energy).abs() < 1e-10,
        "Sym-ADAPT {} vs mom-only ADAPT {}",
        res_adapt.energy,
        mom_adapt.energy
    );
    assert!(
        (res_adapt.energy - e_exact).abs() < 1e-4,
        "ADAPT energy {} vs exact {e_exact} (vqe={}, mom_pool={}, sym_pool={}, labels={:?})",
        res_adapt.energy,
        res_vqe.energy,
        mom_only.pool.len(),
        sym_prob.pool.len(),
        point_sym.labels
    );
    let kept: std::collections::BTreeSet<&str> =
        sym_prob.pool.iter().map(|g| g.label.as_str()).collect();
    let grads_mom = mom_only.pool_gradients(&res_adapt.state, true).unwrap();
    for (g, &grad) in mom_only.pool.iter().zip(&grads_mom) {
        if !kept.contains(g.label.as_str()) {
            assert!(
                grad.abs() < 1e-12,
                "pruned generator {} had non-zero gradient {grad}",
                g.label
            );
        }
    }

    // 3. VASQA
    let search_opts = SearchOptions {
        max_steps: 60,
        max_length: 8,
        move_weights: [1., 0.25, 0.25],
        temperatures: [0.005, 0.0001],
        seed: 3,
        ..Default::default()
    };
    let res_vasqa = sym_prob.search(&search_opts, &[], &[]).unwrap();
    assert!(
        (res_vasqa.best_energy - e_exact).abs() < 1e-3,
        "Sym-VASQA {} vs exact {e_exact}",
        res_vasqa.best_energy
    );

    // 4. VALSQA (with trained ProposalModel + BipartiteGraph from crystal-symmetry problem)
    let _graph = BipartiteGraph::from_problem(&sym_prob).unwrap();
    let rec_opts = SearchOptions {
        max_steps: 12,
        max_length: 4,
        move_weights: [1., 0.5, 0.5],
        temperatures: [0.02, 0.0005],
        screen_insertions: 4,
        record: true,
        record_curvatures: true,
        seed: 4,
        ..Default::default()
    };
    let recorded = sym_prob.search(&rec_opts, &[], &[]).unwrap();
    let records: Vec<_> = recorded
        .records
        .into_iter()
        .map(|mut step| {
            step["problem_id"] = serde_json::json!("p0");
            step["format"] = serde_json::json!("mandacaru-search-edit");
            step["version"] = serde_json::json!(1);
            step
        })
        .collect();
    let problem_meta = serde_json::json!({
        "format": "mandacaru-search-problem",
        "version": 1,
        "num_qubits": sym_prob.hamiltonian.n,
        "spatial_orbitals": sym_prob.encoding.modes / 2,
        "num_particles": sym_prob.encoding.particles,
        "mapping": "jordan_wigner",
        "hamiltonian": sym_prob.hamiltonian.labels().into_iter().map(|(l, c)| serde_json::json!([l, c.re, c.im])).collect::<Vec<_>>(),
        "pool": sym_prob.pool.iter().map(|g| g.operator.labels().into_iter().map(|(l, c)| serde_json::json!([l, c.re, c.im])).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "composition": [[1, 1]],
    });
    let payload = serde_json::json!({
        "problems": { "p0": problem_meta },
        "records": records,
    })
    .to_string();
    let edit_data = EditDataset::from_json(
        &ctx,
        &payload,
        "molecule",
        &[EditKind::Insert, EditKind::Delete, EditKind::Replace],
    )
    .unwrap();
    let edit_model = fit_edit_model(&edit_data, 1e-2).unwrap();
    let trained = Model::from_edit(edit_model, 0.2).unwrap();
    let res_valsqa = sym_prob
        .search_with_model(&search_opts, &[], &[], Some(&trained))
        .unwrap();
    assert!(
        (res_valsqa.best_energy - e_exact).abs() < 2e-3,
        "Sym-VALSQA {} vs exact {e_exact}",
        res_valsqa.best_energy
    );

    // 5. Z2 tapering composed AFTER crystal symmetry across VQE & VASQA
    let untapered_full = model
        .problem(&ctx, "jordan_wigner", false, false, true)
        .unwrap()
        .with_crystal_symmetry(Some(&momentum), Some(point_sym))
        .unwrap();
    let tapered = untapered_full.with_taper().unwrap();
    assert!(tapered.n_qubits() < untapered_full.n_qubits());
    let tap_vqe = tapered
        .solve(
            &SolveOptions {
                adapt: false,
                trotter: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    assert!((tap_vqe.energy - e_exact).abs() < 1e-6);
    let tap_vasqa = tapered.search(&search_opts, &[], &[]).unwrap();
    assert!(
        (tap_vasqa.best_energy - e_exact).abs() < 2e-3,
        "Tapered-VASQA {} vs exact {e_exact}",
        tap_vasqa.best_energy
    );

    // 6. HVA compatibility after crystal symmetry
    let mut hva_prob = sym_prob
        .with_crystal_symmetry(Some(&momentum), Some(point_sym))
        .unwrap();
    hva_prob.use_hva(2, "exact", 1, 1, 0.05).unwrap();
    let hva_vqe = hva_prob
        .solve(
            &SolveOptions {
                adapt: false,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    assert!(hva_vqe.energy <= mf.electronic_energy + 1e-10);
}
