//! Micro-benchmark harness for VASQA kernels and end-to-end search scaling.
//!
//! Run with:
//! `cargo bench -p mandacaru-core --bench vasqa_bench`

use mandacaru_core::{
    quantum::{
        evolution::{self, evaluate, evaluate_from, forward_from_with_mode},
        optimizer,
        search::vasqa::SearchOptions,
        Encoding, Operator, QuantumProblem, SolveOptions, C64,
    },
    Context,
};
use std::time::Instant;

fn build_problem(
    threads: usize,
    modes: usize,
    particles: [usize; 2],
    pool: &str,
) -> QuantumProblem {
    let ctx = Context::new(Some(threads), 2 << 30).unwrap();
    let encoding = Encoding::new("jordan_wigner", modes, particles).unwrap();
    let mut terms = vec![("I".repeat(modes), C64::new(-2.0, 0.))];
    for j in 0..modes {
        let mut z = vec![b'I'; modes];
        z[j] = b'Z';
        terms.push((
            String::from_utf8(z).unwrap(),
            C64::new(0.2 - 0.03 * j as f64, 0.),
        ));
    }
    let mut h = Operator::new(&ctx, modes, &terms).unwrap();
    let half = modes / 2;
    for j in 0..modes - 1 {
        let k = j + 1;
        if j / half != k / half {
            continue;
        }
        let weight = C64::new(0.15, 0.);
        for (a, b) in [(j, k), (k, j)] {
            let hop = encoding.monomial(&ctx, &[(a, true), (b, false)]).unwrap();
            h = h.add(&ctx, &hop, weight).unwrap();
        }
    }
    let mut problem = QuantumProblem::new(&ctx, encoding, h, true, false, true).unwrap();
    problem.use_pool(pool, true).unwrap();
    problem
}

fn bench_eval_scaling() {
    println!("=== 1. Forward + Gradient Scaling (Sparse Tangent vs Dense Adjoint Baseline) ===");
    println!(
        "{:>6} {:>9} {:>5} {:>12} {:>12} {:>9} {:>10}",
        "Qubits", "Dim(D)", "Depth", "Sparse(ms)", "Dense(ms)", "Speedup", "Peak(MiB)"
    );
    let configs = [
        (4usize, [1usize, 1], 4usize),
        (8, [2, 2], 8),
        (12, [3, 3], 12),
        (16, [3, 3], 12),
        (20, [4, 4], 20),
    ];
    for (modes, particles, depth) in configs {
        let problem = build_problem(1, modes, particles, "fermionic");
        let basis = problem.register("bench").unwrap();
        let d = basis.len();
        let len = depth.min(problem.pool.len());
        let indices: Vec<usize> = (0..len)
            .map(|k| (k * (problem.pool.len() / len).max(1)) % problem.pool.len())
            .collect();
        let theta: Vec<f64> = (0..len)
            .map(|k| 0.08 * if k % 2 == 0 { 1. } else { -1. } / (1. + k as f64 * 0.2))
            .collect();
        let ops: Vec<&Operator> = indices.iter().map(|&i| &problem.pool[i].operator).collect();

        // Warm up once.
        let _ = problem.evaluate(&indices, &theta, true).unwrap();

        let iters = if d >= 30_000 {
            3
        } else if d >= 3_000 {
            10
        } else {
            50
        };
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = problem.evaluate(&indices, &theta, true).unwrap();
        }
        let sparse_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

        // Passing `initial: Some(&ref_state)` disables support tracking, exercising the dense
        // `rotate_pairs` + `adjoint_sweep` baseline for a clean before/after comparison.
        let mut ref_state = vec![C64::default(); d];
        ref_state[basis.reference] = C64::new(1., 0.);
        let t1 = Instant::now();
        for _ in 0..iters {
            let _ = evaluate_from::<C64>(
                &problem.context,
                basis,
                &problem.hamiltonian,
                None,
                None,
                &ops,
                &theta,
                true,
                Some(&ref_state),
            )
            .unwrap();
        }
        let dense_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let peak_mib = problem.context.memory_peak() as f64 / (1024. * 1024.);
        let speedup = dense_ms / sparse_ms.max(1e-9);
        println!(
            "{modes:>6} {d:>9} {len:>5} {sparse_ms:>12.3} {dense_ms:>12.3} {speedup:>8.2}x {peak_mib:>10.2}"
        );
    }
}

fn bench_importances_scaling() {
    println!("\n=== 2. Importances Scaling (Sparse 0-D vs Dense 1-Thr & 4-Thr) ===");
    println!(
        "{:>6} {:>9} {:>5} {:>11} {:>11} {:>11} {:>9} {:>10}",
        "Qubits",
        "Dim(D)",
        "Depth",
        "Sp1-Thr(ms)",
        "Sp4-Thr(ms)",
        "Dn4-Thr(ms)",
        "Sp/Dn4",
        "Peak(MiB)"
    );
    let configs = [
        (12usize, [3usize, 3], 12usize),
        (16, [3, 3], 12),
        (20, [4, 4], 16),
    ];
    for (modes, particles, depth) in configs {
        let p1 = build_problem(1, modes, particles, "fermionic");
        let p4 = build_problem(4, modes, particles, "fermionic");
        let d = p1.register("bench").unwrap().len();
        let len = depth.min(p1.pool.len());
        let indices: Vec<usize> = (0..len)
            .map(|k| (k * (p1.pool.len() / len).max(1)) % p1.pool.len())
            .collect();
        let theta: Vec<f64> = (0..len)
            .map(|k| 0.05 * if k % 2 == 0 { 1. } else { -1. })
            .collect();

        let _ = p1.importances_with_mode(&indices, &theta, true).unwrap();
        let _ = p4.importances_with_mode(&indices, &theta, true).unwrap();

        let iters = if d >= 30_000 { 2 } else { 5 };
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = p1.importances_with_mode(&indices, &theta, true).unwrap();
        }
        let sp1_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let t1 = Instant::now();
        for _ in 0..iters {
            let _ = p4.importances_with_mode(&indices, &theta, true).unwrap();
        }
        let sp4_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let peak_mib = p4.context.memory_peak() as f64 / (1024. * 1024.);

        let _ = p4.importances_with_mode(&indices, &theta, false).unwrap();
        let t2 = Instant::now();
        for _ in 0..iters {
            let _ = p4.importances_with_mode(&indices, &theta, false).unwrap();
        }
        let dn4_ms = t2.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!(
            "{modes:>6} {d:>9} {len:>5} {sp1_ms:>11.3} {sp4_ms:>11.3} {dn4_ms:>11.3} {:>8.2}x {peak_mib:>10.2}",
            dn4_ms / sp4_ms.max(1e-9)
        );
    }
}

fn bench_forward_energy_scaling() {
    println!("\n=== 1b. Energy-Only Forward Trial Scaling (Sparse Forward vs Dense Buffer) ===");
    println!(
        "{:>6} {:>9} {:>5} {:>12} {:>12} {:>9} {:>12} {:>12}",
        "Qubits",
        "Dim(D)",
        "Depth",
        "Sparse(ms)",
        "Dense(ms)",
        "Speedup",
        "SpkΔPeak(KB)",
        "DnsΔPeak(KB)"
    );
    let configs = [
        (12usize, [3usize, 3], 12usize),
        (16, [3, 3], 12),
        (20, [4, 4], 20),
    ];
    for (modes, particles, depth) in configs {
        let len_probe = depth;
        let iters = if modes >= 20 { 5 } else { 20 };

        let p_sp = build_problem(1, modes, particles, "fermionic");
        let b_sp = p_sp.register("sp").unwrap();
        let d = b_sp.len();
        let len = len_probe.min(p_sp.pool.len());
        let indices: Vec<usize> = (0..len)
            .map(|k| (k * (p_sp.pool.len() / len).max(1)) % p_sp.pool.len())
            .collect();
        let theta: Vec<f64> = (0..len)
            .map(|k| 0.08 * if k % 2 == 0 { 1. } else { -1. } / (1. + k as f64 * 0.2))
            .collect();
        let ops_sp: Vec<&Operator> = indices.iter().map(|&i| &p_sp.pool[i].operator).collect();
        let before_sp = p_sp.context.memory_peak();
        let t0 = Instant::now();
        for _ in 0..iters {
            let f = forward_from_with_mode::<C64>(
                &p_sp.context,
                b_sp,
                &p_sp.hamiltonian,
                None,
                p_sp.sparse.as_ref(),
                &ops_sp,
                &theta,
                true,
                None,
                true,
            )
            .unwrap();
            std::hint::black_box(f.energy());
        }
        let sp_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let sp_kb = p_sp.context.memory_peak().saturating_sub(before_sp) as f64 / 1024.;

        let p_dn = build_problem(1, modes, particles, "fermionic");
        let b_dn = p_dn.register("dn").unwrap();
        let ops_dn: Vec<&Operator> = indices.iter().map(|&i| &p_dn.pool[i].operator).collect();
        let before_dn = p_dn.context.memory_peak();
        let t1 = Instant::now();
        for _ in 0..iters {
            let f = forward_from_with_mode::<C64>(
                &p_dn.context,
                b_dn,
                &p_dn.hamiltonian,
                None,
                p_dn.sparse.as_ref(),
                &ops_dn,
                &theta,
                true,
                None,
                false,
            )
            .unwrap();
            std::hint::black_box(f.energy());
        }
        let dn_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let dn_kb = p_dn.context.memory_peak().saturating_sub(before_dn) as f64 / 1024.;
        println!(
            "{modes:>6} {d:>9} {len:>5} {sp_ms:>12.3} {dn_ms:>12.3} {:>8.2}x {sp_kb:>12.1} {dn_kb:>12.1}",
            dn_ms / sp_ms.max(1e-9)
        );
    }
}

fn bench_vqe_and_adapt() {
    println!("\n=== 3. VQE & ADAPT-VQE (Baseline vs Sparse + Evaluation Reuse) ===");
    println!(
        "{:>6} {:>9} {:>18} {:>10} {:>14} {:>12} {:>6}",
        "Qubits", "Dim(D)", "Solver/Mode", "Time(ms)", "Energy(Ha)", "MaxGrad", "Depth"
    );
    for (modes, particles, max_growth) in [
        (8usize, [2usize, 2], 8usize),
        (12, [3, 3], 6),
        (16, [3, 3], 6),
    ] {
        let problem = build_problem(1, modes, particles, "fermionic");
        let d = problem.register("bench").unwrap().len();
        let mut configs: Vec<(&str, SolveOptions)> = Vec::new();
        if modes <= 12 {
            configs.push((
                "VQE baseline",
                SolveOptions {
                    adapt: false,
                    trotter: true,
                    sparse_kernels: false,
                    reuse_evaluations: false,
                    ..Default::default()
                },
            ));
            configs.push((
                "VQE default",
                SolveOptions {
                    adapt: false,
                    trotter: true,
                    ..Default::default()
                },
            ));
        }
        configs.extend([
            (
                "ADAPT baseline",
                SolveOptions {
                    adapt: true,
                    max_growth,
                    sparse_kernels: false,
                    reuse_evaluations: false,
                    ..Default::default()
                },
            ),
            (
                "ADAPT default",
                SolveOptions {
                    adapt: true,
                    max_growth,
                    ..Default::default()
                },
            ),
            (
                "ADAPT no-hess",
                SolveOptions {
                    adapt: true,
                    max_growth,
                    carry_hessian: false,
                    ..Default::default()
                },
            ),
            (
                "ADAPT-prune base",
                SolveOptions {
                    adapt: true,
                    prune: true,
                    max_growth,
                    sparse_kernels: false,
                    reuse_evaluations: false,
                    ..Default::default()
                },
            ),
            (
                "ADAPT-prune def",
                SolveOptions {
                    adapt: true,
                    prune: true,
                    max_growth,
                    ..Default::default()
                },
            ),
        ]);
        for (name, opts) in configs {
            let t0 = Instant::now();
            let res = problem.solve(&opts, None).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!(
                "{modes:>6} {d:>9} {name:>18} {ms:>10.2} {:>14.8} {:>12.3e} {:>6}",
                res.energy,
                res.final_gradient,
                res.indices.len()
            );
        }
    }
}

fn bench_search_options() {
    println!("\n=== 4. End-to-End VASQA Search Ablation ===");
    println!(
        "{:>6} {:>9} {:>20} {:>9} {:>8} {:>8} {:>7} {:>7} {:>14}",
        "Qubits",
        "Dim(D)",
        "Mode",
        "Time(ms)",
        "OptEval",
        "GrdEval",
        "Screen",
        "Impt",
        "BestEnergy"
    );
    for (modes, particles, steps, max_len) in
        [(8, [2, 2], 40, 6), (12, [3, 3], 30, 8), (16, [3, 3], 15, 5)]
    {
        let problem = build_problem(1, modes, particles, "fermionic");
        let d = problem.register("bench").unwrap().len();
        let legacy = SearchOptions {
            max_steps: steps,
            max_length: max_len,
            seed: 7,
            warm_start: true,
            memoize_warm: false,
            carry_hessian: false,
            sparse_kernels: false,
            early_reject: false,
            ..Default::default()
        };
        let configs = [
            ("legacy_baseline", legacy.clone()),
            (
                "default(memo+spk)",
                SearchOptions {
                    max_steps: steps,
                    max_length: max_len,
                    seed: 7,
                    ..Default::default()
                },
            ),
            (
                "+early_reject",
                SearchOptions {
                    max_steps: steps,
                    max_length: max_len,
                    seed: 7,
                    early_reject: true,
                    ..Default::default()
                },
            ),
            (
                "+carry_hessian",
                SearchOptions {
                    max_steps: steps,
                    max_length: max_len,
                    seed: 7,
                    carry_hessian: true,
                    ..Default::default()
                },
            ),
            (
                "all_enabled",
                SearchOptions {
                    max_steps: steps,
                    max_length: max_len,
                    seed: 7,
                    carry_hessian: true,
                    early_reject: true,
                    strip_zero_operators: true,
                    ..Default::default()
                },
            ),
        ];
        for (name, opts) in configs {
            let t0 = Instant::now();
            let res = problem.search(&opts, &[], &[]).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!(
                "{modes:>6} {d:>9} {name:>20} {ms:>9.2} {:>8} {:>8} {:>7} {:>7} {:>14.8}",
                res.solution.history.len(),
                res.num_gradient_evaluations,
                res.num_screenings,
                res.num_importance_evaluations,
                res.best_energy
            );
        }
    }

    println!("\n=== 5. Multi-Seed VASQA carry_hessian Check (8q sector, 8 seeds) ===");
    println!(
        "{:>5} {:>10} {:>10} {:>14} {:>14} {:>10}",
        "Seed", "NoH_Evals", "Hs_Evals", "NoH_BestE", "Hs_BestE", "dE(Hs-NoH)"
    );
    let p8 = build_problem(1, 8, [2, 2], "fermionic");
    for seed in 1u64..=8 {
        let base = SearchOptions {
            max_steps: 40,
            max_length: 6,
            seed,
            carry_hessian: false,
            ..Default::default()
        };
        let carried = SearchOptions {
            carry_hessian: true,
            ..base.clone()
        };
        let r0 = p8.search(&base, &[], &[]).unwrap();
        let r1 = p8.search(&carried, &[], &[]).unwrap();
        println!(
            "{seed:>5} {:>10} {:>10} {:>14.8} {:>14.8} {:>+10.2e}",
            r0.solution.history.len(),
            r1.solution.history.len(),
            r0.best_energy,
            r1.best_energy,
            r1.best_energy - r0.best_energy
        );
    }
    // Reference `evolution::evaluate` symbol so benchmark exercises public API cleanly.
    let p = build_problem(1, 4, [1, 1], "fermionic");
    let b = p.register("ref").unwrap();
    let _ = evaluate::<C64>(&p.context, b, &p.hamiltonian, None, None, &[], &[], true).unwrap();
    let _ = evolution::Work::default();
}

fn bench_subspace_scaling() {
    println!("\n=== 6. Subspace-ADAPT / Subspace-VQE / Subspace-VASQA (14q sector [3,3], D=1225, K=2) ===");
    println!(
        "{:>16} {:>16} {:>9} {:>10} {:>14} {:>14}",
        "Algorithm", "Mode", "Time(ms)", "Len", "E0(Ha)", "E1(Ha)"
    );
    let p14 = build_problem(1, 14, [3, 3], "fermionic");
    let weights = [0.65, 0.35];

    for (name, sparse, reuse) in [
        ("baseline(dense)", false, false),
        ("default(sp+re)", true, true),
    ] {
        let adapt_opts = SolveOptions {
            adapt: true,
            trotter: true,
            max_growth: 4,
            sparse_kernels: sparse,
            reuse_evaluations: reuse,
            carry_hessian: reuse,
            optimizer: optimizer::Options {
                method: optimizer::Method::Bfgs,
                max_iterations: 20,
                max_evaluations: 80,
                ..Default::default()
            },
            ..Default::default()
        };
        let t0 = Instant::now();
        let res_adapt = p14.solve_subspace(&adapt_opts, &weights, None).unwrap();
        let ms_adapt = t0.elapsed().as_secs_f64() * 1e3;
        println!(
            "{:>16} {:>16} {ms_adapt:>9.2} {:>10} {:>14.8} {:>14.8}",
            "Subspace-ADAPT",
            name,
            res_adapt.indices.len(),
            res_adapt.energies[0],
            res_adapt.energies[1]
        );

        let search_opts = SearchOptions {
            max_steps: 10,
            max_length: 4,
            seed: 7,
            sparse_kernels: sparse,
            memoize_warm: reuse,
            early_reject: reuse,
            ..Default::default()
        };
        let t0 = Instant::now();
        let res_search = p14
            .search_subspace(&search_opts, &weights, &[], &[])
            .unwrap();
        let ms_search = t0.elapsed().as_secs_f64() * 1e3;
        println!(
            "{:>16} {:>16} {ms_search:>9.2} {:>10} {:>14.8} {:>14.8}",
            "Subspace-VASQA",
            name,
            res_search.solution.indices.len(),
            res_search.energies[0],
            res_search.energies[1]
        );
    }
}

fn main() {
    bench_eval_scaling();
    bench_forward_energy_scaling();
    bench_importances_scaling();
    bench_vqe_and_adapt();
    bench_search_options();
    bench_subspace_scaling();
}
