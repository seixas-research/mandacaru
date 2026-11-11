//! The threaded state-vector actions (`quantum::parallel`): a Pauli sum and its sparse store,
//! applied on a particle sector and on a whole register large enough to split across threads,
//! give the same bits on one thread and on four, and agree with a per-term scatter, for a dense
//! input (the threaded gather) and a sparse one (the serial scatter of its nonzero columns). A
//! product ansatz of excitation rotations (the forward sweep, and the fused adjoint step with its
//! overlap) gives the same energy, gradient and state on every thread count, in both storage
//! precisions.

use mandacaru_core::{
    math::storage::Precision,
    quantum::{sparse::SparseMatrix, Basis, Encoding, Operator, C64},
    Context,
};

/// A deterministic Pauli sum on `n` qubits: `groups` flip masks with several `Z` patterns each,
/// so the strings of one mask must be summed together, and no identity string.
fn operator(ctx: &Context, n: usize, groups: usize) -> (Operator, Vec<(u64, u64, C64)>) {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mask = (1u64 << n) - 1;
    let mut raw = Vec::new();
    for _ in 0..groups {
        let x = next() & mask;
        for _ in 0..1 + next() % 4 {
            let z = next() & mask;
            if x == 0 && z == 0 {
                continue;
            }
            let c = C64::new((next() % 2001) as f64 / 1000. - 1., (next() % 2001) as f64 / 1000. - 1.);
            raw.push((x, z, c));
        }
    }
    let labels: Vec<(String, C64)> = raw
        .iter()
        .map(|&(x, z, c)| {
            let label = (0..n)
                .map(|j| {
                    let b = 1u64 << (n - 1 - j);
                    match (x & b != 0, z & b != 0) {
                        (false, false) => 'I',
                        (true, false) => 'X',
                        (false, true) => 'Z',
                        (true, true) => 'Y',
                    }
                })
                .collect();
            (label, c)
        })
        .collect();
    (Operator::new(ctx, n, &labels).unwrap(), raw)
}

/// `sum_t c_t P_t input` by scattering every column through every string: the reference.
fn scatter(basis: &Basis, raw: &[(u64, u64, C64)], input: &[C64]) -> Vec<C64> {
    let mut out = vec![C64::default(); basis.len()];
    for (column, &ket) in basis.indices.as_slice().iter().enumerate() {
        for &(x, z, c) in raw {
            if let Some(row) = basis.index(ket ^ x) {
                let i = C64::new(0., 1.).powi((x & z).count_ones() as i32);
                let sign = if (ket & z).count_ones() % 2 == 0 { 1. } else { -1. };
                out[row] += c * i * sign * input[column];
            }
        }
    }
    out
}

/// A dense input (six rows in seven nonzero) or a sparse one (one row in forty).
fn state(d: usize, dense: bool) -> Vec<C64> {
    (0..d)
        .map(|k| {
            let zero = if dense { k % 7 == 3 } else { k % 40 != 11 };
            if zero {
                C64::default()
            } else {
                C64::new(((k * 37) % 101) as f64 / 101. - 0.5, ((k * 53) % 89) as f64 / 89. - 0.5)
            }
        })
        .collect()
}

fn bits(values: &[C64]) -> Vec<(u64, u64)> {
    values.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
}

fn check(basis_of: impl Fn(&Context) -> Basis, n: usize) {
    for dense in [true, false] {
        check_input(&basis_of, n, dense);
    }
}

fn check_input(basis_of: &impl Fn(&Context) -> Basis, n: usize, dense: bool) {
    let mut results = Vec::new();
    let mut sparse_results = Vec::new();
    for threads in [1, 4] {
        let ctx = Context::new(Some(threads), 512 << 20).unwrap();
        let basis = basis_of(&ctx);
        let (op, raw) = operator(&ctx, n, 120);
        let input = state(basis.len(), dense);
        assert!(basis.len() * op.terms.len() >= 1 << 20, "the action must cross the parallel threshold");
        let out = op.apply(&ctx, &basis, &input).unwrap();
        let reference = scatter(&basis, &raw, &input);
        let scale = reference.iter().map(|v| v.norm()).fold(0., f64::max);
        let error = out.as_slice().iter().zip(&reference).map(|(a, b)| (a - b).norm()).fold(0., f64::max);
        assert!(error <= 1e-12 * scale, "{threads} threads, dense {dense}: action vs scatter {error:e}");
        let matrix = SparseMatrix::new(&ctx, &basis, &op).unwrap();
        let mut sparse = vec![C64::default(); basis.len()];
        matrix.apply_into(&input, &mut sparse);
        let error = sparse.iter().zip(&reference).map(|(a, b)| (a - b).norm()).fold(0., f64::max);
        assert!(error <= 1e-12 * scale, "{threads} threads, dense {dense}: sparse store vs scatter {error:e}");
        results.push(bits(out.as_slice()));
        sparse_results.push(bits(&sparse));
    }
    assert!(results[0] == results[1], "the Pauli action depends on the thread count");
    assert!(sparse_results[0] == sparse_results[1], "the sparse action depends on the thread count");
}

#[test]
fn a_sector_action_is_the_same_on_one_and_four_threads() {
    let encoding = Encoding::new("jordan_wigner", 18, [4, 4]).unwrap();
    check(|ctx| Basis::new(ctx, &encoding, true).unwrap(), 18);
}

#[test]
fn a_register_action_is_the_same_on_one_and_four_threads() {
    check(|ctx| Basis::register(ctx, 14).unwrap(), 14);
}

#[test]
fn many_actions_spread_over_the_workers_are_each_the_single_action() {
    // Nine states of the 15876-row sector, sparse (one row in forty, the serial scatter) and
    // dense (the gather), each under the many-action route and under the single action: the same
    // bits, on every thread count. This is the Z-vector Hessian's 76 Hamiltonian actions.
    let mut runs = Vec::new();
    for threads in [1, 2, 4] {
        let ctx = Context::new(Some(threads), 512 << 20).unwrap();
        let encoding = Encoding::new("jordan_wigner", 18, [4, 4]).unwrap();
        let basis = Basis::new(&ctx, &encoding, true).unwrap();
        let (op, _) = operator(&ctx, 18, 120);
        let inputs: Vec<Vec<C64>> = (0..9)
            .map(|k| {
                let mut input = state(basis.len(), k % 3 == 0);
                input.rotate_left(7 * k + 1);
                input
            })
            .collect();
        let views: Vec<&[C64]> = inputs.iter().map(Vec::as_slice).collect();
        let each = op.apply_each(&basis, &views, false, |_, row| bits(row)).unwrap();
        for (k, input) in inputs.iter().enumerate() {
            let single = op.apply(&ctx, &basis, input).unwrap();
            assert!(each[k] == bits(single.as_slice()), "{threads} threads, input {k}: the many-action route differs from the action");
        }
        assert!(basis.len() * op.terms.len() >= 1 << 18);
        runs.push(each);
    }
    assert!(runs[0] == runs[1] && runs[0] == runs[2], "the many-action route depends on the thread count");
}

/// A Hubbard chain of `sites` sites (spin-blocked modes: `spin * sites + site`).
fn hubbard(ctx: &Context, encoding: &Encoding, sites: usize) -> Operator {
    let modes = 2 * sites;
    let mut h = Operator::new(ctx, modes, &[]).unwrap();
    for spin in 0..2 {
        for site in 0..sites - 1 {
            let (i, j) = (spin * sites + site, spin * sites + site + 1);
            for (p, q) in [(i, j), (j, i)] {
                h = h.add(ctx, &encoding.monomial(ctx, &[(p, true), (q, false)]).unwrap(), C64::new(-1., 0.)).unwrap();
            }
        }
    }
    for site in 0..sites {
        let (up, down) = (site, sites + site);
        h = h
            .add(ctx, &encoding.monomial(ctx, &[(up, true), (up, false), (down, true), (down, false)]).unwrap(), C64::new(4., 0.))
            .unwrap();
    }
    h
}

/// The energy, gradient and state of a product ansatz of `operators` coupled-exchange operators
/// (the combined four-string ones and their eight-string members, alternating) of a Hubbard
/// chain, on `threads` threads, as bits.
fn product_run(
    threads: usize,
    modes: usize,
    electrons: [usize; 2],
    operators: usize,
    derivatives: usize,
    precision: Precision,
) -> (u64, Vec<u64>, Vec<(u64, u64)>) {
    let ctx = Context::new(Some(threads), 1 << 30).unwrap();
    let encoding = Encoding::new("jordan_wigner", modes, electrons).unwrap();
    let h = hubbard(&ctx, &encoding, modes / 2);
    let mut problem = mandacaru_core::quantum::CachedProblem::new(&ctx, encoding, h, true, false, true)
        .unwrap()
        .with_precision(precision);
    let pool = mandacaru_core::quantum::pools::build(&ctx, &problem.encoding, "ceo", true).unwrap();
    let mut combined = Vec::new();
    let mut members = Vec::new();
    for (g, visible) in pool.operators.into_iter().zip(pool.screenable) {
        if visible && g.label.contains(")+D(") && g.operator.terms.len() == 4 {
            combined.push(g);
        } else if !visible && g.operator.terms.len() == 8 {
            members.push(g);
        }
    }
    let half = operators / 2;
    let (combined_step, member_step) = ((combined.len() / half).max(1), (members.len() / half).max(1));
    let chosen: Vec<_> = combined
        .into_iter()
        .step_by(combined_step)
        .take(half)
        .zip(members.into_iter().step_by(member_step).take(half))
        .flat_map(|(a, b)| [a, b])
        .collect();
    assert_eq!(chosen.len(), operators);
    problem.pool = chosen;
    let indices: Vec<usize> = (0..operators).collect();
    let theta: Vec<f64> = (0..operators).map(|k| 0.3 * ((k as f64) * 1.7 + 0.4).sin()).collect();
    let evaluation = problem.evaluate(&indices, &theta, true).unwrap();
    // The analytic gradient against a central difference of the same energy.
    for k in [0, operators / 2, operators - 1].into_iter().take(derivatives) {
        let step = 1e-5;
        let mut plus = theta.clone();
        plus[k] += step;
        let mut minus = theta.clone();
        minus[k] -= step;
        let difference = (problem.evaluate(&indices, &plus, true).unwrap().energy
            - problem.evaluate(&indices, &minus, true).unwrap().energy)
            / (2. * step);
        assert!((difference - evaluation.gradient[k]).abs() < 1e-7, "{k}: {difference} vs {}", evaluation.gradient[k]);
    }
    let state = match precision {
        Precision::Float64 => bits(evaluation.state.complex128().unwrap()),
        Precision::Float32 => evaluation
            .state
            .complex64()
            .unwrap()
            .iter()
            .map(|v| (v.re.to_bits() as u64, v.im.to_bits() as u64))
            .collect(),
    };
    (evaluation.energy.to_bits(), evaluation.gradient.iter().map(|g| g.to_bits()).collect(), state)
}

fn assert_product_runs_agree(modes: usize, electrons: [usize; 2], operators: usize, rows_at_least: usize, thread_counts: &[usize], derivatives: usize, precision: Precision) {
    let encoding_rows = {
        let ctx = Context::new(Some(1), 1 << 30).unwrap();
        Basis::new(&ctx, &Encoding::new("jordan_wigner", modes, electrons).unwrap(), true).unwrap().len()
    };
    assert!(encoding_rows >= rows_at_least, "{encoding_rows} rows");
    let runs: Vec<_> = thread_counts
        .iter()
        .map(|&threads| (threads, product_run(threads, modes, electrons, operators, if threads == 1 { derivatives } else { 0 }, precision)))
        .collect();
    for (threads, run) in &runs[1..] {
        assert!(runs[0].1 .0 == run.0, "the energy depends on the thread count ({threads})");
        assert!(runs[0].1 .1 == run.1, "the gradient depends on the thread count ({threads})");
        assert!(runs[0].1 .2 == run.2, "the state depends on the thread count ({threads})");
    }
}

#[test]
fn a_product_ansatz_energy_and_gradient_are_the_same_on_one_and_four_threads() {
    // 22 modes, 5 + 6 electrons: 213444 rows, so the excitation rotations and the adjoint
    // overlaps of the product ansatz cross the parallel threshold (row gather and fixed blocks
    // on the threaded runs, the in-place pair update on one).
    assert_product_runs_agree(22, [5, 6], 12, 200_000, &[1, 4], 3, Precision::Float64);
}

#[test]
fn a_growth_sized_product_ansatz_is_the_same_on_every_thread_count() {
    // 20 modes, 4 + 4 electrons: the 44100-row sector of the H2O active space. Its eight-string
    // members cross the threshold (the fused adjoint step and the gathered forward rotation run),
    // its four-string combinations do not (they stay in place), and a long ansatz of both
    // alternates the two within one sweep, on a state that goes from sparse to dense.
    assert_product_runs_agree(20, [4, 4], 40, 40_000, &[1, 2, 4], 3, Precision::Float64);
}

#[test]
fn a_float32_growth_sized_product_ansatz_is_the_same_on_every_thread_count() {
    // The same 44100-row ansatz stored as complex64: the fused kernels are generic over the
    // stored amplitude, and the narrowing after each rotation must not depend on the split.
    assert_product_runs_agree(20, [4, 4], 40, 40_000, &[1, 2, 4], 0, Precision::Float32);
}
