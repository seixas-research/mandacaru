//! The pool screen restricted to the support of the state (`Operator::apply_support_into`,
//! `amplitude::inner_wide_rows`, `CachedProblem::pool_gradients`) gives the dense screen's bits.
//!
//! Every gradient is `2 Re <H psi | G psi>`. A state of a few nonzero amplitudes sends a generator's
//! scatter to a few rows, so the action row is cleared and the dot taken over those rows alone. The
//! restriction is exact, not an approximation: the skipped rows hold `+0`, their products with a
//! finite `H psi` are `+-0`, and a sum that began at `+0` is never `-0`. These tests compare bit
//! patterns against the dense action and dot (`Operator::apply_into`, `inner_wide`), for operators
//! with one flip mask and with several, in a particle sector and on a whole register, in both
//! storage precisions, with `-0.0` amplitudes, around the switch where the dense action turns from
//! a scatter into a gather (which sums in another order, so the restricted route must refuse
//! there), and through the problem's own screen on one thread and on eight.

use mandacaru_core::{
    math::storage::Precision,
    quantum::{
        amplitude::{inner_wide, inner_wide_rows},
        Amplitude, Basis, CachedProblem, Encoding, Operator, State, C64,
    },
    Context,
};
use num_complex::Complex32;

fn xorshift(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn real(seed: &mut u64) -> f64 {
    (xorshift(seed) % 2001) as f64 / 1000. - 1.
}

/// A Pauli sum on `n` qubits: for each flip mask in `masks`, `per_mask` strings with distinct `Z`
/// patterns (the strings of one mask are summed before they multiply an amplitude).
fn operator(ctx: &Context, n: usize, masks: &[u64], per_mask: usize, seed: u64) -> Operator {
    let mut seed = seed;
    let mut labels = Vec::new();
    for &x in masks {
        let mut zs = std::collections::BTreeSet::new();
        while zs.len() < per_mask {
            zs.insert(xorshift(&mut seed) & ((1 << n) - 1));
        }
        for z in zs {
            let label: String = (0..n)
                .map(|j| {
                    let bit = 1u64 << (n - 1 - j);
                    match (x & bit != 0, z & bit != 0) {
                        (false, false) => 'I',
                        (true, false) => 'X',
                        (false, true) => 'Z',
                        (true, true) => 'Y',
                    }
                })
                .collect();
            labels.push((label, C64::new(real(&mut seed), real(&mut seed))));
        }
    }
    Operator::new(ctx, n, &labels).unwrap()
}

/// A finite state of `d` amplitudes with `nonzero` of them nonzero, spread over the basis, and
/// `negative_zeros` more holding a `-0.0` component (a zero the scatter must skip like any other).
fn state<A: Amplitude>(d: usize, nonzero: usize, negative_zeros: usize, seed: u64) -> Vec<A> {
    let mut seed = seed;
    let mut out = vec![A::default(); d];
    let mut placed = 0;
    while placed < nonzero {
        let row = xorshift(&mut seed) as usize % d;
        if out[row] == A::default() {
            out[row] = A::narrow(C64::new(real(&mut seed) + 1.5, real(&mut seed)));
            placed += 1;
        }
    }
    let mut zeros = 0;
    while zeros < negative_zeros.min(d - nonzero) {
        let row = xorshift(&mut seed) as usize % d;
        if out[row] == A::default() {
            let variants = [C64::new(-0., 0.), C64::new(0., -0.), C64::new(-0., -0.)];
            out[row] = A::narrow(variants[zeros % 3]);
            zeros += 1;
        }
    }
    out
}

/// A finite `H psi` of both signs, with signed zeros among its entries.
fn hamiltonian_state<A: Amplitude>(d: usize, seed: u64) -> Vec<A> {
    let mut seed = seed;
    (0..d)
        .map(|k| match k % 11 {
            0 => A::narrow(C64::new(-0., 0.)),
            1 => A::narrow(C64::new(0., -0.)),
            _ => A::narrow(C64::new(real(&mut seed) * 3., real(&mut seed) - 0.7)),
        })
        .collect()
}

fn bits(value: C64) -> (u64, u64) {
    (value.re.to_bits(), value.im.to_bits())
}

fn support<A: Amplitude>(state: &[A]) -> Vec<usize> {
    (0..state.len()).filter(|&k| state[k] != A::default()).collect()
}

/// The restricted action and dot against the dense ones. `None` when the operator refused the
/// restriction; otherwise the number of rows it wrote.
fn compare<A: Amplitude>(basis: &Basis, op: &Operator, psi: &[A], hpsi: &[A]) -> Option<usize> {
    let d = basis.len();
    let support = support(psi);
    let mut dense = vec![C64::default(); d];
    op.apply_into(basis, psi, &mut dense);
    let mut action = vec![C64::default(); d];
    let mut touched = vec![7usize; 3];
    if !op.apply_support_into(basis, psi, &support, &mut action, &mut touched) {
        assert!(action.iter().all(|a| bits(*a) == bits(C64::default())), "a refusal touched the row");
        return None;
    }
    assert!(touched.windows(2).all(|w| w[0] < w[1]), "touched rows must ascend without repeats");
    for (row, (a, b)) in action.iter().zip(&dense).enumerate() {
        assert_eq!(bits(*a), bits(*b), "row {row}");
    }
    for (row, value) in dense.iter().enumerate() {
        if *value != C64::default() {
            assert!(touched.binary_search(&row).is_ok(), "row {row} is nonzero but not touched");
        }
    }
    let by_rows = inner_wide_rows(hpsi, &action, &touched);
    let by_all = inner_wide(hpsi, &dense);
    assert_eq!(bits(by_rows), bits(by_all));
    Some(touched.len())
}

/// One operator (of `masks` flip masks) on one basis, over a range of supports, in both
/// precisions: every support whose written rows stay under an eighth of the basis must take the
/// restricted route (the work here is far below where the dense action gathers), and every route
/// must reproduce the dense bits.
fn sweep(basis: &Basis, op: &Operator, masks: usize, supports: &[usize]) {
    let d = basis.len();
    for &nonzero in supports {
        for zeros in [0, 5] {
            let seed = 17 + nonzero as u64;
            let psi64 = state::<C64>(d, nonzero, zeros, seed);
            let h64 = hamiltonian_state::<C64>(d, seed + 1);
            let restricted = compare(basis, op, &psi64, &h64);
            let psi32 = state::<Complex32>(d, nonzero, zeros, seed);
            let h32 = hamiltonian_state::<Complex32>(d, seed + 1);
            assert_eq!(compare(basis, op, &psi32, &h32).is_some(), restricted.is_some());
            assert_eq!(restricted.is_some(), nonzero * masks * 8 < d, "support {nonzero} of {d}, {masks} masks");
        }
    }
}

#[test]
fn a_sparse_support_on_a_sector_matches_the_dense_screen() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let basis = Basis::new(&ctx, &Encoding::new("jordan_wigner", 14, [3, 3]).unwrap(), true).unwrap();
    assert_eq!(basis.len(), 35 * 35);
    // One mask, then four. The work is far below where the dense action may gather, so the
    // restriction is allowed up to the rows it may write (`support x masks x 8 < d`).
    for masks in [&[0b101_0000_0000_0110u64][..], &[0b11, 0b1100_0000, 0b11_0000_0000_0000, 0b1_0101_0000_0000]] {
        let op = operator(&ctx, 14, masks, 3, 5);
        sweep(&basis, &op, masks.len(), &[1, 6, 30, 38, 39, 150, 153, 154, 600, basis.len()]);
    }
}

#[test]
fn the_restricted_route_follows_the_dense_gather_switch() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let basis = Basis::register(&ctx, 14).unwrap();
    let d = basis.len();
    // One mask of 16 strings: 16384 rows x 16 terms reaches the work at which the dense action
    // gathers when an eighth of the rows is nonzero. `nonzero * 8` is d - 8, d and d + 8.
    let one = operator(&ctx, 14, &[0b1_0010_0010_1001], 16, 3);
    for (nonzero, restricted) in [(1, true), (2047, true), (2048, false), (2049, false), (d, false)] {
        let psi = state::<C64>(d, nonzero, 3, 9);
        let hpsi = hamiltonian_state::<C64>(d, 10);
        assert_eq!(compare(&basis, &one, &psi, &hpsi).is_some(), restricted, "support {nonzero}");
        let psi = state::<Complex32>(d, nonzero, 3, 9);
        let hpsi = hamiltonian_state::<Complex32>(d, 10);
        assert_eq!(compare(&basis, &one, &psi, &hpsi).is_some(), restricted, "support {nonzero} in float32");
    }
    // Eight masks of two strings: the rows a support can write are `support x 8`, so the
    // restriction refuses before the dense action would gather (300 x 8 x 8 > d, 300 x 8 < d).
    let masks = [0b11, 0b1100, 0b11_0000, 0b1100_0000, 0b11_0000_0000, 0b1100_0000_0000, 0b101, 0b1010];
    let many = operator(&ctx, 14, &masks, 2, 4);
    for (nonzero, restricted) in [(1, true), (40, true), (100, true), (300, false)] {
        let psi = state::<C64>(d, nonzero, 2, 11);
        let hpsi = hamiltonian_state::<C64>(d, 12);
        let written = compare(&basis, &many, &psi, &hpsi);
        assert_eq!(written.is_some(), restricted, "support {nonzero}");
        if let Some(written) = written {
            // The skipped work is real: far fewer rows written than the basis holds.
            assert!(written * 8 < d, "support {nonzero} wrote {written} of {d} rows");
        }
    }
}

/// A number-conserving Hermitian Hamiltonian on `modes` modes: hoppings and one-body energies.
fn hamiltonian(ctx: &Context, encoding: &Encoding, modes: usize) -> Operator {
    let identity = "I".repeat(modes);
    let mut h = Operator::new(ctx, modes, &[(identity, C64::new(-3., 0.))]).unwrap();
    for j in 0..modes {
        let term = encoding.monomial(ctx, &[(j, true), (j, false)]).unwrap();
        h = h.add(ctx, &term, C64::new(-1. + 0.13 * j as f64, 0.)).unwrap();
    }
    for j in 0..modes - 2 {
        for k in [j + 1, j + 2] {
            // The first half of the modes is spin up, the second half spin down: a hopping stays
            // in one half, or it leaves the particle sector.
            if k >= modes || j / (modes / 2) != k / (modes / 2) {
                continue;
            }
            let weight = C64::new(0.2 / (1. + (k - j) as f64) + 0.01 * j as f64, 0.);
            for (a, b) in [(j, k), (k, j)] {
                let term = encoding.monomial(ctx, &[(a, true), (b, false)]).unwrap();
                h = h.add(ctx, &term, weight).unwrap();
            }
        }
    }
    h
}

fn problem(threads: usize, pool: &str, precision: Precision) -> CachedProblem {
    let ctx = Context::new(Some(threads), 512 << 20).unwrap();
    let encoding = Encoding::new("jordan_wigner", 16, [3, 3]).unwrap();
    let h = hamiltonian(&ctx, &encoding, 16);
    let mut problem = CachedProblem::new(&ctx, encoding, h, true, false, true).unwrap();
    problem.use_pool(pool, true).unwrap();
    problem.with_precision(precision)
}

fn state_of<A: Amplitude>(problem: &CachedProblem, psi: &[A]) -> State {
    let mut buffer = problem.context.allocate::<A>(psi.len()).unwrap();
    buffer.as_mut_slice().copy_from_slice(psi);
    A::wrap(buffer)
}

/// The dense screen, spelled out: `H psi` stored in the state precision, then every generator's
/// action and a full-length dot.
fn dense_screen<A: Amplitude>(problem: &CachedProblem, psi: &[A]) -> Vec<f64> {
    let basis = problem.basis.as_ref().unwrap();
    let mut wide = vec![C64::default(); psi.len()];
    problem.hamiltonian.apply_into(basis, psi, &mut wide);
    let hpsi: Vec<A> = wide.iter().map(|&w| A::narrow(w)).collect();
    let mut action = vec![C64::default(); psi.len()];
    problem
        .pool
        .iter()
        .map(|g| {
            g.operator.apply_into(basis, psi, &mut action);
            2. * inner_wide(&hpsi, &action).re
        })
        .collect()
}

#[test]
fn the_problems_screen_is_the_dense_screen_on_any_thread_count() {
    for (pool, precision) in [
        ("fermionic", Precision::Float64),
        ("qeb", Precision::Float64),
        ("ceo", Precision::Float64),
        ("fermionic", Precision::Float32),
    ] {
        let (one, eight) = (problem(1, pool, precision), problem(8, pool, precision));
        let d = one.basis.as_ref().unwrap().len();
        assert_eq!(d, 56 * 56);
        let m = one.pool.len();
        assert!(m > 100, "{pool}: a pool of {m} is too small to split across threads");
        // The prepared state of a short ansatz, its support padded with signed zeros, a few
        // dozen amplitudes in the sparse range (`nonzero * 32 < d`), then one past it, an eighth,
        // and a full state.
        let prepared = |problem: &CachedProblem| problem.prepare_state(&[0, 1, 2], &[0.31, -0.2, 0.12], true).unwrap();
        let mut cases: Vec<State> = vec![prepared(&one)];
        let mut dense_states = Vec::new();
        for nonzero in [1, 20, 97, 98, d / 8, d] {
            for zeros in [0, 4] {
                dense_states.push((nonzero, zeros));
            }
        }
        let build = |problem: &CachedProblem, nonzero: usize, zeros: usize| match precision {
            Precision::Float64 => state_of(problem, &state::<C64>(d, nonzero, zeros, 31 + nonzero as u64)),
            Precision::Float32 => state_of(problem, &state::<Complex32>(d, nonzero, zeros, 31 + nonzero as u64)),
        };
        for (nonzero, zeros) in &dense_states {
            cases.push(build(&one, *nonzero, *zeros));
        }
        for (k, case) in cases.iter().enumerate() {
            let on_one = one.pool_gradients(case, false).unwrap();
            let expected = match case {
                State::Float64(buffer) => dense_screen(&one, buffer.as_slice()),
                State::Float32(buffer) => dense_screen(&one, buffer.as_slice()),
            };
            let again = match case {
                // The same amplitudes on the eight-thread problem.
                State::Float64(buffer) => eight.pool_gradients(&state_of(&eight, buffer.as_slice()), false),
                State::Float32(buffer) => eight.pool_gradients(&state_of(&eight, buffer.as_slice()), false),
            }
            .unwrap();
            assert_eq!(on_one.len(), m);
            for i in 0..m {
                assert_eq!(on_one[i].to_bits(), expected[i].to_bits(), "{pool} {precision:?} case {k} generator {i}");
                assert_eq!(again[i].to_bits(), expected[i].to_bits(), "{pool} {precision:?} case {k} generator {i} on 8 threads");
            }
        }
    }
}

#[test]
fn a_nonfinite_h_psi_takes_the_dense_route_and_keeps_its_nan() {
    // `0 * inf` is NaN in the dense dot; the restricted dot would skip it, so a screen whose
    // `H psi` is not finite must not restrict. A huge float32 state overflows `H psi` while the
    // state itself is finite.
    let one = problem(1, "fermionic", Precision::Float32);
    let d = one.basis.as_ref().unwrap().len();
    let mut psi = vec![Complex32::default(); d];
    psi[one.basis.as_ref().unwrap().reference] = Complex32::new(3.0e38, 0.);
    let huge = state_of(&one, &psi);
    let screened = one.pool_gradients(&huge, false).unwrap();
    let expected = dense_screen(&one, &psi);
    for (a, b) in screened.iter().zip(&expected) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn a_generator_the_restriction_refuses_falls_back_inside_a_restricted_screen() {
    // About 90 nonzero amplitudes of 3136: the screen is restricted, but a generator with five
    // flip masks may write `90 x 5 x 8 > d` rows, so its restriction refuses and it runs dense in
    // the middle of the pool. The row it leaves dirty must be cleared before the next generator
    // uses it, and its own value is the dense one.
    let masks: [&[u64]; 3] = [
        &[0b11, 0b1100, 0b11_0000, 0b1100_0000, 0b11_0000_0000],
        &[0b101, 0b1010, 0b1_0100_0000, 0b10_1000_0000, 0b1_0000_0001_0000, 0b11_0000_0000_0000],
        &[0b1_0000_0000_0001, 0b10_0000_0000_0010, 0b100_0000_0000_0100, 0b1000_0000_0000_1000, 0b11_1100],
    ];
    let slots = [7, 8, 300];
    let (mut one, mut eight) = (problem(1, "fermionic", Precision::Float64), problem(8, "fermionic", Precision::Float64));
    for problem in [&mut one, &mut eight] {
        for (&slot, masks) in slots.iter().zip(masks) {
            problem.pool[slot].operator = operator(&problem.context, 16, masks, 2, 40 + slot as u64);
        }
    }
    let basis = one.basis.as_ref().unwrap();
    let d = basis.len();
    let psi = state::<C64>(d, 90, 4, 77);
    let support = support(&psi);
    let mut action = vec![C64::default(); d];
    let mut written = Vec::new();
    for (i, g) in one.pool.iter().enumerate() {
        let applied = g.operator.apply_support_into(basis, &psi, &support, &mut action, &mut written);
        assert_eq!(applied, !slots.contains(&i), "generator {i}");
        for &row in &written {
            action[row] = C64::default();
        }
    }
    let screened = |problem: &CachedProblem, centered: bool| {
        problem.pool_gradients(&state_of(problem, &psi), centered).unwrap()
    };
    let expected = dense_screen(&one, &psi);
    let (on_one, on_eight) = (screened(&one, false), screened(&eight, false));
    for i in 0..one.pool.len() {
        assert_eq!(on_one[i].to_bits(), expected[i].to_bits(), "generator {i}");
        assert_eq!(on_eight[i].to_bits(), expected[i].to_bits(), "generator {i} on 8 threads");
    }
    // The centered screen (the one a search and ADAPT use) differs only in `H psi`, formed before
    // the route is chosen: it is thread invariant and has a refused generator too.
    let (on_one, on_eight) = (screened(&one, true), screened(&eight, true));
    for i in 0..one.pool.len() {
        assert_eq!(on_one[i].to_bits(), on_eight[i].to_bits(), "centered generator {i}");
    }
}
