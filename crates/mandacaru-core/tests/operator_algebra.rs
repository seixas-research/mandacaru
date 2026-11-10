//! The Pauli-operator algebra behind `PauliSum` and `Fermion`: commutators, the Hermitian part,
//! compressed-row matrices, plain encodings and the fermionic term builders (MG-Q5).
use mandacaru_core::{
    quantum::{fermion_terms, Basis, Encoding, Operator, C64},
    Context,
};

fn context() -> Context {
    Context::new(Some(1), 256 << 20).unwrap()
}

/// A deterministic stream in `[-1, 1)`.
struct Stream(u64);
impl Stream {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 52) as f64) - 1.
    }
}

fn random_operator(ctx: &Context, n: usize, count: usize, seed: u64) -> Operator {
    let mut stream = Stream(seed);
    let terms: Vec<(String, C64)> = (0..count)
        .map(|_| {
            let label: String = (0..n)
                .map(|_| ['I', 'X', 'Y', 'Z'][((stream.next() + 1.) * 2.) as usize % 4])
                .collect();
            (label, C64::new(stream.next(), stream.next()))
        })
        .collect();
    Operator::new(ctx, n, &terms).unwrap()
}

fn dense(ctx: &Context, op: &Operator) -> Vec<C64> {
    let basis = Basis::register(ctx, op.n).unwrap();
    op.dense(ctx, &basis).unwrap().as_slice().to_vec()
}

fn largest_difference(a: &Operator, b: &Operator) -> f64 {
    let ctx = context();
    let difference = a.add(&ctx, b, C64::new(-1., 0.)).unwrap();
    difference
        .terms
        .iter()
        .map(|t| t.coefficient.norm())
        .fold(0., f64::max)
}

#[test]
fn the_commutator_is_the_difference_of_the_products() {
    let ctx = context();
    for (n, seed) in [(1, 1), (3, 2), (5, 3)] {
        let a = random_operator(&ctx, n, 14, seed);
        let b = random_operator(&ctx, n, 9, seed + 100);
        let by_products = a
            .compose(&ctx, &b)
            .unwrap()
            .add(&ctx, &b.compose(&ctx, &a).unwrap(), C64::new(-1., 0.))
            .unwrap();
        assert!(largest_difference(&a.commutator(&ctx, &b).unwrap(), &by_products) < 1e-12);
        // Antisymmetric, and zero against itself and against a commuting operator.
        let reversed = b.commutator(&ctx, &a).unwrap();
        let sum = a
            .commutator(&ctx, &b)
            .unwrap()
            .add(&ctx, &reversed, C64::new(1., 0.))
            .unwrap();
        assert!(sum.terms.is_empty());
        assert!(a.commutator(&ctx, &a).unwrap().terms.is_empty());
    }
    let zz = Operator::new(&ctx, 2, &[("ZZ".into(), C64::new(1.5, 0.))]).unwrap();
    let z = Operator::new(&ctx, 2, &[("IZ".into(), C64::new(0., 0.3))]).unwrap();
    assert!(zz.commutator(&ctx, &z).unwrap().terms.is_empty());
    assert!(zz.commutes_with(&ctx, &z).unwrap());
    let x = Operator::new(&ctx, 2, &[("XI".into(), C64::new(1., 0.))]).unwrap();
    let y = Operator::new(&ctx, 2, &[("ZI".into(), C64::new(1., 0.))]).unwrap();
    assert!(!x.commutes_with(&ctx, &y).unwrap());
    assert!(a_register_mismatch_is_refused(&ctx));
}

fn a_register_mismatch_is_refused(ctx: &Context) -> bool {
    let a = random_operator(ctx, 2, 3, 1);
    let b = random_operator(ctx, 3, 3, 2);
    a.commutator(ctx, &b).is_err() && a.compose(ctx, &b).is_err()
}

#[test]
fn the_hermitian_part_is_half_the_sum_with_the_adjoint() {
    let ctx = context();
    let a = random_operator(&ctx, 4, 30, 7);
    let half = a
        .add(&ctx, &a.dagger(&ctx).unwrap(), C64::new(1., 0.))
        .unwrap();
    let expected = Operator::new(&ctx, 4, &[])
        .unwrap()
        .add(&ctx, &half, C64::new(0.5, 0.))
        .unwrap();
    let part = a.hermitian_part(&ctx).unwrap();
    assert!(part.terms.iter().all(|t| t.coefficient.im == 0.));
    assert!(largest_difference(&part, &expected) < 1e-14);
}

#[test]
fn the_csr_matrix_is_the_dense_one_with_sorted_columns() {
    let ctx = context();
    for (n, count, seed) in [(1, 3, 1), (3, 14, 2), (5, 60, 3)] {
        let op = random_operator(&ctx, n, count, seed);
        let basis = Basis::register(&ctx, n).unwrap();
        let csr = op.csr(&ctx, &basis).unwrap();
        let d = basis.len();
        let expected = dense(&ctx, &op);
        let mut rebuilt = vec![C64::default(); d * d];
        for row in 0..d {
            let (lo, hi) = (csr.offsets.as_slice()[row], csr.offsets.as_slice()[row + 1]);
            let columns = &csr.columns.as_slice()[lo..hi];
            assert!(columns.windows(2).all(|w| w[0] < w[1]));
            for k in lo..hi {
                rebuilt[row * d + csr.columns.as_slice()[k]] = csr.values.as_slice()[k];
            }
        }
        let worst = rebuilt
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).norm())
            .fold(0., f64::max);
        assert!(worst < 1e-13);
    }
    let empty = Operator::new(&ctx, 3, &[]).unwrap();
    let csr = empty
        .csr(&ctx, &Basis::register(&ctx, 3).unwrap())
        .unwrap();
    assert_eq!(csr.nnz, 0);
}

#[test]
fn plain_encodings_take_an_odd_mode_count_and_keep_the_anticommutation_relations() {
    let ctx = context();
    for mapping in ["jordan_wigner", "parity", "bravyi_kitaev"] {
        let encoding = Encoding::plain(mapping, 3).unwrap();
        assert_eq!(encoding.n_qubits(), 3);
        let identity = Operator::new(&ctx, 3, &[("III".into(), C64::new(1., 0.))]).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let a = encoding.ladder(&ctx, i, false).unwrap();
                let adagger = encoding.ladder(&ctx, j, true).unwrap();
                let anti = a
                    .compose(&ctx, &adagger)
                    .unwrap()
                    .add(&ctx, &adagger.compose(&ctx, &a).unwrap(), C64::new(1., 0.))
                    .unwrap();
                if i == j {
                    assert!(largest_difference(&anti, &identity) < 1e-12, "{mapping}");
                } else {
                    assert!(anti.terms.is_empty(), "{mapping}");
                }
            }
        }
    }
    assert!(Encoding::plain("parity_reduced", 4).is_err());
    assert!(Encoding::plain("jordan_wigner", 0).is_err());
    assert!(Encoding::plain("nonsense", 3).is_err());
    assert!(Encoding::new("jordan_wigner", 3, [0, 0]).is_err());
}

#[test]
fn a_mapped_fermion_is_the_sum_of_its_mapped_terms() {
    let ctx = context();
    let terms = vec![
        (vec![(0, true), (1, false)], C64::new(0.2, 0.7)),
        (vec![(1, true), (0, false)], C64::new(0.2, -0.7)),
        (vec![(0, true), (3, true), (4, false), (1, false)], C64::new(0.31, 0.)),
        (vec![], C64::new(0.75, 0.)),
    ];
    for mapping in ["jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced"] {
        let encoding = Encoding::new(mapping, 6, [1, 1]).unwrap();
        let mapped = encoding.map_fermion(&ctx, &terms).unwrap();
        let mut by_hand = Operator::new(&ctx, encoding.n_qubits(), &[]).unwrap();
        for (ladders, c) in &terms {
            by_hand = by_hand
                .add(&ctx, &encoding.monomial(&ctx, ladders).unwrap(), *c)
                .unwrap();
        }
        assert!(largest_difference(&mapped, &by_hand) < 1e-14, "{mapping}");
    }
    let reduced = Encoding::new("parity_reduced", 4, [1, 1]).unwrap();
    assert!(reduced
        .map_fermion(&ctx, &[(vec![(0, true)], C64::new(1., 0.))])
        .is_err());
    assert!(Encoding::plain("jordan_wigner", 2)
        .unwrap()
        .map_fermion(&ctx, &[(vec![(2, true)], C64::new(1., 0.))])
        .is_err());
}

fn complex(stream: &mut Stream, len: usize, keep: f64) -> Vec<C64> {
    (0..len)
        .map(|_| {
            let (re, im) = (stream.next(), stream.next());
            if stream.next() < keep {
                C64::new(re, im)
            } else {
                C64::default()
            }
        })
        .collect()
}

#[test]
fn spin_orbital_terms_follow_the_lexicographic_order() {
    let ctx = context();
    let one = [
        C64::new(1., 0.),
        C64::new(0., 0.),
        C64::new(2., -1.),
        C64::new(1e-13, 0.),
    ];
    let terms = fermion_terms::from_spin_orbital(&ctx, 2, &one, None, 1e-12).unwrap();
    let ladders: Vec<_> = terms.iter().map(|t| t.0.clone()).collect();
    assert_eq!(
        ladders,
        vec![vec![(0, true), (0, false)], vec![(1, true), (0, false)]]
    );
    assert_eq!(terms[1].1, C64::new(2., -1.));
    // The kept threshold is strict: an entry of exactly `tol` is dropped, tol = 0 keeps all but 0.
    let all = fermion_terms::from_spin_orbital(&ctx, 2, &one, None, 0.).unwrap();
    assert_eq!(all.len(), 3);
    let mut two = vec![C64::default(); 16];
    two[(((0 * 2 + 1) * 2 + 0) * 2) + 1] = C64::new(0.5, 0.25);
    let terms = fermion_terms::from_spin_orbital(&ctx, 2, &one, Some(&two), 1e-12).unwrap();
    let last = terms.last().unwrap();
    // g[p,q,r,s] enters as 1/2 a+_p a+_q a_s a_r.
    assert_eq!(last.0, vec![(0, true), (1, true), (1, false), (0, false)]);
    assert_eq!(last.1, C64::new(0.25, 0.125));
    assert!(fermion_terms::from_spin_orbital(&ctx, 2, &one[..3], None, 0.).is_err());
}

/// The spin-orbital tensor holding the spatial blocks, for comparison only.
fn spin_blocks(m: usize, two: &[C64], resolved: bool) -> Vec<C64> {
    let n = 2 * m;
    let m4 = m * m * m * m;
    let mut out = vec![C64::default(); n * n * n * n];
    for sigma in 0..2 {
        for tau in 0..2 {
            let block = &two[if resolved { (sigma * 2 + tau) * m4 } else { 0 }..][..m4];
            for p in 0..m {
                for q in 0..m {
                    for r in 0..m {
                        for s in 0..m {
                            let (a, b, c, d) = (sigma * m + p, tau * m + q, sigma * m + r, tau * m + s);
                            out[((a * n + b) * n + c) * n + d] = block[((p * m + q) * m + r) * m + s];
                        }
                    }
                }
            }
        }
    }
    out
}

#[test]
fn spatial_terms_are_those_of_the_spin_blocked_tensor_in_the_same_order() {
    let ctx = context();
    let m = 3;
    let mut stream = Stream(99);
    for resolved in [false, true] {
        let spins = if resolved { 2 } else { 1 };
        let one = complex(&mut stream, spins * m * m, 0.8);
        let two = complex(&mut stream, spins * spins * m * m * m * m, 0.4);
        let extra = complex(&mut stream, 4 * m * m, 0.5);
        let n = 2 * m;
        let mut h = vec![C64::default(); n * n];
        for sigma in 0..2 {
            for p in 0..m {
                for q in 0..m {
                    h[(sigma * m + p) * n + sigma * m + q] =
                        one[if resolved { sigma * m * m } else { 0 } + p * m + q];
                }
            }
        }
        for (value, add) in h.iter_mut().zip(&extra) {
            *value += add;
        }
        let expected = fermion_terms::from_spin_orbital(
            &ctx,
            n,
            &h,
            Some(&spin_blocks(m, &two, resolved)),
            1e-12,
        )
        .unwrap();
        let built =
            fermion_terms::from_spatial(&ctx, m, resolved, &one, Some(&extra), Some(&two), 1e-12)
                .unwrap();
        assert_eq!(built, expected);
        let no_two = fermion_terms::from_spatial(&ctx, m, resolved, &one, None, None, 1e-12)
            .unwrap();
        assert!(no_two.iter().all(|t| t.0.len() == 2));
        assert!(fermion_terms::from_spatial(&ctx, m, resolved, &one[1..], None, None, 0.).is_err());
    }
    assert_eq!(ctx.memory_used(), 0);
}
