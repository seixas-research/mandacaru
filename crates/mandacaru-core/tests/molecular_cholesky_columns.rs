//! The pivoted Cholesky of a pair matrix read one column at a time
//! ([`mandacaru_core::molecular::integrals::cholesky_from_columns`]) on a synthetic positive
//! semidefinite matrix of known rank: its factors reproduce the matrix, a fixed pivot list
//! reproduces the factors, and the reserved peak of the factorization is the factors themselves
//! (grown block by block and assembled while the blocks are released), not twice that.
use mandacaru_core::molecular::integrals::{cholesky_from_columns, pair, Eri, PairColumns};
use mandacaru_core::{Context, Result};

/// `K = A A^T` for a deterministic `A` of `rank` columns whose entries halve every `per` columns.
struct Synthetic {
    m: usize,
    k: Vec<f64>,
}

impl Synthetic {
    fn new(m: usize, rank: usize, per: usize) -> Self {
        let n = m * (m + 1) / 2;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let a: Vec<f64> = (0..n * rank)
            .map(|i| next() * (0.5f64).powi((i % rank / per) as i32))
            .collect();
        let mut k = vec![0.; n * n];
        for x in 0..n {
            for y in 0..n {
                k[x * n + y] = (0..rank).map(|c| a[x * rank + c] * a[y * rank + c]).sum();
            }
        }
        Self { m, k }
    }
    fn n(&self) -> usize {
        self.m * (self.m + 1) / 2
    }
}

impl PairColumns for Synthetic {
    fn m(&self) -> usize {
        self.m
    }
    fn diagonal(&self) -> Result<Vec<f64>> {
        let n = self.n();
        Ok((0..n).map(|a| self.k[a * n + a]).collect())
    }
    fn column(&self, pivot: usize) -> Result<Vec<f64>> {
        let n = self.n();
        Ok((0..n).map(|a| self.k[a * n + pivot]).collect())
    }
    fn departures(&self) -> (f64, f64) {
        (0., 0.)
    }
}

/// `(p, r)` of the packed pair `a`, `p >= r`.
fn unpair(a: usize) -> (usize, usize) {
    let p = (0..).find(|p| (p + 1) * (p + 2) / 2 > a).unwrap();
    (p, a - p * (p + 1) / 2)
}

/// The largest `|factors - K|` over every pair of pairs.
fn miss(eri: &Eri, source: &Synthetic) -> f64 {
    let n = source.n();
    let mut worst = 0f64;
    for a in 0..n {
        let (p, r) = unpair(a);
        for b in 0..n {
            let (q, s) = unpair(b);
            assert_eq!(pair(p, r), a);
            worst = worst.max((eri.value(p, q, r, s) - source.k[a * n + b]).abs());
        }
    }
    worst
}

/// 300 pairs (two blocks of rows) of rank 90: the stride grows twice.
#[test]
fn the_factors_reproduce_a_matrix_of_known_rank() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let source = Synthetic::new(24, 90, 8);
    let eri = cholesky_from_columns(&ctx, &source, 1e-12, None).unwrap();
    let (rank, residual) = eri.report().cholesky.unwrap();
    assert_eq!(rank, 90);
    assert!(residual.unwrap() <= 1e-12);
    let gap = miss(&eri, &source);
    assert!(gap < 1e-11, "{gap:e}");
}

/// The pivots of a factorization, given back as fixed pivots, give the same factors.
#[test]
fn fixed_pivots_reproduce_the_factors() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let source = Synthetic::new(24, 90, 8);
    let free = cholesky_from_columns(&ctx, &source, 1e-10, None).unwrap();
    let fixed = cholesky_from_columns(&ctx, &source, 1e-10, free.pivots()).unwrap();
    assert_eq!(fixed.report().cholesky.unwrap().1, None);
    let gap = miss(&fixed, &source);
    assert!(gap < 1e-9, "{gap:e}");
    assert_eq!(fixed.pivots(), free.pivots());
}

/// A looser tolerance keeps the first pivots of a tighter one (the pivots are nested), and every
/// integral stays within the tolerance.
#[test]
fn a_looser_tolerance_is_a_prefix_within_its_bound() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let source = Synthetic::new(24, 200, 8);
    let tight = cholesky_from_columns(&ctx, &source, 1e-9, None).unwrap();
    let loose = cholesky_from_columns(&ctx, &source, 1e-4, None).unwrap();
    let (tight_pivots, loose_pivots) = (tight.pivots().unwrap(), loose.pivots().unwrap());
    assert!(loose_pivots.len() < tight_pivots.len());
    assert_eq!(loose_pivots, &tight_pivots[..loose_pivots.len()]);
    let gap = miss(&loose, &source);
    assert!(gap <= 1e-4, "{gap:e}");
}

/// A matrix of 2080 pairs and rank above 1000: the reserved peak is the factors and the vectors
/// of one pivot, with the stride's slack of an eighth, never a second copy of the factors (twice
/// their size, as the first streamed version held them).
#[test]
fn the_reserved_peak_of_a_factorization_is_the_factors() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let source = Synthetic::new(64, 2080, 64);
    let eri = cholesky_from_columns(&ctx, &source, 1e-12, None).unwrap();
    let (rank, _) = eri.report().cholesky.unwrap();
    let factors = rank * source.n() * 8;
    let peak = ctx.memory_peak();
    println!("rank {rank}: factors {} B, reserved peak {peak} B", factors);
    assert!(rank > 1000);
    assert!(
        peak < factors + factors / 5,
        "peak {peak} factors {factors}"
    );
}

/// A budget the factors outgrow is a named refusal (not an abort), and nothing beyond the
/// budget is addressed up front: the factors of this matrix need 25 MB, the context holds 8.
#[test]
fn factors_beyond_the_budget_are_refused() {
    let ctx = Context::new(Some(2), 8 << 20).unwrap();
    let source = Synthetic::new(64, 2080, 64);
    let error = cholesky_from_columns(&ctx, &source, 1e-12, None)
        .err()
        .unwrap();
    assert!(
        matches!(error, mandacaru_core::Error::Budget { .. }),
        "{error}"
    );
}
