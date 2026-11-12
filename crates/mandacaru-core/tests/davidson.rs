//! The matrix-free Davidson solvers of `math::davidson` against dense eigensolves.
use mandacaru_core::math::{
    davidson::{lowest, lowest_paired, Options, Paired, Symmetric},
    linalg::real_eigen,
};
use nalgebra::DMatrix;

/// A deterministic pseudo-random matrix (a linear congruential sequence).
fn noise(rows: usize, columns: usize, seed: u64) -> DMatrix<f64> {
    let mut state = seed;
    DMatrix::from_fn(rows, columns, |_, _| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    })
}
/// A diagonally dominant symmetric matrix with a spread diagonal, like a response matrix.
fn response_like(n: usize, seed: u64, coupling: f64) -> DMatrix<f64> {
    let r = noise(n, n, seed);
    let mut a = (&r + r.transpose()) * coupling;
    for i in 0..n {
        a[(i, i)] += 0.3 + 0.05 * i as f64 + 0.01 * ((i * 7) % 5) as f64;
    }
    a
}
struct Dense(DMatrix<f64>);
impl Symmetric for Dense {
    fn dimension(&self) -> usize {
        self.0.nrows()
    }
    fn diagonal(&self) -> Vec<f64> {
        self.0.diagonal().iter().copied().collect()
    }
    fn apply(&self, block: &DMatrix<f64>) -> mandacaru_core::Result<DMatrix<f64>> {
        Ok(&self.0 * block)
    }
}
struct DensePair {
    sum: DMatrix<f64>,
    difference: DMatrix<f64>,
}
impl Paired for DensePair {
    fn dimension(&self) -> usize {
        self.sum.nrows()
    }
    fn diagonal(&self) -> Vec<f64> {
        self.difference.diagonal().iter().copied().collect()
    }
    fn sum(&self, block: &DMatrix<f64>) -> mandacaru_core::Result<DMatrix<f64>> {
        Ok(&self.sum * block)
    }
    fn difference(&self, block: &DMatrix<f64>) -> mandacaru_core::Result<DMatrix<f64>> {
        Ok(&self.difference * block)
    }
}
fn symmetric_root(m: &DMatrix<f64>) -> DMatrix<f64> {
    let (values, vectors) = real_eigen(m.clone()).unwrap();
    let d = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
        values.len(),
        values.iter().map(|v| v.sqrt()),
    ));
    &vectors * d * vectors.transpose()
}

#[test]
fn the_lowest_roots_of_a_symmetric_operator_match_a_dense_solve() {
    let a = response_like(120, 7, 0.02);
    let (exact, _) = real_eigen(a.clone()).unwrap();
    let mut options = Options::new(6);
    options.tolerance = 1e-9;
    let roots = lowest(&Dense(a.clone()), &options).unwrap();
    assert!(roots.converged);
    for k in 0..6 {
        assert!((roots.values[k] - exact[k]).abs() < 1e-12, "root {k}");
        let x = roots.vectors.column(k);
        let residual = (&a * x - x * roots.values[k]).norm();
        assert!(residual <= 1e-9);
    }
    let gram = roots.vectors.transpose() * &roots.vectors;
    assert!((gram - DMatrix::identity(6, 6)).amax() < 1e-12);
    // Every root asked for: the subspace is the whole space and the roots exact.
    let small = response_like(9, 3, 0.05);
    let all = lowest(&Dense(small.clone()), &Options::new(9)).unwrap();
    let (exact, _) = real_eigen(small).unwrap();
    for k in 0..9 {
        assert!((all.values[k] - exact[k]).abs() < 1e-13);
    }
}

#[test]
fn the_solver_is_deterministic_and_signs_are_fixed() {
    let a = response_like(80, 11, 0.03);
    let first = lowest(&Dense(a.clone()), &Options::new(4)).unwrap();
    let second = lowest(&Dense(a), &Options::new(4)).unwrap();
    assert_eq!(first.values, second.values);
    assert_eq!(first.vectors, second.vectors);
    for column in first.vectors.column_iter() {
        let pivot = column.iamax();
        assert!(column[pivot] > 0.);
    }
}

#[test]
fn the_paired_roots_match_the_symmetric_casida_form() {
    let n = 90;
    let difference = response_like(n, 5, 0.01);
    let coupling = noise(n, n, 9);
    let sum = &difference + (&coupling + coupling.transpose()) * 0.01;
    let root = symmetric_root(&difference);
    let (squares, _) = real_eigen(&root * &sum * &root).unwrap();
    let mut options = Options::new(5);
    options.tolerance = 1e-9;
    let roots = lowest_paired(
        &DensePair {
            sum: sum.clone(),
            difference: difference.clone(),
        },
        &options,
    )
    .unwrap();
    assert!(roots.converged);
    for k in 0..5 {
        assert!(
            (roots.values[k] - squares[k].sqrt()).abs() < 1e-11,
            "root {k}"
        );
        let (plus, minus) = (roots.plus.column(k), roots.minus.column(k));
        assert!((plus.dot(&minus) - 1.).abs() < 1e-12, "(X+Y).(X-Y) = 1");
        assert!((&sum * plus - minus * roots.values[k]).norm() < 1e-8);
        assert!((&difference * minus - plus * roots.values[k]).norm() < 1e-8);
    }
    // With B = 0 the paired roots are the eigenvalues of A.
    let alone = lowest_paired(
        &DensePair {
            sum: difference.clone(),
            difference: difference.clone(),
        },
        &options,
    )
    .unwrap();
    let (exact, _) = real_eigen(difference).unwrap();
    for k in 0..5 {
        assert!((alone.values[k] - exact[k]).abs() < 1e-11);
    }
}

#[test]
fn an_unstable_paired_problem_raises() {
    let n = 30;
    let difference = response_like(n, 2, 0.01);
    let mut sum = difference.clone();
    sum[(0, 0)] = -0.5;
    let error = lowest_paired(&DensePair { sum, difference }, &Options::new(2)).unwrap_err();
    assert!(error.to_string().contains("no real root"), "{error}");
}
