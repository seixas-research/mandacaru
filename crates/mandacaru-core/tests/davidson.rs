//! The matrix-free Davidson solvers of `math::davidson` against dense eigensolves.
use mandacaru_core::math::{
    davidson::{lowest, lowest_paired, lowest_root, solve_shifted, Options, Paired, Symmetric},
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

/// An operator split into two blocks the diagonal does not reveal (an orbital Hessian's symmetry
/// blocks): the block of the smallest diagonal entries is positive, the other carries the lowest
/// root, -0.5, along its uniform vector. A start of unit vectors stays in the first block and
/// misses it; `lowest_root` finds it densely and by seeded Davidson alike.
#[test]
fn the_lowest_root_is_found_in_a_block_the_smallest_diagonal_does_not_reach() {
    let (n, split) = (300, 250);
    let mut a = DMatrix::zeros(n, n);
    a.view_mut((0, 0), (split, split))
        .copy_from(&response_like(split, 17, 0.002));
    let width = (n - split) as f64;
    for i in split..n {
        for j in split..n {
            a[(i, j)] = if i == j { 2. } else { 0. } - 2.5 / width;
        }
    }
    let (exact, _) = real_eigen(a.clone()).unwrap();
    assert!((exact[0] + 0.5).abs() < 1e-12, "{}", exact[0]);
    let operator = Dense(a);
    let unit = lowest(&operator, &Options { tolerance: 1e-9, ..Options::new(1) }).unwrap();
    assert!(unit.converged && unit.values[0] > 0.2, "the unit-vector start left its block: {}", unit.values[0]);
    let dense = lowest_root(&operator, n, 1e-9).unwrap().unwrap();
    let seeded = lowest_root(&operator, 0, 1e-9).unwrap().unwrap();
    println!(
        "unit start {:.6}; dense {:.12} ({} applications); seeded Davidson {:.12} ({} applications)",
        unit.values[0], dense.value, dense.applications, seeded.value, seeded.applications
    );
    assert!(dense.dense && dense.converged && (dense.value + 0.5).abs() < 1e-12);
    assert!(!seeded.dense && seeded.converged && (seeded.value + 0.5).abs() < 1e-10);
    assert!((dense.vector.dot(&seeded.vector).abs() - 1.).abs() < 1e-8);
}

/// `solve_shifted` against a dense solve: `(A + shift) x = b` for an indefinite `A` whose shifted
/// form is positive definite (the trust-region step of the SCF's second-order descent).
#[test]
fn the_shifted_solve_matches_a_dense_solve() {
    let n = 60;
    let mut a = response_like(n, 5, 0.01);
    a[(0, 0)] -= 0.6;
    let (values, _) = real_eigen(a.clone()).unwrap();
    let shift = 0.05 - values[0];
    let b = nalgebra::DVector::from_iterator(n, (0..n).map(|i| ((i * 7) % 11) as f64 - 5.));
    let solve = solve_shifted(&Dense(a.clone()), shift, &b, 1e-12, 500).unwrap();
    let x = &solve.x;
    let exact = (&a + DMatrix::identity(n, n) * shift).lu().solve(&b).unwrap();
    println!("lowest eigenvalue {:.4}, shift {shift:.4}: |x - exact| / |exact| {:.2e}", values[0], (x - &exact).norm() / exact.norm());
    assert!(values[0] < 0. && solve.converged && solve.positive);
    assert!((x - &exact).norm() < 1e-10 * exact.norm());
}

/// A shift that leaves the operator indefinite (set from a root above the lowest, as a missed
/// Davidson root would): the solve ends not converged and not positive, without an error.
#[test]
fn a_shift_that_leaves_the_operator_indefinite_is_reported_not_converged() {
    let n = 60;
    let mut a = response_like(n, 5, 0.01);
    a[(0, 0)] -= 0.6;
    let (values, _) = real_eigen(a.clone()).unwrap();
    // Positive above the second root only: the lowest stays negative.
    let shift = 0.05 - values[1];
    assert!(values[0] + shift < 0.);
    let b = nalgebra::DVector::from_iterator(n, (0..n).map(|i| ((i * 7) % 11) as f64 - 5.));
    let solve = solve_shifted(&Dense(a), shift, &b, 1e-12, 500).unwrap();
    println!("lowest {:.4}, second {:.4}, shift {shift:.4}: converged {}, positive {}", values[0], values[1], solve.converged, solve.positive);
    assert!(!solve.converged && !solve.positive);
}
