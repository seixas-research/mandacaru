use mandacaru_core::math::krylov::{minimum_norm, Options};
use nalgebra::{DMatrix, DVector};

fn product(matrix: &DMatrix<f64>) -> impl FnMut(&[f64]) -> mandacaru_core::Result<Vec<f64>> + '_ {
    move |v| Ok((matrix * DVector::from_row_slice(v)).iter().copied().collect())
}

#[test]
fn a_positive_definite_system_is_solved() {
    let a = DMatrix::from_row_slice(3, 3, &[4., 1., 0., 1., 3., 1., 0., 1., 2.]);
    let b = [1., -2., 0.5];
    let solution = minimum_norm(&b, product(&a), &Options::default()).unwrap();
    let exact = a.clone().lu().solve(&DVector::from_row_slice(&b)).unwrap();
    assert!((DVector::from_row_slice(&solution.x) - exact).norm() < 1e-10);
    assert!(!solution.truncated && solution.iterations <= 3);
    assert!(solution.lowest_curvature > 0.);
}

#[test]
fn a_singular_consistent_system_gives_the_minimum_norm_solution() {
    // Rank 2 in 3 dimensions: the null vector (1, -1, 0) / sqrt(2) must not enter.
    let a = DMatrix::from_row_slice(3, 3, &[1., 1., 0., 1., 1., 0., 0., 0., 2.]);
    let b = [2., 2., 4.];
    let solution = minimum_norm(&b, product(&a), &Options::default()).unwrap();
    let pseudo = a.clone().pseudo_inverse(1e-12).unwrap() * DVector::from_row_slice(&b);
    assert!((DVector::from_row_slice(&solution.x) - pseudo).norm() < 1e-9, "{:?}", solution.x);
    assert!((solution.x[0] - solution.x[1]).abs() < 1e-9);
}

#[test]
fn an_indefinite_matrix_is_refused_by_name() {
    let a = DMatrix::from_row_slice(2, 2, &[1., 0., 0., -2.]);
    let error = minimum_norm(&[1., 1.], product(&a), &Options::default()).err().unwrap();
    assert!(error.to_string().contains("negative curvature"), "{error}");
}

#[test]
fn a_right_hand_side_in_the_null_space_is_refused_as_inconsistent() {
    let a = DMatrix::from_row_slice(2, 2, &[1., 0., 0., 0.]);
    let error = minimum_norm(&[0., 1.], product(&a), &Options::default()).err().unwrap();
    assert!(error.to_string().contains("inconsistent"), "{error}");
}

#[test]
fn a_zero_right_hand_side_needs_no_product() {
    let solution = minimum_norm(&[0., 0.], |_| panic!("no product"), &Options::default()).unwrap();
    assert_eq!(solution.x, [0., 0.]);
    assert_eq!(solution.iterations, 0);
}
