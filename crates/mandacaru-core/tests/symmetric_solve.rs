use mandacaru_core::math::symmetric_solve::{solve, Options};
use nalgebra::{DMatrix, DVector};

/// `Q diag(values) Q^T` for a fixed orthogonal `Q` (a Householder reflection chain).
fn with_spectrum(values: &[f64]) -> DMatrix<f64> {
    let n = values.len();
    let mut q = DMatrix::<f64>::identity(n, n);
    for k in 0..3 {
        let v = DVector::from_fn(n, |i, _| ((i * (k + 2) + k + 1) as f64).sin() + 0.3);
        q = q * (DMatrix::identity(n, n) - &v * v.transpose() * (2. / v.norm_squared()));
    }
    &q * DMatrix::from_diagonal(&DVector::from_column_slice(values)) * q.transpose()
}

#[test]
fn an_ill_conditioned_positive_system_is_solved_exactly() {
    // The condition number of a ceo Hessian: 1e6 from the stiffest to the flattest direction.
    let a = with_spectrum(&[6., 2., 0.7, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 7e-6]);
    let b: Vec<f64> = (0..9).map(|i| (i as f64 + 1.).cos()).collect();
    let s = solve(&a, &b, &Options::default()).unwrap();
    let exact = a
        .clone()
        .lu()
        .solve(&DVector::from_column_slice(&b))
        .unwrap();
    assert_eq!((s.kept, s.dropped), (9, 0));
    assert!((DVector::from_column_slice(&s.x) - &exact).norm() < 1e-8 * exact.norm());
    assert!(s.noise < 1e-12);
    assert_eq!(s.neglected_scale(), 0.);
}

#[test]
fn a_redundant_direction_is_dropped_and_the_solution_is_the_minimum_norm_one() {
    // One exact null direction (a redundant parameter), a gradient in the range, FD-like noise.
    let a = with_spectrum(&[3., 1., 0.2, 0.]);
    let null = a.clone().symmetric_eigen();
    let direction = null
        .eigenvectors
        .column(null.eigenvalues.imin())
        .into_owned();
    let b = (&a * DVector::from_column_slice(&[1., -0.5, 0.25, 2.]))
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let mut noisy = a.clone();
    for i in 0..4 {
        for j in 0..4 {
            noisy[(i, j)] += 3e-9 * (((i * 7 + j * 3) as f64).sin()) * if i < j { 1. } else { -1. };
        }
    }
    let s = solve(&noisy, &b, &Options::default()).unwrap();
    assert_eq!((s.kept, s.dropped), (3, 1));
    assert!(s.noise > 1e-9 && s.cutoff >= 10. * s.noise);
    let pseudo = a.pseudo_inverse(1e-9).unwrap() * DVector::from_column_slice(&b);
    assert!(
        (DVector::from_column_slice(&s.x) - &pseudo).norm() < 1e-6,
        "{:?}",
        s.x
    );
    assert!(DVector::from_column_slice(&s.x).dot(&direction).abs() < 1e-6);
    assert!(s.dropped_rhs < 1e-6);
}

#[test]
fn the_cutoff_follows_the_measured_noise_of_the_matrix() {
    let values = [4., 1., 1e-6];
    let b = [1., 1., 1.];
    let quiet = solve(&with_spectrum(&values), &b, &Options::default()).unwrap();
    assert_eq!(quiet.dropped, 0);
    // The same eigenvalue under noise of 1e-5 is not distinguishable from zero.
    let mut noisy = with_spectrum(&values);
    noisy[(0, 1)] += 1e-5;
    noisy[(1, 0)] -= 1e-5;
    let loud = solve(&noisy, &b, &Options::default()).unwrap();
    assert_eq!(loud.dropped, 1);
    assert!(loud.dropped_rhs > 0. && loud.neglected_scale() > 0. && loud.cutoff > 1e-4);
    assert!(
        loud.x.iter().all(|v| v.abs() < 10.),
        "the flat direction must not blow up {:?}",
        loud.x
    );
}

#[test]
fn a_negative_curvature_is_refused_by_name() {
    let error = solve(
        &with_spectrum(&[2., 1., -0.3]),
        &[1., 0., 0.],
        &Options::default(),
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("negative curvature"), "{error}");
    // A negative eigenvalue inside the noise is a flat direction, not a refusal.
    let flat = solve(
        &with_spectrum(&[2., 1., -1e-13]),
        &[1., 0., 0.],
        &Options::default(),
    )
    .unwrap();
    assert_eq!(flat.dropped, 1);
}

#[test]
fn malformed_inputs_are_refused() {
    let a = DMatrix::<f64>::identity(2, 2);
    assert!(solve(&a, &[1.], &Options::default()).is_err());
    assert!(solve(&a, &[1., f64::NAN], &Options::default()).is_err());
}

#[test]
fn the_solution_satisfies_the_kept_system_to_roundoff_on_a_clustered_spectrum() {
    // Clustered eigenvalues are where an unverified eigensolver loses its reconstruction.
    let a = with_spectrum(&[
        2.,
        2. + 1e-9,
        2. - 1e-9,
        1e-3,
        1e-3 * (1. + 1e-8),
        5e-6,
        5e-6,
        0.4,
    ]);
    let b: Vec<f64> = (0..8).map(|i| (0.7 * i as f64 + 0.2).sin()).collect();
    let s = solve(&a, &b, &Options::default()).unwrap();
    assert_eq!((s.kept, s.dropped), (8, 0));
    // Backward stable: the residual is roundoff of |A| |x|, whatever the clustering.
    let x = DVector::from_column_slice(&s.x);
    let residual = &a * &x - DVector::from_column_slice(&b);
    let scale = a.norm() * x.norm();
    assert!(
        residual.norm() < 1e-14 * scale,
        "{:.3e} of {scale:.3e}",
        residual.norm()
    );
}
