use mandacaru_core::{
    math::{
        linalg::real_eigen,
        separable::{eigenpair, Pencil},
    },
    Context,
};
use nalgebra::DMatrix;

#[test]
fn separable_states_match_an_independent_dense_generalized_solve() {
    let ctx = Context::new(Some(1), 4 << 20).unwrap();
    for n in [3, 8, 29] {
        for rank in [1, 2, 3] {
            let diagonal: Vec<_> = (0..n).map(|i| 2. + i as f64 * 0.07).collect();
            let off = vec![-0.9; n - 1];
            let u = DMatrix::from_fn(n, rank, |i, j| {
                ((i + 1) as f64 * (j + 2) as f64).sin() * 0.12
            });
            let d = DMatrix::from_fn(rank, rank, |i, j| {
                if i == j {
                    (i as f64 - 1.) * 1.3
                } else {
                    0.07
                }
            });
            let q = DMatrix::from_fn(rank, rank, |i, j| {
                if i == j {
                    (i as f64 - 1.) * 0.3
                } else {
                    -0.03
                }
            });
            let t = DMatrix::from_fn(n, n, |i, j| {
                if i == j {
                    diagonal[i]
                } else if i.abs_diff(j) == 1 {
                    off[i.min(j)]
                } else {
                    0.
                }
            });
            let h = t + &u * &d * u.transpose();
            let s = DMatrix::identity(n, n) + &u * &q * u.transpose();
            let inverse_l = s.clone().cholesky().unwrap().l().try_inverse().unwrap();
            let (expected, _) = real_eigen(&inverse_l * &h * inverse_l.transpose()).unwrap();
            let projectors: Vec<_> = (0..n)
                .flat_map(|i| {
                    (0..rank).map({
                        let u = &u;
                        move |j| u[(i, j)]
                    })
                })
                .collect();
            let coupling: Vec<_> = (0..rank)
                .flat_map(|i| {
                    (0..rank).map({
                        let d = &d;
                        move |j| d[(i, j)]
                    })
                })
                .collect();
            let metric: Vec<_> = (0..rank)
                .flat_map(|i| {
                    (0..rank).map({
                        let q = &q;
                        move |j| q[(i, j)]
                    })
                })
                .collect();
            for (state, &level) in expected.iter().enumerate() {
                let solved = eigenpair(
                    &ctx,
                    Pencil {
                        diagonal: &diagonal,
                        off_diagonal: &off,
                        projectors: &projectors,
                        coupling: &coupling,
                        metric: &metric,
                        rank,
                    },
                    state,
                    true,
                )
                .unwrap();
                assert!(
                    (solved.energy - level).abs() < 2e-10,
                    "n={n}, rank={rank}, state={state}, got {} expected {level}",
                    solved.energy
                );
                let v = nalgebra::DVector::from_column_slice(
                    solved.vector.as_ref().unwrap().as_slice(),
                );
                assert!((v.dot(&(&s * &v)) - 1.).abs() < 1e-12);
                assert!((&h * &v - &s * &v * solved.energy).norm() < 3e-8);
                let values_only = eigenpair(
                    &ctx,
                    Pencil {
                        diagonal: &diagonal,
                        off_diagonal: &off,
                        projectors: &projectors,
                        coupling: &coupling,
                        metric: &metric,
                        rank,
                    },
                    state,
                    false,
                )
                .unwrap();
                assert_eq!(values_only.energy, solved.energy);
                assert!(values_only.vector.is_none());
            }
            assert_eq!(ctx.memory_used(), 0);
        }
    }
}

#[test]
fn radial_size_storage_is_linear_and_invalid_overlap_releases_the_budget() {
    let ctx = Context::new(Some(1), 3 << 20).unwrap();
    let n = 3000;
    let diagonal = vec![2.; n];
    let off = vec![-1.; n - 1];
    let u: Vec<_> = (0..n)
        .map(|i| ((i + 1) as f64 * 0.003).sin() / (n as f64).sqrt())
        .collect();
    let solve = |metric: &[f64]| {
        eigenpair(
            &ctx,
            Pencil {
                diagonal: &diagonal,
                off_diagonal: &off,
                projectors: &u,
                coupling: &[0.03],
                metric,
                rank: 1,
            },
            0,
            true,
        )
    };
    assert!(solve(&[0.2]).unwrap().energy > 0.);
    assert_eq!(ctx.memory_used(), 0);
    assert!(solve(&[-100.]).is_err());
    assert_eq!(ctx.memory_used(), 0);
    let tiny = Context::new(Some(1), 1024).unwrap();
    assert!(eigenpair(
        &tiny,
        Pencil {
            diagonal: &diagonal,
            off_diagonal: &off,
            projectors: &u,
            coupling: &[0.03],
            metric: &[0.2],
            rank: 1
        },
        0,
        false
    )
    .is_err());
    assert_eq!(tiny.memory_used(), 0);
}

#[test]
fn zero_rank_and_zero_coupling_include_exact_tridiagonal_poles() {
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    let diagonal = [2., 2., 2.];
    let off = [-1., -1.];
    for rank in [0, 1] {
        let u = vec![1.; 3 * rank];
        let zero = vec![0.; rank * rank];
        for (state, expected) in [2. - 2f64.sqrt(), 2., 2. + 2f64.sqrt()]
            .into_iter()
            .enumerate()
        {
            let result = eigenpair(
                &ctx,
                Pencil {
                    diagonal: &diagonal,
                    off_diagonal: &off,
                    projectors: &u,
                    coupling: &zero,
                    metric: &zero,
                    rank,
                },
                state,
                true,
            )
            .unwrap();
            assert!((result.energy - expected).abs() < 1e-11);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_separable_correction_can_cross_an_exact_tridiagonal_pole() {
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    for state in 0..3 {
        let result = eigenpair(
            &ctx,
            Pencil {
                diagonal: &[2., 2., 2.],
                off_diagonal: &[-1., -1.],
                projectors: &[0., 1., 0.],
                coupling: &[1.],
                metric: &[0.],
                rank: 1,
            },
            state,
            true,
        )
        .unwrap();
        assert!((result.energy - [1., 2., 4.][state]).abs() < 1e-10);
    }
}

#[test]
fn disconnected_and_degenerate_tridiagonal_nodes_keep_their_inertia() {
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    for state in 0..4 {
        let result = eigenpair(
            &ctx,
            Pencil {
                diagonal: &[2.; 4],
                off_diagonal: &[0.; 3],
                projectors: &[1.; 4],
                coupling: &[1.],
                metric: &[0.],
                rank: 1,
            },
            state,
            true,
        )
        .unwrap();
        assert!(
            (result.energy - if state == 3 { 6. } else { 2. }).abs() < 1e-10,
            "state={state} energy={}",
            result.energy
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}
