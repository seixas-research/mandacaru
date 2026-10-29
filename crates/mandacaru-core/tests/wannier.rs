use mandacaru_core::{
    math::storage::StoredMatrix,
    periodic::wannier::{self, Mesh, Options},
    quantum::C64,
    Context,
};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;
type Matrix = DMatrix<C64>;

fn entangled_mesh(ctx: &Context) -> (Mesh, Vec<Matrix>, wannier::disentangle::Windows) {
    let centers = [[0., 0.13, 0.1], [0.17, 0., 0.], [-0.17, 0., 0.]];
    let vectors = [
        [2. * PI, 0., 0.],
        [-2. * PI, 0., 0.],
        [0., 2. * PI, 0.],
        [0., -2. * PI, 0.],
        [0., 0., 2. * PI],
        [0., 0., -2. * PI],
    ];
    let links = vectors
        .iter()
        .map(|b| {
            StoredMatrix::new(
                ctx,
                Matrix::from_diagonal(&DVector::from_iterator(
                    3,
                    centers
                        .iter()
                        .map(|r| C64::from_polar(1., -(0..3).map(|a| b[a] * r[a]).sum::<f64>())),
                )),
                false,
            )
            .unwrap()
        })
        .collect();
    let trials = Matrix::from_row_slice(
        3,
        2,
        &[
            C64::new(1., 0.),
            C64::default(),
            C64::default(),
            C64::new(0.8, 0.),
            C64::default(),
            C64::new(0.6, 0.),
        ],
    );
    let mesh = Mesh {
        points: 1,
        bands: 3,
        vectors: ctx.copy_slice(&vectors).unwrap(),
        weights: ctx.copy_slice(&[1. / (8. * PI * PI); 6]).unwrap(),
        neighbors: ctx.copy_slice(&[0; 6]).unwrap(),
        links,
    };
    let windows = wannier::disentangle::Windows {
        inside: ctx.copy_slice(&[1u8; 3]).unwrap(),
        frozen: ctx.copy_slice(&[1u8, 0, 0]).unwrap(),
        functions: 2,
    };
    (mesh, vec![trials], windows)
}

#[test]
fn frozen_window_disentanglement_recovers_an_isolated_atomic_subspace() {
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let (mesh, trials, windows) = entangled_mesh(&ctx);
        let selected = wannier::disentangle::select(
            &ctx,
            &mesh,
            &trials,
            &windows,
            None,
            &wannier::disentangle::Options {
                tolerance: 1e-13,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(selected.converged);
        assert!(selected.history[0] > 1e-3);
        assert!(selected.history.last().unwrap().abs() < 1e-11);
        let gauge = &selected.gauge[0];
        assert_eq!(gauge[(0, 0)], C64::new(1., 0.));
        assert!((gauge[(1, 1)].norm_sqr() - 1.).abs() < 1e-9);
        assert!(gauge[(2, 1)].norm_sqr() < 1e-9);
        assert!((gauge.adjoint() * gauge - Matrix::identity(2, 2)).norm() < 1e-13);
        let spread = wannier::spread(&ctx, &mesh, &selected.gauge, None).unwrap();
        assert!((spread.invariant - selected.history.last().unwrap()).abs() < 1e-14);
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
fn symmetry_keeps_a_wannier_subspace_on_its_symmetric_stationary_solution() {
    use wannier::disentangle::{select, Options, Symmetry};
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let (mesh, trials, windows) = entangled_mesh(&ctx);
        let swap = Matrix::from_fn(3, 3, |i, j| {
            if (i, j) == (0, 0) || (i, j) == (1, 2) || (i, j) == (2, 1) {
                C64::new(1., 0.)
            } else {
                C64::default()
            }
        });
        let group = Symmetry {
            operations: 2,
            mapping: ctx.copy_slice(&[0, 0]).unwrap(),
            rotations: vec![
                StoredMatrix::new(&ctx, Matrix::identity(3, 3), false).unwrap(),
                StoredMatrix::new(&ctx, swap.clone(), false).unwrap(),
            ],
        };
        let selected = select(
            &ctx,
            &mesh,
            &trials,
            &windows,
            Some(&group),
            &Options::default(),
        )
        .unwrap();
        assert!(selected.converged);
        let projector = &selected.gauge[0] * selected.gauge[0].adjoint();
        assert!((&swap * &projector * swap.adjoint() - projector).norm() < 1e-13);
        let expected = (2. * PI * 0.17).sin().powi(2) / (4. * PI * PI);
        assert!((selected.history.last().unwrap() - expected).abs() < 1e-13);
    }
    assert_eq!(ctx.memory_used(), 0);
}

fn generator(k: usize, scale: f64) -> Matrix {
    let z = C64::new(
        scale * (0.4 + k as f64).sin(),
        scale * (0.7 * k as f64 + 0.2).cos(),
    );
    Matrix::from_row_slice(
        2,
        2,
        &[
            C64::new(0., 0.3 * scale),
            z,
            -z.conj(),
            C64::new(0., -0.2 * scale),
        ],
    )
}
fn atomic_mesh(ctx: &Context) -> (Mesh, Vec<Matrix>, [[f64; 3]; 2]) {
    let points = 5;
    let centers = [[0.17, 0.12, 0.23], [0.38, 0.27, 0.34]];
    let vectors = [
        [2. * PI / points as f64, 0., 0.],
        [-2. * PI / points as f64, 0., 0.],
        [0., 2. * PI, 0.],
        [0., -2. * PI, 0.],
        [0., 0., 2. * PI],
        [0., 0., -2. * PI],
    ];
    let weights: Vec<_> = vectors
        .iter()
        .map(|v| 1. / (2. * v.iter().map(|v| v * v).sum::<f64>()))
        .collect();
    let band_gauge: Vec<_> = (0..points)
        .map(|k| wannier::unitary_step(&generator(k, 0.3)).unwrap())
        .collect();
    let mut neighbors = vec![];
    let mut links = vec![];
    for k in 0..points {
        for (b, vector) in vectors.iter().enumerate() {
            let neighbor = match b {
                0 => (k + 1) % points,
                1 => (k + points - 1) % points,
                _ => k,
            };
            neighbors.push(neighbor);
            let phase = Matrix::from_diagonal(&DVector::from_iterator(
                2,
                centers
                    .iter()
                    .map(|c| C64::from_polar(1., -(0..3).map(|a| vector[a] * c[a]).sum::<f64>())),
            ));
            let link = band_gauge[k].adjoint() * phase * &band_gauge[neighbor];
            links.push(StoredMatrix::new(ctx, link, false).unwrap());
        }
    }
    (
        Mesh {
            points,
            bands: 2,
            vectors: ctx.copy_slice(&vectors).unwrap(),
            weights: ctx.copy_slice(&weights).unwrap(),
            neighbors: ctx.copy_slice(&neighbors).unwrap(),
            links,
        },
        band_gauge.iter().map(|u| u.adjoint()).collect(),
        centers,
    )
}
#[test]
fn atomic_limit_has_exact_centers_and_zero_spread_in_any_band_gauge() {
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let (mesh, gauge, centers) = atomic_mesh(&ctx);
        mesh.validate(&ctx).unwrap();
        let spread = wannier::spread(&ctx, &mesh, &gauge, Some(&centers)).unwrap();
        assert!(spread.total().abs() < 1e-13);
        for (a, b) in spread.centers.as_slice().iter().zip(centers) {
            for axis in 0..3 {
                assert!((a[axis] - b[axis]).abs() < 1e-13);
            }
        }
        for (second, center) in spread.second_moments.as_slice().iter().zip(centers) {
            assert!((second - center.iter().map(|r| r * r).sum::<f64>()).abs() < 1e-13);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
fn wannier_gauge_gradient_matches_independent_unitary_displacements() {
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let (mesh, ideal, centers) = atomic_mesh(&ctx);
        let gauge: Vec<_> = ideal
            .iter()
            .enumerate()
            .map(|(k, u)| u * wannier::unitary_step(&generator(k + 1, 0.19)).unwrap())
            .collect();
        let spread = wannier::spread(&ctx, &mesh, &gauge, Some(&centers)).unwrap();
        let gradient = wannier::gradient(&ctx, &mesh, &gauge, spread.centers.as_slice()).unwrap();
        let direction: Vec<_> = (0..mesh.points).map(|k| generator(k + 4, 0.27)).collect();
        let energy = |h: f64| {
            let shifted: Vec<_> = gauge
                .iter()
                .zip(&direction)
                .map(|(u, d)| u * wannier::unitary_step(&(d * C64::new(h, 0.))).unwrap())
                .collect();
            wannier::spread(&ctx, &mesh, &shifted, Some(spread.centers.as_slice()))
                .unwrap()
                .total()
        };
        let fd = (energy(1e-5) - energy(-1e-5)) / (2e-5);
        let analytic = gradient
            .values
            .iter()
            .zip(&direction)
            .map(|(g, d)| (g.adjoint() * d).trace().re)
            .sum::<f64>();
        assert!((fd - analytic).abs() < 2e-9, "{fd} vs {analytic}");
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
fn localization_recovers_the_atomic_limit_and_preserves_unitarity() {
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let (mesh, ideal, centers) = atomic_mesh(&ctx);
        let gauge: Vec<_> = ideal
            .iter()
            .enumerate()
            .map(|(k, u)| u * wannier::unitary_step(&generator(k + 1, 0.19)).unwrap())
            .collect();
        let result = wannier::localize(
            &ctx,
            &mesh,
            &gauge,
            Some(&centers),
            &Options {
                tolerance: 1e-13,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            result.converged,
            "localization did not converge in {} steps",
            result.history.len()
        );
        assert!(
            result.spread.total().abs() < 1e-9,
            "spread {}",
            result.spread.total()
        );
        for pair in result.history.windows(2) {
            assert!(pair[1].iter().sum::<f64>() <= pair[0].iter().sum::<f64>() + 1e-14);
        }
        for u in &result.gauge {
            assert!((u.adjoint() * u - Matrix::identity(2, 2)).norm() < 1e-12);
        }
        for (a, b) in result.spread.centers.as_slice().iter().zip(centers) {
            for axis in 0..3 {
                assert!((a[axis] - b[axis]).abs() < 1e-7);
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}
