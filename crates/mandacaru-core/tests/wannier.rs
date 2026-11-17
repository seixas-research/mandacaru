mod berry_models_support;

use mandacaru_core::{
    math::storage::StoredMatrix,
    periodic::{
        bloch_mesh::MeshIndex,
        wannier::{self, Mesh, Options},
    },
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

/// The subspace links of the lower Qi-Wu-Zhang band on an `n x n x 1` mesh, and the mesh.
fn qwz_links(m: f64, n: usize) -> (MeshIndex, Vec<[Option<Matrix>; 3]>) {
    use berry_models_support::{link, lowest, qwz, sigma};
    let mesh = MeshIndex::new([n, n, 1], [0.; 3]).unwrap();
    let states: Vec<Matrix> = (0..mesh.points())
        .map(|k| {
            let f = mesh.point(k);
            lowest(sigma(qwz(m, 2. * PI * f[0], 2. * PI * f[1])), 1)
        })
        .collect();
    let along = (0..mesh.points())
        .map(|k| {
            [0, 1, 2].map(|j| {
                (mesh.size[j] > 1).then(|| link(&states[k], &states[mesh.advance(k, j, 1)]))
            })
        })
        .collect();
    (mesh, along)
}

#[test]
fn a_chern_band_is_refused_by_name_and_a_trivial_band_passes() {
    // No exponentially localized Wannier functions exist for a band with C != 0: the localization
    // must raise instead of returning functions whose spread grows with the mesh.
    for m in [-1., 1.] {
        let (mesh, along) = qwz_links(m, 12);
        let message = wannier::chern_guard(&mesh, &along).unwrap_err().to_string();
        assert!(message.contains("Chern number"), "{message}");
        assert!(message.contains("normal to G_2"), "{message}");
    }
    for m in [-3., 2.5] {
        let (mesh, along) = qwz_links(m, 12);
        wannier::chern_guard(&mesh, &along).unwrap();
    }
}

#[test]
fn a_saddle_is_escaped_and_an_unfinished_escape_is_not_converged() {
    // Two atomic functions mixed at 45 degrees: a stationary point of the spread (the gradient
    // vanishes by symmetry) that is not a minimum. The saddle check must leave it, and a budget
    // too small for that escape must report converged = false, never the stationary point as a
    // converged minimum.
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let (mesh, ideal, centers) = atomic_mesh(&ctx);
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let mix = Matrix::from_row_slice(
            2,
            2,
            &[
                C64::new(s, 0.),
                C64::new(-s, 0.),
                C64::new(s, 0.),
                C64::new(s, 0.),
            ],
        );
        let saddle: Vec<Matrix> = ideal.iter().map(|u| u * &mix).collect();
        let at = wannier::spread(&ctx, &mesh, &saddle, Some(&centers)).unwrap();
        let g = wannier::gradient(&ctx, &mesh, &saddle, at.centers.as_slice()).unwrap();
        let norm = wannier::gradient_norm(&g.values);
        assert!(at.total() > 1e-3, "the mixed pair is not above the minimum");
        assert!(
            norm < Options::default().gradient_tolerance,
            "precondition: the mixed pair must be stationary (gradient norm {norm:e})"
        );
        let full =
            wannier::localize(&ctx, &mesh, &saddle, Some(&centers), &Options::default()).unwrap();
        assert!(full.converged);
        assert!(
            full.spread.total() < 1e-9,
            "the saddle was not escaped: spread {}",
            full.spread.total()
        );
        let short = wannier::localize(
            &ctx,
            &mesh,
            &saddle,
            Some(&centers),
            &Options {
                max_iterations: 2,
                ..Options::default()
            },
        )
        .unwrap();
        assert!(
            !short.converged,
            "a stationary point whose saddle check ran out of budget was reported converged \
             (spread {})",
            short.spread.total()
        );
        let totals: Vec<f64> = short.history.iter().map(|h| h.iter().sum()).collect();
        assert!(
            totals.windows(2).all(|w| w[1] <= w[0] + 1e-14),
            "{totals:?}"
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}
