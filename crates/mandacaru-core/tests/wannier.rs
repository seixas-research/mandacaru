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

#[test]
fn math_boys_and_pipek_mezey_kernels_localize_synthetic_operators() {
    use mandacaru_core::math::localization::{boys, pipek_mezey};
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let rot0 = DMatrix::from_row_slice(2, 2, &[s, -s, s, s]);
    let mut r_ops = [
        &rot0.transpose() * DMatrix::from_diagonal(&DVector::from_vec(vec![0.0, 3.0])) * &rot0,
        DMatrix::zeros(2, 2),
        DMatrix::zeros(2, 2),
    ];
    let b = boys(&mut r_ops, 200, 1e-12).unwrap();
    assert!(b.converged);
    assert!((b.objective - 9.0).abs() < 1e-12);
    assert!(r_ops[0][(0, 1)].abs() < 1e-10);

    for exponent in [2usize, 4] {
        let mut pops = [
            &rot0.transpose() * DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 0.0])) * &rot0,
            &rot0.transpose() * DMatrix::from_diagonal(&DVector::from_vec(vec![0.0, 1.0])) * &rot0,
        ];
        let pm = pipek_mezey(&mut pops, exponent, 200, 1e-12).unwrap();
        assert!(pm.converged, "exp={exponent}");
        assert!(
            (pm.objective - 2.0).abs() < 1e-10,
            "exp={exponent}: {}",
            pm.objective
        );
    }
}

#[test]
fn periodic_pipek_mezey_gradient_and_atomic_limit_hold_for_mulliken_and_lowdin() {
    use mandacaru_core::periodic::wannier::{
        populations::{AtomPartition, PopulationMesh, PopulationScheme},
        Localization,
    };
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let (mesh, ideal, centers) = atomic_mesh(&ctx);
        let size = [mesh.points, 1, 1];
        // Two orthogonalized Bloch AO states with non-identity overlap S(k) (Mulliken != I, Lowdin symmetric).
        let partition = AtomPartition::from_atom_rows(2, vec![vec![0], vec![1]]).unwrap();
        let s_raw = Matrix::from_row_slice(
            2,
            2,
            &[
                C64::new(1.0, 0.),
                C64::new(0.2, 0.05),
                C64::new(0.2, -0.05),
                C64::new(1.0, 0.),
            ],
        );
        let (s_vals, s_vecs) = mandacaru_core::math::linalg::eigen(s_raw.clone()).unwrap();
        let s_inv_sqrt = &s_vecs
            * Matrix::from_diagonal(&DVector::from_iterator(
                2,
                s_vals.iter().map(|v| C64::new(1. / v.sqrt(), 0.)),
            ))
            * s_vecs.adjoint();
        // Band coefficients C_k = S^{-1/2} * ideal_k satisfy C_k^H S C_k = I.
        let c_mats: Vec<Matrix> = ideal.iter().map(|u| &s_inv_sqrt * u).collect();
        let c_refs: Vec<&Matrix> = c_mats.iter().collect();
        let metrics: Vec<(&Matrix, &Matrix)> = vec![(&s_raw, &s_inv_sqrt); mesh.points];
        let mut rng = berry_models_support::Rng(0x55aa_1234_9876_fedcu64);
        let start_gauge: Vec<Matrix> = (0..mesh.points).map(|_| rng.unitary(2)).collect();
        let dir: Vec<Matrix> = (0..mesh.points).map(|k| generator(k + 1, 0.25)).collect();

        for (scheme, exponent) in [
            (PopulationScheme::Mulliken, 2),
            (PopulationScheme::Lowdin, 2),
            (PopulationScheme::Lowdin, 4),
        ] {
            let pop_mesh = PopulationMesh::new(
                size,
                1,
                &c_refs,
                &metrics,
                partition.clone(),
                scheme,
                exponent,
            )
            .unwrap();
            // 1. Finite-difference derivative check against analytic Riemannian gradient.
            let h = 1e-6;
            let val_at = |step: f64| -> f64 {
                let moved: Vec<Matrix> = start_gauge
                    .iter()
                    .zip(&dir)
                    .map(|(u, d)| u * wannier::unitary_step(&(d * C64::new(step, 0.))).unwrap())
                    .collect();
                pop_mesh.evaluate(&ctx, &moved).unwrap().0
            };
            let fd = (val_at(h) - val_at(-h)) / (2. * h);
            let (_, grad) = pop_mesh.evaluate(&ctx, &start_gauge).unwrap();
            let analytic: f64 = grad
                .iter()
                .zip(&dir)
                .map(|(g, d)| (g.adjoint() * d).trace().re)
                .sum();
            assert!(
                (fd - analytic).abs() < 1e-7 * analytic.abs().max(1.),
                "{scheme:?} p={exponent}: fd={fd} analytic={analytic}"
            );

            // 2. Conjugate gradient recovers localized gauge cleanly and monotonically.
            let result = wannier::localize_with_populations(
                &ctx,
                &mesh,
                Some(&pop_mesh),
                &start_gauge,
                Some(&centers),
                &Options {
                    tolerance: 1e-12,
                    gradient_tolerance: 1e-7,
                    localization: Localization::Populations { scheme, exponent },
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(result.converged, "{scheme:?} p={exponent}");
            let vals: Vec<f64> = result.history.iter().map(|h| h[0]).collect();
            assert!(vals.windows(2).all(|w| w[1] >= w[0] - 1e-12), "{vals:?}");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn interpolation_and_w4_anomalous_hall_match_qwz_and_haldane_models() {
    use mandacaru_core::periodic::{
        berry::hall,
        spectrum::Smearing,
        wannier::interpolation::{anomalous_hall, Model},
    };
    use nalgebra::Matrix3;
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        // Build Qi-Wu-Zhang directly in R-space: H(0) = m sz, H(+-1,0,0) = (sz +- i sx)/2, H(0,+-1,0) = (sz +- i sy)/2.
        let lattice = Matrix3::identity() * 4.0;
        let vectors = vec![[-1, 0, 0], [0, -1, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]];
        let pauli = [
            berry_models_support::sigma([1., 0., 0.]),
            berry_models_support::sigma([0., 1., 0.]),
            berry_models_support::sigma([0., 0., 1.]),
        ];
        for (m, expected_chern) in [
            (-3.0, 0i64),
            (-2.5, 0),
            (-1.5, 1),
            (-0.5, 1),
            (0.5, -1),
            (1.5, -1),
            (2.5, 0),
            (3.0, 0),
        ] {
            let h_r = vec![
                (&pauli[2] - &pauli[0] * C64::i()) * C64::new(0.5, 0.),
                (&pauli[2] - &pauli[1] * C64::i()) * C64::new(0.5, 0.),
                &pauli[2] * C64::new(m, 0.),
                (&pauli[2] + &pauli[1] * C64::i()) * C64::new(0.5, 0.),
                (&pauli[2] + &pauli[0] * C64::i()) * C64::new(0.5, 0.),
            ];
            let zero = vec![Matrix::zeros(2, 2); vectors.len()];
            // Add non-trivial diagonal sublattice center offsets to check basis-position invariance.
            let mut az = zero.clone();
            az[2][(0, 0)] = C64::new(0.3, 0.);
            az[2][(1, 1)] = C64::new(1.7, 0.);
            let model = Model::from_tight_binding(
                lattice,
                vectors.clone(),
                h_r,
                [zero.clone(), zero, az],
                0.,
            )
            .unwrap();
            let qwz_hall = anomalous_hall(
                &ctx,
                &[&model],
                [48, 48, 2],
                1.0,
                0.0,
                Smearing::FermiDirac,
                0.0,
                1.0,
                true,
            )
            .unwrap();
            assert!(!qwz_hall.metallic, "m={m}");
            assert_eq!(qwz_hall.total, [0, 0, expected_chern], "m={m}");
            let (_, sigma_expected) = hall::conductivity(&model.reciprocal, [0, 0, expected_chern]);
            assert!(
                (qwz_hall.sigma[0][1] - sigma_expected[0][1]).abs() < 1e-4,
                "m={m}: {} vs {}",
                qwz_hall.sigma[0][1],
                sigma_expected[0][1]
            );
        }

        // Haldane honeycomb model with t1 = 1, t2 = 0.25, phi = pi/2, M = 0.3 (topological C = +-1) vs M = 2.0 (trivial C = 0).
        let a1 = [1.0, 0.0, 0.0];
        let a2 = [0.5, 3f64.sqrt() * 0.5, 0.0];
        let hex_lat = Matrix3::from_columns(&[
            nalgebra::Vector3::from(a1) * 4.0,
            nalgebra::Vector3::from(a2) * 4.0,
            nalgebra::Vector3::new(0., 0., 8.0),
        ]);
        let h_vectors = vec![
            [-1, 0, 0],
            [-1, 1, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 1, 0],
            [1, -1, 0],
            [1, 0, 0],
        ];
        let t2_val = 0.25;
        let phase = C64::new(0., 1.); // i t2 for +phi = pi/2
        for (mass, expected) in [(0.3, 1i64), (2.0, 0i64)] {
            let mut hh = vec![Matrix::zeros(2, 2); h_vectors.len()];
            // H(R)_{AB} with basis B at (a1+a2)/3 inside cell: NN hoppings A(0)->B(0), B(-a1), B(-a2):
            // B(0) in R=0, A in cell 0 hops to B in R=(0,0), (-1,0), (0,-1).
            hh[3][(0, 0)] = C64::new(mass, 0.);
            hh[3][(1, 1)] = C64::new(-mass, 0.);
            hh[3][(0, 1)] = C64::new(1., 0.);
            hh[0][(0, 1)] = C64::new(1., 0.);
            hh[2][(0, 1)] = C64::new(1., 0.);
            hh[3][(1, 0)] = C64::new(1., 0.);
            hh[6][(1, 0)] = C64::new(1., 0.);
            hh[4][(1, 0)] = C64::new(1., 0.);
            // NNN clockwise/counterclockwise: A hops +-R: (+1,0), (-1,+1), (0,-1) get -t2 i; conjugates get +t2 i. B gets opposite sign.
            for (pos_r, neg_r) in [(6, 0), (1, 5), (2, 4)] {
                let d_a = phase * (-t2_val);
                hh[pos_r][(0, 0)] = d_a;
                hh[neg_r][(0, 0)] = d_a.conj();
                hh[pos_r][(1, 1)] = d_a.conj();
                hh[neg_r][(1, 1)] = d_a;
            }
            let z = vec![Matrix::zeros(2, 2); h_vectors.len()];
            let mut ax = z.clone();
            let mut ay = z.clone();
            // Position of B sublattice relative to A inside unit cell: (hex_lat.col(0) + hex_lat.col(1)) / 3.
            let tau_b = (hex_lat.column(0) + hex_lat.column(1)) / 3.0;
            ax[3][(1, 1)] = C64::new(tau_b[0], 0.);
            ay[3][(1, 1)] = C64::new(tau_b[1], 0.);
            let haldane =
                Model::from_tight_binding(hex_lat, h_vectors.clone(), hh, [ax, ay, z], 0.).unwrap();
            let h_ahc = anomalous_hall(
                &ctx,
                &[&haldane],
                [18, 18, 2],
                1.0,
                0.0,
                Smearing::FermiDirac,
                0.0,
                1.0,
                false,
            )
            .unwrap();
            assert_eq!(
                h_ahc.total[2].abs(),
                expected,
                "mass={mass}: {:?}",
                h_ahc.numbers
            );
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}
