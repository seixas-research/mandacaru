use mandacaru_core::{
    math::storage::MatrixSource,
    periodic::{
        wannier::{
            model,
            sampling::{self, Augmentation, Frame, Trial},
        },
        Crystal, Options,
    },
    quantum::C64,
    Context,
};
use nalgebra::{DMatrix, Matrix3, Vector3};
use std::f64::consts::PI;

#[test]
fn neighbor_shells_reproduce_the_metric_of_skew_meshes_and_wrap_indices() {
    let ctx = Context::new(Some(1), 8 << 20).unwrap();
    for cell in [
        Matrix3::identity() * 8.,
        Matrix3::new(0., 5., 5., 5., 0., 5., 5., 5., 0.),
        Matrix3::new(7., 1., 0.2, 0., 9., 1.1, 0.3, 0., 5.),
    ] {
        let size = [4, 3, 2];
        let shells = sampling::neighbor_shells(cell, size).unwrap();
        let mut metric = Matrix3::zeros();
        let b = 2. * PI * cell.try_inverse().unwrap().transpose();
        for ((v, w), step) in shells
            .vectors
            .iter()
            .zip(&shells.weights)
            .zip(&shells.steps)
        {
            let vector = Vector3::from(*v);
            metric += *w * vector * vector.transpose();
            assert!(
                (vector
                    - b * Vector3::from(std::array::from_fn::<_, 3, _>(
                        |a| step[a] as f64 / size[a] as f64
                    )))
                .norm()
                    < 1e-13
            );
        }
        assert!((metric - Matrix3::identity()).norm() < 1e-8);
        let neighbors = sampling::neighbor_indices(&ctx, size, &shells.steps).unwrap();
        for k in 0..24 {
            let origin = sampling::mesh_point(size, k).unwrap();
            for (d, step) in shells.steps.iter().enumerate() {
                let other =
                    sampling::mesh_point(size, neighbors.as_slice()[k * shells.steps.len() + d])
                        .unwrap();
                for a in 0..3 {
                    let delta = other[a] - origin[a] - step[a] as f64 / size[a] as f64;
                    assert!((delta - delta.round()).abs() < 1e-14);
                }
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn archived_bloch_links_match_independent_ao_contractions_and_boundary_sewing() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test/migration/phase5/si_crystal.json");
        let c = Crystal::load(&ctx, path.to_str().unwrap()).unwrap();
        let state = c.solve(&Options::default()).unwrap();
        let bands = [0, 2, 4, 6];
        let left = Frame::from_point(&c, &state, &c.points[0], 0, &bands).unwrap();
        let right = Frame::from_point(&c, &state, &c.points[1], 0, &bands).unwrap();
        let source: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let augmentation = Augmentation {
            centers: vec![[0.; 3], [2.565303214179483; 3]],
            labels: source["channel_labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| std::array::from_fn(|i| a[i].as_i64().unwrap() as i32))
                .collect(),
        };
        let b: [f64; 3] = std::array::from_fn(|a| c.points[1].k[a] - c.points[0].k[a]);
        let native = sampling::link(&c, &left, &right, b, &augmentation).unwrap();
        let n = c.grid.index.points();
        let first = c.points[0].psi.read(&ctx).unwrap();
        let second = c.points[1].psi.read(&ctx).unwrap();
        let mut ao = DMatrix::<C64>::zeros(c.m, c.m);
        for i in 0..c.m {
            for j in 0..c.m {
                ao[(i, j)] = (0..n)
                    .map(|p| {
                        let r = c.grid.point(p);
                        first[i * n + p].conj()
                            * second[j * n + p]
                            * C64::from_polar(
                                c.grid.index.dv(),
                                -(0..3).map(|a| r[a] * b[a]).sum::<f64>(),
                            )
                    })
                    .sum();
            }
        }
        ao += &c.points[0].projections
            * augmentation.operator(&c, b).unwrap()
            * c.points[1].projections.adjoint();
        let expected = left.coefficients.adjoint() * ao * &right.coefficients;
        assert!((&native - expected).norm() < 1e-12);
        let zero = sampling::link(&c, &left, &left, [0.; 3], &augmentation).unwrap();
        assert!((zero - DMatrix::identity(bands.len(), bands.len())).norm() < 1e-10);
        let reciprocal = 2.
            * PI
            * sampling::lattice(&c.grid)
                .try_inverse()
                .unwrap()
                .transpose();
        let sewn =
            sampling::link(&c, &left, &left, reciprocal.column(0).into(), &augmentation).unwrap();
        assert!((sewn - DMatrix::identity(bands.len(), bands.len())).norm() > 0.1);
        let vectors = state.coefficients[0].matrix(&ctx).unwrap();
        for (i, band) in bands.iter().enumerate() {
            assert!((left.energies.as_slice()[i] - state.eigenvalues[0][*band]).abs() < 1e-10);
            assert!(
                (left.coefficients.column(i) * left.coefficients.column(i).adjoint()
                    - vectors.column(*band) * vectors.column(*band).adjoint())
                .norm()
                    < 1e-9
            );
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn gaussian_trial_radials_are_normalized_and_p_harmonics_point_along_their_axes() {
    for l in 0..5 {
        let trial = Trial {
            center: [0.; 3],
            angular: vec![(l, 0, 1.)],
            width: 0.7,
            spin: None,
            label: format!("l={l}"),
        };
        trial.validate().unwrap();
        let step = 0.0005;
        // At the polar axis |Y_l0|^2=(2l+1)/(4pi), recovering the radial
        // normalization independently of the angular implementation.
        let radial_norm = (0..8400)
            .map(|i| {
                let r = (i as f64 + 0.5) * step;
                trial.value([0., 0., r]).powi(2) * r * r * step * 4. * PI / (2 * l + 1) as f64
            })
            .sum::<f64>();
        assert!((radial_norm - 1.).abs() < 2e-10, "l={l}: {radial_norm}");
    }
    let px = Trial {
        center: [0.; 3],
        angular: vec![(1, 1, 1.)],
        width: 1.,
        spin: None,
        label: "px".into(),
    };
    assert!(px.value([1., 0., 0.]) > 0.);
    assert_eq!(px.value([0., 1., 0.]), 0.);
    assert_eq!(px.value([-1., 0., 0.]), -px.value([1., 0., 0.]));
}

#[test]
fn fourier_hoppings_recover_complex_bands_and_remain_hermitian_off_mesh() {
    let ctx = Context::new(Some(1), 8 << 20).unwrap();
    {
        let size = [4, 2, 2];
        let h = |k: [f64; 3]| {
            let phase = 2. * PI * k[0];
            let off = C64::new(1., 0.) + C64::new(0., 0.2) * C64::from_polar(1., -phase);
            DMatrix::from_row_slice(
                2,
                2,
                &[
                    C64::new(2. * phase.cos(), 0.),
                    off,
                    off.conj(),
                    C64::new(-phase.cos(), 0.),
                ],
            )
        };
        let values: Vec<_> = (0..16)
            .flat_map(|k| {
                let matrix = h(sampling::mesh_point(size, k).unwrap());
                (0..4).map(move |a| matrix[(a / 2, a % 2)])
            })
            .collect();
        let blocks = model::fourier_blocks(&ctx, size, 2, &values).unwrap();
        let cell = Matrix3::new(6., 0.4, 0., 0., 7., 0.5, 0.1, 0., 8.);
        let hopping = model::nearest_hoppings(
            &ctx,
            cell,
            size,
            &[[0.; 3], [0.2, 0., 0.]],
            blocks.as_slice(),
        )
        .unwrap();
        let kpoints: Vec<_> = (0..16)
            .map(|k| sampling::mesh_point(size, k).unwrap())
            .chain([[0.123, 0.293, 0.017], [0.481, 0.1, 0.32]])
            .collect();
        let interpolated = hopping.interpolate(&ctx, &kpoints).unwrap();
        for (k, point) in kpoints.iter().enumerate() {
            let (e, _) = mandacaru_core::math::linalg::eigen(h(*point)).unwrap();
            for i in 0..2 {
                assert!((interpolated.as_slice()[2 * k + i] - e[i]).abs() < 1e-12);
            }
        }
        assert!(model::polar(&DMatrix::from_element(3, 2, C64::new(1., 0.))).is_err());
    }
    assert_eq!(ctx.memory_used(), 0);
}
