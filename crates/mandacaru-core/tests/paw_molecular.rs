use mandacaru_core::{
    basis::Basis,
    math::{
        spline::{simpson, Cubic},
        SpatialGrid,
    },
    molecular::{
        integrals::{Eri, MolecularCorrections, PairCorrection},
        transform, MolecularIntegrals,
    },
    quantum::C64,
    Context, Geometry, LengthUnit,
};
use nalgebra::DMatrix;
use std::sync::Arc;

#[test]
fn nonuniform_splines_reproduce_cubic_values_and_derivatives_and_release_storage() {
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    {
        let x = [0., 0.1, 0.4, 0.9, 1.3, 1.8, 2.];
        let y: Vec<_> = x.iter().map(|x| x * x * x - 2. * x + 3.).collect();
        let spline = Cubic::new(
            &ctx,
            ctx.copy_slice(&x).unwrap(),
            ctx.copy_slice(&y).unwrap(),
        )
        .unwrap();
        for k in -2..=23 {
            let r = k as f64 / 10.;
            let (v, d) = spline.value_slope(r);
            assert!((v - (r * r * r - 2. * r + 3.)).abs() < 1e-11);
            assert!((d - (3. * r * r - 2.)).abs() < 1e-10);
        }
        for n in [5, 6, 7] {
            let y: Vec<_> = x[..n].iter().map(|r| 2. * r * r + 3. * r + 1.).collect();
            let end = x[n - 1];
            let expected = 2. * end.powi(3) / 3. + 1.5 * end * end + end;
            assert!((simpson(&x[..n], &y).unwrap() - expected).abs() < 1e-13);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
    assert!(Cubic::new(
        &ctx,
        ctx.copy_slice(&[0., 1., 1., 2.]).unwrap(),
        ctx.copy_slice(&[0.; 4]).unwrap()
    )
    .is_err());
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn signed_paw_corrections_stay_separate_from_positive_coulomb_cholesky() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    {
        let geometry = Geometry::new(
            vec![1, 1],
            vec![[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]],
            [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
            [false; 3],
            LengthUnit::Angstrom,
        )
        .unwrap();
        let grid = SpatialGrid::from_geometry(&geometry, [1.; 3], false, false).unwrap();
        let basis = Basis::build(&ctx, &geometry, r#"{"name":"HAO"}"#).unwrap();
        let corrections = MolecularCorrections {
            dft: None,
            external: ctx.allocate(grid.index.points()).unwrap(),
            overlap: DMatrix::zeros(2, 2),
            potential: DMatrix::zeros(2, 2),
            pairs: Arc::new(PairCorrection {
                factors: vec![DMatrix::from_row_slice(2, 2, &[1., 0.2, 0.2, -1.])],
                weights: vec![-4.],
                memory: ctx.reserve(128).unwrap(),
            }),
            charges: vec![1., 1.],
            electrons: 2,
        };
        let integrals = MolecularIntegrals::from_basis(
            &ctx,
            geometry,
            grid,
            basis,
            "spectral",
            Some(1e-10),
            Some(corrections),
        )
        .unwrap();
        assert!(!integrals.has_eri());
        let eri = integrals.eri().unwrap();
        assert!(
            matches!(eri.as_ref(),Eri::Augmented{base,..} if matches!(base.as_ref(),Eri::Cholesky{..}))
        );
        assert!(eri.value(0, 0, 0, 0) < 0.); // An indefinite correction is preserved.
        let rotation = DMatrix::from_row_slice(
            2,
            2,
            &[
                C64::new(0.8, 0.),
                C64::new(0., 0.6),
                C64::new(0., 0.6),
                C64::new(0.8, 0.),
            ],
        );
        let transformed = transform::four(&ctx, &eri, [&rotation; 4]).unwrap();
        for p in 0..2 {
            for q in 0..2 {
                for r in 0..2 {
                    for s in 0..2 {
                        let mut expected = C64::default();
                        for a in 0..2 {
                            for b in 0..2 {
                                for c in 0..2 {
                                    for d in 0..2 {
                                        expected += rotation[(a, p)].conj()
                                            * rotation[(b, q)].conj()
                                            * eri.value(a, b, c, d)
                                            * rotation[(c, r)]
                                            * rotation[(d, s)];
                                    }
                                }
                            }
                        }
                        assert!(
                            (transformed.as_slice()[((p * 2 + q) * 2 + r) * 2 + s] - expected)
                                .norm()
                                < 1e-12
                        );
                    }
                }
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}
