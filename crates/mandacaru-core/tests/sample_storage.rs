use mandacaru_core::{
    math::{
        storage::{Precision, RealSamples},
        SpatialGrid,
    },
    molecular::MolecularIntegrals,
    Context, Error, Geometry, LengthUnit,
};

#[test]
fn archived_bloch_samples_preserve_precision_and_release_each_accumulation_view() {
    use mandacaru_core::{math::storage::ComplexSamples, quantum::C64};
    let ctx = Context::new(Some(1), 64 << 10).unwrap();
    let values: Vec<_> = (0..2049)
        .map(|i| C64::new(i as f64 / 7., i as f64 / 13.))
        .collect();
    for precision in [Precision::Double, Precision::Single] {
        {
            let samples = ComplexSamples::archived_copy(&ctx, &values, precision).unwrap();
            assert_eq!(samples.precision(), precision);
            assert_eq!(ctx.memory_used(), 512);
            let read = samples.read(&ctx).unwrap();
            for (a, b) in read.iter().zip(&values) {
                let expected = if precision == Precision::Double {
                    *b
                } else {
                    C64::new(f64::from(b.re as f32), f64::from(b.im as f32))
                };
                assert_eq!(*a, expected);
            }
        }
        assert_eq!(ctx.memory_used(), 0);
    }
    assert!(
        ComplexSamples::archived_copy(&ctx, &[C64::new(f64::NAN, 0.)], Precision::Double).is_err()
    );
    assert!(
        ComplexSamples::archived_copy(&ctx, &[C64::new(1e100, 0.)], Precision::Single).is_err()
    );
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn complex_samples_compact_both_components_and_account_for_accumulation_views() {
    use mandacaru_core::{math::storage::ComplexSamples, quantum::C64};
    let ctx = Context::new(Some(1), 256).unwrap();
    {
        let values = [C64::new(0.1, -0.7); 8];
        let samples = ComplexSamples::copy(&ctx, &values, Precision::Single).unwrap();
        assert_eq!(ctx.memory_used(), 64);
        assert_eq!(samples.precision(), Precision::Single);
        assert_eq!(
            samples.read(&ctx).unwrap()[0],
            C64::new(f64::from(0.1f32), f64::from(-0.7f32))
        );
        {
            let guard = ctx.reserve(65).unwrap();
            assert!(samples.read(&ctx).is_err());
            drop(guard);
        }
        {
            let expanded = samples.read(&ctx).unwrap();
            assert_eq!(ctx.memory_used(), 192);
            assert_eq!(expanded[0], C64::new(f64::from(0.1f32), f64::from(-0.7f32)));
        }
        assert_eq!(ctx.memory_used(), 64);
    }
    assert_eq!(ctx.memory_used(), 0);
    assert!(ComplexSamples::copy(&ctx, &[C64::new(1e100, 0.)], Precision::Single).is_err());
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn compact_samples_and_accumulation_views_obey_the_same_memory_budget() {
    let context = Context::new(Some(1), 128).unwrap();
    let mut source = context.allocate::<f64>(8).unwrap();
    source
        .as_mut_slice()
        .copy_from_slice(&[0.1, -0.7, 3., 1e-20, 5., 6., 7., 8.]);
    let samples = RealSamples::from(source)
        .into_precision(&context, Precision::Single)
        .unwrap();
    assert_eq!(context.memory_used(), 32);
    {
        let guard = context.reserve(33).unwrap();
        assert!(matches!(samples.read(&context), Err(Error::Budget { .. })));
        assert_eq!(context.memory_used(), 65);
        drop(guard);
    }
    {
        let view = samples.read(&context).unwrap();
        assert_eq!(context.memory_used(), 96);
        assert_eq!(view[0], f64::from(0.1f32));
        assert_eq!(view[1], f64::from(-0.7f32));
    }
    assert_eq!(context.memory_used(), 32);
    drop(samples);
    assert_eq!(context.memory_used(), 0);
    let mut overflow = context.allocate::<f64>(1).unwrap();
    overflow.as_mut_slice()[0] = 1e100;
    assert!(RealSamples::from(overflow)
        .into_precision(&context, Precision::Single)
        .is_err());
    assert_eq!(context.memory_used(), 0);
    assert!(Precision::parse("quadruple").is_err());
}

#[test]
fn native_single_precision_keeps_double_matrices_and_follows_geometry() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let geometry = Geometry::new(
            vec![1, 1],
            vec![[4., 4., 3.3], [4., 4., 4.7]],
            [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
            [false; 3],
            LengthUnit::Bohr,
        )
        .unwrap();
        let grid = SpatialGrid::from_geometry(&geometry, [0.5; 3], false, false).unwrap();
        let construct = || {
            MolecularIntegrals::new(
                &context,
                geometry.clone(),
                grid.clone(),
                r#"{"name":"HAO"}"#,
                "spectral",
                None,
            )
            .unwrap()
        };
        let double = construct();
        let single = construct().with_precision(Precision::Single).unwrap();
        assert_eq!(double.one.as_slice(), single.one.as_slice());
        let double = double
            .mean_field("rhf", [1, 1], &Default::default())
            .unwrap();
        let state = single
            .mean_field("rhf", [1, 1], &Default::default())
            .unwrap();
        assert!(state.converged);
        assert!((double.electronic_energy - state.electronic_energy).abs() < 1e-7);
        let shifted = single.at_geometry(geometry).unwrap();
        assert_eq!(shifted.samples.precision(), Precision::Single);
        assert_eq!(shifted.orthonormal_samples.precision(), Precision::Single);
    }
    assert_eq!(context.memory_used(), 0);
}
