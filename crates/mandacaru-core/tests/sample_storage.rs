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
    for precision in [Precision::Float64, Precision::Float32] {
        {
            let samples = ComplexSamples::archived_copy(&ctx, &values, precision).unwrap();
            assert_eq!(samples.precision(), precision);
            assert_eq!(ctx.memory_used(), 512);
            let read = samples.read(&ctx).unwrap();
            for (a, b) in read.iter().zip(&values) {
                let expected = if precision == Precision::Float64 {
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
        ComplexSamples::archived_copy(&ctx, &[C64::new(f64::NAN, 0.)], Precision::Float64).is_err()
    );
    assert!(
        ComplexSamples::archived_copy(&ctx, &[C64::new(1e100, 0.)], Precision::Float32).is_err()
    );
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn complex_samples_compact_both_components_and_account_for_accumulation_views() {
    use mandacaru_core::{math::storage::ComplexSamples, quantum::C64};
    let ctx = Context::new(Some(1), 256).unwrap();
    {
        let values = [C64::new(0.1, -0.7); 8];
        let samples = ComplexSamples::copy(&ctx, &values, Precision::Float32).unwrap();
        assert_eq!(ctx.memory_used(), 64);
        assert_eq!(samples.precision(), Precision::Float32);
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
    assert!(ComplexSamples::copy(&ctx, &[C64::new(1e100, 0.)], Precision::Float32).is_err());
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn real_stacks_round_once_and_block_readers_charge_only_their_block() {
    use mandacaru_core::math::storage::SampleBlocks;
    let context = Context::new(Some(1), 128).unwrap();
    let values = [0.1, -0.7, 3., 1e-20, 5., 6., 7., 8.];
    {
        let samples = RealSamples::from_slice(&context, Precision::Float32, &values).unwrap();
        assert_eq!(samples.precision(), Precision::Float32);
        assert_eq!(context.memory_used(), 32);
        assert_eq!(samples.value(0), f64::from(0.1f32));
        assert_eq!(samples.value(1), f64::from(-0.7f32));
        // Widening one range needs no allocation at all.
        let mut out = [0.; 3];
        samples.widen_range(1, &mut out).unwrap();
        assert_eq!(out, [f64::from(-0.7f32), 3., f64::from(1e-20f32)]);
        assert_eq!(context.memory_used(), 32);
        assert!(samples.widen_range(7, &mut out).is_err());
        {
            // A block reader holds `rows x block` float64 values and no more.
            let mut blocks = SampleBlocks::new(&context, &samples, 2, 4).unwrap();
            assert_eq!(context.memory_used(), 32 + 2 * 4 * 8);
            assert_eq!(blocks.load(2).unwrap(), 2);
            assert_eq!(blocks.row(0), &[3., f64::from(1e-20f32)]);
            assert_eq!(blocks.row(1), &[7., 8.]);
        }
        assert_eq!(context.memory_used(), 32);
        {
            let guard = context.reserve(33).unwrap();
            assert!(matches!(
                SampleBlocks::new(&context, &samples, 2, 4),
                Err(Error::Budget { .. })
            ));
            drop(guard);
        }
        assert!(SampleBlocks::new(&context, &samples, 3, 4).is_err());
    }
    assert_eq!(context.memory_used(), 0);
    // A float64 stack is the identity; non-finite values and values outside
    // the float32 range are refused, in the range check of the stack itself.
    let exact = RealSamples::from_slice(&context, Precision::Float64, &values).unwrap();
    assert_eq!(exact.value(0), 0.1);
    drop(exact);
    assert!(RealSamples::from_slice(&context, Precision::Float32, &[1e100]).is_err());
    assert!(RealSamples::from_slice(&context, Precision::Float32, &[f64::NAN]).is_err());
    assert!(RealSamples::from_slice(&context, Precision::Float64, &[f64::INFINITY]).is_err());
    assert_eq!(context.memory_used(), 0);
    assert!(Precision::parse("quadruple").is_err());
}

fn hydrogen_pair() -> (Geometry, SpatialGrid) {
    let geometry = Geometry::new(
        vec![1, 1],
        vec![[4., 4., 3.3], [4., 4., 4.7]],
        [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
        [false; 3],
        LengthUnit::Bohr,
    )
    .unwrap();
    let grid = SpatialGrid::from_geometry(&geometry, [0.5; 3], false, false).unwrap();
    (geometry, grid)
}

#[test]
fn float32_construction_never_holds_a_float64_stack_and_follows_geometry() {
    let (geometry, grid) = hydrogen_pair();
    let points = grid.index.points();
    let spec = r#"{"name":"HAO","virtual_orbitals":3}"#;
    let build = |context: &Context, spec: &str, precision| {
        MolecularIntegrals::new(
            context,
            geometry.clone(),
            grid.clone(),
            spec,
            "spectral",
            None,
            precision,
        )
        .unwrap()
    };
    let construct = |context: &Context, precision| build(context, spec, precision);
    // Context peak of the build alone, in each precision.
    let (m, peak64, peak32) = {
        let context = Context::new(Some(1), 256 << 20).unwrap();
        let float64 = construct(&context, Precision::Float64);
        let m = float64.m();
        let peak64 = context.memory_peak();
        drop(float64);
        let context32 = Context::new(Some(1), 256 << 20).unwrap();
        let float32 = construct(&context32, Precision::Float32);
        assert_eq!(float32.precision(), Precision::Float32);
        let peak32 = context32.memory_peak();
        drop(float32);
        assert_eq!(context32.memory_used(), 0);
        (m, peak64, peak32)
    };
    let stack = m * points * 8;
    // Two float64 stacks plus scratch for float64; the float32 build stores
    // two float32 stacks (one float64 stack in all) plus the same scratch.
    assert!(peak64 >= 2 * stack, "{peak64} < 2 x {stack}");
    assert!(
        peak32 < peak64 && (peak32 as f64) < 0.75 * peak64 as f64,
        "float32 build peak {peak32} vs float64 {peak64}"
    );
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let plain = r#"{"name":"HAO"}"#;
        let float64 = build(&context, plain, Precision::Float64);
        let float32 = build(&context, plain, Precision::Float32);
        // The float32 build contracts its matrices from the rounded stored
        // rows and orthonormalizes them against their own overlap, so the
        // orthonormal frame is rotated by about 1e-7 and the one-body matrix
        // changes with it. What is frame independent, its spectrum, agrees.
        let spectrum = |integrals: &MolecularIntegrals| {
            let m = integrals.m();
            let matrix = nalgebra::DMatrix::from_row_slice(m, m, integrals.one.as_slice());
            let mut values: Vec<f64> = matrix
                .symmetric_eigen()
                .eigenvalues
                .iter()
                .copied()
                .collect();
            values.sort_by(f64::total_cmp);
            values
        };
        let worst = spectrum(&float64)
            .iter()
            .zip(spectrum(&float32))
            .map(|(a, b)| (a - b).abs() / a.abs().max(1.))
            .fold(0f64, f64::max);
        assert!(
            worst < 1e-7,
            "one-body spectrum relative difference {worst}"
        );
        // The overlap, potential and kinetic matrices come from exact orbital
        // values (re-evaluated for a float32 stack), not from rounded rows.
        for (a, b) in [
            (&float64.overlap, &float32.overlap),
            (&float64.potential, &float32.potential),
            (&float64.kinetic, &float32.kinetic),
        ] {
            let worst = a
                .as_slice()
                .iter()
                .zip(b.as_slice())
                .map(|(a, b)| (a - b).abs())
                .fold(0f64, f64::max);
            assert!(worst < 1e-9, "matrix element difference {worst}");
        }
        let reference = float64
            .mean_field("rhf", [1, 1], &Default::default())
            .unwrap();
        let state = float32
            .mean_field("rhf", [1, 1], &Default::default())
            .unwrap();
        assert!(state.converged);
        assert!((reference.electronic_energy - state.electronic_energy).abs() < 1e-6);
        let shifted = float32.at_geometry(geometry).unwrap();
        assert_eq!(shifted.samples.precision(), Precision::Float32);
        assert_eq!(shifted.orthonormal_samples.precision(), Precision::Float32);
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn precision_names_are_exactly_float32_and_float64() {
    assert_eq!(Precision::parse("float32").unwrap(), Precision::Float32);
    assert_eq!(Precision::parse("float64").unwrap(), Precision::Float64);
    assert_eq!(Precision::Float32.name(), "float32");
    assert_eq!(Precision::Float64.name(), "float64");
    assert_eq!(Precision::default(), Precision::Float64);
    // The retired words, other spellings and the empty name are refused.
    for name in ["single", "double", "Float32", "FLOAT64", " float64", ""] {
        let error = Precision::parse(name).unwrap_err().to_string();
        assert!(error.contains("'float32' or 'float64'"), "{name}: {error}");
    }
}
