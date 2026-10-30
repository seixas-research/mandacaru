use mandacaru_core::{
    math::projection::{hermitian_density, ComplexProjector, RealProjector},
    quantum::C64,
    Context,
};
use nalgebra::DMatrix;

#[test]
fn parallel_projection_respects_thread_and_memory_limits_and_matches_a_direct_sum() {
    let (m, n) = (33, 8201);
    let samples: Vec<_> = (0..m * n).map(|i| (i as f64 * 0.031).sin()).collect();
    let field: Vec<_> = (0..n).map(|i| (i as f64 * 0.017).cos()).collect();
    let before = std::env::var("OMP_NUM_THREADS").ok();
    for (threads, budget) in [(1, 16 << 20), (4, 16 << 20), (4, 3 << 20)] {
        let ctx = Context::new(Some(threads), budget).unwrap();
        {
            let mut projector = RealProjector::new(&ctx, &samples, m, n).unwrap();
            let result = projector.project(&field).unwrap();
            for a in 0..m {
                for b in 0..m {
                    let expected = (0..n)
                        .map(|k| samples[a * n + k] * field[k] * samples[b * n + k])
                        .sum::<f64>();
                    assert!((result[a + b * m] - expected).abs() < 2e-11);
                }
            }
            assert!(ctx.workers_peak() <= threads);
            if threads == 4 && budget == 16 << 20 {
                assert!(ctx.workers_peak() > 1);
            }
            assert!(ctx.memory_peak() <= budget);
        }
        assert_eq!(ctx.memory_used(), 0);
    }
    assert_eq!(std::env::var("OMP_NUM_THREADS").ok(), before);
}

#[test]
fn float32_tiles_widen_per_tile_and_match_the_projection_of_the_widened_values() {
    let (m, n) = (20, 5001);
    let wide: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.031).sin()).collect();
    let narrow: Vec<f32> = wide.iter().map(|v| *v as f32).collect();
    let rounded: Vec<f64> = narrow.iter().map(|v| f64::from(*v)).collect();
    let field: Vec<_> = (0..n).map(|i| (i as f64 * 0.017).cos()).collect();
    let ctx = Context::new(Some(1), 8 << 20).unwrap();
    {
        let mut reference = RealProjector::new(&ctx, &rounded, m, n).unwrap();
        let expected = reference.project(&field).unwrap().to_vec();
        let live = ctx.memory_used();
        let mut projector = RealProjector::new(&ctx, &narrow, m, n).unwrap();
        // The float32 source adds no whole-stack expansion: only its own tile
        // workspace is charged, and the products are those of the widened values.
        assert!(ctx.memory_used() - live <= live);
        let result = projector.project(&field).unwrap();
        assert_eq!(result, expected.as_slice());
        let mut invalid = narrow.clone();
        invalid[7] = f32::NAN;
        assert!(RealProjector::new(&ctx, &invalid, m, n).is_err());
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn streamed_signed_quadrature_matches_direct_outer_products_across_partial_tiles() {
    use mandacaru_core::math::projection::PointAccumulator;
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    {
        let (m, p, n) = (7, 5, 4099);
        let mut accumulator = PointAccumulator::new(&ctx, m, p).unwrap();
        let mut expected = vec![0.; m * p];
        for k in 0..n {
            let left: Vec<_> = (0..m)
                .map(|i| {
                    if k % 11 == 0 {
                        0.
                    } else {
                        ((i + k) as f64 * 0.13).sin()
                    }
                })
                .collect();
            let right: Vec<_> = (0..p).map(|j| ((2 * j + k) as f64 * 0.07).cos()).collect();
            for i in 0..m {
                for j in 0..p {
                    expected[i + j * m] += left[i] * right[j];
                }
            }
            accumulator.push(&left, &right).unwrap();
        }
        let actual = accumulator.finish().unwrap();
        for (a, b) in actual.as_slice().iter().zip(expected) {
            assert!((a - b).abs() < 2e-12);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
    let small = Context::new(Some(1), 64).unwrap();
    assert!(PointAccumulator::new(&small, 7, 5).is_err());
    assert_eq!(small.memory_used(), 0);
}

#[test]
fn complex_grid_projections_retain_coherence_and_match_direct_contractions() {
    let context = Context::new(Some(1), 16 << 20).unwrap();
    let (m, n) = (5, 2051);
    for complex_samples in [false, true] {
        let samples: Vec<_> = (0..m * n)
            .map(|i| {
                C64::new(
                    (i as f64 * 0.031).sin(),
                    if complex_samples {
                        (i as f64 * 0.071).cos()
                    } else {
                        0.
                    },
                )
            })
            .collect();
        let field: Vec<_> = (0..n).map(|i| (i as f64 * 0.013).cos()).collect();
        {
            let mut projector = ComplexProjector::new(&context, &samples, m, n).unwrap();
            let result = projector.project(&field).unwrap();
            for a in 0..m {
                for b in 0..m {
                    let expected = (0..n)
                        .map(|k| samples[a * n + k].conj() * field[k] * samples[b * n + k])
                        .sum::<C64>();
                    assert!((result[a + b * m] - expected).norm() < 2e-11);
                }
            }
            for complex_matrix in [false, true] {
                let matrix = DMatrix::from_fn(m, m, |a, b| {
                    C64::new(
                        ((a + b) as f64 * 0.23).cos(),
                        if complex_matrix {
                            (a as f64 - b as f64) * 0.07
                        } else {
                            0.
                        },
                    )
                });
                let density = hermitian_density(&context, &samples, &matrix, n).unwrap();
                for k in 0..n {
                    let expected = (0..m)
                        .flat_map(|a| (0..m).map(move |b| (a, b)))
                        .map(|(a, b)| {
                            samples[a * n + k] * matrix[(a, b)] * samples[b * n + k].conj()
                        })
                        .sum::<C64>();
                    assert!(expected.im.abs() < 1e-12);
                    assert!((density.as_slice()[k] - expected.re).abs() < 1e-12);
                }
            }
        }
        assert_eq!(context.memory_used(), 0);
    }
    let tiny = Context::new(Some(1), 16).unwrap();
    assert!(ComplexProjector::new(&tiny, &[C64::new(1., 0.); 4], 2, 2).is_err());
    assert_eq!(tiny.memory_used(), 0);
    let invalid = DMatrix::from_row_slice(
        2,
        2,
        &[
            C64::new(1., 0.),
            C64::new(0., 1.),
            C64::new(0., 1.),
            C64::new(1., 0.),
        ],
    );
    assert!(hermitian_density(&context, &[C64::new(1., 0.); 4], &invalid, 2).is_err());
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn tiled_local_projection_matches_direct_signed_quadrature_and_releases_its_budget() {
    let context = Context::new(Some(1), 16 << 20).unwrap();
    for (m, n) in [(2, 7), (17, 2051)] {
        let samples: Vec<_> = (0..m * n).map(|k| (k as f64 * 0.037).sin()).collect();
        let field: Vec<_> = (0..n).map(|k| (k as f64 * 0.071).cos()).collect();
        {
            let mut projector = RealProjector::new(&context, &samples, m, n).unwrap();
            for sign in [1., -1.] {
                let signed: Vec<_> = field.iter().map(|v| sign * v).collect();
                let result = projector.project(&signed).unwrap();
                for a in 0..m {
                    for b in 0..m {
                        let direct = (0..n)
                            .map(|k| samples[a * n + k] * signed[k] * samples[b * n + k])
                            .sum::<f64>();
                        assert!((result[a + b * m] - direct).abs() < 2e-12);
                    }
                }
            }
            assert!(projector.project(&[f64::NAN; 7]).is_err());
        }
        assert_eq!(context.memory_used(), 0);
    }
    let small = Context::new(Some(1), 64).unwrap();
    assert!(RealProjector::new(&small, &[0.; 64], 8, 8).is_err());
    assert_eq!(small.memory_used(), 0);
}
