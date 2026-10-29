use mandacaru_core::{math::projection::RealProjector, Context};

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
