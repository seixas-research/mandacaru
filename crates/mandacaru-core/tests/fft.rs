use mandacaru_core::{math::fft::Fft3, quantum::C64, Context};

#[test]
fn parseval_coulomb_quadratic_form_matches_the_real_space_convolution() {
    use mandacaru_core::math::{fft::Poisson, SpatialGrid};
    let context = Context::new(Some(3), 64 << 20).unwrap();
    {
        let grid = SpatialGrid::new(
            [5, 6, 7],
            [0.; 3],
            [0.4, 0., 0., 0., 0.5, 0., 0., 0., 0.6],
            false,
        )
        .unwrap();
        let n = grid.index.points();
        let density: Vec<_> = (0..n)
            .map(|i| C64::new((i as f64 * 0.2).sin(), (i as f64 * 0.07).cos()))
            .collect();
        let mut output = context.allocate(n).unwrap();
        for periodic in [false, true] {
            let mut solver = if periodic {
                Poisson::periodic(&context, &grid).unwrap()
            } else {
                Poisson::isolated(&context, &grid).unwrap()
            };
            solver.solve_into(&density, output.as_mut_slice()).unwrap();
            let expected = density
                .iter()
                .zip(output.as_slice())
                .map(|(rho, v)| (rho.conj() * v).re)
                .sum::<f64>()
                * grid.index.dv();
            assert!((solver.quadratic_form(&density).unwrap() - expected).abs() < 2e-12);
        }
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn fft_batches_match_independent_dft_and_parallel_roundtrips() {
    for shape in [[3, 4, 5], [32, 33, 35]] {
        let count = shape.iter().product();
        let input: Vec<_> = (0..count)
            .map(|k| C64::new((0.13 * k as f64).sin(), (0.07 * k as f64).cos()))
            .collect();
        let mut reference = None;
        for threads in [1, 3] {
            let context = Context::new(Some(threads), 64 << 20).unwrap();
            {
                let mut fft = Fft3::new(&context, shape).unwrap();
                let mut values = input.clone();
                fft.transform(&mut values, false).unwrap();
                if shape[0] == 3 {
                    for (i, value) in values.iter().enumerate() {
                        let frequency = [
                            i / (shape[1] * shape[2]),
                            (i / shape[2]) % shape[1],
                            i % shape[2],
                        ];
                        let expected: C64 = input
                            .iter()
                            .enumerate()
                            .map(|(k, value)| {
                                let point = [
                                    k / (shape[1] * shape[2]),
                                    (k / shape[2]) % shape[1],
                                    k % shape[2],
                                ];
                                let angle = -2.
                                    * std::f64::consts::PI
                                    * (0..3)
                                        .map(|axis| {
                                            (point[axis] * frequency[axis]) as f64
                                                / shape[axis] as f64
                                        })
                                        .sum::<f64>();
                                value * C64::from_polar(1., angle)
                            })
                            .sum();
                        assert!((value - expected).norm() < 1e-12);
                    }
                }
                if let Some(reference) = &reference {
                    assert!(values
                        .iter()
                        .zip(reference)
                        .all(|(a, b): (&C64, &C64)| (a - b).norm() < 1e-11));
                } else {
                    reference = Some(values.clone());
                }
                fft.transform(&mut values, true).unwrap();
                assert!(values
                    .iter()
                    .zip(&input)
                    .all(|(a, b)| (a - b).norm() < 2e-14));
            }
            assert_eq!(context.memory_used(), 0);
        }
    }
    let context = Context::new(Some(3), 1024).unwrap();
    assert!(Fft3::new(&context, [32; 3]).is_err());
    assert_eq!(context.memory_used(), 0);
}
