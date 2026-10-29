use mandacaru_core::{
    math::{reciprocal, SpatialGrid},
    Context,
};
use std::f64::consts::PI;

#[test]
fn bloch_symbols_reproduce_plane_waves_and_keep_nyquist_kinetic_energy() {
    let context = Context::new(Some(1), 4 << 20).unwrap();
    {
        let grid = SpatialGrid::new(
            [7, 9, 5],
            [0.; 3],
            [0.4, 0., 0., 0., 0.5, 0., 0., 0., 0.6],
            true,
        )
        .unwrap();
        let k = [0.031, -0.043, 0.047];
        let symbols = reciprocal::bloch(&context, &grid, k).unwrap();
        let at = (9 + 2) * 5 + 1;
        let expected = [
            2. * PI / (7. * 0.4) + k[0],
            4. * PI / (9. * 0.5) + k[1],
            2. * PI / (5. * 0.6) + k[2],
        ];
        for (a, value) in expected.iter().enumerate() {
            assert!(
                (symbols.vectors.as_slice()[a * grid.index.points() + at] - value).abs() < 1e-12
            );
        }
        assert!(
            (symbols.squares.as_slice()[at] - expected.iter().map(|v| v * v).sum::<f64>()).abs()
                < 1e-12
        );
        let even = SpatialGrid::new(
            [4; 3],
            [0.; 3],
            [0.4, 0., 0., 0., 0.4, 0., 0., 0., 0.4],
            true,
        )
        .unwrap();
        let symbols = reciprocal::bloch(&context, &even, [0.; 3]).unwrap();
        let at = 2 * 4 * 4;
        for a in 0..3 {
            assert!(symbols.vectors.as_slice()[a * 64 + at].abs() < 1e-12);
        }
        assert!((symbols.squares.as_slice()[at] - (PI / 0.4).powi(2)).abs() < 1e-11);
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn skew_bloch_symbols_obey_time_reversal_and_the_mean_square_bound() {
    let context = Context::new(Some(1), 4 << 20).unwrap();
    {
        let shape = [6, 6, 5];
        let grid = SpatialGrid::new(
            shape,
            [0.; 3],
            [0.5, 0.25, 0., 0., 3f64.sqrt() * 0.25, 0., 0., 0., 0.6],
            true,
        )
        .unwrap();
        let k = [0.031, 0.047, 0.073];
        let plus = reciprocal::bloch(&context, &grid, k).unwrap();
        let minus = reciprocal::bloch(&context, &grid, k.map(|v| -v)).unwrap();
        let n = grid.index.points();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for l in 0..shape[2] {
                    let at = (i * shape[1] + j) * shape[2] + l;
                    let partner = (((shape[0] - i) % shape[0]) * shape[1]
                        + (shape[1] - j) % shape[1])
                        * shape[2]
                        + (shape[2] - l) % shape[2];
                    for a in 0..3 {
                        assert!(
                            (plus.vectors.as_slice()[a * n + at]
                                + minus.vectors.as_slice()[a * n + partner])
                                .abs()
                                < 1e-12
                        );
                    }
                    assert!(
                        (plus.squares.as_slice()[at] - minus.squares.as_slice()[partner]).abs()
                            < 1e-12
                    );
                    let norm = (0..3)
                        .map(|a| plus.vectors.as_slice()[a * n + at].powi(2))
                        .sum::<f64>();
                    assert!(plus.squares.as_slice()[at] + 1e-12 >= norm);
                }
            }
        }
    }
    assert_eq!(context.memory_used(), 0);
}
