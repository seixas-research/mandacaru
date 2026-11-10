use mandacaru_core::{
    math::{reciprocal, SpatialGrid},
    Context,
};
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64;
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

/// The point operations `W` (fractional, unimodular) of a grid's lattice, by the
/// definition alone: an isometry to 2e-2 that maps nodes onto nodes and aliases
/// onto aliases, closed under products. Written for the test, not shared.
fn operations(shape: [usize; 3], step: [f64; 9]) -> Vec<Matrix3<f64>> {
    let cell = Matrix3::from_row_slice(&step)
        * Matrix3::from_diagonal(&Vector3::from_fn(|i, _| shape[i] as f64));
    let inverse = cell.try_inverse().unwrap();
    let kept = |w: &Matrix3<f64>| {
        if (w.determinant().abs() - 1.).abs() > 1e-9 {
            return false;
        }
        let r = cell * w * inverse;
        if (r.transpose() * r - Matrix3::identity()).norm() > 2e-2 {
            return false;
        }
        let v = w.try_inverse().unwrap().transpose();
        (0..3).all(|i| {
            (0..3).all(|j| {
                let nodes = w[(i, j)] * shape[i] as f64 / shape[j] as f64;
                let aliases = v[(i, j)] * shape[j] as f64 / shape[i] as f64;
                (nodes - nodes.round()).abs() < 1e-9 && (aliases - aliases.round()).abs() < 1e-9
            })
        })
    };
    let mut group = Vec::new();
    for code in 0..3usize.pow(9) {
        let entries: Vec<f64> = (0..9)
            .map(|k| ((code / 3usize.pow(k)) % 3) as f64 - 1.)
            .collect();
        let w = Matrix3::from_row_slice(&entries);
        if kept(&w) {
            group.push(w);
        }
    }
    loop {
        let before = group.len();
        for i in 0..before {
            for j in 0..before {
                let w = group[i] * group[j];
                if kept(&w) && !group.iter().any(|g| (g - w).norm() < 1e-9) {
                    group.push(w);
                }
            }
        }
        if group.len() == before {
            return group;
        }
    }
}

fn box_index(m: Vector3<f64>, shape: [usize; 3]) -> usize {
    let mut at = 0;
    for a in 0..3 {
        let n = shape[a] as i64;
        at = at * shape[a] + (m[a].round() as i64).rem_euclid(n) as usize;
    }
    at
}

fn boxed(x: f64, n: f64) -> f64 {
    x - n * ((x + 0.5 * n + 1e-9) / n).floor()
}

fn frequency(i: usize, n: usize) -> f64 {
    if i < n.div_ceil(2) {
        i as f64
    } else {
        i as f64 - n as f64
    }
}

fn mean_over_operations<T>(
    shape: [usize; 3],
    step: [f64; 9],
    b: &Matrix3<f64>,
    f: impl Fn([f64; 3]) -> T,
) -> Vec<T>
where
    T: std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T> + Default + Copy,
{
    let group = operations(shape, step);
    let mut out = Vec::new();
    for at in 0..shape.iter().product::<usize>() {
        let index = [
            at / (shape[1] * shape[2]),
            at / shape[2] % shape[1],
            at % shape[2],
        ];
        let m = Vector3::from_fn(|a, _| frequency(index[a], shape[a]));
        let mut total = T::default();
        for w in &group {
            let v = w.try_inverse().unwrap().transpose();
            let image = v * m;
            let wrapped = Vector3::from_fn(|a, _| boxed(image[a], shape[a] as f64));
            total = total + f((b * (w.transpose() * wrapped)).into());
        }
        out.push(total * (1. / group.len() as f64));
    }
    out
}

const HEX: [f64; 9] = [0.5, 0.25, 0., 0., 0.4330127018922193, 0., 0., 0., 0.6];

#[test]
fn alias_mean_is_the_mean_of_the_function_over_the_lattice_operations() {
    let context = Context::new(Some(1), 8 << 20).unwrap();
    {
        let shape = [6, 6, 5];
        let grid = SpatialGrid::new(shape, [0.; 3], HEX, true).unwrap();
        let b = reciprocal::reciprocal_cell(&grid).unwrap();
        // Anisotropic, complex and non-quadratic: a mean of the function, not a
        // function of a mean vector.
        let f = |g: [f64; 3]| {
            Complex64::new(
                (-0.3 * (g[0] * g[0] + g[1] * g[1] + g[2] * g[2])).exp(),
                g[0] * g[1] + 0.2 * g[2],
            )
        };
        let mean = reciprocal::alias_mean(&context, &grid, f).unwrap();
        let reference = mean_over_operations(shape, HEX, &b, f);
        let boxed_only: Vec<Complex64> = (0..mean.as_slice().len())
            .map(|at| f(reciprocal::box_vector(&b, shape, at)))
            .collect();
        let mut differs = 0.0f64;
        for ((got, want), plain) in mean.as_slice().iter().zip(&reference).zip(&boxed_only) {
            assert!((got - want).norm() < 1e-13, "{got} vs {want}");
            differs = differs.max((got - plain).norm());
        }
        // Non-vacuous: the box value alone is not the mean on this lattice.
        assert!(
            differs > 1e-2,
            "the alias mean never left the box value: {differs:e}"
        );
        println!("hex 6x6x5: max |alias mean - box value| = {differs:.3e}, max |alias mean - definition| < 1e-13");
        let table = reciprocal::alias_table(&context, &grid).unwrap();
        assert!(!table.is_empty() && table.index.windows(2).all(|w| w[0] <= w[1]));
        assert!(table.weights.iter().all(|w| *w > 0. && *w <= 1.));
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn alias_mean_commutes_with_every_operation_and_a_cube_keeps_the_box_but_for_odd_functions_on_the_faces(
) {
    let context = Context::new(Some(1), 8 << 20).unwrap();
    {
        let shape = [6, 6, 5];
        let grid = SpatialGrid::new(shape, [0.; 3], HEX, true).unwrap();
        // Isotropic in the lattice (hexagonal in-plane): the mean is equivariant.
        let f = |g: [f64; 3]| {
            (g[0] * g[0] + g[1] * g[1]).powi(2)
                + g[2].powi(2)
                + (-(g[0] * g[0] + g[1] * g[1]) * 0.1).exp()
        };
        let mean = reciprocal::alias_mean(&context, &grid, f).unwrap();
        let mut worst = 0.0f64;
        for w in operations(shape, HEX) {
            let v = w.try_inverse().unwrap().transpose();
            for at in 0..mean.as_slice().len() {
                let index = [at / 30, at / 5 % 6, at % 5];
                let m = Vector3::from_fn(|a, _| frequency(index[a], shape[a]));
                let image = box_index(v * m, shape);
                worst = worst.max((mean.as_slice()[at] - mean.as_slice()[image]).abs());
            }
        }
        let scale = mean.as_slice().iter().fold(0.0f64, |m, v| m.max(v.abs()));
        println!("hex 6x6x5: equivariance defect of the alias mean = {worst:.3e} (array maximum {scale:.3e})");
        assert!(worst < 1e-13 * scale);

        // An even cube: the Nyquist plane of an odd function averages to zero,
        // everything else is the plain box value.
        let cube = SpatialGrid::new(
            [4; 3],
            [0.; 3],
            [0.4, 0., 0., 0., 0.4, 0., 0., 0., 0.4],
            true,
        )
        .unwrap();
        let odd = reciprocal::alias_mean(&context, &cube, |g| g[0]).unwrap();
        let b = reciprocal::reciprocal_cell(&cube).unwrap();
        for at in 0..64 {
            let box_x = reciprocal::box_vector(&b, [4; 3], at)[0];
            let on_face = at / 16 == 2;
            let expected = if on_face { 0. } else { box_x };
            assert!((odd.as_slice()[at] - expected).abs() < 1e-13, "{at}");
        }
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn symmetric_wavevectors_commute_with_the_operations_and_zero_the_cube_faces() {
    let context = Context::new(Some(1), 8 << 20).unwrap();
    {
        let cube = SpatialGrid::new(
            [4, 4, 5],
            [0.; 3],
            [0.4, 0., 0., 0., 0.4, 0., 0., 0., 0.5],
            true,
        )
        .unwrap();
        let vectors = reciprocal::symmetric_wavevectors(&context, &cube).unwrap();
        let b = reciprocal::reciprocal_cell(&cube).unwrap();
        let n = 80;
        for at in 0..n {
            let index = [at / 20, at / 5 % 4, at % 5];
            let plain = reciprocal::box_vector(&b, [4, 4, 5], at);
            for a in 0..3 {
                let expected = if a < 2 && index[a] == 2 { 0. } else { plain[a] };
                assert!((vectors.as_slice()[a * n + at] - expected).abs() < 1e-13);
            }
        }
        // Hexagonal: fractional vector m~ obeys m~(V m) = V m~(m) for every V.
        let shape = [6, 6, 5];
        let hex = SpatialGrid::new(shape, [0.; 3], HEX, true).unwrap();
        let sym = reciprocal::symmetric_wavevectors(&context, &hex).unwrap();
        let b = reciprocal::reciprocal_cell(&hex).unwrap();
        let inverse = b.try_inverse().unwrap();
        let fractional =
            |at: usize| inverse * Vector3::from_fn(|a, _| sym.as_slice()[a * 180 + at]);
        let (mut worst, mut moved) = (0.0f64, 0.0f64);
        for w in operations(shape, HEX) {
            let v = w.try_inverse().unwrap().transpose();
            for at in 0..180 {
                let index = [at / 30, at / 5 % 6, at % 5];
                let m = Vector3::from_fn(|a, _| frequency(index[a], shape[a]));
                worst =
                    worst.max((fractional(box_index(v * m, shape)) - v * fractional(at)).norm());
                moved = moved.max((fractional(at) - m).norm());
            }
        }
        println!("hex 6x6x5: commutation defect of the symmetric wave vectors = {worst:.3e} (largest move off the box {moved:.2})");
        assert!(worst < 1e-12 && moved > 0.1);
    }
    assert_eq!(context.memory_used(), 0);
}

/// `norm_strain` is the strain derivative of `norms` contracted with per-node weights: a central
/// difference of the weighted norms of the grid whose cell is `(1 + eps) cell`, for a cell whose
/// operations are signed permutations and for a hexagonal one (alias-averaged norms).
#[test]
fn the_norm_strain_is_the_derivative_of_the_alias_averaged_norms() {
    let context = Context::new(Some(1), 4 << 20).unwrap();
    for (shape, step) in [
        ([6, 5, 4], [0.9, 0., 0., 0., 1.1, 0., 0., 0., 1.3]),
        ([6, 6, 4], [1., 0.5, 0., 0., 3f64.sqrt() / 2., 0., 0., 0., 1.2]),
    ] {
        let n = shape[0] * shape[1] * shape[2];
        let weights: Vec<f64> = (0..n).map(|i| 0.3 + ((i * 7) % 11) as f64 / 11.).collect();
        let weighted = |eps: Matrix3<f64>| {
            let moved = (Matrix3::identity() + eps) * Matrix3::from_row_slice(&step);
            let grid = SpatialGrid::new(shape, [0.; 3], std::array::from_fn(|k| moved[(k / 3, k % 3)]), true).unwrap();
            let norms = reciprocal::norms(&context, &grid).unwrap();
            norms.as_slice().iter().zip(&weights).map(|(g, w)| g * w).sum::<f64>()
        };
        let grid = SpatialGrid::new(shape, [0.; 3], step, true).unwrap();
        let analytic = reciprocal::norm_strain(&context, &grid, &weights).unwrap();
        for (a, b) in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)] {
            let h = 1e-5;
            let (mut up, mut down) = (Matrix3::zeros(), Matrix3::zeros());
            up[(a, b)] = h;
            down[(a, b)] = -h;
            let numeric = (weighted(up) - weighted(down)) / (2. * h);
            assert!(
                (analytic[a][b] - numeric).abs() < 1e-7 * (1. + numeric.abs()),
                "{shape:?} ({a},{b}): {} against {numeric}",
                analytic[a][b]
            );
        }
    }
    assert!(reciprocal::norm_strain(&context, &SpatialGrid::new([4; 3], [0.; 3], [0.4, 0., 0., 0., 0.4, 0., 0., 0., 0.4], true).unwrap(), &[1.; 3]).is_err());
    assert_eq!(context.memory_used(), 0);
}
