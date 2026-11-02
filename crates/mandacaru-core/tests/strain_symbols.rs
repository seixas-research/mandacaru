//! The pieces of the strain derivatives of the reciprocal-space symbols and form factors: the
//! angular gradient of the complex harmonics, the alias-mean outer product of the Bloch vectors,
//! the slope of the shape tables and the gradient of the compensation shapes, each against a
//! central difference of the function it differentiates. The index convention is settled here:
//! under `x -> (1 + eps) x` a wave vector moves by `G -> (1 - eps^T) G`, so one entry `eps_ab`
//! moves only the component `G_b`, by `-G_a`, and `d f / d eps_ab = -G_a d_b f`.
//! Needs `MANDACARU_PAW_PATH` for the shape tests.
mod strain_support;

use mandacaru_core::{
    math::{angular, reciprocal},
    periodic::setup::{
        electrostatics::max_wavenumber,
        shapes::{Shapes, SHAPE_TABLE_STEP},
    },
};
use num_complex::Complex64;
use strain_support::*;

/// A deterministic stream of numbers in `[-1, 1)`.
fn stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    move || {
        state = state
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        2. * ((state >> 11) as f64 / (1u64 << 53) as f64) - 1.
    }
}

#[test]
fn the_angular_gradient_is_the_difference_of_the_harmonic_in_the_wave_vector() {
    let mut random = stream(7);
    let mut directions: Vec<[f64; 3]> = (0..24).map(|_| [random(), random(), random()]).collect();
    // On and near the polar axis and the other axes, where the azimuth degenerates.
    directions.extend([
        [0., 0., 1.],
        [0., 0., -2.],
        [1e-7, 2e-7, 1.],
        [1., 0., 0.],
        [0., 3., 0.],
    ]);
    let step = 1e-6;
    let (mut worst_gap, mut largest) = (0f64, 0f64);
    let mut transposed_gap = 0f64;
    for l in 0..=4usize {
        for m in -(l as i32)..=(l as i32) {
            for u in &directions {
                let norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
                let got = angular::complex_gradient(l, m, *u);
                let mut numeric = [Complex64::default(); 3];
                for (b, value) in numeric.iter_mut().enumerate() {
                    let (mut plus, mut minus) = (*u, *u);
                    let h = step * norm;
                    plus[b] += h;
                    minus[b] -= h;
                    *value =
                        (angular::complex(l, m, plus) - angular::complex(l, m, minus)) / (2. * h);
                }
                for b in 0..3 {
                    worst_gap = worst_gap.max((got[b] - numeric[b]).norm() * norm);
                    largest = largest.max(got[b].norm() * norm);
                    transposed_gap =
                        transposed_gap.max((got[(b + 1) % 3] - numeric[b]).norm() * norm);
                }
            }
        }
    }
    println!("angular gradient: worst {worst_gap:.2e} of max {largest:.2e}");
    assert!(largest > 1.);
    assert!(worst_gap < 1e-8 * largest);
    // Control: the components rotated (an axis mix-up) miss by orders.
    assert!(transposed_gap > 100. * 1e-8 * largest);
    // The gradient of the s harmonic vanishes and so does every gradient at the origin.
    assert!(angular::complex_gradient(0, 0, [0.3, -0.2, 0.9])
        .iter()
        .all(|g| g.norm() < 1e-15));
    assert!(angular::complex_gradient(2, 1, [0.; 3])
        .iter()
        .all(|g| g.norm() == 0.));
}

#[test]
fn the_outer_product_symbol_is_the_mean_of_products_and_the_derivative_of_the_squares() {
    let ctx = context();
    // A generic fractional k (no frequency is on a face) and Gamma (the faces alias).
    for name in ["si", "al2", "lih"] {
        let cell = Cell::load(&ctx, name);
        for (fractional, faces) in [([0.137, -0.291, 0.0623], false), ([0.; 3], true)] {
            let n = cell.grid.index.points();
            let b = reciprocal::reciprocal_cell(&cell.grid).unwrap();
            let k: [f64; 3] = (b * nalgebra::Vector3::from(fractional)).into();
            let symbols = reciprocal::bloch(&ctx, &cell.grid, k).unwrap();
            let outer = reciprocal::bloch_outer(&ctx, &cell.grid, k).unwrap();
            let (squares, outer) = (symbols.squares.as_slice(), outer.as_slice());
            let vectors = symbols.vectors.as_slice();
            // The trace of the outer product is the symbol's square.
            let trace = (0..n)
                .map(|at| (outer[at] + outer[n + at] + outer[2 * n + at] - squares[at]).abs())
                .fold(0., f64::max);
            let scale = squares.iter().cloned().fold(0., f64::max);
            assert!(trace < 1e-12 * scale, "{name}: {trace}");
            // The derivative of the squares: central difference of `bloch` on the strained grid
            // at the same FRACTIONAL k, the lattice operations pinned.
            let h = 1e-5;
            let (mut miss, mut vector_miss, mut frozen_miss) = (0f64, 0f64, 0f64);
            for (row, &(a, c)) in strain_voigt().iter().enumerate() {
                let side = |delta: f64, frozen: bool| -> Vec<f64> {
                    let strained = cell.strained(entry(a, c, delta));
                    assert_pinned(&ctx, &cell, &strained);
                    let moved = reciprocal::reciprocal_cell(&strained.grid).unwrap();
                    let at_k: [f64; 3] = if frozen {
                        k
                    } else {
                        (moved * nalgebra::Vector3::from(fractional)).into()
                    };
                    reciprocal::bloch(&ctx, &strained.grid, at_k)
                        .unwrap()
                        .squares
                        .as_slice()
                        .to_vec()
                };
                let (plus, minus) = (side(h, false), side(-h, false));
                let (plus_frozen, minus_frozen) = (side(h, true), side(-h, true));
                for at in 0..n {
                    let numeric = (plus[at] - minus[at]) / (2. * h);
                    miss = miss.max((numeric + 2. * outer[row * n + at]).abs());
                    // The mean-vector symbol is the defect of the old kernel stress.
                    vector_miss = vector_miss
                        .max((numeric + 2. * vectors[a * n + at] * vectors[c * n + at]).abs());
                    let frozen = (plus_frozen[at] - minus_frozen[at]) / (2. * h);
                    frozen_miss = frozen_miss.max((frozen + 2. * outer[row * n + at]).abs());
                }
            }
            println!("{name} k = {fractional:?}: derivative {miss:.2e} of {scale:.2e}; controls: mean vectors {vector_miss:.2e}, Cartesian k frozen {frozen_miss:.2e}");
            assert!(miss < 1e-8 * scale, "{name}");
            // The mean of products differs from the product of the means on the faces, and
            // only a fractional k off Gamma has a k part to freeze.
            if faces {
                assert!(
                    vector_miss > 100. * 1e-8 * scale,
                    "{name}: the mean-vector control is vacuous"
                );
            }
            if !faces {
                assert!(frozen_miss > 100. * 1e-8 * scale, "{name}");
            }
        }
    }
}

fn strain_voigt() -> [(usize, usize); 6] {
    mandacaru_core::periodic::setup::strain_grid::VOIGT
}

#[test]
fn the_shape_slope_is_the_slope_of_the_linear_piece() {
    let ctx = context();
    let cell = Cell::load(&ctx, "si");
    let q_max = max_wavenumber(&ctx, &cell.grid).unwrap();
    let shapes = Shapes::new(&ctx, &cell.sites(), q_max).unwrap();
    let table = shapes.table(cell.atoms[0].compensation_radius, 1).unwrap();
    for x in [0.0123, 0.5049, 3.3351, 17.775, 119.9951] {
        let half = 0.5 * SHAPE_TABLE_STEP * 0.4;
        let numeric = (table.at(x + half) - table.at(x - half)) / (2. * half);
        assert!(
            (table.slope(x) - numeric).abs() < 1e-9 * numeric.abs().max(1e-3),
            "{x}"
        );
    }
    // Clamped outside the table, as `at` is.
    assert_eq!(table.slope(-1.), 0.);
    assert_eq!(table.slope(1e9), 0.);
    // On a node the slope is the right-hand one.
    let j = 100;
    let right = (table.f[j + 1] - table.f[j]) / (table.q[j + 1] - table.q[j]);
    assert_eq!(table.slope(table.q[j]), right);
}

#[test]
fn the_shape_gradient_is_the_difference_of_the_shapes_under_a_strain() {
    use mandacaru_core::math::spline::Cubic;
    use std::f64::consts::PI;
    let ctx = context();
    for name in ["si", "lih"] {
        let cell = Cell::load(&ctx, name);
        let q_max = max_wavenumber(&ctx, &cell.grid).unwrap();
        let sites = cell.sites();
        let shapes = Shapes::new(&ctx, &sites, q_max).unwrap();
        let rows = cell.channels.len();
        // Wave vectors away from the table nodes: the difference cannot straddle a kink.
        let mut random = stream(11);
        let vectors: Vec<[f64; 3]> = (0..300)
            .map(|_| {
                let g = [random(), random(), random()];
                let norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
                let target = 0.3 + 13. * (random() * 0.5 + 0.5);
                let cell_index = (target / SHAPE_TABLE_STEP).floor() + 0.5;
                let radius = cell_index * SHAPE_TABLE_STEP;
                g.map(|v| v / norm * radius)
            })
            .collect();
        let m = vectors.len();
        let mut gradient = vec![Complex64::default(); rows * 3 * m];
        shapes
            .transform_gradients(&sites, &vectors, &mut gradient, m)
            .unwrap();
        let h = 1e-6;
        let (mut gap, mut scale) = (0f64, 0f64);
        let (mut transposed, mut exact_radial, mut no_angular, mut flipped) =
            (0f64, 0f64, 0f64, 0f64);
        // The exact derivative of each (radius, L) transform, the control of the table slope.
        let exact: Vec<Cubic> = (0..rows)
            .map(|c| {
                let (atom, l, _) = cell.channels[c];
                let table = shapes
                    .table(cell.atoms[atom].compensation_radius, l)
                    .unwrap();
                Cubic::new(
                    &ctx,
                    ctx.copy_slice(&table.q).unwrap(),
                    ctx.copy_slice(&table.f).unwrap(),
                )
                .unwrap()
            })
            .collect();
        for a in 0..3 {
            for b in 0..3 {
                let side = |delta: f64| -> Vec<Complex64> {
                    let strained = cell.strained(entry(a, b, delta));
                    // G' = (1 - eps^T) G: only the component b moves, by -eps G_a.
                    let moved: Vec<[f64; 3]> = vectors
                        .iter()
                        .map(|g| {
                            let mut moved = *g;
                            moved[b] = g[b] - delta * g[a];
                            moved
                        })
                        .collect();
                    let mut out = vec![Complex64::default(); rows * m];
                    shapes
                        .transforms(&strained.sites(), &moved, &mut out, m)
                        .unwrap();
                    out
                };
                let (plus, minus) = (side(h), side(-h));
                for c in 0..rows {
                    let (atom, l, mm) = cell.channels[c];
                    let unit = [
                        Complex64::new(1., 0.),
                        Complex64::new(0., -1.),
                        Complex64::new(-1., 0.),
                        Complex64::new(0., 1.),
                    ][l % 4];
                    for (v, g) in vectors.iter().enumerate() {
                        let norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
                        let numeric = (plus[c * m + v] - minus[c * m + v]) / (2. * h);
                        let analytic = gradient[(c * 3 + b) * m + v] * -g[a];
                        gap = gap.max((analytic - numeric).norm());
                        scale = scale.max(analytic.norm());
                        flipped = flipped.max((-analytic - numeric).norm());
                        transposed = transposed
                            .max((gradient[(c * 3 + a) * m + v] * -g[b] - numeric).norm());
                        let angle = -(g[0] * cell.atoms[atom].center[0]
                            + g[1] * cell.atoms[atom].center[1]
                            + g[2] * cell.atoms[atom].center[2]);
                        let common = unit * Complex64::new(angle.cos(), angle.sin()) * (4. * PI);
                        let table = shapes
                            .table(cell.atoms[atom].compensation_radius, l)
                            .unwrap();
                        let radial = angular::complex(l, mm, *g)
                            * ((exact[c].value_slope(norm).1 - table.slope(norm)) * g[b] / norm);
                        exact_radial = exact_radial
                            .max(((analytic + common * radial * -g[a]) - numeric).norm());
                        let angular_part = angular::complex_gradient(l, mm, *g)[b] * table.at(norm);
                        no_angular = no_angular
                            .max(((analytic - common * angular_part * -g[a]) - numeric).norm());
                    }
                }
            }
        }
        println!("{name}: shape gradient {gap:.2e} of max {scale:.2e}; controls: sign {flipped:.2e}, transposed {transposed:.2e}, exact radial derivative {exact_radial:.2e}, no angular gradient {no_angular:.2e}");
        assert!(scale > 1.);
        assert!(gap < 1e-8 * scale, "{name}");
        for control in [flipped, transposed, exact_radial, no_angular] {
            assert!(
                control > 100. * 1e-8 * scale,
                "{name}: a control does not bite"
            );
        }
    }
}
