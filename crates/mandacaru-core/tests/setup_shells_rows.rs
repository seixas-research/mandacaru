//! Rows, derivatives, reach and refusals of the shell sums (`periodic::setup::{images, shells}`):
//! the row sink a float32 store plugs into, the derivative axes against central differences,
//! long-tailed high-`l` shells against the image sum at the same support, the sphere reach, and
//! the inputs the sums refuse. Needs `MANDACARU_PAW_PATH` (the shipped `lda-sr` Si dataset).
mod setup_shells_support;

use mandacaru_core::{
    periodic::setup::{
        images::{reaching, Images, Region},
        shells::{DenseRows, Radial, RowSink, Sampling, Shell, Shells},
    },
    Context, Error, Result,
};
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64;
use setup_shells_support::*;
use std::sync::Arc;

const SUM: f64 = 1e-12;

/// Counts the rows, and rounds each once to single precision as a float32 store would.
struct Rounding {
    seen: Vec<(usize, usize)>,
    stored: Vec<Complex64>,
    functions: usize,
    points: usize,
}
impl RowSink for Rounding {
    fn put(&mut self, k: usize, row: usize, values: &[Complex64]) -> Result<()> {
        self.seen.push((k, row));
        let start = (k * self.functions + row) * self.points;
        for (to, v) in self.stored[start..start + self.points]
            .iter_mut()
            .zip(values)
        {
            *to = Complex64::new(v.re as f32 as f64, v.im as f32 as f64);
        }
        Ok(())
    }
}

#[test]
fn each_row_is_put_once_and_a_sink_can_round_it() {
    let ctx = context();
    let (cell, basis) = silicon(&ctx);
    let frozen = frozen_orbitals();
    let (functions, n) = (basis.shells.functions(), frozen.points.len());
    let exact = bloch(&ctx, &basis, &cell, &frozen.kpoints, &frozen.points);
    let region = region_for(&cell, &frozen.points);
    let columns: Vec<Vec<f64>> = (0..3)
        .map(|axis| frozen.points.iter().map(|p| p[axis]).collect())
        .collect();
    let mut sink = Rounding {
        seen: Vec::new(),
        stored: vec![Complex64::new(0., 0.); exact.len()],
        functions,
        points: n,
    };
    basis
        .shells
        .values(
            &ctx,
            &Sampling {
                lattice: &cell.lattice,
                region,
                kpoints: &frozen.kpoints,
            },
            [&columns[0], &columns[1], &columns[2]],
            &mut sink,
        )
        .unwrap();
    let mut seen = sink.seen.clone();
    seen.sort_unstable();
    let all: Vec<_> = (0..frozen.kpoints.len())
        .flat_map(|k| (0..functions).map(move |row| (k, row)))
        .collect();
    assert_eq!(seen, all, "every (k, row) exactly once");
    let error = relative(&sink.stored, &exact);
    eprintln!("rounded once to float32: {error:e}");
    assert!(error > 1e-9 && error < 1e-6, "{error:e}");
    // The same rounding applied to the finished float64 array agrees bit for bit.
    let after: Vec<_> = exact
        .iter()
        .map(|v| Complex64::new(v.re as f32 as f64, v.im as f32 as f64))
        .collect();
    assert_eq!(sink.stored, after);
}

#[test]
fn the_derivative_axes_are_central_differences_of_the_sums() {
    let ctx = context();
    let (cell, basis) = silicon(&ctx);
    let frozen = frozen_orbitals();
    let (functions, n) = (basis.shells.functions(), frozen.points.len());
    let step = 1e-5;
    let region = region_for(&cell, &frozen.points);
    let columns: Vec<Vec<f64>> = (0..3)
        .map(|axis| frozen.points.iter().map(|p| p[axis]).collect())
        .collect();
    let ks = &frozen.kpoints[..1];
    for axis in 0..3 {
        let mut slope = vec![Complex64::new(0., 0.); functions * n];
        basis
            .shells
            .derivatives(
                &ctx,
                &Sampling {
                    lattice: &cell.lattice,
                    region,
                    kpoints: ks,
                },
                [&columns[0], &columns[1], &columns[2]],
                axis,
                &mut DenseRows::new(&mut slope, functions, n),
            )
            .unwrap();
        let moved = |sign: f64| {
            let points: Vec<[f64; 3]> = frozen
                .points
                .iter()
                .map(|p| {
                    let mut q = *p;
                    q[axis] += sign * step;
                    q
                })
                .collect();
            bloch(&ctx, &basis, &cell, ks, &points)
        };
        let (plus, minus) = (moved(1.), moved(-1.));
        let difference: Vec<_> = plus
            .iter()
            .zip(&minus)
            .map(|(a, b)| (a - b) / (2. * step))
            .collect();
        let error = relative(&slope, &difference);
        eprintln!("derivative axis {axis}: {error:e}");
        assert!(error < 1e-6, "{error:e}");
    }
}

/// A smooth analytic table with a long tail, to exercise the support cut and the confinement
/// radius the way a multiple-zeta shell with a wide table does (`l` up to 3).
fn long_shell(ctx: &Context, l: usize, center: [f64; 3], first_row: usize) -> (Shell, Table) {
    let r: Vec<f64> = (1..=2400).map(|i| i as f64 * 0.0125).collect();
    let values: Vec<f64> = r
        .iter()
        .map(|x| x.powi(l as i32) * (-x / 3.).exp() * (1. + 0.3 * (0.8 * x).cos()))
        .collect();
    let ms: Vec<i32> = (-(l as i32)..=l as i32).collect();
    let rows: Vec<usize> = (first_row..first_row + ms.len()).collect();
    let table = Table {
        center,
        l,
        ms: ms.clone(),
        rows: rows.clone(),
        support: support_of(&r, &values),
        r: r.clone(),
        values: values.clone(),
    };
    let shell = Shell {
        center,
        l,
        ms,
        rows,
        radial: Arc::new(Radial::new(ctx, &r, &values, None).unwrap()),
    };
    (shell, table)
}

#[test]
fn long_tails_and_high_l_equal_the_image_sum_at_the_same_support() {
    let ctx = context();
    // A small cell, so that thousands of images contribute.
    let lattice = Matrix3::from_columns(&[
        Vector3::new(0., 2.6, 2.6),
        Vector3::new(2.6, 0., 2.6),
        Vector3::new(2.6, 2.6, 0.),
    ]);
    let (mut shells, mut tables, mut row) = (Vec::new(), Vec::new(), 0);
    for (l, center) in [
        (0, [0., 0., 0.]),
        (1, [0.7, 0.4, 0.2]),
        (2, [1.3, 1.3, 1.3]),
        (3, [0.2, 1.1, 0.6]),
    ] {
        let (shell, table) = long_shell(&ctx, l, center, row);
        row += 2 * l + 1;
        shells.push(shell);
        tables.push(table);
    }
    let basis = Basis {
        shells: Shells::new(shells).unwrap(),
        tables,
    };
    let cell = Cell {
        region: bounding_ball(&lattice, Vector3::zeros()),
        lattice,
        centers: vec![],
        filter_wavevector: 0.,
        orbitals: vec![],
    };
    let points = [
        [0.31, 0.77, 1.02],
        [2.1, 0.4, 3.3],
        [4.4, 3.9, 0.15],
        [0.9, 2.2, 2.7],
    ];
    let kpoints = [[0.31, -0.27, 0.19], [-0.31, 0.27, -0.19]];
    let got = bloch(&ctx, &basis, &cell, &kpoints, &points);
    let want = image_sum(&ctx, &basis, &cell.lattice, &kpoints, &points);
    let error = relative(&got, &want);
    let images = Images::around(
        &ctx,
        &cell.lattice,
        &region_for(&cell, &points),
        [0.; 3],
        basis.tables[3].support,
    )
    .unwrap()
    .len();
    eprintln!(
        "long tails, l = 0..3, supports {:?}, {images} images: {error:e}",
        basis.tables.iter().map(|t| t.support).collect::<Vec<_>>()
    );
    assert!(images > 1000);
    assert!(error < SUM, "{error:e}");
}

#[test]
fn only_the_shells_that_reach_a_sphere_are_kept() {
    let ctx = context();
    let (cell, basis) = silicon(&ctx);
    // A sphere at an atom is reached by everything in this small cell.
    let near = Region {
        center: cell.centers[0],
        radius: 2.3,
    };
    let reached = basis.shells.reaching(&ctx, &cell.lattice, &near).unwrap();
    assert_eq!(reached.rows, (0..8).collect::<Vec<_>>());
    // A shell whose support is shorter than the gap to the sphere is dropped, and the others are
    // renumbered in order.
    let short = Radial::new(
        &ctx,
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        &[1., 0.9, 0.5, 0.2, 0.05, 0.],
        None,
    )
    .unwrap();
    assert!(short.support() < 0.7);
    let mixed = Shells::new(vec![
        Shell {
            center: [6.2, 6.2, 6.2],
            l: 0,
            ms: vec![0],
            rows: vec![0],
            radial: Arc::new(short),
        },
        basis.shells.shells()[1].clone(),
    ])
    .unwrap();
    // The sphere at the origin is reached by the p shell on the atom there, not by the far one.
    let tight = Region {
        center: [0., 0., 0.],
        radius: 0.05,
    };
    let reached = mixed.reaching(&ctx, &cell.lattice, &tight).unwrap();
    assert_eq!(reached.rows, vec![1, 2, 3]);
    assert_eq!(reached.shells.functions(), 3);
    assert_eq!(reached.shells.shells()[0].rows, vec![0, 1, 2]);
    // The verdict function agrees with a count of images.
    let bodies = [([0.; 3], 3.0), ([2.6, 2.6, 2.6], 0.5)];
    let verdict = reaching(&ctx, &cell.lattice, &tight, &bodies).unwrap();
    assert_eq!(verdict, [true, false]);
}

#[test]
fn images_are_the_lattice_vectors_inside_the_reach() {
    let ctx = context();
    let (cell, _) = silicon(&ctx);
    let region = cell.region;
    let center = cell.centers[1];
    let support = 9.0;
    let images = Images::around(&ctx, &cell.lattice, &region, center, support).unwrap();
    // Brute force over a box that must hold them.
    let mut count = 0;
    for i in -8i32..=8 {
        for j in -8i32..=8 {
            for k in -8i32..=8 {
                let r = cell.lattice * Vector3::new(i as f64, j as f64, k as f64);
                let d = Vector3::from(center) + r - Vector3::from(region.center);
                if d.norm() <= support + region.radius {
                    count += 1;
                }
            }
        }
    }
    assert_eq!(images.len(), count);
    // exp(i sign k.R): the two signs are conjugates.
    let ks = [[0.1, 0.2, -0.3]];
    let plus = images.phases(&ctx, &ks, 1.).unwrap();
    let minus = images.phases(&ctx, &ks, -1.).unwrap();
    for (a, b) in plus.as_slice().iter().zip(minus.as_slice()) {
        assert!((a.conj() - b).norm() < 1e-15);
    }
}

#[test]
fn bad_input_and_a_small_budget_raise() {
    let ctx = context();
    let (cell, basis) = silicon(&ctx);
    let frozen = frozen_orbitals();
    let columns: Vec<Vec<f64>> = (0..3)
        .map(|axis| frozen.points.iter().map(|p| p[axis]).collect())
        .collect();
    let sampling = Sampling {
        lattice: &cell.lattice,
        region: region_for(&cell, &frozen.points),
        kpoints: &frozen.kpoints,
    };
    let mut out = vec![Complex64::new(0., 0.); 3 * 8 * frozen.points.len()];
    let mut sink = DenseRows::new(&mut out, 8, frozen.points.len());
    // No points, ragged coordinates, no k-point, a NaN coordinate, an axis beyond z.
    let empty: [&[f64]; 3] = [&[], &[], &[]];
    assert!(basis
        .shells
        .values(&ctx, &sampling, empty, &mut sink)
        .is_err());
    let ragged = [&columns[0][..3], &columns[1][..4], &columns[2][..4]];
    assert!(basis
        .shells
        .values(&ctx, &sampling, ragged, &mut sink)
        .is_err());
    let none = Sampling {
        kpoints: &[],
        ..sampling
    };
    let all = [&columns[0][..], &columns[1], &columns[2]];
    assert!(basis.shells.values(&ctx, &none, all, &mut sink).is_err());
    let mut bad = columns[0].clone();
    bad[3] = f64::NAN;
    let nan = [&bad[..], &columns[1], &columns[2]];
    assert!(basis
        .shells
        .values(&ctx, &sampling, nan, &mut sink)
        .is_err());
    assert!(basis
        .shells
        .derivatives(&ctx, &sampling, all, 3, &mut sink)
        .is_err());
    // Rows that overlap, or leave a hole, or a component beyond l.
    let one = basis.shells.shells()[0].clone();
    let twice = Shells::new(vec![one.clone(), one.clone()]);
    assert!(matches!(twice, Err(Error::Invalid(_))));
    let high = Shell {
        ms: vec![2],
        ..one.clone()
    };
    assert!(Shells::new(vec![high]).is_err());
    // A budget too small for the point order raises, and releases what it took.
    let small = Context::new(Some(1), 4096).unwrap();
    let result = basis.shells.values(&small, &sampling, all, &mut sink);
    assert!(matches!(result, Err(Error::Budget { .. })), "{result:?}");
    assert_eq!(small.memory_used(), 0);
}
