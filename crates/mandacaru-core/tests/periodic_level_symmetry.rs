//! The levels of k-points that an operation of the crystal (or it with time reversal) relates are
//! equal: the whole mesh, listed point by point and solved without any symmetrization, must give
//! them to roundoff. Two discretizations broke it at the 1e-9 Ha level (review D5 of hand-off 12):
//! the product angular rule of the atom-centered spheres, whose pole axis is not a symmetry of a
//! cubic lattice (fcc Al, any functional), and the face images of the screened compensation shapes,
//! which a skewed FFT box picked per axis instead of by the grid's lattice group (wurtzite AlN with
//! HSE06). Both are now invariant by construction; these gates hold the spread at roundoff.
mod periodic_reduced_mesh_support;

use mandacaru_core::periodic::{solver::Options, Crystal};
use nalgebra::{Matrix3, Vector3};
use periodic_reduced_mesh_support::{build, context, solve, tight, Cell, Points};

/// The largest spread (Ha) of the levels of related points the gates accept. Measured
/// (2026-11-16) with the invariant rules: 1.454e-14 (Al) and 1.588e-14 (AlN); with the product
/// angular rule and the per-axis face images (the control, the same gates on the old code):
/// 1.399e-9 and 5.515e-9.
const SPREAD: f64 = 1e-12;

fn fcc_aluminum(grid: usize, mesh: usize) -> Cell {
    let h = 7.6524 / 2.;
    Cell {
        library: "pbe-sr",
        functional: "pbe",
        symbols: vec!["Al"],
        lattice: Matrix3::from_columns(&[
            Vector3::new(0., h, h),
            Vector3::new(h, 0., h),
            Vector3::new(h, h, 0.),
        ]),
        grid: [grid; 3],
        fractional: vec![[0.; 3]],
        spinor: false,
        spin: false,
        moments: None,
        charge: None,
        mesh: ([mesh; 3], true),
    }
}

/// Wurtzite AlN (P6_3mc, u = 0.39) on the 16 x 16 x 28 grid of the review.
fn wurtzite_aln(functional: &'static str, mesh: [usize; 3]) -> Cell {
    let (a, c, u) = (5.877, 9.411, 0.39);
    Cell {
        library: "pbe-sr",
        functional,
        symbols: vec!["Al", "Al", "N", "N"],
        lattice: Matrix3::from_columns(&[
            Vector3::new(a, 0., 0.),
            Vector3::new(-a / 2., a * 3f64.sqrt() / 2., 0.),
            Vector3::new(0., 0., c),
        ]),
        grid: [16, 16, 28],
        fractional: vec![
            [1. / 3., 2. / 3., 0.],
            [2. / 3., 1. / 3., 0.5],
            [1. / 3., 2. / 3., u],
            [2. / 3., 1. / 3., 0.5 + u],
        ],
        spinor: false,
        spin: false,
        moments: None,
        charge: None,
        mesh: (mesh, true),
    }
}

/// The largest difference between the sorted levels of two points of the whole mesh that an
/// operation of the crystal's group (or it followed by time reversal) relates, and the number of
/// distinct related pairs. `symmetric` is the cell built by the setup with its space group, for
/// the operations it carries: semilocal on the Gamma point alone, which every operation keeps
/// (a hybrid keeps no operations where they do not reduce its mesh).
fn spread(whole: &Crystal, symmetric: &Crystal, levels: &[Vec<f64>]) -> (f64, usize) {
    let operations = symmetric
        .description
        .as_ref()
        .and_then(|d| d.operations.as_ref())
        .expect("the symmetric crystal carries its operations");
    let shape = whole.grid.index.shape();
    let cell = whole.grid.step * Matrix3::from_diagonal(&Vector3::from_fn(|i, _| shape[i] as f64));
    let inverse = cell.try_inverse().unwrap();
    let fractional = |k: &Vector3<f64>| cell.transpose() * k / (2. * std::f64::consts::PI);
    let (mut largest, mut pairs) = (0f64, 0usize);
    for (j, p) in whole.points.iter().enumerate() {
        for w in &operations.rotations {
            let r = cell * Matrix3::from_fn(|a, b| f64::from(w[3 * a + b])) * inverse;
            for sign in [1., -1.] {
                let image = fractional(&(r * Vector3::from(p.k) * sign));
                let Some(i) = whole.points.iter().position(|q| {
                    let g = fractional(&Vector3::from(q.k));
                    (0..3).all(|a| {
                        let d = g[a] - image[a];
                        (d - d.round()).abs() < 1e-7
                    })
                }) else {
                    continue;
                };
                pairs += usize::from(i > j);
                for (x, y) in levels[j].iter().zip(&levels[i]) {
                    largest = largest.max((x - y).abs());
                }
            }
        }
    }
    (largest, pairs)
}

fn gate(cell: &Cell, options: &Options, label: &str) {
    let ctx = context();
    let start = std::time::Instant::now();
    let whole = build(&ctx, &cell.lattice, cell, Points::Whole);
    let gamma = Cell {
        functional: "pbe",
        mesh: ([1; 3], true),
        ..cell.clone()
    };
    let symmetric = build(
        &ctx,
        &cell.lattice,
        &gamma,
        Points::Reduced { symmetry: true },
    );
    let setup = start.elapsed().as_secs_f64();
    let result = solve(&whole, options);
    let (largest, pairs) = spread(&whole, &symmetric, &result.eigenvalues);
    println!(
        "{label}: {} points, {pairs} related pairs, {} iterations, free energy {:.12} Ha, \
         spread of related levels {largest:.3e} Ha (setup {setup:.1} s, SCF {:.1} s)",
        whole.points.len(),
        result.history.len(),
        result.free_energy,
        start.elapsed().as_secs_f64() - setup
    );
    assert!(pairs > 0, "{label}: no two points are related");
    assert!(largest < SPREAD, "{label}: spread {largest:e} Ha");
}

#[test]
#[ignore = "slow: 15 s (measured 2026-11-16: setup 15 s, SCF 0.2 s; the fast gates of the two rules are in paw_local_split.rs and reciprocal.rs); needs the shipped library: MANDACARU_PAW_PATH=<mandacaru-paw> cargo test --profile release-test -p mandacaru-core --test periodic_level_symmetry -- --ignored"]
fn the_levels_of_related_points_of_fcc_aluminum_agree_to_roundoff() {
    // A cubic lattice: the product angular rule of the spheres is not invariant under O_h.
    gate(&fcc_aluminum(16, 2), &tight(), "fcc Al PBE 16^3 2x2x2");
}

#[test]
#[ignore = "slow: 200 s (measured 2026-11-16: setup 27 s, 30 hybrid SCF iterations 172 s); needs the shipped library: MANDACARU_PAW_PATH=<mandacaru-paw> cargo test --profile release-test -p mandacaru-core --test periodic_level_symmetry -- --ignored"]
fn the_levels_of_related_points_of_wurtzite_aln_with_hse06_agree_to_roundoff() {
    // A skewed FFT box: the aliases of the screened compensation shapes must be closed under the
    // grid's lattice group. Converged: the iterates of the hybrid SCF are not symmetric to
    // roundoff before (4e-8 Ha after four iterations, with either rule), its fixed point is.
    gate(
        &wurtzite_aln("hse06", [3, 3, 1]),
        &tight(),
        "wurtzite AlN HSE06 16x16x28 3x3x1",
    );
}
