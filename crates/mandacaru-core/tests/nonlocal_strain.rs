//! X-6: the strain derivative of the rVV10 correlation energy of a periodic grid against a central
//! finite difference of `nonlocal::evaluate` on the strained grid, at fixed node values of the
//! density and its gradient contraction.
//!
//! At fixed nodes the energy depends on the cell through the volume element (the Jacobian
//! `delta_ab E`) and through the wave numbers of the kernel, which is `nonlocal::strain`; the
//! gradient and density responses belong to the xc potential and are not in either. The oracle is
//! the Richardson difference of two central steps on `x -> (1 + eps) x`.
use mandacaru_core::{
    math::SpatialGrid,
    models::nonlocal::{evaluate, strain, Options},
    Context,
};

const STEP: f64 = 1e-4;

fn matrix_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [f64; 9] {
    let mut out = [0.; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[3 * i + j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

/// The grid whose cell is `(1 + eps) cell` (`step` holds the grid vectors as columns).
fn strained(shape: [usize; 3], step: [[f64; 3]; 3], eps: [[f64; 3]; 3]) -> SpatialGrid {
    let mut moved = [[0.; 3]; 3];
    for i in 0..3 {
        moved[i][i] = 1.;
        for j in 0..3 {
            moved[i][j] += eps[i][j];
        }
    }
    SpatialGrid::new(shape, [0.; 3], matrix_mul(moved, step), true).unwrap()
}

/// Smooth, positive node values with a gradient contraction of the same support (a stand-in for
/// a valence density: the energy does not care where the numbers come from).
fn fields(shape: [usize; 3]) -> (Vec<f64>, Vec<f64>) {
    let n = shape[0] * shape[1] * shape[2];
    let (mut density, mut sigma) = (vec![0.; n], vec![0.; n]);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = (i * shape[1] + j) * shape[2] + k;
                let x = [
                    i as f64 / shape[0] as f64,
                    j as f64 / shape[1] as f64,
                    k as f64 / shape[2] as f64,
                ];
                let tau = std::f64::consts::TAU;
                density[at] = 0.06
                    + 0.04 * (tau * x[0]).cos() * (tau * x[1] + 0.3).cos()
                    + 0.02 * (tau * (x[2] + 0.5 * x[0])).sin();
                sigma[at] = 0.004 * (1.2 + (tau * (x[0] - x[1])).sin() * (tau * x[2]).cos());
            }
        }
    }
    (density, sigma)
}

fn check(label: &str, shape: [usize; 3], step: [[f64; 3]; 3], b: f64, entries: &[(usize, usize)]) {
    let ctx = Context::new(Some(1), 1 << 28).unwrap();
    let (density, sigma) = fields(shape);
    let options = Options {
        b,
        ..Default::default()
    };
    let reference = strained(shape, step, [[0.; 3]; 3]);
    let energy = evaluate(&ctx, &reference, &density, &sigma, options)
        .unwrap()
        .energy;
    let kernel = strain(&ctx, &reference, &density, &sigma, options).unwrap();
    let at = |a: usize, c: usize, h: f64| {
        let mut eps = [[0.; 3]; 3];
        eps[a][c] = h;
        let grid = strained(shape, step, eps);
        evaluate(&ctx, &grid, &density, &sigma, options)
            .unwrap()
            .energy
    };
    for &(a, c) in entries {
        let central = |h: f64| (at(a, c, h) - at(a, c, -h)) / (2. * h);
        let oracle = (4. * central(0.5 * STEP) - central(STEP)) / 3.;
        let jacobian = if a == c { energy } else { 0. };
        let got = jacobian + kernel[a][c];
        let miss = (got - oracle).abs();
        eprintln!(
            "{label} g_{a}{c}: analytic {got:+.10e} (jacobian {jacobian:+.4e}, kernel {:+.6e}), oracle {oracle:+.10e}, miss {miss:.2e}",
            kernel[a][c]
        );
        // The kernel term is not vacuous: without it the miss is far above the gate.
        assert!(
            kernel[a][c].abs() > 1e3 * 1e-9,
            "{label} g_{a}{c}: the kernel term {:e} is too small to test",
            kernel[a][c]
        );
        assert!(miss < 1e-8, "{label} g_{a}{c}: miss {miss:e}");
    }
}

#[test]
fn the_kernel_strain_of_a_signed_permutation_cell_matches_the_energy_difference() {
    check(
        "orthorhombic",
        [8, 6, 6],
        [[1.1, 0., 0.], [0., 1.3, 0.], [0., 0., 0.9]],
        6.3,
        &[(0, 0), (1, 1), (2, 2)],
    );
}

#[test]
fn the_kernel_strain_of_an_alias_averaged_skew_cell_matches_the_energy_difference() {
    // A hexagonal grid: the lattice operations are not signed permutations, so each norm is the
    // mean over the alias vectors (`math::reciprocal::norms`), and the strain mixes entries.
    check(
        "hexagonal",
        [6, 6, 4],
        [[1., 0.5, 0.], [0., 3f64.sqrt() / 2., 0.], [0., 0., 1.2]],
        11.95,
        &[(0, 0), (0, 1), (1, 1), (2, 2)],
    );
}
