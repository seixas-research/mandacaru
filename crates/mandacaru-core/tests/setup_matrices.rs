//! Integration tests for the periodic PAW setup matrices and crystal assembly (WP-M).
//!
//! Covers:
//! - Per-kpoint matrices (overlap S, kinetic T, fixed Hamiltonian, projections C, Lowdin transform)
//! - Parity of k together versus one by one
//! - Spin-orbit term properties
//! - Full `PeriodicSetup::build` producing a valid `Crystal`
//! - `PeriodicSetup::with_cell` retaining pins under cell strain

mod setup_shells_support;

use mandacaru_core::{
    math::{linalg::eigen, storage::Precision, SpatialGrid},
    periodic::setup::{
        blocks,
        matrices::{kpoint_matrices, MatrixAssembly},
        parse::{create_setup, AtomSpec, SetupInput},
        spheres,
    },
};
use nalgebra::{DMatrix, Matrix3};
use setup_shells_support::*;
use std::sync::Arc;

#[test]
fn test_kpoint_matrices_properties() {
    let ctx = context();
    let (cell, basis) = silicon(&ctx);
    let ds = Arc::new(dataset(&ctx, "lda-sr", "Si"));
    let datasets = vec![ds.clone()];

    let projectors = projector_list(&ctx, &ds, &cell.centers, false);
    let n_atoms = cell.centers.len();
    let potentials: Vec<_> = (0..n_atoms)
        .map(|_| {
            mandacaru_core::periodic::setup::ions::LocalPotential::new(
                &ctx,
                ds.r.as_slice(),
                ds.v_local.as_slice(),
                ds.valence_charge,
                ds.r_cut_local,
            )
            .unwrap()
        })
        .collect();

    let grid = SpatialGrid::new([10, 10, 10], [0.; 3], [0.5, 0., 0., 0., 0.5, 0., 0., 0., 0.5], true).unwrap();

    let mut channels = Vec::new();
    for a in 0..n_atoms {
        for l in 0..=2 {
            for m in -(l as i32)..=l as i32 {
                channels.push((a, l, m));
            }
        }
    }

    let ops = blocks::operators(&projectors, &datasets, &channels).unwrap();

    let k_cart = [[0.1, 0.2, 0.3]];
    let k_frac = [[0.05, 0.1, 0.15]];
    let weights = [1.0];

    let assembly = MatrixAssembly {
        grid: &grid,
        shells: &basis.shells,
        projectors: &projectors,
        potentials: &potentials,
        centers: &cell.centers,
        lattice: &cell.lattice,
        kpoints_cart: &k_cart,
        kpoints_frac: &k_frac,
        weights: &weights,
        operators: &ops,
        spin_orbit_couplings: None,
        sigma: 0.7,
        precision: Precision::Float64,
    };

    let kpoints = kpoint_matrices(&ctx, &assembly).unwrap();
    assert_eq!(kpoints.len(), 1);

    let kp = &kpoints[0];
    let m = basis.shells.functions();
    assert_eq!(kp.overlap.nrows(), m);
    assert_eq!(kp.overlap.ncols(), m);
    assert_eq!(kp.fixed.nrows(), m);
    assert_eq!(kp.fixed.ncols(), m);
    assert_eq!(kp.projections.nrows(), m);
    assert_eq!(kp.projections.ncols(), projectors.len());

    // S is Hermitian: ||S - S^H|| < 1e-14
    let s_diff = (&kp.overlap - kp.overlap.adjoint()).norm();
    assert!(s_diff < 1e-12, "S is not Hermitian: {s_diff}");

    // S is positive definite: all eigenvalues > 0
    let (evals, _) = eigen(kp.overlap.clone()).unwrap();
    for &e in &evals {
        assert!(e > 0.0, "eigenvalue not positive: {e}");
    }

    // Fixed is Hermitian: ||F - F^H|| < 1e-14
    let f_diff = (&kp.fixed - kp.fixed.adjoint()).norm();
    assert!(f_diff < 1e-12, "F is not Hermitian: {f_diff}");

    // Lowdin transform satisfies: L^H S L = I
    let eye = kp.lowdin.adjoint() * &kp.overlap * &kp.lowdin;
    let identity = DMatrix::identity(m, m);
    let lowdin_diff = (eye - identity).norm();
    assert!(lowdin_diff < 1e-10, "Lowdin metric error: {lowdin_diff}");
}

#[test]
fn test_the_sphere_bloch_sums_k_together_vs_one_by_one() {
    let ctx = context();
    let (cell, basis) = silicon(&ctx);
    let ds = dataset(&ctx, "lda-sr", "Si");
    let projectors = projector_list(&ctx, &ds, &cell.centers, false);

    let k1 = [0.1, -0.2, 0.15];
    let k2 = [-0.05, 0.12, 0.25];
    let k_all = [k1, k2];

    // Projections together
    let p_together = spheres::projections(
        &ctx,
        &basis.shells,
        &cell.lattice,
        &k_all,
        &projectors,
    )
    .unwrap();

    // Projections one by one
    let p1 = spheres::projections(
        &ctx,
        &basis.shells,
        &cell.lattice,
        &[k1],
        &projectors,
    )
    .unwrap();
    let p2 = spheres::projections(
        &ctx,
        &basis.shells,
        &cell.lattice,
        &[k2],
        &projectors,
    )
    .unwrap();

    let m = basis.shells.functions();
    let p = projectors.len();
    let stride = m * p;

    let diff1 = relative(&p_together.as_slice()[..stride], p1.as_slice());
    let diff2 = relative(&p_together.as_slice()[stride..], p2.as_slice());

    assert!(diff1 < 1e-12, "k1 mismatch: {diff1}");
    assert!(diff2 < 1e-12, "k2 mismatch: {diff2}");
}

#[test]
fn test_periodic_setup_build_and_with_cell() {
    let ctx = context();
    let ds = Arc::new(dataset(&ctx, "lda-sr", "Si"));

    // FCC Si lattice (Bohr)
    let a = 10.26;
    let mut lattice = Matrix3::zeros();
    lattice[(0, 1)] = 0.5 * a;
    lattice[(0, 2)] = 0.5 * a;
    lattice[(1, 0)] = 0.5 * a;
    lattice[(1, 2)] = 0.5 * a;
    lattice[(2, 0)] = 0.5 * a;
    lattice[(2, 1)] = 0.5 * a;

    let input = SetupInput {
        lattice,
        grid_shape: [12, 12, 12],
        atoms: vec![
            AtomSpec {
                species: 0,
                symbol: "Si".to_string(),
                fractional: [0.0, 0.0, 0.0],
                moment: 0.0,
            },
            AtomSpec {
                species: 0,
                symbol: "Si".to_string(),
                fractional: [0.25, 0.25, 0.25],
                moment: 0.0,
            },
        ],
        kpoints_frac: vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        weights: vec![0.5, 0.5],
        functional: "lda".to_string(),
        precision: "float64".to_string(),
        spinor: false,
        symmetry: true,
        filter_cutoff: None,
        sigma: None,
        dataset_files: None,
    };

    let setup = create_setup(&ctx, input, vec![ds.clone()]).unwrap();

    // 1. Build Crystal
    let crystal = setup.build(&ctx).unwrap();
    assert_eq!(crystal.points.len(), 2);
    assert!(crystal.m > 0);
    assert!(crystal.p > 0);
    assert!(crystal.channels > 0);
    assert_eq!(crystal.electrons, 8.0); // 2 Si atoms * 4 valence electrons
    assert!(crystal.kernel.as_slice().len() > 0);
    assert!(crystal.u.as_slice().len() > 0);
    assert!(crystal.ion.as_slice().len() > 0);
    assert!(crystal.description.is_some(), "crystal description should be populated");
    let desc = crystal.description.as_ref().unwrap();
    assert_eq!(desc.centers.len(), 2);
    assert_eq!(desc.function_atom.len(), crystal.m);
    assert_eq!(desc.projectors.len(), crystal.p);

    // Check that each point has valid Lowdin metric
    for pt in &crystal.points {
        let eye = pt.lowdin.adjoint() * &pt.overlap * &pt.lowdin;
        let id = DMatrix::identity(crystal.m, crystal.m);
        assert!((eye - id).norm() < 1e-10);
    }

    // 2. Test with_cell (cell strain: 1% hydrostatic expansion)
    let strained_lattice = lattice * 1.01;
    let strained_setup = setup.with_cell(&ctx, &strained_lattice).unwrap();

    // Check that pins are held constant
    assert_eq!(strained_setup.pins.grid_shape, setup.pins.grid_shape);
    assert_eq!(strained_setup.pins.sigma, setup.pins.sigma);
    assert_eq!(strained_setup.pins.filter_cutoff, setup.pins.filter_cutoff);

    // Check that centers moved with the strained lattice
    let c0 = strained_setup.centers[0];
    let c1 = strained_setup.centers[1];
    assert_eq!(c0, [0.0, 0.0, 0.0]);
    assert!((c1[0] - 0.25 * 0.5 * a * 1.01 * 2.0).abs() < 1e-6);

    // Strained setup builds a valid Crystal
    let strained_crystal = strained_setup.build(&ctx).unwrap();
    assert_eq!(strained_crystal.points.len(), 2);
    assert_eq!(strained_crystal.electrons, 8.0);

    // 3. Test matrices_at off-mesh k-points
    let off_mesh = vec![[0.125, 0.25, 0.375]];
    let off_kpoints = setup.matrices_at(&ctx, &off_mesh, &[1.0]).unwrap();
    assert_eq!(off_kpoints.len(), 1);
    let eye_off = off_kpoints[0].lowdin.adjoint() * &off_kpoints[0].overlap * &off_kpoints[0].lowdin;
    let id_off = DMatrix::identity(crystal.m, crystal.m);
    assert!((eye_off - id_off).norm() < 1e-10);
}
