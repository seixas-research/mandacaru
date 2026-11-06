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
fn fused_projector_products_match_general_complex_matrix_products() {
    use mandacaru_core::periodic::setup::matrices::projector_products;
    use num_complex::Complex64;
    let ctx = context();
    let (m, p) = (3, 4);
    let c = DMatrix::from_fn(m, p, |i, j| Complex64::new((i + 2 * j) as f64 * 0.13, (2 * i + j) as f64 * 0.07));
    let q = DMatrix::from_fn(p, p, |i, j| Complex64::new((i + j) as f64 * 0.11, (i as f64 - j as f64) * 0.03));
    let d = DMatrix::from_fn(p, p, |i, j| Complex64::new((i * j) as f64 * 0.05, (i + j) as f64 * 0.04));
    let (actual_q, actual_d) = projector_products(&ctx, m, p,
        |i, j| c[(i, j)], |i, j| q[(i, j)], |i, j| d[(i, j)]).unwrap();
    assert!((&actual_q - &c * &q * c.adjoint()).norm() < 1e-12);
    assert!((&actual_d - &c * &d * c.adjoint()).norm() < 1e-12);
    let hermitian_q = (&q + q.adjoint()) * Complex64::new(0.5, 0.);
    let hermitian_d = (&d + d.adjoint()) * Complex64::new(0.5, 0.);
    let (actual_q, actual_d) = projector_products(&ctx, m, p,
        |i, j| c[(i, j)], |i, j| hermitian_q[(i, j)], |i, j| hermitian_d[(i, j)]).unwrap();
    assert!((&actual_q - &c * &hermitian_q * c.adjoint()).norm() < 1e-12);
    assert!((&actual_d - &c * &hermitian_d * c.adjoint()).norm() < 1e-12);
    let mut sparse_q = DMatrix::zeros(p, p);
    let mut sparse_d = DMatrix::zeros(p, p);
    sparse_q[(0, p - 1)] = Complex64::new(0.4, -0.2);
    sparse_d[(p - 1, 0)] = Complex64::new(-0.3, 0.7);
    let (actual_q, actual_d) = projector_products(&ctx, m, p,
        |i, j| c[(i, j)], |i, j| sparse_q[(i, j)], |i, j| sparse_d[(i, j)]).unwrap();
    assert!((&actual_q - &c * &sparse_q * c.adjoint()).norm() < 1e-12);
    assert!((&actual_d - &c * &sparse_d * c.adjoint()).norm() < 1e-12);
    let (empty_q, empty_d) = projector_products(&ctx, m, 0, |_, _| unreachable!(),
        |_, _| unreachable!(), |_, _| unreachable!()).unwrap();
    assert_eq!(empty_q.norm(), 0.);
    assert_eq!(empty_d.norm(), 0.);
}

#[test]
fn tiled_projector_products_match_general_complex_matrix_products() {
    use mandacaru_core::periodic::setup::matrices::projector_products;
    use num_complex::Complex64;
    let ctx = context();
    let (m, p) = (32, 96);
    let c = DMatrix::from_fn(m, p, |i, j| Complex64::new(((i + 2*j) as f64 * 0.13).sin(), ((2*i + j) as f64 * 0.07).cos()));
    let q = DMatrix::from_fn(p, p, |i, j| if i / 3 == j / 3 { Complex64::new((i+j) as f64 * 0.01, (i as f64-j as f64) * 0.003) } else { Complex64::new(0., 0.) });
    let d = DMatrix::from_fn(p, p, |i, j| if i == j || (i == 0 && j == p-1) { Complex64::new((i+j) as f64 * 0.005, (i+j) as f64 * 0.002) } else { Complex64::new(0., 0.) });
    let (actual_q, actual_d) = projector_products(&ctx, m, p,
        |i, j| c[(i, j)], |i, j| q[(i, j)], |i, j| d[(i, j)]).unwrap();
    assert!(ctx.memory_peak() < 2 * m * p * std::mem::size_of::<Complex64>());
    assert!((&actual_q - &c * &q * c.adjoint()).norm() < 1e-10);
    assert!((&actual_d - &c * &d * c.adjoint()).norm() < 1e-10);
}

#[test]
fn test_kpoint_matrices_properties() {
    let ctx = context();
    let (cell, basis) = silicon(&ctx);
    let ds = Arc::new(dataset(&ctx, "lda-sr", "Si"));
    let datasets = vec![ds.clone(); cell.centers.len()];

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

    let mut assembly = MatrixAssembly {
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
        spin_orbit_projectors: None,
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

    // The archived and narrowed paths retain the same assembly matrices; the
    // samples differ only by their requested storage precision.
    let archived_ctx = mandacaru_core::Context::with_offload(Some(1), 1 << 30, 1).unwrap();
    for precision in [Precision::Float64, Precision::Float32] {
        assembly.precision = precision;
        let stored = kpoint_matrices(&archived_ctx, &assembly).unwrap();
        assert!(stored[0].psi.archived());
        assert!((&stored[0].overlap - &kp.overlap).norm() < 1e-12);
        assert!((&stored[0].fixed - &kp.fixed).norm() < 1e-12);
        let reference = kp.psi.read(&ctx).unwrap();
        let actual = stored[0].psi.read(&archived_ctx).unwrap();
        let tolerance = if precision == Precision::Float32 { 1e-7 } else { 0. };
        for (got, want) in actual.iter().zip(reference.iter()) {
            assert!((got - want).norm() <= tolerance);
        }
    }
    assembly.precision = Precision::Float32;
    let narrowed = kpoint_matrices(&ctx, &assembly).unwrap();
    assert!(!narrowed[0].psi.archived());
    assert!((&narrowed[0].fixed - &kp.fixed).norm() < 1e-12);
}

#[test]
fn skew_lih_native_setup_response_matches_active_setup() {
    use mandacaru_core::{paw::digest::sha256_file, periodic::{response, solver::Options}};
    let ctx = context();
    let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap())
        .join("lda-sr");
    let files: Vec<_> = ["Li", "H"].iter().map(|symbol| {
        let path = root.join(format!("{symbol}.parquet"));
        (path.to_string_lossy().into_owned(), sha256_file(&path).unwrap())
    }).collect();
    let input = SetupInput {
        lattice: Matrix3::from_row_slice(&[8., 1., 0.5, 0., 9., 1., 0., 0., 10.]),
        grid_shape: [8, 9, 10],
        atoms: vec![
            AtomSpec { species: 1, symbol: "Li".into(), fractional: [0.1, 0.2, 0.3], moment: 0. },
            AtomSpec { species: 0, symbol: "H".into(), fractional: [0.55, 0.6, 0.7], moment: 0. },
        ],
        kpoints_frac: vec![[-0.25, 0., 0.], [0.25, 0., 0.]],
        weights: vec![0.5, 0.5], functional: "lda".into(), precision: "float64".into(),
        basis_size: "SZ".into(), spinor: false, spin: false, symmetry: false,
        filter_cutoff: None, sigma: None, dataset_files: Some(files), hubbard_u: None,
    };
    let datasets = vec![Arc::new(dataset(&ctx, "lda-sr", "H")),
        Arc::new(dataset(&ctx, "lda-sr", "Li"))];
    let setup = create_setup(&ctx, input.clone(), datasets.clone()).unwrap();
    let mut crystal = setup.build(&ctx).unwrap();
    let options = Options { tolerance: 1e-11, density_tolerance: 1e-8, ..Options::default() };
    let result = crystal.solve(&options).unwrap();
    assert!(result.converged);
    // Only this test bypasses the capability gate; production responses remain refused.
    crystal.description_unavailable = None;
    let forces = response::forces(&crystal, &result).unwrap();
    let expected = [[-0.0015693448701201258, -0.00041682012015089383, 0.005480404682238612],
        [-0.0011635448014300955, -0.00167223430420251, -0.004134832194651231]];
    for (got, want) in forces.iter().flatten().zip(expected.iter().flatten()) {
        assert!((got - want).abs() < 1e-7, "force {got} differs from active setup {want}");
    }
    let stress = response::stress(&crystal, &result).unwrap();
    let expected_stress = [[1.11316890e-4, 2.08964592e-6, 1.31574895e-6],
        [2.08964592e-6, 1.78219533e-4, -2.84853462e-6],
        [1.31574895e-6, -2.84853462e-6, 2.12973502e-4]];
    for (got, want) in stress.sigma.iter().flatten().zip(expected_stress.iter().flatten()) {
        assert!((got - want).abs() < 1e-10, "stress {got} differs from active setup {want}");
    }
    // Displace Li in Cartesian x with the lattice and all grid pins fixed.
    let h = 2e-3;
    let mut displaced_energy = [0.; 2];
    for (energy, sign) in displaced_energy.iter_mut().zip([-1., 1.]) {
        let mut displaced = input.clone();
        displaced.atoms[0].fractional[0] += sign * h / 8.;
        let moved = create_setup(&ctx, displaced, datasets.clone()).unwrap().build(&ctx).unwrap();
        let state = moved.solve(&options).unwrap();
        assert!(state.converged);
        *energy = state.free_energy;
    }
    let finite_difference = -(displaced_energy[1] - displaced_energy[0]) / (2. * h);
    assert!((forces[0][0] - finite_difference).abs() < 5e-5,
        "analytic force {} differs from finite difference {finite_difference}", forces[0][0]);
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
        basis_size: "SZ".to_string(),
        spinor: false,
        spin: false,
        symmetry: true,
        filter_cutoff: None,
        sigma: None,
        dataset_files: None,
        hubbard_u: None,
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
    assert!(crystal.description_unavailable.as_deref().unwrap().contains("parity"));
    // Each identical atom must receive its own copy of the nonzero monopole
    // operator, even though the setup stores only one Si dataset by species.
    let monopole0 = setup.channels.iter().position(|&c| c == (0, 0, 0)).unwrap();
    let monopole1 = setup.channels.iter().position(|&c| c == (1, 0, 0)).unwrap();
    let first: Vec<_> = setup.projectors.iter().enumerate()
        .filter(|(_, p)| p.atom == 0).map(|(i, _)| i).collect();
    let second: Vec<_> = setup.projectors.iter().enumerate()
        .filter(|(_, p)| p.atom == 1).map(|(i, _)| i).collect();
    assert_eq!(first.len(), second.len());
    let mut largest = 0f64;
    for (&i, &ii) in first.iter().zip(&second) {
        for (&j, &jj) in first.iter().zip(&second) {
            let a = crystal.blocks.as_slice()[(monopole0 * crystal.p + i) * crystal.p + j];
            let b = crystal.blocks.as_slice()[(monopole1 * crystal.p + ii) * crystal.p + jj];
            largest = largest.max(a.norm());
            assert!((a - b).norm() < 1e-12);
        }
    }
    assert!(largest > 1e-10, "the monopole block must not vanish");
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

    // A mesh point rebuilt through the off-mesh entry point uses the same
    // per-atom operators as the original crystal assembly.
    let rebuilt = setup.matrices_at(&ctx, &[[0., 0., 0.]], &[1.0]).unwrap();
    assert!((&rebuilt[0].overlap - &crystal.points[0].overlap).norm() < 1e-10);
    assert!((&rebuilt[0].fixed - &crystal.points[0].fixed).norm() < 1e-10);
}

#[test]
fn polarized_hydrogen_setup_builds_crystal_matrices() {
    let ctx = context();
    let ds = Arc::new(dataset(&ctx, "lda-sr", "H"));
    let input = serde_json::json!({
        "lattice": [8.,0.,0., 0.,8.,0., 0.,0.,8.],
        "grid_shape": [8,8,8],
        "atoms": [{"species":0,"symbol":"H","fractional":[0.,0.,0.]}],
        "basis_size": "DZP", "symmetry": false
    });
    let setup = create_setup(&ctx, mandacaru_core::periodic::setup::parse_input_json(&input).unwrap(), vec![ds]).unwrap();
    assert_eq!(setup.shells.functions(), 5);
    let crystal = setup.build(&ctx).unwrap();
    assert_eq!(crystal.m, 5);
    assert_eq!(crystal.electrons, 1.);
    assert!(crystal.description_unavailable.is_some());
    let point = &crystal.points[0];
    let eye = point.lowdin.adjoint() * &point.overlap * &point.lowdin;
    assert!((eye - DMatrix::identity(5, 5)).norm() < 1e-10);
}
