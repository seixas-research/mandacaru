//! MG-F5b: the analytic stress of a meta-GGA (r2SCAN) crystal against reconverged finite
//! differences of the free energy in the strain.
//!
//! The meta-GGA adds two terms to the semilocal stress:
//! - S-6, the strain of `T_tau = 1/2 <grad chi | v_tau | grad chi>`, with `v_tau` fixed at its nodes;
//! - X-5, the partial core's kinetic density moving with the strained cell.
//! The oracle is the Richardson difference of free energies reconverged at `x -> (1 + eps) x`, in
//! one entry `eps_ab` at a time (atoms in fractional coordinates and fractional k-points carried,
//! the node counts fixed). The Rust setup is pinned to the reference cell's ion width and filter
//! cutoff, which otherwise follow the strained grid spacing. A PBE control runs the same harness
//! with the engine's established terms only.

mod setup_shells_support;

use mandacaru_core::periodic::{
    response,
    setup::parse::{create_setup, AtomSpec, SetupInput},
    solver::Options,
};
use nalgebra::Matrix3;
use setup_shells_support::*;
use std::sync::Arc;

/// The engine gate of the stress lane on `g_ab = dF/d eps_ab`, in Hartree.
const GATE: f64 = 1e-6;

#[derive(Clone, Copy)]
struct Case {
    label: &'static str,
    symbols: [&'static str; 2],
    fractional: [[f64; 3]; 2],
    moments: [f64; 2],
    spin: bool,
    functional: &'static str,
    spinor: bool,
    /// The larger strain step; the oracle extrapolates it and its half.
    step: f64,
}

impl Case {
    fn library(&self) -> &'static str {
        if self.spinor { "pbe-fr" } else { "pbe-sr" }
    }
}

const LATTICE: [f64; 9] = [8., 1., 0.5, 0., 9., 1., 0., 0., 10.];
const SHAPE: [usize; 3] = [8, 9, 10];

/// The reference cell's pins: `pi / spacing` and `1.4 spacing` of its grid (the setup's defaults).
fn pins() -> (f64, f64) {
    let lattice = Matrix3::from_row_slice(&LATTICE);
    let spacing = (0..3)
        .map(|a| lattice.column(a).norm() / SHAPE[a] as f64)
        .fold(0f64, f64::max);
    (std::f64::consts::PI / spacing, 1.4 * spacing)
}

fn input(case: &Case, files: Vec<(String, String)>, eps: Matrix3<f64>) -> SetupInput {
    let (cutoff, sigma) = pins();
    SetupInput {
        mesh: None,
        recipes: None,
        projector_basis: "raw".into(),
        // A skew cell (columns are the lattice vectors), strained as x -> (1 + eps) x.
        lattice: (Matrix3::identity() + eps) * Matrix3::from_row_slice(&LATTICE),
        grid_shape: SHAPE,
        atoms: (0..2)
            .map(|i| AtomSpec {
                species: if case.symbols[i] == case.symbols[0] { 0 } else { 1 },
                symbol: case.symbols[i].into(),
                fractional: case.fractional[i],
                moment: case.moments[i],
            })
            .collect(),
        kpoints_frac: vec![[-0.25, 0., 0.], [0.25, 0., 0.]],
        weights: vec![0.5, 0.5],
        functional: case.functional.into(),
        precision: "float64".into(),
        basis_size: "SZ".into(),
        spinor: case.spinor,
        spin: case.spin,
        symmetry: false,
        filter_cutoff: Some(cutoff),
        sigma: Some(sigma),
        dataset_files: Some(files),
        hubbard_u: None,
        charge: None,
    }
}

/// The analytic `g_ab` against the reconverged oracle for each entry of `entries`.
fn check(case: &Case, entries: &[(usize, usize)]) {
    use mandacaru_core::paw::digest::sha256_file;
    let started = std::time::Instant::now();
    let ctx = context();
    let mut species: Vec<&str> = vec![case.symbols[0]];
    if case.symbols[1] != case.symbols[0] {
        species.push(case.symbols[1]);
    }
    let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap()).join(case.library());
    let files: Vec<_> = species
        .iter()
        .map(|symbol| {
            let path = root.join(format!("{symbol}.parquet"));
            (path.to_string_lossy().into_owned(), sha256_file(&path).unwrap())
        })
        .collect();
    let datasets: Vec<_> = species.iter().map(|s| Arc::new(dataset(&ctx, case.library(), s))).collect();
    let options = Options {
        tolerance: 1e-12,
        density_tolerance: 1e-10,
        ..Options::default()
    };
    let crystal = create_setup(&ctx, input(case, files.clone(), Matrix3::zeros()), datasets.clone())
        .unwrap()
        .build(&ctx)
        .unwrap();
    assert_eq!(crystal.spinor, case.spinor, "{}: the crystal's spinor flag", case.label);
    let result = crystal.solve(&options).unwrap();
    assert!(result.converged, "{}: the reference SCF did not converge", case.label);
    let stress = response::stress(&crystal, &result).unwrap();
    let b = &stress.breakdown;
    for &(a, c) in entries {
        let energy = |h: f64| {
            let mut eps = Matrix3::zeros();
            eps[(a, c)] = h;
            let strained = create_setup(&ctx, input(case, files.clone(), eps), datasets.clone())
                .unwrap()
                .build(&ctx)
                .unwrap();
            let state = strained.solve(&options).unwrap();
            assert!(state.converged, "{}: a strained SCF did not converge", case.label);
            state.free_energy
        };
        let h = case.step;
        let coarse = (energy(h) - energy(-h)) / (2. * h);
        let fine = (energy(0.5 * h) - energy(-0.5 * h)) / h;
        let oracle = (4. * fine - coarse) / 3.;
        let got = stress.gradient[a][c];
        let miss = (got - oracle).abs();
        eprintln!(
            "{} g_{a}{c}: analytic {got:+.10e}, oracle {oracle:+.10e} (central {coarse:+.10e} / {fine:+.10e}), miss {miss:.2e} Ha; S-6 {:+.3e}, X-5 {:+.3e}; {:.0} s",
            case.label,
            b.kinetic_density[a][c],
            b.xc_core_kinetic[a][c],
            started.elapsed().as_secs_f64()
        );
        // The entry is resolved (a small shear entry of H2 is 76 gates); the meta-GGA terms below
        // carry the 100-gate floor.
        assert!(oracle.abs() > 10. * GATE, "{}: the oracle {oracle:e} is not above its gate", case.label);
        if case.functional.starts_with("r2scan") {
            // S-6 is not vacuous: dropping it alone would miss the oracle by far more than the gate.
            assert!(b.kinetic_density[a][c].abs() > 100. * GATE, "{}: S-6 is too small to be tested", case.label);
        } else {
            assert_eq!((b.kinetic_density[a][c], b.xc_core_kinetic[a][c]), (0., 0.), "{}: a GGA has no meta-GGA term", case.label);
        }
        assert!(miss < GATE, "{} g_{a}{c}: {miss:e} against {GATE:e}", case.label);
    }
}

const LIH: Case = Case {
    label: "LiH",
    symbols: ["Li", "H"],
    fractional: [[0.1, 0.2, 0.3], [0.55, 0.6, 0.7]],
    moments: [0., 0.],
    spin: false,
    functional: "r2scan",
    spinor: false,
    step: 2e-4,
};

#[test]
#[ignore = "slow: 13 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_meta -- --ignored the_pbe_control_stress_is_the_reconverged_finite_difference"]
fn the_pbe_control_stress_is_the_reconverged_finite_difference() {
    check(&Case { label: "LiH PBE", functional: "pbe", ..LIH }, &[(0, 0), (0, 1)]);
}

#[test]
#[ignore = "slow: 15 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_meta -- --ignored an_r2scan_crystal_stress_is_the_reconverged_finite_difference"]
fn an_r2scan_crystal_stress_is_the_reconverged_finite_difference() {
    check(&Case { label: "LiH r2SCAN", ..LIH }, &[(0, 0), (0, 1)]);
}

const H2: Case = Case {
    label: "H2 ferromagnetic",
    symbols: ["H", "H"],
    fractional: [[0.1, 0.2, 0.3], [0.45, 0.5, 0.55]],
    moments: [1., 1.],
    spin: true,
    functional: "r2scan",
    spinor: false,
    step: 2e-4,
};

#[test]
#[ignore = "slow: 8 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_meta -- --ignored a_spin_polarized_r2scan_crystal_stress_is_the_reconverged_finite_difference"]
fn a_spin_polarized_r2scan_crystal_stress_is_the_reconverged_finite_difference() {
    check(&Case { label: "H2 ferromagnetic UKS-r2SCAN", ..H2 }, &[(0, 0), (0, 1)]);
}

#[test]
#[ignore = "slow: 12 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_meta -- --ignored a_magnetic_spinor_r2scan_crystal_stress_is_the_reconverged_finite_difference"]
fn a_magnetic_spinor_r2scan_crystal_stress_is_the_reconverged_finite_difference() {
    // Four live Pauli channels of v_tau (no time reversal): S-6 is their Pauli sum. Hydrogen has
    // only s channels, so the spin-orbit term vanishes and the moment stays collinear.
    check(&Case { label: "H2 ferromagnetic spinor r2SCAN", spin: false, spinor: true, ..H2 }, &[(0, 0), (0, 1)]);
}

#[test]
#[ignore = "slow: 44 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_meta -- --ignored a_spin_orbit_r2scan_crystal_stress_is_the_reconverged_finite_difference"]
fn a_spin_orbit_r2scan_crystal_stress_is_the_reconverged_finite_difference() {
    // Gold's semicore and large partial core make S-6 and X-5 heavy, and its 5d/6p channels carry
    // the spin-orbit term (time-reversal symmetric: the charge channel of v_tau).
    let case = Case {
        label: "AuH SOC r2SCAN",
        symbols: ["Au", "H"],
        fractional: [[0.1, 0.2, 0.3], [0.35, 0.4, 0.45]],
        moments: [0., 0.],
        spin: false,
        functional: "r2scan",
        spinor: true,
        step: 2e-5,
    };
    check(&case, &[(0, 0), (0, 1)]);
}
