//! The analytic force of a crystal for the functionals beyond LDA/PBE/HSE06, against reconverged
//! finite differences, unpolarized and spin-polarized: the meta-GGA r2SCAN (MG-F5) and the rVV10
//! nonlocal correlation (MG-F6).
//!
//! For a meta-GGA the engine adds two terms to the semilocal force:
//! - P6, the row-center derivative of `T = 1/2 <grad chi | v_tau | grad chi>`;
//! - E6, the partial core's kinetic density moving with its atom.
//! rVV10 adds none. Its kernel depends on the positions only through the density, and its
//! partials join the semilocal ones in `v_xc`, so the Local and core terms carry it.
//! The oracle is the Richardson difference of free energies reconverged at displaced positions,
//! built by the same Rust setup on the same grid pins.

mod setup_shells_support;

use mandacaru_core::periodic::{
    response,
    setup::parse::{create_setup, AtomSpec, SetupInput},
    solver::Options,
};
use nalgebra::Matrix3;
use setup_shells_support::*;
use std::sync::Arc;

const TO_EV_PER_A: f64 = 27.211386245988 / 0.529177210903;
/// The engine gate of the forces lane, in Ha/Bohr.
const GATE: f64 = 1e-6;

#[derive(Clone, Copy)]
struct Case {
    label: &'static str,
    symbols: [&'static str; 2],
    fractional: [[f64; 3]; 2],
    moments: [f64; 2],
    spin: bool,
    functional: &'static str,
    /// Spin-orbit coupling: a spinor crystal on the fully relativistic set.
    spinor: bool,
    /// The larger finite-difference step (Bohr); the oracle extrapolates it and its half. Gold's
    /// r2SCAN energy has a large third derivative, so its steps must be small: a step scan of AuH
    /// (`step_scan`) shows the central difference O(h^2) only below about 2.5e-4 Bohr; AuH gates use
    /// 1.25e-4 (and its half), where the Richardson residual is 3e-7 Ha/Bohr.
    step: f64,
}

impl Case {
    fn library(&self) -> &'static str {
        if self.spinor { "pbe-fr" } else { "pbe-sr" }
    }
}

fn input(case: &Case, files: Vec<(String, String)>) -> SetupInput {
    SetupInput {
        mesh: None,
        recipes: None,
        projector_basis: "raw".into(),
        // A skew cell (columns are the lattice vectors): no symmetry zeroes a force component.
        lattice: Matrix3::from_row_slice(&[8., 1., 0.5, 0., 9., 1., 0., 0., 10.]),
        grid_shape: [8, 9, 10],
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
        filter_cutoff: None,
        sigma: None,
        dataset_files: Some(files),
        hubbard_u: None,
        charge: None,
    }
}

fn check(case: &Case) -> f64 {
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
    let base = input(case, files);
    let options = Options {
        tolerance: 1e-12,
        density_tolerance: 1e-10,
        ..Options::default()
    };
    let crystal = create_setup(&ctx, base.clone(), datasets.clone()).unwrap().build(&ctx).unwrap();
    assert_eq!(crystal.spinor, case.spinor, "{}: the crystal's spinor flag", case.label);
    let result = crystal.solve(&options).unwrap();
    assert!(result.converged, "{}: the reference SCF did not converge", case.label);
    let forces = response::forces(&crystal, &result).unwrap();
    // Atom 0, Cartesian x: a fractional step along a1 = (8, 0, 0) Bohr.
    let energy = |h: f64| {
        let mut displaced = base.clone();
        displaced.atoms[0].fractional[0] += h / 8.;
        let moved = create_setup(&ctx, displaced, datasets.clone()).unwrap().build(&ctx).unwrap();
        let state = moved.solve(&options).unwrap();
        assert!(state.converged, "{}: a displaced SCF did not converge", case.label);
        state.free_energy
    };
    let h = case.step;
    let coarse = -(energy(h) - energy(-h)) / (2. * h);
    let fine = -(energy(0.5 * h) - energy(-0.5 * h)) / h;
    let oracle = (4. * fine - coarse) / 3.;
    let miss = (forces[0][0] - oracle).abs();
    eprintln!(
        "{}: force {:+.10e}, oracle {oracle:+.10e} (central {coarse:+.10e} / {fine:+.10e}), miss {miss:.2e} Ha/Bohr = {:.2e} eV/A, {:.0} s",
        case.label,
        forces[0][0],
        miss * TO_EV_PER_A,
        started.elapsed().as_secs_f64()
    );
    assert!(oracle.abs() > 100. * GATE, "{}: the oracle {oracle:e} is not above its gate", case.label);
    // The meta-GGA terms are not vacuous: dropping P6 alone would miss the oracle by far more
    // than the gate (E6 is reported; it vanishes for a species without a partial core).
    let breakdown = response::breakdown(&crystal, &result).unwrap();
    let (p6, e6) = (breakdown.kinetic_density[0][0], breakdown.core_kinetic[0][0]);
    eprintln!("{}: P6 kinetic_density {p6:+.3e}, E6 core_kinetic {e6:+.3e} Ha/Bohr (atom 0, x)", case.label);
    if case.functional.starts_with("r2scan") {
        assert!(p6.abs() > 100. * GATE, "{}: P6 {p6:e} is too small to be tested", case.label);
    } else {
        assert_eq!((p6, e6), (0., 0.), "{}: a GGA has no meta-GGA term", case.label);
    }
    assert!(miss < GATE, "{}: {miss:e} against {GATE:e}", case.label);
    forces[0][0]
}

/// The analytic force of atom 0 along x, without the oracle (for a comparison of two models).
fn force_of(case: &Case) -> f64 {
    let ctx = context();
    let mut species: Vec<&str> = vec![case.symbols[0]];
    if case.symbols[1] != case.symbols[0] {
        species.push(case.symbols[1]);
    }
    let datasets: Vec<_> = species.iter().map(|s| Arc::new(dataset(&ctx, case.library(), s))).collect();
    // Forces read the datasets again, so the request carries their files and digests.
    let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap()).join(case.library());
    let files: Vec<_> = species
        .iter()
        .map(|symbol| {
            let path = root.join(format!("{symbol}.parquet"));
            let digest = mandacaru_core::paw::digest::sha256_file(&path).unwrap();
            (path.to_string_lossy().into_owned(), digest)
        })
        .collect();
    let crystal = create_setup(&ctx, input(case, files), datasets).unwrap().build(&ctx).unwrap();
    let options = Options { tolerance: 1e-11, density_tolerance: 1e-9, ..Options::default() };
    let result = crystal.solve(&options).unwrap();
    assert!(result.converged);
    response::forces(&crystal, &result).unwrap()[0][0]
}

#[test]
#[ignore = "slow: five r2SCAN crystal SCFs; run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_xc -- --ignored"]
fn an_r2scan_crystal_force_is_the_reconverged_finite_difference() {
    check(&Case {
        label: "LiH r2SCAN",
        symbols: ["Li", "H"],
        fractional: [[0.1, 0.2, 0.3], [0.55, 0.6, 0.7]],
        moments: [0., 0.],
        spin: false,
        functional: "r2scan",
        spinor: false,
        step: 2e-3,
    });
}

#[test]
#[ignore = "slow: five spin-polarized r2SCAN crystal SCFs; run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_xc -- --ignored"]
fn a_spin_polarized_r2scan_crystal_force_is_the_reconverged_finite_difference() {
    check(&Case {
        label: "H2 ferromagnetic UKS-r2SCAN",
        symbols: ["H", "H"],
        fractional: [[0.1, 0.2, 0.3], [0.45, 0.5, 0.55]],
        moments: [1., 1.],
        spin: true,
        functional: "r2scan",
        spinor: false,
        step: 2e-3,
    });
}

/// The rVV10 energy against an rVV10-free one at the same geometry: the nonlocal term is present
/// (a gate on a functional that silently dropped it would pass for the plain GGA).
fn nonlocal_is_present(case: &Case) {
    let mut plain = Case { ..*case };
    plain.functional = if case.functional.starts_with("r2scan") { "r2scan" } else { "pbe" };
    let energy = |c: &Case| {
        let ctx = context();
        let mut species: Vec<&str> = vec![c.symbols[0]];
        if c.symbols[1] != c.symbols[0] {
            species.push(c.symbols[1]);
        }
        let datasets: Vec<_> = species.iter().map(|s| Arc::new(dataset(&ctx, c.library(), s))).collect();
        let mut plain_input = input(c, Vec::new());
        plain_input.dataset_files = None; // an energy comparison: no provenance needed
        let crystal = create_setup(&ctx, plain_input, datasets).unwrap().build(&ctx).unwrap();
        crystal.solve(&Options::default()).unwrap().free_energy
    };
    let (with, without) = (energy(case), energy(&plain));
    eprintln!("{}: E - E(no rVV10) = {:+.6e} Ha", case.label, with - without);
    assert!((with - without).abs() > 1e-4, "{}: the rVV10 energy is missing", case.label);
}

#[test]
#[ignore = "slow: five PBE+rVV10 crystal SCFs; run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_xc -- --ignored"]
fn a_pbe_rvv10_crystal_force_is_the_reconverged_finite_difference() {
    let case = Case {
        label: "LiH PBE+rVV10",
        symbols: ["Li", "H"],
        fractional: [[0.1, 0.2, 0.3], [0.55, 0.6, 0.7]],
        moments: [0., 0.],
        spin: false,
        functional: "pbe+rvv10",
        spinor: false,
        step: 2e-3,
    };
    nonlocal_is_present(&case);
    check(&case);
}

#[test]
#[ignore = "slow: five spin-polarized PBE+rVV10 crystal SCFs; run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_xc -- --ignored"]
fn a_spin_polarized_pbe_rvv10_crystal_force_is_the_reconverged_finite_difference() {
    check(&Case {
        label: "H2 ferromagnetic UKS-PBE+rVV10",
        symbols: ["H", "H"],
        fractional: [[0.1, 0.2, 0.3], [0.45, 0.5, 0.55]],
        moments: [1., 1.],
        spin: true,
        functional: "pbe+rvv10",
        spinor: false,
        step: 2e-3,
    });
}

#[test]
#[ignore = "slow: five r2SCAN+rVV10 crystal SCFs; run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_xc -- --ignored"]
fn an_r2scan_rvv10_crystal_force_is_the_reconverged_finite_difference() {
    let case = Case {
        label: "LiH r2SCAN+rVV10",
        symbols: ["Li", "H"],
        fractional: [[0.1, 0.2, 0.3], [0.55, 0.6, 0.7]],
        moments: [0., 0.],
        spin: false,
        functional: "r2scan+rvv10",
        spinor: false,
        step: 2e-3,
    };
    nonlocal_is_present(&case);
    check(&case);
}

#[test]
#[ignore = "slow: scalar r2SCAN AuH crystal SCFs; run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_xc -- --ignored a_heavy_r2scan"]
fn a_heavy_r2scan_crystal_force_is_the_reconverged_finite_difference() {
    // Gold's semicore and large partial core make E6 and P6 heavy; the SOC case below is this
    // geometry with the spin-orbit term.
    check(&Case {
        label: "AuH r2SCAN",
        symbols: ["Au", "H"],
        fractional: [[0.1, 0.2, 0.3], [0.35, 0.4, 0.45]],
        moments: [0., 0.],
        spin: false,
        functional: "r2scan",
        spinor: false,
        step: 1.25e-4,
    });
}

#[test]
#[ignore = "slow: spin-orbit r2SCAN crystal SCFs; run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_xc -- --ignored a_spin_orbit_r2scan"]
fn a_spin_orbit_r2scan_crystal_force_is_the_reconverged_finite_difference() {
    // F5c: the spinor meta-GGA in the local spin frame (tau_up/dn = (tau_0 +- m_hat . tau_vec) / 2),
    // P6 per Pauli channel of v_tau, E6 against its charge channel. AuH: gold's 5d/6p channels
    // carry the spin-orbit term.
    let case = Case {
        label: "AuH SOC r2SCAN",
        symbols: ["Au", "H"],
        fractional: [[0.1, 0.2, 0.3], [0.35, 0.4, 0.45]],
        moments: [0., 0.],
        spin: false,
        functional: "r2scan",
        spinor: true,
        step: 1.25e-4,
    };
    let soc = check(&case);
    let scalar = force_of(&Case { spinor: false, ..case });
    eprintln!("{}: F(SOC) - F(scalar) = {:+.3e} Ha/Bohr", case.label, soc - scalar);
    assert!((soc - scalar).abs() > 100. * GATE, "{}: the spin-orbit term is not tested", case.label);
}

#[test]
#[ignore = "slow: five spin-orbit PBE+rVV10 crystal SCFs; run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_xc -- --ignored a_spin_orbit_pbe_rvv10"]
fn a_spin_orbit_pbe_rvv10_crystal_force_is_the_reconverged_finite_difference() {
    // F6c: the spinor XC evaluates rVV10 on the charge in the local spin frame; the spinor force
    // contracts the Pauli channels of that v_xc, so rVV10 again needs no term of its own.
    // AuH: gold's 5d and 6p channels make the spin-orbit term large (Li and H have only s
    // channels, where it vanishes, so a LiH case would not test SOC at all).
    let case = Case {
        label: "AuH SOC PBE+rVV10",
        symbols: ["Au", "H"],
        fractional: [[0.1, 0.2, 0.3], [0.35, 0.4, 0.45]],
        moments: [0., 0.],
        spin: false,
        functional: "pbe+rvv10",
        spinor: true,
        step: 1.25e-4,
    };
    nonlocal_is_present(&case);
    let soc = check(&case);
    let scalar = force_of(&Case { spinor: false, ..case });
    eprintln!("{}: F(SOC) - F(scalar) = {:+.3e} Ha/Bohr", case.label, soc - scalar);
    assert!((soc - scalar).abs() > 100. * GATE, "{}: the spin-orbit term is not tested", case.label);
}

/// A step scan of the central difference of atom 0 along x: whether the oracle of a case is in its
/// asymptotic range (a diagnostic; it asserts nothing).
#[test]
#[ignore = "diagnostic: a step scan of reconverged SCFs; run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_xc -- --ignored --nocapture step_scan"]
fn step_scan() {
    use mandacaru_core::paw::digest::sha256_file;
    for spinor in [false, true] {
        let case = Case {
            label: if spinor { "AuH SOC r2SCAN" } else { "AuH r2SCAN" },
            symbols: ["Au", "H"],
            fractional: [[0.1, 0.2, 0.3], [0.35, 0.4, 0.45]],
            moments: [0., 0.],
            spin: false,
            functional: "r2scan",
            spinor,
            step: 1.25e-4,
        };
        let ctx = context();
        let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap()).join(case.library());
        let files: Vec<_> = ["Au", "H"]
            .iter()
            .map(|s| {
                let path = root.join(format!("{s}.parquet"));
                (path.to_string_lossy().into_owned(), sha256_file(&path).unwrap())
            })
            .collect();
        let datasets: Vec<_> = ["Au", "H"].iter().map(|s| Arc::new(dataset(&ctx, case.library(), s))).collect();
        let base = input(&case, files);
        let options = Options { tolerance: 1e-12, density_tolerance: 1e-10, ..Options::default() };
        let energy = |h: f64| {
            let mut displaced = base.clone();
            displaced.atoms[0].fractional[0] += h / 8.;
            let moved = create_setup(&ctx, displaced, datasets.clone()).unwrap().build(&ctx).unwrap();
            let state = moved.solve(&options).unwrap();
            assert!(state.converged);
            state.free_energy
        };
        let crystal = create_setup(&ctx, base.clone(), datasets.clone()).unwrap().build(&ctx).unwrap();
        let result = crystal.solve(&options).unwrap();
        let analytic = response::forces(&crystal, &result).unwrap()[0][0];
        for h in [1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5] {
            let central = -(energy(h) - energy(-h)) / (2. * h);
            eprintln!("{}: h {h:.1e} central {central:+.10e}, analytic {analytic:+.10e}, diff {:+.2e}", case.label, central - analytic);
        }
    }
}


#[test]
#[ignore = "slow: 12 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_xc -- --ignored a_magnetic_spinor_r2scan"]
fn a_magnetic_spinor_r2scan_crystal_force_is_the_reconverged_finite_difference() {
    // Four live Pauli channels of v_tau (no time reversal; the SCF passed only the charge block
    // before 2026-11-09). Hydrogen has only s channels, so the spin-orbit term vanishes, the moment
    // stays collinear and the force is the spin-polarized one.
    let case = Case {
        label: "H2 ferromagnetic spinor r2SCAN",
        symbols: ["H", "H"],
        fractional: [[0.1, 0.2, 0.3], [0.45, 0.5, 0.55]],
        moments: [1., 1.],
        spin: false,
        functional: "r2scan",
        spinor: true,
        step: 2e-3,
    };
    let spinor = check(&case);
    let collinear = force_of(&Case { spin: true, spinor: false, ..case });
    eprintln!("{}: F(spinor) - F(UKS) = {:+.3e} Ha/Bohr", case.label, spinor - collinear);
    assert!((spinor - collinear).abs() < GATE, "{}: the spinor force is not the UKS one", case.label);
}
