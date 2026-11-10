//! AS-X: the analytic stress of an rVV10 crystal (PBE+rVV10, r2SCAN+rVV10) against reconverged
//! finite differences of the free energy in the strain.
//!
//! The nonlocal correlation adds one term to the semilocal stress, X-6: the kernel's wave numbers
//! `d |G| / d eps_ab = -G_a G_b / |G|` at fixed node values (`models::nonlocal::strain`). Its
//! volume element, gradient and density responses are the xc terms X-1, X-3 and S-3/X-4 already.
//! The fast, default-run gate of X-6 alone is `nonlocal_strain.rs`. The oracle is the Richardson difference of free energies reconverged at `x -> (1 + eps) x`, in
//! one entry `eps_ab` at a time (atoms in fractional coordinates and fractional k-points carried,
//! the node counts fixed). The Rust setup is pinned to the reference cell's ion width and filter
//! cutoff, which otherwise follow the strained grid spacing. The spin layouts and the spinor path are those of the
//! meta-GGA gates.

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
        if self.spinor {
            "pbe-fr"
        } else {
            "pbe-sr"
        }
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
                species: if case.symbols[i] == case.symbols[0] {
                    0
                } else {
                    1
                },
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
    let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap())
        .join(case.library());
    let files: Vec<_> = species
        .iter()
        .map(|symbol| {
            let path = root.join(format!("{symbol}.parquet"));
            (
                path.to_string_lossy().into_owned(),
                sha256_file(&path).unwrap(),
            )
        })
        .collect();
    let datasets: Vec<_> = species
        .iter()
        .map(|s| Arc::new(dataset(&ctx, case.library(), s)))
        .collect();
    let options = Options {
        tolerance: 1e-12,
        density_tolerance: 1e-10,
        ..Options::default()
    };
    let crystal = create_setup(
        &ctx,
        input(case, files.clone(), Matrix3::zeros()),
        datasets.clone(),
    )
    .unwrap()
    .build(&ctx)
    .unwrap();
    assert_eq!(
        crystal.spinor, case.spinor,
        "{}: the crystal's spinor flag",
        case.label
    );
    let result = crystal.solve(&options).unwrap();
    assert!(
        result.converged,
        "{}: the reference SCF did not converge",
        case.label
    );
    let stress = response::stress(&crystal, &result).unwrap();
    let b = &stress.breakdown;
    assert!(
        !b.xc_nonlocal.iter().flatten().all(|x| *x == 0.),
        "{}: X-6 must be live",
        case.label
    );
    for &(a, c) in entries {
        let energy = |h: f64| {
            let mut eps = Matrix3::zeros();
            eps[(a, c)] = h;
            let strained = create_setup(&ctx, input(case, files.clone(), eps), datasets.clone())
                .unwrap()
                .build(&ctx)
                .unwrap();
            let state = strained.solve(&options).unwrap();
            assert!(
                state.converged,
                "{}: a strained SCF did not converge",
                case.label
            );
            state.free_energy
        };
        let h = case.step;
        let coarse = (energy(h) - energy(-h)) / (2. * h);
        let fine = (energy(0.5 * h) - energy(-0.5 * h)) / h;
        let oracle = (4. * fine - coarse) / 3.;
        let got = stress.gradient[a][c];
        let miss = (got - oracle).abs();
        eprintln!(
            "{} g_{a}{c}: analytic {got:+.10e}, oracle {oracle:+.10e} (central {coarse:+.10e} / {fine:+.10e}), miss {miss:.2e} Ha; X-6 {:+.3e}; {:.0} s",
            case.label,
            b.xc_nonlocal[a][c],
            started.elapsed().as_secs_f64()
        );
        // The entry is resolved, and X-6 is not vacuous: dropping it alone would miss the oracle by
        // at least 100 times what the whole stress misses.
        assert!(
            oracle.abs() > 10. * GATE,
            "{}: the oracle {oracle:e} is not above its gate",
            case.label
        );
        assert!(
            b.xc_nonlocal[a][c].abs() > 100. * miss,
            "{}: X-6 {:e} is not resolved by the gate (miss {miss:e})",
            case.label,
            b.xc_nonlocal[a][c]
        );
        assert!(
            (b.kinetic_density[a][c] != 0.) == case.functional.starts_with("r2scan"),
            "{}: S-6 belongs to the meta-GGA alone",
            case.label
        );
        assert!(
            miss < GATE,
            "{} g_{a}{c}: {miss:e} against {GATE:e}",
            case.label
        );
    }
}

const LIH: Case = Case {
    label: "LiH PBE+rVV10",
    symbols: ["Li", "H"],
    fractional: [[0.1, 0.2, 0.3], [0.55, 0.6, 0.7]],
    moments: [0., 0.],
    spin: false,
    functional: "pbe+rvv10",
    spinor: false,
    step: 2e-4,
};

#[test]
#[ignore = "slow: 22 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_rvv10 -- --ignored a_pbe_rvv10_crystal_stress_is_the_reconverged_finite_difference"]
fn a_pbe_rvv10_crystal_stress_is_the_reconverged_finite_difference() {
    check(&LIH, &[(0, 0), (0, 1)]);
}

#[test]
#[ignore = "slow: 25 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_rvv10 -- --ignored an_r2scan_rvv10_crystal_stress_is_the_reconverged_finite_difference"]
fn an_r2scan_rvv10_crystal_stress_is_the_reconverged_finite_difference() {
    check(
        &Case {
            label: "LiH r2SCAN+rVV10",
            functional: "r2scan+rvv10",
            ..LIH
        },
        &[(0, 0), (0, 1)],
    );
}

const H2: Case = Case {
    label: "H2 ferromagnetic",
    symbols: ["H", "H"],
    fractional: [[0.1, 0.2, 0.3], [0.45, 0.5, 0.55]],
    moments: [1., 1.],
    spin: true,
    functional: "pbe+rvv10",
    spinor: false,
    step: 2e-4,
};

#[test]
#[ignore = "slow: 16 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_rvv10 -- --ignored a_spin_polarized_pbe_rvv10_crystal_stress_is_the_reconverged_finite_difference"]
fn a_spin_polarized_pbe_rvv10_crystal_stress_is_the_reconverged_finite_difference() {
    check(
        &Case {
            label: "H2 ferromagnetic UKS-PBE+rVV10",
            ..H2
        },
        &[(0, 0), (0, 1)],
    );
}

#[test]
#[ignore = "slow: 17 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_rvv10 -- --ignored a_spin_polarized_r2scan_rvv10_crystal_stress_is_the_reconverged_finite_difference"]
fn a_spin_polarized_r2scan_rvv10_crystal_stress_is_the_reconverged_finite_difference() {
    // The meta-GGA terms (S-6, X-5) with the nonlocal one, on two spin channels.
    check(
        &Case {
            label: "H2 ferromagnetic UKS-r2SCAN+rVV10",
            functional: "r2scan+rvv10",
            ..H2
        },
        &[(0, 0), (0, 1)],
    );
}

#[test]
#[ignore = "slow: 18 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_rvv10 -- --ignored a_magnetic_spinor_pbe_rvv10_crystal_stress_is_the_reconverged_finite_difference"]
fn a_magnetic_spinor_pbe_rvv10_crystal_stress_is_the_reconverged_finite_difference() {
    // The charge and its gradient the correlation sees are the eigenframe sums of the four-field
    // density (SO-3), exact in that frame: the term is the scalar one.
    check(
        &Case {
            label: "H2 ferromagnetic spinor PBE+rVV10",
            spin: false,
            spinor: true,
            ..H2
        },
        &[(0, 0), (0, 1)],
    );
}

#[test]
#[ignore = "slow: 59 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_rvv10 -- --ignored a_spin_orbit_pbe_rvv10_crystal_stress_is_the_reconverged_finite_difference"]
fn a_spin_orbit_pbe_rvv10_crystal_stress_is_the_reconverged_finite_difference() {
    // Gold's semicore and partial core, and its 5d/6p spin-orbit channels (time-reversal symmetric).
    let case = Case {
        label: "AuH SOC PBE+rVV10",
        symbols: ["Au", "H"],
        fractional: [[0.1, 0.2, 0.3], [0.35, 0.4, 0.45]],
        moments: [0., 0.],
        spin: false,
        functional: "pbe+rvv10",
        spinor: true,
        step: 2e-5,
    };
    check(&case, &[(0, 0), (0, 1)]);
}
