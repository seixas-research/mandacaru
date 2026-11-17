//! The Gaussian ion width `sigma` is a numerical device of the local-potential split, not a physical
//! parameter: the energy of a crystal may not ripple with it.
//!
//! The short-range local potential `v_local - v^lr(sigma)` is integrated on a sphere of radius
//! `6.5 sigma` (`Sphere::short_range`). `v_local + Z / r` has second-derivative corners at the
//! radii where the dataset was joined (the local cutoff, the compensation radius, the projector
//! radii, the partial-core radius, the start of the ionic tail); a radial panel whose nodes slide with `sigma` across one
//! of them makes the integral, and so the energy, a ripple in `sigma`. The installed
//! `pbe-sr/Si` has its compensation corner (2.19990 Bohr) 0.22 Bohr beyond the old first panel
//! edge (the local cutoff, 1.97991 Bohr): `dE/d sigma` was -1.10e-2 eV/Bohr at the default width
//! of the 11^3 diamond cell, rms 9.8e-3 eV/Bohr over `sigma` 0.60-1.35 Bohr. The rule now has a
//! panel edge on every corner ([`LocalPotential::panel_edges`]).
//!
//! The fast gate measures the quadrature itself: the error of `int v^sr f d^3r` (a smooth
//! `f`) against a converged rule, as a function of `sigma`. The slow gate runs the SCF of the
//! cell. Both have the old layout as a control (one panel edge at the local cutoff), which
//! must fail.
//!
//! Run: `MANDACARU_PAW_PATH=<mandacaru-paw> cargo test --profile release-test -p mandacaru-core
//! --test periodic_ion_width_ripple` (the slow one with `-- --ignored`).

use mandacaru_core::{
    derived::free_wavevector,
    paw::{
        digest::dataset_content_sha256,
        local_split::{radial_rule_panels, short_range_potential},
        record::PawDataset,
    },
    periodic::{
        setup::{
            ions::LocalPotential,
            parse::{create_setup, AtomSpec, SetupInput},
            spheres::{Sphere, SHORT_RANGE_AZIMUTHAL, SHORT_RANGE_POLAR},
        },
        solver::Options,
    },
    units::ANGSTROM_TO_BOHR,
    Context,
};
use nalgebra::Matrix3;
use std::{f64::consts::PI, path::PathBuf, sync::Arc};

const HA_TO_EV: f64 = 27.211386245988;
/// The radial points per panel of the old layout (one panel edge, at the local cutoff).
const OLD_RADIAL: usize = 32;
/// The gate on the root mean square of `d (energy error) / d sigma` (eV/Bohr). The slope the
/// width has no business having is 1e-4 eV/Bohr; the old layout measured 1e-2 and more.
const RIPPLE_GATE: f64 = 1e-4;
/// The old layout must miss the gate by at least this factor (it misses it by 100 or more).
const CONTROL_FACTOR: f64 = 10.;

fn library() -> PathBuf {
    PathBuf::from(
        std::env::var_os("MANDACARU_PAW_PATH")
            .filter(|v| !v.is_empty())
            .expect("MANDACARU_PAW_PATH is unset: the ripple gates read the shipped datasets"),
    )
    .join("pbe-sr")
}

fn dataset(ctx: &Context, symbol: &str) -> PawDataset {
    let path = library().join(format!("{symbol}.parquet"));
    PawDataset::read(ctx, &path).unwrap_or_else(|e| panic!("{symbol}: {e}"))
}

/// Root mean square of the central slope of `values` on the uniform grid of step `step`.
fn rms_slope(values: &[f64], step: f64) -> f64 {
    let slopes: Vec<f64> = (1..values.len() - 1)
        .map(|i| (values[i + 1] - values[i - 1]) / (2. * step))
        .collect();
    (slopes.iter().map(|s| s * s).sum::<f64>() / slopes.len() as f64).sqrt()
}

/// `int v^sr(r; sigma) exp(-(r / 1.5)^2) d^3 r` (Hartree Bohr^3) on the points of a rule.
fn integral(potential: &LocalPotential, sigma: f64, radii: &[f64], weights: &[f64]) -> f64 {
    let sampled: Vec<f64> = radii.iter().map(|r| potential.value(*r)).collect();
    let v = short_range_potential(&sampled, potential.charge, sigma, radii).unwrap();
    radii
        .iter()
        .zip(weights)
        .zip(&v)
        .map(|((r, w), v)| w * v * (-(r / 1.5).powi(2)).exp())
        .sum()
}

/// The root mean square slope (eV/Bohr) in `sigma` of the quadrature error of [`integral`] over
/// `sigma` 0.60-1.34 Bohr in steps of 0.02, the rule being `rule(sigma)`.
fn quadrature_ripple(potential: &LocalPotential, rule: impl Fn(f64) -> Sphere) -> f64 {
    let step = 0.02;
    let errors: Vec<f64> = (0..=37)
        .map(|i| {
            let sigma = 0.60 + step * i as f64;
            let sphere = rule(sigma);
            let all = 0..sphere.len();
            let got = integral(
                potential,
                sigma,
                sphere.radii(all.clone()),
                sphere.weights(all),
            );
            let (r, w) = radial_rule_panels(
                potential.reach(sigma).unwrap(),
                &potential.panel_edges(),
                400,
            )
            .unwrap();
            let w: Vec<f64> = w.iter().map(|w| 4. * PI * w).collect();
            (got - integral(potential, sigma, &r, &w)) * HA_TO_EV
        })
        .collect();
    rms_slope(&errors, step)
}

#[test]
fn the_short_range_quadrature_has_no_ripple_in_the_ion_width() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    // Si: the compensation corner beyond the local cutoff. Cu: a projector radius (2.2 Bohr)
    // beyond the compensation radius (1.8). Ar: the tail starts before the projector radius.
    for symbol in ["Si", "Cu", "Ar"] {
        let ds = dataset(&ctx, symbol);
        let potential = LocalPotential::from_dataset(&ctx, &ds).unwrap();
        let center = [0.; 3];
        // A radial integral does not see the lattice: one with no symmetry beyond inversion keeps
        // the plain product rule.
        let lattice = Matrix3::new(9., 1.1, 0.7, 0., 10., 0.9, 0., 0., 11.);
        let production = quadrature_ripple(&potential, |sigma| {
            Sphere::short_range(&ctx, &lattice, center, &potential, sigma).unwrap()
        });
        // The old layout: the same nodes, one panel edge at the local cutoff.
        let old = quadrature_ripple(&potential, |sigma| {
            Sphere::rule(
                &ctx,
                &lattice,
                center,
                potential.reach(sigma).unwrap(),
                OLD_RADIAL,
                SHORT_RANGE_POLAR,
                SHORT_RANGE_AZIMUTHAL,
                ds.r_cut_local,
            )
            .unwrap()
        });
        println!(
            "{symbol}: rms d error / d sigma {production:.3e} eV/Bohr (gate {RIPPLE_GATE:.0e}); \
             old layout {old:.3e}"
        );
        assert!(
            production < RIPPLE_GATE,
            "{symbol}: the short-range quadrature ripples in the ion width, rms {production:e}"
        );
        assert!(
            old > CONTROL_FACTOR * RIPPLE_GATE,
            "{symbol}: the control (the old layout) does not ripple: {old:e}"
        );
    }
}

/// The free energy (eV) of the diamond cell of `pbe-sr/Si` (a = 5.43 A, Gamma, SZ, 11^3 nodes,
/// filter 200 eV, the ion width `sigma`) with the potential `adjust` hands back.
fn cell_energy(ctx: &Context, sigma: f64, adjust: &dyn Fn(&mut LocalPotential)) -> f64 {
    let path = library().join("Si.parquet");
    let files = vec![(
        path.to_string_lossy().into_owned(),
        dataset_content_sha256(ctx, &path).unwrap(),
    )];
    let a = 5.43 * ANGSTROM_TO_BOHR;
    let h = 0.5 * a;
    let input = SetupInput {
        mesh: None,
        recipes: None,
        projector_basis: "raw".into(),
        lattice: Matrix3::from_row_slice(&[0., h, h, h, 0., h, h, h, 0.]),
        grid_shape: [11; 3],
        atoms: [[0.; 3], [0.25; 3]]
            .map(|fractional| AtomSpec {
                species: 0,
                symbol: "Si".into(),
                fractional,
                moment: 0.,
            })
            .to_vec(),
        kpoints_frac: vec![[0.; 3]],
        weights: vec![1.],
        functional: "pbe".into(),
        precision: "float64".into(),
        basis_size: "SZ".into(),
        spinor: false,
        spin: false,
        symmetry: false,
        filter_cutoff: Some(free_wavevector(200.).unwrap()),
        sigma: Some(sigma),
        dataset_files: Some(files),
        hubbard_u: None,
        charge: None,
    };
    let mut setup = create_setup(ctx, input, vec![Arc::new(dataset(ctx, "Si"))]).unwrap();
    setup.potentials.iter_mut().for_each(adjust);
    let result = setup
        .build(ctx)
        .unwrap()
        .solve(&Options {
            max_iterations: 400,
            tolerance: 1e-11,
            density_tolerance: 1e-9,
            width: 0.01 / HA_TO_EV,
            ..Options::default()
        })
        .unwrap();
    assert!(
        result.converged,
        "the SCF at sigma {sigma} did not converge"
    );
    result.free_energy * HA_TO_EV
}

/// rms `dE/d sigma` (eV/Bohr) of the cell over `sigma` 0.80-1.36 Bohr in steps of 0.02, where the
/// Gaussian ions are resolved by the 0.66 Bohr grid (below 0.8 Bohr the aliasing of the ions on
/// the grid is itself 3.6e-4 eV/Bohr with a converged rule).
fn cell_ripple(ctx: &Context, adjust: &dyn Fn(&mut LocalPotential)) -> f64 {
    let step = 0.02;
    let energies: Vec<f64> = (0..=28)
        .map(|i| cell_energy(ctx, 0.80 + step * i as f64, adjust))
        .collect();
    rms_slope(&energies, step)
}

#[test]
#[ignore = "slow: 144 s (measured 2026-11-16; two scans of 29 SCFs of the 11^3 cell); needs the shipped library: MANDACARU_PAW_PATH=<mandacaru-paw> cargo test --profile release-test -p mandacaru-core --test periodic_ion_width_ripple -- --ignored"]
fn the_energy_of_the_silicon_cell_has_no_ripple_in_the_ion_width() {
    let ctx = Context::new(Some(3), 2 << 30).unwrap();
    let production = cell_ripple(&ctx, &|_| {});
    // The old layout: no corner but the local cutoff, no tail edge (the sphere is as large).
    let old = cell_ripple(&ctx, &|potential| {
        potential.corners.clear();
        potential.tail = 0.;
    });
    println!(
        "rms dE/d sigma over sigma 0.80-1.36 Bohr: {production:.3e} eV/Bohr (gate {RIPPLE_GATE:.0e}); \
         old layout {old:.3e}"
    );
    assert!(
        production < RIPPLE_GATE,
        "the energy ripples in the ion width, rms {production:e} eV/Bohr"
    );
    assert!(
        old > CONTROL_FACTOR * RIPPLE_GATE,
        "the control (the old layout) does not ripple: {old:e} eV/Bohr"
    );
}
