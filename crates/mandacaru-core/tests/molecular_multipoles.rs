//! Position and gradient integrals of the molecular basis (`molecular::multipoles`). The position
//! integrals are symmetric, and an origin shift changes them by exactly the identity in the
//! orthonormal frame, with or without the PAW one-center part (whose overlap correction is the
//! frame's metric). The gradient integrals are antisymmetric: on the grid by the skew-adjoint
//! stencil, and in the one-center part by construction (its quadrature residual, 4e-7 of the
//! largest element on the library's water, is projected out).
mod paw_molecular_support;

use mandacaru_core::{
    math::SpatialGrid,
    molecular::{
        multipoles::{gradient_integrals, position_integrals},
        MolecularIntegrals,
    },
    Context, Geometry, LengthUnit,
};
use nalgebra::DMatrix;

fn hydrogen(ctx: &Context) -> MolecularIntegrals {
    let geometry = Geometry::new(
        vec![1, 1],
        vec![[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]],
        [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
        [false; 3],
        LengthUnit::Angstrom,
    )
    .unwrap();
    let grid =
        SpatialGrid::from_geometry(&geometry, [0.35 / 0.529177210903; 3], false, false).unwrap();
    MolecularIntegrals::new(
        ctx,
        geometry,
        grid,
        r#"{"name":"HAO"}"#,
        "spectral",
        None,
        Default::default(),
    )
    .unwrap()
}

fn check_origin(integrals: &MolecularIntegrals, label: &str) {
    let m = integrals.m();
    let at_zero = position_integrals(integrals, [0.; 3]).unwrap();
    let shift = [0.7, -1.3, 2.1];
    let shifted = position_integrals(integrals, shift).unwrap();
    for axis in 0..3 {
        let r = &at_zero[axis];
        assert!((r - r.transpose()).amax() < 1e-14, "{label}: symmetric");
        let expected = r - DMatrix::<f64>::identity(m, m) * shift[axis];
        let error = (&shifted[axis] - expected).amax();
        assert!(
            error < 1e-9,
            "{label} axis {axis}: origin shift misses the identity by {error:e}"
        );
    }
}

#[test]
fn an_origin_shift_moves_the_all_electron_integrals_by_the_identity() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    check_origin(&hydrogen(&ctx), "HAO H2");
}

#[test]
fn an_origin_shift_moves_the_paw_integrals_by_the_identity() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let integrals = paw_molecular_support::integrals(&ctx, &paw_molecular_support::H2, "raw");
    check_origin(&integrals, "PAW H2");
}

#[test]
fn the_dipole_of_a_symmetric_pair_sits_at_its_center() {
    // The mean position of the closed-shell density of H2 is the bond center, by symmetry,
    // with or without the one-center part, to the symmetry of the grid about it.
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let integrals = paw_molecular_support::integrals(&ctx, &paw_molecular_support::H2, "raw");
    let state = integrals
        .kohn_sham("lda", [1, 1], false, &Default::default())
        .unwrap();
    let r = position_integrals(&integrals, [0.; 3]).unwrap();
    let d = state.densities[0].map(|z| z.re);
    let center = [0., 0., 0.];
    for axis in 0..3 {
        let mean = (&d * &r[axis]).trace();
        assert!((mean - center[axis]).abs() < 1e-4, "axis {axis}: {mean}");
    }
}

fn library_water(ctx: &Context, set: &str) -> MolecularIntegrals {
    use paw_molecular_support::{analytic_shells, dataset, molecule};
    let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
    let oxygen = dataset::species(ctx, &mut arrays, &dataset::read(ctx, set, "O"));
    let hydrogen = dataset::species(ctx, &mut arrays, &dataset::read(ctx, set, "H"));
    molecule(
        ctx,
        &[8, 1, 1],
        &[[6.11, 6.07, 6.12], [7.55, 6.04, 5.07], [4.57, 5.95, 5.03]],
        12.,
        0.5,
        (vec![oxygen, hydrogen], vec![0, 1, 1], arrays),
        "raw",
        &analytic_shells(&[0, 1, 2]),
    )
}

fn check_antisymmetry(integrals: &MolecularIntegrals, label: &str, tolerance: f64) -> f64 {
    let g = gradient_integrals(integrals).unwrap();
    let scale = g.iter().map(|g| g.amax()).fold(0f64, f64::max);
    assert!(scale > 1e-3, "{label}: the gradient vanishes");
    let worst = g
        .iter()
        .map(|g| (g + g.transpose()).amax())
        .fold(0f64, f64::max)
        / scale;
    println!("{label}: largest |G + G^T| / |G| = {worst:e}");
    assert!(worst < tolerance, "{label}: |G + G^T| / |G| = {worst:e}");
    worst
}

#[test]
fn the_gradient_integrals_are_antisymmetric() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    check_antisymmetry(&hydrogen(&ctx), "HAO H2", 1e-12);
    check_antisymmetry(
        &paw_molecular_support::integrals(&ctx, &paw_molecular_support::H2, "raw"),
        "synthetic PAW H2",
        1e-12,
    );
    for set in ["lda-sr", "pbe-sr"] {
        check_antisymmetry(&library_water(&ctx, set), &format!("{set} water"), 1e-12);
    }
}

/// Library water with the projectors in `basis` (`"raw"` or `"dual"`).
fn library_water_in(ctx: &Context, basis: &str) -> MolecularIntegrals {
    let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
    let (o, h) = (
        paw_molecular_support::dataset::read(ctx, "lda-sr", "O"),
        paw_molecular_support::dataset::read(ctx, "lda-sr", "H"),
    );
    let oxygen = paw_molecular_support::dataset::species(ctx, &mut arrays, &o);
    let hydrogen = paw_molecular_support::dataset::species(ctx, &mut arrays, &h);
    paw_molecular_support::molecule(
        ctx,
        &[8, 1, 1],
        &[[6.11, 6.07, 6.12], [7.55, 6.04, 5.07], [4.57, 5.95, 5.03]],
        12.,
        0.5,
        (vec![oxygen, hydrogen], vec![0, 1, 1], arrays),
        basis,
        &paw_molecular_support::analytic_shells(&[0, 1, 2]),
    )
}

#[test]
fn the_one_center_operators_do_not_depend_on_the_projector_basis() {
    // The raw and dual projector bases are two coordinates of one PAW transform: every operator
    // in the orthonormal frame (overlap folded in, position, gradient) must agree between them to
    // the quadrature, or one of them applies a one-center block in the wrong basis.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let raw = library_water_in(&ctx, "raw");
    let dual = library_water_in(&ctx, "dual");
    let scale = |m: &DMatrix<f64>| m.amax().max(1e-300);
    let (rp, dp) = (
        position_integrals(&raw, [0.; 3]).unwrap(),
        position_integrals(&dual, [0.; 3]).unwrap(),
    );
    let (rg, dg) = (
        gradient_integrals(&raw).unwrap(),
        gradient_integrals(&dual).unwrap(),
    );
    for axis in 0..3 {
        let position = (&rp[axis] - &dp[axis]).amax() / scale(&rp[axis]);
        let gradient = (&rg[axis] - &dg[axis]).amax() / scale(&rg[axis]);
        eprintln!("axis {axis}: raw - dual, position {position:.3e}, gradient {gradient:.3e}");
        assert!(position < 1e-6, "position axis {axis}: {position:e}");
        assert!(gradient < 1e-6, "gradient axis {axis}: {gradient:e}");
    }
}

#[test]
fn the_dipole_moment_is_the_field_derivative_of_the_energy() {
    // Q10a, `multipoles::dipole_moment` on library water (LDA, partial cores): the electrons'
    // share of the moment is minus the derivative of the SCF energy under a field term `F x_k` in
    // the one-body operator (the energy is variational, so it is `g tr(D x_k)`), by central
    // differences at 2e-3 and 1e-3 a.u. (Richardson), on every axis; the neutral molecule's
    // moment does not depend on the origin; the closed shell run unrestricted (two density
    // matrices, each counted once) has the restricted moment.
    use mandacaru_core::molecular::{multipoles::dipole_moment, ScfOptions};
    use paw_molecular_support::dataset;
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let mut integrals = library_water(&ctx, "lda-sr");
    let electrons = ["O", "H", "H"]
        .iter()
        .map(|e| dataset::read(&ctx, "lda-sr", e).valence_charge)
        .sum::<f64>()
        .round() as usize;
    let options = ScfOptions {
        tolerance: 1e-12,
        max_iterations: 200,
        ..Default::default()
    };
    let particles = [electrons / 2, electrons / 2];
    let state = integrals
        .kohn_sham("lda", particles, false, &options)
        .unwrap();
    assert!(state.converged);
    let moment = dipole_moment(&integrals, &state.densities, [0.; 3]).unwrap();
    let shifted = dipole_moment(&integrals, &state.densities, [1.3, -2.1, 0.7]).unwrap();
    let open = integrals
        .kohn_sham("lda", particles, true, &options)
        .unwrap();
    assert!(open.converged && open.densities.len() == 2);
    let unrestricted = dipole_moment(&integrals, &open.densities, [0.; 3]).unwrap();
    let channels = (0..3)
        .map(|a| (unrestricted[a] - moment[a]).abs())
        .fold(0f64, f64::max);
    let origin = (0..3)
        .map(|a| (moment[a] - shifted[a]).abs())
        .fold(0f64, f64::max);
    let charges = integrals.corrections.as_ref().unwrap().charges.clone();
    let positions = integrals.geometry.positions().to_vec();
    let x = position_integrals(&integrals, [0.; 3]).unwrap();
    let base: Vec<f64> = integrals.one.as_slice().to_vec();
    let m = integrals.m();
    let mut misses = vec![];
    for axis in 0..3 {
        let ions: f64 = charges
            .iter()
            .zip(&positions)
            .map(|(z, r)| z * r[axis])
            .sum();
        let mut energy = |field: f64| {
            for a in 0..m {
                for b in 0..m {
                    integrals.one.as_mut_slice()[a * m + b] =
                        base[a * m + b] + field * x[axis][(a, b)];
                }
            }
            let state = integrals
                .kohn_sham("lda", particles, false, &options)
                .unwrap();
            assert!(state.converged);
            state.electronic_energy
        };
        let slope =
            |f: f64, energy: &mut dyn FnMut(f64) -> f64| (energy(f) - energy(-f)) / (2. * f);
        let (coarse, fine) = (slope(2e-3, &mut energy), slope(1e-3, &mut energy));
        let field = (4. * fine - coarse) / 3.;
        let electronic = moment[axis] - ions;
        println!(
            "water LDA axis {axis}: moment {:+.10} e Bohr, electrons {electronic:+.10}, -dE/dF \
             {:+.10} (central {:+.10} / {:+.10})",
            moment[axis], -field, -coarse, -fine
        );
        misses.push((electronic + field).abs());
    }
    for a in 0..m * m {
        integrals.one.as_mut_slice()[a] = base[a];
    }
    let size = moment.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("water LDA: |mu| {size:.6} e Bohr; origin shift {origin:.2e}; unrestricted {channels:.2e}; field misses {misses:?}");
    assert!(channels < 1e-8, "{channels:e}");
    assert!(origin < 1e-9, "{origin:e}");
    assert!(misses.iter().all(|m| *m < 1e-6), "{misses:?}");
    assert!(size > 0.5 && size < 1., "{size}");
}

/// The spinor branch of `dipole_moment` (Q10a): Au-H on the `pbe-fr` set with gold's spin-orbit
/// term (one s, p and d shell on each atom), spinor PBE. The electrons' share of the moment of the
/// one `2m x 2m` density, its charge component `D_up,up + D_dn,dn`, is minus the derivative of the
/// spinor SCF energy under `F x_k` in the one-body operator, which enters both spin blocks, by
/// central differences at 2e-3 and 1e-3 a.u. (Richardson), along the bond (z) and across it (x).
/// The red control reads the up-up block alone and misses by about half the share.
#[test]
#[ignore = "slow: 75 s (measured 2026-11-16; spinor PBE SCFs of Au-H, four per axis, needs MANDACARU_PAW_PATH); run: cargo test --profile release-test -p mandacaru-core --test molecular_multipoles -- --ignored the_spinor_dipole"]
fn the_spinor_dipole_moment_is_the_field_derivative_of_the_spin_orbit_energy() {
    use mandacaru_core::{
        molecular::{multipoles::dipole_moment, ScfOptions},
        paw::molecular::Arrays,
    };
    use paw_molecular_support::{analytic_shells, dataset, molecule_with};
    let ctx = Context::new(Some(2), 1536 << 20).unwrap();
    let root = std::path::PathBuf::from(
        std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH"),
    );
    let file = |e: &str| {
        root.join("pbe-fr")
            .join(format!("{e}.parquet"))
            .to_string_lossy()
            .into_owned()
    };
    let mut integrals = molecule_with(
        &ctx,
        &[79, 1],
        &[[0.07, -0.04, 0.05], [0.15, 0.11, 2.95]],
        14.,
        0.5,
        (
            vec![
                serde_json::json!({ "file": file("Au") }),
                serde_json::json!({ "file": file("H") }),
            ],
            vec![0, 1],
            Arrays::new(),
        ),
        "raw",
        &analytic_shells(&[0, 1, 2]),
        &[("spin_orbit", serde_json::json!(true))],
    );
    let electrons = (dataset::read(&ctx, "pbe-fr", "Au").valence_charge
        + dataset::read(&ctx, "pbe-fr", "H").valence_charge)
        .round() as usize;
    let options = ScfOptions {
        tolerance: 1e-12,
        max_iterations: 600,
        ..Default::default()
    };
    let state = integrals
        .spinor_kohn_sham("pbe", electrons, &options)
        .unwrap();
    assert!(state.converged && state.densities.len() == 1);
    let m = integrals.m();
    assert_eq!(state.densities[0].shape(), (2 * m, 2 * m));
    let moment = dipole_moment(&integrals, &state.densities, [0.; 3]).unwrap();
    let charges = integrals.corrections.as_ref().unwrap().charges.clone();
    let positions = integrals.geometry.positions().to_vec();
    let x = position_integrals(&integrals, [0.; 3]).unwrap();
    let base: Vec<f64> = integrals.one.as_slice().to_vec();
    let d = state.densities[0].clone();
    let mut misses = vec![];
    for axis in [2, 0] {
        let ions: f64 = charges
            .iter()
            .zip(&positions)
            .map(|(z, r)| z * r[axis])
            .sum();
        let mut energy = |field: f64| {
            for a in 0..m {
                for b in 0..m {
                    integrals.one.as_mut_slice()[a * m + b] =
                        base[a * m + b] + field * x[axis][(a, b)];
                }
            }
            let state = integrals
                .spinor_kohn_sham("pbe", electrons, &options)
                .unwrap();
            assert!(state.converged);
            state.electronic_energy
        };
        let slope =
            |f: f64, energy: &mut dyn FnMut(f64) -> f64| (energy(f) - energy(-f)) / (2. * f);
        let (coarse, fine) = (slope(2e-3, &mut energy), slope(1e-3, &mut energy));
        let field = (4. * fine - coarse) / 3.;
        let electronic = moment[axis] - ions;
        let up: f64 = (0..m)
            .flat_map(|a| (0..m).map(move |b| (a, b)))
            .map(|(a, b)| d[(a, b)].re * x[axis][(b, a)])
            .sum();
        println!(
            "Au-H spinor PBE axis {axis}: moment {:+.10} e Bohr, electrons {electronic:+.10}, \
             -dE/dF {:+.10} (central {:+.10} / {:+.10}); red (up-up block only) {:+.10}",
            moment[axis], -field, -coarse, -fine, -up
        );
        misses.push((electronic + field).abs());
        assert!(
            (-up + field).abs() > 1e-2 * field.abs(),
            "the red control must miss"
        );
    }
    for a in 0..m * m {
        integrals.one.as_mut_slice()[a] = base[a];
    }
    println!("Au-H spinor: moment {moment:?} e Bohr; field misses {misses:?}");
    assert!(misses.iter().all(|m| *m < 1e-6), "{misses:?}");
}
