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
