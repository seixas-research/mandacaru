//! The local potential of `paw::record` against a cubic it must reproduce exactly.

mod paw_record_support;

use mandacaru_core::paw::record::PawDataset;
use mandacaru_core::{Context, Error};
use paw_record_support::*;

// ---------------------------------------------------------------------------
// The local potential.
// ---------------------------------------------------------------------------

#[test]
fn the_local_potential_reproduces_a_cubic_exactly_and_has_the_ionic_tail() {
    let ctx = context();
    let mut ds = PawDataset::read(&ctx, &dataset_path("lda-sr", "O")).unwrap();
    let poly = |r: f64| 0.3 - 1.1 * r + 0.7 * r * r - 0.05 * r * r * r;
    let slope = |r: f64| -1.1 + 1.4 * r - 0.15 * r * r;
    let grid: Vec<f64> = ds.r.as_slice().to_vec();
    for (v, r) in ds.v_local.as_mut_slice().iter_mut().zip(&grid) {
        *v = poly(*r);
    }
    let potential = ds.local_potential(&ctx).unwrap();
    let (r0, rmax, z) = (grid[0], *grid.last().unwrap(), ds.valence_charge);
    for r in linspace(r0, rmax, 237) {
        assert!((potential.value(r) - poly(r)).abs() < 1e-8, "value at {r}");
        assert!((potential.slope(r) - slope(r)).abs() < 1e-7, "slope at {r}");
    }
    // Smooth C^1 even continuation a + b r^2 below the first point, ionic tail beyond the last.
    let h = 0.5 * r0;
    let t = h / r0;
    let y0p = slope(r0);
    assert!((potential.value(h) - (poly(r0) - 0.5 * r0 * y0p * (1. - t * t))).abs() < 1e-8);
    assert!((potential.slope(h) - y0p * t).abs() < 1e-7);
    assert_eq!(potential.slope(0.0), 0.0);
    assert_eq!(potential.value(2.0 * rmax), -z / (2.0 * rmax));
    assert_eq!(potential.slope(2.0 * rmax), z / (4.0 * rmax * rmax));
    drop(potential);
    // A table of fewer than four points has no spline.
    let tiny = ds.local_potential(&Context::new(Some(1), 1 << 12).unwrap());
    assert!(matches!(tiny, Err(Error::Budget { .. })), "{tiny:?}");
}

#[test]
fn a_potential_built_from_the_bare_table_is_the_records_own() {
    use mandacaru_core::paw::record::LocalPotential;
    let ctx = context();
    let recipe = mandacaru_core::paw::generate::recipe::Recipe::for_dataset(
        "H",
        "lda",
        mandacaru_core::models::relativity::Treatment::Scalar,
    )
    .unwrap();
    let ds = mandacaru_core::paw::generate::generate(&ctx, &recipe)
        .unwrap()
        .dataset;
    let (own, bare) = (
        ds.local_potential(&ctx).unwrap(),
        LocalPotential::from_table(
            &ctx,
            ds.r.as_slice(),
            ds.v_local.as_slice(),
            ds.valence_charge,
        )
        .unwrap(),
    );
    let last = *ds.r.as_slice().last().unwrap();
    for radius in [0.5 * ds.r.as_slice()[0], 0.37, 1.9, last, 2. * last] {
        assert_eq!(own.value(radius).to_bits(), bare.value(radius).to_bits());
        assert_eq!(own.slope(radius).to_bits(), bare.slope(radius).to_bits());
    }
    assert!(LocalPotential::from_table(&ctx, &[], &[], 1.0).is_err());
}
