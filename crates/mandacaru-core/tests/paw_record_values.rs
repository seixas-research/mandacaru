//! The local potential of `paw::record` against the frozen Python cases and
//! against a cubic it must reproduce exactly.

mod paw_record_support;

use mandacaru_core::paw::record::PawDataset;
use mandacaru_core::{Context, Error};
use paw_record_support::*;
use serde_json::Value;

// ---------------------------------------------------------------------------
// The local potential.
// ---------------------------------------------------------------------------

fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    let step = (stop - start) / (n - 1) as f64;
    let mut out: Vec<f64> = (0..n).map(|i| i as f64 * step + start).collect();
    out[n - 1] = stop;
    out
}

fn close(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1.0)
}

#[test]
fn the_local_potential_matches_the_frozen_python_cases() {
    let ctx = context();
    let (mut files, mut points, mut worst) = (0usize, 0usize, 0f64);
    for (set, path) in files_of(&library(), &SETS) {
        let element = path.file_stem().unwrap().to_str().unwrap().to_string();
        let Some((doc, _)) = frozen_arrays(&set, &element) else {
            continue;
        };
        let frozen = &doc["functions"]["local_potential"];
        let ds = PawDataset::read(&ctx, &path).unwrap();
        let potential = ds.local_potential(&ctx).unwrap();
        let (r0, rmax) = (ds.r.as_slice()[0], *ds.r.as_slice().last().unwrap());
        assert_eq!(r0, frozen["r0"].as_f64().unwrap());
        assert_eq!(rmax, frozen["r_max"].as_f64().unwrap());
        let inside = linspace(r0, rmax, 41);
        let mut mixed = vec![0.0, 0.5 * r0, r0, 1.0, 2.5];
        mixed.extend_from_slice(&inside[1..40]);
        mixed.extend_from_slice(&[rmax, rmax * 1.0001, rmax * 1.5, 40.0, 100.0]);
        let beyond = [rmax * 1.01, rmax * 1.3, 40.0, 100.0];
        let scalars = [0.5 * r0, 1.7, rmax * 1.2];
        let mut compare = |radii: &[f64], values: &Value, slopes: &Value, label: &str| {
            let (v, s) = potential.evaluate(&ctx, radii).unwrap();
            let (want_v, want_s) = (floats(values), floats(slopes));
            assert_eq!(want_v.len(), radii.len(), "{set}/{element} {label}");
            for i in 0..radii.len() {
                for (got, want, what) in [
                    (v.as_slice()[i], want_v[i], "value"),
                    (s.as_slice()[i], want_s[i], "slope"),
                ] {
                    let error = close(got, want);
                    worst = worst.max(error);
                    assert!(
                        error < 1e-12,
                        "{set}/{element} {label} {what} at r={}: {got:e} vs {want:e}",
                        radii[i]
                    );
                    points += 1;
                }
            }
        };
        compare(
            &mixed,
            &frozen["value_mixed"],
            &frozen["slope_mixed"],
            "mixed",
        );
        compare(
            &beyond,
            &frozen["value_beyond"],
            &frozen["slope_beyond"],
            "beyond",
        );
        compare(
            &scalars,
            &frozen["value_scalars"],
            &frozen["slope_scalars"],
            "scalars",
        );
        let (v, s) = potential.evaluate(&ctx, &inside).unwrap();
        for (got, key) in [
            (v.as_slice(), "value_inside"),
            (s.as_slice(), "slope_inside"),
        ] {
            let max = got.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let want = frozen[key]["max"].as_f64().unwrap();
            assert!(
                close(max, want) < 1e-12,
                "{set}/{element} {key} max {max:e} vs {want:e}"
            );
        }
        files += 1;
    }
    println!("local potential: {files} files, {points} frozen values and slopes, worst relative difference {worst:.2e}");
    assert!(files >= 19);
}

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
    // Flat below the first point, the ionic tail beyond the last.
    assert_eq!(potential.value(0.5 * r0), poly(r0));
    assert_eq!(potential.slope(0.5 * r0), 0.0);
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
