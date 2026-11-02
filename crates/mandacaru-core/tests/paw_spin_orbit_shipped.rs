//! The shipped `lda-fr` library as a read-only cross-check of the Dirac generator: where a
//! shipped file's recipe is reproducible by the current tables (sodium: A0 finding 1), the Rust
//! generator must land on it to the end-to-end value the Python generator did (the spin-orbit
//! block 1.4e-8, the tables 6e-8 of their maxima, the scalars below 4e-10), which is not the
//! port floor. The library is never written. The 35 stale `fr` files and the lanthanide are
//! reported by the heavy test, not gated (their shipped construction is not the current one).
mod paw_assemble_support;
mod paw_generate_support;
mod paw_spin_orbit_support;

use mandacaru_core::paw::record::{Array, PawDataset};
use paw_spin_orbit_support::{context, open};

/// 100 x the end-to-end value measured in A0 for sodium (tables 6.0e-8, spin-orbit 1.4e-8).
const TABLE_GATE: f64 = 6e-6;
const MATRIX_GATE: f64 = 1.5e-6;
const SCALAR_GATE: f64 = 4e-8;

fn shipped(set: &str, symbol: &str) -> PawDataset {
    let root = std::env::var("MANDACARU_PAW_PATH").unwrap_or_else(|_| {
        panic!(
            "MANDACARU_PAW_PATH is not set; run `MANDACARU_PAW_PATH=<mandacaru-paw> cargo test \
             --release -p mandacaru-core --test paw_spin_orbit_shipped -- --ignored`"
        )
    });
    let path = std::path::Path::new(&root)
        .join(set)
        .join(format!("{symbol}.parquet"));
    PawDataset::read(&context(), &path)
        .unwrap_or_else(|e| panic!("{} could not be read: {e}", path.display()))
}

fn worst(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0., |m: f64, x| m.max(x.abs())).max(1e-300);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f64::max)
        / scale
}

fn matrices(a: &Array, b: &Array) -> f64 {
    assert_eq!(a.shape(), b.shape());
    worst(a.data(), b.data())
}

#[test]
#[ignore = "needs the shipped library: MANDACARU_PAW_PATH=<mandacaru-paw> cargo test --release -p mandacaru-core --test paw_spin_orbit_shipped -- --ignored"]
fn the_shipped_sodium_file_is_reproduced_to_its_end_to_end_value() {
    let ctx = context();
    let theirs = shipped("lda-fr", "Na");
    let case = open(&ctx, "lda-fr_Na");
    let ours = case.generate(&ctx).dataset;
    assert_eq!(theirs.relativity, "dirac");
    assert_eq!(ours.points(), 4 * theirs.points());
    assert_eq!(
        ours.spin_orbit.keys().collect::<Vec<_>>(),
        theirs.spin_orbit.keys().collect::<Vec<_>>()
    );
    let mut deviations = Vec::new();
    for (l, so) in &ours.spin_orbit {
        let other = &theirs.spin_orbit[l];
        assert_eq!(so.projectors.len(), other.projectors.len());
        for (i, (p, q)) in so.projectors.iter().zip(&other.projectors).enumerate() {
            // The shipped tables keep every fourth node, ending on the last one of each block.
            let decimated = (0..q.as_slice().len())
                .map(|k| p.as_slice()[4 * k + 3])
                .collect::<Vec<_>>();
            deviations.push((
                format!("l{l} projector {i}"),
                worst(&decimated, q.as_slice()),
                TABLE_GATE,
            ));
        }
        for (name, a, b) in [
            ("average", &so.average, &other.average),
            (
                "average_screened",
                &so.average_screened,
                &other.average_screened,
            ),
            ("coupling", &so.coupling, &other.coupling),
            (
                "coupling_screened",
                &so.coupling_screened,
                &other.coupling_screened,
            ),
            (
                "overlap_average",
                &so.overlap_average,
                &other.overlap_average,
            ),
            (
                "overlap_coupling",
                &so.overlap_coupling,
                &other.overlap_coupling,
            ),
        ] {
            deviations.push((format!("l{l} {name}"), matrices(a, b), MATRIX_GATE));
        }
        deviations.push((
            format!("l{l} reference_energies"),
            matrices(&so.reference_energies, &other.reference_energies),
            MATRIX_GATE,
        ));
        deviations.push((
            format!("l{l} level_errors"),
            matrices(&so.level_errors, &other.level_errors),
            MATRIX_GATE,
        ));
    }
    for (name, a, b) in [
        (
            "one_center_energy",
            ours.one_center_energy,
            theirs.one_center_energy,
        ),
        (
            "compensation_charge",
            ours.compensation_charge,
            theirs.compensation_charge,
        ),
        (
            "hartree_screening",
            ours.hartree_screening,
            theirs.hartree_screening,
        ),
    ] {
        deviations.push((
            name.to_string(),
            (a - b).abs() / b.abs().max(1.),
            SCALAR_GATE,
        ));
    }
    let mut failures = Vec::new();
    for (name, deviation, gate) in &deviations {
        println!("Na lda-fr {name}: {deviation:.3e} (gate {gate:.1e})");
        if !(deviation <= gate) {
            failures.push(format!("{name}: {deviation:e} > {gate:e}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
