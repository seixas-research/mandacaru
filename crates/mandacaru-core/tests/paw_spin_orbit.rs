//! The fully relativistic PAW generator (`paw::generate::spin_orbit`, A4) against the Dirac freeze
//! of the plan (section 3.4): the A0 freeze of the Python generator with `relativity="dirac"`
//! (`test/migration/reference/paw_generate/gen_{lda,pbe}-fr_*`), first construction, the remedy
//! of the ghost search applied as recipe fields where it had to repair.
//!
//! Layers: the Dirac reference atom (iterations, energy, every `(n, l)` average and every
//! `(n, l, kappa)` level), then the whole record: the scalar channels as for any dataset and the
//! j-resolved spin-orbit block (union projectors, average and coupling matrices, overlap split,
//! reference energies, level errors), item by item. Negative controls are in
//! `paw_spin_orbit_controls.rs`, identities and the shipped-file check in
//! `paw_spin_orbit_identities.rs`.
mod paw_assemble_support;
mod paw_generate_support;
mod paw_spin_orbit_support;

use paw_generate_support::compare::{compare_atom, floor, lifted, Scheme};
use paw_generate_support::dataset::compare_dataset_with;
use paw_generate_support::Decimated;
use paw_spin_orbit_support::items::spin_orbit_items;
use paw_spin_orbit_support::{context, open, Case, HEAVY, LIGHT};

/// One frozen Dirac dataset, atom first.
fn parity(name: &str) {
    let ctx = context();
    let case = open(&ctx, name);
    compare_atom(&case.frozen, &case.atom).assert_within(&format!("{name} atom"));
    let data = Decimated::open(name);
    let started = std::time::Instant::now();
    let generated = case.generate(&ctx);
    println!(
        "{name}: generated in {:.1} s, context peak {} B",
        started.elapsed().as_secs_f64(),
        ctx.memory_peak()
    );
    assert!(
        !generated.dataset.spin_orbit.is_empty(),
        "{name}: a Dirac dataset carries a spin-orbit term"
    );
    // The PBE potential of the smooth density moves by up to 5.8e-12 of its maximum under one ulp of
    // the density in the Python generator itself (silicon, `xc_energies` floor of the freeze), so the
    // gate of `v_local` where the density is above the tail is 100 x that floor; LDA keeps 1e-12.
    let v_local_gate = if generated.dataset.xc == "pbe" {
        lifted(floor(&case.frozen, "xc_energies", "tables"))
    } else {
        1e-12
    };
    compare_dataset_with(
        &case.frozen,
        &data,
        &generated,
        Scheme::Legacy,
        v_local_gate,
    )
    .assert_within(name);
    // The spin-orbit block alone, so the log carries its own worst deviation.
    let so = spin_orbit_items(&case, &data, &generated.dataset.spin_orbit);
    let item = so.worst();
    println!(
        "{name}: spin-orbit block, {} items, worst {} {:.3e} (gate {:.1e}, ratio {:.3})",
        so.items.len(),
        item.name,
        item.deviation,
        item.gate,
        item.ratio()
    );
    drop(generated);
    let Case { atom, .. } = case;
    drop(atom);
    assert_eq!(
        ctx.memory_used(),
        0,
        "{name}: the generator gave its memory back"
    );
}

#[test]
#[ignore = "slow: 27 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_spin_orbit -- --ignored the_light_dirac_sample_equals_the_frozen_datasets"]
fn the_light_dirac_sample_equals_the_frozen_datasets() {
    for name in LIGHT {
        parity(name);
    }
}

macro_rules! heavy {
    ($($test:ident => $name:expr),*) => {$(
        #[test]
        #[ignore = "minutes: cargo test --release -p mandacaru-core --test paw_spin_orbit -- --ignored --test-threads=1"]
        fn $test() {
            parity($name);
        }
    )*};
}

heavy!(
    heavy_lda_fr_ce => "lda-fr_Ce",
    heavy_lda_fr_au => "lda-fr_Au",
    heavy_lda_fr_pb => "lda-fr_Pb"
);

#[test]
fn the_dirac_sample_covers_every_frozen_dirac_dataset() {
    let mut frozen = paw_assemble_support::datasets()
        .into_iter()
        .filter(|n| n.starts_with("lda-fr") || n.starts_with("pbe-fr"))
        .collect::<Vec<_>>();
    let mut ours = LIGHT
        .iter()
        .chain(HEAVY.iter())
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    frozen.sort();
    ours.sort();
    assert_eq!(frozen, ours);
}
