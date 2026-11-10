//! The verification gates of `paw::diagnose::quality`, each with a bad case it must catch and a
//! good one it must pass. The bad cases are generated here from the default recipe with one stated
//! change, so a control does not depend on the shipped library (which the gates are used to
//! improve); the good cases are shipped files the gates measured clean:
//!
//! - hardness and working spacing: copper with its 3d forced into a 1.0 Bohr sphere (as compact as
//!   the shipped pbe-sr Br 3d that moved HBr by 186 eV) against shipped pbe-sr O (0.017 Ha, working
//!   spacing 0.15 A) and Cl
//!   (0.20 A; HCl moves 5 meV between 0.20 and 0.15 A, the measured outcome the tolerance is
//!   calibrated on);
//! - sphere against bond: oxygen with 2.0 Bohr spheres against a 0.96 A O-H bond (1.81 Bohr), the
//!   shipped O (1.45 Bohr) against the same bond;
//! - sr/fr partition: sodium without its 2s2p semicore against sodium with it, and shipped pbe-sr
//!   against pbe-fr O;
//! - a bound all-electron level in a channel without projectors: oxygen with its p channel removed
//!   from the smooth view misses the occupied 2p (-0.33 Ha), which the local potential alone does
//!   not bind; the intact oxygen does not;
//! - defects a stale record does not store, on synthetic records.

mod paw_assemble_support;
mod paw_diagnose_support;

use mandacaru_core::models::{atom::Atom, relativity::Treatment};
use mandacaru_core::paw::{
    diagnose::{
        quality::{
            grid_residual, hardness, partition_mismatch, residual_kinetic, sphere_reach,
            unprojected_missing_states, unrecorded_defects, working_spacing, HARDNESS_TOLERANCE,
        },
        reference::AllElectron,
        view::Smooth,
    },
    generate::{
        generate_with_atom,
        recipe::{Cutoffs, Recipe},
        reference::solve_reference,
    },
    record::{Defects, PawDataset},
};
use paw_diagnose_support::*;

const BOHR_PER_ANGSTROM: f64 = 1. / 0.529177210903;

/// A dataset of `symbol` (LDA, scalar-relativistic) from the default recipe with `change`
/// applied, the first construction (no search), and the atom it was built on.
fn generated(
    ctx: &mandacaru_core::Context,
    symbol: &str,
    change: impl FnOnce(&mut Recipe),
) -> (PawDataset, Atom) {
    let mut recipe = Recipe::for_dataset(symbol, "lda", Treatment::Scalar).unwrap();
    change(&mut recipe);
    let atom = solve_reference(ctx, &recipe).unwrap();
    let dataset = generate_with_atom(
        ctx,
        &recipe,
        &atom,
        &mandacaru_core::models::xc_scheme::Variational,
    )
    .unwrap()
    .dataset;
    (dataset, atom)
}

/// Copper's 3d forced into a 1.0 Bohr sphere (the table gives 2.2).
fn compact_d(recipe: &mut Recipe) {
    recipe.cutoff_table.insert(2, 1.0);
}

#[test]
#[ignore = "slow: 11 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test paw_diagnose_quality -- --ignored hardness_flags"]
fn hardness_flags_a_wave_forced_into_a_short_sphere_and_passes_oxygen() {
    let ctx = context();
    let (hard, _) = generated(&ctx, "Cu", compact_d);
    let o = read(&ctx, &dataset_path("pbe-sr", "O"));
    eprintln!(
        "Cu with a 1.0 Bohr d sphere: residual {:?}; shipped O {:?}",
        residual_kinetic(&hard),
        residual_kinetic(&o)
    );
    let d = *hardness(&hard)
        .get(&2)
        .expect("the 3d in a 1.0 Bohr sphere is hard");
    assert!(d > 2. * HARDNESS_TOLERANCE, "{d}");
    assert!(hardness(&o).is_empty());
    let o_worst = residual_kinetic(&o).values().copied().fold(0., nan_max);
    assert!(o_worst < HARDNESS_TOLERANCE / 20., "{o_worst}");
}

#[test]
#[ignore = "slow: 21 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test paw_diagnose_quality -- --ignored the_working_spacing"]
fn the_working_spacing_separates_converging_from_hard_datasets() {
    let ctx = context();
    for (element, expected) in [("Cl", Some(0.2)), ("O", Some(0.15))] {
        let ds = read(&ctx, &dataset_path("pbe-sr", element));
        let got = working_spacing(&ctx, &ds).unwrap();
        eprintln!("pbe-sr {element}: working spacing {got:?}");
        assert_eq!(got, expected, "{element}");
    }
    let (hard, _) = generated(&ctx, "Cu", compact_d);
    let at = grid_residual(&ctx, &hard, 0.1).unwrap();
    eprintln!("Cu with a 1.0 Bohr d sphere: grid residual at 0.10 A {at:?}");
    assert_eq!(working_spacing(&ctx, &hard).unwrap(), None);
}

#[test]
fn sphere_reach_finds_a_neighbor_nucleus_inside_a_wide_sphere() {
    let ctx = context();
    let o_h = 0.96 * BOHR_PER_ANGSTROM;
    let (wide, _) = generated(&ctx, "O", |r| r.cutoffs = Cutoffs::Uniform(2.0));
    let reach = sphere_reach(&wide, o_h).expect("a 2.0 Bohr sphere holds the H of O-H");
    eprintln!("O at 2.0 Bohr: widest sphere {reach:.3} Bohr against O-H {o_h:.3} Bohr");
    let o = read(&ctx, &dataset_path("lda-sr", "O"));
    assert!(sphere_reach(&o, o_h).is_none());
}

#[test]
fn partition_mismatch_and_unprojected_missing_states_on_sodium_without_its_semicore() {
    let ctx = context();
    let (with, atom_with) = generated(&ctx, "Na", |_| {});
    let (without, atom_without) = generated(&ctx, "Na", |r| r.semicore_subshells.clear());
    let message = partition_mismatch(&with, &without).expect("the partitions differ");
    eprintln!("{message}");
    let o_sr = read(&ctx, &dataset_path("pbe-sr", "O"));
    let o_fr = read(&ctx, &dataset_path("pbe-fr", "O"));
    assert_eq!(partition_mismatch(&o_sr, &o_fr), None);
    assert!(partition_mismatch(&o_sr, &with).is_some());
    drop((atom_with, atom_without));
}

#[test]
fn a_bound_level_in_a_channel_without_projectors_is_a_missing_state() {
    let ctx = context();
    let (o, atom) = generated(&ctx, "O", |_| {});
    let intact = Smooth::scalar(&ctx, &o).unwrap();
    let ae = AllElectron::new(
        &atom,
        o.atomic_number as u8,
        &intact.frozen,
        &intact.semicore,
    );
    let complete = unprojected_missing_states(&intact, &ae).unwrap();
    let mut removed = Smooth::scalar(&ctx, &o).unwrap();
    removed.channels.retain(|key, _| key.l != 1);
    let lacking = unprojected_missing_states(&removed, &ae).unwrap();
    eprintln!("O intact: {complete:?}; without its p channel: {lacking:?}");
    assert!(complete.is_empty(), "{complete:?}");
    let offset = *lacking.get(&1).expect("the 2p is missed");
    assert!(offset.abs() > 10. * 1e-2, "{offset}");
}

#[test]
fn unrecorded_defects_are_what_the_fresh_diagnosis_adds() {
    let stored = Defects {
        phases: [(0, [0.06, 0.1])].into_iter().collect(),
        ..Defects::default()
    };
    let fresh = Defects {
        phases: [(0, [0.07, 0.1]), (3, [0.445, 0.5])].into_iter().collect(),
        s_miss: Some(1.2),
        ..Defects::default()
    };
    let extra = unrecorded_defects(&stored, &fresh);
    assert_eq!(extra.phases.keys().copied().collect::<Vec<_>>(), vec![3]);
    assert_eq!(extra.s_miss, Some(1.2));
    assert_eq!(unrecorded_defects(&fresh, &fresh), Defects::default());
}

/// `kinetic_above` at the dataset's own `q_cut` is the residual the generator stored: the same
/// transform of the same expansion, read back from the file (the tail on the stored grid).
#[test]
fn the_kinetic_energy_above_q_cut_is_the_stored_residual() {
    use mandacaru_core::paw::diagnose::quality::kinetic_above;
    let ctx = context();
    for (set, element, l) in [
        ("pbe-sr", "O", 1usize),
        ("pbe-sr", "Br", 2),
        ("lda-sr", "Gd", 3),
    ] {
        let ds = read(&ctx, &dataset_path(set, element));
        let channel = &ds.channels[&(l as u32)];
        let stored = channel.residual_kinetic.data()[0];
        let ours = kinetic_above(&ctx, &ds, l, 0, ds.q_cut).unwrap();
        eprintln!("{set} {element} l={l}: stored {stored:.6e}, recomputed {ours:.6e}");
        assert!(
            (ours - stored).abs() <= 1e-3 * stored.abs().max(1e-4),
            "{ours} vs {stored}"
        );
    }
}

#[test]
#[ignore = "calibration printout; run: cargo test --profile release-test -p mandacaru-core --test paw_diagnose_quality -- --ignored grid_residual_calibration --nocapture"]
fn grid_residual_calibration() {
    use mandacaru_core::paw::diagnose::quality::grid_residual;
    let ctx = context();
    for (set, element) in [
        ("pbe-sr", "O"),
        ("pbe-sr", "Cl"),
        ("pbe-sr", "Br"),
        ("pbe-sr", "Kr"),
        ("pbe-sr", "Xe"),
        ("pbe-sr", "I"),
        ("pbe-sr", "Zn"),
        ("pbe-sr", "Cu"),
        ("pbe-sr", "Fe"),
        ("lda-sr", "Gd"),
        ("lda-sr", "Ce"),
        ("pbe-fr", "Zn"),
        ("pbe-fr", "Ta"),
    ] {
        let ds = read(&ctx, &dataset_path(set, element));
        let line = [0.2, 0.15, 0.12, 0.1]
            .iter()
            .map(|&h| {
                let g = grid_residual(&ctx, &ds, h).unwrap();
                format!(
                    "h={h}: {:.3e} {:?}",
                    g.values().sum::<f64>(),
                    g.iter()
                        .map(|(l, v)| (*l, (v * 1e4).round() / 1e4))
                        .collect::<Vec<_>>()
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        eprintln!("{set} {element}: {line}");
    }
}

/// The atom-free gates of every dataset in `$PAW_CANDIDATE_DIR/<set>/` against its twin in the
/// shipped library: residual kinetic energy, grid residual and working spacing, the widest sphere
/// against a 2.35 Angstrom bond, and the partition against the shipped sr (for an fr file) or fr
/// (for an sr file) dataset. A report for staging candidates, not a gate of the suite.
#[test]
#[ignore = "candidate report; run: PAW_CANDIDATE_DIR=... cargo test --profile release-test -p mandacaru-core --test paw_diagnose_quality -- --ignored candidate_report --nocapture"]
fn candidate_report() {
    use mandacaru_core::paw::diagnose::quality::{
        grid_residual, hardness, partition_mismatch, residual_kinetic, sphere_reach,
        working_spacing,
    };
    let root =
        std::path::PathBuf::from(std::env::var("PAW_CANDIDATE_DIR").expect("PAW_CANDIDATE_DIR"));
    let ctx = context();
    for set in ["lda-sr", "pbe-sr", "lda-fr", "pbe-fr"] {
        let Ok(entries) = std::fs::read_dir(root.join(set)) else {
            continue;
        };
        let mut paths = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths
            .into_iter()
            .filter(|p| p.extension().is_some_and(|x| x == "parquet"))
        {
            let element = path.file_stem().unwrap().to_string_lossy().to_string();
            let ds = read(&ctx, &path);
            let twin_set = if set.ends_with("sr") {
                set.replace("sr", "fr")
            } else {
                set.replace("fr", "sr")
            };
            let twin_path = dataset_path(&twin_set, &element);
            let partition = twin_path
                .exists()
                .then(|| partition_mismatch(&ds, &read(&ctx, &twin_path)))
                .flatten();
            let total = |h: f64| grid_residual(&ctx, &ds, h).unwrap().values().sum::<f64>();
            println!(
                "{set} {element}: Zv {} defects {:?} | residual {:?} hard {:?} | grid 0.2/0.15/0.12/0.1 = {:.2e}/{:.2e}/{:.2e}/{:.2e} working {:?} | sphere vs 4.44 Bohr {:?} | partition vs {twin_set}: {}",
                ds.valence_charge,
                ds.defects,
                residual_kinetic(&ds).iter().map(|(l, v)| (*l, (v * 1e3).round() / 1e3)).collect::<Vec<_>>(),
                hardness(&ds).keys().collect::<Vec<_>>(),
                total(0.2), total(0.15), total(0.12), total(0.1),
                working_spacing(&ctx, &ds).unwrap(),
                sphere_reach(&ds, 2.35 / 0.529177210903),
                partition.unwrap_or_else(|| "same".into()),
            );
        }
    }
}
