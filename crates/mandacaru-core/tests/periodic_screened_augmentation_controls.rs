//! Translation and negative controls of the compensation charges in the periodic screened
//! exchange (W-E4P / P1), and its memory. Every control damages one input of the augmentation
//! and must miss the true operator by more than 100 times the agreement gate; the damage is made
//! on the inputs (blocks, projections, centers, origin), never by a switch in the code.
mod periodic_screened_augmentation_support;
mod setup_support;

use mandacaru_core::{
    periodic::{
        screened_augmentation::Augmentation,
        screened_exchange::{exchange_operators_with, Screening},
        Crystal,
    },
    Context, Error,
};
use num_complex::Complex64;
use periodic_screened_augmentation_support::*;

/// Agreement gate (relative to the largest entry of K) and the factor a control must exceed it by.
const GATE: f64 = 1e-11;
const BITE: f64 = 100. * GATE;
const SHIFT: [f64; 3] = [0.37, -1.10, 0.81];
const SCREENING: Screening = Screening {
    omega: 0.11,
    fraction: 0.25,
};

fn operators(c: &Crystal, d: &[Matrix], rig: &Rig, geometry: &Geometry) -> Vec<Matrix> {
    let sites = geometry.sites();
    let aug = Augmentation::new(c, sites, &rig.shapes).unwrap();
    exchange_operators_with(c, d, SCREENING, GB, Some(&aug))
        .unwrap()
        .operators
}
fn worst(a: &[Matrix], b: &[Matrix]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| relative(a, b))
        .fold(0., f64::max)
}

#[test]
fn translating_the_whole_crystal_leaves_k_unchanged() {
    // The same sampled numbers on a grid whose first node moved by SHIFT, the atoms moved with
    // it: K_{mu nu} = sum rho_mu rho_nu^* is invariant, and it is only so when the phase of the
    // grid origin (at G) and the phase of the atoms (at Q) agree.
    for (fixture, gamma, occupied) in [
        ("h2_skew_pbe.json", false, vec![1usize]),
        ("si_crystal.json", true, vec![4]),
    ] {
        let rig = Rig::new(fixture, gamma, &occupied);
        let moved = hybrid_at(&rig.ctx, fixture, gamma, SHIFT);
        let base = operators(&rig.crystal, &rig.d, &rig, &rig.geometry);
        let same = worst(
            &operators(&moved, &rig.d, &rig, &rig.geometry.translated(SHIFT)),
            &base,
        );
        let forgot_atoms = worst(&operators(&moved, &rig.d, &rig, &rig.geometry), &base);
        println!("{fixture}: translated K differs by {same:.2e}; origin moved without the atoms: {forgot_atoms:.2e}");
        assert!(same < GATE, "{same:e}");
        assert!(forgot_atoms > BITE, "{forgot_atoms:e}");
    }
}

#[test]
fn each_damaged_input_misses_the_true_operator_by_more_than_one_hundred_gates() {
    let mut conjugation_tested = false;
    for (fixture, gamma, occupied) in [
        ("si_crystal.json", true, vec![4usize]),
        ("h2_skew_pbe.json", false, vec![1]),
    ] {
        let mut rig = Rig::new(fixture, gamma, &occupied);
        let truth = operators(&rig.crystal, &rig.d, &rig, &rig.geometry);
        let miss = |rig: &Rig, label: &str| {
            let got = operators(&rig.crystal, &rig.d, rig, &rig.geometry);
            let value = worst(&got, &truth);
            println!(
                "{fixture}: {label}: misses the true K by {value:.2e} ({:.1e} gates)",
                value / GATE
            );
            assert!(value > BITE, "{fixture} {label}: {value:e}");
        };
        // 1. The compensation charges dropped.
        let none = exchange_operators_with(&rig.crystal, &rig.d, SCREENING, GB, None)
            .unwrap()
            .operators;
        let value = worst(&none, &truth);
        println!("{fixture}: compensation dropped: {value:.2e}");
        assert!(value > BITE);
        // 2. Wrong sign of the moments.
        rig.crystal
            .blocks
            .as_mut_slice()
            .iter_mut()
            .for_each(|x| *x = -*x);
        miss(&rig, "wrong sign of the blocks");
        rig.crystal
            .blocks
            .as_mut_slice()
            .iter_mut()
            .for_each(|x| *x = -*x);
        // 3. Every block transposed.
        let p = rig.crystal.p;
        let transpose = |blocks: &mut [Complex64]| {
            for block in blocks.chunks_mut(p * p) {
                for i in 0..p {
                    for j in 0..i {
                        block.swap(i * p + j, j * p + i);
                    }
                }
            }
        };
        transpose(rig.crystal.blocks.as_mut_slice());
        miss(&rig, "transposed blocks");
        transpose(rig.crystal.blocks.as_mut_slice());
        // 4. The phase of the atoms dropped: every center at the origin.
        let flat = Geometry {
            atoms: rig
                .geometry
                .atoms
                .iter()
                .map(|a| mandacaru_core::periodic::setup::shapes::Atom {
                    center: [0.; 3],
                    ..*a
                })
                .collect(),
            channels: rig.geometry.channels.clone(),
            lattice: rig.geometry.lattice,
        };
        let value = worst(&operators(&rig.crystal, &rig.d, &rig, &flat), &truth);
        println!("{fixture}: atom phase dropped: {value:.2e}");
        assert!(value > BITE);
        // 5. The projections conjugated (a transposed C), where they are complex.
        let imaginary = rig
            .crystal
            .points
            .iter()
            .flat_map(|p| p.projections.iter())
            .map(|v| v.im.abs())
            .fold(0., f64::max);
        if imaginary > 1e-6 {
            conjugation_tested = true;
            for point in rig.crystal.points.iter_mut() {
                point.projections = point.projections.conjugate();
            }
            miss(&rig, "conjugated projections");
            for point in rig.crystal.points.iter_mut() {
                point.projections = point.projections.conjugate();
            }
        }
        // The undamaged crystal is back.
        assert!(
            worst(
                &operators(&rig.crystal, &rig.d, &rig, &rig.geometry),
                &truth
            ) == 0.
        );
    }
    assert!(conjugation_tested, "no fixture has complex projections");
}

/// The least headroom (bytes of context budget beyond the loaded crystal) the exchange needs.
fn least_headroom(rig: &Rig, fixture: &str, gamma: bool, augmented: bool) -> usize {
    const LIMIT: usize = 1 << 30;
    let sites = rig.geometry.sites();
    let (mut low, mut high) = (0usize, 64usize << 20);
    while high - low > 256 {
        let middle = (low + high) / 2;
        let ctx = Context::new(Some(1), LIMIT).unwrap();
        let crystal = hybrid(&ctx, fixture, gamma);
        let _filler = ctx.reserve(LIMIT - ctx.memory_used() - middle).unwrap();
        let aug = Augmentation::new(&crystal, sites, &rig.shapes).unwrap();
        let attempt =
            exchange_operators_with(&crystal, &rig.d, SCREENING, GB, augmented.then_some(&aug));
        match attempt {
            Ok(_) => high = middle,
            Err(Error::Budget { .. }) => low = middle,
            Err(other) => panic!("{other:?}"),
        }
    }
    high
}

#[test]
#[ignore = "slow: 11 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_screened_augmentation_controls -- --ignored the_compensation_costs_one_channels_by_grid_block_and_obeys_the_budget"]
fn the_compensation_costs_one_channels_by_grid_block_and_obeys_the_budget() {
    let (fixture, gamma) = ("si_crystal.json", true);
    let rig = Rig::new(fixture, gamma, &[4]);
    let (n, channels) = (rig.crystal.grid.index.points(), rig.crystal.channels);
    let (plain, augmented) = (
        least_headroom(&rig, fixture, gamma, false),
        least_headroom(&rig, fixture, gamma, true),
    );
    let block = channels * n * 16;
    println!("least headroom {plain} B without, {augmented} B with the compensation (one {block} B channels x grid block)");
    // The compensation adds the channels x grid block and the per-state moments, nothing of the
    // size of (basis x occupied states x grid).
    assert!(augmented > plain + block / 2 && augmented <= plain + 2 * block + (1 << 20));
    // The budget refuses it cleanly and the context recovers.
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let crystal = hybrid(&ctx, fixture, gamma);
    let used = ctx.memory_used();
    let _filler = ctx.reserve((1 << 30) - used - (plain + 4096)).unwrap();
    let sites = rig.geometry.sites();
    let aug = Augmentation::new(&crystal, sites, &rig.shapes).unwrap();
    exchange_operators_with(&crystal, &rig.d, SCREENING, GB, None).unwrap();
    let refused = exchange_operators_with(&crystal, &rig.d, SCREENING, GB, Some(&aug));
    assert!(
        matches!(refused, Err(Error::Budget { .. })),
        "{:?}",
        refused.map(|_| ())
    );
    assert_eq!(ctx.memory_used(), (1 << 30) - (plain + 4096));
}
