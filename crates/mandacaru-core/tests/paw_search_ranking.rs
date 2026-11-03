//! The ranking of failed attempts (`paw::generate::ranking`, A6) on synthetic diagnoses: the
//! order of the generator this replaces (a ghost is worst, shallower beats deeper, then fewer
//! channels; a failed s completeness above any scattering defect; otherwise the worst defect
//! over its own tolerance), the earlier attempt winning a tie, and the numbers of the Ce flag
//! of the A0 freeze. Wrong rankings must pick another attempt, and by a large margin.
use mandacaru_core::paw::diagnose::{view::Key, Diagnosis};
use mandacaru_core::paw::generate::ranking::{Badness, Keeper};
use std::cmp::Ordering;
use std::collections::BTreeMap;

fn scattering(phases: &[(usize, [f64; 2])], unprojected: &[(usize, f64)]) -> Diagnosis {
    Diagnosis {
        phases: phases
            .iter()
            .map(|&(l, v)| (Key { l, kappa: None }, v))
            .collect(),
        unprojected: unprojected.iter().copied().collect(),
        ..Diagnosis::default()
    }
}

fn ghosted(levels: &[(usize, f64)]) -> Diagnosis {
    Diagnosis {
        ghosts: levels.iter().copied().collect(),
        ..Diagnosis::default()
    }
}

fn better(a: &Diagnosis, b: &Diagnosis) -> bool {
    Badness::of(a).compare(&Badness::of(b)) == Ordering::Less
}

#[test]
fn a_ghost_is_worse_than_any_scattering_defect() {
    let shallow_ghost = ghosted(&[(1, -2e-4)]);
    let wild = scattering(&[(0, [3.0, 3.0])], &[(2, 1.5)]);
    assert!(better(&wild, &shallow_ghost));
    assert!(!better(&shallow_ghost, &wild));
}

#[test]
fn among_ghosts_the_shallowest_deepest_wins_then_fewer_channels() {
    assert!(better(
        &ghosted(&[(1, -0.133)]),
        &ghosted(&[(0, -4.85), (1, -0.133)])
    ));
    assert!(better(&ghosted(&[(1, -0.133)]), &ghosted(&[(1, -0.419)])));
    // The same deepest ghost: the construction with fewer ghosted channels.
    assert!(better(
        &ghosted(&[(1, -0.3)]),
        &ghosted(&[(0, -0.1), (1, -0.3)])
    ));
}

#[test]
fn a_failed_s_completeness_ranks_above_every_scattering_defect() {
    let missed = Diagnosis {
        s_miss: Some(1.05),
        ..scattering(&[(0, [1e-4, 1e-4])], &[])
    };
    let wrong = scattering(&[(1, [2.0, 2.0])], &[(3, 5.0)]);
    assert!(better(&wrong, &missed));
    // Both failed: the size still counts.
    let worse = Diagnosis {
        s_miss: Some(23.8),
        ..Diagnosis::default()
    };
    assert!(better(&missed, &worse));
    // Below the tolerance the miss is not a defect at all.
    let passed = Diagnosis {
        s_miss: Some(0.99),
        ..Diagnosis::default()
    };
    assert_eq!(Badness::of(&passed).tier, 0);
    assert_eq!(Badness::of(&passed).severity, 0.0);
}

#[test]
fn the_worst_defect_over_its_own_tolerance_decides() {
    // 0.115 rad near is 2.3 tolerances; 0.4 rad of an unprojected channel is 2.0.
    let a = scattering(&[(3, [0.115, 0.115])], &[]);
    let b = scattering(&[], &[(2, 0.4)]);
    assert!(better(&b, &a));
    // The far phase has its own, looser, tolerance: 0.25 far is 0.83 tolerances, inside it.
    let c = scattering(&[(3, [0.0, 0.25])], &[]);
    assert!(better(&c, &a));
    // An unprojected channel inside its tolerance is no defect.
    let d = scattering(&[], &[(2, 0.19)]);
    assert_eq!(Badness::of(&d).severity, 0.0);
}

#[test]
fn the_flagged_cerium_numbers_rank_as_the_freeze_selected_them() {
    // Attempt 8 of the A0 table: phase l=3 0.2795/0.2795 rad, unprojected 1.18, 1.42, 0.206 rad.
    let selected = scattering(
        &[
            (3, [0.27953905359676146, 0.27953905359676146]),
            (0, [0.0158, 0.0158]),
        ],
        &[
            (1, 1.1816210963454088),
            (2, 1.4240461997593101),
            (4, 0.20569389773421332),
        ],
    );
    let badness = Badness::of(&selected);
    assert_eq!(badness.tier, 0);
    let expected = (1.4240461997593101f64 / 0.2).max(0.27953905359676146 / 0.05);
    assert!((badness.severity - expected).abs() < 1e-12);
    // The ghosted first construction of the same element ranks below it.
    let first = ghosted(&[(0, -0.131), (1, -0.133)]);
    assert!(better(&selected, &first));
}

#[test]
fn the_keeper_keeps_the_least_bad_and_the_earlier_one_on_a_tie() {
    let diagnoses = [
        ghosted(&[(1, -0.133)]),
        scattering(&[(3, [0.115, 0.115])], &[]),
        scattering(&[(3, [0.115, 0.115])], &[]),
        ghosted(&[(1, -0.5)]),
    ];
    let mut keeper = Keeper::new();
    for (i, d) in diagnoses.iter().enumerate() {
        keeper.offer(Badness::of(d), i);
    }
    assert_eq!(keeper.into_best(), Some(1));
}

#[test]
fn a_missing_state_only_breaks_a_tie() {
    let near = Diagnosis {
        missing: BTreeMap::from([(2, 0.02)]),
        ..scattering(&[(0, [0.2, 0.2])], &[])
    };
    let far = Diagnosis {
        missing: BTreeMap::from([(2, -0.2)]),
        ..scattering(&[(0, [0.2, 0.2])], &[])
    };
    assert!(better(&near, &far));
    // It never outweighs a scattering defect.
    let milder = scattering(&[(0, [0.19, 0.19])], &[]);
    assert!(better(&milder, &near));
}

/// Negative controls of the ranking: the reversed order, the ghost criterion dropped and the
/// order of the criteria swapped each pick another attempt than the oracle on the same set, and
/// the attempt they pick is much worse than the one the ranking keeps.
#[test]
fn wrong_rankings_choose_another_attempt_by_a_wide_margin() {
    let set = [
        ghosted(&[(0, -0.131), (1, -0.133)]),               // 0: ghost
        scattering(&[(3, [0.115, 0.115])], &[(1, 0.9)]),    // 1: severity 4.5
        scattering(&[(3, [0.0158, 0.0158])], &[(1, 1.18)]), // 2: severity 5.9
        Diagnosis {
            s_miss: Some(1.2),
            ..scattering(&[], &[])
        }, // 3: tier 1
    ];
    let oracle = |rank: &dyn Fn(&Diagnosis, &Diagnosis) -> bool| -> usize {
        let mut best = 0;
        for i in 1..set.len() {
            if rank(&set[i], &set[best]) {
                best = i;
            }
        }
        best
    };
    let right = oracle(&|a, b| better(a, b));
    assert_eq!(right, 1);
    let reversed = oracle(&|a, b| better(b, a));
    assert_eq!(reversed, 0, "the reversed ranking keeps the ghost");
    // The ghost criterion dropped: the ghosted attempt looks perfect (no scattering to speak
    // of) and is kept; the margin is the depth of the ghost over its tolerance.
    let blind = |d: &Diagnosis| Diagnosis {
        ghosts: BTreeMap::new(),
        ..d.clone()
    };
    let without_ghosts = oracle(&|a, b| better(&blind(a), &blind(b)));
    assert_eq!(without_ghosts, 0);
    let tolerance = mandacaru_core::paw::diagnose::ghosts::GHOST_TOLERANCE;
    let depth = 0.133;
    assert!(
        depth / tolerance > 100.0,
        "the ghost the blind ranking keeps is {} tolerances deep",
        depth / tolerance
    );
    // Tier 1 treated as scattering: the failed completeness would beat attempts 1 and 2.
    let flat = oracle(&|a, b| Badness::of(a).severity < Badness::of(b).severity);
    assert_ne!(flat, right);
}
