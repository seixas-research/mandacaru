//! Positive and negative controls of the diagnostics: a WRONG dataset, built from a shipped
//! one by a single stated change, must be caught by the check that exists for it, and the miss
//! (its deviation from the oracle of the clean dataset) must exceed 100 times the gate of that
//! check. The controls of the controls: the clean dataset passes the same check, and the
//! separable eigensolver the spectra stand on equals a dense solve of the same matrices.
//!
//! - a deep local well: a ghost state below the reference (ghost gate `GHOST_TOLERANCE`);
//! - the screened coupling scaled by 1.3: a scattering phase error (gate `PHASE_AGREEMENT`);
//! - a removed projector: a missing bound state (gate `MISSING_TOLERANCE`);
//! - a potential that is not finite: the check is unevaluated, never a pass.

mod paw_assemble_support;
mod paw_diagnose_support;

use mandacaru_core::models::atom::Atom;
use mandacaru_core::models::xc_scheme::AtomSolver;
use mandacaru_core::paw::diagnose::{
    ghosts::{ghost_errors, missing_states, reference_offsets, GHOST_TOLERANCE, MISSING_TOLERANCE},
    reference::{reference_atom_with, AeLogs, AllElectron},
    scattering::{scattering_errors, PHASE_TOLERANCE, RESONANCE_TOLERANCE},
    spectrum::{self, nodes},
    view::{Key, Smooth},
    Error,
};
use paw_diagnose_support::*;

/// Agreement expected of the phase of a dataset with itself across the two resolutions of the
/// tables (rad); the bite of a scattering control is measured against it.
const PHASE_AGREEMENT: f64 = 1e-3;

struct Bench {
    ctx: mandacaru_core::Context,
    smooth: Smooth,
    atom: Atom,
}

fn bench(set: &str, element: &str) -> Bench {
    let ctx = context();
    let ds = read(&ctx, &dataset_path(set, element));
    let atom = reference_atom_with(&ctx, &ds, &AtomSolver).unwrap();
    let smooth = smooth_of(&ctx, &ds);
    Bench { ctx, smooth, atom }
}

fn worst_phase(b: &Bench) -> f64 {
    let ae = AllElectron::new(
        &b.atom,
        b.smooth.atomic_number,
        &b.smooth.frozen,
        &b.smooth.semicore,
    );
    scattering_errors(&b.ctx, &b.smooth, &ae, &mut AeLogs::new())
        .unwrap()
        .values()
        .flat_map(|v| v.iter().copied())
        .fold(0., nan_max)
}

/// A Gaussian well of `depth` Hartree and 1/e radius `width` Bohr added to the local potential.
fn add_well(smooth: &mut Smooth, depth: f64, width: f64) {
    let r = smooth.r.as_slice().to_vec();
    for (v, r) in smooth.v_local_screened.as_mut_slice().iter_mut().zip(r) {
        *v -= depth * (-(r / width).powi(2)).exp();
    }
}

#[test]
fn the_separable_solver_equals_a_dense_solve_of_the_same_matrices() {
    let b = bench("lda-sr", "O");
    let grid = nodes(8., 0.02);
    for l in 0..2 {
        let channel = b.smooth.channel(l).unwrap();
        let prepared = spectrum::prepared(&b.ctx, &b.smooth, channel).unwrap();
        let dense = dense_levels(&b.ctx, &b.smooth, channel, &grid, 3);
        for (state, want) in dense.iter().enumerate() {
            let got = prepared
                .solve_on(&b.ctx, &grid, None, state, false)
                .unwrap()
                .0;
            eprintln!("O l={l} state {state}: separable {got:.12} dense {want:.12}");
            assert!(
                (got - want).abs() < 1e-9,
                "l={l} state {state}: {got} vs {want}"
            );
        }
    }
}

/// The deepest ghost (`first - reference`, Hartree) of a Gaussian well of `depth` Hartree and
/// 1/e radius 0.45 Bohr added to the local potential of `bench`; the potential is restored.
fn deepest_ghost(b: &mut Bench, depth: f64) -> f64 {
    let original = b.smooth.v_local_screened.as_slice().to_vec();
    add_well(&mut b.smooth, depth, 0.45);
    let found = ghost_errors(&b.ctx, &b.smooth, None).unwrap();
    b.smooth
        .v_local_screened
        .as_mut_slice()
        .copy_from_slice(&original);
    found.values().copied().fold(0., |a, b| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    })
}

#[test]
fn a_deep_local_well_binds_a_ghost_at_the_depths_vera_measured() {
    let mut b = bench("lda-sr", "O");
    let clean = ghost_errors(&b.ctx, &b.smooth, None).unwrap();
    assert!(
        clean.is_empty(),
        "the shipped oxygen binds no ghost: {clean:?}"
    );
    let grid = nodes(8., 0.02);
    for target in [-11., -27.] {
        // Bisect the depth of the well until the deepest ghost is at the target.
        let (mut low, mut high) = (0., 200.);
        for _ in 0..40 {
            let middle = 0.5 * (low + high);
            if deepest_ghost(&mut b, middle) > target {
                low = middle;
            } else {
                high = middle;
            }
        }
        let depth = 0.5 * (low + high);
        let ghost = deepest_ghost(&mut b, depth);
        assert!(
            (ghost - target).abs() < 1e-3,
            "target {target}: {ghost} at well {depth}"
        );
        add_well(&mut b.smooth, depth, 0.45);
        let found = ghost_errors(&b.ctx, &b.smooth, None).unwrap();
        for (&l, &error) in &found {
            // The oracle of the ghost level: the dense solve of the very same wrong matrices, on
            // the single grid the separable solver is also asked for (no extrapolation).
            let channel = b.smooth.channel(l).unwrap();
            let prepared = spectrum::prepared(&b.ctx, &b.smooth, channel).unwrap();
            let separable = prepared.solve_on(&b.ctx, &grid, None, 0, false).unwrap().0;
            let dense = dense_levels(&b.ctx, &b.smooth, channel, &grid, 1)[0];
            eprintln!(
                "well {depth:.3} Ha: l={l} ghost {error:+.4} Ha; level {separable:+.9} separable, {dense:+.9} dense"
            );
            assert!(
                (separable - dense).abs() < 1e-8,
                "l={l}: {separable} vs {dense}"
            );
        }
        assert!(
            ghost.abs() / GHOST_TOLERANCE > 100.,
            "the ghost must exceed 100 gates"
        );
        let ds = read(&b.ctx, &dataset_path("lda-sr", "O"));
        b.smooth
            .v_local_screened
            .as_mut_slice()
            .copy_from_slice(ds.v_local_screened.as_slice());
    }
}

/// `(l, near, far)` of every channel.
fn phases(b: &Bench) -> Vec<(usize, f64, f64)> {
    let ae = AllElectron::new(
        &b.atom,
        b.smooth.atomic_number,
        &b.smooth.frozen,
        &b.smooth.semicore,
    );
    scattering_errors(&b.ctx, &b.smooth, &ae, &mut AeLogs::new())
        .unwrap()
        .iter()
        .map(|(k, v)| (k.l, v[0], v[1]))
        .collect()
}

#[test]
#[ignore = "slow: 31 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_diagnose_controls -- --ignored scaling_the_screened_coupling_by_one_point_three_scatters_wrongly"]
fn scaling_the_screened_coupling_by_one_point_three_scatters_wrongly() {
    // vera's figures for this change are the s channel's phase error: 0.19 rad (O), 1.18 rad (Fe).
    for (element, vera) in [("O", 0.19), ("Fe", 1.18)] {
        let mut b = bench("lda-sr", element);
        let clean = phases(&b);
        for channel in b.smooth.channels.values_mut() {
            channel.coupling.iter_mut().for_each(|d| *d *= 1.3);
        }
        let wrong = phases(&b);
        eprintln!("{element}: coupling x1.3, per channel (l, near, far) clean -> wrong:");
        for (c, w) in clean.iter().zip(&wrong) {
            eprintln!(
                "  l={}: ({:.3e}, {:.3e}) -> ({:.4}, {:.4})",
                c.0, c.1, c.2, w.1, w.2
            );
        }
        // The shipped dataset scatters inside the tolerances it was accepted with.
        for (l, near, far) in &clean {
            assert!(
                *near < PHASE_TOLERANCE && *far < RESONANCE_TOLERANCE,
                "{element} l={l}: the shipped channel scatters like the atom"
            );
        }
        let s = wrong.iter().find(|p| p.0 == 0).unwrap();
        assert!(
            (s.1 - vera).abs() < 0.005,
            "{element}: s channel {} rad against vera's {vera}",
            s.1
        );
        // Every channel is wrong by more than 100 gates of the agreement across table resolutions.
        for (c, w) in clean.iter().zip(&wrong) {
            assert!(
                (w.1 - c.1) / PHASE_AGREEMENT > 100.,
                "{element} l={}: the miss must exceed 100 gates",
                c.0
            );
        }
    }
}

/// Remove projector `k` of channel `l` of `b` and report `(offset, distance)` of that channel
/// from `reference_offsets` / `missing_states`; the dataset is restored.
fn remove_projector(b: &mut Bench, l: usize, k: usize) -> (f64, Option<f64>) {
    let n = b.smooth.r.as_slice().len();
    let key = Key { l, kappa: None };
    let broken = without_projector(&b.ctx, b.smooth.channel(l).unwrap(), k, n);
    let kept = b.smooth.channels.insert(key, broken).unwrap();
    let offset = reference_offsets(&b.ctx, &b.smooth).unwrap()[&l];
    let missing = missing_states(&b.ctx, &b.smooth).unwrap().get(&l).copied();
    b.smooth.channels.insert(key, kept);
    (offset, missing)
}

#[test]
fn a_removed_projector_leaves_a_physical_state_unbound() {
    // The tolerance clears the largest offset of a legitimate library channel (a lanthanide 4f at
    // 4.3e-3 Ha). Oxygen's 2p sits at -0.34 Ha, so removing its bound-state projector moves the
    // nearest level by 0.36 Ha: caught at 36 gates. Sodium's 2s semicore state sits at -2 Ha:
    // removing its projector leaves the reference unbound by about 2 Ha, over 100 gates.
    let mut oxygen = bench("lda-sr", "O");
    assert!(missing_states(&oxygen.ctx, &oxygen.smooth)
        .unwrap()
        .is_empty());
    let (offset, missing) = remove_projector(&mut oxygen, 1, 0);
    eprintln!(
        "oxygen p without its 2p projector: offset {offset:+.4} Ha ({:.0} gates)",
        offset.abs() / MISSING_TOLERANCE
    );
    assert!(
        missing
            .expect("the 2p reference is not bound any more")
            .abs()
            / MISSING_TOLERANCE
            > 30.
    );
    assert!(worst_phase(&oxygen) < PHASE_TOLERANCE, "restored");
    let mut sodium = bench("lda-sr", "Na");
    assert!(missing_states(&sodium.ctx, &sodium.smooth)
        .unwrap()
        .is_empty());
    let (offset, missing) = remove_projector(&mut sodium, 0, 0);
    eprintln!(
        "sodium s without its 2s projector: offset {offset:+.4} Ha ({:.0} gates)",
        offset.abs() / MISSING_TOLERANCE
    );
    assert!(
        missing
            .expect("the 2s reference is not bound any more")
            .abs()
            / MISSING_TOLERANCE
            > 100.
    );
    assert!(
        missing_states(&sodium.ctx, &sodium.smooth)
            .unwrap()
            .is_empty(),
        "restored"
    );
}

#[test]
fn a_potential_that_is_not_finite_is_never_a_pass() {
    let mut b = bench("lda-sr", "O");
    b.smooth.v_local_screened.as_mut_slice()[100] = f64::NAN;
    let outcome = ghost_errors(&b.ctx, &b.smooth, None);
    assert!(
        matches!(outcome, Err(Error::Failed(_)) | Err(Error::Unevaluated(_))),
        "a NaN in the potential must fail the check, got {outcome:?}"
    );
}

#[test]
fn the_recording_rule_keeps_ghosts_alone_and_applies_each_tolerance() {
    use mandacaru_core::paw::diagnose::Diagnosis;
    let channel = |l| Key { l, kappa: None };
    let mut d = Diagnosis::default();
    d.phases.insert(channel(1), [0.06, 0.1]); // near over 0.05
    d.phases.insert(channel(2), [0.01, 0.31]); // far over 0.3
    d.phases.insert(channel(3), [0.05, 0.3]); // exactly at the limits: inside
    d.phases.insert(
        Key {
            l: 4,
            kappa: Some(4),
        },
        [9., 9.],
    ); // a branch is not a recorded channel
    d.unprojected.insert(5, 0.21);
    d.unprojected.insert(6, 0.2); // exactly at the limit: inside
    d.s_miss = Some(1.0); // strict: the miss must be below 1
    let defects = d.defects();
    assert_eq!(defects.phases.keys().copied().collect::<Vec<_>>(), [1, 2]);
    assert_eq!(defects.phases[&1], [0.06, 0.1]);
    assert_eq!(defects.unprojected.keys().copied().collect::<Vec<_>>(), [5]);
    assert_eq!(defects.s_miss, Some(1.0));
    assert!(defects.ghosts.is_empty() && defects.residuals.is_empty());
    assert!(!d.is_clean());
    // A ghosted construction records its ghosts and nothing else.
    d.ghosts.insert(0, -3.0);
    let only = d.defects();
    assert_eq!(only.ghosts[&0], -3.0);
    assert!(only.phases.is_empty() && only.unprojected.is_empty() && only.s_miss.is_none());
    // Nothing found, nothing recorded; a missing state alone makes the dataset unclean.
    let mut clean = Diagnosis::default();
    clean.s_miss = Some(0.99);
    assert!(clean.is_clean());
    clean.missing.insert(1, 0.5);
    assert!(!clean.is_clean() && clean.defects().phases.is_empty());
}

/// An all-electron reference atom that did not converge is refused in words that say so, with
/// the iteration count; the functional/treatment mismatch is a separate message. (The audit read
/// "converged false" inside a mismatch sentence for the pbe lanthanides.)
#[test]
fn an_unconverged_reference_atom_is_named_as_such() {
    use mandacaru_core::models::{
        atom::{solve_model, AtomModel, AtomOptions},
        relativity::Treatment,
    };
    use mandacaru_core::paw::diagnose::reference::check_reference;
    let ctx = context();
    let ds = read(&ctx, &dataset_path("lda-sr", "O"));
    let options = |max_iterations| AtomOptions {
        points: 4000,
        r_max: 25.,
        max_iterations,
        tolerance: 1e-7,
        mixing: 0.25,
    };
    let model = |functional: &str| AtomModel {
        functional: functional.into(),
        relativity: Treatment::Scalar,
        polish: 0,
    };
    let stopped = solve_model(&ctx, 8, &options(2), &model("lda"), None, None).unwrap();
    assert!(!stopped.converged);
    let message = check_reference(&ds, &stopped).unwrap_err().to_string();
    assert!(message.contains("did not converge in 2 iterations"), "{message}");
    let other = solve_model(&ctx, 8, &options(200), &model("pbe"), None, None).unwrap();
    assert!(other.converged);
    let message = check_reference(&ds, &other).unwrap_err().to_string();
    assert!(message.contains("is not the one the record asks for") && !message.contains("converge"), "{message}");
}
