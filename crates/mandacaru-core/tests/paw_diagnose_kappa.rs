//! The `kappa`-resolved channels of the shipped fully relativistic datasets: the two `j`
//! branches recovered from the stored spin-orbit blocks are ordinary channels of the
//! diagnostics. The checks: the two ways to recover a branch agree; the branch levels
//! recomputed from the stride-4 tables equal the level errors the generator stored; the
//! branches scatter like the Dirac atom of their own `kappa`, and branches given the wrong
//! `kappa` do not (the control, 100 times over the gate).
//!
//! The generated side of the A0 freeze is not used: the Dirac generation is the spin-orbit
//! chain (A4) and the freeze stores no full-resolution table of it. The shipped files are the
//! read-only oracle (`MANDACARU_PAW_PATH`); Na and Fe are not among the stale lda-fr elements.

mod paw_assemble_support;
mod paw_diagnose_support;

use mandacaru_core::paw::diagnose::{
    kappa::{branches, level_errors, recovery_mismatch},
    reference::{reference_atom, AeLogs, AllElectron},
    scattering::scattering_errors,
    spectrum::branch_level_error,
    view::Key,
};
use paw_diagnose_support::*;

/// Gate on the level errors (Hartree) of a `j` branch between the stored ones (computed from the
/// full-resolution tables) and the ones recomputed from the shipped stride-4 tables. Measured on
/// Na, Ne and Fe: 4.4e-8, 6.1e-5 and 1.4e-6 Ha (worst of the 147 shipped fr files in the sweep:
/// 1.4e-4, lda-fr Yb), and it does not shrink on finer tables in the
/// way an interpolation error would (below): the level of a branch depends on where the sphere
/// edge falls between table nodes. So the gate is the largest measured one, rounded up. The
/// 1e-9 mHa (1e-12 Ha) gate of the plan is a full-resolution gate for the in-memory output of
/// the spin-orbit chain, which this file cannot test.
const LEVEL_GATE: f64 = 2e-4;
/// Gate on the spin-orbit splitting of the level errors, `down - up`: the table-spacing shift is
/// common to both branches and cancels. Measured 9e-9 (Na), 1.5e-6 (Ne), 9e-7 (Fe) Ha.
const SPLITTING_GATE: f64 = 1e-5;
/// Gate on the disagreement of the two recoveries of a branch, relative to the largest entry.
const RECOVERY_GATE: f64 = 1e-9;
/// Agreement expected of a phase with itself across table resolutions (rad).
const PHASE_AGREEMENT: f64 = 1e-3;

#[test]
fn the_two_recoveries_of_a_branch_agree_and_its_level_is_the_stored_one() {
    let mut failures = Vec::new();
    for element in ["Na", "Ne", "Fe"] {
        let ctx = context();
        let ds = read(&ctx, &dataset_path("lda-fr", element));
        assert!(!ds.spin_orbit.is_empty());
        let mismatch = recovery_mismatch(&ds);
        let levels = level_errors(&ctx, &ds).unwrap();
        let mut worst = 0f64;
        let mut splitting = 0f64;
        for (l, so) in &ds.spin_orbit {
            for (b, &stored) in so.level_errors.data().iter().enumerate() {
                let got = levels[&(*l as usize)][b];
                eprintln!("{element} l={l} branch {b}: stored {stored:+.9e} recomputed {got:+.9e}");
                worst = nan_max(worst, (got - stored).abs());
            }
            let [d, u] = levels[&(*l as usize)];
            let stored = so.level_errors.data();
            splitting = nan_max(splitting, ((d - u) - (stored[0] - stored[1])).abs());
        }
        eprintln!("{element}: recovery mismatch {mismatch:.2e}, level deviation {worst:.2e} Ha, splitting deviation {splitting:.2e} Ha");
        if !(splitting <= SPLITTING_GATE) {
            failures.push(format!("{element}: splitting {splitting:.3e} Ha"));
        }
        assert!(mismatch <= RECOVERY_GATE, "{element}: {mismatch:.3e}");
        if !(worst <= LEVEL_GATE) {
            failures.push(format!("{element}: {worst:.3e} Ha"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

/// The deviation of a recomputed branch level from the stored one is of the size of the
/// sensitivity of that level to the spacing of the table it is computed from: on tables twice as
/// coarse the level moves by more than half the deviation (for Ne, Na and Fe it moves by 3e-4,
/// 2.6e-7 and 7e-5 Ha against deviations of 6e-5, 4e-8 and 1e-6), and does not converge toward
/// the stored value as `h^4` would (the extrapolations land at 8e-5, 3e-8 and 4e-6 off). It is
/// not an error of the port: the stored value comes from a table of another spacing.
#[test]
fn the_level_deviation_of_a_shipped_file_is_within_its_sensitivity_to_the_table_spacing() {
    for element in ["Na", "Ne", "Fe"] {
        let ctx = context();
        let ds = read(&ctx, &dataset_path("lda-fr", element));
        let set = branches(&ctx, &ds).unwrap();
        let coarse = decimated(&ctx, &set, 2);
        for (key, channel) in &set.channels {
            let slot = usize::from(key.kappa != Some(key.l as i32));
            let stored = ds.spin_orbit[&(key.l as u32)].level_errors.data()[slot];
            let e4 = branch_level_error(&ctx, &set, channel).unwrap();
            let e8 = branch_level_error(&ctx, &coarse, &coarse.channels[key]).unwrap();
            eprintln!(
                "{element} l={} kappa {:?}: stored {stored:+.6e}; stride 4 {:+.3e} off, stride 8 {:+.3e} off",
                key.l, key.kappa, e4 - stored, e8 - stored
            );
            assert!(
                (e8 - e4).abs() > 0.5 * (e4 - stored).abs(),
                "{element} l={}",
                key.l
            );
        }
    }
}

fn worst_branch_phase(
    ctx: &mandacaru_core::Context,
    ds: &mandacaru_core::paw::record::PawDataset,
    mutate: impl Fn(&mut mandacaru_core::paw::diagnose::view::Smooth),
) -> f64 {
    let atom = reference_atom(ctx, ds).unwrap();
    let mut set = branches(ctx, ds).unwrap();
    mutate(&mut set);
    let ae = AllElectron::new(&atom, ds.atomic_number as u8, &set.frozen, &set.semicore);
    scattering_errors(ctx, &set, &ae, &mut AeLogs::new())
        .unwrap()
        .values()
        .flat_map(|v| v.iter().copied())
        .fold(0., nan_max)
}

#[test]
fn the_branches_scatter_like_the_dirac_atom_and_a_scaled_coupling_does_not() {
    let ctx = context();
    let ds = read(&ctx, &dataset_path("lda-fr", "Na"));
    let right = worst_branch_phase(&ctx, &ds, |_| {});
    let wrong = worst_branch_phase(&ctx, &ds, |set| {
        for channel in set.channels.values_mut() {
            channel.coupling.iter_mut().for_each(|d| *d *= 1.3);
        }
    });
    eprintln!("Na branches: shipped {right:.3e} rad, coupling x1.3 {wrong:.3e} rad");
    assert!(
        right < 0.05,
        "the shipped branches scatter like their Dirac waves: {right}"
    );
    assert!(
        (wrong - right) / PHASE_AGREEMENT > 100.,
        "the scaled coupling must bite"
    );
}

#[test]
#[ignore = "a Dirac lead atom takes minutes; run with --ignored"]
fn the_two_dirac_equations_of_lead_scatter_differently_and_each_branch_fits_its_own() {
    let ctx = context();
    let ds = read(&ctx, &dataset_path("lda-fr", "Pb"));
    let own = worst_branch_phase(&ctx, &ds, |_| {});
    let swapped = worst_branch_phase(&ctx, &ds, |set| {
        let keys = set.channels.keys().copied().collect::<Vec<_>>();
        for key in keys {
            let mut channel = set.channels.remove(&key).unwrap();
            let kappa = if key.kappa == Some(key.l as i32) {
                -(key.l as i32 + 1)
            } else {
                key.l as i32
            };
            channel.kappa = Some(kappa);
            set.channels.insert(
                Key {
                    l: key.l,
                    kappa: Some(kappa),
                },
                channel,
            );
        }
    });
    eprintln!("Pb branches: own kappa {own:.3e} rad, swapped kappa {swapped:.3e} rad");
    assert!(
        (swapped - own) / PHASE_AGREEMENT > 100.,
        "the swapped kappa must bite"
    );
}
