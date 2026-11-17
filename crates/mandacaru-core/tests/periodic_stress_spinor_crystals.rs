//! AS-8: the analytic stress of a spinor (spin-orbit) LDA/PBE crystal on a REAL converged SCF state
//! against the Richardson-reconverged free energy of the same discretized spinor functional
//! (`periodic_freeze_snapshot.rs`: each strained cell a fresh Rust setup solved by the Rust SCF, all
//! nine entries unsymmetrized, the pins of `ANALYTIC_STRESS_PLAN.md` section 2 held).
//!
//! * `si_ldafr_tr3`: the displaced `lda-fr` Si on a 3x1x1 mesh that time reversal halves to two
//!   points (weights 2/3 and 1/3): the spin-orbit share is small here and is reported;
//! * `si_pbefr_tr3`: the same with PBE rotated to the local spin eigenframe (the spectral-gradient
//!   term acts on the rotated spin channels);
//! * `auh_fr` (a molecule in a box): a heavy spinor whose spin-orbit term is large, at the plan's
//!   gates, with the controls of the spinor frame: no spin-orbit term, its sign, `W` with the
//!   reference kept, the kinetic matrix on the alpha block only. "A scalar `W` instead of the spinor
//!   `W`" has no distinct meaning in the doubled frame: `W` meets only `I2 x dS`, i.e. its spin
//!   trace, so a spin-averaged `W` gives the identical stress (the engine's own algebra, as for the
//!   force);
//! * `au_fcc` (a metal, two time-reversal-invariant points, Mermin occupations): the plan's gate
//!   is NOT met and is not claimed. The finite-difference oracle of this cell has a resolution of
//!   its own, measured by three step families and asserted against, see [`resolved`]; its cause is
//!   the shape tables, see [`within_the_resolution`].
//!
//! The 2x2x2 whole-mesh cell (`si_ldafr`, eight points) is `#[ignore]`d: it takes minutes; run it
//! with `cargo test --release -p mandacaru-core --test periodic_stress_spinor_crystals -- --ignored
//! --test-threads=1`. Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;
mod periodic_stress_crystals_support;
mod periodic_stress_spinor_support;

use mandacaru_core::periodic::{
    gradient::Term,
    stress_terms::{contract, Stress, Tensor},
};
use periodic_stress_crystals_support::{
    assert_pins, entry,
    gates::{bites, entries, trace_gate, Entry},
    load, manifest,
    run::{check_with, report, solve_with, weak_terms, SCF_DENSITY_TOLERANCE, SCF_TOLERANCE},
    summary,
};
use periodic_stress_spinor_support::{
    altered::{Altered, Wrong},
    peak, spinor_of, strain_of, stress_of_data,
};
use serde_json::Value;

fn print_share(stress: &Stress, want: &[[Entry; 3]; 3]) {
    let so = &stress.breakdown.spin_orbit;
    let gate = want
        .iter()
        .flatten()
        .map(|e| e.gate)
        .fold(f64::INFINITY, f64::min);
    println!(
        "  spin-orbit term: peak {:.3e} Ha = {:.2} of the tightest gate ({gate:.2e} Ha)",
        peak(so),
        peak(so) / gate
    );
}

#[test]
#[ignore = "slow: 146 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_spinor_crystals -- --ignored silicon_with_spin_orbit_matches_the_reconverged_free_energy"]
fn silicon_with_spin_orbit_matches_the_reconverged_free_energy() {
    let (data, stress, want) = check_with("si_ldafr_tr3", true, &stress_of_data);
    assert!(data.crystal.spinor && data.crystal.time_reversal);
    assert_eq!(
        data.crystal.points.len(),
        2,
        "time reversal halves the mesh"
    );
    print_share(&stress, &want);
    let weak = weak_terms(&stress, &want, &data.crystal.functional);
    println!("  terms below 100 gates when removed: {weak:?}");
}

#[test]
#[ignore = "slow: 146 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_spinor_crystals -- --ignored the_spinor_gradient_correction_matches_the_reconverged_free_energy"]
fn the_spinor_gradient_correction_matches_the_reconverged_free_energy() {
    let (data, stress, want) = check_with("si_pbefr_tr3", true, &stress_of_data);
    assert_eq!(data.crystal.functional, "pbe");
    print_share(&stress, &want);
    // The spectral-gradient term is live on the rotated spin channels and bites.
    let gradient = &stress.breakdown.xc_gradient;
    assert!(peak(gradient) > 1e-3, "the GGA term: {:e}", peak(gradient));
    let without =
        std::array::from_fn(|a| std::array::from_fn(|b| stress.gradient[a][b] - gradient[a][b]));
    let bite = bites(&without, &want);
    println!("  control without the gradient term: bites {bite:.2e} gates");
    assert!(bite > 100.);
}

#[test]
#[ignore = "eight k-points: minutes; cargo test --release -p mandacaru-core --test periodic_stress_spinor_crystals -- --ignored --test-threads=1"]
fn silicon_on_the_whole_mesh_matches_the_reconverged_free_energy() {
    let (_, stress, want) = check_with("si_ldafr", false, &stress_of_data);
    print_share(&stress, &want);
}

/// The finite-difference oracle of the metal as it resolves the derivative: the Richardson value of
/// the finest step family (`<case>_c`, steps 1.25e-4..5e-4) for each of the nine entries and for the
/// series of the equation of state, and the RESOLUTION of each, the largest difference between the
/// values of the three families (`_c`, the case's own with 5e-4..2e-3, and `_b` with 1e-3..4e-3,
/// which differ in the strain steps alone). It is what the oracle cannot tell apart, measured here
/// and not assumed.
struct Resolved {
    finest: [[f64; 3]; 3],
    resolution: [[f64; 3]; 3],
    series_finest: f64,
    series_resolution: f64,
}

fn spread(values: [f64; 3]) -> f64 {
    values
        .iter()
        .flat_map(|a| values.iter().map(move |b| (a - b).abs()))
        .fold(0., f64::max)
}

fn resolved(case: &str) -> Resolved {
    let families = [
        manifest(&format!("{case}_c")),
        manifest(case),
        manifest(&format!("{case}_b")),
    ];
    let at =
        |a: usize, b: usize| -> [f64; 3] { std::array::from_fn(|f| entry(&families[f], a, b).0) };
    let series = |m: &Value| m["isotropic"]["rr"].as_f64().unwrap();
    let rows: [[[f64; 3]; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| at(a, b)));
    let series_of: [f64; 3] = std::array::from_fn(|f| series(&families[f]));
    Resolved {
        finest: std::array::from_fn(|a| std::array::from_fn(|b| rows[a][b][0])),
        resolution: std::array::from_fn(|a| std::array::from_fn(|b| spread(rows[a][b]))),
        series_finest: series_of[0],
        series_resolution: spread(series_of),
    }
}

/// The entries of `got` (a derivative tensor, Hartree) that miss the finest family by more than the
/// larger of the plan's gate of the entry and the oracle's resolution of it, with the miss in units of
/// that limit. The plan's gate is held where the oracle can resolve it; where it cannot, the
/// resolution is all the check can ask, and it is printed against the gate.
fn outside(
    got: &Tensor<f64>,
    oracle: &Resolved,
    plan_gates: &[[Entry; 3]; 3],
) -> Vec<(usize, usize, f64)> {
    let mut out = vec![];
    for a in 0..3 {
        for b in 0..3 {
            let limit = oracle.resolution[a][b].max(plan_gates[a][b].gate);
            let miss = (got[a][b] - oracle.finest[a][b]).abs();
            // A NaN entry is outside.
            if !(miss <= limit) {
                out.push((a, b, miss / limit));
            }
        }
    }
    out
}

/// The metal's analytic stress against its oracle, within the oracle's measured resolution. The
/// analytic stress is the exact derivative of the discretized functional; the central differences of
/// fresh SCFs are not, and their error is not the quadratic truncation a Richardson series removes.
/// It comes from the piecewise-linear shape tables (a node every 0.01 Bohr^-1): a strain moves every
/// wave number `|Q|` of the dense list by `eps |Q|` (half a node for `|Q| = 10` at `eps = 5e-4`), so
/// thousands of vectors cross a node between `-eps` and `+eps`, each crossing a kink of the energy,
/// and the kinks add to a deterministic, irregular function of the step that no step cancels.
/// Measured on this cell (logged in `dev_plan/DECISIONS.md`, hand-off 12, item 1(d)):
///
/// * the same strain solved in two families gives the same free energy to the last digit (the SCF is
///   deterministic), and tightening the SCF from 1e-11 Ha / 1e-9 to 1e-13 / 1e-11 moves the central
///   differences by 1.4e-9 and 2.3e-10 against a scatter of 4e-6;
/// * the central differences of the `[0][0]` entry at 1.25e-4, 2.5e-4 and 5e-4 are -0.444162262,
///   -0.444158302 and -0.444156752 Ha: increments of 4.0e-6 and 1.5e-6, where an even series in the
///   step has increments in the ratio 1:4;
/// * with the table step 8 times finer the same differences scatter by 3e-7 instead of 4e-6 and the
///   entry moves to -0.4441456 Ha: the table is the cause, in proportion to its step.
///
/// So no step family converges the oracle onto the analytic value, and the order of the families'
/// misses is not a property of the stress (the middle family misses the most). What is asserted is
/// what the oracle can resolve: every entry within the larger of the plan's gate and the spread of
/// the three families, measured from the finest, and the controls, which are far outside it, must
/// fail the same check.
fn within_the_resolution(case: &str, volume: f64, stress: &Stress) {
    let oracle = resolved(case);
    let plan_gates = entries(&manifest(&format!("{case}_c")), volume);
    for a in 0..3 {
        for b in 0..3 {
            let miss = (stress.gradient[a][b] - oracle.finest[a][b]).abs();
            let (resolution, gate) = (oracle.resolution[a][b], plan_gates[a][b].gate);
            println!(
                "  resolution [{a}][{b}]: miss {miss:.2e} Ha, families differ by {resolution:.2e}, plan gate {gate:.2e} ({})",
                if resolution <= gate { "resolved" } else { "not resolved: the resolution is the limit" }
            );
        }
    }
    let misses = outside(&stress.gradient, &oracle, &plan_gates);
    assert!(
        misses.is_empty(),
        "outside the oracle's resolution: {misses:?}"
    );
    // The controls, each outside the same check.
    let breakdown = &stress.breakdown;
    let shifted = |t: &Tensor<f64>, k: f64| -> Tensor<f64> {
        std::array::from_fn(|a| std::array::from_fn(|b| stress.gradient[a][b] - k * t[a][b]))
    };
    let mut lifted = stress.gradient;
    lifted[0][1] += 3. * oracle.resolution[0][1].max(plan_gates[0][1].gate);
    for (name, wrong) in [
        ("no spin-orbit term", shifted(&breakdown.spin_orbit, 1.)),
        (
            "spin-orbit term of the other sign",
            shifted(&breakdown.spin_orbit, 2.),
        ),
        ("kinetic term doubled", shifted(&breakdown.kinetic, -1.)),
        ("no Hartree term", shifted(&breakdown.hartree, 1.)),
        ("[0][1] lifted by three limits", lifted),
    ] {
        let missed = outside(&wrong, &oracle, &plan_gates);
        println!(
            "  control ({name}): {} entries outside, worst {:.2e} limits",
            missed.len(),
            missed.iter().map(|m| m.2).fold(0., f64::max)
        );
        assert!(
            !missed.is_empty(),
            "control '{name}' passes the resolution check"
        );
    }
    // The equation of state, the same way: the trace against the finest family's isotropic series.
    let trace: f64 = (0..3).map(|i| stress.gradient[i][i]).sum();
    let family = manifest(&format!("{case}_c"));
    let series = &family["isotropic"];
    let limit = oracle.series_resolution.max(trace_gate(
        series["floor_richardson"].as_f64().unwrap(),
        family["steps"][0].as_f64().unwrap(),
        volume,
    ));
    let miss = (trace - oracle.series_finest).abs();
    println!(
        "  equation of state: trace {trace:+.9e}; miss {miss:.2e} Ha, families differ by {:.2e}, limit {limit:.2e}",
        oracle.series_resolution
    );
    assert!(
        miss <= limit,
        "the trace is outside the oracle's resolution"
    );
    assert!(
        (trace + 2. * limit - oracle.series_finest).abs() > limit,
        "the equation-of-state control passes"
    );
}

/// The controls of the spinor frame on a heavy state, each compared with the oracle of the case: no
/// spin-orbit term, its sign, the kinetic matrix on the alpha block only, and `W` with the reference
/// kept. The spin-orbit controls are REPORTED and asserted only above `resolved` gates (the term is
/// 1.7e-4 Ha for AuH, 2.1e-4 for Au): resolved at 49 gates for AuH (the plan's gate of a 216 A^3 box
/// is large) and at 1.3e3 for Au, and at 1e8 gates by `periodic_stress_spinor_orbit`; the others are
/// far above 100 everywhere.
fn heavy(case: &'static str, metal: bool, resolved: f64) {
    let mut data = load(case);
    assert_pins(&data);
    let volume = data.manifest["volume_bohr3"].as_f64().unwrap();
    let (_, state, stress) = solve_with(
        &mut data,
        SCF_TOLERANCE,
        SCF_DENSITY_TOLERANCE,
        &stress_of_data,
    );
    let want = entries(&data.manifest, volume);
    summary(case, &stress);
    let worst = report(case, &stress.gradient, &want, volume);
    if metal {
        println!(
            "  plan gate: the worst entry misses it by {worst:.2} x (reported: oracle-limited)"
        );
        let finest = report(
            &format!("{case} (finest family, steps 1.25e-4..5e-4)"),
            &stress.gradient,
            &entries(&manifest(&format!("{case}_c")), volume),
            volume,
        );
        println!("  plan gate with the finest family: the worst entry misses it by {finest:.2} x");
        within_the_resolution(case, volume, &stress);
        let partial = state
            .bands
            .iter()
            .flatten()
            .flat_map(|b| &b.occupations)
            .filter(|f| **f > 1e-3 && **f < 1. - 1e-3)
            .count();
        println!("  bands with a partial occupation: {partial}");
        assert!(partial > 0, "a metal needs partial occupations");
    } else {
        assert!(worst < 1., "{case} misses its gate by {worst:.2} x");
        let other = manifest(&format!("{case}_b"));
        let worst_b = report(
            &format!("{case} (second family)"),
            &stress.gradient,
            &entries(&other, volume),
            volume,
        );
        assert!(
            worst_b < 1.,
            "{case} misses the gate of the second family by {worst_b:.2} x"
        );
    }
    print_share(&stress, &want);

    let so = &stress.breakdown.spin_orbit;
    let minus = |t: &Tensor<f64>, k: f64| -> Tensor<f64> {
        std::array::from_fn(|a| std::array::from_fn(|b| stress.gradient[a][b] - k * t[a][b]))
    };
    for (name, wrong) in [
        ("no spin-orbit term", minus(so, 1.)),
        ("spin-orbit term of the other sign", minus(so, 2.)),
    ] {
        let bite = bites(&wrong, &want);
        println!("control {name}: bites {bite:.2e} gates");
        assert!(bite > resolved, "{name}: {bite:.2e}");
    }

    let sp = spinor_of(data);
    let cores = sp.data.cores();
    let c = strain_of(&sp, &cores, &state, &sp.data.options);
    // The kinetic matrix on the alpha block only.
    let wrong = Altered {
        inner: &c.strain,
        wrong: Wrong::AlphaOnly(Term::Kinetic),
    };
    let bite = bites(&contract(&wrong, &c.frozen).unwrap().gradient(), &want);
    println!("control kinetic on the alpha block only: bites {bite:.2e} gates");
    assert!(bite > 100.);
    // `W` with the reference kept (the eigenvalues as the solver returns them).
    let mut frozen = c.frozen;
    let reference = sp.data.crystal.reference;
    for bands in frozen.state.bands.iter_mut().flatten() {
        bands.eigenvalues.iter_mut().for_each(|e| *e += reference);
    }
    let bite = bites(&contract(&c.strain, &frozen).unwrap().gradient(), &want);
    println!("control W with the reference kept: bites {bite:.2e} gates");
    assert!(bite > 100.);
}

#[test]
#[ignore = "slow: 42 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_spinor_crystals -- --ignored gold_hydride_matches_the_reconverged_free_energy_and_the_wrong_frames_bite"]
fn gold_hydride_matches_the_reconverged_free_energy_and_the_wrong_frames_bite() {
    // The spin-orbit term of AuH is 1.7e-4 Ha, 2.8e3 times the tightest (shear) gate but 49 gates in the entry it
    // is largest in relative to ITS gate: the plan's gate of a 216 A^3 box is large. Reported, asserted above 10
    // (the term itself is gated at 1e8 gates by `periodic_stress_spinor_orbit`).
    heavy("auh_fr", false, 10.);
}

#[test]
#[ignore = "slow: 30 s (measured 2026-11-16); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_spinor_crystals -- --ignored gold_metal_agrees_with_the_reconverged_free_energy_within_the_oracles_resolution"]
fn gold_metal_agrees_with_the_reconverged_free_energy_within_the_oracles_resolution() {
    // The spin-orbit term (2.1e-4 Ha) is resolved at 1.3e3 plan gates in the entry it is largest in.
    heavy("au_fcc", true, 100.);
}
