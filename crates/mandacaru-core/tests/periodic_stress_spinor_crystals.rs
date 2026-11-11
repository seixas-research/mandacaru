//! AS-8: the analytic stress of a spinor (spin-orbit) LDA/PBE crystal on a REAL converged SCF state
//! against the Richardson-reconverged free energy of the same discretized spinor functional
//! (`freeze_periodic_strain.py`: each strained cell a fresh Python setup solved by the Rust SCF, all
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
//! * `au_fcc` (a metal, two time-reversal-invariant points, Mermin occupations): the plan's gate is
//!   NOT met and is not claimed. The oracle of this small cell drifts linearly with the strain step,
//!   so the stress is asserted only within the oracle's own spread, see [`within_the_spread`].
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
    gates::{bites, entries, Entry},
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

/// The metal's analytic stress against its oracle, claimed only WITHIN THE ORACLE'S OWN SPREAD (not
/// the plan's gate): the central differences of this small cell drift linearly with the strain step
/// (the piecewise-linear shape tables are crossed by more nodes the larger the strain, so the
/// extrapolation in `h^2` leaves an `O(h)` term), and its step families `au_fcc` (5e-4..2e-3) and
/// `au_fcc_b` (1e-3..4e-3) disagree by up to 5e-6 on an entry against a plan shear gate of 3e-8.
/// Every entry satisfies `|analytic - rr_a| < 3 |rr_a - rr_b|`, and the miss shrinks with the step:
/// the third family `au_fcc_c` (1.25e-4..5e-4) is asserted closer than `au_fcc` in the worst entry,
/// and `au_fcc` closer than `au_fcc_b`.
fn within_the_spread(case: &str, a: &Value, stress: &Stress) {
    let b = manifest(&format!("{case}_b"));
    let c = manifest(&format!("{case}_c"));
    let worst = |m: &Value| -> f64 {
        (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| (stress.gradient[i][j] - entry(m, i, j).0).abs())
            .fold(0., f64::max)
    };
    for i in 0..3 {
        for j in 0..3 {
            let (ra, rb) = (entry(a, i, j).0, entry(&b, i, j).0);
            let miss = (stress.gradient[i][j] - ra).abs();
            let spread = (ra - rb).abs();
            println!("  spread [{i}][{j}]: miss {miss:.2e} Ha, families differ by {spread:.2e}");
            assert!(miss < 3. * spread, "[{i}][{j}] outside the oracle's spread");
        }
    }
    let (wa, wb, wc) = (worst(a), worst(&b), worst(&c));
    println!(
        "  worst miss against the families: {wb:.2e} (1e-3..4e-3), {wa:.2e} (5e-4..2e-3), {wc:.2e} (1.25e-4..5e-4)"
    );
    assert!(wc < wa && wa < wb, "the miss must shrink with the step");
    // The equation of state, the same way: the trace against the isotropic series of each family.
    let trace: f64 = (0..3).map(|i| stress.gradient[i][i]).sum();
    let series = |m: &Value| m["isotropic"]["rr"].as_f64().unwrap();
    let (ta, tb, tc) = (
        (trace - series(a)).abs(),
        (trace - series(&b)).abs(),
        (trace - series(&c)).abs(),
    );
    println!(
        "  equation of state: trace {trace:+.9e}; miss {tb:.2e} / {ta:.2e} / {tc:.2e} against the three families"
    );
    assert!(tc < ta && ta < tb && ta < 3. * (series(a) - series(&b)).abs());
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
        within_the_spread(case, &data.manifest, &stress);
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
#[ignore = "slow: 143 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_spinor_crystals -- --ignored gold_metal_agrees_with_the_reconverged_free_energy_within_the_oracles_spread"]
fn gold_metal_agrees_with_the_reconverged_free_energy_within_the_oracles_spread() {
    // The spin-orbit term (2.1e-4 Ha) is resolved at 1.3e3 plan gates in the entry it is largest in.
    heavy("au_fcc", true, 100.);
}
