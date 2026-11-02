//! The smooth-side diagnostics against the frozen Python ones on shipped files: the logarithmic
//! derivative, the Richardson-extrapolated spectrum, the reconstruction of the all-electron
//! function and the intruding-1s miss (`test/migration/reference/paw_generate/lib_*.json`,
//! taken on the shipped files, digest-checked). A missing frozen input or a changed library
//! file FAILS; nothing skips.
//!
//! Gates: `max(1e-12, 100 x the 1-ulp floor the freeze recorded)`. `paw_spectrum` has a floor of
//! exactly 0 (the Python solver reproduces itself bitwise), so 100 x floor is unusable and its
//! gate is 1e-12 Hartree.

mod paw_assemble_support;
mod paw_diagnose_support;

use mandacaru_core::paw::diagnose::{
    intruding::intruding_s_miss, reconstruct, scattering, spectrum, view::Smooth,
};
use paw_diagnose_support::*;

const GATE: f64 = 1e-12;
/// The levels above zero are the discretized continuum of a Dirichlet box: they follow the
/// wall and the interpolated potential more closely than a bound state does. Measured worst
/// deviation 1.9e-12 Ha (lda-sr Fe, l = 2, the state at +0.119 Ha); every bound state is within
/// 1.3e-13 Ha. The gate on them is 1e-11 Ha.
const BOX_STATE_GATE: f64 = 1e-11;

fn gate(case: &LibraryCase, name: &str) -> f64 {
    GATE.max(case.floor_gate(name))
}

/// Worst `|ours - frozen| / max(|frozen|, 1e-300)` over the logarithmic derivatives of a file.
fn log_derivative_deviation(case: &LibraryCase) -> f64 {
    let ctx = context();
    let ds = read(&ctx, &case.file());
    let smooth = smooth_of(&ctx, &ds);
    let mut worst = 0f64;
    let frozen = case.function("log_derivative_paw");
    for (l, entry) in frozen.as_object().unwrap() {
        let l: usize = l.parse().unwrap();
        let channel = smooth.channel(l).unwrap();
        let energies = numbers(&entry["energies"]);
        let values = numbers(&entry["values"]);
        assert_eq!(
            smooth.match_radius(channel).to_bits() == 0,
            false,
            "{} l={l}",
            case.element
        );
        let radius = channel.projector_radius(smooth.r_cut_local);
        assert!((radius - entry["radius"].as_f64().unwrap()).abs() < 1e-12);
        for (&e, &want) in energies.iter().zip(&values) {
            let got = scattering::log_derivative(&ctx, &smooth, l, Some(channel), e, None).unwrap();
            worst = nan_max(worst, (got - want).abs() / want.abs().max(1e-300));
        }
        let far = entry["radius_explicit"].as_f64().unwrap();
        for (&e, &want) in energies
            .iter()
            .zip(numbers(&entry["values_explicit"]).iter())
        {
            let got =
                scattering::log_derivative(&ctx, &smooth, l, Some(channel), e, Some(far)).unwrap();
            worst = nan_max(worst, (got - want).abs() / want.abs().max(1e-300));
        }
    }
    worst
}

#[test]
fn the_logarithmic_derivative_equals_the_freeze_on_every_recorded_file() {
    let mut worst = (0f64, String::new());
    for (set, element) in library_cases() {
        let case = LibraryCase::open(&set, &element);
        let dev = log_derivative_deviation(&case);
        let ratio = dev / gate(&case, "log_derivative_paw");
        eprintln!("log_derivative_paw {set} {element}: {dev:.2e} ({ratio:.3} of the gate)");
        assert!(ratio <= 1., "{set} {element}: {dev:.3e} over the gate");
        if !(ratio <= worst.0) {
            worst = (ratio, format!("{set} {element}"));
        }
    }
    eprintln!("worst log-derivative ratio {:.3} ({})", worst.0, worst.1);
}

/// The Rust floor of the spectrum: the same spectrum of the same dataset with the screened local
/// potential and every projector moved by one relative ulp (alternating sign).
fn spectrum_floor(ctx: &mandacaru_core::Context, smooth: &mut Smooth, states: usize) -> f64 {
    let before = smooth
        .channels
        .values()
        .map(|c| spectrum::spectrum(ctx, smooth, c, states).unwrap())
        .collect::<Vec<_>>();
    let nudge = |values: &mut [f64]| {
        for (i, v) in values.iter_mut().enumerate() {
            *v *= 1.
                + if i % 2 == 0 {
                    f64::EPSILON
                } else {
                    -f64::EPSILON
                };
        }
    };
    nudge(smooth.v_local_screened.as_mut_slice());
    for c in smooth.channels.values_mut() {
        nudge(c.projectors.as_mut_slice());
    }
    let after = smooth
        .channels
        .values()
        .map(|c| spectrum::spectrum(ctx, smooth, c, states).unwrap())
        .collect::<Vec<_>>();
    before
        .iter()
        .flatten()
        .zip(after.iter().flatten())
        .fold(0., |m, (a, b)| nan_max(m, (a - b).abs()))
}

#[test]
fn the_spectrum_equals_the_freeze_on_every_recorded_file() {
    let mut worst = (0f64, String::new());
    let mut failures = Vec::new();
    for (set, element) in library_cases() {
        let case = LibraryCase::open(&set, &element);
        let ctx = context();
        let ds = read(&ctx, &case.file());
        let mut smooth = smooth_of(&ctx, &ds);
        let mut file_worst = (0f64, String::new());
        for (l, entry) in case.function("paw_spectrum").as_object().unwrap() {
            let channel = smooth.channel(l.parse().unwrap()).unwrap();
            assert_eq!(entry["stride"], 2);
            let states = entry["n_states"].as_u64().unwrap() as usize;
            let got = spectrum::spectrum(&ctx, &smooth, channel, states).unwrap();
            for (k, (g, w)) in got.iter().zip(numbers(&entry["values"])).enumerate() {
                let dev = (g - w).abs();
                let allowed = if w < 0. { GATE } else { BOX_STATE_GATE };
                if !(dev / allowed <= file_worst.0) {
                    file_worst = (
                        dev / allowed,
                        format!("l={l} state {k}: {g:+.15} vs {w:+.15} ({dev:.2e} Ha)"),
                    );
                }
            }
        }
        let floor = spectrum_floor(&ctx, &mut smooth, 3);
        eprintln!(
            "paw_spectrum {set} {element}: {:.3} of its gate ({}); Rust 1-ulp floor {floor:.2e} Ha",
            file_worst.0, file_worst.1
        );
        if !(file_worst.0 <= 1.) {
            failures.push(format!(
                "{set} {element}: {:.3} of its gate ({})",
                file_worst.0, file_worst.1
            ));
        }
        if !(file_worst.0 <= worst.0) {
            worst = (file_worst.0, format!("{set} {element}"));
        }
    }
    eprintln!(
        "worst spectrum deviation {:.3} of its gate at {} (gates {GATE:.0e} Ha bound, {BOX_STATE_GATE:.0e} Ha box states)",
        worst.0, worst.1
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_reconstruction_equals_the_freeze_on_every_recorded_file() {
    for (set, element) in library_cases() {
        let case = LibraryCase::open(&set, &element);
        let ctx = context();
        let ds = read(&ctx, &case.file());
        let smooth = smooth_of(&ctx, &ds);
        let grid: Vec<f64> = (0..449)
            .map(|i| 0.02 + (9.0 - 0.02) * i as f64 / 448.)
            .collect();
        let mut worst = 0f64;
        for (l, entry) in case.function("reconstruct_ae").as_object().unwrap() {
            let l: usize = l.parse().unwrap();
            let channel = smooth.channel(l).unwrap();
            let u: Vec<f64> = grid
                .iter()
                .map(|r| r.powi(l as i32 + 1) * (-0.8 * r).exp())
                .collect();
            let got = reconstruct::reconstruct_ae(&ctx, &smooth, channel, &grid, &u).unwrap();
            let want = case.data.array(entry["blob"].as_str().unwrap());
            assert_eq!(got.len(), want.len());
            let scale = entry["max"].as_f64().unwrap().abs();
            for (g, w) in got.iter().zip(&want) {
                worst = nan_max(worst, (g - w).abs() / scale);
            }
        }
        let allowed = gate(&case, "reconstruct_ae");
        eprintln!(
            "reconstruct_ae {set} {element}: {worst:.2e} ({:.3} of the gate)",
            worst / allowed
        );
        assert!(worst <= allowed, "{set} {element}: {worst:.3e}");
    }
}

#[test]
fn the_intruding_miss_equals_the_freeze_on_every_recorded_file() {
    for (set, element) in library_cases() {
        let case = LibraryCase::open(&set, &element);
        let ctx = context();
        let ds = read(&ctx, &case.file());
        let smooth = smooth_of(&ctx, &ds);
        let got = intruding_s_miss(&smooth).expect("every sampled file has an s channel");
        let want = case.function("intruding_s_miss").as_f64().unwrap();
        let dev = (got - want).abs() / want.abs();
        eprintln!("intruding_s_miss {set} {element}: {got:.9} vs {want:.9} ({dev:.1e})");
        assert!(dev <= 1e-12, "{set} {element}: {dev:.3e}");
    }
}
