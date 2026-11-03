//! The runs the real-crystal stress tests share: the SCF and the analytic stress of a frozen
//! case, the entry-by-entry comparison with the frozen oracle, the equation of state and the
//! controls that remove one term of the stress. The stress itself is a parameter of the runs
//! ([`StressOf`]): [`stress_at`] is the scalar crystal's, the spinor tests pass their own.
#![allow(dead_code)]

use super::{
    assert_pins,
    gates::{bites, entries, trace_gate, Entry, Tensor},
    load, manifest, oracle, summary, to_sigma_ev, TO_EV_PER_A3,
};
use crate::periodic_gradient_crystals_support::{inputs, Data};
use mandacaru_core::periodic::{
    gradient::State,
    stress_terms::{crystal::crystal_stress, Stress},
    Options,
};
use serde_json::Value;

/// The tolerances of the reference state of the analytic stress (the oracle's own SCFs are solved
/// at 1e-11 Ha / 1e-9 in the density).
pub const SCF_TOLERANCE: f64 = 1e-13;
pub const SCF_DENSITY_TOLERANCE: f64 = 1e-11;

/// The analytic stress of the converged `state` of `data`, the frozen state re-diagonalized with
/// `options` (the smearing the occupations are taken with).
pub fn stress_at(data: &Data, state: &State, options: &Options) -> Stress {
    let cores = data.cores();
    crystal_stress(
        &data.crystal.context,
        inputs(data, &cores),
        data.layout(),
        data.operators(),
        &data.crystal,
        options,
        state,
        None,
    )
    .unwrap()
}

/// The analytic stress of the converged `state` of `data` (re-diagonalized with `options`).
pub type StressOf<'a> = &'a dyn Fn(&Data, &State, &Options) -> Stress;

/// The converged SCF of `data` at the given tolerances: its free energy, its bands, its stress.
/// A state that does not reach them is solved again at ten times the tolerances, down to the
/// oracle's own (1e-11 Ha, 1e-9 in the density): the tolerance that was met is printed.
pub fn solve(data: &mut Data, tolerance: f64, density_tolerance: f64) -> (f64, State, Stress) {
    solve_with(data, tolerance, density_tolerance, &stress_at)
}

/// [`solve`] with the stress of `stress_of`.
pub fn solve_with(
    data: &mut Data,
    tolerance: f64,
    density_tolerance: f64,
    stress_of: StressOf,
) -> (f64, State, Stress) {
    let (mut tolerance, mut density_tolerance) = (tolerance, density_tolerance);
    let result = loop {
        data.options.tolerance = tolerance;
        data.options.density_tolerance = density_tolerance;
        let result = data.crystal.solve(&data.options).unwrap();
        if result.converged {
            break result;
        }
        println!("  SCF not converged at {tolerance:.0e} / {density_tolerance:.0e}");
        assert!(
            tolerance < 1e-11,
            "the SCF does not converge to the oracle's tolerances"
        );
        tolerance *= 10.;
        density_tolerance *= 10.;
    };
    println!(
        "  SCF converged at {tolerance:.0e} / {density_tolerance:.0e} in {} iterations",
        result.history.len()
    );
    let state = data.state(&result);
    let stress = stress_of(data, &state, &data.options);
    (result.free_energy, state, stress)
}

/// Prints every entry and returns the worst miss in units of its gate.
pub fn report(case: &str, got: &Tensor, want: &[[Entry; 3]; 3], volume: f64) -> f64 {
    let mut worst = 0f64;
    for a in 0..3 {
        for b in 0..3 {
            let w = &want[a][b];
            let miss = (got[a][b] - w.oracle).abs();
            println!(
                "{case} g[{a}][{b}] {:+.9e} oracle {:+.9e} miss {miss:.2e} Ha = {:.2e} eV/A^3, floor {:.1e} / {:.1e}, gate {:.2e} ({:.2} x)",
                got[a][b],
                w.oracle,
                to_sigma_ev(miss, volume),
                w.floor.abs(),
                w.extrapolated.abs(),
                w.gate,
                miss / w.gate
            );
            // `f64::max` ignores a NaN: a NaN entry must fail the gate, not leave it at zero.
            worst = if miss.is_finite() {
                worst.max(miss / w.gate)
            } else {
                f64::INFINITY
            };
        }
    }
    worst
}

/// One case: the SCF, the analytic stress, the comparison with the frozen oracle entry by entry,
/// and (`second`) with the second step family `<case>_b` of the same snapshot, whose own floors
/// give its own gates (a floor measured with one family can be an underestimate).
pub fn check(case: &'static str, second: bool) -> (Data, Stress, [[Entry; 3]; 3]) {
    check_with(case, second, &stress_at)
}

/// [`check`] with the stress of `stress_of`.
pub fn check_with(
    case: &'static str,
    second: bool,
    stress_of: StressOf,
) -> (Data, Stress, [[Entry; 3]; 3]) {
    let start = std::time::Instant::now();
    let mut data = load(case);
    assert_pins(&data);
    let volume = data.manifest["volume_bohr3"].as_f64().unwrap();
    let (free_energy, _, stress) =
        solve_with(&mut data, SCF_TOLERANCE, SCF_DENSITY_TOLERANCE, stress_of);
    let frozen = data.manifest["free_energy_ha"].as_f64().unwrap();
    assert!(
        (free_energy - frozen).abs() < 1e-9,
        "the SCF is not the oracle's: {:e}",
        free_energy - frozen
    );
    assert!(
        (stress.volume - volume).abs() < 1e-6 * volume,
        "the cell volume"
    );
    println!(
        "{case}: SCF and stress {:.1} s",
        start.elapsed().as_secs_f64()
    );
    summary(case, &stress);
    for (name, term) in stress.breakdown.terms() {
        let peak = term.iter().flatten().fold(0f64, |m, x| m.max(x.abs()));
        println!("  term {name}: peak {peak:.3e} Ha");
    }
    let want = entries(&data.manifest, volume);
    let worst = report(case, &stress.gradient, &want, volume);
    assert!(
        worst < 1.,
        "{case}: the worst entry misses its gate by {worst:.2} x"
    );
    if second {
        let other = manifest(&format!("{case}_b"));
        assert_eq!(
            other["snapshot"].as_str(),
            Some(case),
            "the second family must share the snapshot of {case}"
        );
        assert_ne!(
            other["steps"], data.manifest["steps"],
            "a second family needs other steps"
        );
        let want_b = entries(&other, volume);
        let worst_b = report(
            &format!("{case} (steps {})", other["steps"]),
            &stress.gradient,
            &want_b,
            volume,
        );
        assert!(
            worst_b < 1.,
            "{case}: the worst entry misses the gate of the second family by {worst_b:.2} x"
        );
    }
    (data, stress, want)
}

/// The equation of state: the trace of the nine entries is the derivative of the free energy
/// along the isotropic series `F(e I)`. Returns `(trace, series, gate)`.
pub fn equation_of_state(manifest: &Value, stress: &Stress) -> (f64, f64, f64) {
    let iso = &manifest["isotropic"];
    let trace: f64 = (0..3).map(|a| stress.gradient[a][a]).sum();
    let gate = trace_gate(
        iso["floor_richardson"].as_f64().unwrap(),
        manifest["steps"][0].as_f64().unwrap(),
        stress.volume,
    );
    let oracle = iso["rr"].as_f64().unwrap();
    println!(
        "  equation of state: trace {trace:+.9e}, d F(e I)/de {oracle:+.9e}, miss {:.2e} (gate {gate:.2e}); pressure {:+.6e} eV/A^3 from the series, {:+.6e} analytic",
        (trace - oracle).abs(),
        -oracle / (3. * stress.volume) * TO_EV_PER_A3,
        stress.pressure() * TO_EV_PER_A3
    );
    assert!((trace - oracle).abs() < gate, "the equation of state");
    (trace, oracle, gate)
}

/// A stress without one of its terms must miss the oracle by more than 100 gates; returns the
/// terms whose removal does not (their bite is printed either way).
pub fn weak_terms(stress: &Stress, want: &[[Entry; 3]; 3]) -> Vec<&'static str> {
    let mut weak = Vec::new();
    for (name, term) in stress.breakdown.terms() {
        // The spin-orbit and exchange terms of a crystal that has none are zero by construction, not weak.
        if matches!(name, "spin_orbit" | "exchange") && term.iter().flatten().all(|x| *x == 0.) {
            continue;
        }
        let without: Tensor =
            std::array::from_fn(|a| std::array::from_fn(|b| stress.gradient[a][b] - term[a][b]));
        let bite = bites(&without, want);
        println!("  control without {name}: bites {bite:.2e} gates");
        if bite <= 100. {
            weak.push(name);
        }
    }
    weak
}

/// The stress of a cubic cell is proportional to the identity: the off-diagonal entries of sigma
/// vanish (gated, relative to |sigma|) and the diagonal entries agree to the egg-box floor of the
/// grid terms, which is REPORTED next to the oracle's own diagonal spread (the frozen diagonal
/// entries differ by the same amount; their agreement with the analytic ones is the gate of
/// [`check`]).
pub fn isotropy(manifest: &Value, stress: &Stress) {
    let sigma = stress.sigma;
    let scale = (sigma[0][0] + sigma[1][1] + sigma[2][2]).abs() / 3.;
    let mut off = 0f64;
    let mut spread = 0f64;
    for a in 0..3 {
        spread = spread.max((sigma[a][a] - sigma[0][0]).abs() / scale);
        for b in 0..3 {
            if a != b {
                off = off.max(sigma[a][b].abs() / scale);
            }
        }
    }
    let frozen = oracle(manifest);
    let want = (0..3)
        .map(|a| (frozen[a][a] - frozen[0][0]).abs())
        .fold(0., f64::max)
        / (frozen[0][0].abs());
    println!(
        "  isotropy: off-diagonal {off:.2e} of |sigma| ({:.2e} eV/A^3 of {:.3e}); diagonal spread {spread:.2e} (oracle's own {want:.2e})",
        to_sigma_ev(off * scale * stress.volume, stress.volume),
        to_sigma_ev(scale * stress.volume, stress.volume)
    );
    assert!(
        off < 1e-9,
        "sigma is not proportional to the identity: off-diagonal {off:.2e}"
    );
}
