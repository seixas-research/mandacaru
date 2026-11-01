//! Tests for Model Periodic Coulomb (MPC) finite-size correction (`periodic/finite_size/mpc.rs`).
//!
//! Validates:
//! - T3: The $N = 1$ Gaussian correction and the $v_M/2$ double-counting cancellation against `mpc_n1.json`.
//! - T3-note: Richardson extrapolation meets the 1% gate of $2\pi s^2 / \Omega$ while the raw sampled kernel fails.
//! - T5: Hole trace relation $\sum_q (\Gamma_{pqrq} - D_{pr} D_{qq}) = -D_{pr}$.
//! - Input validation on non-finite, degenerate, or negative inputs.

use mandacaru_core::periodic::finite_size::{
    mpc_from_hole_energies, verify_hole_trace,
};
use serde_json::Value;

fn load(name: &str) -> Value {
    let path = format!(
        "{}/../../test/migration/reference/finite_size/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

#[test]
fn mpc_from_hole_energies_matches_formula_and_freeze() {
    let frozen = load("mpc_n1");
    let cases = frozen["cases"].as_object().unwrap();

    for (name, case) in cases {
        let l = num(&case["L"]);
        let volume = l * l * l;
        let v_m = num(&case["vM"]);
        let hole_model = num(&case["old_hole_truncated"]);
        let hole_periodic = num(&case["old_hole_periodic"]);
        let n_electrons = 1.0;
        let n_cells = 1;

        let res = mpc_from_hole_energies(
            hole_model,
            hole_periodic,
            v_m,
            n_electrons,
            n_cells,
            volume,
        )
        .unwrap();

        let old_correction = num(&case["old_correction"]);
        let corrected_by_formula = num(&case["corrected_by_formula"]);

        assert!(
            (res.raw_correction - old_correction).abs() < 1e-12,
            "{name}: raw correction mismatch"
        );
        assert!(
            (res.correction - corrected_by_formula).abs() < 1e-12,
            "{name}: corrected by formula mismatch"
        );
        assert_eq!(res.correction_per_cell(), res.correction);
        assert_eq!(
            res.corrected_energy_per_cell(-1.0),
            -1.0 + res.correction
        );
    }
}

#[test]
fn richardson_extrapolation_meets_one_percent_gate_and_raw_fails() {
    let frozen = load("mpc_n1");
    let cases = frozen["cases"].as_object().unwrap();

    for &l in &[10.0, 14.0, 18.0] {
        let key_48 = format!("L{l:.1}_s1.0_n48");
        let key_96 = format!("L{l:.1}_s1.0_n96");

        let case_48 = &cases[&key_48];
        let case_96 = &cases[&key_96];

        let coarse = num(&case_48["corrected_by_formula"]);
        let fine = num(&case_96["corrected_by_formula"]);
        let expected = num(&case_48["corrected_expected"]); // 2 * pi * s^2 / Omega

        // Richardson in h for h^2 law: (4 * fine - coarse) / 3
        let richardson = (4.0 * fine - coarse) / 3.0;
        let rel_err_richardson = (richardson - expected).abs() / expected;

        assert!(
            rel_err_richardson < 0.01,
            "L={l}: Richardson error {rel_err_richardson:.2e} must be < 1%"
        );

        // T3-note: raw n=48 fails the 1% gate
        let rel_err_raw_48 = (coarse - expected).abs() / expected;
        assert!(
            rel_err_raw_48 > 0.01,
            "L={l}: raw n=48 error {rel_err_raw_48:.2e} must exceed 1% as noted in plan"
        );
    }
}

#[test]
fn hole_trace_identity_t5() {
    // For an idempotent 1-RDM (D^2 = D) and HF 2-RDM Gamma_{pqrs} = D_{pr} D_{qs} - D_{ps} D_{qr}:
    // sum_q (Gamma_{pqrq} - D_{pr} D_{qq}) = - sum_q D_{pq} D_{qr} = -(D^2)_{pr} = -D_{pr}.
    let m = 2;
    let mut d = vec![0.0; m * m];
    d[0 * m + 0] = 1.0;
    d[1 * m + 1] = 1.0;

    let mut gamma = vec![0.0; m * m * m * m];
    for p in 0..m {
        for q in 0..m {
            for r in 0..m {
                for s in 0..m {
                    let idx = ((p * m + q) * m + r) * m + s;
                    gamma[idx] = d[p * m + r] * d[q * m + s] - d[p * m + s] * d[q * m + r];
                }
            }
        }
    }

    let residual = verify_hole_trace(&gamma, &d, m).unwrap();
    assert!(
        residual < 1e-14,
        "hole trace identity residual: {residual:e} must be < 1e-14"
    );

    // If we perturb Gamma, trace identity fails
    gamma[0] += 0.01;
    let bad_residual = verify_hole_trace(&gamma, &d, m).unwrap();
    assert!(bad_residual > 1e-3);
}

#[test]
fn input_validation_and_errors() {
    assert!(mpc_from_hole_energies(f64::NAN, 0.0, -0.28, 1.0, 1, 1000.0).is_err());
    assert!(mpc_from_hole_energies(0.0, 0.0, f64::INFINITY, 1.0, 1, 1000.0).is_err());
    assert!(mpc_from_hole_energies(0.0, 0.0, -0.28, -1.0, 1, 1000.0).is_err());
    assert!(mpc_from_hole_energies(0.0, 0.0, -0.28, 1.0, 0, 1000.0).is_err());
    assert!(mpc_from_hole_energies(0.0, 0.0, -0.28, 1.0, 1, -10.0).is_err());
}
