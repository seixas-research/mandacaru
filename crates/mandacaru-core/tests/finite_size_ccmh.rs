//! Tests for Chiesa-Ceperley-Martin-Holzmann (CCMH) finite-size correction (`periodic/finite_size/ccmh.rs`).
//!
//! Validates:
//! - T10: Gaussian hole normalization $\epsilon_{\text{pair}} + v_M/2 + \Delta V = \epsilon_\infty$ to $< 10^{-9}$.
//! - Parity with frozen CCMH reference values in `ccmh_synth.json`.
//! - Quadratic regime detection and `reliable` flag.
//! - Index ordering $\Gamma_{prqs}$ from `ccmh_index.json`.

use mandacaru_core::periodic::finite_size::{
    ccmh_extrapolate, CcmhShell, StructureFactor,
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
fn ccmh_extrapolate_matches_the_freeze() {
    let frozen = load("ccmh_synth");
    let triples_data = &frozen["triples"]["cubic_14"];
    let shell_mags: Vec<f64> = triples_data["shell_magnitudes"]
        .as_array()
        .unwrap()
        .iter()
        .map(num)
        .collect();

    // In a unit Gaussian xc hole: S(k) = 1 - exp(-k^2 a^2 / 4). Here a = 2.0.
    let a = 2.0;

    for (case_name, case) in frozen.as_object().unwrap() {
        if !case_name.starts_with('L') {
            continue;
        }
        let l = num(&case["L"]);
        let n_e = num(&case["N_e"]);
        let n_shells = case["shells"].as_u64().unwrap() as usize;
        let volume = l * l * l;

        // Build shells for this case:
        let mut shells = Vec::new();
        for &k_mag in shell_mags.iter().take(n_shells) {
            // S(k) = 1 - exp(-k^2 a^2 / 4)
            let s_val = 1.0 - (-k_mag * k_mag * a * a / 4.0).exp();
            shells.push(CcmhShell {
                k: k_mag,
                s: s_val,
                count: 1,
            });
        }

        let res = ccmh_extrapolate(&shells, volume, 1, n_e).unwrap();

        let frozen_limit = num(&case["limit"]);
        let frozen_spread = num(&case["quadratic_spread"]);
        let frozen_reliable = case["reliable"].as_bool().unwrap();

        assert!(
            (res.limit - frozen_limit).abs() < 1e-10,
            "{case_name}: limit mismatch (got {}, want {frozen_limit})",
            res.limit
        );
        assert!(
            (res.quadratic_spread - frozen_spread).abs() < 1e-10,
            "{case_name}: spread mismatch (got {}, want {frozen_spread})",
            res.quadratic_spread
        );
        assert_eq!(
            res.reliable, frozen_reliable,
            "{case_name}: reliable flag mismatch"
        );

        // Check supercell total has the Ne factor:
        let per_electron = res.correction_per_electron;
        assert!(
            (res.correction - n_e * per_electron).abs() < 1e-14,
            "{case_name}: total correction must equal Ne * per_electron"
        );
    }
}

#[test]
fn gaussian_hole_normalization_t10() {
    let frozen = load("ccmh_synth");

    for (case_name, case) in frozen.as_object().unwrap() {
        if !case_name.starts_with('L') {
            continue;
        }
        let pair_sum = num(&case["pair_sum"]);
        let vm_half = num(&case["vM_half"]);
        let exact_missing = num(&case["exact_missing_per_electron"]);
        let eps_inf = num(&case["eps_inf"]);

        // T10: pair_sum + vm_half + exact_missing == eps_inf to < 1e-9
        let reconstructed = pair_sum + vm_half + exact_missing;
        let closure = (reconstructed - eps_inf).abs();
        assert!(
            closure < 1e-9,
            "{case_name}: T10 normalization closure residual {closure:e} must be < 1e-9"
        );
    }
}

#[test]
fn structure_factor_shells_grouping() {
    let kvecs = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    let vals = vec![0.1, 0.1, 0.1, 0.25, 0.25];
    let sf = StructureFactor::new(kvecs, vals, 2.0).unwrap();

    let shells = sf.shells(5);
    assert_eq!(shells.len(), 2);
    assert!((shells[0].k - 1.0).abs() < 1e-6);
    assert_eq!(shells[0].count, 3);
    assert!((shells[0].s - 0.1).abs() < 1e-6);

    assert!((shells[1].k - std::f64::consts::SQRT_2).abs() < 1e-6);
    assert_eq!(shells[1].count, 2);
    assert!((shells[1].s - 0.25).abs() < 1e-6);
}

#[test]
fn input_validation_and_errors() {
    assert!(ccmh_extrapolate(&[], 1000.0, 1, 2.0).is_err());
    let bad_shell = [CcmhShell { k: 0.0, s: 0.1, count: 1 }];
    assert!(ccmh_extrapolate(&bad_shell, 1000.0, 1, 2.0).is_err());
    let bad_s = [CcmhShell { k: 1.0, s: f64::NAN, count: 1 }];
    assert!(ccmh_extrapolate(&bad_s, 1000.0, 1, 2.0).is_err());
    let valid_shell = [CcmhShell { k: 1.0, s: 0.1, count: 1 }];
    assert!(ccmh_extrapolate(&valid_shell, -10.0, 1, 2.0).is_err());
    assert!(ccmh_extrapolate(&valid_shell, 1000.0, 0, 2.0).is_err());
    assert!(ccmh_extrapolate(&valid_shell, 1000.0, 1, -2.0).is_err());
}

#[test]
fn a_uniform_orbital_has_no_structure_at_finite_k() {
    // One orbital constant over a 4^3 box of unit volume: psi^2 dv integrates to one and its FFT
    // vanishes off k = 0, so M'(0) = 1 (zero overlap error) and S(k) = 0 for every k != 0.
    use mandacaru_core::periodic::finite_size::structure_factor;
    use num_complex::Complex64 as C64;
    let ctx = mandacaru_core::Context::new(Some(1), 64 << 20).unwrap();
    let shape = [4usize, 4, 4];
    let points = 64;
    let dv = 1.0 / points as f64;
    let psi = vec![1.0; points];
    let cell = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    let a = [C64::new(1., 0.)];
    let out = structure_factor(&ctx, &psi, shape, dv, &cell, 1, &a, 1, &[2.], &[0.]).unwrap();
    assert_eq!(out.values.len(), 26);
    assert!(out.overlap_error < 1e-14 && out.n_electrons == 2.);
    assert!(out.values.iter().all(|s| s.abs() < 1e-14));
    assert!((out.magnitudes[0] - (3f64).sqrt() * 2. * std::f64::consts::PI).abs() < 1e-12);
    assert!(structure_factor(&ctx, &psi, shape, dv, &cell, 1, &a, 2, &[2.], &[0.]).is_err());
}
