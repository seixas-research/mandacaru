//! KZK finite-size LDA: parity with the frozen pre-port Python numbers
//! (`test/migration/reference/finite_size/kzk_{table,density,heg}.json`).
//!
//! Table I (a1..a3, g1..g4) is ported AS FROZEN and is TO-VERIFY against Kwee, Zhang and
//! Krakauer, PRL 100, 126404 (2008). These tests pin the port to the old code, not the
//! constants to the paper. The correlation branch has no independent reference (unpinned).
//!
//! Mapping of the 21 non-run functions of `test/core/test_kzk.py`:
//! table_one_is_transcribed -> constants_are_the_frozen_table; a0_is_the_slater_constant ->
//! same + slater_exchange_is_a0_over_rs; gamma_matches_the_figure_one_jumps ->
//! boundaries_and_figure_one; boundaries_are_ordered -> same; infinite_exchange_is_the_lda_one
//! -> slater_exchange_is_a0_over_rs; correlation_helper_is_rydberg and correlation_derivative_is_exact
//! -> perdew_zunger_is_rydberg_and_exact; terms_cancel_exactly, residual_scales_as_one_over_l_cubed,
//! reaches_the_infinite_limit -> designed_cancellation; value/slope continuity ->
//! cubic_is_continuous_in_value_and_slope; correlation_vanishes_beyond_gamma_l and
//! exchange_decays_beyond_gamma -> branches_beyond_the_boundaries; effective_length ->
//! effective_length_is_the_equal_volume_cube; cubic_cell_has_no_deviation and slab_cell_reports ->
//! cubic_deviation_is_reported; empty_density_is_refused -> edge_cases_raise_or_match; scopes_are_kept_apart ->
//! scopes_are_kept_apart; correction_is_the_difference -> density_parity (xc_infinite - xc_finite);
//! huge_cell_needs_almost_no_correction -> DROPPED: it compared two cells at one inconsistent dV;
//! the L -> infinity limit is `designed_cancellation`.
use mandacaru_core::models::xc::lda_terms;
use mandacaru_core::periodic::finite_size::kzk::{self, Fit, Vacuum};
use serde_json::Value;
use std::f64::consts::PI;

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
fn arr(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}

/// Largest `|got - want| / max(|want|, floor)` seen, asserted against `rel`.
struct Worst {
    label: &'static str,
    rel: f64,
    seen: f64,
}
impl Worst {
    fn new(label: &'static str, rel: f64) -> Self {
        Self {
            label,
            rel,
            seen: 0.0,
        }
    }
    fn check(&mut self, got: f64, want: f64, floor: f64, what: &str) {
        let err = (got - want).abs() / want.abs().max(floor);
        self.seen = self.seen.max(err);
        assert!(
            err <= self.rel,
            "{} {what}: got {got:e}, want {want:e} (rel {err:e})",
            self.label
        );
    }
    fn report(&self) {
        println!(
            "{}: max relative difference {:.3e} (gate {:.0e})",
            self.label, self.seen, self.rel
        );
    }
}

fn open() -> Vacuum {
    Vacuum {
        floor: kzk::DENSITY_FLOOR,
        max_fraction: 1.0,
    }
}
fn cube(l: f64) -> [[f64; 3]; 3] {
    [[l, 0., 0.], [0., l, 0.], [0., 0., l]]
}
fn neumaier(values: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0, 0.0);
    for v in values {
        let t = s + v;
        c += if s.abs() >= v.abs() {
            (s - t) + v
        } else {
            (v - t) + s
        };
        s = t;
    }
    s + c
}
/// Sum of periodic minimum-image Gaussians on an `n^3` grid, normalized to `charge`.
fn blob(n: usize, l: f64, centers: &[[f64; 3]], sigma: f64, charge: f64) -> (Vec<f64>, f64) {
    let h = l / n as f64;
    let wrap = |x: f64, c: f64| (x - c + 0.5 * l).rem_euclid(l) - 0.5 * l;
    let mut rho = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let (x, y, z) = (i as f64 * h, j as f64 * h, k as f64 * h);
                rho.push(
                    centers
                        .iter()
                        .map(|c| {
                            let r2 = wrap(x, c[0]).powi(2)
                                + wrap(y, c[1]).powi(2)
                                + wrap(z, c[2]).powi(2);
                            (-r2 / (2.0 * sigma * sigma)).exp()
                        })
                        .sum::<f64>(),
                );
            }
        }
    }
    let scale = charge / (neumaier(rho.iter().copied()) * h.powi(3));
    rho.iter_mut().for_each(|v| *v *= scale);
    (rho, h.powi(3))
}

#[test]
fn constants_are_the_frozen_table() {
    // TO-VERIFY against PRL 100, 126404, Table I: self-consistency with the old code only.
    let c = &load("kzk_table")["constants"];
    let pairs = [
        (kzk::A1, "A1_Ry"),
        (kzk::A2, "A2_Ry"),
        (kzk::A3, "A3_Ry"),
        (kzk::G1, "G1_Ry"),
        (kzk::G2, "G2_Ry"),
        (kzk::G3, "G3_Ry"),
        (kzk::G4, "G4_Ry"),
    ];
    for (value, key) in pairs {
        assert_eq!(value, num(&c[key]), "{key}");
    }
    assert!((kzk::a0() - num(&c["A0_Ry"])).abs() < 1e-14);
    assert!((kzk::a0() - num(&c["A0_independent_Ry"])).abs() < 1e-14);
    assert!((kzk::a0() + 0.916).abs() < 5e-4);
}

#[test]
fn slater_exchange_is_a0_over_rs() {
    for rs in [1.0_f64, 2.0, 5.0] {
        let density = 3.0 / (4.0 * PI * rs.powi(3));
        let slater = -0.75 * (3.0 / PI).cbrt() * density.cbrt();
        let got = kzk::a0() / rs * kzk::HARTREE_PER_RYDBERG;
        assert!((got - slater).abs() <= 1e-12 * slater.abs());
    }
}

#[test]
fn perdew_zunger_is_rydberg_and_exact() {
    let table = load("kzk_table");
    let (pz, ry) = (&table["perdew_zunger"], &table["rydberg_pz"]);
    let rs = arr(&ry["rs"]);
    let (ec, dec) = (arr(&ry["ec_Ry"]), arr(&ry["dec_Ry"]));
    let (ec_native, vc_native) = (arr(&pz["ec_native"]), arr(&pz["vc_native"]));
    let mut worst = Worst::new("rydberg_pz", 1e-12);
    for (i, &r) in rs.iter().enumerate() {
        let (value, slope) = kzk::pz_correlation_rydberg(r);
        worst.check(value, ec[i], 1e-300, "e_c");
        worst.check(slope, dec[i], 1e-300, "de_c/dr_s");
        let (_, _, hartree, v) = lda_terms(3.0 / (4.0 * PI * r.powi(3)));
        worst.check(hartree, ec_native[i], 1e-300, "native e_c");
        worst.check(v, vc_native[i], 1e-300, "native v_c");
        assert!((value - 2.0 * hartree).abs() <= 1e-14 * value.abs());
        // The relation v_c = e_c - (r_s/3) de_c/dr_s, against a central difference.
        let step = 1e-6 * r;
        let numeric = (kzk::pz_correlation_rydberg(r + step).0
            - kzk::pz_correlation_rydberg(r - step).0)
            / (2.0 * step);
        if (r - 1.0).abs() > 2e-6 {
            assert!(
                (slope - numeric).abs() <= 1e-6 * numeric.abs().max(1e-3),
                "rs {r}"
            );
        }
    }
    worst.report();
    let g = &table["g_function"];
    let (grs, gv, dg) = (arr(&g["rs"]), arr(&g["g"]), arr(&g["dg"]));
    let mut worst = Worst::new("g_function", 1e-12);
    for i in 0..grs.len() {
        worst.check(kzk::g_function(grs[i]), gv[i], 1e-300, "g");
        worst.check(kzk::g_derivative(grs[i]), dg[i], 1e-300, "dg");
    }
    worst.report();
}

#[test]
fn boundaries_and_figure_one() {
    for (l, seen) in [(7.5, 3.7), (10.3, 5.0), (18.0, 8.9)] {
        assert!((Fit::new(l).unwrap().gamma - seen).abs() < 0.15, "L {l}");
    }
    let f = Fit::new(12.0).unwrap();
    assert!(f.gamma_h < f.gamma && f.gamma < f.gamma_l);
}

#[test]
fn fit_matches_the_frozen_curves() {
    let table = load("kzk_table");
    let mut worst = Worst::new("by_L", 1e-12);
    for (key, case) in table["by_L"].as_object().unwrap() {
        let l: f64 = key.parse().unwrap();
        let fit = Fit::new(l).unwrap();
        for (got, name) in [
            (fit.alpha, "alpha"),
            (fit.beta, "beta"),
            (fit.gamma, "gamma"),
            (fit.gamma_h, "gamma_h"),
            (fit.gamma_l, "gamma_l"),
        ] {
            worst.check(got, num(&case[name]), 1e-300, name);
        }
        let rs = arr(&case["rs"]);
        let (ex, ec, xc, inf) = (
            arr(&case["ex"]),
            arr(&case["ec"]),
            arr(&case["xc_fs"]),
            arr(&case["xc_inf"]),
        );
        for i in 0..rs.len() {
            worst.check(fit.exchange(rs[i]), ex[i], 1e-300, "e_x");
            worst.check(fit.correlation(rs[i]), ec[i], 1e-18, "e_c");
            worst.check(fit.exchange_correlation(rs[i]), xc[i], 1e-300, "e_xc");
            worst.check(
                kzk::infinite_exchange_correlation(rs[i]),
                inf[i],
                1e-300,
                "e_xc inf",
            );
        }
        assert!(
            (kzk::effective_length(l.powi(3)) - num(&case["effective_length_of_volume"])).abs()
                < 1e-12 * l
        );
        let low: Vec<f64> = rs
            .iter()
            .copied()
            .filter(|&r| r <= fit.gamma.min(fit.gamma_h))
            .collect();
        let cancel = kzk::leading_term_cancellation(&low, l).unwrap();
        assert!(cancel < 1e-12, "L {l}: cancellation {cancel:e}");
    }
    worst.report();
}

#[test]
fn designed_cancellation() {
    for l in [7.5, 18.0, 50.0] {
        let gh = Fit::new(l).unwrap().gamma_h;
        let rs: Vec<f64> = (0..40)
            .map(|i| 0.2 + (0.9 * gh - 0.2) * i as f64 / 39.0)
            .collect();
        assert!(kzk::leading_term_cancellation(&rs, l).unwrap() < 1e-10);
    }
    let diff = |l: f64| {
        (Fit::new(l).unwrap().exchange_correlation(2.0) - kzk::infinite_exchange_correlation(2.0))
            .abs()
    };
    let (e10, e50, e250) = (diff(10.0), diff(50.0), diff(250.0));
    assert!((e10 / e50 / 125.0 - 1.0).abs() < 0.02 && (e50 / e250 / 125.0 - 1.0).abs() < 0.02);
    for rs in [0.8, 2.0, 4.0] {
        let inf = kzk::infinite_exchange_correlation(rs);
        assert!((Fit::new(1e6).unwrap().exchange_correlation(rs) - inf).abs() <= 1e-8 * inf.abs());
    }
    // Frozen 1/L^3 scaling after the a1/L^2 cancellation (slope -3).
    for (key, s) in load("kzk_table")["xc_scaling"].as_object().unwrap() {
        let (ls, frozen, slopes) = (arr(&s["L"]), arr(&s["abs_diff"]), arr(&s["slopes"]));
        let rs: f64 = key.parse().unwrap();
        let mut logs = Vec::new();
        for (i, &l) in ls.iter().enumerate() {
            let d = (Fit::new(l).unwrap().exchange_correlation(rs)
                - kzk::infinite_exchange_correlation(rs))
            .abs();
            assert!((d - frozen[i]).abs() <= 1e-6 * frozen[i], "L {l}");
            logs.push((l.ln(), d.ln()));
        }
        for (i, w) in logs.windows(2).enumerate() {
            let slope = (w[1].1 - w[0].1) / (w[1].0 - w[0].0);
            assert!(
                (slope + 3.0).abs() < 1e-6 && (slope - slopes[i]).abs() < 1e-6,
                "slope {slope}"
            );
        }
    }
}

#[test]
fn cubic_is_continuous_in_value_and_slope() {
    for l in [7.5, 10.0, 12.0, 20.0] {
        let f = Fit::new(l).unwrap();
        let (hv, hs) = f.high_correlation_rydberg(f.gamma_h);
        let (cv, cs) = f.cubic_rydberg(f.gamma_h);
        assert!(
            (hv - cv).abs() <= 1e-12 * hv.abs() && (hs - cs).abs() <= 1e-12 * hs.abs(),
            "gamma_h, L {l}"
        );
        let (zv, zs) = f.cubic_rydberg(f.gamma_l);
        assert!(zv == 0.0 && zs == 0.0, "gamma_l, L {l}");
        for b in [f.gamma_h, f.gamma_l] {
            let (lo, hi) = (
                f.correlation(b * (1.0 - 1e-9)),
                f.correlation(b * (1.0 + 1e-9)),
            );
            assert!((lo - hi).abs() < 1e-9, "value at {b}");
        }
    }
    // Exchange is DISCONTINUOUS at gamma = r_s(N = 2) (a factor ~2.7 at L = 7.5). Pinned as
    // frozen; whether the paper's Eq. (3) does this is TO-VERIFY (see the FS-1 log entry).
    for (key, case) in load("kzk_table")["by_L"].as_object().unwrap() {
        let f = Fit::new(key.parse().unwrap()).unwrap();
        let ex = arr(&case["continuity"]["gamma"]["ex"]);
        let (below, above) = (
            f.exchange(f.gamma * (1.0 - 1e-9)),
            f.exchange(f.gamma * (1.0 + 1e-9)),
        );
        assert!(
            (below - ex[0]).abs() <= 1e-8 * ex[0].abs()
                && (above - ex[2]).abs() <= 1e-8 * ex[2].abs()
        );
        assert!(
            above > below * 0.5 && above.abs() < below.abs(),
            "jump at gamma, L {key}"
        );
    }
}

#[test]
fn branches_beyond_the_boundaries() {
    let f = Fit::new(10.0).unwrap();
    assert_eq!(f.correlation(f.gamma_l * 1.5), 0.0);
    assert!(f.exchange(f.gamma * 3.0).abs() < f.exchange(f.gamma * 1.01).abs());
}

#[test]
fn effective_length_is_the_equal_volume_cube() {
    assert!((kzk::effective_length(27.0) - 3.0).abs() < 1e-14);
    assert!(Fit::new(0.0).is_err() && Fit::new(f64::NAN).is_err() && Fit::new(-1.0).is_err());
}

#[test]
fn cubic_deviation_is_reported() {
    let cubic = kzk::kzk(&[0.05; 8], 1.0, &cube(2.0), 1, open()).unwrap();
    assert!(cubic.cubic_deviation.abs() < 1e-12);
    let slab = [[1., 0., 0.], [0., 5., 0.], [0., 0., 5.]];
    assert!(
        kzk::kzk(&[0.05; 8], 25.0 / 8.0, &slab, 1, open())
            .unwrap()
            .cubic_deviation
            > 0.3
    );
}

#[test]
fn scopes_are_kept_apart() {
    let r = kzk::kzk(&[0.02; 64], 1.0, &cube(4.0), 4, open()).unwrap();
    assert_eq!((r.scope.n_cells, r.scope.volume), (4, 64.0));
    assert!((r.correction_per_cell() - r.correction / 4.0).abs() < 1e-15);
    assert!((r.corrected_energy_per_cell(-2.0) - (-2.0 + r.correction / 4.0)).abs() < 1e-15);
}

#[test]
fn density_parity() {
    let frozen = load("kzk_density");
    let mut worst = Worst::new("kzk_density", 1e-12);
    let mut identity = Worst::new("uniform identity N(eps_inf - eps_FS)", 1e-12);
    for (name, c) in frozen["cases"].as_object().unwrap() {
        let l = c.get("L").map_or(0.0, num);
        let n = num(&c["n"]) as usize;
        let (density, dv, cell) = match c["kind"].as_str().unwrap() {
            "uniform" => (
                vec![3.0 / (4.0 * PI * num(&c["rs"]).powi(3)); n * n * n],
                (l / n as f64).powi(3),
                cube(l),
            ),
            "gaussian" => {
                let (d, dv) = blob(n, l, &[[l / 2.0; 3]], 1.5, 4.0);
                (d, dv, cube(l))
            }
            "two_site" => {
                let m = l / 2.0;
                let (d, dv) = blob(n, l, &[[m - 1.0, m, m], [m + 1.0, m, m]], 0.8, 2.0);
                (d, dv, cube(l))
            }
            _ => {
                let v = c["cell"].as_array().unwrap();
                let mut cell = [[0.0; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        cell[i][j] = num(&v[i][j]);
                    }
                }
                let vol = (cell[0][0] * (cell[1][1] * cell[2][2] - cell[1][2] * cell[2][1])
                    - cell[0][1] * (cell[1][0] * cell[2][2] - cell[1][2] * cell[2][0])
                    + cell[0][2] * (cell[1][0] * cell[2][1] - cell[1][1] * cell[2][0]))
                    .abs();
                (
                    vec![3.0 / (4.0 * PI * 8.0); n * n * n],
                    vol / (n * n * n) as f64,
                    cell,
                )
            }
        };
        let cells = num(&c["n_cells"]) as usize;
        let r = kzk::kzk(&density, dv, &cell, cells, open()).unwrap();
        for (got, key) in [
            (r.correction, "correction"),
            (r.xc_infinite, "xc_infinite"),
            (r.xc_finite, "xc_finite"),
            (r.length, "length"),
            (r.scope.volume, "volume"),
            (r.n_electrons, "n_electrons"),
            (r.rs_range.0, "rs_min"),
            (r.rs_range.1, "rs_max"),
        ] {
            let want = match key {
                "rs_min" => num(&c["rs_range"][0]),
                "rs_max" => num(&c["rs_range"][1]),
                _ => num(&c[key]),
            };
            worst.check(got, want, 1e-300, &format!("{name} {key}"));
        }
        assert!(
            (r.cubic_deviation - num(&c["cubic_deviation"])).abs() < 1e-12,
            "{name}"
        );
        assert!((r.xc_infinite - r.xc_finite - r.correction).abs() <= 1e-12 * r.correction.abs());
        worst.check(
            r.correction_per_cell(),
            num(&c["correction_per_cell"]),
            1e-300,
            name,
        );
        if c["kind"] != "uniform" && c["kind"] != "gaussian" && c["kind"] != "two_site" {
            worst.check(
                r.corrected_energy_per_cell(-1.25),
                num(&c["corrected_energy_per_cell"]),
                1e-300,
                name,
            );
        }
        if c["kind"] == "gaussian" {
            worst.check(
                r.electrons_below_floor + 1.0,
                1.0 + num(&c["electrons_below_floor"]),
                1.0,
                name,
            );
        }
        if c["kind"] == "uniform" {
            let rs = num(&c["rs"]);
            let eps = kzk::infinite_exchange_correlation(rs)
                - Fit::new(l).unwrap().exchange_correlation(rs);
            identity.check(r.correction, r.n_electrons * eps, 1e-300, name);
            assert_eq!(r.electrons_below_floor, 0.0);
        }
    }
    worst.report();
    identity.report();
}

#[test]
fn edge_cases_raise_or_match() {
    let edge = &load("kzk_density")["edge"];
    let (l, n) = (12.0, 24usize);
    let dv = (l / n as f64).powi(3);
    let base = vec![3.0 / (4.0 * PI * 8.0); n * n * n];
    let run = |rho: &[f64], v: Vacuum| kzk::kzk(rho, dv, &cube(l), 1, v);
    let with = |f: &dyn Fn(&mut Vec<f64>)| {
        let mut rho = base.clone();
        f(&mut rho);
        rho
    };
    // Python dropped these silently; the port raises and names the node.
    let neg = with(&|r| r[..100].fill(-1e-3));
    assert!(run(&neg, open())
        .unwrap_err()
        .to_string()
        .contains("negative"));
    let nan = with(&|r| r[5] = f64::NAN);
    assert!(run(&nan, open())
        .unwrap_err()
        .to_string()
        .contains("node 5"));
    let inf = with(&|r| r[7] = f64::INFINITY);
    assert!(run(&inf, open())
        .unwrap_err()
        .to_string()
        .contains("node 7"));
    assert_eq!(
        edge["negative_100_points"]["raised"],
        Value::Null,
        "frozen: the old code dropped them"
    );
    assert!(
        run(&vec![1e-14; base.len()], open()).is_err()
            && run(&vec![-1e-3; base.len()], open()).is_err()
    );
    assert!(run(&vec![0.0; base.len()], open()).is_err());
    // Below, at and exactly at the floor are vacuum and skipped, as before.
    let mut worst = Worst::new("edge", 1e-12);
    for (key, value) in [
        ("zero_50_points", 0.0),
        ("below_floor_50_points", 5e-13),
        ("exactly_at_floor_50_points", kzk::DENSITY_FLOOR),
    ] {
        let rho = with(&|r| r[..50].fill(value));
        let k = run(&rho, open()).unwrap();
        worst.check(k.correction, num(&edge[key]["correction"]), 1e-300, key);
        worst.check(k.n_electrons, num(&edge[key]["n_electrons"]), 1e-300, key);
        assert!((k.electrons_below_floor - 50.0 * value * dv).abs() <= 1e-18);
        assert!(
            (k.n_electrons - k.electrons_below_floor - neumaier(rho[50..].iter().map(|v| v * dv)))
                .abs()
                < 1e-12
        );
    }
    let blob_case = |sigma, charge, floor| {
        let (rho, dv) = blob(24, 12.0, &[[6.0; 3]], sigma, charge);
        kzk::kzk(
            &rho,
            dv,
            &cube(12.0),
            1,
            Vacuum {
                floor,
                max_fraction: 1.0,
            },
        )
        .unwrap()
    };
    let raised = blob_case(1.5, 4.0, 1e-3);
    worst.check(
        raised.correction,
        num(&edge["floor_raised_to_1e-3_blob"]["correction"]),
        1e-300,
        "floor 1e-3",
    );
    let vac = blob_case(1.0, 0.3, kzk::DENSITY_FLOOR);
    worst.check(
        vac.correction,
        num(&edge["vacuum_blob_rs_far_beyond_gamma_l"]["correction"]),
        1e-300,
        "vacuum blob",
    );
    worst.report();
    // The vacuum gate: the bound is an argument with no default; it raises above the measured fraction.
    let fraction = vac.electrons_beyond_gamma_l / vac.n_electrons;
    let (rho, dv2) = blob(24, 12.0, &[[6.0; 3]], 1.0, 0.3);
    let gate = |bound| {
        kzk::kzk(
            &rho,
            dv2,
            &cube(12.0),
            1,
            Vacuum {
                floor: kzk::DENSITY_FLOOR,
                max_fraction: bound,
            },
        )
    };
    assert!(fraction > 0.0 && fraction < 1.0, "fraction {fraction}");
    assert!(gate(fraction * 0.99).is_err() && gate(fraction * 1.01).is_ok());
    let dilute = vec![3.0 / (4.0 * PI * 12f64.powi(3)); base.len()];
    assert!(run(
        &dilute,
        Vacuum {
            floor: 1e-12,
            max_fraction: 0.1
        }
    )
    .is_err());
    assert!(
        (run(&dilute, open()).unwrap().electrons_beyond_gamma_l
            / run(&dilute, open()).unwrap().n_electrons
            - 1.0)
            .abs()
            < 1e-14
    );
}

#[test]
fn inputs_are_validated() {
    let ok = |floor, fraction| Vacuum {
        floor,
        max_fraction: fraction,
    };
    let d = [0.05; 8];
    assert!(kzk::kzk(&d, 1.0, &cube(2.0), 1, ok(1e-12, 1.0)).is_ok());
    for bad in [
        ok(0.0, 1.0),
        ok(f64::NAN, 1.0),
        ok(1e-40, 1.0),
        ok(1e-12, -0.1),
        ok(1e-12, 1.5),
        ok(1e-12, f64::NAN),
    ] {
        assert!(kzk::kzk(&d, 1.0, &cube(2.0), 1, bad).is_err());
    }
    for dv in [0.0, -1.0, f64::NAN, f64::INFINITY, 2.0] {
        assert!(kzk::kzk(&d, dv, &cube(2.0), 1, open()).is_err(), "dv {dv}");
    }
    let flat = [[1., 0., 0.], [2., 0., 0.], [0., 0., 1.]];
    assert!(kzk::kzk(&d, 1.0, &flat, 1, open()).is_err());
    assert!(kzk::kzk(&d, 1.0, &cube(2.0), 0, open()).is_err());
    assert!(kzk::kzk(&[], 1.0, &cube(2.0), 1, open()).is_err());
}

#[test]
fn exchange_branch_reproduces_the_gamma_point_gas_diagnostic() {
    // Exchange branch only, to about 10 %; the correlation branch is UNPINNED. The Gamma-point
    // HF values are the frozen ones (not recomputed here).
    let heg = load("kzk_heg");
    let (ls, e_inf, frozen) = (
        arr(&heg["L"]),
        arr(&heg["e_x_inf"]),
        arr(&heg["kzk_fs_exchange"]),
    );
    let mut worst = Worst::new("kzk_heg", 1e-12);
    let mut diffs = Vec::new();
    for i in 0..ls.len() {
        let fs = Fit::new(ls[i]).unwrap().exchange(2.0);
        worst.check(fs, frozen[i], 1e-300, "fs_exchange");
        worst.check(
            kzk::a0() / 2.0 * kzk::HARTREE_PER_RYDBERG,
            e_inf[i],
            1e-300,
            "e_x_inf",
        );
        diffs.push((ls[i].ln(), (fs - e_inf[i]).abs().ln(), fs - e_inf[i]));
    }
    let mean = diffs.iter().map(|d| d.2).sum::<f64>() / diffs.len() as f64;
    worst.check(mean, num(&heg["mean_kzk_minus_inf"]), 1e-300, "mean");
    let m = diffs.len() as f64;
    let (sx, sy) = (
        diffs.iter().map(|d| d.0).sum::<f64>(),
        diffs.iter().map(|d| d.1).sum::<f64>(),
    );
    let sxy = diffs.iter().map(|d| d.0 * d.1).sum::<f64>();
    let sxx = diffs.iter().map(|d| d.0 * d.0).sum::<f64>();
    let slope = (m * sxy - sx * sy) / (m * sxx - sx * sx);
    worst.check(
        slope,
        num(&heg["loglog_slope_kzk"]),
        1e-300,
        "log-log slope",
    );
    worst.report();
    assert!((mean - num(&heg["mean_hf_minus_inf"])).abs() < 5e-4);
    assert!((slope - num(&heg["loglog_slope_hf"])).abs() < 0.1);
}
