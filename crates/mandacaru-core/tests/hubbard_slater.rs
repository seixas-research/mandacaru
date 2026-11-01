//! Slater coupling tensor of a d or f shell: the tensor against the frozen exact references
//! (`test/migration/reference/hubbard/slater.json`, written by `freeze_hubbard_slater.py`) and
//! against a double-sphere quadrature of the `P_k(cos gamma)` kernel built here, selection rules
//! and symmetries, the sum rules `S_k`, the J weights, the ratios, refusals and `Context` memory.
//! The tests marked "reference file" need the JSON; the others are self-contained.
use mandacaru_core::math::angular;
use mandacaru_core::models::hubbard_slater::{
    real_coefficients, reduced, Basis, Coupling, Ratios, D_F4_F2, F_F4_F2, F_F6_F2,
};
use mandacaru_core::{Context, Error};
use num_complex::Complex64 as C64;
mod hubbard_slater_support;
use hubbard_slater_support::{
    dense, determinant, fraction, hydrogenic, name, number, quadrature, reference, spread,
};

fn context() -> Context {
    Context::new(Some(1), 8 << 20).unwrap()
}
fn build(l: usize, basis: Basis) -> Coupling {
    Coupling::new(&context(), l, basis).unwrap()
}
fn message(e: Error) -> String {
    match e {
        Error::Invalid(m) => m,
        other => panic!("expected Error::Invalid, got {other:?}"),
    }
}

#[test]
fn reduced_coefficients_equal_the_frozen_table() {
    // Reference file. The first check: a sign or argument-order error shows up here, in 25 numbers.
    let r = reference();
    for l in [2usize, 3] {
        let p = 2 * l + 1;
        for (k, table) in r[name(l)]["c_k_complex"].as_object().unwrap() {
            let k: usize = k.parse().unwrap();
            for (a, row) in table.as_array().unwrap().iter().enumerate() {
                for (b, v) in row.as_array().unwrap().iter().enumerate() {
                    let want = number(v.as_str().unwrap());
                    let got = reduced(l, k, a as i32 - l as i32, b as i32 - l as i32).unwrap();
                    assert!(
                        (got - want).abs() < 1e-14,
                        "l={l} k={k} ({a},{b}): {got} vs {want}"
                    );
                }
            }
            assert_eq!(table.as_array().unwrap().len(), p);
        }
    }
}

#[test]
fn tensors_equal_the_frozen_references_in_both_bases() {
    let r = reference();
    for l in [2usize, 3] {
        for (basis, key) in [(Basis::Complex, "a_k_complex"), (Basis::Real, "a_k_real")] {
            let c = build(l, basis);
            let size = c.p().pow(4);
            for &k in c.orders() {
                let want = dense(&r[name(l)][key][k.to_string()], size);
                let err = spread(c.tensor(k).unwrap(), &want);
                assert!(err < 1e-13, "l={l} {key} k={k}: max difference {err:e}");
            }
        }
    }
}

#[test]
fn sums_and_j_weights_equal_the_exact_fractions() {
    let r = reference();
    for l in [2usize, 3] {
        let c = build(l, Basis::Complex);
        let (rules, j) = (&r[name(l)]["sum_rules"], &r[name(l)]["J"]);
        let weights = c.j_weights().unwrap();
        for (i, &k) in c.orders().iter().enumerate() {
            let rule = &rules[k.to_string()];
            let want = |key: &str| fraction(rule[key].as_str().unwrap());
            assert!((c.direct_sum(k).unwrap() - want("sum_ab_abab")).abs() < 1e-13);
            assert!((c.exchange_sum(k).unwrap() - want("sum_ab_abba")).abs() < 1e-13);
            let p2 = (c.p() * c.p()) as f64;
            assert!((c.exchange_sum(k).unwrap() - p2 * want("3j_lkl_000_squared")).abs() < 1e-13);
            let coefficient = fraction(j["J_filled_shell_coeff"][k.to_string()].as_str().unwrap());
            assert!((weights[i] - coefficient).abs() < 1e-14, "l={l} k={k}");
        }
    }
}

#[test]
fn determinant_energies_equal_the_frozen_coefficients() {
    let r = reference();
    for l in [2usize, 3] {
        let (complex, real) = (build(l, Basis::Complex), build(l, Basis::Real));
        let mut checked = 0;
        for det in r[name(l)]["determinants"].as_object().unwrap().values() {
            let Some(list) = det.get("orbitals_index_spin") else {
                continue;
            };
            let occ: Vec<(usize, bool)> = list
                .as_array()
                .unwrap()
                .iter()
                .map(|o| (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() == 1))
                .collect();
            let c = if det["basis"] == "real" {
                &real
            } else {
                &complex
            };
            let (e_int, e_dc) = determinant(c, &occ);
            for (i, &k) in c.orders().iter().enumerate() {
                let want = |key: &str| number(det[key][k.to_string()].as_str().unwrap());
                assert!(
                    (e_int[i] - want("E_int_coeff")).abs() < 1e-13,
                    "l={l} {det} E_int k={k}"
                );
                assert!(
                    (e_dc[i] - want("E_dc_coeff")).abs() < 1e-13,
                    "l={l} E_dc k={k}"
                );
                assert!(
                    (e_int[i] - e_dc[i] - want("E_U_coeff")).abs() < 1e-13,
                    "l={l} E_U k={k}"
                );
            }
            checked += 1;
        }
        assert!(checked >= 10, "only {checked} determinants checked");
    }
}

// ---- self-contained -------------------------------------------------------------------------

#[test]
fn tensors_equal_an_independent_double_sphere_quadrature() {
    for l in [2usize, 3] {
        for basis in [Basis::Complex, Basis::Real] {
            let c = build(l, basis);
            for &k in c.orders() {
                let err = spread(c.tensor(k).unwrap(), &quadrature(l, basis, k));
                assert!(err < 1e-12, "l={l} {basis:?} k={k}: max difference {err:e}");
            }
        }
    }
}

#[test]
fn the_real_basis_is_the_engine_convention() {
    // sum_m T[r][m] Y_m(point) = the real harmonic of math::angular at random directions.
    let mut state = 12345u64;
    let mut unit = || {
        let mut v = [0.; 3];
        for x in &mut v {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *x = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        }
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.map(|x| x / n)
    };
    for l in [2usize, 3] {
        let (t, p) = (real_coefficients(l).unwrap(), 2 * l + 1);
        for _ in 0..6 {
            let point = unit();
            for r in 0..p {
                let sum: C64 = (0..p)
                    .map(|m| t[r * p + m] * angular::complex(l, m as i32 - l as i32, point))
                    .sum();
                let want = angular::cartesian(l, r as i32 - l as i32, point);
                assert!((sum - C64::new(want, 0.)).norm() < 1e-13, "l={l} r={r}");
            }
        }
    }
}

#[test]
fn selection_rules_and_symmetries() {
    for l in [2usize, 3] {
        let p = 2 * l + 1;
        let idx = |a: usize, b: usize, c: usize, d: usize| ((a * p + b) * p + c) * p + d;
        let complex = build(l, Basis::Complex);
        for &k in complex.orders() {
            let t = complex.tensor(k).unwrap();
            for (a, b, c, d) in iproduct(p) {
                let v = t[idx(a, b, c, d)];
                if a + b != c + d {
                    assert!(
                        v.abs() < 1e-14,
                        "m is not conserved at k={k} ({a}{b}{c}{d})"
                    );
                }
                // Hermiticity (1<->3, 2<->4) and the electron exchange (1<->2, 3<->4)
                assert!((v - t[idx(c, d, a, b)]).abs() < 1e-14);
                assert!((v - t[idx(b, a, d, c)]).abs() < 1e-14);
            }
        }
        let real = build(l, Basis::Real);
        for &k in real.orders() {
            let t = real.tensor(k).unwrap();
            for (a, b, c, d) in iproduct(p) {
                let v = t[idx(a, b, c, d)];
                // real orbitals: the eight-fold permutation symmetry
                for w in [
                    t[idx(c, d, a, b)],
                    t[idx(b, a, d, c)],
                    t[idx(c, b, a, d)],
                    t[idx(a, d, c, b)],
                ] {
                    assert!(
                        (v - w).abs() < 1e-13,
                        "real symmetry at k={k} ({a}{b}{c}{d})"
                    );
                }
            }
        }
        // odd multipoles are not in the shell: no tensor for them
        assert!(complex.tensor(1).is_err() && complex.tensor(2 * l + 2).is_err());
    }
}

fn iproduct(p: usize) -> impl Iterator<Item = (usize, usize, usize, usize)> {
    (0..p.pow(4)).map(move |i| (i / (p * p * p), i / (p * p) % p, i / p % p, i % p))
}

#[test]
fn completeness_and_textbook_anchors() {
    for l in [2usize, 3] {
        let c = build(l, Basis::Complex);
        let p = c.p() as f64;
        // sum_k (2k+1) (l k l; 000)^2 = 1  <=>  sum_k (2k+1) S_k = p^2
        let total: f64 = c
            .orders()
            .iter()
            .map(|&k| (2 * k + 1) as f64 * c.exchange_sum(k).unwrap())
            .sum();
        assert!((total - p * p).abs() < 1e-12);
        assert!((c.direct_sum(0).unwrap() - p * p).abs() < 1e-12);
        // c^0(m, m) = 1; the diagonal c^k(m,m) is the Legendre average, symmetric in m -> -m
        for m in -(l as i32)..=l as i32 {
            assert!((reduced(l, 0, m, m).unwrap() - 1.).abs() < 1e-14);
            assert!((reduced(l, 2, m, m).unwrap() - reduced(l, 2, -m, -m).unwrap()).abs() < 1e-14);
        }
    }
    // d shell, Condon-Shortley: c^2(0,0) = c^4(0,0) = 2/7 (so <00|v|00> = F0 + 4/49 F2 + 4/49 F4)
    assert!((reduced(2, 2, 0, 0).unwrap() - 2. / 7.).abs() < 1e-14);
    assert!((reduced(2, 4, 0, 0).unwrap() - 2. / 7.).abs() < 1e-14);
}

#[test]
fn the_one_electron_filled_and_half_filled_shells_have_no_correction() {
    // E_Hub = E_dc for d1, for the filled shell and the half-filled all-up shell, at every k.
    for l in [2usize, 3] {
        for basis in [Basis::Complex, Basis::Real] {
            let c = build(l, basis);
            let p = c.p();
            let shells: [Vec<(usize, bool)>; 3] = [
                vec![(0, true)],
                (0..p).flat_map(|a| [(a, true), (a, false)]).collect(),
                (0..p).map(|a| (a, true)).collect(),
            ];
            for occ in &shells {
                let (e_int, e_dc) = determinant(&c, occ);
                for i in 0..e_int.len() {
                    assert!(
                        (e_int[i] - e_dc[i]).abs() < 1e-13,
                        "l={l} {basis:?} N={}",
                        occ.len()
                    );
                }
            }
            // a negative control: two opposite-spin electrons on one orbital are NOT corrected
            let (e_int, e_dc) = determinant(&c, &[(1, true), (1, false)]);
            assert!(e_int.iter().zip(&e_dc).any(|(a, b)| (a - b).abs() > 1e-3));
        }
    }
}

#[test]
fn the_default_ratios_are_pinned_and_the_hydrogenic_ones_are_derived() {
    // d: 15/23 is hydrogenic 3d and is NOT the default (HL-D2); f defaults are hydrogenic 4f
    assert!((hydrogenic(2, 4) / hydrogenic(2, 2) - 15. / 23.).abs() < 1e-9);
    assert!((hydrogenic(3, 4) / hydrogenic(3, 2) - 451. / 675.).abs() < 1e-9);
    assert!((hydrogenic(3, 6) / hydrogenic(3, 2) - 1001. / 2025.).abs() < 1e-9);
    assert_eq!(D_F4_F2, 0.625);
    assert_eq!((F_F4_F2, F_F6_F2), (451. / 675., 1001. / 2025.));
    assert_eq!(
        Ratios::default_for(2).unwrap(),
        Ratios {
            f4_f2: 5. / 8.,
            f6_f2: None
        }
    );
    assert_eq!(
        Ratios::default_for(3).unwrap(),
        Ratios {
            f4_f2: 451. / 675.,
            f6_f2: Some(1001. / 2025.)
        }
    );
    let over = Ratios::new(3, Some(0.5), None).unwrap();
    assert_eq!((over.f4_f2, over.f6_f2), (0.5, Some(1001. / 2025.)));
}

#[test]
fn integrals_follow_u_j_and_the_ratios_and_j_comes_back() {
    let d = build(2, Basis::Complex);
    let f = build(3, Basis::Complex);
    // exact textbook-form fractions: d J = (F2 + F4)/14; f J = 2/45 F2 + 1/33 F4 + 50/1287 F6
    let r = Ratios::default_for(2).unwrap();
    let ints = d.integrals(6., 0.9, &r).unwrap();
    let f2 = 14. * 0.9 / (1. + 5. / 8.);
    assert_eq!(ints[0], 6.);
    assert!((ints[1] - f2).abs() < 1e-12 && (ints[2] - 5. / 8. * f2).abs() < 1e-12);
    assert!((d.exchange_j(&ints).unwrap() - 0.9).abs() < 1e-13);
    let r = Ratios::new(3, Some(0.7), Some(0.5)).unwrap();
    let ints = f.integrals(7., 1.2, &r).unwrap();
    let weight = 2. / 45. + 0.7 / 33. + 0.5 * 50. / 1287.;
    assert!((ints[1] - 1.2 / weight).abs() < 1e-12);
    assert!((ints[2] / ints[1] - 0.7).abs() < 1e-14 && (ints[3] / ints[1] - 0.5).abs() < 1e-14);
    let j: f64 = 2. / 45. * ints[1] + ints[2] / 33. + 50. / 1287. * ints[3];
    assert!((f.exchange_j(&ints).unwrap() - j).abs() < 1e-13 && (j - 1.2).abs() < 1e-13);
    // J = 0 gives only F0; a different ratio changes F2 (negative control)
    assert_eq!(
        d.integrals(4., 0., &Ratios::default_for(2).unwrap())
            .unwrap(),
        vec![4., 0., 0.]
    );
    let other = Ratios::new(2, Some(0.7), None).unwrap();
    assert!((d.integrals(6., 0.9, &other).unwrap()[1] - f2).abs() > 1e-3);
}

#[test]
fn unsupported_input_raises() {
    let ctx = context();
    for l in [0usize, 1, 4] {
        let m = message(Coupling::new(&ctx, l, Basis::Complex).unwrap_err());
        assert!(m.contains("d or f"), "{m}");
        assert!(Ratios::default_for(l).is_err() && real_coefficients(l).is_err());
    }
    let d = build(2, Basis::Complex);
    let ok = Ratios::default_for(2).unwrap();
    for (u, j) in [(-1., 0.5), (3., -0.1), (f64::NAN, 0.5), (3., f64::INFINITY)] {
        assert!(message(d.integrals(u, j, &ok).unwrap_err()).contains("non-negative"));
    }
    for bad in [-0.5, 0., 1.5, f64::NAN, f64::INFINITY] {
        let m = message(Ratios::new(2, Some(bad), None).unwrap_err());
        assert!(m.contains("f4_f2"), "{m}");
        assert!(Ratios::new(3, None, Some(bad)).is_err());
        let hand = Ratios {
            f4_f2: bad,
            f6_f2: None,
        };
        assert!(d.integrals(1., 0.1, &hand).is_err());
    }
    assert!(message(Ratios::new(2, None, Some(0.5)).unwrap_err()).contains("d shell"));
    assert!(message(
        d.integrals(
            1.,
            0.1,
            &Ratios {
                f4_f2: 0.6,
                f6_f2: Some(0.5)
            }
        )
        .unwrap_err()
    )
    .contains("d shell"));
    let f = build(3, Basis::Complex);
    assert!(message(
        f.integrals(
            1.,
            0.1,
            &Ratios {
                f4_f2: 0.6,
                f6_f2: None
            }
        )
        .unwrap_err()
    )
    .contains("f6_f2"));
    // negative F, wrong length, bad orbital index, unknown order
    assert!(message(d.exchange_j(&[1., -0.1, 0.2]).unwrap_err()).contains("non-negative"));
    assert!(d.exchange_j(&[1., 0.1]).is_err());
    assert!(d.value(0, [5, 0, 0, 0]).is_err() && d.value(3, [0; 4]).is_err());
    assert!(reduced(2, 2, 3, 0).is_err());
}

#[test]
fn memory_is_reserved_from_the_context_and_returned() {
    for (l, basis) in [
        (2usize, Basis::Complex),
        (2, Basis::Real),
        (3, Basis::Complex),
        (3, Basis::Real),
    ] {
        let ctx = context();
        let p4 = (2 * l + 1).pow(4);
        let c = Coupling::new(&ctx, l, basis).unwrap();
        let stored = (l + 1) * p4 * 8;
        assert_eq!(c.bytes(), stored);
        assert_eq!(ctx.memory_used(), stored, "the workspace must be returned");
        let workspace = if basis == Basis::Real { p4 * 16 } else { 0 };
        assert_eq!(ctx.memory_peak(), stored + workspace);
        assert!(ctx.memory_peak() <= 2 << 20 && p4 * 16 <= 2 << 20);
        drop(c);
        assert_eq!(ctx.memory_used(), 0);
        // a budget one byte short keeps nothing; one that fits the tensors but not the
        // real-basis workspace keeps nothing either
        let tiny = Context::new(Some(1), stored - 1).unwrap();
        assert!(matches!(
            Coupling::new(&tiny, l, basis).unwrap_err(),
            Error::Budget { .. }
        ));
        assert_eq!(tiny.memory_used(), 0);
        if basis == Basis::Real {
            let tight = Context::new(Some(1), stored + workspace - 1).unwrap();
            assert!(matches!(
                Coupling::new(&tight, l, basis).unwrap_err(),
                Error::Budget { .. }
            ));
            assert_eq!(tight.memory_used(), 0);
        }
    }
}

#[test]
fn construction_is_deterministic() {
    for basis in [Basis::Complex, Basis::Real] {
        let (a, b) = (build(3, basis), build(3, basis));
        for &k in a.orders() {
            assert!(a
                .tensor(k)
                .unwrap()
                .iter()
                .zip(b.tensor(k).unwrap())
                .all(|(x, y)| x.to_bits() == y.to_bits()));
        }
    }
}
