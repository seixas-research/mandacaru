//! `models::hubbard_spinor`: the j-resolved atomic limit built from
//! `models::spin_orbit::angular`, and the real basis of `math::angular` behind the layout.
mod hubbard_spinor_support;

use hubbard_spinor_support::*;
use mandacaru_core::math::angular;
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

/// `(P_{l+1/2}, P_{l-1/2})` from L.S: eigenvalues l/2 and -(l+1)/2.
fn j_projectors(l: usize, real: bool) -> (DMatrix<C64>, DMatrix<C64>) {
    let ls = ls_matrix(l, real);
    let d = ls.nrows();
    let one = DMatrix::<C64>::identity(d, d);
    let lf = l as f64;
    let hi = (&ls + &one * C64::new(0.5 * (lf + 1.), 0.)) * C64::new(1. / (lf + 0.5), 0.);
    let lo = &one - &hi;
    (hi, lo)
}

#[test]
fn the_j_projectors_from_the_angular_operator_are_the_frozen_ones() {
    let doc = freeze();
    for (name, l) in [("d", 2usize), ("f", 3)] {
        let (hi, lo) = j_projectors(l, false);
        let d = hi.nrows();
        assert!(max_abs(&(&hi * &hi - &hi)) < 1e-14 && max_abs(&(&lo * &lo - &lo)) < 1e-14);
        assert!((hi.trace().re - (2 * l + 2) as f64).abs() < 1e-13);
        assert!((lo.trace().re - (2 * l) as f64).abs() < 1e-13);
        let frozen = &doc[name]["so_atom"]["projectors"];
        for (key, mine) in [("j=l+1/2", &hi), ("j=l-1/2", &lo)] {
            let rows = frozen[key]["P_real_part"].as_array().unwrap();
            let stored = DMatrix::from_fn(d, d, |i, j| C64::new(rows[i][j].as_f64().unwrap(), 0.));
            // The frozen Clebsch-Gordan layout is the layout of angular().
            assert!(max_abs(&(mine - stored)) < 1e-14, "{name} {key}");
        }
    }
}

#[test]
fn the_atomic_limit_reproduces_the_exact_fractions_in_both_bases() {
    let doc = freeze();
    let hb = hub(1.0, 0.);
    for (name, l) in [("d", 2usize), ("f", 3)] {
        for real in [false, true] {
            let (hi, lo) = j_projectors(l, real);
            let lay = layout(l, real);
            let p = lay.width();
            let mut control_moved = false;
            for case in doc[name]["so_atom"]["cases"].as_array().unwrap() {
                let occupations = case["occupation_per_state"].as_object().unwrap();
                let f = |key: &str| occupations.get(key).map_or(0., fraction);
                let (f_hi, f_lo) = (f("j=l+1/2"), f("j=l-1/2"));
                let n = &hi * C64::new(f_hi, 0.) + &lo * C64::new(f_lo, 0.);
                let e = eval(&hb, lay, &n);
                let exact = fraction(&case["E_U_over_U_eff_exact"]);
                assert!(
                    (e.energy - exact).abs() < 1e-13,
                    "{name} real={real} {case}"
                );
                let v = &hi * C64::new(0.5 - f_hi, 0.) + &lo * C64::new(0.5 - f_lo, 0.);
                assert!(max_abs(&(potential(&e) - v)) < 1e-13);
                let mut dropped = n.clone();
                for a in 0..p {
                    for b in 0..p {
                        dropped[(a, p + b)] = C64::default();
                        dropped[(p + a, b)] = C64::default();
                    }
                }
                if !real {
                    let stored = case["control_spin_flip_blocks_dropped_E_U_over_U_eff"]
                        .as_f64()
                        .unwrap();
                    assert!((eval(&hb, lay, &dropped).energy - stored).abs() < 1e-12);
                }
                control_moved |= (eval(&hb, lay, &dropped).energy - exact).abs() > 0.05;
            }
            assert!(
                control_moved,
                "a collinear model must fail some atomic case"
            );
        }
    }
}

#[test]
fn a_pure_single_state_has_no_u_energy_and_a_half_gap_in_v() {
    for (l, real) in CASES {
        let lay = layout(l, real);
        let d = lay.size();
        // The stretched state |m = l, up> is a j = l + 1/2 state in the complex basis; any
        // basis vector is a rank-1 idempotent, which is all this test needs.
        let mut n = DMatrix::<C64>::zeros(d, d);
        let k = lay.index(0, 2 * l).unwrap();
        n[(k, k)] = C64::new(1., 0.);
        let e = eval(&hub(2.0, 0.), lay, &n);
        assert!(e.energy.abs() < 1e-15);
        let v = potential(&e);
        assert!((v[(k, k)].re + 1.0).abs() < 1e-15);
        let other = lay.index(1, 0).unwrap();
        assert!((v[(other, other)].re - 1.0).abs() < 1e-15);
    }
}

#[test]
fn the_real_basis_of_angular_is_the_real_harmonic_of_math_angular() {
    // `angular` writes S_k = sum u[k, k'] Y_k' (complex Y_lm, Condon-Shortley) and
    // L.S(real) = W L.S(complex) W^dagger with W = blockdiag(conj u). The time-reversal
    // operator of the real basis is only i sigma_y K if S_k is `math::angular::cartesian`.
    let mut rng = Lcg(71);
    for l in 1..=3usize {
        let p = 2 * l + 1;
        let root = std::f64::consts::FRAC_1_SQRT_2;
        let mut u = DMatrix::<C64>::zeros(p, p);
        u[(l, l)] = C64::new(1., 0.);
        for m in 1..=l {
            let sign = if m % 2 == 0 { 1. } else { -1. };
            u[(l + m, l + m)] = C64::new(sign * root, 0.);
            u[(l + m, l - m)] = C64::new(root, 0.);
            u[(l - m, l - m)] = C64::new(0., root);
            u[(l - m, l + m)] = C64::new(0., -sign * root);
        }
        let w = DMatrix::from_fn(2 * p, 2 * p, |i, j| {
            if i / p == j / p {
                u[(i % p, j % p)].conj()
            } else {
                C64::default()
            }
        });
        // This copy of u is the one angular() uses ...
        let mapped = &w * ls_matrix(l, false) * w.adjoint();
        assert!(max_abs(&(mapped - ls_matrix(l, true))) < 1e-14, "l={l}");
        // ... and it maps the complex harmonics of math::angular onto its real ones.
        for _ in 0..6 {
            let v = [rng.next(), rng.next(), rng.next()];
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let dir = [v[0] / r, v[1] / r, v[2] / r];
            for k in 0..p {
                let from_complex: C64 = (0..p)
                    .map(|kp| u[(k, kp)] * angular::complex(l, kp as i32 - l as i32, dir))
                    .sum();
                let real = angular::cartesian(l, k as i32 - l as i32, dir);
                assert!(
                    (from_complex - C64::new(real, 0.)).norm() < 1e-12,
                    "l={l} k={k}"
                );
            }
        }
    }
}
