//! `models::hubbard_spinor`: the noncollinear Dudarev functional on the full 2p x 2p
//! occupation matrix. The closed forms, the freeze in
//! `test/migration/reference/hubbard/spinor.json`, rotation invariance, the orientation of
//! V (the energy cannot see it), the collinear embedding, refusals and memory.
mod hubbard_spinor_support;

use hubbard_spinor_support::*;
use mandacaru_core::models::hubbard::{self, Spin};
use mandacaru_core::models::hubbard_spinor::{self as spinor, Layout, Occupation, Potential};
use mandacaru_core::{Context, Error};
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

#[test]
fn the_freeze_reproduces_energy_and_potential_for_d_and_f() {
    let doc = freeze();
    for (name, l) in [("d", 2usize), ("f", 3)] {
        let d = &doc[name]["dudarev"];
        let u_eff = doc[name]["U_eff"].as_f64().unwrap();
        let n = json_matrix(&d["n"]);
        let e = eval(&hub(u_eff, 0.), layout(l, false), &n);
        let want = d["E"].as_f64().unwrap();
        assert!(
            (e.energy - want).abs() < 1e-12 * want.abs().max(1.),
            "{name}"
        );
        let v = json_matrix(&d["V"]);
        assert!(max_abs(&(potential(&e) - v)) < 1e-13, "{name}");
        assert!((e.per_u_eff * u_eff - e.energy).abs() < 1e-14);
    }
}

#[test]
fn the_energy_and_potential_follow_from_the_spectrum() {
    let mut rng = Lcg(11);
    for (l, real) in CASES {
        let lay = layout(l, real);
        let (n, q, lam) = random_state(lay.size(), &mut rng);
        let (u, j) = (3.7, 0.5);
        let u_eff = u - j;
        let e = eval(&hub(u, j), lay, &n);
        let want: f64 = lam.iter().map(|x| 0.5 * u_eff * x * (1. - x)).sum();
        assert!((e.energy - want).abs() < 1e-13, "l={l} real={real}");
        let vd = DMatrix::from_fn(lay.size(), lay.size(), |a, b| {
            C64::new(
                if a == b {
                    0.5 * u_eff * (1. - 2. * lam[a])
                } else {
                    0.
                },
                0.,
            )
        });
        assert!(max_abs(&(potential(&e) - &q * vd * q.adjoint())) < 1e-13);
        let (lo, hi) = spinor::eigenvalue_range(&context(), &occ(lay, &n)).unwrap();
        let (wlo, whi) = (
            lam.iter().cloned().fold(f64::INFINITY, f64::min),
            lam.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        assert!((lo - wlo).abs() < 1e-12 && (hi - whi).abs() < 1e-12);
    }
}

#[test]
fn the_energy_is_invariant_and_the_potential_covariant_under_u_2p() {
    let mut rng = Lcg(23);
    let hb = hub(3.7, 0.);
    for (l, real) in CASES {
        let lay = layout(l, real);
        let (n, _, _) = random_state(lay.size(), &mut rng);
        let base = eval(&hb, lay, &n);
        let q = unitary(lay.size(), &mut rng);
        let rotated = eval(&hb, lay, &(&q * &n * q.adjoint()));
        assert!((rotated.energy - base.energy).abs() < 1e-13);
        let expected = &q * potential(&base) * q.adjoint();
        assert!(max_abs(&(potential(&rotated) - expected)) < 1e-13);
        // The SU(2) spin rotation is one such unitary; the stored one is a second witness.
        let w = spin_rotation(lay.width(), &mut rng);
        let spin = eval(&hb, lay, &(&w * &n * w.adjoint()));
        assert!((spin.energy - base.energy).abs() < 1e-13);
        // Negative control: a non-unitary change of basis is not a symmetry.
        let s = DMatrix::from_fn(lay.size(), lay.size(), |i, j| {
            C64::new(
                if i == j {
                    1. + 0.1 * (i as f64 + 1.) / lay.size() as f64
                } else {
                    0.
                },
                0.,
            )
        });
        let scaled = eval(&hb, lay, &(&s * &n * s.adjoint()));
        assert!((scaled.energy - base.energy).abs() > 1e-3);
    }
}

// ---------------------------------------------------------------- orientation

#[test]
fn the_potential_is_the_derivative_in_the_stated_orientation() {
    // dE = tr[V dn] with V_ab = dE/dn_ba. The energy is blind to n <-> n^T, V is not.
    let doc = freeze();
    let mut rng = Lcg(31);
    for (name, l) in [("d", 2usize), ("f", 3)] {
        let d = &doc[name]["dudarev"];
        let u_eff = doc[name]["U_eff"].as_f64().unwrap();
        let (hb, lay) = (hub(u_eff, 0.), layout(l, false));
        let n = json_matrix(&d["n"]);
        let stored_h = json_matrix(&d["direction"]);
        let random_h = hermitian_direction(lay.size(), &mut rng);
        // purely imaginary Hermitian direction: V^T and conj(V) flip its sign
        let a = DMatrix::from_fn(lay.size(), lay.size(), |_, _| C64::new(rng.next(), 0.));
        let imaginary = (&a - a.transpose()) * C64::new(0., 1.);
        for (label, h) in [
            ("stored", stored_h),
            ("random", random_h),
            ("imaginary", imaginary),
        ] {
            let v = potential(&eval(&hb, lay, &n));
            let fd = finite_difference(&hb, lay, &n, &h);
            let tr = tr_vh(&v, &h);
            assert!((fd - tr).abs() < 1e-9, "{name} {label}: fd {fd} tr {tr}");
            // The model's own first-order helper agrees with the test-local trace.
            let ev = eval(&hb, lay, &n);
            let first = ev.potential.first_order(&flat(&h)).unwrap();
            assert!((first - tr).abs() < 1e-13);
            // Negative controls must fail by far more than the agreement above.
            assert!(
                (fd - tr_vh(&v.transpose(), &h)).abs() > 1e-3,
                "{name} {label}: V^T"
            );
            assert!(
                (fd - tr_vh(&v.conjugate(), &h)).abs() > 1e-3,
                "{name} {label}: conj V"
            );
            assert!((fd - 1.01 * tr).abs() > 1e-4, "{name} {label}: 1.01 V");
        }
        let stored_fd = d["fd_richardson"].as_f64().unwrap();
        let stored_dir = json_matrix(&d["direction"]);
        let model_fd = finite_difference(&hb, lay, &n, &stored_dir);
        assert!(
            (model_fd - stored_fd).abs() < 1e-9,
            "stored Richardson anchor"
        );
    }
}

#[test]
fn a_pure_state_is_an_eigenvector_of_v_and_not_of_its_transpose() {
    // n_ij = c_i conj(c_j) = <i|rho|j> for rho = |c><c|; V = U/2 (1 - 2 |c><c|), so V c = -U/2 c.
    let mut rng = Lcg(37);
    for (l, real) in CASES {
        let lay = layout(l, real);
        let d = lay.size();
        let c: Vec<C64> = (0..d).map(|_| C64::new(rng.next(), rng.next())).collect();
        let norm = c.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        let c = nalgebra::DVector::from_iterator(d, c.into_iter().map(|x| x / norm));
        let n = DMatrix::from_fn(d, d, |i, j| c[i] * c[j].conj());
        let u = 3.0;
        let e = eval(&hub(u, 0.), lay, &n);
        let v = potential(&e);
        assert!(
            e.energy.abs() < 1e-13,
            "a pure state is idempotent: E_U = 0"
        );
        assert!((&v * &c + &c * C64::new(0.5 * u, 0.)).norm() < 1e-13);
        assert!((v.transpose() * &c + &c * C64::new(0.5 * u, 0.)).norm() > 1e-2);
        // The block constructor uses the same orientation: block (s, s') holds <a s|rho|b s'>.
        let p = lay.width();
        let block = |s: usize, t: usize| -> Vec<C64> {
            (0..p * p)
                .map(|k| n[(s * p + k / p, t * p + k % p)])
                .collect()
        };
        let (uu, ud, du, dd) = (block(0, 0), block(0, 1), block(1, 0), block(1, 1));
        let built = Occupation::from_blocks(&context(), lay, [&uu, &ud, &du, &dd]).unwrap();
        assert_eq!(built.as_slice(), &flat(&n)[..]);
        assert_eq!(built.entry(1, p + 2), n[(1, p + 2)]);
    }
}

// ---------------------------------------------------------------- the collinear model

#[test]
fn a_block_diagonal_occupation_reproduces_the_collinear_model() {
    let mut rng = Lcg(41);
    for (l, real) in CASES {
        let lay = layout(l, real);
        let p = lay.width();
        for (u, j) in [(3.7, 0.), (4.2, 0.9)] {
            let hb = hub(u, j);
            let (up, _, _) = random_state(p, &mut rng);
            let (dn, _, _) = random_state(p, &mut rng);
            let mut n = DMatrix::<C64>::zeros(2 * p, 2 * p);
            n.view_mut((0, 0), (p, p)).copy_from(&up);
            n.view_mut((p, p), (p, p)).copy_from(&dn);
            let e = eval(&hb, lay, &n);
            let (fu, fd) = (flat(&up), flat(&dn));
            let c = hubbard::evaluate(
                &context(),
                &hb,
                l,
                Spin::Unrestricted,
                &[fu.as_slice(), fd.as_slice()],
            )
            .unwrap();
            let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.);
            // The spinor sums the same two blocks in the same order and adds exact zeros.
            assert_eq!(e.energy.to_bits(), c.energy.to_bits());
            assert!(rel(e.interaction, c.interaction) < 1e-13);
            assert!(rel(e.double_counting, c.double_counting) < 1e-13);
            let v = potential(&e);
            for (s, block) in c.potentials.iter().enumerate() {
                for a in 0..p {
                    for b in 0..p {
                        // Bit for bit: the same expression on the same Hermitian part.
                        assert_eq!(v[(s * p + a, s * p + b)], block.as_slice()[a * p + b]);
                    }
                }
            }
            for a in 0..p {
                for b in 0..p {
                    assert_eq!(v[(a, p + b)], C64::default());
                    assert_eq!(v[(p + a, b)], C64::default());
                }
            }
            // A spin-flip block is not decoration: the collinear model cannot see it.
            let mut flipped = n.clone();
            flipped[(0, p)] = C64::new(0.1, 0.05);
            flipped[(p, 0)] = C64::new(0.1, -0.05);
            assert!((eval(&hb, lay, &flipped).energy - e.energy).abs() > 1e-4);
        }
    }
}

#[test]
fn the_double_counting_uses_the_spin_vector_and_is_rotation_invariant() {
    let mut rng = Lcg(43);
    let mut control_shift = 0f64;
    for (l, real) in CASES {
        let lay = layout(l, real);
        let p = lay.width();
        let (hb, (u, j)) = (hub(4.2, 0.9), (4.2, 0.9));
        let (n, _, _) = random_state(lay.size(), &mut rng);
        let e = eval(&hb, lay, &n);
        let big_n: f64 = (0..2 * p).map(|a| n[(a, a)].re).sum();
        let m = pauli(p, &n);
        let m2: f64 = m.iter().map(|x| x * x).sum();
        let want_dc =
            0.5 * u * big_n * (big_n - 1.) - 0.5 * j * (0.5 * (big_n * big_n + m2) - big_n);
        assert!((e.double_counting - want_dc).abs() < 1e-12);
        assert!((e.interaction - e.double_counting - e.energy).abs() < 1e-12);
        for (got, want) in e.spin_vector.iter().zip(m) {
            assert!((got - want).abs() < 1e-13);
        }
        let w = spin_rotation(p, &mut rng);
        let r = eval(&hb, lay, &(&w * &n * w.adjoint()));
        assert!((r.double_counting - e.double_counting).abs() < 1e-12);
        assert!((r.interaction - e.interaction).abs() < 1e-12);
        // Control: a z-only moment, the collinear reading, changes under the rotation.
        let mz = pauli(p, &(&w * &n * w.adjoint()))[2];
        control_shift = control_shift.max((mz * mz - m[2] * m[2]).abs());
    }
    assert!(
        control_shift > 1e-3,
        "m_z^2 must not be rotation invariant: {control_shift}"
    );
}

#[test]
fn a_bad_request_raises_and_nothing_is_kept() {
    let ctx = context();
    assert!(matches!(Layout::new(4, false), Err(Error::Invalid(_))));
    let lay = layout(2, false);
    let d = lay.size();
    let good = flat(&DMatrix::<C64>::identity(d, d));
    assert!(matches!(
        Occupation::new(&ctx, lay, &good[1..]),
        Err(Error::Invalid(_))
    ));
    let mut bad = good.clone();
    bad[3] = C64::new(f64::NAN, 0.);
    assert!(matches!(
        Occupation::new(&ctx, lay, &bad),
        Err(Error::Invalid(_))
    ));
    let mut skew = good.clone();
    skew[1] = C64::new(0.2, 0.1);
    skew[d] = C64::new(0.2, 0.1); // not the conjugate
    assert!(
        matches!(Occupation::new(&ctx, lay, &skew), Err(Error::Invalid(m)) if m.contains("Hermitian"))
    );
    let p = lay.width();
    let short = vec![C64::default(); p * p - 1];
    let ok = vec![C64::default(); p * p];
    let r = Occupation::from_blocks(&ctx, lay, [&ok, &ok, &ok, &short]);
    assert!(matches!(r, Err(Error::Invalid(_))));
    assert_eq!(ctx.memory_used(), 0, "a refused request keeps nothing");
    // Budget: the occupation fits, the potential does not, and the failure frees what it took.
    let small = Context::new(Some(1), d * d * 16 + 100).unwrap();
    let o = Occupation::new(&small, lay, &good).unwrap();
    let used = small.memory_used();
    assert!(matches!(
        spinor::evaluate(&small, &hub(2.0, 0.), &o),
        Err(Error::Budget { .. })
    ));
    assert_eq!(small.memory_used(), used);
    assert!(matches!(
        Occupation::new(&Context::new(Some(1), 64).unwrap(), lay, &good),
        Err(Error::Budget { .. })
    ));
}

#[test]
fn zero_u_is_the_identity_and_the_hellmann_feynman_factor_survives() {
    let mut rng = Lcg(67);
    let lay = layout(2, false);
    let (n, _, _) = random_state(lay.size(), &mut rng);
    let e = eval(&hub(2.0, 2.0), lay, &n);
    assert_eq!(e.energy, 0.);
    assert!(potential(&e).iter().all(|x| *x == C64::default()));
    let tr = (0..lay.size()).map(|a| n[(a, a)].re).sum::<f64>();
    let sq = (&n * &n).trace().re;
    assert!((e.per_u_eff - 0.5 * (tr - sq)).abs() < 1e-13);
    // dE/dU = E / U_eff at fixed n.
    let (a, b) = (eval(&hub(3.0, 0.), lay, &n), eval(&hub(3.001, 0.), lay, &n));
    assert!(((b.energy - a.energy) / 0.001 - a.per_u_eff).abs() < 1e-10);
    let _: &Potential = &a.potential;
}
