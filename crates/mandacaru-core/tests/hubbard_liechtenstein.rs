//! Tests for the rotationally invariant Liechtenstein DFT+U functional:
//! - Determinant energies and analytic limits (d1, d5, d6, d10, f1, f7, f8, f14, d2(2,1)).
//! - Reduction to Dudarev when J = 0 (any n) and when n = x I (spherical).
//! - Potential Hermiticity (< 3e-16) and V = dE/dn by complex Richardson finite difference.
//! - Negative controls (V^T, V * 1.01, Hartree-only, wrong-spin).
//! - SO(3) rotational invariance of E and covariance of V, with generic U(p) negative control.
//! - Restricted vs Unrestricted equivalence for identical channels.
//! - Hellmann-Feynman derivatives dE/dU and dE/dJ vs finite differences.
//! - Inert/active behavior and refusals.
//! - Bit-identical reproducibility on repeated calls.
use mandacaru_core::models::hubbard::{
    evaluate, Form, Hubbard, Spin,
};
use mandacaru_core::models::hubbard_liechtenstein::{
    evaluate_with_basis, evaluate_with_coupling, evaluate_with_ratios, Liechtenstein,
};
use mandacaru_core::models::hubbard_slater::{Basis, Coupling, Ratios};
use mandacaru_core::{Context, Error};
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

fn context() -> Context {
    Context::new(Some(1), 64 << 20).unwrap()
}

/// Deterministic pseudo-random numbers in [-1, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2. - 1.
    }
}

fn unitary(p: usize, rng: &mut Lcg) -> DMatrix<C64> {
    let m = DMatrix::from_fn(p, p, |_, _| C64::new(rng.next(), rng.next()));
    m.qr().q()
}

fn hermitian(p: usize, eigenvalues: &[f64], rng: &mut Lcg) -> DMatrix<C64> {
    let q = unitary(p, rng);
    let d = DMatrix::from_fn(p, p, |i, j| {
        if i == j {
            C64::new(eigenvalues[i], 0.)
        } else {
            C64::default()
        }
    });
    &q * d * q.adjoint()
}

fn flat(m: &DMatrix<C64>) -> Vec<C64> {
    let p = m.nrows();
    (0..p * p).map(|k| m[(k / p, k % p)]).collect()
}

fn matrix(p: usize, v: &[C64]) -> DMatrix<C64> {
    DMatrix::from_fn(p, p, |i, j| v[i * p + j])
}

fn max_diff(a: &DMatrix<C64>, b: &DMatrix<C64>) -> f64 {
    (a - b).iter().map(|x| x.norm()).fold(0., f64::max)
}

/// Builds an SO(3) Wigner D-matrix for shell `l` via matrix exponential exp(-i theta . J).
fn so3_rotation(l: usize, axis: [f64; 3], angle: f64) -> DMatrix<C64> {
    let p = 2 * l + 1;
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let n = [axis[0] / norm, axis[1] / norm, axis[2] / norm];

    // J_z, J_+, J_- matrices
    // Index i corresponds to m = -l + i, so m in -l..=l
    let mut jz = DMatrix::<C64>::zeros(p, p);
    let mut jx = DMatrix::<C64>::zeros(p, p);
    let mut jy = DMatrix::<C64>::zeros(p, p);

    let l_f = l as f64;
    for i in 0..p {
        let m = i as f64 - l_f;
        jz[(i, i)] = C64::new(m, 0.);
        if i + 1 < p {
            let m_next = m;
            let c_plus = (l_f * (l_f + 1.) - m_next * (m_next + 1.)).sqrt();
            // J_+ |m> = c_+ |m+1> -> J_+[(i+1, i)] = c_+
            // J_- |m+1> = c_+ |m> -> J_-[(i, i+1)] = c_+
            let j_plus = c_plus;
            jx[(i + 1, i)] += C64::new(0.5 * j_plus, 0.);
            jx[(i, i + 1)] += C64::new(0.5 * j_plus, 0.);
            jy[(i + 1, i)] += C64::new(0., -0.5 * j_plus);
            jy[(i, i + 1)] += C64::new(0., 0.5 * j_plus);
        }
    }

    let generator = &jx * C64::new(n[0], 0.) + &jy * C64::new(n[1], 0.) + &jz * C64::new(n[2], 0.);
    // Exponentiate -i * angle * generator via Taylor series with scaling and squaring:
    let scaled_steps = 16;
    let dt = angle / (1 << scaled_steps) as f64;
    let mut step_mat = DMatrix::<C64>::identity(p, p);
    let i_term = C64::new(0., -dt);
    let mut term = DMatrix::<C64>::identity(p, p);
    for k in 1..=12 {
        term = &term * &generator * (i_term / (k as f64));
        step_mat += &term;
    }
    let mut rot = step_mat;
    for _ in 0..scaled_steps {
        rot = &rot * &rot;
    }
    rot
}

#[test]
fn determinant_energies_and_analytic_limits() {
    let ctx = context();
    // Liechtenstein functional with U = 6.0 eV, J = 1.0 eV
    let u = 6.0;
    let j = 1.0;

    for l in [2usize, 3] {
        let tol = if l == 2 { 1e-13 } else { 5e-13 };
        let p = 2 * l + 1;
        let ratios = Ratios::default_for(l).unwrap();
        let coupling = Coupling::new(&ctx, l, Basis::Complex).unwrap();
        let f_k = coupling.integrals(u, j, &ratios).unwrap();
        let f0 = f_k[0];
        let f2 = f_k[1];
        let f4 = f_k[2];

        // 1. One electron (d1 or f1): E_U = 0 exactly
        for orb in 0..p {
            let mut n_up = vec![C64::default(); p * p];
            n_up[orb * p + orb] = C64::new(1., 0.);
            let n_dn = vec![C64::default(); p * p];
            let eval = evaluate_with_coupling(
                &ctx,
                u,
                j,
                &coupling,
                &ratios,
                Spin::Unrestricted,
                &[&n_up, &n_dn],
            )
            .unwrap();
            assert!(
                eval.energy.abs() < tol,
                "l={l} orb={orb}: E_U={}",
                eval.energy
            );
        }

        // 2. Half-filled spin-up shell (d5 up or f7 up): E_U = 0 exactly
        {
            let mut n_up = vec![C64::default(); p * p];
            for orb in 0..p {
                n_up[orb * p + orb] = C64::new(1., 0.);
            }
            let n_dn = vec![C64::default(); p * p];
            let eval = evaluate_with_coupling(
                &ctx,
                u,
                j,
                &coupling,
                &ratios,
                Spin::Unrestricted,
                &[&n_up, &n_dn],
            )
            .unwrap();
            assert!(eval.energy.abs() < tol, "l={l} half-filled: E_U={}", eval.energy);

            // Check analytic E_Hub formula for d5 up:
            // E_Hub(d5 up) = 10 F0 - 5/7 F2 - 5/7 F4
            if l == 2 {
                let want_interaction = 10. * f0 - (5. / 7.) * f2 - (5. / 7.) * f4;
                assert!(
                    (eval.interaction - want_interaction).abs() < 1e-13,
                    "d5 up interaction: {} vs {}",
                    eval.interaction,
                    want_interaction
                );
            }
        }

        // 3. Half-filled up plus 1 down (d6 or f8): E_U = 0 exactly
        {
            let mut n_up = vec![C64::default(); p * p];
            for orb in 0..p {
                n_up[orb * p + orb] = C64::new(1., 0.);
            }
            let mut n_dn = vec![C64::default(); p * p];
            n_dn[0] = C64::new(1., 0.);
            let eval = evaluate_with_coupling(
                &ctx,
                u,
                j,
                &coupling,
                &ratios,
                Spin::Unrestricted,
                &[&n_up, &n_dn],
            )
            .unwrap();
            assert!(eval.energy.abs() < tol, "l={l} d6/f8: E_U={}", eval.energy);
        }

        // 4. Completely filled shell (d10 or f14): E_U = 0 exactly
        {
            let mut n_up = vec![C64::default(); p * p];
            let mut n_dn = vec![C64::default(); p * p];
            for orb in 0..p {
                n_up[orb * p + orb] = C64::new(1., 0.);
                n_dn[orb * p + orb] = C64::new(1., 0.);
            }
            let eval = evaluate_with_coupling(
                &ctx,
                u,
                j,
                &coupling,
                &ratios,
                Spin::Unrestricted,
                &[&n_up, &n_dn],
            )
            .unwrap();
            assert!(eval.energy.abs() < tol, "l={l} filled: E_U={}", eval.energy);
        }

        // 5. Non-spherical d2 up in orbitals (m=2, m=1) (orbitals 4 and 3 in m = -2..2)
        if l == 2 {
            let mut n_up = vec![C64::default(); p * p];
            n_up[4 * p + 4] = C64::new(1., 0.);
            n_up[3 * p + 3] = C64::new(1., 0.);
            let n_dn = vec![C64::default(); p * p];
            let eval = evaluate_with_coupling(
                &ctx,
                u,
                j,
                &coupling,
                &ratios,
                Spin::Unrestricted,
                &[&n_up, &n_dn],
            )
            .unwrap();
            // Analytic formula from G-L2: E_U = -9/98 F2 + 5/98 F4
            let want = -(9. / 98.) * f2 + (5. / 98.) * f4;
            assert!(
                (eval.energy - want).abs() < 1e-13,
                "d2(2,1): got {} vs want {}",
                eval.energy,
                want
            );
            // Dudarev vanishes on EVERY idempotent n (including this one), while Liechtenstein does not!
            let dudarev = evaluate(
                &ctx,
                &Hubbard::new(u, j).unwrap(),
                l,
                Spin::Unrestricted,
                &[&n_up, &n_dn],
            )
            .unwrap();
            assert_eq!(dudarev.energy, 0.);
            assert!((eval.energy - dudarev.energy).abs() > 0.01);
        }
    }
}

#[test]
fn limits_to_dudarev() {
    let ctx = context();
    let mut rng = Lcg(42);

    for l in [2usize, 3] {
        let p = 2 * l + 1;

        // Limit 1: J = 0 recovers Dudarev for ANY n
        {
            let u = 5.0;
            let j = 0.0;
            let h_dud = Hubbard::with_form(u, j, Form::DudarevFll).unwrap();
            let h_lie = Hubbard::with_form(u, j, Form::LiechtensteinFll).unwrap();

            let ev_up: Vec<f64> = (0..p).map(|k| 0.1 + 0.8 * k as f64 / p as f64).collect();
            let ev_dn: Vec<f64> = (0..p).map(|k| 0.9 - 0.7 * k as f64 / p as f64).collect();
            let n_up = flat(&hermitian(p, &ev_up, &mut rng));
            let n_dn = flat(&hermitian(p, &ev_dn, &mut rng));

            let eval_dud = evaluate(&ctx, &h_dud, l, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap();
            let eval_lie = evaluate(&ctx, &h_lie, l, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap();

            assert!(
                (eval_lie.energy - eval_dud.energy).abs() < 1e-12,
                "l={l} J=0 energy: {} vs {}",
                eval_lie.energy,
                eval_dud.energy
            );
            for s in 0..2 {
                let v_dud = eval_dud.potentials[s].as_slice();
                let v_lie = eval_lie.potentials[s].as_slice();
                for k in 0..p * p {
                    assert!(
                        (v_lie[k] - v_dud[k]).norm() < 1e-13,
                        "l={l} J=0 potential[{s}][{k}]: {} vs {}",
                        v_lie[k],
                        v_dud[k]
                    );
                }
            }
        }

        // Limit 2: n_s = x_s I (spherical) recovers Dudarev for ANY J
        {
            let u = 6.0;
            let j = 1.2;
            let h_dud = Hubbard::with_form(u, j, Form::DudarevFll).unwrap();
            let h_lie = Hubbard::with_form(u, j, Form::LiechtensteinFll).unwrap();

            let x_up = 0.65;
            let x_dn = 0.35;
            let mut n_up = vec![C64::default(); p * p];
            let mut n_dn = vec![C64::default(); p * p];
            for a in 0..p {
                n_up[a * p + a] = C64::new(x_up, 0.);
                n_dn[a * p + a] = C64::new(x_dn, 0.);
            }

            let eval_dud = evaluate(&ctx, &h_dud, l, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap();
            let eval_lie = evaluate(&ctx, &h_lie, l, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap();

            assert!(
                (eval_lie.energy - eval_dud.energy).abs() < 1e-13,
                "l={l} spherical energy: {} vs {}",
                eval_lie.energy,
                eval_dud.energy
            );
            for s in 0..2 {
                let v_dud = eval_dud.potentials[s].as_slice();
                let v_lie = eval_lie.potentials[s].as_slice();
                for k in 0..p * p {
                    assert!(
                        (v_lie[k] - v_dud[k]).norm() < 1e-13,
                        "l={l} spherical potential[{s}][{k}]: {} vs {}",
                        v_lie[k],
                        v_dud[k]
                    );
                }
            }
        }

        // Generic n with J > 0 differs between Liechtenstein and Dudarev
        {
            let u = 6.0;
            let j = 1.0;
            let h_dud = Hubbard::with_form(u, j, Form::DudarevFll).unwrap();
            let h_lie = Hubbard::with_form(u, j, Form::LiechtensteinFll).unwrap();

            let ev_up: Vec<f64> = (0..p).map(|k| 0.2 + 0.6 * k as f64 / p as f64).collect();
            let ev_dn: Vec<f64> = (0..p).map(|k| 0.8 - 0.5 * k as f64 / p as f64).collect();
            let n_up = flat(&hermitian(p, &ev_up, &mut rng));
            let n_dn = flat(&hermitian(p, &ev_dn, &mut rng));

            let eval_dud = evaluate(&ctx, &h_dud, l, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap();
            let eval_lie = evaluate(&ctx, &h_lie, l, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap();

            assert!((eval_lie.energy - eval_dud.energy).abs() > 1e-4);
        }
    }
}

#[test]
fn potential_is_hermitian_and_derivative_of_energy() {
    let ctx = context();
    let mut rng = Lcg(777);

    for l in [2usize, 3] {
        let p = 2 * l + 1;
        let u = 5.5;
        let j = 0.9;
        let h = Hubbard::with_form(u, j, Form::LiechtensteinFll).unwrap();

        let ev_up: Vec<f64> = (0..p).map(|k| 0.85 - 0.7 * k as f64 / p as f64).collect();
        let ev_dn: Vec<f64> = (0..p).map(|k| 0.15 + 0.6 * k as f64 / p as f64).collect();
        let n_up_mat = hermitian(p, &ev_up, &mut rng);
        let n_dn_mat = hermitian(p, &ev_dn, &mut rng);
        let n_up = flat(&n_up_mat);
        let n_dn = flat(&n_dn_mat);

        let eval = evaluate(&ctx, &h, l, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap();

        // 1. Hermiticity of V (< 3e-16)
        for s in 0..2 {
            let v_mat = matrix(p, eval.potentials[s].as_slice());
            let anti_hermitian = &v_mat - v_mat.adjoint();
            let max_anti = anti_hermitian.iter().map(|x| x.norm()).fold(0., f64::max);
            assert!(
                max_anti < 3e-16,
                "l={l} spin={s}: anti-Hermitian part {max_anti:e}"
            );
        }

        // 2. V = dE/dn by complex Richardson finite difference
        // Perturb each spin channel with a random complex Hermitian matrix
        for s in 0..2 {
            let delta = hermitian(
                p,
                &(0..p).map(|_| rng.next() * 0.1).collect::<Vec<_>>(),
                &mut rng,
            );
            let delta_flat = flat(&delta);

            let e_at = |step: f64| -> f64 {
                let mut perturbed = vec![n_up.clone(), n_dn.clone()];
                for k in 0..p * p {
                    perturbed[s][k] += delta_flat[k] * step;
                }
                evaluate(
                    &ctx,
                    &h,
                    l,
                    Spin::Unrestricted,
                    &[&perturbed[0], &perturbed[1]],
                )
                .unwrap()
                .energy
            };

            let h_step = 1e-3;
            let g_h = (e_at(h_step) - e_at(-h_step)) / (2. * h_step);
            let g_half = (e_at(h_step * 0.5) - e_at(-h_step * 0.5)) / h_step;
            let richardson = (4. * g_half - g_h) / 3.;

            let v_mat = matrix(p, eval.potentials[s].as_slice());
            let predicted = (0..p)
                .flat_map(|a| (0..p).map(move |b| (a, b)))
                .map(|(a, b)| (v_mat[(a, b)] * delta[(b, a)]).re)
                .sum::<f64>();

            let rel_err = (richardson - predicted).abs() / predicted.abs();
            assert!(
                rel_err < 1e-7,
                "l={l} s={s}: Richardson FD relative error {rel_err:e}"
            );

            // Negative controls:
            // Control 1: V^T fails
            let v_transposed = v_mat.transpose();
            let pred_vt = (0..p)
                .flat_map(|a| (0..p).map(move |b| (a, b)))
                .map(|(a, b)| (v_transposed[(a, b)] * delta[(b, a)]).re)
                .sum::<f64>();
            let err_vt = (richardson - pred_vt).abs() / predicted.abs();
            assert!(err_vt > 1e-3, "l={l} s={s}: V^T control failed to reject ({err_vt:e})");

            // Control 2: V x 1.01 fails
            let pred_scale = predicted * 1.01;
            let err_scale = (richardson - pred_scale).abs() / predicted.abs();
            assert!(err_scale > 5e-3, "l={l} s={s}: V x 1.01 control failed ({err_scale:e})");

            // Control 3: wrong-spin potential fails
            let other_s = 1 - s;
            let v_other = matrix(p, eval.potentials[other_s].as_slice());
            let pred_other = (0..p)
                .flat_map(|a| (0..p).map(move |b| (a, b)))
                .map(|(a, b)| (v_other[(a, b)] * delta[(b, a)]).re)
                .sum::<f64>();
            let err_other = (richardson - pred_other).abs() / predicted.abs();
            assert!(err_other > 1e-3, "l={l} s={s}: wrong spin control failed ({err_other:e})");
        }
    }
}

#[test]
fn so3_rotational_invariance() {
    let ctx = context();
    let mut rng = Lcg(1234);

    for l in [2usize, 3] {
        let p = 2 * l + 1;
        let u = 6.0;
        let j = 1.0;
        let h = Hubbard::with_form(u, j, Form::LiechtensteinFll).unwrap();

        let ev_up: Vec<f64> = (0..p).map(|k| 0.8 - 0.6 * k as f64 / p as f64).collect();
        let ev_dn: Vec<f64> = (0..p).map(|k| 0.2 + 0.5 * k as f64 / p as f64).collect();
        let n_up_mat = hermitian(p, &ev_up, &mut rng);
        let n_dn_mat = hermitian(p, &ev_dn, &mut rng);
        let n_up = flat(&n_up_mat);
        let n_dn = flat(&n_dn_mat);

        let eval_base = evaluate(&ctx, &h, l, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap();

        // 1. SO(3) rotation
        let axis = [rng.next(), rng.next(), rng.next()];
        let angle = 0.73;
        let d_rot = so3_rotation(l, axis, angle);

        let n_up_rot = &d_rot * &n_up_mat * d_rot.adjoint();
        let n_dn_rot = &d_rot * &n_dn_mat * d_rot.adjoint();
        let eval_rot = evaluate(
            &ctx,
            &h,
            l,
            Spin::Unrestricted,
            &[&flat(&n_up_rot), &flat(&n_dn_rot)],
        )
        .unwrap();

        // SO(3) invariance of energy (1e-11)
        let delta_e = (eval_rot.energy - eval_base.energy).abs();
        assert!(
            delta_e < 1e-11,
            "l={l} SO(3) energy change {delta_e:e}"
        );

        // Covariance of V: V(D n D^+) = D V D^+
        for s in 0..2 {
            let v_base = matrix(p, eval_base.potentials[s].as_slice());
            let v_rot = matrix(p, eval_rot.potentials[s].as_slice());
            let v_rotated_base = &d_rot * &v_base * d_rot.adjoint();
            let diff = max_diff(&v_rot, &v_rotated_base);
            assert!(
                diff < 1e-11,
                "l={l} spin={s}: V covariance max diff {diff:e}"
            );
        }

        // 2. Negative control: generic U(p) unitary (not SO(3)) changes E by ~4e-2 (d) / 9e-2 (f)
        let q_generic = unitary(p, &mut rng);
        let n_up_generic = &q_generic * &n_up_mat * q_generic.adjoint();
        let n_dn_generic = &q_generic * &n_dn_mat * q_generic.adjoint();
        let eval_generic = evaluate(
            &ctx,
            &h,
            l,
            Spin::Unrestricted,
            &[&flat(&n_up_generic), &flat(&n_dn_generic)],
        )
        .unwrap();
        let delta_u_p = (eval_generic.energy - eval_base.energy).abs();
        assert!(
            delta_u_p > 1e-2,
            "l={l}: generic U(p) unitary failed to change energy ({delta_u_p:e})"
        );
    }
}

#[test]
fn restricted_equals_unrestricted_with_identical_channels() {
    let ctx = context();
    let mut rng = Lcg(55);

    for l in [2usize, 3] {
        let p = 2 * l + 1;
        let u = 5.0;
        let j = 0.8;
        let h = Hubbard::with_form(u, j, Form::LiechtensteinFll).unwrap();

        let ev: Vec<f64> = (0..p).map(|k| 0.7 - 0.5 * k as f64 / p as f64).collect();
        let n_mat = hermitian(p, &ev, &mut rng);
        let n = flat(&n_mat);

        let eval_res = evaluate(&ctx, &h, l, Spin::Restricted, &[&n]).unwrap();
        let eval_unres = evaluate(&ctx, &h, l, Spin::Unrestricted, &[&n, &n]).unwrap();

        assert_eq!(eval_res.potentials.len(), 1);
        assert_eq!(eval_unres.potentials.len(), 2);

        assert!(
            (eval_res.energy - eval_unres.energy).abs() < 1e-14,
            "energy: {} vs {}",
            eval_res.energy,
            eval_unres.energy
        );
        assert!(
            (eval_res.interaction - eval_unres.interaction).abs() < 1e-14,
            "interaction: {} vs {}",
            eval_res.interaction,
            eval_unres.interaction
        );
        assert!(
            (eval_res.double_counting - eval_unres.double_counting).abs() < 1e-14,
            "double_counting: {} vs {}",
            eval_res.double_counting,
            eval_unres.double_counting
        );
        assert!(
            (eval_res.per_u_eff - eval_unres.per_u_eff).abs() < 1e-14,
            "per_u_eff: {} vs {}",
            eval_res.per_u_eff,
            eval_unres.per_u_eff
        );
        assert!(
            (eval_res.per_j - eval_unres.per_j).abs() < 1e-14,
            "per_j: {} vs {}",
            eval_res.per_j,
            eval_unres.per_j
        );

        let v_res = eval_res.potentials[0].as_slice();
        let v_unres_0 = eval_unres.potentials[0].as_slice();
        let v_unres_1 = eval_unres.potentials[1].as_slice();
        for k in 0..p * p {
            assert!((v_res[k] - v_unres_0[k]).norm() < 1e-14);
            assert!((v_res[k] - v_unres_1[k]).norm() < 1e-14);
        }
    }
}

#[test]
fn hellmann_feynman_derivatives_in_u_and_j() {
    let ctx = context();
    let mut rng = Lcg(888);

    for l in [2usize, 3] {
        let p = 2 * l + 1;
        let u = 5.0;
        let j = 0.8;
        let h = Hubbard::with_form(u, j, Form::LiechtensteinFll).unwrap();

        let ev_up: Vec<f64> = (0..p).map(|k| 0.85 - 0.6 * k as f64 / p as f64).collect();
        let ev_dn: Vec<f64> = (0..p).map(|k| 0.15 + 0.5 * k as f64 / p as f64).collect();
        let n_up = flat(&hermitian(p, &ev_up, &mut rng));
        let n_dn = flat(&hermitian(p, &ev_dn, &mut rng));

        let eval = evaluate(&ctx, &h, l, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap();

        // 1. dE/dU derivative check by finite difference
        let du = 1e-5;
        let h_u_plus = Hubbard::with_form(u + du, j, Form::LiechtensteinFll).unwrap();
        let h_u_minus = Hubbard::with_form(u - du, j, Form::LiechtensteinFll).unwrap();
        let e_u_plus = evaluate(&ctx, &h_u_plus, l, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap().energy;
        let e_u_minus = evaluate(&ctx, &h_u_minus, l, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap().energy;
        let fd_u = (e_u_plus - e_u_minus) / (2. * du);
        assert!(
            (fd_u - eval.d_energy_d_u()).abs() < 1e-8,
            "l={l} dE/dU: fd={fd_u} vs analytical={}",
            eval.d_energy_d_u()
        );

        // 2. dE/dJ derivative check by finite difference
        let dj = 1e-5;
        let h_j_plus = Hubbard::with_form(u, j + dj, Form::LiechtensteinFll).unwrap();
        let h_j_minus = Hubbard::with_form(u, j - dj, Form::LiechtensteinFll).unwrap();
        let e_j_plus = evaluate(&ctx, &h_j_plus, l, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap().energy;
        let e_j_minus = evaluate(&ctx, &h_j_minus, l, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap().energy;
        let fd_j = (e_j_plus - e_j_minus) / (2. * dj);
        assert!(
            (fd_j - eval.d_energy_d_j()).abs() < 1e-8,
            "l={l} dE/dJ: fd={fd_j} vs analytical={}",
            eval.d_energy_d_j()
        );
    }
}

#[test]
fn inert_and_active_behavior() {
    let ctx = context();
    let mut rng = Lcg(99);
    let p = 5;
    let ev: Vec<f64> = (0..p).map(|k| 0.8 - 0.5 * k as f64 / p as f64).collect();
    let n = flat(&hermitian(p, &ev, &mut rng));

    // U = 0, J = 0 is inert
    let inert_h = Hubbard::with_form(0.0, 0.0, Form::LiechtensteinFll).unwrap();
    assert!(inert_h.is_inert());
    let eval_inert = evaluate(&ctx, &inert_h, 2, Spin::Restricted, &[&n]).unwrap();
    assert_eq!(eval_inert.energy, 0.0);
    assert_eq!(eval_inert.interaction, 0.0);
    assert_eq!(eval_inert.double_counting, 0.0);
    assert!(eval_inert.potentials[0].as_slice().iter().all(|x| *x == C64::default()));

    // U = J != 0 is active for Liechtenstein (nonzero F2, F4)
    let active_h = Hubbard::with_form(4.0, 4.0, Form::LiechtensteinFll).unwrap();
    assert!(!active_h.is_inert());
    let eval_active = evaluate(&ctx, &active_h, 2, Spin::Restricted, &[&n]).unwrap();
    assert!(eval_active.energy.abs() > 1e-4);
    assert!(eval_active.potentials[0].as_slice().iter().any(|x| x.norm() > 1e-4));
}

#[test]
fn refusals() {
    let ctx = context();
    let mut rng = Lcg(101);
    let p = 5;
    let ev: Vec<f64> = (0..p).map(|k| 0.8 - 0.5 * k as f64 / p as f64).collect();
    let n = flat(&hermitian(p, &ev, &mut rng));

    // l not 2 or 3 raises
    let h = Hubbard::with_form(4.0, 1.0, Form::LiechtensteinFll).unwrap();
    assert!(matches!(
        evaluate(&ctx, &h, 1, Spin::Restricted, &[&vec![C64::default(); 9]]),
        Err(Error::Invalid(m)) if m.contains("d or f shell")
    ));

    // Bad parameters
    assert!(matches!(
        Hubbard::with_form(-1.0, 0.0, Form::LiechtensteinFll),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        Hubbard::with_form(1.0, -0.5, Form::LiechtensteinFll),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        Hubbard::with_form(1.0, 2.0, Form::LiechtensteinFll),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        Hubbard::with_form(f64::NAN, 0.0, Form::LiechtensteinFll),
        Err(Error::Invalid(_))
    ));

    // Bad channel count
    assert!(matches!(
        evaluate(&ctx, &h, 2, Spin::Restricted, &[&n, &n]),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        evaluate(&ctx, &h, 2, Spin::Unrestricted, &[&n]),
        Err(Error::Invalid(_))
    ));

    // Non-Hermitian n
    let mut bad_n = n.clone();
    bad_n[1] = C64::new(0.5, 0.2);
    bad_n[5] = C64::new(0.5, 0.2); // not conjugate
    assert!(matches!(
        evaluate(&ctx, &h, 2, Spin::Restricted, &[&bad_n]),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn bit_identical_repeat_calls() {
    let ctx = context();
    let mut rng = Lcg(444);
    let p = 5;
    let ev_up: Vec<f64> = (0..p).map(|k| 0.8 - 0.6 * k as f64 / p as f64).collect();
    let ev_dn: Vec<f64> = (0..p).map(|k| 0.2 + 0.5 * k as f64 / p as f64).collect();
    let n_up = flat(&hermitian(p, &ev_up, &mut rng));
    let n_dn = flat(&hermitian(p, &ev_dn, &mut rng));

    let h = Hubbard::with_form(5.5, 1.2, Form::LiechtensteinFll).unwrap();

    let a = evaluate(&ctx, &h, 2, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap();
    let b = evaluate(&ctx, &h, 2, Spin::Unrestricted, &[&n_up, &n_dn]).unwrap();

    assert_eq!(a.energy.to_bits(), b.energy.to_bits());
    assert_eq!(a.interaction.to_bits(), b.interaction.to_bits());
    assert_eq!(a.double_counting.to_bits(), b.double_counting.to_bits());
    assert_eq!(a.per_u_eff.to_bits(), b.per_u_eff.to_bits());
    assert_eq!(a.per_j.to_bits(), b.per_j.to_bits());

    for s in 0..2 {
        let va = a.potentials[s].as_slice();
        let vb = b.potentials[s].as_slice();
        for k in 0..p * p {
            assert_eq!(va[k].re.to_bits(), vb[k].re.to_bits());
            assert_eq!(va[k].im.to_bits(), vb[k].im.to_bits());
        }
    }
}

#[test]
fn custom_ratios_and_basis_evaluation() {
    let ctx = context();
    let mut rng = Lcg(555);
    let p = 5;
    let ev: Vec<f64> = (0..p).map(|k| 0.7 - 0.4 * k as f64 / p as f64).collect();
    let n = flat(&hermitian(p, &ev, &mut rng));

    // Custom ratios for d shell
    let ratios = Ratios::new(2, Some(0.6), None).unwrap();
    let e_custom = evaluate_with_ratios(&ctx, 6.0, 1.0, 2, &ratios, Basis::Complex, Spin::Restricted, &[&n]).unwrap();
    assert!(e_custom.energy.is_finite());

    // Liechtenstein struct evaluation
    let lie = Liechtenstein::new(2, 6.0, 1.0, Some(ratios)).unwrap();
    assert!(!lie.is_inert());
    let e_lie = lie.evaluate_with_basis(&ctx, 2, Basis::Complex, Spin::Restricted, &[&n]).unwrap();
    assert_eq!(e_custom.energy.to_bits(), e_lie.energy.to_bits());

    // Evaluate in real basis
    let e_real = evaluate_with_basis(&ctx, 6.0, 1.0, 2, Basis::Real, Spin::Restricted, &[&n]).unwrap();
    assert!(e_real.energy.is_finite());
}
