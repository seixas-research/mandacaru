//! `models::hubbard_spinor`: the time-reversal operator Theta and the completion of an
//! occupation matrix to its time-reversal invariant part.
mod hubbard_spinor_support;

use hubbard_spinor_support::*;
use mandacaru_core::models::hubbard_spinor::{self as spinor};
use mandacaru_core::Error;
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

#[test]
fn the_time_reversal_operator_squares_to_minus_one_and_commutes_with_l_dot_s() {
    for l in 1..=3usize {
        for real in [false, true] {
            let lay = layout(l, real);
            let u = theta_matrix(lay);
            let d = lay.size();
            let id = DMatrix::<C64>::identity(d, d);
            assert!(max_abs(&(&u * u.adjoint() - &id)) < 1e-15, "unitary");
            // Theta = U K: Theta^2 = U conj(U).
            assert!(max_abs(&(&u * u.conjugate() + &id)) < 1e-15, "Theta^2 = -1");
            let ls = ls_matrix(l, real);
            let transformed = &u * ls.conjugate() * u.adjoint();
            assert!(max_abs(&(transformed - &ls)) < 1e-14, "l={l} real={real}");
            // Control: drop the orbital (-1)^m in the complex basis, the spin sign in the real
            // one. Each is the wrong antiunitary and must break the L.S symmetry.
            let p = lay.width();
            let wrong = DMatrix::from_fn(d, d, |i, j| {
                let kept = if real {
                    1.
                } else if j / p == 0 {
                    1.
                } else {
                    -1.
                };
                C64::new(kept * u[(i, j)].norm(), 0.)
            });
            let bad = &wrong * ls.conjugate() * wrong.adjoint();
            assert!(max_abs(&(bad - &ls)) > 0.1, "control l={l} real={real}");
        }
    }
}

#[test]
fn the_energy_is_time_reversal_invariant() {
    let mut rng = Lcg(53);
    for (l, real) in CASES {
        let lay = layout(l, real);
        let (n, _, _) = random_state(lay.size(), &mut rng);
        let o = occ(lay, &n);
        let image = spinor::time_reversed(&context(), &o).unwrap();
        let hb = hub(3.7, 0.);
        let a = spinor::evaluate(&context(), &hb, &o).unwrap();
        let b = spinor::evaluate(&context(), &hb, &image).unwrap();
        assert!((a.energy - b.energy).abs() < 1e-13);
        // Theta n Theta^-1 = U conj(n) U^dagger, and V transforms the same way.
        let u = theta_matrix(lay);
        assert!(
            max_abs(&(dm(lay.size(), image.as_slice()) - &u * n.conjugate() * u.adjoint())) < 1e-15
        );
        let tv = &u * potential(&a).conjugate() * u.adjoint();
        assert!(max_abs(&(potential(&b) - tv)) < 1e-13);
    }
}

#[test]
fn the_completion_is_idempotent_trace_preserving_and_kills_the_spin_vector() {
    let mut rng = Lcg(59);
    for (l, real) in CASES {
        let lay = layout(l, real);
        let ctx = context();
        let (n, _, _) = random_state(lay.size(), &mut rng);
        let o = occ(lay, &n);
        assert!(
            spinor::time_reversal_defect(&o) > 0.05,
            "control: a random n breaks it"
        );
        let c = spinor::complete(&ctx, &o).unwrap();
        let again = spinor::complete(&ctx, &c).unwrap();
        assert!(c
            .as_slice()
            .iter()
            .zip(again.as_slice())
            .all(|(x, y)| (x - y).norm() < 1e-15));
        assert!((c.trace() - o.trace()).abs() < 1e-14);
        assert!(c.spin_vector().iter().all(|x| x.abs() < 1e-14));
        assert!(spinor::time_reversal_defect(&c) < 1e-15);
        spinor::require_time_reversal_invariant(&c, 1e-12).unwrap();
        let msg = spinor::require_time_reversal_invariant(&o, 1e-12).unwrap_err();
        assert!(
            matches!(&msg, Error::Invalid(m) if m.contains("time-reversal")),
            "{msg}"
        );
        // The potential of a completed n is itself time-reversal invariant; an uncompleted one is not.
        let u = theta_matrix(lay);
        let hb = hub(3.7, 0.);
        let vc = potential(&spinor::evaluate(&ctx, &hb, &c).unwrap());
        assert!(max_abs(&(&vc - &u * vc.conjugate() * u.adjoint())) < 1e-14);
        let vo = potential(&spinor::evaluate(&ctx, &hb, &o).unwrap());
        assert!(max_abs(&(&vo - &u * vo.conjugate() * u.adjoint())) > 0.05);
    }
}

#[test]
fn a_kramers_pair_is_already_complete() {
    let mut rng = Lcg(61);
    for (l, real) in CASES {
        let lay = layout(l, real);
        let (d, ctx) = (lay.size(), context());
        let u = theta_matrix(lay);
        let c = nalgebra::DVector::from_fn(d, |_, _| C64::new(rng.next(), rng.next()));
        let c = &c / C64::new(c.norm(), 0.);
        let tc = &u * c.conjugate();
        let n = DMatrix::from_fn(d, d, |i, j| c[i] * c[j].conj() + tc[i] * tc[j].conj());
        let o = occ(lay, &n);
        assert!(spinor::time_reversal_defect(&o) < 1e-15);
        let done = spinor::complete(&ctx, &o).unwrap();
        assert!(done
            .as_slice()
            .iter()
            .zip(o.as_slice())
            .all(|(x, y)| (x - y).norm() < 1e-15));
        // Theta c is orthogonal to c (Theta^2 = -1), so the pair is a rank-2 projector: E_U = 0.
        let e = spinor::evaluate(&ctx, &hub(2.0, 0.), &o).unwrap();
        assert!(e.energy.abs() < 1e-13);
    }
}
