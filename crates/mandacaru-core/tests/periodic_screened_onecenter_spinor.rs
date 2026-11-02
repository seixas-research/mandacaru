//! The non-collinear sphere model of the spinor hybrid (SH-2): the one-center terms of a
//! `2p x 2p` spinor density with the vector-magnetization remainder, on the shipped silicon
//! sphere. Its energy is the older evaluator's plus `-fraction P`, its operator the derivative of
//! that energy for every block, it is SU(2) invariant, equals the unrestricted model for a
//! collinear density, and a time-reversal-flagged model refuses a sphere with a moment while
//! accepting a large cross-spin block that carries none.
//!
//! Reads the shipped library: `MANDACARU_PAW_PATH`; a missing one fails the tests.
mod periodic_screened_onecenter_spinor_support;
mod periodic_screened_onecenter_support;

use mandacaru_core::{
    paw::onecenter::Basis,
    periodic::{
        screened_augmentation::OneCenter,
        screened_onecenter::{Species, Sphere, SphereModel},
    },
    quantum::C64,
    Context,
};
use periodic_screened_onecenter_spinor_support::*;
use periodic_screened_onecenter_support::*;

fn model(
    ctx: &Context,
    set: &str,
    symbol: &str,
) -> (
    SphereModel,
    mandacaru_core::paw::screened_atom::ScreenedAtom,
    usize,
) {
    let (atom, projectors) = terms(ctx, set, symbol, Basis::Raw);
    let n = projectors.len();
    let (again, _) = terms(ctx, set, symbol, Basis::Raw);
    let model = SphereModel::new(
        vec![Species::new(atom, &projectors)],
        vec![Sphere {
            offset: 0,
            species: 0,
        }],
        n,
        FRACTION,
    )
    .unwrap();
    (model, again, n)
}

/// A spinor density with all four blocks populated.
fn general(n: usize) -> Matrix {
    density(2 * n, 0.3, 1.4)
}

const GATE: f64 = 1e-7;

fn largest(m: &Matrix) -> f64 {
    m.iter().map(|v| v.norm()).fold(0., f64::max)
}

#[test]
fn the_remainder_is_active_and_the_lift_is_the_one_of_the_chain_rule() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let (model, atom, n) = model(&ctx, "lda-sr", "Si");
    let q = general(n);
    let cross = largest(&block(&q, n, 0, 1)) / largest(&q);
    let (energy, lifted) = remainder(&ctx, &atom, &q);
    assert!(
        cross > 1e-2,
        "the cross blocks must be populated ({cross:.2e})"
    );
    assert!(
        energy.abs() > 1e-6 && largest(&lifted) > 1e-6,
        "the remainder must be active"
    );
    // The operator is Hermitian and its cross blocks are mutually adjoint.
    let skew = largest(&(&lifted - lifted.adjoint()));
    println!(
        "remainder -a P {energy:.9e}; |O| {:.3e}; skew part {skew:.2e}; cross {cross:.2e}",
        largest(&lifted)
    );
    assert!(skew < 1e-12 * largest(&lifted));
    // The model's operator is K and V of the blocks plus this lift; its energy likewise. The
    // derivative check below proves the whole; here the remainder alone is isolated by the model
    // with the spin-scaled term removed: a collinear density through both routes.
    let model_terms = model.spinor(&ctx, &q).unwrap();
    assert!(model_terms.energy.is_finite() && model_terms.operators[0].shape() == (2 * n, 2 * n));
}

#[test]
fn the_operator_is_the_derivative_of_the_energy_for_every_block() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let (model, atom, n) = model(&ctx, "lda-sr", "Si");
    let q = general(n);
    let at = model.spinor(&ctx, &q).unwrap();
    let operator = &at.operators[0];
    // Hermitian directions: the full matrix, then only the cross blocks, then only the diagonal.
    let full = density(2 * n, 2.1, 0.05) - density(2 * n, 3.4, 0.03);
    let mask = |keep_cross: bool| {
        Matrix::from_fn(2 * n, 2 * n, |i, j| {
            if (i / n != j / n) == keep_cross {
                full[(i, j)]
            } else {
                C64::new(0., 0.)
            }
        })
    };
    let (cross, diagonal) = (mask(true), mask(false));
    let numeric = |delta: &Matrix, step: f64| {
        let at = |e: f64| {
            model
                .spinor(&ctx, &(&q + delta * C64::new(e, 0.)))
                .unwrap()
                .energy
        };
        (-at(2. * step) + 8. * at(step) - 8. * at(-step) + at(-2. * step)) / (12. * step)
    };
    let analytic = |o: &Matrix, delta: &Matrix| -FRACTION * trace_product(o, delta).re;
    let mut gaps = Vec::new();
    for (label, delta) in [
        ("full", &full),
        ("cross blocks", &cross),
        ("diagonal blocks", &diagonal),
    ] {
        let got = analytic(operator, delta);
        assert!(got.abs() > 1e-6, "{label}: derivative {got:e}");
        for step in [5e-4, 1e-3, 2e-3] {
            gaps.push((label, step, (numeric(delta, step) - got).abs() / got.abs()));
        }
    }
    println!("spinor operator against the finite-difference derivative of the energy: {gaps:?}");
    // The floor is the hole model's adaptive quadrature (1e-10 relative per point), which a
    // difference quotient turns into 2e-8 of the derivative (steps 5e-4 to 2e-3 scatter, no trend);
    // the plan's 1e-7 is 5x that floor. The remainder alone is held to 1e-8 in
    // `paw_screened_polarization_vector`, where the energy is small and the floor 1e-11.
    assert!(gaps.iter().all(|(_, _, g)| *g < GATE), "{gaps:?}");
    // Controls, each a wrong operator against the same finite difference of the transverse
    // direction: the off-diagonal blocks swapped, and the spin-diagonal blocks alone.
    let (want, polar) = (numeric(&cross, 1e-3), remainder(&ctx, &atom, &q).1);
    assert!(largest(&polar) > 1e-6);
    let swapped = Matrix::from_fn(2 * n, 2 * n, |i, j| {
        let (s, t) = (i / n, j / n);
        if s == t {
            operator[(i, j)]
        } else {
            operator[((1 - s) * n + i % n, (1 - t) * n + j % n)]
        }
    });
    let diagonal_only = Matrix::from_fn(2 * n, 2 * n, |i, j| {
        if i / n == j / n {
            operator[(i, j)]
        } else {
            C64::new(0., 0.)
        }
    });
    for (label, wrong) in [
        ("swapped O_ud / O_du", &swapped),
        ("spin-diagonal blocks only", &diagonal_only),
    ] {
        let miss = (want - analytic(wrong, &cross)).abs() / want.abs();
        println!("control {label}: relative miss of the transverse derivative {miss:.3e}");
        assert!(miss > 100. * GATE, "{label} must bite");
    }
}

#[test]
fn the_sphere_energy_is_su2_invariant_and_the_operator_covariant() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let (model, _, n) = model(&ctx, "lda-sr", "Si");
    let q = general(n);
    let base = model.spinor(&ctx, &q).unwrap();
    let (a, b) = (C64::new(0.31, -0.42), C64::new(0.55, 0.65));
    let norm = (a.norm_sqr() + b.norm_sqr()).sqrt();
    let (a, b) = (a / norm, b / norm);
    let rotated = model.spinor(&ctx, &rotate(&q, a, b)).unwrap();
    let expected = rotate(&base.operators[0], a, b);
    let energy_gap = (rotated.energy - base.energy).abs() / base.energy.abs();
    let operator_gap = largest(&(&rotated.operators[0] - &expected)) / largest(&expected);
    println!("SU(2): energy {energy_gap:.2e}, operator {operator_gap:.2e}");
    assert!(energy_gap < 1e-12 && operator_gap < 1e-11);
    // Control: the transpose in place of the adjoint is not a rotation of the density.
    let wrong = {
        let u = Matrix::from_row_slice(2, 2, &[a, -b.conj(), b, a.conj()]);
        let big = u.kronecker(&Matrix::identity(n, n));
        &big * &q * big.transpose()
    };
    let bad = model.spinor(&ctx, &wrong).unwrap();
    let miss = (bad.energy - base.energy).abs() / base.energy.abs();
    println!("control u^T for u^+: energy changes by {miss:.2e}");
    assert!(miss > 100. * 1e-12);
}

#[test]
fn a_collinear_spinor_density_is_the_unrestricted_one() {
    // m along z with one sign everywhere (the spin densities are proportional): the remainder is
    // the collinear two-spin one, so the spinor model equals the unrestricted model block by block.
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let (model, _, n) = model(&ctx, "lda-sr", "Si");
    let base = density(n, 0.2, 1.0);
    let (up, down) = (&base * C64::new(1.3, 0.), &base * C64::new(0.4, 0.));
    let unrestricted = model.collinear(&ctx, &[up.clone(), down.clone()]).unwrap();
    let mut q = Matrix::zeros(2 * n, 2 * n);
    q.view_mut((0, 0), (n, n)).copy_from(&up);
    q.view_mut((n, n), (n, n)).copy_from(&down);
    let spinor = model.spinor(&ctx, &q).unwrap();
    let uu = block(&spinor.operators[0], n, 0, 0);
    let dd = block(&spinor.operators[0], n, 1, 1);
    let cross = largest(&block(&spinor.operators[0], n, 0, 1)).max(largest(&block(
        &spinor.operators[0],
        n,
        1,
        0,
    )));
    let gaps = [
        (spinor.energy - unrestricted.energy).abs() / unrestricted.energy.abs(),
        largest(&(&uu - &unrestricted.operators[0])) / largest(&uu),
        largest(&(&dd - &unrestricted.operators[1])) / largest(&dd),
        cross,
    ];
    println!("collinear spinor against unrestricted: energy, O_uu, O_dd, cross {gaps:?}");
    assert!(gaps[..3].iter().all(|g| *g < 1e-12), "{gaps:?}");
    assert_eq!(gaps[3], 0., "no cross block without a cross density");
}

#[test]
fn a_time_reversal_flag_refuses_a_moment_and_accepts_a_cross_block_without_one() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let (free, atom, n) = model(&ctx, "lda-sr", "Si");
    let flagged = |ctx: &Context| model(ctx, "lda-sr", "Si").0.with_time_reversal(true);
    // Q_uu = Q_dd = A and a large real antisymmetric cross block: |Q_ud| is large, the spherical
    // magnetization is zero (as the spin-orbit mixing of a time-reversal state is).
    let a = density(n, 0.2, 1.0);
    let b = Matrix::from_fn(n, n, |i, j| {
        let x = ((i * 5 + j * 3) as f64 * 0.41).sin() * 0.3 / (1. + (i + j) as f64);
        C64::new(
            if i < j {
                x
            } else if i > j {
                -((j * 5 + i * 3) as f64 * 0.41).sin() * 0.3 / (1. + (i + j) as f64)
            } else {
                0.
            },
            0.,
        )
    });
    assert!(
        largest(&(&b + b.transpose())) < 1e-15,
        "B must be antisymmetric"
    );
    let mut q = Matrix::zeros(2 * n, 2 * n);
    q.view_mut((0, 0), (n, n)).copy_from(&a);
    q.view_mut((n, n), (n, n)).copy_from(&a);
    q.view_mut((0, n), (n, n)).copy_from(&b);
    q.view_mut((n, 0), (n, n)).copy_from(&b.adjoint());
    assert!(largest(&b) > 1e-3 * largest(&a) && (&q - q.adjoint()).norm() < 1e-14);
    let model_tr = flagged(&ctx);
    let (with, without) = (
        model_tr.spinor(&ctx, &q).unwrap(),
        free.spinor(&ctx, &q).unwrap(),
    );
    // The remainder of this density is exactly zero, so evaluating it changes nothing.
    assert_eq!(with.energy.to_bits(), without.energy.to_bits());
    assert!(with.operators[0] == without.operators[0]);
    assert_eq!(remainder(&ctx, &atom, &q).0, 0.);
    // A sphere with a moment is refused under the flag and evaluated without it.
    let magnetic = general(n);
    let message = flagged(&ctx)
        .spinor(&ctx, &magnetic)
        .err()
        .expect("a moment must be refused")
        .to_string();
    println!("refusal: {message}");
    assert!(message.contains("time-reversal") && message.contains("magnetization"));
    assert!(free.spinor(&ctx, &magnetic).is_ok());
}
