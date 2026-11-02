//! The one-center screened terms of the periodic hybrid (W-E4P / P2) as a model, without a
//! crystal: the complex harmonics as a rotation of the real ones, the species tables of the
//! record against the pinned oxygen terms of the molecular code, the energy and the operators
//! of `SphereModel` against the older `onecenter_hybrid::Hybrid` evaluator fed with the rotated
//! tensor, the operators as the derivative of the energy, and the controls that a wrong
//! convention fails these gates.
//!
//! Reads the shipped library: `MANDACARU_PAW_PATH`; a missing one fails the tests.
mod periodic_screened_onecenter_spinor_support;
mod periodic_screened_onecenter_support;

use mandacaru_core::{
    math::angular,
    paw::{
        onecenter::Basis,
        onecenter_hybrid::{Hybrid, HybridAtom},
    },
    periodic::{
        screened_augmentation::OneCenter,
        screened_onecenter::{harmonic_rotation, Species, Sphere, SphereModel},
    },
    quantum::C64,
    Context,
};
use periodic_screened_onecenter_spinor_support::remainder;
use periodic_screened_onecenter_support::*;

#[test]
fn the_complex_harmonics_are_a_unitary_rotation_of_the_real_ones() {
    let directions: [[f64; 3]; 4] = [
        [0.3, -0.5, 0.81],
        [-0.7, 0.2, -0.4],
        [0.1, 0.9, 0.35],
        [0.6, 0.6, -0.2],
    ];
    let mut worst = 0f64;
    for l in 0..=4usize {
        let s = harmonic_rotation(l);
        worst = worst.max((&s * s.adjoint() - Matrix::identity(2 * l + 1, 2 * l + 1)).norm());
        for d in directions {
            let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let d = [d[0] / norm, d[1] / norm, d[2] / norm];
            for m in -(l as i32)..=l as i32 {
                let mixed: C64 = (-(l as i32)..=l as i32)
                    .map(|k| {
                        s[((m + l as i32) as usize, (k + l as i32) as usize)]
                            * angular::cartesian(l, k, d)
                    })
                    .sum();
                worst = worst.max((mixed - angular::complex(l, m, d)).norm());
            }
        }
    }
    println!("complex harmonics against the rotated real ones, l <= 4: {worst:.2e}");
    assert!(worst < 1e-13);
}

#[test]
fn the_species_tables_of_the_record_reproduce_the_pinned_oxygen_terms() {
    // Pins of the E4 log, section 'What is in Rust now' (pbe-sr oxygen, omega 0.11, raw basis),
    // taken from the molecular adapter's tables: the same radial code on the same waves.
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let (atom, projectors) = terms(&ctx, "pbe-sr", "O", Basis::Raw);
    let p = projectors.len();
    let mut frobenius = 0f64;
    for (a, wa) in atom.factors.iter().zip(&atom.weights) {
        for (b, wb) in atom.factors.iter().zip(&atom.weights) {
            let overlap: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            frobenius += wa * wb * overlap * overlap;
        }
    }
    let frobenius = frobenius.sqrt();
    let block = |l: usize| -> f64 {
        let mut sum = 0.;
        for (i, a) in projectors.iter().enumerate() {
            for (j, b) in projectors.iter().enumerate() {
                if a.l == l && b.l == l && a.m == b.m {
                    sum += atom.v[(i, j)].powi(2);
                }
            }
        }
        // The block of one m: the E4 entry quotes it per (l, m).
        (sum / (2 * l + 1) as f64).sqrt()
    };
    println!(
        "O pbe-sr raw: p {p}, |Delta W^SR| {frobenius:.10}, constant {:.10}, |V_s| {:.8} |V_p| {:.8}",
        atom.constant,
        block(0),
        block(1)
    );
    assert!((frobenius - 80.8910441923).abs() < 1e-6);
    assert!((atom.constant - 0.1761248251).abs() < 1e-9);
    assert!((block(0) - 1.96921590).abs() < 1e-7);
    assert!((block(1) - 4.63398732).abs() < 1e-7);
}

fn model_of(ctx: &Context, set: &str, symbol: &str) -> (SphereModel, Hybrid, usize) {
    let (atom, projectors) = terms(ctx, set, symbol, Basis::Raw);
    let n = projectors.len();
    let members: Vec<usize> = (0..n).collect();
    let hybrid = Hybrid::new(
        vec![HybridAtom {
            members,
            tensor: rotated_tensor(&atom, &projectors),
            v: atom.v.clone(),
        }],
        n,
        FRACTION,
        atom.constant,
    )
    .unwrap();
    let (atom_again, _) = terms(ctx, set, symbol, Basis::Raw);
    let model = SphereModel::new(
        vec![Species::new(atom_again, &projectors)],
        vec![Sphere {
            offset: 0,
            species: 0,
        }],
        n,
        FRACTION,
    )
    .unwrap();
    (model, hybrid, n)
}

fn largest(m: &Matrix) -> f64 {
    m.iter().map(|v| v.norm()).fold(0., f64::max)
}

#[test]
fn the_restricted_and_spinor_terms_equal_the_older_evaluator_on_the_rotated_tensor() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let (model, hybrid, n) = model_of(&ctx, "lda-sr", "Si");
    // Restricted: one matrix holding both spins.
    let total = density(n, 0.4, 2.0);
    let mine = model.collinear(&ctx, &[total.clone()]).unwrap();
    let theirs = hybrid.evaluate(&ctx, &[&slice(&total)]).unwrap();
    // The older evaluator returns the operator of the Hamiltonian (-fraction included).
    let from_theirs = Matrix::from_fn(n, n, |i, j| theirs.operators[0].as_slice()[i * n + j]);
    let operator_gap =
        largest(&(mine.operators[0].clone() * C64::new(-FRACTION, 0.) - &from_theirs));
    let energy_gap = (mine.energy - theirs.energy).abs();
    println!(
        "Si restricted: energy {:.10} against {:.10} ({energy_gap:.2e}); operator {operator_gap:.2e}",
        mine.energy, theirs.energy
    );
    assert!(energy_gap < 1e-12 * theirs.energy.abs().max(1.));
    assert!(operator_gap < 1e-11);

    // A spinor with no spin-orbit coupling and equal spins is the restricted case.
    let mut spinor = Matrix::zeros(2 * n, 2 * n);
    for s in 0..2 {
        spinor
            .view_mut((s * n, s * n), (n, n))
            .copy_from(&(total.clone() * C64::new(0.5, 0.)));
    }
    let terms = model.spinor(&ctx, &spinor).unwrap();
    assert!((terms.energy - mine.energy).abs() < 1e-12);
    // A general spinor against `evaluate_spinor`.
    let blocks = [
        [density(n, 0.1, 0.7), density(n, 0.9, 0.2)],
        [density(n, 0.9, 0.2).adjoint(), density(n, 1.3, 0.5)],
    ];
    let mut general = Matrix::zeros(2 * n, 2 * n);
    for s in 0..2 {
        for t in 0..2 {
            general
                .view_mut((s * n, t * n), (n, n))
                .copy_from(&blocks[s][t]);
        }
    }
    let flat = [
        [slice(&blocks[0][0]), slice(&blocks[0][1])],
        [slice(&blocks[1][0]), slice(&blocks[1][1])],
    ];
    let theirs = hybrid
        .evaluate_spinor(
            &ctx,
            [[&flat[0][0], &flat[0][1]], [&flat[1][0], &flat[1][1]]],
        )
        .unwrap();
    let mine = model.spinor(&ctx, &general).unwrap();
    // The older evaluator has no magnetization remainder: the model adds `-fraction P` of the
    // sphere's vector magnetization (and its operator), written out independently by `remainder`.
    let (atom, _) = periodic_screened_onecenter_support::terms(&ctx, "lda-sr", "Si", Basis::Raw);
    let (extra_energy, extra_operator) = remainder(&ctx, &atom, &general);
    assert!(
        extra_energy.abs() > 1e-6,
        "the remainder must be active here"
    );
    let from_theirs = Matrix::from_fn(2 * n, 2 * n, |i, j| {
        theirs.operators[0].as_slice()[i * 2 * n + j]
    }) - extra_operator * C64::new(FRACTION, 0.);
    let operator_gap =
        largest(&(mine.operators[0].clone() * C64::new(-FRACTION, 0.) - &from_theirs));
    let expected = theirs.energy + extra_energy;
    println!(
        "Si spinor: energy {:.10} against {expected:.10}; operator {operator_gap:.2e}",
        mine.energy
    );
    assert!((mine.energy - expected).abs() < 1e-12 * expected.abs().max(1.));
    assert!(operator_gap < 1e-11);
}

#[test]
fn the_unrestricted_energy_adds_the_polarization_and_its_operators_are_the_derivative() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let (model, hybrid, n) = model_of(&ctx, "lda-sr", "Si");
    let up = density(n, 0.2, 1.1);
    let down = density(n, 0.8, 0.6);
    let both = model.collinear(&ctx, &[up.clone(), down.clone()]).unwrap();
    // The older evaluator has no polarization: the difference is -fraction P, with the
    // same P the species code gives for these spin densities.
    let theirs = hybrid
        .evaluate(&ctx, &[&slice(&up), &slice(&down)])
        .unwrap();
    let (atom, _) = terms(&ctx, "lda-sr", "Si", Basis::Raw);
    let pol = atom
        .polarization
        .evaluate(&ctx, &up.map(|z| z.re), &down.map(|z| z.re))
        .unwrap();
    println!(
        "Si unrestricted: energy {:.10}; older evaluator {:.10}; -a P {:.10}",
        both.energy,
        theirs.energy,
        -FRACTION * pol.energy
    );
    assert!(
        pol.energy.abs() > 1e-6,
        "the polarization must be active here"
    );
    assert!((both.energy - theirs.energy + FRACTION * pol.energy).abs() < 1e-12);
    // Equal spins: the polarization vanishes and the energy is the restricted one.
    let same = model.collinear(&ctx, &[up.clone(), up.clone()]).unwrap();
    let restricted = model.collinear(&ctx, &[&up * C64::new(2., 0.)]).unwrap();
    assert!((same.energy - restricted.energy).abs() < 1e-12);

    // The operators are dE/dP_s: a Hermitian perturbation, central differences.
    let step = 1e-4;
    let mut worst = 0f64;
    for s in 0..2 {
        let delta = density(n, 2.1 + s as f64, 0.05) - density(n, 3.4, 0.03);
        let mut plus = [up.clone(), down.clone()];
        let mut minus = plus.clone();
        plus[s] += &delta * C64::new(step, 0.);
        minus[s] -= &delta * C64::new(step, 0.);
        let numeric = (model.collinear(&ctx, &plus).unwrap().energy
            - model.collinear(&ctx, &minus).unwrap().energy)
            / (2. * step);
        let analytic = -FRACTION * (&both.operators[s] * &delta).trace().re;
        // The operator is in the convention of K: the energy changes by -fraction tr(O dP).
        worst = worst.max((numeric - analytic).abs() / analytic.abs().max(1e-6));
    }
    println!("Si: operators against the finite-difference derivative of the energy: {worst:.2e}");
    assert!(worst < 1e-5);
}

#[test]
fn a_zero_fraction_adds_no_energy_and_the_model_is_hermitian() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let (atom, projectors) = terms(&ctx, "lda-sr", "Si", Basis::Raw);
    let n = projectors.len();
    let none = SphereModel::new(
        vec![Species::new(atom, &projectors)],
        vec![Sphere {
            offset: 0,
            species: 0,
        }],
        n,
        0.,
    )
    .unwrap();
    let (up, down) = (density(n, 0.2, 1.1), density(n, 0.8, 0.6));
    let terms = none.collinear(&ctx, &[up.clone(), down.clone()]).unwrap();
    // The energy is -fraction times the terms: exactly nothing at fraction 0 (-0.0 == 0.0).
    assert!(terms.energy == 0.);
    // The operators (in the convention of K, which the solver multiplies by the fraction) are
    // Hermitian, as the exchange matrices they join.
    let (model, _, _) = model_of(&ctx, "lda-sr", "Si");
    let both = model.collinear(&ctx, &[up, down]).unwrap();
    for operator in &both.operators {
        let skew = largest(&(operator - operator.adjoint()));
        println!("operator skew part {skew:.2e}");
        assert!(skew < 1e-12 * largest(operator));
    }
}
