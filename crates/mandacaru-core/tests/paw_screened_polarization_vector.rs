//! The non-collinear remainder of the spheres' semilocal short-range exchange
//! (`paw::screened_polarization::vector`) on a toy species: no molecule, no crystal.
//!
//! `P = (F[2 n_+] + F[2 n_-]) / 2 - F[max(n_0, 0)]`, `n_pm = max((n_0 +- |m|) / 2, 0)`, is checked
//! against an oracle that rebuilds the radial fields from the waves, against finite differences
//! of that oracle (the operators `O^a = dP/dQ^a`, also with an active clamp), against the
//! collinear `Polarization::evaluate` for a one-signed `m_z`, and for its symmetries: rotation
//! invariance, evenness, quadratic scaling and exactly zero without a magnetization.
mod paw_screened_polarization_vector_support;

use nalgebra::DMatrix;
use paw_screened_polarization_vector_support::*;

const AMPLITUDE: [f64; 3] = [0.25, 0.2, 0.3];
/// The operator against the finite difference of the oracle: measured 5e-11 (clamp inactive) and
/// 1e-11 (clamp active); 2e-10 of fourth-order truncation at the step used.
const GATE: f64 = 1e-8;

fn relative(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-300)
}
fn zeros(toy: &Toy) -> Pauli {
    std::array::from_fn(|_| DMatrix::zeros(toy.size(), toy.size()))
}
/// A direction with only component `a` nonzero (a different radial shape per component).
fn along(toy: &Toy, a: usize) -> Pauli {
    let mut dq = zeros(toy);
    dq[a] = toy.varied(0.53 + 0.29 * a as f64);
    dq
}

#[test]
fn the_energy_is_the_oracle_of_the_local_rotation_prescription() {
    let (ctx, toy) = (context(), toy());
    let q = toy.pauli(AMPLITUDE);
    let seen = oracle(&ctx, &toy, &q, Total::Unclamped);
    let terms = toy.polarization.evaluate_vector(&ctx, &q).unwrap();
    println!(
        "P {:.12e} oracle {:.12e} ({:.2e}); F[n_0] {:.9e}; clamped nodes {}",
        terms.energy,
        seen.energy,
        relative(terms.energy, seen.energy),
        terms.unpolarized,
        seen.clamped
    );
    assert!(terms.energy.abs() > 1e-6, "the remainder must be active");
    assert_eq!(seen.clamped, 0, "no clamp may be active in this gate");
    assert!(relative(terms.energy, seen.energy) < 1e-10);
}

#[test]
fn a_smooth_core_is_charge_on_the_smooth_side_and_its_operators_stay_the_derivative() {
    // The grid evaluates the functional on the smooth valence density plus the smooth core, so
    // the sphere's smooth side carries it as charge: the energy is the oracle's with the core,
    // it differs from the coreless sphere's, and the operators are still dP/dQ.
    let (ctx, toy) = (context(), toy_with_core(0.4));
    let bare = toy_with_core(0.);
    let q = toy.pauli(AMPLITUDE);
    let seen = oracle(&ctx, &toy, &q, Total::Unclamped);
    let terms = toy.polarization.evaluate_vector(&ctx, &q).unwrap();
    let without = bare.polarization.evaluate_vector(&ctx, &q).unwrap();
    println!(
        "with core P {:.12e} oracle {:.12e}; without {:.12e}",
        terms.energy, seen.energy, without.energy
    );
    assert_eq!(seen.clamped, 0);
    assert!(relative(terms.energy, seen.energy) < 1e-10);
    assert!(
        relative(terms.energy, without.energy) > 1e-3,
        "the core must matter"
    );
    let gaps = operator_against_the_oracle(&toy, &q, Total::Unclamped, "smooth core");
    assert!(gaps.iter().all(|g| *g < GATE), "{gaps:?}");
}

#[test]
fn a_one_signed_m_z_is_the_collinear_two_spin_evaluation() {
    // m along z: Q^0 = D_up + D_dn, Q^3 = D_up - D_dn.
    let (ctx, toy) = (context(), toy());
    let d0 = toy.reference();
    let (up, down) = (&d0 * 0.8, &d0 * 0.3);
    let collinear = toy.polarization.evaluate(&ctx, &up, &down).unwrap();
    let mut q = zeros(&toy);
    q[0] = &up + &down;
    q[3] = &up - &down;
    let terms = toy.polarization.evaluate_vector(&ctx, &q).unwrap();
    let operator_up = &terms.operators[0] + &terms.operators[3];
    let operator_down = &terms.operators[0] - &terms.operators[3];
    let worst = [
        relative(terms.energy, collinear.energy),
        relative(terms.unpolarized, collinear.unpolarized),
        (&operator_up - &collinear.up).abs().max() / collinear.up.abs().max(),
        (&operator_down - &collinear.down).abs().max() / collinear.down.abs().max(),
        terms.operators[1].abs().max() + terms.operators[2].abs().max(),
    ];
    println!(
        "one-signed m_z against evaluate(D_up, D_dn): energy, F, O_up, O_dn, transverse {worst:?}"
    );
    assert!(collinear.energy.abs() > 1e-6 && collinear.up.abs().max() > 1e-6);
    // Measured 1.6e-14 at worst (the plan's gate is 1e-12).
    assert!(worst[..4].iter().all(|w| *w < 1e-12), "{worst:?}");
    assert_eq!(worst[4], 0., "m along z has no transverse operator");
    // Along -z the pointwise |m| gives the same energy and the spin-swapped operators.
    q[3] = -&q[3];
    let flipped = toy.polarization.evaluate_vector(&ctx, &q).unwrap();
    assert_eq!(flipped.energy, terms.energy);
    assert_eq!(flipped.operators[3], -&terms.operators[3]);
}

#[test]
fn no_magnetization_gives_exactly_nothing() {
    let (ctx, toy) = (context(), toy());
    for scale in [0.3, 1., 1.7] {
        let mut q = zeros(&toy);
        q[0] = toy.reference() * scale;
        let terms = toy.polarization.evaluate_vector(&ctx, &q).unwrap();
        assert_eq!(terms.energy, 0., "scale {scale}");
        assert!(terms.operators.iter().all(|o| o.iter().all(|v| *v == 0.)));
        assert!(terms.unpolarized != 0.);
        assert_eq!(toy.polarization.magnetization_ratio(&q).unwrap(), 0.);
    }
    let terms = toy
        .polarization
        .evaluate_vector(&ctx, &zeros(&toy))
        .unwrap();
    assert_eq!((terms.energy, terms.unpolarized), (0., 0.));
    // A magnetization is a ratio of the charge, and it scales with the moment.
    let q = toy.pauli(AMPLITUDE);
    let ratio = toy.polarization.magnetization_ratio(&q).unwrap();
    let half = toy
        .polarization
        .magnetization_ratio(&scaled(&q, 0.5))
        .unwrap();
    println!("spherical-average magnetization / charge: {ratio:.4e} (half the moment {half:.4e})");
    assert!(ratio > 1e-2 && relative(ratio, 2. * half) < 1e-12);
}

fn operator_against_the_oracle(toy: &Toy, q: &Pauli, total: Total, label: &str) -> [f64; 5] {
    let ctx = context();
    let terms = toy.polarization.evaluate_vector(&ctx, q).unwrap();
    // The five directions: each component alone, then all of them at once.
    let mut directions: Vec<Pauli> = (0..4).map(|a| along(toy, a)).collect();
    directions.push(std::array::from_fn(|a| {
        toy.varied(0.71 + 0.17 * a as f64) * 0.5
    }));
    let mut gaps = [0.; 5];
    for (i, dq) in directions.iter().enumerate() {
        let numeric = derivative(toy, q, dq, 2e-3, &|s| oracle(&ctx, toy, s, total).energy);
        let analytic = contract(&terms.operators, dq);
        assert!(
            analytic.abs() > 1e-6,
            "{label} direction {i}: derivative {analytic:e}"
        );
        gaps[i] = relative(numeric, analytic);
    }
    println!(
        "{label}: operator against the finite difference of the oracle, per direction {gaps:?}"
    );
    gaps
}

#[test]
fn the_operators_are_the_derivative_of_the_energy_for_every_component() {
    let toy = toy();
    let q = toy.pauli(AMPLITUDE);
    let seen = oracle(&context(), &toy, &q, Total::Unclamped);
    assert_eq!(seen.clamped, 0);
    let gaps = operator_against_the_oracle(&toy, &q, Total::Unclamped, "inactive clamp");
    assert!(gaps.iter().all(|g| *g < GATE), "{gaps:?}");
}

#[test]
fn with_an_active_clamp_the_operators_are_the_derivative_of_the_clamped_energy() {
    // |m| > n_0 at some nodes (not physical for a density matrix, but arbitrary Q reach it):
    // the operator is the derivative of the clamped energy with the UNCLAMPED total.
    let toy = toy();
    let q = toy.pauli([0.9, 0.7, 1.1]);
    let seen = oracle(&context(), &toy, &q, Total::Unclamped);
    println!(
        "clamp gate: {} nodes clamped inside the sphere",
        seen.clamped
    );
    assert!(seen.clamped > 3, "the clamp must be active");
    let right = operator_against_the_oracle(&toy, &q, Total::Unclamped, "active clamp");
    assert!(right.iter().all(|g| *g < GATE), "{right:?}");
    // Control: the clamped-sum form of the third term has another energy and another derivative.
    let wrong = operator_against_the_oracle(&toy, &q, Total::ClampedSum, "clamped-sum control");
    assert!(wrong.iter().all(|g| *g > 100. * GATE), "{wrong:?}");
}

fn rotation(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let norm = axis.iter().map(|x| x * x).sum::<f64>().sqrt();
    let u: Vec<f64> = axis.iter().map(|x| x / norm).collect();
    let (s, c) = angle.sin_cos();
    let mut r = [[0.; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let cross = if a == b {
                0.
            } else {
                // epsilon_{a b k} u_k with the Levi-Civita sign
                let k = 3 - a - b;
                let sign = if (b + 3 - a) % 3 == 1 { 1. } else { -1. };
                sign * u[k]
            };
            r[a][b] = c * f64::from(a == b) + (1. - c) * u[a] * u[b] - s * cross;
        }
    }
    r
}
fn rotate(q: &Pauli, r: &[[f64; 3]; 3]) -> Pauli {
    std::array::from_fn(|a| {
        if a == 0 {
            q[0].clone()
        } else {
            (0..3).map(|b| &q[b + 1] * r[a - 1][b]).sum()
        }
    })
}

#[test]
fn the_energy_is_invariant_and_the_operator_covariant_under_a_global_rotation() {
    let (ctx, toy) = (context(), toy());
    let q = toy.pauli(AMPLITUDE);
    let base = toy.polarization.evaluate_vector(&ctx, &q).unwrap();
    let transverse = q[1].abs().max().min(q[2].abs().max());
    assert!(
        transverse > 1e-2,
        "the transverse components must be populated"
    );
    let (mut energy_gap, mut operator_gap) = (0f64, 0f64);
    for (axis, angle) in [
        ([1., 0.4, -0.7], 0.9),
        ([0., 1., 0.], 1.3),
        ([0.3, -0.5, 0.2], 2.6),
    ] {
        let r = rotation(axis, angle);
        let rotated = toy
            .polarization
            .evaluate_vector(&ctx, &rotate(&q, &r))
            .unwrap();
        energy_gap = energy_gap.max(relative(rotated.energy, base.energy));
        // dP = sum_a O^a dQ^a is invariant, so the operator rotates as the components do.
        let expected = rotate(&base.operators, &r);
        for a in 0..4 {
            operator_gap = operator_gap.max(
                (&rotated.operators[a] - &expected[a]).abs().max() / base.operators[a].abs().max(),
            );
        }
    }
    println!("rotation: energy {energy_gap:.2e}, operator {operator_gap:.2e}");
    // Measured 1.4e-14 and 1.1e-14 (the rotated fields differ at round-off): the plan's 1e-12.
    assert!(energy_gap < 1e-12 && operator_gap < 1e-12);
    // Control: a rotation that drops the transverse components (the spin-diagonal blocks only)
    // loses the polarization of an x-polarized sphere.
    let mut along_x = zeros(&toy);
    along_x[0] = q[0].clone();
    along_x[1] = q[3].clone();
    let along_z = {
        let mut z = zeros(&toy);
        z[0] = q[0].clone();
        z[3] = q[3].clone();
        z
    };
    let (x, z) = (
        toy.polarization
            .evaluate_vector(&ctx, &along_x)
            .unwrap()
            .energy,
        toy.polarization
            .evaluate_vector(&ctx, &along_z)
            .unwrap()
            .energy,
    );
    let dropped = {
        let mut d = along_x.clone();
        d[1] = DMatrix::zeros(toy.size(), toy.size());
        toy.polarization.evaluate_vector(&ctx, &d).unwrap().energy
    };
    println!("m along x {x:.9e}, along z {z:.9e}, transverse part dropped {dropped:.3e}");
    assert!(relative(x, z) < 1e-13 && z.abs() > 1e-6);
    assert_eq!(dropped, 0.);
}

#[test]
fn the_remainder_is_even_and_quadratic_and_the_operators_linear_in_the_moment() {
    let (ctx, toy) = (context(), toy());
    let q = toy.pauli(AMPLITUDE);
    let at = |t: f64| {
        toy.polarization
            .evaluate_vector(&ctx, &scaled(&q, t))
            .unwrap()
    };
    let (small, large) = (at(0.01), at(0.02));
    assert_eq!(at(-0.02).energy, large.energy, "P is even in the moment");
    let ratio = large.energy / small.energy;
    println!(
        "P(0.01) {:.6e}, P(0.02) {:.6e}, ratio {ratio:.6}",
        small.energy, large.energy
    );
    assert!(small.energy.abs() > 1e-12 && (ratio - 4.).abs() < 1e-3);
    for a in 1..4 {
        let ratio = large.operators[a].sum() / small.operators[a].sum();
        assert!(small.operators[a].abs().max() > 1e-9, "component {a}");
        assert!((ratio - 2.).abs() < 1e-3, "O^{a} ratio {ratio}");
    }
    // O^0 is even: quadratic in the moment.
    let ratio = large.operators[0].sum() / small.operators[0].sum();
    assert!((ratio - 4.).abs() < 2e-3, "O^0 ratio {ratio}");
}

#[test]
fn a_sign_changing_m_z_differs_from_the_collinear_two_spin_evaluation() {
    // MEASURED, NOT GATED: for a one-center m_z(r) that changes sign the pointwise |m| swaps
    // n_+ and n_- over that range, the collinear formula does not. It is a property of the
    // prescription (the grid also uses |m|), recorded for the d-shell and oxygen magnets.
    let (ctx, toy) = (context(), toy());
    let mut q = zeros(&toy);
    q[0] = toy.reference() * 1.5;
    q[3] = {
        // the s channel up, the p channel down: m_z changes sign across the sphere
        let d0 = toy.reference();
        DMatrix::from_fn(toy.size(), toy.size(), |a, b| {
            let sign = if toy.projectors[a].channel == 0 {
                1.
            } else {
                -1.8
            };
            0.3 * sign * d0[(a, b)]
        })
    };
    let seen = oracle(&ctx, &toy, &q, Total::Unclamped);
    assert!(
        seen.sign_changes >= 1,
        "m_z must change sign (changes {})",
        seen.sign_changes
    );
    let terms = toy.polarization.evaluate_vector(&ctx, &q).unwrap();
    let (up, down) = ((&q[0] + &q[3]) * 0.5, (&q[0] - &q[3]) * 0.5);
    let collinear = toy.polarization.evaluate(&ctx, &up, &down).unwrap();
    println!(
        "sign-changing m_z ({} changes): P pointwise |m| {:.9e}, collinear {:.9e}, difference {:.2} percent",
        seen.sign_changes,
        terms.energy,
        collinear.energy,
        100. * (terms.energy - collinear.energy).abs() / collinear.energy.abs()
    );
    assert!(relative(terms.energy, seen.energy) < 1e-10);
}

#[test]
fn a_component_that_does_not_match_the_layout_is_refused() {
    let (ctx, toy) = (context(), toy());
    let mut q = toy.pauli(AMPLITUDE);
    q[2] = DMatrix::zeros(toy.size() - 1, toy.size() - 1);
    assert!(toy.polarization.evaluate_vector(&ctx, &q).is_err());
    assert!(toy.polarization.magnetization_ratio(&q).is_err());
}
