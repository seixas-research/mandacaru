//! The electrostatic strain terms of the crystal stress (`periodic::stress_terms::electrostatic`,
//! plan E-1..E-7) on a hand-built frozen state of Si, Al2 and the skew LiH cell: each term of the
//! `Breakdown`, all nine `(a, b)` entries unsymmetrized, against the central difference of the
//! same expression rebuilt at the strained cell with the pins held (`periodic_stress_support::oracle`),
//! and the negative controls. No SCF. Needs `MANDACARU_PAW_PATH`.
//!
//! The Hartree self term is differenced ingredient by ingredient (`dv` and `Omega`, the kernel, the
//! hats, the ions; the sum of the four partial derivatives is the derivative of `E_G`). The dense
//! sums are summed over the pinned list of reciprocal vectors that no table node of the shape
//! tables straddles at the step (the table is piecewise linear, so a whole-list difference has a
//! kink floor far above the gate; see `strain_sums.rs`); the analytic sum is fed the SAME list.
mod periodic_gradient_crystals_support;
mod periodic_stress_support;
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;
mod strain_samples_support;
mod strain_support;

use mandacaru_core::periodic::stress_terms::{electrostatic, Strain};
use periodic_stress_support::{
    entry,
    gates::{bites, gate, negated, transposed, Tensor},
    oracle, Eps, World,
};

/// Relative to the peak entry of the term (the plan's 1e-9 for grid arrays, 1e-7 for the dense
/// sums; measured floors are in the log).
const GATE: f64 = 1e-8;
const DENSE_GATE: f64 = 1e-7;
const STEP: f64 = 1e-5;
const DENSE_STEP: f64 = 2e-5;

const ZERO: Eps = [[0.; 3]; 3];

fn central(step: f64, mut f: impl FnMut(&Eps) -> f64) -> Tensor {
    std::array::from_fn(|a| {
        std::array::from_fn(|b| (f(&entry(a, b, step)) - f(&entry(a, b, -step))) / (2. * step))
    })
}

fn run(case: &'static str) {
    let ks: &[usize] = if case == "lih" { &[0] } else { &[0, 1] };
    let mut w = World::new(case, ks, "lda");
    w.taint(STEP);
    println!("{case}");
    let (list, vectors) = oracle::pinned_dense(&w, DENSE_STEP);
    let (hartree, d, constants_got) = w.with_setup(Some(vectors), |setup, frozen| {
        let hartree = setup.hartree(&frozen.fields, &frozen.q).unwrap();
        let dense = electrostatic::dense_terms(&setup.dense().unwrap(), &frozen.q).unwrap();
        (hartree, dense, setup.constants().unwrap())
    });
    // E-1 .. E-4: one ingredient strained at a time.
    let part = |i: usize| {
        central(STEP, |e| {
            let mut s = [&ZERO; 4];
            s[i] = e;
            oracle::hartree_energy(&w, s)
        })
    };
    gate("E-1 hartree jacobian", &hartree.jacobian, &part(0), GATE);
    gate("E-2 kernel", &hartree.kernel, &part(1), GATE);
    gate("E-3 hats", &hartree.hats, &part(2), GATE);
    gate("E-4 ions", &hartree.ions, &part(3), GATE);
    // E-5, E-6, E-7.
    let dense = central(DENSE_STEP, |e| oracle::dense_energy(&w, e, &list));
    gate("E-5 dense", &d.dense, &dense, DENSE_GATE);
    let linear = central(DENSE_STEP, |e| oracle::linear_energy(&w, e));
    gate("E-6 linear", &d.linear, &linear, DENSE_GATE);
    let constants = central(STEP, |e| oracle::constants_energy(&w, e));
    gate("E-7 constants", &constants_got, &constants, GATE);
    // Controls: each wrong variant against the SAME oracle term.
    bites("hartree jacobian dropped", &[[0.; 3]; 3], &part(0), GATE);
    bites("kernel sign", &negated(&hartree.kernel), &part(1), GATE);
    bites("hats sign", &negated(&hartree.hats), &part(2), GATE);
    bites("ions sign", &negated(&hartree.ions), &part(3), GATE);
    bites("dense sign", &negated(&d.dense), &dense, DENSE_GATE);
    bites("constants sign", &negated(&constants_got), &constants, GATE);
    let hats = part(2);
    bites("hats transposed", &transposed(&hartree.hats), &hats, GATE);
}

#[test]
fn the_electrostatic_terms_si() {
    run("si");
}

#[test]
fn the_electrostatic_terms_al2() {
    run("al2");
}

#[test]
fn the_electrostatic_terms_lih() {
    run("lih");
}
