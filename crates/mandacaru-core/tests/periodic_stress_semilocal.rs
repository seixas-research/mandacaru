//! The whole semilocal crystal stress (`periodic::stress_terms::stress`, plan AS-4 + AS-5) on a
//! hand-built frozen PBE state of the skew LiH cell: every one of the fifteen `Breakdown` terms
//! against the sum of the oracle pieces (each one the central difference, in one strain entry at a
//! time, of the frozen energy piece rebuilt at the strained cell with the pins held), the SUM of
//! all nine `(a, b)` entries unsymmetrized, the sigma convention (`(g_ab + g_ba) / 2 Omega`, the
//! antisymmetric part reported), and the refusals of what the module does not differentiate yet.
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_stress_support;
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;
mod strain_samples_support;
mod strain_support;

use mandacaru_core::{periodic::stress_terms::stress, Error};
use periodic_stress_support::{
    gates::{bites, gate, map, Tensor},
    oracle, Eps, World,
};

const GATE: f64 = 1e-8;
const DENSE_GATE: f64 = 1e-7;
const STEP: f64 = 1e-5;
const DENSE_STEP: f64 = 2e-5;
const ZERO: Eps = [[0.; 3]; 3];

#[test]
fn the_stress_is_the_derivative_of_the_frozen_energy() {
    let mut w = World::new("lih", &[0], "pbe");
    w.taint(STEP);
    let vbar = w.fields().xc.potential;
    let (list, vectors) = oracle::pinned_dense(&w, DENSE_STEP);
    let result = w.with_setup(Some(vectors), |setup, frozen| {
        stress(setup, frozen).unwrap()
    });
    let b = &result.breakdown;
    let terms = result.breakdown.terms();
    let mut fd: Vec<Tensor> = vec![[[0.; 3]; 3]; terms.len()];
    for a in 0..3 {
        for c in 0..3 {
            let mut put = |slot: usize, x: f64| fd[slot][a][c] = x;
            let m = oracle::central(a, c, STEP, |e| oracle::matrix_energies(&w, e, None));
            for (i, x) in m.iter().enumerate() {
                put(i, *x);
            }
            for i in 0..4 {
                let part = oracle::central(a, c, STEP, |e| {
                    let mut s = [&ZERO; 4];
                    s[i] = e;
                    [oracle::hartree_energy(&w, s)]
                })[0];
                put(5 + i, part);
            }
            let dense = oracle::central(a, c, DENSE_STEP, |e| [oracle::dense_energy(&w, e, &list)]);
            put(9, dense[0]);
            let rest = oracle::central(a, c, DENSE_STEP, |e| [oracle::linear_energy(&w, e)]);
            put(10, rest[0]);
            let k = oracle::central(a, c, STEP, |e| [oracle::constants_energy(&w, e)]);
            put(11, k[0]);
            let x = oracle::central(a, c, STEP, |e| oracle::xc_energies(&w, e, &vbar));
            put(12, x[0]);
            put(13, oracle::xc_wave_derivative(&w, a, c, STEP));
            put(14, x[2]);
        }
    }
    let mut total: Tensor = [[0.; 3]; 3];
    for (i, (name, term)) in b.terms().iter().enumerate() {
        // A scalar-relativistic semilocal crystal has no spin-orbit and no exchange term: exactly zero.
        if matches!(*name, "spin_orbit" | "exchange") {
            assert!(
                term.iter().flatten().all(|x| *x == 0.),
                "{name} must vanish"
            );
            continue;
        }
        let gate_for = if matches!(*name, "dense" | "linear") {
            DENSE_GATE
        } else {
            GATE
        };
        gate(name, term, &fd[i], gate_for);
        total = map(&total, |a, c, x| x + fd[i][a][c]);
    }
    let sum = result.gradient;
    let m = gate("sum of the terms", &sum, &total, DENSE_GATE);
    println!("  sum miss {m:.2e}");
    // A single missing term in the sum is far above the gate.
    for (i, (name, _)) in b.terms().iter().enumerate() {
        if matches!(*name, "spin_orbit" | "exchange") {
            continue;
        }
        let without = map(&sum, |a, c, x| x - b.terms()[i].1[a][c]);
        bites(&format!("sum without {name}"), &without, &total, DENSE_GATE);
    }
    // The convention: sigma = (g_ab + g_ba) / (2 Omega), the antisymmetric part is a diagnostic.
    for a in 0..3 {
        for c in 0..3 {
            let want = 0.5 * (sum[a][c] + sum[c][a]) / result.volume;
            assert_eq!(result.sigma[a][c], want);
            assert_eq!(result.sigma[a][c], result.sigma[c][a]);
            assert_eq!(result.antisymmetric[a][c], 0.5 * (sum[a][c] - sum[c][a]));
        }
    }
    let trace = result.sigma[0][0] + result.sigma[1][1] + result.sigma[2][2];
    assert_eq!(result.pressure(), -trace / 3.);
    println!("  antisymmetric part {:?}", result.antisymmetric);
}

#[test]
fn what_is_not_differentiated_yet_raises_with_its_cause() {
    let w = World::new("lih", &[0], "pbe");
    let refused = |edit: &dyn Fn(&mut mandacaru_core::periodic::gradient::Layout)| {
        w.with_layout(None, |l| edit(l), |setup, frozen| stress(setup, frozen))
    };
    for (what, edit) in [
        (
            "hybrid",
            Box::new(|l: &mut _| set(l, "hse06")) as Box<dyn Fn(&mut _)>,
        ),
        ("meta-GGA", Box::new(|l: &mut _| set(l, "r2scan"))),
        ("rVV10", Box::new(|l: &mut _| set(l, "pbe+rvv10"))),
        (
            "float32",
            Box::new(|l: &mut mandacaru_core::periodic::gradient::Layout| {
                l.request.single_precision = true
            }),
        ),
        (
            "spin polarized",
            Box::new(|l: &mut mandacaru_core::periodic::gradient::Layout| l.spins = 2),
        ),
    ] {
        match refused(&*edit) {
            Err(Error::Unsupported(message)) => {
                assert!(
                    message.contains("analytic crystal stress"),
                    "{what}: {message}"
                );
            }
            other => panic!("{what}: expected Unsupported, got {:?}", other.map(|_| ())),
        }
    }
}

fn set(layout: &mut mandacaru_core::periodic::gradient::Layout, functional: &str) {
    layout.request.functional = functional.to_string();
}
