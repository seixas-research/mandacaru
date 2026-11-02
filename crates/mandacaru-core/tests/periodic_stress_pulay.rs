//! The matrix terms of the crystal stress (`periodic::stress_terms::pulay`, plan S-1..S-5) on a
//! hand-built frozen state: two k-points (the first is Gamma on the periodic fixtures, where the
//! alias-mean symbol of the kinetic term differs from the mean-vector one), fractional
//! occupations, a rough local potential, random channel potentials. Each term of the
//! `Breakdown` is compared, for ALL nine `(a, b)` entries unsymmetrized, with the central
//! difference of the frozen energy piece rebuilt at the strained cell (`oracle::matrix_energies`:
//! Bloch sums, `kinetic_form`, short-range spheres and projections of the carried cell, the
//! node values of the potential and `D`, `W` fixed). Needs `MANDACARU_PAW_PATH`.
//!
//! The LiH case also carries a +U operator `O` in `D_ion + sum_a m_a B_a + O`: the projection
//! contraction must reproduce the difference with `O` in the oracle's operator as well.
mod periodic_gradient_crystals_support;
mod periodic_stress_support;
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;
mod strain_samples_support;
mod strain_support;

use mandacaru_core::periodic::{
    gradient::{Frozen, Matrix, Term},
    stress_terms::{pulay, Strain},
};
use num_complex::Complex64;
use periodic_stress_support::{
    gates::{bites, gate, map, miss, negated, transposed, Tensor},
    oracle, seq, Eps, World, GAMMA, GENERIC_K,
};

const GATE: f64 = 1e-8;
const STEP: f64 = 1e-5;
const ZERO: Eps = [[0.; 3]; 3];

fn hubbard(p: usize) -> Matrix {
    let a = Matrix::from_fn(p, p, |i, j| {
        Complex64::new(seq(i * p + j, 71), seq(i * p + j, 72)) * 0.2
    });
    (&a + a.adjoint()) * Complex64::new(0.5, 0.)
}

fn run(case: &'static str, kpoints: &[[f64; 3]], plus_u: bool) {
    let started = std::time::Instant::now();
    let w = World::new(case, &vec![0; kpoints.len()], "lda").at(kpoints);
    let o = plus_u.then(|| hubbard(w.data.crystal.p));
    let (got, without_u) = w.with_setup(None, |setup, frozen| {
        let with = |hubbard: Option<Vec<Matrix>>| Frozen {
            state: frozen.state.clone(),
            fields: frozen.fields.clone(),
            q: frozen.q.clone(),
            hubbard,
        };
        let got = pulay::contract(setup, &with(o.clone().map(|o| vec![o]))).unwrap();
        // The +U branch is taken: without the operator the projection term is another number.
        let without = o
            .is_some()
            .then(|| pulay::contract(setup, &with(None)).unwrap().projection);
        (got, without)
    });
    println!(
        "  analytic contraction done at {:.1} s",
        started.elapsed().as_secs_f64()
    );
    // The oracle: every entry, every piece, one central difference at a time.
    let mut fd = [[[0f64; 5]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            fd[a][b] = oracle::central(a, b, STEP, |e| oracle::matrix_energies(&w, e, o.as_ref()));
        }
    }
    println!("  oracle done at {:.1} s", started.elapsed().as_secs_f64());
    let term = |i: usize| -> Tensor { map(&[[0.; 3]; 3], |a, b, _| fd[a][b][i]) };
    let names = ["overlap", "kinetic", "local", "short_range", "projection"];
    let analytic = [
        got.overlap,
        got.kinetic,
        got.local,
        got.short_range,
        got.projection,
    ];
    for i in 0..5 {
        gate(names[i], &analytic[i], &term(i), GATE);
    }
    // Controls, each against the oracle term it damages.
    if let Some(without) = &without_u {
        bites(
            "projection without the +U operator",
            without,
            &term(4),
            GATE,
        );
    }
    // The nine entries are compared unsymmetrized: the transposed tensor must miss (an entry
    // whose antisymmetric part is below the gate cannot discriminate and is reported).
    let mut discriminating = 0;
    for i in 0..5 {
        let bite = miss(&transposed(&analytic[i]), &term(i)) / GATE;
        println!("  control {} transposed: bites {bite:.2e} gates", names[i]);
        discriminating += usize::from(bite > 100.);
    }
    // The cells with a symmetric overlap, kinetic and local derivative (the skew LiH at Gamma)
    // cannot tell the order there: the projections and the spheres still do.
    let needed = if case == "lih" { 2 } else { 4 };
    assert!(
        discriminating >= needed,
        "a transposed index order must miss the oracle"
    );
    let base = oracle::matrix_energies(&w, &ZERO, o.as_ref());
    let jacobian = |t: &Tensor, e: f64| map(t, |a, b, x| if a == b { x - e } else { x });
    // -W dS without its dv: the overlap term minus delta_ab (-W S) is the tangent part alone.
    bites(
        "overlap without delta_ab",
        &jacobian(&got.overlap, base[0]),
        &term(0),
        GATE,
    );
    bites("no -W dS", &[[0.; 3]; 3], &term(0), GATE);
    bites(
        "kinetic without delta_ab",
        &jacobian(&got.kinetic, base[1]),
        &term(1),
        GATE,
    );
    bites(
        "local potential with delta_ab",
        &map(&got.local, |a, b, x| if a == b { x + base[2] } else { x }),
        &term(2),
        GATE,
    );
    bites("local sign", &negated(&got.local), &term(2), GATE);
    bites(
        "short-range sign",
        &negated(&got.short_range),
        &term(3),
        GATE,
    );
    bites("projection sign", &negated(&got.projection), &term(4), GATE);
}

#[test]
fn the_matrix_terms_lih() {
    run("lih", &[GAMMA, GENERIC_K], true);
}

#[test]
fn the_matrix_terms_si() {
    run("si", &[GENERIC_K], false);
}

#[test]
fn the_matrix_terms_al2() {
    run("al2", &[GENERIC_K], false);
}

/// The kinetic symbol is the alias MEAN of `Q_a Q_b`. At Gamma a frequency sits on the Nyquist
/// faces, where the product of the mean vectors (zero there) is another number; the control forms
/// the term with the wrong symbol and must miss the central difference of `kinetic_form` by more
/// than 100 gates, the right symbol must meet the gate. Needs a cell whose aliases differ.
fn symbol(case: &'static str) {
    let w = World::new(case, &[0], "lda").at(&[GAMMA]);
    let got = w.with_setup(None, |setup, frozen| {
        let dm = setup
            .matrix_strain(0, 0, Term::Kinetic, &frozen.fields)
            .unwrap();
        let d = w.bands[0].density();
        let weight = 2.0 * w.weights[0];
        map(&[[0.; 3]; 3], |a, b, _| {
            let m = &dm[a][b];
            let mut t = Complex64::new(0., 0.);
            for i in 0..d.nrows() {
                for j in 0..d.ncols() {
                    t += d[(i, j)] * m[(j, i)];
                }
            }
            weight * t.re
        })
    });
    let mut fd = [[0.; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            fd[a][b] = oracle::central(a, b, STEP, |e| [oracle::kinetic_energy(&w, e)])[0];
        }
    }
    gate("kinetic at Gamma", &got, &fd, GATE);
    let [right, wrong] = oracle::symbol_terms(&w);
    let mean_vector = map(&got, |a, b, x| x - right[a][b] + wrong[a][b]);
    bites(
        "kinetic symbol from the mean vectors",
        &mean_vector,
        &fd,
        GATE,
    );
}

#[test]
fn the_alias_mean_kinetic_symbol_si() {
    symbol("si");
}

#[test]
fn the_alias_mean_kinetic_symbol_al2() {
    symbol("al2");
}
