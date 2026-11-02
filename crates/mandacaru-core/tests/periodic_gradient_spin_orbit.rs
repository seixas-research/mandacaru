//! The spin-orbit union-projector term of the spinor crystal force (`periodic::gradient::spin_orbit`,
//! CS-2) on the toy of the force engine: the HU-10a toy carries `C(X)` and its exact atom
//! derivative, a spin-orbit channel (`l = 1`, one radial projector) sits on every shell, and a
//! fixed Hermitian spinor density `D` with populated spin blocks replaces the SCF (the term is
//! `D_SO` constant, no hats, no `dD_SO`).
//!
//! Gates: the ownership-identity contraction of the engine (row-center `G`, owner masks, the
//! `Z` of the spin trace) equals the oracle built from the toy's exact `dC_X` through
//! `models::spin_orbit::pair` to 1e-10; a Richardson central difference of `sum_k w_k Re tr[D h^SO]`
//! confirms it at its measured floor; the sum over atoms vanishes to 1e-12; the term is linear
//! in the coupling strength and zero without coupling. Every negative control misses the oracle
//! by more than 100 times the gate. The atom sum of the force is zero by construction of the
//! ownership scatter (a structural check); the physical translation invariance is the value
//! invariance of the energy under a rigid shift of every center. The term sees only the
//! Hermitian part of `D`, checked against the oracle with a general `D`. The hook in the engine
//! is `periodic_gradient_spin_orbit_hook.rs`.
mod hubbard_gradient_support;
mod periodic_gradient_spin_orbit_support;

use hubbard_gradient_support::{Kind, Matrix, Toy};
use mandacaru_core::{
    periodic::gradient::spin_orbit::{Gradient, SpinOrbit},
    quantum::C64,
};
use periodic_gradient_spin_orbit_support::*;

const GATE: f64 = 1e-10;
const SUM_RULE: f64 = 1e-12;
/// 100 x the measured floor of the Richardson difference (1.2e-14).
const FD_GATE: f64 = 1e-12;
const CONTROL: f64 = 100. * GATE;
const LAMBDA: [f64; 2] = [0.31, 0.57];

fn toys() -> [Toy; 2] {
    [Toy::two_site(), Toy::three_site()]
}

#[test]
fn the_operator_and_the_matrix_are_the_pair_of_the_molecule() {
    let ctx = context();
    for toy in toys() {
        let chs = channels(&ctx, &toy, &LAMBDA);
        let so = spin_orbit(&ctx, &toy, &LAMBDA);
        for k in toy.kpoints() {
            let c = toy.bloch(Kind::Projection, k);
            let want = pair_of(&ctx, &chs, &c, &c);
            let got = so.matrix(&c);
            let defect = (0..want.len())
                .map(|i| (got[(i / got.ncols(), i % got.ncols())] - want[i]).norm())
                .fold(0., f64::max);
            assert!(defect < 1e-13, "{defect:e}");
            // Hermitian and not trivial.
            assert!((&got - got.adjoint()).norm() < 1e-13);
            assert!(got.norm() > 1e-3);
        }
        // The energy of a k-point is the trace of D against the pair.
        let d = density(toy.basis(), 2);
        let c = toy.bloch(Kind::Projection, toy.kpoints()[1]);
        let want = 0.4 * trace_with(&d, &pair_of(&ctx, &chs, &c, &c));
        let got = so.energy(0.4, &c, &d).unwrap();
        assert!(
            (got - want).abs() < 1e-13 && got.abs() > 1e-3,
            "{got:e} {want:e}"
        );
        // The B of the identity is Hermitian and spin-mixing.
        let b = so.operator();
        assert!((b - b.adjoint()).norm() < 1e-14);
        let p = toy.projectors();
        assert!(b.view((0, p), (p, p)).norm() > 1e-3);
    }
}

#[test]
fn the_contraction_equals_the_exact_projection_derivative() {
    let ctx = context();
    for (index, toy) in toys().iter().enumerate() {
        let (chs, so) = (channels(&ctx, toy, &LAMBDA), spin_orbit(&ctx, toy, &LAMBDA));
        let d = density(toy.basis(), 7 + index as u64);
        let want = oracle(&ctx, toy, &chs, &d);
        let gradient = engine(toy, &so, &d);
        let got = gradient.total();
        let error = largest(&got, &want);
        eprintln!(
            "toy {index}: |engine - oracle| = {error:e}, peak {:e}",
            peak(&want)
        );
        assert!(peak(&want) > 1e-3, "non-vacuous");
        assert!(error < GATE, "{error:e}");
        // Both the moving rows and the moving projectors carry weight.
        assert!(peak(&gradient.rows) > 1e-3 && peak(&gradient.columns) > 1e-3);
    }
}

#[test]
fn a_central_difference_of_the_energy_agrees_at_its_measured_floor() {
    let ctx = context();
    let toy = Toy::two_site();
    let (chs, so) = (
        channels(&ctx, &toy, &LAMBDA),
        spin_orbit(&ctx, &toy, &LAMBDA),
    );
    let d = density(toy.basis(), 11);
    let got = engine(&toy, &so, &d).total();
    let (fd, spread) = finite_difference(&ctx, &toy, &chs, &d, 2e-3);
    let error = largest(&got, &fd);
    // The plain central difference misses by `spread` (the h^2 term); the Richardson step
    // leaves the h^4 remainder and the round-off, measured at 1.2e-14: the gate is 100 x that.
    eprintln!("|engine - FD| = {error:e}, spread c(h) - c(2h) = {spread:e}");
    assert!(
        spread > 1e-9,
        "the plain difference has an h^2 term to remove"
    );
    assert!(error < FD_GATE, "{error:e}");
    let negative: Vec<[f64; 3]> = got.iter().map(|g| g.map(|x| -x)).collect();
    assert!(largest(&negative, &fd) > 100. * FD_GATE);
    // The spread is what a plain difference would have missed by: far above the gate.
    assert!(spread > 100. * FD_GATE);
}

#[test]
fn the_atom_sum_is_structurally_zero_and_the_energy_is_translation_invariant() {
    let ctx = context();
    for toy in toys() {
        let so = spin_orbit(&ctx, &toy, &LAMBDA);
        let gradient = engine(&toy, &so, &density(toy.basis(), 3));
        for term in [&gradient.rows, &gradient.columns] {
            // Neither share vanishes alone; it is their sum over the atoms that does.
            let alone: f64 = term
                .iter()
                .map(|g| g[0].abs() + g[1].abs() + g[2].abs())
                .sum();
            assert!(alone > 1e-3);
        }
        // Structural: rows and columns carry opposite signs (`out[a] += t`, `out[b] -= t`), so
        // the atoms sum to zero for ANY `C`, `G` and `D`; this does not test the physics.
        let g = gradient.total();
        for axis in 0..3 {
            let sum: f64 = g.iter().map(|x| x[axis]).sum();
            assert!(sum.abs() < SUM_RULE, "axis {axis}: {sum:e}");
        }
        // The rows alone break the rule: dropping the projector term must show.
        let rows: f64 = (0..3)
            .map(|a| gradient.rows.iter().map(|x| x[a]).sum::<f64>().abs())
            .sum();
        assert!(rows > CONTROL);
        // The physical statement: E_SO is unchanged by a rigid shift of every center.
        let chs = channels(&ctx, &toy, &LAMBDA);
        let d = density(toy.basis(), 3);
        let base = energy(&ctx, &toy, &chs, &d);
        assert!(base.abs() > 1e-3);
        for shift in [[0.4, 0., 0.], [-0.3, 0.7, 0.2], [0., 0., -1.1]] {
            let moved = energy(&ctx, &translated(&toy, shift), &chs, &d);
            assert!(
                (moved - base).abs() < SUM_RULE,
                "{shift:?}: {:e}",
                moved - base
            );
        }
        // A shift of one atom alone is visible: the invariance is not an insensitive energy.
        let one = energy(&ctx, &toy.displaced(0, 0, 0.4), &chs, &d);
        assert!((one - base).abs() > CONTROL);
    }
}

/// `Re tr[D h]` of a Hermitian `h` sees the Hermitian part of `D` only, so the term built from a
/// general `D` equals the oracle (which traces the general `D` against the exact `dh`) and the
/// term built from its Hermitian part.
#[test]
fn only_the_hermitian_part_of_the_density_contributes() {
    let ctx = context();
    for (index, toy) in toys().iter().enumerate() {
        let (chs, so) = (channels(&ctx, toy, &LAMBDA), spin_orbit(&ctx, toy, &LAMBDA));
        let d = general(toy.basis(), 17 + index as u64);
        let part = (&d + d.adjoint()) * C64::new(0.5, 0.);
        let anti = (&d - d.adjoint()) * C64::new(0.5, 0.);
        assert!(anti.norm() > 1e-1, "the anti-Hermitian part is populated");
        let want = oracle(&ctx, toy, &chs, &d);
        let got = engine(toy, &so, &d).total();
        assert!(peak(&want) > 1e-3);
        assert!(largest(&got, &want) < GATE, "{:e}", largest(&got, &want));
        assert!(largest(&got, &engine(toy, &so, &part).total()) < SUM_RULE);
        // The anti-Hermitian part alone gives nothing to the real part.
        assert!(peak(&engine(toy, &so, &anti).total()) < SUM_RULE);
        assert!(peak(&oracle(&ctx, toy, &chs, &anti)) < SUM_RULE);
    }
}

#[test]
fn the_term_is_linear_in_the_coupling_and_zero_without_it() {
    let ctx = context();
    let toy = Toy::three_site();
    let d = density(toy.basis(), 5);
    let at = |scale: f64| {
        let lambda = LAMBDA.map(|x| x * scale);
        engine(&toy, &spin_orbit(&ctx, &toy, &lambda), &d).total()
    };
    let (one, two, none) = (at(1.), at(2.), at(0.));
    let twice: Vec<[f64; 3]> = one.iter().map(|g| g.map(|x| 2. * x)).collect();
    let quadruple: Vec<[f64; 3]> = one.iter().map(|g| g.map(|x| 4. * x)).collect();
    assert!(peak(&one) > 1e-3);
    assert!(largest(&two, &twice) < SUM_RULE);
    // A quadratic dependence (lambda^2) would have given four times.
    assert!(largest(&two, &quadruple) > CONTROL);
    assert_eq!(peak(&none), 0.);
    // The energy itself is linear too, so the force is the gradient of a linear functional.
    let chs = |s: f64| channels(&ctx, &toy, &LAMBDA.map(|x| x * s));
    let e = |s: f64| energy(&ctx, &toy, &chs(s), &d);
    assert!((e(2.) - 2. * e(1.)).abs() < SUM_RULE && e(1.).abs() > 1e-3);
}

fn miss(term: &[[f64; 3]], oracle: &[[f64; 3]]) -> f64 {
    largest(term, oracle)
}

#[test]
fn every_negative_control_misses_the_oracle_by_more_than_one_hundred_gates() {
    let ctx = context();
    let toy = Toy::two_site();
    let (chs, so) = (
        channels(&ctx, &toy, &LAMBDA),
        spin_orbit(&ctx, &toy, &LAMBDA),
    );
    let d = density(toy.basis(), 7);
    let want = oracle(&ctx, &toy, &chs, &d);
    let gradient = engine(&toy, &so, &d);
    assert!(miss(&gradient.total(), &want) < GATE);

    // The projector-moving term dropped.
    assert!(miss(&gradient.rows, &want) > CONTROL);
    // The sign flipped.
    let flipped: Vec<[f64; 3]> = gradient.total().iter().map(|g| g.map(|x| -x)).collect();
    assert!(miss(&flipped, &want) > CONTROL);
    // The projections in place of their center derivative.
    let mut wrong = Gradient::new(toy.atoms());
    for (&k, &w) in toy.kpoints().iter().zip(&toy.weights) {
        let c = toy.bloch(Kind::Projection, k);
        so.add_k(&mut wrong, w, &c, &[c.clone(), c.clone(), c.clone()], &d)
            .unwrap();
    }
    assert!(miss(&wrong.total(), &want) > CONTROL);
    // The transverse spin blocks of the density dropped: only an answer that depends on them.
    let m = toy.basis();
    let mut diagonal = d.clone();
    diagonal.view_mut((0, m), (m, m)).fill(C64::default());
    diagonal.view_mut((m, 0), (m, m)).fill(C64::default());
    assert!(d.view((0, m), (m, m)).norm() > 1e-3);
    assert!(miss(&engine(&toy, &so, &diagonal).total(), &want) > CONTROL);
    // The spin blocks of B transposed (B_st -> B_ts).
    let p = toy.projectors();
    let b = so.operator();
    let swapped = Matrix::from_fn(2 * p, 2 * p, |i, j| {
        b[((j / p) * p + i % p, (i / p) * p + j % p)]
    });
    let transposed = SpinOrbit::new(
        swapped,
        toy.basis_owner(),
        toy.projector_owner(),
        toy.atoms(),
    )
    .unwrap();
    assert!(miss(&engine(&toy, &transposed, &d).total(), &want) > CONTROL);
    // Ownership of the rows taken from the wrong atom.
    let mut owner = toy.basis_owner();
    owner.rotate_left(2);
    let confused = SpinOrbit::new(
        so.operator().clone(),
        owner,
        toy.projector_owner(),
        toy.atoms(),
    )
    .unwrap();
    assert!(miss(&engine(&toy, &confused, &d).total(), &want) > CONTROL);
}

#[test]
fn malformed_input_is_refused() {
    let ctx = context();
    let toy = Toy::two_site();
    let so = spin_orbit(&ctx, &toy, &LAMBDA);
    let mut gradient = Gradient::new(toy.atoms());
    let k = toy.kpoints()[0];
    let c = toy.bloch(Kind::Projection, k);
    let g = row_derivative(&toy, k);
    let d = density(toy.basis(), 1);
    // A density of the wrong size, a projection of the wrong size, an owner out of range.
    assert!(so
        .add_k(&mut gradient, 1., &c, &g, &density(toy.basis() + 1, 1))
        .is_err());
    assert!(so
        .add_k(&mut gradient, 1., &c.rows(0, 2).into_owned(), &g, &d)
        .is_err());
    // An output with one share too short, either of them.
    for short in [0, 1] {
        let mut out = Gradient::new(toy.atoms());
        [&mut out.rows, &mut out.columns][short].pop();
        assert!(so.add_k(&mut out, 1., &c, &g, &d).is_err(), "share {short}");
    }
    assert!(SpinOrbit::new(
        so.operator().clone(),
        toy.basis_owner(),
        vec![9; toy.projectors()],
        toy.atoms()
    )
    .is_err());
    assert!(SpinOrbit::new(
        Matrix::zeros(3, 3),
        toy.basis_owner(),
        toy.projector_owner(),
        toy.atoms()
    )
    .is_err());
}
