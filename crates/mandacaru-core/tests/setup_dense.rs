//! The dense compensation sums at a Bloch shift under a kernel (`periodic::setup::dense`): the
//! frozen Python sum and its `i (G + p)` weighted gradient sums (Al, screened kernel,
//! `omega = 0.11`, two shifts), the bare-Coulomb special case against `couplings`, the
//! Hermiticity of the sum, and the ownership identity `dU/dR_A = out * (delta_aA - delta_bA)`
//! against central differences of the sum itself on the two-atom cells. Needs
//! `MANDACARU_PAW_PATH`.
mod setup_gradients_support;

use mandacaru_core::periodic::setup::{
    dense::{dense_compensation, dense_compensation_gradient},
    electrostatics::{coulomb, couplings},
    shapes::{Shapes, COMPENSATION_CUTOFF},
};
use num_complex::Complex64;
use setup_gradients_support::*;
use std::f64::consts::PI;

const OMEGA: f64 = 0.11;
const SHIFTS: [[f64; 3]; 2] = [[0.13, 0.07, -0.05], [0., 0., 0.3]];

/// `4 pi (1 - exp(-q^2 / 4 omega^2)) / q^2`, `pi / omega^2` at zero.
fn screened(q2: f64) -> f64 {
    if q2 > 1e-12 {
        4. * PI * (1. - (-q2 / (4. * OMEGA * OMEGA)).exp()) / q2
    } else {
        PI / (OMEGA * OMEGA)
    }
}

#[test]
fn the_dense_sum_and_its_gradient_sums_equal_the_frozen_python() {
    for name in ["al", "si_sym"] {
        frozen_dense(name);
    }
}

fn frozen_dense(name: &'static str) {
    let case = Case::load(name);
    let ctx = context();
    let cell = case.cell();
    let sites = cell.sites();
    let shapes = Shapes::new(&ctx, &sites, 20.).unwrap();
    let rows = cell.channels.len();
    for (i, shift) in SHIFTS.iter().enumerate() {
        let u = dense_compensation(
            &ctx,
            &shapes,
            &sites,
            COMPENSATION_CUTOFF,
            *shift,
            &screened,
        )
        .unwrap();
        let key = format!("dense_compensation.shift{i}");
        let miss = relative(&u, &case.quantity(&key).complex());
        let gate = case.gate(&key);
        let g = dense_compensation_gradient(
            &ctx,
            &shapes,
            &sites,
            COMPENSATION_CUTOFF,
            *shift,
            &screened,
        )
        .unwrap();
        let flat: Vec<Complex64> = g.iter().flatten().copied().collect();
        let gkey = format!("dense_compensation_gradient.shift{i}");
        let gmiss = relative(&flat, &case.quantity(&gkey).complex());
        let ggate = case.gate(&gkey);
        println!("{name} shift {i}: sum {miss:.2e} (gate {gate:.1e}), gradient sums {gmiss:.2e} (gate {ggate:.1e})");
        assert!(miss <= gate && gmiss <= ggate);
        // Hermitian, and the shift matters (a shift moved by 0.01 misses by orders).
        let norm = u.iter().map(|x| x.norm()).fold(0., f64::max);
        let skew = (0..rows)
            .flat_map(|a| (0..rows).map(move |b| (a, b)))
            .map(|(a, b)| (u[a * rows + b] - u[b * rows + a].conj()).norm())
            .fold(0., f64::max);
        assert!(skew < 1e-12 * norm);
        let moved = [shift[0] + 0.01, shift[1] + 0.01, shift[2] + 0.01];
        let other =
            dense_compensation(&ctx, &shapes, &sites, COMPENSATION_CUTOFF, moved, &screened)
                .unwrap();
        assert!(relative(&other, &case.quantity(&key).complex()) > 1e3 * gate);
    }
}

#[test]
fn the_bare_coulomb_special_case_is_the_couplings_of_the_setup() {
    let case = Case::load("si");
    let ctx = context();
    let cell = case.cell();
    let sites = cell.sites();
    let shapes = Shapes::new(&ctx, &sites, 20.).unwrap();
    let bare = |q2: f64| coulomb([q2.sqrt(), 0., 0.]);
    let u = dense_compensation(&ctx, &shapes, &sites, COMPENSATION_CUTOFF, [0.; 3], &bare).unwrap();
    let c = couplings(&ctx, &shapes, &sites, COMPENSATION_CUTOFF).unwrap();
    let miss = relative(&u, c.u.as_slice());
    println!("bare Coulomb against couplings: {miss:.2e}");
    assert!(miss < 1e-12);
}

#[test]
fn the_gradient_sums_are_the_difference_of_the_dense_sum_under_ownership() {
    for name in ["si", "al2"] {
        let case = Case::load(name);
        let ctx = context();
        let cell = case.cell();
        let sites = cell.sites();
        let shapes = Shapes::new(&ctx, &sites, 20.).unwrap();
        let rows = cell.channels.len();
        let shift = SHIFTS[0];
        let g = dense_compensation_gradient(
            &ctx,
            &shapes,
            &sites,
            COMPENSATION_CUTOFF,
            shift,
            &screened,
        )
        .unwrap();
        let fd_step = 1e-4;
        let (mut worst, mut scale, mut crude) = (0.0f64, 0.0f64, 0.0f64);
        for atom in 0..cell.atoms.len() {
            for axis in 0..3 {
                let sum = |delta: f64| {
                    let c = cell.moved(atom, axis, delta);
                    dense_compensation(
                        &ctx,
                        &shapes,
                        &c.sites(),
                        COMPENSATION_CUTOFF,
                        shift,
                        &screened,
                    )
                    .unwrap()
                };
                let (plus, minus) = (sum(fd_step), sum(-fd_step));
                for a in 0..rows {
                    for b in 0..rows {
                        let own = |c: usize| f64::from(u8::from(cell.channels[c].0 == atom));
                        let analytic = g[axis][a * rows + b] * (own(a) - own(b));
                        let numeric = (plus[a * rows + b] - minus[a * rows + b]) / (2. * fd_step);
                        worst = worst.max((analytic - numeric).norm());
                        crude = crude.max((g[axis][a * rows + b] - numeric).norm());
                        scale = scale.max(analytic.norm());
                    }
                }
            }
        }
        println!(
            "{name}: ownership identity against central differences {:.2e} of max {scale:.2e}",
            worst / scale
        );
        assert!(scale > 1e-3 && worst < 1e-7 * scale);
        // Control: without the ownership factor the identity is far off.
        assert!(crude > 100. * 1e-7 * scale, "{name}");
    }
}
