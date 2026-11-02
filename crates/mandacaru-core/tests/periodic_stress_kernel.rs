//! The strain derivative of the screened Coulomb kernel of the exact exchange
//! (`periodic::stress::screened_kernel_strain`): the kernel the exchange evaluates for a pair
//! `(k, q)` is `v^omega` of the alias-MEAN square of `k_diff + G` (and, at the head, a
//! Gygi-Baldereschi average over the pair's own weight), so its derivative is checked against
//! the central difference of THAT function over a strained grid at fixed fractional `k_diff`,
//! with the head's cell volume strained too. Needs `MANDACARU_PAW_PATH`.
mod strain_support;

use mandacaru_core::{
    math::reciprocal,
    periodic::{
        screened_exchange::{gygi_baldereschi_screened_kernel, screened_kernel, ExxDivTreatment},
        stress::{kernel_strain_from_moments, screened_kernel_strain},
    },
};
use nalgebra::Vector3;
use strain_support::*;

const OMEGA: f64 = 0.11;
const GATE: f64 = 1e-9;
const VOIGT: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];

/// The kernel `kernel_weights` evaluates before its weight factor: `v^omega(squares)`, the head
/// `v_bar(q_c)` or the point limit at `squares <= 1e-12`.
fn kernel_of(cell: &Cell, fractional: [f64; 3], weight: f64, exxdiv: ExxDivTreatment) -> Vec<f64> {
    let ctx = context();
    let n = cell.grid.index.points();
    let b = reciprocal::reciprocal_cell(&cell.grid).unwrap();
    let k: [f64; 3] = (b * Vector3::from(fractional)).into();
    let squares = reciprocal::bloch(&ctx, &cell.grid, k).unwrap();
    let volume = cell.grid.index.dv() * n as f64;
    squares
        .squares
        .as_slice()
        .iter()
        .map(|&q2| {
            if q2 > 1e-12 {
                screened_kernel(q2, OMEGA)
            } else {
                match exxdiv {
                    ExxDivTreatment::GygiBaldereschi => gygi_baldereschi_screened_kernel(
                        (6. * std::f64::consts::PI.powi(2) * weight / volume).cbrt(),
                        OMEGA,
                    ),
                    ExxDivTreatment::None => screened_kernel(0., OMEGA),
                }
            }
        })
        .collect()
}

fn strain_of(cell: &Cell, k_diff: [f64; 3], weight: f64, exxdiv: ExxDivTreatment) -> Vec<f64> {
    let ctx = context();
    let b = reciprocal::reciprocal_cell(&cell.grid).unwrap();
    let k: [f64; 3] = (b * Vector3::from(k_diff)).into();
    screened_kernel_strain(&ctx, &cell.grid, k, weight, OMEGA, exxdiv)
        .unwrap()
        .as_slice()
        .to_vec()
}

#[test]
fn the_kernel_strain_is_the_difference_of_the_kernel_through_the_alias_averaged_symbol() {
    let ctx = context();
    for name in ["si", "al2", "lih"] {
        let cell = Cell::load(&ctx, name);
        let n = cell.grid.index.points();
        let nk = 4.;
        // Gamma (the head is on the grid) and a generic fractional difference.
        for (fractional, label) in [([0.; 3], "Gamma"), ([0.137, -0.291, 0.0623], "generic")] {
            for exxdiv in [ExxDivTreatment::None, ExxDivTreatment::GygiBaldereschi] {
                // A pair weight that is NOT 1 / nk, as on a reduced mesh.
                let weight = 0.3;
                let got = strain_of(&cell, fractional, weight, exxdiv);
                let (mut miss, mut scale_all) = (0f64, 0f64);
                let (mut mean_vector, mut uniform_weight, mut flipped) = (0f64, 0f64, 0f64);
                let uniform = strain_of(&cell, fractional, 1. / nk, exxdiv);
                let b = reciprocal::reciprocal_cell(&cell.grid).unwrap();
                let k: [f64; 3] = (b * Vector3::from(fractional)).into();
                let symbols = reciprocal::bloch(&ctx, &cell.grid, k).unwrap();
                for (row, &(a, c)) in VOIGT.iter().enumerate() {
                    let side = |h: f64| {
                        let strained = cell.strained(entry(a, c, h));
                        assert_pinned(&ctx, &cell, &strained);
                        kernel_of(&strained, fractional, weight, exxdiv)
                    };
                    let difference = |h: f64| -> Vec<f64> {
                        side(h)
                            .iter()
                            .zip(side(-h))
                            .map(|(p, m)| (p - m) / (2. * h))
                            .collect()
                    };
                    let (one, two) = (difference(1e-4), difference(2e-4));
                    let want: Vec<f64> = one
                        .iter()
                        .zip(&two)
                        .map(|(x, y)| (4. * x - y) / 3.)
                        .collect();
                    let have = &got[row * n..(row + 1) * n];
                    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
                    miss = miss.max(worst(have, &want, &|_| false) / scale);
                    scale_all = scale_all.max(scale);
                    flipped = flipped.max(
                        have.iter()
                            .zip(&want)
                            .map(|(x, y)| (-x - y).abs())
                            .fold(0., f64::max)
                            / scale,
                    );
                    uniform_weight = uniform_weight.max(
                        uniform[row * n..(row + 1) * n]
                            .iter()
                            .zip(&want)
                            .map(|(x, y)| (x - y).abs())
                            .fold(0., f64::max)
                            / scale,
                    );
                    // The outer product of the MEAN vector in place of the mean of products.
                    let wrong: Vec<f64> = (0..n)
                        .map(|at| {
                            let q = [0, 1, 2].map(|d| symbols.vectors.as_slice()[d * n + at]);
                            let outer = [
                                q[0] * q[0],
                                q[1] * q[1],
                                q[2] * q[2],
                                q[1] * q[2],
                                q[0] * q[2],
                                q[0] * q[1],
                            ];
                            let qc = (6. * std::f64::consts::PI.powi(2) * weight
                                / (cell.grid.index.dv() * n as f64))
                                .cbrt();
                            kernel_strain_from_moments(
                                q.iter().map(|v| v * v).sum::<f64>(),
                                outer,
                                OMEGA,
                                qc,
                                exxdiv,
                            )[row]
                        })
                        .collect();
                    mean_vector = mean_vector.max(worst(&wrong, &want, &|_| false) / scale);
                }
                println!(
                    "{name} {label} {exxdiv:?}: dv {miss:.2e} of max {scale_all:.2e}; controls: sign {flipped:.2e}, mean vectors {mean_vector:.2e}, weight 1/nk {uniform_weight:.2e}"
                );
                assert!(miss < GATE, "{name} {label} {exxdiv:?}");
                assert!(flipped > 100. * GATE, "{name}");
                if label == "Gamma" {
                    // The faces alias at Gamma, so the mean-vector symbol is wrong there.
                    assert!(
                        mean_vector > 100. * GATE,
                        "{name}: the mean-vector control is vacuous"
                    );
                    // The head carries the pair weight only for Gygi-Baldereschi.
                    if exxdiv == ExxDivTreatment::GygiBaldereschi {
                        assert!(uniform_weight > 100. * GATE, "{name}");
                    }
                }
            }
        }
    }
}
