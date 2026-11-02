//! The unit-charge sum rule of the screened pair integrals on a real dataset.
//!
//! Every augmented pair density of the orthonormal basis carries the charge
//! `delta_ab`, so as the screening vanishes its long-range interaction is the
//! constant `2 omega / sqrt(pi)` of the kernel times `delta_ab delta_cd`: the grid
//! pairs, the short-range compensation charges and the one-center tensor must
//! together lose exactly that between two small screenings, on every channel the
//! basis carries. `paw_molecular_screened` checks it on a synthetic species with
//! s and p waves; this file runs it on the oxygen dataset of the shipped library
//! (s, p and d basis functions), the case where the first-row p shell matters.
//! The library is `$MANDACARU_PAW_PATH`; a missing one FAILS the test.
mod paw_molecular_support;

use mandacaru_core::{paw::molecular::Arrays, Context};
use paw_molecular_support::{analytic_shells, dataset, grid_pairs, molecule};
use std::f64::consts::PI;

/// The angular momentum of each function of an atom with s, p and d shells.
fn shells() -> Vec<usize> {
    [0, 1, 1, 1, 2, 2, 2, 2, 2].to_vec()
}

#[test]
fn the_screened_pair_integrals_of_oxygen_lose_the_unit_charge_long_range_part() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let oxygen = dataset::read(&ctx, "pbe-sr", "O");
    println!(
        "O pbe-sr: channels {:?}, compensation radius {}",
        oxygen.channels.keys().collect::<Vec<_>>(),
        oxygen.compensation_radius
    );
    let l_of = shells();
    for kind in ["dual", "raw"] {
        let mut arrays = Arrays::new();
        let species = dataset::species(&ctx, &mut arrays, &oxygen);
        let base = molecule(
            &ctx,
            &[8],
            &[[0., 0., 0.]],
            12.,
            0.45,
            (vec![species], vec![0], arrays),
            kind,
            &analytic_shells(&[0, 1, 2]),
        );
        let setup = base.paw_setup.as_ref().unwrap();
        let m = base.m();
        assert_eq!(m, l_of.len());
        let (w1, w2) = (0.005, 0.01);
        let total = |omega: f64| -> Vec<f64> {
            let mut grid = grid_pairs(&ctx, &base, omega);
            let screened = setup.screened_exchange(&base, omega).unwrap();
            for a in 0..m {
                for b in 0..m {
                    for c in 0..m {
                        for d in 0..m {
                            grid[((a * m + b) * m + c) * m + d] += screened.pairs.value(a, b, c, d);
                        }
                    }
                }
            }
            grid
        };
        let (first, second) = (total(w1), total(w2));
        let unit = 2. * (w2 - w1) / PI.sqrt();
        // The worst deviation by the angular momenta of the pair (a, b).
        let mut worst = [[0f64; 3]; 3];
        for a in 0..m {
            for b in 0..m {
                for c in 0..m {
                    for d in 0..m {
                        let i = ((a * m + b) * m + c) * m + d;
                        let expected = if a == b && c == d { unit } else { 0. };
                        let slot = &mut worst[l_of[a].max(l_of[b])][l_of[c].max(l_of[d])];
                        *slot = slot.max((first[i] - second[i] - expected).abs());
                    }
                }
            }
        }
        println!("{kind}: unit {unit:.3e}; worst deviation by (max l of ab, max l of cd):");
        for (l, row) in worst.iter().enumerate() {
            println!("  l_ab {l}: {:?}", row.map(|v| format!("{v:.3e}")));
        }
        for row in worst {
            for value in row {
                assert!(value < 2e-5, "{kind}: {value:e}");
            }
        }
    }
    drop(ctx);
}
