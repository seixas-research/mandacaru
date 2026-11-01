//! `models::hubbard_response::states`: the bare response as a sum over states against the
//! finite difference of the frozen problem at fixed electron number.
mod hubbard_response_support;

use hubbard_response_support::*;
use mandacaru_core::models::hubbard_response::{sum_over_states, StateBlock};
use mandacaru_core::Error;
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

// ---- sum over states ---------------------------------------------------------------

/// Occupation of site `s` at fixed `electrons` and `kt`, Fermi level re-found.
fn site_occupation(h: &[f64], n: usize, electrons: f64, kt: f64, site: usize) -> f64 {
    let (e, v) = jacobi(h, n);
    let (f, _) = fermi_dirac(&e, electrons, kt);
    2. * (0..n).map(|k| f[k] * v[k * n + site].powi(2)).sum::<f64>()
}

fn frozen_difference(electrons: f64, kt: f64, sites: &[usize]) -> DMatrix<f64> {
    let n = 12;
    let a = 1e-4;
    let mut out = DMatrix::zeros(sites.len(), sites.len());
    for (j, &sj) in sites.iter().enumerate() {
        let shifted = |sign: f64| {
            let mut h = clean_ring(n);
            h[sj * n + sj] += sign * a;
            h
        };
        for (i, &si) in sites.iter().enumerate() {
            let plus = site_occupation(&shifted(1.), n, electrons, kt, si);
            let minus = site_occupation(&shifted(-1.), n, electrons, kt, si);
            out[(i, j)] = (plus - minus) / (2. * a);
        }
    }
    out
}

fn sum_for(f: &[f64], fp: &[f64], sites: &[usize]) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = 12;
    let (e, v) = jacobi(&clean_ring(n), n);
    let elements: Vec<Vec<C64>> = sites
        .iter()
        .map(|&s| {
            (0..n * n)
                .map(|x| C64::new(v[(x / n) * n + s] * v[(x % n) * n + s], 0.))
                .collect()
        })
        .collect();
    let slices: Vec<&[C64]> = elements.iter().map(|m| m.as_slice()).collect();
    let block = StateBlock {
        weight: 1.,
        levels: &e,
        occupations: f,
        slopes: fp,
        elements: &slices,
    };
    let sum = sum_over_states(&[block], 2.).unwrap();
    (sum.total(), sum.pairs)
}

fn states(electrons: f64, kt: f64, sites: &[usize]) -> (DMatrix<f64>, DMatrix<f64>) {
    let (e, _) = jacobi(&clean_ring(12), 12);
    let (f, fp) = fermi_dirac(&e, electrons, kt);
    sum_for(&f, &fp, sites)
}

#[test]
fn the_sum_over_states_matches_the_frozen_difference_and_the_fermi_term_is_needed() {
    let sites = [0, 1, 5];
    let reference = frozen_difference(12., 0.15, &sites);
    let (total, pairs) = states(12., 0.15, &sites);
    let with = worst(&total, &reference);
    let without = worst(&pairs, &reference);
    println!("metal: with Fermi term {with:.2e}, without {without:.3}");
    assert!(with < 1e-6, "sum over states vs frozen difference: {with}");
    assert!(
        without > 0.05,
        "dropping the Fermi-level term is the failing control: {without}"
    );
    assert!(worst(&total, &total.transpose()) < 1e-14);
}

#[test]
fn an_insulator_at_tiny_width_skips_the_fermi_term_instead_of_dividing_by_zero() {
    let sites = [0, 3];
    let reference = frozen_difference(10., 5e-4, &sites);
    // Zero-width limit: five doubly occupied orbitals, every slope exactly zero.
    let f: Vec<f64> = (0..12).map(|k| if k < 5 { 1. } else { 0. }).collect();
    let (total, pairs) = sum_for(&f, &[0.; 12], &sites);
    assert!(total.iter().all(|v| v.is_finite()));
    assert_eq!(
        total, pairs,
        "the sum of slopes underflows to zero: no Fermi term"
    );
    assert!(worst(&total, &reference) < 1e-6);
}

#[test]
fn inconsistent_state_blocks_are_refused() {
    let (levels, occupations, slopes) = ([0., 1.], [1., 0.], [0., 0.]);
    let m = [C64::new(1., 0.); 4];
    let one = [&m[..]];
    let base = StateBlock {
        weight: 1.,
        levels: &levels,
        occupations: &occupations,
        slopes: &slopes,
        elements: &one,
    };
    let short = [&m[..3]];
    let mut bad = [
        StateBlock {
            elements: &short,
            ..base
        },
        StateBlock {
            weight: -1.,
            ..base
        },
        StateBlock {
            occupations: &occupations[..1],
            ..base
        },
    ];
    for block in bad.iter_mut() {
        assert!(matches!(
            sum_over_states(&[*block], 2.),
            Err(Error::Invalid(_))
        ));
    }
    assert!(matches!(
        sum_over_states(&[base], 0.),
        Err(Error::Invalid(_))
    ));
}
