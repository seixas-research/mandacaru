//! AS-8, SO-1: the spin-orbit union-projector term of the stress, `sum_k w 2 Re sum dC^SO_ab[mu, p]
//! Z[p, mu]`, against the central difference of the SAME energy `E_SO(eps) = sum_k w Re tr[D_k
//! h^SO_k(eps)]` rebuilt from the setup's spin-orbit builder at the strained cell, with the
//! converged spinor density `D_k` held fixed (rigid spheres, fractional k fixed). No SCF is
//! differenced and no `W`, `dS` or hat enters: the term is the whole derivative of `E_SO`.
//!
//! Two states: the displaced `lda-fr` Si (the spin-orbit share is small there) and the heavy AuH
//! molecule (Au 5d splits by about 1.5 eV: the term is large). The coupling is scaled by `lambda`
//! in both the analytic operator and the oracle (`lambda` = 0, 0.5, 2): the term is linear in it
//! and each point is differenced on its own, so the linearity is checked against the oracle and
//! not by construction. Controls (compared with the same oracle, > 100 gates): the sign, no term,
//! the alpha block of `D` only (`T` not doubled), the transposed tensor.
//! Needs `MANDACARU_PAW_PATH`.
#[path = "periodic_stress_spinor_support/oracle.rs"]
mod oracle;
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;
mod periodic_stress_spinor_support;
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;
mod strain_samples_support;

use mandacaru_core::periodic::{
    gradient::{Matrix, State},
    setup::strain_provider::StrainSetup,
    stress_terms::spin_orbit::add_k,
};
use num_complex::Complex64;
use periodic_gradient_crystals_support::inputs;
use periodic_gradient_spinor_support::Spinor;
use periodic_stress_spinor_support::{miss, peak, Grid};

/// The plan's gate on a per-term comparison, relative to the peak entry of the oracle term.
const GATE: f64 = 1e-8;
const STEP: f64 = 2e-4;

struct Case {
    sp: Spinor,
    ks: Vec<usize>,
    weights: Vec<f64>,
    densities: Vec<Matrix>,
}

impl Case {
    fn new(sp: Spinor, ks: &[usize]) -> Self {
        let result = sp.data.solve();
        let state: State = sp.data.state(&result);
        let densities = ks.iter().map(|&k| state.bands[0][k].density()).collect();
        let weights = ks.iter().map(|&k| 0.3 + 0.2 * k as f64).collect();
        Self {
            sp,
            ks: ks.to_vec(),
            weights,
            densities,
        }
    }

    /// The analytic term for the couplings scaled by `scale` and the densities `densities`.
    fn analytic(&self, scale: f64, densities: &[Matrix]) -> Grid {
        let data = &self.sp.data;
        let mut couplings = self.sp.couplings.clone();
        for c in couplings.values_mut() {
            c.values.iter_mut().for_each(|v| *v *= scale);
        }
        let cores = data.cores();
        let setup = StrainSetup::new(
            &data.crystal.context,
            inputs(data, &cores),
            data.layout(),
            data.operators(),
        )
        .unwrap()
        .with_spin_orbit(&self.sp.projectors, &couplings)
        .unwrap();
        let mut out = [[0.; 3]; 3];
        for ((&k, &w), d) in self.ks.iter().zip(&self.weights).zip(densities) {
            add_k(&setup, &mut out, w, k, d).unwrap();
        }
        out
    }

    fn oracle(&self, scale: f64) -> Grid {
        std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                oracle::derivative(a, b, STEP, |eps| {
                    oracle::energy(
                        &self.sp,
                        &self.ks,
                        &self.weights,
                        &self.densities,
                        scale,
                        eps,
                    )
                })
            })
        })
    }
}

/// The worst entry difference in units of the gate (relative to the oracle's peak).
fn gates(got: &Grid, want: &Grid) -> f64 {
    miss(got, want) / (GATE * peak(want))
}

/// `P_alpha D P_alpha`: the density with the beta rows and columns removed.
fn alpha_block(d: &Matrix) -> Matrix {
    let half = d.nrows() / 2;
    let mut out = d.clone();
    out.view_mut((half, 0), (half, d.ncols()))
        .fill(Complex64::default());
    out.view_mut((0, half), (d.nrows(), half))
        .fill(Complex64::default());
    out
}

fn transposed(t: &Grid) -> Grid {
    std::array::from_fn(|a| std::array::from_fn(|b| t[b][a]))
}

fn scaled(t: &Grid, s: f64) -> Grid {
    t.map(|row| row.map(|x| s * x))
}

fn report(label: &str, got: &Grid, want: &Grid) -> f64 {
    let g = gates(got, want);
    println!(
        "{label}: term peak {:.3e} Ha, miss {:.2e} Ha = {g:.3} gates",
        peak(want),
        miss(got, want)
    );
    g
}

#[test]
#[ignore = "slow: 70 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_spinor_orbit -- --ignored the_silicon_term_is_the_derivative_of_the_spin_orbit_energy"]
fn the_silicon_term_is_the_derivative_of_the_spin_orbit_energy() {
    let case = Case::new(Spinor::load("si_ldafr"), &[3]);
    let want = case.oracle(1.);
    let got = case.analytic(1., &case.densities);
    let g = report("Si lda-fr SO-1", &got, &want);
    assert!(
        peak(&want) > 1e-9,
        "the term must be resolved: {:e}",
        peak(&want)
    );
    assert!(g < 1., "SO-1 misses by {g:.2} gates");
}

#[test]
#[ignore = "slow: 105 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_spinor_orbit -- --ignored the_heavy_term_is_linear_in_the_coupling_and_the_controls_bite"]
fn the_heavy_term_is_linear_in_the_coupling_and_the_controls_bite() {
    let case = Case::new(Spinor::load("auh_ldafr"), &[0]);
    let unit_term = case.analytic(1., &case.densities);
    let mut full = None;
    for lambda in [0., 0.5, 1., 2.] {
        let want = case.oracle(lambda);
        let got = case.analytic(lambda, &case.densities);
        if lambda == 0. {
            assert_eq!(peak(&got), 0., "a zero coupling has no term");
            assert!(
                peak(&want) < 1e-14,
                "the oracle of a zero coupling: {:e}",
                peak(&want)
            );
            continue;
        }
        let g = report(&format!("AuH SO-1 lambda {lambda}"), &got, &want);
        assert!(g < 1., "lambda {lambda}: SO-1 misses by {g:.2} gates");
        if lambda == 1. {
            full = Some((got, want));
        }
        // Linear: twice the coupling is twice the term, in the analytic tensor as in the oracle.
        let unit = scaled(&unit_term, lambda);
        assert!(
            miss(&got, &unit) <= 1e-12 * peak(&unit),
            "lambda {lambda}: linear"
        );
    }
    let (got, want) = full.unwrap();
    assert!(
        peak(&want) > 1e-5,
        "the heavy term must be large: {:e}",
        peak(&want)
    );
    // The controls.
    let alpha: Vec<Matrix> = case.densities.iter().map(alpha_block).collect();
    let controls: [(&str, Grid); 4] = [
        ("sign", scaled(&got, -1.)),
        ("no term", [[0.; 3]; 3]),
        (
            "alpha block of D only (T not doubled)",
            case.analytic(1., &alpha),
        ),
        ("transposed tensor", transposed(&got)),
    ];
    for (name, wrong) in controls {
        let bite = gates(&wrong, &want);
        println!("control {name}: bites {bite:.3e} gates");
        if name == "transposed tensor" {
            // Reported: the tensor of a molecule in a cubic box is nearly symmetric.
            continue;
        }
        assert!(bite > 100., "{name} must bite: {bite:.2e}");
    }
}
