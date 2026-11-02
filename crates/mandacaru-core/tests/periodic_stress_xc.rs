//! The xc strain terms of the crystal stress (`periodic::stress_terms::xc`, plan X-1, X-3, X-4)
//! on a hand-built node density of Si, Al2 and the skew LiH cell, for a PBE and an LDA
//! functional: the analytic term, all nine `(a, b)` entries unsymmetrized, against the central
//! difference of the xc energy of FIXED node values rebuilt with only the `dv` strained (X-1),
//! only the wave vectors of the spectral gradient strained (X-3) or only the smooth core carried
//! (X-4, the dataset core placed on the strained grid about the strained centers). The xc
//! energy is written from the public point functions (`periodic_stress_support::xc`), the same
//! evaluation that hands the provider its gradients and flux. Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_stress_support;
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;
mod strain_samples_support;
mod strain_support;

use mandacaru_core::periodic::stress_terms::{xc as terms, Strain};
use periodic_stress_support::{
    gates::{bites, gate, map, negated, peak, Tensor},
    oracle, xc, World,
};

const GATE: f64 = 1e-8;
/// The cubic of the core table has third-derivative jumps: the strain of its placement needs the
/// small step (`strain_grid.rs`).
const STEP: f64 = 1e-5;

fn run(case: &'static str, functional: &'static str) {
    let w = World::new(case, &[0], functional);
    let got = w.with_setup(None, |setup, frozen| setup.xc(&frozen.fields).unwrap());
    let vbar = w.fields().xc.potential;
    let mut fd = [[[0f64; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let d = oracle::central(a, b, STEP, |e| oracle::xc_energies(&w, e, &vbar));
            for t in 0..3 {
                fd[t][a][b] = d[t];
            }
            fd[1][a][b] = oracle::xc_wave_derivative(&w, a, b, STEP);
        }
    }
    gate("X-1 jacobian", &got.jacobian, &fd[0], GATE);
    gate("X-4 core", &got.core, &fd[2], GATE);
    if functional == "pbe" {
        gate("X-3 gradient", &got.gradient, &fd[1], GATE);
    } else {
        assert!(
            peak(&fd[1]) < 1e-9 && peak(&got.gradient) == 0.,
            "a local functional has no gradient term (oracle {:.2e})",
            peak(&fd[1])
        );
    }
    // Controls against the oracle term they damage.
    bites("jacobian dropped", &[[0.; 3]; 3], &fd[0], GATE);
    bites("core sign", &negated(&got.core), &fd[2], GATE);
    if functional == "pbe" {
        bites(
            "no GGA gradient term (LDA on a PBE crystal)",
            &[[0.; 3]; 3],
            &fd[1],
            GATE,
        );
        bites("gradient sign", &negated(&got.gradient), &fd[1], GATE);
        // A restricted run still has two rotated spin channels: counting one is half.
        bites(
            "one spin channel",
            &map(&got.gradient, |_, _, x| 0.5 * x),
            &fd[1],
            GATE,
        );
        // The gradient of the valence alone (the core omitted from n_s).
        let (ctx, grid) = (w.ctx(), w.grid());
        let valence = xc::evaluate(
            ctx,
            grid,
            &xc::wave(ctx, grid),
            grid.index.dv(),
            functional,
            w.data.crystal.relativistic,
            &w.rho,
            &vec![0.; w.rho.len()],
        );
        let wrong: Tensor = terms::gradient(
            grid.index.dv(),
            w.rho.len(),
            valence.gradients.as_deref(),
            valence.flux.as_deref(),
        )
        .unwrap();
        bites("gradient of the valence only", &wrong, &fd[1], GATE);
    }
}

#[test]
fn the_xc_terms_of_a_gga_si() {
    run("si", "pbe");
}

#[test]
fn the_xc_terms_of_a_gga_al2() {
    run("al2", "pbe");
}

#[test]
fn the_xc_terms_of_a_gga_lih() {
    run("lih", "pbe");
}

#[test]
fn the_xc_terms_of_a_local_functional_lih() {
    run("lih", "lda");
}

/// The potential the core term contracts with is the functional derivative of the energy it
/// differences: `dE = dv sum_i vbar_i d core_i` for a perturbation of the core nodes (the
/// divergence term of a GGA included), by a Richardson pair of central differences.
#[test]
fn the_xc_potential_is_the_functional_derivative_of_the_energy() {
    for (case, functional) in [("lih", "pbe"), ("si", "pbe"), ("lih", "lda")] {
        let w = World::new(case, &[0], functional);
        let (ctx, grid) = (w.ctx(), w.grid());
        let vbar = w.fields().xc.potential;
        let core0 = w.core(grid, &w.centers());
        let delta: Vec<f64> = (0..core0.len())
            .map(|j| 0.01 * ((0.05 * j as f64).sin() + 0.3 * (0.011 * j as f64).cos()))
            .collect();
        let wave = xc::wave(ctx, grid);
        let energy = |t: f64| {
            let core: Vec<f64> = core0.iter().zip(&delta).map(|(c, d)| c + t * d).collect();
            xc::evaluate(
                ctx,
                grid,
                &wave,
                grid.index.dv(),
                functional,
                w.data.crystal.relativistic,
                &w.rho,
                &core,
            )
            .energy
        };
        let step = 1e-3;
        let one = (energy(step) - energy(-step)) / (2. * step);
        let two = (energy(2. * step) - energy(-2. * step)) / (4. * step);
        let fd = (4. * one - two) / 3.;
        let analytic = grid.index.dv() * vbar.iter().zip(&delta).map(|(v, d)| v * d).sum::<f64>();
        let miss = (fd - analytic).abs() / analytic.abs().max(1e-6);
        println!("{case} {functional}: dE {analytic:.9e}, difference {fd:.9e}, miss {miss:.2e}");
        assert!(
            miss < 1e-8,
            "{case} {functional}: the xc potential is not the derivative of the energy"
        );
    }
}
