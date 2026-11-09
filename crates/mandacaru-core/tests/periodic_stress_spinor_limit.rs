//! AS-8, the zero spin-orbit limit: a scalar-relativistic crystal rewritten as a time-reversal
//! spinor with a zero `h^SO` and zero couplings has the stress of the scalar crystal, term by term.
//!
//! The spinor eigenproblem of a zero coupling is two copies of the scalar one, the doubled frame
//! contracts `D_aa + D_bb` and `W_aa + W_bb`, and the spin-orbit term is exactly zero, so every one
//! of the sixteen terms and the sum must agree with the scalar engine's to rounding. This is the
//! strongest gate of the spinor frame: the spin trace of the overlap, the kinetic, the
//! short-range and the projection terms, the single Pauli channel of a time-reversal local term,
//! the eigenframe xc of the rotated (spin-unpolarized) fields, and `W` built with `blockdiag(L, L)`.
//!
//! The controls are the wrong frames compared with the same scalar stress: `T` (or `S`) acting on the
//! alpha block only, which is half the term. Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;
mod periodic_stress_crystals_support;
mod periodic_stress_spinor_support;

use mandacaru_core::periodic::{
    gradient::Term,
    stress_terms::{contract, Breakdown},
};
use periodic_gradient_crystals_support::{inputs, Data};
use periodic_gradient_spinor_support::Spinor;
use periodic_stress_crystals_support::run::{stress_at, SCF_DENSITY_TOLERANCE, SCF_TOLERANCE};
use periodic_stress_spinor_support::{
    altered::{Altered, Wrong},
    miss, peak, spinor_of, strain_of,
};

/// The plan's gate of a term that must agree to rounding (the engine's own zero-limit gate is 1e-9).
const GATE: f64 = 1e-9;

fn converge(data: &mut Data) -> (f64, mandacaru_core::periodic::gradient::State) {
    data.options.tolerance = SCF_TOLERANCE;
    data.options.density_tolerance = SCF_DENSITY_TOLERANCE;
    let result = data.crystal.solve(&data.options).unwrap();
    assert!(result.converged, "{}: the SCF must converge", data.case);
    (result.free_energy, data.state(&result))
}

#[test]
#[ignore = "slow: 20 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_spinor_limit -- --ignored the_zero_coupling_spinor_stress_is_the_scalar_stress"]
fn the_zero_coupling_spinor_stress_is_the_scalar_stress() {
    let mut scalar = Data::load("lih");
    let (e_scalar, state) = converge(&mut scalar);
    let reference = stress_at(&scalar, &state, &scalar.options);

    let mut twin = Spinor::zero_soc_twin("lih");
    let (e_twin, twin_state) = converge(&mut twin.data);
    assert!(
        (e_twin - e_scalar).abs() < 1e-11,
        "the zero-coupling spinor free energy is the scalar one: {:e}",
        e_twin - e_scalar
    );
    let got = periodic_stress_spinor_support::stress_at(&twin, &twin_state, &twin.data.options);

    let mut worst = 0f64;
    for ((name, a), (_, b)) in got
        .breakdown
        .terms()
        .iter()
        .zip(reference.breakdown.terms().iter())
    {
        let scale = peak(b).max(1e-6);
        let relative = miss(a, b) / scale;
        println!(
            "{name:12} peak {:.3e}  spinor - scalar {:.2e}  (relative {relative:.2e})",
            peak(b),
            miss(a, b)
        );
        assert!(relative < GATE, "{name}: relative miss {relative:.2e}");
        worst = worst.max(relative);
    }
    // The spin-orbit term of a zero coupling is exactly zero (B = 0).
    assert_eq!(peak(&got.breakdown.spin_orbit), 0., "the spin-orbit term");
    let total = miss(&got.gradient, &reference.gradient) / peak(&reference.gradient);
    println!(
        "sum: relative miss {total:.2e} of {:.3e}; worst term {worst:.2e}",
        peak(&reference.gradient)
    );
    assert!(total < GATE, "the sum");
    assert!(
        (got.volume - reference.volume).abs() < 1e-9 * reference.volume,
        "the cell volume"
    );
    // The convention is the scalar one.
    for a in 0..3 {
        for b in 0..3 {
            assert_eq!(
                got.sigma[a][b],
                0.5 * (got.gradient[a][b] + got.gradient[b][a]) / got.volume
            );
        }
    }

    // The controls: the same frozen spinor state with one tangent on the alpha block only.
    let cores = twin.data.cores();
    let c = strain_of(&twin, &cores, &twin_state, &twin.data.options);
    let _ = inputs;
    for term in [Term::Kinetic, Term::Overlap] {
        let wrong = Altered {
            inner: &c.strain,
            wrong: Wrong::AlphaOnly(term),
        };
        let b: Breakdown = contract(&wrong, &c.frozen).unwrap();
        let bite = miss(&b.gradient(), &reference.gradient) / (GATE * peak(&reference.gradient));
        println!("control {term:?} on the alpha block only: bites {bite:.2e} gates");
        assert!(
            bite > 100.,
            "{term:?} on the alpha block only must bite: {bite:.2e}"
        );
    }
    // The control of the controls: the undamaged delegate is the right stress.
    let same = Altered {
        inner: &c.strain,
        wrong: Wrong::Nothing,
    };
    let b = contract(&same, &c.frozen).unwrap();
    assert!(miss(&b.gradient(), &reference.gradient) / peak(&reference.gradient) < GATE);
    let _ = spinor_of;
}
