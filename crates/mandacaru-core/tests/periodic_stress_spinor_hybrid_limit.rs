//! AS-10, the zero spin-orbit limit of the HYBRID stress: the scalar HSE06 crystal `lih_hse06`
//! rewritten as a time-reversal spinor with a zero `h^SO` and zero couplings has the stress of the
//! scalar crystal of AS-9, term by term, the seventeen terms of the breakdown and the eight
//! pieces of the exchange (a spinor eigenproblem of zero coupling is two copies of the scalar
//! one, the spin-summed pair density of the `2m` rows is the scalar pair density, and the exchange
//! response `Z` of the spinor is the scalar one). Two SCFs, two stresses; needs
//! `MANDACARU_PAW_PATH`.
//!
//! The control compares a WRONG spin structure with the same scalar stress: the exchange of the
//! spinor with the second spin component's rows dropped (the energy and every term of a
//! single-component pass are half of the spin-summed ones for a spin-degenerate state).
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;
mod periodic_stress_crystals_support;
mod periodic_stress_spinor_support;

use periodic_gradient_crystals_support::Data;
use periodic_gradient_spinor_support::Spinor;
use periodic_stress_crystals_support::run::{stress_at, SCF_DENSITY_TOLERANCE, SCF_TOLERANCE};
use periodic_stress_spinor_support::{miss, peak};

/// The plan's gate of a term that must agree to rounding.
const GATE: f64 = 1e-9;

fn converge(data: &mut Data) -> (f64, mandacaru_core::periodic::gradient::State) {
    data.options.tolerance = SCF_TOLERANCE;
    data.options.density_tolerance = SCF_DENSITY_TOLERANCE;
    let result = data.crystal.solve(&data.options).unwrap();
    assert!(result.converged, "{}: the SCF must converge", data.case);
    (result.free_energy, data.state(&result))
}

#[test]
fn the_zero_coupling_spinor_hybrid_stress_is_the_scalar_hybrid_stress() {
    let mut scalar = Data::load("lih_hse06");
    let (e_scalar, state) = converge(&mut scalar);
    let reference = stress_at(&scalar, &state, &scalar.options);

    let mut twin = Spinor::zero_soc_twin("lih_hse06");
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
        let relative = miss(a, b) / peak(b).max(1e-6);
        println!(
            "{name:22} peak {:.3e}  spinor - scalar {:.2e}  (relative {relative:.2e})",
            peak(b),
            miss(a, b)
        );
        assert!(relative < GATE, "{name}: relative miss {relative:.2e}");
        worst = worst.max(relative);
    }
    // The pieces of the exchange, each against its scalar counterpart, and each live.
    let (g, r) = (
        got.breakdown.exchange_parts.as_ref().unwrap(),
        reference.breakdown.exchange_parts.as_ref().unwrap(),
    );
    for ((name, a), (_, b)) in g.parts().iter().zip(r.parts().iter()) {
        let relative = miss(a, b) / peak(b).max(1e-6);
        println!(
            "{name:22} peak {:.3e}  spinor - scalar {:.2e}  (relative {relative:.2e})",
            peak(b),
            miss(a, b)
        );
        assert!(peak(b) > 1e-5, "{name} is live in the scalar reference");
        assert!(relative < GATE, "{name}: relative miss {relative:.2e}");
    }
    assert!(
        (g.energy - r.energy).abs() < 1e-9 * r.energy.abs(),
        "the pass energy"
    );
    assert_eq!(peak(&got.breakdown.spin_orbit), 0., "the spin-orbit term");
    let total = miss(&got.gradient, &reference.gradient) / peak(&reference.gradient);
    println!(
        "sum: relative miss {total:.2e} of {:.3e}; worst term {worst:.2e}",
        peak(&reference.gradient)
    );
    assert!(total < GATE, "the sum");
    // The wrong spin structure: the second component's rows dropped halve every exchange term
    // of a spin-degenerate state; the stress with half the exchange misses by the other half.
    let exchange = peak(&reference.breakdown.exchange);
    let half: Vec<f64> = (0..9)
        .map(|i| {
            let (a, b) = (i / 3, i % 3);
            reference.gradient[a][b] - 0.5 * reference.breakdown.exchange[a][b]
        })
        .collect();
    let wrong_miss = (0..9)
        .map(|i| (half[i] - got.gradient[i / 3][i % 3]).abs())
        .fold(0f64, f64::max);
    let bite = wrong_miss / (GATE * peak(&reference.gradient));
    println!("control one spin component of the exchange only: bites {bite:.2e} gates (exchange peak {exchange:.3e})");
    assert!(bite > 100., "{bite:e}");
}
