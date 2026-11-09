//! A screened hybrid on an open-shell PAW molecule: the spheres' semilocal
//! short-range exchange follows the spin densities (`paw::screened_polarization`),
//! so the unrestricted energy must (a) not depend on the spin labels, (b) equal
//! the restricted one for a closed shell, and (c) have forces equal to the finite
//! difference of the energy, which only holds if the polarization's Fock matrix
//! is the derivative of its energy. All on the synthetic hydrogen-like species.
mod paw_molecular_support;

use mandacaru_core::{molecular::ScfOptions, Context, Geometry, LengthUnit};
use paw_molecular_support::{integrals, H2};

fn options() -> ScfOptions {
    ScfOptions {
        tolerance: 1e-11,
        max_iterations: 300,
        guesses: 1,
        ..Default::default()
    }
}

#[test]
#[ignore = "slow: 11 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_molecular_spin -- --ignored an_unrestricted_closed_shell_is_the_restricted_one_and_spin_labels_do_not_matter"]
fn an_unrestricted_closed_shell_is_the_restricted_one_and_spin_labels_do_not_matter() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let base = integrals(&ctx, &H2, "dual");
    let restricted = base.kohn_sham("hse06", [1, 1], false, &options()).unwrap();
    let unrestricted = base.kohn_sham("hse06", [1, 1], true, &options()).unwrap();
    assert!(restricted.converged && unrestricted.converged);
    println!(
        "closed shell: restricted {:.12}, unrestricted {:.12}",
        restricted.electronic_energy, unrestricted.electronic_energy
    );
    assert!((restricted.electronic_energy - unrestricted.electronic_energy).abs() < 1e-9);
    let up = base.kohn_sham("hse06", [1, 0], true, &options()).unwrap();
    let down = base.kohn_sham("hse06", [0, 1], true, &options()).unwrap();
    assert!(up.converged && down.converged);
    println!(
        "one electron: up {:.12}, down {:.12}",
        up.electronic_energy, down.electronic_energy
    );
    assert!((up.electronic_energy - down.electronic_energy).abs() < 1e-10);
    drop((restricted, unrestricted, up, down, base));
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_forces_of_an_open_shell_hybrid_are_the_finite_difference_of_the_energy() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let base = integrals(&ctx, &H2, "dual");
    let state = base.kohn_sham("hse06", [1, 0], true, &options()).unwrap();
    assert!(state.converged);
    let gradient = base.gradient(&state).unwrap();
    let energy = |delta: f64| {
        let mut positions = base.geometry.positions().to_vec();
        positions[1][2] += delta;
        let geometry = Geometry::new(
            base.geometry.numbers().to_vec(),
            positions,
            *base.geometry.cell(),
            base.geometry.pbc(),
            LengthUnit::Bohr,
        )
        .unwrap();
        let moved = base.at_geometry(geometry).unwrap();
        let state = moved.kohn_sham("hse06", [1, 0], true, &options()).unwrap();
        assert!(state.converged);
        state.electronic_energy + moved.nuclear
    };
    let h = 1e-3;
    let coarse = (energy(h) - energy(-h)) / (2. * h);
    let fine = (energy(0.5 * h) - energy(-0.5 * h)) / h;
    let reference = (4. * fine - coarse) / 3.;
    let analytic = gradient.total.as_slice()[5];
    let to_ev = 27.211386245988 / 0.529177210903;
    println!(
        "H2+ HSE06 PAW force: analytic {analytic:.8e}, finite difference {reference:.8e} Ha/Bohr"
    );
    assert!(analytic.abs() > 1e-4, "the force is not trivially zero");
    assert!(
        (analytic - reference).abs() * to_ev < 1e-5,
        "{analytic:e} vs {reference:e}"
    );
    drop((gradient, state, base));
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_exchange_pieces_agree_between_restricted_and_unrestricted_bookkeeping() {
    // The same closed-shell density as one restricted matrix and as two equal
    // spin matrices: every piece is the same, and the polarization vanishes.
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let base = integrals(&ctx, &H2, "dual");
    let state = base.kohn_sham("hse06", [1, 1], false, &options()).unwrap();
    let one = &state.densities[0];
    let restricted = base
        .exchange_pieces("hse06", std::slice::from_ref(one))
        .unwrap()
        .unwrap();
    let unrestricted = base
        .exchange_pieces("hse06", &[one.clone(), one.clone()])
        .unwrap()
        .unwrap();
    println!("pieces: {restricted:?}\ntotal {:.9}", restricted.total());
    assert!(restricted.grid < 0. && restricted.constant.is_finite());
    assert_eq!(restricted.polarization, 0.);
    assert!(unrestricted.polarization.abs() < 1e-14);
    for (a, b) in [
        (restricted.grid, unrestricted.grid),
        (restricted.one_center, unrestricted.one_center),
        (restricted.compensation, unrestricted.compensation),
        (restricted.frozen_linear, unrestricted.frozen_linear),
        (restricted.constant, unrestricted.constant),
    ] {
        assert!((a - b).abs() < 1e-12 * (1. + a.abs()), "{a} vs {b}");
    }
    assert!(base
        .exchange_pieces("pbe", &state.densities)
        .unwrap()
        .is_none());
    drop((state, base));
    assert_eq!(ctx.memory_used(), 0);
}
