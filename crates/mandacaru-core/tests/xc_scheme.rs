//! The radial exchange-correlation scheme of the atom and of the PAW one-center terms (XC-SW,
//! D-2/D-2b): production is the variational scheme (`spherical::exchange_correlation`, the
//! scheme of the molecular PAW path) for the reference-atom SCF, the unscreening and the
//! one-center terms alike; the atom-solver scheme the 431 shipped datasets were produced with
//! (`radial_xc`, derivatives on subset nodes, splined back) survives only as the `doc(hidden)`
//! `AtomSolver` seam that proves the port against the frozen Python numbers (`paw_generate`,
//! `paw_assemble`, `paw_one_center`, `paw_search*`, `paw_diagnose*`).
//!
//! This file holds the independent evidence that the new scheme is right and the measured size of
//! the difference:
//!
//! * its potential is the functional derivative of its own discrete energy (finite differences of
//!   the energy in one node of the density, edge nodes included);
//! * against the continuum functional derivative evaluated with sixth-order derivatives it is
//!   closer than the atom-solver scheme (`CONTINUUM_*`);
//! * LDA is identical to the atom-solver scheme everywhere the density is above the kernel's
//!   exchange floor (2e-12), and differs by the uniform-gas continuation below it; PBE holds the
//!   uniform-gas exchange at the floor density and so carries a constant `-1.874e-4 Ha` to
//!   infinity, which the variational scheme does not (the production `v_local` of a PBE
//!   dataset lost that offset);
//! * the atom is solved with the scheme it is asked for.
use mandacaru_core::models::{
    radial_xc,
    relativity::Treatment,
    spherical, xc,
    xc_scheme::{AtomSolver, Variational, XcSource},
};
use mandacaru_core::paw::generate::{
    generate, generate_with,
    recipe::Recipe,
    reference::{solve_reference, solve_reference_with},
};
use mandacaru_core::Context;
use std::f64::consts::PI;

fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

fn atom_of(ctx: &Context, symbol: &str, functional: &str) -> mandacaru_core::models::atom::Atom {
    let recipe = Recipe::for_dataset(symbol, functional, Treatment::Scalar).unwrap();
    solve_reference(ctx, &recipe).unwrap()
}

/// A miss that is not a number fails every gate.
fn worst(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// `dE/d rho_i / w_i` by a central difference of the discrete energy in node `i`, Richardson
/// extrapolated over the relative steps `1e-4` and `2e-4` (`w_i` is the node's quadrature
/// weight, half a step at the ends). The potential of the scheme must be this.
fn energy_derivative(
    ctx: &Context,
    r: &[f64],
    rho: &[f64],
    functional: &str,
    relativistic: bool,
    i: usize,
) -> f64 {
    let n = r.len();
    let energy = |v: f64| {
        let mut d = rho.to_vec();
        d[i] = v;
        spherical::exchange_correlation(ctx, r, &d, functional, relativistic)
            .unwrap()
            .energy
    };
    let central = |relative: f64| {
        let step = relative * rho[i];
        (energy(rho[i] + step) - energy(rho[i] - step)) / (2. * step)
    };
    let width = if i == 0 {
        0.5 * (r[1] - r[0])
    } else if i == n - 1 {
        0.5 * (r[n - 1] - r[n - 2])
    } else {
        0.5 * (r[i + 1] - r[i - 1])
    };
    let weight = 4. * PI * r[i] * r[i] * width;
    (4. * central(1e-4) - central(2e-4)) / 3. / weight
}

/// Gate of the finite-difference check, relative to `max(1, |v|)` (Ha): the energy is smooth in
/// one node of the density; the Richardson-extrapolated central difference leaves the
/// round-off of the energy difference divided by the node's weight (5e-7 to 2.3e-6 of the
/// potential, measured on oxygen and silicon), which a miss of the potential has to exceed to be seen.
const DERIVATIVE_GATE: f64 = 5e-6;

#[test]
fn the_potential_is_the_functional_derivative_of_the_energy_it_integrates() {
    let ctx = context();
    for (symbol, functional, relativistic) in [
        ("O", "pbe", false),
        ("O", "pbe", true),
        ("O", "lda", false),
        ("Si", "pbe", false),
    ] {
        let atom = atom_of(&ctx, symbol, functional);
        let (r, rho) = (atom.r.as_slice(), atom.density.as_slice());
        let v = spherical::exchange_correlation(&ctx, r, rho, functional, relativistic).unwrap();
        let n = r.len();
        // The edge nodes, the steep core region, the valence and a tail node above the floor.
        let tail = rho.iter().rposition(|d| *d > 1e-5).unwrap();
        let mut miss = 0f64;
        for i in [
            0,
            1,
            2,
            50,
            n / 40,
            n / 12,
            n / 6,
            n / 3,
            tail,
            n - 2,
            n - 1,
        ] {
            if rho[i] < 1e-6 {
                continue; // the round-off of the energy difference exceeds the node's share
            }
            let fd = energy_derivative(&ctx, r, rho, functional, relativistic, i);
            let at = v.potential.as_slice()[i];
            miss = worst(miss, (fd - at).abs() / at.abs().max(1.));
        }
        eprintln!(
            "{symbol} {functional} relativistic={relativistic}: worst relative miss {miss:.3e}"
        );
        assert!(miss <= DERIVATIVE_GATE, "{symbol} {functional}: {miss:e}");
    }
}

/// The control: the atom-solver potential is not the derivative of any discrete energy (its
/// derivatives live on subset nodes). Against the same finite differences it misses by 100 times
/// the gate and more, so the check can tell the two schemes apart.
#[test]
fn the_atom_solver_potential_is_not_the_derivative_of_the_discrete_energy() {
    let ctx = context();
    let atom = atom_of(&ctx, "O", "pbe");
    let (r, rho) = (atom.r.as_slice(), atom.density.as_slice());
    let legacy = radial_xc::xc_potential(&ctx, r, rho, "pbe", false)
        .unwrap()
        .1;
    let n = r.len();
    let mut miss = 0f64;
    for i in [0, 1, n / 40, n / 12, n / 6, n / 3] {
        let fd = energy_derivative(&ctx, r, rho, "pbe", false, i);
        miss = worst(miss, (fd - legacy[i]).abs() / legacy[i].abs().max(1.));
    }
    eprintln!(
        "atom-solver potential against dE/drho: relative {miss:.3e} ({:.0} gates)",
        miss / DERIVATIVE_GATE
    );
    assert!(miss / DERIVATIVE_GATE > 100.0);
}

fn sixth(y: &[f64], i: usize, h: f64) -> f64 {
    (45. * (y[i + 1] - y[i - 1]) - 9. * (y[i + 2] - y[i - 2]) + (y[i + 3] - y[i - 3])) / (60. * h)
}

/// The continuum functional derivative `df/drho - (1/r^2) d/dr (r^2 df/d(drho/dr))` of a smooth
/// density on the uniform grid, with sixth-order derivatives: the slow, accurate reference.
fn continuum(r: &[f64], rho: &[f64], functional: &str) -> Vec<f64> {
    let (n, h) = (r.len(), r[1] - r[0]);
    let mut dfdrho = vec![0.; n];
    let mut flux = vec![0.; n];
    for i in 3..n - 3 {
        let g = sixth(rho, i, h);
        let (s, sigma) = (0.5 * rho[i], 0.25 * g * g);
        let (_, d) =
            xc::point_partials(functional, [s, s, sigma, sigma, sigma, 0., 0.], false).unwrap();
        dfdrho[i] = 0.5 * (d[0] + d[1]);
        flux[i] = r[i] * r[i] * 0.5 * g * (d[2] + d[3] + d[4]);
    }
    (0..n)
        .map(|i| {
            if !(6..n - 6).contains(&i) {
                f64::NAN
            } else if functional == "lda" {
                dfdrho[i]
            } else {
                dfdrho[i] - sixth(&flux, i, h) / (r[i] * r[i])
            }
        })
        .collect()
}

/// Worst `|v - reference|` over the nodes with `lo <= r < hi`.
fn deviation(r: &[f64], v: &[f64], reference: &[f64], lo: f64, hi: f64) -> f64 {
    (0..r.len())
        .filter(|&i| r[i] >= lo && r[i] < hi && reference[i].is_finite())
        .map(|i| (v[i] - reference[i]).abs())
        .fold(0., worst)
}

/// Gates on the distance (Ha) to the continuum derivative, per window of the radius (Bohr), for
/// the variational scheme: measured O/Si 7.5e-5 / 5.0e-5 in `[0.2, 1)` (the cusp region; a
/// second-order difference on a grid of 0.0025 Bohr), 6.1e-7 / 1.8e-6 in `[1, 4)` and
/// 1.5e-7 / 2.4e-8 in `[4, 10)`. The atom-solver scheme measures 9.5e-4 / 5.3e-4, 5.8e-5 /
/// 2.9e-4 and 3.2e-5 / 3.0e-5.
const CONTINUUM_GATES: [(f64, f64, f64); 3] = [(0.2, 1., 1.5e-4), (1., 4., 3e-6), (4., 10., 5e-7)];

#[test]
fn the_variational_potential_is_closer_to_the_continuum_derivative_than_the_atom_solver_one() {
    let ctx = context();
    for symbol in ["O", "Si"] {
        let atom = atom_of(&ctx, symbol, "pbe");
        let (r, rho) = (atom.r.as_slice(), atom.density.as_slice());
        let reference = continuum(r, rho, "pbe");
        let variational = spherical::exchange_correlation(&ctx, r, rho, "pbe", false).unwrap();
        let legacy = radial_xc::xc_potential(&ctx, r, rho, "pbe", false)
            .unwrap()
            .1;
        for (lo, hi, gate) in CONTINUUM_GATES {
            let ours = deviation(r, variational.potential.as_slice(), &reference, lo, hi);
            let theirs = deviation(r, &legacy, &reference, lo, hi);
            eprintln!(
                "{symbol} [{lo}, {hi}): variational {ours:.2e}, atom-solver {theirs:.2e} ({:.0}x)",
                theirs / ours
            );
            assert!(ours <= gate, "{symbol} [{lo}, {hi}): {ours:e} > {gate:e}");
            // The atom-solver scheme is the less accurate one in every window, by a factor the
            // measurement bounds from below (12 in the cusp window, 90 and more outside it).
            assert!(
                theirs > 10. * ours,
                "{symbol} [{lo}, {hi}): {ours:e} against {theirs:e}"
            );
        }
        // The control: the same potential without the gradient (flux) term is far off.
        let lda = spherical::exchange_correlation(&ctx, r, rho, "lda", false).unwrap();
        let blind = deviation(r, lda.potential.as_slice(), &reference, 1., 4.);
        eprintln!(
            "{symbol}: without the gradient term the miss is {blind:.2e} Ha ({:.0} gates)",
            blind / CONTINUUM_GATES[1].2
        );
        assert!(blind / CONTINUUM_GATES[1].2 > 100.0, "{blind:e}");
    }
}

#[test]
fn the_schemes_agree_for_lda_above_the_floor_and_for_pbe_in_the_stride_one_region() {
    let ctx = context();
    // LDA: identical wherever the density is above the exchange floor; below it the atom-solver
    // scheme continues the uniform gas (a step of 1.86e-4 Ha at 2e-12) and the kernel drops it.
    let atom = atom_of(&ctx, "O", "lda");
    let (r, rho) = (atom.r.as_slice(), atom.density.as_slice());
    let legacy = radial_xc::xc_potential(&ctx, r, rho, "lda", false)
        .unwrap()
        .1;
    let variational = spherical::exchange_correlation(&ctx, r, rho, "lda", false).unwrap();
    let mut above = 0f64;
    let mut below = 0f64;
    for i in 0..r.len() {
        let d = (legacy[i] - variational.potential.as_slice()[i]).abs();
        if rho[i] > 2e-12 {
            above = worst(above, d);
        } else {
            below = worst(below, d);
        }
    }
    eprintln!("O lda: above the floor {above:.2e} Ha, below it up to {below:.3e} Ha");
    assert!(above <= 1e-12, "{above:e}");
    assert!((1.8e-4..1.9e-4).contains(&below), "{below:e}");
    // PBE: the same up to the stride-one region of the atom-solver derivative (r < 0.2 Bohr, the
    // first node aside) and different, by a measured amount, beyond it.
    let atom = atom_of(&ctx, "O", "pbe");
    let (r, rho) = (atom.r.as_slice(), atom.density.as_slice());
    let legacy = radial_xc::xc_potential(&ctx, r, rho, "pbe", false)
        .unwrap()
        .1;
    let variational = spherical::exchange_correlation(&ctx, r, rho, "pbe", false).unwrap();
    let inner = deviation(r, variational.potential.as_slice(), &legacy, 0.01, 0.2);
    let core = deviation(r, variational.potential.as_slice(), &legacy, 0.2, 1.);
    let valence = deviation(r, variational.potential.as_slice(), &legacy, 1., 14.);
    eprintln!("O pbe: atom-solver against variational: [0.01, 0.2) {inner:.2e}, [0.2, 1) {core:.2e}, [1, 14) {valence:.2e}");
    assert!(inner <= 1e-12, "{inner:e}");
    assert!((9e-4..1e-3).contains(&core), "{core:e}");
    assert!((1e-4..4e-4).contains(&valence), "{valence:e}");
}

#[test]
fn the_pbe_tail_decays_to_zero_instead_of_holding_the_uniform_gas_value() {
    let ctx = context();
    let atom = atom_of(&ctx, "O", "pbe");
    let (r, rho) = (atom.r.as_slice(), atom.density.as_slice());
    let legacy = radial_xc::xc_potential(&ctx, r, rho, "pbe", false)
        .unwrap()
        .1;
    let variational = spherical::exchange_correlation(&ctx, r, rho, "pbe", false).unwrap();
    for x in [20.0, 25.0, 29.9] {
        let i = r.iter().position(|v| *v >= x).unwrap();
        let (old, new) = (legacy[i], variational.potential.as_slice()[i]);
        eprintln!(
            "O pbe r={:.1}: rho {:.2e}, atom-solver {old:.6e}, variational {new:.3e}",
            r[i], rho[i]
        );
        assert!(
            (old + 1.873829e-4).abs() < 1e-9,
            "the constant of the legacy tail: {old:e}"
        );
        assert_eq!(new, 0.0, "the variational tail beyond the floor");
    }
}

#[test]
fn the_atom_is_solved_with_the_scheme_it_is_asked_for() {
    let ctx = context();
    let recipe = Recipe::for_dataset("O", "pbe", Treatment::Scalar).unwrap();
    let production = solve_reference(&ctx, &recipe).unwrap();
    let seam = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
    assert!(production.converged && seam.converged);
    let on = |atom: &mandacaru_core::models::atom::Atom, scheme: &dyn XcSource| {
        let rho = atom
            .density
            .as_slice()
            .iter()
            .map(|d| d.max(0.))
            .collect::<Vec<_>>();
        // A scalar-relativistic atom is solved with the relativistic exchange factor.
        let relativistic = atom.relativity.relativistic_exchange();
        assert!(
            relativistic,
            "the reference atom of a PAW dataset is scalar relativistic"
        );
        let field = scheme
            .evaluate(&ctx, atom.r.as_slice(), &rho, "pbe", relativistic)
            .unwrap();
        field
            .potential
            .iter()
            .zip(atom.xc.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0., worst)
    };
    let own = (on(&production, &Variational), on(&seam, &AtomSolver));
    let crossed = (on(&production, &AtomSolver), on(&seam, &Variational));
    eprintln!("atom potential against its own scheme {own:?}, against the other {crossed:?}");
    assert!(own.0 <= 1e-12 && own.1 <= 1e-12, "{own:?}");
    // The control: the other scheme on the same density differs by the tail offset and more.
    assert!(
        crossed.0 > 100. * 1e-12 && crossed.1 > 100. * 1e-12,
        "{crossed:?}"
    );
    // What the scheme change does to the atom itself, for the record: the PBE eigenvalues move by
    // up to 1.2 mHa, the energy by 2.2 mHa (oxygen).
    let shift = production
        .eigenvalues
        .iter()
        .zip(&seam.eigenvalues)
        .map(|(a, b)| (a - b).abs())
        .fold(0., worst);
    eprintln!(
        "O pbe: eigenvalues move by {shift:.3e} Ha, energy by {:.3e} Ha",
        (production.energy - seam.energy).abs()
    );
    assert!((5e-4..3e-3).contains(&shift), "{shift:e}");
}

/// The offset that came from the legacy atom scheme: the PBE `v_local` of the production path is
/// `-Z_ion/r` at the edge of the box again (it carried `-1.874e-4 Ha` to infinity while the atom
/// was solved with the atom-solver scheme and the unscreening used the variational one), exactly
/// as the legacy seam, which is consistent, has it. The old production is the control: the
/// atom-solver atom with the variational energies, built here from the pieces, is 1.87e-4 off.
#[test]
fn the_production_v_local_of_a_pbe_dataset_has_no_constant_offset() {
    let ctx = context();
    let recipe = Recipe::for_dataset("O", "pbe", Treatment::Scalar).unwrap();
    let production = generate(&ctx, &recipe).unwrap().dataset;
    let seam = generate_with(&ctx, &recipe, &AtomSolver).unwrap().dataset;
    let tail_error = |d: &mandacaru_core::paw::record::PawDataset| {
        let r = d.r.as_slice();
        (0..r.len())
            .filter(|&i| r[i] > 26.)
            .map(|i| (d.v_local.as_slice()[i] + d.valence_charge / r[i]).abs())
            .fold(0., worst)
    };
    let (new, consistent) = (tail_error(&production), tail_error(&seam));
    // The inconsistent old production: the legacy atom unscreened with the variational scheme.
    let atom = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
    let old = mandacaru_core::paw::generate::generate_with_atom(&ctx, &recipe, &atom, &Variational)
        .unwrap()
        .dataset;
    let before = tail_error(&old);
    eprintln!("O pbe v_local + Z_ion/r beyond 26 Bohr: production {new:.2e}, legacy seam {consistent:.2e}, old production {before:.2e}");
    assert!(new <= 2e-6 && consistent <= 2e-6, "{new:e} {consistent:e}");
    assert!(
        before / 2e-6 > 50.0 && (1.8e-4..1.95e-4).contains(&before),
        "{before:e}"
    );
}
