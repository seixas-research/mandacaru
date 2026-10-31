//! DFT+U in the periodic SCF (HU-3): parity with the frozen DFT+U numbers of the deleted
//! Python route (`test/migration/reference/hubbard/frozen.json`), the inertness of U = 0, the
//! Hellmann-Feynman identity with a negative control, restricted == unrestricted, explicit
//! seeding, persistence and refusals.
//!
//! The crystals are written by `python test/migration/export_hubbard_crystal.py` (a metadata JSON
//! and a git-ignored `.npz`); every SCF here is serial (one thread) and under 150 MB.
//!
//! Parity tolerances (measured, see `PAW_MIGRATION_LOG.md`, HU-3): the frozen runs come from
//! the old Python mixing the on-site operator, this port mixes the occupations, so the paths
//! differ and only the converged state is compared. The energy is variational, so it agrees to
//! second order in the density error (`ENERGY_TOLERANCE_EV`, measured 1.4e-10); E_U, the gap and
//! n are first order in it. The SCF stops at `density_tolerance` 1e-5 and the four seeds of
//! HU-0 already spread E_U, the gap and n by 2e-6 (`FIRST_ORDER_EV`, `N_TOLERANCE`).
mod hubbard_support;

use hubbard_support::{directory, load, try_load, HARTREE_EV};
use mandacaru_core::{
    math::linalg::eigen,
    periodic::{
        hubbard::{Seed, Spec, MAX_SHELLS, MAX_SHELL_BYTES, MAX_TABLE_BYTES},
        solver::{diagonalize, diagonalize_with},
        Crystal, Options, PeriodicResult,
    },
    quantum::C64,
    Context,
};
use nalgebra::DMatrix;
use serde_json::Value;

const ENERGY_TOLERANCE_EV: f64 = 1e-8;
const FIRST_ORDER_EV: f64 = 1e-5;
const N_TOLERANCE: f64 = 1e-5;

fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

fn frozen() -> Value {
    let text = std::fs::read_to_string(directory().join("frozen.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn solve(c: &Crystal, width_ev: f64) -> PeriodicResult {
    let state = c
        .solve(&Options {
            width: width_ev / HARTREE_EV,
            ..Default::default()
        })
        .unwrap();
    assert!(state.converged, "the SCF did not converge");
    state
}

fn with_u(c: &mut Crystal, u_ev: f64) {
    c.hubbard
        .as_mut()
        .unwrap()
        .set_interaction(u_ev / HARTREE_EV, 0.)
        .unwrap();
}

/// Lowest level above the Fermi level minus the highest at or below it, over every spin and k.
fn gap_ev(r: &PeriodicResult) -> f64 {
    let (mut above, mut below) = (f64::INFINITY, f64::NEG_INFINITY);
    for e in r.eigenvalues.iter().flatten() {
        if *e > r.fermi {
            above = above.min(*e)
        } else {
            below = below.max(*e)
        }
    }
    (above - below) * HARTREE_EV
}

fn eigenvalues(n: &[C64]) -> Vec<f64> {
    let p = (n.len() as f64).sqrt() as usize;
    eigen(DMatrix::from_fn(p, p, |a, b| n[a * p + b]))
        .unwrap()
        .0
}

fn trace(n: &[C64]) -> f64 {
    let p = (n.len() as f64).sqrt() as usize;
    (0..p).map(|a| n[a * p + a].re).sum()
}

fn energy_ev(r: &PeriodicResult) -> f64 {
    r.extrapolated_energy * HARTREE_EV
}

fn nio_state(u_ev: f64) -> PeriodicResult {
    let ctx = context();
    let mut c = load(&ctx, "nio_afm2");
    with_u(&mut c, u_ev);
    solve(&c, 0.05)
}

#[test]
fn a_zero_u_shell_table_is_plain_dft_to_the_last_bit() {
    let ctx = context();
    let mut c = load(&ctx, "nio_afm2");
    with_u(&mut c, 0.);
    let with_table = solve(&c, 0.05);
    c.hubbard = None;
    let plain = solve(&c, 0.05);
    assert_eq!(
        with_table.free_energy.to_bits(),
        plain.free_energy.to_bits()
    );
    assert_eq!(with_table.history.len(), plain.history.len());
    assert_eq!(with_table.terms, plain.terms);
    assert_eq!(plain.hubbard_energy, None);
    assert!(plain.hubbard.is_none());
    // The table is reported even when U = 0, and then E_U is exactly zero.
    assert_eq!(with_table.hubbard_energy, Some(0.));
    let report = with_table.hubbard.as_ref().unwrap();
    assert!(report
        .operators
        .iter()
        .all(|o| o.iter().all(|x| *x == C64::default())));
}

#[test]
fn nio_at_zero_u_reproduces_the_frozen_plain_dft_energy_and_gap() {
    let reference = frozen();
    let case = &reference["cases"]["nio_U0_A"];
    let r = nio_state(0.);
    let e = energy_ev(&r);
    let expected = case["energy_eV"].as_f64().unwrap();
    println!(
        "U = 0: E = {e:.12} eV (frozen {expected:.12}), diff {:.2e}",
        e - expected
    );
    assert!((e - expected).abs() < 1e-8, "energy {e} vs {expected}");
    let gap = gap_ev(&r);
    let frozen_gap = case["gap_from_levels_eV"].as_f64().unwrap();
    assert!((gap - frozen_gap).abs() < 1e-6, "gap {gap} vs {frozen_gap}");
    assert_eq!(
        r.history.len(),
        case["n_iterations"].as_u64().unwrap() as usize
    );
}

#[test]
fn nio_at_u_3_and_6_matches_the_frozen_energy_gap_occupations_and_moment() {
    let reference = frozen();
    let mut energies = vec![];
    for (u, key) in [(3., "nio_U3_A"), (6., "nio_U6_A")] {
        let case = &reference["cases"][key];
        let r = nio_state(u);
        let e = energy_ev(&r);
        let e_u = r.hubbard_energy.unwrap() * HARTREE_EV;
        let gap = gap_ev(&r);
        let report = r.hubbard.as_ref().unwrap();
        let (frozen_e, frozen_gap, frozen_u) = (
            case["energy_eV"].as_f64().unwrap(),
            case["gap_from_levels_eV"].as_f64().unwrap(),
            case["hubbard_eV"].as_f64().unwrap(),
        );
        println!(
            "U = {u}: E {e:.9} (frozen {frozen_e:.9}, diff {:.2e}) E_U {e_u:.9} (diff {:.2e}) gap {gap:.9} \
             (diff {:.2e}) iterations {} (frozen {})",
            e - frozen_e,
            e_u - frozen_u,
            gap - frozen_gap,
            r.history.len(),
            case["n_iterations"]
        );
        assert!((e - frozen_e).abs() < ENERGY_TOLERANCE_EV);
        assert!((e_u - frozen_u).abs() < FIRST_ORDER_EV);
        assert!(
            (gap - frozen_gap).abs() < FIRST_ORDER_EV,
            "gap {gap} vs {frozen_gap}"
        );
        // Full occupation spectra and the d-shell moment tr n_up - tr n_down on both Ni sites.
        let mut worst: f64 = 0.;
        for spin in 0..2 {
            for shell in 0..2 {
                let mine = eigenvalues(report.occupation(spin, shell));
                let theirs: Vec<f64> = case["n_eigenvalues"][spin][shell]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect();
                for (a, b) in mine.iter().zip(&theirs) {
                    worst = worst.max((a - b).abs());
                }
                let trace_diff = trace(report.occupation(spin, shell))
                    - case["n_trace"][spin][shell].as_f64().unwrap();
                worst = worst.max(trace_diff.abs());
            }
        }
        println!("U = {u}: worst |n eigenvalue or trace - frozen| = {worst:.2e}");
        assert!(worst < N_TOLERANCE);
        let moment = trace(report.occupation(0, 0)) - trace(report.occupation(1, 0));
        let frozen_moment =
            case["n_trace"][0][0].as_f64().unwrap() - case["n_trace"][1][0].as_f64().unwrap();
        assert!((moment - frozen_moment).abs() < N_TOLERANCE);
        // AFM-II: the two Ni sites are spin mirrors.
        for (a, b) in eigenvalues(report.occupation(0, 0))
            .iter()
            .zip(eigenvalues(report.occupation(1, 1)))
        {
            assert!((a - b).abs() < N_TOLERANCE);
        }
        energies.push((u, e, gap, e_u));
    }
    // E(U) increases and is concave, the gap opens with a falling slope, E_U is positive.
    let e0 = energy_ev(&nio_state(0.));
    let g0 = gap_ev(&nio_state(0.));
    let (d1, d2) = (energies[0].1 - e0, energies[1].1 - energies[0].1);
    println!("increments E(3)-E(0) = {d1:.6}, E(6)-E(3) = {d2:.6} eV");
    assert!(d1 > d2 && d2 > 0.);
    assert!(g0 < energies[0].2 && energies[0].2 < energies[1].2);
    assert!(energies[1].2 - energies[0].2 < energies[0].2 - g0);
    assert!(energies.iter().all(|x| x.3 > 0.));
    // n eigenvalues stay in the measured range of HU-0 (0.2563 .. 0.9588).
    let r = nio_state(6.);
    let report = r.hubbard.as_ref().unwrap();
    for spin in 0..2 {
        for shell in 0..2 {
            let ev = eigenvalues(report.occupation(spin, shell));
            assert!(ev[0] > 0.25 && ev[4] < 0.96, "{ev:?}");
        }
    }
}

#[test]
fn the_total_is_the_four_terms_plus_e_u_and_the_free_energy_adds_the_entropy() {
    let r = nio_state(3.);
    let e_u = r.hubbard_energy.unwrap();
    assert!(e_u > 0.);
    let four = r.terms[0] + r.terms[1] + r.terms[2] + r.terms[3];
    assert!((r.energy - (four + e_u)).abs() < 1e-12);
    assert!((r.free_energy - (r.energy + r.terms[4])).abs() < 1e-12);
}

/// The NiO AFM-II state at `u` eV: `(F, E_U)` in eV, `F` the Mermin free energy.
fn free_and_hubbard(ctx: &Context, u: f64) -> (f64, f64) {
    let mut c = load(ctx, "nio_afm2");
    with_u(&mut c, u);
    let state = solve(&c, 0.05);
    (
        state.free_energy * HARTREE_EV,
        state.hubbard_energy.unwrap() * HARTREE_EV,
    )
}

/// `dF/dU` against `E_U / U` for the Mermin free energy: Richardson over the central
/// differences at `dU` and `dU / 2`. `reported` multiplies the Hubbard energy that the test
/// reports, `F' = F + (reported - 1) E_U` and `E_U' = reported E_U`, while the SCF keeps the
/// production potential. That makes the pair `(E', V)` inconsistent on the test side, so no
/// production type carries a verification knob. Returns `(analytic, Richardson, central dU =
/// 0.2 error)`.
fn hellmann_feynman(reported: f64) -> (f64, f64, f64) {
    let ctx = context();
    let u0 = 3.;
    let state = |d: f64| free_and_hubbard(&ctx, u0 + d);
    let [m2, m1, z, p1, p2] = [-0.2, -0.1, 0., 0.1, 0.2].map(state);
    let report = |(f, e_u): (f64, f64)| f + (reported - 1.) * e_u;
    let analytic = reported * z.1 / u0;
    let central = |lo: (f64, f64), hi: (f64, f64), d: f64| (report(hi) - report(lo)) / (2. * d);
    let (wide, narrow) = (central(m2, p2, 0.2), central(m1, p1, 0.1));
    (analytic, (4. * narrow - wide) / 3., (wide - analytic).abs())
}

#[test]
fn the_hellmann_feynman_identity_holds_by_richardson_and_an_inconsistent_pair_fails_it() {
    let (analytic, richardson, central_error) = hellmann_feynman(1.);
    let error = (richardson - analytic).abs();
    println!(
        "dF/dU = {richardson:.9} against E_U/U = {analytic:.9}: Richardson error {error:.2e} \
         (relative {:.2e}), central dU = 0.2 error {central_error:.2e}",
        error / analytic
    );
    assert!(error / analytic < 1e-6);
    assert!(
        central_error > 10. * error,
        "the central difference is the weak gate"
    );
    // Negative control: the energy reported 1 % too large against the production potential is
    // not the integral of V, and the same Richardson test must reject it.
    let (analytic, richardson, _) = hellmann_feynman(1.01);
    let wrong = (richardson - analytic).abs() / analytic;
    println!("E_U reported 1 % high: relative error {wrong:.2e}");
    assert!(wrong > 1e-5, "the negative control must fail the identity");
}

#[test]
fn restricted_and_unrestricted_agree_at_zero_moment() {
    let ctx = context();
    let mut restricted = load(&ctx, "cuh_7");
    let mut unrestricted = load(&ctx, "cuh_7_spin");
    for c in [&mut restricted, &mut unrestricted] {
        with_u(c, 4.);
    }
    let (a, b) = (solve(&restricted, 0.01), solve(&unrestricted, 0.01));
    let de = (energy_ev(&a) - energy_ev(&b)).abs();
    let du = (a.hubbard_energy.unwrap() - b.hubbard_energy.unwrap()).abs() * HARTREE_EV;
    println!(
        "restricted - unrestricted: dE {de:.2e} eV, dE_U {du:.2e} eV, magnetization {:.1e}",
        b.magnetization
    );
    assert!(de < ENERGY_TOLERANCE_EV && du < FIRST_ORDER_EV);
    let (ra, rb) = (a.hubbard.as_ref().unwrap(), b.hubbard.as_ref().unwrap());
    for (x, y) in ra.occupation(0, 0).iter().zip(rb.occupation(0, 0)) {
        assert!((x - y).norm() < N_TOLERANCE);
    }
    for (x, y) in rb.occupation(0, 0).iter().zip(rb.occupation(1, 0)) {
        assert!((x - y).norm() < N_TOLERANCE);
    }
}

#[test]
fn the_cuh_gamma_box_shifts_by_the_frozen_amount_at_u_4() {
    let reference = frozen();
    let (zero, four) = (
        &reference["cases"]["cuh_crystal_L7_U0"],
        &reference["cases"]["cuh_crystal_L7_U4"],
    );
    let ctx = context();
    let mut c = load(&ctx, "cuh_7");
    let plain = solve(&c, 0.01);
    with_u(&mut c, 4.);
    let r = solve(&c, 0.01);
    let shift = energy_ev(&r) - energy_ev(&plain);
    let frozen_shift = four["energy_eV"].as_f64().unwrap() - zero["energy_eV"].as_f64().unwrap();
    println!(
        "CuH 7 A box: shift {shift:.9} eV (frozen {frozen_shift:.9}, diff {:.2e}); gap {:.9} (frozen {:.9}); \
         E(U=0) diff {:.2e}",
        shift - frozen_shift,
        gap_ev(&r),
        four["gap_from_levels_eV"].as_f64().unwrap(),
        energy_ev(&plain) - zero["energy_eV"].as_f64().unwrap()
    );
    assert!((shift - frozen_shift).abs() < ENERGY_TOLERANCE_EV);
    assert!((energy_ev(&plain) - zero["energy_eV"].as_f64().unwrap()).abs() < 1e-8);
    assert!((gap_ev(&r) - four["gap_from_levels_eV"].as_f64().unwrap()).abs() < FIRST_ORDER_EV);
    let e_u = r.hubbard_energy.unwrap();
    assert!((e_u - four["hubbard_Ha"].as_f64().unwrap()).abs() < 1e-7);
    let ev = eigenvalues(r.hubbard.as_ref().unwrap().occupation(0, 0));
    let expected: Vec<f64> = four["n_eigenvalues"][0][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    for (a, b) in ev.iter().zip(&expected) {
        assert!((a - b).abs() < N_TOLERANCE);
    }
}

/// Off-mesh bands and Wannier frames diagonalize at new k-points with the converged potential;
/// the reported operator must reproduce the SCF spectrum at a mesh point, and leaving it out
/// must not (the shift of the shell is far above the tolerance).
#[test]
fn the_reported_operator_reproduces_the_converged_spectrum_and_the_shell_energies_add_up() {
    let ctx = context();
    let mut c = load(&ctx, "cuh_7");
    with_u(&mut c, 4.);
    let r = solve(&c, 0.01);
    let report = r.hubbard.as_ref().unwrap();
    let n = c.grid.index.points();
    let spectrum = |operator: Option<&nalgebra::DMatrix<C64>>| {
        diagonalize_with(
            &c,
            &c.points[0],
            &r.potential.as_slice()[..n],
            r.moment_potential.as_slice(),
            None,
            operator,
        )
        .unwrap()
        .0
    };
    let with: Vec<f64> = spectrum(Some(&report.operators[0]))
        .iter()
        .map(|e| e + c.reference)
        .collect();
    let without: Vec<f64> = spectrum(None).iter().map(|e| e + c.reference).collect();
    let worst = with
        .iter()
        .zip(&r.eigenvalues[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let missed = without
        .iter()
        .zip(&r.eigenvalues[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    println!("operator replay {worst:.2e} Ha, without the operator {missed:.2e} Ha");
    assert!(worst < 1e-12, "{worst}");
    assert!(missed > 1e-3, "{missed}");
    let spec = c.hubbard.as_ref().unwrap();
    let shells = spec.shell_energies(&c, &report.occupations).unwrap();
    assert_eq!(shells.len(), spec.shells.len());
    let total: f64 = shells.iter().sum();
    assert!((total - r.hubbard_energy.unwrap()).abs() < 1e-14);
}

#[test]
fn the_filled_shell_of_a_ni_atom_is_the_sphere_truncated_projection() {
    let reference = frozen();
    let fraction = reference["cases"]["weights"]["rows"]["Ni"]["fraction_in_sphere"]
        .as_f64()
        .unwrap();
    let ctx = context();
    let mut c = load(&ctx, "ni_atom_7");
    with_u(&mut c, 3.);
    let r = solve(&c, 0.02);
    let n = r.hubbard.as_ref().unwrap();
    let majority = eigenvalues(n.occupation(0, 0));
    let minority = eigenvalues(n.occupation(1, 0));
    println!("Ni atom: majority {majority:?} minority {minority:?} sphere fraction {fraction}");
    let frozen_majority = 0.9532466936455168;
    for x in &majority {
        assert!((x - frozen_majority).abs() < N_TOLERANCE);
        // the KS orbital is a little more localized than the reference orbital: 0.2 % above
        assert!(*x > fraction && *x < fraction + 0.003 && *x < 1.);
    }
    assert!(minority[0].abs() < 1e-6 && minority[1].abs() < 1e-6);
    for x in &minority[2..] {
        assert!((x - frozen_majority).abs() < N_TOLERANCE);
    }
    assert!((r.magnetization - 2.).abs() < 1e-6);
    let (up, down) = (trace(n.occupation(0, 0)), trace(n.occupation(1, 0)));
    let (frozen_up, frozen_down) = (
        reference["cases"]["ni_atom_U3"]["n_trace"][0][0]
            .as_f64()
            .unwrap(),
        reference["cases"]["ni_atom_U3"]["n_trace"][1][0]
            .as_f64()
            .unwrap(),
    );
    assert!((up - frozen_up).abs() < N_TOLERANCE && (down - frozen_down).abs() < N_TOLERANCE);
}

#[test]
fn an_explicit_seed_is_the_first_input_and_the_solution_does_not_depend_on_a_perturbed_seed() {
    let ctx = context();
    let mut c = load(&ctx, "nio_afm2");
    with_u(&mut c, 3.);
    let default = solve(&c, 0.05);
    // 1) the explicit matrices are what the first diagonalization sees
    let seeds: Vec<Vec<C64>> = {
        let spec = c.hubbard.as_ref().unwrap();
        let mut seeds = spec.seed(2).unwrap();
        for n in &mut seeds {
            for a in 0..5 {
                n[a * 5 + a] += C64::new(0.02 * a as f64, 0.);
            }
        }
        seeds
    };
    for (shell, atom) in c.hubbard.as_mut().unwrap().shells.iter_mut().enumerate() {
        atom.seed = Seed::Occupations(vec![seeds[shell].clone(), seeds[2 + shell].clone()]);
    }
    let one = c
        .solve(&Options {
            width: 0.05 / HARTREE_EV,
            max_iterations: 1,
            ..Default::default()
        })
        .unwrap();
    let expected = c.hubbard.as_ref().unwrap().operators(&c, &seeds).unwrap();
    for (a, b) in one
        .hubbard
        .as_ref()
        .unwrap()
        .operators
        .iter()
        .zip(&expected)
    {
        assert_eq!(a, b);
    }
    // 2) from this orbital-broken seed the SCF returns to the same state (HU-0 found no second basin)
    let broken = solve(&c, 0.05);
    let de = (energy_ev(&broken) - energy_ev(&default)).abs();
    let dn = (0..2)
        .flat_map(|s| (0..2).map(move |k| (s, k)))
        .map(|(s, k)| {
            eigenvalues(broken.hubbard.as_ref().unwrap().occupation(s, k))
                .iter()
                .zip(eigenvalues(
                    default.hubbard.as_ref().unwrap().occupation(s, k),
                ))
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max)
        })
        .fold(0., f64::max);
    println!(
        "perturbed explicit seed: dE {de:.2e} eV, dn {dn:.2e}, iterations {} vs {}",
        broken.history.len(),
        default.history.len()
    );
    assert!(de < ENERGY_TOLERANCE_EV && dn < N_TOLERANCE);
}

fn ni_atom_with_minority_diagonal(diagonal: Option<[f64; 5]>) -> (f64, Vec<f64>) {
    let ctx = context();
    let mut c = load(&ctx, "ni_atom_7");
    with_u(&mut c, 3.);
    if let Some(d) = diagonal {
        let matrix = |d: [f64; 5]| -> Vec<C64> {
            (0..25)
                .map(|i| C64::new(if i / 5 == i % 5 { d[i / 5] } else { 0. }, 0.))
                .collect()
        };
        c.hubbard.as_mut().unwrap().shells[0].seed =
            Seed::Occupations(vec![matrix([0.95; 5]), matrix(d)]);
    }
    let r = solve(&c, 0.02);
    let minority = eigenvalues(r.hubbard.as_ref().unwrap().occupation(1, 0));
    (energy_ev(&r), minority)
}

/// An isolated d8 atom has several self-consistent orbital patterns for the two minority holes,
/// all with the same n spectrum (two empty, three filled) and different orientations: the
/// seed chooses between them. The default spherical seed finds the lowest one probed.
#[test]
fn an_explicit_seed_selects_between_the_orbital_patterns_of_a_ni_atom() {
    let (default, spectrum) = ni_atom_with_minority_diagonal(None);
    let (broken, other) = ni_atom_with_minority_diagonal(Some([0., 0.95, 0.95, 0.95, 0.]));
    println!("Ni atom U = 3: default {default:.9} eV, holes on m = -2, +2: {broken:.9} eV ({:.3} eV higher)", broken - default);
    for s in [&spectrum, &other] {
        assert!(s[0].abs() < 1e-6 && s[1].abs() < 1e-6);
        assert!(s[2..]
            .iter()
            .all(|x| (x - 0.9532466936).abs() < N_TOLERANCE));
    }
    assert!(
        broken - default > 0.05,
        "the broken seed must stay in its own fixed point"
    );
}

/// The same crystal with every spin mirrored: the initial magnetization density and the
/// signed moment of each shell change sign.
fn mirrored(ctx: &Context, u_ev: f64) -> Crystal {
    let mut c = load(ctx, "nio_afm2");
    with_u(&mut c, u_ev);
    c.initial_m.as_mut_slice().iter_mut().for_each(|m| *m = -*m);
    for shell in &mut c.hubbard.as_mut().unwrap().shells {
        match &mut shell.seed {
            Seed::Moment { moment, .. } => *moment = -*moment,
            Seed::Occupations(_) => unreachable!("the fixture seeds by moment"),
        }
    }
    c
}

#[test]
fn a_flipped_antiferromagnetic_seed_converges_to_the_mirror_state() {
    let ctx = context();
    for u in [3., 6.] {
        let original = nio_state(u);
        let mirror = solve(&mirrored(&ctx, u), 0.05);
        let de = (energy_ev(&original) - energy_ev(&mirror)).abs();
        let (a, b) = (
            original.hubbard.as_ref().unwrap(),
            mirror.hubbard.as_ref().unwrap(),
        );
        let mut worst: f64 = 0.;
        let mut flipped: f64 = 0.;
        for shell in 0..2 {
            // the mirror's spin-up shell is the original's spin-down shell and vice versa
            for (x, y) in [(0, 1), (1, 0)] {
                let (mine, theirs) = (a.occupation(x, shell), b.occupation(y, shell));
                for (p, q) in eigenvalues(mine).iter().zip(eigenvalues(theirs)) {
                    worst = worst.max((p - q).abs());
                }
                worst = worst.max((trace(mine) - trace(theirs)).abs());
            }
            flipped =
                flipped.max((trace(a.occupation(0, shell)) - trace(b.occupation(0, shell))).abs());
        }
        println!(
            "U = {u}: mirror dE {de:.2e} eV, worst |n - swapped n| {worst:.2e}, spin-up trace changes by {flipped:.3}"
        );
        assert!(de < ENERGY_TOLERANCE_EV, "U = {u}: dE {de}");
        assert!(worst < N_TOLERANCE, "U = {u}: n is not the spin swap");
        assert!(
            flipped > 1.,
            "U = {u}: the mirror seed stayed in the original state"
        );
    }
}

/// `count` distinct f shells of eight radial projectors each, with an explicit two-spin seed:
/// the largest table a snapshot can hold.
fn widest_table(count: usize) -> Value {
    let identity = |p: usize| -> Vec<Value> {
        (0..p * p)
            .map(|i| serde_json::json!([if i / p == i % p { 0.5 } else { 0. }, 0.]))
            .collect()
    };
    let shell = |atom: usize| {
        serde_json::json!({"atom": atom, "symbol": "Ce", "l": 3, "u": 0.1, "j": 0.,
            "projectors": (0..56).map(|i| atom * 56 + i).collect::<Vec<_>>(),
            "weights": vec![0.1; 8], "projector_basis": "raw",
            "seed": {"occupations": [identity(7), identity(7)]}})
    };
    serde_json::json!({"form": "dudarev-fll", "shells": (0..count).map(shell).collect::<Vec<_>>()})
}

#[test]
fn the_shell_table_is_bounded_by_the_documented_size() {
    let parse = |count: usize| {
        Spec::parse(
            &widest_table(count),
            count * 56,
            2,
            false,
            mandacaru_core::math::storage::Precision::Float64,
            0,
        )
    };
    let spec = parse(MAX_SHELLS).unwrap().unwrap();
    assert_eq!(spec.shells.len(), MAX_SHELLS);
    // The widest shell is the one the bound is written for.
    assert_eq!(spec.table_bytes(), MAX_TABLE_BYTES);
    assert_eq!(spec.table_bytes() / MAX_SHELLS, MAX_SHELL_BYTES);
    assert!(MAX_TABLE_BYTES < 16 << 20, "{MAX_TABLE_BYTES}");
    // One shell more raises, with its own message.
    let message = parse(MAX_SHELLS + 1).err().expect("must raise").to_string();
    assert!(message.contains("at most 4096"), "{message}");
}

#[test]
fn a_saved_crystal_keeps_its_shell_table() {
    let ctx = context();
    let mut c = load(&ctx, "cuh_7");
    with_u(&mut c, 4.);
    let path = std::env::temp_dir().join(format!("mandacaru-hubbard-{}.json", std::process::id()));
    c.save(path.to_str().unwrap()).unwrap();
    let again = Crystal::load(&ctx, path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(again.hubbard, c.hubbard);
    assert_eq!(
        again.hubbard.as_ref().unwrap().shells[0].basis.as_deref(),
        Some("raw")
    );
    assert_eq!(again.hubbard.as_ref().unwrap().shells[0].u, 4. / HARTREE_EV);
}

#[test]
fn unsupported_combinations_raise_and_no_full_mesh_crystal_has_a_symmetry_table() {
    let ctx = context();
    for case in ["nio_afm2", "ni_atom_7", "cuh_7", "cuh_7_spin"] {
        let c = load(&ctx, case);
        assert!(
            c.permutations.is_empty(),
            "{case}: a full mesh must carry no symmetry table"
        );
    }
    let error = |r: mandacaru_core::Result<Crystal>| r.err().expect("must raise").to_string();
    // hybrid functional
    let message = error(try_load(&ctx, "cuh_7", |m| {
        m["functional"] = "hse06".into()
    }));
    assert!(message.contains("DFT+U"), "{message}");
    // float32 samples
    let message = error(try_load(&ctx, "cuh_7", |m| {
        m["precision"] = "float32".into()
    }));
    assert!(message.contains("float64"), "{message}");
    // a shell outside the basis, a wrong projector count, a shared projector, a wrong form, s/p shells
    let table =
        |edit: fn(&mut Value)| error(try_load(&ctx, "nio_afm2", move |m| edit(&mut m["hubbard"])));
    assert!(table(|h| h["shells"][0]["projectors"][3] = 9999.into()).contains("outside the basis"));
    assert!(table(|h| {
        h["shells"][0]["projectors"].as_array_mut().unwrap().pop();
    })
    .contains("projector indices"));
    assert!(
        table(|h| h["shells"][1]["projectors"] = h["shells"][0]["projectors"].clone())
            .contains("share")
    );
    assert!(table(|h| h["form"] = "liechtenstein".into()).contains("form"));
    assert!(table(|h| h["shells"][0]["l"] = 1.into()).contains("d or f"));
    assert!(table(|h| h["shells"][0]["u"] = (-1.).into()).contains("negative"));
    assert!(table(|h| h["shells"][0]["j"] = 1.0.into()).contains("exceeds"));
    assert!(table(|h| h["shells"][0]["seed"]["n_electrons"] = 50.into()).contains("seed"));
    assert!(table(|h| h["shells"][1]["atom"] = 0.into()).contains("per atom"));
    assert!(table(|h| h["shells"] = Value::Array(vec![])).contains("at least one shell"));
    // a restricted run has no spin polarization: a seed moment would be ignored, so it raises
    let message = error(try_load(&ctx, "cuh_7", |m| {
        m["hubbard"]["shells"][0]["seed"]["moment"] = 1.0.into()
    }));
    assert!(
        message.contains("restricted") && message.contains("moment"),
        "{message}"
    );
    // spin-orbit crystals, and a symmetry table without the shell-occupation map
    let spec = load(&ctx, "cuh_7").hubbard.unwrap().to_json();
    let parse = |spinor, permutations| {
        mandacaru_core::periodic::hubbard::Spec::parse(
            &spec,
            26,
            1,
            spinor,
            mandacaru_core::math::storage::Precision::Float64,
            permutations,
        )
    };
    assert!(parse(false, 0).unwrap().is_some());
    assert!(parse(true, 0).err().unwrap().to_string().contains("spinor"));
    assert!(parse(false, 2)
        .err()
        .unwrap()
        .to_string()
        .contains("symmetry"));
    // the plain entry point would drop the operator
    let mut c = load(&ctx, "cuh_7");
    with_u(&mut c, 4.);
    let potential = vec![0.; c.grid.index.points()];
    let moments = vec![C64::default(); c.channels];
    let message = diagonalize(&c, &c.points[0], &potential, &moments, None)
        .err()
        .expect("must raise")
        .to_string();
    assert!(message.contains("diagonalize_with"), "{message}");
    assert!(try_load(&ctx, "cuh_7", |_| {}).is_ok());
}
