//! HX-W: the force of a HYBRID (HSE06) crystal on a real converged SCF, against the reconverged
//! free-energy finite difference of the same discretized energy (the frozen Richardson oracle of
//! `freeze_periodic_gradients.py`, 14.4 item 1): a pair in a triclinic box at Gamma, and the
//! displaced Si cell on the whole 2x2x2 mesh with both divergence treatments of the exact
//! exchange. Collinear scalar, whole k mesh, no Python. Needs `MANDACARU_PAW_PATH`.
//!
//! The energy tie and the hybrid eigenvalues are checked first: the force differentiates THE
//! energy the SCF minimizes (`hybrid_state_energy` equals the SCF energy), and the bands the
//! engine builds `W` from are those of `H_semilocal - a K[D_out]` (they reproduce the SCF's).
mod periodic_gradient_crystals_support;

use mandacaru_core::periodic::{
    gradient::{contract, freeze, gradient, scf},
    setup::provider::Core,
    Options,
};
use periodic_gradient_crystals_support::{gates::*, *};

/// Bound on the SCF energy and the hybrid tie (Hartree).
const ENERGY_GATE: f64 = 1e-9;
/// The re-diagonalized bands against the SCF's: the SCF's own input exchange differs from the one
/// of its output density by the exchange residual it stops at (1e-11 in the oracle options).
const BAND_GATE: f64 = 1e-9;

fn sum(term: &[[f64; 3]]) -> f64 {
    (0..3)
        .map(|a| term.iter().map(|g| g[a]).sum::<f64>())
        .fold(0f64, |m, s| m.max(s.abs()))
}

/// One case: the SCF, the energy tie, the engine, the comparison with the frozen oracle.
fn check(case: &'static str) {
    let start = std::time::Instant::now();
    let data = Data::load(case);
    let result = data.solve();
    let scf_seconds = start.elapsed().as_secs_f64();
    let reference = data.manifest["free_energy_ha"].as_f64().unwrap();
    assert!(
        (result.free_energy - reference).abs() < ENERGY_GATE,
        "{case}: the SCF is not the oracle's: {:e}",
        result.free_energy - reference
    );
    let tie = scf::hybrid_state_energy(
        &data.crystal,
        &result.matrices,
        data.options.exxdiv_treatment,
    )
    .unwrap()
        - result.energy;
    eprintln!("{case}: hybrid_state_energy - SCF energy = {tie:e}");
    assert!(tie.abs() < 1e-12, "{case}: energy tie {tie:e}");

    let state = data.state(&result);
    let cores = data.cores();
    let provider = provider(&data, &cores);
    let frozen = freeze(&provider, &state, None).unwrap();
    // The re-diagonalized hybrid bands are the SCF's: the eigenvalues carry the exchange.
    let nk = data.crystal.points.len();
    let mut band_gap = 0f64;
    for (index, eps) in result.eigenvalues.iter().enumerate() {
        let again = &frozen.state.bands[index / nk][index % nk].eigenvalues;
        for (a, b) in eps.iter().zip(again) {
            band_gap = band_gap.max((a - b).abs());
        }
    }
    eprintln!("{case}: re-diagonalized eigenvalues - SCF eigenvalues: {band_gap:e}");
    assert!(band_gap < BAND_GATE, "{case}: bands {band_gap:e}");

    let breakdown = contract(&provider, &frozen).unwrap();
    eprintln!(
        "{case}: SCF {scf_seconds:.1} s ({} iterations), engine {:.1} s",
        result.history.len(),
        start.elapsed().as_secs_f64() - scf_seconds
    );
    let forces = breakdown.forces();
    for entry in data.manifest["coordinates"].as_array().unwrap() {
        let (atom, axis) = (
            entry["atom"].as_u64().unwrap() as usize,
            entry["axis"].as_u64().unwrap() as usize,
        );
        let oracle = -entry["rr"].as_f64().unwrap();
        let floor = entry["floor_richardson"].as_f64().unwrap();
        let extrapolated = entry["floor_extrapolated"].as_f64().unwrap();
        let miss = (forces[atom][axis] - oracle).abs();
        eprintln!(
            "{case} atom {atom} axis {axis}: force {:+.9e}, oracle {oracle:+.9e}, miss {miss:.2e} Ha/Bohr = {:.2e} eV/A, floor {:.1e} (extrapolation {:.1e}), gate {:.2e}",
            forces[atom][axis],
            miss * TO_EV_PER_A,
            floor.abs(),
            extrapolated.abs(),
            gate(floor)
        );
        assert!(
            miss < gate(floor),
            "{case} atom {atom} axis {axis}: {miss:e} against {:e}",
            gate(floor)
        );
        // The component is not vacuous: the oracle itself is far above its gate.
        assert!(
            oracle.abs() > 100. * gate(floor),
            "{case} atom {atom} axis {axis}: the oracle {oracle:e} is not above its gate"
        );
    }
    // The exchange terms are reported per atom: they are what this package added.
    for (name, term) in [
        ("exchange_phase", &breakdown.exchange_phase),
        ("exchange_functions", &breakdown.exchange_functions),
        ("exchange_rows", &breakdown.exchange_rows),
        ("exchange_columns", &breakdown.exchange_columns),
    ] {
        eprintln!("{case}: {name} {term:?}");
    }
    // Mesh-free terms stay exactly translation invariant; the projection chain of the exchange
    // (rows + columns) is too, because the ownership identity moves a row and a column together.
    let chain: Vec<[f64; 3]> = breakdown
        .exchange_rows
        .iter()
        .zip(&breakdown.exchange_columns)
        .map(|(r, c)| [r[0] + c[0], r[1] + c[1], r[2] + c[2]])
        .collect();
    let (dense, constants, short_range) = (
        sum(&breakdown.dense),
        sum(&breakdown.constants),
        sum(&breakdown.short_range),
    );
    let (bound, exchange_mesh) = (sum(&chain), sum(&breakdown.exchange_functions));
    let phase = sum(&breakdown.exchange_phase);
    let net = sum(&breakdown.gradient());
    eprintln!(
        "{case}: mesh-free sums dense {dense:e}, constants {constants:e}, short_range {short_range:e}; \
         exchange projection chain {bound:e}; mesh-bound exchange (egg-box ripple, reported) \
         functions {exchange_mesh:e}, phase {phase:e}; net force {net:e} Ha/Bohr = {:.2e} eV/A",
        net * TO_EV_PER_A
    );
    for (name, s) in [
        ("dense", dense),
        ("constants", constants),
        ("short_range", short_range),
        ("exchange projection chain", bound),
    ] {
        assert!(s < 1e-12, "{case}: {name} sum rule {s:e}");
    }
}

#[test]
#[ignore = "slow: 19 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_hybrid -- --ignored lih_hse06_force_is_the_reconverged_finite_difference"]
fn lih_hse06_force_is_the_reconverged_finite_difference() {
    check("lih_hse06");
}

#[test]
#[ignore = "slow: 171 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_hybrid -- --ignored si_hse06_force_is_the_reconverged_finite_difference"]
fn si_hse06_force_is_the_reconverged_finite_difference() {
    check("si_hse06");
}

#[test]
#[ignore = "slow: 172 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_hybrid -- --ignored si_hse06_with_gygi_baldereschi_is_the_reconverged_finite_difference"]
fn si_hse06_with_gygi_baldereschi_is_the_reconverged_finite_difference() {
    check("si_hse06_gb");
}

/// The worst miss over the frozen coordinates of the force of `case` solved at `options`.
fn miss_at(data: &Data, cores: &[Option<Core<'_>>], options: &Options) -> f64 {
    let result = data.crystal.solve(options).unwrap();
    assert!(result.converged);
    let provider = provider(data, cores);
    let forces = gradient(&provider, &data.state(&result), None)
        .unwrap()
        .forces();
    data.manifest["coordinates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let (atom, axis) = (
                e["atom"].as_u64().unwrap() as usize,
                e["axis"].as_u64().unwrap() as usize,
            );
            (forces[atom][axis] + e["rr"].as_f64().unwrap()).abs()
        })
        .fold(0., f64::max)
}

/// 14.4 item 5 (iv) for a hybrid: the force is first order in the SCF residual (the density and
/// the exchange operators the SCF stops at), so a force at the gate needs a converged state.
/// LiH solved at three tolerances.
#[test]
#[ignore = "slow: 36 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_hybrid -- --ignored the_hybrid_force_follows_the_scf_residual"]
fn the_hybrid_force_follows_the_scf_residual() {
    let data = Data::load("lih_hse06");
    let cores = data.cores();
    let misses: Vec<f64> = [(1e-4, 1e-3), (1e-7, 1e-5), (1e-11, 1e-9)]
        .iter()
        .map(|&(tolerance, density)| {
            let options = Options {
                tolerance,
                density_tolerance: density,
                ..data.options.clone()
            };
            let miss = miss_at(&data, &cores, &options);
            eprintln!("LiH HSE06 SCF density tolerance {density:e}: worst miss {miss:.2e} Ha/Bohr");
            miss
        })
        .collect();
    assert!(
        misses[0] > 2. * PLAN_GATE && misses[0] > 100. * misses[2],
        "the loose state is not outside the gate: {misses:?}"
    );
    assert!(misses[0] > misses[1] && misses[1] > misses[2], "{misses:?}");
    assert!(misses[2] < PLAN_GATE);
}

/// The hybrid Si miss at the oracle tolerances (1.4e-8 Ha/Bohr, 0.07 gates) against a state
/// converged 1000x tighter in density: the part of the miss that is the SCF residual.
#[test]
#[ignore = "two Si gradients of about 2 minutes each: measured on 2026-11-02, see the log"]
fn the_hybrid_si_miss_is_the_scf_residual() {
    let data = Data::load("si_hse06");
    let cores = data.cores();
    let tight = Options {
        tolerance: 1e-12,
        density_tolerance: 1e-12,
        ..data.options.clone()
    };
    let miss = miss_at(&data, &cores, &tight);
    eprintln!("Si HSE06 SCF density tolerance 1e-12: worst miss {miss:.2e} Ha/Bohr");
    assert!(miss < PLAN_GATE);
}
