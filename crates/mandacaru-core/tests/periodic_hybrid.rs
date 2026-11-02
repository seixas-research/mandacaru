//! The SCF of a hybrid crystal: the exact exchange K(k) is part of the mixed state, the stopping
//! rule waits for it, and the run does not depend on how the workers are scheduled.
//!
//! The crystal is the silicon fixture on the Gamma point alone, which is a whole mesh (the wedge
//! of the 2x2x2 mesh is refused by a hybrid). It is a mechanics fixture, not a physical reference.
use mandacaru_core::{
    periodic::{
        exchange_state,
        solver::{diagonalize_with_exchange, Iteration},
        Crystal, Options, PeriodicResult,
    },
    Context,
};
mod setup_support;
use serde_json::{json, Value};
use setup_support::{library_of, sites};

fn gamma_hybrid(ctx: &Context) -> Crystal {
    hybrid(ctx, "si_crystal.json")
}

/// The fixture as a hybrid on its first k-point alone (Gamma is a whole mesh).
fn hybrid(ctx: &Context, fixture: &str) -> Crystal {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/phase5")
        .join(fixture);
    let mut value: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    value["functional"] = json!("hse06");
    value["sites"] = sites(&value, library_of(fixture));
    let mut gamma = value["kpoints"][0].clone();
    gamma["weight"] = json!(1.0);
    value["kpoints"] = json!([gamma]);
    value["permutations"] = json!([]);
    Crystal::from_json(ctx, &value.to_string()).unwrap()
}

fn context(threads: usize) -> Context {
    Context::new(Some(threads), 512 << 20).unwrap()
}

/// The three-term rule that the loop applied before the exchange operator was part of it.
fn density_rule(row: &Iteration, options: &Options) -> bool {
    row.delta.is_some_and(|d| d.abs() < options.tolerance)
        && row.residual < options.density_tolerance
        && row.kinetic_residual < options.density_tolerance
        && row.moment_residual < options.density_tolerance
}

/// Largest distance between the SCF eigenvalues and the bands of the converged density matrices
/// (exchange rebuilt from them, as the band and Wannier paths do), in Hartree.
fn band_mismatch(crystal: &Crystal, result: &PeriodicResult, options: &Options) -> f64 {
    let n = crystal.grid.index.points();
    let field = if crystal.spinor {
        crystal.field_channels() * n
    } else {
        n
    };
    let mut worst = 0f64;
    for (k, point) in crystal.points.iter().enumerate() {
        let exchange = crystal
            .hybrid_exchange_at_k(&result.matrices, point, 0, options.exxdiv_treatment)
            .unwrap();
        let (bands, _, _) = diagonalize_with_exchange(
            crystal,
            point,
            &result.potential.as_slice()[..field],
            result.moment_potential.as_slice(),
            None,
            None,
            Some(&exchange),
        )
        .unwrap();
        for (band, scf) in bands.iter().zip(&result.eigenvalues[k]) {
            worst = worst.max((band + crystal.reference - scf).abs());
        }
    }
    worst
}

#[test]
fn a_hybrid_scf_repeats_bitwise_at_one_thread() {
    let options = Options::default();
    let runs: Vec<_> = (0..2)
        .map(|_| {
            let ctx = context(1);
            let crystal = gamma_hybrid(&ctx);
            let result = crystal.solve(&options).unwrap();
            assert!(result.converged);
            assert!(ctx.workers_peak() <= 1, "{}", ctx.workers_peak());
            result
        })
        .collect();
    assert_eq!(runs[0].free_energy.to_bits(), runs[1].free_energy.to_bits());
    assert_eq!(runs[0].eigenvalues, runs[1].eigenvalues);
    assert_eq!(runs[0].history.len(), runs[1].history.len());
    for (a, b) in runs[0].history.iter().zip(&runs[1].history) {
        assert_eq!(a.exchange_residual, b.exchange_residual);
    }
}

#[test]
fn the_stopping_rule_waits_for_the_exchange_operator() {
    let options = Options::default();
    let row = |exchange_residual| Iteration {
        iteration: 9,
        free_energy: -1.,
        delta: Some(1e-9),
        residual: 1e-7,
        moment_residual: 1e-7,
        magnetization: 0.,
        kinetic_residual: 0.,
        exchange_residual,
    };
    // Every other residual passes: only the operator decides, in Hartree against `tolerance`.
    assert!(density_rule(&row(None), &options));
    assert!(row(None).converged(&options));
    assert!(row(Some(0.5 * options.tolerance)).converged(&options));
    assert!(!row(Some(2. * options.tolerance)).converged(&options));
    assert!(!row(Some(1e-3)).converged(&options));
    let mut slow = row(Some(1e-3));
    slow.delta = None;
    assert!(!slow.converged(&options));

    // Through the loop: with the density residual left loose, the old rule is met while K still
    // moves by more than the tolerance; the run goes on until K has stopped.
    let loose = Options {
        tolerance: 1e-8,
        density_tolerance: 1e3,
        ..Options::default()
    };
    let ctx = context(1);
    let crystal = gamma_hybrid(&ctx);
    let result = crystal.solve(&loose).unwrap();
    assert!(result.converged);
    let old_stop = result
        .history
        .iter()
        .position(|r| density_rule(r, &loose))
        .expect("the old rule is met");
    let stop = result.history.len() - 1;
    assert!(
        result.history[old_stop].exchange_residual.unwrap() >= loose.tolerance,
        "the old rule stops with the operator already converged: choose a tighter case"
    );
    assert!(old_stop < stop, "old stop {old_stop}, stop {stop}");
    assert!(result.history[stop].exchange_residual.unwrap() < loose.tolerance);
    println!(
        "old rule would stop at iteration {} (exchange residual {:.3e} Ha); the run stops at {} ({:.3e} Ha)",
        old_stop + 1,
        result.history[old_stop].exchange_residual.unwrap(),
        stop + 1,
        result.history[stop].exchange_residual.unwrap()
    );
}

#[test]
fn the_scf_eigenvalues_are_the_bands_of_the_converged_density_and_the_energy_does_not_move() {
    let ctx = context(1);
    let crystal = gamma_hybrid(&ctx);
    let options = Options::default();
    let result = crystal.solve(&options).unwrap();
    assert!(result.converged);
    // First order in the operator change, bounded by the Frobenius norm that stopped the loop.
    let last = result.history.last().unwrap().exchange_residual.unwrap();
    let mismatch = band_mismatch(&crystal, &result, &options);
    println!(
        "default stop: {} iterations, K residual {last:.3e}, band mismatch {mismatch:.3e} Ha",
        result.history.len()
    );
    assert!(last < options.tolerance);
    assert!(mismatch <= last + 1e-12, "{mismatch:e} vs {last:e}");
    let tight = Options {
        tolerance: 1e-11,
        density_tolerance: 1e-9,
        ..Options::default()
    };
    let exact = crystal.solve(&tight).unwrap();
    assert!(exact.converged);
    let moved = (exact.free_energy - result.free_energy).abs();
    println!(
        "tight stop: {} iterations, energy difference {moved:.3e} Ha",
        exact.history.len()
    );
    assert!(moved < 1e-9, "{moved:e}");
    assert!(band_mismatch(&crystal, &exact, &tight) < 1e-10);
    assert_eq!(exchange_state::len(&crystal), 8 * 8);
}

#[test]
fn a_spinor_hybrid_mixes_a_two_component_exchange_operator() {
    let ctx = context(1);
    let crystal = hybrid(&ctx, "si_so.json");
    assert!(crystal.spinor);
    // The operator is 2m x 2m: the spin blocks of the basis, one per k-point.
    assert_eq!(exchange_state::dimension(&crystal), 2 * crystal.m);
    assert_eq!(exchange_state::len(&crystal), 4 * crystal.m * crystal.m);
    let options = Options::default();
    let result = crystal.solve(&options).unwrap();
    assert!(result.converged);
    let last = result.history.last().unwrap().exchange_residual.unwrap();
    let mismatch = band_mismatch(&crystal, &result, &options);
    println!(
        "spinor hybrid: {} iterations, K residual {last:.3e}, band mismatch {mismatch:.3e} Ha",
        result.history.len()
    );
    assert!(last < options.tolerance);
    assert!(mismatch <= last + 1e-12, "{mismatch:e} vs {last:e}");
}
