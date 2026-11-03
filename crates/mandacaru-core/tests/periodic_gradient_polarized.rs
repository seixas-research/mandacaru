//! F1: the force of a COLLINEAR SPIN-POLARIZED (`spins = 2`, unrestricted) LDA/PBE crystal on a real
//! converged SCF, against the reconverged free-energy finite difference of the same discretized
//! energy (the frozen Richardson oracle of `freeze_periodic_gradients.py`, 14.4 item 1, two step
//! families): the BeH radical (Be carries a large partial core and the spin density) in a box at
//! Gamma, LDA and PBE, with every coordinate of both atoms. Whole k mesh, no Python. Needs
//! `MANDACARU_PAW_PATH`.
//!
//! What spin adds to the engine, and what each test pins:
//! * the local-potential term contracts `D_s` with the potential of ITS spin (`v_ee + v_xc,s`) and
//!   the projection terms use `D_s`, `W_s` per spin (the compensation moments and the hats see the
//!   spin SUM);
//! * the partial core is split in halves between the spin densities, so it is differentiated
//!   against `(v_up + v_dn) / 2`;
//! * the restricted limit: a `spins = 2` run with zero moment is the `spins = 1` run, and its force
//!   equals the `spins = 1` force of the same crystal.
mod periodic_gradient_crystals_support;

use mandacaru_core::periodic::gradient::gradient;
use periodic_gradient_crystals_support::{gates::*, *};
use serde_json::Value;

/// One case: the SCF, the engine, the comparison with the frozen oracle on every coordinate, the
/// sum rules and the egg-box ripple. One SCF serves all of it.
fn check(case: &'static str, moment: f64) {
    let start = std::time::Instant::now();
    let data = Data::load(case);
    assert_eq!(data.crystal.spins, 2, "{case}: a spin-polarized snapshot");
    let result = data.solve();
    let reference = data.manifest["free_energy_ha"].as_f64().unwrap();
    assert!(
        (result.free_energy - reference).abs() < 1e-9,
        "{case}: the SCF is not the oracle's: {:e}",
        result.free_energy - reference
    );
    // The magnetic state, not a collapsed one: the moment is asserted, not assumed.
    assert!(
        (result.magnetization - moment).abs() < 0.05,
        "{case}: moment {}",
        result.magnetization
    );
    let scf_seconds = start.elapsed().as_secs_f64();
    let state = data.state(&result);
    let cores = data.cores();
    let provider = provider(&data, &cores);
    let breakdown = gradient(&provider, &state, None).unwrap();
    eprintln!(
        "{case}: moment {:.6}, SCF {scf_seconds:.1} s ({} iterations), engine {:.1} s",
        result.magnetization,
        result.history.len(),
        start.elapsed().as_secs_f64() - scf_seconds
    );
    let worst = compare(case, &data, Some(&second_family(case)), &breakdown.forces());
    eprintln!("{case}: worst miss {worst:.3} gates");
    // Mesh-free terms are exactly translation invariant, the projection chain moves a row and a
    // column together; the mesh-bound ones carry the egg-box ripple (reported).
    for (name, term) in [
        ("dense", &breakdown.dense),
        ("constants", &breakdown.constants),
        ("short_range", &breakdown.short_range),
    ] {
        assert!(net(term) < 1e-12, "{case}: {name} sum rule {:e}", net(term));
    }
    let projections: Vec<[f64; 3]> = breakdown
        .projection_rows
        .iter()
        .zip(&breakdown.projection_columns)
        .map(|(r, c)| [r[0] + c[0], r[1] + c[1], r[2] + c[2]])
        .collect();
    assert!(net(&projections) < 1e-12, "{case}: projection chain");
    eprintln!(
        "{case}: egg-box ripple (sum of a term over the atoms, Ha/Bohr): overlap {:.1e}, kinetic {:.1e}, local {:.1e}, hats {:.1e}, ion {:.1e}, core {:.1e}; net force {:.1e}",
        net(&breakdown.overlap),
        net(&breakdown.kinetic),
        net(&breakdown.local),
        net(&breakdown.hats),
        net(&breakdown.ion),
        net(&breakdown.core),
        net(&breakdown.gradient()),
    );
}

#[test]
fn beh_lda_force_is_the_reconverged_finite_difference() {
    check("beh_lda", 1.0);
}

/// A metal on a 2x2x2 mesh (8 k-points, the whole mesh) with two spin channels: the common Fermi
/// level, the partial occupations and the complex phases of the mesh through the polarized route.
/// The SCF started with a ferromagnetic moment (1 + 1) converges to moment 0 (asserted: the
/// frozen oracle is that non-magnetic state; a ferromagnetic Fe cell did not converge), so this is
/// the restricted limit on a mesh, not a magnetic gate.
#[test]
fn ni2_lda_zero_moment_mesh_force_is_the_reconverged_finite_difference() {
    check("ni2_lda", 0.0);
}

#[test]
fn beh_pbe_force_is_the_reconverged_finite_difference() {
    check("beh_pbe", 1.0);
}

/// A snapshot with another start: `initial_m` scaled (zero keeps the two spin densities equal for
/// the whole run).
fn restarted(case: &'static str, scale: f64) -> Data {
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(root().join(format!("{case}.json"))).unwrap(),
    )
    .unwrap();
    let mut snapshot = snapshot_of(case, &manifest);
    for x in snapshot["initial_m"].as_array_mut().unwrap() {
        *x = Value::from(x.as_f64().unwrap() * scale);
    }
    Data::build(case, manifest, snapshot)
}

/// The moment matters: the same crystal started without a moment stays non-magnetic, and its
/// force differs from the magnetic one by far more than the gate. A force built from the spin sum
/// alone would not tell the two states apart.
#[test]
fn the_magnetic_and_the_non_magnetic_force_differ() {
    let magnetic = Data::load("beh_lda");
    let flat = restarted("beh_lda", 0.);
    let (a, b) = (magnetic.solve(), flat.solve());
    assert!(a.magnetization > 0.9 && b.magnetization.abs() < 1e-6);
    let cores = magnetic.cores();
    let fa = gradient(&provider(&magnetic, &cores), &magnetic.state(&a), None)
        .unwrap()
        .forces();
    let fb = gradient(&provider(&flat, &cores), &flat.state(&b), None)
        .unwrap()
        .forces();
    let worst = (0..2)
        .flat_map(|atom| (0..3).map(move |axis| (atom, axis)))
        .map(|(atom, axis)| (fa[atom][axis] - fb[atom][axis]).abs())
        .fold(0f64, f64::max);
    eprintln!("beh_lda: |F(magnetic) - F(non-magnetic)| = {worst:.3e} Ha/Bohr");
    assert!(worst > 100. * PLAN_GATE, "{worst:e}");
}
