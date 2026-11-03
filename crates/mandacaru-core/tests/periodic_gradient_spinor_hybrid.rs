//! F2a (SH-F1 + SH-F2): the force of a HYBRID (HSE06) crystal WITH SPIN-ORBIT COUPLING, time-reversal
//! symmetric (non-magnetic), on a real converged SCF over the WHOLE complex-k mesh, against the
//! reconverged free-energy finite difference of the same discretized energy (two step families of
//! `freeze_periodic_gradients.py`, 14.4 item 1): the displaced Si cell with the `pbe-fr` datasets
//! on the 3x1x1 Gamma-centered mesh (k = 0, +-1/3: two non-time-reversal-invariant points), every
//! coordinate of both atoms. No Python. Needs `MANDACARU_PAW_PATH`.
//!
//! Pinned before the force: the force differentiates THE energy the SCF minimizes
//! (`hybrid_state_energy` equals the SCF energy), the bands `W` is built from are those of the
//! hybrid spinor `H - a K[D_out]` (they reproduce the SCF's), and Kramers holds on the mesh
//! (`E_n(+k) = E_n(-k)`). A halved mesh (what the Python build makes of a time-reversal spinor) is
//! not a hybrid crystal: the build refuses it, so the half-versus-whole comparison of the semilocal
//! spinor force (A2c) has no hybrid counterpart.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;

use mandacaru_core::periodic::gradient::{contract, freeze, scf, Breakdown};
use periodic_gradient_crystals_support::gates::*;
use periodic_gradient_spinor_support::Spinor;

const CASE: &str = "si_hsefr_tr3";
const ENERGY_GATE: f64 = 1e-9;
const BAND_GATE: f64 = 1e-9;

fn net(term: &[[f64; 3]]) -> f64 {
    (0..3)
        .map(|a| term.iter().map(|g| g[a]).sum::<f64>().abs())
        .fold(0f64, f64::max)
}

fn peak(term: &[[f64; 3]]) -> f64 {
    term.iter().flatten().fold(0f64, |m, v| m.max(v.abs()))
}

fn chain(rows: &[[f64; 3]], columns: &[[f64; 3]]) -> Vec<[f64; 3]> {
    rows.iter()
        .zip(columns)
        .map(|(r, c)| [r[0] + c[0], r[1] + c[1], r[2] + c[2]])
        .collect()
}

#[test]
fn si_hse06_fr_force_is_the_reconverged_finite_difference() {
    let start = std::time::Instant::now();
    let spinor = Spinor::load(CASE);
    let data = &spinor.data;
    let c = &data.crystal;
    assert!(c.spinor && c.time_reversal && c.spins == 1);
    // A whole mesh with complex points: 0 and +-1/3 along x, equal weights.
    assert_eq!(c.points.len(), 3);
    assert!(c.points.iter().all(|p| (p.weight - 1. / 3.).abs() < 1e-12));
    let length = |k: [f64; 3]| (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
    let off: Vec<usize> = (0..3).filter(|&i| length(c.points[i].k) > 1e-9).collect();
    assert_eq!(off.len(), 2, "Gamma and one +-k pair");
    assert!(
        (0..3).all(|a| (c.points[off[0]].k[a] + c.points[off[1]].k[a]).abs() < 1e-9),
        "the two points are k and -k, not time-reversal invariant"
    );
    let result = data.solve();
    let reference = data.manifest["free_energy_ha"].as_f64().unwrap();
    assert!(
        (result.free_energy - reference).abs() < ENERGY_GATE,
        "the SCF is not the oracle's: {:e}",
        result.free_energy - reference
    );
    let tie = scf::hybrid_state_energy(c, &result.matrices, data.options.exxdiv_treatment).unwrap()
        - result.energy;
    eprintln!("{CASE}: hybrid_state_energy - SCF energy = {tie:e}");
    assert!(tie.abs() < 1e-12, "energy tie {tie:e}");

    let state = data.state(&result);
    let cores = data.cores();
    let provider = spinor.provider(&cores);
    let frozen = freeze(&provider, &state, None).unwrap();
    let mut band_gap = 0f64;
    for (k, eps) in result.eigenvalues.iter().enumerate() {
        for (a, b) in eps.iter().zip(&frozen.state.bands[0][k].eigenvalues) {
            band_gap = band_gap.max((a - b).abs());
        }
    }
    eprintln!("{CASE}: re-diagonalized eigenvalues - SCF eigenvalues: {band_gap:e}");
    assert!(band_gap < BAND_GATE, "bands {band_gap:e}");
    // Kramers: E_n(+k) = E_n(-k) between the two complex points (and not a trivial list).
    let (plus, minus) = (off[0], off[1]);
    let kramers = result.eigenvalues[plus]
        .iter()
        .zip(&result.eigenvalues[minus])
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    let spread = result.eigenvalues[plus]
        .iter()
        .fold(0f64, |m, e| m.max(e.abs()));
    eprintln!("{CASE}: Kramers E_n(+k) - E_n(-k): {kramers:e} (band scale {spread:.3})");
    // The semilocal spinor holds Kramers to 1e-15 (A2c). The hybrid holds it to 5e-8 Ha, the
    // size of the Nyquist-plane asymmetry of the compensation form factor in the discretized
    // exchange (SH-1a: `conj(T_(A,L,M)(Q)) = (-1)^M T_(A,L,-M)(-Q)` fails on the Nyquist
    // planes). It is a property of the ENERGY the force differentiates, so the force follows
    // it (the displaced-crystal gate of `periodic_gradient_spinor_hybrid_mesh.rs` differences
    // the same asymmetric energy); the gate here is 10x the measured value.
    assert!(spread > 0.1 && kramers < 5e-7, "{kramers:e}");

    let breakdown: Breakdown = contract(&provider, &frozen).unwrap();
    eprintln!(
        "{CASE}: SCF + engine {:.1} s ({} iterations)",
        start.elapsed().as_secs_f64(),
        result.history.len()
    );
    let other = second_family(CASE);
    let worst = compare(CASE, data, Some(&other), &breakdown.forces());
    eprintln!("{CASE}: worst miss {worst:.3} gates");
    // The exchange terms are resolved: each is far above the plan gate on the largest coordinate.
    for (name, term) in [
        ("exchange_phase", &breakdown.exchange_phase),
        ("exchange_functions", &breakdown.exchange_functions),
        ("exchange_rows", &breakdown.exchange_rows),
        ("exchange_columns", &breakdown.exchange_columns),
        ("spin_orbit_rows", &breakdown.spin_orbit_rows),
        ("spin_orbit_columns", &breakdown.spin_orbit_columns),
    ] {
        eprintln!(
            "{CASE}: {name} peak {:.3e} Ha/Bohr = {:.1} plan gates",
            peak(term),
            peak(term) / PLAN_GATE
        );
    }
    // The spin-orbit share of the force on Si is not resolved by this oracle: a heavy cell
    // would be (see the log of F2a), and the zero-coupling limit and the displaced-crystal gate
    // carry the spin structure instead.
    let so = chain(&breakdown.spin_orbit_rows, &breakdown.spin_orbit_columns);
    eprintln!(
        "{CASE}: net spin-orbit share of the force {:.3e} Ha/Bohr = {:.2} plan gates",
        peak(&so),
        peak(&so) / PLAN_GATE
    );
    // Mesh-free sum rules.
    for (name, term) in [
        ("dense", &breakdown.dense),
        ("constants", &breakdown.constants),
        ("short_range", &breakdown.short_range),
    ] {
        assert!(net(term) < 1e-12, "{name} sum rule {:e}", net(term));
    }
    assert!(
        net(&chain(
            &breakdown.projection_rows,
            &breakdown.projection_columns
        )) < 1e-12,
        "projection chain"
    );
    assert!(
        net(&chain(
            &breakdown.exchange_rows,
            &breakdown.exchange_columns
        )) < 1e-12,
        "exchange projection chain"
    );
    eprintln!(
        "{CASE}: egg-box (sum over atoms) exchange phase {:.2e}, functions {:.2e}; net force {:.2e} Ha/Bohr",
        net(&breakdown.exchange_phase),
        net(&breakdown.exchange_functions),
        net(&breakdown.gradient())
    );
}
