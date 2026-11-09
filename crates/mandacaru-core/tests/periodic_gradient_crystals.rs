//! CF-4: the crystal force of the Rust setup on a REAL converged SCF crystal, against the
//! reconverged free-energy finite difference of the same discretized energy (the frozen Richardson
//! oracle of `freeze_periodic_gradients.py`, 14.4 item 1), plus the translation sum rule, the
//! mesh-free terms, the egg-box floor and the SCF-tolerance sensitivity. Collinear scalar LDA,
//! whole k mesh, no Python. Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;

use mandacaru_core::periodic::gradient::gradient;
use periodic_gradient_crystals_support::{gates::*, *};

/// One case: the SCF, the engine, the comparison with the frozen oracle, the sum rules and the
/// egg-box floor. One SCF serves all of it (a test per case keeps each under the 180 s budget).
fn check(case: &'static str) {
    let start = std::time::Instant::now();
    let data = Data::load(case);
    let result = data.solve();
    let scf_seconds = start.elapsed().as_secs_f64();
    let reference = data.manifest["free_energy_ha"].as_f64().unwrap();
    assert!(
        (result.free_energy - reference).abs() < 1e-9,
        "the SCF is not the oracle's: {:e}",
        result.free_energy - reference
    );
    let state = data.state(&result);
    let cores = data.cores();
    let provider = provider(&data, &cores);
    let breakdown = gradient(&provider, &state, None).unwrap();
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
        let miss = (forces[atom][axis] - oracle).abs();
        eprintln!(
            "{case} atom {atom} axis {axis}: force {:+.9e}, oracle {oracle:+.9e}, miss {miss:.2e} Ha/Bohr = {:.2e} eV/A, floor {:.1e}, gate {:.2e}",
            forces[atom][axis],
            miss * TO_EV_PER_A,
            floor.abs(),
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
    let sum = |term: &Vec<[f64; 3]>| {
        (0..3)
            .map(|a| term.iter().map(|g| g[a]).sum::<f64>())
            .fold(0f64, |m, s| m.max(s.abs()))
    };
    // Mesh-free terms (the lattice sums and the sphere rules) are exactly translation invariant.
    let free = [
        ("dense", sum(&breakdown.dense)),
        ("constants", sum(&breakdown.constants)),
        ("short_range", sum(&breakdown.short_range)),
    ];
    let projections: Vec<[f64; 3]> = breakdown
        .projection_rows
        .iter()
        .zip(&breakdown.projection_columns)
        .map(|(r, c)| [r[0] + c[0], r[1] + c[1], r[2] + c[2]])
        .collect();
    let bound = sum(&projections);
    // Mesh-bound terms carry the egg-box ripple: REPORTED, not gated.
    let mesh = [
        ("overlap", sum(&breakdown.overlap)),
        ("kinetic", sum(&breakdown.kinetic)),
        ("local", sum(&breakdown.local)),
        ("hats", sum(&breakdown.hats)),
        ("ion", sum(&breakdown.ion)),
        ("core", sum(&breakdown.core)),
    ];
    let net = sum(&breakdown.gradient());
    eprintln!(
        "{case}: mesh-free sums {free:?}, projections {bound:e}; mesh-bound (egg-box) {mesh:?}; net force {net:e} Ha/Bohr = {:.2e} eV/A",
        net * TO_EV_PER_A
    );
    for (name, s) in free {
        assert!(s < 1e-12, "{case}: {name} sum rule {s:e}");
    }
    assert!(bound < 1e-12, "{case}: projection rows + columns {bound:e}");
    // The net force is the sum of the egg-box ripples of the mesh terms plus what the mesh-free
    // terms leave: REPORTED above, not asserted (an inequality against the sum of the ripples
    // would hold for any net force the triangle inequality allows).
}

#[test]
#[ignore = "slow: 84 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_crystals -- --ignored si_force_is_the_reconverged_finite_difference"]
fn si_force_is_the_reconverged_finite_difference() {
    check("si");
}

#[test]
#[ignore = "slow: 139 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_crystals -- --ignored al2_force_is_the_reconverged_finite_difference"]
fn al2_force_is_the_reconverged_finite_difference() {
    check("al2");
}

#[test]
fn lih_force_is_the_reconverged_finite_difference() {
    check("lih");
}

/// The LiH cell must keep a lattice that is not symmetric AND whose transpose is another lattice
/// (the Si cell of the fixture is a circulant fcc primitive cell: `L != L^T`, but both generate the
/// same lattice, so a transposed Si lattice is invisible to every sum over images; the three
/// older frozen cells are symmetric). `a_transposed_lattice_is_detected` and the controls of
/// `periodic_gradient_crystals_controls.rs` use this cell.
#[test]
fn the_lih_cell_is_not_symmetric() {
    let data = Data::load("lih");
    let asymmetry = (data.lattice - data.lattice.transpose()).norm() / data.lattice.norm();
    eprintln!("lattice asymmetry {asymmetry:.3}");
    assert!(asymmetry > 0.05);
}

/// 14.4 item 5 (iv), as corrected by the S1 packages: the force is first order in the SCF
/// residual (the output-density route only shrinks the constant), so a force at the gate needs a
/// converged state. LiH solved at three tolerances: the miss against the oracle falls with the
/// tolerance, and a loose state is far outside the gate.
#[test]
fn the_force_follows_the_scf_residual() {
    let data = Data::load("lih");
    let cores = data.cores();
    let mut misses = Vec::new();
    for (tolerance, density) in [(1e-4, 1e-3), (1e-7, 1e-5), (1e-11, 1e-9)] {
        let options = mandacaru_core::periodic::Options {
            tolerance,
            density_tolerance: density,
            ..data.options.clone()
        };
        let result = data.crystal.solve(&options).unwrap();
        assert!(result.converged);
        let provider = provider(&data, &cores);
        let forces = gradient(&provider, &data.state(&result), None)
            .unwrap()
            .forces();
        let miss = data.manifest["coordinates"]
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
            .fold(0., f64::max);
        eprintln!("LiH SCF density tolerance {density:e}: worst miss {miss:.2e} Ha/Bohr");
        misses.push(miss);
    }
    // Outside the plan gate (measured 10x it), and 1.6e3 x the converged miss: the force is first
    // order in the residual, so a state a thousand times looser in density is orders worse.
    assert!(
        misses[0] > 2. * PLAN_GATE && misses[0] > 100. * misses[2],
        "the loose state is not outside the gate: {misses:?}"
    );
    assert!(misses[0] > misses[1] && misses[1] > misses[2], "{misses:?}");
    assert!(misses[2] < PLAN_GATE);
}
