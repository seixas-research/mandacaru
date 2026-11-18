//! Q11/SH-5a: a time-reversal-symmetric (nonmagnetic) spinor crystal on the k mesh the setup reduces
//! by the space group of its cell.
//!
//! Such a crystal carries the charge density and the charge multipoles only (`Crystal::time_reversal`:
//! the magnetization is zero by symmetry and no Pauli component is stored), and both are scalars
//! under the spin rotation of an operation, so the SCF symmetrizes them with the spatial operation
//! alone (grid-node permutation, harmonic rotation of the multipoles) and the star of a k-point
//! needs no SU(2) factor: the band energies of `Sk` equal those of `k`. A magnetic or spin-polarized
//! spinor still keeps the whole mesh (its magnetization needs the spin rotation).
//!
//! Gate, Si `pbe-fr` with the second atom moved along the cubic z axis (four operations kept by the
//! 16^3 grid, 8 points of the 2 x 2 x 2 mesh reduced to a smaller set):
//!
//! * the free energy of the reduced mesh is the whole mesh's to 1e-10 Ha, the levels of the kept
//!   points to 1e-9 Ha;
//! * the analytic force of the reduced mesh (averaged over the group, CF-6) is the whole mesh's to
//!   1e-8 Ha/Bohr and matches the central differences of the reconverged free energy of the WHOLE
//!   mesh (step 1e-3 Bohr, filter cutoff and ion width pinned) to 1e-6;
//! * controls that bite: without the group average, or with a wrong atom map, the force misses.
//!
//! Needs `MANDACARU_PAW_PATH`. Run: `MANDACARU_PAW_PATH=... cargo test --profile release-test -p
//! mandacaru-core --test periodic_spinor_reduced_mesh -- --ignored --nocapture --test-threads=1`.
mod periodic_reduced_mesh_support;

use mandacaru_core::periodic::{
    point_group::PointGroup,
    response::{forces, forces_with_group, stress},
    Crystal,
};
use nalgebra::{Matrix3, Vector3};
use periodic_reduced_mesh_support::{
    build, context, silicon, solve, tight, Cell, Points, HARTREE_EV,
};

/// The largest `|a - b|` over matching entries.
fn worst(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .fold(0f64, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
#[ignore = "slow: 36 s (measured 2026-11-16); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_spinor_reduced_mesh -- --ignored"]
fn a_reduced_spinor_mesh_equals_the_whole_mesh_in_energy_and_force_and_the_finite_differences() {
    let ctx = context();
    let cell = silicon("pbe-fr", "pbe", true, 16, 0.08);
    let whole = build(&ctx, &cell.lattice, &cell, Points::Whole);
    let reduced = build(
        &ctx,
        &cell.lattice,
        &cell,
        Points::Reduced { symmetry: true },
    );
    let group = reduced
        .description
        .as_ref()
        .and_then(|d| d.point_group.clone())
        .expect("a reduced mesh carries its operations");
    println!(
        "Si pbe-fr: {} points whole, {} reduced by {} operations (time reversal included)",
        whole.points.len(),
        reduced.points.len(),
        group.len()
    );
    assert!(group.len() >= 2 && reduced.points.len() < whole.points.len());
    assert!(reduced.time_reversal && reduced.spinor);

    let (rw, rr) = (solve(&whole, &tight()), solve(&reduced, &tight()));
    let gap = (rw.free_energy - rr.free_energy).abs();
    // The levels of every kept point against the whole mesh's (matched modulo the reciprocal
    // lattice and -k).
    let same = |a: [f64; 3], b: [f64; 3], sign: f64| {
        (0..3).all(|i| {
            let d = a[i] - sign * b[i];
            (d - d.round()).abs() < 1e-8
        })
    };
    let mut levels = 0f64;
    for (k, point) in reduced.points.iter().enumerate() {
        let at = (0..whole.points.len())
            .find(|&j| {
                same(whole.points[j].fractional, point.fractional, 1.)
                    || same(whole.points[j].fractional, point.fractional, -1.)
            })
            .expect("a kept point of the whole mesh");
        for (x, y) in rr.eigenvalues[k].iter().zip(&rw.eigenvalues[at]) {
            levels = levels.max((x - y).abs());
        }
    }
    println!(
        "free energy: whole {:.12} Ha, reduced {:.12} Ha, difference {gap:.3e}; levels {levels:.3e}",
        rw.free_energy, rr.free_energy
    );
    assert!(gap < 1e-10, "energy {gap:e}");
    assert!(levels < 1e-9, "levels {levels:e}");

    let (f_whole, f_reduced) = (forces(&whole, &rw).unwrap(), forces(&reduced, &rr).unwrap());
    let miss = worst(&f_whole, &f_reduced);
    let peak = f_whole.iter().flatten().fold(0f64, |m, x| m.max(x.abs()));
    println!("forces whole {f_whole:?}\nforces reduced {f_reduced:?}\nreduced vs whole {miss:.3e} Ha/Bohr (peak {peak:.3e})");
    assert!(miss < 1e-8, "reduced force vs whole {miss:e}");
    assert!(peak > 1e-3, "the force must be a signal");

    // Controls: the group average and the atom map are not vacuous.
    let raw = forces_with_group(&reduced, &rr, None).unwrap();
    let broken = worst(&f_whole, &raw);
    println!("control, no average: miss {broken:.3e}");
    assert!(
        broken > 1e-6,
        "no average must break the force ({broken:e})"
    );
    let shifted: Vec<Vec<usize>> = group
        .atom_maps()
        .iter()
        .map(|m| m.iter().map(|&a| (a + 1) % m.len()).collect())
        .collect();
    let wrong = PointGroup::from_parts(group.rotations().to_vec(), shifted);
    let bad = forces_with_group(&reduced, &rr, Some(wrong)).unwrap();
    let broken = worst(&f_whole, &bad);
    println!("control, wrong atom map: miss {broken:.3e}");
    assert!(
        broken > 1e-6,
        "a wrong atom map must break the force ({broken:e})"
    );

    // The reconverged central differences of the whole mesh's free energy.
    let step = 1e-3;
    let free_energy = |atom: usize, sign: f64| {
        let mut moved = cell.clone();
        moved.move_atom(atom, Vector3::new(0., 0., sign * step));
        let c = build(&ctx, &cell.lattice, &moved, Points::Whole);
        solve(&c, &tight()).free_energy
    };
    for atom in [0, 1] {
        let slope = -(free_energy(atom, 1.) - free_energy(atom, -1.)) / (2. * step);
        let analytic = f_reduced[atom][2];
        println!(
            "atom {atom} z: analytic (reduced) {analytic:+.9e}, whole-mesh central difference {slope:+.9e}, miss {:.2e}",
            (analytic - slope).abs()
        );
        assert!(
            (analytic - slope).abs() < 1e-6,
            "atom {atom}: {analytic:e} vs {slope:e}"
        );
    }
}

fn fm_nickel_cell(functional: &'static str, grid: usize, mesh: ([usize; 3], bool)) -> Cell {
    let a = 6.65 / 2.;
    // Slightly strained along c so the stress tensor has a clear non-zero traceless signal.
    let c = a * 1.03;
    Cell {
        library: "pbe-fr",
        functional,
        symbols: vec!["Ni"],
        lattice: Matrix3::from_columns(&[
            Vector3::new(0., a, c),
            Vector3::new(a, 0., c),
            Vector3::new(a, a, 0.),
        ]),
        grid: [grid; 3],
        fractional: vec![[0., 0., 0.]],
        spinor: true,
        spin: false,
        moments: Some(vec![0.8]),
        charge: None,
        mesh,
    }
}

fn afm_iron_cell(grid: usize, mesh: ([usize; 3], bool)) -> Cell {
    let a = 5.42;
    Cell {
        library: "pbe-fr",
        functional: "pbe",
        symbols: vec!["Fe", "Fe"],
        lattice: Matrix3::from_diagonal(&Vector3::new(a, a, a)),
        grid: [grid; 3],
        fractional: vec![[0., 0., 0.], [0.5, 0.5, 0.5]],
        spinor: true,
        spin: false,
        moments: Some(vec![2.2, -2.2]),
        charge: None,
        mesh,
    }
}

fn displaced_fm_fesi_cell(functional: &'static str, grid: usize, mesh: ([usize; 3], bool)) -> Cell {
    Cell {
        library: "pbe-fr",
        functional,
        symbols: vec!["Fe", "Si"],
        lattice: Matrix3::from_diagonal(&Vector3::new(6.2, 6.2, 6.8)),
        grid: [grid; 3],
        fractional: vec![[0., 0., 0.], [0.5, 0.5, 0.56]],
        spinor: true,
        spin: false,
        moments: Some(vec![2.2, 0.0]),
        charge: None,
        mesh,
    }
}

#[test]
fn magnetic_spinor_setup_classifies_unitary_and_anti_unitary_operations() {
    let ctx = context();
    // Unstrained FM bcc Fe has O_h spatial group (48 ops); magnetization along +z breaks 3-folds
    // and keeps magnetic point group 4/mm'm' (8 unitary + 8 anti-unitary = 16 ops).
    let a = 5.42;
    let fm = Cell {
        library: "pbe-fr",
        functional: "pbe",
        symbols: vec!["Fe"],
        lattice: Matrix3::from_columns(&[
            Vector3::new(-0.5 * a, 0.5 * a, 0.5 * a),
            Vector3::new(0.5 * a, -0.5 * a, 0.5 * a),
            Vector3::new(0.5 * a, 0.5 * a, -0.5 * a),
        ]),
        grid: [12; 3],
        fractional: vec![[0., 0., 0.]],
        spinor: true,
        spin: false,
        moments: Some(vec![2.2]),
        charge: None,
        mesh: ([3, 3, 3], true),
    };
    let reduced = build(&ctx, &fm.lattice, &fm, Points::Reduced { symmetry: true });
    let whole = build(&ctx, &fm.lattice, &fm, Points::Whole);
    assert!(reduced.spinor && !reduced.time_reversal);
    assert!(
        reduced.points.len() < whole.points.len(),
        "FM Fe [3,3,3] must shrink ({} vs {})",
        reduced.points.len(),
        whole.points.len()
    );
    let ops = reduced
        .description
        .as_ref()
        .and_then(|d| d.operations.as_ref())
        .expect("magnetic spinor carries setup operations");
    let flags = ops
        .anti_unitary
        .as_ref()
        .expect("magnetic spinor carries anti_unitary flags");
    assert_eq!(ops.rotations.len(), 16);
    assert_eq!(flags.len(), 16);
    assert_eq!(flags.iter().filter(|&&b| !b).count(), 8);
    assert_eq!(flags.iter().filter(|&&b| b).count(), 8);

    // Save/load JSON roundtrip preserves anti_unitary flags.
    let path = std::env::temp_dir().join(format!(
        "mandacaru-magnetic-spinor-{}.json",
        std::process::id()
    ));
    reduced.save(path.to_str().unwrap()).unwrap();
    let reloaded = Crystal::load(&ctx, path.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        reloaded
            .description
            .as_ref()
            .and_then(|d| d.operations.as_ref())
            .and_then(|o| o.anti_unitary.as_ref()),
        Some(flags)
    );

    // AFM BCC Fe (two sublattices +2.2 / -2.2): half-cell translation is anti-unitary (T * t).
    let afm = afm_iron_cell(12, ([2, 2, 2], true));
    let afm_red = build(&ctx, &afm.lattice, &afm, Points::Reduced { symmetry: true });
    let afm_ops = afm_red
        .description
        .as_ref()
        .and_then(|d| d.operations.as_ref())
        .unwrap();
    let afm_flags = afm_ops.anti_unitary.as_ref().unwrap();
    assert!(afm_flags.contains(&true) && afm_flags.contains(&false));
    let has_tr_translation = afm_ops
        .rotations
        .iter()
        .zip(&afm_ops.atom_maps)
        .zip(afm_flags)
        .any(|((r, map), &anti)| r == &[1, 0, 0, 0, 1, 0, 0, 0, 1] && map == &[1, 0] && anti);
    assert!(
        has_tr_translation,
        "AFM sublattice swap under pure translation must be anti-unitary"
    );
}

#[test]
fn magnetic_spinor_reduced_mesh_matches_whole_mesh_for_semilocal_and_hybrid() {
    let ctx = context();
    // 1) Semilocal strained FM Ni ([2,2,2] shifted: reduced vs whole, metallic smearing 0.1 eV):
    let ni = fm_nickel_cell("pbe", 16, ([2, 2, 2], false));
    let whole = build(&ctx, &ni.lattice, &ni, Points::Whole);
    let reduced = build(&ctx, &ni.lattice, &ni, Points::Reduced { symmetry: true });
    assert!(!reduced.time_reversal && reduced.spinor);
    assert!(reduced.points.len() < whole.points.len());
    let mut metal_opts = tight();
    metal_opts.width = 0.1 / HARTREE_EV;
    metal_opts.density_tolerance = 1e-9;
    metal_opts.tolerance = 1e-11;
    let rw = solve(&whole, &metal_opts);
    let rr = solve(&reduced, &metal_opts);
    let gap = (rw.free_energy - rr.free_energy).abs();
    let mag_gap = (rw.magnetization - rr.magnetization).abs();
    let mag_norm = rr.magnetization.abs();
    println!(
        "FM Ni PBE spinor: {} whole -> {} reduced pts, |dE|={gap:.2e} Ha, |M|={mag_norm:.4}, |dM|={mag_gap:.2e}",
        whole.points.len(),
        reduced.points.len(),
    );
    assert!(mag_norm > 0.1, "state must stay magnetic, got {mag_norm}");
    assert!(gap < 5e-6, "energy gap {gap:e}");
    assert!(mag_gap < 5e-4, "magnetization gap {mag_gap:e}");

    let sw = stress(&whole, &rw).expect("whole magnetic spinor stress");
    let sr = stress(&reduced, &rr).expect("reduced magnetic spinor stress");
    let stress_miss = (0..3)
        .flat_map(|i| (0..3).map(move |j| (sr.gradient[i][j] - sw.gradient[i][j]).abs()))
        .fold(0f64, f64::max);
    let stress_peak = sw
        .gradient
        .iter()
        .flatten()
        .fold(0f64, |m, x| m.max(x.abs()));
    println!(
        "FM Ni PBE spinor stress peak {stress_peak:.3e} Ha, diff vs whole: {stress_miss:.2e} Ha"
    );
    assert!(
        stress_peak > 1e-3,
        "stress must be non-trivial ({stress_peak:e})"
    );
    assert!(stress_miss < 5e-6, "stress miss {stress_miss:e}");

    // 2) Hybrid HSE06 magnetic spinor star reduction (displaced FeSi, 8 -> 1 pts via 4m'm'):
    let fesi_hse = displaced_fm_fesi_cell("hse06", 12, ([2, 2, 2], false));
    let hse_whole = build(&ctx, &fesi_hse.lattice, &fesi_hse, Points::Whole);
    let hse_red = build(
        &ctx,
        &fesi_hse.lattice,
        &fesi_hse,
        Points::Reduced { symmetry: true },
    );
    assert!(hse_red.points.len() < hse_whole.points.len());
    hse_red
        .image_census()
        .expect("magnetic spinor HSE06 crystal has its star images");
    let mut hse_opts = metal_opts.clone();
    hse_opts.density_tolerance = 1e-8;
    hse_opts.tolerance = 1e-10;
    let hrw = solve(&hse_whole, &hse_opts);
    let hrr = solve(&hse_red, &hse_opts);
    let hse_gap = (hrw.free_energy - hrr.free_energy).abs();
    let hse_mag_gap = (hrw.magnetization - hrr.magnetization).abs();
    println!(
        "FM FeSi HSE06 spinor: {} whole -> {} reduced pts, |dE|={hse_gap:.2e} Ha, |M|={:.4}, |dM|={hse_mag_gap:.2e}",
        hse_whole.points.len(),
        hse_red.points.len(),
        hrr.magnetization.abs(),
    );
    assert!(hrr.magnetization.abs() > 0.5, "state must stay magnetic");
    assert!(hse_gap < 1e-7, "HSE06 energy gap {hse_gap:e}");
    assert!(
        hse_mag_gap < 1e-5,
        "HSE06 magnetization gap {hse_mag_gap:e}"
    );
    let hse_fw = forces(&hse_whole, &hrw).expect("whole magnetic spinor HSE06 forces");
    let hse_fr = forces(&hse_red, &hrr).expect("reduced magnetic spinor HSE06 forces");
    let hse_fmiss = worst(&hse_fw, &hse_fr);
    let hse_fpeak = hse_fw.iter().flatten().fold(0f64, |m, x| m.max(x.abs()));
    println!(
        "FM FeSi HSE06 spinor peak force {hse_fpeak:.3e}, diff vs whole: {hse_fmiss:.2e} Ha/Bohr"
    );
    assert!(
        hse_fpeak > 1e-3,
        "HSE06 force must be non-trivial ({hse_fpeak:e})"
    );
    assert!(hse_fmiss < 1e-6, "HSE06 force miss {hse_fmiss:e}");
}

#[test]
fn magnetic_spinor_reduced_mesh_forces_match_whole_mesh() {
    let ctx = context();
    let cell = displaced_fm_fesi_cell("pbe", 14, ([2, 2, 2], true));
    let whole = build(&ctx, &cell.lattice, &cell, Points::Whole);
    let reduced = build(
        &ctx,
        &cell.lattice,
        &cell,
        Points::Reduced { symmetry: true },
    );
    assert!(!reduced.time_reversal && reduced.spinor);
    assert!(reduced.points.len() < whole.points.len());
    let mut opts = tight();
    opts.width = 0.1 / HARTREE_EV;
    opts.density_tolerance = 1e-8;
    opts.tolerance = 1e-10;
    let rw = solve(&whole, &opts);
    let rr = solve(&reduced, &opts);
    assert!(rr.magnetization.abs() > 0.5, "FeSi must stay magnetic");
    let fw = forces(&whole, &rw).expect("whole forces");
    let fr = forces(&reduced, &rr).expect("reduced magnetic spinor forces");
    let miss = worst(&fw, &fr);
    let peak = fw.iter().flatten().fold(0f64, |m, x| m.max(x.abs()));
    println!(
        "displaced FM FeSi spinor: {} -> {} pts, |M|={:.4}, |dE|={:.2e}, peak force {peak:.3e}, force diff {miss:.2e}",
        whole.points.len(),
        reduced.points.len(),
        rr.magnetization.abs(),
        (rw.free_energy - rr.free_energy).abs(),
    );
    assert!(peak > 1e-3, "force must be non-trivial ({peak:e})");
    assert!(miss < 1e-5, "force miss {miss:e}");
}
