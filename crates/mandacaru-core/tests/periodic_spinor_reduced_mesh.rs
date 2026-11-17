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
    response::{forces, forces_with_group},
};
use nalgebra::Vector3;
use periodic_reduced_mesh_support::{build, context, silicon, solve, tight, Points};

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
