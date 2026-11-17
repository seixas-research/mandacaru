//! Q11/SH-5b, 5c and 5e: a collinear (also with two spin channels, one polarized by a hole) and a spin-orbit (spinor, `pbe-fr`) hybrid (HSE06) crystal on the k mesh the setup reduces by the space group
//! of its cell. The exact exchange sums over the occupied states of every mesh point, so the states
//! of the points a star lists no longer are rotated from the listed one (`periodic::star`: grid-node
//! permutation with the phase of the fractional translation, the PAW projections through the
//! complex-harmonic rotation of every operation and the atom map, time reversal where the star needs
//! it; a spinor state also turns its spin by the SU(2) matrix of the operation), and the projected
//! density of the one-center terms is the average over the star. Every gate runs for both.
//!
//! * fixed state: the whole mesh's density matrices handed to the reduced crystal give the same
//!   exchange energy and the same `K(k)` at every listed point (1e-12 Ha), with the energy split into
//!   the grid pair, the cross term and the compensation charges, and controls that bite (a wrong
//!   atom map, no rotation phase);
//! * SCF: Si HSE06 with the second atom moved along the cubic z axis (a 2 x 2 x 2 mesh whose
//!   points the operations send to other points): the free energy of the reduced mesh is the whole
//!   mesh's to 1e-10 Ha, the levels to 1e-9;
//! * forces: the reduced mesh's analytic force is the whole mesh's to 1e-8 Ha/Bohr and matches the
//!   central differences of the reconverged free energy of the whole mesh to 1e-6 (step 1e-3 Bohr,
//!   filter cutoff and ion width pinned by the undisplaced cell).
//! * stress (SH-5d, both): the reduced mesh's analytic strain derivative equals the whole mesh's to
//!   the plan's gates (3e-6 eV/A^3 of the diagonal, 5e-8 of a shear) and the central differences of the
//!   whole mesh's reconverged free energy under a strain of the cell (node counts, filter cutoff and
//!   ion width pinned) to 1e-6, with controls (no group average, wrong rotations). A wrong atom map
//!   is not a control of a stress: the average of a rank-2 tensor reads the rotations alone.
//!
//! Needs `MANDACARU_PAW_PATH`. Run: `MANDACARU_PAW_PATH=... cargo test --profile release-test -p
//! mandacaru-core --test periodic_hybrid_reduced_mesh -- --ignored --nocapture --test-threads=1`.
mod periodic_reduced_mesh_support;

use mandacaru_core::units::{BOHR_TO_ANGSTROM, HARTREE_TO_EV};
use mandacaru_core::{
    math::storage::MatrixSource,
    periodic::{
        point_group::PointGroup,
        response::{forces, forces_with_group, stress, stress_with_group},
        screened_exchange::ExxDivTreatment,
        solver::Options,
        Crystal,
    },
};
use nalgebra::{DMatrix, Matrix3, Vector3};
use num_complex::Complex64;
use periodic_reduced_mesh_support::{
    build, context, silicon, solve, tight, Cell, Points, HARTREE_EV,
};

type Matrix = DMatrix<Complex64>;

fn worst(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .fold(0f64, |m, (x, y)| m.max((x - y).abs()))
}

/// The index in `whole` of the listed point `k` of the reduced crystal (the same representative).
fn at(whole: &Crystal, fractional: [f64; 3]) -> usize {
    (0..whole.points.len())
        .find(|&j| (0..3).all(|i| (whole.points[j].fractional[i] - fractional[i]).abs() < 1e-12))
        .expect("a listed point is a point of the whole mesh")
}

/// The density matrices of the whole crystal's first few SCF iterations (a symmetric state), and
/// the reduced crystal's share of them (its listed points).
fn state(whole: &Crystal, reduced: &Crystal) -> (Vec<Matrix>, Vec<Matrix>) {
    let options = Options {
        width: 0.01 / HARTREE_EV,
        tolerance: 1e-6,
        density_tolerance: 1e-4,
        max_iterations: 4,
        ..Default::default()
    };
    let result = whole.solve(&options).unwrap();
    let ctx = &whole.context;
    let all: Vec<Matrix> = result
        .matrices
        .iter()
        .map(|m| (*m.matrix(ctx).unwrap()).clone())
        .collect();
    let listed = reduced
        .points
        .iter()
        .map(|p| all[at(whole, p.fractional)].clone())
        .collect();
    (all, listed)
}

/// One cell's fixed-state comparison: the exchange of the whole mesh and of the reduced one for the
/// same symmetric density matrices, with a control that bites.
fn fixed_state(label: &str, cell: Cell, needs_translation: bool) {
    let ctx = context();
    let whole = build(&ctx, &cell.lattice, &cell, Points::Whole);
    let reduced = build(
        &ctx,
        &cell.lattice,
        &cell,
        Points::Reduced { symmetry: true },
    );
    let operations = reduced
        .description
        .as_ref()
        .and_then(|d| d.point_group.as_ref())
        .map_or(0, |g| g.len());
    println!(
        "{label}: {} points whole, {} reduced by {operations} operations",
        whole.points.len(),
        reduced.points.len()
    );
    assert!(operations >= 2 && reduced.points.len() < whole.points.len());
    let census = reduced.image_census().unwrap();
    println!("{label}: images {census:?}");
    if needs_translation {
        // A gate for the fractional translations must USE them: an image reached by an operation
        // with a translation whose grid phase is not 1.
        assert!(
            census.nontrivial_phase > 0,
            "{label}: no image uses a translation phase"
        );
    }
    let (all, listed) = state(&whole, &reduced);
    let exxdiv = ExxDivTreatment::None;
    let reference = whole.hybrid_exchange(&all, exxdiv).unwrap();
    let rotated = reduced.hybrid_exchange(&listed, exxdiv).unwrap();
    let gap = (reference.energy - rotated.energy).abs();
    let mut worst_operator = 0f64;
    for (k, point) in reduced.points.iter().enumerate() {
        let (a, b) = (
            &reference.operators[at(&whole, point.fractional)],
            &rotated.operators[k],
        );
        worst_operator = (a - b).iter().fold(worst_operator, |m, z| m.max(z.norm()));
    }
    println!(
        "{label}: exchange energy whole {:.14} Ha, reduced {:.14} Ha, difference {gap:.3e}; K(k) at the listed points: {worst_operator:.3e}",
        reference.energy, rotated.energy
    );
    assert!(gap < 1e-11, "{label}: energy {gap:e}");
    assert!(worst_operator < 1e-10, "{label}: K {worst_operator:e}");

    // Controls. (1) A wrong atom map is refused: an operation must send every atom onto its image.
    let mut wrong = build(
        &ctx,
        &cell.lattice,
        &cell,
        Points::Reduced { symmetry: true },
    );
    for map in &mut operations_of(&mut wrong).atom_maps {
        let count = map.len();
        for atom in map.iter_mut() {
            *atom = (*atom + 1) % count;
        }
    }
    let cause = wrong
        .hybrid_exchange(&listed, exxdiv)
        .err()
        .map(|e| e.to_string());
    println!("{label}: control, wrong atom map: {cause:?}");
    assert!(
        cause.is_some_and(|c| c.contains("atom map")),
        "{label}: a wrong atom map must be refused"
    );
    // (2) The operations of the setup block no longer in the order of the node permutations are
    // refused, naming the operation (before, the grid was rotated by one operation and the
    // projections by another, and the exchange missed by 1e-1 without an error).
    let mut shuffled = build(
        &ctx,
        &cell.lattice,
        &cell,
        Points::Reduced { symmetry: true },
    );
    let operations = operations_of(&mut shuffled);
    operations.rotations.reverse();
    operations.translations.reverse();
    operations.atom_maps.reverse();
    let cause = shuffled
        .hybrid_exchange(&listed, exxdiv)
        .err()
        .map(|e| e.to_string());
    println!("{label}: control, operations out of order with the node permutations: {cause:?}");
    assert!(
        cause.is_some_and(|c| c.contains("node permutation")),
        "{label}: operations out of order must be refused"
    );
    // (3) A translation shifted by a lattice vector is the same operation: the phase of the grid
    // and the phase of the projections move together.
    let mut shifted = build(
        &ctx,
        &cell.lattice,
        &cell,
        Points::Reduced { symmetry: true },
    );
    for t in &mut operations_of(&mut shifted).translations {
        t[0] += 1.;
        t[2] -= 1.;
    }
    let same = shifted.hybrid_exchange(&listed, exxdiv).unwrap();
    let invariance = same
        .operators
        .iter()
        .zip(&rotated.operators)
        .fold((same.energy - rotated.energy).abs(), |m, (a, b)| {
            (a - b).iter().fold(m, |m, z| m.max(z.norm()))
        });
    println!("{label}: control, translations shifted by a lattice vector: {invariance:.3e}");
    assert!(
        invariance < 1e-10,
        "{label}: the same operation gave {invariance:e}"
    );
}

fn operations_of(c: &mut Crystal) -> &mut mandacaru_core::periodic::description::Operations {
    c.description
        .as_mut()
        .and_then(|d| d.operations.as_mut())
        .expect("the operations of the setup block")
}

#[test]
#[ignore = "slow: 72 s (measured 2026-11-16); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_hybrid_reduced_mesh -- --ignored the_exchange_of_a_reduced_mesh"]
fn the_exchange_of_a_reduced_mesh_is_the_whole_meshs_for_the_same_state() {
    fixed_state(
        "Si displaced (four operations)",
        silicon("pbe-sr", "hse06", false, 16, 0.08),
        false,
    );
    fixed_state(
        "Si ideal (every operation)",
        silicon("pbe-sr", "hse06", false, 16, 0.0),
        false,
    );
    fixed_state(
        "Si opposite displacements (fractional translations used)",
        opposite("pbe-sr", false),
        true,
    );
}

/// Silicon with its two atoms moved 0.08 Bohr in opposite directions along z: the inversion about
/// the bond centre (a fractional translation) stays, and on the 2 x 2 x 2 Gamma mesh three of the
/// eight images of the stars are reached only through it.
fn opposite(library: &'static str, spinor: bool) -> Cell {
    let mut cell = silicon(library, "hse06", spinor, 16, 0.);
    cell.move_atom(0, Vector3::new(0., 0., 0.08));
    cell.move_atom(1, Vector3::new(0., 0., -0.08));
    cell
}

#[test]
#[ignore = "slow: 111 s (measured 2026-11-16); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_hybrid_reduced_mesh -- --ignored the_spinor_exchange"]
fn the_spinor_exchange_of_a_reduced_mesh_is_the_whole_meshs_for_the_same_state() {
    fixed_state(
        "Si pbe-fr displaced (four operations)",
        silicon("pbe-fr", "hse06", true, 16, 0.08),
        false,
    );
    fixed_state(
        "Si pbe-fr ideal (every operation)",
        silicon("pbe-fr", "hse06", true, 16, 0.0),
        false,
    );
    fixed_state(
        "Si pbe-fr opposite displacements (fractional translations used)",
        opposite("pbe-fr", true),
        true,
    );
}

#[test]
#[ignore = "slow: 286 s (measured 2026-11-16); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_hybrid_reduced_mesh -- --ignored a_reduced_hybrid_mesh_equals"]
fn a_reduced_hybrid_mesh_equals_the_whole_mesh_in_energy_and_force_and_the_finite_differences() {
    gate("pbe-sr", false);
}

#[test]
#[ignore = "slow: 604 s (measured 2026-11-16); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_hybrid_reduced_mesh -- --ignored a_reduced_spinor_hybrid_mesh"]
fn a_reduced_spinor_hybrid_mesh_equals_the_whole_mesh_in_energy_and_force_and_the_finite_differences(
) {
    gate("pbe-fr", true);
}

fn gate(library: &'static str, spinor: bool) {
    run_gate(silicon(library, "hse06", spinor, 16, 0.08), true, 0., false);
}

#[test]
#[ignore = "slow: 870 s (measured 2026-11-16); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_hybrid_reduced_mesh -- --ignored a_reduced_two_channel"]
fn a_reduced_two_channel_hybrid_mesh_equals_the_whole_mesh_in_energy_force_and_stress() {
    let mut cell = silicon("pbe-sr", "hse06", false, 16, 0.08);
    cell.spin = true;
    run_gate(cell, true, 0., false);
}

#[test]
#[ignore = "slow: 883 s (measured 2026-11-16); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_hybrid_reduced_mesh -- --ignored a_reduced_polarized"]
fn a_reduced_polarized_hybrid_mesh_equals_the_whole_mesh_in_energy_and_force() {
    // A hole in the valence band of Si (one electron less per cell, with the compensating
    // background) with a seeded moment on each atom: two channels that differ.
    // (The hole softens the bonds: the atom is moved further for the force to be a signal.)
    let mut cell = silicon("pbe-sr", "hse06", false, 16, 0.4);
    cell.spin = true;
    cell.moments = Some(vec![0.5, 0.5]);
    cell.charge = Some(1.);
    run_gate(cell, true, 0.1, false);
}

#[test]
#[ignore = "slow: 571 s (measured 2026-11-16); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_hybrid_reduced_mesh -- --ignored a_reduced_hybrid_mesh_whose_stars"]
fn a_reduced_hybrid_mesh_whose_stars_need_fractional_translations_equals_the_whole_mesh() {
    run_gate(opposite("pbe-sr", false), true, 0., true);
}

fn run_gate(cell: Cell, with_stress: bool, least_moment: f64, needs_translation: bool) {
    let ctx = context();
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
        "Si HSE06: {} points whole, {} reduced by {} operations",
        whole.points.len(),
        reduced.points.len(),
        group.len()
    );
    assert!(group.len() >= 2 && reduced.points.len() < whole.points.len());
    let census = reduced.image_census().unwrap();
    println!("images {census:?}");
    if needs_translation {
        assert!(
            census.nontrivial_phase > 0,
            "no image uses a translation phase"
        );
    }
    let (rw, rr) = (solve(&whole, &tight()), solve(&reduced, &tight()));
    let gap = (rw.free_energy - rr.free_energy).abs();
    println!(
        "magnetization: whole {:.6}, reduced {:.6}",
        rw.magnetization, rr.magnetization
    );
    assert!(
        rw.magnetization >= least_moment,
        "the state must be polarized ({})",
        rw.magnetization
    );
    let mut levels = 0f64;
    let channels = rr.eigenvalues.len() / reduced.points.len();
    for (k, point) in reduced.points.iter().enumerate() {
        let j = at(&whole, point.fractional);
        for s in 0..channels {
            let (kept, all) = (s * reduced.points.len() + k, s * whole.points.len() + j);
            for (x, y) in rr.eigenvalues[kept].iter().zip(&rw.eigenvalues[all]) {
                levels = levels.max((x - y).abs());
            }
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

    if !with_stress {
        return;
    }
    // The stress (SH-5d). The plan's gates on `g_ab` (Hartree): 3e-6 eV/A^3 of the diagonal, 5e-8 of
    // a shear.
    let (s_whole, s_reduced) = (stress(&whole, &rw).unwrap(), stress(&reduced, &rr).unwrap());
    let volume = s_whole.volume;
    let gate = |a: usize, b: usize| {
        (if a == b { 3e-6 } else { 5e-8 }) * BOHR_TO_ANGSTROM.powi(3) / HARTREE_TO_EV * volume
    };
    let ratio = |x: &[[f64; 3]; 3], y: &[[f64; 3]; 3]| {
        (0..9).fold((0f64, 0f64), |(r, m), i| {
            let (a, b) = (i / 3, i % 3);
            let d = (x[a][b] - y[a][b]).abs();
            (r.max(d / gate(a, b)), m.max(d))
        })
    };
    let (bite, miss) = ratio(&s_reduced.gradient, &s_whole.gradient);
    println!(
        "stress gradient whole {:?}\nstress gradient reduced {:?}\nreduced vs whole {miss:.3e} Ha = {bite:.3e} of the plan's gates",
        s_whole.gradient, s_reduced.gradient
    );
    assert!(bite < 1., "reduced stress vs whole: {bite:e} gates");
    let raw = stress_with_group(&reduced, &rr, None).unwrap();
    let (broken, miss_raw) = ratio(&raw.gradient, &s_whole.gradient);
    println!("control, no average: miss {miss_raw:.3e} Ha = {broken:.3e} gates");
    assert!(
        broken > 100.,
        "no average must break the stress ({broken:e} gates)"
    );
    // A wrong group: the operations conjugated by a quarter turn about x (the atoms and the tensor
    // average no longer agree with the mesh that was reduced).
    let quarter = Matrix3::new(1., 0., 0., 0., 0., -1., 0., 1., 0.);
    let turned: Vec<Matrix3<f64>> = group
        .rotations()
        .iter()
        .map(|r| quarter * r * quarter.transpose())
        .collect();
    let wrong = PointGroup::from_parts(turned, group.atom_maps().to_vec());
    let bad = stress_with_group(&reduced, &rr, Some(wrong)).unwrap();
    let (broken, miss_bad) = ratio(&bad.gradient, &s_whole.gradient);
    println!("control, wrong rotations: miss {miss_bad:.3e} Ha = {broken:.3e} gates");
    assert!(
        broken > 100.,
        "wrong rotations must break the stress ({broken:e} gates)"
    );

    // The reconverged central differences of the whole mesh under a strain of the cell (the
    // node counts, the filter cutoff and the ion width stay those of the unstrained cell).
    for (a, b) in [(0, 0), (2, 2), (0, 1)] {
        let free_energy = |sign: f64| {
            let mut e = Matrix3::<f64>::zeros();
            e[(a, b)] += 0.5 * sign * step;
            e[(b, a)] += 0.5 * sign * step;
            let mut strained = cell.clone();
            strained.lattice = (Matrix3::identity() + e) * cell.lattice;
            let c = build(&ctx, &cell.lattice, &strained, Points::Whole);
            solve(&c, &tight()).free_energy
        };
        let slope = (free_energy(1.) - free_energy(-1.)) / (2. * step);
        let analytic = 0.5 * (s_reduced.gradient[a][b] + s_reduced.gradient[b][a]);
        println!(
            "stress gradient ({a},{b}): analytic (reduced) {analytic:+.9e} Ha, whole-mesh central difference {slope:+.9e}, miss {:.2e}",
            (analytic - slope).abs()
        );
        assert!(
            (analytic - slope).abs() < 1e-6,
            "stress ({a},{b}): {analytic:e} vs {slope:e}"
        );
    }
}

/// The stars of a reduced mesh include the time-reversed image of every operation, so a magnetic
/// spin-orbit crystal (not time-reversal symmetric) is refused naming the cause.
#[test]
fn a_magnetic_spinor_crystal_has_no_star_images() {
    let ctx = context();
    let cell = silicon("pbe-fr", "hse06", true, 16, 0.08);
    let mut crystal = build(
        &ctx,
        &cell.lattice,
        &cell,
        Points::Reduced { symmetry: true },
    );
    assert!(
        crystal.image_census().is_ok(),
        "a nonmagnetic spinor crystal has its stars"
    );
    crystal.time_reversal = false;
    let cause = crystal.image_census().unwrap_err().to_string();
    assert!(
        cause.contains("not time-reversal symmetric") && cause.contains("symmetry=False"),
        "{cause}"
    );
}

/// Operation `s` must reproduce node permutation `s`: operations out of order, a rotation or a
/// translation the permutation does not carry are refused naming the operation.
#[test]
#[ignore = "slow: 13 s (measured 2026-11-16); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_hybrid_reduced_mesh -- --ignored operations_that_do_not_reproduce"]
fn operations_that_do_not_reproduce_the_node_permutations_are_refused() {
    let ctx = context();
    let cell = silicon("pbe-sr", "hse06", false, 16, 0.08);
    let base = build(
        &ctx,
        &cell.lattice,
        &cell,
        Points::Reduced { symmetry: true },
    );
    assert!(base.image_census().is_ok());
    let edits: [(
        &str,
        &dyn Fn(&mut mandacaru_core::periodic::description::Operations),
    ); 3] = [
        ("order", &|o| {
            o.rotations.reverse();
            o.translations.reverse();
            o.atom_maps.reverse();
        }),
        ("translation", &|o| o.translations[1][0] += 0.25),
        ("rotation", &|o| o.rotations[1] = o.rotations[0]),
    ];
    for (what, edit) in edits {
        let mut crystal = build(
            &ctx,
            &cell.lattice,
            &cell,
            Points::Reduced { symmetry: true },
        );
        edit(operations_of(&mut crystal));
        let cause = crystal.image_census().unwrap_err().to_string();
        assert!(
            cause.contains("node permutation") && cause.contains("symmetry operation"),
            "{what}: {cause}"
        );
    }
}
