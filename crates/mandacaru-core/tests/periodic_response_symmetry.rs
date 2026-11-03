//! CF-6: the force and the stress on a k mesh reduced by the cell's space group equal those of
//! the whole mesh. The pairs of snapshots are written by the setup (`freeze_periodic_gradients.py
//! symmetry`): the same crystal solved on the whole mesh and on the irreducible one, each carrying
//! its `setup` block (the reduced one lists the kept operations). Controls: the unaveraged
//! reduced-mesh k-sum (the symmetrizer is not vacuous), a wrong atom map, and a snapshot that
//! lists no operations (raises). Needs `MANDACARU_PAW_PATH`.
//!
//! The five tests that run the engines of a whole and a reduced mesh and took 159 to 312 s each (alone, three threads, at a
//! machine load of 15-20; 592 s for the file two at a time) are `#[ignore]`d with their time. The Si [110] force test
//! (107 s) and the three refusal and NaN tests run by default. Si [111] force (159 s, 13 percent under the 180 s budget at
//! that load) is the one borderline case, ignored because it carries the controls (no average, wrong atom map). All of
//! them: `MANDACARU_PAW_PATH=... cargo test --release -p mandacaru-core --test periodic_response_symmetry -- --include-ignored --test-threads=1`.
mod periodic_gradient_crystals_support;
mod periodic_response_support;

use mandacaru_core::periodic::{
    point_group::PointGroup,
    response::{forces, forces_with_group, stress, stress_with_group},
};
use mandacaru_core::units::{BOHR_TO_ANGSTROM, HARTREE_TO_EV};
use periodic_gradient_crystals_support::gates::PLAN_GATE;
use periodic_response_support::{solve_at, solve_snapshot, worst, Pair};

/// The plan's stress gate on `g_ab` (Hartree), 3e-6 eV/A^3 of the diagonal and 5e-8 of a shear.
fn stress_gate(a: usize, b: usize, volume: f64) -> f64 {
    let sigma = if a == b { 3e-6 } else { 5e-8 };
    sigma * BOHR_TO_ANGSTROM.powi(3) / HARTREE_TO_EV * volume
}

fn worst_force(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    worst(
        a.iter()
            .zip(b)
            .flat_map(|(x, y)| (0..3).map(move |i| (x[i] - y[i]).abs())),
    )
}

/// The worst entry of `g(reduced) - g(full)` in units of its gate.
fn worst_stress(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3], volume: f64) -> f64 {
    worst(
        (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| (a[i][j] - b[i][j]).abs() / stress_gate(i, j, volume)),
    )
}

type Solved = (
    mandacaru_core::periodic::Crystal,
    mandacaru_core::periodic::PeriodicResult,
);

/// The whole-mesh and the reduced crystal of `case`, solved to the stress tests' reference
/// tolerances (a force or a stress is first order in the SCF residual).
fn solved(case: &str) -> (Solved, Solved) {
    let pair = Pair::load(case);
    let tight = Some((1e-13, 1e-11));
    let full = solve_at(&pair.full, &pair.manifest, 3, tight);
    let reduced = solve_at(&pair.reduced, &pair.manifest, 3, tight);
    let (nf, nr) = (full.0.points.len(), reduced.0.points.len());
    let group = reduced.0.description.as_ref().unwrap().point_group.as_ref();
    let operations = group.unwrap().len();
    eprintln!("{case}: {nf} k-points whole, {nr} reduced by {operations} operations");
    assert!(nr < nf && operations >= 2, "{case}: nothing was reduced");
    assert!(full.0.description.as_ref().unwrap().point_group.is_none());
    (full, reduced)
}

/// The force of the reduced mesh against the whole mesh's and, with `controls`, the two ways to
/// get it wrong: no average, and every atom sent to the next one.
fn check_forces(case: &str, controls: bool) {
    let ((cf, rf), (cr, rr)) = solved(case);
    let whole = forces(&cf, &rf).unwrap();
    let reduced = forces(&cr, &rr).unwrap();
    let miss = worst_force(&whole, &reduced);
    let scale = worst(whole.iter().flatten().map(|x| x.abs()));
    eprintln!(
        "{case}: forces whole {whole:?}\n{case}: forces reduced {reduced:?}\n{case}: miss {miss:.2e} Ha/Bohr (gate {PLAN_GATE:.2e}), largest force {scale:.2e}"
    );
    assert!(
        miss < PLAN_GATE,
        "{case}: reduced-mesh force misses by {miss:e}"
    );
    assert!(scale > 1e-3, "{case}: the force must be a real signal");
    if controls {
        // The unaveraged reduced-mesh k-sum is a different, symmetry-broken number: the average is not vacuous.
        let raw = forces_with_group(&cr, &rr, None).unwrap();
        let broken = worst_force(&whole, &raw);
        eprintln!(
            "{case}: control, no average: miss {broken:.2e} = {:.0} gates",
            broken / PLAN_GATE
        );
        assert!(
            broken > 100. * PLAN_GATE,
            "{case}: no average must break the force"
        );
        // A wrong atom map: every operation sends each atom to the next one.
        let group = cr
            .description
            .as_ref()
            .unwrap()
            .point_group
            .as_ref()
            .unwrap();
        let shifted: Vec<Vec<usize>> = group
            .atom_maps()
            .iter()
            .map(|m| m.iter().map(|&a| (a + 1) % m.len()).collect())
            .collect();
        let wrong = PointGroup::from_parts(group.rotations().to_vec(), shifted);
        let bad = forces_with_group(&cr, &rr, Some(wrong)).unwrap();
        let broken = worst_force(&whole, &bad);
        eprintln!(
            "{case}: control, wrong atom map: miss {broken:.2e} = {:.0} gates",
            broken / PLAN_GATE
        );
        assert!(
            broken > 100. * PLAN_GATE,
            "{case}: a wrong atom map must break the force"
        );
    }
}

/// The stress of the reduced mesh against the whole mesh's, entry by entry at the plan's gates.
fn check_stress(case: &str) {
    let ((cf, rf), (cr, rr)) = solved(case);
    let (sf, sr) = (stress(&cf, &rf).unwrap(), stress(&cr, &rr).unwrap());
    let volume = sf.volume;
    assert!((volume - sr.volume).abs() < 1e-9 * volume);
    // Where the two differ, term by term (the breakdown of both is averaged over the group).
    for ((name, reduced), (_, whole)) in sr.breakdown.terms().iter().zip(sf.breakdown.terms()) {
        let miss = worst(
            (0..3)
                .flat_map(|a| (0..3).map(move |b| (a, b)))
                .map(|(a, b)| (reduced[a][b] - whole[a][b]).abs()),
        );
        let peak = worst(whole.iter().flatten().map(|x| x.abs()));
        eprintln!("{case}: term {name}: miss {miss:.2e} Ha of {peak:.2e}");
    }
    let bite = worst_stress(&sr.gradient, &sf.gradient, volume);
    eprintln!(
        "{case}: stress gradient whole {:?}\n{case}: stress gradient reduced {:?}\n{case}: worst miss {bite:.2} gates",
        sf.gradient, sr.gradient
    );
    assert!(
        bite < 1.,
        "{case}: reduced-mesh stress misses by {bite} gates"
    );
}

#[test]
#[ignore = "159 s alone, three threads, at load 15-20 (the pair of SCFs and the force or stress engines of both meshes); run: MANDACARU_PAW_PATH=... cargo test --release -p mandacaru-core --test periodic_response_symmetry -- --ignored --test-threads=1"]
fn si_displaced_along_111_force_on_the_reduced_mesh_equals_the_whole_mesh() {
    check_forces("si111", true);
}

#[test]
#[ignore = "229 s alone, three threads, at load 15-20 (the pair of SCFs and the force or stress engines of both meshes); run: MANDACARU_PAW_PATH=... cargo test --release -p mandacaru-core --test periodic_response_symmetry -- --ignored --test-threads=1"]
fn si_displaced_along_111_stress_on_the_reduced_mesh_equals_the_whole_mesh() {
    check_stress("si111");
}

/// Measured 107 s alone (three threads, load 15-20): under the 180 s budget, so it runs by default.
#[test]
fn si_displaced_along_110_force_on_the_reduced_mesh_equals_the_whole_mesh() {
    check_forces("si110", false);
}

#[test]
#[ignore = "312 s alone, three threads, at load 15-20 (the pair of SCFs and the force or stress engines of both meshes); run: MANDACARU_PAW_PATH=... cargo test --release -p mandacaru-core --test periodic_response_symmetry -- --ignored --test-threads=1"]
fn si_displaced_along_110_stress_on_the_reduced_mesh_equals_the_whole_mesh() {
    check_stress("si110");
}

#[test]
#[ignore = "293 s alone, three threads, at load 15-20 (the pair of SCFs and the force or stress engines of both meshes); run: MANDACARU_PAW_PATH=... cargo test --release -p mandacaru-core --test periodic_response_symmetry -- --ignored --test-threads=1"]
fn cubic_aluminum_stress_on_the_reduced_mesh_equals_the_whole_mesh() {
    check_stress("al_prim");
}

/// The unaveraged reduced-mesh stress is a different number on a cell whose stress is not
/// already invariant: the average is not vacuous for the stress either.
#[test]
#[ignore = "229 s alone, three threads, at load 15-20 (an SCF pair and two stress engines); run: MANDACARU_PAW_PATH=... cargo test --release -p mandacaru-core --test periodic_response_symmetry -- --ignored --test-threads=1"]
fn the_stress_average_is_not_vacuous() {
    let ((cf, rf), (cr, rr)) = solved("si111");
    let whole = stress(&cf, &rf).unwrap();
    let raw = stress_with_group(&cr, &rr, None).unwrap();
    let broken = worst_stress(&raw.gradient, &whole.gradient, whole.volume);
    eprintln!("si111: control, no average: stress miss {broken:.1} gates");
    assert!(
        broken > 100.,
        "no average must break the stress by 100 gates, not {broken}"
    );
}

/// The cause a force raises with, for a reduced-mesh snapshot whose `setup` block `edit` changed.
fn refused_cause(edit: &dyn Fn(&mut serde_json::Value)) -> String {
    let mut pair = Pair::load("si111");
    edit(&mut pair.reduced);
    // The energy stands: the crystal loads and solves whatever its block says.
    let (crystal, result) = solve_snapshot(&pair.reduced, &pair.manifest);
    let free = pair.manifest["free_energy_ha"]["reduced"].as_f64().unwrap();
    assert!((result.free_energy - free).abs() < 1e-6);
    let message = forces(&crystal, &result).unwrap_err().to_string();
    assert!(stress(&crystal, &result).is_err());
    message
}

#[test]
fn a_reduced_mesh_without_its_operations_raises_with_the_cause() {
    // The block lists none although the snapshot symmetrizes with six: refused naming the operations.
    let listed_none = refused_cause(&|s| s["setup"]["operations"] = serde_json::Value::Null);
    assert!(
        listed_none.contains("space-group operations"),
        "{listed_none}"
    );
    // An old snapshot (no setup block at all).
    let old = refused_cause(&|s| {
        s.as_object_mut().unwrap().remove("setup");
    });
    assert!(old.contains("space-group symmetry"), "{old}");
}

#[test]
fn operations_that_do_not_reproduce_the_node_permutations_are_refused() {
    // A translation the grid cannot carry: the node permutation no longer matches.
    let message = refused_cause(&|s| {
        s["setup"]["operations"]["translations"][1] = serde_json::json!([0.5, 0.0, 0.0])
    });
    assert!(message.contains("operation 1"), "{message}");
}

/// The folds do not swallow a NaN: a miss that is not a number fails every gate that follows
/// (`f64::max` would have returned the other operand and passed).
#[test]
fn a_miss_that_is_not_a_number_is_the_worst_one() {
    let a = [[0.; 3], [0.; 3]];
    let mut b = a;
    b[1][2] = f64::NAN;
    assert!(worst_force(&a, &b).is_nan());
    assert!(!(worst_force(&a, &b) < PLAN_GATE));
    let g = [[0.; 3]; 3];
    let mut h = g;
    h[2][0] = f64::NAN;
    assert!(worst_stress(&g, &h, 100.).is_nan());
    assert!(!(worst_stress(&g, &h, 100.) < 1.));
    // The wrong variant (`f64::max`) passes the same input: the control bites.
    let swallowed = [0.0f64, f64::NAN, 0.0]
        .iter()
        .fold(0., |m: f64, x| m.max(*x));
    assert!(swallowed < PLAN_GATE);
}
