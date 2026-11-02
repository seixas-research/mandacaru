//! The explicit terms of the hybrid crystal force on a DISPLACED crystal (HX-1): the basis
//! functions of the grid pair, the compensation phase, and the projection response chained
//! through the ownership identity of CF-2 (`dC_X = P_X G_C - G_C Q_X`), against central
//! differences of the production energy `Crystal::hybrid_exchange` of crystals rebuilt at the
//! moved geometry (`periodic_hybrid_gradient_motion_support`), and the negative controls of the
//! chain. Si, 2 atoms, SZ, Gamma, 10^3 grid, atom 1 off its site. The occupied coefficients are
//! held fixed (the fixed-state derivative). Needs `MANDACARU_PAW_PATH`.
mod periodic_hybrid_gradient_motion_support;
mod periodic_hybrid_gradient_support;
mod periodic_screened_augmentation_support;
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;
mod setup_support;

use mandacaru_core::periodic::{
    hybrid_gradient::{explicit, projection_forces, Gradient},
    screened_exchange::Screening,
    setup::pulay::projection_gradient,
};
use periodic_hybrid_gradient_motion_support::*;
use periodic_hybrid_gradient_support::*;
use periodic_screened_augmentation_support::GB;
use std::sync::OnceLock;

/// The gate of the chain: about 100 x the measured step floor (1.4e-11 at 2e-3 Bohr; the worst
/// deviation, 1.8e-11, sits at that floor); forces are 5e-4 .. 1.7e-3. A control must miss it by
/// more than 100 x the gate.
const GATE: f64 = 1.5e-9;
const STEP: f64 = 2e-3;

/// Everything the tests compare, computed once per test binary.
struct Reference {
    /// Central differences `dE/dR` per atom and axis.
    slope: Vec<[f64; 3]>,
    /// The worst over the six entries of `|R(STEP) - R(STEP/2)|`: the measured step floor of
    /// every comparison.
    floor: f64,
    gradient: Gradient,
    projection: Vec<[f64; 3]>,
    /// The projection chain of the compensation response alone, and with the owners swapped.
    compensation: Vec<[f64; 3]>,
    swapped: Vec<[f64; 3]>,
    rows: Vec<[f64; 3]>,
    columns: Vec<[f64; 3]>,
}

fn reference() -> &'static Reference {
    static REFERENCE: OnceLock<Reference> = OnceLock::new();
    REFERENCE.get_or_init(|| {
        let si = Si::new();
        let base = si.centers(true, 0, 0, 0.);
        let crystal = si.crystal(&base);
        let shells = si.shells(&base);
        let mesh = si.fx.mesh(&si.ctx);
        let projectors = si.projectors(&base);
        let motion = ShellMotion {
            ctx: &si.ctx,
            fx: &si.fx,
            shells: &shells,
            mesh: &mesh,
        };
        let aug = crystal
            .hybrid
            .as_ref()
            .unwrap()
            .augmentation(&crystal)
            .unwrap();
        let screening = Screening::of(&crystal).unwrap();
        let gradient = explicit(&crystal, &si.rig.d, screening, GB, &aug, Some(&motion)).unwrap();
        let gc =
            projection_gradient(&si.ctx, &shells, &si.fx.lattice, [0.; 3], &projectors).unwrap();
        let owners: Vec<usize> = projectors.iter().map(|p| p.atom).collect();
        let chain = |z: &[_], rows: &[usize], columns: &[usize]| {
            projection_forces(z, std::slice::from_ref(&gc), rows, columns, 2).unwrap()
        };
        let all = chain(&gradient.response(), &si.fx.owner, &owners);
        let swap = |v: &[usize]| v.iter().map(|a| 1 - a).collect::<Vec<_>>();
        let swapped = chain(&gradient.response(), &swap(&si.fx.owner), &swap(&owners));
        let compensation = chain(&gradient.compensation, &si.fx.owner, &owners);
        let mut slope = Vec::new();
        let mut floor = 0f64;
        for atom in 0..2 {
            let mut row = [0.; 3];
            for (axis, entry) in row.iter_mut().enumerate() {
                let mut energy = |d: f64| {
                    let moved = si.centers(true, atom, axis, d);
                    si.crystal(&moved)
                        .hybrid_exchange(&si.rig.d, GB)
                        .unwrap()
                        .energy
                };
                let (coarse, gap) = central(STEP, &mut energy);
                *entry = coarse;
                floor = floor.max(gap);
            }
            slope.push(row);
        }
        Reference {
            slope,
            floor,
            projection: all.total(),
            compensation: compensation.total(),
            swapped: swapped.total(),
            rows: all.rows,
            columns: all.columns,
            gradient,
        }
    })
}

/// The worst `|slope - variant|` over atoms and axes.
fn miss(r: &Reference, variant: &dyn Fn(usize, usize) -> f64) -> f64 {
    (0..2)
        .flat_map(|a| (0..3).map(move |x| (a, x)))
        .map(|(a, x)| (r.slope[a][x] - variant(a, x)).abs())
        .fold(0., f64::max)
}

fn terms(r: &Reference) -> (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let f = r.gradient.functions.as_ref().unwrap();
    (f.bra.clone(), f.ket.clone(), r.gradient.phase.clone())
}

#[test]
fn the_rust_setup_reproduces_the_snapshots_samples_and_projections() {
    let si = Si::new();
    let (psi, c) = si.arrays(&si.ideal);
    let crystal = &si.rig.crystal;
    let point = &crystal.points[0];
    let stored = point.psi.read(&crystal.context).unwrap();
    let gap = psi
        .iter()
        .zip(stored.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0., f64::max);
    let c_gap = (&c - &point.projections)
        .iter()
        .map(|z| z.norm())
        .fold(0., f64::max);
    let rebuilt = si.crystal(&si.ideal);
    let e0 = full_energy(&si.rig, GB);
    let e1 = rebuilt.hybrid_exchange(&si.rig.d, GB).unwrap().energy;
    println!(
        "samples {gap:.2e}, projections {c_gap:.2e}, energy {e0:.12e} rebuilt {e1:.12e} ({:.1e})",
        (e0 - e1).abs()
    );
    assert!(gap < 1e-10 && c_gap < 1e-10, "{gap:e} {c_gap:e}");
    assert!((e0 - e1).abs() < 1e-9, "{:e}", (e0 - e1).abs());
}

#[test]
fn the_explicit_terms_and_the_chain_equal_the_energy_slope_of_the_displaced_crystal() {
    let r = reference();
    let (bra, ket, phase) = terms(r);
    let total = |a: usize, x: usize| phase[a][x] + bra[a][x] + ket[a][x] + r.projection[a][x];
    let worst = miss(r, &total);
    println!("worst deviation {worst:.2e}, step floor {:.2e}", r.floor);
    for a in 0..2 {
        for x in 0..3 {
            println!(
                "atom {a} axis {x}: fd {:+.9e} | phase {:+.3e} bra {:+.3e} ket {:+.3e} rows {:+.3e} columns {:+.3e}",
                r.slope[a][x], phase[a][x], bra[a][x], ket[a][x], r.rows[a][x], r.columns[a][x]
            );
        }
    }
    assert!(worst < GATE, "{worst:e}");
    assert!(r.floor < GATE / 10., "the step floor {:e}", r.floor);
    // Non-vacuous: the sphere response is part of the chain and every term is large.
    for (name, values) in [
        ("phase", &phase),
        ("bra", &bra),
        ("ket", &ket),
        ("rows", &r.rows),
        ("columns", &r.columns),
    ] {
        let peak = values.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
        assert!(peak > 1e3 * GATE, "{name}: {peak:e}");
    }
}

#[test]
fn a_wrong_chain_misses_the_energy_slope_by_far_more_than_the_gate() {
    let r = reference();
    let (bra, ket, phase) = terms(r);
    let right = |a: usize, x: usize| phase[a][x] + bra[a][x] + ket[a][x] + r.projection[a][x];
    assert!(miss(r, &right) < GATE);
    let variants: Vec<(&str, Box<dyn Fn(usize, usize) -> f64 + '_>)> = vec![
        ("phase dropped", Box::new(|a, x| right(a, x) - phase[a][x])),
        (
            "phase sign",
            Box::new(|a, x| right(a, x) - 2. * phase[a][x]),
        ),
        (
            "phase owners swapped",
            Box::new(|a, x| right(a, x) - phase[a][x] + phase[1 - a][x]),
        ),
        ("bra dropped", Box::new(|a, x| right(a, x) - bra[a][x])),
        ("ket dropped", Box::new(|a, x| right(a, x) - ket[a][x])),
        (
            "function sign",
            Box::new(|a, x| right(a, x) - 2. * (bra[a][x] + ket[a][x])),
        ),
        (
            "function owners swapped",
            Box::new(|a, x| right(a, x) - (bra[a][x] + ket[a][x]) + bra[1 - a][x] + ket[1 - a][x]),
        ),
        (
            "projector-moving part dropped",
            Box::new(|a, x| right(a, x) - r.columns[a][x]),
        ),
        (
            "basis-moving part dropped",
            Box::new(|a, x| right(a, x) - r.rows[a][x]),
        ),
        (
            "projection chain owners swapped",
            Box::new(|a, x| right(a, x) - r.projection[a][x] + r.swapped[a][x]),
        ),
        (
            "sphere response dropped",
            Box::new(|a, x| right(a, x) - r.projection[a][x] + r.compensation[a][x]),
        ),
        (
            "projection chain dropped",
            Box::new(|a, x| right(a, x) - r.projection[a][x]),
        ),
    ];
    for (name, variant) in variants {
        let gap = miss(r, &*variant);
        println!("{name}: misses by {gap:.2e}");
        assert!(gap > 100. * GATE, "{name}: misses by {gap:e}");
    }
}
