//! Collinear magnetic crystals on a symmetry-reduced k mesh: the setup keeps the operations that
//! map the moment pattern onto itself (the moment-preserving subgroup of the magnetic group; an
//! operation that sends a moment to its opposite would need a spin flip of the channels and is
//! not kept), so the density of each spin channel is symmetrized channel by channel.
//!
//! The reference of every gate is the same crystal on the whole mesh, listed point by point with
//! `symmetry = false` (a magnetic crystal with symmetry off keeps the whole mesh): the total
//! energy, the total magnetization, and with +U the occupation matrices and the moment of each
//! shell (the trace of the up occupations minus the down). Forces and stress are in
//! `test/algorithms/test_magnetic_symmetry.py`.
//!
//! Gates as in `periodic_hubbard_symmetry.rs`: 1e-10 on the occupations and 1e-8 eV on the
//! energy where the discretization carries the operations (the hexagonal cell), the floor of the
//! projector-sphere quadrature above it for the cubic cells (fcc Ni, NiO).
mod periodic_symmetry_support;

use mandacaru_core::periodic::setup::{create_setup, parse_input_json};
use periodic_symmetry_support::*;
use serde_json::{json, Value};

/// `(energy eV, magnetization in mu_B, occupations and shell moments, density tolerance)`.
type Gates = (f64, f64, f64, f64);
const EXACT_MAGNETIC: Gates = (1e-8, 1e-8, 1e-10, 5e-10);
const CUBIC_MAGNETIC: Gates = (2e-5, 1e-6, 3e-6, 2e-8);
/// The type-II cell's plain-DFT floor is 3.15e-5 eV (`periodic_hubbard_symmetry.rs`).
const AFM_CELL: Gates = (1e-4, 1e-6, 3e-6, 2e-8);

fn with_moments(mut cell: Cell, moments: &[f64]) -> Cell {
    cell.moments = moments.to_vec();
    cell
}

/// Number of operations the setup keeps for `cell` (`None`: no group).
fn kept_operations(cell: &Cell, mesh: [usize; 3]) -> usize {
    let ctx = context();
    let mut raw = cell.request(Value::Null, false);
    raw["mesh"] = json!({"size": mesh, "gamma": true});
    raw["symmetry"] = json!(true);
    let setup = create_setup(
        &ctx,
        parse_input_json(&raw).unwrap(),
        datasets(&ctx, &cell.symbols),
    )
    .unwrap();
    setup.symmetry.as_ref().map_or(0, |s| s.len())
}

/// Reduced against whole; returns (k wedge, k whole).
fn compare_magnetic(
    cell: &Cell,
    hubbard: Value,
    mesh: [usize; 3],
    (energy_gate, moment_gate, occupation_gate, tolerance): Gates,
) -> (usize, usize) {
    let (reduced, whole) = cell.crystals(hubbard.clone(), false, mesh);
    assert_eq!(reduced.spins, 2);
    let (a, b) = (solve(&reduced, tolerance), solve(&whole, tolerance));
    let energy = (a.energy - b.energy).abs() * HARTREE_EV;
    let moment = (a.magnetization - b.magnetization).abs();
    let mut occupations = 0f64;
    let mut shell_moment = 0f64;
    let mut survives = 0f64;
    if let (Some(x), Some(y)) = (a.hubbard.as_ref(), b.hubbard.as_ref()) {
        occupations = occupation_error(&a, &b);
        let shells = x.shells;
        let trace = |r: &mandacaru_core::periodic::hubbard::Report, spin: usize, shell: usize| {
            let n = r.occupation(spin, shell);
            let p = (n.len() as f64).sqrt() as usize;
            (0..p).map(|i| n[i * p + i].re).sum::<f64>()
        };
        for shell in 0..shells {
            let up_down = |r| trace(r, 0, shell) - trace(r, 1, shell);
            shell_moment = shell_moment.max((up_down(x) - up_down(y)).abs());
            survives = survives.max(up_down(y).abs());
        }
    }
    eprintln!(
        "{} k-points against {}: |dE| = {energy:.2e} eV, |dM| = {moment:.2e}, |dn| = {occupations:.2e}, \
         |d shell moment| = {shell_moment:.2e}, M = {:.4}",
        reduced.points.len(),
        whole.points.len(),
        b.magnetization
    );
    assert!(
        b.magnetization.abs() > 1e-3 || cell.moments.iter().sum::<f64>().abs() < 1e-9,
        "the magnetic state must survive"
    );
    if cell.moments.iter().any(|m| *m != 0.) && b.hubbard.is_some() {
        assert!(survives > 0.1, "the shell moment must survive: {survives}");
    }
    assert!(energy < energy_gate, "energies differ by {energy:.2e} eV");
    assert!(
        moment < moment_gate,
        "magnetizations differ by {moment:.2e}"
    );
    assert!(
        occupations < occupation_gate,
        "occupations differ by {occupations:.2e}"
    );
    assert!(
        shell_moment < occupation_gate.max(moment_gate),
        "shell moments differ by {shell_moment:.2e}"
    );
    (reduced.points.len(), whole.points.len())
}

fn triangle(moments: &[f64]) -> Cell {
    with_moments(nickel_triangle(), moments)
}

#[test]
fn the_moment_pattern_selects_the_operations_and_a_flipped_site_gets_the_smaller_group() {
    // Three Ni around a threefold axis: ferromagnetic keeps the 12 operations of D3h; flipping one
    // site leaves only the mirror that fixes it and the plane of the atoms; the antiferromagnetic
    // pair-plus-one pattern is the same flipped pattern with the sign reversed.
    let ferro = kept_operations(&triangle(&[0.5, 0.5, 0.5]), [2, 2, 2]);
    let flipped = kept_operations(&triangle(&[0.5, 0.5, -0.5]), [2, 2, 2]);
    let none = kept_operations(&triangle(&[]), [2, 2, 2]);
    assert_eq!(
        (ferro, none),
        (12, 12),
        "a uniform moment keeps the group of the empty pattern"
    );
    assert_eq!(
        flipped, 4,
        "one flipped site keeps the mirror through it and the atomic plane"
    );
    // The control of the control: opposite moments on the swapped pair are the same group again.
    assert_eq!(
        kept_operations(&triangle(&[-0.5, -0.5, 0.5]), [2, 2, 2]),
        flipped
    );
}

#[test]
#[ignore = "slow: 10 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test periodic_magnetic_symmetry -- --ignored ferromagnetic_triangle_plain_dft"]
fn ferromagnetic_triangle_plain_dft_on_the_wedge_equals_the_whole_mesh() {
    let (wedge, whole) = compare_magnetic(
        &triangle(&[0.5, 0.5, 0.5]),
        Value::Null,
        [2, 2, 2],
        EXACT_MAGNETIC,
    );
    assert!(wedge < whole);
}

#[test]
#[ignore = "slow: 10 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test periodic_magnetic_symmetry -- --ignored the_flipped_site_triangle"]
fn the_flipped_site_triangle_with_its_smaller_group_equals_the_whole_mesh_with_u() {
    let (wedge, whole) = compare_magnetic(
        &triangle(&[0.5, 0.5, -0.5]),
        dudarev(4.),
        [2, 2, 2],
        EXACT_MAGNETIC,
    );
    assert!(wedge < whole);
}

#[test]
fn ferromagnetic_fcc_nickel_with_u_on_the_wedge_equals_the_whole_mesh() {
    let cell = with_moments(nickel_fcc(), &[0.6]);
    let (wedge, whole) = compare_magnetic(&cell, dudarev(4.), [3, 3, 3], CUBIC_MAGNETIC);
    assert!(wedge < whole);
}

#[test]
#[ignore = "slow: 15 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test periodic_magnetic_symmetry -- --ignored antiferromagnetic_nickel_oxide"]
fn antiferromagnetic_nickel_oxide_type_ii_with_u_on_the_wedge_equals_the_whole_mesh() {
    let cell = with_moments(nickel_oxide_antiferromagnetic_cell(), &[2., -2., 0., 0.]);
    // The moment-preserving group does not swap the two Ni atoms (the plain pattern does).
    let kept = kept_operations(&cell, [2, 2, 2]);
    let plain = kept_operations(&nickel_oxide_antiferromagnetic_cell(), [2, 2, 2]);
    assert!(
        (kept, plain) == (12, 24),
        "{kept} operations with the moments, {plain} without"
    );
    let (wedge, whole) = compare_magnetic(&cell, dudarev(6.), [2, 2, 2], AFM_CELL);
    assert!(wedge < whole);
}
