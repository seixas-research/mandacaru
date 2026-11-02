//! The periodic setup's symmetry against checks that need no frozen number:
//! characters and closure of `T^L` up to L = 6 on a cubic and a hexagonal cell,
//! atom maps and node permutations from the positions, closure of the kept set,
//! the `(species, moment)` label, the grid rule, the `symprec` rule for a noisy
//! shift, the refused mesh, and the memory budget.
mod setup_symmetry_support;

use mandacaru_core::{
    periodic::{
        setup::symmetry::Symmetry,
        space_group::{find, Cell as SpaceCell},
    },
    Context, Error,
};
use num_complex::Complex64;
use setup_symmetry_support::*;
use std::collections::HashMap;

#[test]
fn harmonic_rotations_are_characters_and_a_representation_to_l_6() {
    for (name, mut cell) in [("Al fcc", aluminum()), ("Mg hcp", magnesium(6))] {
        cell.max_l = 6;
        let sym = cell.symmetry();
        let (mut character, mut unitary, mut closure) = (0f64, 0f64, 0f64);
        for k in 0..sym.len() {
            let r = cartesian(&cell, &sym, k);
            let det = r.determinant().signum();
            let proper = r * det;
            let theta = ((proper.trace() - 1.) / 2.).clamp(-1., 1.).acos();
            for l in 0..=6usize {
                let t = sym.harmonic(k, l);
                let chi = if theta < 1e-12 {
                    (2 * l + 1) as f64
                } else {
                    ((l as f64 + 0.5) * theta).sin() / (theta / 2.).sin()
                };
                character = character
                    .max((t.trace() - Complex64::new(det.powi(l as i32) * chi, 0.)).norm());
                let id = nalgebra::DMatrix::<Complex64>::identity(2 * l + 1, 2 * l + 1);
                unitary = unitary.max(
                    (t.adjoint() * t - id)
                        .iter()
                        .map(|v| v.norm())
                        .fold(0., f64::max),
                );
            }
        }
        let keys: HashMap<_, _> = (0..sym.len())
            .map(|k| (key(sym.rotation(k), [0.; 3]).0, k))
            .collect();
        for a in 0..sym.len() {
            for b in 0..sym.len() {
                let product = sym.rotation(a) * sym.rotation(b);
                if let Some(&c) = keys.get(&key(&product, [0.; 3]).0) {
                    // R(ab) = R(a) R(b): Y(R_a R_b u) = Y(R_b u) T_a... = Y(u) T_b T_a.
                    let d = (sym.harmonic(c, 6) - sym.harmonic(b, 6) * sym.harmonic(a, 6))
                        .iter()
                        .map(|v| v.norm())
                        .fold(0., f64::max);
                    closure = closure.max(d);
                }
            }
        }
        println!("{name}: {} operations, L <= 6: character {character:.1e}, unitarity {unitary:.1e}, closure {closure:.1e}", sym.len());
        assert!(character < 1e-12 && unitary < 1e-12 && closure < 1e-12);
    }
}

#[test]
fn atom_maps_and_permutations_follow_the_positions() {
    // Si on an 8^3 grid keeps all 48 operations; half of them swap the two atoms.
    for (name, cell) in [("Si 8^3", silicon(8)), ("Mg hcp 6x6x10", magnesium(6))] {
        let sym = cell.symmetry();
        let n = cell.fractional.len();
        let mut swapping = 0;
        let mut map = vec![0; n];
        let nodes: usize = cell.grid.iter().product();
        let mut perm = vec![0; nodes];
        for k in 0..sym.len() {
            sym.atom_map(k, &mut map);
            let (w, t) = (sym.rotation(k).map(f64::from), sym.translation(k));
            swapping += usize::from(map[0] != 0);
            for a in 0..n {
                let x =
                    w * nalgebra::Vector3::from(cell.fractional[a]) + nalgebra::Vector3::from(t);
                let d = x - nalgebra::Vector3::from(cell.fractional[map[a]]);
                assert!(
                    d.iter().all(|v| (v - v.round()).abs() < 1e-8),
                    "{name} op {k} atom {a}"
                );
            }
            sym.permutation(k, &mut perm);
            let size = cell.grid;
            for flat in (0..nodes).step_by(7) {
                let node = [
                    flat / (size[1] * size[2]),
                    flat / size[2] % size[1],
                    flat % size[2],
                ];
                let x = w * nalgebra::Vector3::new(
                    node[0] as f64 / size[0] as f64,
                    node[1] as f64 / size[1] as f64,
                    node[2] as f64 / size[2] as f64,
                ) + nalgebra::Vector3::from(t);
                let image: Vec<usize> = (0..3)
                    .map(|i| {
                        ((x[i] * size[i] as f64).round() as i64).rem_euclid(size[i] as i64) as usize
                    })
                    .collect();
                assert_eq!(
                    perm[flat],
                    (image[0] * size[1] + image[1]) * size[2] + image[2],
                    "{name} op {k}"
                );
            }
        }
        println!("{name}: {} operations, {swapping} swap atoms", sym.len());
        assert!(swapping > 0 && swapping < sym.len());
        closed_under_composition(&sym, nodes);
    }
}

#[test]
fn the_species_moment_label_halves_an_antiferromagnet() {
    // bcc Fe in its simple-cubic cell: 96 operations by species, 48 once the
    // two moments are told apart (the pure translation swaps them).
    let mut cell = Cell::new(
        [[2.87, 0., 0.], [0., 2.87, 0.], [0., 0., 2.87]],
        &[[0.; 3], [0.5; 3]],
        26,
        [8; 3],
    );
    let alike = cell.symmetry();
    assert_eq!((alike.n_space_group, alike.len()), (96, 96));
    cell.labels = vec![0, 1];
    let sym = cell.symmetry();
    println!(
        "AFM bcc Fe: {} of {} operations kept",
        sym.len(),
        sym.n_space_group
    );
    assert_eq!((sym.n_space_group, sym.len()), (96, 48));
    let mut map = vec![0; 2];
    for k in 0..sym.len() {
        sym.atom_map(k, &mut map);
        assert_eq!(map, [0, 1]);
    }
    // The finer label can also be handed to the search: same operations, same count.
    cell.species = vec![0, 1];
    assert_eq!(cell.symmetry().len(), 48);
}

#[test]
fn the_grid_decides_the_operations() {
    // A 7x7x8 grid does not carry the fcc rotations that mix the third axis
    // with the others: W_ij N_i / N_j is not an integer for them.
    let mut cell = aluminum();
    cell.grid = [7, 7, 8];
    let sym = cell.symmetry();
    println!("Al fcc on 7x7x8: {} of {}", sym.len(), sym.n_space_group);
    assert!(sym.len() < sym.n_space_group && sym.len() > 1);
    closed_under_composition(&sym, 7 * 7 * 8);
    for k in 0..sym.len() {
        let w = sym.rotation(k);
        for i in 0..3 {
            for j in 0..3 {
                assert!((w[(i, j)] * cell.grid[i] as i32) % cell.grid[j] as i32 == 0);
            }
        }
    }
}

#[test]
fn a_noisy_translation_is_judged_by_symprec_not_by_one_millionth_of_a_node() {
    // The second Si atom is moved by 2e-6 of a lattice vector.  The glide-like
    // operations then have N t = 2 + 1.6e-5 nodes: the old 1e-6 test (grid units)
    // drops them, the symprec rule keeps them.
    let mut noisy = silicon(8);
    noisy.fractional[1] = [0.25 + 2e-6, 0.25, 0.25];
    noisy.symprec = 1e-4;
    let sym = noisy.symmetry();
    let old_rule = (0..sym.len()).filter(|&k| {
        let t = sym.translation(k);
        (0..3).all(|i| (t[i] * 8. - (t[i] * 8.).round()).abs() <= 1e-6)
    });
    let (new, old) = (sym.len(), old_rule.count());
    println!(
        "noisy Si 8^3 (2e-6 fractional): symprec rule keeps {new} operations, the 1e-6 rule {old}"
    );
    assert_eq!(new, 48);
    assert!(old < new);
    closed_under_composition(&sym, 512);
    // The same operations judged at a tighter symprec: the grid test refuses a
    // shift 1.6e-5 nodes off, it never rounds it away.
    let ops = find(
        &SpaceCell {
            lattice: noisy.lattice.transpose(),
            fractional: &noisy.fractional,
            labels: &noisy.species,
        },
        1e-4,
    )
    .unwrap();
    noisy.symprec = 1e-7;
    let tight = Symmetry::from_operations(ops, &noisy.geometry()).unwrap();
    println!("same operations at symprec 1e-7 Bohr: {} kept", tight.len());
    assert!(tight.len() < new);
}

#[test]
fn a_supercell_keeps_every_pure_translation_and_reduces_through_48_rotations() {
    // Al in a 3x3x3 conventional cell: 108 atoms, 27 x 4 pure translations.
    let mut atoms = vec![];
    for (i, j, k) in itertools_product(3) {
        for b in [[0., 0., 0.], [0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]] {
            atoms.push([
                (i as f64 + b[0]) / 3.,
                (j as f64 + b[1]) / 3.,
                (k as f64 + b[2]) / 3.,
            ]);
        }
    }
    let a = 3. * 4.05;
    let cell = Cell::new([[a, 0., 0.], [0., a, 0.], [0., 0., a]], &atoms, 13, [24; 3]);
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let start = std::time::Instant::now();
    let sym = cell.symmetry().keeping_mesh([2, 2, 2], true).unwrap();
    assert_eq!((sym.n_space_group, sym.len()), (5184, 5184));
    assert_eq!(sym.distinct_rotations().len(), 48);
    let wedge = sym.mesh(&ctx, [2, 2, 2], true).unwrap();
    // Simple cubic: Gamma, X, M and R of the 2x2x2 mesh.
    assert_eq!(wedge.points.len(), 4);
    let channels: Vec<_> = (0..108).map(|a| (a, 0, 0)).collect();
    let q = sym.qsym(&ctx, &channels).unwrap();
    // Every atom is equivalent: each row of the average is 1/108 on all columns.
    assert!(q
        .as_slice()
        .iter()
        .all(|v| (v.re - 1. / 108.).abs() < 1e-12 && v.im.abs() < 1e-12));
    println!(
        "Al 108 atoms, 5184 operations: setup, wedge and qsym in {:.2} s, context peak {:.1} MiB",
        start.elapsed().as_secs_f64(),
        ctx.memory_peak() as f64 / (1 << 20) as f64
    );
}
fn itertools_product(n: usize) -> Vec<(usize, usize, usize)> {
    (0..n)
        .flat_map(|i| (0..n).flat_map(move |j| (0..n).map(move |k| (i, j, k))))
        .collect()
}

#[test]
fn a_mesh_the_rotations_break_is_refused_until_filtered() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let sym = silicon(10).symmetry();
    assert!(matches!(
        sym.mesh(&ctx, [2, 1, 1], true),
        Err(Error::Invalid(_))
    ));
    let sym = sym.keeping_mesh([2, 1, 1], true).unwrap();
    assert_eq!(sym.len(), 6);
    assert!(sym.mesh(&ctx, [2, 1, 1], true).is_ok());
    assert_eq!(sym.distinct_rotations().len(), 6);
}

#[test]
fn the_symmetry_tables_obey_the_memory_budget_and_refuse_bad_input() {
    let sym = silicon(10).symmetry();
    let channels: Vec<(usize, usize, i32)> = (0..2)
        .flat_map(|a| {
            (0..=2usize).flat_map(move |l| (-(l as i32)..=l as i32).map(move |m| (a, l, m)))
        })
        .collect();
    let small = Context::new(Some(1), 1000).unwrap();
    assert!(matches!(
        sym.qsym(&small, &channels),
        Err(Error::Budget { .. })
    ));
    assert_eq!(small.memory_used(), 0);
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    assert!(
        sym.qsym(&ctx, &[(0, 5, 0)]).is_err(),
        "L above the tables raises"
    );
    assert!(
        sym.qsym(&ctx, &[(0, 1, 0), (0, 0, 0)]).is_err(),
        "unsorted channels raise"
    );
    assert_eq!(ctx.memory_used(), 0);
    let mut bad = silicon(10);
    bad.labels.pop();
    assert!(Symmetry::new(&bad.geometry()).is_err());
}
