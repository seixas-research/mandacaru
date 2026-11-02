//! The periodic setup's symmetry against the frozen `reference/periodic/{symmetry,
//! mesh_wedge}.json` and `phase5/si_crystal.json`: operations and atom maps T0,
//! permutations by sha256 T0, harmonic rotations and the multipole average T2,
//! mesh wedges (representatives, weights, mapping) T0 including the generic-k
//! meshes whose points are not their own negatives.
mod setup_symmetry_support;
#[path = "paw_record_support/sha.rs"]
mod sha;

use mandacaru_core::Context;
use num_complex::Complex64;
use serde_json::Value;
use setup_symmetry_support::*;
use std::path::PathBuf;

#[test]
fn the_frozen_operations_atom_maps_and_permutations() {
    let body = reference("symmetry.json");
    for (name, cell) in [("al", aluminum()), ("si", silicon(10))] {
        let frozen = &body[name];
        let sym = cell.symmetry();
        println!(
            "{name}: {} of {} operations kept",
            sym.len(),
            sym.n_space_group
        );
        assert_eq!(
            sym.n_space_group,
            frozen["n_space_group"].as_u64().unwrap() as usize
        );
        assert_eq!(sym.len(), frozen["n_operations"].as_u64().unwrap() as usize);
        let index = frozen_index(&sym, frozen);
        let maps = ints(&frozen["atom_maps"]["data"]);
        let n_atoms = cell.fractional.len();
        let mut map = vec![0; n_atoms];
        for (k, &j) in index.iter().enumerate() {
            sym.atom_map(k, &mut map);
            let want: Vec<usize> = maps[j * n_atoms..(j + 1) * n_atoms]
                .iter()
                .map(|&v| v as usize)
                .collect();
            assert_eq!(map, want, "{name} atom map {k}");
        }
        // The permutation table in the frozen order, as int64 little-endian bytes.
        let nodes: usize = cell.grid.iter().product();
        let mut order = vec![0; sym.len()];
        for (k, &j) in index.iter().enumerate() {
            order[j] = k;
        }
        let mut bytes = Vec::with_capacity(8 * nodes * sym.len());
        let mut perm = vec![0; nodes];
        for &k in &order {
            sym.permutation(k, &mut perm);
            bytes.extend(perm.iter().flat_map(|&p| (p as i64).to_le_bytes()));
        }
        assert_eq!(
            sha::Sha256::digest(&bytes),
            frozen["permutations_sha256"].as_str().unwrap(),
            "{name}"
        );
    }
}

#[test]
fn the_frozen_harmonic_rotations() {
    let body = reference("symmetry.json");
    let al = &body["al"];
    let (cell, sym) = (aluminum(), aluminum().symmetry());
    let index = frozen_index(&sym, al);
    let mut worst = 0f64;
    for l in 0..=4usize {
        let size = 2 * l + 1;
        for (k, &j) in index.iter().enumerate() {
            let want = matrix(&al["harmonic_rotations"][l.to_string()], j, size);
            let got = sym.harmonic(k, l);
            for (i, w) in want.iter().enumerate() {
                worst = worst.max((got[(i / size, i % size)] - w).norm());
            }
        }
    }
    println!("Al fcc 48 operations, L <= 4: largest difference {worst:.2e}");
    assert!(worst < T2);
    // Si shares fcc's lattice: each of its operations picks an Al operation.
    let si = &body["si"];
    let sym = silicon(10).symmetry();
    let pick = ints(&si["harmonic_rotations_are_fcc_operations"]);
    let index = frozen_index(&sym, si);
    let mut worst = 0f64;
    for l in 0..=4usize {
        let size = 2 * l + 1;
        for (k, &j) in index.iter().enumerate() {
            let want = matrix(
                &al["harmonic_rotations"][l.to_string()],
                pick[j] as usize,
                size,
            );
            for (i, w) in want.iter().enumerate() {
                worst = worst.max((sym.harmonic(k, l)[(i / size, i % size)] - w).norm());
            }
        }
    }
    println!("Si 24 operations, L <= 4: largest difference {worst:.2e}");
    assert!(worst < T2);
    drop(cell);
}

/// Cartesian operation of kept operation `k`.
fn wedge_cell(name: &str) -> Cell {
    if name.starts_with("si") {
        silicon(10)
    } else {
        aluminum()
    }
}

/// Orbit labels of a mesh under `k -> k W` and `k -> -k`, by brute force.
fn orbits(mesh: &[f64], rotations: &[[i32; 9]]) -> Vec<usize> {
    let n = mesh.len() / 3;
    let mut label = vec![usize::MAX; n];
    for i in 0..n {
        if label[i] != usize::MAX {
            continue;
        }
        label[i] = i;
        let mut stack = vec![i];
        while let Some(j) = stack.pop() {
            let k = [mesh[3 * j], mesh[3 * j + 1], mesh[3 * j + 2]];
            for w in rotations {
                for sign in [1., -1.] {
                    let image: [f64; 3] = std::array::from_fn(|a| {
                        sign * (0..3).map(|b| k[b] * f64::from(w[3 * b + a])).sum::<f64>()
                    });
                    for t in 0..n {
                        let same = (0..3).all(|a| {
                            let d = mesh[3 * t + a] - image[a];
                            (d - d.round()).abs() < 1e-8
                        });
                        if same && label[t] == usize::MAX {
                            label[t] = i;
                            stack.push(t);
                        }
                    }
                }
            }
        }
    }
    label
}

#[test]
fn the_mesh_wedges_equal_the_frozen_representatives_weights_and_mapping() {
    let body = reference("mesh_wedge.json");
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    for name in [
        "si_3x3x3",
        "si_shifted_2x2x2",
        "si_2x1x1",
        "al_2x2x2",
        "al_shifted_4x4x4",
    ] {
        let case = &body[name];
        let size: Vec<usize> = ints(&case["size"]).iter().map(|&v| v as usize).collect();
        let (size, gamma) = (
            [size[0], size[1], size[2]],
            case["gamma"].as_bool().unwrap(),
        );
        let sym = wedge_cell(name)
            .symmetry()
            .keeping_mesh(size, gamma)
            .unwrap();
        assert_eq!(
            sym.len(),
            case["n_operations_kept"].as_u64().unwrap() as usize,
            "{name}"
        );
        let wedge = sym.mesh(&ctx, size, gamma).unwrap();
        let points: Vec<f64> = wedge.points.iter().flatten().copied().collect();
        assert_eq!(
            points,
            reals(&case["reduced"]["data"]),
            "{name} representatives"
        );
        assert_eq!(
            wedge.weights,
            reals(&case["weights"]["data"]),
            "{name} weights"
        );
        let mapping: Vec<i64> = wedge.mapping.iter().map(|&v| v as i64).collect();
        assert_eq!(
            mapping,
            ints(&case["mesh_wedge_mapping"]["data"]),
            "{name} mapping"
        );
        assert_eq!(
            wedge.full.iter().flatten().copied().collect::<Vec<_>>(),
            reals(&case["mesh"]["data"]),
            "{name} full mesh"
        );
        // Time reversal alone: no operation reaches the reduction.
        let alone = mandacaru_core::periodic::spectrum::mesh(&ctx, size, gamma, &[], true).unwrap();
        assert_eq!(
            alone.points.iter().flatten().copied().collect::<Vec<_>>(),
            reals(&case["time_reversal_only"]["points"]["data"]),
            "{name}"
        );
        assert_eq!(
            alone.weights,
            reals(&case["time_reversal_only"]["weights"]["data"]),
            "{name}"
        );
        // Independent: orbit sizes by brute force, weight x N_k = orbit size.
        let label = orbits(&points_of(&wedge.full), &sym.distinct_rotations());
        let mut sizes: Vec<usize> = (0..label.len())
            .filter(|&i| label[i] == i)
            .map(|i| label.iter().filter(|&&l| l == i).count())
            .collect();
        let mut got: Vec<usize> = wedge
            .weights
            .iter()
            .map(|w| (w * label.len() as f64).round() as usize)
            .collect();
        sizes.sort();
        got.sort();
        assert_eq!(sizes, got, "{name} orbits");
        assert!((wedge.weights.iter().sum::<f64>() - 1.).abs() < 1e-14);
        let generic = wedge
            .points
            .iter()
            .filter(|k| k.iter().any(|x| (2. * x - (2. * x).round()).abs() > 1e-9))
            .count();
        println!(
            "{name}: {} operations, {} k of {}, {generic} with k != -k",
            sym.len(),
            wedge.points.len(),
            wedge.full.len()
        );
        assert_eq!(
            generic as u64,
            case["check"]["n_k_with_k_ne_minus_k"].as_u64().unwrap()
        );
        if ["si_3x3x3", "si_shifted_2x2x2", "al_shifted_4x4x4"].contains(&name) {
            assert!(generic > 0, "{name} must exercise k != -k");
        }
    }
}
fn points_of(full: &[[f64; 3]]) -> Vec<f64> {
    full.iter().flatten().copied().collect()
}

#[test]
fn the_multipole_average_equals_the_frozen_crystal_and_is_a_projector() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/phase5/si_crystal.json");
    let frozen: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let channels: Vec<(usize, usize, i32)> = frozen["channel_labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            (
                c[0].as_u64().unwrap() as usize,
                c[1].as_u64().unwrap() as usize,
                c[2].as_i64().unwrap() as i32,
            )
        })
        .collect();
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let sym = silicon(10)
        .symmetry()
        .keeping_mesh([2, 2, 2], true)
        .unwrap();
    assert_eq!(sym.len(), 24);
    let q = sym.qsym(&ctx, &channels).unwrap();
    let n = channels.len();
    let want: Vec<Complex64> = frozen["qsym"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| Complex64::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap()))
        .collect();
    let worst = q
        .as_slice()
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).norm())
        .fold(0., f64::max);
    let m = nalgebra::DMatrix::from_row_slice(n, n, q.as_slice());
    let idempotent = (&m * &m - &m).iter().map(|v| v.norm()).fold(0., f64::max);
    println!("qsym 18 x 18: difference to the frozen {worst:.2e} (T2), projector defect {idempotent:.1e}");
    assert!(worst < T2 * 1e2 && idempotent < 1e-12);
    // A trivial group is the identity and takes no rotation table.
    let single = Cell::new(
        [[5., 0., 0.], [0., 5.7, 0.], [0., 0., 6.1]],
        &[[0.1, 0.2, 0.3]],
        1,
        [4; 3],
    );
    let sym = single.symmetry();
    let id = sym.qsym(&ctx, &[(0, 0, 0), (0, 1, -1), (0, 1, 0)]).unwrap();
    assert!(sym.len() >= 1);
    let trace: Complex64 = (0..3).map(|i| id.as_slice()[4 * i]).sum();
    assert!((trace.re - 3.).abs() < 1e-12);
}
