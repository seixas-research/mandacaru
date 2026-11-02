//! Bloch sums of atom-centered shells (`periodic::setup::{images, shells}`), against the plain
//! image sum written in `setup_shells_support` and against the frozen diagnostics of
//! `reference/periodic/orbitals_si_sz.json`, `basis_table.json` and the `si_crystal.json` grid
//! sums. Needs `MANDACARU_PAW_PATH` (the shipped `lda-sr` Si dataset).
mod setup_shells_support;

use nalgebra::Matrix3;
use num_complex::Complex64;
use setup_shells_support::*;

/// Frozen kernel-versus-reference gate of the diagnostics (T2).
const T2: f64 = 1e-9;
/// The same kernel algorithm against an image sum that cuts at the same support.
const SUM: f64 = 1e-12;

#[test]
fn the_rebuilt_basis_has_the_frozen_order_and_support() {
    let ctx = context();
    let (cell, basis) = silicon(&ctx);
    let mut frozen = cell.orbitals.iter();
    for table in &basis.tables {
        for m in &table.ms {
            let &(atom, l, mm, support) = frozen.next().unwrap();
            assert_eq!((l, mm), (table.l, *m));
            assert_eq!(cell.centers[atom], table.center);
            // The frozen file keeps ten digits.
            assert!(
                (support - table.support).abs() < 1e-8,
                "frozen {support} against {} (l = {l})",
                table.support
            );
        }
    }
    assert!(frozen.next().is_none());
    // The kernel's view of the support is the table's.
    for (shell, table) in basis.shells.shells().iter().zip(&basis.tables) {
        assert_eq!(shell.radial.support(), table.support);
    }
}

#[test]
fn bloch_sums_equal_the_independent_image_sum() {
    let ctx = context();
    let (cell, basis) = silicon(&ctx);
    let frozen = frozen_orbitals();
    let got = bloch(&ctx, &basis, &cell, &frozen.kpoints, &frozen.points);
    let want = image_sum(&ctx, &basis, &cell.lattice, &frozen.kpoints, &frozen.points);
    let error = relative(&got, &want);
    eprintln!("kernel versus the image sum at the same support: {error:e}");
    assert!(error < SUM, "{error:e}");
}

#[test]
fn bloch_sums_equal_the_frozen_values() {
    let ctx = context();
    let (cell, basis) = silicon(&ctx);
    let frozen = frozen_orbitals();
    let got = bloch(&ctx, &basis, &cell, &frozen.kpoints, &frozen.points);
    let error = relative(&got, &frozen.values);
    // The frozen image sum keeps every image's whole table tail, the kernel cuts at the support.
    let tail = relative(&got, &frozen.image_sum);
    eprintln!("frozen values {error:e}; frozen full-table image sum {tail:e}");
    assert!(error < T2, "{error:e}");
    assert!(tail < SUM, "{tail:e}");
}

#[test]
fn the_phase_conventions_are_pinned_by_non_vacuous_controls() {
    let ctx = context();
    let (cell, basis) = silicon(&ctx);
    let frozen = frozen_orbitals();
    let n = frozen.points.len();
    let functions = basis.shells.functions();
    let got = bloch(&ctx, &basis, &cell, &frozen.kpoints, &frozen.points);
    // The wrong sign of k.R is far from the sum (the frozen control is 1.48).
    let flipped: Vec<[f64; 3]> = frozen.kpoints.iter().map(|k| k.map(|x| -x)).collect();
    let wrong = bloch(&ctx, &basis, &cell, &flipped, &frozen.points);
    assert!(relative(&wrong, &got) > 0.5, "{}", relative(&wrong, &got));
    // chi(r + A_j) = e^{i k.A_j} chi(r) at the three lattice vectors.
    for j in 0..3 {
        let a = cell.lattice.column(j).into_owned();
        let moved: Vec<[f64; 3]> = frozen
            .points
            .iter()
            .map(|p| [p[0] + a[0], p[1] + a[1], p[2] + a[2]])
            .collect();
        let shifted = bloch(&ctx, &basis, &cell, &frozen.kpoints, &moved);
        let mut law = got.clone();
        let mut unphased = 0f64;
        let scale = got.iter().map(|v| v.norm()).fold(0., f64::max);
        for (k, q) in frozen.kpoints.iter().enumerate() {
            let angle = q[0] * a[0] + q[1] * a[1] + q[2] * a[2];
            let phase = Complex64::new(angle.cos(), angle.sin());
            for v in &mut law[k * functions * n..(k + 1) * functions * n] {
                *v *= phase;
            }
        }
        for (s, g) in shifted.iter().zip(&got) {
            unphased = unphased.max((s - g).norm() / scale);
        }
        let error = relative(&shifted, &law);
        eprintln!("translation law, vector {j}: {error:e} (without the phase {unphased:e})");
        assert!(error < SUM, "{error:e}");
        assert!(unphased > 1e-3, "the law must not hold without its phase");
    }
}

#[test]
fn minus_k_is_the_conjugate_with_m_flipped() {
    let ctx = context();
    let (cell, basis) = silicon(&ctx);
    let frozen = frozen_orbitals();
    // Frozen k[1] = -k[0].
    assert!(frozen.kpoints[0]
        .iter()
        .zip(&frozen.kpoints[1])
        .all(|(a, b)| (a + b).abs() < 1e-15));
    let n = frozen.points.len();
    let functions = basis.shells.functions();
    let got = bloch(&ctx, &basis, &cell, &frozen.kpoints, &frozen.points);
    let scale = got.iter().map(|v| v.norm()).fold(0., f64::max);
    let (mut worst, mut naive) = (0f64, 0f64);
    for shell in basis.shells.shells() {
        for (&m, &row) in shell.ms.iter().zip(&shell.rows) {
            let partner = shell.rows[shell.ms.iter().position(|x| *x == -m).unwrap()];
            let sign = if m % 2 == 0 { 1. } else { -1. };
            for j in 0..n {
                let plus = got[row * n + j];
                let minus = got[(functions + partner) * n + j];
                worst = worst.max((minus - sign * plus.conj()).norm() / scale);
                // Without the m flip or without the conjugate the pair is not related.
                naive = naive.max((got[(functions + row) * n + j] - plus.conj()).norm() / scale);
            }
        }
    }
    eprintln!("time reversal with the m flip: {worst:e} (naive {naive:e})");
    let _ = cell;
    assert!(worst < 1e-14 && naive > 1e-3);
}

/// The nodes of the snapshot's grid, `C` order (the layout of psi).
fn grid_nodes(snapshot: &serde_json::Value) -> Vec<[f64; 3]> {
    let shape: Vec<usize> = snapshot["grid"]["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let step = Matrix3::from_row_slice(&reals(&snapshot["grid"]["step"]));
    // The first node is the origin, which the file keeps to round-off.
    assert!(reals(&snapshot["grid"]["origin"])
        .iter()
        .all(|x| x.abs() < 1e-14));
    let mut points = Vec::new();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let r = step.row(0) * i as f64 + step.row(1) * j as f64 + step.row(2) * k as f64;
                points.push([r[0], r[1], r[2]]);
            }
        }
    }
    points
}

#[test]
fn the_grid_sums_equal_the_frozen_psi() {
    let ctx = context();
    let (cell, basis) = silicon(&ctx);
    let snapshot = snapshot("si_crystal.json");
    let points = grid_nodes(&snapshot);
    let kpoints = snapshot["kpoints"].as_array().unwrap();
    let ks: Vec<[f64; 3]> = kpoints
        .iter()
        .map(|k| {
            let k = reals(&k["k"]);
            [k[0], k[1], k[2]]
        })
        .collect();
    let got = bloch(&ctx, &basis, &cell, &ks, &points);
    let functions = basis.shells.functions();
    let n = points.len();
    for (k, entry) in kpoints.iter().enumerate() {
        let want = pairs(&entry["psi"]);
        let error = relative(&got[k * functions * n..(k + 1) * functions * n], &want);
        eprintln!("frozen psi at k{k}: {error:e}");
        assert!(error < T2, "{error:e}");
    }
    let _ = cell;
}

#[test]
fn the_grid_sums_at_irrational_k_equal_every_37th_node_of_the_live_freeze() {
    let ctx = context();
    let (cell, basis) = silicon(&ctx);
    // The grid of `kpoint_matrices.json` is the 10^3 grid of the snapshot.
    let points = grid_nodes(&snapshot("si_crystal.json"));
    let frozen = reference("kpoint_matrices.json");
    let ks = triples(&real_array(&frozen["si"]["k_cartesian"]));
    let got = bloch(&ctx, &basis, &cell, &ks, &points);
    let (functions, n) = (basis.shells.functions(), points.len());
    let stride = frozen["psi_stride"].as_u64().unwrap() as usize;
    let want = complex_array(&frozen["si"]["psi_stride"]);
    let kept = n.div_ceil(stride);
    let mut sampled = Vec::new();
    for k in 0..ks.len() {
        for row in 0..functions {
            for j in (0..n).step_by(stride) {
                sampled.push(got[(k * functions + row) * n + j]);
            }
        }
    }
    assert_eq!(sampled.len(), want.len());
    assert_eq!(want.len(), ks.len() * functions * kept);
    let error = relative(&sampled, &want);
    eprintln!("kpoint_matrices.json psi on every {stride}th node, 3 irrational k: {error:e}");
    assert!(error < T2, "{error:e}");
}
