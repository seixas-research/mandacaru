//! The spin-orbit block between the spinor Bloch sums (`periodic::setup::spin_orbit`) of Si
//! `lda-fr`, SZ, against `reference/periodic/spin_orbit.json` (the `(2, M, 2, M)` block, alpha
//! first) and the mesh points of `phase5/si_so.json`. Needs `MANDACARU_PAW_PATH`.
mod setup_shells_support;

use mandacaru_core::{
    math::linalg::eigen,
    periodic::setup::spin_orbit::{channels, matrices, Coupling},
    Context, Error,
};
use nalgebra::{Cholesky, DMatrix, Matrix3};
use num_complex::Complex64;
use setup_shells_support::*;
use std::collections::BTreeMap;

const T2: f64 = 1e-9;

struct Silicon {
    ctx: Context,
    cell: Cell,
    basis: Basis,
    projectors: Vec<mandacaru_core::periodic::setup::spheres::Projector>,
    couplings: BTreeMap<(usize, usize), Coupling>,
}
fn silicon() -> Silicon {
    let ctx = context();
    let cell = Cell::read("si_sz_lda_fr");
    let ds = dataset(&ctx, "lda-fr", "Si");
    let basis = minimal_basis(&ctx, &ds, &cell.centers, cell.filter_wavevector);
    let (projectors, couplings) = spin_orbit_list(&ctx, &ds, &cell.centers);
    Silicon {
        ctx,
        cell,
        basis,
        projectors,
        couplings,
    }
}
impl Silicon {
    fn block(&self, ks: &[[f64; 3]]) -> Vec<Complex64> {
        matrices(
            &self.ctx,
            &self.basis.shells,
            &self.cell.lattice,
            ks,
            &self.projectors,
            &self.couplings,
        )
        .unwrap()
        .as_slice()
        .to_vec()
    }
}

/// The eigenvalues of `H x = e (S kron I2) x`, ascending.
fn spectrum(h: &[Complex64], s: &[Complex64], m: usize) -> Vec<f64> {
    let n = 2 * m;
    let metric = DMatrix::from_fn(n, n, |i, j| {
        if i / m == j / m {
            s[(i % m) * m + j % m]
        } else {
            Complex64::new(0., 0.)
        }
    });
    let l = Cholesky::new(metric).expect("S is positive definite").l();
    let h = DMatrix::from_fn(n, n, |i, j| h[i * n + j]);
    let left = l.solve_lower_triangular(&h).unwrap();
    let both = l.solve_lower_triangular(&left.adjoint()).unwrap().adjoint();
    eigen(both).unwrap().0
}

#[test]
fn the_block_equals_the_frozen_spin_orbit_matrix() {
    let t = silicon();
    let frozen = reference("spin_orbit.json");
    let ks = triples(&real_array(&frozen["k_cartesian"]));
    let got = t.block(&ks);
    let want = complex_array(&frozen["spin_orbit"]);
    let m = t.basis.shells.functions();
    assert_eq!(m, 8);
    let size = 4 * m * m;
    for k in 0..ks.len() {
        let error = relative(
            &got[k * size..(k + 1) * size],
            &want[k * size..(k + 1) * size],
        );
        eprintln!("spin_orbit.json h^SO at k{k}: {error:e}");
        assert!(error < T2, "{error:e}");
    }
    // The block is not empty, couples the spins and is not symmetric in alpha and beta.
    let scale = got.iter().map(|v| v.norm()).fold(0., f64::max);
    assert!(scale > 1e-4);
    let (mut off, mut swap) = (0f64, 0f64);
    for i in 0..m {
        for j in 0..m {
            off = off.max(got[i * 2 * m + m + j].norm());
            swap = swap.max((got[i * 2 * m + j] - got[(m + i) * 2 * m + m + j]).norm());
        }
    }
    assert!(off > 1e-5 && swap > 1e-5, "{off:e} {swap:e}");
}

#[test]
fn the_block_equals_the_mesh_points_of_the_boundary_snapshot() {
    let t = silicon();
    let snap = snapshot("si_so.json");
    let entries = snap["kpoints"].as_array().unwrap();
    let ks: Vec<[f64; 3]> = entries
        .iter()
        .map(|e| {
            let k = reals(&e["k"]);
            [k[0], k[1], k[2]]
        })
        .collect();
    let got = t.block(&ks);
    let size = 4 * 64;
    for (k, entry) in entries.iter().enumerate() {
        let error = relative(&got[k * size..(k + 1) * size], &pairs(&entry["spin_orbit"]));
        eprintln!("si_so.json h^SO at k{k}: {error:e}");
        assert!(error < T2, "{error:e}");
    }
}

#[test]
fn the_block_is_hermitian_periodic_and_kramers_degenerate() {
    let t = silicon();
    let frozen = reference("spin_orbit.json");
    let ks = triples(&real_array(&frozen["k_cartesian"]));
    let m = 8;
    let overlap = complex_array(&frozen["overlap"]);
    let got = t.block(&ks);
    let size = 4 * m * m;
    // Hermitian.
    for k in 0..ks.len() {
        let h = &got[k * size..(k + 1) * size];
        let mut defect = 0f64;
        for i in 0..2 * m {
            for j in 0..2 * m {
                defect = defect.max((h[i * 2 * m + j] - h[j * 2 * m + i].conj()).norm());
            }
        }
        assert!(defect < 1e-14, "{defect:e}");
    }
    // Periodic in G: k and k + G give the same matrix.
    let a: Matrix3<f64> = t.cell.lattice;
    let b = a.try_inverse().unwrap().transpose() * (2. * std::f64::consts::PI);
    let g = b * nalgebra::Vector3::new(1., -1., 2.);
    let moved = [ks[2][0] + g[0], ks[2][1] + g[1], ks[2][2] + g[2]];
    let shifted = t.block(&[moved]);
    let defect = relative(&shifted, &got[2 * size..3 * size]);
    eprintln!("h^SO(k + G) against h^SO(k): {defect:e}");
    assert!(defect < 1e-11, "{defect:e}");
    // Time reversal: the spectrum at k and at -k, in the metric of the overlap, is the same.
    let metric = |k: usize| &overlap[k * m * m..(k + 1) * m * m];
    let plus = spectrum(&got[2 * size..3 * size], metric(2), m);
    let minus = spectrum(&got[3 * size..4 * size], metric(3), m);
    let kramers = plus
        .iter()
        .zip(&minus)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    eprintln!("Kramers, k against -k: {kramers:e}");
    assert!(kramers < 1e-11);
    let gamma = spectrum(&got[..size], metric(0), m);
    let pairs_at_gamma = (0..m)
        .map(|i| (gamma[2 * i] - gamma[2 * i + 1]).abs())
        .fold(0., f64::max);
    assert!(pairs_at_gamma < 1e-12, "{pairs_at_gamma:e}");
    // And the frozen spectra.
    let want = real_array(&frozen["spectra_in_overlap_metric"]);
    let scale = want.iter().fold(0f64, |a, v| a.max(v.abs()));
    for (k, expected) in want.chunks(2 * m).enumerate() {
        let spec = spectrum(&got[k * size..(k + 1) * size], metric(k), m);
        let worst = spec
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        assert!(worst < T2 * scale, "{worst:e}");
    }
}

#[test]
fn channels_follow_the_projectors_and_refuse_incomplete_ones() {
    let t = silicon();
    let found = channels(&t.ctx, &t.projectors, &t.couplings).unwrap();
    // One channel per atom and l > 0 that carries the term.
    assert_eq!(found.len(), t.couplings.keys().filter(|k| k.1 > 0).count());
    for c in &found {
        assert_eq!(c.indices.as_slice().len(), (2 * c.l + 1) * c.radial);
        assert!(!c.real);
    }
    // A missing projector of the rectangle, a duplicated label, a coupling of the wrong size.
    let mut missing = t.projectors.clone();
    missing.remove(5);
    assert!(matches!(
        channels(&t.ctx, &missing, &t.couplings),
        Err(Error::Invalid(_))
    ));
    let mut twice = t.projectors.clone();
    twice.push(t.projectors[2].clone());
    assert!(channels(&t.ctx, &twice, &t.couplings).is_err());
    let mut wrong = t.couplings.clone();
    for c in wrong.values_mut() {
        c.values.pop();
    }
    assert!(channels(&t.ctx, &t.projectors, &wrong).is_err());
    // A crystal without the term: zeros of the same shape.
    let none = matrices(
        &t.ctx,
        &t.basis.shells,
        &t.cell.lattice,
        &[[0.1, 0.2, 0.3]],
        &[],
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(none.as_slice().len(), 4 * 64);
    assert!(none.as_slice().iter().all(|v| v.norm() == 0.));
}
