//! Shared by the `periodic_screened_onecenter*` tests: the shipped species as `ScreenedAtom`s, the
//! rotated dense tensor the older `Hybrid` evaluator takes, and reproducible densities.
#![allow(dead_code)]

use mandacaru_core::{
    paw::{
        onecenter::Basis,
        pair_tables::{layout, Projector},
        record::PawDataset,
        screened_atom::{screened, ScreenedAtom},
        species_tables::SpeciesTables,
    },
    periodic::screened_onecenter::harmonic_rotation,
    quantum::C64,
    Context,
};
use nalgebra::DMatrix;
use std::path::PathBuf;

pub type Matrix = DMatrix<C64>;
pub const OMEGA: f64 = 0.11;
pub const FRACTION: f64 = 0.25;

pub fn read(ctx: &Context, set: &str, symbol: &str) -> PawDataset {
    let root = std::env::var_os("MANDACARU_PAW_PATH")
        .filter(|v| !v.is_empty())
        .expect("MANDACARU_PAW_PATH is unset");
    let path = PathBuf::from(root).join(format!("{set}/{symbol}.parquet"));
    PawDataset::read(ctx, &path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}
pub fn terms(
    ctx: &Context,
    set: &str,
    symbol: &str,
    basis: Basis,
) -> (ScreenedAtom, Vec<Projector>) {
    let record = read(ctx, set, symbol);
    let tables = SpeciesTables::from_record(ctx, &record, basis).unwrap();
    let view = tables.view();
    let projectors = layout(&view);
    (
        screened(ctx, &view, &projectors, OMEGA).unwrap(),
        projectors,
    )
}

/// A reproducible Hermitian positive matrix of order `n`.
pub fn density(n: usize, seed: f64, scale: f64) -> Matrix {
    let a = Matrix::from_fn(n, n, |i, j| {
        let x = (i * 7 + j * 3) as f64 * 0.37 + seed;
        C64::new(x.sin(), (1.7 * x + 0.3).cos()) * (0.35 / (1. + 0.5 * (i + j) as f64))
    });
    &a * a.adjoint() * C64::new(scale, 0.)
}

/// `Delta W^c[(ij), (kl)]` of the model's species rotated to complex harmonics, as a dense
/// `n^4` tensor in `Hybrid`'s `[a][b][c][d]` order, by the explicit sum over the real basis.
pub fn rotated_tensor(atom: &ScreenedAtom, layout: &[Projector]) -> Vec<C64> {
    let n = layout.len();
    let mut s = Matrix::zeros(n, n);
    for (i, a) in layout.iter().enumerate() {
        for (j, b) in layout.iter().enumerate() {
            if a.channel == b.channel && a.radial == b.radial {
                let rotation = harmonic_rotation(a.l);
                s[(i, j)] = rotation[((a.m + a.l as i32) as usize, (b.m + a.l as i32) as usize)];
            }
        }
    }
    let mut real = vec![0.; n * n * n * n];
    for (f, w) in atom.factors.iter().zip(&atom.weights) {
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    for d in 0..n {
                        real[((a * n + b) * n + c) * n + d] += w * f[(a, b)] * f[(c, d)];
                    }
                }
            }
        }
    }
    let mut out = vec![C64::default(); n * n * n * n];
    // One index at a time: W^c = S^* (x) S (x) S^* (x) S applied to W.
    let mut current: Vec<C64> = real.iter().map(|x| C64::new(*x, 0.)).collect();
    for axis in 0..4 {
        let stride = n.pow(3 - axis as u32);
        for at in 0..n * n * n * n {
            let index = at / stride % n;
            let base = at - index * stride;
            out[at] = (0..n)
                .map(|k| {
                    let entry = if axis % 2 == 0 {
                        s[(index, k)].conj()
                    } else {
                        s[(index, k)]
                    };
                    entry * current[base + k * stride]
                })
                .sum();
        }
        current.clone_from(&out);
    }
    current
}

pub fn slice(m: &Matrix) -> Vec<C64> {
    (0..m.nrows())
        .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]).collect::<Vec<_>>())
        .collect()
}
