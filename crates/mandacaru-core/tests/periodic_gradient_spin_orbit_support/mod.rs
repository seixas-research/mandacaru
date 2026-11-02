//! The toy of `periodic_gradient_spin_orbit.rs`: the HU-10a toy (`hubbard_gradient_support`, one
//! owner of `C(X)` and its exact atom derivative) carrying one spin-orbit channel per shell
//! (`l = 1`, one radial projector, so the three toy projectors of a shell are `m = -1, 0, 1`),
//! a fixed Hermitian spinor density, and the independent oracle of the force term: the exact
//! `dC_X` of the toy (`bloch_atom`, the ownership applied inside the toy) pushed through
//! `models::spin_orbit::pair`.
#![allow(dead_code)]

use crate::hubbard_gradient_support::{Kind, Matrix, Toy, NP};
use mandacaru_core::{
    models::spin_orbit::{pair, Channel, Projectors},
    periodic::gradient::spin_orbit::{operator, Gradient, SpinOrbit},
    quantum::C64,
    Context,
};

pub fn context() -> Context {
    Context::new(Some(1), 1 << 28).unwrap()
}

/// One channel per shell with the coupling strengths `lambda[shell]`.
pub fn channels(ctx: &Context, toy: &Toy, lambda: &[f64]) -> Vec<Channel> {
    (0..toy.shell_atoms.len())
        .map(|s| {
            let mut indices = ctx.allocate::<usize>(NP).unwrap();
            for (i, to) in indices.as_mut_slice().iter_mut().enumerate() {
                *to = s * NP + i;
            }
            let mut coupling = ctx.allocate::<C64>(1).unwrap();
            coupling.as_mut_slice()[0] = C64::new(lambda[s], 0.);
            Channel {
                l: 1,
                real: false,
                indices,
                radial: 1,
                coupling,
            }
        })
        .collect()
}

pub fn spin_orbit(ctx: &Context, toy: &Toy, lambda: &[f64]) -> SpinOrbit {
    let b = operator(ctx, toy.projectors(), &channels(ctx, toy, lambda)).unwrap();
    SpinOrbit::new(b, toy.basis_owner(), toy.projector_owner(), toy.atoms()).unwrap()
}

/// A fixed Hermitian `2m x 2m` matrix with every spin block populated (a deterministic
/// congruential generator; `seed` selects the matrix).
pub fn density(m: usize, seed: u64) -> Matrix {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    let raw = Matrix::from_fn(2 * m, 2 * m, |_, _| C64::new(next(), next()));
    (&raw + raw.adjoint()) * C64::new(0.5, 0.)
}

/// `G^a(k) = dC/dc_row` of the toy's projection block, read off the image blocks (the toy's
/// sign `s = 1` for a projection; no ownership mask).
pub fn row_derivative(toy: &Toy, k: [f64; 3]) -> [Matrix; 3] {
    let blocks = toy.blocks(Kind::Projection);
    std::array::from_fn(|axis| {
        toy.sum(Kind::Projection, k, |image, i, j| {
            blocks.gradients[image][axis][(i, j)]
        })
    })
}

pub fn row_major(m: &Matrix) -> Vec<C64> {
    m.transpose().as_slice().to_vec()
}

pub fn trace_with(d: &Matrix, h: &[C64]) -> f64 {
    let n = d.nrows();
    (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| d[(i, j)] * h[j * n + i])
        .sum::<C64>()
        .re
}

/// `pair(left, right)` of two `m x p` blocks, not symmetrized.
pub fn pair_of(ctx: &Context, chs: &[Channel], left: &Matrix, right: &Matrix) -> Vec<C64> {
    let (l, r) = (row_major(left), row_major(right));
    pair(
        ctx,
        Projectors {
            left: &l,
            right: &r,
            rows: left.nrows(),
            columns: right.nrows(),
            count: left.ncols(),
        },
        chs,
        false,
    )
    .unwrap()
    .as_slice()
    .to_vec()
}

/// `E_SO = sum_k w_k Re tr[D h^SO(k)]` of the toy geometry, `h` through `pair`.
pub fn energy(ctx: &Context, toy: &Toy, chs: &[Channel], d: &Matrix) -> f64 {
    toy.kpoints()
        .iter()
        .zip(&toy.weights)
        .map(|(&k, w)| {
            let c = toy.bloch(Kind::Projection, k);
            w * trace_with(d, &pair_of(ctx, chs, &c, &c))
        })
        .sum()
}

/// The oracle: `sum_k w_k Re tr[D (pair(dC_X, C) + pair(C, dC_X))]` with the toy's exact
/// `dC_X`; per atom and axis.
pub fn oracle(ctx: &Context, toy: &Toy, chs: &[Channel], d: &Matrix) -> Vec<[f64; 3]> {
    (0..toy.atoms())
        .map(|atom| {
            std::array::from_fn(|axis| {
                toy.kpoints()
                    .iter()
                    .zip(&toy.weights)
                    .map(|(&k, w)| {
                        let c = toy.bloch(Kind::Projection, k);
                        let dc = toy.bloch_atom(Kind::Projection, k, atom, axis);
                        let h1 = pair_of(ctx, chs, &dc, &c);
                        let h2 = pair_of(ctx, chs, &c, &dc);
                        w * (trace_with(d, &h1) + trace_with(d, &h2))
                    })
                    .sum()
            })
        })
        .collect()
}

/// The engine's force term over the toy's k-points.
pub fn engine(toy: &Toy, so: &SpinOrbit, d: &Matrix) -> Gradient {
    let mut gradient = Gradient::new(toy.atoms());
    for (&k, &w) in toy.kpoints().iter().zip(&toy.weights) {
        let c = toy.bloch(Kind::Projection, k);
        so.add_k(&mut gradient, w, &c, &row_derivative(toy, k), d)
            .unwrap();
    }
    gradient
}

pub fn largest(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| (0..3).map(move |i| (x[i] - y[i]).abs()))
        .fold(0., f64::max)
}

pub fn peak(a: &[[f64; 3]]) -> f64 {
    a.iter().flatten().fold(0., |m, v| f64::max(m, v.abs()))
}

/// Central difference of `energy` over the atom's coordinate, Richardson-extrapolated
/// `(4 c(h) - c(2h)) / 3`, error `O(h^4)`; returns it with the `c(h) - c(2h)` spread.
pub fn finite_difference(
    ctx: &Context,
    toy: &Toy,
    chs: &[Channel],
    d: &Matrix,
    step: f64,
) -> (Vec<[f64; 3]>, f64) {
    let central = |atom: usize, axis: usize, h: f64| {
        (energy(ctx, &toy.displaced(atom, axis, h), chs, d)
            - energy(ctx, &toy.displaced(atom, axis, -h), chs, d))
            / (2. * h)
    };
    let mut spread = 0f64;
    let out = (0..toy.atoms())
        .map(|atom| {
            std::array::from_fn(|axis| {
                let (a, b) = (central(atom, axis, step), central(atom, axis, 2. * step));
                spread = spread.max((a - b).abs());
                (4. * a - b) / 3.
            })
        })
        .collect();
    (out, spread)
}

/// A fixed general (non-Hermitian) `2m x 2m` matrix: the anti-Hermitian part is populated.
pub fn general(m: usize, seed: u64) -> Matrix {
    let hermitian = density(m, seed);
    let other = density(m, seed + 1000);
    // H1 + i H2 is neither Hermitian nor anti-Hermitian.
    &hermitian + other * C64::new(0., 1.)
}

/// Rigid translation of every center: `energy` must not change (the toy's blocks depend on
/// displacements only; this is the value form of the translation invariance the atom sum of the
/// force stands for).
pub fn translated(toy: &Toy, shift: [f64; 3]) -> Toy {
    let mut out = toy.clone();
    for atom in 0..toy.atoms() {
        for (axis, delta) in shift.iter().enumerate() {
            out = out.displaced(atom, axis, *delta);
        }
    }
    out
}

pub mod spinor;
