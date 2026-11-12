//! Shared by `berry_links` and `berry_weighted`: lattice models built directly (no crystal), a
//! deterministic random generator and the link of two state matrices.
#![allow(dead_code)]
use mandacaru_core::quantum::C64;
use nalgebra::{DMatrix, SymmetricEigen};
use std::f64::consts::PI;

pub type Matrix = DMatrix<C64>;

/// A deterministic generator, so a failing gauge is reproducible.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-300), self.next());
        (-2. * u.ln()).sqrt() * (2. * PI * v).cos()
    }
    /// A Haar-ish random unitary: the Q of a QR of a complex Gaussian matrix.
    pub fn unitary(&mut self, n: usize) -> Matrix {
        let g = Matrix::from_fn(n, n, |_, _| C64::new(self.normal(), self.normal()));
        g.qr().q()
    }
}

/// The `n` lowest eigenvectors (columns) of a Hermitian matrix.
pub fn lowest(h: Matrix, n: usize) -> Matrix {
    let eig = SymmetricEigen::new(h);
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|a, b| eig.eigenvalues[*a].total_cmp(&eig.eigenvalues[*b]));
    Matrix::from_fn(eig.eigenvectors.nrows(), n, |i, j| {
        eig.eigenvectors[(i, order[j])]
    })
}

pub fn sigma(d: [f64; 3]) -> Matrix {
    Matrix::from_row_slice(
        2,
        2,
        &[
            C64::new(d[2], 0.),
            C64::new(d[0], -d[1]),
            C64::new(d[0], d[1]),
            C64::new(-d[2], 0.),
        ],
    )
}

pub fn link(left: &Matrix, right: &Matrix) -> Matrix {
    left.adjoint() * right
}

/// The closed string of links of the states `states[s]` (the last links back to the first).
pub fn string(states: &[Matrix]) -> Vec<Matrix> {
    (0..states.len())
        .map(|s| link(&states[s], &states[(s + 1) % states.len()]))
        .collect()
}

/// Qi-Wu-Zhang: `d = (sin kx, sin ky, m + cos kx + cos ky)`.
pub fn qwz(m: f64, kx: f64, ky: f64) -> [f64; 3] {
    [kx.sin(), ky.sin(), m + kx.cos() + ky.cos()]
}
