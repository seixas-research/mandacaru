//! The toy of `periodic_hubbard_gradient.rs`: a one-dimensional chain of atoms in a periodic
//! cell, a non-orthogonal basis with Gaussian overlaps, Hamiltonian blocks and projector
//! overlaps that are analytic functions of the geometry, and a few k-points. No engine: the
//! overlap `S(X)`, the Hamiltonian `H0(X)` and the projections `C(X)` are written here, so the
//! converged toy energy is an exact function of the coordinates and the strain.
//!
//! Conventions. `S[i,j](k) = sum_R e^{+ik.R} f(c_j + R - c_i)` and the same for `H0`;
//! `C[mu,p](k) = sum_R e^{-ik.R} f(c_mu + R - c_p)` (the basis function sits in the image `R`).
#![allow(dead_code)]

pub mod functional;
pub mod reference;
pub mod scf;

use mandacaru_core::quantum::C64;
use nalgebra::DMatrix;

pub type Matrix = DMatrix<C64>;

/// Basis functions per atom, projectors per shell, orbitals per shell.
pub const NB: usize = 2;
pub const NP: usize = 3;
pub const NM: usize = 2;
const IMAGES: i32 = 2;

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Overlap,
    Hopping,
    Projection,
}

/// The radial weights `W` of a shell (`NP x NM`), the same for every shell.
pub fn shell_weights() -> DMatrix<f64> {
    DMatrix::from_row_slice(NP, NM, &[0.8, 0.1, 0.3, 0.7, 0.1, 0.2])
}

fn sigma(i: usize, j: usize) -> f64 {
    if i == j {
        return 1.0;
    }
    let (x, y) = (i as f64, j as f64);
    0.3 / (1.0 + 0.5 * (x - y).abs()) * (1.0 + 0.2 * (x + y).sin())
}

/// Coefficient and Gaussian exponent of the pair.
fn pair(kind: Kind, i: usize, j: usize) -> (f64, f64) {
    let (x, y) = (i as f64, j as f64);
    match kind {
        Kind::Overlap => (sigma(i, j), 0.10),
        Kind::Hopping => (-0.45 * sigma(i, j) * (1.0 + 0.15 * (x * y).cos()), 0.09),
        Kind::Projection => (0.5 * (1.0 + 0.4 * (1.3 * x + 0.7 * y + 0.2).sin()), 0.07),
    }
}

/// Real blocks per image: values and the derivative with respect to the displacement `d`.
pub struct Blocks {
    pub values: Vec<DMatrix<f64>>,
    pub gradients: Vec<[DMatrix<f64>; 3]>,
}

#[derive(Clone)]
pub struct Toy {
    pub tau: Vec<[f64; 3]>,
    pub a1: [f64; 3],
    pub shell_atoms: Vec<usize>,
    pub fractions: Vec<f64>,
    pub weights: Vec<f64>,
}

impl Toy {
    pub fn two_site() -> Self {
        Toy {
            tau: vec![[0.0, 0.0, 0.0], [2.3, 0.7, -0.4]],
            a1: [5.6, 0.5, -0.3],
            shell_atoms: vec![0, 1],
            fractions: vec![0.0, 0.3, 0.7],
            weights: vec![0.2, 0.5, 0.3],
        }
    }

    /// Three atoms; only the first and the last carry a shell.
    pub fn three_site() -> Self {
        Toy {
            tau: vec![[0.0, 0.0, 0.0], [2.0, 0.6, -0.3], [3.9, -0.5, 0.8]],
            a1: [6.2, 0.4, 0.2],
            shell_atoms: vec![0, 2],
            fractions: vec![0.1, 0.45, 0.8],
            weights: vec![0.25, 0.4, 0.35],
        }
    }

    pub fn atoms(&self) -> usize {
        self.tau.len()
    }
    pub fn basis(&self) -> usize {
        NB * self.tau.len()
    }
    pub fn projectors(&self) -> usize {
        NP * self.shell_atoms.len()
    }
    pub fn basis_centers(&self) -> Vec<[f64; 3]> {
        (0..self.basis()).map(|i| self.tau[i / NB]).collect()
    }
    pub fn projector_centers(&self) -> Vec<[f64; 3]> {
        (0..self.projectors())
            .map(|q| self.tau[self.shell_atoms[q / NP]])
            .collect()
    }
    pub fn basis_owner(&self) -> Vec<usize> {
        (0..self.basis()).map(|i| i / NB).collect()
    }
    pub fn projector_owner(&self) -> Vec<usize> {
        (0..self.projectors())
            .map(|q| self.shell_atoms[q / NP])
            .collect()
    }
    pub fn images(&self) -> Vec<[f64; 3]> {
        (-IMAGES..=IMAGES)
            .map(|n| self.a1.map(|x| f64::from(n) * x))
            .collect()
    }

    /// Cartesian k with `k.R = 2 pi f n` for the image `n`: fractional k stays fixed in a strain.
    pub fn kpoints(&self) -> Vec<[f64; 3]> {
        let norm: f64 = self.a1.iter().map(|x| x * x).sum();
        self.fractions
            .iter()
            .map(|f| self.a1.map(|x| 2.0 * std::f64::consts::PI * f * x / norm))
            .collect()
    }

    pub fn displaced(&self, atom: usize, axis: usize, delta: f64) -> Toy {
        let mut toy = self.clone();
        toy.tau[atom][axis] += delta;
        toy
    }

    /// Affine strain `x_a -> x_a + eps x_b` of the atoms and of the lattice vector.
    pub fn deformed(&self, a: usize, b: usize, eps: f64) -> Toy {
        let mut toy = self.clone();
        for x in toy.tau.iter_mut().chain(std::iter::once(&mut toy.a1)) {
            x[a] += eps * x[b];
        }
        toy
    }

    /// Row and column centers and owners, and the sign `s` of `d = s (row - col) + R`.
    fn frame(&self, kind: Kind) -> (Vec<[f64; 3]>, Vec<usize>, Vec<[f64; 3]>, Vec<usize>, f64) {
        let rows = (self.basis_centers(), self.basis_owner());
        if kind == Kind::Projection {
            (
                rows.0,
                rows.1,
                self.projector_centers(),
                self.projector_owner(),
                1.0,
            )
        } else {
            (rows.0.clone(), rows.1.clone(), rows.0, rows.1, -1.0)
        }
    }

    fn displacement(&self, kind: Kind, image: usize, i: usize, j: usize) -> [f64; 3] {
        let (rc, _, cc, _, s) = self.frame(kind);
        let r = self.images()[image];
        std::array::from_fn(|b| s * (rc[i][b] - cc[j][b]) + r[b])
    }

    pub fn blocks(&self, kind: Kind) -> Blocks {
        let (rows, _, cols, _, _) = self.frame(kind);
        let (m, n) = (rows.len(), cols.len());
        let (mut values, mut gradients) = (vec![], vec![]);
        for image in 0..self.images().len() {
            let mut v = DMatrix::zeros(m, n);
            let mut g = [
                DMatrix::zeros(m, n),
                DMatrix::zeros(m, n),
                DMatrix::zeros(m, n),
            ];
            for i in 0..m {
                for j in 0..n {
                    let d = self.displacement(kind, image, i, j);
                    let (coefficient, kappa) = pair(kind, i, j);
                    let value = coefficient * (-kappa * d.iter().map(|x| x * x).sum::<f64>()).exp();
                    v[(i, j)] = value;
                    for a in 0..3 {
                        g[a][(i, j)] = -2.0 * kappa * value * d[a];
                    }
                }
            }
            values.push(v);
            gradients.push(g);
        }
        Blocks { values, gradients }
    }

    fn phase(&self, kind: Kind, k: [f64; 3], image: usize) -> C64 {
        let s = self.frame(kind).4;
        let r = self.images()[image];
        let arg = -s * (0..3).map(|b| k[b] * r[b]).sum::<f64>();
        C64::new(arg.cos(), arg.sin())
    }

    fn sum(&self, kind: Kind, k: [f64; 3], cell: impl Fn(usize, usize, usize) -> f64) -> Matrix {
        let (rows, _, cols, _, _) = self.frame(kind);
        let mut out = Matrix::zeros(rows.len(), cols.len());
        for image in 0..self.images().len() {
            let phase = self.phase(kind, k, image);
            for i in 0..rows.len() {
                for j in 0..cols.len() {
                    out[(i, j)] += phase * cell(image, i, j);
                }
            }
        }
        out
    }

    /// `S(k)`, `H0(k)` (without the on-site level) or `C(k)`.
    pub fn bloch(&self, kind: Kind, k: [f64; 3]) -> Matrix {
        let b = self.blocks(kind);
        self.sum(kind, k, |image, i, j| b.values[image][(i, j)])
    }

    /// `d bloch / d tau_atom^axis`.
    pub fn bloch_atom(&self, kind: Kind, k: [f64; 3], atom: usize, axis: usize) -> Matrix {
        let (_, ro, _, co, s) = self.frame(kind);
        let b = self.blocks(kind);
        self.sum(kind, k, |image, i, j| {
            let own = f64::from(u8::from(ro[i] == atom)) - f64::from(u8::from(co[j] == atom));
            s * own * b.gradients[image][axis][(i, j)]
        })
    }

    /// `d bloch / d eps_ab` at fixed fractional k.
    pub fn bloch_strain(&self, kind: Kind, k: [f64; 3], a: usize, b: usize) -> Matrix {
        let blocks = self.blocks(kind);
        self.sum(kind, k, |image, i, j| {
            blocks.gradients[image][a][(i, j)] * self.displacement(kind, image, i, j)[b]
        })
    }

    /// The constant on-site part of `H0` for the spin `sign` (+1 or -1): levels and splitting.
    pub fn onsite(&self, sign: f64) -> Matrix {
        Matrix::from_fn(self.basis(), self.basis(), |i, j| {
            if i != j {
                return C64::new(0.0, 0.0);
            }
            let (atom, orbital) = (i / NB, i % NB);
            let level = (if orbital == 0 { -0.5 } else { 0.3 }) + 0.04 * atom as f64;
            let split = (if orbital == 0 { 1.0 } else { -0.7 }) * (1.0 + 0.2 * atom as f64);
            C64::new(level + sign * 0.06 * split, 0.0)
        })
    }
}
