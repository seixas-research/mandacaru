//! The Si `lda-fr` SZ cell of the spin-orbit derivative tests (`setup_spin_orbit_gradient.rs`,
//! `periodic_gradient_spin_orbit_silicon.rs`): the basis, the spin-orbit union projectors, the
//! block `h^SO(k)` with the basis and/or the projectors of one atom displaced, its Richardson
//! central difference, and the analytic `matrix_derivative`.
#![allow(dead_code)]

use crate::setup_shells_support::*;
use mandacaru_core::{
    periodic::setup::{
        pulay::Matrix,
        shells::{Shell, Shells},
        spheres::Projector,
        spin_orbit::{matrices, matrix_derivative, Block},
    },
    Context,
};
use nalgebra::Matrix3;
use num_complex::Complex64;
use std::collections::BTreeMap;

pub const STEP: f64 = 2e-4;
pub const KS: [[f64; 3]; 2] = [[0.173205, -0.141421, 0.0866025], [-0.31, 0.05, 0.22]];
/// `(2M)^2` entries of one k-point's block (`M = 8` functions).
pub const SIZE: usize = 4 * 64;

pub struct Silicon {
    pub ctx: Context,
    pub lattice: Matrix3<f64>,
    pub shells: Shells,
    pub row_owner: Vec<usize>,
    pub projectors: Vec<Projector>,
    pub couplings: BTreeMap<(usize, usize), mandacaru_core::models::spin_orbit::Coupling>,
}

pub fn silicon() -> Silicon {
    let ctx = context();
    let cell = Cell::read("si_sz_lda_fr");
    let ds = dataset(&ctx, "lda-fr", "Si");
    let basis = minimal_basis(&ctx, &ds, &cell.centers, cell.filter_wavevector);
    let (projectors, couplings) = spin_orbit_list(&ctx, &ds, &cell.centers);
    let row_owner: Vec<usize> = cell.orbitals.iter().map(|o| o.0).collect();
    assert_eq!(row_owner.len(), basis.shells.functions());
    Silicon {
        ctx,
        lattice: cell.lattice,
        shells: basis.shells,
        row_owner,
        projectors,
        couplings,
    }
}

impl Silicon {
    pub fn atoms(&self) -> usize {
        self.projectors.iter().map(|p| p.atom).max().unwrap() + 1
    }

    pub fn block_of(&self) -> Block<'_> {
        Block {
            shells: &self.shells,
            lattice: &self.lattice,
            row_owner: &self.row_owner,
            projectors: &self.projectors,
            couplings: &self.couplings,
        }
    }

    /// The basis with the functions of `atom` (every function when `None`) moved by `delta`.
    pub fn moved_basis(&self, atom: Option<usize>, axis: usize, delta: f64) -> Shells {
        let shells = self
            .shells
            .shells()
            .iter()
            .map(|s| {
                let mut s: Shell = s.clone();
                if atom.is_none_or(|a| self.row_owner[s.rows[0]] == a) {
                    s.center[axis] += delta;
                }
                s
            })
            .collect();
        Shells::new(shells).unwrap()
    }

    pub fn moved_projectors(&self, atom: Option<usize>, axis: usize, delta: f64) -> Vec<Projector> {
        self.projectors
            .iter()
            .map(|p| {
                let mut p = p.clone();
                if atom.is_none_or(|a| p.atom == a) {
                    p.center[axis] += delta;
                }
                p
            })
            .collect()
    }

    /// `h^SO(k)` for the basis and/or the projectors of `atom` displaced, `(k, 2M, 2M)`.
    pub fn block(
        &self,
        atom: Option<usize>,
        axis: usize,
        delta: f64,
        basis: bool,
        projectors: bool,
    ) -> Vec<Complex64> {
        let shells = if basis {
            self.moved_basis(atom, axis, delta)
        } else {
            self.shells.clone()
        };
        let list = if projectors {
            self.moved_projectors(atom, axis, delta)
        } else {
            self.projectors.clone()
        };
        matrices(
            &self.ctx,
            &shells,
            &self.lattice,
            &KS,
            &list,
            &self.couplings,
        )
        .unwrap()
        .as_slice()
        .to_vec()
    }

    /// The Richardson central difference of the block along the move; also its `c(h) - c(2h)`.
    pub fn difference(
        &self,
        atom: usize,
        axis: usize,
        basis: bool,
        projectors: bool,
    ) -> (Vec<Complex64>, f64) {
        let central = |step: f64| -> Vec<Complex64> {
            let (up, down) = (
                self.block(Some(atom), axis, step, basis, projectors),
                self.block(Some(atom), axis, -step, basis, projectors),
            );
            up.iter()
                .zip(&down)
                .map(|(a, b)| (a - b) / (2. * step))
                .collect()
        };
        let (one, two) = (central(STEP), central(2. * STEP));
        let spread = one
            .iter()
            .zip(&two)
            .map(|(a, b)| (a - b).norm())
            .fold(0., f64::max);
        let rich = one
            .iter()
            .zip(&two)
            .map(|(a, b)| (a * 4. - b) / 3.)
            .collect();
        (rich, spread)
    }

    pub fn analytic(&self, k: usize, atom: usize, axis: usize) -> Vec<Complex64> {
        matrix_derivative(&self.ctx, &self.block_of(), KS[k], atom, axis)
            .unwrap()
            .as_slice()
            .to_vec()
    }
}

pub fn gap(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0., f64::max)
}

pub fn peak(a: &[Complex64]) -> f64 {
    a.iter().map(|x| x.norm()).fold(0., f64::max)
}

/// `sum_ij D_ij h_ji` real part.
pub fn trace(d: &Matrix, h: &[Complex64]) -> f64 {
    let n = d.nrows();
    (0..n * n)
        .map(|x| d[(x / n, x % n)] * h[(x % n) * n + x / n])
        .sum::<Complex64>()
        .re
}

/// A fixed Hermitian matrix of size `n` (a congruential generator).
pub fn hermitian(n: usize) -> Matrix {
    let mut state = 0x2545F4914F6CDD1Du64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    let raw = Matrix::from_fn(n, n, |_, _| Complex64::new(next(), next()));
    (&raw + raw.adjoint()) * Complex64::new(0.5, 0.)
}
