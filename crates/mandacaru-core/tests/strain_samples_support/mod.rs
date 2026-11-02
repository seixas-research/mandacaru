//! Shared by the `strain_*` tests: a frozen WP-X case carried by a strain `x -> (1 + eps) x`. The
//! atoms, the basis and projector centers, the lattice and the grid nodes scale with the cell; the
//! radial tables, the cutoff radii and the sphere quadratures do not, and the Cartesian k-point
//! follows the fractional one (`k' = (1 + eps)^{-T} k`, so `k.T` is invariant). This is the
//! discretized energy's own strain, the one the analytic tangent differentiates.
#![allow(dead_code)]

use crate::setup_pulay_support::*;
use mandacaru_core::{
    math::SpatialGrid,
    periodic::setup::{
        pulay::{Matrix, Mesh},
        shells::{DenseRows, Sampling, Shell, Shells},
        spheres::Projector,
    },
    Context,
};
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64;

/// The strain with the single entry `eps_ab = e` (the entries of a strain are independent).
pub fn entry(a: usize, b: usize, e: f64) -> Matrix3<f64> {
    let mut m = Matrix3::zeros();
    m[(a, b)] = e;
    m
}

/// A basis, its projectors, the lattice and the k-point carried by `eps`.
pub struct Carried {
    pub map: Matrix3<f64>,
    pub lattice: Matrix3<f64>,
    pub shells: Shells,
    pub projectors: Vec<Projector>,
    pub k: [f64; 3],
}

impl Carried {
    pub fn new(
        lattice: &Matrix3<f64>,
        shells: &Shells,
        projectors: &[Projector],
        eps: &Matrix3<f64>,
        k: [f64; 3],
    ) -> Self {
        let map = Matrix3::identity() + eps;
        let at = |c: [f64; 3]| -> [f64; 3] { (map * Vector3::from(c)).into() };
        let moved = shells
            .shells()
            .iter()
            .map(|s| {
                let mut s: Shell = s.clone();
                s.center = at(s.center);
                s
            })
            .collect();
        let projectors = projectors
            .iter()
            .map(|p| {
                let mut p = p.clone();
                p.center = at(p.center);
                p
            })
            .collect();
        Self {
            map,
            lattice: map * lattice,
            shells: Shells::new(moved).unwrap(),
            projectors,
            k: (map.transpose().try_inverse().unwrap() * Vector3::from(k)).into(),
        }
    }
}

/// A fixture carried by `eps`, with its grid.
pub struct Strained {
    pub carried: Carried,
    pub grid: SpatialGrid,
}

impl Strained {
    pub fn new(fx: &Fixture, eps: &Matrix3<f64>, k: [f64; 3]) -> Self {
        let carried = Carried::new(&fx.lattice, &fx.shells, &fx.projectors, eps, k);
        let step = carried.map * fx.grid.step;
        let mut rows = [0.; 9];
        for (i, v) in rows.iter_mut().enumerate() {
            *v = step[(i / 3, i % 3)];
        }
        let grid = SpatialGrid::new(
            fx.grid.index.shape(),
            (carried.map * Vector3::from(fx.grid.origin)).into(),
            rows,
            fx.grid.periodic,
        )
        .unwrap();
        Self { carried, grid }
    }
    pub fn mesh(&self, ctx: &Context) -> Mesh {
        Mesh::new(ctx, &self.grid).unwrap()
    }
    /// The Bloch sums on the strained mesh at the carried k-point, `(function, node)`.
    pub fn bloch_values(&self, ctx: &Context) -> Matrix {
        let mesh = self.mesh(ctx);
        let (m, n) = (self.carried.shells.functions(), mesh.len());
        let mut out = vec![Complex64::new(0., 0.); m * n];
        self.carried
            .shells
            .values(
                ctx,
                &Sampling {
                    lattice: &self.carried.lattice,
                    region: mesh.region(),
                    kpoints: &[self.carried.k],
                },
                mesh.points(),
                &mut DenseRows::new(&mut out, m, n),
            )
            .unwrap();
        Matrix::from_row_slice(m, n, &out)
    }
}
