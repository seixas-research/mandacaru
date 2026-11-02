//! The +U oracle of `strain_spheres_hubbard.rs`: a real geometry (a cell, its basis and its
//! projectors restricted to the `m = 0` components, whose images blocks are REAL), synthetic
//! Hubbard data (shell weights, Hermitian shell potentials, Hermitian densities), the image
//! blocks `G^a_R = dB_R/dd^a` from the per-image projection kernel (the test supplies them, as
//! `hubbard_gradient::Input.gradients` takes them), and the contraction of the library's `dC_ab`
//! with the same operator.
#![allow(dead_code)]

use super::{columns, groups};
use mandacaru_core::{
    kernels::{bloch_project, ShellImages},
    periodic::{
        hubbard_gradient::{operator, Channel, Geometry, Input, Shell},
        setup::{
            basis_gradient::subset,
            images::{Images, Region},
            pulay::Matrix,
            shells::Shells,
            spheres::{chunk_size, Projector, Sphere},
        },
    },
    quantum::C64,
    Context,
};
use nalgebra::{DMatrix, Matrix3};
use num_complex::Complex64;
use std::collections::BTreeMap;

/// Orbitals of every Hubbard shell.
const ORBITALS: usize = 2;

/// The restricted real geometry and the data the layer takes.
pub struct Toy {
    pub lattice: Matrix3<f64>,
    pub shells: Shells,
    pub projectors: Vec<Projector>,
    pub atoms: usize,
    pub basis_centers: Vec<[f64; 3]>,
    pub basis_owner: Vec<usize>,
    pub projector_centers: Vec<[f64; 3]>,
    pub projector_owner: Vec<usize>,
    pub images: Vec<[f64; 3]>,
    /// `gradients[image][axis]`, `m x p`.
    pub gradients: Vec<[DMatrix<f64>; 3]>,
    pub hubbard: Vec<Shell>,
    pub potentials: Vec<Matrix>,
    pub kpoints: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub density: Vec<Matrix>,
    /// The largest imaginary part met in an image block (real by construction for `m = 0`).
    pub imaginary: f64,
}

fn hermitian(entries: [f64; 3]) -> Matrix {
    Matrix::from_row_slice(
        ORBITALS,
        ORBITALS,
        &[
            C64::new(entries[0], 0.),
            C64::new(entries[1], -0.4 * entries[1]),
            C64::new(entries[1], 0.4 * entries[1]),
            C64::new(entries[2], 0.),
        ],
    )
}

/// `shells` and `owner` (the atom of each basis function) of the `m = 0` components, the `m = 0`
/// projectors, their image blocks and the synthetic Hubbard data on the k-points `kpoints`.
pub fn build(
    ctx: &Context,
    lattice: Matrix3<f64>,
    shells: &Shells,
    owner: &[usize],
    projectors: &[Projector],
    atoms: usize,
    kpoints: Vec<[f64; 3]>,
    density: impl Fn(usize) -> Matrix,
) -> Toy {
    let mut kept_rows = Vec::new();
    for shell in shells.shells() {
        for (&m, &row) in shell.ms.iter().zip(&shell.rows) {
            if m == 0 {
                kept_rows.push(row);
            }
        }
    }
    kept_rows.sort_unstable();
    let reduced = subset(shells, &kept_rows).unwrap();
    let kept: Vec<Projector> = projectors.iter().filter(|p| p.m == 0).cloned().collect();
    let (m, p) = (reduced.functions(), kept.len());
    // Centers of the reduced functions, by their new row.
    let mut basis_centers = vec![[0.; 3]; m];
    for shell in reduced.shells() {
        for &row in &shell.rows {
            basis_centers[row] = shell.center;
        }
    }
    let basis_owner: Vec<usize> = kept_rows.iter().map(|&r| owner[r]).collect();
    // The image blocks, one entry per distinct lattice vector met by a (shell, sphere) pair.
    let key = |t: &[f64; 3]| t.map(|x| (x * 1e8).round() as i64);
    let mut slot: BTreeMap<[i64; 3], usize> = BTreeMap::new();
    let mut images: Vec<[f64; 3]> = Vec::new();
    let mut gradients: Vec<[DMatrix<f64>; 3]> = Vec::new();
    let mut imaginary = 0f64;
    for members in groups(&kept) {
        let first = &kept[members[0]];
        let region = Region {
            center: first.center,
            radius: first.r_cut,
        };
        let reached = reduced.reaching(ctx, &lattice, &region).unwrap();
        if reached.rows.is_empty() {
            continue;
        }
        let sphere = Sphere::projector(ctx, first.center, first.r_cut).unwrap();
        for range in sphere.chunks(chunk_size(1, 4 * reached.rows.len())) {
            let points = sphere.points(range.clone());
            let cols = columns(ctx, &kept, &members, points, sphere.weights(range));
            for shell in reached.shells.shells() {
                let around =
                    Images::around(ctx, &lattice, &region, shell.center, shell.radial.support())
                        .unwrap();
                if around.is_empty() {
                    continue;
                }
                let spline = shell.radial.spline(shell.l).unwrap();
                let nm = shell.ms.len();
                for axis in 0..3 {
                    let blocks = bloch_project(
                        ctx,
                        &ShellImages {
                            xyz: points,
                            center: shell.center,
                            images: around.vectors(),
                            phases: &vec![Complex64::new(1., 0.); around.len()],
                            support: shell.radial.support(),
                            spline: &spline,
                            ms: &shell.ms,
                            derivative: axis as i32,
                        },
                        &cols,
                    )
                    .unwrap();
                    for (r, t) in around.vectors().iter().enumerate() {
                        let at = *slot.entry(key(t)).or_insert_with(|| {
                            images.push(*t);
                            gradients.push(std::array::from_fn(|_| DMatrix::zeros(m, p)));
                            images.len() - 1
                        });
                        for (i, &row) in shell.rows.iter().enumerate() {
                            for (j, &q) in members.iter().enumerate() {
                                let v = blocks.as_slice()[(r * nm + i) * members.len() + j];
                                imaginary = imaginary.max(v.im.abs());
                                // The center derivative of the overlap with the image: minus
                                // `<d_a chi_T | w p>`.
                                gradients[at][axis][(reached.rows[row], q)] -= v.re;
                            }
                        }
                    }
                }
            }
        }
    }
    // One Hubbard shell per atom that carries projectors.
    let mut hubbard = Vec::new();
    let mut potentials = Vec::new();
    for atom in 0..atoms {
        let list: Vec<usize> = (0..p).filter(|&q| kept[q].atom == atom).collect();
        if list.is_empty() {
            continue;
        }
        let np = list.len();
        let weights = DMatrix::from_fn(np, ORBITALS, |i, a| {
            0.6 / (1. + i as f64) + 0.25 * ((i + 2 * a + atom) as f64).sin()
        });
        hubbard.push(Shell {
            projectors: list,
            weights,
        });
        let s = 1. + 0.3 * atom as f64;
        potentials.push(hermitian([0.31 * s, 0.12, -0.2 / s]));
    }
    let weights: Vec<f64> = (0..kpoints.len())
        .map(|i| (1. + i as f64) / (kpoints.len() * (kpoints.len() + 1) / 2) as f64)
        .collect();
    let density: Vec<Matrix> = (0..kpoints.len()).map(density).collect();
    Toy {
        lattice,
        shells: reduced,
        projector_centers: kept.iter().map(|q| q.center).collect(),
        projector_owner: kept.iter().map(|q| q.atom).collect(),
        projectors: kept,
        atoms,
        basis_centers,
        basis_owner,
        images,
        gradients,
        hubbard,
        potentials,
        kpoints,
        weights,
        density,
        imaginary,
    }
}

impl Toy {
    /// `dE_U/d eps_ab` of the layer (`Gradient::strain`) with the displacement map `d_R` given.
    pub fn layer(
        &self,
        projections: &[Matrix],
        displacement: fn(&Geometry, usize, usize, usize) -> [f64; 3],
    ) -> [[f64; 3]; 3] {
        let channels = [Channel {
            potentials: &self.potentials,
            density: &self.density,
        }];
        let input = Input {
            geometry: Geometry {
                atoms: self.atoms,
                images: &self.images,
                basis_centers: &self.basis_centers,
                projector_centers: &self.projector_centers,
                basis_owner: &self.basis_owner,
                projector_owner: &self.projector_owner,
            },
            shells: &self.hubbard,
            weights: &self.weights,
            kpoints: &self.kpoints,
            projections,
            gradients: &self.gradients,
            channels: &channels,
        };
        mandacaru_core::periodic::hubbard_gradient::gradient_with(&input, displacement)
            .unwrap()
            .strain
    }
    /// The same energy derivative from a `dC_ab` supplier: `sum_k w 2 Re sum dC[mu, p] Y[p, mu]`
    /// with `Y = O C^+ D`, `d_c[k][a][b]` the `m x p` strain derivative of `C_k`.
    pub fn contract(&self, projections: &[Matrix], d_c: &[[[Matrix; 3]; 3]]) -> [[f64; 3]; 3] {
        let o = operator(self.projectors.len(), &self.hubbard, &self.potentials).unwrap();
        let mut out = [[0.; 3]; 3];
        for (k, w) in self.weights.iter().enumerate() {
            let y = &(&o * projections[k].adjoint()) * &self.density[k];
            for a in 0..3 {
                for b in 0..3 {
                    let tangent = &d_c[k][a][b];
                    let mut sum = 0.;
                    for mu in 0..tangent.nrows() {
                        for q in 0..tangent.ncols() {
                            sum += (tangent[(mu, q)] * y[(q, mu)]).re;
                        }
                    }
                    out[a][b] += 2. * w * sum;
                }
            }
        }
        out
    }
}

/// `d_R` with the lattice vector dropped: `c_mu - c_p` (right at Gamma-like contacts, wrong
/// across cells).
pub fn without_image(g: &Geometry, _image: usize, mu: usize, p: usize) -> [f64; 3] {
    std::array::from_fn(|b| g.basis_centers[mu][b] - g.projector_centers[p][b])
}
