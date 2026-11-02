//! Shared by the `setup_pulay*` tests: a frozen WP-X case (Si, Al2 or Al, SZ, `lda-sr`) with the
//! grid of its recorded shape, an atom-by-atom way of moving the basis, the projectors and the
//! short-range spheres, and the central difference that every gate of CF-2 is.
#![allow(dead_code)]

use crate::{setup_gradients_support::*, setup_shells_support::projector_list};
use mandacaru_core::{
    math::SpatialGrid,
    periodic::setup::{
        basis_gradient::{short_range_cross, ShortRange},
        ions::LocalPotential,
        pulay::{Matrix, Mesh},
        shells::{Shell, Shells},
        spheres::Projector,
    },
    Context,
};
use nalgebra::Matrix3;
use num_complex::Complex64;

/// An irrational k-point (Cartesian, Bohr^-1) inside the first zone of every fixture.
pub const KPOINT: [f64; 3] = [0.173205, -0.141421, 0.0866025];

pub struct Fixture {
    pub case: Case,
    pub lattice: Matrix3<f64>,
    pub grid: SpatialGrid,
    pub shells: Shells,
    /// The atom each basis function sits on.
    pub owner: Vec<usize>,
    pub centers: Vec<[f64; 3]>,
    pub potentials: Vec<LocalPotential>,
    pub projectors: Vec<Projector>,
    pub sigma: f64,
}

impl Fixture {
    pub fn new(ctx: &Context, name: &'static str) -> Self {
        Self::with_shape(ctx, name, None)
    }
    /// The case on a grid of `shape` nodes (the recorded shape when `None`).
    pub fn with_shape(ctx: &Context, name: &'static str, shape: Option<[usize; 3]>) -> Self {
        let case = Case::load(name);
        let cell = case.cell();
        let recorded: Vec<usize> = case.inputs()["grid_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let shape = shape.unwrap_or([recorded[0], recorded[1], recorded[2]]);
        // Step vectors are the columns of the lattice divided by the node counts.
        let mut step = [0.; 9];
        for row in 0..3 {
            for col in 0..3 {
                step[row * 3 + col] = cell.lattice[(row, col)] / shape[col] as f64;
            }
        }
        let grid = SpatialGrid::new(shape, [0.; 3], step, true).unwrap();
        let (shells, list) = case.basis(ctx);
        let centers: Vec<[f64; 3]> = cell.atoms.iter().map(|a| a.center).collect();
        let mut projectors = Vec::new();
        for (atom, center) in centers.iter().enumerate() {
            let ds = case.dataset(ctx, atom);
            for mut p in projector_list(ctx, &ds, &[*center], true) {
                p.atom = atom;
                projectors.push(p);
            }
        }
        Self {
            lattice: cell.lattice,
            sigma: cell.sigma,
            owner: list.iter().map(|f| f.0).collect(),
            potentials: case.potentials(ctx),
            grid,
            shells,
            centers,
            projectors,
            case,
        }
    }
    pub fn mesh(&self, ctx: &Context) -> Mesh {
        Mesh::new(ctx, &self.grid).unwrap()
    }
    pub fn atoms(&self) -> usize {
        self.centers.len()
    }
    pub fn functions(&self) -> usize {
        self.shells.functions()
    }
    /// The basis with every shell of `atom` moved by `delta` along `axis`.
    pub fn moved_basis(&self, atom: usize, axis: usize, delta: f64) -> Shells {
        let shells = self
            .shells
            .shells()
            .iter()
            .map(|s| {
                let mut s: Shell = s.clone();
                if self.owner[s.rows[0]] == atom {
                    s.center[axis] += delta;
                }
                s
            })
            .collect();
        Shells::new(shells).unwrap()
    }
    /// The projectors with those of `atom` (and so their spheres) moved.
    pub fn moved_projectors(&self, atom: usize, axis: usize, delta: f64) -> Vec<Projector> {
        self.projectors
            .iter()
            .map(|p| {
                let mut p = p.clone();
                if p.atom == atom {
                    p.center[axis] += delta;
                }
                p
            })
            .collect()
    }
    /// Rows of the basis that belong to `atom` (the diagonal of `P_X`).
    pub fn rows_of(&self, atom: usize) -> Vec<bool> {
        self.owner.iter().map(|&o| o == atom).collect()
    }
}

/// `P G + G^+ P` for the rows selected by `own`.
pub fn ownership(g: &Matrix, own: &[bool]) -> Matrix {
    let mut p = g.clone();
    for (mu, keep) in own.iter().enumerate() {
        if !keep {
            p.row_mut(mu).fill(Complex64::new(0., 0.));
        }
    }
    &p + p.adjoint()
}

/// The Richardson-extrapolated central difference `d f / d delta` at 0 from `f(+-h)` and
/// `f(+-2h)` (error `O(h^4)`); `f` returns a matrix.
pub fn central(h: f64, f: &mut dyn FnMut(f64) -> Matrix) -> Matrix {
    let one = (f(h) - f(-h)) / Complex64::new(2. * h, 0.);
    let two = (f(2. * h) - f(-2. * h)) / Complex64::new(4. * h, 0.);
    (one * Complex64::new(4., 0.) - two) / Complex64::new(3., 0.)
}

/// Largest absolute entry.
pub fn peak(m: &Matrix) -> f64 {
    m.iter().map(|x| x.norm()).fold(0., f64::max)
}

/// Largest entry of `a - b` over the largest entry of `b`.
pub fn gap(a: &Matrix, b: &Matrix) -> f64 {
    peak(&(a - b)) / peak(b).max(1e-300)
}

/// The Bloch sums of `shells` at `k` on the mesh, `(function, node)`, from the kernel alone.
pub fn bloch_values(
    ctx: &Context,
    fx: &Fixture,
    shells: &Shells,
    mesh: &Mesh,
    k: [f64; 3],
) -> Vec<Complex64> {
    use mandacaru_core::periodic::setup::shells::{DenseRows, Sampling};
    let (m, n) = (shells.functions(), mesh.len());
    let mut out = vec![Complex64::new(0., 0.); m * n];
    shells
        .values(
            ctx,
            &Sampling {
                lattice: &fx.lattice,
                region: mesh.region(),
                kpoints: &[k],
            },
            mesh.points(),
            &mut DenseRows::new(&mut out, m, n),
        )
        .unwrap();
    out
}

/// `weight * sum_x conj(a[mu, x]) w_x b[nu, x]` for row-major `(function, node)` arrays.
pub fn form(a: &[Complex64], b: &[Complex64], n: usize, weight: f64, w: Option<&[f64]>) -> Matrix {
    let (ma, mb) = (a.len() / n, b.len() / n);
    Matrix::from_fn(ma, mb, |i, j| {
        (0..n)
            .map(|x| a[i * n + x].conj() * b[j * n + x] * w.map_or(1., |w| w[x]))
            .sum::<Complex64>()
            * weight
    })
}

/// A rough, non-symmetric potential on the mesh (any weights are a legitimate local term).
pub fn rough_potential(n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| 0.4 + (0.013 * j as f64).sin() + 0.5 * (0.0071 * j as f64).cos())
        .collect()
}

/// The short-range matrix of every atom's sphere summed, with the atoms in `centers` and the
/// functions in `shells`, at `k`.
pub fn short_range_total(
    fx: &Fixture,
    ctx: &Context,
    shells: &Shells,
    centers: &[[f64; 3]],
    k: [f64; 3],
) -> Matrix {
    let m = fx.functions();
    let mut total = Matrix::zeros(m, m);
    for (atom, center) in centers.iter().enumerate() {
        let sphere = ShortRange {
            center: *center,
            potential: &fx.potentials[atom],
            sigma: fx.sigma,
        };
        let block =
            short_range_cross(ctx, &fx.lattice, &[k], &sphere, shells, None, shells).unwrap();
        total += Matrix::from_row_slice(m, m, block.as_slice());
    }
    total
}

/// A deterministic Hermitian matrix with no symmetry of its own.
pub fn density(m: usize) -> Matrix {
    let a = Matrix::from_fn(m, m, |i, j| {
        Complex64::new(
            ((3 * i + 5 * j + 1) as f64).sin(),
            ((7 * i + 2 * j + 3) as f64).cos() * 0.5,
        )
    });
    (&a + a.adjoint()) * Complex64::new(0.5, 0.)
}

/// `Re tr[D M]` with `tr[D M] = sum_{mu nu} D_{nu mu} M_{mu nu}`.
pub fn trace(d: &Matrix, m: &Matrix) -> f64 {
    (d * m).trace().re
}
