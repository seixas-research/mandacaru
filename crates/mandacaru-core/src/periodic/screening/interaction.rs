//! The statically screened interaction of a crystal's whole mesh, full RPA (no target subspace):
//!
//! ```text
//! W_{GG'}(q) = v^1/2(q + G) eps^-1_{GG'}(q) v^1/2(q + G'),   eps = 1 - v^1/2 chi v^1/2
//! ```
//!
//! on the vectors `|q + G| <= cutoff` of every mesh vector `q`, each represented in the
//! Gamma-centered cell of the mesh (`q = B (i - N round(i / N)) / N`), with `chi` the static
//! polarizability of [`super::medium`] over the pairs `(k, k + q)` of every channel, every state
//! a frame holds. At `q = 0` the Coulomb interaction has no `G = 0` term: the head and wings of
//! that block are zero here, and the caller adds the long-wavelength head (its average over the
//! mesh's mini-zone, [`head_average`]); the body there is the inverse of the body alone. Every
//! diagonal element is `eps^-1_GG(q)` times the mean of `4 pi / |q + G|^2` over the mini-zone
//! cell around `q + G` ([`cell_average`]), so that a mesh sum integrates the Coulomb singularity
//! cell by cell; the off-diagonal elements keep the point values.
//! Atomic units: `W` in Hartree Bohr^3, before the `1 / (N_k Omega)` of a mesh sum.
//!
//! The Wannier downfold builds the same blocks for a fragment (`wannier::downfold::screened`,
//! with the constrained-RPA weights); this one needs no fragment. Both walk their mesh through
//! `solve`, which owns the loop and the Dyson solve.
use super::{
    bloch::Bloch,
    dyson,
    medium::{within, Medium},
    ScreenedInteraction,
};
use crate::{
    periodic::{bloch_mesh::MeshIndex, invalid},
    quantum::C64,
    Context, Result,
};
use nalgebra::{DMatrix, Matrix3, Vector3};
use std::f64::consts::PI;

/// `W(q)` on the vectors of one mesh vector.
pub struct Block {
    /// The mesh vector in integer mesh steps, folded to the Gamma-centered cell.
    pub steps: [i32; 3],
    /// `q` (Cartesian, Bohr^-1).
    pub q: [f64; 3],
    /// Integer reciprocal-lattice coordinates `G` of the vectors.
    pub g: Vec<[i64; 3]>,
    /// `q + G` (Cartesian, Bohr^-1).
    pub vectors: Vec<[f64; 3]>,
    /// `W_{GG'}(q)` (Hartree Bohr^3).
    pub screened: DMatrix<C64>,
}

/// The mesh steps of point `index` folded to the Gamma-centered cell (`-N/2 < i <= N/2`).
pub fn folded(mesh: &MeshIndex, index: usize) -> [i32; 3] {
    let at = mesh.indices(index);
    std::array::from_fn(|a| {
        let (i, n) = (at[a] as i64, mesh.size[a] as i64);
        (if 2 * i > n { i - n } else { i }) as i32
    })
}

/// The Cartesian vector of integer mesh steps `steps` (`reciprocal` the columns `B`).
pub fn vector(reciprocal: &Matrix3<f64>, mesh: &MeshIndex, steps: [i32; 3]) -> [f64; 3] {
    let fractional = Vector3::from_fn(|a, _| steps[a] as f64 / mesh.size[a] as f64);
    (reciprocal * fractional).into()
}

/// The polarizability of one mesh vector: `(chi, q + G, G)`.
pub(in crate::periodic) type Response = (DMatrix<C64>, Vec<[f64; 3]>, Vec<[i64; 3]>);

/// The Dyson solution of the mesh vectors `0..count` one after another, in mesh order: each
/// `polarizability(t)` runs on the medium's workers, and `finish(t, solution, q + G, G)` keeps
/// what the caller needs of it before the next vector starts (its sum does not depend on the
/// number of workers). Running the vectors on the context's lanes as well nests the pool inside
/// its own lanes, and that deadlocked under the Python calculator's automatic layout (every lane
/// waiting in `pool.install`).
pub(in crate::periodic) fn solve<T>(
    ctx: &Context,
    count: usize,
    polarizability: impl Fn(usize) -> Result<Response>,
    finish: impl Fn(usize, ScreenedInteraction, Vec<[f64; 3]>, Vec<[i64; 3]>) -> Result<T>,
) -> Result<Vec<T>> {
    (0..count)
        .map(|t| {
            let (chi, vectors, g) = polarizability(t)?;
            finish(t, dyson(ctx, &chi, &vectors)?, vectors, g)
        })
        .collect()
}

/// `W(q)` of every mesh vector, in mesh order, from `states[s][k]` (channel `s`, mesh point `k`
/// of `mesh`, Gamma-anchored), the cell `lattice` (columns, Bohr) and its `reciprocal`.
pub fn screened(
    medium: &Medium<'_>,
    states: &[Vec<Bloch<'_>>],
    mesh: &MeshIndex,
    lattice: &Matrix3<f64>,
    reciprocal: &Matrix3<f64>,
    cutoff: f64,
) -> Result<Vec<Block>> {
    if !(cutoff.is_finite() && cutoff > 0.) {
        return Err(invalid("the screening cutoff must be finite and positive"));
    }
    if mesh.origin.iter().any(|x| *x != 0.) {
        return Err(invalid(
            "the screened interaction needs a Gamma-anchored mesh",
        ));
    }
    let n = mesh.points();
    if states.is_empty() || states.iter().any(|c| c.len() != n) {
        return Err(invalid(
            "the screened interaction needs the states of every mesh point",
        ));
    }
    let response = |t: usize| -> Result<Response> {
        let steps = folded(mesh, t);
        let q = vector(reciprocal, mesh, steps);
        let g = within(lattice, q, cutoff);
        let mut pairs: Vec<(&Bloch<'_>, &Bloch<'_>)> = Vec::with_capacity(states.len() * n);
        for channel in states {
            for k in 0..n {
                pairs.push((&channel[k], &channel[mesh.shift(k, steps)]));
            }
        }
        let (chi, vectors) = medium.polarizability(q, &g, &pairs, n)?;
        Ok((chi, vectors, g))
    };
    let block = |t: usize,
                 solved: ScreenedInteraction,
                 vectors: Vec<[f64; 3]>,
                 g: Vec<[i64; 3]>|
     -> Result<Block> {
        let steps = folded(mesh, t);
        let q = vector(reciprocal, mesh, steps);
        let inverse = solved.inverse_dielectric;
        let root: Vec<f64> = vectors
            .iter()
            .map(|v| {
                let q2 = v.iter().map(|x| x * x).sum::<f64>();
                if q2 < 1e-14 {
                    0.
                } else {
                    (4. * PI / q2).sqrt()
                }
            })
            .collect();
        let mut screened =
            DMatrix::from_fn(g.len(), g.len(), |i, j| root[i] * inverse[(i, j)] * root[j]);
        // Each diagonal element stands for its mini-zone cell: eps^-1_GG(q) times the cell's mean
        // of 4 pi / |q + G|^2 (smooth away from q + G = 0; the q = 0, G = 0 head is the
        // caller's). Every vector of the block is averaged alike, so the block of a zone-boundary
        // q, which stands for -q too, keeps the symmetry of W.
        for (i, v) in g.iter().enumerate() {
            let centre: [i32; 3] =
                std::array::from_fn(|a| steps[a] + mesh.size[a] as i32 * v[a] as i32);
            if centre == [0; 3] {
                continue;
            }
            let average = cell_average(reciprocal, mesh, centre, &Matrix3::identity(), 8)?;
            screened[(i, i)] = inverse[(i, i)] * average;
        }
        Ok(Block {
            steps,
            q,
            g,
            vectors,
            screened,
        })
    };
    solve(medium.ctx, n, response, block)
}

/// The mean of `4 pi / (q . eps . q)` over the mesh's mini-zone, the parallelepiped
/// `sum_a x_a B_a / N_a`, `|x_a| <= 1/2`, around `q = 0`, with `eps` the macroscopic static
/// dielectric tensor (the identity for the bare interaction): the `G = G' = 0` element the
/// `q = 0` term of a mesh sum stands for. [`cell_average`] at the origin.
pub fn head_average(
    reciprocal: &Matrix3<f64>,
    mesh: &MeshIndex,
    dielectric: &Matrix3<f64>,
    order: usize,
) -> Result<f64> {
    cell_average(reciprocal, mesh, [0; 3], dielectric, order)
}

/// The mean of `4 pi / (q . eps . q)` over the mini-zone cell of the mesh vector `steps` (the
/// parallelepiped of [`head_average`] moved to `q = B steps / N`): the `G = G' = 0` element a
/// mesh sum's term at `q` stands for, so that the sum integrates the `1 / q^2` singularity cell
/// by cell instead of sampling it.
///
/// Away from the origin the integrand is smooth on the cell (the nearest singularity lies half a
/// cell outside it) and a tensor Gauss-Legendre rule of `order` points per direction integrates
/// it. At the origin it is homogeneous of degree `-2`, so the integral over the zone scaled by
/// `1/3` is a third of the integral over the zone: `I = S / (1 - 1/3)` with `S` the integral over
/// the 26 outer cells of the zone's `3 x 3 x 3` division, each by the same rule.
pub fn cell_average(
    reciprocal: &Matrix3<f64>,
    mesh: &MeshIndex,
    steps: [i32; 3],
    dielectric: &Matrix3<f64>,
    order: usize,
) -> Result<f64> {
    let eps = 0.5 * (dielectric + dielectric.transpose());
    if eps.iter().any(|v| !v.is_finite()) || eps.symmetric_eigenvalues().min() <= 0. {
        return Err(invalid(
            "the head average needs a finite positive-definite dielectric tensor",
        ));
    }
    let nodes = crate::math::radial::gauss_legendre(order)?;
    let edges = Matrix3::from_fn(|i, a| reciprocal[(i, a)] / mesh.size[a] as f64);
    let centre = Vector3::from_fn(|a, _| steps[a] as f64);
    // The mean over the fractional box `[low, low + width]^3` (cell units around `centre`).
    let mean = |low: [f64; 3], width: f64| -> f64 {
        let mut sum = 0.;
        for &(x0, w0) in &nodes {
            for &(x1, w1) in &nodes {
                for &(x2, w2) in &nodes {
                    let x = [x0, x1, x2];
                    let f = Vector3::from_fn(|a, _| low[a] + width * (x[a] + 1.) / 2.);
                    let q = edges * (centre + f);
                    sum += w0 * w1 * w2 / 8. * 4. * PI / (q.transpose() * eps * q)[0];
                }
            }
        }
        sum
    };
    if steps != [0; 3] {
        return Ok(mean([-0.5; 3], 1.));
    }
    let mut outer = 0.;
    for cell in 0..27usize {
        let at = [cell / 9, (cell / 3) % 3, cell % 3];
        if at == [1, 1, 1] {
            continue;
        }
        let low = at.map(|i| (i as f64 - 1.5) / 3.);
        outer += mean(low, 1. / 3.) / 27.;
    }
    Ok(outer / (1. - 1. / 3.))
}
