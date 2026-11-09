//! Strain derivatives of the grid forms of the basis: the overlap, the local potential and the
//! alias-averaged kinetic matrix at one k-point.
//!
//! Convention (see [`super::strain_samples`]): `x -> (1 + eps) x` carries the lattice, the atoms,
//! the grid nodes and the image centers, fractional `k` is held (so every phase `k.T`, `k.r` is
//! invariant), the node count `N` is fixed, so `dv -> dv (1 + tr eps)` and `Omega -> Omega (1 +
//! tr eps)`. With `A_ab = d chi / d eps_ab` ([`StrainSamples`]):
//!
//! * grid form `F = dv sum_i conj(chi_mu) w_i chi_nu`:
//!   `dF_ab = delta_ab F (the dv of the form) + G + G^+`, `G = dv sum_i conj(A_ab,mu) w_i chi_nu`;
//!   an energy that carries its own `dv` (the density of a fixed potential) uses `jacobian = false`;
//! * kinetic `T = sum_G conj(u_mu) (s/2 Omega) u_nu`, `u = dv FFT[chi e^{-ik.r}]`, `s` the alias
//!   mean `|k + G|^2` (`reciprocal::bloch`): `dT_ab = delta_ab T + sum_G conj(u) (-Q_a Q_b/Omega) u
//!   + G^T + G^T+`, with `Q_a Q_b` the alias MEAN of the product (`reciprocal::bloch_outer`, not the
//!   product of the mean vectors) and `G^T` the same form with the bra replaced by the periodic part of
//!   `A_ab`. The prefactor `dv^2 / Omega` gives the net Jacobian `+delta_ab`.
//!
//! Every function returns `[a][b]`, the matrix `d M / d eps_ab` for the nine ordered pairs.
use super::{
    pulay::{kinetic_weights, periodic_parts, Matrix, Mesh},
    shells::{DenseRows, Sampling, Shells},
    strain_grid::VOIGT,
    strain_samples::StrainSamples,
};
use crate::{
    math::{fft::Fft3, reciprocal, SpatialGrid},
    periodic::invalid,
    product, Buffer, Context, Result,
};
use nalgebra::{DMatrixView, Matrix3};
use num_complex::Complex64;

pub type Tensor = [[Matrix; 3]; 3];

/// The Bloch sums of every function of `shells` at `k` on the mesh, `(function, node)` row-major.
pub fn bloch_values(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    mesh: &Mesh,
    k: [f64; 3],
) -> Result<Buffer<Complex64>> {
    let (m, n) = (shells.functions(), mesh.len());
    let mut psi = ctx.allocate::<Complex64>(product(&[m, n])?)?;
    shells.values(
        ctx,
        &Sampling {
            lattice,
            region: mesh.region(),
            kpoints: &[k],
        },
        mesh.points(),
        &mut DenseRows::new(psi.as_mut_slice(), m, n),
    )?;
    Ok(psi)
}

/// The `(points, count)` column-major view of a `(count, points)` row-major array.
fn columns(rows: &[Complex64], points: usize, count: usize) -> DMatrixView<'_, Complex64> {
    DMatrixView::from_slice(rows, points, count)
}

fn zeros(m: usize) -> Tensor {
    std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(m, m)))
}

fn delta(a: usize, b: usize) -> f64 {
    if a == b {
        1.
    } else {
        0.
    }
}

/// `d/d eps_ab` of `dv sum_i conj(chi_mu) w_i chi_nu` (`w = 1` without a potential): the tangent
/// terms `G + G^+`, plus `delta_ab F` when `jacobian` is set. One tangent at a time, each
/// reserved from the `Context` until the next is formed.
pub fn grid_strain(
    ctx: &Context,
    mesh: &Mesh,
    strain: &StrainSamples,
    values: &[Complex64],
    potential: Option<&[f64]>,
    jacobian: bool,
) -> Result<Tensor> {
    let (m, n) = (strain.functions(), mesh.len());
    if values.len() != m * n || potential.is_some_and(|v| v.len() != n) {
        return Err(invalid(
            "a grid form needs the Bloch sums and one potential value per node",
        ));
    }
    let _held = ctx.reserve(product(&[m, n, 16])?)?;
    let ket = Matrix::from_fn(n, m, |x, nu| {
        values[nu * n + x] * (mesh.dv * potential.map_or(1., |v| v[x]))
    });
    let base = jacobian.then(|| columns(values, n, m).adjoint() * &ket);
    let mut out = zeros(m);
    for (a, row) in out.iter_mut().enumerate() {
        for (b, to) in row.iter_mut().enumerate() {
            let tangent = strain.tangent(ctx, a, b)?;
            let g = columns(tangent.as_slice(), n, m).adjoint() * &ket;
            *to = &g + g.adjoint();
            if let Some(f) = &base {
                *to += f * Complex64::new(delta(a, b), 0.);
            }
        }
    }
    Ok(out)
}

/// `d T / d eps_ab` of [`super::pulay::kinetic_form`] with the basis on both sides at `k` (the
/// Cartesian wave vector of the reference cell): the Jacobian, the symbol and the tangent terms.
pub fn kinetic_strain(
    ctx: &Context,
    grid: &SpatialGrid,
    mesh: &Mesh,
    strain: &StrainSamples,
    values: &[Complex64],
    k: [f64; 3],
) -> Result<Tensor> {
    let (m, n) = (strain.functions(), mesh.len());
    if values.len() != m * n {
        return Err(invalid(
            "the kinetic strain needs the Bloch sums of the mesh",
        ));
    }
    let mut fft = Fft3::new(ctx, grid.index.shape())?;
    let weight = kinetic_weights(ctx, grid, mesh, k)?;
    let outer = reciprocal::bloch_outer(ctx, grid, k)?;
    let (u, _held) = periodic_parts(ctx, &mut fft, mesh, k, values, m)?;
    let mut weighted = u.clone();
    for mut column in weighted.column_iter_mut() {
        for (v, s) in column.iter_mut().zip(&weight) {
            *v *= *s;
        }
    }
    let t = u.adjoint() * &weighted;
    // The symbol term for the six distinct pairs: `ds / 2 Omega = -Q_a Q_b / Omega`.
    let symbol: Vec<Matrix> = (0..6)
        .map(|pair| {
            let row = &outer.as_slice()[pair * n..(pair + 1) * n];
            let mut scaled = u.clone();
            for mut column in scaled.column_iter_mut() {
                for (v, s) in column.iter_mut().zip(row) {
                    *v *= -*s / mesh.volume;
                }
            }
            u.adjoint() * scaled
        })
        .collect();
    let mut out = zeros(m);
    for (a, row) in out.iter_mut().enumerate() {
        for (b, to) in row.iter_mut().enumerate() {
            let tangent = strain.tangent(ctx, a, b)?;
            let (bra, _bra) = periodic_parts(ctx, &mut fft, mesh, k, tangent.as_slice(), m)?;
            let g = bra.adjoint() * &weighted;
            let pair = VOIGT
                .iter()
                .position(|&(p, q)| (p, q) == (a, b) || (q, p) == (a, b))
                .expect("a pair of axes is one of the six");
            *to = &g + g.adjoint() + &symbol[pair] + &t * Complex64::new(delta(a, b), 0.);
        }
    }
    Ok(out)
}

/// `d T_tau / d eps_ab` of the meta-GGA operator `T_tau = sum_c g_c^+ diag(dv v_tau / 2) g_c`
/// ([`crate::periodic::fields::kinetic_operator`]), `g_c` the SCF's spectral gradient of the Bloch
/// sums along `c` ([`crate::periodic::fields::bloch_gradients`], the alias-MEAN wave vector `Q`),
/// with the node values of `v_tau` fixed. No `delta_ab` term: like the local form, the operator's
/// `dv` belongs to `E_xc = dv sum_i f(..., tau_i)`, whose Jacobian is X-1. The wave vectors move as
/// `Q -> (1 - eps^T) Q`, so `d g_c / d eps_ab = -delta_cb g_a` at fixed samples (the symbol term
/// `-(g_a^+ W g_b + g_b^+ W g_a)`, `W = diag(dv v_tau / 2)`), and the samples move by the tangent
/// `A_ab` (the term `G + G^+`, `G = sum_c (g_c A_ab)^+ W g_c`).
pub fn tau_strain(
    ctx: &Context,
    grid: &SpatialGrid,
    mesh: &Mesh,
    strain: &StrainSamples,
    values: &[Complex64],
    k: [f64; 3],
    potential: &[f64],
) -> Result<Tensor> {
    use crate::periodic::fields::bloch_gradients;
    let (m, n) = (strain.functions(), mesh.len());
    if values.len() != m * n || potential.len() != n || n != grid.index.points() {
        return Err(invalid(
            "the kinetic-density strain needs the Bloch sums and one v_tau value per grid node",
        ));
    }
    // The untracked copies: the weighted ket and the two column-major views (three axes each).
    let _held = ctx.reserve(product(&[m, n, 16, 9])?)?;
    let half_dv = 0.5 * mesh.dv;
    // [axis][function][point]: the gradients, and the ket weighted by dv v_tau / 2.
    let gradients = bloch_gradients(ctx, grid, k, values, m)?;
    let mut ket = gradients.as_slice().to_vec();
    for row in ket.chunks_mut(n) {
        for (value, v) in row.iter_mut().zip(potential) {
            *value *= half_dv * v;
        }
    }
    let axis = |data: &[Complex64], c: usize| columns(&data[c * m * n..(c + 1) * m * n], n, m).into_owned();
    let g: [Matrix; 3] = std::array::from_fn(|c| axis(gradients.as_slice(), c));
    let w: [Matrix; 3] = std::array::from_fn(|c| axis(&ket, c));
    let mut out = zeros(m);
    for (a, row) in out.iter_mut().enumerate() {
        for (b, to) in row.iter_mut().enumerate() {
            let tangent = strain.tangent(ctx, a, b)?;
            let moved = bloch_gradients(ctx, grid, k, tangent.as_slice(), m)?;
            let mut cross = Matrix::zeros(m, m);
            for c in 0..3 {
                cross += axis(moved.as_slice(), c).adjoint() * &w[c];
            }
            *to = &cross + cross.adjoint() - g[a].adjoint() * &w[b] - g[b].adjoint() * &w[a];
        }
    }
    Ok(out)
}
