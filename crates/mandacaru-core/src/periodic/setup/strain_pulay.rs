//! Strain derivatives of the grid forms of the basis: the overlap, the local potential and the
//! alias-averaged kinetic matrix at one k-point, contracted with a Hermitian matrix `H` (the density
//! or the energy-weighted density of the bands) BEFORE the matrices are formed.
//!
//! Convention (see [`super::strain_samples`]): `x -> (1 + eps) x` carries the lattice, the atoms,
//! the grid nodes and the image centers, fractional `k` is held (so every phase `k.T`, `k.r` is
//! invariant), the node count `N` is fixed, so `dv -> dv (1 + tr eps)` and `Omega -> Omega (1 +
//! tr eps)`. With `A_E = sum_ab E_ab d chi / d eps_ab` ([`StrainSamples::tangent_along`]) for a
//! direction `E`:
//!
//! * grid form `F = dv sum_i conj(chi_mu) w_i chi_nu`:
//!   `dF_E = tr(E) F (the dv of the form) + G + G^+`, `G = dv sum_i conj(A_E,mu) w_i chi_nu`;
//!   an energy that carries its own `dv` (the density of a fixed potential) has no `tr(E) F`;
//! * kinetic `T = sum_G conj(u_mu) (s/2 Omega) u_nu`, `u = dv FFT[chi e^{-ik.r}]`, `s` the alias
//!   mean `|k + G|^2` (`reciprocal::bloch`): `dT_E = tr(E) T + sum_G conj(u) (-Q_E/Omega) u
//!   + G^T + G^T+`, with `Q_E = sum_ab E_ab Q_a Q_b` the alias MEAN of the product
//!   (`reciprocal::bloch_outer`, not the product of the mean vectors) and `G^T` the same form with
//!   the bra replaced by the periodic part of `A_E`.
//!
//! What the stress needs is `Re tr[H dM_E]`, and `tr[H G] = sum_{x, mu} conj(A_mu(x)) Y(x, mu)` with
//! `Y = (dv w) (chi H)`: one `n x m x m` product per matrix and k-point, then every direction is a
//! pointwise sum over the mesh. No `m x m` tangent matrix is formed. The mesh is streamed in
//! blocks ([`super::pulay::block_len`]) and the kinetic operator is applied through its
//! real-space ket (`G^T` is `sum_G conj(bra) w (u H) = sum_x conj(A) (K H)` by Parseval,
//! [`super::pulay::kinetic_setup`]), so the only array of the size of the mesh and the basis is
//! that ket; the tangent-independent symbol terms are `m x m` matrices.
use super::{
    pulay::{Matrix, Mesh},
    shells::{DenseRows, Sampling, Shells},
    strain_samples::StrainSamples,
};
use crate::{math::SpatialGrid, periodic::invalid, product, Buffer, Context, Result};
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

/// `sum_i conj(a_i) b_i` over two equal-length slices.
pub fn dotc(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter().zip(b).map(|(x, y)| x.conj() * y).sum()
}

/// The Hermitian part `(M + M^+) / 2`: `Re tr[M (G + G^+)] = 2 Re tr[H G]` for any `M`.
pub fn hermitian(m: &Matrix) -> Matrix {
    (m + m.adjoint()) * Complex64::new(0.5, 0.)
}

/// `d T_tau_E` of the meta-GGA operator `T_tau = sum_c g_c^+ diag(dv v_tau / 2) g_c`
/// ([`crate::periodic::fields::kinetic_operator`]), `g_c` the SCF's spectral gradient of the Bloch
/// sums along `c` ([`crate::periodic::fields::bloch_gradients`], the alias-MEAN wave vector `Q`),
/// with the node values of `v_tau` fixed, for each direction of `directions`. No `delta_ab` term:
/// like the local form, the operator's `dv` belongs to `E_xc = dv sum_i f(..., tau_i)`, whose
/// Jacobian is X-1. The wave vectors move as `Q -> (1 - eps^T) Q`, so `d g_c / d eps_ab = -delta_cb
/// g_a` at fixed samples (the symbol term `-sum_ab (E_ab + E_ba) g_a^+ W g_b`, `W = diag(dv v_tau /
/// 2)`), and the samples move by the tangent `A_E` (the term `G + G^+`, `G = sum_c (g_c A_E)^+ W g_c`).
pub fn tau_strain(
    ctx: &Context,
    grid: &SpatialGrid,
    mesh: &Mesh,
    strain: &StrainSamples,
    values: &[Complex64],
    k: [f64; 3],
    potential: &[f64],
    directions: &[[[f64; 3]; 3]],
) -> Result<Vec<Matrix>> {
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
    let axis = |data: &[Complex64], c: usize| {
        columns(&data[c * m * n..(c + 1) * m * n], n, m).into_owned()
    };
    let g: [Matrix; 3] = std::array::from_fn(|c| axis(gradients.as_slice(), c));
    let w: [Matrix; 3] = std::array::from_fn(|c| axis(&ket, c));
    directions
        .iter()
        .map(|e| {
            let tangent = strain.tangent_along(ctx, e)?;
            let moved = bloch_gradients(ctx, grid, k, tangent.as_slice(), m)?;
            let mut cross = Matrix::zeros(m, m);
            for c in 0..3 {
                cross += axis(moved.as_slice(), c).adjoint() * &w[c];
            }
            let mut out = &cross + cross.adjoint();
            for a in 0..3 {
                for b in 0..3 {
                    let weight = e[a][b] + e[b][a];
                    if weight != 0. {
                        out -= g[a].adjoint() * &w[b] * Complex64::new(weight, 0.);
                    }
                }
            }
            Ok(out)
        })
        .collect()
}
