//! The Lanczos recursion of a Hermitian operator from one start vector, with full
//! reorthogonalization: the Gauss quadrature of the operator's spectral measure in that vector.
//!
//! For `K` steps from `|u>` the tridiagonal `T` has the Ritz values `E_j` and eigenvectors `U`;
//! the poles are `E_j` with weights `<u|u> U_0j^2`. Their resolvent `sum_j w_j / (z - E_j)` is the
//! `K`-level continued fraction of `<u| (z - H)^-1 |u>` (Haydock's recursion) and reproduces the
//! first `2K` moments of the measure; it is exact once `K` reaches the dimension of the Krylov
//! space (the recursion stops there on its own). The total weight is `<u|u>` whatever `K` is.
//!
//! Nothing here knows what the operator is; every step is serial and in a fixed order. Callers:
//! the Lehmann spectral functions of a correlated state (`quantum::spectral`) and the
//! Bethe-Salpeter spectrum of a crystal (`periodic::optics::bse`).
use super::invalid;
use crate::{product, Context, Result};
use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64 as C64;

/// The Ritz poles of [`poles`]: ascending energies and their weights.
#[derive(Clone, Debug, Default)]
pub struct Ritz {
    pub energies: Vec<f64>,
    pub weights: Vec<f64>,
}

impl Ritz {
    /// `sum_j w_j / (z - E_j)`.
    pub fn resolvent(&self, z: C64) -> C64 {
        self.energies
            .iter()
            .zip(&self.weights)
            .map(|(e, w)| C64::new(*w, 0.) / (z - e))
            .sum()
    }
}

/// [`Ritz`] of the Hermitian operator `apply` (`apply(v, out)` writes `H v`) from `start` (not
/// normalized), with at most `krylov` vectors. A zero start has no poles.
pub fn poles(
    ctx: &Context,
    start: &[C64],
    krylov: usize,
    mut apply: impl FnMut(&[C64], &mut [C64]) -> Result<()>,
) -> Result<Ritz> {
    let d = start.len();
    if start.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
        return Err(invalid("a Lanczos start vector must be finite"));
    }
    let norm2: f64 = start.iter().map(|v| v.norm_sqr()).sum();
    if norm2 <= 1e-28 {
        return Ok(Ritz::default());
    }
    let steps = krylov.min(d).max(1);
    let _memory = ctx.reserve(product(&[steps + 2, d, 16])?)?;
    let mut basis: Vec<Vec<C64>> = Vec::with_capacity(steps);
    let scale = 1. / norm2.sqrt();
    basis.push(start.iter().map(|v| v * scale).collect());
    let mut alpha = Vec::with_capacity(steps);
    let mut beta: Vec<f64> = Vec::with_capacity(steps);
    let mut w = vec![C64::default(); d];
    for i in 0..steps {
        apply(&basis[i], &mut w)?;
        let a: f64 = basis[i]
            .iter()
            .zip(&w)
            .map(|(v, x)| (v.conj() * x).re)
            .sum();
        if !a.is_finite() {
            return Err(invalid("the Lanczos operator is not finite"));
        }
        alpha.push(a);
        if i + 1 == steps {
            break;
        }
        // Full reorthogonalization, twice (classical Gram-Schmidt is stable when repeated).
        for _ in 0..2 {
            for v in &basis {
                let overlap: C64 = v.iter().zip(&w).map(|(v, x)| v.conj() * x).sum();
                for (x, v) in w.iter_mut().zip(v) {
                    *x -= overlap * v;
                }
            }
        }
        let b = w.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        let spread = alpha.iter().map(|v| v.abs()).fold(1f64, f64::max);
        if b <= 1e-12 * spread {
            break;
        }
        beta.push(b);
        basis.push(w.iter().map(|v| v / b).collect());
    }
    let k = alpha.len();
    let t = DMatrix::from_fn(k, k, |i, j| {
        if i == j {
            alpha[i]
        } else if i + 1 == j {
            beta[i]
        } else if j + 1 == i {
            beta[j]
        } else {
            0.
        }
    });
    let eigen = SymmetricEigen::new(t);
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
    Ok(Ritz {
        energies: order.iter().map(|&j| eigen.eigenvalues[j]).collect(),
        weights: order
            .iter()
            .map(|&j| norm2 * eigen.eigenvectors[(0, j)].powi(2))
            .collect(),
    })
}
