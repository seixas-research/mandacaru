//! The exact exchange K_s(k) of a hybrid crystal as part of the periodic SCF state.
//!
//! The operator that defines the next diagonalization is the same kind of object as the
//! density it was built from, so it is mixed with the density and checked for convergence with
//! it. `Crystal::screened_exchange` is the one place the periodic code reads the range
//! separation and the exact-exchange fraction: the functional registry, as the molecular path.
use super::{invalid, spinor, Crystal};
use crate::{quantum::C64, Result};
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

impl Crystal {
    /// `(omega, exact fraction)` of a hybrid functional, `None` for a semilocal one.
    pub fn screened_exchange(&self) -> Result<Option<(f64, f64)>> {
        Ok(crate::models::functional(&self.functional)?.screened_exchange)
    }
    /// The screened exchange of a crystal that must be a hybrid.
    pub fn require_screened_exchange(&self) -> Result<(f64, f64)> {
        self.screened_exchange()?.ok_or_else(|| {
            invalid(format!(
                "exact exchange needs a hybrid functional, not '{}'",
                self.functional
            ))
        })
    }
}

/// Number of operators: one per k-point, for each collinear spin channel (one for a spinor).
pub fn operators(c: &Crystal) -> usize {
    c.points.len() * if c.spinor { 1 } else { c.spins }
}

/// Side of one operator: the basis, doubled for a spinor.
pub fn dimension(c: &Crystal) -> usize {
    if c.spinor {
        2 * c.m
    } else {
        c.m
    }
}

/// Complex numbers in the mixed state.
pub fn len(c: &Crystal) -> usize {
    operators(c) * dimension(c) * dimension(c)
}

/// The start of the SCF: no exchange, as the first diagonalization of a semilocal functional.
pub fn zeros(c: &Crystal) -> Vec<Matrix> {
    let d = dimension(c);
    (0..operators(c)).map(|_| Matrix::zeros(d, d)).collect()
}

/// The operators as one vector, column-major per operator, for the mixer.
pub fn flatten(k: &[Matrix]) -> Vec<C64> {
    k.iter().flat_map(|m| m.iter().copied()).collect()
}

/// Inverse of [`flatten`]; the matrices are made exactly Hermitian, as K is.
pub fn restore(k: &mut [Matrix], flat: &[C64]) {
    let mut at = 0;
    for matrix in k.iter_mut() {
        let d = matrix.nrows();
        let raw = &flat[at..at + d * d];
        for j in 0..d {
            for i in 0..d {
                matrix[(i, j)] = 0.5 * (raw[i + j * d] + raw[j + i * d].conj());
            }
        }
        at += d * d;
    }
}

/// Largest change of the Hamiltonian that the exchange makes from `used` to `built`, in Hartree.
///
/// The diagonalization added `-fraction * K(used)`; the density it produced asks for
/// `-fraction * K(built)`. The difference is taken in the orthonormal frame the eigenproblem is
/// solved in, and measured by its Frobenius norm, which bounds the shift of every eigenvalue
/// (Hoffman-Wielandt): when it is below `t`, no band of the SCF is more than `t` Hartree away
/// from the band of the Hamiltonian that the converged density defines. Maximum over k and spin.
pub fn residual(c: &Crystal, fraction: f64, used: &[Matrix], built: &[Matrix]) -> f64 {
    let nk = c.points.len();
    used.iter()
        .zip(built)
        .enumerate()
        .map(|(index, (used, built))| {
            let frame = spinor::lowdin_block(&c.points[index % nk], c.m, c.spinor);
            let change = (built - used) * C64::new(fraction, 0.);
            (frame.adjoint() * change * &frame).norm()
        })
        .fold(0., f64::max)
}
