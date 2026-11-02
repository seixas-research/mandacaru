//! Shared by `periodic_screened_onecenter_spinor*` (and by the older spinor comparison of
//! `periodic_screened_onecenter`): an independent computation of the non-collinear remainder of one
//! sphere and its lift to the `2p x 2p` operator (own loops over the blocks, not `component`), the
//! SU(2) rotation of a spinor density and the Hermitian directions of the finite differences.
#![allow(dead_code)]

use super::periodic_screened_onecenter_support::*;
use mandacaru_core::{paw::screened_atom::ScreenedAtom, quantum::C64, Context};
use nalgebra::DMatrix;

/// The blocks `Q_st` of a `2n x 2n` spinor density.
pub fn block(q: &Matrix, n: usize, s: usize, t: usize) -> Matrix {
    q.view((s * n, t * n), (n, n)).into_owned()
}

/// The four Pauli components (real parts) of a sphere's density, written out block by block.
pub fn pauli(q: &Matrix, n: usize) -> [DMatrix<f64>; 4] {
    let (uu, ud, du, dd) = (
        block(q, n, 0, 0),
        block(q, n, 0, 1),
        block(q, n, 1, 0),
        block(q, n, 1, 1),
    );
    let i = C64::i();
    [
        (&uu + &dd).map(|z| z.re),
        (&ud + &du).map(|z| z.re),
        ((&ud - &du) * i).map(|z| z.re),
        (&uu - &dd).map(|z| z.re),
    ]
}

/// `-fraction P` and the operator `O` (the convention of the model: the Hamiltonian takes
/// `-fraction O`) of the remainder for the `2n x 2n` density `q`, lifted as
/// `O_uu = O^0 + O^3`, `O_dd = O^0 - O^3`, `O_ud = O^1 - i O^2`, `O_du = O^1 + i O^2`.
pub fn remainder(ctx: &Context, atom: &ScreenedAtom, q: &Matrix) -> (f64, Matrix) {
    let n = q.nrows() / 2;
    let terms = atom
        .polarization
        .evaluate_vector(ctx, &pauli(q, n))
        .unwrap();
    let o: Vec<Matrix> = terms
        .operators
        .iter()
        .map(|o| o.map(|x| C64::new(x, 0.)))
        .collect();
    let i = C64::i();
    let mut lifted = Matrix::zeros(2 * n, 2 * n);
    lifted.view_mut((0, 0), (n, n)).copy_from(&(&o[0] + &o[3]));
    lifted.view_mut((n, n), (n, n)).copy_from(&(&o[0] - &o[3]));
    lifted
        .view_mut((0, n), (n, n))
        .copy_from(&(&o[1] - &o[2] * i));
    lifted
        .view_mut((n, 0), (n, n))
        .copy_from(&(&o[1] + &o[2] * i));
    (-FRACTION * terms.energy, lifted)
}

/// `(u (x) 1) q (u (x) 1)^+` for the SU(2) matrix `u` of parameters `(a, b)` (`|a|^2 + |b|^2 = 1`).
pub fn rotate(q: &Matrix, a: C64, b: C64) -> Matrix {
    let n = q.nrows() / 2;
    let u = Matrix::from_row_slice(2, 2, &[a, -b.conj(), b, a.conj()]);
    let big = u.kronecker(&Matrix::identity(n, n));
    &big * q * big.adjoint()
}

/// `tr(a b)` for two matrices of the same order.
pub fn trace_product(a: &Matrix, b: &Matrix) -> C64 {
    (a * b).trace()
}
