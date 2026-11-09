//! The spinor frame of the crystal force (CS-3): the one place where a scalar description of a
//! crystal becomes the doubled frame the engine contracts a spinor in.
//!
//! The spinor Hamiltonian of a k-point is, in the spin-major basis (`alpha` block first),
//!
//! ```text
//! H = I2 x fixed + I2 x C (sum_a m_a B_a) C^+ + sum_{a=0..3} sigma_a x V^a + h^SO,
//! ```
//!
//! with `V^a` the matrix of the local potential `v^a(r)` of Pauli channel `a` (`a = 0` carries
//! the electrostatic and the charge-channel xc potential; a time-reversal crystal has only
//! `a = 0`) and `h^SO = (I2 x C^SO) B (I2 x C^SO)^+` the spin-orbit term. The engine contracts
//! `D = C f C^+` and `W = C f (eps - reference) C^+` (both `2M x 2M`, `C = blockdiag(L, L) y`)
//! with the derivative of this `H` and of `S = I2 x S`, so a provider presents:
//!
//! * `Layout::m = 2M`, `Layout::p = 2P`, the owners repeated per spin ([`doubled_layout`]);
//! * every spin-diagonal matrix as `I2 x (.)` ([`doubled`]): the overlap, the kinetic and
//!   short-range blocks, the projections `I2 x C` and their row-center derivative, the channel
//!   blocks `B_a`, `q` and `D_ion`. The contraction `tr[(I2 x G) D] = tr[G (D_aa + D_bb)]` is then
//!   the spin-traced `D^0`, the moments are `q_a = tr[B_a C^+ D^0 C]` and `-tr[W (I2 x dS)]` is the
//!   spin-traced `W^0` term, with no change of the algebra;
//! * the local term as the Pauli sum of the per-channel row-center derivatives
//!   ([`pauli_sum`]): `sum_a sigma_a x G^a`. Its contraction is
//!   `sum_a tr[component_a(D) G^a]` with the Hermitian spatial combinations of
//!   [`crate::models::noncollinear::component`], the same four fields the Hamiltonian scatters in
//!   `spinor::diagonalize_with_exchange`;
//! * the spin-orbit term through [`SpinOrbit`](super::SpinOrbit), whose projections stay in the
//!   scalar frame.
//!
//! The explicit terms (hats, ions, dense sums, constants, the partial core) read the charge
//! channel only. Magnetization enters the energy through the grid xc alone (the augmentation
//! charge is built from the spin-summed density), so there is no on-site magnetization term to
//! differentiate.
use super::{invalid, Layout, Matrix};
use crate::{quantum::C64, Result};

/// `I2 x a`: `a` on both spin blocks, nothing between them.
pub fn doubled(a: &Matrix) -> Matrix {
    let (r, c) = a.shape();
    let mut out = Matrix::zeros(2 * r, 2 * c);
    for s in 0..2 {
        out.view_mut((s * r, s * c), (r, c)).copy_from(a);
    }
    out
}

/// The owners of the doubled frame: the scalar owners repeated per spin.
pub fn doubled_owners(owner: &[usize]) -> Vec<usize> {
    owner.iter().chain(owner).copied().collect()
}

/// The doubled-frame layout of a scalar description (`m` functions, `p` projectors) of a spinor
/// crystal: `2m`, `2p`, owners repeated per spin, one spin channel of degeneracy 1.
pub fn doubled_layout(scalar: &Layout) -> Result<Layout> {
    if !scalar.request.spinor || scalar.spins != 1 || scalar.degeneracy != 1.0 {
        return Err(invalid(
            "spinor layout: a spinor request has one spin channel of degeneracy 1",
        ));
    }
    Ok(Layout {
        m: 2 * scalar.m,
        p: 2 * scalar.p,
        row_owner: doubled_owners(&scalar.row_owner),
        projector_owner: doubled_owners(&scalar.projector_owner),
        ..scalar.clone()
    })
}

/// `sum_a sigma_a x G^a` for the Pauli channels `parts` (`M x M` each): one channel is the
/// time-reversal case (`I2 x G^0`), four are `(n, m_x, m_y, m_z)` with the block structure
/// `spinor::diagonalize_with_exchange` scatters,
///
/// ```text
/// aa = G0 + G3,  bb = G0 - G3,  ab = G1 - i G2,  ba = G1 + i G2.
/// ```
pub fn pauli_sum(parts: &[Matrix]) -> Result<Matrix> {
    let shape = parts.first().map(Matrix::shape);
    if !matches!(parts.len(), 1 | 4) || parts.iter().any(|g| Some(g.shape()) != shape) {
        return Err(invalid(
            "spinor local term: one (time-reversal) or four Pauli channels of one shape",
        ));
    }
    if parts.len() == 1 {
        return Ok(doubled(&parts[0]));
    }
    let (m, i) = (parts[0].nrows(), C64::i());
    let mut out = Matrix::zeros(2 * m, 2 * m);
    out.view_mut((0, 0), (m, m))
        .copy_from(&(&parts[0] + &parts[3]));
    out.view_mut((m, m), (m, m))
        .copy_from(&(&parts[0] - &parts[3]));
    out.view_mut((0, m), (m, m))
        .copy_from(&(&parts[1] - &parts[2] * i));
    out.view_mut((m, 0), (m, m))
        .copy_from(&(&parts[1] + &parts[2] * i));
    Ok(out)
}
