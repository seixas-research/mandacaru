//! SO-1: the spin-orbit union-projector term of the stress of a spinor crystal (AS-8).
//!
//! The spin-orbit matrix of a k-point is `h^SO = (I2 x C^SO) B (I2 x C^SO)^+` with the projections
//! `C^SO` of the Bloch sums on the dataset's union projectors and the CONSTANT operator `B`
//! (`gradient::spin_orbit`: the angular `L.S` and the radial `D^SO`, no position in it). The
//! strain moves the energy `E_SO = sum_k w Re tr[D h^SO]` only through `C^SO`: no `dB`, no hat, no
//! `dS`, no `W` (the overlap of the spinor basis is `I2 x S`, whose strain is the scalar one). With
//! the frozen spinor density `D` (`2M x 2M`, the Lowdin frame carried)
//!
//! `dE_SO/d eps_ab = sum_k w 2 Re sum_{mu p} dC^SO_ab[mu, p] Z[p, mu]`,
//! `Z = sum_s [B (I2 x C^SO)^+ D]_{(s,p),(s,mu)}`,
//!
//! the force engine's `Z` ([`SpinOrbit::strain_k`] reuses it) with the strain tangent of the
//! projections on the UNION list ([`Strain::spin_orbit_strain`], the same `projection_strain` as the
//! scalar term) in place of the center derivative `P_X G - G Q_X`. The term is linear in the
//! coupling strength: every entry of `B` scales with it.
use super::{Strain, Tensor};
use crate::{periodic::gradient::Matrix, Result};

/// Adds this k-point's share of SO-1 to `out`; `density` is the spinor `D_k` (`2M x 2M`) and
/// `weight` its k-point weight.
pub fn add_k<S: Strain>(
    s: &S,
    out: &mut Tensor<f64>,
    weight: f64,
    k: usize,
    density: &Matrix,
) -> Result<()> {
    let (so, c, dc) = (
        s.spin_orbit()?,
        s.spin_orbit_projection(k)?,
        s.spin_orbit_strain(k)?,
    );
    let share = so.strain_k(weight, &c, &dc, density)?;
    for a in 0..3 {
        for b in 0..3 {
            out[a][b] += share[a][b];
        }
    }
    Ok(())
}
