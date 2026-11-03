//! H-5 and H-6: the projections `C(k)` of the compensation moments and of the one-center terms
//! move with the strain (`dC_ab`, the spheres rigid about their atoms); their response `Z` is
//! the one the force engine contracts with `dC / dR` (`dE = 2 Re sum_k tr[dC_k Z_k]`, the weights
//! `w_k`, the spin factor and `-a/2` already inside). It is contracted here per k-point, never
//! added to the semilocal response of the pulay terms (that would weight it twice).
use super::{Tangents, Terms};
use crate::{
    periodic::{hybrid_gradient::Gradient, invalid, Crystal},
    Result,
};

/// `Re tr[left right]` of an `m x p` and a `p x m` matrix.
fn trace(
    left: &nalgebra::DMatrix<crate::quantum::C64>,
    right: &nalgebra::DMatrix<crate::quantum::C64>,
) -> f64 {
    let (m, p) = left.shape();
    (0..m)
        .map(|i| {
            (0..p)
                .map(|j| left[(i, j)] * right[(j, i)])
                .sum::<crate::quantum::C64>()
                .re
        })
        .sum()
}

pub(super) fn contract(
    c: &Crystal,
    tangents: &Tangents,
    gradient: &Gradient,
    terms: &mut Terms,
) -> Result<()> {
    let nk = c.points.len();
    if gradient.compensation.len() != c.spins * nk || gradient.sphere.len() != c.spins * nk {
        return Err(invalid(
            "exchange strain: the response must hold one matrix per spin and k-point",
        ));
    }
    for k in 0..nk {
        let dc = tangents.projection_strain(k)?;
        for s in 0..c.spins {
            let at = s * nk + k;
            let (comp, sphere) = (&gradient.compensation[at], &gradient.sphere[at]);
            if comp.shape() != (c.p, c.m) || sphere.shape() != (c.p, c.m) {
                return Err(invalid("exchange strain: the response is p x m"));
            }
            for a in 0..3 {
                for b in 0..3 {
                    terms.compensation[a][b] += 2. * trace(&dc[a][b], comp);
                    terms.sphere[a][b] += 2. * trace(&dc[a][b], sphere);
                }
            }
        }
    }
    Ok(())
}
