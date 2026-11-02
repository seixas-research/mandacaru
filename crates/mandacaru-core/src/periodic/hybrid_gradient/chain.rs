//! The chain from the response `Z` of the projections to the force on the atoms: the ownership
//! identity of CF-2, `dC_X = P_X G_C - G_C Q_X`, with `G_C = dC/dc_row` and `Z` as the engine
//! contracts it (`2 Re sum_{mu,p} G_C[mu,p] Z[p,mu]`, added to the owner of the row `mu` and
//! subtracted from the owner of the projector `p`).
use super::Matrix;
use crate::{periodic::invalid, Result};

/// The projection chain per atom, rows (the basis functions moving) and columns (the projectors
/// moving) apart so a caller can gate each.
pub struct ProjectionForces {
    pub rows: Vec<[f64; 3]>,
    pub columns: Vec<[f64; 3]>,
}

impl ProjectionForces {
    pub fn total(&self) -> Vec<[f64; 3]> {
        self.rows
            .iter()
            .zip(&self.columns)
            .map(|(r, c)| std::array::from_fn(|a| r[a] + c[a]))
            .collect()
    }
}

/// `response`: `Z` per (spin, k), `spin * nk + k` (`p x m`); `gradients[k]`: `G_C^axis(k)`
/// (`m x p`); `row_owner` / `projector_owner`: the atom of each basis function / projector.
pub fn projection_forces(
    response: &[Matrix],
    gradients: &[[Matrix; 3]],
    row_owner: &[usize],
    projector_owner: &[usize],
    atoms: usize,
) -> Result<ProjectionForces> {
    let nk = gradients.len();
    if nk == 0 || response.len() % nk != 0 {
        return Err(invalid(
            "the response must hold one matrix per spin channel and k-point",
        ));
    }
    let mut out = ProjectionForces {
        rows: vec![[0.; 3]; atoms],
        columns: vec![[0.; 3]; atoms],
    };
    for (at, z) in response.iter().enumerate() {
        let g = &gradients[at % nk];
        if z.shape() != (projector_owner.len(), row_owner.len()) {
            return Err(invalid("a response must be p x m"));
        }
        for (axis, ga) in g.iter().enumerate() {
            if ga.shape() != (row_owner.len(), projector_owner.len()) {
                return Err(invalid("a projection gradient must be m x p"));
            }
            for (mu, &row) in row_owner.iter().enumerate() {
                for (p, &column) in projector_owner.iter().enumerate() {
                    let t = 2.0 * (ga[(mu, p)] * z[(p, mu)]).re;
                    out.rows[row][axis] += t;
                    out.columns[column][axis] -= t;
                }
            }
        }
    }
    Ok(out)
}
