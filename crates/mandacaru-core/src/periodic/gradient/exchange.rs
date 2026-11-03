//! The exact-exchange terms of the force of a hybrid crystal: the explicit derivative of the
//! screened exchange at fixed occupied states ([`crate::periodic::hybrid_gradient`]) as the
//! provider returns it ([`Derivatives::exchange_gradient`]), checked against the layout and
//! chained to the atoms.
//!
//! The response `Z` of the projections carries its own weights (`w_k`, the spin factor, `-a/2`),
//! so it is contracted with `dC/dR` here, one k-point at a time, by the ownership identity
//! `dC_X = P_X G_C - G_C Q_X` of [`super::pulay`], and never joins the engine's own `z`.
//!
//! A spinor crystal (SH-F2) is presented to the engine in the doubled frame (`2M` rows, owners
//! repeated per spin, `dC = I2 x dC^0`), but the exchange response is already summed over the
//! two spin components in the scalar frame of one block ([`crate::periodic::hybrid_gradient`]:
//! `Z` is `P x M`). It is therefore contracted with the scalar block `G_C^0` of the doubled
//! gradient (its upper left `M x P` corner, the lower right one is the same matrix) and the
//! scalar owners (the first `M` and `P` entries of the doubled ones): the spin trace is in `Z`,
//! so no factor 2 enters here.
use super::{Derivatives, State};
use crate::{
    periodic::{hybrid_gradient::projection_forces, invalid},
    Result,
};

/// `dE_x/dtau` per atom, term by term (derivatives, not forces).
pub struct Terms {
    pub phase: Vec<[f64; 3]>,
    pub functions: Vec<[f64; 3]>,
    pub rows: Vec<[f64; 3]>,
    pub columns: Vec<[f64; 3]>,
}

impl Terms {
    pub fn new(atoms: usize) -> Self {
        let zero = || vec![[0.0; 3]; atoms];
        Terms {
            phase: zero(),
            functions: zero(),
            rows: zero(),
            columns: zero(),
        }
    }
}

/// The exchange terms of the density of `state` (the re-diagonalized bands of the frozen state).
pub fn contract<D: Derivatives>(d: &D, state: &State) -> Result<Terms> {
    let l = d.layout();
    let nk = l.weights.len();
    // One spin block of the frame the response is in.
    let (m, p) = if l.request.spinor {
        (l.m / 2, l.p / 2)
    } else {
        (l.m, l.p)
    };
    let gradient = d.exchange_gradient(state)?;
    let functions = gradient.functions.as_ref().ok_or_else(|| {
        invalid("crystal gradient: the exchange derivative must carry the basis-function terms")
    })?;
    let response = gradient.response();
    if gradient.phase.len() != l.atoms
        || functions.bra.len() != l.atoms
        || functions.ket.len() != l.atoms
        || response.len() != l.spins * nk
        || response.iter().any(|z| z.shape() != (p, m))
    {
        return Err(invalid(
            "crystal gradient: the exchange derivative disagrees with the layout \
             (atoms, spins, k-points, p x m responses)",
        ));
    }
    let mut out = Terms::new(l.atoms);
    out.phase.clone_from(&gradient.phase);
    out.functions = functions.total();
    for k in 0..nk {
        let mut gc = d.projection_gradient(k)?;
        if l.request.spinor {
            for g in gc.iter_mut() {
                *g = g.view((0, 0), (m, p)).into_owned();
            }
        }
        let z: Vec<_> = (0..l.spins).map(|s| response[s * nk + k].clone()).collect();
        let chain = projection_forces(
            &z,
            &[gc],
            &l.row_owner[..m],
            &l.projector_owner[..p],
            l.atoms,
        )?;
        for atom in 0..l.atoms {
            for a in 0..3 {
                out.rows[atom][a] += chain.rows[atom][a];
                out.columns[atom][a] += chain.columns[atom][a];
            }
        }
    }
    Ok(out)
}
