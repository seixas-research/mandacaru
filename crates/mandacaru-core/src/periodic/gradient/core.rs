//! E5: the partial-core density against the xc potential.
//!
//! The xc energy is evaluated on `rho + rho_core(X)`, so at the frozen output density the
//! position enters through the core alone: `dE_xc = <v_xc | d rho_core>`. The provider samples
//! the analytic radial slope of the core on its mesh; the engine only collects the atoms.
use super::{Derivatives, Frozen};
use crate::Result;

pub fn contract<D: Derivatives>(d: &D, frozen: &Frozen<D::Fields>) -> Result<Vec<[f64; 3]>> {
    let atoms = d.layout().atoms;
    let mut out = vec![[0.0; 3]; atoms];
    for (atom, row) in out.iter_mut().enumerate() {
        for (axis, x) in row.iter_mut().enumerate() {
            *x = d.core_overlap(&frozen.fields, atom, axis)?;
        }
    }
    Ok(out)
}
