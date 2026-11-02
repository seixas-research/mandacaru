//! The explicit terms of the force: everything that differentiates the energy at the frozen
//! state without moving a basis function (E1-E5). The matrix terms live in `pulay.rs`.
use super::{core, electrostatic, Derivatives, Frozen};
use crate::Result;

/// `dE/dtau` per atom, term by term.
pub struct Explicit {
    pub hats: Vec<[f64; 3]>,
    pub ion: Vec<[f64; 3]>,
    pub dense: Vec<[f64; 3]>,
    pub constants: Vec<[f64; 3]>,
    pub core: Vec<[f64; 3]>,
}

pub fn explicit<D: Derivatives>(d: &D, frozen: &Frozen<D::Fields>) -> Result<Explicit> {
    let es = electrostatic::contract(d, frozen)?;
    Ok(Explicit {
        hats: es.hats,
        ion: es.ion,
        dense: es.dense,
        constants: es.constants,
        core: core::contract(d, frozen)?,
    })
}
