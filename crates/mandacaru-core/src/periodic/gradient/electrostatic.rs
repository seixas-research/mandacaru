//! E1-E4: the explicit derivatives of the electrostatic energy at the frozen output density.
//!
//! The energy is `E = e_ion + c_ion + sum_G K (|rho|^2/2 + Re rho^* (hats + ion))/V
//! + Re (q^+ U q / 2 + q^+ u_ion + q linear)` with `hats = sum_a q_a hat_a(G)`; at fixed
//! `rho` and `q` the position enters through `hat_a(G)` (structure factors, E1), `ion(G)` and the
//! dense `e_ion` (E2), `U`, `u_ion`, `linear` (E3) and the ion constants (E4). The mesh
//! contractions with `rho` belong to the provider (`hat_overlap`, `ion_overlap`); the dense
//! contractions with `q` are done here, with the formulas of the solver's energy.
use super::{Derivatives, Frozen};
use crate::{quantum::C64, Result};

/// `dE/dtau` per atom of E1-E4.
pub struct Electrostatic {
    pub hats: Vec<[f64; 3]>,
    pub ion: Vec<[f64; 3]>,
    pub dense: Vec<[f64; 3]>,
    pub constants: Vec<[f64; 3]>,
}

/// `Re (q^+ dU q / 2 + q^+ d u_ion + q d linear)`.
fn dense_term<D: Derivatives>(d: &D, q: &[C64], atom: usize, axis: usize) -> Result<f64> {
    let dense = d.dense(atom, axis)?;
    let mut sum = C64::new(0.0, 0.0);
    for (a, qa) in q.iter().enumerate() {
        let uq: C64 = q
            .iter()
            .enumerate()
            .map(|(b, qb)| dense.u[(a, b)] * qb)
            .sum();
        sum += 0.5 * qa.conj() * uq + qa.conj() * dense.uion[a] + qa * dense.linear[a];
    }
    Ok(sum.re)
}

pub fn contract<D: Derivatives>(d: &D, frozen: &Frozen<D::Fields>) -> Result<Electrostatic> {
    let atoms = d.layout().atoms;
    let mut out = Electrostatic {
        hats: vec![[0.0; 3]; atoms],
        ion: vec![[0.0; 3]; atoms],
        dense: vec![[0.0; 3]; atoms],
        constants: vec![[0.0; 3]; atoms],
    };
    for atom in 0..atoms {
        for axis in 0..3 {
            let overlaps = d.hat_overlap(&frozen.fields, atom, axis)?;
            out.hats[atom][axis] = frozen
                .q
                .iter()
                .zip(&overlaps)
                .map(|(q, o)| (q * o).re)
                .sum();
            out.ion[atom][axis] =
                d.ion_overlap(&frozen.fields, atom, axis)? + d.ion_energy(atom, axis)?;
            out.dense[atom][axis] = dense_term(d, &frozen.q, atom, axis)?;
            out.constants[atom][axis] = d.constant(atom, axis)?;
        }
    }
    Ok(out)
}
