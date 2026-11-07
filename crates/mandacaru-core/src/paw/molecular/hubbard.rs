//! Molecular PAW Hubbard setup builder.
use super::{data::Species, onecenter::AtomTerms};
use crate::{
    models::{elements::SYMBOLS, hubbard_request::Request},
    molecular::hubbard::{MolecularHubbard, MolecularHubbardShell},
    Geometry, Result,
};
use nalgebra::DMatrix;

/// Build the molecular Hubbard specification and subspace projectors `q_ao`
/// from the validated user request.
pub fn build(
    request: &Request,
    species: &[Species],
    species_terms: &[AtomTerms],
    atoms: &[usize],
    geometry: &Geometry,
    c: &[DMatrix<f64>],
    m: usize,
) -> Result<Option<MolecularHubbard>> {
    if request.entries.is_empty() {
        return Ok(None);
    }
    let mut shells = Vec::new();
    let form = request.entries[0].form;

    for (a, &s) in atoms.iter().enumerate() {
        let z = geometry.numbers()[a];
        if z == 0 || z as usize > SYMBOLS.len() {
            continue;
        }
        let symbol = SYMBOLS[z as usize - 1];
        if let Some(entry) = request.entries.iter().find(|e| e.symbol == symbol) {
            let terms = &species_terms[s];
            let sp = &species[s];
            let ch_idx = sp
                .channels
                .iter()
                .position(|ch| ch.l == entry.l)
                .ok_or_else(|| {
                    crate::Error::Invalid(format!(
                        "PAW dataset for element {symbol} does not have channel l = {}",
                        entry.l
                    ))
                })?;
            let weights = sp.channels[ch_idx]
                .weights
                .as_ref()
                .ok_or_else(|| {
                    crate::Error::Invalid(format!(
                        "PAW dataset channel l = {} for element {symbol} has no bound reference wave",
                        entry.l
                    ))
                })?;

            let mut q_ao = DMatrix::zeros(m, 2 * entry.l + 1);
            for (p_idx, p) in terms.projectors.iter().enumerate() {
                if p.l == entry.l {
                    let m_idx = (p.m + entry.l as i32) as usize;
                    let w = weights[p.radial];
                    for mu in 0..m {
                        q_ao[(mu, m_idx)] += c[a][(mu, p_idx)] * w;
                    }
                }
            }

            shells.push(MolecularHubbardShell {
                atom: a,
                symbol: symbol.to_string(),
                l: entry.l,
                u: entry.u,
                j: entry.j,
                weights: weights.clone(),
                q: q_ao.clone(),
                q_ao,
            });
        }
    }

    if shells.is_empty() {
        return Ok(None);
    }

    Ok(Some(MolecularHubbard { form, shells }))
}
