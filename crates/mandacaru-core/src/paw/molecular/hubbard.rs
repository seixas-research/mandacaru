//! Molecular PAW Hubbard setup builder.
use super::{data::Species, onecenter::AtomTerms};
use crate::{
    models::{elements::SYMBOLS, hubbard_request::Request},
    molecular::hubbard::{MolecularHubbard, MolecularHubbardShell},
    Geometry, Result,
};
use nalgebra::DMatrix;

/// The projectors of `terms` that form the shell of angular momentum `l`, as
/// `(projector index, m index in 0..2l+1, weight of its radial partial wave)`: the one
/// definition of `q_ao[mu, m] = sum_p C[mu, p] w_p` that the energy and its gradient share.
pub(in crate::paw) fn shell_projectors(
    terms: &AtomTerms,
    l: usize,
    weights: &[f64],
) -> Vec<(usize, usize, f64)> {
    terms
        .projectors
        .iter()
        .enumerate()
        .filter(|(_, p)| p.l == l)
        .map(|(index, p)| (index, (p.m + l as i32) as usize, weights[p.radial]))
        .collect()
}

/// Build the molecular Hubbard specification and subspace projectors `q_ao`
/// from the validated user request. Every element the request names must be in the
/// molecule and carry the shell: the request is never partly applied.
pub fn build(
    request: &Request,
    species: &[Species],
    species_terms: &[AtomTerms],
    atoms: &[usize],
    geometry: &Geometry,
    c: &[DMatrix<f64>],
    m: usize,
) -> Result<MolecularHubbard> {
    if request.entries.is_empty() {
        return Err(crate::Error::Invalid("hubbard_u names no element".into()));
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
            let weights = sp.channels[ch_idx].weights.as_ref().ok_or_else(|| {
                crate::Error::Invalid(format!(
                    "PAW dataset channel l = {} for element {symbol} has no bound reference wave",
                    entry.l
                ))
            })?;

            let mut q_ao = DMatrix::zeros(m, 2 * entry.l + 1);
            for (p_idx, m_idx, w) in shell_projectors(terms, entry.l, weights) {
                for mu in 0..m {
                    q_ao[(mu, m_idx)] += c[a][(mu, p_idx)] * w;
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

    if let Some(entry) = request
        .entries
        .iter()
        .find(|e| !shells.iter().any(|shell| shell.symbol == e.symbol))
    {
        return Err(crate::Error::Invalid(format!(
            "hubbard_u names {}, which is not in the system",
            entry.symbol
        )));
    }

    Ok(MolecularHubbard { form, shells })
}
