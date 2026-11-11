//! Coefficient algebra on ordered term lists: the arithmetic behind the Python `PauliSum` and
//! `Fermion` builders, whose terms are an insertion-ordered `{key: coefficient}` mapping.
//!
//! Every function keeps that order: a sum lists each key where it first occurs (duplicates
//! added there, in order), a product visits the pairs left factor outer, right factor inner,
//! and nothing is reordered or dropped unless asked (`drop_small`). The keys are Pauli labels or
//! fermion ladder products (`(mode, creation)` factors); nothing here maps or orders operators.
use super::C64;
use crate::{Error, Result};
use std::collections::HashMap;
use std::hash::Hash;

/// A fermion ladder product: `(mode, creation)` factors, left to right.
pub type Ladders = Vec<(usize, bool)>;

fn finite(value: C64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

/// `terms` with equal keys added where the key first occurs.
pub fn sum<K: Clone + Eq + Hash>(terms: impl IntoIterator<Item = (K, C64)>) -> Result<Vec<(K, C64)>> {
    let mut out: Vec<(K, C64)> = Vec::new();
    let mut index: HashMap<K, usize> = HashMap::new();
    for (key, value) in terms {
        if !finite(value) {
            return Err(Error::Invalid("non-finite operator coefficient".into()));
        }
        match index.get(&key) {
            Some(&at) => out[at].1 += value,
            None => {
                index.insert(key.clone(), out.len());
                out.push((key, value));
            }
        }
    }
    if out.iter().any(|(_, v)| !finite(*v)) {
        return Err(Error::Invalid("operator coefficient overflow".into()));
    }
    Ok(out)
}

/// Every coefficient times `factor`, in order.
pub fn scale<K: Clone>(terms: &[(K, C64)], factor: C64) -> Result<Vec<(K, C64)>> {
    if !finite(factor) {
        return Err(Error::Invalid("non-finite scalar".into()));
    }
    Ok(terms.iter().map(|(k, v)| (k.clone(), v * factor)).collect())
}

/// The terms whose coefficient magnitude exceeds `atol`, in order.
pub fn drop_small<K: Clone>(terms: &[(K, C64)], atol: f64) -> Result<Vec<(K, C64)>> {
    if !atol.is_finite() || atol < 0. {
        return Err(Error::Invalid("the drop tolerance must be finite and nonnegative".into()));
    }
    Ok(terms.iter().filter(|(_, v)| v.norm() > atol).cloned().collect())
}

/// Whether every coefficient above `drop` in magnitude has an imaginary part below `atol`: a
/// sum of Hermitian strings with real weights.
pub fn hermitian<K>(terms: &[(K, C64)], atol: f64, drop: f64) -> bool {
    terms.iter().filter(|(_, v)| v.norm() > drop).all(|(_, v)| v.im.abs() < atol)
}

/// The Hermitian Pauli sum a propagator compiles: the terms of `n_qubits`-letter `I/X/Y/Z` labels
/// whose real part is nonzero, as `(label, real part)` in order, with the sum of their magnitudes
/// (the norm bound, summed as Python's `sum` adds floats). A malformed label, a non-finite coefficient or an imaginary part above `1e-12`
/// is refused, in the order the terms are met, as is a sum that overflows.
pub fn real_pauli_terms(n_qubits: usize, terms: &[(String, C64)]) -> Result<(Vec<(String, f64)>, f64)> {
    let mut real = Vec::new();
    for (label, coefficient) in terms {
        if label.chars().count() != n_qubits || label.chars().any(|c| !"IXYZ".contains(c)) {
            return Err(Error::Invalid("invalid Pauli label".into()));
        }
        if !finite(*coefficient) || coefficient.im.abs() > 1e-12 {
            return Err(Error::Invalid("hamiltonian must be finite and Hermitian".into()));
        }
        if coefficient.re != 0. {
            real.push((label.clone(), coefficient.re));
        }
    }
    let bound = crate::derived::python_sum(&real.iter().map(|(_, c)| c.abs()).collect::<Vec<_>>());
    if !bound.is_finite() {
        return Err(Error::Invalid("sum of absolute Pauli coefficients must be finite".into()));
    }
    Ok((real, bound))
}

/// The product of two fermion operators: each pair of terms concatenated (left term first),
/// coefficients multiplied, pairs visited left outer and right inner, equal products added.
pub fn fermion_product(left: &[(Ladders, C64)], right: &[(Ladders, C64)]) -> Result<Vec<(Ladders, C64)>> {
    sum(left.iter().flat_map(|(a, ca)| {
        right.iter().map(move |(b, cb)| {
            let mut ladders = a.clone();
            ladders.extend_from_slice(b);
            (ladders, ca * cb)
        })
    }))
}

/// The Hermitian conjugate of a fermion operator: each product reversed with its creation and
/// annihilation exchanged, coefficients conjugated, equal products added.
pub fn fermion_dagger(terms: &[(Ladders, C64)]) -> Result<Vec<(Ladders, C64)>> {
    sum(terms.iter().map(|(ladders, value)| {
        (ladders.iter().rev().map(|&(mode, creation)| (mode, !creation)).collect(), value.conj())
    }))
}

/// One more than the largest mode any term acts on (zero for no ladder operator).
pub fn fermion_modes(terms: &[(Ladders, C64)]) -> usize {
    terms.iter().flat_map(|(l, _)| l.iter().map(|&(mode, _)| mode + 1)).max().unwrap_or(0)
}
