//! Per-dataset repairs of the d-block and 6p libraries, keyed by element and functional.
//!
//! Every entry is the exact combination a scan measured, all built unitary (norm deficit 0), and
//! fixes one of two defects of the scalar libraries: the **s channel** misses a hydrogen 1s
//! entering its sphere (the miss of the projector expansion of that tail, above 1 before): a
//! local-potential shift and a tenth Bessel function, and for Tl, Po, Bi, Os, Re and Ta a shorter
//! s sphere, bring it to 0.1-0.95; the **semicore d** of Ga-Kr, I and Xe is so compact that its
//! projectors amplify the basis filter (a filtered 3d/4d function projected onto its own wave at
//! 0.20 Angstrom was off by 0.1-0.9): a larger d sphere with a higher second reference energy
//! brings it to 0.02-0.16. The large-sphere elements Cs, Ba, Ra, La, Lu, Ir, W, Au and the
//! actinides carry their (n-1)s (n-1)p semicore, an f channel (empty for Cs-Th), the 4f frozen
//! (Lu, Ir, W, Au) and spheres of 2.0-3.45 Bohr. Sb (LDA) and Sn take no local shift: the 10 Ha
//! raise widened their local potential and left the f channel, which has no projectors, scattering
//! 0.157 rad off the atom; Sn and Bi in PBE take their LDA repairs. The three derived tables of
//! the generator (local shift, second-reference offset and norm deficit by dataset) are the
//! functions below.

const S_CHANNEL: &[(&str, &str, usize, f64)] = &[
    ("Ac", "lda", 10, 16.0),
    ("Ac", "pbe", 10, 16.0),
    ("Au", "lda", 10, 20.0),
    ("Au", "pbe", 10, 20.0),
    ("Ba", "lda", 10, 12.0),
    ("Ba", "pbe", 10, 12.0),
    ("Bi", "lda", 10, 10.0),
    ("Bi", "pbe", 10, 10.0),
    ("Cr", "lda", 10, 40.0),
    ("Cr", "pbe", 10, 0.0),
    ("Cs", "lda", 10, 12.0),
    ("Cs", "pbe", 10, 12.0),
    ("Ir", "lda", 10, 20.0),
    ("Ir", "pbe", 10, 20.0),
    ("La", "lda", 10, 16.0),
    ("La", "pbe", 10, 16.0),
    ("Lu", "lda", 8, 0.0),
    ("Lu", "pbe", 8, 0.0),
    ("Os", "lda", 10, 10.0),
    ("Os", "pbe", 10, 50.0),
    ("Pa", "lda", 10, 12.0),
    ("Pa", "pbe", 10, 12.0),
    ("Pd", "lda", 9, 50.0),
    ("Pd", "pbe", 10, 50.0),
    ("Pt", "lda", 10, 0.0),
    ("Pt", "pbe", 10, 0.0),
    ("Ra", "lda", 10, 16.0),
    ("Ra", "pbe", 10, 16.0),
    ("Re", "lda", 10, 10.0),
    ("Re", "pbe", 10, 10.0),
    ("Rh", "lda", 10, 50.0),
    ("Rh", "pbe", 10, 50.0),
    ("Ru", "lda", 9, 50.0),
    ("Ru", "pbe", 10, 0.0),
    ("Sb", "lda", 9, 0.0),
    ("Sn", "lda", 9, 0.0),
    ("Sn", "pbe", 9, 0.0),
    ("Th", "lda", 10, 12.0),
    ("Th", "pbe", 10, 12.0),
    ("Tl", "lda", 10, 10.0),
    ("Tl", "pbe", 8, 10.0),
    ("U", "lda", 10, 12.0),
    ("U", "pbe", 10, 12.0),
    ("W", "lda", 10, 10.0),
    ("W", "pbe", 10, 10.0),
];

const D_CHANNEL: &[(&str, &str, f64, f64)] = &[
    ("As", "lda", 2.2, 3.0),
    ("As", "pbe", 1.612, 3.0),
    ("Br", "lda", 1.006, 3.0),
    ("Br", "pbe", 1.009, 3.0),
    ("Ga", "lda", 2.6, 2.0),
    ("Ga", "pbe", 2.6, 2.0),
    ("Ge", "lda", 1.696, 3.0),
    ("Ge", "pbe", 1.696, 3.0),
    ("I", "lda", 1.113, 2.0),
    ("I", "pbe", 1.113, 2.0),
    ("Kr", "lda", 1.066, 1.0),
    ("Kr", "pbe", 1.07, 2.0),
    ("Se", "lda", 2.0, 2.0),
    ("Se", "pbe", 2.0, 2.0),
    ("Xe", "lda", 1.07, 2.0),
    ("Xe", "pbe", 1.07, 2.0),
];

/// `(n_bessel, local_shift)` of a dataset whose s channel is repaired.
pub fn s_channel(symbol: &str, functional: &str) -> Option<(usize, f64)> {
    S_CHANNEL
        .iter()
        .find(|e| e.0 == symbol && e.1 == functional)
        .map(|e| (e.2, e.3))
}

/// `(d cutoff in Bohr, second reference energy in Hartree)` of a repaired semicore d.
pub fn d_channel(symbol: &str, functional: &str) -> Option<(f64, f64)> {
    D_CHANNEL
        .iter()
        .find(|e| e.0 == symbol && e.1 == functional)
        .map(|e| (e.2, e.3))
}

/// The local shift of a repaired dataset (the s-channel repair's).
pub fn local_shift(symbol: &str, functional: &str) -> Option<f64> {
    s_channel(symbol, functional).map(|e| e.1)
}

/// The second reference offset of a repaired dataset (the d-channel repair's).
pub fn energy_offset(symbol: &str, functional: &str) -> Option<f64> {
    d_channel(symbol, functional).map(|e| e.1)
}

/// Every repaired dataset is built unitary.
pub fn norm_deficit(symbol: &str, functional: &str) -> Option<f64> {
    (s_channel(symbol, functional).is_some() || d_channel(symbol, functional).is_some())
        .then_some(0.0)
}

/// Every s-channel entry, for the policy-table freeze.
pub fn s_entries() -> impl Iterator<Item = (&'static str, &'static str, usize, f64)> {
    S_CHANNEL.iter().copied()
}

/// Every d-channel entry, for the policy-table freeze.
pub fn d_entries() -> impl Iterator<Item = (&'static str, &'static str, f64, f64)> {
    D_CHANNEL.iter().copied()
}
