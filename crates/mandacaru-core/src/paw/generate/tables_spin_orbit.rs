//! The measured calibration of the spin-orbit coupling of the Dirac datasets (gate G13 of the
//! 2026-11 regeneration, `dev_plan/PAW_REGENERATION_PLAN.md`).
//!
//! A Dirac dataset is the scalar (`j`-averaged) construction plus an `L.S` coupling built on the
//! two `j` branches' own projectors (`generate::spin_orbit`). The engine hands those projectors
//! the *scalar* smooth waves of its basis, not the branch waves they were built for, so the
//! splitting of a heavy shell it produces is smaller than the all-electron Dirac one (bismuth 6p:
//! 1.61 eV against 1.84 eV; lead 6p 1.17 against 1.36; the deficit grows with the relativistic
//! contraction of the shell). The engine's isolated-atom splitting is linear in the coupling, so one number
//! per `(element, functional, l)` closes the gap: `scale = Dirac splitting / engine splitting`
//! of the uncalibrated record, measured with the engine's own atom (a periodic box, Gamma,
//! DZP, Fermi-Dirac 0.02 eV) and checked again on the calibrated file. The scale multiplies the
//! stored `coupling` of that `l` (the operator the engine reads); the diagnostic arrays
//! (`coupling_screened`, `overlap_coupling`, the averages) keep the branch construction, whose
//! levels the ghost checks verify. A shell absent from the table is within 2 percent or 10 meV
//! as built.
//!
//! Data, not a rule: regenerating with another recipe or functional needs the table measured
//! again (`scratchpad` procedure in the plan, `G13`).

/// `(symbol, functional, l, scale)`.
/// The trailing word of each entry says how the scale was obtained: `engine` (measured on the
/// element's own atom), `twin` (measured on the other functional's file of the element, whose own
/// atom does not converge) or `proxy` (the first-order matrix element of the stored coupling on the
/// scalar smooth wave, which reproduces the engine within 1-3 percent where both exist, for atoms
/// the engine's spinor SCF does not converge: open d and f shells).
#[rustfmt::skip]
const SCALES: &[(&str, &str, usize, f64)] = &[
    ("Ac", "lda", 1, 1.1077),  // engine
    ("Ac", "pbe", 1, 1.1093),  // engine
    ("Ag", "lda", 1, 1.0246),  // engine
    ("Ag", "lda", 2, 1.04),  // engine
    ("Ag", "pbe", 1, 1.0242),  // engine
    ("Ag", "pbe", 2, 1.0392),  // engine
    ("At", "lda", 1, 1.1219),  // engine
    ("At", "pbe", 1, 1.127),  // engine
    ("Au", "lda", 1, 1.0506),  // engine
    ("Au", "lda", 2, 1.0274),  // engine
    ("Au", "pbe", 1, 1.0515),  // engine
    ("Au", "pbe", 2, 1.0282),  // engine
    ("Ba", "lda", 1, 1.0241),  // engine
    ("Ba", "pbe", 1, 1.0245),  // engine
    ("Bi", "lda", 1, 1.1416),  // engine
    ("Bi", "pbe", 1, 1.1427),  // engine
    ("Ce", "lda", 1, 1.0386),  // proxy
    ("Ce", "pbe", 1, 1.0378),  // proxy
    ("Cs", "lda", 1, 1.0318),  // engine
    ("Cs", "pbe", 1, 1.0334),  // engine
    ("Dy", "lda", 1, 1.0407),  // proxy
    ("Dy", "pbe", 1, 1.0397),  // proxy
    ("Er", "lda", 1, 1.0418),  // proxy
    ("Er", "pbe", 1, 1.0415),  // proxy
    ("Eu", "lda", 1, 1.041),  // proxy
    ("Eu", "pbe", 1, 1.04),  // proxy
    ("Fr", "lda", 1, 1.1115),  // engine
    ("Fr", "pbe", 1, 1.1146),  // engine
    ("Gd", "lda", 1, 1.0434),  // proxy
    ("Gd", "pbe", 1, 1.0407),  // proxy
    ("Hf", "lda", 1, 1.0339),  // engine
    ("Hf", "pbe", 1, 1.0333),  // engine
    ("Hg", "lda", 1, 1.0315),  // engine
    ("Hg", "pbe", 1, 1.0331),  // engine
    ("Ho", "lda", 1, 1.0432),  // proxy
    ("Ho", "pbe", 1, 1.0402),  // proxy
    ("Ir", "lda", 1, 1.0419),  // engine
    ("Ir", "pbe", 1, 1.0419),  // twin
    ("La", "lda", 1, 1.0355),  // proxy
    ("La", "pbe", 1, 1.0312),  // proxy
    ("Lu", "lda", 1, 1.0346),  // engine
    ("Lu", "pbe", 1, 1.0316),  // engine
    ("Mg", "lda", 1, 1.0935),  // engine
    ("Mg", "pbe", 1, 1.0906),  // engine
    ("Na", "lda", 1, 1.1311),  // engine
    ("Na", "pbe", 1, 1.1291),  // engine
    ("Nd", "lda", 1, 1.0395),  // proxy
    ("Nd", "pbe", 1, 1.0387),  // proxy
    ("Pa", "lda", 1, 1.0686),  // engine
    ("Pa", "pbe", 1, 1.0686),  // twin
    ("Pb", "lda", 1, 1.153),  // engine
    ("Pb", "pbe", 1, 1.1627),  // engine
    ("Pd", "lda", 2, 1.0839),  // engine
    ("Pd", "pbe", 2, 1.0817),  // engine
    ("Pm", "lda", 1, 1.04),  // proxy
    ("Pm", "pbe", 1, 1.0391),  // proxy
    ("Po", "lda", 1, 1.1457),  // engine
    ("Po", "pbe", 1, 1.1483),  // engine
    ("Pr", "lda", 1, 1.0391),  // proxy
    ("Pr", "pbe", 1, 1.0383),  // proxy
    ("Pt", "lda", 1, 1.0497),  // engine
    ("Pt", "pbe", 1, 1.0504),  // engine
    ("Ra", "lda", 1, 1.0889),  // engine
    ("Ra", "pbe", 1, 1.094),  // engine
    ("Re", "lda", 1, 1.044),  // engine
    ("Re", "pbe", 1, 1.0403),  // engine
    ("Rh", "lda", 2, 1.0744),  // engine
    ("Rn", "lda", 1, 1.1456),  // engine
    ("Rn", "pbe", 1, 1.1455),  // engine
    ("Ru", "lda", 2, 1.0782),  // engine
    ("Ru", "pbe", 2, 1.0782),  // twin
    ("Sb", "lda", 1, 1.0288),  // engine
    ("Sb", "pbe", 1, 1.0335),  // engine
    ("Sm", "lda", 1, 1.0404),  // proxy
    ("Sm", "pbe", 1, 1.0395),  // proxy
    ("Sn", "lda", 1, 1.0242),  // engine
    ("Sn", "pbe", 1, 1.034),  // engine
    ("Ta", "lda", 1, 1.0327),  // engine
    ("Ta", "pbe", 1, 1.0318),  // engine
    ("Tb", "lda", 1, 1.0399),  // proxy
    ("Tb", "pbe", 1, 1.0401),  // proxy
    ("Tc", "lda", 1, 1.0215),  // engine
    ("Tc", "pbe", 1, 1.0223),  // engine
    ("Th", "lda", 1, 1.0402),  // engine
    ("Th", "pbe", 1, 1.0441),  // engine
    ("Tl", "lda", 1, 1.1135),  // engine
    ("Tl", "pbe", 1, 1.1304),  // engine
    ("Tm", "lda", 1, 1.0998),  // twin
    ("Tm", "pbe", 1, 1.0998),  // engine
    ("U", "lda", 1, 1.109),  // engine
    ("U", "pbe", 1, 1.109),  // twin
    ("W", "lda", 1, 1.0454),  // engine
    ("W", "pbe", 1, 1.0457),  // engine
    ("Yb", "lda", 1, 1.0422),  // proxy
    ("Yb", "pbe", 1, 1.0415),  // proxy
];

/// The factor the stored spin-orbit `coupling` of channel `l` carries (`1.0` when none is
/// tabulated).
pub fn scale(symbol: &str, functional: &str, l: usize) -> f64 {
    SCALES
        .iter()
        .find(|e| e.0 == symbol && e.1 == functional && e.2 == l)
        .map_or(1., |e| e.3)
}

/// Every entry, for the table's own test.
pub fn entries() -> impl Iterator<Item = (&'static str, &'static str, usize, f64)> {
    SCALES.iter().copied()
}
