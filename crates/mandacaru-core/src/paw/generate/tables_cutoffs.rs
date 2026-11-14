//! Augmentation radii of the policy tables (Bohr): per element, and per dataset on top.
//!
//! `by_element` carries the radii that the molecular tests showed to be resolvable on a
//! 0.25-0.30 Angstrom grid (hydrogen's compact 1.0 Bohr s sphere beside its empty p channel;
//! lithium's 2.6; the second-row 1.40-1.50 spheres; tin's shortened s sphere; the d spheres of
//! the 3d metals, Pb and Po, Zn, Ag, Cd and Hg at 2.2 Bohr, the vanadium way: semicore s p in
//! valence, a larger d sphere and a second reference at +2 Ha, which the grid projection of a
//! filtered 3d function needs). `by_dataset` holds the radius of every channel of the repaired
//! datasets, as each scan built it: a channel left to the generic rule can come out elsewhere
//! (Os-PBE's f at 0.53 Bohr, 1.2 rad off in phase). A dataset entry is laid over the element's.
//! Both tables are data copied from the generator they replace (the policy-table freeze of
//! `test/migration/reference/paw_recipe` compares every entry).

/// `(l, r_cut)` pairs of one entry.
pub type Radii = &'static [(usize, f64)];

const BY_ELEMENT: &[(&str, Radii)] = &[
    ("Ag", &[(2, 2.2)]),
    ("Al", &[(0, 2.35), (1, 2.35)]),
    ("Be", &[(0, 1.97), (1, 1.97)]),
    ("C", &[(0, 1.5), (1, 1.5)]),
    ("Cd", &[(2, 2.2)]),
    ("Ce", &[(0, 2.2), (1, 2.2), (2, 2.4), (3, 2.0)]),
    ("Co", &[(2, 2.2)]),
    ("Cu", &[(0, 1.8), (1, 1.8), (2, 2.2)]),
    ("Dy", &[(0, 2.2), (1, 2.2), (2, 2.4), (3, 2.0)]),
    ("Er", &[(0, 2.2), (1, 2.2), (2, 2.4), (3, 2.0)]),
    ("Eu", &[(0, 2.2), (1, 2.2), (2, 2.4), (3, 2.0)]),
    ("F", &[(0, 1.4), (1, 1.4)]),
    ("Fe", &[(2, 2.2)]),
    ("Gd", &[(0, 2.2), (1, 2.2), (2, 2.4), (3, 2.0)]),
    ("H", &[(0, 1.0)]),
    ("Hg", &[(2, 2.2)]),
    ("Ho", &[(0, 2.2), (1, 2.2), (2, 2.4), (3, 2.0)]),
    ("Li", &[(0, 2.6)]),
    ("Mn", &[(2, 2.2)]),
    ("N", &[(0, 1.45), (1, 1.45)]),
    ("Nd", &[(0, 2.2), (1, 2.2), (2, 2.4), (3, 2.0)]),
    ("Ne", &[(0, 1.5), (1, 1.5)]),
    ("Ni", &[(0, 1.8), (1, 1.8), (2, 2.2)]),
    ("O", &[(0, 1.45), (1, 1.45)]),
    ("Pb", &[(2, 2.2)]),
    ("Pm", &[(0, 2.2), (1, 2.2), (2, 2.4), (3, 2.0)]),
    ("Po", &[(2, 2.2)]),
    ("Pr", &[(0, 2.2), (1, 2.2), (2, 2.4), (3, 2.0)]),
    ("Rn", &[(2, 2.2)]),
    ("Sc", &[(2, 2.07)]),
    ("Si", &[(0, 2.2), (1, 2.2)]),
    ("Sm", &[(0, 2.2), (1, 2.2), (2, 2.4), (3, 2.0)]),
    ("Sn", &[(0, 2.33)]),
    ("Tb", &[(0, 2.2), (1, 2.2), (2, 2.4), (3, 2.0)]),
    ("Ti", &[(2, 1.86)]),
    ("Tm", &[(0, 2.2), (1, 2.2), (2, 2.4), (3, 2.0)]),
    ("V", &[(2, 1.67)]),
    ("Yb", &[(0, 2.2), (1, 2.2), (2, 2.4), (3, 2.0)]),
    ("Zn", &[(0, 1.8), (1, 1.8), (2, 2.2)]),
];

const BY_DATASET: &[(&str, &str, Radii)] = &[
    ("Ac", "lda", &[(0, 2.6), (1, 2.6), (2, 2.6), (3, 2.6)]),
    ("Ac", "pbe", &[(0, 2.6), (1, 2.6), (2, 2.6), (3, 2.6)]),
    ("As", "lda", &[(0, 2.081), (1, 2.564), (2, 2.2)]),
    ("As", "pbe", &[(0, 2.095), (1, 2.582), (2, 2.2)]),
    ("Au", "lda", &[(0, 2.4), (1, 1.58), (2, 2.4), (3, 2.4)]),
    ("Au", "pbe", &[(0, 2.4), (1, 1.58), (2, 2.4), (3, 2.4)]),
    ("Ba", "lda", &[(0, 3.2), (1, 3.2), (2, 3.2), (3, 3.2)]),
    ("Ba", "pbe", &[(0, 3.2), (1, 3.2), (2, 3.2), (3, 3.2)]),
    ("Bi", "lda", &[(0, 2.294), (1, 3.183), (2, 2.2), (3, 3.183)]),
    ("Bi", "pbe", &[(0, 2.294), (1, 3.183), (2, 2.2), (3, 3.183)]),
    ("Br", "lda", &[(0, 1.829), (1, 2.159), (2, 1.006)]),
    ("Br", "pbe", &[(0, 1.839), (1, 2.171), (2, 1.009)]),
    // Cr (2026-11-10): the 1.06 Bohr d sphere kept 1.15 Ha above q_cut and left no working grid
    // spacing; the 3.38-3.41 Bohr s sphere held the H of a 1.70 Angstrom Cr-H bond. At s 2.9 and
    // d 2.0 Bohr every gate passes (d 0.016 Ha, working spacing 0.12 Angstrom).
    ("Cr", "lda", &[(0, 2.9), (2, 2.0)]),
    ("Cr", "pbe", &[(0, 2.9), (2, 2.0)]),
    ("Cs", "lda", &[(0, 3.0), (1, 3.0), (2, 3.0), (3, 3.0)]),
    ("Cs", "pbe", &[(0, 3.0), (1, 3.0), (2, 3.0), (3, 3.0)]),
    ("Fr", "lda", &[(0, 2.9), (1, 2.9), (2, 2.9), (3, 2.9)]),
    ("Fr", "pbe", &[(0, 2.9), (1, 2.9), (2, 2.9), (3, 2.9)]),
    ("Ga", "lda", &[(0, 2.37), (1, 2.37), (2, 2.6)]),
    ("Ga", "pbe", &[(0, 2.37), (1, 2.37), (2, 2.6)]),
    ("Ge", "lda", &[(0, 2.25), (1, 2.34), (2, 2.0)]),
    ("Ge", "pbe", &[(0, 2.25), (1, 2.34), (2, 2.0)]),
    ("I", "lda", &[(0, 2.194), (1, 2.628), (2, 1.113)]),
    ("I", "pbe", &[(0, 2.206), (1, 2.641), (2, 1.113)]),
    ("In", "lda", &[(0, 2.68), (1, 2.68), (2, 2.0)]),
    ("In", "pbe", &[(0, 2.68), (1, 2.68), (2, 2.0)]),
    ("Ir", "lda", &[(0, 2.4), (1, 1.67), (2, 2.4), (3, 2.4)]),
    ("Ir", "pbe", &[(0, 2.4), (1, 1.67), (2, 2.4), (3, 2.4)]),
    ("Kr", "lda", &[(0, 1.727), (1, 2.011), (2, 1.066)]),
    ("Kr", "pbe", &[(0, 1.736), (1, 2.019), (2, 1.07)]),
    ("La", "lda", &[(0, 2.5), (1, 2.5), (2, 2.5), (3, 2.5)]),
    ("La", "pbe", &[(0, 2.6), (1, 2.6), (2, 2.6), (3, 2.6)]),
    ("Lu", "lda", &[(0, 2.0), (1, 2.0), (2, 2.2), (3, 2.2)]),
    ("Lu", "pbe", &[(0, 2.0), (1, 2.0), (2, 2.2), (3, 2.2)]),
    ("Os", "lda", &[(0, 2.871), (2, 3.19), (3, 3.19)]),
    ("Os", "pbe", &[(0, 3.041), (2, 3.201), (3, 3.201)]),
    ("Pa", "lda", &[(0, 2.8), (1, 2.8), (2, 2.8), (3, 2.4)]),
    ("Pa", "pbe", &[(0, 2.8), (1, 2.8), (2, 2.8), (3, 2.4)]),
    ("Pd", "lda", &[(0, 3.178), (2, 2.2)]),
    ("Pd", "pbe", &[(0, 3.196), (2, 2.2)]),
    ("Pt", "lda", &[(0, 3.005), (2, 3.005), (3, 3.005)]),
    ("Pt", "pbe", &[(0, 3.011), (2, 3.011), (3, 3.011)]),
    ("Ra", "lda", &[(0, 2.9), (1, 2.9), (2, 2.9), (3, 2.9)]),
    ("Ra", "pbe", &[(0, 2.9), (1, 2.9), (2, 2.9), (3, 2.9)]),
    ("Re", "lda", &[(0, 2.968), (2, 3.298), (3, 3.298)]),
    ("Re", "pbe", &[(0, 2.981), (2, 3.312), (3, 3.312)]),
    ("Rh", "lda", &[(0, 3.279), (2, 2.2)]),
    ("Rh", "pbe", &[(0, 3.299), (2, 2.2)]),
    ("Ru", "lda", &[(0, 3.39), (2, 2.2)]),
    ("Ru", "pbe", &[(0, 3.412), (2, 2.2)]),
    ("Sb", "lda", &[(0, 2.439), (1, 3.026), (2, 2.0)]),
    ("Sb", "pbe", &[(0, 2.46), (1, 3.05), (2, 2.0)]),
    ("Se", "lda", &[(0, 1.945), (1, 2.34), (2, 2.3)]),
    ("Se", "pbe", &[(0, 1.958), (1, 2.353), (2, 2.3)]),
    ("Sn", "lda", &[(0, 2.33), (1, 2.63), (2, 2.0)]),
    ("Sn", "pbe", &[(0, 2.33), (1, 2.63), (2, 2.0)]),
    ("Th", "lda", &[(0, 3.0), (1, 3.0), (2, 3.0), (3, 3.0)]),
    ("Th", "pbe", &[(0, 3.0), (1, 3.0), (2, 3.0), (3, 3.0)]),
    ("Tl", "lda", &[(0, 2.68), (1, 2.73), (2, 2.2), (3, 2.73)]),
    ("Tl", "pbe", &[(0, 2.68), (1, 2.73), (2, 2.2), (3, 2.73)]),
    ("U", "lda", &[(0, 2.6), (1, 2.6), (2, 2.6), (3, 2.4)]),
    ("U", "pbe", &[(0, 2.6), (1, 2.6), (2, 2.6), (3, 2.4)]),
    ("W", "lda", &[(0, 2.97), (1, 1.78), (2, 3.3), (3, 3.3)]),
    ("W", "pbe", &[(0, 2.97), (1, 1.78), (2, 3.3), (3, 3.3)]),
    ("Xe", "lda", &[(0, 2.093), (1, 2.476), (2, 1.07)]),
    ("Xe", "pbe", &[(0, 2.104), (1, 2.487), (2, 1.07)]),
];

/// The element's radii, if the table has any.
pub fn by_element(symbol: &str) -> Option<Radii> {
    BY_ELEMENT.iter().find(|e| e.0 == symbol).map(|e| e.1)
}

/// The radii of a repaired dataset (`functional` is `"lda"` or `"pbe"`).
pub fn by_dataset(symbol: &str, functional: &str) -> Option<Radii> {
    BY_DATASET
        .iter()
        .find(|e| e.0 == symbol && e.1 == functional)
        .map(|e| e.2)
}

/// Every element entry, for the policy-table freeze.
pub fn element_entries() -> impl Iterator<Item = (&'static str, Radii)> {
    BY_ELEMENT.iter().copied()
}

/// Every dataset entry, for the policy-table freeze.
pub fn dataset_entries() -> impl Iterator<Item = (&'static str, &'static str, Radii)> {
    BY_DATASET.iter().copied()
}
