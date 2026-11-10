//! Per-element and per-dataset defaults of the generator: Bessel counts, norm deficits, second
//! reference energies, local shifts, the frozen and semicore subshells, the empty channels and the
//! reference configurations. Data copied from the generator this replaces; the policy-table
//! freeze (`test/migration/reference/paw_recipe`) compares every entry.
//!
//! The reasons the entries exist, from the scans that measured them:
//!
//! * **Bessel count.** The norm-constrained minimization has two nearly degenerate minima for
//!   lithium's 2s pair (one nodeless, one with a node inside `r_cut`); at 8 functions the PBE pair
//!   has a node at 1.22 Bohr and broke LiH (minimum at 2.4 Angstrom instead of 1.6), at 9 it is
//!   nodeless. Fe, Co and Ni (and the heavier d-block, Sn through its repair) take 9 for a related
//!   reason: at 8 their s pair represents a hydrogen orbital entering the sphere badly and the
//!   hydride collapses.
//! * **Norm deficit.** H is kept small (a larger deficit grows a ghost in its s channel); lithium
//!   and copper are unitary (copper's PBE dataset at 0.1 passed every atomic check and still
//!   collapsed CuH); Co, Ni and the 4d/5d/6p block follow.
//! * **Energy offset.** Lithium's scattering wave at +1 Ha sits at a pole of the logarithmic
//!   derivative; hydrogen at +0.5 Ha gives a softer second projector; the d block and the heavy
//!   p block take +2 Ha.
//! * **Second references.** A compact shell more than 1 Ha deep put the second reference below
//!   zero and the channel bound a spurious level just above it (the 4f of W-Hg, the 4p of Tc, the
//!   semicore 5p of Ta): at +0.25 Ha each channel binds only its own state. The empty valence p of
//!   the closed-shell s2 metals sits at the valence energy.
//! * **Local shift.** The polynomial continuation of the screened all-electron potential is deep
//!   enough to bind a spurious 1s-like state in the s channel of the first row; the local
//!   potential is raised until the s spectrum has nothing between the bound state and the box.
//! * **Frozen 4f.** The 4f14 of Lu-Rn sits 4-8 Ha deep inside a 2.9-3.9 Bohr sphere; frozen, it
//!   joins the nonlinear core correction and an empty f channel scatters at 0.25 Ha.
//! * **Semicore.** The `(n-1)s (n-1)p` shell of the alkali, alkaline-earth and early d-block
//!   atoms gives the p channel a bound reference and its projectors, and the compact shell sets
//!   small spheres.
//! * **Highest channel.** The heavier alkali and alkaline-earth atoms have a low-lying empty d,
//!   lithium and beryllium an empty p (their local potential alone scatters it 0.28-0.42 rad
//!   off), hydrogen a p channel for the bonded neighbor's tail.
//! * **Core radius.** Lithium's 1s overlaps the 2s out to 1.5 Bohr; pseudized at the
//!   compensation radius 2.6 its smooth core holds 0.02 electrons.

/// Spherical Bessel functions per smooth partial wave.
pub const N_BESSEL: usize = 8;
/// Wave-vector cutoff of the residual kinetic energy (Bohr^-1).
pub const Q_CUT: f64 = 5.0;
/// Second reference energy above the bound state (Hartree).
pub const ENERGY_OFFSET: f64 = 1.0;
/// Fraction of the all-electron inner norm the smooth waves give up.
pub const NORM_DEFICIT: f64 = 0.1;
/// Default augmentation radius as a multiple of the outermost maximum of `rR`.
pub const RC_FACTOR: f64 = 1.3;
/// Local-potential radius as a multiple of the largest augmentation radius.
pub const LOCAL_FACTOR: f64 = 0.9;
/// Cutoff of a channel holding a semicore state, as a fraction of the valence s orbital's peak
/// radius: the default rule would take it from the compact semicore shell (sodium's 2s gave 0.79
/// Bohr and a ghost at -7.8 Ha, while 1.8-2.6 Bohr are all ghost-free).
pub const SEMICORE_CUTOFF_FRACTION: f64 = 0.7;
/// First reference energy (Hartree) of the empty channel a frozen shell leaves.
pub const FROZEN_SCATTERING_ENERGY: f64 = 0.25;
/// First references an empty channel falls back to when the anchor's smooth waves gain a node:
/// the channel's own bound level, then these scattering energies (Hartree).
pub const EXTRA_ANCHOR_ENERGIES: [f64; 3] = [0.0, 0.1, 0.25];
/// Half-width of the fit that reads a gradient-corrected potential's derivatives at the local
/// radius, as a fraction of that radius: its potential is a cubic spline through derivative nodes,
/// and a fit over the default 25 grid points spans two or three of them and reads their kinks as a
/// fourth derivative (PBE aluminum came out 13.4 Ha deep at the origin, LDA 1.5).
pub const GGA_LOCAL_FIT_FRACTION: f64 = 0.15;
/// A fallback cutoff beyond this fraction of the radial grid is a box state, not an atomic one.
pub const CUTOFF_GRID_FRACTION: f64 = 0.5;
/// Convergence of the reference atom's density.
pub const REFERENCE_TOLERANCE: f64 = 1e-7;
/// Mixing of the reference atom's SCF.
pub const REFERENCE_MIXING: f64 = 0.25;
/// Iterations the reference atom's SCF may take before the atom is refused as unconverged. An
/// atom stops at its first converged iteration, so the cap changes no converged atom; it decides
/// only whether a slow one is refused. The atom-solver scheme (the legacy seam) needs 251 for
/// the PBE Dirac polonium atom (the generator's earlier cap of 200 refused it; the variational
/// scheme converges in 27); an atom that does not converge in 600 is still refused by name.
pub const REFERENCE_MAX_ITERATIONS: usize = 600;
/// Radial extent of the reference atom (Bohr).
pub const R_MAX: f64 = 30.0;

/// Uniform radial grid points for the reference atom of element `Z`: the Numerov partial waves
/// must satisfy the all-electron radial equation to about 1e-7, which with the core scale `1/Z`
/// needs `1500 Z` points out to 30 Bohr (12000 for oxygen, where 6000 leaves a 6e-6 asymmetry).
pub fn generation_points(z: u8) -> usize {
    6000usize.max(1500 * z as usize)
}

/// A subshell `(n, l)`.
pub type Subshell = (usize, usize);

const N_BESSEL_BY_DATASET: &[(&str, &str, usize)] = &[
    ("Ag", "lda", 9),
    ("Ag", "pbe", 9),
    ("Cd", "lda", 9),
    ("Cd", "pbe", 9),
    ("Co", "lda", 9),
    ("Co", "pbe", 9),
    ("Cu", "lda", 9),
    ("Cu", "pbe", 9),
    ("Fe", "lda", 9),
    ("Fe", "pbe", 9),
    ("Hg", "lda", 9),
    ("Hg", "pbe", 9),
    ("Li", "pbe", 9),
    ("Mn", "lda", 9),
    ("Mn", "pbe", 9),
    ("Ni", "lda", 9),
    ("Ni", "pbe", 9),
    ("Pb", "lda", 9),
    ("Pb", "pbe", 9),
    ("Po", "lda", 9),
    ("Po", "pbe", 9),
    ("Zn", "lda", 9),
    ("Zn", "pbe", 9),
];

const NORM_DEFICITS: &[(&str, f64)] = &[
    ("Ag", 0.0),
    ("C", 0.1),
    ("Cd", 0.0),
    ("Co", 0.0),
    ("Cu", 0.0),
    ("F", 0.15),
    ("Fe", 0.0),
    ("H", 0.05),
    ("Hg", 0.0),
    ("Li", 0.0),
    ("Mg", 0.065),
    ("Mn", 0.0),
    ("N", 0.15),
    ("Ni", 0.0),
    ("O", 0.15),
    ("Pb", 0.0),
    ("Po", 0.0),
    ("Zn", 0.0),
];

const ENERGY_OFFSETS: &[(&str, f64)] = &[
    ("Ag", 2.0),
    ("Cd", 2.0),
    ("Co", 2.0),
    ("Cu", 2.0),
    ("Fe", 2.0),
    ("H", 0.5),
    ("In", 2.0),
    ("Li", 0.5),
    ("Mn", 2.0),
    ("Ni", 2.0),
    ("Pb", 2.0),
    ("Po", 2.0),
    ("Sb", 2.0),
    ("Sc", 2.0),
    ("Sn", 2.0),
    ("Ti", 2.0),
    ("V", 2.0),
    ("Zn", 2.0),
];

const LOCAL_SHIFTS: &[(&str, f64)] = &[("C", 12.0), ("F", 8.0), ("N", 10.0), ("O", 6.0)];

const CORE_RADII: &[(&str, f64)] = &[("Li", 1.5)];

const SECOND_REFERENCES: &[(&str, &str, &[(usize, f64)])] = &[
    ("Au", "lda", &[(3, 0.25)]),
    ("Au", "pbe", &[(3, 0.25)]),
    ("Hg", "lda", &[(3, 0.25)]),
    ("Hg", "pbe", &[(3, 0.25)]),
    ("Ir", "lda", &[(3, 0.25)]),
    ("Ir", "pbe", &[(3, 0.25)]),
    ("Mg", "lda", &[(1, -0.12)]),
    ("Mg", "pbe", &[(1, -0.12)]),
    ("Os", "lda", &[(3, 0.25)]),
    ("Os", "pbe", &[(3, 0.25)]),
    ("Pt", "lda", &[(3, 0.25)]),
    ("Pt", "pbe", &[(3, 0.25)]),
    ("Re", "lda", &[(3, 0.25)]),
    ("Re", "pbe", &[(3, 0.25)]),
    ("Ta", "lda", &[(1, 0.25)]),
    ("Tc", "lda", &[(1, 0.25)]),
    ("Tc", "pbe", &[(1, 0.25)]),
    ("W", "lda", &[(3, 0.25)]),
    ("W", "pbe", &[(3, 0.25)]),
    ("Zn", "pbe", &[(1, -0.16)]),
];

const FROZEN: &[&str] = &[
    "Lu", "Hf", "Ta", "W", "Re", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
];

const SEMICORE: &[(&str, &[Subshell])] = &[
    ("Ac", &[(6, 0), (6, 1)]),
    ("Ag", &[(4, 0), (4, 1)]),
    ("Au", &[(5, 0), (5, 1)]),
    ("Ba", &[(5, 0), (5, 1)]),
    ("Ca", &[(3, 0), (3, 1)]),
    ("Cd", &[(4, 0), (4, 1)]),
    ("Ce", &[(5, 0), (5, 1)]),
    ("Co", &[(3, 0), (3, 1)]),
    ("Cs", &[(5, 0), (5, 1)]),
    ("Cu", &[(3, 0), (3, 1)]),
    ("Dy", &[(5, 0), (5, 1)]),
    ("Er", &[(5, 0), (5, 1)]),
    ("Eu", &[(5, 0), (5, 1)]),
    ("Fe", &[(3, 0), (3, 1)]),
    ("Gd", &[(5, 0), (5, 1)]),
    ("Hf", &[(5, 0), (5, 1)]),
    ("Hg", &[(5, 0), (5, 1)]),
    ("Ho", &[(5, 0), (5, 1)]),
    ("Ir", &[(5, 0), (5, 1)]),
    ("K", &[(3, 0), (3, 1)]),
    ("La", &[(5, 0), (5, 1)]),
    ("Lu", &[(5, 0), (5, 1)]),
    ("Mg", &[(2, 0), (2, 1)]),
    ("Mn", &[(3, 0), (3, 1)]),
    ("Mo", &[(4, 0), (4, 1)]),
    ("Na", &[(2, 0), (2, 1)]),
    ("Nb", &[(4, 0), (4, 1)]),
    ("Nd", &[(5, 0), (5, 1)]),
    ("Ni", &[(3, 0), (3, 1)]),
    ("Pa", &[(6, 0), (6, 1)]),
    ("Pb", &[(5, 0), (5, 1)]),
    ("Pm", &[(5, 0), (5, 1)]),
    ("Po", &[(5, 0), (5, 1)]),
    ("Pr", &[(5, 0), (5, 1)]),
    ("Pt", &[(5, 0), (5, 1)]),
    ("Ra", &[(6, 0), (6, 1)]),
    ("Rb", &[(4, 0), (4, 1)]),
    ("Re", &[(5, 0), (5, 1)]),
    ("Sc", &[(3, 0), (3, 1)]),
    ("Sm", &[(5, 0), (5, 1)]),
    ("Sr", &[(4, 0), (4, 1)]),
    ("Ta", &[(5, 0), (5, 1)]),
    ("Tb", &[(5, 0), (5, 1)]),
    ("Tc", &[(4, 0), (4, 1)]),
    ("Th", &[(6, 0), (6, 1)]),
    ("Ti", &[(3, 0), (3, 1)]),
    ("Tm", &[(5, 0), (5, 1)]),
    ("U", &[(6, 0), (6, 1)]),
    ("V", &[(3, 0), (3, 1)]),
    ("W", &[(5, 0), (5, 1)]),
    ("Y", &[(4, 0), (4, 1)]),
    ("Yb", &[(5, 0), (5, 1)]),
    ("Zn", &[(3, 0), (3, 1)]),
    ("Zr", &[(4, 0), (4, 1)]),
];

const HIGHEST_CHANNEL: &[(&str, usize)] = &[
    ("Ac", 3),
    ("Ba", 3),
    ("Be", 1),
    ("Ca", 2),
    ("Cs", 3),
    ("H", 1),
    ("K", 2),
    ("La", 3),
    ("Li", 1),
    ("Ra", 3),
    ("Rb", 2),
    ("Sr", 2),
    ("Th", 3),
];

const REFERENCE_CONFIGURATIONS: &[(&str, &[(Subshell, f64)])] = &[
    ("Ce", &[((4, 3), 1.0), ((5, 2), 1.0)]),
    ("Dy", &[((4, 3), 9.0), ((5, 2), 1.0)]),
    ("Er", &[((4, 3), 11.0), ((5, 2), 1.0)]),
    ("Eu", &[((4, 3), 6.0), ((5, 2), 1.0)]),
    ("Gd", &[((4, 3), 7.0), ((5, 2), 1.0)]),
    ("Ho", &[((4, 3), 10.0), ((5, 2), 1.0)]),
    ("Nd", &[((4, 3), 3.0), ((5, 2), 1.0)]),
    ("Pa", &[((5, 3), 2.0), ((6, 2), 1.0)]),
    ("Pm", &[((4, 3), 4.0), ((5, 2), 1.0)]),
    ("Pr", &[((4, 3), 2.0), ((5, 2), 1.0)]),
    ("Sm", &[((4, 3), 5.0), ((5, 2), 1.0)]),
    ("Tb", &[((4, 3), 8.0), ((5, 2), 1.0)]),
    ("Tm", &[((4, 3), 12.0), ((5, 2), 1.0)]),
    ("U", &[((5, 3), 3.0), ((6, 2), 1.0)]),
    ("Yb", &[((4, 3), 13.0), ((5, 2), 1.0)]),
];

fn find<T: Copy>(table: &[(&str, T)], symbol: &str) -> Option<T> {
    table.iter().find(|e| e.0 == symbol).map(|e| e.1)
}

/// The Bessel count fixed for a dataset (before its s-channel repair's).
pub fn n_bessel(symbol: &str, functional: &str) -> Option<usize> {
    N_BESSEL_BY_DATASET
        .iter()
        .find(|e| e.0 == symbol && e.1 == functional)
        .map(|e| e.2)
}

/// The norm deficit of an element.
pub fn norm_deficit(symbol: &str) -> Option<f64> {
    find(NORM_DEFICITS, symbol)
}

/// The second reference energy above the bound state of an element (Hartree).
pub fn energy_offset(symbol: &str) -> Option<f64> {
    find(ENERGY_OFFSETS, symbol)
}

/// The raise of the local potential at the origin of an element (Hartree).
pub fn local_shift(symbol: &str) -> Option<f64> {
    find(LOCAL_SHIFTS, symbol)
}

/// The radius at which an element's frozen core is pseudized, when not the compensation radius.
pub fn core_radius(symbol: &str) -> Option<f64> {
    find(CORE_RADII, symbol)
}

/// Absolute second reference energies `(l, energy)` of the single channels of a dataset.
pub fn second_references(symbol: &str, functional: &str) -> &'static [(usize, f64)] {
    SECOND_REFERENCES
        .iter()
        .find(|e| e.0 == symbol && e.1 == functional)
        .map_or(&[], |e| e.2)
}

/// The filled semicore d shells generated into the frozen core (2026-11-09). In valence, the 3d10
/// of Br and Kr and the 4d10 of Te, I and Xe sit 1.5-3.3 Ha deep inside 1.0-1.1 Bohr spheres: their
/// smooth waves kept 1.6-8.2 Ha of kinetic energy above `q_cut`, and HBr moved 186 eV between grid
/// spacings of 0.20 and 0.15 Angstrom. Frozen, they join the core correction and an empty d
/// channel scatters at the frozen-shell energy: pbe-sr HBr then moves 2.0 meV over that step, and
/// its binding energy is 4.012 eV against 4.021 eV all-electron (X2C-PBE, def2-QZVPP).
const FROZEN_D: &[(&str, &[Subshell])] = &[
    ("Br", &[(3, 2)]),
    ("I", &[(4, 2)]),
    ("Kr", &[(3, 2)]),
    ("Te", &[(4, 2)]),
    ("Xe", &[(4, 2)]),
];

/// Occupied valence subshells generated into the frozen core by default.
pub fn frozen_subshells(symbol: &str) -> &'static [Subshell] {
    if FROZEN.contains(&symbol) {
        &[(4, 3)]
    } else {
        find(FROZEN_D, symbol).unwrap_or(&[])
    }
}

/// Filled core subshells generated into the valence by default.
pub fn semicore_subshells(symbol: &str) -> &'static [Subshell] {
    find(SEMICORE, symbol).unwrap_or(&[])
}

/// The highest channel an element's dataset carries at least, empty ones added above its valence.
pub fn highest_channel(symbol: &str) -> Option<usize> {
    find(HIGHEST_CHANNEL, symbol)
}

/// Occupancy overrides laid over the relaxed ground-state configuration of the reference atom
/// (Pa and U keep their 5f in valence and so had no occupied 6d).
pub fn reference_configuration(symbol: &str) -> &'static [(Subshell, f64)] {
    find(REFERENCE_CONFIGURATIONS, symbol).unwrap_or(&[])
}

/// Every element of the per-element scalar tables, for the policy-table freeze.
pub fn scalar_entries() -> [(&'static str, &'static [(&'static str, f64)]); 4] {
    [
        ("norm_deficit", NORM_DEFICITS),
        ("energy_offset", ENERGY_OFFSETS),
        ("local_shift", LOCAL_SHIFTS),
        ("core_radius", CORE_RADII),
    ]
}

/// Every `(symbol, functional, count)` of the Bessel table, for the policy-table freeze.
pub fn n_bessel_entries() -> impl Iterator<Item = (&'static str, &'static str, usize)> {
    N_BESSEL_BY_DATASET.iter().copied()
}

/// Every `(symbol, functional, references)` of the second-reference table.
pub fn second_reference_entries(
) -> impl Iterator<Item = (&'static str, &'static str, &'static [(usize, f64)])> {
    SECOND_REFERENCES.iter().copied()
}
