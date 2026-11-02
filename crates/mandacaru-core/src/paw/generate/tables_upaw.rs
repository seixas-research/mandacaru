//! Per-dataset recipes of the UPAW-LCAO family only (norm deficit 0), laid under the caller's own
//! options. PAW-LCAO never reads them: a dataset at a nonzero norm deficit and the unitary one need
//! different spheres, channels and reference configurations.
//!
//! * An empty channel above the valence one (`extra_l`): without projectors the d of Al-Ar and the
//!   f of Te, I and Xe scatter off the local potential alone, 0.10-0.21 rad off the atom (they
//!   pass at 0.01 with projectors). Aluminum's local potential is raised 5 Ha: with its d channel
//!   the longer local radius takes the s projectors' miss of an intruding hydrogen 1s from 0.5 to
//!   1.19, and 5 Ha brings it to 0.08. Argon's d resonance sits 0.47 rad off at the default 1 Ha
//!   second reference and 0.000 at 1.5 Ha.
//! * Tantalum's empty f (4f frozen) is 0.133 rad off 0.75 Ha above its reference at the default
//!   second reference; 0.6 Ha gives 0.008. Radium's empty f has an overlap-operator minimum of
//!   0.60 at the default (huge projectors); at 2 Ha it is 0.97.
//! * Cr, Ru, Rh, Pd the vanadium way: the `(n-1)s (n-1)p` semicore in valence gives the p channel
//!   projectors, without the 40-50 Ha raise and the 3.2-3.4 Bohr s sphere that repair needs.
//!   Pd's s and p spheres are 2.2 Bohr (at 1.7 the local radius is so short that its polynomial
//!   continuation binds a state at -1000 Ha). Francium the radium way (2 Ha, 2.9 Bohr spheres, d
//!   and f channels above the 6s 6p semicore).
//! * The lanthanides with their 4f in valence: the trivalent reference configuration `4f(n) 5d
//!   6s2` (a 5d binds the d channel) and the 5s 5p semicore (a p channel; without it the p
//!   scattered off the local potential and every build bound a p ghost at -0.13 Ha). Dy, Ho and Er
//!   need a 20 Ha local raise.
use super::tables_elements::Subshell;

/// The options a UPAW-LCAO dataset overrides; `None` leaves the generator's default.
#[derive(Debug, Clone, Copy, Default)]
pub struct Upaw {
    pub extra_l: Option<usize>,
    pub local_shift: Option<f64>,
    pub energy_offset: Option<f64>,
    pub n_bessel: Option<usize>,
    pub semicore_subshells: Option<&'static [Subshell]>,
    pub radii: Option<&'static [(usize, f64)]>,
    pub reference_configuration: Option<&'static [(Subshell, f64)]>,
}

const RECIPES: &[(&str, &str, Upaw)] = &[
    (
        "Al",
        "lda",
        Upaw {
            extra_l: Some(1),
            local_shift: Some(5.0),
            ..Upaw::NONE
        },
    ),
    (
        "Ar",
        "lda",
        Upaw {
            extra_l: Some(1),
            energy_offset: Some(1.5),
            ..Upaw::NONE
        },
    ),
    (
        "Ce",
        "lda",
        Upaw {
            semicore_subshells: Some(&[(5, 0), (5, 1)]),
            reference_configuration: Some(&[((4, 3), 1.0), ((5, 2), 1.0), ((6, 0), 2.0)]),
            ..Upaw::NONE
        },
    ),
    (
        "Cl",
        "lda",
        Upaw {
            extra_l: Some(1),
            ..Upaw::NONE
        },
    ),
    (
        "Cr",
        "lda",
        Upaw {
            extra_l: Some(1),
            local_shift: Some(0.0),
            energy_offset: Some(2.0),
            n_bessel: Some(8),
            semicore_subshells: Some(&[(3, 0), (3, 1)]),
            radii: Some(&[(0, 1.819), (1, 1.819), (2, 1.062)]),
            ..Upaw::NONE
        },
    ),
    (
        "Dy",
        "lda",
        Upaw {
            local_shift: Some(20.0),
            semicore_subshells: Some(&[(5, 0), (5, 1)]),
            reference_configuration: Some(&[((4, 3), 9.0), ((5, 2), 1.0), ((6, 0), 2.0)]),
            ..Upaw::NONE
        },
    ),
    (
        "Er",
        "lda",
        Upaw {
            local_shift: Some(20.0),
            semicore_subshells: Some(&[(5, 0), (5, 1)]),
            reference_configuration: Some(&[((4, 3), 11.0), ((5, 2), 1.0), ((6, 0), 2.0)]),
            ..Upaw::NONE
        },
    ),
    (
        "Eu",
        "lda",
        Upaw {
            semicore_subshells: Some(&[(5, 0), (5, 1)]),
            reference_configuration: Some(&[((4, 3), 6.0), ((5, 2), 1.0), ((6, 0), 2.0)]),
            ..Upaw::NONE
        },
    ),
    (
        "Fr",
        "lda",
        Upaw {
            extra_l: Some(2),
            local_shift: Some(8.0),
            energy_offset: Some(2.0),
            n_bessel: Some(10),
            semicore_subshells: Some(&[(6, 0), (6, 1)]),
            radii: Some(&[(0, 2.9), (1, 2.9)]),
            ..Upaw::NONE
        },
    ),
    (
        "Gd",
        "lda",
        Upaw {
            semicore_subshells: Some(&[(5, 0), (5, 1)]),
            reference_configuration: Some(&[((4, 3), 7.0), ((5, 2), 1.0), ((6, 0), 2.0)]),
            ..Upaw::NONE
        },
    ),
    (
        "Ho",
        "lda",
        Upaw {
            local_shift: Some(20.0),
            semicore_subshells: Some(&[(5, 0), (5, 1)]),
            reference_configuration: Some(&[((4, 3), 10.0), ((5, 2), 1.0), ((6, 0), 2.0)]),
            ..Upaw::NONE
        },
    ),
    (
        "I",
        "lda",
        Upaw {
            extra_l: Some(1),
            ..Upaw::NONE
        },
    ),
    (
        "Nd",
        "lda",
        Upaw {
            semicore_subshells: Some(&[(5, 0), (5, 1)]),
            reference_configuration: Some(&[((4, 3), 3.0), ((5, 2), 1.0), ((6, 0), 2.0)]),
            ..Upaw::NONE
        },
    ),
    (
        "P",
        "lda",
        Upaw {
            extra_l: Some(1),
            ..Upaw::NONE
        },
    ),
    (
        "Pd",
        "lda",
        Upaw {
            extra_l: Some(1),
            local_shift: Some(0.0),
            energy_offset: Some(2.0),
            n_bessel: Some(8),
            semicore_subshells: Some(&[(4, 0), (4, 1)]),
            radii: Some(&[(0, 2.2), (1, 2.2), (2, 1.4)]),
            ..Upaw::NONE
        },
    ),
    (
        "Pm",
        "lda",
        Upaw {
            semicore_subshells: Some(&[(5, 0), (5, 1)]),
            reference_configuration: Some(&[((4, 3), 4.0), ((5, 2), 1.0), ((6, 0), 2.0)]),
            ..Upaw::NONE
        },
    ),
    (
        "Pr",
        "lda",
        Upaw {
            semicore_subshells: Some(&[(5, 0), (5, 1)]),
            reference_configuration: Some(&[((4, 3), 2.0), ((5, 2), 1.0), ((6, 0), 2.0)]),
            ..Upaw::NONE
        },
    ),
    (
        "Ra",
        "lda",
        Upaw {
            energy_offset: Some(2.0),
            ..Upaw::NONE
        },
    ),
    (
        "Rh",
        "lda",
        Upaw {
            extra_l: Some(1),
            local_shift: Some(0.0),
            energy_offset: Some(2.0),
            n_bessel: Some(8),
            semicore_subshells: Some(&[(4, 0), (4, 1)]),
            radii: Some(&[(0, 1.768), (1, 1.768), (2, 1.473)]),
            ..Upaw::NONE
        },
    ),
    (
        "Ru",
        "lda",
        Upaw {
            extra_l: Some(1),
            local_shift: Some(0.0),
            energy_offset: Some(2.0),
            n_bessel: Some(8),
            semicore_subshells: Some(&[(4, 0), (4, 1)]),
            radii: Some(&[(0, 1.829), (1, 1.829), (2, 1.558)]),
            ..Upaw::NONE
        },
    ),
    (
        "S",
        "lda",
        Upaw {
            extra_l: Some(1),
            ..Upaw::NONE
        },
    ),
    (
        "Si",
        "lda",
        Upaw {
            extra_l: Some(1),
            ..Upaw::NONE
        },
    ),
    (
        "Sm",
        "lda",
        Upaw {
            semicore_subshells: Some(&[(5, 0), (5, 1)]),
            reference_configuration: Some(&[((4, 3), 5.0), ((5, 2), 1.0), ((6, 0), 2.0)]),
            ..Upaw::NONE
        },
    ),
    (
        "Ta",
        "lda",
        Upaw {
            local_shift: Some(10.0),
            energy_offset: Some(0.6),
            ..Upaw::NONE
        },
    ),
    (
        "Tb",
        "lda",
        Upaw {
            semicore_subshells: Some(&[(5, 0), (5, 1)]),
            reference_configuration: Some(&[((4, 3), 8.0), ((5, 2), 1.0), ((6, 0), 2.0)]),
            ..Upaw::NONE
        },
    ),
    (
        "Te",
        "lda",
        Upaw {
            extra_l: Some(1),
            ..Upaw::NONE
        },
    ),
    (
        "Tm",
        "lda",
        Upaw {
            semicore_subshells: Some(&[(5, 0), (5, 1)]),
            reference_configuration: Some(&[((4, 3), 12.0), ((5, 2), 1.0), ((6, 0), 2.0)]),
            ..Upaw::NONE
        },
    ),
    (
        "Xe",
        "lda",
        Upaw {
            extra_l: Some(1),
            ..Upaw::NONE
        },
    ),
    (
        "Yb",
        "lda",
        Upaw {
            semicore_subshells: Some(&[(5, 0), (5, 1)]),
            reference_configuration: Some(&[((4, 3), 13.0), ((5, 2), 1.0), ((6, 0), 2.0)]),
            ..Upaw::NONE
        },
    ),
];

impl Upaw {
    /// The recipe that changes nothing.
    pub const NONE: Upaw = Upaw {
        extra_l: None,
        local_shift: None,
        energy_offset: None,
        n_bessel: None,
        semicore_subshells: None,
        radii: None,
        reference_configuration: None,
    };
}

/// The UPAW-LCAO recipe of a dataset (none for most: the defaults apply).
pub fn recipe(symbol: &str, functional: &str) -> Upaw {
    RECIPES
        .iter()
        .find(|e| e.0 == symbol && e.1 == functional)
        .map_or(Upaw::NONE, |e| e.2)
}

/// Every `(symbol, functional, recipe)`, for the policy-table freeze.
pub fn entries() -> impl Iterator<Item = (&'static str, &'static str, Upaw)> {
    RECIPES.iter().copied()
}
