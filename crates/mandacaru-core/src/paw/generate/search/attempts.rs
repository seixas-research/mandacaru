//! The remedies of the search, in the order they are tried.
//!
//! Every attempt is the first construction's recipe with some fields changed (never the
//! previous attempt's: the remedies do not accumulate), and every one is built at zero norm
//! deficit. A raise of the local potential acts on every angular momentum the projectors do not
//! cover, so the shorter local radius at no raise comes first; the raises follow with the
//! channels at their own cutoffs, then with more Bessel functions and a shorter s sphere, and
//! last with every channel at the largest cutoff (Hamann's `dvloc0`). The measured reasons are
//! in `tables_repairs` and the generator's history; this file is the order.
use crate::paw::{
    generate::recipe::{Cutoffs, Recipe},
    record::PawDataset,
};
use std::collections::BTreeMap;

/// Fractions of the local radius tried, at no raise, before any raise. A raise acts on every
/// angular momentum the projectors do not cover: aluminum's 5 Ha raise scattered d 0.52 rad off
/// and put fcc Al at 4.48 Angstrom, a 0.6x radius at no raise passes every check.
pub const SHORTER_LOCAL_FACTORS: [f64; 3] = [0.75, 0.6, 0.45];
/// Local-potential raises (Ha) tried with every channel at its own cutoff.
pub const OWN_CUTOFF_SHIFTS: [f64; 4] = [5.0, 10.0, 20.0, 40.0];
/// Local-potential raises (Ha) tried once every channel is given the largest cutoff.
pub const GHOST_REMEDY_SHIFTS: [f64; 5] = [0.0, 10.0, 20.0, 40.0, 80.0];
/// Bessel-function counts tried with each raise (the s channel's completeness grows with them).
pub const BESSEL_COUNTS: [usize; 2] = [9, 10];
/// Fractions of the s cutoff tried, with ten Bessel functions (Tl, Po, Bi, Os, Re, Ta and V
/// were repaired at 0.85-0.95 of their s cutoff).
pub const S_CUTOFF_FACTORS: [f64; 3] = [0.95, 0.9, 0.85];
/// Raises (Ha) tried with each shorter s sphere.
pub const S_CUTOFF_SHIFTS: [f64; 3] = [0.0, 10.0, 20.0];

/// One construction to try: the label of its remedy and the recipe it is built from.
#[derive(Debug, Clone)]
pub struct Attempt {
    pub label: String,
    pub recipe: Recipe,
}

/// The augmentation radius of every channel the first construction built, empty channels
/// included.
pub fn radii_of(first: &PawDataset) -> BTreeMap<usize, f64> {
    first
        .channels
        .iter()
        .map(|(&l, c)| (l as usize, c.r_cut))
        .collect()
}

/// The attempts after the first construction of `recipe`, which built the channels of radii
/// `own`, in the order they are tried.
pub fn plan(recipe: &Recipe, own: &BTreeMap<usize, f64>) -> Vec<Attempt> {
    let unitary = || Recipe {
        norm_deficit: 0.,
        ..recipe.clone()
    };
    let mut out = vec![Attempt {
        label: "norm_deficit=0".to_string(),
        recipe: unitary(),
    }];
    for factor in SHORTER_LOCAL_FACTORS {
        out.push(Attempt {
            label: format!("local radius x {factor}, no shift"),
            recipe: Recipe {
                local_shift: 0.,
                local_factor: recipe.local_factor * factor,
                ..unitary()
            },
        });
    }
    for shift in OWN_CUTOFF_SHIFTS {
        out.push(Attempt {
            label: format!("own cutoffs, shift {shift}"),
            recipe: Recipe {
                local_shift: shift,
                ..unitary()
            },
        });
    }
    for count in BESSEL_COUNTS {
        for shift in std::iter::once(0.).chain(OWN_CUTOFF_SHIFTS) {
            out.push(Attempt {
                label: format!("{count} Bessel functions, shift {shift}"),
                recipe: Recipe {
                    n_bessel: count,
                    local_shift: shift,
                    ..unitary()
                },
            });
        }
    }
    for factor in S_CUTOFF_FACTORS {
        let shorter = own
            .iter()
            .map(|(&l, &radius)| (l, if l == 0 { radius * factor } else { radius }))
            .collect::<BTreeMap<_, _>>();
        for shift in S_CUTOFF_SHIFTS {
            out.push(Attempt {
                label: format!("s cutoff x {factor}, 10 Bessel functions, shift {shift}"),
                recipe: Recipe {
                    cutoffs: Cutoffs::PerChannel(shorter.clone()),
                    n_bessel: 10,
                    local_shift: shift,
                    ..unitary()
                },
            });
        }
    }
    let radius = own.values().copied().fold(f64::NEG_INFINITY, f64::max);
    let balanced = own.keys().map(|&l| (l, radius)).collect::<BTreeMap<_, _>>();
    for shift in GHOST_REMEDY_SHIFTS {
        out.push(Attempt {
            label: format!("balanced at {radius:.3} Bohr, shift {shift}"),
            recipe: Recipe {
                cutoffs: Cutoffs::PerChannel(balanced.clone()),
                local_radius: Some(recipe.local_factor * radius),
                local_shift: shift,
                ..unitary()
            },
        });
    }
    out
}
