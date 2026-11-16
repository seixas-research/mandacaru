//! G13: the engine's isolated-atom `j` splitting of every valence `l` with spin-orbit against the
//! all-electron Dirac one stored in the record. The measurement is
//! [`crate::paw::engine_spin_orbit::splittings`], the engine's own operator in its own basis, and
//! its verdict [`Splitting::passes`] with the thresholds that module owns (2 percent or 10 meV):
//! this gate only reports them, so there is one implementation of the measurement and one owner of
//! the threshold, whatever the engine's assembly of the operator becomes.
use super::report::{Row, Status};
use crate::{
    paw::{
        engine_spin_orbit::{splittings, splittings_from, ABSOLUTE, RELATIVE},
        record::PawDataset,
    },
    Context,
};

const HARTREE_EV: f64 = 27.211_386_245_988;
const L_LABELS: [char; 6] = ['s', 'p', 'd', 'f', 'g', 'h'];

/// The rows of G13: one per spin-orbit channel with a bound occupied reference.
pub fn rows(ctx: &Context, ds: &PawDataset, width: Option<f64>) -> Vec<Row> {
    if ds.spin_orbit.is_empty() {
        return vec![Row::not_applicable(
            "G13",
            "spin-orbit splitting",
            format!("the {} dataset has no spin-orbit entry", ds.relativity),
        )];
    }
    let first = match width {
        Some(w) => splittings_from(ctx, ds, Some(w)).or_else(|_| splittings(ctx, ds)),
        None => splittings(ctx, ds),
    };
    let mut instability: Option<String> = None;
    let found = match first {
        Ok(found) => found,
        // The atom's SCF converges at no width of the ladder: nothing to measure, said, not passed.
        Err(error) if crate::paw::atom_scf::not_converged(&error) => {
            return vec![Row::measured_without_value(
                "G13",
                "spin-orbit splitting",
                error.to_string(),
            )]
        }
        Err(error) => {
            return vec![Row::unevaluated(
                "G13",
                "spin-orbit splitting",
                error.to_string(),
            )]
        }
    };
    if found.is_empty() {
        return vec![Row::not_applicable(
            "G13",
            "spin-orbit splitting",
            "no bound occupied reference has a Dirac splitting above the 10 meV floor of the gate",
        )];
    }
    // The splitting of an atom with an open shell depends on the occupations the SCF converges at
    // (the Fermi-Dirac width that converges): a shell outside the gate at the first converged
    // width that is inside it at another converged width of the ladder is reported with both, not
    // refused (the occupation, not the file, moved it).
    let mut found = found;
    // An atom with a fractionally filled d or f shell: the occupations of that shell move every
    // splitting of the atom, so a shell of it that stays out of the gate at every converged width
    // is reported with the open shell named, not refused.
    let open_atom = found.iter().any(|s| s.unreliable());
    if found.iter().any(|s| s.engine.is_some() && !s.passes() && !s.unreliable()) {
        let first_width = found[0].width;
        for &w in crate::paw::atom_scf::Controls::default().widths.iter().filter(|&&w| w > first_width) {
            if let Ok(alt) = splittings_from(ctx, ds, Some(w)) {
                if alt.len() == found.len()
                    && alt.iter().all(|a| a.engine.is_none() || a.passes() || a.unreliable())
                {
                    instability = Some(format!(
                        "outside the gate at the SCF width {first_width} eV, inside it at {w} eV: the occupations of the open shell move the splitting"
                    ));
                    found = alt;
                    break;
                }
            }
        }
    }
    found
        .iter()
        .map(|s| {
            let label = format!(
                "{} splitting, reference {}: |engine - Dirac| (Ha)",
                L_LABELS.get(s.l).copied().unwrap_or('?'),
                s.state
            );
            let Some(engine) = s.engine else {
                return Row::measured_without_value(
                    "G13",
                    label,
                    format!(
                        "the isolated atom shows no pair of j multiplets for this shell within a factor of 0.6 to 1.6 of the Dirac splitting (an open shell filled j by j): the coupling stays as built (l={})",
                        s.l
                    ),
                );
            };
            let allowed = (RELATIVE * s.dirac.abs()).max(ABSOLUTE);
            let mut row = Row::at_most("G13", label, (engine - s.dirac).abs(), allowed);
            // The verdict is the engine module's own, not a second comparison.
            row.status = if s.passes() && instability.is_none() {
                Status::Pass
            } else if s.passes() {
                Status::Reported
            } else if s.unreliable() || open_atom {
                // An open d or f shell: the atom's splitting includes the exchange of the
                // fractional filling, the coupling stays as built, the row says so.
                Status::Reported
            } else {
                Status::Fail
            };
            row.with_detail(format!(
                "{}engine {:.4} eV, Dirac {:.4} eV ({:+.1} %); {} percent or {:.0} meV; width {} eV",
                instability
                    .as_deref()
                    .map(|i| format!("{i}; "))
                    .or_else(|| (open_atom && !s.passes() && !s.unreliable()).then(|| "an open d or f shell of this atom moves the splitting at every converged width; ".to_string()))
                    .unwrap_or_default(),
                engine * HARTREE_EV,
                s.dirac * HARTREE_EV,
                100. * (engine - s.dirac) / s.dirac,
                100. * RELATIVE,
                1000. * ABSOLUTE * HARTREE_EV,
                s.width,
            ))
        })
        .collect()
}
