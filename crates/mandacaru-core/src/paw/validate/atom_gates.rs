//! The gates that read the smooth Hamiltonian against the all-electron atom: ghosts (G2),
//! scattering phases (G3) and bound states (G4). The numbers come from [`Diagnosis`]; this module
//! only compares them with the thresholds the diagnostics own.
use super::report::{Row, Status};
use crate::paw::{
    diagnose::{
        ghosts::{GHOST_TOLERANCE, MISSING_TOLERANCE},
        intruding::INTRUDING_S_TOLERANCE,
        scattering::{PHASE_TOLERANCE, RESONANCE_TOLERANCE, UNPROJECTED_TOLERANCE},
        Diagnosis,
    },
    record::PawDataset,
};
use std::collections::BTreeMap;

/// Largest offset (Ha) of the smooth level from the all-electron one for a bound level in a
/// channel without projectors, which the local potential alone has to reproduce. Entries between
/// [`MISSING_TOLERANCE`] and this are listed in the row's detail and do not refuse the file.
pub const UNPROJECTED_MISSING_TOLERANCE: f64 = 0.05;

/// What the diagnostics measured on one dataset: the checks of the recording rule and the bound
/// levels of the channels without projectors.
#[derive(Debug, Clone, Default)]
pub struct Measured {
    pub diagnosis: Diagnosis,
    /// `{l: offset}` of `diagnose::quality::unprojected_missing_states`.
    pub unprojected_missing: BTreeMap<usize, f64>,
}

/// The largest magnitude, a NaN kept.
fn worst(values: impl Iterator<Item = f64>) -> f64 {
    values.map(f64::abs).fold(0., |m, v| {
        if m.is_nan() || v.is_nan() {
            f64::NAN
        } else {
            m.max(v)
        }
    })
}

fn listed<K: std::fmt::Display>(entries: impl Iterator<Item = (K, f64)>) -> String {
    entries
        .map(|(k, v)| format!("{k}: {v:+.3e}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Rows of G2, G3 and G4. `recorded`: the file records the defects the diagnosis found
/// (`ghosts="flag"`), so the rows built from them are waived rather than refused.
pub fn rows(ds: &PawDataset, measured: &Measured, recorded: bool) -> Vec<Row> {
    let d = &measured.diagnosis;
    let mut rows = Vec::new();
    // G2: ghosts, in the channels with projectors, in the angular momenta without, and in the j branches.
    let (own, bare): (Vec<_>, Vec<_>) = d
        .ghosts
        .iter()
        .partition(|(l, _)| ds.channels.contains_key(&(**l as u32)));
    for (check, found) in [
        (
            "ghosts: deepest extra level below the atom (Ha), channels with projectors",
            own,
        ),
        (
            "ghosts: deepest extra level below the atom (Ha), angular momenta without projectors",
            bare,
        ),
    ] {
        let mut row = Row::at_most(
            "G2",
            check,
            worst(found.iter().map(|(_, &e)| e)),
            GHOST_TOLERANCE,
        );
        if !found.is_empty() {
            row.status = Status::Fail;
            row.detail = listed(found.iter().map(|(l, &e)| (format!("l={l}"), e)));
        }
        rows.push(row.waived_if(recorded));
    }
    let mut branch = Row::at_most(
        "G2",
        "ghosts: deepest extra level, j branches (Ha)",
        worst(d.branch_ghosts.values().copied()),
        GHOST_TOLERANCE,
    );
    if !d.branch_ghosts.is_empty() {
        branch.status = Status::Fail;
        branch.detail = listed(
            d.branch_ghosts
                .iter()
                .map(|(k, &e)| (format!("{}:{}", k.l, k.kappa.unwrap_or(0)), e)),
        );
    }
    rows.push(branch.waived_if(recorded));
    let stored = ds.defects != crate::paw::record::Defects::default();
    rows.push(Row::flag(
        "G2",
        "record stores no defect",
        !stored || recorded,
        if recorded && stored {
            "ghosts=\"flag\": the defects are the file's own record"
        } else if stored {
            "the file carries recorded defects"
        } else {
            ""
        },
    ));

    // G3: scattering phases, worst over the channels.
    let scalar: Vec<_> = d.phases.iter().filter(|(k, _)| k.kappa.is_none()).collect();
    let branches: Vec<_> = d.phases.iter().filter(|(k, _)| k.kappa.is_some()).collect();
    for (label, set) in [("", scalar), ("j-branch ", branches)] {
        if set.is_empty() && !label.is_empty() {
            continue;
        }
        for (window, index, tolerance) in [
            ("near, +-0.5 Ha", 0, PHASE_TOLERANCE),
            ("far, +-1 Ha", 1, RESONANCE_TOLERANCE),
        ] {
            let value = worst(set.iter().map(|(_, p)| p[index]));
            let at = set
                .iter()
                .max_by(|a, b| a.1[index].total_cmp(&b.1[index]))
                .map(|(k, _)| format!("worst at l={}", k.l))
                .unwrap_or_default();
            rows.push(
                Row::at_most(
                    "G3",
                    format!("{label}phase error, {window} (rad)"),
                    value,
                    tolerance,
                )
                .with_detail(at)
                .waived_if(recorded),
            );
        }
    }
    let unprojected = worst(d.unprojected.values().copied());
    rows.push(
        Row::at_most(
            "G3",
            "phase error, channels without projectors (rad)",
            unprojected,
            UNPROJECTED_TOLERANCE,
        )
        .with_detail(listed(
            d.unprojected.iter().map(|(l, &e)| (format!("l={l}"), e)),
        ))
        .waived_if(recorded),
    );

    // G4: states.
    let mut missing = Row::at_most(
        "G4",
        "missing bound states: worst offset (Ha)",
        worst(d.missing.values().copied()),
        MISSING_TOLERANCE,
    );
    if !d.missing.is_empty() {
        missing.status = Status::Fail;
        missing.detail = listed(d.missing.iter().map(|(l, &e)| (format!("l={l}"), e)));
    }
    rows.push(missing.waived_if(recorded));
    rows.push(
        Row::at_most(
            "G4",
            "bound levels in channels without projectors: worst offset (Ha)",
            worst(measured.unprojected_missing.values().copied()),
            UNPROJECTED_MISSING_TOLERANCE,
        )
        .with_detail(listed(
            measured
                .unprojected_missing
                .iter()
                .map(|(l, &e)| (format!("l={l}"), e)),
        )),
    );
    rows.push(match d.s_miss {
        Some(miss) => Row::below(
            "G4",
            "intruding 1s: s-channel miss",
            miss,
            INTRUDING_S_TOLERANCE,
        )
        .waived_if(recorded),
        None => Row::not_applicable(
            "G4",
            "intruding 1s: s-channel miss",
            "the dataset has no s channel",
        ),
    });
    rows
}
