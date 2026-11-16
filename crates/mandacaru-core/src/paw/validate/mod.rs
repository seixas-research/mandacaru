//! The validation entry of a PAW-LCAO / UPAW-LCAO dataset file: every per-file gate of the
//! regeneration plan (G1-G11, G13 and G14; G12 is a crystal calculation) evaluated on one record and
//! returned as a [`GateReport`] with each row's value, threshold and status.
//!
//! The generator calls [`validate`] on the file it has read back, before it moves the file into
//! place ([`crate::paw::generate::build`]), and never releases a file that [`GateReport::passed`]
//! is false for. The same entry runs on a record read from disk (`paw_validate` in the Python
//! extension), so a library on disk can be audited by the code that would have refused it.
//!
//! One owner for each threshold: the diagnostics own G2-G4 and G7 (`paw::diagnose`), the
//! assembler G6 (`paw::generate::assemble`), the record G1 (`paw::record`), the engine's spin-orbit
//! module G13 (`paw::engine_spin_orbit`), and [`record_gates`] and [`atom_gates`] the rest. No
//! consumer states a threshold again.
//!
//! - [`report`]: rows, statuses and the report.
//! - [`record_gates`]: G1, G5-G11, from the record's own tables.
//! - [`atom_gates`]: G2-G4, from the diagnostics against the all-electron atom.
//! - [`spin_orbit`]: G13, through the engine's spin-orbit operator.
pub mod atom_gates;
pub mod record_gates;
pub mod report;
pub mod spin_orbit;

pub use atom_gates::Measured;
pub use report::{GateReport, Relation, Row, Status, GATES};

use crate::{
    paw::{
        diagnose::{self, quality::unrecorded_defects, reference::AeLogs},
        record::{Defects, PawDataset},
    },
    Context, Result,
};

/// What the caller can add to a validation of a record.
#[derive(Default, Clone, Copy)]
pub struct Options<'a> {
    /// The diagnosis the caller already has (the generator's, on the atom its search used). When
    /// `None`, [`validate`] solves the reference atom from the record and diagnoses it.
    pub measured: Option<&'a Measured>,
    /// G1: the largest bound-level shift between the generated record and the file read back
    /// ([`crate::paw::record::reload_shift`]); `None` for a file read alone.
    pub reload_shift: Option<f64>,
    /// G8: the twin record of the same element (the scalar-relativistic set for a fully
    /// relativistic one, or the other functional) whose core/valence partition must be equal.
    pub twin: Option<&'a PawDataset>,
    /// G8: the bond (Bohr) the widest sphere is held against; by default the element's own bond to
    /// hydrogen (`diagnose::quality::shortest_bond`).
    pub bond: Option<f64>,
    /// The file records its own defects (`ghosts="flag"`): a G2-G4 row that fails on a defect the
    /// record carries is waived, not refused. A defect the record does not carry still refuses.
    pub recorded_defects: bool,
    /// G13: the Fermi-Dirac width (eV) to measure at first (the one the generator's calibration
    /// closed at); the ladder follows when the SCF does not converge there.
    pub spin_orbit_width: Option<f64>,
}

/// Every per-file gate of `dataset`.
///
/// An error is a failure to run (a malformed record, a budget); a check that cannot be evaluated
/// is a row with [`Status::Unevaluated`], which refuses the file like a failure.
pub fn validate(ctx: &Context, dataset: &PawDataset, options: &Options<'_>) -> Result<GateReport> {
    let mut rows = vec![record_gates::reload(options.reload_shift)];
    let solved;
    let atom: std::result::Result<&Measured, String> = match options.measured {
        Some(measured) => Ok(measured),
        None => {
            let outcome =
                diagnose::diagnose_record_with_unprojected(ctx, dataset, &mut AeLogs::new());
            match outcome {
                Ok((diagnosis, unprojected_missing)) => {
                    solved = Measured {
                        diagnosis,
                        unprojected_missing,
                    };
                    Ok(&solved)
                }
                Err(diagnose::Error::Unevaluated(why)) => Err(why),
                Err(diagnose::Error::Failed(error)) => return Err(error),
            }
        }
    };
    match atom {
        Ok(measured) => {
            let recorded = options.recorded_defects
                && unrecorded_defects(&dataset.defects, &measured.diagnosis.defects())
                    == Defects::default();
            rows.extend(atom_gates::rows(dataset, measured, recorded));
        }
        Err(why) => {
            for (gate, check) in [
                ("G2", "ghosts"),
                ("G3", "scattering phases"),
                ("G4", "bound states"),
            ] {
                rows.push(Row::unevaluated(gate, check, why.clone()));
            }
        }
    }
    rows.extend(record_gates::core(dataset));
    rows.extend(record_gates::conditioning(dataset));
    rows.extend(record_gates::hardness_and_grid(ctx, dataset)?);
    rows.extend(record_gates::sphere_and_partition(
        dataset,
        options.bond,
        options.twin,
    ));
    rows.extend(record_gates::smooth_core(dataset));
    rows.push(record_gates::transferability());
    rows.push(record_gates::grid_uniformity(dataset));
    rows.extend(record_gates::ionic_tail(dataset));
    rows.extend(spin_orbit::rows(ctx, dataset, options.spin_orbit_width));
    let order = |gate: &str| {
        GATES
            .iter()
            .position(|(g, _)| *g == gate)
            .unwrap_or(GATES.len())
    };
    rows.sort_by_key(|row| order(row.gate));
    Ok(GateReport {
        symbol: dataset.symbol.clone(),
        xc: dataset.xc.clone(),
        relativity: dataset.relativity.clone(),
        rows,
    })
}
