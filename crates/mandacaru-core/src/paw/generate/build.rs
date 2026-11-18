//! Generate a dataset, write it, read it back and check the file: the whole of one element of
//! `mandacaru-build`.
//!
//! What is released is the file, not the record in memory. The searched dataset is written beside
//! its final name (`<path>.new`, itself staged and verified by `PawDataset::write`). It is read
//! back, and moved into place only when every channel's ghost-test levels are those of the
//! generated record to [`RELOAD_LEVEL_TOLERANCE`]. The checks reported are those of the file read
//! back: its ghosts and, on request, its scattering phases against the all-electron atom. Before
//! it is moved into place the file read back passes every per-file gate of
//! [`crate::paw::validate`] (G1-G11 and G13); a file that fails one is never written. Because
//! the record never crosses into another language, everything it carries (the core states of
//! format version 3 included) reaches the file.
use super::{
    recipe::Recipe,
    reference::solve_reference,
    search::{self, Mode, Options, Searched},
    spin_orbit_scale::{self, Scale},
};
use crate::models::{atom::Atom, xc_scheme::Variational};
use crate::paw::{
    diagnose::{self, reference::AeLogs, spectrum::ghost_levels, view::Smooth, Diagnosis},
    engine_spin_orbit, invalid,
    record::{
        digits::LIBRARY_STRIDE, reload_errors, reload_shift, PawDataset, WriteSpec,
        RELOAD_LEVEL_TOLERANCE,
    },
    validate::{self, GateReport, Measured},
};
use crate::Context;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

/// The file layout of a written dataset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Format {
    /// The library's: tables as `double` columns, the payload as metadata.
    Parquet,
    /// The payload with the tables inline.
    Json,
}

/// One channel of the file read back, as the build reports it.
#[derive(Clone, Debug)]
pub struct ChannelCheck {
    pub l: usize,
    pub r_cut: f64,
    /// The first reference energy of the channel (Hartree).
    pub reference_energy: f64,
    /// The lowest level of the channel's ghost test (Hartree).
    pub lowest: f64,
    pub ghost: bool,
    /// `(near, far)` phase error (rad), when the phases were asked for.
    pub phase: Option<[f64; 2]>,
}

/// A dataset written and checked.
#[derive(Debug)]
pub struct Built {
    /// The search and the record it accepted (in memory, before the file's stride and digits).
    pub searched: Searched,
    /// The file read back.
    pub loaded: PawDataset,
    /// Where it now is.
    pub path: PathBuf,
    /// The stride the file was written at.
    pub stride: usize,
    /// Every channel of the file read back, ascending in `l`.
    pub channels: Vec<ChannelCheck>,
    /// Ghosts of angular momenta without projectors: `(l, depth below the atom)`.
    pub unprojected_ghosts: Vec<(usize, f64)>,
    /// Every check of the file read back, against the atom the search used (it is solved once).
    pub diagnosis: Diagnosis,
    /// The bound all-electron levels a channel without projectors misses, `{l: offset}`
    /// (`diagnose::quality::unprojected_missing_states`).
    pub unprojected_missing: std::collections::BTreeMap<usize, f64>,
    /// Every per-file gate of the file read back ([`crate::paw::validate`]); none refuses it.
    pub report: GateReport,
    /// The factor of the stored spin-orbit coupling of every `l` of a Dirac record
    /// ([`spin_orbit_scale`]); empty for a scalar one.
    pub spin_orbit_scales: Vec<Scale>,
}

/// Why nothing was written.
#[derive(Debug)]
pub enum Error {
    /// The search refused, or exhausted its attempts (`search::Error`, whose text names why).
    Search(search::Error),
    /// The file read back moved bound levels by more than the tolerance: `{l: shift}`.
    Reload(BTreeMap<usize, f64>),
    /// The file read back fails a per-file gate ([`crate::paw::validate`]); the report holds every row.
    Gates(GateReport),
    /// Anything else: writing, reading, a check that could not be evaluated, a budget.
    Failed(crate::Error),
}

impl From<crate::Error> for Error {
    fn from(error: crate::Error) -> Self {
        Self::Failed(error)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Search(error) => write!(f, "{error}"),
            Self::Reload(moved) => {
                write!(f, "the file read back does not reproduce the dataset: ")?;
                for (i, (l, shift)) in moved.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "l={l} levels moved {shift:.1e} Ha")?;
                }
                write!(f, " (tolerance {RELOAD_LEVEL_TOLERANCE} Ha); not written")
            }
            Self::Gates(report) => f.write_str(&report.refusal()),
            Self::Failed(error) => write!(f, "{error}"),
        }
    }
}

/// The ghost-test levels of every channel of `dataset`, ascending in `l`.
fn levels(ctx: &Context, dataset: &PawDataset) -> crate::Result<Vec<(usize, Vec<f64>)>> {
    let smooth = Smooth::scalar(ctx, dataset)?;
    smooth
        .angular_momenta()
        .into_iter()
        .map(|l| {
            let channel = smooth.channel(l).expect("listed channel");
            Ok((l, ghost_levels(ctx, &smooth, channel)?))
        })
        .collect()
}

fn staged_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".new");
    PathBuf::from(name)
}

/// Search `recipe` under `options`, write the accepted record to `path` in `format` at `stride`,
/// read it back, refuse it if its levels moved or a gate fails, check it (`phases`: the scattering
/// phases too) and move it into place. Nothing is left beside `path` on any failure.
pub fn build(
    ctx: &Context,
    recipe: &Recipe,
    options: &Options,
    path: &Path,
    format: Format,
    stride: usize,
    phases: bool,
) -> Result<Built, Error> {
    // One reference atom per element: the search, and the check of the file it writes, share it.
    let atom = solve_reference(ctx, recipe).map_err(|e| Error::Search(search::Error::Failed(e)))?;
    let mut logs = AeLogs::new();
    let mut searched =
        search::search_with_atom_logged(ctx, recipe, options, &atom, &Variational, &mut logs)
            .map_err(Error::Search)?;
    let how = Release {
        format,
        stride,
        phases,
        // `ghosts="flag"` writes a file whose defects it records: those rows are waived.
        recorded_defects: options.mode == Mode::Flag,
        spin_orbit_width: None,
    };
    let scales = calibrate_spin_orbit(ctx, &mut searched.generated.dataset, path, how)?;
    let how = Release {
        spin_orbit_width: scales.iter().find_map(|s| s.width),
        ..how
    };
    let released = release(
        ctx,
        &searched.generated.dataset,
        &atom,
        &mut logs,
        path,
        how,
    )?;
    Ok(Built {
        spin_orbit_scales: scales,
        searched,
        loaded: released.loaded,
        path: path.to_path_buf(),
        stride,
        channels: released.channels,
        unprojected_ghosts: released.unprojected_ghosts,
        diagnosis: released.diagnosis,
        unprojected_missing: released.unprojected_missing,
        report: released.report,
    })
}

/// The strides a library file is tried at, coarsest first ([`build_at_library_stride`]).
pub const LIBRARY_STRIDES: [usize; 3] = [LIBRARY_STRIDE, LIBRARY_STRIDE / 2, 1];

/// [`build`] at the coarsest of [`LIBRARY_STRIDES`] whose file reads back with the generated
/// levels: a [`Error::Reload`] refusal builds again at the next stride, and any other outcome is
/// final. This is how the library is written (`mandacaru-build`, the 2026-11 regeneration of the
/// four PAW-LCAO sets); a file that moves its levels at stride 1 too is refused.
pub fn build_at_library_stride(
    ctx: &Context,
    recipe: &Recipe,
    options: &Options,
    path: &Path,
    format: Format,
    phases: bool,
) -> Result<Built, Error> {
    let mut refused = None;
    for stride in LIBRARY_STRIDES {
        match build(ctx, recipe, options, path, format, stride, phases) {
            Err(moved @ Error::Reload(_)) => refused = Some(moved),
            done => return done,
        }
    }
    Err(refused.expect("at least one stride is tried"))
}

/// The record as the file will hold it (written at `how.stride`, read back) and its engine
/// splittings, through a scratch file beside `path` that never outlives the call.
fn splittings_as_released(
    ctx: &Context,
    ds: &PawDataset,
    path: &Path,
    how: Release,
    only_width: Option<f64>,
) -> crate::Result<Vec<engine_spin_orbit::Splitting>> {
    let mut name = path.as_os_str().to_owned();
    name.push(".calibrate");
    let scratch = PathBuf::from(name);
    let outcome = write_record(ctx, ds, &scratch, how)
        .and_then(|()| PawDataset::read(ctx, &scratch))
        .and_then(|loaded| engine_spin_orbit::splittings_from(ctx, &loaded, only_width));
    let _ = std::fs::remove_file(&scratch);
    outcome
}

/// Scale the stored spin-orbit coupling of a Dirac record until the engine's isolated atom, in
/// the record as it will be written, reproduces the Dirac splittings ([`spin_orbit_scale`]).
fn calibrate_spin_orbit(
    ctx: &Context,
    ds: &mut PawDataset,
    path: &Path,
    how: Release,
) -> Result<Vec<Scale>, Error> {
    if ds.spin_orbit.is_empty() {
        return Ok(Vec::new());
    }
    match spin_orbit_scale::calibrate(ds, |scaled, floor| {
        splittings_as_released(ctx, scaled, path, how, floor)
    }) {
        Ok(scales) => Ok(scales),
        // An open-shell atom whose SCF converges at no width: nothing to measure, the coupling
        // stays as built and gate G13 reports the shell (never a silent pass).
        Err(error) if crate::paw::atom_scf::not_converged(&error) => Ok(Vec::new()),
        Err(error) => Err(error.into()),
    }
}

/// Derive the spin-orbit scale of the Dirac file at `path` again and write it back, through the
/// same release as a generation (every gate, nothing left beside the path on a refusal). The
/// calibration is a measurement, so it is idempotent: a file already at the Dirac splittings keeps
/// its coupling. The record is read, not generated, so it is written at stride 1 (its tables are
/// the file's) unless `stride` says otherwise.
pub fn recalibrate(
    ctx: &Context,
    path: &Path,
    format: Format,
    stride: usize,
    phases: bool,
) -> Result<(Released, Vec<Scale>), Error> {
    let mut record = PawDataset::read(ctx, path)?;
    if record.spin_orbit.is_empty() {
        return Err(invalid(format!(
            "the {} dataset has no spin-orbit entry to calibrate",
            record.symbol
        ))
        .into());
    }
    let recipe = Recipe::for_dataset(
        &record.symbol,
        &record.xc,
        crate::models::relativity::Treatment::Dirac,
    )?;
    let atom =
        solve_reference(ctx, &recipe).map_err(|e| Error::Search(search::Error::Failed(e)))?;
    let how = Release {
        format,
        stride,
        phases,
        recorded_defects: record.defects != Default::default(),
        spin_orbit_width: None,
    };
    let scales = calibrate_spin_orbit(ctx, &mut record, path, how)?;
    let how = Release {
        spin_orbit_width: scales.iter().find_map(|s| s.width),
        ..how
    };
    let released = release(ctx, &record, &atom, &mut AeLogs::new(), path, how)?;
    Ok((released, scales))
}

/// Make the ionic tail of the file at `path` the bare Coulomb potential `-Zv/r` beyond the local
/// radius ([`super::ionic_tail`]) and write it back through the generator's own release (every
/// gate; nothing left beside the path on a refusal). Only `v_local` changes: the fix a file made
/// before the tail was enforced needs, without searching its partial waves again. The record is
/// read, not generated, so it is written at stride 1 unless `stride` says otherwise.
pub fn reionize(
    ctx: &Context,
    path: &Path,
    format: Format,
    stride: usize,
    phases: bool,
) -> Result<Released, Error> {
    let mut record = PawDataset::read(ctx, path)?;
    let (zv, r_local) = (record.valence_charge, record.r_cut_local);
    let r: Vec<f64> = record.r.as_slice().to_vec();
    super::ionic_tail::enforce(&r, record.v_local.as_mut_slice(), zv, r_local);
    let treatment = crate::models::relativity::Treatment::parse(&record.relativity)?;
    let recipe = Recipe::for_dataset(&record.symbol, &record.xc, treatment)?;
    let atom =
        solve_reference(ctx, &recipe).map_err(|e| Error::Search(search::Error::Failed(e)))?;
    let how = Release {
        format,
        stride,
        phases,
        recorded_defects: record.defects != Default::default(),
        spin_orbit_width: None,
    };
    release(ctx, &record, &atom, &mut AeLogs::new(), path, how)
}

/// How a generated record is written and what its gates may waive.
#[derive(Clone, Copy, Debug)]
pub struct Release {
    pub format: Format,
    pub stride: usize,
    /// Also compare the scattering phases with the all-electron atom.
    pub phases: bool,
    /// The record carries its own defects (`ghosts="flag"`), so G2-G4 rows that fail on them are
    /// waived instead of refusing the file.
    pub recorded_defects: bool,
    /// The Fermi-Dirac width (eV) the spin-orbit calibration closed at, when the record was
    /// calibrated: G13 measures the file there first (the splitting of an open-shell atom depends
    /// on the occupations the SCF converges at).
    pub spin_orbit_width: Option<f64>,
}

/// What [`release`] checked on the file it moved into place.
#[derive(Debug)]
pub struct Released {
    /// The file read back.
    pub loaded: PawDataset,
    pub channels: Vec<ChannelCheck>,
    pub unprojected_ghosts: Vec<(usize, f64)>,
    pub diagnosis: Diagnosis,
    pub unprojected_missing: BTreeMap<usize, f64>,
    /// Every per-file gate; none refuses the file.
    pub report: GateReport,
}

/// Write `generated` beside `path`, read it back, check it ([`write_and_check`]) and the gates of
/// [`crate::paw::validate`], and move it into place only when every one passes. The record is
/// taken, not generated, so the refusal of a record that fails a gate is testable on its own.
/// Nothing is left beside `path` on any failure.
pub fn release(
    ctx: &Context,
    generated: &PawDataset,
    atom: &Atom,
    logs: &mut AeLogs,
    path: &Path,
    how: Release,
) -> Result<Released, Error> {
    let staged = staged_path(path);
    let checked = write_and_check(ctx, generated, atom, logs, &staged, how);
    let released = match checked {
        Ok(found) => found,
        Err(error) => {
            let _ = std::fs::remove_file(&staged);
            return Err(error);
        }
    };
    if let Err(error) = std::fs::rename(&staged, path) {
        let _ = std::fs::remove_file(&staged);
        return Err(Error::Failed(invalid(format!(
            "cannot move {} into place: {error}",
            staged.display()
        ))));
    }
    Ok(released)
}

fn write_and_check(
    ctx: &Context,
    generated: &PawDataset,
    atom: &Atom,
    logs: &mut AeLogs,
    staged: &Path,
    how: Release,
) -> Result<Released, Error> {
    write_record(ctx, generated, staged, how)?;
    let loaded = PawDataset::read(ctx, staged)?;
    let before = levels(ctx, generated)?;
    let after = levels(ctx, &loaded)?;
    let moved = reload_errors(&before, &after);
    if !moved.is_empty() {
        return Err(Error::Reload(moved));
    }
    let (mut diagnosis, unprojected_missing) =
        diagnose::diagnose_record_with_unprojected_on(ctx, &loaded, atom, logs)
            .map_err(|error| Error::Failed(error.into_core()))?;
    if how.phases && !loaded.spin_orbit.is_empty() {
        let smooth = diagnose::view::Smooth::scalar(ctx, &loaded)?;
        let ae = diagnose::reference::AllElectron::new(
            atom,
            smooth.atomic_number,
            &smooth.frozen,
            &smooth.semicore,
        );
        diagnosis = diagnosis
            .with_branch_phases(ctx, &loaded, &ae, logs)
            .map_err(|error| Error::Failed(error.into_core()))?;
    }
    let measured = Measured {
        diagnosis,
        unprojected_missing,
    };
    let report = validate::validate(
        ctx,
        &loaded,
        &validate::Options {
            measured: Some(&measured),
            reload_shift: Some(reload_shift(&before, &after)),
            recorded_defects: how.recorded_defects,
            spin_orbit_width: how.spin_orbit_width,
            ..Default::default()
        },
    )?;
    if !report.passed() {
        return Err(Error::Gates(report));
    }
    let Measured {
        diagnosis,
        unprojected_missing,
    } = measured;
    let mut channels = Vec::with_capacity(loaded.channels.len());
    for (&l, channel) in &loaded.channels {
        let l = l as usize;
        let lowest = after
            .iter()
            .find(|(k, _)| *k == l)
            .and_then(|(_, values)| values.first().copied())
            .ok_or_else(|| invalid(format!("the file read back has no levels for l={l}")))?;
        channels.push(ChannelCheck {
            l,
            r_cut: channel.r_cut,
            reference_energy: *channel
                .reference_energies
                .data()
                .first()
                .ok_or_else(|| invalid(format!("channel l={l} has no reference energy")))?,
            lowest,
            ghost: diagnosis.ghosts.contains_key(&l),
            phase: if how.phases {
                diagnosis
                    .phases
                    .iter()
                    .find(|(key, _)| key.l == l && key.kappa.is_none())
                    .map(|(_, phase)| *phase)
            } else {
                None
            },
        });
    }
    let unprojected_ghosts = diagnosis
        .ghosts
        .iter()
        .filter(|(l, _)| !loaded.channels.contains_key(&(**l as u32)))
        .map(|(&l, &depth)| (l, depth))
        .collect();
    Ok(Released {
        loaded,
        channels,
        unprojected_ghosts,
        diagnosis,
        unprojected_missing,
        report,
    })
}

fn write_record(ctx: &Context, ds: &PawDataset, path: &Path, how: Release) -> crate::Result<()> {
    match how.format {
        Format::Parquet => ds.write(
            ctx,
            path,
            &WriteSpec {
                stride: how.stride,
                codec: crate::parquet::Codec::Zstd,
            },
        ),
        Format::Json => ds.write_json(ctx, path, how.stride),
    }
}
