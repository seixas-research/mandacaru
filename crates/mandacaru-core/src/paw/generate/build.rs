//! Generate a dataset, write it, read it back and check the file: the whole of one element of
//! `mandacaru-build`.
//!
//! What is released is the file, not the record in memory. The searched dataset is written beside
//! its final name (`<path>.new`, itself staged and verified by `PawDataset::write`). It is read
//! back, and moved into place only when every channel's ghost-test levels are those of the
//! generated record to [`RELOAD_LEVEL_TOLERANCE`]. The checks reported are those of the file read
//! back: its ghosts and, on request, its scattering phases against the all-electron atom. Because
//! the record never crosses into another language, everything it carries (the core states of
//! format version 3 included) reaches the file.
use super::{
    recipe::Recipe,
    reference::solve_reference,
    search::{self, Options, Searched},
};
use crate::models::{atom::Atom, xc_scheme::Variational};
use crate::paw::{
    diagnose::{self, reference::AeLogs, spectrum::ghost_levels, view::Smooth, Diagnosis},
    invalid,
    record::{reload_errors, PawDataset, WriteSpec, RELOAD_LEVEL_TOLERANCE},
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
    /// Every channel of the file read back, ascending in `l`.
    pub channels: Vec<ChannelCheck>,
    /// Ghosts of angular momenta without projectors: `(l, depth below the atom)`.
    pub unprojected_ghosts: Vec<(usize, f64)>,
    /// Every check of the file read back, against the atom the search used (it is solved once).
    pub diagnosis: Diagnosis,
    /// The bound all-electron levels a channel without projectors misses, `{l: offset}`
    /// (`diagnose::quality::unprojected_missing_states`).
    pub unprojected_missing: std::collections::BTreeMap<usize, f64>,
}

/// Why nothing was written.
#[derive(Debug)]
pub enum Error {
    /// The search refused, or exhausted its attempts (`search::Error`, whose text names why).
    Search(search::Error),
    /// The file read back moved bound levels by more than the tolerance: `{l: shift}`.
    Reload(BTreeMap<usize, f64>),
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
/// read it back, refuse it if its levels moved, check it (`phases`: the scattering phases too)
/// and move it into place. Nothing is left beside `path` on any failure.
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
    let searched =
        search::search_with_atom_logged(ctx, recipe, options, &atom, &Variational, &mut logs)
            .map_err(Error::Search)?;
    let staged = staged_path(path);
    let checked = write_and_check(
        ctx,
        &searched.generated.dataset,
        &atom,
        &mut logs,
        &staged,
        format,
        stride,
        phases,
    );
    let (loaded, channels, unprojected_ghosts, diagnosis, unprojected_missing) = match checked {
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
    Ok(Built {
        searched,
        loaded,
        path: path.to_path_buf(),
        channels,
        unprojected_ghosts,
        diagnosis,
        unprojected_missing,
    })
}

type Checked = (
    PawDataset,
    Vec<ChannelCheck>,
    Vec<(usize, f64)>,
    Diagnosis,
    std::collections::BTreeMap<usize, f64>,
);

fn write_and_check(
    ctx: &Context,
    generated: &PawDataset,
    atom: &Atom,
    logs: &mut AeLogs,
    staged: &Path,
    format: Format,
    stride: usize,
    phases: bool,
) -> Result<Checked, Error> {
    match format {
        Format::Parquet => generated.write(
            ctx,
            staged,
            &WriteSpec {
                stride,
                codec: crate::parquet::Codec::Zstd,
            },
        )?,
        Format::Json => generated.write_json(ctx, staged, stride)?,
    }
    let loaded = PawDataset::read(ctx, staged)?;
    let before = levels(ctx, generated)?;
    let after = levels(ctx, &loaded)?;
    let moved = reload_errors(&before, &after);
    if !moved.is_empty() {
        return Err(Error::Reload(moved));
    }
    let (diagnosis, unprojected_missing) =
        diagnose::diagnose_record_with_unprojected_on(ctx, &loaded, atom, logs)
            .map_err(|error| Error::Failed(error.into_core()))?;
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
            phase: if phases {
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
    Ok((
        loaded,
        channels,
        unprojected_ghosts,
        diagnosis,
        unprojected_missing,
    ))
}
