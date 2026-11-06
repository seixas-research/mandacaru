//! The DFT+U shell table of a crystal: the user's option meeting the crystal's projectors.
//!
//! [`build`] turns a [`Request`] (element, shell, `U`, `J`, form) and the labels of a crystal
//! (which projector belongs to which atom, `l`, `m` and radial index; the reference-atom
//! channels of each species) into the [`Spec`] that the crystal snapshot carries: one shell per
//! atom of each requested element, its projectors ordered `m` then radial index (global
//! positions in the crystal's projector list), the PAW partial-wave weights of
//! [`paw::shell`](crate::paw::shell), and the seed the SCF starts from (the reference electrons
//! of the shell and the atom's signed initial moment). [`check_loaded`] compares the option with
//! the table of a crystal file, which owns the correction.
//!
//! Nothing here knows Python: the labels are plain slices, the refusals are
//! [`Error`](crate::Error) values naming `hubbard_u` and the element.
use super::hubbard::{Seed, Shell, Spec, MAX_SHELLS};
use super::invalid;
use crate::{
    models::hubbard_request::Request,
    paw::{
        onecenter::{Basis, Channel},
        shell,
    },
    Context, Error, Result,
};
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

/// One channel of the reference atom of a species, as the dataset stores it.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelData {
    pub waves: usize,
    /// `waves x grid`, row major.
    pub ae: Vec<f64>,
    /// `waves x waves`, row major.
    pub vanderbilt: Vec<f64>,
    pub r_cut: f64,
    /// One number per partial wave (see [`shell::weights`]).
    pub occupations: Vec<f64>,
}

/// The reference-atom data of one species: the radial grid and its channels by `l`.
#[derive(Debug, Clone, PartialEq)]
pub struct Species {
    pub r: Vec<f64>,
    pub channels: BTreeMap<usize, ChannelData>,
    /// The file the dataset was read from (empty for a dataset that has none).
    pub source: String,
}

/// The label of one projector of the crystal, in the crystal's projector order.
#[derive(Debug, Clone, PartialEq)]
pub struct Projector {
    pub symbol: String,
    /// The projector basis the crystal's projections were built in (`"raw"` or `"dual"`).
    pub basis: String,
    pub atom: usize,
    pub l: usize,
    pub m: i64,
    /// Radial index within the `(atom, l)` channel.
    pub index: usize,
}

/// Everything about the crystal that the table needs.
pub struct Layout<'a> {
    pub projectors: &'a [Projector],
    /// For each atom, the index of its species in `species`.
    pub species_of_atom: &'a [usize],
    pub species: &'a [Species],
    /// The signed initial magnetic moment of each atom.
    pub moments: &'a [f64],
    /// The crystal carries spin-orbit matrices.
    pub spin_orbit: bool,
}

/// Layout backed directly by already-loaded PAW records.
pub struct PawLayout<'a> {
    pub projectors: &'a [Projector],
    pub species_of_atom: &'a [usize],
    pub species: &'a [std::sync::Arc<crate::paw::record::PawDataset>],
    /// Dataset file for each species, used in diagnostic messages.
    pub sources: &'a [String],
    pub moments: &'a [f64],
    pub spin_orbit: bool,
}

/// What the geometry check needs of one element's dataset: the file it came from and the
/// reference occupations of each channel.
pub struct Dataset<'a> {
    pub symbol: &'a str,
    pub source: &'a str,
    pub occupations: BTreeMap<usize, &'a [f64]>,
}

impl<'a> Dataset<'a> {
    pub fn of(symbol: &'a str, species: &'a Species) -> Self {
        Self {
            symbol,
            source: &species.source,
            occupations: species
                .channels
                .iter()
                .map(|(l, channel)| (*l, channel.occupations.as_slice()))
                .collect(),
        }
    }

    /// The dataset set (the folder of the dataset file, for example `lda-sr`).
    fn set(&self) -> String {
        dataset_set(self.source)
    }
}

/// The dataset set a dataset file belongs to: the name of its folder, or `unknown` for a
/// dataset that has no file (for example one generated in memory).
fn dataset_set(source: &str) -> String {
    Path::new(source)
        .parent()
        .and_then(Path::file_name)
        .map_or_else(
            || "unknown".into(),
            |name| name.to_string_lossy().into_owned(),
        )
}

/// The refusals that need only the elements, their datasets and the kind of calculation, not
/// the crystal's projectors: [`build`] runs them first, and the calculator runs them at
/// construction when the atoms are known, before any crystal is assembled.
///
/// A spin-orbit crystal, an element of the option that no dataset covers, a dataset without
/// the requested channel and a channel whose reference atom holds no electron (no bound
/// orbital to project on: Ba 5d, Au 5f in `lda-sr`) raise, naming the element, the shell and
/// the dataset set.
pub fn check_datasets(request: &Request, spin_orbit: bool, datasets: &[Dataset<'_>]) -> Result<()> {
    if spin_orbit {
        return Err(Error::Unsupported(
            "hubbard_u with a spin-orbit (fully relativistic) dataset: DFT+U is implemented for \
             scalar-relativistic datasets only"
                .into(),
        ));
    }
    let missing: Vec<&str> = request
        .entries
        .iter()
        .filter(|e| datasets.iter().all(|d| d.symbol != e.symbol))
        .map(|e| e.symbol.as_str())
        .collect();
    if !missing.is_empty() {
        return Err(invalid(format!(
            "hubbard_u names {}, which is not in the system",
            missing.join(", ")
        )));
    }
    for entry in &request.entries {
        for dataset in datasets.iter().filter(|d| d.symbol == entry.symbol) {
            let Some(occupations) = dataset.occupations.get(&entry.l) else {
                return Err(invalid(format!(
                    "hubbard_u: the {} dataset has no '{}' channel (dataset set {})",
                    entry.symbol,
                    entry.letter(),
                    dataset.set()
                )));
            };
            if shell::reference_wave(occupations).is_none() {
                return Err(invalid(format!(
                    "hubbard_u: the '{}' shell of {} has no occupied reference wave in the {} \
                     dataset set (its reference atom holds no electron there, so there is no \
                     bound orbital to project on); use another shell, another element or \
                     another dataset set",
                    entry.letter(),
                    entry.symbol,
                    dataset.set()
                )));
            }
        }
    }
    Ok(())
}

/// One dataset per element of the crystal, in order of first appearance.
fn datasets_of<'a>(layout: &'a Layout<'a>) -> Result<Vec<Dataset<'a>>> {
    let mut out: Vec<Dataset<'a>> = Vec::new();
    for p in layout.projectors {
        if out.iter().any(|d| d.symbol == p.symbol) {
            continue;
        }
        let species = layout
            .species_of_atom
            .get(p.atom)
            .and_then(|i| layout.species.get(*i))
            .ok_or_else(|| invalid(format!("hubbard_u: atom {} has no dataset", p.atom)))?;
        out.push(Dataset::of(&p.symbol, species));
    }
    Ok(out)
}

/// The first projector of an atom's shell and the projectors by `(m, radial index)`.
struct Slots {
    atom: usize,
    first: usize,
    by_label: HashMap<(i64, usize), usize>,
}

/// Builds the table of `layout` for `request`.
pub fn build(ctx: &Context, request: &Request, layout: &Layout<'_>) -> Result<Spec> {
    check_datasets(request, layout.spin_orbit, &datasets_of(layout)?)?;
    build_table(request, layout.projectors, layout.species_of_atom, layout.moments,
        |species_index, l, basis, entry| {
            let species = layout.species.get(species_index)
                .ok_or_else(|| invalid("hubbard_u: atom has no dataset"))?;
            let set = dataset_set(&species.source);
            let named = |error: Error| match error {
                Error::Invalid(message) => invalid(format!(
                    "hubbard_u: the '{}' shell of {} in the {set} dataset set: {message}",
                    entry.letter(), entry.symbol)),
                other => other,
            };
            let channel = species.channels.get(&l).ok_or_else(|| invalid(format!(
                "hubbard_u: the {} dataset has no '{}' channel (dataset set {set})",
                entry.symbol, entry.letter())))?;
            let view = Channel { l, waves: channel.waves, ae: &channel.ae, ps: &[],
                vanderbilt: &channel.vanderbilt, r_cut: channel.r_cut };
            let weights = shell::weights(ctx, &view, &species.r, &channel.occupations,
                Basis::parse(basis)?).map_err(named)?;
            let electrons = shell::reference_electrons(&channel.occupations).map_err(named)?;
            Ok((weights, electrons))
        })
}

/// Build the same table while borrowing radial arrays from PAW records.
pub fn build_paw(ctx: &Context, request: &Request, layout: &PawLayout<'_>) -> Result<Spec> {
    if layout.sources.len() != layout.species.len() {
        return Err(invalid("hubbard_u: one dataset source per species is required"));
    }
    let mut datasets = Vec::new();
    for p in layout.projectors {
        let index = *layout.species_of_atom.get(p.atom)
            .ok_or_else(|| invalid(format!("hubbard_u: atom {} has no dataset", p.atom)))?;
        let ds = layout.species.get(index)
            .ok_or_else(|| invalid(format!("hubbard_u: atom {} has no dataset", p.atom)))?;
        if p.symbol != ds.symbol {
            return Err(invalid("hubbard_u: projector symbol disagrees with its dataset"));
        }
        if datasets.iter().any(|d: &Dataset<'_>| d.symbol == p.symbol) { continue; }
        datasets.push(Dataset { symbol: &ds.symbol, source: &layout.sources[index],
            occupations: ds.channels.iter().map(|(&l, c)| (l as usize, c.occupations.data())).collect() });
    }
    check_datasets(request, layout.spin_orbit, &datasets)?;
    build_table(request, layout.projectors, layout.species_of_atom, layout.moments,
        |species_index, l, basis, entry| {
            let ds = layout.species.get(species_index)
                .ok_or_else(|| invalid("hubbard_u: atom has no dataset"))?;
            let set = dataset_set(&layout.sources[species_index]);
            let channel = ds.channels.get(&(l as u32)).ok_or_else(|| invalid(format!(
                "hubbard_u: the {} dataset has no '{}' channel (dataset set {set})",
                entry.symbol, entry.letter())))?;
            let named = |error: Error| match error {
                Error::Invalid(message) => invalid(format!(
                    "hubbard_u: the '{}' shell of {} in the {set} dataset set: {message}",
                    entry.letter(), entry.symbol)),
                other => other,
            };
            let weights = shell::dataset_weights(ctx, channel, ds.r.as_slice(), Basis::parse(basis)?)
                .map_err(named)?;
            let electrons = shell::reference_electrons(channel.occupations.data()).map_err(named)?;
            Ok((weights, electrons))
        })
}

fn build_table(
    request: &Request,
    projectors: &[Projector],
    species_of_atom: &[usize],
    moments: &[f64],
    weights_for: impl Fn(usize, usize, &str, &crate::models::hubbard_request::Entry)
        -> Result<(Vec<f64>, f64)>,
) -> Result<Spec> {
    // The shells in order of the first projector of each atom, as the crystal lists them.
    let mut shells: Vec<Slots> = Vec::new();
    let mut at: HashMap<usize, usize> = HashMap::new();
    for (position, p) in projectors.iter().enumerate() {
        if request.entry(&p.symbol).is_some_and(|e| e.l == p.l) {
            let slot = *at.entry(p.atom).or_insert_with(|| {
                shells.push(Slots {
                    atom: p.atom,
                    first: position,
                    by_label: HashMap::new(),
                });
                shells.len() - 1
            });
            shells[slot].by_label.insert((p.m, p.index), position);
        }
    }
    for entry in &request.entries {
        if shells
            .iter()
            .all(|s| projectors[s.first].symbol != entry.symbol)
        {
            return Err(invalid(format!(
                "hubbard_u: the {} dataset has no '{}' channel",
                entry.symbol,
                entry.letter()
            )));
        }
    }
    if shells.len() > MAX_SHELLS {
        return Err(invalid(format!(
            "hubbard_u covers {} atoms; a Hubbard table holds at most {MAX_SHELLS} shells",
            shells.len()
        )));
    }
    let mut cache: HashMap<(usize, usize, &str), (Vec<f64>, f64)> = HashMap::new();
    let mut table = Vec::with_capacity(shells.len());
    for slots in &shells {
        let first = &projectors[slots.first];
        let entry = request
            .entry(&first.symbol)
            .ok_or_else(|| invalid("hubbard_u: a projector names an element the option lacks"))?;
        let (atom, l, basis) = (slots.atom, entry.l, first.basis.as_str());
        let species_index = *species_of_atom
            .get(atom)
            .ok_or_else(|| invalid(format!("hubbard_u: atom {atom} has no dataset")))?;
        let key = (species_index, l, basis);
        if !cache.contains_key(&key) {
            cache.insert(key, weights_for(species_index, l, basis, entry)?);
        }
        let (weights, electrons) = &cache[&key];
        let moment = *moments
            .get(atom)
            .filter(|m| m.is_finite())
            .ok_or_else(|| {
                invalid(format!(
                    "hubbard_u: atom {atom} needs a finite initial magnetic moment"
                ))
            })?;
        let width = l as i64;
        let listed = (-width..=width)
            .flat_map(|m| (0..weights.len()).map(move |i| (m, i)))
            .map(|label| {
                slots.by_label.get(&label).copied().ok_or_else(|| {
                    invalid(format!(
                        "hubbard_u: the projectors of atom {atom} ({}) lack m = {}, radial index {} \
                         of the '{}' shell",
                        entry.symbol,
                        label.0,
                        label.1,
                        entry.letter()
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        table.push(Shell {
            atom,
            symbol: entry.symbol.clone(),
            l,
            u: entry.u,
            j: entry.j,
            projectors: listed,
            weights: weights.clone(),
            basis: Some(basis.to_string()),
            seed: Seed::Moment {
                n_electrons: *electrons,
                moment,
            },
        });
    }
    Ok(Spec {
        form: request.form(),
        shells: table,
    })
}

/// `symbol: l, U, J` of each element of a table, in Hartree; the shells of one element must
/// agree, otherwise `None` marks the element as inconsistent.
fn describe<'a>(
    shells: impl Iterator<Item = (&'a str, usize, f64, f64)>,
) -> BTreeMap<&'a str, Option<(usize, f64, f64)>> {
    let mut out: BTreeMap<&str, Option<(usize, f64, f64)>> = BTreeMap::new();
    for (symbol, l, u, j) in shells {
        let value = Some((l, u, j));
        out.entry(symbol)
            .and_modify(|seen| {
                if *seen != value {
                    *seen = None;
                }
            })
            .or_insert(value);
    }
    out
}

fn show(map: &BTreeMap<&str, Option<(usize, f64, f64)>>) -> String {
    if map.is_empty() {
        return "no correction".into();
    }
    map.iter()
        .map(|(symbol, v)| match v {
            Some((l, u, j)) => format!("{symbol}: (l = {l}, U = {u}, J = {j})"),
            None => format!("{symbol}: (shells that differ)"),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Compares the option with the table of a loaded crystal file (`None` for a file with no
/// correction). The file owns the correction: an option that names other shells, other
/// energies (compared exactly: the file stores the Hartree values [`Request::parse`] computes
/// from the eV option, and the JSON round trip of a float is exact) or another form raises.
pub fn check_loaded(request: &Request, loaded: Option<&Spec>) -> Result<()> {
    let wanted = describe(
        request
            .entries
            .iter()
            .map(|e| (e.symbol.as_str(), e.l, e.u, e.j)),
    );
    let file = describe(
        loaded
            .into_iter()
            .flat_map(|spec| spec.shells.iter())
            .map(|s| (s.symbol.as_str(), s.l, s.u, s.j)),
    );
    let same_form = loaded.is_none_or(|spec| spec.form == request.form());
    if file == wanted && same_form {
        return Ok(());
    }
    Err(invalid(format!(
        "hubbard_u disagrees with the crystal file, which owns the DFT+U correction (file: {}; \
         option: {}, energies in Hartree, form '{}'); drop the option to use the file's",
        show(&file),
        show(&wanted),
        request.form().name()
    )))
}
