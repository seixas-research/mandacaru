//! The PAW-LCAO basis of one element: the radial shells a molecule and a crystal both use.
//!
//! One recipe for both geometries. The basis states are every occupied reference state of the
//! dataset (two in a semicore channel) and the bound level of every empty channel that has one
//! (except the scattering-only p channel of Li and Be). The first zeta of a state is the orbital
//! confined so that its level rises by `energy_shift` (eV) in the confining wall, or, unconfined,
//! the dataset's stored wave (an empty channel's bound eigenstate). Valence states are split into
//! `zetas` (`zeta_split`: every extra zeta from the first by tail norms, or from the previous one
//! by a squared-norm fraction); semicore states keep one. Polarization is either the
//! quasi-Gaussian shell sized from the confined orbital (`r_char = 0.25 r_c` at a fixed 0.3 eV in
//! the default wall) with `polarizations` zetas of one `l`, or `r^k R_outer`, one shell per extra
//! `l`. Every table is Fourier-filtered at `cutoff` unless the basis is unfiltered.
use super::confinement::{orbital, ChannelData, Orbital, Prepared};
use super::record::PawDataset;
use crate::{basis::filter, Context, Error, Result};

/// The default energy shift (eV) and confining wall `(amplitude Ha, r_i / r_c)` of the basis.
pub const DEFAULT_ENERGY_SHIFT: f64 = 0.1;
pub const DEFAULT_WALL: (f64, f64) = (12., 0.6);
/// Diffuse shells: the highest `l` a `"present"` recipe covers, how far past the widest support
/// their grid reaches, and the squared-norm residual below which one is dropped as dependent.
pub const DIFFUSE_PRESENT_L_MAX: usize = 2;
pub const DIFFUSE_GRID_REACH: f64 = 1.6;
pub const DIFFUSE_LINEAR_DEPENDENCE_TOL: f64 = 1e-4;
/// The tail norms of `zeta_split="first_zeta"` and the split norm of `"last_zeta"`.
pub const DEFAULT_TAIL_NORMS: [f64; 3] = [0.15, 0.31, 0.54];
pub const DEFAULT_SPLIT_NORM: f64 = 0.15;
/// The shift (eV, default wall) the Gaussian polarization width is sized at, and its fraction.
const POLARIZATION_REFERENCE_SHIFT: f64 = 0.3;
const POLARIZATION_CHARACTER_FRACTION: f64 = 0.25;

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

/// What one element's basis is built with.
#[derive(Clone, Debug)]
pub struct Recipe {
    pub zetas: usize,
    pub polarizations: usize,
    /// `None`: unconfined.
    pub energy_shift: Option<f64>,
    pub wall: (f64, f64),
    pub split_norm: f64,
    /// `Some`: every extra zeta split from the first (`first_zeta`); `None`: from the previous one.
    pub tail_norms: Option<Vec<f64>>,
    /// `None`: Gaussian when confined, `r^k R_outer` when not.
    pub gaussian_polarization: Option<bool>,
    /// The filter wave-vector (Bohr^-1); `None`: unfiltered.
    pub cutoff: Option<f64>,
    /// Diffuse shells after the ordinary ones: the channels (`None`: those present, up to
    /// `DIFFUSE_PRESENT_L_MAX`) and their sizing; `None`: no diffuse shell.
    pub diffuse: Option<(Option<Vec<usize>>, crate::basis::diffuse::Recipe)>,
}

impl Recipe {
    /// The defaults of a basis size `(zetas, polarizations)` at the filter `cutoff`.
    pub fn new(zetas: usize, polarizations: usize, cutoff: Option<f64>) -> Self {
        Self {
            zetas,
            polarizations,
            energy_shift: Some(DEFAULT_ENERGY_SHIFT),
            wall: DEFAULT_WALL,
            split_norm: DEFAULT_SPLIT_NORM,
            tail_norms: Some(DEFAULT_TAIL_NORMS.to_vec()),
            gaussian_polarization: None,
            cutoff,
            diffuse: None,
        }
    }

    /// The recipe of a request: `zetas`, `polarizations`, `cutoff` (Bohr^-1, `null` unfiltered,
    /// `"auto"` the caller's `auto_cutoff`), `energy_shift` (eV, `null` unconfined), `wall`,
    /// `split_norm`, `tail_norms`, `polarization` (`"gaussian"` / `"orbital"`) and `diffuse`.
    pub fn from_json(v: &serde_json::Value, auto_cutoff: Option<f64>) -> Result<Self> {
        use serde_json::Value;
        let count = |key: &str| -> Result<usize> {
            v[key].as_u64().map(|n| n as usize).ok_or_else(|| invalid(format!("the recipe needs {key}")))
        };
        let number = |value: &Value, what: &str| -> Result<f64> {
            value.as_f64().filter(|x| x.is_finite())
                .ok_or_else(|| invalid(format!("{what} must be a finite number")))
        };
        let mut out = Recipe::new(count("zetas")?, count("polarizations")?, None);
        out.cutoff = match &v["cutoff"] {
            Value::Null => None,
            Value::String(text) if text == "auto" => Some(auto_cutoff
                .ok_or_else(|| invalid("an \"auto\" filter cutoff needs the grid"))?),
            x => Some(number(x, "cutoff")?),
        };
        out.energy_shift = match &v["energy_shift"] {
            Value::Null => None,
            x => Some(number(x, "energy_shift")?),
        };
        if let Some(wall) = v["wall"].as_array() {
            if wall.len() != 2 {
                return Err(invalid("wall is [amplitude in Hartree, r_i / r_c]"));
            }
            out.wall = (number(&wall[0], "the wall amplitude")?, number(&wall[1], "the wall radius")?);
        }
        if !v["split_norm"].is_null() {
            out.split_norm = number(&v["split_norm"], "split_norm")?;
        }
        out.tail_norms = match &v["tail_norms"] {
            Value::Null => None,
            Value::Array(list) => Some(list.iter().map(|x| number(x, "a tail norm")).collect::<Result<_>>()?),
            _ => return Err(invalid("tail_norms must be a list or null")),
        };
        out.gaussian_polarization = match v["polarization"].as_str() {
            None => None,
            Some("gaussian") => Some(true),
            Some("orbital") => Some(false),
            Some(other) => return Err(invalid(format!("unknown polarization {other:?}"))),
        };
        out.diffuse = match &v["diffuse"] {
            Value::Null => None,
            d => {
                let channels = match &d["shells"] {
                    Value::Null => None,
                    Value::Array(list) => Some(list.iter().map(|x| x.as_u64().map(|l| l as usize)
                        .ok_or_else(|| invalid("a diffuse channel is an angular momentum")))
                        .collect::<Result<Vec<_>>>()?),
                    _ => return Err(invalid("diffuse shells are null or a list of l")),
                };
                let sizing = crate::basis::diffuse::Recipe {
                    count: d["count"].as_u64().map(|n| n as usize)
                        .ok_or_else(|| invalid("the diffuse recipe needs count"))?,
                    ratio: number(&d["ratio"], "the diffuse ratio")?,
                    cutoff_norm: number(&d["cutoff_norm"], "the diffuse cutoff_norm")?,
                    rms_ratio: number(&d["rms_ratio"], "the diffuse rms_ratio")?,
                    rms_radius: match &d["rms_radius"] {
                        Value::Null => None,
                        x => Some(number(x, "the diffuse rms_radius")?),
                    },
                };
                Some((channels, sizing))
            }
        };
        out.check()?;
        Ok(out)
    }

    fn check(&self) -> Result<()> {
        let (amplitude, inner) = self.wall;
        if !(amplitude.is_finite() && amplitude > 0. && inner > 0. && inner < 1.) {
            return Err(invalid("the confinement potential needs a positive amplitude and r_i / r_c in (0, 1)"));
        }
        if self.energy_shift.is_some_and(|x| !(x.is_finite() && x >= 1e-3)) {
            return Err(invalid("an energy_shift must be at least 1e-3 eV"));
        }
        if !(self.split_norm > 0. && self.split_norm < 1.)
            || self.tail_norms.as_ref().is_some_and(|t| t.is_empty() || t.iter().any(|x| !(*x > 0. && *x < 1.)))
        {
            return Err(invalid("the split norm and tail norms must lie in (0, 1)"));
        }
        if self.zetas == 0 {
            return Err(invalid("a basis needs at least one zeta"));
        }
        Ok(())
    }
}

/// One radial shell: `R(r)` on `r` (the dataset's table; a diffuse shell's own, longer grid).
#[derive(Clone, Debug)]
pub struct Shell {
    pub n: usize,
    pub l: usize,
    /// 1 for the first zeta.
    pub zeta: usize,
    pub polarization: bool,
    pub diffuse: bool,
    pub r: Vec<f64>,
    pub values: Vec<f64>,
}

/// The confined orbital of one basis state, for the run log.
#[derive(Clone, Debug)]
pub struct Confined {
    pub n: usize,
    pub l: usize,
    pub state: usize,
    pub orbital: Orbital,
}

/// The Gaussian polarization shell, for the run log.
#[derive(Clone, Debug)]
pub struct GaussianShell {
    pub l: usize,
    pub base_l: usize,
    pub r_cut: f64,
    pub r_char: f64,
    pub reference_radius: f64,
}

/// What the recipe resolved, beside the shells.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub confined: Vec<Confined>,
    pub polarization: Option<GaussianShell>,
    /// Every diffuse shell built, accepted or not.
    pub diffuse: Vec<crate::basis::diffuse::Row>,
}

/// The basis shells of the element of `ds`, in build order (states, then polarization).
pub fn species(ctx: &Context, ds: &PawDataset, recipe: &Recipe) -> Result<(Vec<Shell>, Report)> {
    recipe.check()?;
    let r = ds.r.as_slice();
    let band_limit = |values: &[f64], l: usize| -> Result<Vec<f64>> {
        match recipe.cutoff {
            None => Ok(values.to_vec()),
            Some(k_c) => Ok(filter::filter(ctx, r, values, l, k_c, &filter::Options::default())?.0),
        }
    };
    let states = basis_states(ctx, ds)?;
    let tails = recipe.tail_norms.as_deref();
    let (mut shells, mut report) = (Vec::new(), Report::default());
    let mut valence = std::collections::BTreeMap::new();
    let mut cutoffs = std::collections::BTreeMap::new();
    let mut first_radials = std::collections::BTreeMap::new();
    for (l, state, empty) in states {
        let lu = l as usize;
        let n = ds.channels[&l].n as usize + state;
        let radial = match recipe.energy_shift {
            Some(shift) => {
                let confined = confined_orbital(ctx, ds, l, state, shift, recipe.wall)?;
                let radial = confined.radial.clone();
                if !is_semicore(ds, n, l) {
                    cutoffs.insert((l, state), confined.r_c);
                }
                report.confined.push(Confined { n, l: lu, state, orbital: confined });
                radial
            }
            None if empty => empty_channel_radial(ctx, ds, l)?,
            None => ds.channels[&l].pseudo_waves.get(state)
                .ok_or_else(|| invalid("occupied state has no pseudo wave"))?
                .as_slice().to_vec(),
        };
        let semicore = is_semicore(ds, n, l);
        if !semicore {
            valence.insert(l, state);
            first_radials.entry(l).or_insert_with(|| radial.clone());
        }
        let count = if semicore { 1 } else { recipe.zetas };
        let zetas = crate::basis::radial::zeta_values(ctx, r, &radial, lu, count, recipe.split_norm, tails)?;
        for (index, zeta) in zetas.iter().enumerate() {
            shells.push(Shell {
                n, l: lu, zeta: index + 1, polarization: false, diffuse: false, r: r.to_vec(),
                values: band_limit(zeta.as_slice(), lu)?,
            });
        }
    }
    polarize(ctx, ds, recipe, &valence, &cutoffs, &first_radials, &band_limit, &mut shells, &mut report)?;
    if let Some((channels, sizing)) = &recipe.diffuse {
        add_diffuse(ctx, ds, recipe, channels.as_deref(), sizing, &mut shells, &mut report)?;
    }
    Ok((shells, report))
}

#[allow(clippy::too_many_arguments)]
fn polarize(
    ctx: &Context,
    ds: &PawDataset,
    recipe: &Recipe,
    valence: &std::collections::BTreeMap<u32, usize>,
    cutoffs: &std::collections::BTreeMap<(u32, usize), f64>,
    first_radials: &std::collections::BTreeMap<u32, Vec<f64>>,
    band_limit: &dyn Fn(&[f64], usize) -> Result<Vec<f64>>,
    shells: &mut Vec<Shell>,
    report: &mut Report,
) -> Result<()> {
    let r = ds.r.as_slice();
    let tails = recipe.tail_norms.as_deref();
    if recipe.polarizations == 0 {
        return Ok(());
    }
    let l_max = *valence.keys().max()
        .ok_or_else(|| invalid("polarization needs a valence channel"))?;
    if !recipe.gaussian_polarization.unwrap_or(recipe.energy_shift.is_some()) {
        // r^k R_outer: one shell per extra l, each a single zeta.
        let outermost = &first_radials[&l_max];
        for offset in 0..recipe.polarizations {
            let l = l_max as usize + 1 + offset;
            let shape = crate::basis::pao::power_shape(r, outermost, offset + 1)?;
            for zeta in crate::basis::radial::zeta_values(ctx, r, &shape, l, 1, recipe.split_norm, None)? {
                shells.push(Shell {
                    n: l + 1, l, zeta: 1, polarization: true, diffuse: false, r: r.to_vec(),
                    values: band_limit(zeta.as_slice(), l)?,
                });
            }
        }
        return Ok(());
    }
    if recipe.energy_shift.is_none() {
        return Err(invalid(format!(
            "the Gaussian polarization shell is sized from a confined orbital; {} is unconfined",
            ds.symbol)));
    }
    let l = (0..=l_max).find(|l| !valence.contains_key(l)).unwrap_or(l_max + 1);
    let base = l.checked_sub(1)
        .and_then(|l| valence.get(&l).map(|&state| (l, state)))
        .ok_or_else(|| invalid("polarization needs the preceding valence channel"))?;
    let r_cut = cutoffs[&base];
    let reference = confined_orbital(ctx, ds, base.0, base.1, POLARIZATION_REFERENCE_SHIFT, DEFAULT_WALL)?.r_c;
    let r_char = POLARIZATION_CHARACTER_FRACTION * reference;
    let shape = crate::basis::pao::polarization_radial(r, l as usize, r_cut, r_char)?;
    let zetas = crate::basis::radial::zeta_values_by_fraction(
        ctx, r, &shape, l as usize, recipe.polarizations, recipe.split_norm, tails,
    )?;
    for (index, zeta) in zetas.iter().enumerate() {
        shells.push(Shell {
            n: l as usize + 1, l: l as usize, zeta: index + 1, polarization: true, diffuse: false,
            r: r.to_vec(), values: band_limit(zeta.as_slice(), l as usize)?,
        });
    }
    report.polarization = Some(GaussianShell {
        l: l as usize, base_l: base.0 as usize, r_cut, r_char, reference_radius: reference,
    });
    Ok(())
}

/// The diffuse shells of the element, after its ordinary ones: sized from the free-atom valence rms
/// radius of each channel (the stored bound wave of each valence state, the outermost of a channel;
/// an empty channel's bound level), or of the outermost valence orbital for a channel without one.
fn add_diffuse(
    ctx: &Context,
    ds: &PawDataset,
    recipe: &Recipe,
    channels: Option<&[usize]>,
    sizing: &crate::basis::diffuse::Recipe,
    shells: &mut Vec<Shell>,
    report: &mut Report,
) -> Result<()> {
    let r = ds.r.as_slice();
    let mut reference = std::collections::BTreeMap::<usize, f64>::new();
    for (l, state, empty) in basis_states(ctx, ds)? {
        if is_semicore(ds, ds.channels[&l].n as usize + state, l) {
            continue;
        }
        let radial = if empty {
            empty_channel_radial(ctx, ds, l)?
        } else {
            ds.channels[&l].pseudo_waves[state].as_slice().to_vec()
        };
        let rms = crate::basis::diffuse::rms(r, &radial);
        let entry = reference.entry(l as usize).or_insert(0.);
        if rms > *entry {
            *entry = rms;
        }
    }
    let (&outermost, _) = reference.iter()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .ok_or_else(|| invalid(format!("{} has no valence orbital to size diffuse shells from", ds.symbol)))?;
    let present: std::collections::BTreeSet<usize> = shells.iter().map(|s| s.l).collect();
    let wanted: Vec<usize> = match channels {
        None => present.into_iter().filter(|&l| l <= DIFFUSE_PRESENT_L_MAX).collect(),
        Some(list) => list.to_vec(),
    };
    let info: Vec<(usize, usize, f64)> = wanted.iter().map(|&l| {
        let source = if reference.contains_key(&l) { l } else { outermost };
        (l, source, reference[&source])
    }).collect();
    let existing: Vec<(usize, usize, &[f64], &[f64])> = shells.iter()
        .map(|s| (s.n, s.l, s.r.as_slice(), s.values.as_slice())).collect();
    let (tables, rows) = crate::basis::diffuse::build_tables(
        ctx, r, &existing, &info, sizing, recipe.cutoff, DIFFUSE_GRID_REACH, DIFFUSE_LINEAR_DEPENDENCE_TOL,
    )?;
    for (grid, values, n, l) in tables {
        shells.push(Shell { n, l, zeta: 1, polarization: false, diffuse: true, r: grid, values });
    }
    report.diffuse = rows;
    Ok(())
}

/// The basis states `(l, state, empty)`: every occupied reference state, then the bound level of
/// every empty channel that has one (Li and Be keep their p channel for scattering only).
fn basis_states(ctx: &Context, ds: &PawDataset) -> Result<Vec<(u32, usize, bool)>> {
    let mut states = Vec::new();
    for (&l, channel) in &ds.channels {
        for (state, &occupation) in channel.occupations.data().iter().enumerate() {
            if occupation > 0. {
                states.push((l, state, false));
            }
        }
    }
    for (&l, channel) in &ds.channels {
        if channel.occupations.data().iter().any(|&o| o > 0.)
            || (l == 1 && (ds.symbol == "Li" || ds.symbol == "Be"))
        {
            continue;
        }
        if empty_channel_level(ctx, ds, l)? < -1e-3 {
            states.push((l, 0, true));
        }
    }
    Ok(states)
}

fn is_semicore(ds: &PawDataset, n: usize, l: u32) -> bool {
    ds.semicore_subshells.iter()
        .any(|shell| shell.len() >= 2 && shell[0] == n as i64 && shell[1] == l as i64)
}

fn channel_data<'a>(ds: &'a PawDataset, l: u32, wave: &'a [f64], projectors: &'a [f64]) -> ChannelData<'a> {
    let channel = &ds.channels[&l];
    ChannelData {
        table: ds.r.as_slice(),
        local: ds.v_local_screened.as_slice(),
        projectors,
        rank: channel.projectors.len(),
        coupling: channel.coupling_screened.data(),
        overlap: channel.overlap_correction.data(),
        wave,
        l: l as usize,
        r_cut: channel.r_cut,
    }
}

fn flat_projectors(ds: &PawDataset, l: u32) -> Vec<f64> {
    ds.channels[&l].projectors.iter().flat_map(|p| p.as_slice().iter().copied()).collect()
}

fn spectrum_box(data: &ChannelData) -> Result<f64> {
    let (lo, hi) = super::diagnose::spectrum::EXTENT_BOUNDS;
    super::checks::spectrum_extent(data.table, data.wave, super::diagnose::spectrum::EXTENT_FLOOR, lo, hi)
}

/// The lowest level (Hartree) of an empty channel in the box of the spectrum rule.
pub fn empty_channel_level(ctx: &Context, ds: &PawDataset, l: u32) -> Result<f64> {
    let wave = ds.channels[&l].pseudo_waves.first()
        .ok_or_else(|| invalid("empty channel has no reference pseudo wave"))?;
    let projectors = flat_projectors(ds, l);
    let data = channel_data(ds, l, wave.as_slice(), &projectors);
    let extent = spectrum_box(&data)?;
    super::pao::bound_level(ctx, data, extent)
}

/// The lowest eigenstate of an empty channel, `R(r)` on the dataset's table.
pub fn empty_channel_radial(ctx: &Context, ds: &PawDataset, l: u32) -> Result<Vec<f64>> {
    let wave = ds.channels[&l].pseudo_waves.first()
        .ok_or_else(|| invalid("empty channel has no reference pseudo wave"))?;
    let projectors = flat_projectors(ds, l);
    let data = channel_data(ds, l, wave.as_slice(), &projectors);
    let extent = spectrum_box(&data)?;
    Ok(super::pao::bound_radial(ctx, data, extent)?.0)
}

/// The orbital of channel `l`, `state`, confined so that its level rises by `shift` eV in `wall`.
pub fn confined_orbital(ctx: &Context, ds: &PawDataset, l: u32, state: usize, shift: f64,
                        (amplitude, inner_fraction): (f64, f64)) -> Result<Orbital> {
    let wave = ds.channels[&l].pseudo_waves.get(state)
        .ok_or_else(|| invalid("occupied state has no corresponding pseudo wave"))?;
    let projectors = flat_projectors(ds, l);
    let prepared = Prepared::new(ctx, channel_data(ds, l, wave.as_slice(), &projectors))?;
    let wall = crate::basis::confinement::Potential { amplitude, inner_fraction };
    orbital(ctx, &prepared, shift, &wall, state, None)?
        .map_err(|reach| invalid(format!("orbital does not fit table at reach {reach}")))
}
