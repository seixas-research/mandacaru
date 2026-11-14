//! Adding the core states (dataset format 3) to a dataset file written before they existed
//! (format 2): stage 4-0b of the core-valence exchange (`dev_plan/BLOCH_VQE_PLAN.md`).
//!
//! A format-2 file stores only the total `core_density`. Its core orbitals are those of the
//! reference atom it was generated on, which this module solves again: the dataset's own recipe
//! (`Recipe::for_dataset` / `Recipe::upaw`, with the record's frozen and semicore subshells, which
//! are the record's facts), its functional and relativistic treatment, and the radial
//! exchange-correlation scheme the shipped datasets were produced with (`AtomSolver`, the legacy
//! seam the module documentation of `models::xc_scheme` keeps for exactly this: diagnosing a
//! shipped file against the atom it was made on). Nothing is taken on trust:
//!
//! - the atom's grid, at the file's stride and under the ten-digit rule, must be the stored `r`
//!   exactly;
//! - the solved core, `sum occupation u^2 / (4 pi r^2)` over the core subshells, must reproduce
//!   the stored `core_density` ([`DENSITY_TOLERANCE`] of its peak at every point,
//!   [`CHARGE_TOLERANCE`] in the integrated charge).
//!
//! A file that fails either check is refused, by name and number, and nothing is written for it.
//! [`upgrade_file`] then writes the upgraded record beside nothing it read, and proves that only
//! the core states were added: every column of the input is in the output, by name and bit for
//! bit (a file written in generation order comes back in the record's ascending order); the
//! payload is the input's but for `version`, `core_states` and the keys the reader filled with its
//! documented defaults (`defaults_applied`, now written explicitly); and the content digest
//! (`PawDataset::content_sha256`) is unchanged.
use super::{
    recipe::Recipe,
    reference::solve_reference_with,
    shells::{orbital, partition},
};
use crate::models::{
    relativity::Treatment,
    xc_scheme::{AtomSolver, Variational, XcSource},
};
use crate::paw::{
    invalid,
    record::{digits::round_digits, CoreState, Family, PawDataset, WriteSpec, METADATA_KEY},
};
use crate::{parquet, Context, Result};
use std::{f64::consts::PI, path::Path};

/// Largest pointwise difference between the solved and the stored core density, relative to the
/// stored density's peak.
pub const DENSITY_TOLERANCE: f64 = 1e-7;
/// Largest difference (electrons) between the solved and the stored core charge.
pub const CHARGE_TOLERANCE: f64 = 1e-7;
/// When both radial schemes reproduce the stored core density (every LDA file, every core-free
/// file), the largest difference between their core orbitals, relative to each orbital's peak,
/// and between their core energies (Ha) for the file to be upgraded without a scheme (owner Q23).
pub const TIE_ORBITAL_TOLERANCE: f64 = 1e-7;
pub const TIE_ENERGY_TOLERANCE: f64 = 1e-6;

/// What the upgrade of one dataset found.
#[derive(Clone, Debug)]
pub struct Upgrade {
    pub symbol: String,
    /// The radial exchange-correlation scheme whose atom reproduces the stored core density
    /// (`record::CORE_SCHEMES`), identified by the file's own data (owner Q22); `None` when both
    /// schemes reproduce it and agree with each other (owner Q23).
    pub scheme: Option<&'static str>,
    /// On such a tie, the largest difference between the two schemes' core orbitals (relative to
    /// their peak) and core energies (Ha).
    pub tie: Option<(f64, f64)>,
    /// Every `stride`-th point of the generation grid is stored.
    pub stride: usize,
    /// `(n, l, occupation)` of the core subshells, in the order the density is summed.
    pub states: Vec<(u32, u32, f64)>,
    /// `max |n_solved - n_stored| / max n_stored`.
    pub density_deviation: f64,
    /// `|Q_solved - Q_stored|`, electrons.
    pub charge_deviation: f64,
    /// The stored core charge, electrons.
    pub core_charge: f64,
    /// The largest augmentation radius of the dataset (Bohr) and the stored core charge beyond it
    /// (electrons): reported, never refused (owner Q14).
    pub radius: f64,
    pub charge_outside: f64,
}

/// `int 4 pi r^2 f(r) dr` over `r[from..]` by trapezoids; from the origin (`from == 0`) the
/// segment `[0, r[0]]` is added with `f` taken constant there.
fn charge(r: &[f64], f: &[f64], from: usize) -> f64 {
    let origin = if from == 0 {
        4. * PI * f[0] * r[0].powi(3) / 3.
    } else {
        0.
    };
    origin
        + (from.max(1)..r.len())
            .map(|i| {
                0.5 * (r[i] - r[i - 1])
                    * 4.
                    * PI
                    * (r[i - 1] * r[i - 1] * f[i - 1] + r[i] * r[i] * f[i])
            })
            .sum::<f64>()
}

fn subshells(rows: &[Vec<i64>], what: &str) -> Result<Vec<(usize, usize)>> {
    rows.iter()
        .map(|row| match row.as_slice() {
            [n, l] if *n >= 1 && *l >= 0 => Ok((*n as usize, *l as usize)),
            _ => Err(invalid(format!("a {what} entry is not an (n, l) pair"))),
        })
        .collect()
}

/// The core states of the format-2 record `dataset`, solved again and checked against its stored
/// core density; the record with them (format 3) and what was found.
pub fn add_core_states(ctx: &Context, mut dataset: PawDataset) -> Result<(PawDataset, Upgrade)> {
    if dataset.core_states.is_some() {
        return Err(invalid(format!(
            "{} already carries its core states (format 3)",
            dataset.symbol
        )));
    }
    let relativity = Treatment::parse(&dataset.relativity)?;
    let mut recipe = match dataset.family {
        Family::PawLcao => Recipe::for_dataset(&dataset.symbol, &dataset.xc, relativity)?,
        Family::UpawLcao => Recipe::upaw(&dataset.symbol, &dataset.xc, relativity)?,
    };
    recipe.frozen_subshells = subshells(&dataset.frozen_subshells, "frozen subshell")?;
    recipe.semicore_subshells = subshells(&dataset.semicore_subshells, "semicore subshell")?;
    // The scheme is identified by the file's own data (owner Q22): the shipped library was
    // produced with the legacy `AtomSolver`, and the files regenerated since with the production
    // `Variational`; exactly one of them reproduces a file's stored core density.
    let mut matches = Vec::new();
    let mut report = Vec::new();
    for (scheme, source) in [
        ("atom-solver", &AtomSolver as &dyn XcSource),
        ("variational", &Variational),
    ] {
        // A scheme whose atom fails to converge or to land on the stored grid did not make the
        // file: a miss. Every scheme's numbers go into a refusal.
        match solve_core(ctx, &dataset, &recipe, scheme, source) {
            Ok((states, upgrade)) => {
                report.push(format!(
                    "{scheme}: {:.2e} of the peak, {:.2e} e",
                    upgrade.density_deviation, upgrade.charge_deviation
                ));
                if upgrade.density_deviation <= DENSITY_TOLERANCE
                    && upgrade.charge_deviation <= CHARGE_TOLERANCE
                {
                    matches.push((states, upgrade));
                }
            }
            Err(error) => report.push(format!("{scheme}: {error}")),
        }
    }
    let refuse = |what: &str, report: &[String]| {
        invalid(format!(
            "the reference atom of {} solved with {what} (tolerances {DENSITY_TOLERANCE:e} of the \
             peak, {CHARGE_TOLERANCE:e} e; {}): its core states cannot be added",
            dataset.symbol,
            report.join("; "),
        ))
    };
    let (states, upgrade) = match matches.len() {
        1 => matches.pop().expect("one match"),
        2 => {
            // Both schemes solve the same atom here (LDA has no gradient for them to differ on;
            // a core-free file has nothing to compare). The file cannot name its scheme, so the
            // upgrade records none, provided the two cores are the same to the tie tolerances.
            let (orbital, energy) = tie_gap(&matches[0].0, &matches[1].0);
            report.push(format!(
                "the two schemes' core orbitals differ by {orbital:.2e} of their peak and their \
                 energies by {energy:.2e} Ha (tolerances {TIE_ORBITAL_TOLERANCE:e}, \
                 {TIE_ENERGY_TOLERANCE:e} Ha)"
            ));
            if !(orbital <= TIE_ORBITAL_TOLERANCE && energy <= TIE_ENERGY_TOLERANCE) {
                return Err(refuse(
                    "each radial scheme reproduces its stored core density, but the two cores \
                     disagree",
                    &report,
                ));
            }
            // The states of the scheme the shipped library was produced with, `AtomSolver`
            // (listed first); the other's agree with them to the tolerances above.
            let (states, mut upgrade) = matches.swap_remove(0);
            upgrade.scheme = None;
            upgrade.tie = Some((orbital, energy));
            (states, upgrade)
        }
        _ => {
            return Err(refuse(
                "neither radial scheme reproduces its stored core density",
                &report,
            ))
        }
    };
    dataset.core_states = Some(states);
    dataset.core_scheme = upgrade.scheme.map(str::to_string);
    Ok((dataset, upgrade))
}

/// The largest difference between two solutions of the same core: orbitals relative to each
/// orbital's peak, energies in Ha. Infinite when the two do not list the same subshells.
fn tie_gap(a: &[CoreState], b: &[CoreState]) -> (f64, f64) {
    if a.len() != b.len()
        || a.iter()
            .zip(b)
            .any(|(x, y)| (x.n, x.l, x.occupation) != (y.n, y.l, y.occupation))
    {
        return (f64::INFINITY, f64::INFINITY);
    }
    a.iter().zip(b).fold((0f64, 0f64), |(o, e), (x, y)| {
        let (u, v) = (x.orbital.as_slice(), y.orbital.as_slice());
        let peak = u.iter().fold(0f64, |m, t| m.max(t.abs()));
        let gap = if u.len() == v.len() {
            u.iter().zip(v).fold(0f64, |m, (p, q)| m.max((p - q).abs()))
        } else {
            f64::INFINITY
        };
        (
            o.max(if peak > 0. { gap / peak } else { gap }),
            e.max((x.energy - y.energy).abs()),
        )
    })
}

/// The core of `dataset`'s reference atom solved with `source`: its states on the stored grid and
/// what they give against the stored core density.
fn solve_core(
    ctx: &Context,
    dataset: &PawDataset,
    recipe: &Recipe,
    scheme: &'static str,
    source: &dyn XcSource,
) -> Result<(Vec<CoreState>, Upgrade)> {
    let atom = solve_reference_with(ctx, recipe, source)?;
    if !atom.converged {
        return Err(invalid(format!(
            "the reference atom of {} did not converge",
            dataset.symbol
        )));
    }
    let shells = partition(&atom, recipe)?;
    let full = atom.r.as_slice();
    let stored = dataset.r.as_slice();
    let stride = (stored[0] / full[0]).round() as usize;
    let kept = |values: &[f64]| -> Vec<f64> {
        values
            .iter()
            .skip(stride.max(1) - 1)
            .step_by(stride.max(1))
            .copied()
            .collect()
    };
    let grid = kept(full);
    if stride == 0
        || grid.len() != stored.len()
        || grid
            .iter()
            .zip(stored)
            .any(|(a, b)| round_digits(*a).to_bits() != b.to_bits())
    {
        return Err(invalid(format!(
            "the reference atom of {} is not on the stored radial grid (stride {stride}): the \
             file was not generated with this recipe",
            dataset.symbol
        )));
    }
    let mut states = Vec::with_capacity(shells.core.len());
    let mut solved = vec![0.; stored.len()];
    for &(key, occupation) in &shells.core {
        let u = kept(orbital(&atom, key)?);
        for (i, rho) in solved.iter_mut().enumerate() {
            *rho += occupation * u[i] * u[i] / (4. * PI * stored[i] * stored[i]);
        }
        let at = atom
            .occupations
            .iter()
            .position(|s| (s.0, s.1) == key)
            .ok_or_else(|| invalid("a core subshell without its level"))?;
        states.push(CoreState {
            n: u32::try_from(key.0).map_err(|_| invalid("a core n out of range"))?,
            l: u32::try_from(key.1).map_err(|_| invalid("a core l out of range"))?,
            occupation,
            energy: *atom
                .eigenvalues
                .get(at)
                .ok_or_else(|| invalid("a core level without an eigenvalue"))?,
            orbital: ctx.copy_slice(&u)?,
        });
    }
    let core = dataset.core_density.as_slice();
    let peak = core.iter().fold(0f64, |m, x| m.max(x.abs()));
    let density_deviation = if peak > 0. {
        solved
            .iter()
            .zip(core)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max)
            / peak
    } else {
        solved.iter().fold(0f64, |m, x| m.max(x.abs()))
    };
    let core_charge = charge(stored, core, 0);
    let charge_deviation = (charge(stored, &solved, 0) - core_charge).abs();
    let radius = dataset
        .channels
        .values()
        .map(|c| c.r_cut)
        .fold(dataset.r_cut_local, f64::max);
    let from = stored.partition_point(|&r| r <= radius);
    let charge_outside = charge(stored, core, from);
    let upgrade = Upgrade {
        symbol: dataset.symbol.clone(),
        scheme: Some(scheme),
        tie: None,
        stride,
        states: states.iter().map(|s| (s.n, s.l, s.occupation)).collect(),
        density_deviation,
        charge_deviation,
        core_charge,
        radius,
        charge_outside,
    };
    Ok((states, upgrade))
}

/// Upgrade the format-2 dataset file `input` to format 3 at `output` (never the same file), and
/// prove that only the core states were added. Nothing is written when a check fails.
pub fn upgrade_file(ctx: &Context, input: &Path, output: &Path) -> Result<Upgrade> {
    let same = |a: &Path, b: &Path| match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if output.exists() || same(input, output) {
        return Err(invalid(format!(
            "{} exists: an upgrade never writes over a file",
            output.display()
        )));
    }
    let source = PawDataset::read(ctx, input)?;
    let before = source.content_sha256(ctx)?;
    let (upgraded, upgrade) = add_core_states(ctx, source)?;
    upgraded.write(
        ctx,
        output,
        &WriteSpec {
            stride: 1,
            codec: parquet::Codec::Zstd,
        },
    )?;
    drop(upgraded);
    let proved = prove_only_core_states(ctx, input, output, &before);
    if let Err(error) = proved {
        let _ = std::fs::remove_file(output);
        return Err(error);
    }
    Ok(upgrade)
}

/// The output's columns are the input's, at the same places and bit for bit, followed by the
/// core states; its payload is the input's but for `version` and `core_states`; its content
/// digest is the input's.
fn prove_only_core_states(ctx: &Context, input: &Path, output: &Path, before: &str) -> Result<()> {
    let old = parquet::read_table(ctx, input)?;
    let new = parquet::read_table(ctx, output)?;
    // By name: a file written in generation order (`lda-sr/As` lists l = 2 first) comes back in
    // the record's ascending order, as every write of a record does. The order is layout; the
    // values are the content.
    let find = |name: &str| new.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c);
    for (name, a) in &old.columns {
        let b = find(name)
            .ok_or_else(|| invalid(format!("the upgraded file lost the column {name}")))?;
        let (Some(a), Some(b)) = (a.as_f64(), b.as_f64()) else {
            return Err(invalid(format!("column {name} is not a float64 column")));
        };
        if a.len() != b.len() || a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits()) {
            return Err(invalid(format!(
                "column {name} of the upgraded file is not the input's bit for bit"
            )));
        }
    }
    let added = new.columns.len() - old.columns.len();
    if new.columns.len() < old.columns.len()
        || new
            .columns
            .iter()
            .filter(|(name, _)| !old.columns.iter().any(|(n, _)| n == name))
            .any(|(name, _)| !name.starts_with("core_wave_"))
        || new
            .columns
            .iter()
            .filter(|(name, _)| name.starts_with("core_wave_"))
            .count()
            != added
    {
        return Err(invalid(
            "the upgraded file added a column that is not a core state",
        ));
    }
    let payload = |table: &parquet::Table| -> Result<serde_json::Value> {
        let text = table
            .metadata
            .get(METADATA_KEY)
            .ok_or_else(|| invalid("a dataset file without its payload"))?;
        let mut value: serde_json::Value =
            serde_json::from_str(text).map_err(|e| invalid(e.to_string()))?;
        let map = value
            .as_object_mut()
            .ok_or_else(|| invalid("the payload is not an object"))?;
        map.remove("version");
        map.remove("core_states");
        map.remove("core_scheme");
        Ok(value)
    };
    // The keys the reader filled with its documented defaults (`PawDataset::defaults_applied`:
    // `semicore_subshells`, `channels.<l>.occupations` of a file written before they existed)
    // are written explicitly by the upgrade: removed from the upgraded payload before the
    // comparison, they are the only other difference allowed.
    let defaults = PawDataset::read(ctx, input)?.defaults_applied;
    let mut upgraded = payload(&new)?;
    for path in &defaults {
        let parts: Vec<&str> = path.split('.').collect();
        let removed = match parts.as_slice() {
            [key] => upgraded.as_object_mut().and_then(|m| m.remove(*key)),
            ["channels", l, key] => upgraded["channels"][*l]
                .as_object_mut()
                .and_then(|m| m.remove(*key)),
            _ => None,
        };
        if removed.is_none() {
            return Err(invalid(format!(
                "the default {path} of the input is not in the upgraded payload"
            )));
        }
    }
    if payload(&old)? != upgraded {
        return Err(invalid(
            "the upgraded payload differs from the input's beyond version, core_states and the \
             reader's defaults",
        ));
    }
    drop((old, new));
    if PawDataset::read(ctx, output)?.content_sha256(ctx)? != before {
        return Err(invalid("the upgraded file's content digest differs"));
    }
    Ok(())
}
