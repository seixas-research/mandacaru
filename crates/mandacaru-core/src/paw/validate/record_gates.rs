//! The gates a record answers from its own tables and scalars, with no all-electron atom: reload
//! (G1), the core record (G5), conditioning (G6), hardness and working spacing (G7), sphere and
//! partition (G8), the smooth-core charge (G9), transferability (G10, external) and grid
//! uniformity (G11). Every number is measured on the record as stored (the ten-digit tables of the
//! file), never on the generator's in-memory copy.
use super::report::{Row, Status};
use crate::{
    paw::{
        diagnose::quality::{
            hardness, partition_mismatch, residual_kinetic, shortest_bond, working_spacing,
            HARDNESS_TOLERANCE,
        },
        generate::assemble::{COUPLING_ASYMMETRY_TOLERANCE, DUALITY_TOLERANCE, OVERLAP_MINIMUM},
        record::{PawDataset, FORMAT_VERSION, RELOAD_LEVEL_TOLERANCE},
    },
    Context, Result,
};
use std::f64::consts::PI;

/// Largest difference (electrons) between the stored core occupations' sum and `Z - Zv`.
pub const CORE_CHARGE_TOLERANCE: f64 = 1e-8;
/// Largest pointwise difference between the stored core orbitals' density and the stored
/// `core_density`, relative to the peak of the latter. Measured on the 368 regenerated files
/// (2026-11-13): 1.4e-9 at most, the ten-digit storage of the tables.
pub const CORE_DENSITY_TOLERANCE: f64 = 1e-8;
/// From this multiple of the widest sphere radius out, the ionic potential is Coulombic (G14).
pub const IONIC_TAIL_RADIUS: f64 = 1.6;
/// The deviation of `r v_local` from `-Zv` tolerated there (e): the 92 scalar-relativistic PBE files
/// have 1.1e-3 at worst (Lu) and 2e-6 at the median; the fully relativistic files of the first
/// regeneration had 4e-3 at the median and 0.09 at worst (Pb), and the second 8e-3 (Bi, 3.5 Bohr) and
/// 5e-3 at 20 Bohr, which over the whole table sum to -1.4 Ha bohr^3.
pub const IONIC_TAIL_TOLERANCE: f64 = 2e-3;
/// The largest ionic constant tolerated beyond 1.6 r_c (Ha bohr^3): the first two regenerations had
/// 30 (Bi) to 110 (Tl) and -1.4 (Bi); 0.1 moves the energy of a 100 bohr^3 per atom crystal with
/// 16 valence electrons by 4 meV per atom and its pressure by 0.2 GPa (the 184 scalar-relativistic files: -0.069 at worst, Pt).
pub const IONIC_CONSTANT_TOLERANCE: f64 = 0.1;
/// Largest relative deviation of the radial grid from `r_i = (i + 1) h` (the core-valence exchange
/// reads the nodes that way).
pub const GRID_UNIFORMITY_TOLERANCE: f64 = 1e-4;
/// Largest difference (electrons) between the integral of the stored smooth core density and the
/// record's own `partial_core_electrons`. The stored table is decimated by the library stride and
/// has no node at the origin, where the generator's integral adds a cap: measured on the 92
/// regenerated `pbe-sr` files, 3.2e-6 at most (Li, the largest smooth core at the origin: the cap is
/// 2.2e-6 of it, the rest the stride), 1e-9 elsewhere. A file kept at a stride ten times coarser
/// misses by 2e-4 (a test), so the threshold stands at 1e-4.
pub const SMOOTH_CORE_TOLERANCE: f64 = 1e-4;
/// The working spacing (Angstrom) a file is expected to converge at; a coarser one is recorded in
/// the table, not refused.
pub const WORKING_SPACING_TARGET: f64 = 0.15;

/// `np.trapezoid(y, x)`: the trapezoid rule on the stored nodes, the first one at `r[0]`.
fn trapezoid(x: &[f64], y: impl Fn(usize) -> f64) -> f64 {
    (1..x.len())
        .map(|i| 0.5 * (y(i - 1) + y(i)) * (x[i] - x[i - 1]))
        .sum()
}

/// G1: the file read back reproduces the generated levels. `None`: a file read alone has no
/// generated record to compare with.
pub fn reload(shift: Option<f64>) -> Row {
    match shift {
        Some(shift) => Row::at_most(
            "G1",
            "reload: largest bound-level shift (Ha)",
            shift,
            RELOAD_LEVEL_TOLERANCE,
        ),
        None => Row::not_applicable(
            "G1",
            "reload",
            "a file read alone has no generated record to compare with",
        ),
    }
}

/// G11: `r_i = (i + 1) h` to [`GRID_UNIFORMITY_TOLERANCE`] of `h`.
pub fn grid_uniformity(ds: &PawDataset) -> Row {
    let r = ds.r.as_slice();
    if r.len() < 2 {
        return Row::flag("G11", "radial grid", false, "fewer than two nodes");
    }
    let h = r[1] - r[0];
    let step = (1..r.len())
        .map(|i| (r[i] - r[i - 1] - h).abs())
        .fold(0f64, f64::max);
    let value = step.max((r[0] - h).abs()) / h;
    Row::at_most(
        "G11",
        "radial grid deviation from (i+1)h, relative to h",
        value,
        GRID_UNIFORMITY_TOLERANCE,
    )
}

/// G14: the ionic local potential is the bare Coulomb `-Zv/r` wherever the smooth valence density
/// has handed the electrostatics over to the augmentation. Two rows over the table from
/// [`IONIC_TAIL_RADIUS`] times the widest sphere out, where a crystal's valence density is smooth:
///
/// - the tail: `max |r v_local + Zv|` is at most [`IONIC_TAIL_TOLERANCE`] (electrons);
/// - the constant: `alpha = int (v_local + Zv/r) 4 pi r^2 dr` over the same range is at most
///   [`IONIC_CONSTANT_TOLERANCE`] in magnitude. A crystal's smooth density `n` meets it as
///   `n alpha` per atom: an energy shift of `N_valence alpha / V` and a pressure of the same size
///   per volume. The first fully relativistic datasets, whose Dirac atom filled an open subshell
///   `j` by `j` while the scalar partial waves are its spherical average, had `alpha` of 30 Ha bohr^3
///   (Bi) to 110 (Tl) and measured 105 eV and -90 GPa on Bi2Se3; the second, whose Dirac and scalar
///   densities still differed by 1e-3 electrons beyond 10 Bohr, -1.4 (Bi) and -6 GPa. An isolated
///   atom in a box does not show it (its density is negligible out there): only the table does.
pub fn ionic_tail(ds: &PawDataset) -> Vec<Row> {
    let widest = ds
        .channels
        .values()
        .map(|c| c.r_cut)
        .fold(ds.r_cut_local, f64::max);
    let (r, v) = (ds.r.as_slice(), ds.v_local.as_slice());
    let from = IONIC_TAIL_RADIUS * widest;
    let deviation = (0..r.len())
        .filter(|&i| r[i] >= from)
        .map(|i| (r[i] * v[i] + ds.valence_charge).abs())
        .fold(0f64, f64::max);
    let alpha = trapezoid(r, |i| {
        if r[i] >= from {
            (v[i] + ds.valence_charge / r[i]) * 4. * PI * r[i] * r[i]
        } else {
            0.
        }
    });
    vec![
        Row::at_most(
            "G14",
            "ionic tail: max |r v_local + Zv| from 1.6 r_c to the end of the table (e)",
            deviation,
            IONIC_TAIL_TOLERANCE,
        ),
        Row::at_most(
            "G14",
            "ionic constant: |int (v_local + Zv/r) d3r| from 1.6 r_c out (Ha bohr^3)",
            alpha.abs(),
            IONIC_CONSTANT_TOLERANCE,
        )
        .with_detail(format!("{alpha:.4}")),
    ]
}

/// G5: the core record.
pub fn core(ds: &PawDataset) -> Vec<Row> {
    let mut rows = vec![Row::flag(
        "G5",
        "format version 3 with the core states",
        ds.core_states.is_some() && ds.format_version() == FORMAT_VERSION,
        format!("format {}", ds.format_version()),
    )];
    rows.push(Row::flag(
        "G5",
        "no reader default applied",
        ds.defaults_applied.is_empty(),
        ds.defaults_applied.join(", "),
    ));
    rows.push(Row::flag(
        "G5",
        "no core_scheme field",
        ds.core_scheme.is_none(),
        ds.core_scheme.clone().unwrap_or_default(),
    ));
    let Some(states) = &ds.core_states else {
        for check in ["core charge", "core density reproduction"] {
            rows.push(Row::not_applicable(
                "G5",
                check,
                "the record has no core states",
            ));
        }
        return rows;
    };
    let r = ds.r.as_slice();
    let charge = states.iter().map(|s| s.occupation).sum::<f64>();
    let expected = ds.atomic_number as f64 - ds.valence_charge;
    rows.push(
        Row::at_most(
            "G5",
            "core charge: |sum occupation - (Z - Zv)| (e)",
            (charge - expected).abs(),
            CORE_CHARGE_TOLERANCE,
        )
        .with_detail(format!("{charge} stored, {expected} expected")),
    );
    if states.iter().any(|s| s.orbital.as_slice().len() != r.len()) {
        rows.push(Row::flag(
            "G5",
            "core density reproduction",
            false,
            "a core orbital is not on the radial grid",
        ));
        return rows;
    }
    let mut density = vec![0f64; r.len()];
    for state in states {
        for (i, rho) in density.iter_mut().enumerate() {
            let u = state.orbital.as_slice()[i];
            *rho += state.occupation * u * u / (4. * PI * r[i] * r[i]);
        }
    }
    let core = ds.core_density.as_slice();
    let peak = core.iter().fold(0f64, |m, x| m.max(x.abs()));
    let gap = density.iter().zip(core).fold(0f64, |m, (a, b)| {
        let d = (a - b).abs();
        if m.is_nan() || d.is_nan() {
            f64::NAN
        } else {
            m.max(d)
        }
    });
    rows.push(Row::at_most(
        "G5",
        "core density from the orbitals, relative to the peak",
        gap / peak.max(1e-300),
        CORE_DENSITY_TOLERANCE,
    ));
    let widest = ds.channels.values().map(|c| c.r_cut).fold(0f64, f64::max);
    let outside = trapezoid(r, |i| {
        if r[i] > widest {
            4. * PI * r[i] * r[i] * core[i]
        } else {
            0.
        }
    });
    rows.push(Row::measured(
        "G5",
        "stored core beyond the widest sphere (e)",
        outside,
        format!("beyond r_c = {widest:.4} Bohr; reported, never refused"),
    ));
    rows
}

/// G6: the worst conditioning figure over the scalar channels, as the generator stored it.
pub fn conditioning(ds: &PawDataset) -> Vec<Row> {
    let nan_max = |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    };
    let nan_min = |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    };
    let channels = ds.channels.values();
    vec![
        Row::at_most(
            "G6",
            "projector duality error",
            channels
                .clone()
                .fold(0., |m, c| nan_max(m, c.duality_error)),
            DUALITY_TOLERANCE,
        ),
        Row::at_most(
            "G6",
            "coupling asymmetry (Ha)",
            channels.clone().fold(0., |m, c| nan_max(m, c.asymmetry)),
            COUPLING_ASYMMETRY_TOLERANCE,
        ),
        Row::at_least(
            "G6",
            "smallest overlap-operator eigenvalue",
            channels.fold(f64::INFINITY, |m, c| nan_min(m, c.overlap_minimum)),
            OVERLAP_MINIMUM,
        ),
    ]
}

/// G7: hardness (a channel's residual kinetic energy above `q_cut`) and the working spacing.
pub fn hardness_and_grid(ctx: &Context, ds: &PawDataset) -> Result<Vec<Row>> {
    let worst = residual_kinetic(ds).values().fold(0., |m: f64, &e| {
        if m.is_nan() || e.is_nan() {
            f64::NAN
        } else {
            m.max(e)
        }
    });
    let hard = hardness(ds);
    let mut hardness_row = Row::at_most(
        "G7",
        "hardness: residual kinetic above q_cut (Ha)",
        worst,
        HARDNESS_TOLERANCE,
    );
    if !hard.is_empty() {
        hardness_row.status = Status::Fail;
        hardness_row.detail = format!("hard channels {hard:?}");
    }
    let spacing = match working_spacing(ctx, ds)? {
        Some(h) => Row::at_most(
            "G7",
            "working spacing (Angstrom)",
            h,
            WORKING_SPACING_TARGET,
        )
        .advisory()
        .with_detail("recorded in the table when coarser"),
        None => {
            let mut row = Row::not_applicable(
                "G7",
                "working spacing (Angstrom)",
                "no gate spacing down to 0.10 A converges the grid residual",
            );
            row.status = Status::Reported;
            row
        }
    };
    Ok(vec![hardness_row, spacing])
}

/// G8: the widest sphere against the shortest bond (`bond`, Bohr, or the element's own bond to
/// hydrogen), and the partition against a twin record of the same element.
pub fn sphere_and_partition(
    ds: &PawDataset,
    bond: Option<f64>,
    twin: Option<&PawDataset>,
) -> Vec<Row> {
    let widest = ds
        .channels
        .values()
        .map(|c| c.r_cut)
        .fold(ds.r_cut_local, f64::max);
    let sphere = match bond.or_else(|| shortest_bond(ds.atomic_number)) {
        Some(bond) => Row::below(
            "G8",
            "widest sphere against the shortest bond (Bohr)",
            widest,
            bond,
        )
        .advisory(),
        None => Row::not_applicable(
            "G8",
            "widest sphere against the shortest bond",
            "no covalent radius past Z = 96",
        ),
    };
    let partition = match twin {
        Some(other) => match partition_mismatch(ds, other) {
            None => Row::flag(
                "G8",
                "core/valence partition equals the twin's",
                true,
                format!("{} {}", other.xc, other.relativity),
            ),
            Some(text) => Row::flag(
                "G8",
                "core/valence partition equals the twin's",
                false,
                text,
            ),
        },
        None => Row::not_applicable(
            "G8",
            "core/valence partition equals the twin's",
            "no twin record named",
        ),
    };
    vec![sphere, partition]
}

/// G9: the electrons of the stored smooth core density (reported), and their distance from the
/// record's own `partial_core_electrons` (zero when the correction is not applied).
pub fn smooth_core(ds: &PawDataset) -> Vec<Row> {
    let r = ds.r.as_slice();
    let sc = ds.smooth_core_density.as_slice();
    let electrons = trapezoid(r, |i| 4. * PI * r[i] * r[i] * sc[i]);
    let measured = Row::measured(
        "G9",
        "smooth-core electrons, integrated (e)",
        electrons,
        "reported: no threshold",
    );
    let expected = match (ds.nlcc.applied, ds.nlcc.partial_core_electrons) {
        (false, _) => Some(0.),
        (true, stored) => stored,
    };
    match expected {
        Some(expected) => vec![
            measured,
            Row::at_most(
                "G9",
                "smooth-core charge against partial_core_electrons (e)",
                (electrons - expected).abs(),
                SMOOTH_CORE_TOLERANCE,
            )
            .with_detail(format!("{electrons:.8} integrated, {expected:.8} stored")),
        ],
        None => vec![measured],
    }
}

/// G10: the transferability check compares with an external all-electron code; it is a research
/// check, not a property the file can prove alone.
pub fn transferability() -> Row {
    Row::not_applicable(
        "G10",
        "transferability (IP2 against X2C)",
        "needs an external all-electron code: recorded by the pilot, not a per-file gate",
    )
}
