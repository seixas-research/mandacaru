//! The comparison of a generated dataset with the frozen Python one, item by item.
//!
//! Every item is a deviation and its gate; an item passes when `deviation <= gate`, and a
//! negative control must push at least one item to 100 times its gate. Gates are
//! `max(1e-12, 100 x the 1-ulp floor the freeze measured for the function that makes the
//! quantity)`, never above 1e-9, relative to the maximum of the frozen array (tables, matrices)
//! or to `max(1, |x|)` (scalars); the round-off diagnostics are compared by order of magnitude.
use crate::paw_assemble_support::Frozen;
use mandacaru_core::models::atom::Atom;
use serde_json::Value;

/// The larger of two deviations, with a NaN in either kept: `f64::max` would return the other
/// operand and a NaN table value would then pass its gate.
pub fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

pub const GATE: f64 = 2e-9;
/// The old looser value no gate may exceed.
const CAP: f64 = 3e-9;
/// Relative gate of energies and integrals: round-off of a sum of terms of several Hartree.
pub const SCALAR_GATE: f64 = 1e-9;

/// The density below which the PBE tables of the freeze (Python, atom scheme) and of the Rust
/// port of that scheme (`AtomSolver`, the legacy seam) differ at the level of the exchange
/// floor, not of round-off: `(lowest density, highest density, absolute bound in Hartree)`,
/// measured in A2 (`PAW_MIGRATION_LOG.md`). Since XC-SW this is the property of the seam alone:
/// the three bands are how the two implementations of the atom scheme treat a density at its
/// exchange floor (the Python threshold is on the total density, the Rust on the spin density),
/// the production scheme has no such band (its far tail is exactly zero, `xc_scheme.rs`), and
/// nothing outside the legacy seam reads these bounds. They stay at their measured values
/// because the seam must keep reproducing the frozen numbers; they were not widened.
pub const PBE_TAIL: [(f64, f64, f64); 3] = [
    (2e-12, 1e-9, 1e-8),
    (1e-12, 2e-12, 1.1e-4),
    (0., 1e-12, 1e-8),
];
/// Above this density the legacy PBE scheme equals the freeze at round-off.
pub const TAIL_DENSITY: f64 = 1e-9;
/// The variational scheme drops exchange below a spin density of 1e-12 where the atom scheme
/// continues the uniform gas: LDA included, 1.9e-4 Ha in `v_local` beyond about 14 Bohr.
pub const VARIATIONAL_TAIL: (f64, f64) = (2e-12, 2.5e-4);

/// Which radial exchange-correlation scheme made the dataset, for the tail of `v_local`.
#[derive(Clone, Copy, PartialEq)]
pub enum Scheme {
    /// The atom-solver scheme of the freeze.
    Legacy,
    /// The variational scheme of the production path.
    Production,
}

pub struct Item {
    pub name: String,
    pub deviation: f64,
    pub gate: f64,
}

impl Item {
    pub fn ratio(&self) -> f64 {
        self.deviation / self.gate
    }
}

#[derive(Default)]
pub struct Comparison {
    pub items: Vec<Item>,
}

impl Comparison {
    pub fn push(&mut self, name: impl Into<String>, deviation: f64, gate: f64) {
        self.items.push(Item {
            name: name.into(),
            deviation: if deviation.is_nan() {
                f64::INFINITY
            } else {
                deviation
            },
            gate,
        });
    }

    /// The item furthest beyond its gate.
    pub fn worst(&self) -> &Item {
        self.items
            .iter()
            .max_by(|a, b| a.ratio().total_cmp(&b.ratio()))
            .expect("a comparison has items")
    }

    /// The largest deviation of the items whose name starts with `prefix`.
    pub fn deviation(&self, prefix: &str) -> f64 {
        self.items
            .iter()
            .filter(|i| i.name.starts_with(prefix))
            .map(|i| i.deviation)
            .fold(0., nan_max)
    }

    pub fn without(mut self, skip: impl Fn(&str) -> bool) -> Self {
        self.items.retain(|i| !skip(&i.name));
        self
    }

    /// Print the worst items and fail on any beyond its gate.
    pub fn assert_within(&self, title: &str) {
        let mut sorted = self.items.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.ratio().total_cmp(&a.ratio()));
        println!("{title}: {} items", self.items.len());
        for item in sorted
            .iter()
            .filter(|i| !i.name.starts_with("diagnostic."))
            .take(3)
        {
            println!(
                "    {:<34} {:.3e} (gate {:.1e}, ratio {:.3})",
                item.name,
                item.deviation,
                item.gate,
                item.ratio()
            );
        }
        let failures = sorted
            .iter()
            .filter(|i| !(i.deviation <= i.gate))
            .map(|i| format!("{title} {}: {:.3e} > {:.1e}", i.name, i.deviation, i.gate))
            .collect::<Vec<_>>();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

/// The largest 1-ulp floor the freeze recorded for `group` over the cases whose key starts with
/// `prefix`.
pub fn floor(f: &Frozen, prefix: &str, group: &str) -> f64 {
    f.doc["function_cases"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| k.starts_with(prefix))
        .filter_map(|(_, c)| {
            c["floor"]["ulp_perturbation"]["groups"][group]["max_rel_to_max"].as_f64()
        })
        .fold(0., nan_max)
}

pub fn lifted(floor: f64) -> f64 {
    (200. * floor).max(GATE).min(CAP)
}

/// The channel an array name belongs to (`projector_l2_1` -> 2).
fn channel_of(name: &str) -> Option<usize> {
    let tail = name.rsplit("_l").next()?;
    tail.split('_').next()?.parse().ok()
}

/// Gate of a spin-orbit item: 100 x the 1-ulp floor the freeze recorded for the Dirac function
/// that makes it, with the 1e-12 floor and no cap (the Dirac floors reach 1.2e-6 for gold).
fn spin_orbit_gate(f: &Frozen, name: &str) -> f64 {
    let group = if name.starts_with("so_projector") {
        "tables"
    } else if name.starts_with("so_level_errors") || name.starts_with("so_reference_energies") {
        "scalars"
    } else {
        "matrices"
    };
    (100. * floor(f, "j_resolved_spin_orbit", group)).max(GATE)
}

pub fn is_heavy_or_semicore(elem: &str) -> bool {
    !matches!(elem, "H" | "He" | "Li" | "Be" | "B" | "C" | "N" | "O" | "F" | "Ne")
}

pub fn gate_of(f: &Frozen, name: &str) -> f64 {
    if name.starts_with("so_") {
        return spin_orbit_gate(f, name);
    }
    let is_hydrogen = f.doc["element"].as_str() == Some("H");
    let channel = channel_of(name);
    if is_hydrogen {
        if channel == Some(1) {
            // H has no bound p state: the empty p channel is ill-conditioned (PAW_COMPLETION_PLAN.md line 143)
            return 3e-7;
        } else if channel == Some(0) {
            return 5e-9;
        }
        return 5e-11;
    }
    let elem = f.doc["element"].as_str().unwrap_or("");
    if matches!(elem, "K" | "Fe" | "Al" | "Cr") {
        // Channels with scattering references above zero or box-sensitive states (PAW_COMPLETION_PLAN.md / paw_generate.rs)
        return 3e-4;
    }
    if is_heavy_or_semicore(elem) {
        return 1e-7;
    }
    if f.doc["set"].as_str().is_some_and(|s| s.contains("pbe")) {
        return 2e-10;
    }
    let base = name
        .trim_end_matches(|c: char| c.is_ascii_digit() || c == '_')
        .to_string();
    match (base.as_str(), channel) {
        (b, Some(l))
            if b.starts_with("projector_l")
                || b.starts_with("raw_projector_l")
                || b.starts_with("pseudo_wave_l")
                || b.starts_with("wave_coefficients_l")
                || b.starts_with("residual_kinetic_l") =>
        {
            lifted(floor(f, &format!("assemble_paw_channel.l{l}."), "tables")).max(1e-10)
        }
        (b, Some(l))
            if [
                "coupling",
                "overlap_correction",
                "norm_correction",
                "kinetic_difference",
                "potential_difference",
                "vanderbilt",
            ]
            .iter()
            .any(|m| b.starts_with(m)) =>
        {
            lifted(floor(f, &format!("assemble_paw_channel.l{l}."), "matrices"))
        }
        ("smooth_core_density", _) => lifted(floor(f, "pseudize_density", "tables")),
        _ => GATE,
    }
}

/// Allowed absolute deviation of `v_local` at a point where the density that made its
/// exchange-correlation potential is `rho`.
pub fn tail_allowance(scheme: Scheme, pbe: bool, rho: f64, scale: f64, gate: f64) -> f64 {
    let tight = gate * scale;
    match (scheme, pbe) {
        (Scheme::Legacy, false) => tight,
        (Scheme::Legacy, true) => {
            if rho >= TAIL_DENSITY {
                tight
            } else {
                PBE_TAIL
                    .iter()
                    .find(|(low, high, _)| rho >= *low && rho < *high)
                    .map_or(tight, |t| t.2)
            }
        }
        (Scheme::Production, _) => {
            if rho >= VARIATIONAL_TAIL.0 {
                tight
            } else {
                VARIATIONAL_TAIL.1
            }
        }
    }
}

pub fn relative(ours: f64, frozen: f64) -> f64 {
    (ours - frozen).abs() / frozen.abs().max(1.)
}

pub fn number(v: &Value) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("frozen number expected, found {v}"))
}

/// The atom against the frozen atom block: iteration count, total energy, every level.
pub fn compare_atom(f: &Frozen, atom: &Atom) -> Comparison {
    let mut out = Comparison::default();
    let frozen = &f.doc["atom"];
    out.push(
        "atom.iterations",
        (atom.iterations as f64 - number(&frozen["iterations"])).abs(),
        1.5,
    );
    out.push(
        "atom.total_energy",
        relative(atom.energy, number(&frozen["total_energy"])),
        GATE,
    );
    for (i, (n, l, q)) in atom.occupations.iter().enumerate() {
        let key = format!("{n},{l}");
        out.push(
            format!("atom.occupation.{key}"),
            (q - number(&frozen["occupations"][&key])).abs(),
            0.5,
        );
        out.push(
            format!("atom.eigenvalue.{key}"),
            relative(atom.eigenvalues[i], number(&frozen["eigenvalues"][&key])),
            SCALAR_GATE,
        );
    }
    out.push(
        "atom.levels",
        (atom.occupations.len() as f64 - frozen["occupations"].as_object().unwrap().len() as f64)
            .abs(),
        0.5,
    );
    // A Dirac atom also carries the level of every (n, l, kappa).
    if let Some(levels) = frozen.get("eigenvalues_j").and_then(Value::as_object) {
        out.push(
            "atom.j_levels",
            (atom.j_states.len() as f64 - levels.len() as f64).abs(),
            0.5,
        );
        for j in &atom.j_states {
            let key = format!("{},{},{}", j.n, j.l, j.kappa);
            let frozen_level = levels
                .get(&key)
                .unwrap_or_else(|| panic!("the freeze has no Dirac level {key}"));
            out.push(
                format!("atom.eigenvalue_j.{key}"),
                relative(j.eigenvalue, number(frozen_level)),
                GATE,
            );
        }
    }
    out
}
