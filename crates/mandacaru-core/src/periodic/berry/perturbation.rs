//! B3: Born effective charges and the clamped-ion (proper) piezoelectric tensor, by central
//! finite differences of the Berry-phase polarization over perturbed self-consistent crystals
//! (`dev_plan/BERRY_PLAN.md` section 10). Atomic units.
//!
//! A perturbation (an atom displaced along a Cartesian axis, or a Voigt strain of the cell) is
//! applied to the setup by a [`Perturber`], the perturbed crystal is solved from the converged
//! ground state (a warm start: the grid and its node count are the same, so the density, the
//! moments, the DFT+U occupations and the exchange state transfer), and its polarization is taken
//! on the string mesh. Perturbations run one after another, each SCF on the calculation's whole
//! processor layout.
//!
//! Every difference is of the `raw` coordinates in units of the quanta, FOLDED to the nearest
//! lattice point (`d - round(d)`): `raw` is continuous only modulo `f/g` (an electronic branch
//! jump) and `Z_a / g` (an ion crossing the cell boundary). A step whose folded change exceeds a
//! quarter quantum is refused: beyond it the fold may pick the wrong whole quantum.
//!
//! * `Z*_{a,ij} = (Omega / e) dP_i / du_{a,j}`, from `(P(+h) - P(-h)) / 2h`, the change of `P`
//!   being `sum_alpha d_alpha quanta_alpha` of the ground cell.
//! * The proper piezoelectric tensor `e_{i,v} = dP_i / d eta_v` (Voigt, engineering shear): the
//!   derivative of the reduced coordinates times the quanta of the UNSTRAINED cell, which does not
//!   depend on the branch.
//! * The acoustic sum rule `sum_a Z*_a = 0` is reported, and imposed (the per-atom mean removed)
//!   only on request.
use super::polarization::{polarization, Polarization};
use crate::{
    periodic::{bloch_mesh::Sampler, invalid, setup::PeriodicSetup, PeriodicResult},
    Context, Error, Result,
};
use nalgebra::Matrix3;
use std::f64::consts::PI;

/// The largest folded change of a coordinate (quanta) a step may make.
pub const QUARTER: f64 = 0.25;
/// Above this residual of the acoustic sum rule (e) the Born charges are returned with a warning.
pub const ASR_WARNING: f64 = 0.05;
/// The largest displacement (Bohr) and strain a step may take.
pub const MAX_DISPLACEMENT: f64 = 0.1;
pub const MAX_STRAIN: f64 = 0.02;
/// A perturbed run must keep the ground state's magnetization within this (Bohr magnetons).
pub const MAGNETIZATION_TOLERANCE: f64 = 1e-3;
/// The step (Bohr, or a strain times the shortest lattice vector) must be at least this multiple
/// of the SCF density tolerance: the per-run error of `P` is of the order of that tolerance.
pub const TOLERANCE_RATIO: f64 = 1e3;

/// One perturbation of the ground crystal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Perturbation {
    /// Atom `atom` moved by `delta` Bohr along the Cartesian `axis`.
    Displacement {
        atom: usize,
        axis: usize,
        delta: f64,
    },
    /// The cell strained by `delta` in the Voigt component `voigt` (xx, yy, zz, yz, xz, xy; a
    /// shear `delta` is the engineering strain, `eta_jk = eta_kj = delta / 2`), the atoms at
    /// fixed fractional coordinates (clamped ions).
    Strain { voigt: usize, delta: f64 },
}

impl Perturbation {
    /// How an error names the run.
    pub fn name(&self) -> String {
        match self {
            Self::Displacement { atom, axis, delta } => {
                format!(
                    "atom {atom} displaced by {delta:+e} Bohr along {}",
                    ["x", "y", "z"][*axis]
                )
            }
            Self::Strain { voigt, delta } => format!(
                "the cell strained by {delta:+e} in {}",
                ["xx", "yy", "zz", "yz", "xz", "xy"][*voigt]
            ),
        }
    }
}

/// The symmetric strain `eta` of a Voigt component (`I + eta` deforms the cell).
pub fn voigt_strain(voigt: usize, delta: f64) -> Result<Matrix3<f64>> {
    let mut eta = Matrix3::zeros();
    match voigt {
        0..=2 => eta[(voigt, voigt)] = delta,
        3..=5 => {
            let (j, k) = [(1, 2), (0, 2), (0, 1)][voigt - 3];
            eta[(j, k)] = delta / 2.;
            eta[(k, j)] = delta / 2.;
        }
        _ => return Err(invalid("a Voigt strain component is 0..6")),
    }
    Ok(eta)
}

/// Applies a perturbation to the setup of the ground crystal.
pub trait Perturber {
    fn setup(&self, ctx: &Context, perturbation: Perturbation) -> Result<PeriodicSetup>;
}

/// The production perturber over the Rust setup: the ground recipe rebuilt on the perturbed
/// geometry ([`PeriodicSetup::with_geometry`]) with the discretization of the ground setup held,
/// so every perturbed SCF is symmetrized with its own (lower) group and its k mesh reduced by it.
/// A strained cell is `(I + eta) L` with the atoms at their fractional coordinates; a displaced
/// atom moves by `delta` along a Cartesian axis.
pub struct SetupPerturber<'a> {
    pub setup: &'a PeriodicSetup,
}

impl Perturber for SetupPerturber<'_> {
    fn setup(&self, ctx: &Context, perturbation: Perturbation) -> Result<PeriodicSetup> {
        let ground = self.setup;
        let mut fractional = ground.fractional()?;
        match perturbation {
            Perturbation::Strain { voigt, delta } => {
                let eta = voigt_strain(voigt, delta)?;
                let lattice = (Matrix3::identity() + eta) * ground.lattice;
                ground.with_geometry(ctx, &lattice, &fractional)
            }
            Perturbation::Displacement { atom, axis, delta } => {
                if atom >= fractional.len() || axis > 2 {
                    return Err(invalid(
                        "a displacement names an atom of the cell and an axis 0..3",
                    ));
                }
                let inverse = ground
                    .lattice
                    .try_inverse()
                    .ok_or_else(|| invalid("singular lattice"))?;
                let mut step = nalgebra::Vector3::zeros();
                step[axis] = delta;
                let shift = inverse * step;
                for j in 0..3 {
                    fractional[atom][j] += shift[j];
                }
                ground.with_positions(ctx, &fractional)
            }
        }
    }
}

/// What a perturbed run keeps.
pub struct Run {
    pub polarization: Polarization,
    pub iterations: usize,
    pub magnetization: f64,
}

/// Prefixes an error with the run it came from, keeping its kind.
fn named(error: Error, name: &str) -> Error {
    match error {
        Error::Invalid(m) => Error::Invalid(format!("{name}: {m}")),
        Error::Unsupported(m) => Error::Unsupported(format!("{name}: {m}")),
        Error::Kernel(m) => Error::Kernel(format!("{name}: {m}")),
        other => other,
    }
}

/// The SCF of a perturbed setup from the ground state and its polarization on `size`.
pub fn perturbed_run(
    ctx: &Context,
    ground: &PeriodicResult,
    setup: &PeriodicSetup,
    size: [usize; 3],
    perturbation: Perturbation,
) -> Result<Run> {
    let name = perturbation.name();
    let crystal = setup.build(ctx).map_err(|e| named(e, &name))?;
    let state = crystal
        .solve_with(&ground.options, Some(ground), None)
        .map_err(|e| named(e, &name))?;
    if !state.converged {
        return Err(Error::Kernel(format!(
            "{name}: the perturbed self-consistent field did not converge in {} iterations",
            state.history.len()
        )));
    }
    if (state.magnetization - ground.magnetization).abs() > MAGNETIZATION_TOLERANCE {
        return Err(invalid(format!(
            "{name}: the magnetization moved from {} to {} Bohr magnetons, another magnetic \
             state; use a smaller step",
            ground.magnetization, state.magnetization
        )));
    }
    let sampler = Sampler::new(&crystal, &state, setup).map_err(|e| named(e, &name))?;
    let polarization = polarization(&sampler, size).map_err(|e| named(e, &name))?;
    Ok(Run {
        polarization,
        iterations: state.history.len(),
        magnetization: state.magnetization,
    })
}

/// `d - round(d)`.
pub fn fold(d: f64) -> f64 {
    d - d.round()
}

/// The folded change of the raw coordinates from `minus` to `plus` (quanta), refused beyond
/// [`QUARTER`].
pub fn folded_step(plus: &Polarization, minus: &Polarization, name: &str) -> Result<[f64; 3]> {
    if plus.quantum != minus.quantum || plus.degeneracy != minus.degeneracy {
        return Err(invalid(format!(
            "{name}: the two runs of a step have different polarization quanta"
        )));
    }
    let step: [f64; 3] = std::array::from_fn(|j| fold(plus.raw[j] - minus.raw[j]));
    if let Some(j) = (0..3).find(|&j| step[j].abs() > QUARTER) {
        return Err(invalid(format!(
            "{name}: the step changes P along a_{j} by {:.3} of a quantum, more than a quarter: \
             use a smaller step",
            step[j]
        )));
    }
    Ok(step)
}

/// The folded change of each channel's electronic coordinate (quanta; a channel's coordinate is
/// continuous modulo `f/g`).
pub fn channel_steps(plus: &Polarization, minus: &Polarization) -> Vec<[f64; 3]> {
    let period = plus.degeneracy / plus.quantum as f64;
    plus.phases
        .iter()
        .zip(&minus.phases)
        .map(|(p, m)| {
            std::array::from_fn(|j| {
                let d = -period * (p[j] - m[j]) / (2. * PI);
                period * fold(d / period)
            })
        })
        .collect()
}

/// `sum_alpha step_alpha quanta_alpha` (e/Bohr^2).
fn cartesian(quanta: &[[f64; 3]; 3], step: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|alpha| step[alpha] * quanta[alpha][i]).sum())
}

/// `Z*_{a,ij} = Omega dP_i / (2 delta)` from the folded steps `steps[a][j]` of the central
/// difference of atom `a` along `j` (quanta of the ground cell, volume `Omega`).
pub fn born_charges(
    quanta: &[[f64; 3]; 3],
    volume: f64,
    steps: &[[[f64; 3]; 3]],
    delta: f64,
) -> Vec<[[f64; 3]; 3]> {
    steps
        .iter()
        .map(|atom| {
            let mut z = [[0.; 3]; 3];
            for (j, step) in atom.iter().enumerate() {
                let dp = cartesian(quanta, step);
                for i in 0..3 {
                    z[i][j] = volume * dp[i] / (2. * delta);
                }
            }
            z
        })
        .collect()
}

/// `e_{i,v} = dP_i / d eta_v` from the folded steps `steps[v]` of the central difference of the
/// Voigt strain `v` (quanta of the UNSTRAINED cell), e/Bohr^2.
pub fn piezoelectric(quanta: &[[f64; 3]; 3], steps: &[[f64; 3]; 6], delta: f64) -> [[f64; 6]; 3] {
    let mut e = [[0.; 6]; 3];
    for (v, step) in steps.iter().enumerate() {
        let dp = cartesian(quanta, step);
        for i in 0..3 {
            e[i][v] = dp[i] / (2. * delta);
        }
    }
    e
}

/// The Richardson combination `(4 fine - coarse) / 3` of central differences at `h` and `h / 2`.
pub fn richardson(coarse: f64, fine: f64) -> f64 {
    (4. * fine - coarse) / 3.
}

/// The acoustic sum rule of a set of Born charges.
#[derive(Clone, Debug, PartialEq)]
pub struct SumRule {
    /// `max_ij |sum_a Z*_{a,ij}|` (e).
    pub residual: f64,
    /// `(1 / N) sum_a Z*_a`, what imposing the rule removes from every atom.
    pub mean: [[f64; 3]; 3],
    /// Set when `residual` exceeds [`ASR_WARNING`].
    pub warning: Option<String>,
}

pub fn acoustic_sum_rule(charges: &[[[f64; 3]; 3]]) -> SumRule {
    let n = charges.len().max(1) as f64;
    let mut sum = [[0.; 3]; 3];
    for z in charges {
        for i in 0..3 {
            for j in 0..3 {
                sum[i][j] += z[i][j];
            }
        }
    }
    let residual = sum.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
    let warning = (residual > ASR_WARNING).then(|| {
        format!(
            "the Born charges break the acoustic sum rule by {residual:.3} e (above \
             {ASR_WARNING} e): refine grid_spacing and kpts, or impose the rule"
        )
    });
    SumRule {
        residual,
        mean: sum.map(|row| row.map(|v| v / n)),
        warning,
    }
}

/// The charges with the mean of [`acoustic_sum_rule`] removed from every atom.
pub fn impose(charges: &[[[f64; 3]; 3]], rule: &SumRule) -> Vec<[[f64; 3]; 3]> {
    charges
        .iter()
        .map(|z| std::array::from_fn(|i| std::array::from_fn(|j| z[i][j] - rule.mean[i][j])))
        .collect()
}

/// Checks a step size and that the SCF tolerance resolves it (`length` the step as a length,
/// Bohr).
fn check_step(
    delta: f64,
    largest: f64,
    length: f64,
    ground: &PeriodicResult,
    what: &str,
) -> Result<()> {
    if !(delta.is_finite() && delta > 0. && delta <= largest) {
        return Err(invalid(format!(
            "a {what} step must be positive and at most {largest}; got {delta}"
        )));
    }
    let tolerance = ground.options.density_tolerance;
    if length < TOLERANCE_RATIO * tolerance {
        return Err(invalid(format!(
            "the SCF density tolerance {tolerance:e} is too loose for a {what} step of {delta}: \
             tighten the density tolerance below {:e}",
            length / TOLERANCE_RATIO
        )));
    }
    Ok(())
}

/// The central difference of one perturbation: the folded total step and the channel steps,
/// with the SCF iterations of both runs.
struct Central {
    step: [f64; 3],
    channels: Vec<[f64; 3]>,
    iterations: [usize; 2],
}

fn central(
    ctx: &Context,
    ground: &PeriodicResult,
    perturber: &dyn Perturber,
    size: [usize; 3],
    plus: Perturbation,
    minus: Perturbation,
) -> Result<Central> {
    let up = perturbed_run(ctx, ground, &perturber.setup(ctx, plus)?, size, plus)?;
    let down = perturbed_run(ctx, ground, &perturber.setup(ctx, minus)?, size, minus)?;
    Ok(Central {
        step: folded_step(&up.polarization, &down.polarization, &plus.name())?,
        channels: channel_steps(&up.polarization, &down.polarization),
        iterations: [up.iterations, down.iterations],
    })
}

/// The ground polarization: its quanta and volume define every difference.
fn ground_polarization(sampler: &Sampler, size: [usize; 3]) -> Result<Polarization> {
    polarization(sampler, size)
}

/// Born effective charges of every atom.
pub struct BornCharges {
    /// `(N, 3, 3)`, `[a][i][j] = Z*_{a,ij}` (e), as computed.
    pub charges: Vec<[[f64; 3]; 3]>,
    /// The per-channel electronic parts (spin-resolved), `[channel][a][i][j]`.
    pub channels: Vec<Vec<[[f64; 3]; 3]>>,
    pub sum_rule: SumRule,
    /// The charges with the sum rule imposed, when asked for.
    pub corrected: Option<Vec<[[f64; 3]; 3]>>,
    /// The folded steps `[a][j]` (quanta) of the central differences at `delta`.
    pub steps: Vec<[[f64; 3]; 3]>,
    pub delta: f64,
    /// SCF iterations of every run, in the order they ran.
    pub iterations: Vec<usize>,
}

/// `Z*` of every atom of the converged crystal of `sampler`, by central differences of `delta`
/// Bohr (and `delta / 2`, combined, with `richardson`).
pub fn born_effective_charges(
    ctx: &Context,
    sampler: &Sampler,
    perturber: &dyn Perturber,
    size: [usize; 3],
    delta: f64,
    richardson_step: bool,
    impose_rule: bool,
) -> Result<BornCharges> {
    check_step(
        delta,
        MAX_DISPLACEMENT,
        delta,
        sampler.state,
        "displacement",
    )?;
    let ground = ground_polarization(sampler, size)?;
    let atoms = sampler.setup.centers.len();
    let mut iterations = Vec::new();
    let mut charges_at = |h: f64| -> Result<(
        Vec<[[f64; 3]; 3]>,
        Vec<Vec<[[f64; 3]; 3]>>,
        Vec<[[f64; 3]; 3]>,
    )> {
        let mut steps = vec![[[0.; 3]; 3]; atoms];
        let mut channel_steps_all: Vec<Vec<[[f64; 3]; 3]>> = Vec::new();
        for atom in 0..atoms {
            for axis in 0..3 {
                let at = |delta| Perturbation::Displacement { atom, axis, delta };
                let c = central(ctx, sampler.state, perturber, size, at(h), at(-h))?;
                steps[atom][axis] = c.step;
                if channel_steps_all.is_empty() {
                    channel_steps_all = vec![vec![[[0.; 3]; 3]; atoms]; c.channels.len()];
                }
                for (s, step) in c.channels.iter().enumerate() {
                    channel_steps_all[s][atom][axis] = *step;
                }
                iterations.extend(c.iterations);
            }
        }
        let charges = born_charges(&ground.quanta, ground.volume, &steps, h);
        let channels = channel_steps_all
            .iter()
            .map(|s| born_charges(&ground.quanta, ground.volume, s, h))
            .collect();
        Ok((charges, channels, steps))
    };
    let (coarse, coarse_channels, steps) = charges_at(delta)?;
    let (charges, channels) = if richardson_step {
        let (fine, fine_channels, _) = charges_at(delta / 2.)?;
        let combine = |c: &[[[f64; 3]; 3]], f: &[[[f64; 3]; 3]]| -> Vec<[[f64; 3]; 3]> {
            c.iter()
                .zip(f)
                .map(|(c, f)| {
                    std::array::from_fn(|i| std::array::from_fn(|j| richardson(c[i][j], f[i][j])))
                })
                .collect()
        };
        let channels = coarse_channels
            .iter()
            .zip(&fine_channels)
            .map(|(c, f)| combine(c, f))
            .collect();
        (combine(&coarse, &fine), channels)
    } else {
        (coarse, coarse_channels)
    };
    let sum_rule = acoustic_sum_rule(&charges);
    let corrected = impose_rule.then(|| impose(&charges, &sum_rule));
    Ok(BornCharges {
        charges,
        channels,
        sum_rule,
        corrected,
        steps,
        delta,
        iterations,
    })
}

/// The clamped-ion proper piezoelectric tensor.
pub struct Piezoelectric {
    /// `[i][v]`, e/Bohr^2 (Voigt, engineering shear).
    pub tensor: [[f64; 6]; 3],
    /// The folded steps `[v]` (quanta) of the central differences at `delta`.
    pub steps: [[f64; 3]; 6],
    pub delta: f64,
    pub iterations: Vec<usize>,
}

/// The clamped-ion tensor of the converged crystal of `sampler`, by central differences of the
/// strain `delta` (and `delta / 2`, combined, with `richardson`).
pub fn clamped_piezoelectric(
    ctx: &Context,
    sampler: &Sampler,
    perturber: &dyn Perturber,
    size: [usize; 3],
    delta: f64,
    richardson_step: bool,
) -> Result<Piezoelectric> {
    let shortest = (0..3)
        .map(|j| sampler.lattice.column(j).norm())
        .fold(f64::INFINITY, f64::min);
    check_step(delta, MAX_STRAIN, delta * shortest, sampler.state, "strain")?;
    let ground = ground_polarization(sampler, size)?;
    let mut iterations = Vec::new();
    let mut tensor_at = |h: f64| -> Result<([[f64; 6]; 3], [[f64; 3]; 6])> {
        let mut steps = [[0.; 3]; 6];
        for (voigt, step) in steps.iter_mut().enumerate() {
            let at = |delta| Perturbation::Strain { voigt, delta };
            let c = central(ctx, sampler.state, perturber, size, at(h), at(-h))?;
            *step = c.step;
            iterations.extend(c.iterations);
        }
        Ok((piezoelectric(&ground.quanta, &steps, h), steps))
    };
    let (coarse, steps) = tensor_at(delta)?;
    let tensor = if richardson_step {
        let (fine, _) = tensor_at(delta / 2.)?;
        std::array::from_fn(|i| std::array::from_fn(|v| richardson(coarse[i][v], fine[i][v])))
    } else {
        coarse
    };
    Ok(Piezoelectric {
        tensor,
        steps,
        delta,
        iterations,
    })
}
