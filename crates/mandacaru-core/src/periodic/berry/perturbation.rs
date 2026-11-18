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
    math::{linalg::real_eigen, storage::Precision},
    models::ts::tkatchenko_scheffler,
    periodic::{
        bloch_mesh::Sampler, invalid, point_group::PointGroup, response::Response,
        setup::PeriodicSetup, PeriodicResult,
    },
    Context, Error, Geometry, LengthUnit, Result,
};
use nalgebra::{DMatrix, Matrix3, Vector3};
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

/// Non-acoustic force-constant eigenvalues below this (Ha/Bohr^2) are refused for relaxed-ion
/// piezoelectricity (soft or unstable modes diverge `Gamma = K^+ Lambda`).
pub const SOFT_MODE_THRESHOLD: f64 = 1e-4;

/// What a perturbed run keeps.
pub struct Run {
    pub polarization: Polarization,
    pub iterations: usize,
    pub magnetization: f64,
    pub forces: Option<Vec<[f64; 3]>>,
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

fn total_forces(
    crystal: &crate::periodic::Crystal,
    state: &PeriodicResult,
    setup: &PeriodicSetup,
    dispersion_xc: Option<&str>,
) -> Result<Vec<[f64; 3]>> {
    let mut forces = Response::new(crystal)?.forces(crystal, state)?;
    if let Some(xc) = dispersion_xc {
        let numbers: Vec<u8> = setup
            .atoms_species
            .iter()
            .map(|&s| {
                u8::try_from(setup.datasets[s].atomic_number)
                    .map_err(|_| invalid("atomic number out of u8 range"))
            })
            .collect::<Result<_>>()?;
        let cell_rows: [[f64; 3]; 3] =
            std::array::from_fn(|a| std::array::from_fn(|x| setup.lattice[(x, a)]));
        let geom = Geometry::new(
            numbers,
            setup.centers.clone(),
            cell_rows,
            [true; 3],
            LengthUnit::Bohr,
        )?;
        let ts = tkatchenko_scheffler(&geom, xc, None, None, None, None)?;
        for (f, ts_f) in forces.iter_mut().zip(&ts.forces_ha_bohr) {
            for axis in 0..3 {
                f[axis] += ts_f[axis];
            }
        }
    }
    Ok(forces)
}

/// The SCF of a perturbed setup from the ground state and its polarization on `size` (plus analytic
/// forces when `want_forces` is set).
pub fn perturbed_run(
    ctx: &Context,
    ground: &PeriodicResult,
    setup: &PeriodicSetup,
    size: [usize; 3],
    perturbation: Perturbation,
) -> Result<Run> {
    perturbed_run_with(ctx, ground, setup, size, perturbation, false, None)
}

fn perturbed_run_with(
    ctx: &Context,
    ground: &PeriodicResult,
    setup: &PeriodicSetup,
    size: [usize; 3],
    perturbation: Perturbation,
    want_forces: bool,
    dispersion_xc: Option<&str>,
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
    let forces = if want_forces {
        Some(total_forces(&crystal, &state, setup, dispersion_xc).map_err(|e| named(e, &name))?)
    } else {
        None
    };
    Ok(Run {
        polarization,
        iterations: state.history.len(),
        magnetization: state.magnetization,
        forces,
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

/// The central difference of one perturbation: the folded total step, channel steps, SCF
/// iterations of both runs, and optional central force difference `-(F(+) - F(-))` or `(F(+) - F(-))`.
struct Central {
    step: [f64; 3],
    channels: Vec<[f64; 3]>,
    iterations: [usize; 2],
    /// `(F(+h) - F(-h))` in Hartree/Bohr (per atom), when `want_forces` was set.
    force_diff: Option<Vec<[f64; 3]>>,
}

fn central(
    ctx: &Context,
    ground: &PeriodicResult,
    perturber: &dyn Perturber,
    size: [usize; 3],
    plus: Perturbation,
    minus: Perturbation,
    want_forces: bool,
    dispersion_xc: Option<&str>,
) -> Result<Central> {
    let up = perturbed_run_with(
        ctx,
        ground,
        &perturber.setup(ctx, plus)?,
        size,
        plus,
        want_forces,
        dispersion_xc,
    )?;
    let down = perturbed_run_with(
        ctx,
        ground,
        &perturber.setup(ctx, minus)?,
        size,
        minus,
        want_forces,
        dispersion_xc,
    )?;
    let force_diff = match (up.forces, down.forces) {
        (Some(fp), Some(fm)) => Some(
            fp.iter()
                .zip(&fm)
                .map(|(a, b)| std::array::from_fn(|l| a[l] - b[l]))
                .collect(),
        ),
        _ => None,
    };
    Ok(Central {
        step: folded_step(&up.polarization, &down.polarization, &plus.name())?,
        channels: channel_steps(&up.polarization, &down.polarization),
        iterations: [up.iterations, down.iterations],
        force_diff,
    })
}

fn require_float64(sampler: &Sampler, what: &str) -> Result<()> {
    if sampler.crystal.precision == Precision::Float32 {
        return Err(invalid(format!(
            "{what} require precision='float64': finite differences of P are below float32 \
             precision"
        )));
    }
    Ok(())
}

/// The ground polarization: its quanta and volume define every difference.
fn ground_polarization(sampler: &Sampler, size: [usize; 3]) -> Result<Polarization> {
    polarization(sampler, size)
}

/// Symmetry orbit decomposition for displacement reduction over a [`PointGroup`].
struct OrbitPlan {
    rotations: Vec<Matrix3<f64>>,
    atom_maps: Vec<Vec<usize>>,
    inv_maps: Vec<Vec<usize>>,
    orbits: Vec<AtomOrbit>,
}

struct AtomOrbit {
    rep: usize,
    axes: Vec<usize>,
    /// Three generating `(op_index, computed_axis)` pairs whose rotated basis vectors form `basis`.
    generators: [(usize, usize); 3],
    /// Inverse of the 3x3 matrix `[R_{op_m} e_{k_m}]` (columns).
    basis_inv: Matrix3<f64>,
    stabilizer: Vec<usize>,
    members: Vec<usize>,
}

impl OrbitPlan {
    fn new(atoms: usize, group: Option<&PointGroup>) -> Result<Self> {
        let (rotations, atom_maps) = match group {
            Some(pg) if !pg.rotations().is_empty() => {
                (pg.rotations().to_vec(), pg.atom_maps().to_vec())
            }
            _ => (vec![Matrix3::identity()], vec![(0..atoms).collect()]),
        };
        let mut inv_maps = Vec::with_capacity(atom_maps.len());
        for map in &atom_maps {
            if map.len() != atoms {
                return Err(invalid("symmetry atom map does not match atom count"));
            }
            let mut inv = vec![0usize; atoms];
            for (a, &b) in map.iter().enumerate() {
                inv[b] = a;
            }
            inv_maps.push(inv);
        }
        let mut assigned = vec![false; atoms];
        let mut orbits = Vec::new();
        for start in 0..atoms {
            if assigned[start] {
                continue;
            }
            let mut members = Vec::new();
            let mut in_orbit = vec![false; atoms];
            for map in &atom_maps {
                let b = map[start];
                if !in_orbit[b] {
                    in_orbit[b] = true;
                    members.push(b);
                }
            }
            members.sort_unstable();
            for &m in &members {
                assigned[m] = true;
            }
            let rep = members[0];
            let stabilizer: Vec<usize> = atom_maps
                .iter()
                .enumerate()
                .filter_map(|(op, map)| (map[rep] == rep).then_some(op))
                .collect();
            let mut q_basis: Vec<Vector3<f64>> = Vec::with_capacity(3);
            let mut cols: Vec<Vector3<f64>> = Vec::with_capacity(3);
            let mut generators_vec: Vec<(usize, usize)> = Vec::with_capacity(3);
            let mut axes = Vec::new();
            for k in 0..3 {
                if q_basis.len() == 3 {
                    break;
                }
                let e = Vector3::ith(k, 1.);
                let mut proj = e;
                for q in &q_basis {
                    proj -= *q * q.dot(&e);
                }
                if proj.norm() <= 1e-8 {
                    continue;
                }
                axes.push(k);
                for &op in &stabilizer {
                    let v = rotations[op] * e;
                    let mut rem = v;
                    for q in &q_basis {
                        rem -= *q * q.dot(&v);
                    }
                    if rem.norm() > 1e-8 {
                        q_basis.push(rem / rem.norm());
                        cols.push(v);
                        generators_vec.push((op, k));
                        if q_basis.len() == 3 {
                            break;
                        }
                    }
                }
            }
            if cols.len() != 3 {
                return Err(invalid("symmetry stabilizer basis failed to span 3D space"));
            }
            let basis = Matrix3::from_columns(&[cols[0], cols[1], cols[2]]);
            let basis_inv = basis
                .try_inverse()
                .ok_or_else(|| invalid("singular symmetry basis"))?;
            orbits.push(AtomOrbit {
                rep,
                axes,
                generators: [generators_vec[0], generators_vec[1], generators_vec[2]],
                basis_inv,
                stabilizer,
                members,
            });
        }
        Ok(Self {
            rotations,
            atom_maps,
            inv_maps,
            orbits,
        })
    }

    /// Expand computed `[orbit][computed_axis_idx]` 3x3-column vectors (`Z*` columns) to `(N, 3, 3)`
    /// by stabilizer span and space-group averaging.
    fn expand_charges(&self, rep_cols: &[Vec<[f64; 3]>]) -> Vec<[[f64; 3]; 3]> {
        let atoms = self.atom_maps[0].len();
        let mut out = vec![[[0.; 3]; 3]; atoms];
        for (orbit, computed) in self.orbits.iter().zip(rep_cols) {
            let axis_pos = |k: usize| orbit.axes.iter().position(|&a| a == k).unwrap();
            let mut z_rep = Matrix3::zeros();
            for j in 0..3 {
                let coeff = orbit.basis_inv * Vector3::ith(j, 1.);
                let mut col = Vector3::zeros();
                for (m, &(op, k)) in orbit.generators.iter().enumerate() {
                    let raw = Vector3::from(computed[axis_pos(k)]);
                    col += (self.rotations[op] * raw) * coeff[m];
                }
                z_rep.set_column(j, &col);
            }
            let mut z_stab = Matrix3::zeros();
            for &op in &orbit.stabilizer {
                let r = &self.rotations[op];
                z_stab += r * z_rep * r.transpose();
            }
            z_stab /= orbit.stabilizer.len() as f64;
            for &b in &orbit.members {
                let mut acc = Matrix3::zeros();
                let mut count = 0usize;
                for (op, map) in self.atom_maps.iter().enumerate() {
                    if map[orbit.rep] == b {
                        let r = &self.rotations[op];
                        acc += r * z_stab * r.transpose();
                        count += 1;
                    }
                }
                acc /= count.max(1) as f64;
                out[b] = std::array::from_fn(|i| std::array::from_fn(|j| acc[(i, j)]));
            }
        }
        out
    }

    /// Expand computed force-constant blocks `[orbit][computed_axis_idx][atom_c]` (`Ha/Bohr^2`)
    /// to `(3N, 3N)` (`K_{(c,i), (b,j)} = -dF_{c,i} / du_{b,j}`).
    fn expand_force_constants(&self, rep_cols: &[Vec<Vec<[f64; 3]>>]) -> Vec<Vec<f64>> {
        let atoms = self.atom_maps[0].len();
        let dim = 3 * atoms;
        let mut k_mat = vec![vec![0.; dim]; dim];
        for (orbit, computed) in self.orbits.iter().zip(rep_cols) {
            let axis_pos = |k: usize| orbit.axes.iter().position(|&a| a == k).unwrap();
            let mut k_rep = vec![Matrix3::<f64>::zeros(); atoms];
            for j in 0..3 {
                let coeff = orbit.basis_inv * Vector3::ith(j, 1.);
                for c in 0..atoms {
                    let mut col = Vector3::zeros();
                    for (m, &(op, k)) in orbit.generators.iter().enumerate() {
                        let c_pre = self.inv_maps[op][c];
                        let raw = Vector3::from(computed[axis_pos(k)][c_pre]);
                        col += (self.rotations[op] * raw) * coeff[m];
                    }
                    k_rep[c].set_column(j, &col);
                }
            }
            let mut k_stab = vec![Matrix3::<f64>::zeros(); atoms];
            for &op in &orbit.stabilizer {
                let r = &self.rotations[op];
                for c in 0..atoms {
                    let c_pre = self.inv_maps[op][c];
                    k_stab[c] += r * k_rep[c_pre] * r.transpose();
                }
            }
            for mat in &mut k_stab {
                *mat /= orbit.stabilizer.len() as f64;
            }
            for &b in &orbit.members {
                let mut acc = vec![Matrix3::<f64>::zeros(); atoms];
                let mut count = 0usize;
                for (op, map) in self.atom_maps.iter().enumerate() {
                    if map[orbit.rep] == b {
                        let r = &self.rotations[op];
                        for c in 0..atoms {
                            let c_pre = self.inv_maps[op][c];
                            acc[c] += r * k_stab[c_pre] * r.transpose();
                        }
                        count += 1;
                    }
                }
                let scale = 1. / count.max(1) as f64;
                for c in 0..atoms {
                    let block = acc[c] * scale;
                    for i in 0..3 {
                        for j in 0..3 {
                            k_mat[3 * c + i][3 * b + j] = block[(i, j)];
                        }
                    }
                }
            }
        }
        k_mat
    }
}

/// Convert Cartesian Born charges `Z*_{a,ij}` back into folded polarization quanta steps
/// `[a][axis][alpha]` of `ground` (consistent with symmetry-expanded `charges`).
fn steps_from_charges(
    quanta: &[[f64; 3]; 3],
    volume: f64,
    charges: &[[[f64; 3]; 3]],
    delta: f64,
) -> Result<Vec<[[f64; 3]; 3]>> {
    let q_cols = Matrix3::from_fn(|i, alpha| quanta[alpha][i]);
    let q_inv = q_cols
        .try_inverse()
        .ok_or_else(|| invalid("singular polarization quanta matrix"))?;
    let scale = (2. * delta) / volume;
    Ok(charges
        .iter()
        .map(|z| {
            std::array::from_fn(|j| {
                let dp = Vector3::from_fn(|i, _| z[i][j] * scale);
                let step = q_inv * dp;
                [step[0], step[1], step[2]]
            })
        })
        .collect())
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
/// Bohr (and `delta / 2`, combined, with `richardson`). Uses the ground crystal's space group to
/// displace only symmetry-inequivalent `(atom, axis)` pairs when available.
pub fn born_effective_charges(
    ctx: &Context,
    sampler: &Sampler,
    perturber: &dyn Perturber,
    size: [usize; 3],
    delta: f64,
    richardson_step: bool,
    impose_rule: bool,
) -> Result<BornCharges> {
    born_effective_charges_with(
        ctx,
        sampler,
        perturber,
        size,
        delta,
        richardson_step,
        impose_rule,
        true,
    )
}

/// Like [`born_effective_charges`], with explicit control over `symmetry_reduce`.
#[allow(clippy::too_many_arguments)]
pub fn born_effective_charges_with(
    ctx: &Context,
    sampler: &Sampler,
    perturber: &dyn Perturber,
    size: [usize; 3],
    delta: f64,
    richardson_step: bool,
    impose_rule: bool,
    symmetry_reduce: bool,
) -> Result<BornCharges> {
    let (charges, _) = born_charges_and_force_constants(
        ctx,
        sampler,
        perturber,
        size,
        delta,
        richardson_step,
        impose_rule,
        symmetry_reduce,
        false,
        None,
    )?;
    Ok(charges)
}

#[allow(clippy::too_many_arguments)]
fn born_charges_and_force_constants(
    ctx: &Context,
    sampler: &Sampler,
    perturber: &dyn Perturber,
    size: [usize; 3],
    delta: f64,
    richardson_step: bool,
    impose_rule: bool,
    symmetry_reduce: bool,
    want_forces: bool,
    dispersion_xc: Option<&str>,
) -> Result<(BornCharges, Option<Vec<Vec<f64>>>)> {
    require_float64(sampler, "Born effective charges")?;
    check_step(
        delta,
        MAX_DISPLACEMENT,
        delta,
        sampler.state,
        "displacement",
    )?;
    let ground = ground_polarization(sampler, size)?;
    let atoms = sampler.setup.centers.len();
    let group = if symmetry_reduce {
        sampler
            .setup
            .symmetry
            .as_ref()
            .map(|s| s.to_point_group(atoms))
            .transpose()?
    } else {
        None
    };
    let plan = OrbitPlan::new(atoms, group.as_ref())?;
    let mut iterations = Vec::new();
    let mut charges_at = |h: f64| -> Result<(
        Vec<[[f64; 3]; 3]>,
        Vec<Vec<[[f64; 3]; 3]>>,
        Vec<[[f64; 3]; 3]>,
        Option<Vec<Vec<f64>>>,
    )> {
        let mut rep_cols: Vec<Vec<[f64; 3]>> = Vec::with_capacity(plan.orbits.len());
        let mut rep_channels: Vec<Vec<Vec<[f64; 3]>>> = Vec::new();
        let mut rep_forces: Vec<Vec<Vec<[f64; 3]>>> = Vec::with_capacity(plan.orbits.len());
        let scale = ground.volume / (2. * h);
        for orbit in &plan.orbits {
            let mut axis_cols = Vec::with_capacity(orbit.axes.len());
            let mut axis_chan_cols: Vec<Vec<[f64; 3]>> = Vec::new();
            let mut axis_force_cols = Vec::with_capacity(orbit.axes.len());
            for &axis in &orbit.axes {
                let at = |d| Perturbation::Displacement {
                    atom: orbit.rep,
                    axis,
                    delta: d,
                };
                let c = central(
                    ctx,
                    sampler.state,
                    perturber,
                    size,
                    at(h),
                    at(-h),
                    want_forces,
                    dispersion_xc,
                )?;
                let dp = cartesian(&ground.quanta, &c.step);
                axis_cols.push([dp[0] * scale, dp[1] * scale, dp[2] * scale]);
                if rep_channels.is_empty() {
                    rep_channels = vec![Vec::with_capacity(plan.orbits.len()); c.channels.len()];
                }
                if axis_chan_cols.is_empty() {
                    axis_chan_cols = vec![Vec::with_capacity(orbit.axes.len()); c.channels.len()];
                }
                for (s, step) in c.channels.iter().enumerate() {
                    let c_dp = cartesian(&ground.quanta, step);
                    axis_chan_cols[s].push([c_dp[0] * scale, c_dp[1] * scale, c_dp[2] * scale]);
                }
                if let Some(diff) = c.force_diff {
                    axis_force_cols.push(
                        diff.iter()
                            .map(|f| std::array::from_fn(|l| -f[l] / (2. * h)))
                            .collect(),
                    );
                }
                iterations.extend(c.iterations);
            }
            rep_cols.push(axis_cols);
            for (s, cols) in axis_chan_cols.into_iter().enumerate() {
                rep_channels[s].push(cols);
            }
            if want_forces {
                rep_forces.push(axis_force_cols);
            }
        }
        let charges = plan.expand_charges(&rep_cols);
        let channels = rep_channels
            .iter()
            .map(|r| plan.expand_charges(r))
            .collect();
        let steps = steps_from_charges(&ground.quanta, ground.volume, &charges, h)?;
        let force_constants = want_forces.then(|| plan.expand_force_constants(&rep_forces));
        Ok((charges, channels, steps, force_constants))
    };
    let (coarse, coarse_channels, steps, coarse_k) = charges_at(delta)?;
    let (charges, channels, k_mat) = if richardson_step {
        let (fine, fine_channels, _, fine_k) = charges_at(delta / 2.)?;
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
        let k = match (coarse_k, fine_k) {
            (Some(ck), Some(fk)) => Some(
                ck.iter()
                    .zip(&fk)
                    .map(|(cr, fr)| {
                        cr.iter()
                            .zip(fr)
                            .map(|(&cv, &fv)| richardson(cv, fv))
                            .collect()
                    })
                    .collect(),
            ),
            _ => None,
        };
        (combine(&coarse, &fine), channels, k)
    } else {
        (coarse, coarse_channels, coarse_k)
    };
    let sum_rule = acoustic_sum_rule(&charges);
    let corrected = impose_rule.then(|| impose(&charges, &sum_rule));
    Ok((
        BornCharges {
            charges,
            channels,
            sum_rule,
            corrected,
            steps,
            delta,
            iterations,
        },
        k_mat,
    ))
}

/// Output of [`relaxed_piezoelectric_from_responses`].
#[derive(Clone, Debug)]
pub struct RelaxedResponse {
    pub relaxed: [[f64; 6]; 3],
    pub internal_strain: Vec<[f64; 6]>,
    pub force_constants: Vec<Vec<f64>>,
    pub acoustic_eigenvalues: [f64; 3],
}

/// Computes the relaxed-ion piezoelectric tensor `e_{i,v} = e^(0)_{i,v} + (1 / Omega) sum_{a,l} Z*_{a,il} Gamma_{al,v}`
/// with internal strain `Gamma = K^+ Lambda`, projecting out the three rigid translations and
/// refusing non-acoustic eigenvalues below [`SOFT_MODE_THRESHOLD`].
pub fn relaxed_piezoelectric_from_responses(
    clamped: &[[f64; 6]; 3],
    volume: f64,
    charges: &[[[f64; 3]; 3]],
    force_constants: &[Vec<f64>],
    lambda: &[[f64; 6]],
) -> Result<RelaxedResponse> {
    let atoms = charges.len();
    let dim = 3 * atoms;
    if atoms == 0 || force_constants.len() != dim || force_constants.iter().any(|r| r.len() != dim)
    {
        return Err(invalid(
            "force-constant matrix must match 3N x 3N atom coordinates",
        ));
    }
    if lambda.len() != dim {
        return Err(invalid(
            "internal-strain force response must match 3N atom coordinates",
        ));
    }
    let mut k_sym = DMatrix::<f64>::from_fn(dim, dim, |r, c| {
        0.5 * (force_constants[r][c] + force_constants[c][r])
    });
    let symmetrized: Vec<Vec<f64>> = (0..dim)
        .map(|r| (0..dim).map(|c| k_sym[(r, c)]).collect())
        .collect();
    let norm = 1. / (atoms as f64).sqrt();
    let mut trans = DMatrix::<f64>::zeros(dim, 3);
    for a in 0..atoms {
        for l in 0..3 {
            trans[(3 * a + l, l)] = norm;
        }
    }
    let k_trans = trans.transpose() * &k_sym * &trans;
    let (ac_vals, _) = real_eigen(k_trans)?;
    let acoustic_eigenvalues = [ac_vals[0], ac_vals[1], ac_vals[2]];

    let gamma_rows = if dim > 3 {
        // Orthonormal basis `U` (dim x (dim - 3)) of the complement of rigid translations.
        let projector = DMatrix::<f64>::identity(dim, dim) - &trans * trans.transpose();
        let (p_vals, p_vecs) = real_eigen(projector)?;
        let mut comp_cols = Vec::with_capacity(dim - 3);
        for (idx, &val) in p_vals.iter().enumerate() {
            if val > 0.5 {
                comp_cols.push(p_vecs.column(idx).into_owned());
            }
        }
        if comp_cols.len() != dim - 3 {
            return Err(invalid(
                "failed to construct translational complement of force constants",
            ));
        }
        let u = DMatrix::from_columns(&comp_cols);
        let k_red = u.transpose() * &k_sym * &u;
        let (eigs, vecs) = real_eigen(k_red)?;
        if let Some(&min_eig) = eigs.iter().find(|&&v| v < SOFT_MODE_THRESHOLD) {
            return Err(invalid(format!(
                "the force-constant matrix has a soft or unstable mode of {min_eig:.3e} Ha/Bohr^2 \
                 (below {SOFT_MODE_THRESHOLD:e}); the relaxed-ion piezoelectric tensor diverges"
            )));
        }
        let mut inv_red = DMatrix::<f64>::zeros(dim - 3, dim - 3);
        for (m, &eig) in eigs.iter().enumerate() {
            let v = vecs.column(m);
            inv_red += (&v * v.transpose()) * (1. / eig);
        }
        let k_pinv = &u * inv_red * u.transpose();
        k_sym = k_pinv;
        let lam_mat = DMatrix::<f64>::from_fn(dim, 6, |r, c| lambda[r][c]);
        let g = &k_sym * lam_mat;
        (0..dim)
            .map(|r| std::array::from_fn(|v| g[(r, v)]))
            .collect()
    } else {
        vec![[0.; 6]; dim]
    };

    let mut relaxed = *clamped;
    for a in 0..atoms {
        for i in 0..3 {
            for l in 0..3 {
                let z = charges[a][i][l];
                for v in 0..6 {
                    relaxed[i][v] += (z * gamma_rows[3 * a + l][v]) / volume;
                }
            }
        }
    }
    Ok(RelaxedResponse {
        relaxed,
        internal_strain: gamma_rows,
        force_constants: symmetrized,
        acoustic_eigenvalues,
    })
}

/// The proper piezoelectric tensor (clamped-ion, and optionally relaxed-ion).
pub struct Piezoelectric {
    /// Active tensor (`relaxed` when computed, else `clamped`), `[i][v]`, e/Bohr^2 (Voigt).
    pub tensor: [[f64; 6]; 3],
    pub clamped: [[f64; 6]; 3],
    pub relaxed: Option<[[f64; 6]; 3]>,
    pub born_charges: Option<BornCharges>,
    /// Internal strain `Gamma = K^+ Lambda` (`3N x 6`), Bohr.
    pub internal_strain: Option<Vec<[f64; 6]>>,
    /// Symmetrized force-constant matrix `K` (`3N x 3N`), Ha/Bohr^2.
    pub force_constants: Option<Vec<Vec<f64>>>,
    /// Eigenvalues of the translational subspace removed from `K^+`, Ha/Bohr^2.
    pub acoustic_eigenvalues: Option<[f64; 3]>,
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
    piezoelectric_tensor(
        ctx,
        sampler,
        perturber,
        size,
        delta,
        MAX_DISPLACEMENT.min(0.02),
        richardson_step,
        false,
        false,
        true,
        None,
    )
}

/// Clamped- or relaxed-ion proper piezoelectric tensor of the converged crystal of `sampler`.
#[allow(clippy::too_many_arguments)]
pub fn piezoelectric_tensor(
    ctx: &Context,
    sampler: &Sampler,
    perturber: &dyn Perturber,
    size: [usize; 3],
    strain: f64,
    displacement: f64,
    richardson_step: bool,
    impose_rule: bool,
    relaxed_ions: bool,
    symmetry_reduce: bool,
    dispersion_xc: Option<&str>,
) -> Result<Piezoelectric> {
    require_float64(sampler, "the piezoelectric tensor")?;
    let shortest = (0..3)
        .map(|j| sampler.lattice.column(j).norm())
        .fold(f64::INFINITY, f64::min);
    check_step(
        strain,
        MAX_STRAIN,
        strain * shortest,
        sampler.state,
        "strain",
    )?;
    let ground = ground_polarization(sampler, size)?;
    let atoms = sampler.setup.centers.len();
    let mut iterations = Vec::new();
    let mut tensor_at = |h: f64| -> Result<([[f64; 6]; 3], [[f64; 3]; 6], Option<Vec<[f64; 6]>>)> {
        let mut steps = [[0.; 3]; 6];
        let mut lambda = relaxed_ions.then(|| vec![[0.; 6]; 3 * atoms]);
        for (voigt, step) in steps.iter_mut().enumerate() {
            let at = |delta| Perturbation::Strain { voigt, delta };
            let c = central(
                ctx,
                sampler.state,
                perturber,
                size,
                at(h),
                at(-h),
                relaxed_ions,
                dispersion_xc,
            )?;
            *step = c.step;
            if let (Some(lam), Some(diff)) = (lambda.as_mut(), c.force_diff) {
                for a in 0..atoms {
                    for l in 0..3 {
                        lam[3 * a + l][voigt] = diff[a][l] / (2. * h);
                    }
                }
            }
            iterations.extend(c.iterations);
        }
        Ok((piezoelectric(&ground.quanta, &steps, h), steps, lambda))
    };
    let (coarse, steps, coarse_lam) = tensor_at(strain)?;
    let (clamped, lambda) = if richardson_step {
        let (fine, _, fine_lam) = tensor_at(strain / 2.)?;
        let combined =
            std::array::from_fn(|i| std::array::from_fn(|v| richardson(coarse[i][v], fine[i][v])));
        let lam = match (coarse_lam, fine_lam) {
            (Some(cl), Some(fl)) => Some(
                cl.iter()
                    .zip(&fl)
                    .map(|(cr, fr)| std::array::from_fn(|v| richardson(cr[v], fr[v])))
                    .collect(),
            ),
            _ => None,
        };
        (combined, lam)
    } else {
        (coarse, coarse_lam)
    };
    if !relaxed_ions {
        return Ok(Piezoelectric {
            tensor: clamped,
            clamped,
            relaxed: None,
            born_charges: None,
            internal_strain: None,
            force_constants: None,
            acoustic_eigenvalues: None,
            steps,
            delta: strain,
            iterations,
        });
    }
    let (z_charges, k_opt) = born_charges_and_force_constants(
        ctx,
        sampler,
        perturber,
        size,
        displacement,
        richardson_step,
        impose_rule,
        symmetry_reduce,
        true,
        dispersion_xc,
    )?;
    iterations.extend(z_charges.iterations.iter().copied());
    let z_for_relax = z_charges.corrected.as_ref().unwrap_or(&z_charges.charges);
    let k_mat =
        k_opt.ok_or_else(|| invalid("missing force constants for relaxed piezoelectric"))?;
    let lam = lambda.ok_or_else(|| invalid("missing internal strain response"))?;
    let rel =
        relaxed_piezoelectric_from_responses(&clamped, ground.volume, z_for_relax, &k_mat, &lam)?;
    Ok(Piezoelectric {
        tensor: rel.relaxed,
        clamped,
        relaxed: Some(rel.relaxed),
        born_charges: Some(z_charges),
        internal_strain: Some(rel.internal_strain),
        force_constants: Some(rel.force_constants),
        acoustic_eigenvalues: Some(rel.acoustic_eigenvalues),
        steps,
        delta: strain,
        iterations,
    })
}
