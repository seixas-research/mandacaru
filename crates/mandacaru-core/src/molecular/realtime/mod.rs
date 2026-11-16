//! Real-time propagation of a molecular mean-field state after a delta kick (`TDDFT_PLAN.md`,
//! sections 13.3 and 13.4).
//!
//! The occupied orbitals of a converged reference are propagated in the orthonormal (Loewdin)
//! frame, `i dC/dt = F[D(t)] C`, by `math::propagator`, with `F` the reference's own Fock builder
//! ([`fock`]). The basis is fixed, so the overlap is constant and no basis-derivative term exists.
//!
//! - **Density kick (length gauge):** `C -> exp(-i kappa X) C` at `t = 0+`, `X = e . r` in the
//!   orthonormal frame (`multipoles::position_integrals`: the grid part and the PAW one-center
//!   dipole blocks), the operator that gives T1 its transition dipoles. For `t > 0` there is no
//!   external term. The induced dipole `delta mu(t) = -tr[D X] + tr[D_0 X]` (electrons, charge
//!   -1) then obeys `delta mu / kappa -> sum_n 2 abs(mu_n . e)^2 sin(omega_n t)` over every Casida
//!   root (V9) as `kappa -> 0`.
//! - **Spin kick:** `exp(-i kappa X)` on the up spin and `exp(+i kappa X)` on the down spin; the
//!   observable is the spin dipole `-tr[(D_up - D_down) X]`. A closed shell is expanded to two
//!   equal channels, and the kick excites its `M_s = 0` triplets.
//! - **Velocity gauge:** no change of the orbitals; a constant vector potential `A = kappa`
//!   (`A_phys / c`) along `e` from `t = 0+` adds `A e.P + A^2 / 2`. The observable is still the
//!   dipole; its poles are the same, its weights the velocity strengths.
//!
//! +U enters through the complex occupations of its shells (`MolecularHubbard::evaluate_complex`),
//! which carry the current through the shell. A meta-GGA (r2SCAN) sees the gauge-invariant
//! kinetic density `tau~ = tau - |j|^2 / (2 n)` of the complex orbitals, with its current-density
//! Fock term (`dft::evaluate_with`, owner decision Q19a), the same functional whose second
//! derivative the amended linear-response kernel is. The run refuses what it cannot do: nonlocal
//! correlation, fractional occupations, a spinor reference and float32, and a step above the
//! scheme's `dt W` bound.
mod fock;

use crate::math::propagator::{evolve, Channel, Options, Propagator, Scheme};
use crate::molecular::{multipoles, MeanField, MolecularIntegrals};
use crate::{quantum::C64, Error, Result};
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

/// The default kick, 0.002 / Angstrom in Bohr^-1 (the value the user interface defaults to),
/// against which [`Request::corrector_tolerance`] is scaled.
pub fn default_strength() -> f64 {
    crate::units::from_user(0.002, crate::units::Quantity::Wavenumber, false)
}

/// The corrector tolerance a run of `request` applies: [`Request::corrector_tolerance`] times
/// `abs(strength) / default_strength()`.
pub fn effective_corrector_tolerance(request: &Request) -> f64 {
    request.corrector_tolerance * (request.strength.abs() / default_strength())
}

/// What the kick perturbs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kick {
    Density,
    Spin,
}
impl Kick {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "density" => Ok(Self::Density),
            "spin" => Ok(Self::Spin),
            _ => Err(invalid(format!(
                "unknown kick {name:?}; expected 'density' or 'spin'"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Density => "density",
            Self::Spin => "spin",
        }
    }
}

/// How the kick enters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gauge {
    Length,
    Velocity,
}
impl Gauge {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "length" => Ok(Self::Length),
            "velocity" => Ok(Self::Velocity),
            _ => Err(invalid(format!(
                "unknown gauge {name:?}; expected 'length' or 'velocity'"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Velocity => "velocity",
        }
    }
}

/// A real-time run: a kick of `strength` (Bohr^-1) along `direction`, then `steps` steps of
/// `time_step` (atomic units).
#[derive(Clone, Debug)]
pub struct Request {
    pub direction: [f64; 3],
    pub strength: f64,
    pub kick: Kick,
    pub gauge: Gauge,
    pub scheme: Scheme,
    pub time_step: f64,
    pub steps: usize,
    /// The ETRS corrector's bound on the change of any density element **at the default kick**
    /// ([`default_strength`]); the run applies it scaled by `abs(strength) / default_strength()`,
    /// because the induced density scales with the kick (owner decision Q18c, 2026-11-12). 1e-6
    /// was measured on water to keep the trace within 5e-5 of the 1e-10 run at half the builds;
    /// the absolute 1e-6 at a kick of 1e-4 / Bohr cost 1.1e-3.
    pub corrector_tolerance: f64,
    pub max_corrector_iterations: usize,
    /// The largest energy deviation from its value after the kick, in units of the kick energy;
    /// `None` checks nothing.
    pub max_energy_drift: Option<f64>,
}
impl Request {
    /// The defaults of owner decision Q18b (2026-11-12): ETRS, corrector tolerance 1e-6, at most
    /// 8 corrector iterations, an energy drift of at most 0.05 kick energies. Every run is also
    /// held to the scheme's `dt W` bound ([`Scheme::step_width_bound`]).
    pub fn new(direction: [f64; 3], strength: f64, time_step: f64, steps: usize) -> Self {
        Self {
            direction,
            strength,
            kick: Kick::Density,
            gauge: Gauge::Length,
            scheme: Scheme::Etrs,
            time_step,
            steps,
            corrector_tolerance: 1e-6,
            max_corrector_iterations: 8,
            max_energy_drift: Some(0.05),
        }
    }
    /// [`Request::new`] from user units (`atomic`: already atomic): `strength` in 1/Angstrom,
    /// `time_step` and `duration` in femtoseconds; the step count is `duration / time_step`
    /// rounded, and a duration shorter than one step is refused.
    pub fn from_user_units(
        direction: [f64; 3],
        strength: f64,
        time_step: f64,
        duration: f64,
        atomic: bool,
    ) -> Result<Self> {
        use crate::units::{from_user, Quantity};
        let strength = from_user(strength, Quantity::Wavenumber, atomic);
        let time_step = from_user(time_step, Quantity::Time, atomic);
        let duration = from_user(duration, Quantity::Time, atomic);
        if !(time_step.is_finite() && time_step > 0. && duration.is_finite()) {
            return Err(invalid(
                "time_step must be positive and finite, and duration finite",
            ));
        }
        let steps = (duration / time_step).round();
        if !(steps >= 1.) || steps > 1e9 {
            return Err(invalid(format!(
                "duration must hold at least one time_step (and at most 1e9 steps); it holds {steps}"
            )));
        }
        Ok(Self::new(direction, strength, time_step, steps as usize))
    }
}

/// The trajectory of a run and its record.
#[derive(Clone, Debug)]
pub struct RealTime {
    /// `n dt`, `n = 0 .. steps`.
    pub times: Vec<f64>,
    /// The induced dipole (the spin dipole for a spin kick), electrons only, atomic units; zero
    /// at `t = 0`.
    pub dipole: Vec<[f64; 3]>,
    /// The electrons' kinetic momentum `sum_s g tr[D_s (P + A)]` (`g` 2 for one restricted
    /// channel; the spin momentum for a spin kick), atomic units; zero in the ground state.
    pub momentum: Vec<[f64; 3]>,
    /// The induced dipole along the kick direction (the signal of the spectra).
    pub along: Vec<f64>,
    /// The energy at each time (the field terms included in the velocity gauge).
    pub energy: Vec<f64>,
    pub norm_error: Vec<f64>,
    /// The energy of the unkicked reference from the same builder, and the kick's energy above it.
    pub ground_energy: f64,
    pub kick_energy: f64,
    /// `max abs(E(t) - E(0+))` in units of the kick energy.
    pub energy_drift: f64,
    /// `W = e_max - e_min` of the reference Fock matrix (the largest over the channels), and the
    /// largest width exponentiated during the run.
    pub width: f64,
    pub largest_width: f64,
    /// `dt W`, checked against the scheme's bound.
    pub step_width: f64,
    /// The largest `abs(C^dagger C - 1)` of the run.
    pub worst_norm_error: f64,
    /// The corrector tolerance applied, scaled with the kick ([`effective_corrector_tolerance`]).
    pub corrector_tolerance: f64,
    /// The bare Kohn-Sham oscillator weight along the kick above the Nyquist frequency
    /// `pi / dt`, as a fraction of the bare total (a proxy: interacting roots can lie above).
    pub aliased_weight: f64,
    /// Fock builds, the initial ones included, and the largest corrector iteration count.
    pub builds: usize,
    pub corrector_iterations: usize,
}

/// What a reference needs to be for real-time propagation, or the reason it is refused.
fn check_reference(integrals: &MolecularIntegrals, reference: &MeanField) -> Result<()> {
    let unsupported = |what: String| Err(Error::Unsupported(what));
    if reference.method == "ghf" {
        if reference.functional.is_some() {
            return unsupported(
                "real-time propagation of spinor Kohn-Sham (a spin-orbit dataset) is not \
                 implemented yet; a scalar-relativistic dataset runs restricted Kohn-Sham"
                    .into(),
            );
        }
        return Err(invalid(
            "real-time propagation needs an rhf, uhf or dft reference; method='ghf' gives a \
             two-component state it does not take",
        ));
    }
    if integrals.grid.periodic || integrals.supercell.is_some() {
        return Err(invalid(
            "molecular real-time propagation needs an isolated molecule, not a periodic supercell",
        ));
    }
    if integrals.precision() == crate::math::storage::Precision::Float32 {
        return Err(invalid(
            "real-time propagation runs in float64 only; precision='float32' is not supported",
        ));
    }
    if !reference.converged {
        return Err(invalid(
            "real-time propagation needs a converged reference; the SCF did not converge",
        ));
    }
    reference.require_determinant("real-time propagation")?;
    if reference.aufbau_inversion.is_some() {
        return unsupported(
            "real-time propagation needs an aufbau reference; this state holds an occupied level \
             above an empty one"
                .into(),
        );
    }
    if let Some(name) = reference.functional.as_deref() {
        let definition = crate::models::functional(name)?;
        if definition.nonlocal_b.is_some() {
            return unsupported(format!(
                "real-time propagation with xc='{name}' (nonlocal rVV10 correlation) is not \
                 implemented"
            ));
        }
    }
    Ok(())
}

fn unit(direction: [f64; 3]) -> Result<[f64; 3]> {
    let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !norm.is_finite() || norm == 0. {
        return Err(invalid(
            "the kick direction must be a finite nonzero vector",
        ));
    }
    Ok(direction.map(|v| v / norm))
}

/// Propagates `reference` after the kick of `request`.
/// Refuses the crystal options a molecule does not take: the propagation mesh `kpts`, and a
/// `kernel` other than the reference's own adiabatic one.
pub fn check_molecular_options(kernel: &str, kpts: Option<[usize; 3]>) -> Result<()> {
    if kpts.is_some() {
        return Err(invalid(
            "kpts is a crystal's propagation mesh; a molecule has none",
        ));
    }
    if kernel != "adiabatic" {
        return Err(Error::Unsupported(format!(
            "kernel={kernel:?} is offered for crystals only; a molecule propagates its own \
             adiabatic Fock matrix"
        )));
    }
    Ok(())
}

/// The matrices the propagation steps with, without a field, and the energy, at the complex
/// orthonormal-frame `densities` (one per-spin matrix of a restricted reference, or both spins):
/// the builder of the reference read at complex densities (the meta-GGA's `tau~` with its
/// current, +U on complex occupations, the complex exchange). The oracle of the antisymmetric
/// kernel checks (V1b, R16a).
pub fn fock_matrices(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    densities: &[Matrix],
) -> Result<(Vec<Matrix>, f64)> {
    check_reference(integrals, reference)?;
    let m = integrals.m();
    if densities.is_empty() || densities.len() > 2 || densities.iter().any(|d| d.shape() != (m, m))
    {
        return Err(invalid(
            "fock_matrices needs one or two m x m density matrices",
        ));
    }
    integrals.context.run(|| {
        fock::MolecularMap::new(integrals, reference.functional.as_deref())?.bare(densities)
    })
}

pub fn propagate(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    request: &Request,
) -> Result<RealTime> {
    check_reference(integrals, reference)?;
    let direction = unit(request.direction)?;
    if !(request.strength.is_finite() && request.strength != 0.) {
        return Err(invalid("the kick strength must be finite and nonzero"));
    }
    if !(request.time_step.is_finite() && request.time_step > 0.) || request.steps == 0 {
        return Err(invalid(
            "real-time propagation needs a positive time step and at least one step",
        ));
    }
    let m = integrals.m();
    let unrestricted = reference.method == "uhf";
    if !unrestricted && reference.particles[0] != reference.particles[1] {
        return Err(invalid(
            "a restricted reference must be a closed shell; an open shell runs unrestricted",
        ));
    }
    if reference.densities.len() != if unrestricted { 2 } else { 1 }
        || reference.densities.iter().any(|d| d.shape() != (m, m))
    {
        return Err(invalid(
            "the reference density matrices do not match its method",
        ));
    }
    integrals
        .context
        .run(|| run(integrals, reference, request, direction, unrestricted))
}

fn run(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    request: &Request,
    direction: [f64; 3],
    unrestricted: bool,
) -> Result<RealTime> {
    let ctx = &integrals.context;
    let m = integrals.m();
    let mut map = fock::MolecularMap::new(integrals, reference.functional.as_deref())?;
    // A spin kick of a closed shell propagates two equal channels.
    let expand = !unrestricted && request.kick == Kick::Spin;
    let densities: Vec<Matrix> = if expand {
        vec![
            reference.densities[0].clone(),
            reference.densities[0].clone(),
        ]
    } else {
        reference.densities.clone()
    };
    let (focks, ground_energy) = map.bare(&densities)?;
    // The orbitals and levels of the reference from the same builder; a reference that is not
    // its own fixed point would drift without a kick.
    let mut channels = Vec::with_capacity(densities.len());
    let mut levels = Vec::with_capacity(densities.len());
    let mut width = 0f64;
    for (s, (fock, density)) in focks.iter().zip(&densities).enumerate() {
        let o = reference.particles[if unrestricted { s } else { 0 }];
        if o > m {
            return Err(invalid("a spin channel holds more electrons than orbitals"));
        }
        let (energies, vectors) =
            crate::math::linalg::eigen((fock + fock.adjoint()) * C64::new(0.5, 0.))?;
        let occupied = vectors.columns(0, o).into_owned();
        let stationarity = (&occupied * occupied.adjoint() - density)
            .iter()
            .map(|v| v.norm())
            .fold(0., f64::max);
        if stationarity > 1e-6 {
            return Err(invalid(format!(
                "the reference is not stationary: the occupied orbitals of channel {s} differ \
                 from the lowest eigenvectors of its Fock matrix by {stationarity:.2e} (above 1e-6)"
            )));
        }
        width = width.max(energies[m - 1] - energies[0]);
        channels.push(Channel {
            orbitals: occupied,
            occupations: vec![1.; o],
        });
        levels.push((energies, vectors, o));
    }
    let limit = request.scheme.step_width_bound();
    if request.time_step * width > limit {
        return Err(invalid(format!(
            "time_step = {} gives dt W = {:.3} with the basis width W = {width:.3} Hartree, above \
             the stability bound {limit} of propagator='{}'; lower time_step",
            request.time_step,
            request.time_step * width,
            request.scheme.name()
        )));
    }
    let positions = multipoles::position_integrals(integrals, [0.; 3])?;
    let along = |matrices: &[DMatrix<f64>; 3]| -> Matrix {
        DMatrix::from_fn(m, m, |a, b| {
            C64::new(
                (0..3)
                    .map(|k| direction[k] * matrices[k][(a, b)])
                    .sum::<f64>(),
                0.,
            )
        })
    };
    let x = along(&positions);
    let aliased_weight = aliased(&levels, &x, request.time_step);
    let gradients = multipoles::gradient_integrals(integrals)?;
    match request.gauge {
        Gauge::Length => {
            for (s, channel) in channels.iter_mut().enumerate() {
                let sign = if request.kick == Kick::Spin && s == 1 {
                    -1.
                } else {
                    1.
                };
                channel.orbitals = evolve(&x, sign * request.strength, &channel.orbitals)?.0;
            }
        }
        Gauge::Velocity => {
            if request.kick == Kick::Spin {
                return Err(invalid(
                    "a spin kick is a length-gauge kick; gauge='velocity' takes kick='density'",
                ));
            }
            // P = -i <a|grad|b>, Hermitian for the antisymmetric gradient integrals.
            let g = along(&gradients);
            map.set_field(g * C64::new(0., -1.), request.strength);
        }
    }
    let spin_sign = |s: usize| {
        if request.kick == Kick::Spin && s == 1 {
            -1.
        } else {
            1.
        }
    };
    let degeneracy = if densities.len() == 1 { 2. } else { 1. };
    let dipole_of = |states: &[Matrix]| -> [f64; 3] {
        let mut out = [0.; 3];
        for (k, value) in out.iter_mut().enumerate() {
            *value = -degeneracy
                * states
                    .iter()
                    .enumerate()
                    .map(|(s, d)| {
                        spin_sign(s)
                            * (0..m)
                                .flat_map(|a| (0..m).map(move |b| (a, b)))
                                .map(|(a, b)| d[(a, b)].re * positions[k][(b, a)])
                                .sum::<f64>()
                    })
                    .sum::<f64>();
        }
        out
    };
    // The electrons' kinetic momentum sum_s g tr[D_s (P + A)], P = -i <a|grad|b>: the velocity
    // observable (the spin momentum for a spin kick). Re tr[D P_k] = sum_ab Im D_ab G_k,ba.
    let field = match request.gauge {
        Gauge::Velocity => request.strength,
        Gauge::Length => 0.,
    };
    let momentum_of = |states: &[Matrix]| -> [f64; 3] {
        let mut out = [0.; 3];
        for (k, value) in out.iter_mut().enumerate() {
            *value = degeneracy
                * states
                    .iter()
                    .enumerate()
                    .map(|(s, d)| {
                        spin_sign(s)
                            * ((0..m)
                                .flat_map(|a| (0..m).map(move |b| (a, b)))
                                .map(|(a, b)| d[(a, b)].im * gradients[k][(b, a)])
                                .sum::<f64>()
                                + field * direction[k] * d.trace().re)
                    })
                    .sum::<f64>();
        }
        out
    };
    let origin = dipole_of(&densities);
    let mut options = Options::new(request.scheme, request.time_step);
    options.corrector_tolerance = effective_corrector_tolerance(request);
    options.max_corrector_iterations = request.max_corrector_iterations;
    let mut propagator = Propagator::new(ctx, map, channels, 0., options)?;
    let initial = propagator.energy();
    let kick_energy = initial - ground_energy;
    let mut times = vec![0.];
    let mut dipole = vec![[0.; 3]];
    let mut momentum = vec![momentum_of(&propagator.densities())];
    let mut energy = vec![initial];
    let mut norm_error = vec![propagator
        .channels()
        .iter()
        .map(Channel::norm_error)
        .fold(0., f64::max)];
    let mut largest_width = 0f64;
    let mut iterations = 0;
    let mut drift = 0f64;
    for _ in 0..request.steps {
        let step = propagator.step()?;
        largest_width = largest_width.max(step.width);
        iterations = iterations.max(step.corrector_iterations);
        let now = dipole_of(&propagator.densities());
        times.push(step.time);
        dipole.push([now[0] - origin[0], now[1] - origin[1], now[2] - origin[2]]);
        momentum.push(momentum_of(&propagator.densities()));
        energy.push(step.energy);
        norm_error.push(step.norm_error);
        if kick_energy > 0. {
            drift = drift.max((step.energy - initial).abs() / kick_energy);
        }
        if let Some(limit) = request.max_energy_drift {
            if drift > limit {
                return Err(Error::Kernel(format!(
                    "the energy drifted by {drift:.3e} kick energies by t = {} (above \
                     max_energy_drift = {limit}); lower time_step",
                    step.time
                )));
            }
        }
    }
    let worst_norm_error = norm_error.iter().fold(0f64, |t, e| t.max(*e));
    Ok(RealTime {
        times,
        along: dipole
            .iter()
            .map(|d| (0..3).map(|k| d[k] * direction[k]).sum())
            .collect(),
        dipole,
        momentum,
        energy,
        norm_error,
        ground_energy,
        kick_energy,
        energy_drift: drift,
        width,
        largest_width,
        step_width: request.time_step * width,
        worst_norm_error,
        corrector_tolerance: effective_corrector_tolerance(request),
        aliased_weight,
        builds: propagator.builds() + 1,
        corrector_iterations: iterations,
    })
}

/// The bare oscillator weight `sum 2 (e_a - e_i) abs(X_ia)^2` above `pi / dt`, as a fraction of
/// the whole bare weight (0 when it is all below).
fn aliased(levels: &[(Vec<f64>, Matrix, usize)], x: &Matrix, time_step: f64) -> f64 {
    let nyquist = std::f64::consts::PI / time_step;
    let (mut above, mut total) = (0f64, 0f64);
    for (energies, vectors, o) in levels {
        let transformed = vectors.adjoint() * x * vectors;
        for i in 0..*o {
            for a in *o..energies.len() {
                let gap = energies[a] - energies[i];
                let weight = 2. * gap * transformed[(a, i)].norm_sqr();
                total += weight;
                if gap > nyquist {
                    above += weight;
                }
            }
        }
    }
    if total > 0. {
        above / total
    } else {
        0.
    }
}
