//! Unitary propagation of orbitals under a state-dependent Hermitian matrix.
//!
//! The equation is `i dC/dt = F(t, D) C` per channel, with `D = C f C^dagger` the channel's
//! density matrix in an orthonormal frame and `F` Hermitian. Nothing here knows what `F` is: a
//! [`FockMap`] returns it, with the energy, for the densities of every channel at a time. Every
//! half-step is the exact exponential `exp(-i tau F) = Y exp(-i tau e) Y^dagger` of the Hermitized
//! matrix ([`evolve`]), so the orbitals stay orthonormal to round-off at any step, and a constant
//! `F` is propagated exactly at any step. That is **not** unconditional stability: `F` depends on
//! the propagated state, and the self-consistent map can amplify the error. The step must be
//! chosen against the basis width `W = e_max - e_min` of `F`, which every step reports.
//!
//! Three schemes, with `F_n = F(t_n, D_n)` and `tau = dt / 2`:
//!
//! - [`Scheme::Midpoint`]: `C_{n+1} = exp(-i dt (3 F_n - F_{n-1}) / 2) C_n`; one build.
//! - [`Scheme::Aetrs`]: `C_{n+1} = exp(-i tau (2 F_n - F_{n-1})) exp(-i tau F_n) C_n`; one build.
//! - [`Scheme::Etrs`]: `C_{n+1} = exp(-i tau F_{n+1}) exp(-i tau F_n) C_n` with `F_{n+1}` built at
//!   the corrected state until the density changes by less than the corrector tolerance; a build
//!   per corrector iteration. Converged, the step is time-reversal symmetric.
//!
//! (Enforced time-reversal symmetry and its approximation: Castro, Marques and Rubio,
//! J. Chem. Phys. 121, 3425 (2004).) The step that has no `F_{n-1}` (the first one, and the first
//! one after [`Propagator::reverse`]) is an ETRS step with one corrector iteration, so every
//! scheme is second order from the first step.
//!
//! Every build happens at the end of a step, at the new state: `F_{n+1}` is never extrapolated
//! into the history. Every step is serial and in a fixed order, so the trajectory depends only on
//! the map's values.
use super::{invalid, linalg::eigen};
use crate::{Context, Reservation, Result};
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

type Matrix = DMatrix<C64>;

/// The stepping rule of a [`Propagator`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    /// Exponential midpoint with the linearly extrapolated midpoint matrix.
    Midpoint,
    /// Approximate enforced time-reversal symmetry: ETRS with an extrapolated `F_{n+1}`.
    Aetrs,
    /// Enforced time-reversal symmetry with a self-consistent `F_{n+1}`.
    Etrs,
}
impl Scheme {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "midpoint" => Ok(Self::Midpoint),
            "aetrs" => Ok(Self::Aetrs),
            "etrs" => Ok(Self::Etrs),
            _ => Err(invalid(format!(
                "unknown propagator {name:?}; expected 'aetrs', 'etrs' or 'midpoint'"
            ))),
        }
    }
    /// The largest `dt W` (`W = e_max - e_min` of the matrix) a run of this scheme takes: 4 for
    /// ETRS, 1 for AETRS and the midpoint. Measured on an interacting many-level model and on
    /// water (TDDFT_PLAN.md, the T6.0 record): past 1 the extrapolated schemes lose 4e-2 kick
    /// energies on water and diverge from 2 to 8; ETRS stays within 1e-1 to 4 (owner decision
    /// Q18b, 2026-11-12).
    pub fn step_width_bound(self) -> f64 {
        match self {
            Self::Etrs => 4.,
            Self::Aetrs | Self::Midpoint => 1.,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Midpoint => "midpoint",
            Self::Aetrs => "aetrs",
            Self::Etrs => "etrs",
        }
    }
}

/// What a [`FockMap`] returns: one Hermitian matrix per channel, in the orthonormal frame of the
/// orbitals, and the energy of the state.
#[derive(Clone, Debug)]
pub struct Evaluation {
    pub matrices: Vec<Matrix>,
    pub energy: f64,
}

/// The state-dependent Hermitian matrix of the equation of motion.
pub trait FockMap {
    /// The matrices and the energy at `time` for the channel densities `D = C f C^dagger`.
    fn evaluate(&mut self, time: f64, densities: &[Matrix]) -> Result<Evaluation>;
}

/// The controls of a [`Propagator`].
#[derive(Clone, Debug)]
pub struct Options {
    pub scheme: Scheme,
    /// The time step; negative runs backward.
    pub time_step: f64,
    /// The largest change of any density element that ends the ETRS corrector.
    pub corrector_tolerance: f64,
    /// The largest number of ETRS corrector iterations in a step.
    pub max_corrector_iterations: usize,
    /// The largest `abs(C^dagger C - 1)` a step may leave.
    pub max_norm_error: f64,
    /// The largest anti-Hermitian part of a returned matrix, relative to its largest element
    /// (at least 1); it is removed before the exponential.
    pub hermitian_tolerance: f64,
}
impl Options {
    pub fn new(scheme: Scheme, time_step: f64) -> Self {
        Self {
            scheme,
            time_step,
            corrector_tolerance: 1e-10,
            max_corrector_iterations: 8,
            max_norm_error: 1e-8,
            hermitian_tolerance: 1e-10,
        }
    }
    fn validate(&self) -> Result<()> {
        if !self.time_step.is_finite() || self.time_step == 0. {
            return Err(invalid("the time step must be finite and nonzero"));
        }
        if !(self.corrector_tolerance.is_finite() && self.corrector_tolerance > 0.)
            || self.max_corrector_iterations == 0
            || !(self.max_norm_error.is_finite() && self.max_norm_error > 0.)
            || !(self.hermitian_tolerance.is_finite() && self.hermitian_tolerance >= 0.)
        {
            return Err(invalid(
                "the corrector tolerance, norm bound and Hermitian tolerance must be positive \
                 and finite, with at least one corrector iteration",
            ));
        }
        Ok(())
    }
}

/// The orbitals of one channel: `m x n` columns, orthonormal, with their occupations.
#[derive(Clone, Debug)]
pub struct Channel {
    pub orbitals: Matrix,
    pub occupations: Vec<f64>,
}
impl Channel {
    /// `D = C f C^dagger`.
    pub fn density(&self) -> Matrix {
        let mut weighted = self.orbitals.clone();
        for (j, &f) in self.occupations.iter().enumerate() {
            weighted.column_mut(j).scale_mut(f);
        }
        &weighted * self.orbitals.adjoint()
    }
    /// `max abs(C^dagger C - 1)`.
    pub fn norm_error(&self) -> f64 {
        let overlap = self.orbitals.adjoint() * &self.orbitals;
        let n = overlap.nrows();
        let mut worst = 0f64;
        for j in 0..n {
            for i in 0..n {
                let target = if i == j { 1. } else { 0. };
                worst = worst.max((overlap[(i, j)] - target).norm());
            }
        }
        worst
    }
}

/// The record of one step.
#[derive(Clone, Debug)]
pub struct Step {
    /// The time at the end of the step.
    pub time: f64,
    /// The energy the map returned at the new state.
    pub energy: f64,
    /// `max abs(C^dagger C - 1)` over the channels after the step.
    pub norm_error: f64,
    /// The largest basis width `e_max - e_min` of the matrices exponentiated in the step.
    pub width: f64,
    /// The map evaluations of the step.
    pub builds: usize,
    /// The ETRS corrector iterations (0 for a one-build step).
    pub corrector_iterations: usize,
    /// The last density change of the corrector (0 for a one-build step).
    pub corrector_change: f64,
    /// Whether the corrector reached its tolerance (true for a one-build step).
    pub converged: bool,
}

/// The Hermitian part `(H + H^dagger) / 2`, refused when the anti-Hermitian part exceeds
/// `tolerance` times the largest element (at least 1).
pub fn hermitian_part(matrix: &Matrix, tolerance: f64) -> Result<Matrix> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(invalid("a Hermitian matrix must be square"));
    }
    let mut scale = 1f64;
    let mut defect = 0f64;
    for j in 0..n {
        for i in 0..n {
            let value = matrix[(i, j)];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(invalid("nonfinite matrix in the equation of motion"));
            }
            scale = scale.max(value.norm());
            defect = defect.max((value - matrix[(j, i)].conj()).norm());
        }
    }
    if defect > tolerance * scale {
        return Err(invalid(format!(
            "the matrix of the equation of motion is not Hermitian: anti-Hermitian part {defect:.3e} \
             against a largest element {scale:.3e}"
        )));
    }
    let mut out = (matrix + matrix.adjoint()) * C64::new(0.5, 0.);
    for i in 0..n {
        out[(i, i)].im = 0.;
    }
    Ok(out)
}

/// `exp(-i duration H) block` for a Hermitian `H` (already Hermitized), and the width
/// `e_max - e_min` of `H`. One eigendecomposition; exact for any `duration`.
pub fn evolve(hamiltonian: &Matrix, duration: f64, block: &Matrix) -> Result<(Matrix, f64)> {
    if block.nrows() != hamiltonian.nrows() {
        return Err(invalid(
            "the orbitals and the matrix have different dimensions",
        ));
    }
    let (values, vectors) = eigen(hamiltonian.clone())?;
    let width = match (values.first(), values.last()) {
        (Some(low), Some(high)) => high - low,
        _ => 0.,
    };
    let mut projected = vectors.adjoint() * block;
    for (i, &e) in values.iter().enumerate() {
        let phase = C64::from_polar(1., -duration * e);
        for j in 0..projected.ncols() {
            projected[(i, j)] *= phase;
        }
    }
    Ok((vectors * projected, width))
}

/// Propagates the channels of a state under a [`FockMap`].
pub struct Propagator<M: FockMap> {
    map: M,
    options: Options,
    channels: Vec<Channel>,
    time: f64,
    steps: usize,
    builds: usize,
    energy: f64,
    current: Vec<Matrix>,
    previous: Option<Vec<Matrix>>,
    _workspace: Reservation,
}

impl<M: FockMap> Propagator<M> {
    /// Starts at `time` from `channels` (orthonormal columns, one occupation per column) and builds
    /// the matrices of the initial state.
    pub fn new(
        ctx: &Context,
        mut map: M,
        channels: Vec<Channel>,
        time: f64,
        options: Options,
    ) -> Result<Self> {
        options.validate()?;
        if channels.is_empty() || !time.is_finite() {
            return Err(invalid(
                "a propagation needs at least one channel and a finite start",
            ));
        }
        let mut bytes = 0usize;
        for channel in &channels {
            let (m, n) = channel.orbitals.shape();
            // An empty channel (the minority spin of a one-electron state) has no columns.
            if m == 0 || n > m || channel.occupations.len() != n {
                return Err(invalid(
                    "each channel needs m x n orbitals with n <= m and one occupation per column",
                ));
            }
            if channel
                .occupations
                .iter()
                .any(|f| !f.is_finite() || *f < 0.)
            {
                return Err(invalid("occupations must be finite and nonnegative"));
            }
            if channel
                .orbitals
                .iter()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
            {
                return Err(invalid("nonfinite orbitals"));
            }
            let norm = channel.norm_error();
            if norm > options.max_norm_error {
                return Err(invalid(format!(
                    "the initial orbitals are not orthonormal: max |C^dagger C - 1| = {norm:.3e}"
                )));
            }
            // The two stored matrices, the density, two exponential workspaces (the matrix and
            // its eigenvectors) and two copies of the orbitals (the step and the corrector).
            let per = crate::product(&[m, m, 5])?
                .checked_add(crate::product(&[m, n, 2])?)
                .ok_or(crate::Error::Overflow)?;
            bytes = bytes
                .checked_add(crate::product(&[per, std::mem::size_of::<C64>()])?)
                .ok_or(crate::Error::Overflow)?;
        }
        let workspace = ctx.reserve(bytes)?;
        let densities: Vec<_> = channels.iter().map(Channel::density).collect();
        let evaluation = map.evaluate(time, &densities)?;
        let current = Self::checked(&channels, evaluation.matrices, options.hermitian_tolerance)?;
        if !evaluation.energy.is_finite() {
            return Err(invalid("the map returned a nonfinite energy"));
        }
        Ok(Self {
            map,
            options,
            channels,
            time,
            steps: 0,
            builds: 1,
            energy: evaluation.energy,
            current,
            previous: None,
            _workspace: workspace,
        })
    }

    fn checked(channels: &[Channel], matrices: Vec<Matrix>, tolerance: f64) -> Result<Vec<Matrix>> {
        if matrices.len() != channels.len() {
            return Err(invalid(
                "the map returned a matrix count unlike the channel count",
            ));
        }
        matrices
            .iter()
            .zip(channels)
            .map(|(matrix, channel)| {
                if matrix.nrows() != channel.orbitals.nrows() {
                    return Err(invalid("the map returned a matrix of the wrong dimension"));
                }
                hermitian_part(matrix, tolerance)
            })
            .collect()
    }

    fn build(&mut self, time: f64, channels: &[Channel]) -> Result<(Vec<Matrix>, f64)> {
        let densities: Vec<_> = channels.iter().map(Channel::density).collect();
        let evaluation = self.map.evaluate(time, &densities)?;
        self.builds += 1;
        if !evaluation.energy.is_finite() {
            return Err(invalid("the map returned a nonfinite energy"));
        }
        let matrices = Self::checked(
            channels,
            evaluation.matrices,
            self.options.hermitian_tolerance,
        )?;
        Ok((matrices, evaluation.energy))
    }

    /// Each channel advanced by `exp(-i tau second) exp(-i tau first)` (or one exponential when
    /// `second` is `None`, with `tau` the whole duration).
    fn apply(
        &self,
        first: &[Matrix],
        second: Option<&[Matrix]>,
        tau: f64,
        width: &mut f64,
    ) -> Result<Vec<Channel>> {
        let mut out = Vec::with_capacity(self.channels.len());
        for (s, channel) in self.channels.iter().enumerate() {
            let (mut orbitals, w) = evolve(&first[s], tau, &channel.orbitals)?;
            *width = width.max(w);
            if let Some(second) = second {
                let (next, w) = evolve(&second[s], tau, &orbitals)?;
                *width = width.max(w);
                orbitals = next;
            }
            out.push(Channel {
                orbitals,
                occupations: channel.occupations.clone(),
            });
        }
        Ok(out)
    }

    fn combine(a: &[Matrix], wa: f64, b: &[Matrix], wb: f64) -> Vec<Matrix> {
        a.iter()
            .zip(b)
            .map(|(a, b)| a * C64::new(wa, 0.) + b * C64::new(wb, 0.))
            .collect()
    }

    fn density_change(a: &[Channel], b: &[Channel]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(a, b)| {
                (a.density() - b.density())
                    .iter()
                    .map(|v| v.norm())
                    .fold(0., f64::max)
            })
            .fold(0., f64::max)
    }

    /// One step, recorded; a corrector that misses its tolerance is reported in
    /// [`Step::converged`] and its last iterate is kept. [`Propagator::step`] refuses it instead.
    pub fn advance(&mut self) -> Result<Step> {
        let dt = self.options.time_step;
        let tau = 0.5 * dt;
        let next_time = self.time + dt;
        let builds_before = self.builds;
        let mut width = 0f64;
        let current = self.current.clone();
        let previous = self.previous.clone();
        let (corrected, matrices, energy, iterations, change, converged) =
            match (self.options.scheme, previous.as_ref()) {
                (Scheme::Midpoint, Some(previous)) => {
                    let middle = Self::combine(&current, 1.5, previous, -0.5);
                    let channels = self.apply(&middle, None, dt, &mut width)?;
                    let (matrices, energy) = self.build(next_time, &channels)?;
                    (channels, matrices, energy, 0, 0., true)
                }
                (Scheme::Aetrs, Some(previous)) => {
                    let predicted = Self::combine(&current, 2., previous, -1.);
                    let channels = self.apply(&current, Some(&predicted), tau, &mut width)?;
                    let (matrices, energy) = self.build(next_time, &channels)?;
                    (channels, matrices, energy, 0, 0., true)
                }
                (scheme, previous) => {
                    // ETRS, or the start of any scheme: predict with the extrapolated (or the
                    // current) matrix, then correct.
                    let predicted = match previous {
                        Some(previous) => Self::combine(&current, 2., previous, -1.),
                        None => current.clone(),
                    };
                    let mut channels = self.apply(&current, Some(&predicted), tau, &mut width)?;
                    let (mut matrices, mut energy) = self.build(next_time, &channels)?;
                    let limit = if scheme == Scheme::Etrs {
                        self.options.max_corrector_iterations
                    } else {
                        1
                    };
                    let mut iterations = 0;
                    let mut change = f64::INFINITY;
                    while iterations < limit {
                        let next = match scheme {
                            Scheme::Midpoint if previous.is_none() => {
                                let middle = Self::combine(&current, 0.5, &matrices, 0.5);
                                self.apply(&middle, None, dt, &mut width)?
                            }
                            _ => self.apply(&current, Some(&matrices), tau, &mut width)?,
                        };
                        iterations += 1;
                        change = Self::density_change(&next, &channels);
                        channels = next;
                        (matrices, energy) = self.build(next_time, &channels)?;
                        if change < self.options.corrector_tolerance {
                            break;
                        }
                    }
                    let converged =
                        scheme != Scheme::Etrs || change < self.options.corrector_tolerance;
                    (channels, matrices, energy, iterations, change, converged)
                }
            };
        let norm_error = corrected.iter().map(Channel::norm_error).fold(0., f64::max);
        if norm_error > self.options.max_norm_error {
            return Err(crate::Error::Kernel(format!(
                "the orbitals lost orthonormality: max |C^dagger C - 1| = {norm_error:.3e} after \
                 step {} (bound {:.1e})",
                self.steps + 1,
                self.options.max_norm_error
            )));
        }
        self.channels = corrected;
        self.previous = Some(std::mem::replace(&mut self.current, matrices));
        self.time = next_time;
        self.steps += 1;
        self.energy = energy;
        Ok(Step {
            time: next_time,
            energy,
            norm_error,
            width,
            builds: self.builds - builds_before,
            corrector_iterations: iterations,
            corrector_change: if iterations == 0 { 0. } else { change },
            converged,
        })
    }

    /// One step; an ETRS corrector that misses its tolerance raises, naming the option.
    pub fn step(&mut self) -> Result<Step> {
        let step = self.advance()?;
        if !step.converged {
            return Err(crate::Error::Kernel(format!(
                "the ETRS corrector did not reach corrector_tolerance = {:.1e} in {} iterations at \
                 t = {} (last density change {:.3e}); lower time_step or raise the tolerance",
                self.options.corrector_tolerance,
                step.corrector_iterations,
                step.time,
                step.corrector_change
            )));
        }
        Ok(step)
    }

    /// Runs backward from here: the step changes sign and the history restarts.
    pub fn reverse(&mut self) {
        self.options.time_step = -self.options.time_step;
        self.previous = None;
    }

    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }
    pub fn densities(&self) -> Vec<Matrix> {
        self.channels.iter().map(Channel::density).collect()
    }
    /// The matrices at the current state.
    pub fn matrices(&self) -> &[Matrix] {
        &self.current
    }
    pub fn time(&self) -> f64 {
        self.time
    }
    pub fn energy(&self) -> f64 {
        self.energy
    }
    pub fn steps(&self) -> usize {
        self.steps
    }
    /// Map evaluations so far, the initial one included.
    pub fn builds(&self) -> usize {
        self.builds
    }
    pub fn options(&self) -> &Options {
        &self.options
    }
    pub fn map(&self) -> &M {
        &self.map
    }
}
