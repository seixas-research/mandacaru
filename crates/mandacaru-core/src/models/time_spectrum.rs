//! Response functions from a sampled time signal after a delta kick.
//!
//! A signal `s(t)` sampled at `t_n = n dt`, `n = 0 .. N` (`T = N dt`), is transformed on the
//! caller's frequency axis as
//!
//! ```text
//! s(omega) = int_0^T s(t) e^{i omega t} w(t) dt
//! ```
//!
//! by the trapezoid rule: the sample at `t = 0` is the value at `0+` (a current jumps at the
//! kick) and carries the endpoint weight `dt / 2`, as does the last one. The quadrature error is
//! `O(dt^2)`. Two windows:
//!
//! - [`Window::Exponential`], `w(t) = exp(-gamma t)`: the transform is the response at the complex
//!   frequency `z = omega + i gamma`, so every line becomes a Lorentzian of half width `gamma`
//!   and the response functions below carry `z` wherever the closed form has `omega`.
//! - [`Window::Gaussian`], `w(t) = exp(-(gamma t)^2 / 2)`: a Gaussian line of standard deviation
//!   `gamma`. It has no complex-frequency form, so only the line-shape quantities accept it.
//!
//! The window's value at `T` (the truncation of the signal, [`Window::tail`]) is reported with
//! every transform; a tail that is not small leaves truncation ripples on the spectrum.
//!
//! Units are atomic and the signs are those of a kick `exp(-i kappa x)` (an electric field
//! `kappa delta(t)` along `x` for an electron of charge `-1`) or a vector potential
//! `A(t) = A_0 theta(t)` (`E = -A_0 delta(t)`): a line `omega_n` with oscillator strength `f_n`
//! gives `delta mu(t) = kappa sum_n (f_n / omega_n) sin(omega_n t)` and
//! `alpha(z) = sum_n f_n / (omega_n^2 - z^2)`.
use super::invalid;
use crate::{finite, Buffer, Context, Result};
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// The damping window of a transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    Exponential,
    Gaussian,
}
impl Window {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "exponential" => Ok(Self::Exponential),
            "gaussian" => Ok(Self::Gaussian),
            _ => Err(invalid(format!(
                "unknown window {name:?}; expected 'exponential' or 'gaussian'"
            ))),
        }
    }
    /// `w(t)` for the damping `gamma`.
    pub fn weight(self, damping: f64, time: f64) -> f64 {
        match self {
            Self::Exponential => (-damping * time).exp(),
            Self::Gaussian => (-0.5 * (damping * time).powi(2)).exp(),
        }
    }
    /// `w(T)`: the weight the window leaves on the last sample.
    pub fn tail(self, damping: f64, duration: f64) -> f64 {
        self.weight(damping, duration)
    }
}

/// A transform on a frequency axis and the window's weight at the end of the signal.
pub struct Transform {
    pub values: Buffer<C64>,
    pub tail: f64,
}

/// `int_0^T s(t) e^{i omega t} w(t) dt` on `frequencies` (module documentation).
pub fn damped_transform(
    ctx: &Context,
    samples: &[f64],
    time_step: f64,
    frequencies: &[f64],
    damping: f64,
    window: Window,
) -> Result<Transform> {
    if samples.len() < 2 {
        return Err(invalid("a time transform needs at least two samples"));
    }
    if !(time_step.is_finite() && time_step > 0.) {
        return Err(invalid(
            "the time step of a transform must be positive and finite",
        ));
    }
    if !(damping.is_finite() && damping >= 0.) {
        return Err(invalid("the damping must be finite and nonnegative"));
    }
    finite(samples, "time samples")?;
    finite(frequencies, "frequencies")?;
    let last = samples.len() - 1;
    let duration = time_step * last as f64;
    let mut values = ctx.allocate::<C64>(frequencies.len())?;
    for (value, &omega) in values.as_mut_slice().iter_mut().zip(frequencies) {
        // The phase advances by a fixed rotation per sample: the recurrence would accumulate
        // rounding over 10^4 samples, so each phase is evaluated directly.
        let mut sum = C64::new(0., 0.);
        for (n, &s) in samples.iter().enumerate() {
            let time = time_step * n as f64;
            let weight = if n == 0 || n == last { 0.5 } else { 1. };
            sum += C64::from_polar(weight * s * window.weight(damping, time), omega * time);
        }
        *value = sum * time_step;
    }
    let tail = window.tail(damping, duration);
    Ok(Transform { values, tail })
}

/// `alpha(omega) = delta mu(omega) / kappa`: the dynamic polarizability from the induced dipole
/// after a kick of `strength` `kappa` (with the exponential window, `alpha(omega + i gamma)`).
pub fn polarizability(
    ctx: &Context,
    dipole: &[f64],
    time_step: f64,
    strength: f64,
    frequencies: &[f64],
    damping: f64,
    window: Window,
) -> Result<Transform> {
    if !(strength.is_finite() && strength != 0.) {
        return Err(invalid("the kick strength must be finite and nonzero"));
    }
    let mut transform = damped_transform(ctx, dipole, time_step, frequencies, damping, window)?;
    for value in transform.values.as_mut_slice() {
        *value /= strength;
    }
    Ok(transform)
}

/// `S(omega) = (2 omega / pi) Im alpha(omega)`: the dipole strength function (per unit
/// frequency), whose lines have areas `f_n` and whose integral is the oscillator-strength sum.
pub fn strength_function(
    ctx: &Context,
    dipole: &[f64],
    time_step: f64,
    strength: f64,
    frequencies: &[f64],
    damping: f64,
    window: Window,
) -> Result<(Buffer<f64>, f64)> {
    let alpha = polarizability(
        ctx,
        dipole,
        time_step,
        strength,
        frequencies,
        damping,
        window,
    )?;
    let mut out = ctx.allocate::<f64>(frequencies.len())?;
    for ((value, &omega), a) in out
        .as_mut_slice()
        .iter_mut()
        .zip(frequencies)
        .zip(alpha.values.as_slice())
    {
        *value = 2. * omega / PI * a.im;
    }
    Ok((out, alpha.tail))
}

/// `epsilon(z) = 1 - 4 pi i j(z) / (z A_0)`, `z = omega + i gamma`, from the macroscopic current
/// density after a vector-potential kick of `strength` `A_0` (`A = A_phys / c`). The prefactor
/// is `1 / z`, not `1 / omega`: the exponential window evaluates the response at `z`. Only the
/// exponential window has this form; the Gaussian one is refused.
pub fn dielectric_function(
    ctx: &Context,
    current: &[f64],
    time_step: f64,
    strength: f64,
    frequencies: &[f64],
    damping: f64,
    window: Window,
) -> Result<Transform> {
    if window != Window::Exponential {
        return Err(invalid(
            "the dielectric function needs window='exponential': only it gives the response at \
             omega + i damping",
        ));
    }
    if !(strength.is_finite() && strength != 0.) {
        return Err(invalid("the kick strength must be finite and nonzero"));
    }
    if frequencies.iter().any(|&omega| omega == 0.) && damping == 0. {
        return Err(invalid(
            "the dielectric function at zero frequency needs a positive damping",
        ));
    }
    let mut transform = damped_transform(ctx, current, time_step, frequencies, damping, window)?;
    for (value, &omega) in transform.values.as_mut_slice().iter_mut().zip(frequencies) {
        let z = C64::new(omega, damping);
        *value = C64::new(1., 0.) - C64::new(0., 4. * PI) * *value / (z * strength);
    }
    Ok(transform)
}

/// What [`spectrum_in_user_units`] forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Strength,
    Polarizability,
    Dielectric,
}
impl Kind {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "strength" => Ok(Self::Strength),
            "polarizability" => Ok(Self::Polarizability),
            "dielectric" => Ok(Self::Dielectric),
            _ => Err(invalid(format!(
                "unknown spectrum kind {name:?}; expected 'strength', 'polarizability' or \
                 'dielectric'"
            ))),
        }
    }
}

/// The response function `kind` of a trace sampled in atomic units (`samples` at `n time_step`
/// after a kick of `strength`), on `frequencies` with `damping` given in the user's energy unit
/// (eV, or Hartree when `atomic`), returned in the user's units: `S` in 1/eV (real part),
/// `alpha` in Angstrom^3, `epsilon` dimensionless; and the window's tail.
#[allow(clippy::too_many_arguments)]
pub fn spectrum_in_user_units(
    ctx: &Context,
    kind: Kind,
    samples: &[f64],
    time_step: f64,
    strength: f64,
    frequencies: &[f64],
    damping: f64,
    window: Window,
    atomic: bool,
) -> Result<(Vec<C64>, f64)> {
    use crate::units::{from_user, Quantity};
    let omega: Vec<f64> = frequencies
        .iter()
        .map(|&w| from_user(w, Quantity::Energy, atomic))
        .collect();
    let gamma = from_user(damping, Quantity::Energy, atomic);
    let (values, tail, factor) = match kind {
        Kind::Strength => {
            let (s, tail) =
                strength_function(ctx, samples, time_step, strength, &omega, gamma, window)?;
            let values = s.as_slice().iter().map(|v| C64::new(*v, 0.)).collect();
            (values, tail, Quantity::InverseEnergy.factor(atomic))
        }
        Kind::Polarizability => {
            let t = polarizability(ctx, samples, time_step, strength, &omega, gamma, window)?;
            (
                t.values.as_slice().to_vec(),
                t.tail,
                Quantity::Volume.factor(atomic),
            )
        }
        Kind::Dielectric => {
            let t = dielectric_function(ctx, samples, time_step, strength, &omega, gamma, window)?;
            (t.values.as_slice().to_vec(), t.tail, 1.)
        }
    };
    Ok((values.into_iter().map(|v: C64| v * factor).collect(), tail))
}
