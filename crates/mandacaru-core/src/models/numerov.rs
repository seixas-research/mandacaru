//! Fourth-order Numerov shooting on a uniform radial grid: scattering waves and bound states.
//!
//! The atomic grid starts at one step; the recursion runs on the grid with `r = 0`
//! prepended (`with_origin`) and never reads index 0. Relativistically the integrated
//! variable is `W` (`P = sqrt(M) W`); the origin seed comes from the tridiagonal solution.
use super::relativity::{mass_factor, relativistic_f, solve_radial, PotentialTerms, Treatment};
use crate::{kernels, math::roots::brent, Context, Result};

/// Search window (Hartree) around the energy guess of a bound-state shoot.
pub const DEFAULT_WINDOW: f64 = 5e-3;

fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::Invalid(message.into())
}

/// The atomic grid with `r = 0` prepended.
pub fn with_origin(values: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len() + 1);
    out.push(0.);
    out.extend_from_slice(values);
    out
}

/// `f = 2[V + l(l+1)/2r^2 - E]` on the origin-extended grid; `f[0] = 0`.
pub fn radial_f(r0: &[f64], v0: &[f64], l: usize, energy: f64) -> Vec<f64> {
    let ll = (l * (l + 1)) as f64;
    let mut f = (0..r0.len())
        .map(|i| 2.0 * (v0[i] + ll / (2.0 * r0[i] * r0[i]) - energy))
        .collect::<Vec<_>>();
    for value in f.iter_mut() {
        if !value.is_finite() {
            *value = 0.;
        }
    }
    f[0] = 0.;
    f
}

/// Power-series start `u = r^{l+1} sum a_k r^k` of the regular solution at the first two
/// nodes (index 1 and 2 of the extended grid), carrying the screening of `V`.
pub fn origin_seed(
    r0: &[f64],
    l: usize,
    z: f64,
    potential0: Option<&[f64]>,
    energy: f64,
) -> (f64, f64) {
    let c0 = potential0.map_or(0., |v| v[1] + z / r0[1] - energy);
    let lf = l as f64;
    let mut a = vec![1.0, -z / (lf + 1.)];
    for k in 2..=6 {
        let kf = k as f64;
        let next = (-2.0 * z * a[k - 1] + 2.0 * c0 * a[k - 2]) / (kf * (kf + 2. * lf + 1.));
        a.push(next);
    }
    let power = lf + 1.;
    let series = |x: f64| {
        x.powf(power)
            * a.iter()
                .enumerate()
                .map(|(k, ak)| ak * x.powi(k as i32))
                .sum::<f64>()
    };
    (series(r0[1]), series(r0[2]))
}

/// Fourth-order centered first derivative at one index (needs `2 <= index <= len - 3`).
pub fn derivative4(u: &[f64], h: f64, index: usize) -> f64 {
    (u[index - 2] - 8. * u[index - 1] + 8. * u[index + 1] - u[index + 2]) / (12. * h)
}

/// Arrays of one shoot at one energy.
struct Shoot {
    f: Vec<f64>,
    sqrt_mass: Option<Vec<f64>>,
}

/// Everything that does not depend on the trial energy of a shoot.
struct Setup<'a> {
    r: &'a [f64],
    r0: Vec<f64>,
    v0: Vec<f64>,
    l: usize,
    z: f64,
    treatment: Treatment,
    kappa: Option<i32>,
    terms: Option<PotentialTerms>,
    seed: Option<(f64, f64)>,
}

impl<'a> Setup<'a> {
    fn new(
        ctx: &Context,
        r: &'a [f64],
        potential: &[f64],
        l: usize,
        z: f64,
        treatment: Treatment,
        kappa: Option<i32>,
    ) -> Result<Self> {
        let r0 = with_origin(r);
        let v0 = with_origin(potential);
        let (terms, seed) = if treatment == Treatment::None {
            (None, None)
        } else {
            let terms = PotentialTerms::new(r, potential, z)?;
            // The tridiagonal solution carries the right power and screening at the origin;
            // two of its nodes start the recursion, serving every trial energy of the shoot.
            let solved = solve_radial(
                ctx,
                r,
                potential,
                l,
                0,
                if treatment == Treatment::Dirac {
                    Treatment::Dirac
                } else {
                    Treatment::Scalar
                },
                kappa,
                z,
                None,
                None,
            )?;
            let (mass, _) = mass_factor(potential, solved.1, &terms);
            let w = solved
                .0
                .iter()
                .zip(&mass)
                .map(|(p, m)| p / m.max(1e-30).sqrt())
                .collect::<Vec<_>>();
            let scale = if w[1].abs() > 0. { w[1] } else { 1. };
            (Some(terms), Some((w[0] / scale, w[1] / scale)))
        };
        Ok(Self {
            r,
            r0,
            v0,
            l,
            z,
            treatment,
            kappa,
            terms,
            seed,
        })
    }

    /// `f` and `sqrt(M)` on the extended grid. The first three entries of the relativistic `f`
    /// are zero: the closed-form origin derivatives are not finite there.
    fn arrays(&self, energy: f64) -> Shoot {
        let Some(terms) = &self.terms else {
            return Shoot {
                f: radial_f(&self.r0, &self.v0, self.l, energy),
                sqrt_mass: None,
            };
        };
        let potential = &self.v0[1..];
        let f = relativistic_f(
            self.r,
            potential,
            self.l,
            self.kappa.unwrap_or(-1),
            energy,
            terms,
            self.treatment,
        );
        let mut f0 = vec![0.; f.len() + 1];
        f0[3..].copy_from_slice(&f[2..]);
        let (mass, _) = mass_factor(potential, energy, terms);
        let mut sqrt_mass = Vec::with_capacity(mass.len() + 1);
        sqrt_mass.push(mass[0].sqrt());
        sqrt_mass.extend(mass.iter().map(|m| m.sqrt()));
        Shoot {
            f: f0,
            sqrt_mass: Some(sqrt_mass),
        }
    }

    fn outward(&self, ctx: &Context, shoot: &Shoot, energy: f64) -> Result<Vec<f64>> {
        let seed = self
            .seed
            .unwrap_or_else(|| origin_seed(&self.r0, self.l, self.z, Some(&self.v0), energy));
        let mut start = vec![0.; self.r0.len()];
        start[1] = seed.0;
        start[2] = seed.1;
        let h = self.r0[1] - self.r0[0];
        let out = kernels::numerov(
            ctx,
            &shoot.f,
            &vec![0.; self.r0.len()],
            &start,
            h * h,
            1,
            false,
        )?;
        Ok(out.as_slice().to_vec())
    }
}

#[allow(clippy::too_many_arguments)]
/// Outward Numerov solution `u(r)` at `energy` (arbitrary scale), on the atomic grid.
pub fn scattering_wave(
    ctx: &Context,
    r: &[f64],
    potential: &[f64],
    l: usize,
    energy: f64,
    z: f64,
    treatment: Treatment,
    kappa: Option<i32>,
) -> Result<Vec<f64>> {
    let setup = Setup::new(ctx, r, potential, l, z, treatment, kappa)?;
    let shoot = setup.arrays(energy);
    let mut u = setup.outward(ctx, &shoot, energy)?;
    if let Some(m) = &shoot.sqrt_mass {
        for (u, m) in u.iter_mut().zip(m) {
            *u *= m;
        }
    }
    Ok(u[1..].to_vec())
}

#[allow(clippy::too_many_arguments)]
/// Numerov bound state `(u, energy)` near `energy_guess`; `None` when no bracket is found
/// (the caller keeps its tridiagonal solution).
///
/// Outward and inward integrations are matched at the outermost classical turning point
/// (read off the nonrelativistic `f`) and the logarithmic-derivative mismatch is brought to
/// zero with Brent's method on the energy, from a bracket of `+/- window` around the guess,
/// widened up to twelve times. `u` is normalized to `int u^2 dr = 1`, positive near the origin.
pub fn bound_state(
    ctx: &Context,
    r: &[f64],
    potential: &[f64],
    l: usize,
    energy_guess: f64,
    z: f64,
    window: f64,
    treatment: Treatment,
    kappa: Option<i32>,
) -> Result<Option<(Vec<f64>, f64)>> {
    let setup = Setup::new(ctx, r, potential, l, z, treatment, kappa)?;
    let n0 = setup.r0.len();
    let h = setup.r0[1] - setup.r0[0];
    // (mismatch, outward, inward, match index) at one trial energy.
    let mismatch = |energy: f64| -> Result<(f64, Vec<f64>, Vec<f64>, usize)> {
        let shoot = setup.arrays(energy);
        let plain = radial_f(&setup.r0, &setup.v0, l, energy);
        let turning = plain[10..].iter().rposition(|f| *f < 0.);
        let matched = turning.map_or(n0 / 2, |i| i + 10);
        let matched = matched.max(20).min(n0 - 20);
        let mut out = setup.outward(ctx, &shoot, energy)?;
        let decay = (-2.0 * energy).max(1e-6).sqrt();
        let mut start = vec![0.; n0];
        start[n0 - 1] = (-decay * setup.r0[n0 - 1]).exp();
        start[n0 - 2] = (-decay * setup.r0[n0 - 2]).exp();
        let mut inward = kernels::numerov(
            ctx,
            &shoot.f,
            &vec![0.; n0],
            &start,
            h * h,
            matched - 3,
            true,
        )?
        .as_slice()
        .to_vec();
        let scale = out[matched] / inward[matched];
        for v in inward.iter_mut() {
            *v *= scale;
        }
        if let Some(m) = &shoot.sqrt_mass {
            for (v, m) in out.iter_mut().zip(m) {
                *v *= m;
            }
            for (v, m) in inward.iter_mut().zip(m) {
                *v *= m;
            }
        }
        let value =
            (derivative4(&out, h, matched) - derivative4(&inward, h, matched)) / out[matched];
        Ok((value, out, inward, matched))
    };
    let value_at = |energy: f64| mismatch(energy).map_or(f64::NAN, |m| m.0);
    let (mut lo, mut hi) = (energy_guess - window, energy_guess + window);
    let mut bracketed = false;
    for _ in 0..12 {
        if value_at(lo) * value_at(hi) < 0. {
            bracketed = true;
            break;
        }
        lo -= 2. * window;
        hi += 2. * window;
    }
    if !bracketed {
        return Ok(None);
    }
    let Ok(energy) = brent(|e| mismatch(e).map(|m| m.0), lo, hi, 1e-13, 1e-13, 100) else {
        return Ok(None);
    };
    let (_, out, inward, matched) = mismatch(energy).map_err(|e| invalid(e.to_string()))?;
    let mut u = out[..matched]
        .iter()
        .chain(&inward[matched..])
        .copied()
        .collect::<Vec<_>>();
    u.remove(0);
    // The sign is fixed on the first twenty nodes; `normalize` then sees a positive sum.
    let head = u[..20].iter().sum::<f64>();
    normalize_trapezoid(&mut u, r);
    if head < 0. {
        for v in u.iter_mut() {
            *v = -*v;
        }
    }
    Ok(Some((u, energy)))
}

/// Divide by the trapezoid norm without touching the sign.
fn normalize_trapezoid(u: &mut [f64], r: &[f64]) {
    let mut sum = 0.;
    for i in 1..u.len() {
        sum += 0.5 * (r[i] - r[i - 1]) * (u[i - 1] * u[i - 1] + u[i] * u[i]);
    }
    let scale = sum.sqrt();
    for v in u.iter_mut() {
        *v /= scale;
    }
}
