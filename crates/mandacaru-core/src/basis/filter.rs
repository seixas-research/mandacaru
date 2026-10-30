//! The Fourier filter of a radial basis function: the spherical Bessel pair
//! `F_l(k) = int R(r) j_l(kr) r^2 dr`, `R(r) = (2/pi) int F_l(k) j_l(kr) k^2 dk`
//! with a raised-cosine window on `F_l`, evaluated on the function's own radial mesh.
//!
//! The kernel `j_l(k_i r_j)` is never stored. Each output element is a reduction over
//! the other index that evaluates the kernel on the fly, so the memory is a fixed
//! number of arrays of the radial length and of the wave-vector length, independent
//! of the cutoff `k_c` (the wave-vector mesh length grows with `k_c`, but only
//! three arrays of that length are held; there is no `n_k x n_r` matrix). Every
//! array is reserved from the `Context` before it is allocated.
use super::invalid;
use crate::{Buffer, Context, Error, Result};
use mandacaru_kernels::two_center::spherical_bessel;
use std::f64::consts::PI;

/// Width of the `k`-space roll-off, as a fraction of `k_c`: the window is 1 below
/// `(1 - rolloff) k_c` and reaches 0 at `k_c`. A hard step rings with a
/// `sin(k_c r)/r` tail that only decays as `1/r`; a raised cosine over a quarter of
/// the band decays as `1/r^3`. Measured with the switch-off in place, a hard step
/// leaves 5x more residual weight above `k_c` on the compact split zetas.
pub const ROLLOFF_FRACTION: f64 = 0.25;
/// Points per oscillation of `j_l(k r_max)` in the `k` quadrature. The round trip
/// does not move between 4 and 16; 8 is margin.
pub const TRANSFORM_OVERSAMPLE: usize = 8;
/// A function's support radius is where its remaining tail norm drops below this
/// fraction of the total: where the switch-off (or the mask) is placed.
pub const SUPPORT_TAIL_NORM: f64 = 1e-6;
/// The switch-off reaches zero at this multiple of the support radius, clipped to
/// the end of the table. The filtered function needs room for its ringing tail:
/// 1.25 / 1.5 / 2.0 / 3.0 leave 5.6e-4 / 2.8e-4 / 9.9e-5 / 1.9e-5 of the norm above
/// `k_c` (O 2s zeta 3) at 1.44 / 1.73 / 2.30 / 3.46 Bohr of range; 1.5 is the knee.
pub const SWITCH_MARGIN: f64 = 1.5;
/// The mask method's window is exactly 1 below this fraction of its outer radius,
/// i.e. it starts decaying inside the support.
pub const MASK_INNER_FRACTION: f64 = 0.6;
/// Largest angular momentum: the upward recurrence of `j_l` is stable for `x >= 8 > l`.
pub const MAX_L: usize = 6;
/// Largest wave-vector mesh length (a larger one is a request, not a calculation).
const MAX_WAVEVECTORS: f64 = 1e9;

/// Real-space treatment of the filtered function; see `mandacaru.basis.filtering`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Filter and keep the tail.
    Plain,
    /// Filter, then multiply by a window that is 1 out to the support radius.
    Switch,
    /// Divide by a window that starts decaying inside the support, filter, multiply back.
    Mask,
}
impl Method {
    /// The names, default first.
    pub const NAMES: [&'static str; 3] = ["switch", "plain", "mask"];
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "plain" => Ok(Self::Plain),
            "switch" => Ok(Self::Switch),
            "mask" => Ok(Self::Mask),
            _ => Err(invalid(format!(
                "unknown filter method {name:?}; use one of {:?}",
                Self::NAMES
            ))),
        }
    }
}

/// Options of [`filter`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    pub method: Method,
    pub rolloff: f64,
    pub oversample: usize,
    /// Rescale the result to the original norm.
    pub renormalize: bool,
    pub switch_margin: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            method: Method::Switch,
            rolloff: ROLLOFF_FRACTION,
            oversample: TRANSFORM_OVERSAMPLE,
            renormalize: true,
            switch_margin: SWITCH_MARGIN,
        }
    }
}

/// What the filter measured: the numbers the guide quotes and the tests assert.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Report {
    pub residual_before: f64,
    pub residual_after: f64,
    pub support_radius: f64,
    pub mask_radius: f64,
    pub masked: bool,
    pub norm_kept: f64,
    pub max_change: f64,
}

fn check_table(r: &[f64], values: &[f64]) -> Result<()> {
    if r.len() != values.len() {
        return Err(invalid("the radial mesh and the values differ in length"));
    }
    if r.len() < 2 {
        return Err(invalid("a radial table needs at least two nodes to filter"));
    }
    if r.iter().chain(values).any(|x| !x.is_finite()) {
        return Err(invalid("the radial table must be finite"));
    }
    if r[0] < 0. || r.windows(2).any(|w| w[1] <= w[0]) {
        return Err(invalid(
            "the radial mesh must be non-negative and strictly increasing",
        ));
    }
    Ok(())
}

/// `r` and `values` with the node `r = 0` prepended using the `r^l` continuation
/// (the tables start at `r_0 > 0`; a plain quadrature would drop `[0, r_0]`).
fn augment(
    ctx: &Context,
    r: &[f64],
    values: &[f64],
    l: usize,
) -> Result<(Buffer<f64>, Buffer<f64>)> {
    let extra = usize::from(r[0] > 0.);
    let mut ra = ctx.allocate::<f64>(r.len() + extra)?;
    let mut va = ctx.allocate::<f64>(r.len() + extra)?;
    if extra == 1 {
        va.as_mut_slice()[0] = if l == 0 { values[0] } else { 0. };
    }
    ra.as_mut_slice()[extra..].copy_from_slice(r);
    va.as_mut_slice()[extra..].copy_from_slice(values);
    Ok((ra, va))
}

fn trapezoid(f: impl Fn(usize) -> f64, x: &[f64]) -> f64 {
    let mut total = 0.;
    let mut previous = f(0);
    for i in 1..x.len() {
        let current = f(i);
        total += 0.5 * (previous + current) * (x[i] - x[i - 1]);
        previous = current;
    }
    total
}

/// `int R^2 r^2 dr`.
fn radial_norm(r: &[f64], values: &[f64]) -> f64 {
    trapezoid(|i| values[i] * values[i] * r[i] * r[i], r)
}

/// Run `f(start, chunk)` over contiguous chunks of `out`, one per context thread.
fn parallel(ctx: &Context, out: &mut [f64], f: impl Fn(usize, &mut [f64]) + Sync) -> Result<()> {
    let workers = ctx.threads().max(1).min(out.len().max(1));
    if workers == 1 {
        f(0, out);
        return Ok(());
    }
    let chunk = out.len().div_ceil(workers);
    let f = &f;
    ctx.run(|| {
        std::thread::scope(|scope| {
            let handles = out
                .chunks_mut(chunk)
                .enumerate()
                .map(|(i, part)| {
                    mandacaru_kernels::runtime::spawn(scope, move || f(i * chunk, part))
                })
                .collect::<Vec<_>>();
            let mut outcome = Ok(());
            for handle in handles {
                if handle.join().is_err() && outcome.is_ok() {
                    outcome = Err(Error::Kernel("radial filter worker panicked".into()));
                }
            }
            outcome
        })
    })
}

/// The uniform wave-vector mesh `[0, k_max]` that resolves `j_l(k r_max)`.
fn wavevectors(ctx: &Context, r_max: f64, k_max: f64, oversample: usize) -> Result<Buffer<f64>> {
    let step = PI / (oversample as f64 * r_max);
    let count = (k_max / step).ceil() + 1.;
    if !(count <= MAX_WAVEVECTORS) {
        return Err(invalid(format!(
            "the cutoff {k_max} needs {count:e} wave-vectors on a table reaching {r_max}"
        )));
    }
    let count = (count as usize).max(8);
    let mut k = ctx.allocate::<f64>(count)?;
    let step = k_max / (count - 1) as f64;
    for (i, x) in k.as_mut_slice().iter_mut().enumerate() {
        *x = i as f64 * step;
    }
    k.as_mut_slice()[count - 1] = k_max;
    Ok(k)
}

/// `F(k_i) = int values(r) j_l(k_i r) r^2 dr` for every `k_i`.
fn forward(ctx: &Context, l: usize, k: &[f64], r: &[f64], values: &[f64]) -> Result<Buffer<f64>> {
    let mut weighted = ctx.allocate::<f64>(r.len())?;
    for ((w, &v), &x) in weighted.as_mut_slice().iter_mut().zip(values).zip(r) {
        *w = v * (x * x);
    }
    let weighted = weighted.as_slice();
    let mut out = ctx.allocate::<f64>(k.len())?;
    let order = l as i32;
    parallel(ctx, out.as_mut_slice(), |start, part| {
        for (slot, &ki) in part.iter_mut().zip(&k[start..]) {
            *slot = trapezoid(|j| spherical_bessel(order, ki * r[j]) * weighted[j], r);
        }
    })?;
    Ok(out)
}

/// `(2/pi) int G(k) j_l(k r_j) dk` for every `r_j`, with `G = F k^2`.
fn inverse(ctx: &Context, l: usize, k: &[f64], g: &[f64], r: &[f64]) -> Result<Buffer<f64>> {
    let mut out = ctx.allocate::<f64>(r.len())?;
    let order = l as i32;
    parallel(ctx, out.as_mut_slice(), |start, part| {
        for (slot, &rj) in part.iter_mut().zip(&r[start..]) {
            *slot = (2. / PI) * trapezoid(|i| spherical_bessel(order, k[i] * rj) * g[i], k);
        }
    })?;
    Ok(out)
}

/// Fraction of `total` carried below `k_c`, from the transform `f` on the mesh `k`.
fn inside_fraction(k: &[f64], f: &[f64], total: f64) -> f64 {
    (2. / PI) * trapezoid(|i| f[i] * f[i] * k[i] * k[i], k) / total
}

/// Fraction of `int R^2 r^2 dr` carried by wave-vectors below `k_c`.
///
/// Computed from below, never by integrating a truncated high-`k` tail: the
/// quadrature of `j_l(kr)` on the tabulated radial mesh degrades once `k`
/// approaches `pi / dr`.
pub fn band_weight(
    ctx: &Context,
    r: &[f64],
    values: &[f64],
    l: usize,
    k_c: f64,
    oversample: usize,
) -> Result<f64> {
    check_table(r, values)?;
    check_cutoff(l, k_c, oversample)?;
    let (ra, va) = augment(ctx, r, values, l)?;
    let total = radial_norm(ra.as_slice(), va.as_slice());
    if total <= 0. {
        return Ok(1.);
    }
    let ra = ra.as_slice();
    let k = wavevectors(ctx, ra[ra.len() - 1], k_c, oversample)?;
    let f = forward(ctx, l, k.as_slice(), ra, va.as_slice())?;
    Ok(inside_fraction(k.as_slice(), f.as_slice(), total))
}

fn check_cutoff(l: usize, k_c: f64, oversample: usize) -> Result<()> {
    if l > MAX_L {
        return Err(invalid(format!(
            "the radial filter supports l <= {MAX_L}, got {l}"
        )));
    }
    if !(k_c.is_finite() && k_c > 0.) {
        return Err(invalid(format!(
            "the filter cutoff must be positive and finite, got {k_c}"
        )));
    }
    if oversample == 0 {
        return Err(invalid("the transform oversampling must be at least 1"));
    }
    Ok(())
}

/// Radius beyond which the function carries less than `tail` of its norm.
fn support_radius(r: &[f64], values: &[f64], tail: f64) -> f64 {
    let last = r[r.len() - 1];
    let density = |i: usize| values[i] * values[i] * r[i] * r[i];
    let total = trapezoid(density, r);
    if total <= 0. {
        return last;
    }
    let mut cumulative = 0.;
    for i in 0..r.len() {
        if i > 0 {
            cumulative += 0.5 * (density(i) + density(i - 1)) * (r[i] - r[i - 1]);
        }
        if (total - cumulative) / total <= tail {
            return r[i.max(1)];
        }
    }
    last
}

/// Raised-cosine mask: 1 below `inner`, 0 at and beyond `outer`.
pub fn mask_at(x: f64, inner: f64, outer: f64) -> f64 {
    if !(outer > inner) {
        return if x < outer { 1. } else { 0. };
    }
    if x >= outer {
        return 0.;
    }
    let s = ((x - inner) / (outer - inner)).clamp(0., 1.);
    0.5 * (1. + (PI * s).cos())
}

/// Raised-cosine window in `k`: 1 below `(1 - rolloff) k_c`, 0 at `k_c`.
pub fn window_at(k: f64, k_c: f64, rolloff: f64) -> f64 {
    if k >= k_c {
        return 0.;
    }
    let inner = (1. - rolloff) * k_c;
    if k_c > inner {
        let s = ((k - inner) / (k_c - inner)).clamp(0., 1.);
        0.5 * (1. + (PI * s).cos())
    } else {
        1.
    }
}

/// Remove the content above `k_c` from one radial table.
///
/// Returns the filtered values and the [`Report`]. The result is rescaled to the
/// original norm (not to 1) when `renormalize` is set, so the filter is the
/// identity in the `k_c -> infinity` limit.
pub fn filter(
    ctx: &Context,
    r: &[f64],
    values: &[f64],
    l: usize,
    k_c: f64,
    options: &Options,
) -> Result<(Vec<f64>, Report)> {
    check_table(r, values)?;
    check_cutoff(l, k_c, options.oversample)?;
    if !(options.rolloff.is_finite() && (0. ..=1.).contains(&options.rolloff)) {
        return Err(invalid("the roll-off fraction must lie in [0, 1]"));
    }
    if !(options.switch_margin.is_finite() && options.switch_margin > 0.) {
        return Err(invalid("the switch margin must be positive and finite"));
    }
    let (ra, va) = augment(ctx, r, values, l)?;
    let (ra, va) = (ra.as_slice(), va.as_slice());
    let n = ra.len();
    let r_end = ra[n - 1];
    let original_norm = radial_norm(ra, va);

    let r_support = support_radius(ra, va, SUPPORT_TAIL_NORM);
    let r_outer = (options.switch_margin * r_support).min(r_end);
    let r_inner = if options.method == Method::Mask {
        MASK_INNER_FRACTION * r_outer
    } else {
        r_support.min(r_outer)
    };
    let windowed = options.method != Method::Plain && r_outer < r_end;

    let k = wavevectors(ctx, r_end, k_c, options.oversample)?;
    let k = k.as_slice();
    let (before, mut spectrum) = {
        let f = forward(ctx, l, k, ra, va)?;
        let before = if original_norm <= 0. {
            0.
        } else {
            1. - inside_fraction(k, f.as_slice(), original_norm)
        };
        (before, f)
    };

    let mut mask = ctx.allocate::<f64>(if windowed { n } else { 0 })?;
    if windowed {
        for (m, &x) in mask.as_mut_slice().iter_mut().zip(ra) {
            *m = mask_at(x, r_inner, r_outer);
        }
        if options.method == Method::Mask {
            // Pre-compensate the multiplication that follows; the mask is zero
            // beyond the support, so the divisor is never small where the function
            // is not.
            let mut work = ctx.allocate::<f64>(n)?;
            for (i, w) in work.as_mut_slice().iter_mut().enumerate() {
                if ra[i] < r_outer && mask.as_slice()[i] > 0. {
                    *w = va[i] / mask.as_slice()[i];
                }
            }
            spectrum = forward(ctx, l, k, ra, work.as_slice())?;
        }
    }

    // F(k) times the window, times k^2 for the inverse measure.
    for (g, &ki) in spectrum.as_mut_slice().iter_mut().zip(k) {
        *g = *g * window_at(ki, k_c, options.rolloff) * ki * ki;
    }
    let mut filtered = inverse(ctx, l, k, spectrum.as_slice(), ra)?;
    drop(spectrum);
    if windowed {
        for (f, &m) in filtered.as_mut_slice().iter_mut().zip(mask.as_slice()) {
            *f *= m;
        }
    }
    drop(mask);

    let new_norm = radial_norm(ra, filtered.as_slice());
    let kept = if original_norm > 0. {
        new_norm / original_norm
    } else {
        1.
    };
    if options.renormalize && new_norm > 0. && original_norm > 0. {
        let scale = (original_norm / new_norm).sqrt();
        for f in filtered.as_mut_slice() {
            *f *= scale;
        }
    }
    let offset = n - r.len();
    let out = ctx.copy_slice(&filtered.as_slice()[offset..])?;
    drop(filtered);
    let (out, _held) = out.into_parts();

    let after = 1. - band_weight(ctx, r, &out, l, k_c, options.oversample)?;
    let scale = values.iter().fold(0., |a: f64, v| a.max(v.abs()));
    let scale = if scale > 0. { scale } else { 1. };
    let change = out
        .iter()
        .zip(values)
        .fold(0., |a: f64, (f, v)| a.max((f - v).abs()))
        / scale;
    let report = Report {
        residual_before: before,
        residual_after: after,
        support_radius: r_support,
        mask_radius: r_outer,
        masked: windowed,
        norm_kept: kept,
        max_change: change,
    };
    Ok((out, report))
}
