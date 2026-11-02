//! Checked uniform-grid radial eigensolves and not-a-knot cubic interpolation.
use super::invalid;
use crate::{kernels, math::spline::Cubic, product, Buffer, Context, Result};

#[derive(Debug)]
pub struct Table {
    pub step: f64,
    end: f64,
    pub values: Buffer<f64>,
    second: Buffer<f64>,
    pub energy: f64,
    pub kinetic: f64,
}
impl Table {
    pub fn new(
        ctx: &Context,
        step: f64,
        values: Buffer<f64>,
        energy: f64,
        l: usize,
    ) -> Result<Self> {
        let end = (values.as_slice().len().saturating_sub(1)) as f64 * step;
        Self::with_radius(ctx, step, values, energy, l, end)
    }
    /// The final wall knot need not be an integer multiple of the radial step.
    pub fn with_radius(
        ctx: &Context,
        step: f64,
        values: Buffer<f64>,
        energy: f64,
        l: usize,
        end: f64,
    ) -> Result<Self> {
        let n = values.as_slice().len();
        if n < 4
            || !step.is_finite()
            || step <= 0.
            || !end.is_finite()
            || end <= (n.saturating_sub(2)) as f64 * step
            || values.as_slice().iter().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid radial spline table"));
        }
        let knot = |i: usize| if i == n - 1 { end } else { i as f64 * step };
        let mut second = ctx.allocate::<f64>(n)?;
        let mut upper = ctx.allocate::<f64>(n)?;
        let y = values.as_slice();
        // Eliminate the two not-a-knot boundary equations before Thomas solve.
        for i in 1..n - 1 {
            let left = knot(i) - knot(i - 1);
            let right = knot(i + 1) - knot(i);
            let (lower, diagonal, up) = if i == 1 {
                (
                    0.,
                    left * (1. + left / right) + 2. * (left + right),
                    right - left * left / right,
                )
            } else if i == n - 2 {
                (
                    left - right * right / left,
                    2. * (left + right) + right * (1. + right / left),
                    0.,
                )
            } else {
                (left, 2. * (left + right), right)
            };
            let rhs = 6. * ((y[i + 1] - y[i]) / right - (y[i] - y[i - 1]) / left);
            let pivot = diagonal - lower * upper.as_slice()[i - 1];
            upper.as_mut_slice()[i] = up / pivot;
            second.as_mut_slice()[i] = (rhs - lower * second.as_slice()[i - 1]) / pivot;
        }
        for i in (1..n - 2).rev() {
            second.as_mut_slice()[i] -= upper.as_slice()[i] * second.as_slice()[i + 1];
        }
        second.as_mut_slice()[0] = 2. * second.as_slice()[1] - second.as_slice()[2];
        let ratio = (end - knot(n - 2)) / step;
        second.as_mut_slice()[n - 1] =
            (1. + ratio) * second.as_slice()[n - 2] - ratio * second.as_slice()[n - 3];
        let mut kinetic = 0.;
        for i in 1..n - 1 {
            let r = i as f64 * step;
            let u = r * y[i];
            let left = (r - step) * y[i - 1];
            let hr = knot(i + 1) - r;
            let right = knot(i + 1) * y[i + 1];
            kinetic += u
                * (-((right - u) / hr - (u - left) / step) / (step + hr)
                    + l as f64 * (l + 1) as f64 / (2. * r * r) * u)
                * step;
        }
        Ok(Self {
            step,
            end,
            values,
            second,
            energy,
            kinetic,
        })
    }
    pub fn radius(&self) -> f64 {
        self.end
    }
    pub fn value_slope(&self, r: f64) -> (f64, f64) {
        if r >= self.radius() || !r.is_finite() || r < 0. {
            return (0., 0.);
        }
        let i = ((r / self.step).floor() as usize).min(self.values.as_slice().len() - 2);
        let h = if i == self.values.as_slice().len() - 2 {
            self.end - i as f64 * self.step
        } else {
            self.step
        };
        let b = (r - i as f64 * self.step) / h;
        let a = 1. - b;
        let y = self.values.as_slice();
        let m = self.second.as_slice();
        (
            a * y[i]
                + b * y[i + 1]
                + ((a * a * a - a) * m[i] + (b * b * b - b) * m[i + 1]) * h * h / 6.,
            (y[i + 1] - y[i]) / h
                + ((-3. * a * a + 1.) * m[i] + (3. * b * b - 1.) * m[i + 1]) * h / 6.,
        )
    }
}
pub fn confined(
    ctx: &Context,
    n: usize,
    l: usize,
    z: f64,
    radius: f64,
    intervals: usize,
) -> Result<Table> {
    if n == 0
        || n >= intervals
        || intervals > i32::MAX as usize
        || l >= n
        || l > 8
        || !z.is_finite()
        || z <= 0.
        || !radius.is_finite()
        || radius <= 0.
        || intervals < 4
        || intervals <= n - l + 1
    {
        return Err(invalid(
            "invalid confined radial state, charge, radius or grid",
        ));
    }
    let step = radius / intervals as f64;
    let mut diagonal = ctx.allocate::<f64>(intervals - 1)?;
    let mut off = ctx.allocate::<f64>(intervals - 2)?;
    for i in 0..intervals - 1 {
        let r = (i + 1) as f64 * step;
        diagonal.as_mut_slice()[i] =
            1. / (step * step) - z / r + (l * (l + 1)) as f64 / (2. * r * r);
    }
    off.as_mut_slice().fill(-0.5 / (step * step));
    let (energy, u) = kernels::eigenpair(ctx, diagonal.as_slice(), off.as_slice(), n - l - 1)?;
    let norm = (u
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v * v
                * if i == 0 || i == intervals - 2 {
                    0.5
                } else {
                    1.
                }
        })
        .sum::<f64>()
        * step)
        .sqrt()
        .copysign(u.as_slice()[0]);
    let mut values =
        ctx.allocate::<f64>(intervals.checked_add(1).ok_or(crate::Error::Overflow)?)?;
    for i in 1..intervals {
        values.as_mut_slice()[i] = u.as_slice()[i - 1] / (norm * i as f64 * step);
    }
    values.as_mut_slice()[0] = 2. * values.as_slice()[1] - values.as_slice()[2];
    Table::new(ctx, step, values, energy, l)
}
/// The extra zetas of a first-zeta table `(r, first)` by the split-valence
/// construction; `r` is any increasing grid with `r[0] >= 0` (a PAW table starts
/// at its step, not at zero). The result holds a copy of `first` followed by up to
/// `count - 1` normalized refinements, one fewer when the split leaves nothing.
///
/// Without `tail_norms` each zeta splits the previous one at the radius leaving
/// `split_norm` of its squared norm outside, halved for every further zeta. With
/// them every zeta splits the first one, leaving `tail_norms[k]^2` outside.
pub fn zeta_values(
    ctx: &Context,
    r: &[f64],
    first: &[f64],
    l: usize,
    count: usize,
    split_norm: f64,
    tail_norms: Option<&[f64]>,
) -> Result<Vec<Buffer<f64>>> {
    let fractions = tail_norms.map(|v| v.iter().map(|x| x.powi(2)).collect::<Vec<_>>());
    zeta_values_by_fraction(ctx, r, first, l, count, split_norm, fractions.as_deref())
}
/// [`zeta_values`] with the first-zeta tails given as the squared-norm fractions
/// left outside the split radius (`tail_fractions[k]` rather than its square
/// root): the convention of the polarization zetas.
pub fn zeta_values_by_fraction(
    ctx: &Context,
    r: &[f64],
    first: &[f64],
    l: usize,
    count: usize,
    split_norm: f64,
    tail_fractions: Option<&[f64]>,
) -> Result<Vec<Buffer<f64>>> {
    if count == 0
        || count > 4
        || !split_norm.is_finite()
        || !(0. ..1.).contains(&split_norm)
        || split_norm == 0.
        || tail_fractions.is_some_and(|v| {
            v.len() < count - 1 || v.iter().any(|x| !x.is_finite() || *x <= 0. || *x >= 1.)
        })
    {
        return Err(invalid("invalid zeta count or tail norms"));
    }
    let n = r.len();
    if n < 4 || first.len() != n || r[0] < 0. || r.windows(2).any(|w| w[1] <= w[0]) {
        return Err(invalid(
            "the zeta split needs matching values on an increasing grid from r >= 0",
        ));
    }
    crate::finite(r, "zeta grid")?;
    crate::finite(first, "zeta values")?;
    let l_i = i32::try_from(l).map_err(|_| invalid("angular momentum too large"))?;
    let trapezoid = |f: &[f64]| {
        (1..n)
            .map(|i| 0.5 * (f[i - 1] + f[i]) * (r[i] - r[i - 1]))
            .sum::<f64>()
    };
    let _work = ctx.reserve(product(&[n, 24])?)?;
    let mut out = vec![ctx.copy_slice(first)?];
    let mut fraction = split_norm;
    for zeta in 1..count {
        let previous = out[if tail_fractions.is_some() {
            0
        } else {
            out.len() - 1
        }]
        .as_slice();
        if let Some(tails) = tail_fractions {
            fraction = tails[zeta - 1];
        }
        let density = (0..n)
            .map(|i| previous[i].powi(2) * r[i].powi(2))
            .collect::<Vec<_>>();
        let total = trapezoid(&density);
        if total <= 0. {
            return Err(invalid("zero-norm radial function"));
        }
        let mut inner = 0.;
        let mut best = 0;
        let mut error = f64::INFINITY;
        for i in 0..n {
            if i > 0 {
                inner += 0.5 * (density[i - 1] + density[i]) * (r[i] - r[i - 1]);
            }
            let diff = ((total - inner) / total - fraction).abs();
            if diff < error {
                error = diff;
                best = i;
            }
        }
        let radius = r[best.max(1)];
        let spline = Cubic::new(ctx, ctx.copy_slice(r)?, ctx.copy_slice(previous)?)?;
        let (value, slope) = spline.value_slope(radius);
        let b = (l as f64 * value / radius - slope) / (2. * radius.powi(l_i + 1));
        let a = value / radius.powi(l_i) + b * radius * radius;
        let mut values = ctx.allocate::<f64>(n)?;
        for i in 0..n {
            if r[i] < radius {
                values.as_mut_slice()[i] = previous[i] - r[i].powi(l_i) * (a - b * r[i] * r[i]);
            }
        }
        if !values.as_slice().iter().any(|v| v.abs() > 1e-12) {
            break;
        }
        let weight = (0..n)
            .map(|i| values.as_slice()[i].powi(2) * r[i].powi(2))
            .collect::<Vec<_>>();
        let norm = trapezoid(&weight).sqrt();
        for value in values.as_mut_slice() {
            *value /= norm;
        }
        out.push(values);
        fraction *= 0.5;
    }
    Ok(out)
}
/// [`zeta_values`] on the uniform grid `i * step` of a table, as spline tables.
pub fn zetas(
    ctx: &Context,
    first: Table,
    l: usize,
    count: usize,
    split_norm: f64,
    tail_norms: Option<&[f64]>,
) -> Result<Vec<std::sync::Arc<Table>>> {
    let n = first.values.as_slice().len();
    let h = first.step;
    if first.radius() != (n - 1) as f64 * h {
        return Err(invalid("the zeta split needs a table on a uniform grid"));
    }
    let r = (0..n).map(|i| i as f64 * h).collect::<Vec<_>>();
    let refined = zeta_values(
        ctx,
        &r,
        first.values.as_slice(),
        l,
        count,
        split_norm,
        tail_norms,
    )?;
    let mut out = vec![std::sync::Arc::new(first)];
    for values in refined.into_iter().skip(1) {
        out.push(std::sync::Arc::new(Table::new(ctx, h, values, 0., l)?));
    }
    Ok(out)
}
