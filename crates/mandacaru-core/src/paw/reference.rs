//! Fourth-order radial reference waves in the converged atomic potential.
use super::invalid;
use crate::{kernels, Buffer, Context, Result};

fn derivative(values: &[f64], index: usize, h: f64) -> f64 {
    (values[index - 2] - 8. * values[index - 1] + 8. * values[index + 1] - values[index + 2])
        / (12. * h)
}
fn coefficients(
    ctx: &Context,
    r: &[f64],
    potential: &[f64],
    l: usize,
    energy: f64,
) -> Result<Buffer<f64>> {
    let mut f = ctx.allocate::<f64>(r.len() + 1)?;
    for i in 0..r.len() {
        f.as_mut_slice()[i + 1] =
            (l * (l + 1)) as f64 / r[i].powi(2) + 2. * (potential[i] - energy);
    }
    Ok(f)
}
fn seed(r: f64, l: usize, z: f64, constant: f64) -> f64 {
    let mut coefficients = [0.; 7];
    coefficients[0] = 1.;
    coefficients[1] = -z / (l + 1) as f64;
    for k in 2..=6 {
        coefficients[k] = (-2. * z * coefficients[k - 1] + 2. * constant * coefficients[k - 2])
            / (k * (k + 2 * l + 1)) as f64;
    }
    r.powi(l as i32 + 1) * coefficients.iter().rev().fold(0., |sum, a| sum * r + a)
}
pub fn scattering(
    ctx: &Context,
    r: &[f64],
    potential: &[f64],
    l: usize,
    energy: f64,
    z: f64,
) -> Result<Buffer<f64>> {
    crate::math::radial::validate(r, potential)?;
    let h = r[0];
    let f = coefficients(ctx, r, potential, l, energy)?;
    let mut initial = ctx.allocate::<f64>(r.len() + 1)?;
    let zero = ctx.allocate::<f64>(r.len() + 1)?;
    let constant = potential[0] + z / h - energy;
    initial.as_mut_slice()[1] = seed(h, l, z, constant);
    initial.as_mut_slice()[2] = seed(2. * h, l, z, constant);
    let output = kernels::numerov(
        ctx,
        f.as_slice(),
        zero.as_slice(),
        initial.as_slice(),
        h * h,
        1,
        false,
    )?;
    let mut u = ctx.allocate::<f64>(r.len())?;
    u.as_mut_slice().copy_from_slice(&output.as_slice()[1..]);
    Ok(u)
}
pub fn bound(
    ctx: &Context,
    r: &[f64],
    potential: &[f64],
    l: usize,
    guess: f64,
    z: f64,
    nodes: usize,
) -> Result<(Buffer<f64>, f64)> {
    crate::math::radial::validate(r, potential)?;
    if r.len() < 64 || !guess.is_finite() || guess >= 0. || l > 8 || r[0] <= 0. {
        return Err(invalid("invalid bound PAW reference"));
    }
    let h = r[0];
    if r.iter()
        .enumerate()
        .any(|(i, x)| (x - (i + 1) as f64 * h).abs() > 1e-10 * r[r.len() - 1])
    {
        return Err(invalid(
            "reference waves require a uniform radial grid starting one step from the origin",
        ));
    }
    type Match = (f64, Buffer<f64>, Buffer<f64>, usize, usize);
    let mismatch = |energy: f64| -> Result<Match> {
        if energy >= 0. {
            return Err(invalid("bound reference energy must be negative"));
        }
        let f = coefficients(ctx, r, potential, l, energy)?;
        let matching = f
            .as_slice()
            .iter()
            .enumerate()
            .skip(10)
            .filter(|(_, v)| **v < 0.)
            .map(|(i, _)| i)
            .next_back()
            .unwrap_or(r.len() / 2)
            .clamp(20, r.len() - 20);
        let decay = (-2. * energy).sqrt();
        let end = ((matching as f64 + 40. / (h * decay)).ceil() as usize)
            .min(r.len())
            .max(matching + 10);
        let constant = potential[0] + z / h - energy;
        let mut initial = ctx.allocate::<f64>(matching + 4)?;
        let zero = ctx.allocate::<f64>(matching + 4)?;
        initial.as_mut_slice()[1] = seed(h, l, z, constant);
        initial.as_mut_slice()[2] = seed(2. * h, l, z, constant);
        let out = kernels::numerov(
            ctx,
            &f.as_slice()[..matching + 4],
            zero.as_slice(),
            initial.as_slice(),
            h * h,
            1,
            false,
        )?;
        let mut initial = ctx.allocate::<f64>(end + 1)?;
        let zero = ctx.allocate::<f64>(end + 1)?;
        initial.as_mut_slice()[end] = 1e-100;
        initial.as_mut_slice()[end - 1] = 1e-100 * (h * decay).exp();
        let mut inn = kernels::numerov(
            ctx,
            &f.as_slice()[..end + 1],
            zero.as_slice(),
            initial.as_slice(),
            h * h,
            matching - 3,
            true,
        )?;
        let scale = out.as_slice()[matching] / inn.as_slice()[matching];
        for value in inn.as_mut_slice() {
            *value *= scale;
        }
        let residual = (derivative(out.as_slice(), matching, h)
            - derivative(inn.as_slice(), matching, h))
            / out.as_slice()[matching];
        if !residual.is_finite() {
            return Err(invalid("nonfinite PAW bound-state matching residual"));
        }
        Ok((residual, out, inn, matching, end))
    };
    let window = 0.005 * guess.abs().max(1.);
    let mut lo = guess - window;
    let mut hi = (guess + window).min(-1e-8);
    let mut flo = mismatch(lo)?.0;
    let mut fhi = mismatch(hi)?.0;
    for _ in 0..12 {
        if flo * fhi < 0. {
            break;
        }
        lo -= 2. * window;
        hi = (hi + 2. * window).min(-1e-8);
        flo = mismatch(lo)?.0;
        fhi = mismatch(hi)?.0;
    }
    if flo * fhi >= 0. {
        return Err(invalid("could not bracket the PAW bound reference"));
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let fm = mismatch(mid)?.0;
        if fm.abs() < 1e-12 || (hi - lo).abs() < 1e-13 {
            lo = mid;
            hi = mid;
            break;
        }
        if flo * fm <= 0. {
            hi = mid;
        } else {
            lo = mid;
            flo = fm;
        }
    }
    let energy = 0.5 * (lo + hi);
    let (residual, out, inn, matching, end) = mismatch(energy)?;
    if residual.abs() > 1e-6 {
        return Err(invalid(
            "PAW reference matching converged to a pole instead of a bound state",
        ));
    }
    let mut u = ctx.allocate::<f64>(r.len())?;
    for i in 1..=r.len() {
        u.as_mut_slice()[i - 1] = if i < matching {
            out.as_slice()[i]
        } else if i <= end {
            inn.as_slice()[i]
        } else {
            inn.as_slice()[end] * (-(-2. * energy).sqrt() * (i - end) as f64 * h).exp()
        };
    }
    let mut squares = ctx.allocate::<f64>(r.len())?;
    for i in 0..r.len() {
        squares.as_mut_slice()[i] = u.as_slice()[i].powi(2);
    }
    let norm = crate::math::radial::integrate(r, squares.as_slice())?
        .sqrt()
        .copysign(u.as_slice()[..20].iter().sum::<f64>());
    for value in u.as_mut_slice() {
        *value /= norm;
    }
    let count = u.as_slice().windows(2).filter(|w| w[0] * w[1] < 0.).count();
    if count != nodes {
        return Err(invalid(format!(
            "PAW reference has {count} nodes, expected {nodes}"
        )));
    }
    Ok((u, energy))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hydrogenic_reference_obeys_the_coulomb_spectrum() {
        let ctx = Context::new(Some(1), 32 << 20).unwrap();
        let r = (1..6001).map(|i| i as f64 * 0.005).collect::<Vec<_>>();
        let v = r.iter().map(|r| -1. / r).collect::<Vec<_>>();
        let (u, e) = bound(&ctx, &r, &v, 0, -0.499, 1., 0).unwrap();
        assert!((e + 0.5).abs() < 1e-7);
        assert!((u.as_slice()[199] - 2. * (-1f64).exp()).abs() < 1e-6);
        let (_, e) = bound(&ctx, &r, &v, 1, -0.124, 1., 0).unwrap();
        assert!((e + 0.125).abs() < 1e-7);
    }
}
