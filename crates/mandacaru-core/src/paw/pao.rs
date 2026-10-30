//! The bound state of an empty channel in a PAW dataset's own operator: the
//! lowest level and the orbital of a channel the atom leaves empty.
use super::confinement::{ChannelData, Prepared};
use crate::{math::spline::Cubic, Context, Result};

/// Spacing (Bohr) of the finest grid of a bound-state solve.
pub const SPECTRUM_SPACING: f64 = 0.005;
/// Strides (in units of [`SPECTRUM_SPACING`]) of the two Richardson grids.
pub const STRIDES: [usize; 2] = [2, 4];

fn uniform(r_max: f64, stride: usize) -> Result<Vec<f64>> {
    let h = stride as f64 * SPECTRUM_SPACING;
    if !(r_max.is_finite() && r_max >= 2. * h) {
        return Err(super::invalid("the bound-state box is too small"));
    }
    Ok((1..=(r_max / h) as usize).map(|i| i as f64 * h).collect())
}

/// Lowest eigenvalue (Hartree) of the channel in the dataset's own operator,
/// Richardson-extrapolated from the two spectrum grids in a box `r_max`.
pub fn bound_level(ctx: &Context, data: ChannelData, r_max: f64) -> Result<f64> {
    let prepared = Prepared::new(ctx, data)?;
    let mut levels = [0.; 2];
    for (level, stride) in levels.iter_mut().zip(STRIDES) {
        let r = uniform(r_max, stride)?;
        *level = prepared.solve_on(ctx, &r, None, 0, false)?.0;
    }
    Ok((4. * levels[0] - levels[1]) / 3.)
}

/// `R(r)` on the dataset grid of the channel's lowest eigenstate in the
/// dataset's own operator, zero past the box it was solved in; with its level.
pub fn bound_radial(ctx: &Context, data: ChannelData, r_max: f64) -> Result<(Vec<f64>, f64)> {
    let prepared = Prepared::new(ctx, data)?;
    let mut solved = Vec::with_capacity(2);
    for stride in STRIDES {
        let r = uniform(r_max, stride)?;
        let (energy, u) = prepared.solve_on(ctx, &r, None, 0, true)?;
        let u = u.expect("vectors requested");
        let sign = prepared.sign_of(&r, u.as_slice());
        solved.push((r, u, sign, energy));
    }
    let (r_c, u_c, sign_c, e_c) = solved.pop().expect("two grids");
    let (r_f, u_f, sign_f, e_f) = solved.pop().expect("two grids");
    if r_f.len() < 2 * r_c.len() {
        return Err(super::invalid("the two bound-state grids do not nest"));
    }
    let mut knots = Vec::with_capacity(r_c.len() + 1);
    knots.push(0.);
    knots.extend_from_slice(&r_c);
    let mut values = Vec::with_capacity(r_c.len() + 1);
    values.push(0.);
    values.extend(
        (0..r_c.len())
            .map(|i| (4. * sign_f * u_f.as_slice()[2 * i + 1] - sign_c * u_c.as_slice()[i]) / 3.),
    );
    let curve = Cubic::new(ctx, ctx.copy_slice(&knots)?, ctx.copy_slice(&values)?)?;
    let end = *r_c.last().expect("nodes");
    let radial = data
        .table
        .iter()
        .map(|&x| {
            if x < end {
                curve.value_slope(x.min(end)).0 / x
            } else {
                0.
            }
        })
        .collect();
    Ok((radial, (4. * e_f - e_c) / 3.))
}
