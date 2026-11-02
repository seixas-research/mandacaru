//! Bloechl's transformation of a smooth radial function back to the all-electron one.
use super::view::{Channel, Smooth};
use crate::{
    math::spline::{simpson, Cubic},
    paw::invalid,
    Context, Result,
};

fn spline(ctx: &Context, knots: &[f64], values: &[f64]) -> Result<Cubic> {
    Cubic::new(ctx, ctx.copy_slice(knots)?, ctx.copy_slice(values)?)
}

/// `u~ + sum_i (u_i - u~_i) <p~_i|u~>` for a smooth radial function `u = r R` given on the
/// nodes `r` (strictly increasing): the all-electron function it stands for. The dual projectors
/// and the partial waves are splined from the dataset's table; the projections are Simpson
/// integrals on `r`.
pub fn reconstruct_ae(
    ctx: &Context,
    smooth: &Smooth,
    channel: &Channel,
    r: &[f64],
    u: &[f64],
) -> Result<Vec<f64>> {
    if r.len() != u.len() || r.len() < 3 {
        return Err(invalid(
            "diagnose: reconstruct needs matching radial arrays",
        ));
    }
    let table = smooth.r.as_slice();
    let n = table.len();
    let mut out = u.to_vec();
    for k in 0..channel.rank {
        let projector = spline(
            ctx,
            table,
            &channel.projectors.as_slice()[k * n..(k + 1) * n],
        )?;
        let ae = spline(ctx, table, channel.ae_waves[k].as_slice())?;
        let pseudo = spline(ctx, table, channel.pseudo_waves[k].as_slice())?;
        let integrand = r
            .iter()
            .zip(u)
            .map(|(&x, &v)| projector.value_slope(x).0 * x * v)
            .collect::<Vec<_>>();
        let weight = simpson(r, &integrand)?;
        for (i, &x) in r.iter().enumerate() {
            out[i] += weight * (ae.value_slope(x).0 - pseudo.value_slope(x).0) * x;
        }
    }
    Ok(out)
}
