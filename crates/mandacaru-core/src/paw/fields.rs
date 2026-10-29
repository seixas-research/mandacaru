//! Radial core models and bounded placement on molecular or periodic grids.
use super::invalid;
use crate::{
    finite,
    math::{spline::Cubic, SpatialGrid},
    product, Buffer, Context, Result,
};
use nalgebra::{Matrix3, Vector3};

pub struct RadialSource {
    pub knots: Buffer<f64>,
    pub values: Buffer<f64>,
    pub centers: Buffer<[f64; 3]>,
}
pub fn sample_many(
    ctx: &Context,
    grid: &SpatialGrid,
    sources: &[RadialSource],
    kinetic: bool,
) -> Result<Buffer<f64>> {
    ctx.run(|| {
        let mut total = ctx.allocate::<f64>(grid.index.points())?;
        for source in sources {
            let profile = Cubic::new(
                ctx,
                ctx.copy_slice(source.knots.as_slice())?,
                ctx.copy_slice(source.values.as_slice())?,
            )?;
            let profile = if kinetic {
                core_kinetic(ctx, &profile)?
            } else {
                profile
            };
            let field = place(ctx, grid, &profile, source.centers.as_slice())?;
            for (to, from) in total.as_mut_slice().iter_mut().zip(field.as_slice()) {
                *to += from;
            }
        }
        Ok(total)
    })
}

/// Radial von Weizsaecker kinetic density, with the same core-tail floor as
/// the molecular one-center model. The returned spline owns its tables.
pub fn core_kinetic(ctx: &Context, density: &Cubic) -> Result<Cubic> {
    let floor = density
        .values
        .as_slice()
        .iter()
        .map(|v| v.abs())
        .fold(0f64, f64::max)
        * 1e-12;
    let mut values = ctx.allocate::<f64>(density.knots.as_slice().len())?;
    for (to, &r) in values
        .as_mut_slice()
        .iter_mut()
        .zip(density.knots.as_slice())
    {
        let (value, slope) = density.value_slope(r);
        *to = if value > floor {
            slope * slope / (8. * value)
        } else {
            0.
        };
    }
    Cubic::new(ctx, ctx.copy_slice(density.knots.as_slice())?, values)
}

pub fn place(
    ctx: &Context,
    grid: &SpatialGrid,
    radial: &Cubic,
    centers: &[[f64; 3]],
) -> Result<Buffer<f64>> {
    if radial.knots.as_slice()[0] < 0. {
        return Err(invalid("radial field nodes must be nonnegative"));
    }
    for center in centers {
        finite(center, "radial field centers")?;
    }
    let peak = radial
        .values
        .as_slice()
        .iter()
        .map(|v| v.abs())
        .fold(0f64, f64::max);
    let support = radial
        .knots
        .as_slice()
        .iter()
        .zip(radial.values.as_slice())
        .filter_map(|(&r, &v)| (v.abs() > 1e-14 * peak).then_some(r))
        .next_back()
        .unwrap_or(0.);
    let shape = grid.index.shape();
    let origin = Vector3::from(grid.origin);
    let step_inverse = grid
        .step
        .try_inverse()
        .ok_or_else(|| invalid("singular radial placement grid"))?;
    let cell = grid.step
        * Matrix3::from_diagonal(&Vector3::from_fn(|a, _| {
            (shape[a] - usize::from(!grid.periodic)) as f64
        }));
    let inverse = if grid.periodic {
        cell.try_inverse()
            .ok_or_else(|| invalid("singular radial placement cell"))?
    } else {
        Matrix3::identity()
    };
    let midpoint = origin + cell * Vector3::repeat(0.5);
    let radius = (0..8)
        .map(|bits| {
            let corner = Vector3::from_fn(|a, _| if bits & (1 << a) == 0 { -0.5 } else { 0.5 });
            (cell * corner).norm()
        })
        .fold(0f64, f64::max);
    let mut limits = [0i32; 3];
    if grid.periodic {
        for (a, limit) in limits.iter_mut().enumerate() {
            let count = ((support + 2. * radius) * inverse.row(a).norm()).ceil() + 1.;
            if count > i32::MAX as f64 / 2. {
                return Err(crate::Error::Overflow);
            }
            *limit = count as i32;
        }
    }
    product(&limits.map(|v| 2 * v as usize + 1))?;
    let mut field = ctx.allocate::<f64>(grid.index.points())?;
    if peak == 0. {
        return Ok(field);
    }
    for center in centers {
        let center = if grid.periodic {
            let fractional = inverse * (Vector3::from(*center) - origin);
            origin + cell * fractional.map(|v| v - v.floor())
        } else {
            Vector3::from(*center)
        };
        for i in -limits[0]..=limits[0] {
            for j in -limits[1]..=limits[1] {
                for k in -limits[2]..=limits[2] {
                    let position = center + cell * Vector3::new(i as f64, j as f64, k as f64);
                    if (position - midpoint).norm() > support + radius {
                        continue;
                    }
                    let fractional = step_inverse * (position - origin);
                    let mut lower = [0usize; 3];
                    let mut upper = [0usize; 3];
                    let mut intersects = true;
                    for a in 0..3 {
                        let reach = support * step_inverse.row(a).norm();
                        let low = (fractional[a] - reach - 1e-12).ceil().max(0.);
                        let high = (fractional[a] + reach + 1e-12)
                            .floor()
                            .min((shape[a] - 1) as f64);
                        if high < low {
                            intersects = false;
                            break;
                        }
                        lower[a] = low as usize;
                        upper[a] = high as usize;
                    }
                    if !intersects {
                        continue;
                    }
                    for x in lower[0]..=upper[0] {
                        for y in lower[1]..=upper[1] {
                            for z in lower[2]..=upper[2] {
                                let at = (x * shape[1] + y) * shape[2] + z;
                                let distance = (Vector3::from(grid.point(at)) - position).norm();
                                if distance <= support {
                                    field.as_mut_slice()[at] += radial.value_slope(distance).0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(field)
}
