//! PAW local potentials, compact compensation charges and radial quadrature.
use super::invalid;
use crate::math::radial::{derivatives, gauss_legendre, validate};
use crate::{
    finite,
    models::{spherical::hartree, xc},
    Buffer, Context, Result,
};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

pub fn shape(r: f64, radius: f64) -> f64 {
    if r >= radius {
        0.
    } else {
        315. / (64. * PI * radius.powi(3)) * (1. - (r / radius).powi(2)).powi(3)
    }
}
pub fn compensation_potential(r: f64, radius: f64) -> f64 {
    if r >= radius {
        return 1. / r;
    }
    let x = r / radius;
    let enclosed_over_x =
        315. / 16. * (x.powi(2) / 3. - 3. * x.powi(4) / 5. + 3. * x.powi(6) / 7. - x.powi(8) / 9.);
    (enclosed_over_x + 315. / 128. * (1. - x * x).powi(4)) / radius
}

pub fn compensation_coulomb(
    a: f64,
    b: f64,
    distance: f64,
    radial: usize,
    angular: usize,
) -> Result<f64> {
    if [a, b, distance].iter().any(|v| !v.is_finite()) || a <= 0. || b <= 0. || distance < 0. {
        return Err(invalid(
            "compensation radii must be positive and distance nonnegative",
        ));
    }
    if distance >= a + b {
        return Ok(1. / distance);
    }
    let rule = gauss_legendre(radial)?;
    let angles = gauss_legendre(angular)?;
    let result = rule
        .iter()
        .map(|(x, w)| {
            let r = 0.5 * b * (x + 1.);
            let angular = angles
                .iter()
                .map(|(mu, w)| {
                    w * compensation_potential(
                        (r * r + distance * distance + 2. * r * distance * mu)
                            .max(0.)
                            .sqrt(),
                        a,
                    )
                })
                .sum::<f64>();
            0.5 * b * w * r * r * shape(r, b) * angular
        })
        .sum::<f64>();
    Ok(2. * PI * result)
}

/// Smooth local continuation: match four derivatives and independently control v(0).
pub fn local_potential(
    ctx: &Context,
    r: &[f64],
    ae: &[f64],
    radius: f64,
    shift: f64,
) -> Result<Buffer<f64>> {
    validate(r, ae)?;
    if !radius.is_finite() || radius <= r[0] || radius >= r[r.len() - 1] || !shift.is_finite() {
        return Err(invalid("invalid local PAW radius or shift"));
    }
    let target = derivatives(r, ae, radius, 4, 25)?;
    let row = |derivative: usize, power: usize| {
        if power < derivative {
            0.
        } else {
            (0..derivative).map(|j| (power - j) as f64).product::<f64>()
        }
    };
    let first = DMatrix::from_fn(5, 5, |i, j| row(i, 2 * j));
    let scaled = DVector::from_iterator(
        5,
        target
            .iter()
            .enumerate()
            .map(|(i, v)| v * radius.powi(i as i32)),
    );
    let unconstrained = first
        .lu()
        .solve(&scaled)
        .ok_or_else(|| invalid("singular local-potential matching system"))?;
    let system = DMatrix::from_fn(6, 6, |i, j| {
        if i == 5 {
            if j == 0 {
                1.
            } else {
                0.
            }
        } else {
            row(i, 2 * j)
        }
    });
    let mut rhs = DVector::zeros(6);
    rhs.rows_mut(0, 5).copy_from(&scaled);
    rhs[5] = unconstrained[0] + shift;
    let coefficients = system
        .lu()
        .solve(&rhs)
        .ok_or_else(|| invalid("singular shifted-potential matching system"))?;
    let mut out = ctx.allocate::<f64>(r.len())?;
    for i in 0..r.len() {
        out.as_mut_slice()[i] = if r[i] <= radius {
            coefficients
                .iter()
                .rev()
                .fold(0., |value, c| value * (r[i] / radius).powi(2) + c)
        } else {
            ae[i]
        };
    }
    finite(out.as_slice(), "local PAW potential")?;
    Ok(out)
}

/// Remove Hartree and XC screening from a PAW local potential in atomic units.
pub fn unscreen(
    ctx: &Context,
    r: &[f64],
    screened: &[f64],
    smooth: &[f64],
    core: &[f64],
    radius: f64,
    charge: f64,
) -> Result<(Buffer<f64>, Buffer<f64>)> {
    validate(r, screened)?;
    validate(r, smooth)?;
    validate(r, core)?;
    if !radius.is_finite()
        || radius <= 0.
        || !charge.is_finite()
        || smooth.iter().chain(core).any(|v| *v < 0.)
    {
        return Err(invalid("invalid PAW unscreening densities or compensation"));
    }
    let mut augmented = ctx.allocate::<f64>(r.len())?;
    for i in 0..r.len() {
        augmented.as_mut_slice()[i] = smooth[i] + charge * shape(r[i], radius);
    }
    let vh = hartree(ctx, r, augmented.as_slice())?;
    let mut ionic = ctx.allocate::<f64>(r.len())?;
    for i in 0..r.len() {
        let density = 0.5 * (smooth[i] + core[i]);
        let (_, partial) =
            xc::point_partials("lda", [density, density, 0., 0., 0., 0., 0.], false)?;
        ionic.as_mut_slice()[i] = screened[i] - vh.as_slice()[i] - 0.5 * (partial[0] + partial[1]);
    }
    Ok((ionic, vh))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_charge_normalization_and_coulomb_tail() {
        let rule = gauss_legendre(64).unwrap();
        let radius = 1.7;
        let q = rule
            .iter()
            .map(|(x, w)| {
                let r = 0.5 * radius * (x + 1.);
                0.5 * radius * w * 4. * PI * r * r * shape(r, radius)
            })
            .sum::<f64>();
        assert!((q - 1.).abs() < 1e-13);
        assert!((compensation_potential(0., radius) - 315. / 128. / radius).abs() < 1e-14);
        assert_eq!(compensation_coulomb(1., 2., 4., 64, 32).unwrap(), 0.25);
        let ab = compensation_coulomb(1.1, 1.7, 0.8, 160, 80).unwrap();
        let ba = compensation_coulomb(1.7, 1.1, 0.8, 160, 80).unwrap();
        assert!((ab - ba).abs() < 2e-9);
    }
    #[test]
    fn polynomial_potential_matches_boundary_and_origin_shift() {
        let ctx = Context::new(Some(1), 1 << 20).unwrap();
        let r = (1..1001).map(|i| i as f64 * 0.005).collect::<Vec<_>>();
        let v = r.iter().map(|r| -3. / r).collect::<Vec<_>>();
        let a = local_potential(&ctx, &r, &v, 1.5, 0.).unwrap();
        let b = local_potential(&ctx, &r, &v, 1.5, 0.4).unwrap();
        assert_eq!(&a.as_slice()[300..], &v[300..]);
        for (i, _) in v.iter().enumerate().take(306).skip(290) {
            assert!((a.as_slice()[i] - v[i]).abs() < 1e-5);
        }
        assert!((b.as_slice()[0] - a.as_slice()[0] - 0.4).abs() < 3e-5);
    }
}
