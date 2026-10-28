//! Smooth partial waves from constrained Bessel expansions.
use super::invalid;
use crate::math::radial::interpolate;
use crate::{math::constrained, product, Buffer, Context, Result};
use mandacaru_kernels::two_center::spherical_bessel as j;
use nalgebra::{DMatrix as M, DVector as V};
use std::f64::consts::PI;

pub struct SmoothWaves {
    pub wavevectors: Vec<f64>,
    pub coefficients: Buffer<f64>,
    pub norms: M<f64>,
    pub residuals: Vec<f64>,
}
fn bessel(l: usize, x: f64) -> [f64; 4] {
    let value = j(l as i32, x);
    let first = if l == 0 {
        -j(1, x)
    } else {
        j(l as i32 - 1, x) - (l + 1) as f64 * value / x
    };
    let ll = (l * (l + 1)) as f64;
    let second = -2. * first / x - (1. - ll / (x * x)) * value;
    let third = 2. * first / (x * x)
        - 2. * second / x
        - 2. * ll * value / x.powi(3)
        - (1. - ll / (x * x)) * first;
    [value, first, second, third]
}
pub fn wavevectors(l: usize, radius: f64, count: usize) -> Result<Vec<f64>> {
    if l > 8 || !radius.is_finite() || radius <= 0. || !(5..=32).contains(&count) {
        return Err(invalid("invalid PAW Bessel basis"));
    }
    let mut roots = vec![];
    for derivative in [false, true] {
        let f = |x| bessel(l, x)[usize::from(derivative)];
        let mut lower = 0.1;
        let mut last = f(lower);
        let mut found = 0;
        while lower < 400. && found < count {
            let upper = lower + 0.01;
            let next = f(upper);
            if last * next < 0. {
                let (mut a, mut b) = (lower, upper);
                let mut fa = last;
                for _ in 0..60 {
                    let c = 0.5 * (a + b);
                    if c == a || c == b {
                        break;
                    }
                    let fc = f(c);
                    if fa * fc <= 0. {
                        b = c;
                    } else {
                        a = c;
                        fa = fc;
                    }
                }
                roots.push(0.5 * (a + b) / radius);
                found += 1;
            }
            lower = upper;
            last = next;
        }
    }
    roots.sort_by(f64::total_cmp);
    if roots.len() < count {
        return Err(invalid("insufficient Bessel matching roots"));
    }
    roots.truncate(count);
    Ok(roots)
}
fn product_integral(l: usize, a: f64, b: f64, r: f64) -> Result<f64> {
    if (a - b).abs() < 1e-6 * a.abs().max(b.abs()).max(1.) {
        return Ok(crate::math::radial::gauss_legendre(64)?
            .iter()
            .map(|(x, w)| {
                let t = 0.5 * r * (x + 1.);
                0.5 * r * w * t * t * j(l as i32, a * t) * j(l as i32, b * t)
            })
            .sum());
    }
    Ok(r * r
        * (a * j(l as i32 + 1, a * r) * j(l as i32, b * r)
            - b * j(l as i32, a * r) * j(l as i32 + 1, b * r))
        / (a * a - b * b))
}
pub struct SmoothInput<'a> {
    pub r: &'a [f64],
    pub potential: &'a [f64],
    pub ae: &'a [f64],
    pub energies: &'a [f64],
    pub l: usize,
    pub radius: f64,
    pub count: usize,
    pub q_cut: f64,
    pub norm_factor: f64,
}
pub fn construct(ctx: &Context, input: &SmoothInput<'_>) -> Result<SmoothWaves> {
    let SmoothInput {
        r,
        potential,
        ae,
        energies,
        l,
        radius,
        count,
        q_cut,
        norm_factor,
    } = *input;
    crate::math::radial::validate(r, potential)?;
    let n = r.len();
    let waves = energies.len();
    if waves == 0
        || waves > 4
        || ae.len() != product(&[n, waves])?
        || !q_cut.is_finite()
        || q_cut <= 0.
        || !norm_factor.is_finite()
        || norm_factor <= 0.
        || norm_factor > 1.
        || radius <= r[2]
        || radius >= r[n - 3]
    {
        return Err(invalid("invalid PAW smooth-wave construction"));
    }
    crate::finite(ae, "reference partial waves")?;
    crate::finite(energies, "reference energies")?;
    let qs = wavevectors(l, radius, count)?;
    let _workspace = ctx.reserve(product(&[count, count, 8, 24])?)?;
    let mut norms = M::<f64>::zeros(waves, waves);
    for (x, w) in crate::math::radial::gauss_legendre(256)? {
        let x = 0.5 * radius * (x + 1.);
        let weight = 0.5 * radius * w * x * x;
        for i in 0..waves {
            for k in 0..waves {
                norms[(i, k)] += weight
                    * interpolate(r, &ae[i * n..(i + 1) * n], x)
                    * interpolate(r, &ae[k * n..(k + 1) * n], x);
            }
        }
    }
    let g = M::from_fn(count, count, |a, b| {
        product_integral(l, qs[a], qs[b], radius).unwrap()
    });
    let nq = 601;
    let dq = 0.1;
    let prefactor = (2. / PI).sqrt();
    let mut transform = ctx.allocate::<f64>(product(&[count, nq])?)?;
    for (a, _) in qs.iter().enumerate() {
        for k in 0..nq {
            transform.as_mut_slice()[a * nq + k] =
                prefactor * product_integral(l, qs[a], k as f64 * dq, radius)?;
        }
    }
    let weight = |index: usize| {
        let q = index as f64 * dq;
        if q >= q_cut {
            0.5 * q.powi(4) * dq
        } else {
            0.
        }
    };
    let k = M::from_fn(count, count, |a, b| {
        (0..nq)
            .map(|q| {
                weight(q) * transform.as_slice()[a * nq + q] * transform.as_slice()[b * nq + q]
            })
            .sum()
    });
    let mut coefficients = ctx.allocate::<f64>(product(&[waves, count])?)?;
    let mut residuals = vec![];
    for wave in 0..waves {
        let reference = &ae[wave * n..(wave + 1) * n];
        let mut u = ctx.allocate::<f64>(n)?;
        for i in 0..n {
            u.as_mut_slice()[i] = r[i] * reference[i];
        }
        let derivatives = crate::math::radial::derivatives(r, u.as_slice(), radius, 1, 3)?;
        let value = derivatives[0];
        let first = derivatives[1];
        let v = interpolate(r, potential, radius);
        let vp = crate::math::radial::derivatives(r, potential, radius, 1, 3)?[1];
        let centrifugal = (l * (l + 1)) as f64;
        let a = centrifugal / (radius * radius) + 2. * (v - energies[wave]);
        let second = a * value;
        let third = (-2. * centrifugal / radius.powi(3) + 2. * vp) * value + a * first;
        let target = [
            value / radius,
            first / radius - value / radius.powi(2),
            second / radius - 2. * first / radius.powi(2) + 2. * value / radius.powi(3),
            third / radius - 3. * second / radius.powi(2) + 6. * first / radius.powi(3)
                - 6. * value / radius.powi(4),
        ];
        let matrix = M::from_fn(4 + wave, count, |row, col| {
            if row < 4 {
                bessel(l, qs[col] * radius)[row] * qs[col].powi(row as i32)
            } else {
                (0..count)
                    .map(|a| g[(col, a)] * coefficients.as_slice()[(row - 4) * count + a])
                    .sum()
            }
        });
        let target = V::from_iterator(
            4 + wave,
            (0..4 + wave).map(|row| {
                if row < 4 {
                    target[row]
                } else {
                    norm_factor * norms[(wave, row - 4)]
                }
            }),
        );
        let bound = energies[wave] < 0. && (reference[n - 1] * r[n - 1]).abs() < 1e-6;
        let end = if bound {
            r[reference
                .iter()
                .zip(r)
                .rposition(|(p, r)| (p * r).abs() > 1e-10)
                .unwrap_or(n - 1)
                .saturating_add(1)
                .min(n - 1)]
        } else {
            (5. * radius).min(r[n - 1])
        };
        let step = (0.25 * (r[1] - r[0])).min(0.00125);
        let requested = ((end - radius) / step).ceil().max(2.);
        if !requested.is_finite() || requested >= (usize::MAX - 2) as f64 {
            return Err(crate::Error::Overflow);
        }
        let intervals = (requested as usize).next_multiple_of(2);
        let h = (end - radius) / intervals as f64;
        let mut tail = ctx.allocate::<f64>(intervals + 1)?;
        for i in 0..=intervals {
            let x = radius + i as f64 * h;
            let taper = if bound {
                1.
            } else {
                (0.5 * PI * ((x - 3. * radius) / (2. * radius)).clamp(0., 1.))
                    .cos()
                    .powi(2)
            };
            tail.as_mut_slice()[i] = interpolate(r, reference, x) * x * x * taper * h / 3.
                * if i == 0 || i == intervals {
                    1.
                } else if i % 2 == 0 {
                    2.
                } else {
                    4.
                };
        }
        let mut linear = V::zeros(count);
        let mut constant = 0.;
        for q in 0..nq {
            let reciprocal = q as f64 * dq;
            let value = prefactor
                * (0..=intervals)
                    .map(|i| tail.as_slice()[i] * j(l as i32, reciprocal * (radius + i as f64 * h)))
                    .sum::<f64>();
            let w = weight(q);
            constant += w * value * value;
            for (a, _) in qs.iter().enumerate() {
                linear[a] += w * value * transform.as_slice()[a * nq + q];
            }
        }
        let solution = constrained::minimum(
            &k,
            &linear,
            &matrix,
            &target,
            &g,
            norm_factor * norms[(wave, wave)],
        )?;
        coefficients.as_mut_slice()[wave * count..(wave + 1) * count]
            .copy_from_slice(solution.as_slice());
        residuals.push(solution.dot(&(&k * &solution)) + 2. * linear.dot(&solution) + constant);
    }
    Ok(SmoothWaves {
        wavevectors: qs,
        coefficients,
        norms,
        residuals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bessel_product_has_the_correct_analytic_s_wave_limit() {
        let r = 1.7;
        let a = 1.2;
        let expected = r * r * j(1, a * r) / a;
        assert!((product_integral(0, a, 0., r).unwrap() - expected).abs() < 1e-14);
        let roots = wavevectors(0, r, 8).unwrap();
        assert!((roots[0] * r - PI).abs() < 1e-13);
        assert!(roots.windows(2).all(|v| v[1] > v[0]));
    }
}
