//! Dual PAW projectors, overlap correction and screened one-center operators.
use super::invalid;
use crate::{finite, math::radial::interpolate, product, Buffer, Context, Result};
use mandacaru_kernels::two_center::spherical_bessel;
use nalgebra::DMatrix;

pub struct Channel {
    pub pseudo: Buffer<f64>,
    pub projectors: Buffer<f64>,
    pub overlap: DMatrix<f64>,
    pub screened: DMatrix<f64>,
    pub kinetic: DMatrix<f64>,
    pub potential: DMatrix<f64>,
    pub duality_error: f64,
    pub asymmetry: f64,
}

pub struct ChannelInput<'a> {
    pub r: &'a [f64],
    pub ae: &'a [f64],
    pub vae: &'a [f64],
    pub local: &'a [f64],
    pub l: usize,
    pub radius: f64,
    pub local_radius: f64,
    pub energies: &'a [f64],
    pub wavevectors: &'a [f64],
    pub coefficients: &'a [f64],
}

pub fn construct(ctx: &Context, input: &ChannelInput<'_>) -> Result<Channel> {
    let ChannelInput {
        r,
        ae,
        vae,
        local,
        l,
        radius,
        local_radius,
        energies,
        wavevectors,
        coefficients,
    } = *input;
    crate::math::radial::validate(r, vae)?;
    crate::math::radial::validate(r, local)?;
    let n = r.len();
    let count = energies.len();
    let basis = wavevectors.len();
    if count == 0
        || count > 8
        || !(4..=64).contains(&basis)
        || l > 8
        || ae.len() != product(&[count, n])?
        || coefficients.len() != product(&[count, basis])?
        || !radius.is_finite()
        || radius <= r[0]
        || radius > r[n - 1]
        || !local_radius.is_finite()
        || local_radius < 0.
        || local_radius > r[n - 1]
        || wavevectors.iter().any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err(invalid(
            "invalid PAW channel dimensions, radii or wave vectors",
        ));
    }
    finite(ae, "all-electron partial waves")?;
    finite(energies, "reference energies")?;
    finite(coefficients, "Bessel coefficients")?;
    let _workspace = ctx.reserve(product(&[count, count, 8, 24])?)?;
    let mut b = DMatrix::<f64>::zeros(count, count);
    let mut q = b.clone();
    let mut dt = b.clone();
    let mut dv = b.clone();
    let psi = |wave: usize, x: f64| -> (f64, f64) {
        if x > radius {
            return (interpolate(r, &ae[wave * n..(wave + 1) * n], x), 0.);
        }
        let mut value = 0.;
        let mut kinetic = 0.;
        for k in 0..basis {
            let term =
                coefficients[wave * basis + k] * spherical_bessel(l as i32, wavevectors[k] * x);
            value += term;
            kinetic += 0.5 * wavevectors[k].powi(2) * term;
        }
        (value, kinetic)
    };
    for (lower, upper, is_shell) in [(0., radius, false), (radius, local_radius, true)] {
        if upper <= lower {
            continue;
        }
        for (node, weight) in crate::math::radial::gauss_legendre(256)? {
            let x = lower + 0.5 * (upper - lower) * (node + 1.);
            let w = 0.5 * (upper - lower) * weight * x * x;
            let v = interpolate(r, vae, x);
            let vl = interpolate(r, local, x);
            let av = (0..count)
                .map(|i| interpolate(r, &ae[i * n..(i + 1) * n], x))
                .collect::<Vec<_>>();
            let smooth = (0..count).map(|i| psi(i, x)).collect::<Vec<_>>();
            for i in 0..count {
                for j in 0..count {
                    let (pi, _) = smooth[i];
                    let (pj, tj) = smooth[j];
                    if is_shell {
                        let term = av[i] * (v - vl) * av[j] * w;
                        b[(i, j)] += term;
                        dv[(i, j)] += term;
                    } else {
                        b[(i, j)] += pi * ((energies[j] - vl) * pj - tj) * w;
                        q[(i, j)] += (av[i] * av[j] - pi * pj) * w;
                        dt[(i, j)] += (av[i] * (energies[j] - v) * av[j] - pi * tj) * w;
                        dv[(i, j)] += (av[i] * v * av[j] - pi * vl * pj) * w;
                    }
                }
            }
        }
    }
    let inverse = b
        .clone()
        .try_inverse()
        .ok_or_else(|| invalid("PAW projector overlap matrix is singular"))?;
    let raw = DMatrix::from_fn(count, count, |i, j| b[(i, j)] + q[(i, j)] * energies[j]);
    let asymmetry = (&raw - raw.transpose()).amax();
    if !asymmetry.is_finite() || asymmetry > 5e-5 {
        return Err(invalid(format!("PAW screened coupling is asymmetric by {asymmetry:e}; partial waves or matching are inconsistent")));
    }
    let duality_error = (&b * &inverse - DMatrix::identity(count, count)).amax();
    if duality_error > 1e-8 {
        return Err(invalid("PAW projectors fail their duality check"));
    }
    let mut pseudo = ctx.allocate::<f64>(product(&[count, n])?)?;
    let mut projectors = ctx.allocate::<f64>(product(&[count, n])?)?;
    for k in 0..n {
        let mut chi = vec![0.; count];
        for i in 0..count {
            let (value, kinetic) = psi(i, r[k]);
            pseudo.as_mut_slice()[i * n + k] = value;
            chi[i] = if r[k] <= radius {
                (energies[i] - local[k]) * value - kinetic
            } else if r[k] <= local_radius {
                (vae[k] - local[k]) * ae[i * n + k]
            } else {
                0.
            };
        }
        for i in 0..count {
            projectors.as_mut_slice()[i * n + k] =
                (0..count).map(|j| inverse[(j, i)] * chi[j]).sum();
        }
    }
    let sym = |a: DMatrix<f64>| 0.5 * (&a + a.transpose());
    Ok(Channel {
        pseudo,
        projectors,
        overlap: q,
        screened: sym(raw),
        kinetic: sym(dt),
        potential: sym(dv),
        duality_error,
        asymmetry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn separable_operator_reproduces_a_free_reference_wave() {
        let ctx = Context::new(Some(1), 1 << 20).unwrap();
        let r = (1..1001).map(|i| i as f64 * 0.005).collect::<Vec<_>>();
        let ae = r
            .iter()
            .map(|x| spherical_bessel(0, *x))
            .collect::<Vec<_>>();
        let zero = vec![0.; r.len()];
        let local = r
            .iter()
            .map(|x| if *x <= 2. { 0.3 } else { 0. })
            .collect::<Vec<_>>();
        let input = ChannelInput {
            r: &r,
            ae: &ae,
            vae: &zero,
            local: &local,
            l: 0,
            radius: 2.,
            local_radius: 2.,
            energies: &[0.5],
            wavevectors: &[1., 2., 3., 4.],
            coefficients: &[1., 0., 0., 0.],
        };
        let out = construct(&ctx, &input).unwrap();
        let norm = 1. - 4f64.sin() / 4.;
        // Interpolation at the discontinuity of the deliberately stepped local potential
        // is excluded from an exact normalization check by using its native quadrature.
        assert!(out.duality_error < 1e-12);
        assert!(out.overlap[(0, 0)].abs() < 1e-8);
        assert!((out.screened[(0, 0)] + 0.3 * norm).abs() < 3e-4);
        let k = 100;
        assert!((out.projectors.as_slice()[k] * out.screened[(0, 0)] + 0.3 * ae[k]).abs() < 1e-12);
        drop(out);
        assert_eq!(ctx.memory_used(), 0);
    }
    #[test]
    fn singular_projector_basis_is_refused_without_leaking_memory() {
        let ctx = Context::new(Some(1), 1 << 20).unwrap();
        let r = (1..501).map(|i| i as f64 * 0.01).collect::<Vec<_>>();
        let ae = r
            .iter()
            .map(|x| spherical_bessel(0, *x))
            .collect::<Vec<_>>();
        let zero = vec![0.; r.len()];
        let input = ChannelInput {
            r: &r,
            ae: &ae,
            vae: &zero,
            local: &zero,
            l: 0,
            radius: 2.,
            local_radius: 2.,
            energies: &[0.5],
            wavevectors: &[1., 2., 3., 4.],
            coefficients: &[1., 0., 0., 0.],
        };
        assert!(construct(&ctx, &input).is_err());
        assert_eq!(ctx.memory_used(), 0);
    }
}
