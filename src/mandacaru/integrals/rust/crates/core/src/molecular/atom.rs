//! Nonrelativistic spherical LDA reference atoms, including density-metric DIIS.
use super::{basis::configuration, invalid, xc};
use crate::{kernels, product, Buffer, Context, Reservation, Result};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;
pub struct AtomOptions {
    pub points: usize,
    pub r_max: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub mixing: f64,
}
pub struct Atom {
    pub r: Buffer<f64>,
    pub density: Buffer<f64>,
    pub hartree: Buffer<f64>,
    pub xc: Buffer<f64>,
    pub effective: Buffer<f64>,
    pub orbitals: Vec<Buffer<f64>>,
    pub eigenvalues: Vec<f64>,
    pub occupations: Vec<(usize, usize, f64)>,
    pub energy: f64,
    pub converged: bool,
    pub iterations: usize,
    _memory: Reservation,
}
fn integrate(values: impl Iterator<Item = f64>, n: usize, h: f64) -> f64 {
    values
        .enumerate()
        .map(|(i, v)| v * if i == 0 || i == n - 1 { 0.5 } else { 1. })
        .sum::<f64>()
        * h
}
fn potentials(
    r: &[f64],
    density: &[f64],
    h: f64,
    hartree: &mut [f64],
    vxc: &mut [f64],
) -> Result<f64> {
    let n = r.len();
    let outer_total = integrate(r.iter().zip(density).map(|(r, d)| r * d), n, h);
    let mut inner = 0.;
    let mut outer = 0.;
    let mut exc = 0.;
    for i in 0..n {
        if i > 0 {
            inner += 0.5 * (density[i - 1] * r[i - 1].powi(2) + density[i] * r[i].powi(2)) * h;
            outer += 0.5 * (density[i - 1] * r[i - 1] + density[i] * r[i]) * h;
        }
        hartree[i] = 4. * PI * (inner / r[i] + outer_total - outer);
        let (f, df) = xc::partials_key("lda", [0.5 * density[i], 0.5 * density[i], 0., 0., 0.])?;
        vxc[i] = 0.5 * (df[0] + df[1]);
        exc += f * 4. * PI * r[i] * r[i] * h * if i == 0 || i == n - 1 { 0.5 } else { 1. };
    }
    Ok(exc)
}
pub fn solve(
    ctx: &Context,
    z: u8,
    options: &AtomOptions,
    configuration_spec: Option<&[(usize, usize, f64)]>,
    confinement: Option<&[f64]>,
) -> Result<Atom> {
    let n = options.points;
    let h = options.r_max / (n as f64 + 1.);
    if z == 0
        || z > 118
        || n < 4
        || n > i32::MAX as usize
        || !options.r_max.is_finite()
        || options.r_max <= 0.
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.
        || !options.mixing.is_finite()
        || options.mixing <= 0.
        || options.mixing > 1.
        || options.max_iterations == 0
    {
        return Err(invalid("invalid spherical atom or SCF options"));
    }
    if confinement.is_some_and(|v| v.len() != n || v.iter().any(|v| !v.is_finite())) {
        return Err(invalid(
            "confinement must be a finite array on the radial grid",
        ));
    }
    if configuration_spec.is_none() && configuration(z).iter().any(|s| s.1 == 3) {
        return Err(invalid("native f-block atoms require an explicit configuration; relaxed configuration selection is not migrated"));
    }
    let occupations = configuration_spec.map_or_else(
        || {
            configuration(z)
                .into_iter()
                .map(|(n, l, q)| (n, l, q as f64))
                .collect::<Vec<_>>()
        },
        |v| v.to_vec(),
    );
    if occupations.is_empty()
        || occupations.iter().any(|(nn, l, q)| {
            *nn == 0
                || *l >= *nn
                || *l > 8
                || nn - l > n
                || !q.is_finite()
                || *q <= 0.
                || *q > 2. * (2 * l + 1) as f64
        })
    {
        return Err(invalid("invalid atomic configuration"));
    }
    for (i, state) in occupations.iter().enumerate() {
        if occupations[..i]
            .iter()
            .any(|s| (s.0, s.1) == (state.0, state.1))
        {
            return Err(invalid("duplicate atomic subshell"));
        }
    }
    let electrons = occupations.iter().map(|s| s.2).sum::<f64>();
    let memory = ctx.reserve(product(&[occupations.len(), 128])?)?;
    let _workspace = ctx.reserve(product(&[n, 8, 40])?)?;
    let mut r = ctx.allocate::<f64>(n)?;
    let mut density = ctx.allocate::<f64>(n)?;
    let scale = (z as f64).cbrt().max(1.);
    for i in 0..n {
        r.as_mut_slice()[i] = (i + 1) as f64 * h;
        density.as_mut_slice()[i] =
            z as f64 * scale.powi(3) / PI * (-2. * scale * r.as_slice()[i]).exp();
    }
    let charge = integrate(
        density
            .as_slice()
            .iter()
            .zip(r.as_slice())
            .map(|(d, r)| 4. * PI * d * r * r),
        n,
        h,
    );
    for d in density.as_mut_slice() {
        *d *= electrons / charge;
    }
    let mut hartree = ctx.allocate::<f64>(n)?;
    let mut vxc = ctx.allocate::<f64>(n)?;
    let mut effective = ctx.allocate::<f64>(n)?;
    let mut diagonal = ctx.allocate::<f64>(n)?;
    let mut off = ctx.allocate::<f64>(n - 1)?;
    off.as_mut_slice().fill(-0.5 / (h * h));
    let mut history: Vec<(Vec<f64>, Vec<f64>)> = vec![];
    let mut orbitals = vec![];
    let mut eigenvalues = vec![];
    let mut converged = false;
    let mut iterations = 0;
    for iteration in 1..=options.max_iterations {
        potentials(
            r.as_slice(),
            density.as_slice(),
            h,
            hartree.as_mut_slice(),
            vxc.as_mut_slice(),
        )?;
        for i in 0..n {
            effective.as_mut_slice()[i] =
                -(z as f64) / r.as_slice()[i] + hartree.as_slice()[i] + vxc.as_slice()[i];
        }
        let mut next = ctx.allocate::<f64>(n)?;
        orbitals.clear();
        eigenvalues.clear();
        for &(nn, l, q) in &occupations {
            for i in 0..n {
                let radius = r.as_slice()[i];
                diagonal.as_mut_slice()[i] = 1. / (h * h)
                    + effective.as_slice()[i]
                    + (l * (l + 1)) as f64 / (2. * radius * radius)
                    + confinement.map_or(0., |v| v[i]);
            }
            let (e, mut u) =
                kernels::eigenpair(ctx, diagonal.as_slice(), off.as_slice(), nn - l - 1)?;
            let norm = integrate(u.as_slice().iter().map(|v| v * v), n, h)
                .sqrt()
                .copysign(u.as_slice()[0]);
            for i in 0..n {
                u.as_mut_slice()[i] /= norm;
                next.as_mut_slice()[i] +=
                    q * u.as_slice()[i].powi(2) / (4. * PI * r.as_slice()[i].powi(2));
            }
            orbitals.push(u);
            eigenvalues.push(e);
        }
        let residual = next
            .as_slice()
            .iter()
            .zip(density.as_slice())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let change = residual
            .iter()
            .zip(r.as_slice())
            .map(|(d, r)| d.abs().max((4. * PI * r * r * d).abs()))
            .fold(0f64, f64::max);
        let linear = density
            .as_slice()
            .iter()
            .zip(&residual)
            .map(|(d, r)| d + options.mixing * r)
            .collect::<Vec<_>>();
        if history.len() == 8 {
            history.remove(0);
        }
        history.push((density.as_slice().to_vec(), residual));
        let length = history.len();
        let mut b = DMatrix::<f64>::zeros(length + 1, length + 1);
        for i in 0..length {
            for j in 0..length {
                b[(i, j)] = (0..n)
                    .map(|k| {
                        4. * PI * r.as_slice()[k].powi(2) * h * history[i].1[k] * history[j].1[k]
                    })
                    .sum();
            }
            b[(i, length)] = 1.;
            b[(length, i)] = 1.;
        }
        let mut rhs = DVector::zeros(length + 1);
        rhs[length] = 1.;
        let weights = b
            .lu()
            .solve(&rhs)
            .filter(|w| w.iter().all(|v| v.is_finite()));
        let mixed = weights.map(|weights| {
            (0..n)
                .map(|k| {
                    (0..length)
                        .map(|i| weights[i] * (history[i].0[k] + options.mixing * history[i].1[k]))
                        .sum::<f64>()
                })
                .collect::<Vec<_>>()
        });
        if let Some(mixed) = mixed {
            let deficit = mixed
                .iter()
                .zip(r.as_slice())
                .map(|(d, r)| (-d).max(0.) * 4. * PI * r * r * h)
                .sum::<f64>();
            if deficit <= 1e-9 * electrons {
                for (out, value) in density.as_mut_slice().iter_mut().zip(mixed) {
                    *out = value.max(0.);
                }
            } else {
                density.as_mut_slice().copy_from_slice(&linear);
            }
        } else {
            density.as_mut_slice().copy_from_slice(&linear);
            let last = history.pop().unwrap();
            history.clear();
            history.push(last);
        }
        iterations = iteration;
        if change < options.tolerance {
            converged = true;
            break;
        }
    }
    let exc = potentials(
        r.as_slice(),
        density.as_slice(),
        h,
        hartree.as_mut_slice(),
        vxc.as_mut_slice(),
    )?;
    for i in 0..n {
        effective.as_mut_slice()[i] =
            -(z as f64) / r.as_slice()[i] + hartree.as_slice()[i] + vxc.as_slice()[i];
    }
    let band = eigenvalues
        .iter()
        .zip(&occupations)
        .map(|(e, s)| e * s.2)
        .sum::<f64>();
    let correction = integrate(
        (0..n).map(|i| {
            (0.5 * hartree.as_slice()[i] + vxc.as_slice()[i])
                * density.as_slice()[i]
                * 4.
                * PI
                * r.as_slice()[i].powi(2)
        }),
        n,
        h,
    );
    Ok(Atom {
        r,
        density,
        hartree,
        xc: vxc,
        effective,
        orbitals,
        eigenvalues,
        occupations,
        energy: band - correction + exc,
        converged,
        iterations,
        _memory: memory,
    })
}
