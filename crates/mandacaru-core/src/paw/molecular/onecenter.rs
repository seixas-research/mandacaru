//! Multipole moments and exact valence Coulomb/exchange corrections.
use super::*;
use crate::math::{linalg::real_eigen, spline::simpson};

#[derive(Clone, Copy)]
pub struct Projector {
    pub channel: usize,
    pub radial: usize,
    pub l: usize,
    pub m: i32,
}
pub fn layout(species: &Species) -> Vec<Projector> {
    let mut out = Vec::new();
    for (channel, ch) in species.channels.iter().enumerate() {
        for m in -(ch.l as i32)..=ch.l as i32 {
            for radial in 0..ch.projectors.len() {
                out.push(Projector {
                    channel,
                    radial,
                    l: ch.l,
                    m,
                });
            }
        }
    }
    out
}
pub fn harmonic(l: usize, m: i32, point: [f64; 3]) -> f64 {
    crate::math::angular::cartesian(l, m, point)
}
pub fn shape(r: f64, radius: f64, l: usize) -> f64 {
    if r >= radius {
        return 0.;
    }
    let l = l as i32;
    let bracket = 1. / (2 * l + 3) as f64 - 3. / (2 * l + 5) as f64 + 3. / (2 * l + 7) as f64
        - 1. / (2 * l + 9) as f64;
    r.powi(l) * (1. - (r / radius).powi(2)).powi(3) / (radius.powi(2 * l + 3) * bracket)
}
pub fn potential(r: f64, radius: f64, l: usize) -> f64 {
    let prefactor = 4. * PI / (2 * l + 1) as f64;
    if r >= radius {
        return prefactor / r.powi((l + 1) as i32);
    }
    let x = r / radius;
    let norm = 1.
        / (radius.powi((2 * l + 3) as i32)
            * (1. / (2 * l + 3) as f64 - 3. / (2 * l + 5) as f64 + 3. / (2 * l + 7) as f64
                - 1. / (2 * l + 9) as f64));
    let mut enclosed = 0.;
    let mut tail = 0.;
    for (k, c) in [1., -3., 3., -1.].into_iter().enumerate() {
        let power = (2 * l + 3 + 2 * k) as i32;
        enclosed += c * x.powi(power) / power as f64;
        tail += c * (1. - x.powi((2 * k + 2) as i32)) / (2 * k + 2) as f64;
    }
    prefactor
        * norm
        * (if r == 0. {
            0.
        } else {
            radius.powi((2 * l + 3) as i32) * enclosed / r.powi((l + 1) as i32)
        } + r.powi(l as i32) * radius * radius * tail)
}
pub fn angular_rule(polar: usize, azimuthal: usize) -> Result<Vec<([f64; 3], f64)>> {
    let mut out = Vec::new();
    for (mu, w) in crate::math::radial::gauss_legendre(polar)? {
        for k in 0..azimuthal {
            let phi = 2. * PI * (k as f64 + 0.5) / azimuthal as f64;
            let sin = (1. - mu * mu).max(0.).sqrt();
            out.push((
                [sin * phi.cos(), sin * phi.sin(), mu],
                w * 2. * PI / azimuthal as f64,
            ));
        }
    }
    Ok(out)
}
pub struct Multipole {
    pub l: usize,
    pub m: i32,
    pub block: DMatrix<f64>,
}
pub struct AtomTerms {
    pub projectors: Vec<Projector>,
    pub coupling: DMatrix<f64>,
    pub ks_difference: DMatrix<f64>,
    pub overlap: DMatrix<f64>,
    pub multipoles: Vec<Multipole>,
    pub factors: Vec<DMatrix<f64>>,
    pub weights: Vec<f64>,
    pub _memory: crate::Reservation,
}
fn linear(x: &[f64], y: &[f64], at: f64) -> f64 {
    let i = x.partition_point(|v| *v <= at);
    if i == 0 {
        y[0]
    } else if i == x.len() {
        y[x.len() - 1]
    } else {
        y[i - 1] + (y[i] - y[i - 1]) * (at - x[i - 1]) / (x[i] - x[i - 1])
    }
}
pub(super) fn coulomb(r: &[f64], a: &[f64], b: &[f64], l: usize) -> f64 {
    // SAFETY: callers supply equal-length live, finite slices, of at least
    // four nodes and a count checked by the radial input boundary to fit i32;
    // the raw routine writes no caller memory and retains no pointers.
    unsafe {
        mandacaru_kernels::onecenter::mandacaru_radial_coulomb(
            l as i32,
            r.len() as i32,
            r.as_ptr(),
            a.as_ptr(),
            b.as_ptr(),
        )
    }
}
pub fn build(ctx: &Context, species: &Species) -> Result<AtomTerms> {
    let projectors = layout(species);
    let p = projectors.len();
    let pp = product(&[p, p])?;
    if p == 0 {
        return Err(invalid("PAW species has no projectors"));
    }
    let lmax = projectors.iter().map(|p| p.l).max().unwrap() * 2;
    let memory = ctx.reserve(
        product(&[pp, pp, 8, 6])?
            .checked_add(product(&[pp, 600, 16, lmax + 1])?)
            .ok_or(crate::Error::Overflow)?,
    )?;
    let angular = angular_rule(24, 48)?;
    let mut gaunts = Vec::new();
    for l in 0..=lmax {
        for m in -(l as i32)..=l as i32 {
            let mut g = DMatrix::<f64>::zeros(p, p);
            for (direction, weight) in &angular {
                let top = harmonic(l, m, *direction) * weight;
                let values: Vec<_> = projectors
                    .iter()
                    .map(|a| harmonic(a.l, a.m, *direction))
                    .collect();
                for a in 0..p {
                    for b in 0..p {
                        if (l + projectors[a].l + projectors[b].l) % 2 == 0
                            && l >= projectors[a].l.abs_diff(projectors[b].l)
                            && l <= projectors[a].l + projectors[b].l
                        {
                            g[(a, b)] += top * values[a] * values[b];
                        }
                    }
                }
            }
            for value in g.iter_mut() {
                if value.abs() < 1e-12 {
                    *value = 0.;
                }
            }
            gaunts.push((l, m, g));
        }
    }
    let source = species.ionic.knots.as_slice();
    let radius = species
        .channels
        .iter()
        .map(|ch| ch.radius)
        .fold(0., f64::max);
    let r: Vec<_> = (0..600).map(|i| radius * i as f64 / 599.).collect();
    let mut ae_pair = vec![vec![0.; 600]; pp];
    let mut ps_pair = vec![vec![0.; 600]; pp];
    for (a, pa) in projectors.iter().enumerate() {
        for (b, pb) in projectors.iter().enumerate() {
            let ca = &species.channels[pa.channel];
            let cb = &species.channels[pb.channel];
            for k in 0..600 {
                ae_pair[a * p + b][k] = linear(source, &ca.ae[pa.radial], r[k])
                    * linear(source, &cb.ae[pb.radial], r[k]);
                ps_pair[a * p + b][k] = linear(source, &ca.ps[pa.radial], r[k])
                    * linear(source, &cb.ps[pb.radial], r[k]);
            }
        }
    }
    let mut multipoles = Vec::new();
    let mut compensated = Vec::new();
    for l in 0..=lmax {
        let mut pairs = Vec::new();
        for ab in 0..pp {
            let difference: Vec<_> = (0..600)
                .map(|k| (ae_pair[ab][k] - ps_pair[ab][k]) * r[k].powi((l + 2) as i32))
                .collect();
            let moment = simpson(&r, &difference)?;
            pairs.push(
                (0..600)
                    .map(|k| ps_pair[ab][k] + moment * shape(r[k], species.compensation_radius, l))
                    .collect::<Vec<_>>(),
            );
        }
        compensated.push(pairs);
    }
    for (l, m, g) in &gaunts {
        if g.iter().all(|&v| v == 0.) {
            continue;
        }
        let mut block = DMatrix::zeros(p, p);
        for (a, pa) in projectors.iter().enumerate() {
            for (b, pb) in projectors.iter().enumerate() {
                if g[(a, b)] == 0. {
                    continue;
                }
                let ca = &species.channels[pa.channel];
                let cb = &species.channels[pb.channel];
                let moment = if *l == 0 && pa.channel == pb.channel {
                    ca.overlap[(pa.radial, pb.radial)]
                } else {
                    let inside = source.partition_point(|r| *r <= ca.radius.max(cb.radius));
                    let values: Vec<_> = (0..inside)
                        .map(|k| {
                            (ca.ae[pa.radial][k] * cb.ae[pb.radial][k]
                                - ca.ps[pa.radial][k] * cb.ps[pb.radial][k])
                                * source[k].powi((*l + 2) as i32)
                        })
                        .collect();
                    simpson(&source[..inside], &values)?
                };
                block[(a, b)] = moment * g[(a, b)];
            }
        }
        if block.iter().any(|v| v.abs() > 1e-14) {
            multipoles.push(Multipole {
                l: *l,
                m: *m,
                block,
            });
        }
    }
    let mut w = DMatrix::zeros(pp, pp);
    for ab in 0..pp {
        for cd in ab..pp {
            let mut value = 0.;
            for (l, compensated) in compensated.iter().enumerate() {
                let angular: f64 = gaunts
                    .iter()
                    .filter(|(level, _, _)| *level == l)
                    .map(|(_, _, g)| g[(ab / p, ab % p)] * g[(cd / p, cd % p)])
                    .sum::<f64>()
                    * 4.
                    * PI
                    / (2 * l + 1) as f64;
                if angular.abs() < 1e-14 {
                    continue;
                }
                let bare = 0.5
                    * (coulomb(&r, &ae_pair[ab], &ae_pair[cd], l)
                        + coulomb(&r, &ae_pair[cd], &ae_pair[ab], l));
                let smooth = 0.5
                    * (coulomb(&r, &compensated[ab], &compensated[cd], l)
                        + coulomb(&r, &compensated[cd], &compensated[ab], l));
                value += angular * (bare - smooth);
            }
            w[(ab, cd)] = value;
            w[(cd, ab)] = value;
        }
    }
    let mut coupling = DMatrix::zeros(p, p);
    let mut ks_coupling = DMatrix::zeros(p, p);
    let mut overlap = DMatrix::zeros(p, p);
    let mut reference = DMatrix::zeros(p, p);
    for (a, pa) in projectors.iter().enumerate() {
        for (b, pb) in projectors.iter().enumerate() {
            if pa.channel == pb.channel && pa.m == pb.m {
                let ch = &species.channels[pa.channel];
                coupling[(a, b)] = ch.coupling[(pa.radial, pb.radial)]
                    - ch.xc[(pa.radial, pb.radial)]
                    + ch.core_exchange[(pa.radial, pb.radial)];
                ks_coupling[(a, b)] = ch.coupling[(pa.radial, pb.radial)];
                overlap[(a, b)] = ch.overlap[(pa.radial, pb.radial)];
                reference[(a, b)] = ch.reference[(pa.radial, pb.radial)];
            }
        }
    }
    for a in 0..p {
        for b in 0..p {
            coupling[(a, b)] -= (0..p)
                .map(|c| {
                    (0..p)
                        .map(|d| w[(a * p + b, c * p + d)] * reference[(c, d)])
                        .sum::<f64>()
                })
                .sum::<f64>();
        }
    }
    let (values, vectors) = real_eigen(w)?;
    let floor = values.iter().map(|v| v.abs()).fold(0., f64::max) * 1e-10;
    let keep: Vec<_> = values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.abs() > floor)
        .map(|(i, _)| i)
        .collect();
    let factors = keep
        .iter()
        .map(|&k| {
            DMatrix::from_fn(p, p, |a, b| {
                0.5 * (vectors[(a * p + b, k)] + vectors[(b * p + a, k)])
            })
        })
        .collect();
    let weights = keep.iter().map(|&k| values[k]).collect();
    Ok(AtomTerms {
        ks_difference: ks_coupling - &coupling,
        projectors,
        coupling,
        overlap,
        multipoles,
        factors,
        weights,
        _memory: memory,
    })
}
