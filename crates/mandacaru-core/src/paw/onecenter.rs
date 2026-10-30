//! The one-center Coulomb tensor `Delta W[a, b, c, d]` of a PAW sphere: the
//! all-electron pair-pair Coulomb energy minus the smooth one with the
//! compensation charges, assembled from the multipole expansion of `1/r12`
//! (a radial double integral per `L` times the angular couplings).
use super::{
    checks::interpolate_linear,
    invalid,
    multipole::{gaunt, gaunt_grid, multipole_range, shape_function},
    onecenter_radial::{long_range_radial_kernel, radial_coulomb},
};
use crate::{math::spline::simpson, product, Context, Error, Result};
use nalgebra::DMatrix;
use num_complex::Complex64;
use std::{collections::HashMap, f64::consts::PI};

/// Radial points of the one-center quadrature (uniform, inside the sphere).
pub const ONE_CENTER_POINTS: usize = 600;

/// The projector basis the partial waves multiplying `<chi_k|` are taken in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    /// `sum_i (B^-1)_{ki} phi_i`, with `B` the Vanderbilt matrix.
    Raw,
    /// The dual partial waves themselves.
    Dual,
}

impl Basis {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "raw" => Ok(Self::Raw),
            "dual" => Ok(Self::Dual),
            other => Err(invalid(format!("unknown projector basis {other:?}"))),
        }
    }
}

/// One channel's partial waves on the dataset grid (`waves x grid`, row major).
#[derive(Clone, Copy)]
pub struct Channel<'a> {
    pub l: usize,
    pub waves: usize,
    pub ae: &'a [f64],
    pub ps: &'a [f64],
    /// `waves x waves`.
    pub vanderbilt: &'a [f64],
    pub r_cut: f64,
}

impl Channel<'_> {
    pub fn check(&self, grid: usize) -> Result<()> {
        if self.ae.len() != self.waves * grid
            || self.ps.len() != self.waves * grid
            || self.vanderbilt.len() != self.waves * self.waves
        {
            return Err(invalid("partial-wave arrays do not match the radial grid"));
        }
        Ok(())
    }
    pub fn ae_wave(&self, i: usize, grid: usize) -> &[f64] {
        &self.ae[i * grid..(i + 1) * grid]
    }
    pub fn ps_wave(&self, i: usize, grid: usize) -> &[f64] {
        &self.ps[i * grid..(i + 1) * grid]
    }
    /// The matrix taking the dual partial waves to `basis`'s, or `None` for the dual basis.
    pub fn transform(&self, basis: Basis) -> Result<Option<DMatrix<f64>>> {
        match basis {
            Basis::Dual => Ok(None),
            Basis::Raw => DMatrix::from_row_slice(self.waves, self.waves, self.vanderbilt)
                .try_inverse()
                .map(Some)
                .ok_or_else(|| invalid("the Vanderbilt matrix is singular")),
        }
    }
}

/// A dataset's sphere: its radial grid, channels and compensation radius.
pub struct Sphere<'a> {
    pub r: &'a [f64],
    pub channels: Vec<Channel<'a>>,
    pub compensation_radius: f64,
}

/// One projector of a sphere: `chi_{index}(r) Y_{lm}`.
#[derive(Clone, Copy, Debug)]
pub struct Projector {
    pub l: usize,
    pub m: i32,
    pub index: usize,
    pub real: bool,
}

/// `n^4` entries in `[a][b][c][d]` order; real tensors carry zero imaginary parts.
pub struct Tensor {
    pub n: usize,
    pub real: bool,
    pub data: Vec<Complex64>,
}

impl Tensor {
    /// `(W + W.transpose(2, 3, 0, 1)) / 2`: the symmetry under exchanging the two electrons.
    pub fn exchange_symmetrized(&self) -> Self {
        let n = self.n;
        let mut data = self.data.clone();
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    for d in 0..n {
                        data[((a * n + b) * n + c) * n + d] = (self.data
                            [((a * n + b) * n + c) * n + d]
                            + self.data[((c * n + d) * n + a) * n + b])
                            * 0.5;
                    }
                }
            }
        }
        Self {
            n,
            real: self.real,
            data,
        }
    }
}

/// `np.linspace(0, stop, num)`.
fn linspace(stop: f64, num: usize) -> Vec<f64> {
    let step = stop / (num - 1) as f64;
    let mut out = (0..num).map(|i| i as f64 * step).collect::<Vec<_>>();
    out[num - 1] = stop;
    out
}

/// The waves of `channels` interpolated onto the uniform sphere grid, in `basis`.
struct Tables {
    r: Vec<f64>,
    channels: Vec<Waves>,
}

/// `(l, ae waves, smooth waves)` of a channel at the nodes of a sphere quadrature.
pub type Waves = (usize, Vec<Vec<f64>>, Vec<Vec<f64>>);

/// The partial waves of `sphere` interpolated linearly onto `r` and carried to the
/// projector `basis`, in increasing `l`.
pub fn waves_on(sphere: &Sphere<'_>, basis: Basis, r: &[f64]) -> Result<Vec<Waves>> {
    let grid = sphere.r.len();
    let mut channels = sphere.channels.clone();
    channels.sort_by_key(|c| c.l);
    let mut out = Vec::new();
    for channel in &channels {
        channel.check(grid)?;
        let mut ae = Vec::new();
        let mut ps = Vec::new();
        for i in 0..channel.waves {
            ae.push(interpolate_linear(r, sphere.r, channel.ae_wave(i, grid))?);
            ps.push(interpolate_linear(r, sphere.r, channel.ps_wave(i, grid))?);
        }
        if let Some(t) = channel.transform(basis)? {
            let mix = |waves: &[Vec<f64>]| -> Vec<Vec<f64>> {
                (0..waves.len())
                    .map(|i| {
                        (0..r.len())
                            .map(|p| (0..waves.len()).fold(0., |s, k| s + t[(i, k)] * waves[k][p]))
                            .collect()
                    })
                    .collect()
            };
            let (a, s) = (mix(&ae), mix(&ps));
            ae = a;
            ps = s;
        }
        out.push((channel.l, ae, ps));
    }
    Ok(out)
}

fn radial_tables(sphere: &Sphere<'_>, basis: Basis) -> Result<Tables> {
    let r_cut = sphere.channels.iter().map(|c| c.r_cut).fold(0., f64::max);
    let r = linspace(r_cut, ONE_CENTER_POINTS);
    let channels = waves_on(sphere, basis, &r)?;
    Ok(Tables { r, channels })
}

/// Gaunt coefficients cached per `(L, M, l1, m1, l2, m2)` on the default product rule.
pub struct Couplings {
    grid: (Vec<[f64; 3]>, Vec<f64>),
    cache: HashMap<(bool, [i64; 6]), Complex64>,
}

impl Couplings {
    pub fn new() -> Result<Self> {
        Ok(Self {
            grid: gaunt_grid()?,
            cache: HashMap::new(),
        })
    }
    fn gaunt(&mut self, l: usize, m: i32, p: &Projector, q: &Projector, real: bool) -> Complex64 {
        let key = (
            real,
            [
                l as i64, m as i64, p.l as i64, p.m as i64, q.l as i64, q.m as i64,
            ],
        );
        let grid = &self.grid;
        *self
            .cache
            .entry(key)
            .or_insert_with(|| gaunt([l, p.l, q.l], [m, p.m, q.m], real, grid))
    }
    /// The angular factor multiplying the radial integral for the pair densities
    /// `a* b` and `c* d`, in the harmonics `a` carries.
    pub fn angular(
        &mut self,
        l: usize,
        a: &Projector,
        b: &Projector,
        c: &Projector,
        d: &Projector,
    ) -> Complex64 {
        let real = a.real;
        let mut total = Complex64::new(0., 0.);
        for m in -(l as i32)..=l as i32 {
            let left = self.gaunt(l, m, a, b, real);
            if left.norm() == 0. {
                continue;
            }
            if real {
                total += left * self.gaunt(l, m, c, d, true).re;
            } else {
                let right = self.gaunt(l, -m, c, d, false);
                total += left * right * if m % 2 == 0 { 1. } else { -1. };
            }
        }
        total * (4. * PI / (2 * l + 1) as f64)
    }
}

type Pair = (usize, usize, usize, usize, usize);

/// `Delta W[a, b, c, d]` under `1/r12` (`omega = 0`) or `erfc(omega r12)/r12`.
/// Not symmetrized: the cumulative trapezoid inside the Slater integral breaks the
/// exchange of the two electrons at the 1e-5 level.
pub fn one_center_tensor(
    ctx: &Context,
    sphere: &Sphere<'_>,
    basis: Basis,
    projectors: &[Projector],
    omega: f64,
) -> Result<Tensor> {
    if !omega.is_finite() || omega < 0. {
        return Err(invalid(format!("omega must be >= 0, got {omega:?}")));
    }
    let n = projectors.len();
    let _memory = ctx.reserve(product(&[n, n, n, n, 16])?)?;
    let tables = radial_tables(sphere, basis)?;
    let r = &tables.r;
    let points = r.len();
    let lookup = |l: usize| {
        tables
            .channels
            .iter()
            .find(|c| c.0 == l)
            .ok_or_else(|| invalid(format!("the dataset has no channel l = {l}")))
    };
    for p in projectors {
        if p.index >= lookup(p.l)?.1.len() {
            return Err(invalid("a projector index exceeds its channel's waves"));
        }
    }
    let mut levels = Vec::new();
    for a in projectors {
        for b in projectors {
            levels.extend(multipole_range(a.l, b.l));
        }
    }
    levels.sort_unstable();
    levels.dedup();
    let long_range = if omega > 0. {
        let kernels = long_range_radial_kernel(ctx, r, r, &levels, omega, None)?;
        let weights = (0..points)
            .map(|i| {
                let mut unit = vec![0.; points];
                unit[i] = 1.;
                simpson(r, &unit).map(|w| w * r[i] * r[i])
            })
            .collect::<Result<Vec<_>>>()?;
        Some((kernels, weights))
    } else {
        None
    };
    let r_g = sphere.compensation_radius;
    // The caches are keyed by `(l, index)` of the projectors, not by `m`: at most
    // `distinct^2` pair tables (two `points`-long vectors each) per level and
    // `distinct^4` radial values per level. Reserve that bound (a hash map entry
    // is counted at 160 bytes: key, value, control byte and load factor).
    let distinct = {
        let mut keys = projectors
            .iter()
            .map(|p| (p.l, p.index))
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();
        keys.len()
    };
    let _caches = ctx.reserve(
        product(&[distinct, distinct, levels.len(), 2 * points * 8 + 160])?
            .checked_add(product(&[
                distinct,
                distinct,
                distinct,
                distinct,
                levels.len(),
                160,
            ])?)
            .ok_or(Error::Overflow)?,
    )?;
    let mut pair_cache: HashMap<Pair, (Vec<f64>, Vec<f64>)> = HashMap::new();
    let mut radial_cache: HashMap<(Pair, Pair, usize), f64> = HashMap::new();
    let mut couplings = Couplings::new()?;
    let mut data = vec![Complex64::new(0., 0.); n * n * n * n];
    let real = projectors.iter().all(|p| p.real);

    let pair_key =
        |a: &Projector, b: &Projector, l: usize| -> Pair { (a.l, a.index, b.l, b.index, l) };
    for ia in 0..n {
        for ib in 0..n {
            let (a, b) = (&projectors[ia], &projectors[ib]);
            let levels_ab = multipole_range(a.l, b.l);
            for ic in 0..n {
                for id in 0..n {
                    let (c, d) = (&projectors[ic], &projectors[id]);
                    let mut total = Complex64::new(0., 0.);
                    for l in multipole_range(c.l, d.l) {
                        if !levels_ab.contains(&l) {
                            continue;
                        }
                        let factor = couplings.angular(l, a, b, c, d);
                        if factor.norm() == 0. {
                            continue;
                        }
                        let key = (pair_key(a, b, l), pair_key(c, d, l), l);
                        if !radial_cache.contains_key(&key) {
                            for (p, q) in [(a, b), (c, d)] {
                                let k = pair_key(p, q, l);
                                if pair_cache.contains_key(&k) {
                                    continue;
                                }
                                let (_, ae_p, ps_p) = lookup(p.l)?;
                                let (_, ae_q, ps_q) = lookup(q.l)?;
                                let ae = (0..points)
                                    .map(|i| ae_p[p.index][i] * ae_q[q.index][i])
                                    .collect::<Vec<_>>();
                                let ps = (0..points)
                                    .map(|i| ps_p[p.index][i] * ps_q[q.index][i])
                                    .collect::<Vec<_>>();
                                let power = (l + 2) as i32;
                                let moment = simpson(
                                    r,
                                    &(0..points)
                                        .map(|i| ae[i] * r[i].powi(power))
                                        .collect::<Vec<_>>(),
                                )? - simpson(
                                    r,
                                    &(0..points)
                                        .map(|i| ps[i] * r[i].powi(power))
                                        .collect::<Vec<_>>(),
                                )?;
                                let shape = shape_function(r, r_g, l)?;
                                let compensated = (0..points)
                                    .map(|i| ps[i] + moment * shape[i])
                                    .collect::<Vec<_>>();
                                pair_cache.insert(k, (ae, compensated));
                            }
                            let (ae_ab, ps_ab) = &pair_cache[&pair_key(a, b, l)];
                            let (ae_cd, ps_cd) = &pair_cache[&pair_key(c, d, l)];
                            let mut value = radial_coulomb(r, ae_ab, ae_cd, l)?
                                - radial_coulomb(r, ps_ab, ps_cd, l)?;
                            if let Some((kernels, weights)) = &long_range {
                                let kernel = &kernels[levels.binary_search(&l).unwrap()];
                                let form = |left: &[f64], right: &[f64]| -> f64 {
                                    (0..points)
                                        .map(|i| {
                                            let row = &kernel[i * points..(i + 1) * points];
                                            left[i]
                                                * weights[i]
                                                * (0..points)
                                                    .map(|j| row[j] * right[j] * weights[j])
                                                    .sum::<f64>()
                                        })
                                        .sum()
                                };
                                value -= form(ae_ab, ae_cd) - form(ps_ab, ps_cd);
                            }
                            radial_cache.insert(key, value);
                        }
                        total += factor * radial_cache[&key];
                    }
                    data[((ia * n + ib) * n + ic) * n + id] = total;
                }
            }
        }
    }
    Ok(Tensor { n, real, data })
}
