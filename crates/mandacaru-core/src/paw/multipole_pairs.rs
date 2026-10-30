//! Partial-wave pairs inside the augmentation sphere: the smooth waves rebuilt
//! from their Bessel expansions, the radial multipole moments and the part of
//! the pair density the compensation shape leaves.
use super::invalid;
use crate::math::spline::simpson;
use crate::{finite, product, Context, Result};
use mandacaru_kernels::two_center::spherical_bessel as bessel;

/// All-electron waves `ae[i][k] = R_i(r_k)` of one channel with the Bessel
/// expansion (`coefficients`, `wavevectors`, both `waves x terms`) of each
/// smooth wave.
pub struct ChannelWaves<'a> {
    pub l: usize,
    pub r_cut: f64,
    pub ae: &'a [f64],
    pub coefficients: &'a [f64],
    pub wavevectors: &'a [f64],
    pub waves: usize,
}

fn validate(r: &[f64], waves: &ChannelWaves<'_>) -> Result<usize> {
    let (n, terms) = (r.len(), waves.coefficients.len() / waves.waves.max(1));
    if waves.waves == 0
        || n < 4
        || waves.ae.len() != product(&[waves.waves, n])?
        || waves.coefficients.len() != product(&[waves.waves, terms])?
        || waves.wavevectors.len() != waves.coefficients.len()
        || !waves.r_cut.is_finite()
    {
        return Err(invalid("inconsistent partial-wave arrays"));
    }
    finite(waves.ae, "all-electron partial waves")?;
    finite(waves.coefficients, "Bessel coefficients")?;
    finite(waves.wavevectors, "Bessel wave vectors")?;
    Ok(terms)
}

/// The smooth waves on `r`: the Bessel expansion inside the channel's
/// `r_cut`, the all-electron wave beyond it (they match there in value and
/// three derivatives; the expansion itself diverges outside).
pub fn smooth_waves(r: &[f64], waves: &ChannelWaves<'_>) -> Result<Vec<f64>> {
    let terms = validate(r, waves)?;
    let n = r.len();
    let mut out = waves.ae.to_vec();
    for i in 0..waves.waves {
        let (c, q) = (
            &waves.coefficients[i * terms..(i + 1) * terms],
            &waves.wavevectors[i * terms..(i + 1) * terms],
        );
        for (k, &x) in r.iter().enumerate() {
            if x <= waves.r_cut {
                out[i * n + k] = (0..terms)
                    .map(|t| c[t] * bessel(waves.l as i32, q[t] * x))
                    .sum();
            }
        }
    }
    Ok(out)
}

/// A pair of channels with their all-electron and smooth waves.
pub struct Pair<'a> {
    pub r: &'a [f64],
    pub r_cut: f64,
    pub ae: [&'a [f64]; 2],
    pub smooth: [&'a [f64]; 2],
    pub waves: [usize; 2],
}

fn density(pair: &Pair<'_>, i: usize, j: usize, inside: usize) -> Vec<f64> {
    let n = pair.r.len();
    (0..inside)
        .map(|k| {
            pair.ae[0][i * n + k] * pair.ae[1][j * n + k]
                - pair.smooth[0][i * n + k] * pair.smooth[1][j * n + k]
        })
        .collect()
}

fn check(pair: &Pair<'_>) -> Result<usize> {
    let n = pair.r.len();
    for side in 0..2 {
        if pair.waves[side] == 0
            || pair.ae[side].len() != product(&[pair.waves[side], n])?
            || pair.smooth[side].len() != pair.ae[side].len()
        {
            return Err(invalid("inconsistent partial-wave pair"));
        }
    }
    Ok(pair.r.partition_point(|&x| x <= pair.r_cut))
}

/// `Delta^(L)_ij = int (R_i R_j - R~_i R~_j) r^(L+2) dr` over `r <= r_cut`.
pub fn radial_moments(pair: &Pair<'_>, l: usize) -> Result<Vec<f64>> {
    let inside = check(pair)?;
    let weight = pair.r[..inside]
        .iter()
        .map(|r| r.powi(l as i32 + 2))
        .collect::<Vec<_>>();
    let mut out = Vec::with_capacity(pair.waves[0] * pair.waves[1]);
    for i in 0..pair.waves[0] {
        for j in 0..pair.waves[1] {
            let integrand = density(pair, i, j, inside)
                .iter()
                .zip(&weight)
                .map(|(d, w)| d * w)
                .collect::<Vec<_>>();
            out.push(simpson(&pair.r[..inside], &integrand)?);
        }
    }
    Ok(out)
}

/// `E^(L)_ij(q) = int (R_i R_j - R~_i R~_j) j_L(q r) r^2 dr - Delta_ij g(q)`,
/// laid out `[q][i][j]`; `delta` is `waves0 x waves1`, `shape` is `g(q)`.
pub fn radial_remainder(
    ctx: &Context,
    pair: &Pair<'_>,
    l: usize,
    q: &[f64],
    delta: &[f64],
    shape: &[f64],
) -> Result<Vec<f64>> {
    let inside = check(pair)?;
    let (n1, n2) = (pair.waves[0], pair.waves[1]);
    if delta.len() != product(&[n1, n2])? || shape.len() != q.len() {
        return Err(invalid(
            "moments and shape transform must match the waves and wave vectors",
        ));
    }
    finite(q, "wave vectors")?;
    finite(delta, "multipole moments")?;
    finite(shape, "shape transform")?;
    let _workspace = ctx.reserve(product(&[inside, n1 * n2 + 2, 8])?)?;
    let x = &pair.r[..inside];
    let densities = (0..n1 * n2)
        .map(|index| {
            density(pair, index / n2, index % n2, inside)
                .iter()
                .zip(x)
                .map(|(d, r)| d * r * r)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut out = vec![0.; q.len() * n1 * n2];
    for (iq, &wavevector) in q.iter().enumerate() {
        let table = x
            .iter()
            .map(|r| bessel(l as i32, wavevector * r))
            .collect::<Vec<_>>();
        for (index, d) in densities.iter().enumerate() {
            let integrand = d.iter().zip(&table).map(|(a, b)| a * b).collect::<Vec<_>>();
            out[iq * n1 * n2 + index] = simpson(x, &integrand)? - delta[index] * shape[iq];
        }
    }
    Ok(out)
}
