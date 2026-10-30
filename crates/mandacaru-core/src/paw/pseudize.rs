//! Optimized smooth partial waves (Hamann, PRB 88, 085117): inside `r_c` a sum
//! of spherical Bessel functions matched in value and three derivatives to the
//! all-electron wave, with the residual kinetic energy beyond a wave-vector
//! cutoff minimized under the norm condition.
use super::{
    invalid,
    pseudize_tables::{
        bessel_table, gram, inner_grid, q_grid, resample, tail_transform, transform_table,
        weighted_overlap, Q_STEP,
    },
    smooth::{bessel, wavevectors},
};
use crate::models::numerov::{bound_state, derivative4, scattering_wave, DEFAULT_WINDOW};
use crate::models::relativity::{mass_factor, PotentialTerms, Treatment, SCALAR_KAPPA};
use crate::{math::constrained, product, Context, Result};
use nalgebra::{DMatrix as M, DVector as V};

/// The radial equation the all-electron waves obey.
#[derive(Clone, Copy, Debug)]
pub struct Equation {
    pub treatment: Treatment,
    pub kappa: Option<i32>,
    pub z: f64,
}

fn nearest(r: &[f64], radius: f64) -> usize {
    (0..r.len())
        .min_by(|&a, &b| (r[a] - radius).abs().total_cmp(&(r[b] - radius).abs()))
        .unwrap_or(0)
}

/// The grid node nearest to `radius`.
pub fn snap(r: &[f64], radius: f64) -> f64 {
    r[nearest(r, radius)]
}

fn stencil(u: &[f64], h: f64, index: usize) -> Result<f64> {
    if index < 2 || index + 3 > u.len() {
        return Err(invalid(format!(
            "the fourth-order derivative stencil needs a grid index in [2, {}] and was asked for {index} on a {}-point grid; a channel cutoff was placed at the edge of the radial grid",
            u.len().saturating_sub(3),
            u.len()
        )));
    }
    Ok(derivative4(u, h, index))
}

fn kappa_of(eq: &Equation) -> Result<i32> {
    match eq.treatment {
        Treatment::Scalar => Ok(SCALAR_KAPPA),
        Treatment::Dirac => eq
            .kappa
            .ok_or_else(|| invalid("the Dirac treatment needs kappa")),
        Treatment::None => Ok(SCALAR_KAPPA),
    }
}

/// `[R, R', R'', R''']` of a partial wave at the node nearest `r_cut`. `R'`
/// comes from a fourth-order stencil on `u = rR`; the next two derivatives
/// follow from the radial equation the wave satisfies,
/// `u'' = a u + b u'` with `a = l(l+1)/r^2 + kappa M'/(M r) + 2M(V - E)`, `b = M'/M`.
pub fn matching_targets(
    r: &[f64],
    u: &[f64],
    v: &[f64],
    l: usize,
    energy: f64,
    r_cut: f64,
    eq: &Equation,
) -> Result<[f64; 4]> {
    let n = r.len();
    if n < 5 || u.len() != n || v.len() != n {
        return Err(invalid("matching targets need matching radial arrays"));
    }
    let h = r[1] - r[0];
    let k = nearest(r, r_cut);
    let (rk, uk) = (r[k], u[k]);
    let up = stencil(u, h, k)?;
    let centrifugal = |i: usize| (l * (l + 1)) as f64 / (r[i] * r[i]);
    let (a, b) = if eq.treatment == Treatment::None {
        (
            (0..n)
                .map(|i| centrifugal(i) + 2. * (v[i] - energy))
                .collect::<Vec<_>>(),
            vec![0.; n],
        )
    } else {
        let kk = kappa_of(eq)? as f64;
        let terms = PotentialTerms::new(r, v, eq.z)?;
        let (m, dm) = mass_factor(v, energy, &terms);
        (
            (0..n)
                .map(|i| centrifugal(i) + kk * dm[i] / (m[i] * r[i]) + 2. * m[i] * (v[i] - energy))
                .collect(),
            (0..n).map(|i| dm[i] / m[i]).collect(),
        )
    };
    let (ap, bp) = (stencil(&a, h, k)?, stencil(&b, h, k)?);
    let upp = a[k] * uk + b[k] * up;
    let uppp = ap * uk + a[k] * up + bp * up + b[k] * upp;
    Ok([
        uk / rk,
        up / rk - uk / rk.powi(2),
        upp / rk - 2. * up / rk.powi(2) + 2. * uk / rk.powi(3),
        uppp / rk - 3. * upp / rk.powi(2) + 6. * up / rk.powi(3) - 6. * uk / rk.powi(4),
    ])
}

/// The waves, one per row of `waves` (`count x len(r)`), resampled to the inner grid.
fn inner_waves(
    ctx: &Context,
    r: &[f64],
    waves: &[f64],
    count: usize,
    r_in: &[f64],
) -> Result<Vec<Vec<f64>>> {
    (0..count)
        .map(|i| resample(ctx, r, &waves[i * r.len()..(i + 1) * r.len()], r_in))
        .collect()
}

fn overlap_matrix(r_in: &[f64], inside: &[Vec<f64>]) -> Result<M<f64>> {
    let n = inside.len();
    let mut out = M::zeros(n, n);
    for a in 0..n {
        for b in 0..n {
            out[(a, b)] = weighted_overlap(r_in, &inside[a], &inside[b])?;
        }
    }
    Ok(out)
}

/// `<phi_i|phi_j>` inside `r_cut` on the refined inner grid: the true
/// all-electron overlap, whatever equation the waves solve.
pub fn inner_overlaps(
    ctx: &Context,
    r: &[f64],
    waves: &[f64],
    count: usize,
    r_cut: f64,
) -> Result<M<f64>> {
    let r_in = inner_grid(r_cut);
    overlap_matrix(&r_in, &inner_waves(ctx, r, waves, count, &r_in)?)
}

/// The generalized-norm matrix the smooth waves must reproduce: the inner
/// overlap non-relativistically; relativistically the off-diagonal terms are
/// Wronskians, `-W_ij(r_c) / 2(E_j - E_i)`, and the diagonal carries
/// `M(r_c) int (2M - 1)/M u^2 dr`, so the smooth system reproduces the
/// logarithmic derivative and its energy derivative.
pub fn norm_targets(
    ctx: &Context,
    r: &[f64],
    waves: &[f64],
    energies: &[f64],
    r_cut: f64,
    v: &[f64],
    eq: &Equation,
) -> Result<M<f64>> {
    let count = energies.len();
    let r_in = inner_grid(r_cut);
    let inside = inner_waves(ctx, r, waves, count, &r_in)?;
    let mut targets = overlap_matrix(&r_in, &inside)?;
    if eq.treatment == Treatment::None {
        return Ok(targets);
    }
    let n = r.len();
    let (h, k) = (r[1] - r[0], nearest(r, r_cut));
    let terms = PotentialTerms::new(r, v, eq.z)?;
    let us = (0..count)
        .map(|i| (0..n).map(|x| waves[i * n + x] * r[x]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    for i in 0..count {
        let m = mass_factor(v, energies[i], &terms).0;
        let m_in = resample(ctx, r, &m, &r_in)?;
        let weighted = (0..r_in.len())
            .map(|x| {
                (2. * m_in[x] - 1.) / m_in[x] * inside[i][x] * inside[i][x] * r_in[x] * r_in[x]
            })
            .collect::<Vec<_>>();
        targets[(i, i)] = m[k] * crate::math::spline::simpson(&r_in, &weighted)?;
        for j in 0..count {
            let gap = energies[j] - energies[i];
            if i == j || gap.abs() < 1e-12 {
                continue;
            }
            let wronskian = us[i][k] * stencil(&us[j], h, k)? - us[j][k] * stencil(&us[i], h, k)?;
            targets[(i, j)] = -wronskian / (2. * gap);
        }
    }
    Ok(targets)
}

/// The scattering wave at `energy` as a radial function `R = u / r`, normalized
/// to `int u^2 dr = 1` over the nodes inside `radius` (trapezoid rule).
pub fn scattering_reference(
    ctx: &Context,
    r: &[f64],
    v: &[f64],
    l: usize,
    energy: f64,
    equation: &Equation,
    radius: f64,
) -> Result<Vec<f64>> {
    let u = scattering_wave(
        ctx,
        r,
        v,
        l,
        energy,
        equation.z,
        equation.treatment,
        equation.kappa,
    )?;
    let inside = r.partition_point(|x| *x <= radius);
    let norm = (1..inside)
        .map(|i| 0.5 * (r[i] - r[i - 1]) * (u[i - 1].powi(2) + u[i].powi(2)))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= 0. {
        return Err(invalid(
            "the scattering reference has no weight inside the cutoff",
        ));
    }
    Ok(u.iter().zip(r).map(|(u, r)| u / norm / r).collect())
}

/// The bound reference `(R, energy)` with `R = u / r`: the Numerov shoot near
/// `guess`, at the level of theory of the equation.
pub fn bound_reference(
    ctx: &Context,
    r: &[f64],
    v: &[f64],
    l: usize,
    guess: f64,
    equation: &Equation,
    window: Option<f64>,
) -> Result<(Vec<f64>, f64)> {
    let (u, energy) = bound_state(
        ctx,
        r,
        v,
        l,
        guess,
        equation.z,
        window.unwrap_or(DEFAULT_WINDOW),
        equation.treatment,
        equation.kappa,
    )?
    .ok_or_else(|| {
        invalid(format!(
            "no Numerov bound state near {guess:.6} Ha for l={l}"
        ))
    })?;
    Ok((u.iter().zip(r).map(|(u, r)| u / r).collect(), energy))
}

/// How the norm of the smooth waves is fixed.
#[derive(Clone, Copy, Debug)]
pub enum NormCondition {
    /// `<phi~_i|phi~_j> = factor <phi_i|phi_j>` (1 is norm conservation).
    Factor(f64),
    /// No norm condition: the minimum of the residual energy in the null space
    /// of the matching conditions, regularized by a tiny multiple of the metric.
    Free,
}

pub struct PseudoInput<'a> {
    pub r: &'a [f64],
    pub v_ae: &'a [f64],
    pub l: usize,
    pub waves: &'a [f64],
    pub energies: &'a [f64],
    pub r_cut: f64,
    pub q_cut: f64,
    pub n_bessel: usize,
    pub norm: NormCondition,
    pub equation: Equation,
}

pub struct PseudoWaves {
    pub r_cut: f64,
    pub wavevectors: Vec<f64>,
    pub coefficients: M<f64>,
    pub residuals: Vec<f64>,
    pub norms: M<f64>,
    pub achieved: M<f64>,
    pub targets: M<f64>,
}

/// Null space (columns of the result), minimum-norm particular solution
/// operator and the SVD rank of the matching rows `a` (`rows x terms`).
fn null_space(a: &M<f64>) -> (M<f64>, M<f64>) {
    let n = a.ncols();
    let mut padded = M::zeros(n, n);
    padded.rows_mut(0, a.nrows()).copy_from(a);
    let svd = padded.svd(true, true);
    let (u, vt, sigma) = (svd.u.unwrap(), svd.v_t.unwrap(), svd.singular_values);
    let top = sigma.max();
    let kept = (0..n)
        .filter(|&i| sigma[i] > 1e-12 * top)
        .collect::<Vec<_>>();
    let dropped = (0..n)
        .filter(|&i| sigma[i] <= 1e-12 * top)
        .collect::<Vec<_>>();
    let z = M::from_fn(n, dropped.len(), |i, j| vt[(dropped[j], i)]);
    // pinv(A) = V S^+ U^T, restricted to the kept singular values.
    let pinv = M::from_fn(n, n, |i, j| {
        kept.iter()
            .map(|&k| vt[(k, i)] * u[(j, k)] / sigma[k])
            .sum()
    });
    (z, pinv)
}

/// The Bessel expansions of the smooth partial waves of one channel.
pub fn optimize(ctx: &Context, input: &PseudoInput<'_>) -> Result<PseudoWaves> {
    let PseudoInput {
        r,
        v_ae,
        l,
        waves,
        energies,
        q_cut,
        n_bessel,
        norm,
        equation,
        ..
    } = *input;
    let count = energies.len();
    let n = r.len();
    if count == 0
        || n < 8
        || waves.len() != product(&[count, n])?
        || v_ae.len() != n
        || !q_cut.is_finite()
    {
        return Err(invalid("invalid pseudo-wave optimization input"));
    }
    let r_cut = snap(r, input.r_cut);
    let (r_in, q) = (inner_grid(r_cut), q_grid());
    let _workspace = ctx.reserve(product(&[n_bessel, q.len(), r_in.len(), 8])?)?;
    let weight = q
        .iter()
        .map(|&q| if q >= q_cut { 0.5 * q.powi(4) } else { 0. })
        .collect::<Vec<_>>();
    let norms = inner_overlaps(ctx, r, waves, count, r_cut)?;
    let targets = match norm {
        NormCondition::Factor(_) => norm_targets(ctx, r, waves, energies, r_cut, v_ae, &equation)?,
        NormCondition::Free => norms.clone(),
    };
    let qs = wavevectors(l, r_cut, n_bessel)?;
    let nb = qs.len();
    let rows = {
        let derivs = qs.iter().map(|&k| bessel(l, k * r_cut)).collect::<Vec<_>>();
        M::from_fn(4, nb, |row, col| {
            qs[col].powi(row as i32) * derivs[col][row]
        })
    };
    let basis = bessel_table(l, &qs, &r_in);
    let gram_matrix = M::from_row_slice(nb, nb, &gram(l, &qs, &r_in)?);
    let transform = M::from_row_slice(nb, q.len(), &transform_table(l, &qs, &r_in, &q)?);
    let weighted = M::from_fn(nb, q.len(), |a, k| transform[(a, k)] * weight[k]);
    let kernel = &weighted * transform.transpose() * Q_STEP;
    let free = match norm {
        NormCondition::Free => {
            let (z, pinv) = null_space(&rows);
            let reg = 1e-10 * kernel.trace() / nb as f64;
            let kz = z.transpose() * &kernel * &z + reg * (z.transpose() * &gram_matrix * &z);
            Some((z, pinv, kz))
        }
        NormCondition::Factor(_) => None,
    };
    let mut coefficients = M::zeros(count, nb);
    let mut residuals = Vec::with_capacity(count);
    let mut pseudo = Vec::with_capacity(count);
    for i in 0..count {
        let wave = &waves[i * n..(i + 1) * n];
        let u = wave.iter().zip(r).map(|(w, r)| w * r).collect::<Vec<_>>();
        let target = matching_targets(r, &u, v_ae, l, energies[i], r_cut, &equation)?;
        let bound = energies[i] < 0. && (wave[n - 1] * r[n - 1]).abs() < 1e-6;
        let tail = V::from_vec(tail_transform(ctx, l, r, wave, r_cut, bound, &q)?);
        let linear = (&weighted * &tail) * Q_STEP;
        let constant = (0..q.len())
            .map(|k| weight[k] * tail[k] * tail[k])
            .sum::<f64>()
            * Q_STEP;
        let c = match (&free, norm) {
            (Some((z, pinv, kz)), _) => {
                let mut padded = V::zeros(nb);
                padded.rows_mut(0, 4).copy_from(&V::from_row_slice(&target));
                let c0 = pinv * padded;
                let rhs = -(z.transpose() * (&kernel * &c0 + &linear));
                let y = kz
                    .clone()
                    .lu()
                    .solve(&rhs)
                    .ok_or_else(|| invalid("singular reduced residual-energy system"))?;
                &c0 + z * y
            }
            (None, NormCondition::Factor(factor)) => {
                let mut a = M::zeros(4 + i, nb);
                a.rows_mut(0, 4).copy_from(&rows);
                let mut b = V::zeros(4 + i);
                b.rows_mut(0, 4).copy_from(&V::from_row_slice(&target));
                for k in 0..i {
                    let previous = coefficients.row(k).transpose();
                    a.set_row(4 + k, &(&gram_matrix * previous).transpose());
                    b[4 + k] = factor * targets[(i, k)];
                }
                constrained::minimum(
                    &kernel,
                    &linear,
                    &a,
                    &b,
                    &gram_matrix,
                    factor * targets[(i, i)],
                )?
            }
            _ => unreachable!(),
        };
        residuals.push(c.dot(&(&kernel * &c)) + 2. * linear.dot(&c) + constant);
        pseudo.push(
            (0..r_in.len())
                .map(|x| {
                    (0..nb)
                        .map(|a| c[a] * basis[a * r_in.len() + x])
                        .sum::<f64>()
                })
                .collect::<Vec<_>>(),
        );
        coefficients.set_row(i, &c.transpose());
    }
    let achieved = overlap_matrix(&r_in, &pseudo)?;
    Ok(PseudoWaves {
        r_cut,
        wavevectors: qs,
        coefficients,
        residuals,
        norms,
        achieved,
        targets,
    })
}
