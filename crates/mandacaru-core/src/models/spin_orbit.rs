//! Angular L.S operators and their separable radial-projector contractions.
use super::invalid;
use crate::{product, Buffer, Context, Result};
use std::collections::BTreeMap;
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

pub fn angular(ctx: &Context, l: usize, real: bool) -> Result<Buffer<C64>> {
    if l > 16 {
        return Err(invalid("spin-orbit angular degree must be at most 16"));
    }
    let width = 2 * l + 1;
    let size = 2 * width;
    let _work = ctx.reserve(product(&[size, size, 16, 5])?)?;
    let mut matrix = DMatrix::<C64>::zeros(size, size);
    for a in 0..width {
        let m = a as f64 - l as f64;
        matrix[(a, a)] = C64::new(m / 2., 0.);
        matrix[(width + a, width + a)] = C64::new(-m / 2., 0.);
        if a + 1 < width {
            let value = C64::new(0.5 * ((l * (l + 1)) as f64 - m * (m + 1.)).sqrt(), 0.);
            matrix[(a, width + a + 1)] = value;
            matrix[(width + a + 1, a)] = value;
        }
    }
    if real {
        // S_lm = sum U_mm' Y_lm'. Bra coefficients are conjugated.
        let mut u = DMatrix::<C64>::zeros(width, width);
        u[(l, l)] = C64::new(1., 0.);
        let root = std::f64::consts::FRAC_1_SQRT_2;
        for m in 1..=l {
            let sign = if m % 2 == 0 { 1. } else { -1. };
            u[(l + m, l + m)] = C64::new(sign * root, 0.);
            u[(l + m, l - m)] = C64::new(root, 0.);
            u[(l - m, l - m)] = C64::new(0., root);
            u[(l - m, l + m)] = C64::new(0., -sign * root);
        }
        let w = DMatrix::from_fn(size, size, |i, j| {
            if i / width == j / width {
                u[(i % width, j % width)].conj()
            } else {
                C64::default()
            }
        });
        matrix = &w * matrix * w.adjoint();
    }
    let mut out = ctx.allocate::<C64>(size * size)?;
    for i in 0..size {
        for j in 0..size {
            out.as_mut_slice()[i * size + j] = matrix[(i, j)];
        }
    }
    Ok(out)
}

pub struct Channel {
    pub l: usize,
    pub real: bool,
    /// Projector positions, ordered first by m=-l..l, then radial index.
    pub indices: Buffer<usize>,
    pub radial: usize,
    pub coupling: Buffer<C64>,
}

pub struct Projectors<'a> {
    pub left: &'a [C64],
    pub right: &'a [C64],
    pub rows: usize,
    pub columns: usize,
    pub count: usize,
}
pub fn pair(
    ctx: &Context,
    projections: Projectors<'_>,
    channels: &[Channel],
    hermitian: bool,
) -> Result<Buffer<C64>> {
    let Projectors {
        left,
        right,
        rows,
        columns,
        count: projectors,
    } = projections;
    if left.len() != product(&[rows, projectors])?
        || right.len() != product(&[columns, projectors])?
        || left
            .iter()
            .chain(right)
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        || (hermitian && (rows != columns || left != right))
    {
        return Err(invalid(
            "invalid spin-orbit projection dimensions, values or Hermitian request",
        ));
    }
    let mut out = ctx.allocate::<C64>(product(&[2, rows, 2, columns])?)?;
    ctx.run(|| {
        for channel in channels {
            if channel.l > 16
                || channel.radial == 0
                || channel.indices.as_slice().len()
                    != product(&[2 * channel.l + 1, channel.radial])?
                || channel.coupling.as_slice().len() != product(&[channel.radial, channel.radial])?
                || channel
                    .coupling
                    .as_slice()
                    .iter()
                    .any(|v| !v.re.is_finite() || !v.im.is_finite())
                || channel.indices.as_slice().iter().any(|i| *i >= projectors)
            {
                return Err(invalid(
                    "invalid spin-orbit channel labels or radial coupling",
                ));
            }
            let width = 2 * channel.l + 1;
            let count = channel.indices.as_slice().len();
            let bytes = product(&[rows + columns, count, 16, 2])?
                .checked_add(product(&[count, count, 16, 2])?)
                .and_then(|v| v.checked_add(product(&[rows, columns, 16, 2]).ok()?))
                .ok_or(crate::Error::Overflow)?;
            let _work = ctx.reserve(bytes)?;
            let angular = angular(ctx, channel.l, channel.real)?;
            let l = DMatrix::from_fn(rows, count, |i, j| {
                left[i * projectors + channel.indices.as_slice()[j]]
            });
            let r = DMatrix::from_fn(columns, count, |i, j| {
                right[i * projectors + channel.indices.as_slice()[j]]
            });
            for sigma in 0..2 {
                for tau in 0..2 {
                    let block = DMatrix::from_fn(count, count, |i, j| {
                        let angular =
                            angular.as_slice()[(sigma * width + i / channel.radial) * 2 * width
                                + tau * width
                                + j / channel.radial];
                        angular
                            * channel.coupling.as_slice()
                                [(i % channel.radial) * channel.radial + j % channel.radial]
                    });
                    let value = &l * block * r.adjoint();
                    for i in 0..rows {
                        for j in 0..columns {
                            out.as_mut_slice()
                                [(sigma * rows + i) * 2 * columns + tau * columns + j] +=
                                value[(i, j)];
                        }
                    }
                }
            }
        }
        if hermitian {
            let size = 2 * rows;
            for i in 0..size {
                for j in 0..=i {
                    let value =
                        (out.as_slice()[i * size + j] + out.as_slice()[j * size + i].conj()) * 0.5;
                    out.as_mut_slice()[i * size + j] = value;
                    out.as_mut_slice()[j * size + i] = value.conj();
                }
            }
        }
        Ok(())
    })?;
    Ok(out)
}

/// The radial coupling `D^{SO}` of one `(atom, l)`: `radial x radial`, row-major.
#[derive(Clone, Debug)]
pub struct Coupling {
    pub radial: usize,
    pub values: Vec<f64>,
}

/// The label of one spin-orbit projector: its atom, `l`, `m` and radial index.
#[derive(Clone, Copy, Debug)]
pub struct Label {
    pub atom: usize,
    pub l: usize,
    pub m: i32,
    pub index: usize,
}

/// The channels of the term for projectors labeled `labels` (in projector order): for each
/// `(atom, l)` with a coupling and `l > 0`, in order of first appearance, the projector positions
/// ordered `m = -l..l` outer and radial index inner, with complex (`real = false`, a crystal) or
/// real (`real = true`, a molecule) harmonics. A channel that lacks any `(m, index)` of its
/// rectangle, or a duplicated label, is refused.
pub fn channels(
    ctx: &Context,
    labels: &[Label],
    couplings: &BTreeMap<(usize, usize), Coupling>,
    real: bool,
) -> Result<Vec<Channel>> {
    let mut order: Vec<(usize, usize)> = Vec::new();
    let mut positions: BTreeMap<(usize, usize), BTreeMap<(i32, usize), usize>> = BTreeMap::new();
    for (position, p) in labels.iter().enumerate() {
        let group = positions.entry((p.atom, p.l)).or_insert_with(|| {
            order.push((p.atom, p.l));
            BTreeMap::new()
        });
        if group.insert((p.m, p.index), position).is_some() {
            return Err(invalid("duplicate spin-orbit projector label"));
        }
    }
    let mut out = Vec::new();
    for key in order {
        let l = key.1;
        let Some(coupling) = couplings.get(&key) else {
            continue;
        };
        if l == 0 {
            continue;
        }
        let group = &positions[&key];
        let mut radial: Vec<usize> = group.keys().map(|(_, i)| *i).collect();
        radial.sort_unstable();
        radial.dedup();
        let expected: Vec<(i32, usize)> = (-(l as i32)..=l as i32)
            .flat_map(|m| radial.iter().map(move |&i| (m, i)))
            .collect();
        if expected.len() != group.len() || expected.iter().any(|e| !group.contains_key(e)) {
            return Err(invalid("spin-orbit channel needs every angular and radial projector"));
        }
        if coupling.radial != radial.len()
            || coupling.values.len() != product(&[coupling.radial, coupling.radial])?
        {
            return Err(invalid("spin-orbit block dimensions must match its radial projectors"));
        }
        let mut indices = ctx.allocate::<usize>(expected.len())?;
        for (to, e) in indices.as_mut_slice().iter_mut().zip(&expected) {
            *to = group[e];
        }
        let mut matrix = ctx.allocate::<C64>(coupling.values.len())?;
        for (to, v) in matrix.as_mut_slice().iter_mut().zip(&coupling.values) {
            *to = C64::new(*v, 0.);
        }
        out.push(Channel { l, real, indices, radial: radial.len(), coupling: matrix });
    }
    Ok(out)
}

