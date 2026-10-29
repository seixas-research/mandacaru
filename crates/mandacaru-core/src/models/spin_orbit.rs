//! Angular L.S operators and their separable radial-projector contractions.
use super::invalid;
use crate::{product, Buffer, Context, Result};
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
