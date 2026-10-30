//! Bounded tiled projection of a local field into a real orbital basis.
use super::gemm::{self, MatrixMut, MatrixRef};
use super::storage::Sample;
use crate::{finite, product, Buffer, Context, Error, Reservation, Result};
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

/// A bounded sum of outer products of streamed quadrature samples. Matrix
/// products retain signed weights and do not change the integration rule.
pub struct PointAccumulator {
    rows: usize,
    columns: usize,
    tile: usize,
    filled: usize,
    left: Buffer<f64>,
    right: Buffer<f64>,
    matrix: Buffer<f64>,
}
impl PointAccumulator {
    pub fn new(ctx: &Context, rows: usize, columns: usize) -> Result<Self> {
        let tile = 2048;
        Ok(Self {
            rows,
            columns,
            tile,
            filled: 0,
            left: ctx.allocate(product(&[rows, tile])?)?,
            right: ctx.allocate(product(&[columns, tile])?)?,
            matrix: ctx.allocate(product(&[rows, columns])?)?,
        })
    }
    pub fn push(&mut self, left: &[f64], right: &[f64]) -> Result<()> {
        if left.len() != self.rows
            || right.len() != self.columns
            || left.iter().chain(right).any(|v| !v.is_finite())
        {
            return Err(Error::Invalid(
                "quadrature sample dimensions or values disagree with the accumulator".into(),
            ));
        }
        if left.iter().all(|v| *v == 0.) || right.iter().all(|v| *v == 0.) {
            return Ok(());
        }
        self.left.as_mut_slice()[self.filled * self.rows..(self.filled + 1) * self.rows]
            .copy_from_slice(left);
        for (j, &value) in right.iter().enumerate() {
            self.right.as_mut_slice()[self.filled + j * self.tile] = value;
        }
        self.filled += 1;
        if self.filled == self.tile {
            self.flush()?;
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<()> {
        if self.filled == 0 {
            return Ok(());
        }
        gemm::multiply(
            MatrixRef {
                values: self.left.as_slice(),
                rows: self.rows,
                columns: self.filled,
                leading: self.rows,
            },
            MatrixRef {
                values: self.right.as_slice(),
                rows: self.filled,
                columns: self.columns,
                leading: self.tile,
            },
            MatrixMut {
                values: self.matrix.as_mut_slice(),
                rows: self.rows,
                columns: self.columns,
                leading: self.rows,
            },
            1.,
            1.,
        )?;
        self.filled = 0;
        Ok(())
    }
    pub fn finish(mut self) -> Result<Buffer<f64>> {
        self.flush()?;
        finite(self.matrix.as_slice(), "quadrature contraction")?;
        Ok(self.matrix)
    }
}

pub struct RealProjector<'a, T: Sample = f64> {
    context: Context,
    samples: &'a [T],
    orbitals: usize,
    points: usize,
    tile: usize,
    workers: Vec<RealWorkspace>,
    matrix: Buffer<f64>,
}
struct RealWorkspace {
    left: Buffer<f64>,
    right: Buffer<f64>,
    matrix: Buffer<f64>,
    _packing: Reservation,
}
struct RealTile<'a, T> {
    samples: &'a [T],
    field: &'a [f64],
    orbitals: usize,
    points: usize,
    tile: usize,
    begin: usize,
    end: usize,
}
impl<T> Clone for RealTile<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for RealTile<'_, T> {}
impl RealWorkspace {
    /// The samples stay in their stored precision; each tile widens its own
    /// values to float64 for the products.
    fn compute<T: Sample>(&mut self, input: RealTile<'_, T>) -> Result<()> {
        let RealTile {
            samples,
            field,
            orbitals: m,
            points,
            tile,
            begin,
            end,
        } = input;
        self.matrix.as_mut_slice().fill(0.);
        for start in (begin..end).step_by(tile) {
            let count = tile.min(end - start);
            for a in 0..m {
                for k in 0..count {
                    let value = samples[a * points + start + k].widen();
                    self.left.as_mut_slice()[k * m + a] = value;
                    self.right.as_mut_slice()[a * tile + k] = value * field[start + k];
                }
            }
            gemm::multiply(
                MatrixRef {
                    values: self.left.as_slice(),
                    rows: m,
                    columns: count,
                    leading: m,
                },
                MatrixRef {
                    values: self.right.as_slice(),
                    rows: count,
                    columns: m,
                    leading: tile,
                },
                MatrixMut {
                    values: self.matrix.as_mut_slice(),
                    rows: m,
                    columns: m,
                    leading: m,
                },
                1.,
                1.,
            )?;
        }
        Ok(())
    }
}

/// Local scalar-field projection into complex Bloch orbitals. Real GEMMs
/// accumulate both components without allocating a grid-sized pair tensor.
pub struct ComplexProjector<'a> {
    samples: &'a [C64],
    orbitals: usize,
    points: usize,
    tile: usize,
    left: Buffer<f64>,
    right: Buffer<f64>,
    real: Buffer<f64>,
    imaginary: Buffer<f64>,
    matrix: Buffer<C64>,
    _packing: Reservation,
}
impl<'a> ComplexProjector<'a> {
    pub fn new(ctx: &Context, samples: &'a [C64], orbitals: usize, points: usize) -> Result<Self> {
        if orbitals == 0
            || points == 0
            || samples.len() != product(&[orbitals, points])?
            || samples
                .iter()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(Error::Invalid(
                "complex orbital samples have inconsistent dimensions or values".into(),
            ));
        }
        let tile = points.min(2048);
        Ok(Self {
            samples,
            orbitals,
            points,
            tile,
            left: ctx.allocate(product(&[orbitals, tile])?)?,
            right: ctx.allocate(product(&[orbitals, tile])?)?,
            real: ctx.allocate(product(&[orbitals, orbitals])?)?,
            imaginary: ctx.allocate(product(&[orbitals, orbitals])?)?,
            matrix: ctx.allocate(product(&[orbitals, orbitals])?)?,
            _packing: ctx.reserve(product(&[orbitals, tile, 16])?)?,
        })
    }
    /// Column-major matrix `sum_k conj(phi[a,k]) field[k] phi[b,k]`.
    pub fn project(&mut self, field: &[f64]) -> Result<&[C64]> {
        if field.len() != self.points {
            return Err(Error::Invalid(
                "local field does not match the complex orbital grid".into(),
            ));
        }
        finite(field, "local field")?;
        let m = self.orbitals;
        self.real.as_mut_slice().fill(0.);
        self.imaginary.as_mut_slice().fill(0.);
        for start in (0..self.points).step_by(self.tile) {
            let count = self.tile.min(self.points - start);
            for (left_imag, right_imag, imaginary, sign) in [
                (false, false, false, 1.),
                (true, true, false, 1.),
                (false, true, true, 1.),
                (true, false, true, -1.),
            ] {
                for a in 0..m {
                    for k in 0..count {
                        let sample = self.samples[a * self.points + start + k];
                        self.left.as_mut_slice()[k * m + a] =
                            if left_imag { sample.im } else { sample.re };
                        self.right.as_mut_slice()[a * self.tile + k] =
                            (if right_imag { sample.im } else { sample.re }) * field[start + k];
                    }
                }
                let target = if imaginary {
                    &mut self.imaginary
                } else {
                    &mut self.real
                };
                gemm::multiply(
                    MatrixRef {
                        values: self.left.as_slice(),
                        rows: m,
                        columns: count,
                        leading: m,
                    },
                    MatrixRef {
                        values: self.right.as_slice(),
                        rows: count,
                        columns: m,
                        leading: self.tile,
                    },
                    MatrixMut {
                        values: target.as_mut_slice(),
                        rows: m,
                        columns: m,
                        leading: m,
                    },
                    sign,
                    1.,
                )?;
            }
        }
        for (i, value) in self.matrix.as_mut_slice().iter_mut().enumerate() {
            *value = C64::new(self.real.as_slice()[i], self.imaginary.as_slice()[i]);
        }
        Ok(self.matrix.as_slice())
    }
}

/// Grid diagonal `phi^T D conj(phi)` for a Hermitian density matrix.
/// Tiled real products retain complex coherence, including imaginary D.
pub fn hermitian_density(
    ctx: &Context,
    samples: &[C64],
    matrix: &DMatrix<C64>,
    points: usize,
) -> Result<Buffer<f64>> {
    let m = matrix.nrows();
    if m == 0
        || matrix.ncols() != m
        || points == 0
        || samples.len() != product(&[m, points])?
        || samples
            .iter()
            .chain(matrix.iter())
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(Error::Invalid(
            "complex density projection has invalid dimensions or values".into(),
        ));
    }
    let scale = matrix.iter().map(|v| v.norm()).fold(1f64, f64::max);
    if (0..m)
        .any(|a| (0..m).any(|b| (matrix[(a, b)] - matrix[(b, a)].conj()).norm() > 1e-10 * scale))
    {
        return Err(Error::Invalid(
            "grid density needs a Hermitian matrix".into(),
        ));
    }
    let tile = points.min(2048);
    let mut x = ctx.allocate::<f64>(product(&[m, tile])?)?;
    let mut y = ctx.allocate::<f64>(product(&[m, tile])?)?;
    let mut work = ctx.allocate::<f64>(product(&[m, tile])?)?;
    let mut a = ctx.allocate::<f64>(product(&[m, m])?)?;
    let mut b = ctx.allocate::<f64>(product(&[m, m])?)?;
    for j in 0..m {
        for i in 0..m {
            a.as_mut_slice()[i + j * m] = matrix[(i, j)].re;
            b.as_mut_slice()[i + j * m] = matrix[(i, j)].im;
        }
    }
    let imaginary_matrix = b.as_slice().iter().any(|v| *v != 0.);
    let imaginary_samples = samples.iter().any(|v| v.im != 0.);
    let _packing = ctx.reserve(product(&[m, tile, 16])?)?;
    let mut density = ctx.allocate::<f64>(points)?;
    for start in (0..points).step_by(tile) {
        let count = tile.min(points - start);
        for j in 0..count {
            for i in 0..m {
                let value = samples[i * points + start + j];
                x.as_mut_slice()[i + j * m] = value.re;
                y.as_mut_slice()[i + j * m] = value.im;
            }
        }
        for (operator, input, contraction, factor) in [
            (a.as_slice(), x.as_slice(), x.as_slice(), 1.),
            (a.as_slice(), y.as_slice(), y.as_slice(), 1.),
            (b.as_slice(), y.as_slice(), x.as_slice(), 2.),
        ]
        .into_iter()
        .take(if !imaginary_samples {
            1
        } else if !imaginary_matrix {
            2
        } else {
            3
        }) {
            gemm::multiply(
                MatrixRef {
                    values: operator,
                    rows: m,
                    columns: m,
                    leading: m,
                },
                MatrixRef {
                    values: input,
                    rows: m,
                    columns: count,
                    leading: m,
                },
                MatrixMut {
                    values: work.as_mut_slice(),
                    rows: m,
                    columns: count,
                    leading: m,
                },
                1.,
                0.,
            )?;
            for j in 0..count {
                density.as_mut_slice()[start + j] += factor
                    * (0..m)
                        .map(|i| contraction[i + j * m] * work.as_slice()[i + j * m])
                        .sum::<f64>();
            }
        }
    }
    Ok(density)
}
impl<'a, T: Sample> RealProjector<'a, T> {
    pub fn new(ctx: &Context, samples: &'a [T], orbitals: usize, points: usize) -> Result<Self> {
        if orbitals == 0 || points == 0 || samples.len() != product(&[orbitals, points])? {
            return Err(Error::Invalid(
                "orbital samples have inconsistent projection dimensions".into(),
            ));
        }
        if samples.iter().any(|v| !v.widen().is_finite()) {
            return Err(Error::Invalid(
                "orbital samples must contain only finite values".into(),
            ));
        }
        let tile = points.min(2048);
        let matrix = ctx.allocate(product(&[orbitals, orbitals])?)?;
        let per_worker = product(&[orbitals, tile, 32])?
            .checked_add(product(&[orbitals, orbitals, 8])?)
            .ok_or(Error::Overflow)?;
        let count = if orbitals >= 32 && points >= 8192 {
            ctx.threads()
                .min(points.div_ceil(tile))
                .min((ctx.memory_limit() - ctx.memory_used()) / per_worker / 2)
                .max(1)
        } else {
            1
        };
        let mut workers = Vec::with_capacity(count);
        for _ in 0..count {
            workers.push(RealWorkspace {
                left: ctx.allocate(product(&[orbitals, tile])?)?,
                right: ctx.allocate(product(&[orbitals, tile])?)?,
                matrix: ctx.allocate(product(&[orbitals, orbitals])?)?,
                _packing: ctx.reserve(product(&[orbitals, tile, 16])?)?,
            });
        }
        Ok(Self {
            context: ctx.clone(),
            samples,
            orbitals,
            points,
            tile,
            workers,
            matrix,
        })
    }
    /// Factorizations can return extra scratch storage as retained columns
    /// grow, while keeping the same mathematical projection.
    pub(crate) fn release_extra_workspaces(&mut self) -> bool {
        if self.workers.len() > 1 {
            self.workers.truncate(1);
            true
        } else {
            false
        }
    }
    /// Column-major local-field matrix. Independent tiles share only immutable
    /// input; each calling-thread BLAS kernel stays single-threaded. Partial
    /// matrices are reduced in a fixed order within the owner's thread budget.
    pub fn project(&mut self, field: &[f64]) -> Result<&[f64]> {
        if field.len() != self.points {
            return Err(Error::Invalid(
                "local field does not match the orbital grid".into(),
            ));
        }
        finite(field, "local field")?;
        let ctx = self.context.clone();
        ctx.run(|| {
            let input = RealTile {
                samples: self.samples,
                field,
                orbitals: self.orbitals,
                points: self.points,
                tile: self.tile,
                begin: 0,
                end: self.points,
            };
            if self.workers.len() == 1 {
                self.workers[0].compute(input)?;
            } else {
                let count = self.workers.len();
                let tiles = self.points.div_ceil(self.tile);
                std::thread::scope(|scope| {
                    let mut handles = Vec::with_capacity(count);
                    for (i, worker) in self.workers.iter_mut().enumerate() {
                        let begin = (i * tiles / count) * self.tile;
                        let end = (((i + 1) * tiles / count) * self.tile).min(self.points);
                        handles.push(mandacaru_kernels::runtime::spawn(scope, move || {
                            worker.compute(RealTile {
                                begin,
                                end,
                                ..input
                            })
                        }));
                    }
                    let mut outcome = Ok(());
                    for handle in handles {
                        let result = handle
                            .join()
                            .map_err(|_| Error::Kernel("local projection worker panicked".into()))
                            .and_then(|r| r);
                        if outcome.is_ok() {
                            outcome = result;
                        }
                    }
                    outcome
                })?;
            }
            self.matrix.as_mut_slice().fill(0.);
            for worker in &self.workers {
                for (to, from) in self
                    .matrix
                    .as_mut_slice()
                    .iter_mut()
                    .zip(worker.matrix.as_slice())
                {
                    *to += from;
                }
            }
            Ok(())
        })?;
        Ok(self.matrix.as_slice())
    }
}
