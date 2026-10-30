//! Calculation-owned sample storage with float64 accumulation views.
use crate::{Buffer, Context, Error, Result};
use num_complex::{Complex32, Complex64};
use std::ops::Deref;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    #[default]
    Float64,
    Float32,
}
impl Precision {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "float64" => Ok(Self::Float64),
            "float32" => Ok(Self::Float32),
            _ => Err(Error::Invalid(
                "precision must be 'float32' or 'float64'".into(),
            )),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Float64 => "float64",
            Self::Float32 => "float32",
        }
    }
}
/// A stored real sample component. Row kernels are generic over it and widen
/// one element (or one block, see [`SampleBlocks`]) to float64 as they
/// accumulate; no operation expands a whole stack.
pub trait Sample: Copy + Send + Sync + Into<f64> + 'static {
    const PRECISION: Precision;
    /// Round once from the float64 value.
    fn narrow(value: f64) -> Self;
    fn widen(self) -> f64 {
        self.into()
    }
}
impl Sample for f64 {
    const PRECISION: Precision = Precision::Float64;
    fn narrow(value: f64) -> Self {
        value
    }
}
impl Sample for f32 {
    const PRECISION: Precision = Precision::Float32;
    fn narrow(value: f64) -> Self {
        value as f32
    }
}
/// Typed view of a stored stack: a kernel matches once and runs generic over
/// the component type.
#[derive(Clone, Copy)]
pub enum Rows<'a> {
    Float64(&'a [f64]),
    Float32(&'a [f32]),
}
impl<'a> Rows<'a> {
    pub fn len(&self) -> usize {
        match self {
            Self::Float64(v) => v.len(),
            Self::Float32(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The float64 storage itself, borrowed; `None` for a float32 stack. This
    /// never expands anything.
    pub fn float64(self) -> Option<&'a [f64]> {
        match self {
            Self::Float64(v) => Some(v),
            Self::Float32(_) => None,
        }
    }
}
/// Run `$body` once per stored component type with `$slice` bound to the
/// typed slice (`&[f64]` or `&[f32]`); the body widens with `f64::from`.
macro_rules! with_rows {
    ($rows:expr, |$slice:ident| $body:expr) => {
        match $rows {
            $crate::math::storage::Rows::Float64($slice) => $body,
            $crate::math::storage::Rows::Float32($slice) => $body,
        }
    };
}
pub(crate) use with_rows;

/// Orbital sample stack in the calculation's precision, filled row by row.
pub enum RealSamples {
    Float64(Buffer<f64>),
    Float32(Buffer<f32>),
}
impl RealSamples {
    /// A zero stack of `len` values: the precision is a construction argument.
    pub fn zeros(ctx: &Context, precision: Precision, len: usize) -> Result<Self> {
        Ok(match precision {
            Precision::Float64 => Self::Float64(ctx.allocate::<f64>(len)?),
            Precision::Float32 => Self::Float32(ctx.allocate::<f32>(len)?),
        })
    }
    /// A stack rounded once from float64 values.
    pub fn from_slice(ctx: &Context, precision: Precision, values: &[f64]) -> Result<Self> {
        let mut out = Self::zeros(ctx, precision, values.len())?;
        out.fill_range(0, values)?;
        Ok(out)
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Float64(v) => v.as_slice().len(),
            Self::Float32(v) => v.as_slice().len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn precision(&self) -> Precision {
        match self {
            Self::Float64(_) => Precision::Float64,
            Self::Float32(_) => Precision::Float32,
        }
    }
    pub fn value(&self, index: usize) -> f64 {
        match self {
            Self::Float64(values) => values.as_slice()[index],
            Self::Float32(values) => f64::from(values.as_slice()[index]),
        }
    }
    pub fn rows(&self) -> Rows<'_> {
        match self {
            Self::Float64(v) => Rows::Float64(v.as_slice()),
            Self::Float32(v) => Rows::Float32(v.as_slice()),
        }
    }
    /// Round `values` once into the stack starting at `offset`. A value that
    /// is not finite, or leaves the finite range of float32, is refused.
    pub fn fill_range(&mut self, offset: usize, values: &[f64]) -> Result<()> {
        let end = offset
            .checked_add(values.len())
            .filter(|end| *end <= self.len())
            .ok_or_else(|| Error::Invalid("sample range is out of bounds".into()))?;
        match self {
            Self::Float64(to) => {
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(Error::Invalid("orbital samples must be finite".into()));
                }
                to.as_mut_slice()[offset..end].copy_from_slice(values);
            }
            Self::Float32(to) => {
                for (to, &from) in to.as_mut_slice()[offset..end].iter_mut().zip(values) {
                    *to = from as f32;
                    if !to.is_finite() {
                        return Err(Error::Invalid(
                            "orbital samples exceed the finite range of float32".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
    /// Row `row` of a stack with `points` values per row.
    pub fn fill_row(&mut self, row: usize, points: usize, values: &[f64]) -> Result<()> {
        if values.len() != points {
            return Err(Error::Invalid(
                "a sample row must hold one value per grid point".into(),
            ));
        }
        self.fill_range(row.checked_mul(points).ok_or(Error::Overflow)?, values)
    }
    /// Widen `out.len()` values starting at `offset` into float64.
    pub fn widen_range(&self, offset: usize, out: &mut [f64]) -> Result<()> {
        let end = offset
            .checked_add(out.len())
            .filter(|end| *end <= self.len())
            .ok_or_else(|| Error::Invalid("sample range is out of bounds".into()))?;
        match self {
            Self::Float64(from) => out.copy_from_slice(&from.as_slice()[offset..end]),
            Self::Float32(from) => {
                for (to, &from) in out.iter_mut().zip(&from.as_slice()[offset..end]) {
                    *to = f64::from(from);
                }
            }
        }
        Ok(())
    }
}

/// Point-block reader of a stack of `rows` rows with `points` values each. It
/// widens one block of every row into a float64 buffer of `rows x block`
/// values (row `r` at `r * stride()`), which is all an accumulation needs: the
/// stack itself is never expanded.
pub struct SampleBlocks<'a> {
    source: &'a RealSamples,
    rows: usize,
    points: usize,
    block: usize,
    count: usize,
    values: Buffer<f64>,
}
impl<'a> SampleBlocks<'a> {
    pub const BLOCK: usize = 2048;
    pub fn new(ctx: &Context, source: &'a RealSamples, rows: usize, points: usize) -> Result<Self> {
        if rows == 0
            || points == 0
            || source.len() != rows.checked_mul(points).ok_or(Error::Overflow)?
        {
            return Err(Error::Invalid(
                "sample stack does not match the block reader dimensions".into(),
            ));
        }
        let block = points.min(Self::BLOCK);
        Ok(Self {
            source,
            rows,
            points,
            block,
            count: 0,
            values: ctx.allocate(rows.checked_mul(block).ok_or(Error::Overflow)?)?,
        })
    }
    /// Points per block (and the row stride of [`values`](Self::values)).
    pub fn block(&self) -> usize {
        self.block
    }
    pub fn stride(&self) -> usize {
        self.block
    }
    /// Widen points `start..start + count` of every row; returns `count`.
    pub fn load(&mut self, start: usize) -> Result<usize> {
        if start >= self.points {
            return Err(Error::Invalid("sample block starts past the grid".into()));
        }
        let count = self.block.min(self.points - start);
        for row in 0..self.rows {
            self.source.widen_range(
                row * self.points + start,
                &mut self.values.as_mut_slice()[row * self.block..row * self.block + count],
            )?;
        }
        self.count = count;
        Ok(count)
    }
    /// All rows of the loaded block, row `r` at `r * stride()`.
    pub fn values(&self) -> &[f64] {
        self.values.as_slice()
    }
    pub fn row(&self, row: usize) -> &[f64] {
        &self.values.as_slice()[row * self.block..row * self.block + self.count]
    }
}

pub enum ComplexSamples {
    Float64(Buffer<Complex64>),
    Float32(Buffer<Complex32>),
    ArchivedFloat64(super::archive::Archive<Complex64>),
    ArchivedFloat32(super::archive::Archive<Complex32>),
}
impl ComplexSamples {
    pub fn copy_float32(
        ctx: &Context,
        values: &[Complex32],
        precision: Precision,
        archive: bool,
    ) -> Result<Self> {
        use super::archive::Archive;
        if values
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(Error::Invalid("Bloch samples must be finite".into()));
        }
        match (precision, archive) {
            (Precision::Float32, true) => Ok(Self::ArchivedFloat32(Archive::create(
                ctx,
                values.len(),
                |i| Ok(values[i]),
            )?)),
            (Precision::Float32, false) => Ok(Self::Float32(ctx.copy_slice(values)?)),
            (Precision::Float64, true) => Ok(Self::ArchivedFloat64(Archive::create(
                ctx,
                values.len(),
                |i| {
                    Ok(Complex64::new(
                        f64::from(values[i].re),
                        f64::from(values[i].im),
                    ))
                },
            )?)),
            (Precision::Float64, false) => {
                let mut out = ctx.allocate::<Complex64>(values.len())?;
                for (to, from) in out.as_mut_slice().iter_mut().zip(values) {
                    *to = Complex64::new(f64::from(from.re), f64::from(from.im));
                }
                Ok(Self::Float64(out))
            }
        }
    }
    pub fn archived_copy(
        ctx: &Context,
        values: &[Complex64],
        precision: Precision,
    ) -> Result<Self> {
        use super::archive::Archive;
        if values
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(Error::Invalid("Bloch samples must be finite".into()));
        }
        match precision {
            Precision::Float64 => Ok(Self::ArchivedFloat64(Archive::create(
                ctx,
                values.len(),
                |i| Ok(values[i]),
            )?)),
            Precision::Float32 => Ok(Self::ArchivedFloat32(Archive::create(
                ctx,
                values.len(),
                |i| {
                    let v = Complex32::new(values[i].re as f32, values[i].im as f32);
                    if !v.re.is_finite() || !v.im.is_finite() {
                        return Err(Error::Invalid(
                            "Bloch samples exceed the finite range of float32".into(),
                        ));
                    }
                    Ok(v)
                },
            )?)),
        }
    }
    pub fn finite(&self) -> bool {
        match self {
            Self::Float64(v) => v
                .as_slice()
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite()),
            Self::Float32(v) => v
                .as_slice()
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite()),
            Self::ArchivedFloat64(_) | Self::ArchivedFloat32(_) => true, // checked before archive creation
        }
    }
    pub fn copy(ctx: &Context, values: &[Complex64], precision: Precision) -> Result<Self> {
        match precision {
            Precision::Float64 => Ok(Self::Float64(ctx.copy_slice(values)?)),
            Precision::Float32 => {
                let mut compact = ctx.allocate::<Complex32>(values.len())?;
                for (to, from) in compact.as_mut_slice().iter_mut().zip(values) {
                    *to = Complex32::new(from.re as f32, from.im as f32);
                    if !to.re.is_finite() || !to.im.is_finite() {
                        return Err(Error::Invalid(
                            "Bloch samples exceed the finite range of float32".into(),
                        ));
                    }
                }
                Ok(Self::Float32(compact))
            }
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Float64(v) => v.as_slice().len(),
            Self::Float32(v) => v.as_slice().len(),
            Self::ArchivedFloat64(v) => v.len(),
            Self::ArchivedFloat32(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn precision(&self) -> Precision {
        match self {
            Self::Float64(_) | Self::ArchivedFloat64(_) => Precision::Float64,
            Self::Float32(_) | Self::ArchivedFloat32(_) => Precision::Float32,
        }
    }
    pub fn into_precision(self, ctx: &Context, precision: Precision) -> Result<Self> {
        if self.precision() == precision {
            return Ok(self);
        }
        match self {
            Self::Float64(values) => Self::copy(ctx, values.as_slice(), precision),
            Self::Float32(values) => {
                let mut expanded = ctx.allocate::<Complex64>(values.as_slice().len())?;
                for (to, from) in expanded.as_mut_slice().iter_mut().zip(values.as_slice()) {
                    *to = Complex64::new(f64::from(from.re), f64::from(from.im));
                }
                Ok(Self::Float64(expanded))
            }
            values => Self::archived_copy(ctx, &values.read(ctx)?, precision),
        }
    }
    pub fn read<'a>(&'a self, ctx: &Context) -> Result<ComplexRead<'a>> {
        self.read_range(ctx, 0, self.len())
    }
    pub fn read_range<'a>(
        &'a self,
        ctx: &Context,
        start: usize,
        length: usize,
    ) -> Result<ComplexRead<'a>> {
        let end = start
            .checked_add(length)
            .filter(|end| *end <= self.len())
            .ok_or_else(|| Error::Invalid("sample read range is out of bounds".into()))?;
        match self {
            Self::Float64(values) => Ok(ComplexRead::Borrowed(&values.as_slice()[start..end])),
            Self::Float32(values) => {
                let mut expanded = ctx.allocate::<Complex64>(length)?;
                for (to, from) in expanded
                    .as_mut_slice()
                    .iter_mut()
                    .zip(&values.as_slice()[start..end])
                {
                    *to = Complex64::new(f64::from(from.re), f64::from(from.im));
                }
                Ok(ComplexRead::Expanded(expanded))
            }
            Self::ArchivedFloat64(values) => {
                let mut out = ctx.allocate::<Complex64>(length)?;
                values.read_range_into(start, out.as_mut_slice(), |v| v)?;
                Ok(ComplexRead::Expanded(out))
            }
            Self::ArchivedFloat32(values) => {
                let mut out = ctx.allocate::<Complex64>(length)?;
                values.read_range_into(start, out.as_mut_slice(), |v| {
                    Complex64::new(f64::from(v.re), f64::from(v.im))
                })?;
                Ok(ComplexRead::Expanded(out))
            }
        }
    }
}
pub enum ComplexRead<'a> {
    Borrowed(&'a [Complex64]),
    Expanded(Buffer<Complex64>),
}
impl Deref for ComplexRead<'_> {
    type Target = [Complex64];
    fn deref(&self) -> &[Complex64] {
        match self {
            Self::Borrowed(v) => v,
            Self::Expanded(v) => v.as_slice(),
        }
    }
}

pub enum StoredMatrix {
    Resident {
        value: nalgebra::DMatrix<Complex64>,
        _memory: crate::Reservation,
    },
    Archived {
        rows: usize,
        columns: usize,
        values: super::archive::Archive<Complex64>,
    },
}
impl StoredMatrix {
    pub fn new(ctx: &Context, value: nalgebra::DMatrix<Complex64>, archive: bool) -> Result<Self> {
        if archive {
            Ok(Self::Archived {
                rows: value.nrows(),
                columns: value.ncols(),
                values: super::archive::Archive::create(ctx, value.len(), |i| Ok(value[i]))?,
            })
        } else {
            let memory = ctx.reserve(crate::product(&[value.len(), 16])?)?;
            Ok(Self::Resident {
                value,
                _memory: memory,
            })
        }
    }
}
pub enum MatrixRead<'a> {
    Borrowed(&'a nalgebra::DMatrix<Complex64>),
    Owned {
        value: nalgebra::DMatrix<Complex64>,
        _memory: crate::Reservation,
    },
}
impl Deref for MatrixRead<'_> {
    type Target = nalgebra::DMatrix<Complex64>;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(v) => v,
            Self::Owned { value, .. } => value,
        }
    }
}
pub trait MatrixSource {
    fn matrix<'a>(&'a self, ctx: &Context) -> Result<MatrixRead<'a>>;
}
impl MatrixSource for nalgebra::DMatrix<Complex64> {
    fn matrix<'a>(&'a self, _ctx: &Context) -> Result<MatrixRead<'a>> {
        Ok(MatrixRead::Borrowed(self))
    }
}
impl MatrixSource for StoredMatrix {
    fn matrix<'a>(&'a self, ctx: &Context) -> Result<MatrixRead<'a>> {
        match self {
            Self::Resident { value, .. } => Ok(MatrixRead::Borrowed(value)),
            Self::Archived {
                rows,
                columns,
                values,
            } => {
                let memory = ctx.reserve(crate::product(&[*rows, *columns, 16])?)?;
                let mut value = nalgebra::DMatrix::zeros(*rows, *columns);
                values.read_into(value.as_mut_slice(), |v| v)?;
                Ok(MatrixRead::Owned {
                    value,
                    _memory: memory,
                })
            }
        }
    }
}
