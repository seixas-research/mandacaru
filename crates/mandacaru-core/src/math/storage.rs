//! Calculation-owned sample storage with double-precision accumulation views.
use crate::{Buffer, Context, Error, Result};
use num_complex::{Complex32, Complex64};
use std::ops::Deref;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    #[default]
    Double,
    Single,
}
impl Precision {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "double" => Ok(Self::Double),
            "single" => Ok(Self::Single),
            _ => Err(Error::Invalid(
                "precision must be 'double' or 'single'".into(),
            )),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Double => "double",
            Self::Single => "single",
        }
    }
}
pub enum RealSamples {
    Double(Buffer<f64>),
    Single(Buffer<f32>),
}
impl From<Buffer<f64>> for RealSamples {
    fn from(values: Buffer<f64>) -> Self {
        Self::Double(values)
    }
}
impl RealSamples {
    pub fn value(&self, index: usize) -> f64 {
        match self {
            Self::Double(values) => values.as_slice()[index],
            Self::Single(values) => f64::from(values.as_slice()[index]),
        }
    }
    pub fn precision(&self) -> Precision {
        match self {
            Self::Double(_) => Precision::Double,
            Self::Single(_) => Precision::Single,
        }
    }
    pub fn into_precision(self, context: &Context, precision: Precision) -> Result<Self> {
        match (self, precision) {
            (Self::Double(values), Precision::Single) => {
                let mut compact = context.allocate::<f32>(values.as_slice().len())?;
                for (to, &from) in compact.as_mut_slice().iter_mut().zip(values.as_slice()) {
                    *to = from as f32;
                    if !to.is_finite() {
                        return Err(Error::Invalid(
                            "orbital samples exceed the finite range of single precision".into(),
                        ));
                    }
                }
                Ok(Self::Single(compact))
            }
            (Self::Single(values), Precision::Double) => {
                let mut expanded = context.allocate::<f64>(values.as_slice().len())?;
                for (to, &from) in expanded.as_mut_slice().iter_mut().zip(values.as_slice()) {
                    *to = f64::from(from);
                }
                Ok(Self::Double(expanded))
            }
            (values, _) => Ok(values),
        }
    }
    /// Expansion, when needed, is temporary and charged to the same budget.
    /// Integrals and SCF matrices always accumulate in double precision.
    pub fn read<'a>(&'a self, context: &Context) -> Result<SampleRead<'a>> {
        match self {
            Self::Double(values) => Ok(SampleRead::Borrowed(values.as_slice())),
            Self::Single(values) => {
                let mut expanded = context.allocate::<f64>(values.as_slice().len())?;
                for (to, &from) in expanded.as_mut_slice().iter_mut().zip(values.as_slice()) {
                    *to = f64::from(from);
                }
                Ok(SampleRead::Expanded(expanded))
            }
        }
    }
}
pub enum SampleRead<'a> {
    Borrowed(&'a [f64]),
    Expanded(Buffer<f64>),
}
impl Deref for SampleRead<'_> {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        match self {
            Self::Borrowed(values) => values,
            Self::Expanded(values) => values.as_slice(),
        }
    }
}

pub enum ComplexSamples {
    Double(Buffer<Complex64>),
    Single(Buffer<Complex32>),
    ArchivedDouble(super::archive::Archive<Complex64>),
    ArchivedSingle(super::archive::Archive<Complex32>),
}
impl ComplexSamples {
    pub fn copy_single(
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
            (Precision::Single, true) => Ok(Self::ArchivedSingle(Archive::create(
                ctx,
                values.len(),
                |i| Ok(values[i]),
            )?)),
            (Precision::Single, false) => Ok(Self::Single(ctx.copy_slice(values)?)),
            (Precision::Double, true) => Ok(Self::ArchivedDouble(Archive::create(
                ctx,
                values.len(),
                |i| {
                    Ok(Complex64::new(
                        f64::from(values[i].re),
                        f64::from(values[i].im),
                    ))
                },
            )?)),
            (Precision::Double, false) => {
                let mut out = ctx.allocate::<Complex64>(values.len())?;
                for (to, from) in out.as_mut_slice().iter_mut().zip(values) {
                    *to = Complex64::new(f64::from(from.re), f64::from(from.im));
                }
                Ok(Self::Double(out))
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
            Precision::Double => Ok(Self::ArchivedDouble(Archive::create(
                ctx,
                values.len(),
                |i| Ok(values[i]),
            )?)),
            Precision::Single => Ok(Self::ArchivedSingle(Archive::create(
                ctx,
                values.len(),
                |i| {
                    let v = Complex32::new(values[i].re as f32, values[i].im as f32);
                    if !v.re.is_finite() || !v.im.is_finite() {
                        return Err(Error::Invalid(
                            "Bloch samples exceed the finite range of single precision".into(),
                        ));
                    }
                    Ok(v)
                },
            )?)),
        }
    }
    pub fn finite(&self) -> bool {
        match self {
            Self::Double(v) => v
                .as_slice()
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite()),
            Self::Single(v) => v
                .as_slice()
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite()),
            Self::ArchivedDouble(_) | Self::ArchivedSingle(_) => true, // checked before archive creation
        }
    }
    pub fn copy(ctx: &Context, values: &[Complex64], precision: Precision) -> Result<Self> {
        match precision {
            Precision::Double => Ok(Self::Double(ctx.copy_slice(values)?)),
            Precision::Single => {
                let mut compact = ctx.allocate::<Complex32>(values.len())?;
                for (to, from) in compact.as_mut_slice().iter_mut().zip(values) {
                    *to = Complex32::new(from.re as f32, from.im as f32);
                    if !to.re.is_finite() || !to.im.is_finite() {
                        return Err(Error::Invalid(
                            "Bloch samples exceed the finite range of single precision".into(),
                        ));
                    }
                }
                Ok(Self::Single(compact))
            }
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Double(v) => v.as_slice().len(),
            Self::Single(v) => v.as_slice().len(),
            Self::ArchivedDouble(v) => v.len(),
            Self::ArchivedSingle(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn precision(&self) -> Precision {
        match self {
            Self::Double(_) | Self::ArchivedDouble(_) => Precision::Double,
            Self::Single(_) | Self::ArchivedSingle(_) => Precision::Single,
        }
    }
    pub fn into_precision(self, ctx: &Context, precision: Precision) -> Result<Self> {
        if self.precision() == precision {
            return Ok(self);
        }
        match self {
            Self::Double(values) => Self::copy(ctx, values.as_slice(), precision),
            Self::Single(values) => {
                let mut expanded = ctx.allocate::<Complex64>(values.as_slice().len())?;
                for (to, from) in expanded.as_mut_slice().iter_mut().zip(values.as_slice()) {
                    *to = Complex64::new(f64::from(from.re), f64::from(from.im));
                }
                Ok(Self::Double(expanded))
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
            Self::Double(values) => Ok(ComplexRead::Borrowed(&values.as_slice()[start..end])),
            Self::Single(values) => {
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
            Self::ArchivedDouble(values) => {
                let mut out = ctx.allocate::<Complex64>(length)?;
                values.read_range_into(start, out.as_mut_slice(), |v| v)?;
                Ok(ComplexRead::Expanded(out))
            }
            Self::ArchivedSingle(values) => {
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
