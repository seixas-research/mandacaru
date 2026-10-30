//! State-vector storage precision. A state is stored as complex128 (float64) or
//! complex64 (float32). Hamiltonian coefficients, sparse values, parameters and
//! every reduction (norms, inner products, energies, gradients) stay float64:
//! a state element is widened on read and rounded once on write.
use super::C64;
use crate::{math::storage::Precision, product, Buffer, Context, Error, Result};
use num_complex::Complex32;

/// One stored state element. The kernels of this module are written once,
/// generic over this trait; the two implementations differ only in rounding.
pub trait Amplitude: Copy + Default + PartialEq + Send + Sync + std::fmt::Debug + 'static {
    const PRECISION: Precision;
    /// Relative size below which a Taylor term no longer changes the sum.
    const TAYLOR_STOP: f64;
    /// Allowed drift of `<psi|psi>` from one after `applications` generator
    /// exponentials (each rounds the state once, so float32 grows as sqrt).
    fn norm_tolerance(applications: usize) -> f64;
    /// Allowed imaginary part of a Hermitian expectation value.
    const HERMITIAN_TOLERANCE: f64;
    /// Allowed deviation of a user-supplied state's norm from one. A complex64
    /// state must be normalized in float64 before it is narrowed: a norm summed
    /// in complex64 at 20+ qubits is off by about sqrt(d) * 6e-8.
    const INPUT_NORM_TOLERANCE: f64;
    fn widen(self) -> C64;
    fn narrow(value: C64) -> Self;
    /// Run `f` on a float64 row that accumulates a many-term sum, then store
    /// it once into `row`. For float64 the stored row is the accumulator
    /// itself and `acc` is unused (it may be empty).
    fn with_wide<R>(row: &mut [Self], acc: &mut [C64], f: impl FnOnce(&mut [C64]) -> R) -> R;
    /// Length of the float64 accumulator row a caller must provide.
    fn accumulator_len(dimension: usize) -> usize;
    /// Length of an extra float64 row a kernel needs when it reads a
    /// many-term sum before rounding it. For float32 the accumulator row
    /// itself serves, so none is allocated.
    fn own_wide_len(dimension: usize) -> usize;
    /// The float64 row of `own_wide_len`, or else the accumulator.
    fn wide_row<'a>(own: &'a mut [C64], acc: &'a mut [C64]) -> &'a mut [C64];
    fn wrap(buffer: Buffer<Self>) -> State;
    fn view(state: &State) -> Option<&[Self]>;
}

impl Amplitude for C64 {
    const PRECISION: Precision = Precision::Float64;
    const TAYLOR_STOP: f64 = 2e-16;
    const HERMITIAN_TOLERANCE: f64 = 1e-10;
    const INPUT_NORM_TOLERANCE: f64 = 1e-10;
    fn norm_tolerance(_applications: usize) -> f64 {
        1e-10
    }
    fn widen(self) -> C64 {
        self
    }
    fn narrow(value: C64) -> Self {
        value
    }
    fn with_wide<R>(row: &mut [Self], _acc: &mut [C64], f: impl FnOnce(&mut [C64]) -> R) -> R {
        row.fill(C64::default());
        f(row)
    }
    fn accumulator_len(_dimension: usize) -> usize {
        0
    }
    fn own_wide_len(dimension: usize) -> usize {
        dimension
    }
    fn wide_row<'a>(own: &'a mut [C64], _acc: &'a mut [C64]) -> &'a mut [C64] {
        own
    }
    fn wrap(buffer: Buffer<Self>) -> State {
        State::Float64(buffer)
    }
    fn view(state: &State) -> Option<&[Self]> {
        match state {
            State::Float64(buffer) => Some(buffer.as_slice()),
            State::Float32(_) => None,
        }
    }
}

/// Measured safety factor of the float32 norm and Hermiticity tolerances:
/// `tolerance = factor * 2^-24 * sqrt(number of rounded steps)`; the
/// measurement is recorded in `dev_plan/PRECISION_LOG.md`.
pub const NORM_FACTOR_FLOAT32: f64 = 64.;

impl Amplitude for Complex32 {
    const PRECISION: Precision = Precision::Float32;
    // 2 * 2^-24: the unit roundoff of float32, doubled like the float64 value.
    const TAYLOR_STOP: f64 = 1.2e-7;
    const HERMITIAN_TOLERANCE: f64 = NORM_FACTOR_FLOAT32 * 5.960_464_477_539_063e-8;
    const INPUT_NORM_TOLERANCE: f64 = 1e-5;
    fn norm_tolerance(applications: usize) -> f64 {
        NORM_FACTOR_FLOAT32 * 5.960_464_477_539_063e-8 * (applications.max(1) as f64).sqrt()
    }
    fn widen(self) -> C64 {
        C64::new(f64::from(self.re), f64::from(self.im))
    }
    fn narrow(value: C64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }
    fn with_wide<R>(row: &mut [Self], acc: &mut [C64], f: impl FnOnce(&mut [C64]) -> R) -> R {
        let acc = &mut acc[..row.len()];
        acc.fill(C64::default());
        let out = f(acc);
        for (to, &from) in row.iter_mut().zip(acc.iter()) {
            *to = Self::narrow(from);
        }
        out
    }
    fn accumulator_len(dimension: usize) -> usize {
        dimension
    }
    fn own_wide_len(_dimension: usize) -> usize {
        0
    }
    fn wide_row<'a>(_own: &'a mut [C64], acc: &'a mut [C64]) -> &'a mut [C64] {
        acc
    }
    fn wrap(buffer: Buffer<Self>) -> State {
        State::Float32(buffer)
    }
    fn view(state: &State) -> Option<&[Self]> {
        match state {
            State::Float32(buffer) => Some(buffer.as_slice()),
            State::Float64(_) => None,
        }
    }
}

/// A calculation-owned state (or a stack of rows) in either precision.
#[derive(Debug)]
pub enum State {
    Float64(Buffer<C64>),
    Float32(Buffer<Complex32>),
}
impl State {
    pub fn zeros(ctx: &Context, precision: Precision, len: usize) -> Result<Self> {
        Ok(match precision {
            Precision::Float64 => Self::Float64(ctx.allocate(len)?),
            Precision::Float32 => Self::Float32(ctx.allocate(len)?),
        })
    }
    pub fn precision(&self) -> Precision {
        match self {
            Self::Float64(_) => Precision::Float64,
            Self::Float32(_) => Precision::Float32,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Float64(buffer) => buffer.as_slice().len(),
            Self::Float32(buffer) => buffer.as_slice().len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// One amplitude, widened to complex128.
    pub fn amplitude(&self, index: usize) -> C64 {
        match self {
            Self::Float64(buffer) => buffer.as_slice()[index],
            Self::Float32(buffer) => buffer.as_slice()[index].widen(),
        }
    }
    /// The complex128 storage; a float32 state refuses instead of widening.
    pub fn complex128(&self) -> Result<&[C64]> {
        C64::view(self).ok_or_else(|| {
            Error::Invalid("this state is stored in float32 (complex64); it is not widened".into())
        })
    }
    /// The complex64 storage; a float64 state refuses instead of narrowing.
    pub fn complex64(&self) -> Result<&[Complex32]> {
        Complex32::view(self).ok_or_else(|| {
            Error::Invalid(
                "this state is stored in float64 (complex128); it is not narrowed".into(),
            )
        })
    }
    /// A float64 copy for an observer that needs complex128, charged to the
    /// budget. Quantum-method forces and checkpoints use this only under float64.
    pub fn widen(&self, ctx: &Context) -> Result<Buffer<C64>> {
        let mut out = ctx.allocate::<C64>(self.len())?;
        match self {
            Self::Float64(buffer) => out.as_mut_slice().copy_from_slice(buffer.as_slice()),
            Self::Float32(buffer) => {
                for (to, from) in out.as_mut_slice().iter_mut().zip(buffer.as_slice()) {
                    *to = from.widen();
                }
            }
        }
        Ok(out)
    }
    /// `sum |psi|^2` in float64.
    pub fn norm_squared(&self) -> f64 {
        match self {
            Self::Float64(buffer) => norm_squared(buffer.as_slice()),
            Self::Float32(buffer) => norm_squared(buffer.as_slice()),
        }
    }
    /// Rows `[row * dimension, (row + 1) * dimension)` of a stack, as a new state.
    pub fn row(&self, ctx: &Context, row: usize, dimension: usize) -> Result<Self> {
        let range = row * dimension..(row + 1) * dimension;
        Ok(match self {
            Self::Float64(buffer) => {
                let mut out = ctx.allocate::<C64>(dimension)?;
                out.as_mut_slice()
                    .copy_from_slice(&buffer.as_slice()[range]);
                Self::Float64(out)
            }
            Self::Float32(buffer) => {
                let mut out = ctx.allocate::<Complex32>(dimension)?;
                out.as_mut_slice()
                    .copy_from_slice(&buffer.as_slice()[range]);
                Self::Float32(out)
            }
        })
    }
}

pub fn norm_squared<A: Amplitude>(a: &[A]) -> f64 {
    a.iter().map(|c| c.widen().norm_sqr()).sum()
}
/// `sum conj(a) b` in float64.
pub fn inner<A: Amplitude>(a: &[A], b: &[A]) -> C64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.widen().conj() * y.widen())
        .sum()
}
/// `sum conj(a) b` of a stored state with a float64 row.
pub fn inner_wide<A: Amplitude>(a: &[A], b: &[C64]) -> C64 {
    a.iter().zip(b).map(|(x, y)| x.widen().conj() * y).sum()
}
pub fn finite<A: Amplitude>(a: &[A]) -> bool {
    a.iter().all(|c| {
        let c = c.widen();
        c.re.is_finite() && c.im.is_finite()
    })
}
/// Allocate a stack of `rows` states of `dimension` amplitudes.
pub fn stack<A: Amplitude>(ctx: &Context, rows: usize, dimension: usize) -> Result<Buffer<A>> {
    ctx.allocate(product(&[rows, dimension])?)
}

/// The smallest settings a complex64 state resolves. A setting below a floor
/// would chase rounding noise, so it is refused. Measured on 12 frozen 8-14
/// qubit molecular Hamiltonians (`dev_plan/PRECISION_LOG.md`, section WP2):
/// - energy: BFGS with `tol` 1e-9 fails in every inner optimization (up to
///   4420 steps against 544), 1e-8 in some, while 1e-7 behaves like float64
///   (SLSQP resolved all of 1e-9..1e-3); the Rayleigh-quotient energy
///   differs from float64 by at most 3e-10 Ha at an ADAPT stop and by at most
///   7.7e-7 Ha at arbitrary parameters. The floor is 10x the smallest tol
///   that behaved.
/// - gradient: a pool gradient differs from float64 by at most 3.3e-8 at an
///   ADAPT state and a parameter derivative by at most 4.9e-7 at arbitrary
///   parameters; the floor is 30x the former.
///
/// Rounding noise grows with the size of the Hamiltonian: the molecules
/// measured have total energies down to -30.2 Ha.
#[derive(Clone, Copy, Debug)]
pub struct Floors {
    /// Optimizer `tol` and ADAPT `convergence["energy"]` (Hartree).
    pub energy: f64,
    /// ADAPT pool-gradient threshold (Hartree per radian).
    pub gradient: f64,
}
pub const FLOAT32_FLOORS: Floors = Floors {
    energy: 1e-6,
    gradient: 1e-6,
};
pub fn floors(precision: Precision) -> Option<Floors> {
    match precision {
        Precision::Float64 => None,
        Precision::Float32 => Some(FLOAT32_FLOORS),
    }
}
