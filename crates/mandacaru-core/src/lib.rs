//! Scientific engine foundation. This crate has no dependency on Python or NumPy.

pub mod basis;
pub mod context;
pub mod execution;
pub mod geometry;
pub mod kernels;
pub mod math;
pub mod models;
pub mod molecular;
pub mod parquet;
pub mod paw;
pub mod periodic;
pub mod persistence;
pub mod quantum;
pub mod snapshot_table;
pub mod units;
pub mod volumetric;

pub use context::{Buffer, Context, Reservation};
pub use geometry::Geometry;
pub use mandacaru_kernels::bloch::Complex64;
pub use mandacaru_kernels::runtime::WorkerStats;
pub use math::index::Grid;
pub use units::LengthUnit;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    Invalid(String),
    /// A well-formed request for something the engine does not implement: the caller
    /// raises it, never substitutes another calculation.
    Unsupported(String),
    Overflow,
    Budget {
        requested: usize,
        available: usize,
    },
    Allocation,
    Kernel(String),
    /// An external executor ended the run on purpose (for example, its run
    /// budget is exhausted). Not a fault: solvers that can keep their best
    /// point catch it, every other caller propagates it.
    Interrupted(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message)
            | Self::Unsupported(message)
            | Self::Kernel(message)
            | Self::Interrupted(message) => {
                f.write_str(message)
            }
            Self::Overflow => f.write_str("array shape or byte count exceeds addressable size"),
            Self::Budget { requested, available } => write!(f, "calculation memory budget exceeded: requested {requested} bytes, {available} available"),
            Self::Allocation => f.write_str("could not allocate calculation buffer"),
        }
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

pub fn product(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v).ok_or(Error::Overflow))
}

pub fn finite(values: &[f64], name: &str) -> Result<()> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(Error::Invalid(format!(
            "{name} must contain only finite values"
        )))
    }
}
