//! Periodic engine and portable crystal snapshots. Atomic units throughout.
pub mod analysis;
pub mod ewald;
pub mod fields;
pub mod finite_size;
pub mod hubbard;
pub mod hubbard_gradient;
pub mod hubbard_table;
pub mod input;
pub mod screening;
pub mod screened_exchange;
pub mod solver;
pub mod space_group;
pub mod spectrum;
pub mod spinor;
pub mod stress;
pub mod wannier;
use crate::Error;
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
pub use input::{Crystal, KPoint};
pub use solver::{Options, PeriodicResult};
