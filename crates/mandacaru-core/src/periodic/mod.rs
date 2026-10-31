//! Periodic engine and portable crystal snapshots. Atomic units throughout.
pub mod analysis;
pub mod ewald;
pub mod fields;
pub mod finite_size;
pub mod hubbard;
pub mod hubbard_table;
pub mod input;
pub mod screening;
pub mod solver;
pub mod spectrum;
pub mod spinor;
pub mod wannier;
use crate::Error;
fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
pub use input::{Crystal, KPoint};
pub use solver::{Options, PeriodicResult};
