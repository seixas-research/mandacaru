//! Periodic engine and portable crystal snapshots. Atomic units throughout.
pub mod analysis;
pub mod ewald;
pub mod input;
pub mod solver;
pub mod spectrum;
use crate::Error;
fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
pub use input::{Crystal, KPoint};
pub use solver::{Options, PeriodicResult};
