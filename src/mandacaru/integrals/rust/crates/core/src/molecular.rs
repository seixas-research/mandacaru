//! All-electron molecular engine. Atomic units and physicists' ERI notation.
pub mod basis;
pub mod fft;
pub mod grid;
pub mod integrals;
pub mod scf;
pub mod transform;
use crate::Error;
pub use grid::SpatialGrid;
pub use integrals::MolecularIntegrals;
pub use scf::{MeanField, ScfOptions};
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
