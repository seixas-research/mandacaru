//! All-electron molecular engine. Atomic units and physicists' ERI notation.
pub mod correlation;
pub mod dft;
pub mod forces;
pub mod integrals;
pub mod scf;
pub mod transform;
use crate::math::SpatialGrid;
use crate::Error;
pub use integrals::MolecularIntegrals;
pub use scf::{MeanField, ScfOptions};
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
