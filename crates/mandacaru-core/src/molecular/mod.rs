//! All-electron molecular engine. Atomic units and physicists' ERI notation.
pub mod active;
pub mod correlation;
pub mod dft;
pub mod forces;
pub mod integrals;
pub mod local_correlation;
pub mod observables;
pub mod response;
pub mod scf;
pub mod symmetry;
pub mod transform;
use crate::math::SpatialGrid;
use crate::Error;
pub use integrals::MolecularIntegrals;
pub use scf::{MeanField, ScfOptions};
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
