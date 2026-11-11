//! All-electron molecular engine. Atomic units and physicists' ERI notation.
pub mod active;
pub mod bloch_path;
pub mod correlation;
pub mod dft;
pub mod double_factorized;
mod exchange_pieces;
pub mod forces;
pub mod hubbard;
pub mod integrals;
pub mod local_correlation;
pub mod observables;
pub mod occupation;
pub mod periodic;
pub mod spinor_dft;
mod paw_forces;
pub mod response;
pub mod scf;
pub mod symmetry;
pub mod transform;
pub mod variational_gradient;
use crate::math::SpatialGrid;
use crate::Error;
pub use exchange_pieces::ExchangePieces;
pub use integrals::MolecularIntegrals;
pub use scf::{MeanField, ScfOptions};
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
