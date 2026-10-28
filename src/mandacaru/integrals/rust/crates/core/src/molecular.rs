//! All-electron molecular engine. Atomic units and physicists' ERI notation.
pub mod atom;
pub mod basis;
pub mod correlation;
pub mod dft;
pub mod fft;
pub mod forces;
pub mod grid;
pub mod integrals;
pub mod nao_ae;
pub mod nonlocal;
pub mod radial;
pub mod reciprocal;
pub mod scf;
pub mod transform;
pub mod xc;
use crate::Error;
pub use grid::SpatialGrid;
pub use integrals::MolecularIntegrals;
pub use scf::{MeanField, ScfOptions};
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
