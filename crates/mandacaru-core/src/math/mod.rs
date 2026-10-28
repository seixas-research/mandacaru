//! Shared grids, transforms, quadrature and linear algebra.
pub mod constrained;
pub mod fft;
pub mod grid;
pub mod index;
pub mod linalg;
pub mod radial;
pub mod reciprocal;

pub use grid::SpatialGrid;

fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::Invalid(message.into())
}
