//! Shared grids, transforms, quadrature and linear algebra.
pub mod angular;
pub mod assignment;
pub mod autodiff;
pub mod constrained;
pub mod fft;
pub mod gemm;
pub mod grid;
pub mod index;
pub mod linalg;
pub mod projection;
pub mod radial;
pub mod random;
pub mod reciprocal;
pub mod rigid;
pub mod roots;
pub mod separable;
pub mod special;
pub mod spline;
pub mod storage;
pub mod symmetry;
pub mod wigner_seitz;

pub use grid::SpatialGrid;

fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::Invalid(message.into())
}
pub mod archive;
