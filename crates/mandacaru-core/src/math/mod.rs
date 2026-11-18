//! Shared grids, transforms, quadrature and linear algebra.
pub mod angular;
pub mod assignment;
pub mod autodiff;
pub mod constrained;
pub mod csr;
pub mod davidson;
pub mod fft;
pub mod gemm;
pub mod grid;
pub mod harmonics;
pub mod index;
pub mod isosurface;
pub mod lanczos;
pub mod lattice;
pub mod linalg;
pub mod localization;
pub mod numpy_rules;
pub mod partial_charges;
pub mod peaks;
pub mod projection;
pub mod propagator;
pub mod radial;
pub mod random;
pub mod reciprocal;
pub mod rigid;
pub mod roots;
pub mod separable;
pub mod smearing;
pub mod special;
pub mod spline;
pub mod storage;
pub mod symmetric_solve;
pub mod symmetry;
pub mod viewer;
pub mod wigner_seitz;

pub use grid::SpatialGrid;

fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::Invalid(message.into())
}
pub mod archive;
