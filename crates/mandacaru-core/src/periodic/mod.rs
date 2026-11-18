//! Periodic engine and portable crystal snapshots. Atomic units throughout.
pub mod analysis;
pub mod berry;
pub mod bloch_mesh;
pub mod description;
pub mod ewald;
pub mod exchange_state;
pub mod fields;
pub mod finite_size;
pub mod gradient;
pub mod hubbard;
pub mod hubbard_gradient;
pub mod hubbard_response;
pub mod hubbard_symmetry;
pub mod hubbard_table;
pub mod hybrid_gradient;
pub mod hybrid_setup;
pub mod input;
pub mod irreps;
pub mod kpath;
pub mod optics;
pub mod point_group;
pub mod response;
pub mod response_scf;
pub mod screened_augmentation;
pub mod screened_exchange;
pub mod screened_onecenter;
pub mod screening;
pub mod setup;
pub mod solver;
pub mod space_group;
pub mod spectrum;
pub mod spinor;
mod star;
#[doc(hidden)]
pub use star::ImageCensus;
pub mod stress;
pub mod stress_terms;
pub mod tetrahedron;
pub mod wannier;
use crate::Error;
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
pub use hubbard_response::{hubbard_response, PeriodicProbe, ResponseResult};
pub use input::{Crystal, KPoint};
pub use solver::{Eigensolver, Options, PeriodicResult, ScfAlgorithm};
