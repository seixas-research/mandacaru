//! Projector augmented-wave datasets, potentials and projector construction.
pub mod channel;
pub mod dataset;
pub mod generate;
pub mod radial;
pub mod reference;
pub mod smooth;

fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::Invalid(message.into())
}
