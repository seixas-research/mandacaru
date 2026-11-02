//! Projector augmented-wave datasets, potentials and projector construction.
pub mod augmentation;
pub mod channel;
pub mod checks;
pub mod confinement;
pub mod core_correction;
pub mod dataset;
pub mod digest;
pub mod fields;
pub mod generate;
pub mod local_split;
pub mod molecular;
pub mod multipole;
pub mod multipole_pairs;
pub mod onecenter;
pub mod onecenter_hybrid;
pub mod onecenter_radial;
pub mod onecenter_terms;
pub mod pair_tables;
pub mod pao;
pub mod pseudize;
pub mod pseudize_tables;
pub mod radial;
pub mod record;
pub mod screened_atom;
pub mod screened_polarization;
pub mod shell;
pub mod smooth;
pub mod species_tables;
pub mod spectrum;

fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::Invalid(message.into())
}
