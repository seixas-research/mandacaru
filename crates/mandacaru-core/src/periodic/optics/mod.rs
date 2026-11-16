//! Optical response of a crystal in its static LCAO basis: the momentum matrix of the Bloch states
//! and the diamagnetic band sum (`TDDFT_PLAN.md`, T6b0), and the real-time propagation in the
//! velocity gauge that reads it (T6b). The frequency-domain dielectric function (T5a,
//! [`dielectric`]) reads the momentum too, and the Bethe-Salpeter excitons (T5b, [`bse`]) read
//! its columns.
pub mod bse;
pub mod dielectric;
pub mod momentum;
pub mod realtime;
