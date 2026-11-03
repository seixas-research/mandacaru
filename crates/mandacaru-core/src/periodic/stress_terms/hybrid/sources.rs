//! The strain derivatives of the setup data the exchange pass reads that are not tangents of the
//! basis: the Coulomb kernel and the pair shapes. A protocol, so a test can hand the pass a
//! wrong derivative (a negative control) and compare the stress with the oracle; the stress
//! itself uses [`Production`] and nothing else.
use crate::{
    periodic::{
        screened_augmentation::{Augmentation, ShapeStrain},
        screened_exchange::ExxDivTreatment,
        stress::screened_kernel_strain,
        Crystal,
    },
    Buffer, Result,
};

pub trait Sources {
    /// `d v / d eps_ab` of the kernel the exchange evaluates for the pair wave vector
    /// `k_diff = q - k`, `(6, N)` in the order xx, yy, zz, yz, xz, xy.
    fn kernel(
        &self,
        c: &Crystal,
        k_diff: [f64; 3],
        pair_weight: f64,
        omega: f64,
        exxdiv: ExxDivTreatment,
    ) -> Result<Buffer<f64>>;

    /// The gradients of the pair shapes in `Q` at the wrapped `Q`.
    fn shapes(&self, aug: &Augmentation, c: &Crystal, k_diff: [f64; 3]) -> Result<ShapeStrain>;
}

/// The derivatives of the energy: the alias-mean kernel with the pair's own weight at the head,
/// and the table slope of the shapes.
pub struct Production;

impl Sources for Production {
    fn kernel(
        &self,
        c: &Crystal,
        k_diff: [f64; 3],
        pair_weight: f64,
        omega: f64,
        exxdiv: ExxDivTreatment,
    ) -> Result<Buffer<f64>> {
        screened_kernel_strain(&c.context, &c.grid, k_diff, pair_weight, omega, exxdiv)
    }

    fn shapes(&self, aug: &Augmentation, c: &Crystal, k_diff: [f64; 3]) -> Result<ShapeStrain> {
        aug.pair_shape_strain(c, k_diff)
    }
}
