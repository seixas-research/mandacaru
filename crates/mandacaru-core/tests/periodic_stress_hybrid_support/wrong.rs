//! The wrong derivatives of the data the exchange pass reads, handed to it through the
//! `Sources` protocol: the negative controls of the kernel (the mean vector, no head, the pair
//! `k = q` for every pair). Each is the right `Production` with ONE thing replaced.
#![allow(dead_code)]

use mandacaru_core::{
    math::reciprocal,
    periodic::{
        screened_augmentation::{Augmentation, ShapeStrain},
        screened_exchange::ExxDivTreatment,
        stress::{kernel_strain_derivative, screened_kernel_strain},
        stress_terms::hybrid::{Production, Sources},
        Crystal,
    },
    Buffer, Result,
};

/// The kernel strain of the OUTER product of the mean vector (the defect of the wired function
/// the plan replaced): `v'(|<Q>|^2)` and `<Q> (x) <Q>` instead of the alias means of the square
/// and of the outer product.
pub struct MeanVector;

impl Sources for MeanVector {
    fn kernel(
        &self,
        c: &Crystal,
        k_diff: [f64; 3],
        pair_weight: f64,
        omega: f64,
        exxdiv: ExxDivTreatment,
    ) -> Result<Buffer<f64>> {
        let n = c.grid.index.points();
        let volume = c.grid.index.dv() * n as f64;
        let qc = (6. * std::f64::consts::PI.powi(2) * pair_weight / volume).cbrt();
        let symbols = reciprocal::bloch(&c.context, &c.grid, k_diff)?;
        let mut out = c.context.allocate::<f64>(6 * n)?;
        for at in 0..n {
            let q = std::array::from_fn(|a| symbols.vectors.as_slice()[a * n + at]);
            for (row, v) in kernel_strain_derivative(&q, omega, qc, exxdiv)
                .iter()
                .enumerate()
            {
                out.as_mut_slice()[row * n + at] = *v;
            }
        }
        Ok(out)
    }
    fn shapes(&self, aug: &Augmentation, c: &Crystal, k_diff: [f64; 3]) -> Result<ShapeStrain> {
        Production.shapes(aug, c, k_diff)
    }
}

/// The right kernel with the head (the `Q = 0` entry, where the squares vanish) removed. Only
/// the pairs with `k_diff = 0` have one: `heads` counts the entries removed over all pairs, so a
/// test can assert that the control removed something.
#[derive(Default)]
pub struct NoHead {
    pub heads: std::cell::Cell<usize>,
}

impl Sources for NoHead {
    fn kernel(
        &self,
        c: &Crystal,
        k_diff: [f64; 3],
        pair_weight: f64,
        omega: f64,
        exxdiv: ExxDivTreatment,
    ) -> Result<Buffer<f64>> {
        let n = c.grid.index.points();
        let mut out = Production.kernel(c, k_diff, pair_weight, omega, exxdiv)?;
        let symbols = reciprocal::bloch(&c.context, &c.grid, k_diff)?;
        for at in 0..n {
            if symbols.squares.as_slice()[at] <= 1e-12 {
                self.heads.set(self.heads.get() + 1);
                for row in 0..6 {
                    out.as_mut_slice()[row * n + at] = 0.;
                }
            }
        }
        Ok(out)
    }
    fn shapes(&self, aug: &Augmentation, c: &Crystal, k_diff: [f64; 3]) -> Result<ShapeStrain> {
        Production.shapes(aug, c, k_diff)
    }
}

/// The kernel strain of the pair `k = q` for every pair (the wave vector `q - k` replaced by 0).
pub struct ZeroPair;

impl Sources for ZeroPair {
    fn kernel(
        &self,
        c: &Crystal,
        _k_diff: [f64; 3],
        pair_weight: f64,
        omega: f64,
        exxdiv: ExxDivTreatment,
    ) -> Result<Buffer<f64>> {
        screened_kernel_strain(&c.context, &c.grid, [0.; 3], pair_weight, omega, exxdiv)
    }
    fn shapes(&self, aug: &Augmentation, c: &Crystal, k_diff: [f64; 3]) -> Result<ShapeStrain> {
        Production.shapes(aug, c, k_diff)
    }
}
