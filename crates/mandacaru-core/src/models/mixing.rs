//! Pulay (DIIS) density mixing of a spherical atom, in the radial metric.
//!
//! `n(r)` spans ten orders of magnitude between the nucleus and the valence tail, so a least
//! squares in the plain Euclidean norm optimizes the core and leaves the valence to bare
//! linear mixing. The inner product here is `<A, B> = int A B 4 pi r^2 dr`, the one in which
//! a charge is a charge. With one density in hand the step is exactly linear mixing.
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

/// Densities kept by the mixer; more history buys nothing once the residuals span the slow
/// subspace.
pub const HISTORY: usize = 8;
/// Negative charge, as a fraction of the electron count, an extrapolated density may carry
/// before the mixer falls back to a linear step.
pub const NEGATIVE_CHARGE_TOLERANCE: f64 = 1e-9;

/// How far two radial densities are apart, as the SCF loop measures it: the maximum of the
/// deviation in `n(r)` **and** in the radial density `4 pi r^2 n(r)`, so the test is never
/// blind to the valence tail (a branch flip is invisible in `n(r)` of a heavy atom).
pub fn density_change(previous: &[f64], current: &[f64], r: &[f64]) -> f64 {
    previous
        .iter()
        .zip(current)
        .zip(r)
        .map(|((p, c), r)| {
            let d = c - p;
            d.abs().max((4. * PI * r * r * d).abs())
        })
        .fold(0., f64::max)
}

pub struct PulayMixer {
    weight: Vec<f64>,
    beta: f64,
    electrons: f64,
    history: usize,
    inputs: Vec<Vec<f64>>,
    residuals: Vec<Vec<f64>>,
}

impl PulayMixer {
    /// `weight` is the quadrature weight of the radial metric (`4 pi r^2 h` on a uniform grid).
    pub fn new(weight: Vec<f64>, mixing: f64, electrons: f64, history: usize) -> Self {
        Self {
            weight,
            beta: mixing,
            electrons,
            history,
            inputs: Vec::new(),
            residuals: Vec::new(),
        }
    }

    fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        self.weight
            .iter()
            .zip(a)
            .zip(b)
            .map(|((w, a), b)| w * a * b)
            .sum()
    }

    fn restart(&mut self) {
        let input = self.inputs.pop().unwrap();
        let residual = self.residuals.pop().unwrap();
        self.inputs = vec![input];
        self.residuals = vec![residual];
    }

    /// The next input density from the current input and the density it produced.
    pub fn mix(&mut self, density_in: &[f64], density_out: &[f64]) -> Vec<f64> {
        let residual = density_out
            .iter()
            .zip(density_in)
            .map(|(o, i)| o - i)
            .collect::<Vec<_>>();
        let linear = density_in
            .iter()
            .zip(&residual)
            .map(|(d, r)| d + self.beta * r)
            .collect::<Vec<_>>();
        self.inputs.push(density_in.to_vec());
        self.residuals.push(residual);
        if self.inputs.len() > self.history {
            self.inputs.remove(0);
            self.residuals.remove(0);
        }
        let m = self.inputs.len();
        if m < 2 {
            return linear;
        }
        let mut matrix = DMatrix::<f64>::zeros(m + 1, m + 1);
        for i in 0..m {
            for j in i..m {
                let value = self.dot(&self.residuals[i], &self.residuals[j]);
                matrix[(i, j)] = value;
                matrix[(j, i)] = value;
            }
            matrix[(i, m)] = 1.;
            matrix[(m, i)] = 1.;
        }
        let mut rhs = DVector::<f64>::zeros(m + 1);
        rhs[m] = 1.;
        let coefficients = matrix
            .lu()
            .solve(&rhs)
            .filter(|c| c.iter().all(|v| v.is_finite()));
        let Some(coefficients) = coefficients else {
            // Linearly dependent residuals: nothing left to extrapolate along.
            self.restart();
            return linear;
        };
        let mut mixed = vec![0.; linear.len()];
        for k in 0..m {
            let c = coefficients[k];
            for (out, (i, r)) in mixed
                .iter_mut()
                .zip(self.inputs[k].iter().zip(&self.residuals[k]))
            {
                *out += c * (i + self.beta * r);
            }
        }
        self.accept(mixed, linear)
    }

    /// `mixed`, or `linear` when the extrapolation is not a density. Charge needs no repair
    /// (every input holds `N`, every residual zero, the coefficients sum to one); positivity
    /// is the one property Pulay can violate, and clipping plus rescaling instead stalls the
    /// SCF outright.
    fn accept(&self, mixed: Vec<f64>, linear: Vec<f64>) -> Vec<f64> {
        let deficit = self
            .weight
            .iter()
            .zip(&mixed)
            .map(|(w, m)| if *m < 0. { w * -m } else { 0. })
            .sum::<f64>();
        if deficit > NEGATIVE_CHARGE_TOLERANCE * self.electrons {
            return linear;
        }
        mixed.into_iter().map(|v| v.max(0.)).collect()
    }
}
