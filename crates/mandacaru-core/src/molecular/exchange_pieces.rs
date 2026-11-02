//! The pieces of a screened hybrid's exact-exchange energy on a PAW molecule.
//!
//! The pair integrals of the hybrid are a sum of independent sets of pair
//! vectors, so its exchange energy is the sum of one term per set: the grid's
//! screened Cholesky factors, the one-center tensor `Delta W^SR` and the
//! compensation charges. Beside them stand the spheres' frozen semilocal
//! short-range exchange the hybrid replaces (its linear term and its constant)
//! and the spin polarization of that term. [`ExchangePieces`] reports each,
//! already multiplied by `-fraction`, so they add up to the hybrid's
//! `exact_energy` that the Kohn-Sham energy includes. It exists for diagnosis:
//! which piece of the exchange energy moves with the grid, the basis or the
//! dataset.
use super::{integrals::Eri, scf, MolecularIntegrals};
use crate::{quantum::C64, Result};
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

/// The terms of the hybrid's exact-exchange energy (Hartree), each with its sign.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExchangePieces {
    /// `-a E_x[grid pairs]`: the smooth pair densities on the grid.
    pub grid: f64,
    /// `-a E_x[Delta W^SR]`: the one-center remainder of the spheres.
    pub one_center: f64,
    /// `-a E_x[compensation]`: the compensation charges (their grid matrices
    /// and mutual repulsion).
    pub compensation: f64,
    /// `-a V . D`: the linear term of the frozen semilocal exchange.
    pub frozen_linear: f64,
    /// `-a (X0 - V . D0)`: its constant.
    pub constant: f64,
    /// `-a P`: the polarization of the semilocal term (zero for a restricted density).
    pub polarization: f64,
    /// Not part of [`total`](Self::total): `-a (exact - frozen)`, the energy the
    /// spheres' semilocal short-range exchange would add if it were evaluated on
    /// the current density instead of linearized at the reference atom.
    pub linearization: f64,
}

impl ExchangePieces {
    /// The sum, the hybrid's exact-exchange energy.
    pub fn total(&self) -> f64 {
        self.grid
            + self.one_center
            + self.compensation
            + self.frozen_linear
            + self.constant
            + self.polarization
    }
}

impl MolecularIntegrals {
    /// The pieces of the exact-exchange energy of `functional` at the orthonormal
    /// frame spin densities `d` (one restricted or two); `None` for a functional
    /// without screened exchange.
    pub fn exchange_pieces(
        &self,
        functional: &str,
        d: &[Matrix],
    ) -> Result<Option<ExchangePieces>> {
        let Some(screening) = self.screening(functional)? else {
            return Ok(None);
        };
        let m = self.m();
        let spins = if d.len() == 1 { 2. } else { 1. };
        let a = screening.fraction;
        let mut pieces = ExchangePieces {
            grid: 0.,
            one_center: 0.,
            compensation: 0.,
            frozen_linear: 0.,
            constant: -a * screening.constant,
            polarization: 0.,
            linearization: 0.,
        };
        let Eri::Augmented {
            base, correction, ..
        } = &*screening.eri
        else {
            for density in d {
                let (_, k) = scf::jk(&screening.eri, m, density);
                pieces.grid -= 0.5 * a * (density * &k).trace().re * spins;
            }
            return Ok(Some(pieces));
        };
        for density in d {
            let (_, k) = scf::jk(base, m, density);
            pieces.grid -= 0.5 * a * (density * &k).trace().re * spins;
            let real = density.map(|v| v.re);
            for (index, (factor, weight)) in correction
                .factors
                .iter()
                .zip(&correction.weights)
                .enumerate()
            {
                let product = &real * factor;
                let value = -0.5 * a * weight * (&product * &product).trace() * spins;
                if index < screening.one_center_factors {
                    pieces.one_center += value;
                } else {
                    pieces.compensation += value;
                }
            }
            if let Some(frozen) = &screening.one_body {
                pieces.frozen_linear -= a * (&real * frozen).trace() * spins;
            }
        }
        if let Some(spin) = &screening.spin {
            let up = d[0].map(|v| v.re);
            let down = d.get(1).map_or_else(|| up.clone(), |d| d.map(|v| v.re));
            if d.len() == 2 {
                pieces.polarization = -a * spin.polarization(&self.context, &up, &down)?.0;
            }
            let (exact, frozen) = spin.semilocal(&self.context, &up, &down)?;
            pieces.linearization = -a * (exact - frozen);
        }
        Ok(Some(pieces))
    }
}
