//! The wrong variants of the spinor stress the controls contract: a strain provider that
//! delegates to the right one and damages one ingredient. Each is compared with the same oracle as
//! the right stress and must miss it by more than 100 gates.
#![allow(dead_code)]

use mandacaru_core::{
    periodic::{
        gradient::{Layout, Matrix, SpinOrbit, Term},
        stress_terms::{
            electrostatic::Hartree, xc, Dense, Direction, Directions, SpinMatrices, Strain, Tensor,
        },
    },
    quantum::C64,
    Result,
};

/// What a [`Altered`] provider gets wrong.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Wrong {
    /// Nothing: the delegate itself (the control of the controls).
    Nothing,
    /// The matrix tangent of `term` acts on the alpha block only (`T` not doubled).
    AlphaOnly(Term),
    /// The spin-orbit tangent is zero (no SO term).
    NoSpinOrbit,
    /// The spin-orbit tangent has the other sign.
    SpinOrbitSign,
}

pub struct Altered<'a, S: Strain> {
    pub inner: &'a S,
    pub wrong: Wrong,
}

/// The same densities with the beta block zeroed: the contraction sees the alpha block only.
fn alpha_only(m: &[SpinMatrices]) -> Vec<SpinMatrices> {
    let cut = |matrix: &Matrix| {
        let mut m = matrix.clone();
        let half = m.nrows() / 2;
        m.view_mut((half, half), (half, half)).fill(C64::default());
        m
    };
    m.iter()
        .map(|m| SpinMatrices {
            density: cut(&m.density),
            weighted: cut(&m.weighted),
        })
        .collect()
}

impl<S: Strain> Strain for Altered<'_, S> {
    type Fields = S::Fields;

    fn layout(&self) -> &Layout {
        self.inner.layout()
    }
    fn directions(&self) -> &Directions {
        self.inner.directions()
    }
    fn lane_bytes(&self, terms: usize) -> Result<usize> {
        self.inner.lane_bytes(terms)
    }
    fn blocks(&self) -> &[Matrix] {
        self.inner.blocks()
    }
    fn augmentation(&self) -> &Matrix {
        self.inner.augmentation()
    }
    fn ion_operator(&self) -> &Matrix {
        self.inner.ion_operator()
    }
    fn volume(&self) -> f64 {
        self.inner.volume()
    }
    fn projection(&self, k: usize) -> Result<Matrix> {
        self.inner.projection(k)
    }
    fn projection_strain(&self, k: usize, directions: &[Direction]) -> Result<Vec<Matrix>> {
        self.inner.projection_strain(k, directions)
    }
    fn matrix_strain(
        &self,
        k: usize,
        fields: &S::Fields,
        matrices: &[SpinMatrices],
        terms: &[Term],
        directions: &[Direction],
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let mut out = self
            .inner
            .matrix_strain(k, fields, matrices, terms, directions)?;
        if let Wrong::AlphaOnly(term) = self.wrong {
            if let Some(at) = terms.iter().position(|t| *t == term) {
                let damaged = self.inner.matrix_strain(
                    k,
                    fields,
                    &alpha_only(matrices),
                    &[term],
                    directions,
                )?;
                for (spin, per_term) in damaged.into_iter().enumerate() {
                    out[spin][at] = per_term.into_iter().next().expect("one term");
                }
            }
        }
        Ok(out)
    }
    fn moments<'f>(&self, fields: &'f S::Fields) -> &'f [C64] {
        self.inner.moments(fields)
    }
    fn hartree(&self, fields: &S::Fields, q: &[C64]) -> Result<Hartree> {
        self.inner.hartree(fields, q)
    }
    fn dense(&self) -> Result<Dense> {
        self.inner.dense()
    }
    fn constants(&self) -> Result<Tensor<f64>> {
        self.inner.constants()
    }
    fn xc(&self, fields: &S::Fields) -> Result<xc::Terms> {
        self.inner.xc(fields)
    }
    fn spin_orbit(&self) -> Result<&SpinOrbit> {
        self.inner.spin_orbit()
    }
    fn spin_orbit_projection(&self, k: usize) -> Result<Matrix> {
        self.inner.spin_orbit_projection(k)
    }
    fn spin_orbit_strain(&self, k: usize, directions: &[Direction]) -> Result<Vec<Matrix>> {
        let t = self.inner.spin_orbit_strain(k, directions)?;
        Ok(match self.wrong {
            Wrong::NoSpinOrbit => t
                .into_iter()
                .map(|m| Matrix::zeros(m.nrows(), m.ncols()))
                .collect(),
            Wrong::SpinOrbitSign => t.into_iter().map(|m| m * C64::new(-1., 0.)).collect(),
            _ => t,
        })
    }
}
