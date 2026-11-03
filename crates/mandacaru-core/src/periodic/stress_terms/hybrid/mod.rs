//! The strain derivative of the periodic screened exact exchange (AS-9, AS-10): scalar collinear
//! and spinor crystals on the whole k mesh.
//!
//! The Mermin free energy of a hybrid is `F = E_semilocal[D; eps] + E_x[D; eps] - T S` with the
//! exchange `E_x = -(a/2) mult sum_{k,q} w_k w_q (dv/N) sum_{m,n} sum_G v(Q_G) |F_mn(G)|^2 + E_oc[P]`,
//! quadratic in the density matrix `D` (the AO coefficients and occupations, which a strain does
//! not move). The stress is the semilocal chain rule of [`super`] with `W` built from the bands of
//! THE HYBRID `H` (the engine's job), plus the explicit derivative of `E_x` at fixed `D`. Its
//! strain dependences, each a [`Terms`] entry:
//!
//! | entry | what strains |
//! |---|---|
//! | `jacobian` (H-1) | `dv` of the prefactor `dv / N`: `delta_ab E_x^pair` (grid, cross and compensation parts; NOT `E_oc`) |
//! | `kernel` (H-2) | `v(squares)`, `squares = mean_alias |k_diff + G|^2`: `v'(s) ds_ab`, `ds_ab = -2 mean_alias(Q_a Q_b)`; at `Q = 0` nothing for `exxdiv = none`, `delta_ab (v_bar - v(q_c))` for Gygi-Baldereschi |
//! | `shape_volume`, `shape_form` (H-3) | `T_a = hat_a(Q) e^{iG.r0} / dv`: `-delta_ab T_a` and `-Q_a d_b f_a` (the slope of the shape table at the plain wrapped `Q`) |
//! | `bra`, `ket` (H-4) | the grid pair `conj(u_m(k)) psi_n(q)`, both slots: `A_ab` of [`StrainSamples`] |
//! | `compensation`, `sphere` (H-5, H-6) | the projections `C(k)`, `C(q)`: `2 Re tr[dC_ab Z]` with the response `Z` of [`crate::periodic::hybrid_gradient::explicit`] (its own weights folded in) |
//!
//! There is NO strain analogue of the force's phase term: `Q.R`, `G.r0` and `k.r` do not strain.
//! The one-center energy `E_oc` is a function of the projected density alone (rigid spheres,
//! frozen potentials): it enters through `Z` and `dC`.
//!
//! Convention and units: those of [`super`]; every entry is `dE/d eps_ab` in Hartree, nine
//! unsymmetrized entries `[a][b]`.
//!
//! Memory: the periodic parts of the nine tangents of every k-point are held at once (`9 nk m N`
//! complex values charged to the `Context`; the allocation raises when it does not fit).
mod chain;
mod pass;
mod sources;

pub use sources::{Production, Sources};

use super::Tensor;
use crate::{
    math::storage::MatrixSource,
    periodic::{
        gradient::{scf, Bands, Layout},
        hybrid_gradient::explicit,
        invalid,
        screened_exchange::{ExxDivTreatment, Screening},
        setup::{
            provider::Inputs,
            pulay::{Matrix, Mesh},
            shells::Shells,
            spheres::Projector,
            strain_samples::StrainSamples,
            strain_spheres::projection_strain,
        },
        Crystal,
    },
    quantum::C64,
    Buffer, Context, Result,
};
use nalgebra::Matrix3;

/// Whether the functional of the request has screened exact exchange.
pub fn is_hybrid(layout: &Layout) -> Result<bool> {
    Ok(crate::models::functional(&layout.request.functional)?
        .screened_exchange
        .is_some())
}

/// `dE_x / d eps_ab` term by term (Hartree).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Terms {
    /// The grid-plus-compensation exchange energy the pass differentiated (`E_x - E_oc`); equal to
    /// [`crate::periodic::hybrid_gradient::Gradient::energy`].
    pub energy: f64,
    pub jacobian: Tensor<f64>,
    pub kernel: Tensor<f64>,
    pub shape_volume: Tensor<f64>,
    pub shape_form: Tensor<f64>,
    pub bra: Tensor<f64>,
    pub ket: Tensor<f64>,
    pub compensation: Tensor<f64>,
    pub sphere: Tensor<f64>,
}

impl Terms {
    pub fn parts(&self) -> [(&'static str, &Tensor<f64>); 8] {
        [
            ("exchange_jacobian", &self.jacobian),
            ("exchange_kernel", &self.kernel),
            ("exchange_shape_volume", &self.shape_volume),
            ("exchange_shape_form", &self.shape_form),
            ("exchange_bra", &self.bra),
            ("exchange_ket", &self.ket),
            ("exchange_compensation", &self.compensation),
            ("exchange_sphere", &self.sphere),
        ]
    }

    /// The sum of every entry.
    pub fn total(&self) -> Tensor<f64> {
        let mut total = [[0.; 3]; 3];
        for (_, part) in self.parts() {
            for a in 0..3 {
                for b in 0..3 {
                    total[a][b] += part[a][b];
                }
            }
        }
        total
    }
}

/// The setup's strain tangents at the crystal's k-points: the Bloch tangents `A_ab` of the basis
/// ([`StrainSamples`]) and the projection tangents `dC_ab` with rigid spheres.
pub struct Tangents<'a> {
    ctx: &'a Context,
    shells: &'a Shells,
    lattice: Matrix3<f64>,
    mesh: Mesh,
    projectors: &'a [Projector],
    kpoints: &'a [[f64; 3]],
}

impl<'a> Tangents<'a> {
    pub fn new(ctx: &'a Context, inputs: &Inputs<'a>) -> Result<Self> {
        Ok(Self {
            ctx,
            shells: inputs.shells,
            lattice: inputs.sites.lattice,
            mesh: Mesh::new(ctx, inputs.grid)?,
            projectors: inputs.projectors,
            kpoints: inputs.kpoints,
        })
    }

    fn point(&self, k: usize) -> Result<[f64; 3]> {
        self.kpoints
            .get(k)
            .copied()
            .ok_or_else(|| invalid("exchange strain: k-point out of range"))
    }

    /// `A_ab` (`m x N`, function-major) at k-point `k`.
    pub fn samples(&self, k: usize, a: usize, b: usize) -> Result<Buffer<C64>> {
        StrainSamples::new(self.shells, &self.lattice, &self.mesh, self.point(k)?)
            .tangent(self.ctx, a, b)
    }

    /// `dC_ab` (`m x p`) at k-point `k`.
    pub fn projection_strain(&self, k: usize) -> Result<Tensor<Matrix>> {
        projection_strain(
            self.ctx,
            self.shells,
            &self.lattice,
            self.point(k)?,
            self.projectors,
        )
    }

    /// Raises unless the setup these tangents come from is the crystal's: its k-points, basis and
    /// projector counts and mesh.
    fn check(&self, c: &Crystal) -> Result<()> {
        let fine = self.kpoints.len() == c.points.len()
            && self.shells.functions() == c.m
            && self.projectors.len() == c.p
            && self.mesh.len() == c.grid.index.points()
            && self
                .kpoints
                .iter()
                .zip(&c.points)
                .all(|(k, p)| (0..3).all(|a| (k[a] - p.k[a]).abs() < 1e-12));
        if fine {
            Ok(())
        } else {
            Err(invalid(
                "exchange strain: the setup is not the crystal's (k-points, basis, projectors or mesh)",
            ))
        }
    }
}

/// The exchange strain of the density matrices `matrices` (the layout of
/// `Crystal::hybrid_exchange`) of a scalar collinear hybrid crystal: the pair pass of
/// [`pass`], and the projection chain with the response `Z` of
/// [`explicit`]. `sources` supplies the two strain derivatives of the setup data that are not
/// tangents ([`Production`] is the one the stress uses).
///
/// The pass energy is tied to the energy `explicit` differentiates; they must agree to 1e-9
/// (relative to `max(|E|, 1)`), else the pass is not differentiating the exchange of the SCF.
pub fn exchange_strain<M: MatrixSource>(
    c: &Crystal,
    matrices: &[M],
    exxdiv: ExxDivTreatment,
    tangents: &Tangents,
    sources: &dyn Sources,
) -> Result<Terms> {
    tangents.check(c)?;
    let screening = Screening::of(c)?;
    let augmentation = c.hybrid_augmentation()?;
    let gradient = explicit(c, matrices, screening, exxdiv, &augmentation, None)?;
    let mut terms = pass::run(
        c,
        matrices,
        screening,
        exxdiv,
        &augmentation,
        tangents,
        sources,
    )?;
    if (terms.energy - gradient.energy).abs() > 1e-9 * gradient.energy.abs().max(1.) {
        return Err(invalid(format!(
            "exchange strain: the pass energy {} is not the exchange energy {} it differentiates",
            terms.energy, gradient.energy
        )));
    }
    chain::contract(c, tangents, &gradient, &mut terms)?;
    Ok(terms)
}

/// [`exchange_strain`] of the density matrices of the (re-diagonalized) `bands`, with the
/// production derivatives.
pub fn exchange_terms(
    c: &Crystal,
    bands: &[Vec<Bands>],
    exxdiv: ExxDivTreatment,
    tangents: &Tangents,
) -> Result<Terms> {
    let densities = scf::densities(c, bands)?;
    exchange_strain(c, &densities, exxdiv, tangents, &Production)
}
