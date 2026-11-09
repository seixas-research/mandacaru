//! The spin-orbit term between the spinor Bloch sums.
//!
//! The molecule's operator ([`crate::models::spin_orbit::pair`]) with the projections of the
//! Bloch sums on the home cell's spin-orbit projectors in place of the molecular ones, the same
//! unfolding as the scalar nonlocal term `C(k) D C(k)^dagger`:
//!
//! ```text
//! h^SO_{mu s, nu t}(k) = sum_{a,l} sum_{ij, m m'} C^SO_{mu,lmi}(k) D^{SO,a,l}_{ij}
//!                        <l m s| L.S |l m' t> C^SO_{nu,lm'j}(k)^*
//! ```
//!
//! The matrix is `(2M, 2M)` over `(spin, function)`, alpha block first; the crystal's projectors
//! carry complex harmonics.
use super::shells::Shells;
use super::spheres::{projection_slopes, projections, Projector};
use crate::{
    models::spin_orbit::{channels, pair, Coupling, Label, Projectors},
    product, Buffer, Context, Error, Result,
};
use nalgebra::Matrix3;
use num_complex::Complex64;
use std::collections::BTreeMap;

/// The labels of the crystal's spin-orbit projectors, for [`crate::models::spin_orbit::channels`].
pub fn labels(projectors: &[Projector]) -> Vec<Label> {
    projectors
        .iter()
        .map(|p| Label { atom: p.atom, l: p.l, m: p.m, index: p.index })
        .collect()
}

/// `(k, 2M, 2M)` row-major: `h^{SO}(k)` at the Cartesian `kpoints`, Hermitian. Zero (and still
/// that shape) when `couplings` is empty, a crystal without the term.
pub fn matrices(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    kpoints: &[[f64; 3]],
    projectors: &[Projector],
    couplings: &BTreeMap<(usize, usize), Coupling>,
) -> Result<Buffer<Complex64>> {
    let m = shells.functions();
    let mut out = ctx.allocate::<Complex64>(product(&[kpoints.len(), 2 * m, 2 * m])?)?;
    if couplings.is_empty() {
        return Ok(out);
    }
    let found = channels(ctx, &labels(projectors), couplings, false)?;
    let c = projections(ctx, shells, lattice, kpoints, projectors)?;
    let count = projectors.len();
    for k in 0..kpoints.len() {
        let block = &c.as_slice()[k * m * count..(k + 1) * m * count];
        let h = pair(
            ctx,
            Projectors {
                left: block,
                right: block,
                rows: m,
                columns: m,
                count,
            },
            &found,
            true,
        )?;
        out.as_mut_slice()[k * 4 * m * m..(k + 1) * 4 * m * m].copy_from_slice(h.as_slice());
    }
    Ok(out)
}

/// What a spin-orbit block derivative holds fixed: the basis, the cell, who owns each basis
/// function, the union projectors and the radial couplings.
#[derive(Clone, Copy)]
pub struct Block<'a> {
    pub shells: &'a Shells,
    pub lattice: &'a Matrix3<f64>,
    pub row_owner: &'a [usize],
    pub projectors: &'a [Projector],
    pub couplings: &'a BTreeMap<(usize, usize), Coupling>,
}

/// `d h^SO(k) / d tau_atom^axis`, `(2M, 2M)` row-major: the derivative of [`matrices`] at one
/// k-point when the atom's basis functions (`row_owner[mu] == atom`) and its union projectors
/// (`Projector::atom`) move together. The projections' derivative is the ownership identity of
/// the scalar set, `dC = P G - G Q` (`G = -<d chi | p>` the row-center derivative of
/// [`projection_slopes`], a projector-set parameter of the same code), and the coupling stays
/// in [`pair`]: `dh = pair(dC, C) + pair(C, dC)`. The oracle of the force term
/// (`periodic::gradient::spin_orbit`), not a force.
pub fn matrix_derivative(
    ctx: &Context,
    block: &Block,
    k: [f64; 3],
    atom: usize,
    axis: usize,
) -> Result<Buffer<Complex64>> {
    let Block {
        shells,
        lattice,
        row_owner,
        projectors,
        couplings,
    } = *block;
    let (m, count) = (shells.functions(), projectors.len());
    if row_owner.len() != m || axis > 2 {
        return Err(Error::Invalid(
            "spin-orbit derivative needs one owner per basis function and an axis 0..2".into(),
        ));
    }
    let mut out = ctx.allocate::<Complex64>(product(&[2 * m, 2 * m])?)?;
    if couplings.is_empty() {
        return Ok(out);
    }
    let found = channels(ctx, &labels(projectors), couplings, false)?;
    let c = projections(ctx, shells, lattice, &[k], projectors)?;
    let mut moved = projection_slopes(ctx, shells, lattice, &[k], projectors, Some(axis))?;
    for (mu, row) in moved.as_mut_slice().chunks_mut(count).enumerate() {
        for (value, projector) in row.iter_mut().zip(projectors) {
            let own = f64::from(u8::from(row_owner[mu] == atom))
                - f64::from(u8::from(projector.atom == atom));
            *value *= -own;
        }
    }
    let parts = [
        (moved.as_slice(), c.as_slice()),
        (c.as_slice(), moved.as_slice()),
    ];
    for (left, right) in parts {
        let h = pair(
            ctx,
            Projectors {
                left,
                right,
                rows: m,
                columns: m,
                count,
            },
            &found,
            false,
        )?;
        for (to, from) in out.as_mut_slice().iter_mut().zip(h.as_slice()) {
            *to += *from;
        }
    }
    Ok(out)
}
