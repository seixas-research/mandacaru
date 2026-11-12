//! Position integrals `<a| r - O |b>` and gradient integrals `<a| grad |b>` of the molecular basis
//! in its orthonormal frame.
//!
//! The smooth part is a quadrature over the grid of the orthonormal samples. On PAW-LCAO the
//! one-center part (`paw::molecular::Setup::position_blocks`) is added through the projections,
//! so the integrals are those of the all-electron pair densities. Because the orthonormal frame is
//! orthonormal in the same metric (grid overlap plus the one-center `q`), moving the origin by
//! `c` changes every matrix by exactly `c` times the identity: a transition moment between
//! orthogonal states does not depend on the origin.
//!
//! Atomic units (Bohr).
use super::{invalid, MolecularIntegrals};
use crate::{math::storage::SampleBlocks, product, Result};
use nalgebra::DMatrix;

/// The three position matrices about `origin` (Bohr), symmetric, in the orthonormal frame.
pub fn position_integrals(
    integrals: &MolecularIntegrals,
    origin: [f64; 3],
) -> Result<[DMatrix<f64>; 3]> {
    if origin.iter().any(|v| !v.is_finite()) {
        return Err(invalid("the dipole origin must be finite"));
    }
    if integrals.grid.periodic {
        return Err(invalid(
            "position integrals are defined for an isolated molecule, not a periodic grid",
        ));
    }
    let ctx = &integrals.context;
    let m = integrals.m();
    let n = integrals.grid.index.points();
    let dv = integrals.grid.index.dv();
    let _work = ctx.reserve(product(&[m, m, 8, 6])?)?;
    let mut blocks = SampleBlocks::new(ctx, &integrals.orthonormal_samples, m, n)?;
    let stride = blocks.stride();
    let mut out = [
        DMatrix::<f64>::zeros(m, m),
        DMatrix::<f64>::zeros(m, m),
        DMatrix::<f64>::zeros(m, m),
    ];
    for start in (0..n).step_by(stride) {
        let count = blocks.load(start)?;
        let samples = nalgebra::DMatrixView::from_slice_with_strides_generic(
            blocks.values(),
            nalgebra::Dyn(count),
            nalgebra::Dyn(m),
            nalgebra::Const::<1>,
            nalgebra::Dyn(stride),
        );
        for (axis, sum) in out.iter_mut().enumerate() {
            let weighted = DMatrix::from_fn(count, m, |i, a| {
                samples[(i, a)] * (integrals.grid.point(start + i)[axis] - origin[axis])
            });
            *sum += samples.transpose() * weighted;
        }
    }
    for sum in &mut out {
        *sum = (&*sum + sum.transpose()) * (0.5 * dv);
    }
    if let Some(setup) = integrals.paw_setup.as_ref() {
        let projections = integrals
            .corrections
            .as_ref()
            .and_then(|c| c.projections.as_ref())
            .ok_or_else(|| invalid("PAW position integrals need the atoms' projections"))?;
        let atoms = setup.position_blocks();
        if atoms.len() != projections.len() || atoms.len() != integrals.geometry.positions().len() {
            return Err(invalid("PAW projections and atoms disagree"));
        }
        for axis in 0..3 {
            let mut raw = DMatrix::<f64>::zeros(m, m);
            for ((blocks, c), position) in atoms
                .iter()
                .zip(projections)
                .zip(integrals.geometry.positions())
            {
                if c.nrows() != m || c.ncols() != blocks.overlap.nrows() {
                    return Err(invalid("PAW projections and one-center blocks disagree"));
                }
                let local =
                    &blocks.overlap * (position[axis] - origin[axis]) + &blocks.dipole[axis];
                raw += c * local * c.transpose();
            }
            let frame = integrals.lowdin.transpose() * raw * &integrals.lowdin;
            out[axis] += (&frame + frame.transpose()) * 0.5;
        }
    }
    Ok(out)
}

/// The gradient integrals `<a| d/dx_k |b>` (Bohr^-1) in the orthonormal frame: the grid part with
/// the molecular derivative stencil of the SCF (skew-adjoint, so exactly antisymmetric) and, on
/// PAW-LCAO, the one-center part (`paw::molecular::Setup::gradient_blocks`) through the
/// projections. These are the matrix elements of the velocity form of a transition moment.
pub fn gradient_integrals(integrals: &MolecularIntegrals) -> Result<[DMatrix<f64>; 3]> {
    if integrals.grid.periodic {
        return Err(invalid(
            "gradient integrals are defined for an isolated molecule, not a periodic grid",
        ));
    }
    let ctx = &integrals.context;
    let m = integrals.m();
    let n = integrals.grid.index.points();
    let dv = integrals.grid.index.dv();
    let gradients = super::dft::orbital_gradients(integrals)?;
    let mut values = SampleBlocks::new(ctx, &integrals.orthonormal_samples, m, n)?;
    let mut slopes = SampleBlocks::new(ctx, &gradients, 3 * m, n)?;
    let stride = values.stride();
    let mut out = [
        DMatrix::<f64>::zeros(m, m),
        DMatrix::<f64>::zeros(m, m),
        DMatrix::<f64>::zeros(m, m),
    ];
    for start in (0..n).step_by(stride) {
        let count = values.load(start)?;
        slopes.load(start)?;
        let view = |data: &[f64], offset: usize| {
            nalgebra::DMatrixView::from_slice_with_strides_generic(
                &data[offset..],
                nalgebra::Dyn(count),
                nalgebra::Dyn(m),
                nalgebra::Const::<1>,
                nalgebra::Dyn(stride),
            )
            .into_owned()
        };
        let phi = view(values.values(), 0);
        for (axis, sum) in out.iter_mut().enumerate() {
            let slope = view(slopes.values(), axis * m * stride);
            *sum += phi.transpose() * slope;
        }
    }
    for sum in &mut out {
        *sum *= dv;
    }
    if let Some(setup) = integrals.paw_setup.as_ref() {
        let projections = integrals
            .corrections
            .as_ref()
            .and_then(|c| c.projections.as_ref())
            .ok_or_else(|| invalid("PAW gradient integrals need the atoms' projections"))?;
        let atoms = setup.gradient_blocks(ctx)?;
        if atoms.len() != projections.len() {
            return Err(invalid("PAW projections and atoms disagree"));
        }
        for axis in 0..3 {
            let mut raw = DMatrix::<f64>::zeros(m, m);
            for (blocks, c) in atoms.iter().zip(projections) {
                if c.ncols() != blocks[axis].nrows() {
                    return Err(invalid("PAW projections and one-center blocks disagree"));
                }
                raw += c * &blocks[axis] * c.transpose();
            }
            out[axis] += integrals.lowdin.transpose() * raw * &integrals.lowdin;
        }
    }
    Ok(out)
}
