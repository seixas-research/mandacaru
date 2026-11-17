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
    if let Some(raw) = one_center_positions(integrals, origin)? {
        for (sum, raw) in out.iter_mut().zip(raw) {
            let frame = integrals.lowdin.transpose() * raw * &integrals.lowdin;
            *sum += (&frame + frame.transpose()) * 0.5;
        }
    }
    Ok(out)
}

/// The PAW one-center part of the position integrals about `origin` (Bohr) over the raw atomic
/// basis, `sum_A P^A ((R_A - O) Delta^(0) + Delta^(1)) P^A^T` per axis (`None` off PAW-LCAO): the
/// charge and dipole of an atomic-basis pair density that its smooth field leaves out.
pub fn one_center_positions(
    integrals: &MolecularIntegrals,
    origin: [f64; 3],
) -> Result<Option<[DMatrix<f64>; 3]>> {
    let Some(setup) = integrals.paw_setup.as_ref() else {
        return Ok(None);
    };
    let m = integrals.m();
    let projections = integrals
        .corrections
        .as_ref()
        .and_then(|c| c.projections.as_ref())
        .ok_or_else(|| invalid("PAW position integrals need the atoms' projections"))?;
    let atoms = setup.position_blocks();
    if atoms.len() != projections.len() || atoms.len() != integrals.geometry.positions().len() {
        return Err(invalid("PAW projections and atoms disagree"));
    }
    let mut out = [
        DMatrix::<f64>::zeros(m, m),
        DMatrix::<f64>::zeros(m, m),
        DMatrix::<f64>::zeros(m, m),
    ];
    for (axis, raw) in out.iter_mut().enumerate() {
        for ((blocks, c), position) in atoms
            .iter()
            .zip(projections)
            .zip(integrals.geometry.positions())
        {
            if c.nrows() != m || c.ncols() != blocks.overlap.nrows() {
                return Err(invalid("PAW projections and one-center blocks disagree"));
            }
            let local = &blocks.overlap * (position[axis] - origin[axis]) + &blocks.dipole[axis];
            *raw += c * local * c.transpose();
        }
    }
    Ok(Some(out))
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

/// The electric dipole moment `sum_A Z_A (R_A - O) - sum_s g tr(D_s X(O))` (Bohr times the
/// elementary charge) of the molecular state with the orthonormal-frame densities `densities`,
/// about `origin` (Bohr): the ions' charges (the PAW valence charges, the nuclear numbers on an
/// all-electron basis) at their positions, less the electrons through [`position_integrals`]
/// (the PAW one-center dipole included). One per-spin matrix is a restricted state's (`g = 2`),
/// two are both spins (`g = 1`), and a spinor state's `2m x 2m` matrix contributes its charge
/// component. A neutral molecule's moment does not depend on `origin`.
pub fn dipole_moment(
    integrals: &MolecularIntegrals,
    densities: &[DMatrix<crate::quantum::C64>],
    origin: [f64; 3],
) -> Result<[f64; 3]> {
    let m = integrals.m();
    let spinor = densities.len() == 1 && densities[0].shape() == (2 * m, 2 * m);
    if densities.is_empty()
        || densities.len() > 2
        || (!spinor && densities.iter().any(|d| d.shape() != (m, m)))
    {
        return Err(invalid(
            "a dipole moment needs one or two m x m density matrices or one 2m x 2m spinor one",
        ));
    }
    let x = position_integrals(integrals, origin)?;
    let degeneracy = if densities.len() == 1 && !spinor {
        2.
    } else {
        1.
    };
    let charge = |d: &DMatrix<crate::quantum::C64>, a: usize, b: usize| {
        if spinor {
            (d[(a, b)] + d[(m + a, m + b)]).re
        } else {
            d[(a, b)].re
        }
    };
    let numbers = integrals.geometry.numbers();
    let positions = integrals.geometry.positions();
    let charges: Vec<f64> = (0..numbers.len())
        .map(|i| {
            integrals
                .corrections
                .as_ref()
                .map_or(numbers[i] as f64, |c| c.charges[i])
        })
        .collect();
    Ok(std::array::from_fn(|axis| {
        let ions: f64 = charges
            .iter()
            .zip(positions)
            .map(|(z, r)| z * (r[axis] - origin[axis]))
            .sum();
        let electrons: f64 = densities
            .iter()
            .map(|d| {
                (0..m)
                    .flat_map(|a| (0..m).map(move |b| (a, b)))
                    .map(|(a, b)| charge(d, a, b) * x[axis][(b, a)])
                    .sum::<f64>()
            })
            .sum();
        ions - degeneracy * electrons
    }))
}
