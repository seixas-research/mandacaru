//! Orbital-aware molecular ansatz handover between independent geometries.
use super::{CachedProblem, C64};
use crate::{
    basis::{Basis, Radial},
    math::storage::SampleBlocks,
    molecular::{response, transform::Model, MeanField, MolecularIntegrals},
    product, Error, Reservation, Result,
};
use nalgebra::DMatrix;
use std::collections::BTreeMap;
pub struct Frame<'a> {
    pub integrals: &'a MolecularIntegrals,
    pub reference: &'a MeanField,
    pub model: &'a Model,
    pub problem: &'a CachedProblem,
}
/// Immutable handover data without an integral cache, Hamiltonian or solver workspace.
pub struct Handover {
    basis: Basis,
    coefficients: DMatrix<C64>,
    numbers: Vec<u8>,
    particles: [usize; 2],
    mapping: String,
    labels: Vec<String>,
    parameters: Vec<f64>,
    _memory: Reservation,
}
pub fn capture(previous: Frame<'_>, indices: &[usize], parameters: &[f64]) -> Result<Handover> {
    if indices.len() != parameters.len()
        || indices.iter().any(|i| *i >= previous.problem.pool.len())
        || parameters.iter().any(|p| !p.is_finite())
    {
        return Err(Error::Invalid(
            "ansatz handover needs a valid previous operator sequence".into(),
        ));
    }
    let ctx = &previous.integrals.context;
    ctx.run(|| {
        let m = previous.integrals.m();
        let labels_memory = indices
            .iter()
            .map(|&i| previous.problem.pool[i].label.len() + std::mem::size_of::<String>() + 8)
            .sum::<usize>();
        let radial_memory = previous
            .integrals
            .basis
            .orbitals
            .iter()
            .map(|o| match &o.radial {
                Radial::Gaussian {
                    exponents,
                    weighted,
                } => 8 * (exponents.len() + weighted.len()),
                _ => 0,
            })
            .sum::<usize>();
        let memory = ctx.reserve(
            product(&[m, previous.model.m, 16])?
                .checked_add(labels_memory)
                .and_then(|v| v.checked_add(radial_memory))
                .and_then(|v| {
                    v.checked_add(
                        previous.integrals.geometry.numbers().len()
                            + previous.problem.encoding.mapping.len(),
                    )
                })
                .ok_or(crate::Error::Overflow)?,
        )?;
        let _work = ctx.reserve(product(&[m, m, 16, 8])?)?;
        let coefficients = previous.integrals.lowdin.map(|v| C64::new(v, 0.))
            * response::orbitals(previous.reference, previous.model)?;
        let basis = Basis::from_orbitals(ctx, previous.integrals.basis.orbitals.clone())?;
        Ok(Handover {
            basis,
            coefficients,
            numbers: previous.integrals.geometry.numbers().to_vec(),
            particles: previous.model.particles,
            mapping: previous.problem.encoding.mapping.clone(),
            labels: indices
                .iter()
                .map(|&i| previous.problem.pool[i].label.clone())
                .collect(),
            parameters: parameters.to_vec(),
            _memory: memory,
        })
    })
}
pub struct Seed {
    pub indices: Vec<usize>,
    pub parameters: Vec<f64>,
    pub confidence: f64,
    pub accepted: bool,
    pub reason: String,
}
fn refused(reason: impl Into<String>, confidence: f64) -> Seed {
    Seed {
        indices: vec![],
        parameters: vec![],
        confidence,
        accepted: false,
        reason: reason.into(),
    }
}
fn normalized_overlap(previous: &Handover, current: &Frame<'_>) -> Result<DMatrix<C64>> {
    let ctx = &current.integrals.context;
    let grid = &current.integrals.grid;
    let points = grid.index.points();
    let a = previous.basis.orbitals.len();
    let b = current.integrals.m();
    let _memory = ctx.reserve(product(&[a.max(b), a.max(b), 16, 12])?)?;
    // Raw samples are real. All large grid contractions remain real and run
    // over point blocks (the previous orbitals evaluated per point, the
    // current stored rows widened per block): no sample stack of either basis
    // exists. The small orbital transforms retain complex phases without
    // clipping.
    let mut blocks = SampleBlocks::new(ctx, &current.integrals.samples, b, points)?;
    let stride = blocks.stride();
    let _old_block = ctx.reserve(product(&[stride, a, 8])?)?;
    let mut old = DMatrix::<f64>::zeros(stride, a);
    let mut raw = vec![0.; a];
    let mut cross = DMatrix::<f64>::zeros(a, b);
    let mut old_metric = DMatrix::<f64>::zeros(a, a);
    let mut new_metric = DMatrix::<f64>::zeros(b, b);
    for start in (0..points).step_by(stride) {
        let count = blocks.load(start)?;
        for i in 0..count {
            previous
                .basis
                .evaluate_at(0, grid.point(start + i), &mut raw);
            for (j, value) in raw.iter().enumerate() {
                old[(i, j)] = *value;
            }
        }
        let old_block = old.rows(0, count);
        let new_block = nalgebra::DMatrixView::from_slice_with_strides_generic(
            blocks.values(),
            nalgebra::Dyn(count),
            nalgebra::Dyn(b),
            nalgebra::Const::<1>,
            nalgebra::Dyn(stride),
        );
        cross += old_block.transpose() * new_block;
        old_metric += old_block.transpose() * old_block;
        new_metric += new_block.transpose() * new_block;
    }
    let dv = grid.index.dv();
    let cross = cross * dv;
    let old_metric = old_metric * dv;
    let new_metric = new_metric * dv;
    let old_coeff = &previous.coefficients;
    let new_coeff = current.integrals.lowdin.map(|v| C64::new(v, 0.))
        * response::orbitals(current.reference, current.model)?;
    let mut overlap = old_coeff.adjoint() * cross.map(|v| C64::new(v, 0.)) * &new_coeff;
    let old_norm = old_coeff.adjoint() * old_metric.map(|v| C64::new(v, 0.)) * old_coeff;
    let new_norm = new_coeff.adjoint() * new_metric.map(|v| C64::new(v, 0.)) * new_coeff;
    for i in 0..overlap.nrows() {
        for j in 0..overlap.ncols() {
            let norm = (old_norm[(i, i)].re * new_norm[(j, j)].re).sqrt();
            if !norm.is_finite() || norm <= 0. {
                return Err(Error::Invalid(
                    "orbital handover has a zero or nonfinite sampled norm".into(),
                ));
            }
            overlap[(i, j)] /= norm;
        }
    }
    Ok(overlap)
}
fn rename(label: &str, permutation: &[usize], signs: &[f64]) -> Option<(String, f64)> {
    let (prefix, body) = label.split_once('(')?;
    if !["S", "D", "QS", "QD"].contains(&prefix) {
        return None;
    }
    let (occupied, virtuals) = body.strip_suffix(')')?.split_once("->")?;
    let mut sign = 1.;
    let m = permutation.len();
    let mut map = |part: &str| -> Option<Vec<usize>> {
        part.split(',')
            .map(|v| {
                let i = v.parse::<usize>().ok()?;
                if i >= 2 * m {
                    return None;
                }
                sign *= signs[i % m];
                Some(permutation[i % m] + (i / m) * m)
            })
            .collect()
    };
    let mut occupied = map(occupied)?;
    let mut virtuals = map(virtuals)?;
    if occupied.len() != virtuals.len()
        || occupied.len() != if prefix.ends_with('S') { 1 } else { 2 }
    {
        return None;
    }
    for indices in [&mut occupied, &mut virtuals] {
        if indices.len() == 2 && indices[0] > indices[1] {
            indices.swap(0, 1);
            if !prefix.starts_with('Q') {
                sign = -sign;
            }
        }
    }
    let list = |indices: &[usize]| {
        indices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    };
    Some((
        format!("{prefix}({}->{})", list(&occupied), list(&virtuals)),
        sign,
    ))
}
pub fn carry(previous: &Handover, current: Frame<'_>, threshold: f64) -> Result<Seed> {
    if !threshold.is_finite() || !(0. ..=1.).contains(&threshold) {
        return Err(Error::Invalid(
            "ansatz handover needs a valid previous sequence and an overlap threshold in [0,1]"
                .into(),
        ));
    }
    if previous.numbers != current.integrals.geometry.numbers()
        || previous.coefficients.ncols() != current.model.m
        || previous.particles != current.model.particles
        || previous.mapping != current.problem.encoding.mapping
    {
        return Ok(refused(
            "geometry changes the atomic composition, active register, occupations or mapping",
            0.,
        ));
    }
    if previous.labels.is_empty() {
        return Ok(refused("the previous ansatz is empty", 1.));
    }
    let overlap = normalized_overlap(previous, &current)?;
    let identity = DMatrix::identity(overlap.nrows(), overlap.ncols());
    let (permutation, phases) = response::match_orbitals(
        &current.integrals.context,
        &identity,
        &overlap,
        current.model.particles,
        0.,
    )?;
    let confidence = permutation
        .iter()
        .enumerate()
        .map(|(i, &j)| overlap[(i, j)].norm())
        .fold(1., f64::min);
    if confidence < threshold {
        return Ok(refused(
            format!("smallest matched orbital overlap {confidence:.3} is below {threshold:.3}"),
            confidence,
        ));
    }
    if phases.iter().any(|p| p.im.abs() > 1e-9) {
        return Ok(refused(
            "complex orbital phases require a complex excitation-frame handover",
            confidence,
        ));
    }
    let signs: Vec<_> = phases
        .iter()
        .map(|p| if p.re < 0. { -1. } else { 1. })
        .collect();
    let unchanged =
        permutation.iter().enumerate().all(|(i, &j)| i == j) && signs.iter().all(|s| *s == 1.);
    let available: BTreeMap<_, _> = current
        .problem
        .pool
        .iter()
        .enumerate()
        .map(|(i, g)| (g.label.as_str(), i))
        .collect();
    let mut moved = Vec::with_capacity(previous.labels.len());
    let mut angles = Vec::with_capacity(previous.labels.len());
    for (label, &angle) in previous.labels.iter().zip(&previous.parameters) {
        let renamed = if unchanged {
            Some((label.clone(), 1.))
        } else {
            rename(label, &permutation, &signs)
        };
        let Some((label, sign)) = renamed else {
            return Ok(refused(
                format!("operator {label} cannot follow this orbital matching"),
                confidence,
            ));
        };
        let Some(&index) = available.get(label.as_str()) else {
            return Ok(refused(
                format!("renamed operator {label} is absent from the current pool"),
                confidence,
            ));
        };
        moved.push(index);
        angles.push(angle * sign);
    }
    Ok(Seed {indices:moved,parameters:angles,confidence,accepted:true,
        reason:format!("previous geometry's ansatz matched in occupied and virtual domains; minimum normalized pseudo-orbital overlap {confidence:.3}")})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn excitation_renaming_preserves_fermionic_parity_and_qubit_commutation() {
        let permutation = [1, 0, 2, 3];
        let signs = [-1., 1., 1., 1.];
        assert_eq!(
            rename("D(0,1->2,3)", &permutation, &signs),
            Some(("D(0,1->2,3)".into(), 1.))
        );
        assert_eq!(
            rename("QD(0,1->2,3)", &permutation, &signs),
            Some(("QD(0,1->2,3)".into(), -1.))
        );
        assert_eq!(
            rename("S(0->2)", &permutation, &signs),
            Some(("S(1->2)".into(), -1.))
        );
        assert_eq!(rename("iP[XYZI]", &permutation, &signs), None);
    }
}
