//! Geometry derivatives of the same reduced Hamiltonian used by a state.
//! SCF and orbital selection are rebuilt in Rust, retaining their response.
use super::{invalid, transform::Model, MeanField, MolecularIntegrals, ScfOptions};
use crate::{
    math::{assignment, rigid},
    product,
    quantum::{
        properties::{self, SpatialRdms},
        CachedProblem, C64,
    },
    Buffer, Geometry, LengthUnit, Result,
};
use nalgebra::DMatrix;
use std::sync::Arc;
type Matrix = DMatrix<C64>;

pub struct Gradient {
    pub gradient: Buffer<f64>,
    pub energy: f64,
    pub directions: usize,
    pub step: f64,
    pub estimate: Option<f64>,
    pub scheme: &'static str,
}
pub struct Options<'a> {
    pub active_spec: Option<&'a str>,
    pub step: f64,
    pub internal: bool,
    pub extrapolate: bool,
}
impl Default for Options<'_> {
    fn default() -> Self {
        Self {
            active_spec: None,
            step: 1e-3,
            internal: false,
            extrapolate: true,
        }
    }
}
pub fn orbitals(reference: &MeanField, model: &Model) -> Result<Matrix> {
    let source = &reference.model_orbitals;
    let rotated = if let Some(rotation) = &model.orbital_rotation {
        source * &rotation.matrix
    } else {
        source.clone()
    };
    if model.spinor || model.orbital_indices.iter().any(|i| *i >= rotated.ncols()) {
        return Err(invalid(
            "spatial geometry response requires collinear molecular orbitals",
        ));
    }
    Ok(Matrix::from_fn(rotated.nrows(), model.m, |r, c| {
        rotated[(r, model.orbital_indices[c])]
    }))
}
/// Match the same occupied/singly-occupied/virtual orbital domains and phases.
/// This removes eigenvector gauge changes, while retaining continuous SCF and
/// selector rotation response in the displaced Hamiltonian.
pub fn match_orbitals(
    context: &crate::Context,
    old: &Matrix,
    new: &Matrix,
    particles: [usize; 2],
    threshold: f64,
) -> Result<(Vec<usize>, Vec<C64>)> {
    if old.shape() != new.shape() || !threshold.is_finite() || !(0. ..=1.).contains(&threshold) {
        return Err(invalid(
            "orbital matching needs equally sized orbital frames and an overlap threshold in [0,1]",
        ));
    }
    let n = old.ncols();
    if particles.iter().any(|p| *p > n) {
        return Err(invalid("orbital occupations exceed their frame"));
    }
    let _memory = context.reserve(product(&[n, n, 32])?)?;
    let overlap = old.adjoint() * new;
    let domain = |i| {
        if i < particles[0].min(particles[1]) {
            0
        } else if i < particles[0].max(particles[1]) {
            1
        } else {
            2
        }
    };
    let cost = DMatrix::from_fn(n, n, |i, j| {
        if domain(i) == domain(j) {
            -overlap[(i, j)].norm_sqr()
        } else {
            1e6
        }
    });
    let assignment = assignment::minimum(context, &cost)?;
    let mut phases = Vec::with_capacity(n);
    for (i, &j) in assignment.iter().enumerate() {
        let value = overlap[(i, j)];
        if domain(i) != domain(j) || value.norm() < threshold || value.norm() == 0. {
            return Err(invalid("a geometry displacement cannot follow an orbital through its occupation domain; reduce the step or resolve the crossing"));
        }
        phases.push(value.conj() / value.norm());
    }
    Ok((assignment, phases))
}
pub fn contracted(
    model: &Model,
    rdm: &SpatialRdms,
    indices: &[usize],
    phases: &[C64],
) -> Result<f64> {
    let m = rdm.m;
    if model.m != m || indices.len() != m || phases.len() != m || indices.iter().any(|i| *i >= m) {
        return Err(invalid(
            "reduced Hamiltonian and spatial RDM dimensions disagree",
        ));
    }
    let mut energy = C64::new(model.constant, 0.);
    for p in 0..m {
        for q in 0..m {
            energy += rdm.one.as_slice()[p * m + q]
                * model.one.as_slice()[indices[p] * m + indices[q]]
                * phases[p].conj()
                * phases[q];
            for r in 0..m {
                for s in 0..m {
                    energy += 0.5
                        * rdm.two.as_slice()[((p * m + q) * m + r) * m + s]
                        * model.two.as_slice()
                            [((indices[p] * m + indices[q]) * m + indices[r]) * m + indices[s]]
                        * phases[p].conj()
                        * phases[q].conj()
                        * phases[r]
                        * phases[s];
                }
            }
        }
    }
    if !energy.re.is_finite() || energy.im.abs() > 1e-8 * energy.re.abs().max(1.) {
        return Err(invalid(
            "RDM contraction produced a nonfinite or complex energy",
        ));
    }
    Ok(energy.re)
}
pub fn gradient(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    model: &Model,
    problem: &CachedProblem,
    state: &[C64],
    options: &Options<'_>,
) -> Result<Gradient> {
    let Options {
        active_spec,
        step,
        internal,
        extrapolate,
    } = *options;
    if !step.is_finite()
        || step <= 0.
        || reference.functional.is_some()
        || reference.method == "ghf"
        || !reference.converged
        || model.m * 2 != problem.encoding.modes
        || model.particles != problem.encoding.particles
    {
        return Err(invalid("variational geometry response needs a converged collinear HF reference, matching molecular state, and positive displacement step"));
    }
    integrals.context.run(|| {
        let ctx = &integrals.context;
        let rdm = properties::spatial_rdms(ctx, &problem.encoding, &problem.basis, state)?;
        let old = orbitals(reference, model)?;
        let identity: Vec<_> = (0..model.m).collect();
        let unit = vec![C64::new(1., 0.); model.m];
        let energy = contracted(model, &rdm, &identity, &unit)?;
        let action=problem.hamiltonian.apply(ctx,&problem.basis,state)?;
        let expected=crate::quantum::evolution::inner(state,action.as_slice()).re;
        if (energy-expected).abs()>1e-7 {
            return Err(invalid("the force RDMs and molecular orbitals do not reconstruct the variational Hamiltonian energy"));
        }
        drop(action);
        let norm = state.iter().map(|v| v.norm_sqr()).sum::<f64>();
        if (norm - 1.).abs() > 1e-9 {
            return Err(invalid("variational forces require a normalized state"));
        }
        let geometry = &integrals.geometry;
        let size = geometry.numbers().len() * 3;
        let _workspace = ctx.reserve(product(&[size, size + 8, 16])?)?;
        let internal_basis = if internal {
            Some(rigid::internal(ctx, geometry.positions())?)
        } else {
            None
        };
        let directions = internal_basis.as_ref().map_or_else(
            || DMatrix::identity(size, size),
            |basis| basis.matrix.clone(),
        );
        let mut gradient = ctx.allocate::<f64>(size)?;
        let mut estimate = 0f64;
        for direction in 0..directions.ncols() {
            let mut values = [0.; 4];
            let displacements = if extrapolate { &[-1., 1., -0.5, 0.5][..] } else { &[-1., 1.][..] };
            for (at, &sign) in displacements.iter().enumerate() {
                let positions = geometry
                    .positions()
                    .iter()
                    .enumerate()
                    .map(|(a, r)| {
                        std::array::from_fn(|i| {
                            r[i] + sign * step * directions[(3 * a + i, direction)]
                        })
                    })
                    .collect();
                let displaced = Geometry::new(
                    geometry.numbers().to_vec(),
                    positions,
                    *geometry.cell(),
                    geometry.pbc(),
                    LengthUnit::Bohr,
                )?;
                let rebuilt = integrals.at_geometry_with_fixed_pivots(displaced)?;
                let scf = rebuilt.continue_mean_field(
                    reference,
                    &ScfOptions {
                        tolerance: 1e-11,
                        guesses: 1,
                        max_iterations: 500,
                        diis: true,
                    },
                )?;
                if !scf.converged {
                    return Err(invalid(
                        "displaced SCF failed while differentiating the variational Hamiltonian",
                    ));
                }
                let reduced = if let Some(spec) = active_spec {
                    Arc::new(scf.active_model(&rebuilt, spec)?)
                } else {
                    scf.model(&rebuilt)?
                };
                if reduced.particles != model.particles
                    || reduced.frozen_indices.len() != model.frozen_indices.len()
                {
                    return Err(invalid(
                        "a geometry displacement changes the active particle or frozen-core count",
                    ));
                }
                let frame = orbitals(&scf, &reduced)?;
                let (indices, phases) = match_orbitals(ctx, &old, &frame, model.particles, 0.5)?;
                values[at] = contracted(&reduced, &rdm, &indices, &phases)?;
            }
            let coarse = (values[1] - values[0]) / (2. * step);
            let derivative = if extrapolate {
                let fine = (values[3] - values[2]) / step;
                estimate = estimate.max((fine - coarse).abs() / 3.);
                (4. * fine - coarse) / 3.
            } else { coarse };
            for row in 0..size {
                gradient.as_mut_slice()[row] += directions[(row, direction)] * derivative;
            }
        }
        Ok(Gradient {
            gradient,
            energy,
            directions: directions.ncols(),
            step,
            estimate: extrapolate.then_some(estimate),
            scheme: if extrapolate { "Richardson-reduced-Hamiltonian" } else { "Central-reduced-Hamiltonian" },
        })
    })
}
