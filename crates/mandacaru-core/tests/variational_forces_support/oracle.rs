//! The finite-difference route the analytic variational force replaced (MG-ZA, ZA-6), kept only
//! as a test oracle: a Richardson difference of the reduced Hamiltonian at fixed RDMs over
//! displaced SCF solutions, MP2 selections and transformations, the displaced orbitals followed
//! by assignment and, inside exactly degenerate clusters, by the polar factor of their overlap.
#![allow(dead_code)]
use mandacaru_core::{
    math::rigid,
    molecular::{
        response::{match_orbitals, orbitals, Stationarity, DEGENERACY_TOLERANCE},
        transform::Model,
        MeanField, MolecularIntegrals, ScfOptions,
    },
    product,
    quantum::{
        lagrangian::{self, Ansatz, Residual},
        properties::{self, SpatialRdms},
        CachedProblem, C64,
    },
    Buffer, Error, Geometry, LengthUnit, Result,
};
use nalgebra::DMatrix;
use std::sync::Arc;
type Matrix = DMatrix<C64>;
fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

pub struct Gradient {
    pub gradient: Buffer<f64>,
    pub uncorrected: Buffer<f64>,
    pub stationarity: Stationarity,
    pub energy: f64,
    pub directions: usize,
    pub step: f64,
    pub estimate: Option<f64>,
}
pub struct Options<'a> {
    pub ansatz: Option<Ansatz<'a>>,
    pub lagrangian: lagrangian::Options,
    pub active_spec: Option<&'a str>,
    pub step: f64,
    pub internal: bool,
    pub extrapolate: bool,
}
impl Default for Options<'_> {
    fn default() -> Self {
        Self {
            ansatz: None,
            lagrangian: lagrangian::Options::default(),
            active_spec: None,
            step: 1e-3,
            internal: false,
            extrapolate: true,
        }
    }
}
/// How the displaced frame is aligned to the reference one: aligned orbital `p` is
/// `sum_a new_a W[a, p]`, stored by column as its nonzero `(a, W[a, p])`. Outside degenerate
/// clusters a column holds one entry (the matched orbital and its phase).
pub struct Alignment {
    pub columns: Vec<Vec<(usize, C64)>>,
}
impl Alignment {
    pub fn from_matches(indices: &[usize], phases: &[C64]) -> Self {
        Self {
            columns: indices
                .iter()
                .zip(phases)
                .map(|(&i, &w)| vec![(i, w)])
                .collect(),
        }
    }
}

/// The label of each model orbital that makes "exactly degenerate" well defined: the orbital
/// energy of a canonical RHF frame, the natural occupation of a UHF frame or of a selector
/// rotation. `None` when no such label belongs to the frame.
pub fn degeneracy_labels(reference: &MeanField, model: &Model) -> Option<Vec<f64>> {
    let source: &[f64] = if let Some(rotation) = &model.orbital_rotation {
        &rotation.occupations
    } else if reference.method == "uhf" {
        &reference.natural_occupations
    } else if reference.method == "rhf" {
        reference.energies.first()?
    } else {
        return None;
    };
    model
        .orbital_indices
        .iter()
        .map(|&i| source.get(i).copied())
        .collect()
}

/// [`match_orbitals`], then, inside every cluster of exactly degenerate orbitals (labels within
/// [`DEGENERACY_TOLERANCE`] in both frames), the unitary that best maps the displaced cluster onto
/// the reference one (the polar factor of their overlap) instead of a permutation and phases. A
/// degenerate cluster's SCF orbitals are any rotation of one another, so following them by
/// assignment alone picks an arbitrary frame at each displacement; a state that is not symmetric
/// in the cluster then sees an energy that jumps between displacements. A split pair keeps its
/// physical, continuous rotation.
#[allow(clippy::too_many_arguments)]
pub fn align_orbitals(
    context: &mandacaru_core::Context,
    old: &Matrix,
    new: &Matrix,
    particles: [usize; 2],
    threshold: f64,
    old_labels: Option<&[f64]>,
    new_labels: Option<&[f64]>,
) -> Result<Alignment> {
    let (indices, phases) = match_orbitals(context, old, new, particles, threshold)?;
    let mut alignment = Alignment::from_matches(&indices, &phases);
    let (Some(old_labels), Some(new_labels)) = (old_labels, new_labels) else {
        return Ok(alignment);
    };
    let n = old.ncols();
    if old_labels.len() != n || new_labels.len() != n {
        return Err(invalid("orbital labels must have one entry per orbital"));
    }
    let mut visited = vec![false; n];
    for start in 0..n {
        if visited[start] {
            continue;
        }
        // A cluster: the orbitals degenerate with `start` in the reference frame and in its
        // occupation domain (mixing an occupied orbital with an empty one changes the state, not
        // the frame).
        let domain = |i: usize| {
            usize::from(i >= particles[0].min(particles[1]))
                + usize::from(i >= particles[0].max(particles[1]))
        };
        let cluster: Vec<usize> = (start..n)
            .filter(|&p| {
                !visited[p]
                    && domain(p) == domain(start)
                    && (old_labels[p] - old_labels[start]).abs() < DEGENERACY_TOLERANCE
            })
            .collect();
        for &p in &cluster {
            visited[p] = true;
        }
        let matched: Vec<usize> = cluster.iter().map(|&p| indices[p]).collect();
        let degenerate_after = matched
            .iter()
            .all(|&a| (new_labels[a] - new_labels[matched[0]]).abs() < DEGENERACY_TOLERANCE);
        if cluster.len() < 2 || !degenerate_after {
            continue;
        }
        let k = cluster.len();
        let overlap = Matrix::from_fn(k, k, |i, j| {
            (old.column(cluster[i]).adjoint() * new.column(matched[j]))[(0, 0)]
        });
        let svd = overlap.svd(true, true);
        let (u, v_t) = (svd.u.expect("requested"), svd.v_t.expect("requested"));
        if svd.singular_values.iter().any(|&s| s < threshold) {
            return Err(invalid("a geometry displacement cannot follow a degenerate orbital cluster; reduce the step"));
        }
        // Maximize Re tr(O W) over unitary W: W = V U^H for O = U S V^H.
        let w = v_t.adjoint() * u.adjoint();
        for (j, &p) in cluster.iter().enumerate() {
            alignment.columns[p] = (0..k).map(|i| (matched[i], w[(i, j)])).collect();
        }
    }
    Ok(alignment)
}

pub fn contracted(
    model: &Model,
    rdm: &SpatialRdms,
    indices: &[usize],
    phases: &[C64],
) -> Result<f64> {
    if indices.len() != rdm.m || phases.len() != rdm.m {
        return Err(invalid(
            "reduced Hamiltonian and spatial RDM dimensions disagree",
        ));
    }
    contracted_aligned(model, rdm, &Alignment::from_matches(indices, phases))
}

/// `<rdm, H>` with the Hamiltonian taken in the aligned frame:
/// `h'_pq = sum conj(W_ap) W_bq h_ab`, `g'_pqrs = sum conj(W_ap) conj(W_bq) W_cr W_ds g_abcd`.
pub fn contracted_aligned(model: &Model, rdm: &SpatialRdms, alignment: &Alignment) -> Result<f64> {
    let m = rdm.m;
    let w = &alignment.columns;
    if model.m != m || w.len() != m || w.iter().flatten().any(|(a, _)| *a >= m) {
        return Err(invalid(
            "reduced Hamiltonian and spatial RDM dimensions disagree",
        ));
    }
    let one = model.one.as_slice();
    let two = model.two.as_slice();
    let mut energy = C64::new(model.constant, 0.);
    for p in 0..m {
        for q in 0..m {
            let mut h = C64::default();
            for &(a, wa) in &w[p] {
                for &(b, wb) in &w[q] {
                    h += wa.conj() * wb * one[a * m + b];
                }
            }
            energy += rdm.one.as_slice()[p * m + q] * h;
            for r in 0..m {
                for s in 0..m {
                    let density = rdm.two.as_slice()[((p * m + q) * m + r) * m + s];
                    if density == C64::default() {
                        continue;
                    }
                    let mut g = C64::default();
                    for &(a, wa) in &w[p] {
                        for &(b, wb) in &w[q] {
                            let left = wa.conj() * wb.conj();
                            for &(c, wc) in &w[r] {
                                for &(d, wd) in &w[s] {
                                    g += left * wc * wd * two[((a * m + b) * m + c) * m + d];
                                }
                            }
                        }
                    }
                    energy += 0.5 * density * g;
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
        ansatz,
        lagrangian: lagrangian_options,
        active_spec,
        step,
        internal,
        extrapolate,
    } = *options;
    if integrals.samples.precision() != mandacaru_core::math::storage::Precision::Float64 {
        return Err(invalid(
            "variational forces difference the energy over a finite step and require float64 orbital samples",
        ));
    }
    if problem.taper.is_some() {
        return Err(invalid(
            "variational forces are not implemented on a Z2-tapered register: its state is read through the untapered density matrices",
        ));
    }
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
    let basis = problem.register("variational geometry response")?;
    (|| {
        let ctx = &integrals.context;
        let rdm = properties::spatial_rdms(ctx, &problem.encoding, basis, state)?;
        let old = orbitals(reference, model)?;
        let old_labels = degeneracy_labels(reference, model);
        let identity: Vec<_> = (0..model.m).collect();
        let unit = vec![C64::new(1., 0.); model.m];
        let energy = contracted(model, &rdm, &identity, &unit)?;
        let action = problem.hamiltonian.apply(ctx, basis, state)?;
        let expected = mandacaru_core::quantum::evolution::inner(state, action.as_slice()).re;
        if (energy - expected).abs() > 1e-7 {
            return Err(invalid("the force RDMs and molecular orbitals do not reconstruct the variational Hamiltonian energy"));
        }
        drop(action);
        let norm = state.iter().map(|v| v.norm_sqr()).sum::<f64>();
        if (norm - 1.).abs() > 1e-9 {
            return Err(invalid("variational forces require a normalized state"));
        }
        let residual = ansatz
            .map(|ansatz| problem.lagrangian(&ansatz, state, &lagrangian_options))
            .transpose()?;
        // The RDMs of phi± = psi ± s psi', whose contraction with H(R) gives z . g(R) exactly
        // (a quadratic form; the constant cancels in the difference).
        let shifted = match &residual {
            Some(Residual::Corrected { z, .. }) => {
                let plus =
                    properties::spatial_rdms(ctx, &problem.encoding, basis, z.plus.complex128()?)?;
                let minus =
                    properties::spatial_rdms(ctx, &problem.encoding, basis, z.minus.complex128()?)?;
                let at_reference = (contracted(model, &plus, &identity, &unit)?
                    - contracted(model, &minus, &identity, &unit)?)
                    / (2. * z.scale);
                if (at_reference - z.z_dot_g).abs() > 1e-8 * z.z_dot_g.abs() + 1e-9 {
                    return Err(invalid("the shifted-state RDMs do not reproduce z . g: the parameter gradient and the state disagree"));
                }
                Some((plus, minus, z.scale))
            }
            _ => None,
        };
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
        let mut uncorrected = ctx.allocate::<f64>(size)?;
        let mut estimate = 0f64;
        for direction in 0..directions.ncols() {
            let mut values = [0.; 4];
            let mut plain = [0.; 4];
            let displacements = if extrapolate {
                &[-1., 1., -0.5, 0.5][..]
            } else {
                &[-1., 1.][..]
            };
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
                let alignment = align_orbitals(
                    ctx,
                    &old,
                    &frame,
                    model.particles,
                    0.5,
                    old_labels.as_deref(),
                    degeneracy_labels(&scf, &reduced).as_deref(),
                )?;
                plain[at] = contracted_aligned(&reduced, &rdm, &alignment)?;
                values[at] = plain[at];
                if let Some((plus, minus, scale)) = &shifted {
                    // E - z . g at this geometry.
                    values[at] -= (contracted_aligned(&reduced, plus, &alignment)?
                        - contracted_aligned(&reduced, minus, &alignment)?)
                        / (2. * scale);
                }
            }
            let richardson = |values: &[f64; 4]| {
                let coarse = (values[1] - values[0]) / (2. * step);
                if extrapolate {
                    let fine = (values[3] - values[2]) / step;
                    ((4. * fine - coarse) / 3., (fine - coarse).abs() / 3.)
                } else {
                    (coarse, 0.)
                }
            };
            let (derivative, error) = richardson(&values);
            estimate = estimate.max(error);
            let (plain_derivative, _) = richardson(&plain);
            for row in 0..size {
                let direction_row = directions[(row, direction)];
                gradient.as_mut_slice()[row] += direction_row * derivative;
                uncorrected.as_mut_slice()[row] += direction_row * plain_derivative;
            }
        }
        let stationarity = match &residual {
            None => Stationarity::Assumed,
            Some(Residual::Stationary { gradient_norm }) => Stationarity::Stationary {
                gradient_norm: *gradient_norm,
            },
            Some(Residual::Corrected { gradient_norm, z }) => Stationarity::Lagrangian {
                gradient_norm: *gradient_norm,
                z_norm: z.z_norm,
                solve: z.solve.clone(),
            },
        };
        Ok(Gradient {
            gradient,
            uncorrected,
            stationarity,
            energy,
            directions: directions.ncols(),
            step,
            estimate: extrapolate.then_some(estimate),
        })
    })()
}
