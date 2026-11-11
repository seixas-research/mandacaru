//! The analytic nuclear gradient of a variational state (MG-ZA): the state's density matrices,
//! with the Lagrangian (Z-vector) correction of its residual parameter gradient folded in, are
//! relaxed for the response of the orbitals the Hamiltonian was built in (canonical
//! Hartree-Fock, MP2 natural orbitals, frozen orbitals; `molecular::variational_gradient`) and
//! contracted once with the derivative integrals of the PAW-LCAO model. No displaced SCF and no
//! finite difference: the route it replaced is a test oracle (`tests/variational_forces_support`).
use super::{invalid, transform::Model, variational_gradient, MeanField, MolecularIntegrals};
use crate::{
    math::{assignment, rigid},
    product,
    quantum::{
        lagrangian::{self, Ansatz, Residual},
        properties, CachedProblem, C64,
    },
    Buffer, Result,
};
use nalgebra::DMatrix;
type Matrix = DMatrix<C64>;

/// What the response knows about the parameter stationarity of the state it differentiated.
pub enum Stationarity {
    /// No ansatz was given: the state is taken as the exact optimum (an eigenvector, say), and
    /// the gradient is the Hellmann-Feynman one with the state's RDMs held fixed.
    Assumed,
    /// The parameter gradient `|g|` is under the floor; no correction was applied.
    Stationary { gradient_norm: f64 },
    /// `gradient` is the geometry derivative of `E - z . g`, with `H z = g` held fixed.
    Lagrangian {
        gradient_norm: f64,
        z_norm: f64,
        solve: lagrangian::Solve,
    },
}
pub struct Gradient {
    /// The derivative of the optimized surface, the Lagrangian correction included.
    pub gradient: Buffer<f64>,
    pub stationarity: Stationarity,
    pub energy: f64,
    /// The directions the gradient is given in: `3 N` Cartesian ones, or the internal ones.
    pub directions: usize,
    pub scheme: &'static str,
    /// The largest occupied-virtual Fock element of the reference (its own stationarity) and the
    /// residual of the CPHF solve.
    pub reference_stationarity: f64,
    pub cphf_residual: f64,
    /// The Coulomb forms of the relaxed two-body density.
    pub forms: usize,
}
pub struct Options<'a> {
    /// The optimized ansatz of the state; `None` takes the state as exactly stationary.
    pub ansatz: Option<Ansatz<'a>>,
    pub lagrangian: lagrangian::Options,
    /// Project onto the internal (non-rigid) molecular directions.
    pub internal: bool,
}
impl Default for Options<'_> {
    fn default() -> Self {
        Self {
            ansatz: None,
            lagrangian: lagrangian::Options::default(),
            internal: false,
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
/// Two orbitals whose labels (orbital energies, or natural occupations) differ by less than this
/// are one exactly degenerate cluster, inside which the SCF fixes no frame.
pub const DEGENERACY_TOLERANCE: f64 = 1e-8;

fn real(values: &[C64], what: &str) -> Result<Vec<f64>> {
    let scale = values.iter().fold(1f64, |a, v| a.max(v.norm()));
    if values.iter().any(|v| v.im.abs() > 1e-10 * scale) {
        return Err(invalid(format!(
            "the analytic variational force needs real {what}"
        )));
    }
    Ok(values.iter().map(|v| v.re).collect())
}

/// The analytic gradient (Hartree per Bohr) of the state `state` of `problem`, built on
/// `model` of `reference`. PAW-LCAO only: the all-electron basis raises by name.
pub fn gradient(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    model: &Model,
    problem: &CachedProblem,
    state: &[C64],
    options: &Options<'_>,
) -> Result<Gradient> {
    model.require_dense("an analytic variational force")?;
    if integrals.paw_setup.is_none() || integrals.corrections.is_none() {
        return Err(crate::Error::Unsupported(
            "variational forces are analytic and PAW-LCAO only; the all-electron (HAO) basis has none: \
             use basis=\"PAW-LCAO\""
                .into(),
        ));
    }
    if problem.taper.is_some() {
        return Err(invalid(
            "variational forces are not implemented on a Z2-tapered register: its state is read through the untapered density matrices",
        ));
    }
    if model.m * 2 != problem.encoding.modes || model.particles != problem.encoding.particles {
        return Err(invalid(
            "the variational state, its model and its reference disagree",
        ));
    }
    let basis = problem.register("variational geometry response")?;
    integrals.context.run(|| {
        let ctx = &integrals.context;
        let norm = state.iter().map(|v| v.norm_sqr()).sum::<f64>();
        if (norm - 1.).abs() > 1e-9 {
            return Err(invalid("variational forces require a normalized state"));
        }
        let action = problem.hamiltonian.apply(ctx, basis, state)?;
        let energy = crate::quantum::evolution::inner(state, action.as_slice()).re;
        drop(action);
        let residual = options
            .ansatz
            .map(|ansatz| problem.lagrangian(&ansatz, state, &options.lagrangian))
            .transpose()?;
        // E - z . g in one pass: with psi' the exact tangent along z (psi± = psi ± s psi'), the
        // densities of E - z . g are the symmetrized transition densities <psi| . |psi - 2 psi'>.
        let (chi, expected) = match &residual {
            Some(Residual::Corrected { z, .. }) => {
                let (plus, minus) = (z.plus.complex128()?, z.minus.complex128()?);
                let mut chi = ctx.allocate::<C64>(state.len())?;
                for (((c, s), p), m) in chi.as_mut_slice().iter_mut().zip(state).zip(plus).zip(minus) {
                    *c = s - (p - m) / z.scale;
                }
                (Some(chi), energy - z.z_dot_g)
            }
            _ => (None, energy),
        };
        let ket = chi.as_ref().map_or(state, |c| c.as_slice());
        let rdm = properties::spatial_transition_rdms(ctx, &problem.encoding, basis, state, ket)?;
        drop(chi);
        let m = rdm.m;
        let (t1, t2) = (rdm.one.as_slice(), rdm.two.as_slice());
        let one = real(
            &(0..m * m).map(|k| 0.5 * (t1[k] + t1[(k % m) * m + k / m].conj())).collect::<Vec<_>>(),
            "one-body density matrices",
        )?;
        let two = real(
            &(0..m.pow(4))
                .map(|k| {
                    let (p, q, r, s) = (k / (m * m * m), (k / (m * m)) % m, (k / m) % m, k % m);
                    0.5 * (t2[k] + t2[((r * m + s) * m + p) * m + q].conj())
                })
                .collect::<Vec<_>>(),
            "two-body density matrices",
        )?;
        drop(rdm);
        let relaxed = variational_gradient::relaxed_densities(integrals, reference, model, &one, &two)?;
        if (relaxed.energy - expected).abs() > 1e-7 {
            return Err(invalid("the force RDMs and molecular orbitals do not reconstruct the variational Hamiltonian energy"));
        }
        let analytic = integrals.fixed_density_gradient(&relaxed.densities)?;
        let size = 3 * integrals.geometry.numbers().len();
        let _workspace = ctx.reserve(product(&[size, size + 8, 16])?)?;
        let projector = if options.internal {
            let q = rigid::internal(ctx, integrals.geometry.positions())?.matrix;
            Some((&q * q.transpose(), q.ncols()))
        } else {
            None
        };
        let output = |values: &[f64]| -> Result<Buffer<f64>> {
            let mut out = ctx.allocate::<f64>(size)?;
            match &projector {
                Some((p, _)) => {
                    let projected = p * nalgebra::DVector::from_column_slice(values);
                    out.as_mut_slice().copy_from_slice(projected.as_slice());
                }
                None => out.as_mut_slice().copy_from_slice(values),
            }
            Ok(out)
        };
        let total = output(analytic.total.as_slice())?;
        let stationarity = match &residual {
            None => Stationarity::Assumed,
            Some(Residual::Stationary { gradient_norm }) => Stationarity::Stationary { gradient_norm: *gradient_norm },
            Some(Residual::Corrected { gradient_norm, z }) => Stationarity::Lagrangian {
                gradient_norm: *gradient_norm,
                z_norm: z.z_norm,
                solve: z.solve.clone(),
            },
        };
        Ok(Gradient {
            gradient: total,
            stationarity,
            energy,
            directions: projector.as_ref().map_or(size, |(_, k)| *k),
            scheme: "analytic",
            reference_stationarity: relaxed.stationarity,
            cphf_residual: relaxed.cphf_residual,
            forms: relaxed.densities.two.len(),
        })
    })
}

/// Spin-summed spatial density matrices measured on a device, with the standard error of every
/// element (`None` when the device reports none: an exact circuit device).
pub struct MeasuredDensities<'a> {
    pub one: &'a [f64],
    pub two: &'a [f64],
    pub one_error: Option<&'a [f64]>,
    pub two_error: Option<&'a [f64]>,
}

/// The Z-vector correction of a device force: `z` (solved from the measured gradient and
/// Hessian), its covariance from the measured energies' errors (row-major `n x n`; `None` on an
/// exact device), and the shifted circuits' densities, each with its parameter `k` and
/// coefficient `c` (its weight is `z_k c`; `quantum::execution::shifted_circuits`).
pub struct DeviceCorrection<'a> {
    pub z: &'a [f64],
    pub z_covariance: Option<&'a [f64]>,
    pub shifted: Vec<(usize, f64, MeasuredDensities<'a>)>,
}

/// The analytic force of measured density matrices and its statistical error.
pub struct DeviceGradient {
    pub gradient: Buffer<f64>,
    /// The standard error of every component: the RDM part `diag(J Sigma J^T)` and the Z-vector
    /// part `diag(G Cov(z) G^T)` added (independent jobs).
    pub error: Buffer<f64>,
    /// The Z-vector part of the error alone (zero without a covariance).
    pub z_vector_error: Buffer<f64>,
    /// The unique measured elements whose errors were propagated.
    pub elements: usize,
    pub directions: usize,
    pub reference_stationarity: f64,
    pub cphf_residual: f64,
}

/// The largest number of unique measured elements whose errors [`device_gradient`] propagates.
pub const DEVICE_ERROR_ELEMENTS: usize = 256;

/// The force of a device state from its measured density matrices: `base` the state's and the
/// `correction`'s shifted circuits, whose energies give `z . g` (`None` when no correction is
/// applied), so that the densities of `E - z . g` are `base - sum_a z_k c_a shifted_a`.
/// The force is affine in the densities, so its error is exact: one force per unique element
/// (an orbit of the real-RDM symmetries) gives the Jacobian `J`, and
/// `Var F = J Sigma J^T` with `Sigma` the elements' variances (`base` plus `weight^2` times each
/// shifted circuit's, the circuits being independent jobs of independent shots). Elements of one
/// circuit are taken as uncorrelated with each other; each element's own error carries the
/// in-basis covariances of its strings. More than [`DEVICE_ERROR_ELEMENTS`] elements with an
/// error raise by name. The error of `z` (from the Hessian job, independent of the RDM job) adds
/// `G Cov(z) G^T`, with `G_k = dF/dz_k = -J sum_{a in k} c_a shifted_a`, one force per parameter.
/// Both parts are first order: the product of the two noises is left out.
pub fn device_gradient(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    model: &Model,
    base: &MeasuredDensities<'_>,
    correction: Option<&DeviceCorrection<'_>>,
    internal: bool,
) -> Result<DeviceGradient> {
    model.require_dense("an analytic variational force")?;
    if integrals.paw_setup.is_none() || integrals.corrections.is_none() {
        return Err(crate::Error::Unsupported(
            "variational forces are analytic and PAW-LCAO only; the all-electron (HAO) basis has none: \
             use basis=\"PAW-LCAO\""
                .into(),
        ));
    }
    let m = model.m;
    let (m2, m4) = (m * m, m.pow(4));
    let sized = |d: &MeasuredDensities<'_>| {
        d.one.len() == m2
            && d.two.len() == m4
            && d.one_error.is_none_or(|e| e.len() == m2)
            && d.two_error.is_none_or(|e| e.len() == m4)
            && d.one.iter().chain(d.two).all(|v| v.is_finite())
    };
    let no_shift = Vec::new();
    let (z, shifted_circuits) = match correction {
        Some(c) => (c.z, &c.shifted),
        None => (&[][..], &no_shift),
    };
    let parameters = z.len();
    if !sized(base) || shifted_circuits.iter().any(|(_, c, d)| !c.is_finite() || !sized(d)) {
        return Err(invalid("measured density matrices must be finite, m^2 and m^4, over the model's active orbitals"));
    }
    if z.iter().any(|v| !v.is_finite())
        || shifted_circuits.iter().any(|(k, _, _)| *k >= parameters)
        || correction
            .and_then(|c| c.z_covariance)
            .is_some_and(|cov| cov.len() != parameters * parameters || cov.iter().any(|v| !v.is_finite()))
    {
        return Err(invalid(
            "the Z vector, its covariance (n x n) and the shifted circuits' parameters must be finite and agree",
        ));
    }
    let shifted: Vec<(f64, &MeasuredDensities<'_>)> =
        shifted_circuits.iter().map(|(k, c, d)| (z[*k] * c, d)).collect();
    integrals.context.run(|| {
        let ctx = &integrals.context;
        let mut one = base.one.to_vec();
        let mut two = base.two.to_vec();
        let square = |e: Option<&[f64]>, k: usize| e.map_or(0., |e| e[k] * e[k]);
        let mut var_one: Vec<f64> = (0..m2).map(|k| square(base.one_error, k)).collect();
        let mut var_two: Vec<f64> = (0..m4).map(|k| square(base.two_error, k)).collect();
        for &(weight, d) in &shifted {
            for k in 0..m2 {
                one[k] -= weight * d.one[k];
                var_one[k] += weight * weight * square(d.one_error, k);
            }
            for k in 0..m4 {
                two[k] -= weight * d.two[k];
                var_two[k] += weight * weight * square(d.two_error, k);
            }
        }
        let size = 3 * integrals.geometry.numbers().len();
        let projector = if internal {
            let q = rigid::internal(ctx, integrals.geometry.positions())?.matrix;
            Some((&q * q.transpose(), q.ncols()))
        } else {
            None
        };
        let project = |values: &[f64]| -> Vec<f64> {
            match &projector {
                Some((p, _)) => (p * nalgebra::DVector::from_column_slice(values)).iter().copied().collect(),
                None => values.to_vec(),
            }
        };
        let force = |one: &[f64], two: &[f64]| -> Result<(Vec<f64>, variational_gradient::VariationalDensities)> {
            let relaxed = variational_gradient::relaxed_densities(integrals, reference, model, one, two)?;
            let g = integrals.fixed_density_gradient(&relaxed.densities)?;
            Ok((project(g.total.as_slice()), relaxed))
        };
        let (total, relaxed) = force(&one, &two)?;
        // The orbits of the real-RDM symmetries with a nonzero variance.
        let mut orbits: Vec<(bool, Vec<usize>, f64)> = Vec::new();
        for p in 0..m {
            for q in p..m {
                let variance = var_one[p * m + q];
                if variance > 0. {
                    let mut members = vec![p * m + q, q * m + p];
                    members.dedup();
                    orbits.push((false, members, variance));
                }
            }
        }
        let at = |p: usize, q: usize, r: usize, s: usize| ((p * m + q) * m + r) * m + s;
        let mut seen = vec![false; m4];
        for p in 0..m {
            for q in 0..m {
                for r in 0..m {
                    for s in 0..m {
                        let k = at(p, q, r, s);
                        if seen[k] {
                            continue;
                        }
                        let mut members = vec![k, at(q, p, s, r), at(r, s, p, q), at(s, r, q, p)];
                        members.sort_unstable();
                        members.dedup();
                        for &member in &members {
                            seen[member] = true;
                        }
                        let variance = members.iter().map(|&i| var_two[i]).fold(0f64, f64::max);
                        if variance > 0. {
                            orbits.push((true, members, variance));
                        }
                    }
                }
            }
        }
        let z_covariance = correction.and_then(|c| c.z_covariance).filter(|cov| cov.iter().any(|v| *v != 0.));
        if orbits.len() > DEVICE_ERROR_ELEMENTS {
            return Err(crate::Error::Unsupported(format!(
                "the statistical error of a device force is propagated exactly up to {DEVICE_ERROR_ELEMENTS} unique \
                 measured RDM elements; this state has {} with an error",
                orbits.len()
            )));
        }
        let mut variance = vec![0.; total.len()];
        let zero = if !orbits.is_empty() || z_covariance.is_some() {
            Some(force(&vec![0.; m2], &vec![0.; m4])?.0)
        } else {
            None
        };
        if let Some(zero) = &zero {
            for (is_two, members, orbit_variance) in &orbits {
                let (mut unit_one, mut unit_two) = (vec![0.; m2], vec![0.; m4]);
                for &k in members {
                    if *is_two { unit_two[k] = 1. } else { unit_one[k] = 1. }
                }
                let (column, _) = force(&unit_one, &unit_two)?;
                for c in 0..total.len() {
                    let j = column[c] - zero[c];
                    variance[c] += j * j * orbit_variance;
                }
            }
        }
        let mut z_variance = vec![0.; total.len()];
        if let (Some(cov), Some(zero)) = (z_covariance, &zero) {
            // G_k = dF/dz_k = -(F(sum_{a in k} c_a D_a) - F(0)).
            let mut sensitivity = vec![vec![0.; total.len()]; parameters];
            for (k, row) in sensitivity.iter_mut().enumerate() {
                if (0..parameters).all(|l| cov[k * parameters + l] == 0.) {
                    continue;
                }
                let (mut d_one, mut d_two) = (vec![0.; m2], vec![0.; m4]);
                for (_, c, d) in shifted_circuits.iter().filter(|(j, _, _)| *j == k) {
                    d_one.iter_mut().zip(d.one).for_each(|(x, v)| *x += c * v);
                    d_two.iter_mut().zip(d.two).for_each(|(x, v)| *x += c * v);
                }
                let (column, _) = force(&d_one, &d_two)?;
                for (g, (f, f0)) in row.iter_mut().zip(column.iter().zip(zero)) {
                    *g = -(f - f0);
                }
            }
            for (c, v) in z_variance.iter_mut().enumerate() {
                let mut sum = 0.;
                for k in 0..parameters {
                    for l in 0..parameters {
                        sum += sensitivity[k][c] * cov[k * parameters + l] * sensitivity[l][c];
                    }
                }
                *v = sum.max(0.);
            }
        }
        let mut gradient = ctx.allocate::<f64>(size)?;
        gradient.as_mut_slice().copy_from_slice(&total);
        let mut error = ctx.allocate::<f64>(size)?;
        for (e, (v, vz)) in error.as_mut_slice().iter_mut().zip(variance.iter().zip(&z_variance)) {
            *e = (v + vz).sqrt();
        }
        let mut z_vector_error = ctx.allocate::<f64>(size)?;
        for (e, v) in z_vector_error.as_mut_slice().iter_mut().zip(&z_variance) {
            *e = v.sqrt();
        }
        Ok(DeviceGradient {
            gradient,
            error,
            z_vector_error,
            elements: orbits.len(),
            directions: projector.as_ref().map_or(size, |(_, k)| *k),
            reference_stationarity: relaxed.stationarity,
            cphf_residual: relaxed.cphf_residual,
        })
    })
}
