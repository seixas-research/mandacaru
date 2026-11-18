//! Gauge-invariant subspace projection and projected densities of states (Berry phases and
//! Wilson loops are `periodic::berry`).
use super::{invalid, Crystal, PeriodicResult};
use crate::{
    math::{linalg::eigen, storage::MatrixSource},
    product,
    quantum::C64,
    Buffer, Result,
};
use nalgebra::{DMatrix, DVector};
pub struct Projection {
    pub weights: Buffer<f64>,
    pub hamiltonians: Buffer<C64>,
}
/// Projected DOS streams a single k-point's orbitals. The retained result
/// need not carry a dense (spin,k,basis,band) probability tensor.
pub fn projected_dos(
    c: &Crystal,
    result: &PeriodicResult,
    axis: &[f64],
    width: f64,
) -> Result<Buffer<f64>> {
    let ctx = &c.context;
    let m = c.bands();
    let mut out = ctx.allocate::<f64>(product(&[axis.len(), m])?)?;
    let _work = ctx.reserve(product(&[m, m, 16, 8])?)?;
    for spin in 0..c.spins {
        for (k, point) in c.points.iter().enumerate() {
            let coefficients = result.coefficients[spin * c.points.len() + k].matrix(ctx)?;
            let transform = &point.lowdin * &point.overlap;
            let vectors = if c.spinor {
                let mut vectors = DMatrix::<C64>::zeros(m, m);
                for s in 0..2 {
                    let block = &transform * coefficients.rows(s * c.m, c.m);
                    vectors.view_mut((s * c.m, 0), (c.m, m)).copy_from(&block);
                }
                vectors
            } else {
                transform * &*coefficients
            };
            let weights: Vec<_> = (0..m)
                .flat_map(|i| {
                    let v = &vectors;
                    (0..m).map(move |j| v[(i, j)].norm_sqr())
                })
                .collect();
            let values = super::spectrum::broaden(
                ctx,
                axis,
                std::slice::from_ref(&result.eigenvalues[spin * c.points.len() + k]),
                &[point.weight],
                width,
                c.degeneracy(),
                Some((&weights, m)),
            )?;
            for (to, from) in out.as_mut_slice().iter_mut().zip(values.as_slice()) {
                *to += from;
            }
        }
    }
    Ok(out)
}
pub fn projectability(
    c: &Crystal,
    result: &PeriodicResult,
    trials: &[C64],
    targets: usize,
) -> Result<Projection> {
    let m = c.bands();
    if targets == 0
        || targets > m
        || trials.len() != product(&[m, targets])?
        || trials
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(invalid("invalid periodic trial-space dimensions or values"));
    }
    let ctx = &c.context;
    let _memory = ctx.reserve(product(&[m, m, 16, 16])?)?;
    let t = DMatrix::from_row_slice(m, targets, trials);
    let nk = c.points.len();
    let mut weights = ctx.allocate::<f64>(product(&[c.spins, nk, m])?)?;
    let mut hamiltonians = ctx.allocate::<C64>(product(&[c.spins, nk, targets, targets])?)?;
    for (k, point) in c.points.iter().enumerate() {
        let overlap = if c.spinor {
            DMatrix::from_fn(m, m, |i, j| {
                if i / c.m == j / c.m {
                    point.overlap[(i % c.m, j % c.m)]
                } else {
                    C64::default()
                }
            })
        } else {
            point.overlap.clone()
        };
        let gram = t.adjoint() * &overlap * &t;
        let (s, u) = eigen(gram)?;
        if s[0] <= 1e-10 {
            return Err(invalid(
                "periodic trial functions are linearly dependent in the overlap metric",
            ));
        }
        let normalized = &t
            * &u
            * DMatrix::from_diagonal(&DVector::from_iterator(
                targets,
                s.iter().map(|s| C64::new(1. / s.sqrt(), 0.)),
            ))
            * u.adjoint();
        for spin in 0..c.spins {
            let a =
                result.coefficients[spin * nk + k].matrix(ctx)?.adjoint() * &overlap * &normalized;
            for b in 0..m {
                weights.as_mut_slice()[(spin * nk + k) * m + b] =
                    (0..targets).map(|j| a[(b, j)].norm_sqr()).sum();
            }
            let h = a.adjoint()
                * DMatrix::from_diagonal(&DVector::from_iterator(
                    m,
                    result.eigenvalues[spin * nk + k]
                        .iter()
                        .map(|e| C64::new(*e, 0.)),
                ))
                * &a;
            for i in 0..targets {
                for j in 0..targets {
                    hamiltonians.as_mut_slice()[((spin * nk + k) * targets + i) * targets + j] =
                        h[(i, j)];
                }
            }
        }
    }
    Ok(Projection {
        weights,
        hamiltonians,
    })
}

/// The bands of a converged crystal at fractional k-points off its mesh (P6): `eigenvalues`
/// `(spins, n_k, bands)` in Hartree with the crystal's reference added, and `projections`
/// `(spins, n_k, bands, bands)`, the `|y_ab|^2` of the eigenvectors in the orthogonalized basis.
pub struct OffMeshBands {
    pub eigenvalues: Vec<f64>,
    pub projections: Vec<f64>,
}

/// Diagonalizes the converged potential of `result` at `fractional_k` (with the hybrid exchange
/// at each point for a hybrid crystal). Each `(spin, k)` is one task in spin-major order, a serial
/// computation whatever lane runs it, so the numbers do not depend on the layout. A semilocal
/// crystal runs the tasks through [`crate::Context::map_independent`] (the layout's k-point lanes,
/// or the calling thread's workers), at most one lane per task (one point under `kpts = 4` still
/// runs); a hybrid crystal runs them in order on the calling thread, since its exchange opens its
/// own band lanes and the fast tests cannot gate the nesting.
pub fn bands_at(
    c: &Crystal,
    setup: &super::setup::PeriodicSetup,
    result: &PeriodicResult,
    fractional_k: &[[f64; 3]],
) -> Result<OffMeshBands> {
    let n_k = fractional_k.len();
    if n_k == 0 {
        return Err(invalid("fractional_k cannot be empty"));
    }
    let weights = vec![1.0 / n_k as f64; n_k];
    let points = setup.matrices_at(&c.context, fractional_k, &weights)?;
    bands_of_points(c, result, &points)
}

/// [`bands_at`] at k-points whose matrices are already assembled (`points`, in that order).
pub fn bands_of_points(
    c: &Crystal,
    result: &PeriodicResult,
    points: &[super::KPoint],
) -> Result<OffMeshBands> {
    let ctx = &c.context;
    let n_k = points.len();
    if n_k == 0 {
        return Err(invalid("bands need a nonempty k-point list"));
    }
    let m = c.bands();
    let _memory = ctx.reserve(product(&[
        n_k,
        c.spins,
        m,
        m.checked_add(c.p).ok_or(crate::Error::Overflow)?,
        16,
        16,
    ])?)?;
    let n = c.grid.index.points();
    let is_hybrid = c.screened_exchange()?.is_some();
    let tasks = product(&[c.spins, n_k])?;
    let solve = |task: usize| -> Result<(Vec<f64>, Vec<f64>)> {
        let (s, point) = (task / n_k, &points[task % n_k]);
        let exchange = if is_hybrid {
            Some(c.hybrid_exchange_at_k(
                &result.matrices,
                point,
                s,
                result.options.exxdiv_treatment,
            )?)
        } else {
            None
        };
        let potential = result.potential.as_slice();
        let (e, _, y) = super::solver::diagonalize_with_exchange(
            c,
            point,
            if c.spinor {
                potential
            } else {
                &potential[s * n..(s + 1) * n]
            },
            result.moment_potential.as_slice(),
            result
                .kinetic_potential
                .as_ref()
                .map(|v| &v.as_slice()[s * n..(s + 1) * n]),
            result.hubbard.as_ref().map(|h| &h.operators[s]),
            exchange.as_ref(),
        )?;
        let eigenvalues: Vec<f64> = e.iter().map(|e| e + c.reference).collect();
        let projections: Vec<f64> = (0..m)
            .flat_map(|a| (0..m).map(move |b| (a, b)))
            .map(|(a, b)| y[(a, b)].norm_sqr())
            .collect();
        Ok((eigenvalues, projections))
    };
    let solved = if is_hybrid {
        (0..tasks).map(solve).collect::<Result<Vec<_>>>()?
    } else {
        ctx.map_independent(tasks, solve)?
    };
    let mut out = OffMeshBands {
        eigenvalues: Vec::with_capacity(tasks * m),
        projections: Vec::with_capacity(tasks * m * m),
    };
    for (eigenvalues, projections) in solved {
        out.eigenvalues.extend(eigenvalues);
        out.projections.extend(projections);
    }
    Ok(OffMeshBands {
        eigenvalues: out.eigenvalues,
        projections: out.projections,
    })
}

/// Partition periodic crystal smooth electron density plus PAW on-site augmentation charges into
/// atomic partial charges (`"bader"`, `"hirshfeld"`, or `"voronoi"`).
pub fn partial_charges(
    c: &Crystal,
    result: &PeriodicResult,
    method: &str,
    numbers: &[u8],
    positions: Option<&[[f64; 3]]>,
    valence_charges: Option<&[f64]>,
    interpolate: bool,
) -> Result<crate::math::partial_charges::PartialCharges> {
    use std::f64::consts::PI;
    let centers: Vec<[f64; 3]> = match (positions, &c.description) {
        (Some(pos), _) => pos.to_vec(),
        (None, Some(desc)) => desc.centers.clone(),
        (None, None) => {
            return Err(invalid(
                "periodic partial charges need atomic positions or a setup description",
            ))
        }
    };
    let n_atoms = centers.len();
    if numbers.len() != n_atoms {
        return Err(invalid("atomic numbers count must match atomic positions"));
    }
    let valences: Vec<f64> = match valence_charges {
        Some(q) => {
            if q.len() != n_atoms {
                return Err(invalid("valence charges count must match atomic positions"));
            }
            q.to_vec()
        }
        None => numbers.iter().map(|&z| f64::from(z)).collect(),
    };
    let n = c.grid.index.points();
    let rho_slice = result.density.as_slice();
    let total_rho: Vec<f64> = if c.spinor || c.spins == 1 {
        rho_slice[..n].to_vec()
    } else {
        (0..n).map(|k| rho_slice[k] + rho_slice[n + k]).collect()
    };
    let mut augmentation = vec![0.; n_atoms];
    if let Some(desc) = &c.description {
        let sqrt_4pi = (4. * PI).sqrt();
        for (idx, &(atom, l, m)) in desc.channels.iter().enumerate() {
            if l == 0 && m == 0 && atom < n_atoms && idx < result.moments.as_slice().len() {
                augmentation[atom] += sqrt_4pi * result.moments.as_slice()[idx].re;
            }
        }
    }
    let proatomic = if method.eq_ignore_ascii_case("hirshfeld") {
        let mut radials = Vec::with_capacity(n_atoms);
        for (atom, (&z, &qv)) in numbers.iter().zip(&valences).enumerate() {
            let mut shells = Vec::new();
            if let Some(desc) = &c.description {
                let mut seen_tables = Vec::new();
                for (&fa, &ft) in desc.function_atom.iter().zip(&desc.function_table) {
                    if fa == atom && !seen_tables.contains(&ft) {
                        seen_tables.push(ft);
                        let t = &desc.tables[ft];
                        shells.push((t.l, t.r.as_slice(), t.values.as_slice()));
                    }
                }
            }
            radials.push(crate::models::configuration::proatomic_radial_from_shells(
                z, qv, &shells,
            ));
        }
        Some(radials)
    } else {
        None
    };
    crate::math::partial_charges::compute_partial_charges(
        &c.context,
        method,
        &c.grid,
        &total_rho,
        &centers,
        &valences,
        &augmentation,
        proatomic.as_deref(),
        interpolate,
    )
}
