//! Gauge-invariant subspace projection and Wilson-loop analysis primitives.
use super::{invalid, Crystal, PeriodicResult};
use crate::{
    math::{linalg::eigen, storage::MatrixSource},
    product,
    quantum::C64,
    Buffer, Context, Result,
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
pub fn wilson_loop(
    ctx: &Context,
    links: &[C64],
    count: usize,
    occupied: usize,
    tolerance: f64,
) -> Result<(f64, f64)> {
    if count == 0
        || occupied == 0
        || links.len() != product(&[count, occupied, occupied])?
        || !tolerance.is_finite()
        || tolerance <= 0.
        || tolerance >= 1.
        || links.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(invalid(
            "invalid Wilson-loop dimensions, values or tolerance",
        ));
    }
    let _memory = ctx.reserve(product(&[occupied, occupied, 16, 20])?)?;
    let mut phase = 0.;
    let mut minimum = 1f64;
    for i in 0..count {
        let a = DMatrix::from_row_slice(
            occupied,
            occupied,
            &links[i * occupied * occupied..(i + 1) * occupied * occupied],
        );
        let svd = a.svd(true, true);
        let singular = svd.singular_values.min();
        minimum = minimum.min(singular);
        if !singular.is_finite() || singular < tolerance {
            return Err(invalid(
                "singular occupied-space link: Berry phase is undefined",
            ));
        }
        // Polar unitarization keeps the determinant near unit modulus even
        // when a nonsingular raw determinant would underflow or overflow.
        let unitary = svd.u.unwrap() * svd.v_t.unwrap();
        phase += unitary.lu().determinant().arg();
    }
    Ok((
        (phase + std::f64::consts::PI).rem_euclid(2. * std::f64::consts::PI) - std::f64::consts::PI,
        minimum,
    ))
}
