//! Gauge-invariant subspace projection and Wilson-loop analysis primitives.
use super::{invalid, Crystal, PeriodicResult};
use crate::{molecular::scf::eigen, product, quantum::C64, Buffer, Context, Result};
use nalgebra::{DMatrix, DVector};
pub struct Projection {
    pub weights: Buffer<f64>,
    pub hamiltonians: Buffer<C64>,
}
pub fn projectability(
    c: &Crystal,
    result: &PeriodicResult,
    trials: &[C64],
    targets: usize,
) -> Result<Projection> {
    let m = c.m;
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
        let gram = t.adjoint() * &point.overlap * &t;
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
            let a = result.coefficients[spin * nk + k].adjoint() * &point.overlap * &normalized;
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
