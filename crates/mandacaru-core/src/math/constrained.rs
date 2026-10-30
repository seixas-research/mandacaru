//! Quadratic minimization with linear constraints and a fixed positive metric norm.
use crate::{Error, Result};
use nalgebra::{DMatrix as M, DVector as V};

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

/// Global minimizer on the feasible ellipsoid, including the trust-region hard case.
pub fn minimum(
    k: &M<f64>,
    linear: &V<f64>,
    a: &M<f64>,
    b: &V<f64>,
    g: &M<f64>,
    norm: f64,
) -> Result<V<f64>> {
    let n = k.nrows();
    if n == 0
        || k.ncols() != n
        || g.shape() != (n, n)
        || linear.len() != n
        || a.ncols() != n
        || a.nrows() != b.len()
        || !norm.is_finite()
        || norm < 0.
        || k.iter()
            .chain(g.iter())
            .chain(linear.iter())
            .chain(a.iter())
            .chain(b.iter())
            .any(|x| !x.is_finite())
    {
        return Err(invalid(
            "invalid constrained quadratic dimensions or values",
        ));
    }
    // Eliminate the linear constraints exactly: c = c0 + Z y, with c0 the
    // minimum-norm solution of A c = b and Z an orthonormal basis of null(A).
    let (rows, columns) = a.shape();
    let size = rows.max(columns);
    let mut padded = M::zeros(size, columns);
    padded.rows_mut(0, rows).copy_from(a);
    let svd = padded.svd(true, true);
    let (u, vt, sigma) = (svd.u.unwrap(), svd.v_t.unwrap(), svd.singular_values);
    let top = sigma.max();
    if top == 0. && b.iter().any(|v| v.abs() > 1e-12) {
        return Err(invalid("inconsistent zero constraint"));
    }
    let mut rhs = V::zeros(size);
    rhs.rows_mut(0, rows).copy_from(b);
    let projection = u.transpose() * rhs;
    let mut origin = V::zeros(n);
    for i in 0..sigma.len() {
        if sigma[i] > f64::EPSILON * size as f64 * top {
            origin += vt.row(i).transpose() * (projection[i] / sigma[i]);
        }
    }
    let rank = (0..sigma.len()).filter(|&i| sigma[i] > 1e-10 * top).count();
    let mut order = (0..sigma.len()).collect::<Vec<_>>();
    order.sort_by(|&x, &y| sigma[y].total_cmp(&sigma[x]));
    let null = order[rank..]
        .iter()
        .map(|&i| vt.row(i).transpose())
        .collect::<Vec<_>>();
    if null.is_empty() {
        if (origin.dot(&(g * &origin)) - norm).abs() > 1e-9 * norm.max(1.) {
            return Err(invalid(
                "norm is incompatible with fixed linear constraints",
            ));
        }
        return Ok(origin);
    }
    let z = M::from_columns(&null);
    let kz = z.transpose() * k * &z;
    let gz = z.transpose() * g * &z;
    let kp = z.transpose() * (k * &origin + linear);
    let gp = z.transpose() * (g * &origin);
    // The pencil (Kz, Gz) diagonalized once, V^T Gz V = 1 and V^T Kz V = diag(mu), so
    // y(lambda) = V (V^T r) / (mu - lambda) needs no solve and stays exact arbitrarily
    // close to the pole, where a linear solve of Kz - lambda Gz is singular.
    let cholesky = gz.clone().cholesky().ok_or_else(|| {
        invalid("norm metric is not positive definite on the constraint null space")
    })?;
    let l = cholesky.l();
    let l_inverse = l
        .solve_lower_triangular(&M::identity(l.nrows(), l.nrows()))
        .ok_or_else(|| invalid("singular norm metric"))?;
    // Rounding makes the reduced matrix slightly asymmetric; the symmetric part is what
    // the eigensolver is asked about.
    let reduced = &l_inverse * &kz * l_inverse.transpose();
    let (values, w) = super::linalg::real_eigen(0.5 * (&reduced + reduced.transpose()))?;
    let vectors = l_inverse.transpose() * w;
    let solution = |lambda: f64| -> V<f64> {
        let coefficients = vectors.transpose() * (gp.clone() * lambda - &kp);
        let scaled = V::from_iterator(
            values.len(),
            coefficients
                .iter()
                .zip(&values)
                .map(|(c, mu)| c / (mu - lambda)),
        );
        &origin + &z * (&vectors * scaled)
    };
    let residual = |lambda: f64| {
        let c = solution(lambda);
        c.dot(&(g * &c)) - norm
    };
    let mu = values[0];
    let mut hi = mu - 1e-9 * mu.abs().max(1.);
    let mut lo = hi - 1.;
    while residual(lo) > 0. {
        lo = hi - 2. * (hi - lo);
        if hi - lo > 1e12 {
            return Err(invalid(
                "linear constraints require more norm than requested",
            ));
        }
    }
    // Close to the pole the norm grows without bound; unless it stays finite there
    // and below its target (the hard case).
    if residual(hi) < 0. {
        for _ in 0..60 {
            let nearer = mu - 0.5 * (mu - hi);
            if !(hi < nearer && nearer < mu) {
                break;
            }
            hi = nearer;
            if residual(hi) > 0. {
                break;
            }
        }
    }
    let out = if residual(hi) < 0. {
        // Lambda at the pole, the other components as usual, plus the multiple of the lowest
        // eigenvector that meets the norm (of its two signs, the lower objective).
        let mut coefficients = vectors.transpose() * (gp.clone() * mu - &kp);
        coefficients[0] = 0.;
        for i in 1..coefficients.len() {
            coefficients[i] /= values[i] - mu;
        }
        let base = &origin + &z * (&vectors * coefficients);
        let direction = &z * vectors.column(0);
        let aa = direction.dot(&(g * &direction));
        let bb = 2. * direction.dot(&(g * &base));
        let cc = base.dot(&(g * &base)) - norm;
        let discriminant = bb * bb - 4. * aa * cc;
        if discriminant < -1e-9 * bb.abs().max(1.) {
            return Err(invalid("fixed norm has no feasible solution"));
        }
        let root = discriminant.max(0.).sqrt();
        let plus = &base + &direction * ((-bb + root) / (2. * aa));
        let minus = &base + &direction * ((-bb - root) / (2. * aa));
        let objective = |c: &V<f64>| c.dot(&(k * c)) + 2. * linear.dot(c);
        if objective(&plus) < objective(&minus) {
            plus
        } else {
            minus
        }
    } else {
        let lambda = super::roots::brent(|x| Ok(residual(x)), lo, hi, 1e-15, 1e-15, 500)?;
        solution(lambda)
    };
    if out.iter().any(|v| !v.is_finite())
        || (a * &out - b).amax() > 1e-7 * b.amax().max(1.)
        || (out.dot(&(g * &out)) - norm).abs() > 1e-7 * norm.max(1.)
    {
        return Err(invalid(
            "constrained quadratic failed its feasibility checks",
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_coordinate_and_hard_case() {
        let k = M::from_diagonal(&V::from_vec(vec![4., 1., 3.]));
        let g = M::identity(3, 3);
        let a = M::from_row_slice(1, 3, &[1., 0., 0.]);
        let b = V::from_vec(vec![0.5]);
        let c = minimum(&k, &V::zeros(3), &a, &b, &g, 1.).unwrap();
        assert!((c[0] - 0.5).abs() < 1e-12);
        assert!((c[1].abs() - 0.75f64.sqrt()).abs() < 1e-10);
        assert!(c[2].abs() < 1e-10);
    }
    #[test]
    fn linear_objective_and_infeasible_norm() {
        let k = M::identity(2, 2);
        let g = k.clone();
        let a = M::zeros(0, 2);
        let b = V::zeros(0);
        let c = minimum(&k, &V::from_vec(vec![3., 4.]), &a, &b, &g, 4.).unwrap();
        assert!((c - V::from_vec(vec![-1.2, -1.6])).norm() < 1e-9);
        assert!(minimum(
            &k,
            &V::zeros(2),
            &M::identity(2, 2),
            &V::from_vec(vec![1., 1.]),
            &g,
            1.
        )
        .is_err());
    }
}
