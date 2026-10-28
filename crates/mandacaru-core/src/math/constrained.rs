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
    // Orthogonalize the constraint rows and their right-hand sides together.
    let mut rows: Vec<V<f64>> = vec![];
    let mut targets: Vec<f64> = vec![];
    for i in 0..a.nrows() {
        let mut row = a.row(i).transpose().into_owned();
        let scale = row.norm();
        if scale == 0. {
            if b[i].abs() > 1e-12 {
                return Err(invalid("inconsistent zero constraint"));
            } else {
                continue;
            }
        }
        row /= scale;
        let mut target = b[i] / scale;
        for _ in 0..2 {
            for (prior, rhs) in rows.iter().zip(&targets) {
                let projection = prior.dot(&row);
                row -= prior * projection;
                target -= rhs * projection;
            }
        }
        let length = row.norm();
        if length < 1e-11 {
            if target.abs() > 1e-9 {
                return Err(invalid("inconsistent linear constraints"));
            }
        } else {
            rows.push(row / length);
            targets.push(target / length);
        }
    }
    let mut origin = V::zeros(n);
    for (row, target) in rows.iter().zip(&targets) {
        origin += row * *target;
    }
    let mut null = vec![];
    for axis in 0..n {
        let mut direction = V::zeros(n);
        direction[axis] = 1.;
        for _ in 0..2 {
            for row in rows.iter().chain(null.iter()) {
                direction -= row * row.dot(&direction);
            }
        }
        let length = direction.norm();
        if length > 1e-9 {
            null.push(direction / length);
        }
    }
    if null.is_empty() {
        if (origin.dot(&(g * &origin)) - norm).abs() > 1e-9 * norm.max(1.) {
            return Err(invalid(
                "norm is incompatible with fixed linear constraints",
            ));
        }
        return Ok(origin);
    }
    let z = M::from_columns(&null);
    let gz = z.transpose() * g * &z;
    let (gv, gu) = super::linalg::real_eigen(gz)?;
    if gv[0] <= 0. {
        return Err(invalid(
            "norm metric is not positive definite on the constraint null space",
        ));
    }
    let whitening = &gu
        * M::from_diagonal(&V::from_iterator(
            gv.len(),
            gv.iter().map(|v| 1. / v.sqrt()),
        ));
    let kz = z.transpose() * k * &z;
    let (values, vectors) = super::linalg::real_eigen(whitening.transpose() * kz * &whitening)?;
    let transform = &z * whitening * vectors;
    let gp = transform.transpose() * g * &origin;
    let kp = transform.transpose() * (k * &origin + linear);
    let solution = |lambda: f64| -> V<f64> {
        &origin
            + &transform
                * V::from_iterator(
                    values.len(),
                    values
                        .iter()
                        .enumerate()
                        .map(|(i, mu)| (lambda * gp[i] - kp[i]) / (mu - lambda)),
                )
    };
    let residual = |lambda| {
        let c = solution(lambda);
        c.dot(&(g * &c)) - norm
    };
    let mu = values[0];
    let mut hi = mu - 1e-9 * mu.abs().max(1.);
    let mut lo = hi - 1.;
    while residual(lo) > 0. {
        lo = hi - 2. * (hi - lo);
        if hi - lo > 1e14 {
            return Err(invalid(
                "linear constraints require more norm than requested",
            ));
        }
    }
    for _ in 0..60 {
        if residual(hi) >= 0. {
            break;
        }
        let nearer = mu - 0.5 * (mu - hi);
        if nearer <= hi || nearer >= mu {
            break;
        }
        hi = nearer;
    }
    let out = if residual(hi) < 0. {
        // At the smallest eigenvalue, complete the norm along its eigenspace.
        let mut coefficients = V::zeros(values.len());
        let tol = 1e-10 * mu.abs().max(1.);
        for i in 0..values.len() {
            if values[i] - mu > tol {
                coefficients[i] = (mu * gp[i] - kp[i]) / (values[i] - mu);
            }
        }
        let base = &origin + &transform * coefficients;
        let direction = transform.column(0).into_owned();
        let aa = direction.dot(&(g * &direction));
        let bb = 2. * direction.dot(&(g * &base));
        let cc = base.dot(&(g * &base)) - norm;
        let discriminant = bb * bb - 4. * aa * cc;
        if discriminant < -1e-9 {
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
        for _ in 0..500 {
            let middle = 0.5 * (lo + hi);
            if middle == lo || middle == hi {
                break;
            }
            if residual(middle) > 0. {
                hi = middle;
            } else {
                lo = middle;
            }
        }
        solution(0.5 * (lo + hi))
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
