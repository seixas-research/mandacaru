//! Generalized tridiagonal eigenproblems with separable corrections.
//!
//! H = T + U D U^t, S = I + U Q U^t. Only the tridiagonal and
//! projector-sized matrices are stored; no radial-size dense pencil is built.
use super::{invalid, linalg::real_eigen};
use crate::{finite, product, Buffer, Context, Error, Result};
use nalgebra::{DMatrix, DVector};

pub struct Pencil<'a> {
    pub diagonal: &'a [f64],
    pub off_diagonal: &'a [f64],
    /// Row-major (radial nodes, projectors).
    pub projectors: &'a [f64],
    /// Row-major symmetric (projectors, projectors).
    pub coupling: &'a [f64],
    pub metric: &'a [f64],
    pub rank: usize,
}

pub struct Eigenpair {
    pub energy: f64,
    /// Normalized in the generalized metric, when requested.
    pub vector: Option<Buffer<f64>>,
}

// LDL inertia of T - shift. Signed tiny pivots implement the limiting
// convention at a pole without dividing by zero.
fn sturm(diagonal: &[f64], off: &[f64], shift: f64) -> usize {
    let tiny = f64::MIN_POSITIVE.sqrt();
    let mut pivot = diagonal[0] - shift;
    let mut count = 0;
    for i in 0..diagonal.len() {
        if i > 0 {
            pivot = diagonal[i] - shift - off[i - 1] * (off[i - 1] / pivot);
        }
        if pivot.abs() < tiny {
            pivot = -tiny;
        }
        count += usize::from(pivot < 0.);
    }
    count
}

/// Adjacent-row pivoting for a tridiagonal system, with multiple RHS.
/// Pivoting matters here: a bisection shift may coincide with a leading
/// principal minor even when the full matrix is nonsingular.
fn solve_tridiagonal(
    diagonal: &[f64],
    off: &[f64],
    shift: f64,
    mut rhs: DMatrix<f64>,
) -> Result<DMatrix<f64>> {
    let n = diagonal.len();
    let mut d: Vec<_> = diagonal.iter().map(|v| v - shift).collect();
    let mut upper = off.to_vec();
    let mut second = vec![0.; n.saturating_sub(2)];
    for i in 0..n - 1 {
        if d[i].abs() >= off[i].abs() {
            if d[i] == 0. {
                return Err(Error::Kernel("singular tridiagonal shift".into()));
            }
            let factor = off[i] / d[i];
            d[i + 1] -= factor * upper[i];
            for j in 0..rhs.ncols() {
                rhs[(i + 1, j)] -= factor * rhs[(i, j)];
            }
        } else {
            let factor = d[i] / off[i];
            d[i] = off[i];
            let next = d[i + 1];
            d[i + 1] = upper[i] - factor * next;
            upper[i] = next;
            if i + 2 < n {
                second[i] = upper[i + 1];
                upper[i + 1] *= -factor;
            }
            for j in 0..rhs.ncols() {
                let previous = rhs[(i, j)];
                rhs[(i, j)] = rhs[(i + 1, j)];
                rhs[(i + 1, j)] = previous - factor * rhs[(i + 1, j)];
            }
        }
    }
    if d[n - 1] == 0. {
        return Err(Error::Kernel("singular tridiagonal shift".into()));
    }
    for j in 0..rhs.ncols() {
        rhs[(n - 1, j)] /= d[n - 1];
        if n > 1 {
            rhs[(n - 2, j)] = (rhs[(n - 2, j)] - upper[n - 2] * rhs[(n - 1, j)]) / d[n - 2];
        }
        for i in (0..n.saturating_sub(2)).rev() {
            rhs[(i, j)] =
                (rhs[(i, j)] - upper[i] * rhs[(i + 1, j)] - second[i] * rhs[(i + 2, j)]) / d[i];
        }
    }
    if rhs.iter().any(|v| !v.is_finite()) {
        return Err(Error::Kernel("nonfinite tridiagonal solution".into()));
    }
    Ok(rhs)
}

struct Correction {
    weights: Vec<f64>,
    columns: DMatrix<f64>,
}
fn correction(
    u: &DMatrix<f64>,
    d: &DMatrix<f64>,
    q: &DMatrix<f64>,
    shift: f64,
) -> Result<Correction> {
    let (beta, vectors) = real_eigen(d - q * shift)?;
    let scale = beta.iter().map(|v| v.abs()).fold(0., f64::max);
    let keep: Vec<_> = (0..beta.len())
        .filter(|&i| beta[i].abs() > 1e-14 * scale)
        .collect();
    let rotated = DMatrix::from_fn(beta.len(), keep.len(), |i, j| vectors[(i, keep[j])]);
    Ok(Correction {
        weights: keep.iter().map(|&i| beta[i]).collect(),
        columns: u * rotated,
    })
}

// Inertia of the bordered matrix [[T-shift,W],[W^t,-beta^-1]].
// Eliminate radial rows in 1x1/2x2 blocks. Unlike subtracting two
// near-singular resolvents, this is defined even at an exact pole of T.
fn below(
    p: &Pencil<'_>,
    u: &DMatrix<f64>,
    d: &DMatrix<f64>,
    q: &DMatrix<f64>,
    shift: f64,
) -> Result<usize> {
    let c = correction(u, d, q, shift)?;
    if c.weights.is_empty() {
        return Ok(sturm(p.diagonal, p.off_diagonal, shift));
    }
    let n = p.diagonal.len();
    let mut diagonal: Vec<_> = p.diagonal.iter().map(|v| v - shift).collect();
    let mut w = c.columns;
    let mut border = DMatrix::from_diagonal(&DVector::from_iterator(
        c.weights.len(),
        c.weights.iter().map(|b| -1. / b),
    ));
    let removed = c.weights.iter().filter(|&&v| v > 0.).count();
    let mut negative = 0;
    let mut i = 0;
    while i < n {
        let rank = border.nrows();
        let a = diagonal[i];
        let e = p.off_diagonal.get(i).copied().unwrap_or(0.);
        if a != 0. && (a.abs() >= 0.64 * e.abs() || e == 0.) {
            negative += usize::from(a < 0.);
            if i + 1 < n {
                diagonal[i + 1] -= e * (e / a);
            }
            for j in 0..rank {
                for k in 0..rank {
                    border[(j, k)] -= w[(i, j)] * (w[(i, k)] / a);
                }
            }
            if i + 1 < n {
                for j in 0..rank {
                    w[(i + 1, j)] -= (e / a) * w[(i, j)];
                }
            }
            i += 1;
        } else if e != 0. {
            let b = diagonal[i + 1];
            let determinant = a * b - e * e;
            // A singular 2x2 block with a!=0 can still be eliminated one
            // row at a time; the next iteration handles its zero Schur pivot.
            if determinant == 0. && a != 0. {
                negative += usize::from(a < 0.);
                diagonal[i + 1] = 0.;
                for j in 0..rank {
                    for k in 0..rank {
                        border[(j, k)] -= w[(i, j)] * (w[(i, k)] / a);
                    }
                }
                for j in 0..rank {
                    w[(i + 1, j)] -= (e / a) * w[(i, j)];
                }
                i += 1;
                continue;
            }
            negative += if determinant < 0. {
                1
            } else if a + b < 0. {
                2
            } else {
                0
            };
            let aa = b / determinant;
            let ab = -e / determinant;
            let bb = a / determinant;
            for j in 0..rank {
                for k in 0..rank {
                    border[(j, k)] -= w[(i, j)] * (aa * w[(i, k)] + ab * w[(i + 1, k)])
                        + w[(i + 1, j)] * (ab * w[(i, k)] + bb * w[(i + 1, k)]);
                }
            }
            if i + 2 < n {
                let next = p.off_diagonal[i + 1];
                diagonal[i + 2] -= next * next * bb;
                for j in 0..rank {
                    w[(i + 2, j)] -= next * (ab * w[(i, j)] + bb * w[(i + 1, j)]);
                }
            }
            i += 2;
        } else {
            // An isolated zero diagonal can couple to the border. Pivot
            // against its largest border entry: the inverse's border-border
            // entry is zero, so the remaining tridiagonal structure survives.
            let pivot = (0..rank).max_by(|&j, &k| w[(i, j)].abs().total_cmp(&w[(i, k)].abs()));
            if let Some(pivot) = pivot.filter(|&j| w[(i, j)] != 0.) {
                let value = w[(i, pivot)];
                let keep: Vec<_> = (0..rank).filter(|&j| j != pivot).collect();
                let next_border = DMatrix::from_fn(rank - 1, rank - 1, |j, k| {
                    let (j, k) = (keep[j], keep[k]);
                    border[(j, k)]
                        + border[(pivot, pivot)] * (w[(i, j)] / value) * (w[(i, k)] / value)
                        - (w[(i, j)] * border[(pivot, k)] + border[(j, pivot)] * w[(i, k)]) / value
                });
                let next_w = DMatrix::from_fn(n, rank - 1, |j, k| {
                    w[(j, keep[k])] - w[(j, pivot)] * (w[(i, keep[k])] / value)
                });
                border = next_border;
                w = next_w;
                negative += 1;
            }
            i += 1;
        }
    }
    if i == n && border.nrows() != 0 {
        let (values, _) = real_eigen((&border + border.transpose()) * 0.5)?;
        negative += values.iter().filter(|&&v| v < 0.).count();
    }
    if negative < removed || negative - removed > n {
        return Err(Error::Kernel("unstable separable-pencil inertia".into()));
    }
    Ok(negative - removed)
}

fn apply_metric(u: &DMatrix<f64>, q: &DMatrix<f64>, v: &DVector<f64>) -> DVector<f64> {
    v + u * (q * (u.transpose() * v))
}
fn apply_h(p: &Pencil<'_>, u: &DMatrix<f64>, d: &DMatrix<f64>, v: &DVector<f64>) -> DVector<f64> {
    let mut out = u * (d * (u.transpose() * v));
    for i in 0..v.len() {
        out[i] += p.diagonal[i] * v[i];
        if i > 0 {
            out[i] += p.off_diagonal[i - 1] * v[i - 1];
        }
        if i + 1 < v.len() {
            out[i] += p.off_diagonal[i] * v[i + 1];
        }
    }
    out
}

pub fn eigenpair(ctx: &Context, p: Pencil<'_>, state: usize, vectors: bool) -> Result<Eigenpair> {
    let n = p.diagonal.len();
    let r = p.rank;
    if n == 0
        || state >= n
        || p.off_diagonal.len() != n - 1
        || p.projectors.len() != product(&[n, r])?
        || p.coupling.len() != product(&[r, r])?
        || p.metric.len() != product(&[r, r])?
    {
        return Err(invalid(
            "invalid separable-pencil dimensions or state index",
        ));
    }
    for values in [
        p.diagonal,
        p.off_diagonal,
        p.projectors,
        p.coupling,
        p.metric,
    ] {
        finite(values, "separable pencil")?;
    }
    for matrix in [p.coupling, p.metric] {
        for a in 0..r {
            for b in 0..a {
                if (matrix[a * r + b] - matrix[b * r + a]).abs()
                    > 1e-12 * (1. + matrix[a * r + b].abs().max(matrix[b * r + a].abs()))
                {
                    return Err(invalid("separable coupling and metric must be symmetric"));
                }
            }
        }
    }
    // Covers all nalgebra temporaries, inverse iteration, and tridiagonal LU.
    // This bound is linear in radial nodes at fixed projector rank.
    let _workspace = ctx.reserve(
        product(&[n, r.checked_add(8).ok_or(Error::Overflow)?, 8, 12])?
            .checked_add(product(&[r, r, 8, 32])?)
            .ok_or(Error::Overflow)?,
    )?;
    if r == 0 {
        let (energy, vector) = crate::kernels::eigenpair(ctx, p.diagonal, p.off_diagonal, state)?;
        return Ok(Eigenpair {
            energy,
            vector: vectors.then_some(vector),
        });
    }
    let u = DMatrix::from_row_slice(n, r, p.projectors);
    let d = DMatrix::from_row_slice(r, r, p.coupling);
    let q = DMatrix::from_row_slice(r, r, p.metric);
    let (gram, rotation) = real_eigen(u.transpose() * &u)?;
    let sqrt_gram = &rotation
        * DMatrix::from_diagonal(&DVector::from_iterator(
            r,
            gram.iter().map(|v| v.max(0.).sqrt()),
        ))
        * rotation.transpose();
    let (metric_values, _) = real_eigen(DMatrix::identity(r, r) + &sqrt_gram * &q * &sqrt_gram)?;
    if metric_values[0] <= 1e-12 {
        return Err(invalid("separable overlap must be positive definite"));
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for i in 0..n {
        let radius = if i > 0 {
            p.off_diagonal[i - 1].abs()
        } else {
            0.
        } + if i + 1 < n {
            p.off_diagonal[i].abs()
        } else {
            0.
        };
        lo = lo.min(p.diagonal[i] - radius);
        hi = hi.max(p.diagonal[i] + radius);
    }
    let padding = 1. + lo.abs().max(hi.abs());
    lo -= padding;
    hi += padding;
    for _ in 0..128 {
        if below(&p, &u, &d, &q, lo)? <= state {
            break;
        }
        lo -= 2. * (1. + lo.abs());
        if !lo.is_finite() {
            return Err(invalid("unbounded separable eigenvalue bracket"));
        }
    }
    for _ in 0..128 {
        if below(&p, &u, &d, &q, hi)? > state {
            break;
        }
        hi += 2. * (1. + hi.abs());
        if !hi.is_finite() {
            return Err(invalid("unbounded separable eigenvalue bracket"));
        }
    }
    for _ in 0..160 {
        let mid = lo + 0.5 * (hi - lo);
        if mid == lo || mid == hi || hi - lo < 2e-13 * (1. + mid.abs()) {
            break;
        }
        if below(&p, &u, &d, &q, mid)? > state {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let energy = lo + 0.5 * (hi - lo);
    if !vectors {
        return Ok(Eigenpair {
            energy,
            vector: None,
        });
    }
    let shift = energy - 1e-7 * (1. + energy.abs());
    let c = correction(&u, &d, &q, shift)?;
    let inverse_u = solve_tridiagonal(p.diagonal, p.off_diagonal, shift, c.columns.clone())?;
    let inner = DMatrix::from_diagonal(&DVector::from_iterator(
        c.weights.len(),
        c.weights.iter().map(|b| 1. / b),
    )) + c.columns.transpose() * &inverse_u;
    let lu = inner.lu();
    let mut v = DVector::from_fn(n, |i, _| ((i + 1) as f64 * 1.618_033_988_749_895).sin());
    for _ in 0..24 {
        let rhs = apply_metric(&u, &q, &v);
        let solved = solve_tridiagonal(
            p.diagonal,
            p.off_diagonal,
            shift,
            DMatrix::from_column_slice(n, 1, rhs.as_slice()),
        )?;
        let mut next = solved.column(0).into_owned();
        if !c.weights.is_empty() {
            let projected = c.columns.transpose() * &next;
            let small = lu
                .solve(&projected)
                .ok_or_else(|| Error::Kernel("singular separable inverse iteration".into()))?;
            next -= &inverse_u * small;
        }
        let norm = next.dot(&apply_metric(&u, &q, &next));
        if !norm.is_finite() || norm <= 0. {
            return Err(Error::Kernel("invalid generalized eigenvector norm".into()));
        }
        v = next / norm.sqrt();
        let h_v = apply_h(&p, &u, &d, &v);
        let s_v = apply_metric(&u, &q, &v);
        let residual = (&h_v - &s_v * energy).norm();
        if residual <= 2e-9 * (1. + energy.abs()) * s_v.norm() {
            let pivot = (0..n)
                .max_by(|&a, &b| v[a].abs().total_cmp(&v[b].abs()))
                .unwrap();
            if v[pivot] < 0. {
                v *= -1.;
            }
            let mut vector = ctx.allocate(n)?;
            vector.as_mut_slice().copy_from_slice(v.as_slice());
            return Ok(Eigenpair {
                energy,
                vector: Some(vector),
            });
        }
    }
    Err(Error::Kernel(
        "separable eigenvector did not converge".into(),
    ))
}
