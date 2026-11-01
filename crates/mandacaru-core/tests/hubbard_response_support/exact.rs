//! Independent numerical helpers: eigenpairs, inverse, bisection.
use nalgebra::DMatrix;

/// Cyclic Jacobi. Returns ascending eigenvalues and `vectors[k * n + i]`, the
/// component `i` of eigenvector `k`.
pub fn jacobi(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut v = vec![0.; n * n];
    for i in 0..n {
        v[i * n + i] = 1.;
    }
    for _ in 0..100 {
        let mut off = 0.;
        for p in 0..n {
            for q in p + 1..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off < 1e-32 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p * n + q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2. * a[p * n + q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.).sqrt());
                let c = 1. / (t * t + 1.).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (kp, kq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * kp - s * kq;
                    a[k * n + q] = s * kp + c * kq;
                }
                for k in 0..n {
                    let (pk, qk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * pk - s * qk;
                    a[q * n + k] = s * pk + c * qk;
                }
                for k in 0..n {
                    let (kp, kq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * kp - s * kq;
                    v[k * n + q] = s * kp + c * kq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| a[x * n + x].total_cmp(&a[y * n + y]));
    let values = order.iter().map(|&k| a[k * n + k]).collect();
    let mut vectors = vec![0.; n * n];
    for (rank, &k) in order.iter().enumerate() {
        for i in 0..n {
            vectors[rank * n + i] = v[i * n + k];
        }
    }
    (values, vectors)
}

/// Gauss-Jordan inverse with partial pivoting (not nalgebra's, not an eigen route).
pub fn invert(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let n = matrix.nrows();
    let mut a = matrix.clone();
    let mut b = DMatrix::<f64>::identity(n, n);
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&x, &y| a[(x, col)].abs().total_cmp(&a[(y, col)].abs()))
            .unwrap();
        a.swap_rows(col, pivot);
        b.swap_rows(col, pivot);
        let d = a[(col, col)];
        for j in 0..n {
            a[(col, j)] /= d;
            b[(col, j)] /= d;
        }
        for r in 0..n {
            if r != col {
                let f = a[(r, col)];
                for j in 0..n {
                    let (x, y) = (a[(col, j)], b[(col, j)]);
                    a[(r, j)] -= f * x;
                    b[(r, j)] -= f * y;
                }
            }
        }
    }
    b
}

/// Bisection of a function that changes sign on `[lo, hi]`.
pub fn bisect(mut lo: f64, mut hi: f64, f: impl Fn(f64) -> f64) -> f64 {
    let sign = f(lo) < 0.;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if (f(mid) < 0.) == sign {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Number of eigenvalues of the symmetric `h` (3 x 3) below `x`: negative pivots of the
/// LDL^T factorization of `h - x`.
fn count_below(h: &[f64; 9], x: f64) -> usize {
    let mut a = *h;
    for i in 0..3 {
        a[i * 3 + i] -= x;
    }
    let mut negatives = 0;
    for k in 0..3 {
        let mut pivot = a[k * 3 + k];
        if pivot.abs() < 1e-300 {
            pivot = -1e-300;
        }
        if pivot < 0. {
            negatives += 1;
        }
        for i in k + 1..3 {
            let f = a[i * 3 + k] / pivot;
            for j in k + 1..3 {
                a[i * 3 + j] -= f * a[k * 3 + j];
            }
        }
    }
    negatives
}

/// Lowest eigenpair of a symmetric 3 x 3 matrix by Sturm bisection and a cross product.
pub fn lowest_3x3(h: &[f64; 9]) -> (f64, [f64; 3]) {
    let radius = h.iter().fold(0f64, |m, v| m.max(v.abs())) * 3. + 1.;
    let (mut lo, mut hi) = (-radius, radius);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if count_below(h, mid) >= 1 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let e = 0.5 * (lo + hi);
    let row = |i: usize| {
        [
            h[i * 3] - if i == 0 { e } else { 0. },
            h[i * 3 + 1] - if i == 1 { e } else { 0. },
            h[i * 3 + 2] - if i == 2 { e } else { 0. },
        ]
    };
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let candidates = [
        cross(row(0), row(1)),
        cross(row(0), row(2)),
        cross(row(1), row(2)),
    ];
    let best = candidates
        .into_iter()
        .max_by(|a, b| norm(*a).total_cmp(&norm(*b)))
        .unwrap();
    let n = norm(best);
    (e, [best[0] / n, best[1] / n, best[2] / n])
}

/// Fermi-Dirac occupations with the chemical potential found by bisection so that
/// `2 sum f = electrons`. Returns `(f, df/de)`.
pub fn fermi_dirac(levels: &[f64], electrons: f64, kt: f64) -> (Vec<f64>, Vec<f64>) {
    let f = |e: f64, mu: f64| 1. / (1. + ((e - mu) / kt).exp());
    let count = |mu: f64| 2. * levels.iter().map(|&e| f(e, mu)).sum::<f64>() - electrons;
    let lo = levels[0] - 50. * kt - 1.;
    let hi = levels[levels.len() - 1] + 50. * kt + 1.;
    let mu = bisect(lo, hi, count);
    let occupations: Vec<f64> = levels.iter().map(|&e| f(e, mu)).collect();
    let slopes = occupations.iter().map(|&o| -o * (1. - o) / kt).collect();
    (occupations, slopes)
}
