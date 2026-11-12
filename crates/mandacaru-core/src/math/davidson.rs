//! Matrix-free Davidson solvers for the lowest roots of large symmetric eigenproblems.
//!
//! [`lowest`] finds the lowest roots of a real symmetric operator `A`, given only its action on a
//! block of vectors and its diagonal. [`lowest_paired`] finds the lowest positive roots `omega`
//! of the product form `(A - B)(A + B) z = omega^2 z` with `A + B` symmetric and `A - B`
//! symmetric positive definite, given both actions, without square roots of either. The
//! subspace is shared by `z = X + Y` and `X - Y` (the paired driver of Stratmann, Scuseria and
//! Frisch, J. Chem. Phys. 109, 8218 (1998)), and the projected problem is the symmetric
//! `L^T P L u = omega^2 u` with `M = L L^T`.
//!
//! Nothing here knows what the operator is. Every step is serial and in a fixed order, so the
//! result depends only on the operator's values. A subspace that reaches the whole space gives
//! the exact roots; a request for as many roots as the dimension starts there.
//!
//! [`linalg::davidson`](super::linalg::davidson) is a different routine: the dense refinement of
//! a full eigenbasis that the periodic eigensolver uses.
use super::{invalid, linalg::real_eigen};
use crate::Result;
use nalgebra::{DMatrix, DVector};

/// What [`lowest`] and [`lowest_paired`] are asked for.
#[derive(Clone, Debug)]
pub struct Options {
    /// The number of lowest roots.
    pub roots: usize,
    /// The largest residual norm a root may keep.
    pub tolerance: f64,
    /// The largest number of expansion steps.
    pub max_iterations: usize,
    /// The largest subspace before a restart from the current roots (at least `2 roots + 2`).
    pub max_subspace: usize,
}
impl Options {
    pub fn new(roots: usize) -> Self {
        Self {
            roots,
            tolerance: 1e-5,
            max_iterations: 200,
            max_subspace: (10 * roots).max(48),
        }
    }
}

/// A symmetric operator given by its action.
pub trait Symmetric {
    fn dimension(&self) -> usize;
    /// The diagonal (the preconditioner).
    fn diagonal(&self) -> Vec<f64>;
    /// `A V` for the columns of `block`.
    fn apply(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>>;
}

/// The two symmetric operators of a paired problem.
pub trait Paired {
    fn dimension(&self) -> usize;
    /// An approximation to the diagonals of both `A + B` and `A - B` (the preconditioner).
    fn diagonal(&self) -> Vec<f64>;
    /// `(A + B) V`.
    fn sum(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>>;
    /// `(A - B) V`.
    fn difference(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>>;
}

/// The roots of [`lowest`].
#[derive(Clone, Debug)]
pub struct Roots {
    pub values: Vec<f64>,
    /// Orthonormal columns, one per root; each sign makes its largest component positive.
    pub vectors: DMatrix<f64>,
    pub residuals: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    /// Operator applications (vectors), for the record.
    pub applications: usize,
}

/// The roots of [`lowest_paired`]: `omega` and the columns `X + Y`, `X - Y` normalized to
/// `(X + Y)^T (X - Y) = 1`.
#[derive(Clone, Debug)]
pub struct PairedRoots {
    pub values: Vec<f64>,
    pub plus: DMatrix<f64>,
    pub minus: DMatrix<f64>,
    pub residuals: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub applications: usize,
}

fn check(options: &Options, n: usize) -> Result<()> {
    if n == 0 || options.roots == 0 || options.roots > n {
        return Err(invalid(format!(
            "Davidson needs between 1 and {n} roots of a nonempty operator, not {}",
            options.roots
        )));
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0. || options.max_iterations == 0 {
        return Err(invalid(
            "Davidson needs a positive finite tolerance and at least one iteration",
        ));
    }
    Ok(())
}

/// The first `count` indices of `diagonal` in ascending order (ties by index): the unit
/// vectors the search starts from.
fn start(diagonal: &[f64], count: usize) -> DMatrix<f64> {
    let mut order: Vec<usize> = (0..diagonal.len()).collect();
    order.sort_by(|&a, &b| diagonal[a].total_cmp(&diagonal[b]).then(a.cmp(&b)));
    let mut v = DMatrix::zeros(diagonal.len(), count);
    for (column, &row) in order.iter().take(count).enumerate() {
        v[(row, column)] = 1.;
    }
    v
}

/// Appends the columns of `candidates` to the orthonormal `basis`, each orthogonalized twice
/// against what is there and dropped when less than `1e-8` of it remains.
fn extend(basis: &DMatrix<f64>, candidates: &[DVector<f64>]) -> DMatrix<f64> {
    let mut columns: Vec<DVector<f64>> = basis.column_iter().map(|c| c.into_owned()).collect();
    let old = columns.len();
    for candidate in candidates {
        let norm = candidate.norm();
        if !(norm > 0.) || !norm.is_finite() {
            continue;
        }
        let mut t = candidate / norm;
        for _ in 0..2 {
            for c in &columns {
                let overlap = c.dot(&t);
                t -= c * overlap;
            }
        }
        let remaining = t.norm();
        if remaining > 1e-8 {
            columns.push(t / remaining);
        }
    }
    if columns.len() == old {
        return basis.clone();
    }
    DMatrix::from_columns(&columns)
}

/// Orthonormal columns spanning the columns of `coefficients` (in subspace coordinates).
fn orthonormal(coefficients: &DMatrix<f64>) -> DMatrix<f64> {
    let candidates: Vec<DVector<f64>> =
        coefficients.column_iter().map(|c| c.into_owned()).collect();
    extend(&DMatrix::zeros(coefficients.nrows(), 0), &candidates)
}

fn precondition(residual: &DVector<f64>, diagonal: &[f64], shift: f64) -> DVector<f64> {
    DVector::from_fn(residual.len(), |i, _| {
        let denominator = diagonal[i] - shift;
        let safe = if denominator.abs() < 1e-8 {
            1e-8f64.copysign(denominator)
        } else {
            denominator
        };
        residual[i] / safe
    })
}

/// A deterministic sign: the largest component (lowest index on ties) is positive.
fn fix_sign(column: &mut DVector<f64>) -> f64 {
    let mut pivot = 0;
    for i in 1..column.len() {
        if column[i].abs() > column[pivot].abs() {
            pivot = i;
        }
    }
    let sign = if column[pivot] < 0. { -1. } else { 1. };
    *column *= sign;
    sign
}

fn symmetric_part(m: DMatrix<f64>) -> DMatrix<f64> {
    (&m + m.transpose()) * 0.5
}

fn initial_count(options: &Options, n: usize) -> usize {
    (2 * options.roots).max(options.roots + 4).min(n)
}

/// The lowest `options.roots` eigenpairs of the symmetric operator `op`.
pub fn lowest(op: &impl Symmetric, options: &Options) -> Result<Roots> {
    let n = op.dimension();
    check(options, n)?;
    let k = options.roots;
    let diagonal = op.diagonal();
    if diagonal.len() != n || diagonal.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "the Davidson diagonal must be finite and match the operator",
        ));
    }
    let limit = options.max_subspace.max(2 * k + 2).min(n);
    let mut v = start(&diagonal, initial_count(options, n));
    let mut av = op.apply(&v)?;
    let mut applications = v.ncols();
    let mut iterations = 0;
    loop {
        iterations += 1;
        if av.shape() != v.shape() {
            return Err(invalid("the operator returned a block of the wrong shape"));
        }
        let projected = symmetric_part(v.transpose() * &av);
        let (theta, s) = real_eigen(projected)?;
        let sk = s.columns(0, k).into_owned();
        let x = &v * &sk;
        let ax = &av * &sk;
        let mut residuals = Vec::with_capacity(k);
        let mut corrections = Vec::new();
        for root in 0..k {
            let r = ax.column(root) - x.column(root) * theta[root];
            let norm = r.norm();
            residuals.push(norm);
            if norm > options.tolerance {
                corrections.push(precondition(&r.into_owned(), &diagonal, theta[root]));
            }
        }
        let whole = v.ncols() == n;
        if corrections.is_empty() || whole || iterations >= options.max_iterations {
            let converged = corrections.is_empty() || whole;
            let mut vectors = x;
            for mut column in vectors.column_iter_mut() {
                let mut owned = column.clone_owned();
                fix_sign(&mut owned);
                column.copy_from(&owned);
            }
            return Ok(Roots {
                values: theta[..k].to_vec(),
                vectors,
                residuals,
                iterations,
                converged,
                applications,
            });
        }
        if v.ncols() + corrections.len() > limit {
            // Restart from the current roots and the next ones the subspace holds.
            let keep = (k + corrections.len()).min(v.ncols());
            let c = orthonormal(&s.columns(0, keep).into_owned());
            v = &v * &c;
            av = &av * &c;
        }
        let grown = extend(&v, &corrections);
        if grown.ncols() == v.ncols() {
            return Err(invalid(
                "Davidson stalled: its corrections are inside the subspace before the residuals \
                 reached the tolerance",
            ));
        }
        let new = grown
            .columns(v.ncols(), grown.ncols() - v.ncols())
            .into_owned();
        let anew = op.apply(&new)?;
        applications += new.ncols();
        av = DMatrix::from_columns(
            &av.column_iter()
                .chain(anew.column_iter())
                .map(|c| c.into_owned())
                .collect::<Vec<_>>(),
        );
        v = grown;
    }
}

/// The lowest `options.roots` positive roots `omega` of `(A - B)(A + B) z = omega^2 z`.
///
/// Raises when the projected `A - B` is not positive definite or a projected `omega^2` is not
/// positive: the eigenproblem then has no real root there (an unstable reference, in the
/// language of linear response), and the caller decides what that means.
pub fn lowest_paired(op: &impl Paired, options: &Options) -> Result<PairedRoots> {
    let n = op.dimension();
    check(options, n)?;
    let k = options.roots;
    let diagonal = op.diagonal();
    if diagonal.len() != n || diagonal.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "the Davidson diagonal must be finite and match the operator",
        ));
    }
    let limit = options.max_subspace.max(4 * k + 4).min(n);
    let mut v = start(&diagonal, initial_count(options, n));
    let mut sum = op.sum(&v)?;
    let mut difference = op.difference(&v)?;
    let mut applications = v.ncols();
    let mut iterations = 0;
    loop {
        iterations += 1;
        if sum.shape() != v.shape() || difference.shape() != v.shape() {
            return Err(invalid("the operator returned a block of the wrong shape"));
        }
        let p = symmetric_part(v.transpose() * &sum);
        let m = symmetric_part(v.transpose() * &difference);
        let l = m.clone().cholesky().ok_or_else(|| {
            invalid(
                "the projected A - B is not positive definite: the paired eigenproblem has no \
                 real root (an unstable reference)",
            )
        })?;
        let l = l.l();
        let (squares, u) = real_eigen(symmetric_part(l.transpose() * &p * &l))?;
        if let Some((root, value)) = squares[..k].iter().enumerate().find(|(_, w)| **w <= 0.) {
            return Err(invalid(format!(
                "root {root} of the paired eigenproblem has omega^2 = {value:.3e} <= 0: no real \
                 root (an unstable reference)"
            )));
        }
        let omega: Vec<f64> = squares[..k].iter().map(|w| w.sqrt()).collect();
        let mut a = &l * u.columns(0, k);
        let mut b = &p * &a;
        for root in 0..k {
            let mut column = b.column_mut(root);
            column /= omega[root];
            let scale = a.column(root).dot(&b.column(root)).sqrt();
            a.column_mut(root).scale_mut(1. / scale);
            b.column_mut(root).scale_mut(1. / scale);
        }
        let plus = &v * &a;
        let minus = &v * &b;
        let sum_plus = &sum * &a;
        let difference_minus = &difference * &b;
        let mut residuals = Vec::with_capacity(k);
        let mut corrections = Vec::new();
        for root in 0..k {
            let r1 = sum_plus.column(root) - minus.column(root) * omega[root];
            let r2 = difference_minus.column(root) - plus.column(root) * omega[root];
            let norm = (r1.norm_squared() + r2.norm_squared()).sqrt();
            residuals.push(norm);
            if norm > options.tolerance {
                corrections.push(precondition(&r1.into_owned(), &diagonal, omega[root]));
                corrections.push(precondition(&r2.into_owned(), &diagonal, omega[root]));
            }
        }
        let whole = v.ncols() == n;
        if corrections.is_empty() || whole || iterations >= options.max_iterations {
            let converged = corrections.is_empty() || whole;
            let (mut plus, mut minus) = (plus, minus);
            for root in 0..k {
                let mut column = plus.column(root).clone_owned();
                let sign = fix_sign(&mut column);
                plus.column_mut(root).copy_from(&column);
                minus.column_mut(root).scale_mut(sign);
            }
            return Ok(PairedRoots {
                values: omega,
                plus,
                minus,
                residuals,
                iterations,
                converged,
                applications,
            });
        }
        if v.ncols() + corrections.len() > limit {
            let coefficients = DMatrix::from_columns(
                &a.column_iter()
                    .chain(b.column_iter())
                    .map(|c| c.into_owned())
                    .collect::<Vec<_>>(),
            );
            let c = orthonormal(&coefficients);
            v = &v * &c;
            sum = &sum * &c;
            difference = &difference * &c;
        }
        let grown = extend(&v, &corrections);
        if grown.ncols() == v.ncols() {
            return Err(invalid(
                "Davidson stalled: its corrections are inside the subspace before the residuals \
                 reached the tolerance",
            ));
        }
        let new = grown
            .columns(v.ncols(), grown.ncols() - v.ncols())
            .into_owned();
        let (snew, dnew) = (op.sum(&new)?, op.difference(&new)?);
        applications += new.ncols();
        let join = |old: &DMatrix<f64>, add: &DMatrix<f64>| {
            DMatrix::from_columns(
                &old.column_iter()
                    .chain(add.column_iter())
                    .map(|c| c.into_owned())
                    .collect::<Vec<_>>(),
            )
        };
        sum = join(&sum, &snew);
        difference = join(&difference, &dnew);
        v = grown;
    }
}
