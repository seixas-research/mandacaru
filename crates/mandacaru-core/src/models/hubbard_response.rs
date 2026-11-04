//! Linear-response kernel of the Hubbard correction: the numbers that turn a pair of
//! shifted calculations into a response matrix and an interaction matrix.
//!
//! The model sees an engine only through [`Probe`]: given a shift `alpha` on the
//! correlated shells (and spin channels), it returns the shell occupations. The
//! occupation index is `shell * channels + channel`, in the probe's energy unit; the
//! interaction comes back in the same unit.
//!
//! ```text
//! chi0_IJ = dN_I / dalpha_J   Probe::frozen  (potential held at the ground state)
//! chi_IJ  = dN_I / dalpha_J   Probe::scf     (potential re-converged)
//! U       = chi0^-1 - chi^-1  the full matrix over every (shell, channel) pair
//! ```
//!
//! `U` is the shell-projected screened kernel, a definition: it equals a Hubbard-model
//! interaction only when the feedback depends on a shell through its occupation alone
//! (exact in the toy models of `tests/hubbard_response.rs`). Whether a probe holds the
//! Hubbard operator itself frozen in both passes (the engine definition of the plan,
//! literature comparison TO-VERIFY) or lets it respond is the probe's choice, not the
//! kernel's. A probe must also keep the electron number fixed in both passes, which means
//! re-finding the chemical potential: a fixed-potential probe passes every finite
//! difference here and breaks the sum rule.
//!
//! Each column is measured at symmetric shifts `+-alpha`, so the constant and every even
//! term cancel in the odd part `(N(+a) - N(-a)) / 2`. The quotient `D(a)` of that part by
//! `a` carries only `a^2, a^4, ...`: [`Fit::Central`] takes the smallest alpha (error
//! `O(a^2)`), [`Fit::Richardson`] eliminates the `a^2` term from the two smallest (error
//! `O(a^4)`). Several alphas give a linearity diagnostic; one alpha is reported as having
//! none.
//!
//! The matrices are inverted whole. Scalar, single-site and sub-block shortcuts are not
//! offered: they are wrong when the shells couple (`tests/hubbard_response.rs` measures by
//! how much).
mod kernel;
mod states;

use super::invalid;
use crate::math::linalg::real_eigen;
use crate::{product, Context, Result};
use nalgebra::DMatrix;
use std::mem::size_of;

pub use kernel::{Group, Kernel};
pub use states::{sum_over_states, StateBlock, StateSum, DEGENERATE, NO_SLOPE};

/// Largest condition number of `chi0` or `chi` that is inverted.
pub const CONDITION_LIMIT: f64 = 1e8;
/// Largest drift of the difference quotient across the alphas, relative to the smallest.
pub const NONLINEARITY_LIMIT: f64 = 1e-2;
/// Largest `max|A - A^T| / max|A|` of `chi0` or `chi`.
pub const ASYMMETRY_LIMIT: f64 = 1e-3;

/// Occupations of the correlated shells under a shift.
pub trait Probe {
    fn shells(&self) -> usize;
    /// Spin channels per shell: 1 (spin-summed or spinor trace) or 2 (collinear).
    fn channels(&self) -> usize;
    /// Occupations with the potential frozen at the ground state, electron number fixed.
    fn frozen(&mut self, shifts: &[f64]) -> Result<Vec<f64>>;
    /// Occupations after the self-consistent cycle is restarted with the shift, electron
    /// number fixed. A cycle that does not converge is an error, never a number.
    fn scf(&mut self, shifts: &[f64]) -> Result<Vec<f64>>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    Central,
    Richardson,
}

#[derive(Clone, Debug)]
pub struct Options {
    /// Positive shifts, strictly increasing.
    pub alphas: Vec<f64>,
    pub fit: Fit,
    pub condition_limit: f64,
    pub nonlinearity_limit: f64,
    pub asymmetry_limit: f64,
    /// When the correlated shells hold every electron, each column of `chi0` and `chi`
    /// sums to zero. `Some(limit)` refuses a larger column-sum residual. Leave `None`
    /// when other sites hold electrons: the residual is then only reported.
    pub sum_rule_limit: Option<f64>,
}

impl Options {
    /// The fit is a required choice; the limits take the plan's defaults.
    pub fn new(alphas: &[f64], fit: Fit) -> Result<Self> {
        let mut alphas = alphas.to_vec();
        alphas.sort_by(f64::total_cmp);
        let options = Self {
            alphas,
            fit,
            condition_limit: CONDITION_LIMIT,
            nonlinearity_limit: NONLINEARITY_LIMIT,
            asymmetry_limit: ASYMMETRY_LIMIT,
            sum_rule_limit: None,
        };
        options.validate()?;
        Ok(options)
    }
    fn validate(&self) -> Result<()> {
        let a = &self.alphas;
        if a.is_empty() || a.iter().any(|x| !x.is_finite() || *x <= 0.) {
            return Err(invalid("response shifts alpha must be positive and finite"));
        }
        if a.windows(2).any(|w| w[1] <= w[0]) {
            return Err(invalid(
                "response shifts alpha must be distinct and increasing",
            ));
        }
        if self.fit == Fit::Richardson && a.len() < 2 {
            return Err(invalid("a Richardson fit needs at least two shifts alpha"));
        }
        let limits = [
            self.condition_limit,
            self.nonlinearity_limit,
            self.asymmetry_limit,
        ];
        if limits.iter().any(|l| l.is_nan() || *l <= 0.)
            || self.sum_rule_limit.is_some_and(|l| l.is_nan() || l <= 0.)
        {
            return Err(invalid("response limits must be positive"));
        }
        Ok(())
    }
}

/// The measured matrices, over the `shells * channels` pairs in probe order.
#[derive(Clone, Debug)]
pub struct Response {
    pub shells: usize,
    pub channels: usize,
    /// Symmetrized frozen response.
    pub chi0: DMatrix<f64>,
    /// Symmetrized self-consistent response.
    pub chi: DMatrix<f64>,
    /// `chi0^-1 - chi^-1`.
    pub u: DMatrix<f64>,
    /// Relative drift of the difference quotient across the alphas (larger of the two
    /// passes); `None` for a single alpha, which has no linearity control.
    pub nonlinearity: Option<f64>,
    /// `max|A - A^T| / max|A|` before symmetrizing (larger of the two).
    pub asymmetry: f64,
    /// Largest column-sum residual relative to `max|A|` (larger of the two).
    pub sum_rule: f64,
    pub condition_chi0: f64,
    pub condition_chi: f64,
    /// Number of probe calls.
    pub evaluations: usize,
    limit: f64,
}

#[derive(Clone, Copy)]
enum Pass {
    Frozen,
    Scf,
}

fn peak(m: &DMatrix<f64>) -> f64 {
    m.iter().fold(0., |a, v| f64::max(a, v.abs()))
}

fn relative(numerator: f64, scale: f64) -> f64 {
    if scale > 0. {
        numerator / scale
    } else if numerator > 0. {
        f64::INFINITY
    } else {
        0.
    }
}

/// `D(a)` for every alpha: column `J` is `(N(+a e_J) - N(-a e_J)) / 2a`.
fn quotients(
    probe: &mut dyn Probe,
    pass: Pass,
    alphas: &[f64],
    n: usize,
    calls: &mut usize,
) -> Result<Vec<DMatrix<f64>>> {
    let mut out = Vec::with_capacity(alphas.len());
    for &alpha in alphas {
        let mut d = DMatrix::zeros(n, n);
        for j in 0..n {
            let mut shifts = vec![0.; n];
            let mut columns = Vec::with_capacity(2);
            for sign in [1., -1.] {
                shifts[j] = sign * alpha;
                let occupations = match pass {
                    Pass::Frozen => probe.frozen(&shifts)?,
                    Pass::Scf => probe.scf(&shifts)?,
                };
                *calls += 1;
                if occupations.len() != n {
                    return Err(invalid(format!(
                        "the probe returned {} occupations for {n} shell-channel pairs",
                        occupations.len()
                    )));
                }
                if occupations.iter().any(|x| !x.is_finite()) {
                    return Err(invalid("the probe returned a non-finite occupation"));
                }
                columns.push(occupations);
            }
            for i in 0..n {
                d[(i, j)] = (columns[0][i] - columns[1][i]) / (2. * alpha);
            }
        }
        out.push(d);
    }
    Ok(out)
}

fn estimate(fit: Fit, alphas: &[f64], q: &[DMatrix<f64>]) -> DMatrix<f64> {
    match fit {
        Fit::Central => q[0].clone(),
        Fit::Richardson => {
            let r2 = (alphas[1] / alphas[0]).powi(2);
            (&q[0] * r2 - &q[1]) / (r2 - 1.)
        }
    }
}

fn drift(q: &[DMatrix<f64>]) -> Option<f64> {
    let (first, last) = (q.first()?, q.last()?);
    (q.len() > 1).then(|| relative(peak(&(last - first)), peak(first)))
}

fn asymmetry(a: &DMatrix<f64>) -> f64 {
    relative(peak(&(a - a.transpose())), peak(a))
}

fn column_sums(a: &DMatrix<f64>) -> f64 {
    let worst = (0..a.ncols())
        .map(|j| a.column(j).sum().abs())
        .fold(0., f64::max);
    relative(worst, peak(a))
}

/// An inverse that knows when it should not exist.
pub struct Inverse {
    pub matrix: DMatrix<f64>,
    pub condition: f64,
    /// Largest eigenvalue (negative for a negative definite matrix).
    pub largest: f64,
}

/// Inverse of a symmetric matrix through its eigendecomposition. Refuses a zero or an
/// ill-conditioned spectrum (`max|l| / min|l|` above `limit`), naming the condition
/// number; `hint` is appended to that message.
pub fn checked_inverse(
    matrix: &DMatrix<f64>,
    limit: f64,
    name: &str,
    hint: &str,
) -> Result<Inverse> {
    let (values, vectors) = real_eigen(matrix.clone())?;
    let high = values.iter().fold(0f64, |m, v| m.max(v.abs()));
    let low = values.iter().fold(f64::INFINITY, |m, v| m.min(v.abs()));
    let condition = if low > 0. { high / low } else { f64::INFINITY };
    if low == 0. || high == 0. || !(condition <= limit) {
        return Err(invalid(format!(
            "{name} is singular or ill conditioned: condition number {condition:.3e} exceeds {limit:e}{hint}"
        )));
    }
    let diagonal = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
        values.len(),
        values.iter().map(|v| 1. / v),
    ));
    Ok(Inverse {
        matrix: &vectors * diagonal * vectors.transpose(),
        condition,
        largest: values[values.len() - 1],
    })
}

/// Measures `chi0`, `chi` and `U` with `probe`. The matrices are reserved from `ctx`
/// before the first probe call.
pub fn measure(ctx: &Context, probe: &mut dyn Probe, options: &Options) -> Result<Response> {
    options.validate()?;
    let (shells, channels) = (probe.shells(), probe.channels());
    if shells == 0 || !(1..=2).contains(&channels) {
        return Err(invalid(format!(
            "a response needs at least one shell and one or two spin channels, got {shells} and {channels}"
        )));
    }
    let n = shells * channels;
    let copies = 2 * options.alphas.len() + 14;
    let _reservation = ctx.reserve(product(&[n, n, size_of::<f64>(), copies])?)?;
    let mut calls = 0;
    let q0 = quotients(probe, Pass::Frozen, &options.alphas, n, &mut calls)?;
    let q = quotients(probe, Pass::Scf, &options.alphas, n, &mut calls)?;
    let nonlinearity = match (drift(&q0), drift(&q)) {
        (Some(a), Some(b)) => Some(a.max(b)),
        _ => None,
    };
    if let Some(x) = nonlinearity.filter(|x| !(*x <= options.nonlinearity_limit)) {
        return Err(invalid(format!(
            "the response is not linear in the shifts: nonlinearity {x:.3e} exceeds {:e}; use smaller alphas",
            options.nonlinearity_limit
        )));
    }
    let (raw0, raw) = (
        estimate(options.fit, &options.alphas, &q0),
        estimate(options.fit, &options.alphas, &q),
    );
    let skew = asymmetry(&raw0).max(asymmetry(&raw));
    if !(skew <= options.asymmetry_limit) {
        return Err(invalid(format!(
            "the response is asymmetric: max|A - A^T| / max|A| = {skew:.3e} exceeds {:e}",
            options.asymmetry_limit
        )));
    }
    let symmetrize = |a: &DMatrix<f64>| (a + a.transpose()) * 0.5;
    let (chi0, chi) = (symmetrize(&raw0), symmetrize(&raw));
    let sum_rule = column_sums(&chi0).max(column_sums(&chi));
    if let Some(limit) = options.sum_rule_limit.filter(|l| !(sum_rule <= *l)) {
        return Err(invalid(format!(
            "the fixed-N sum rule fails: column sums are {sum_rule:.3e} of the largest entry, more than {limit:e}; the probe does not conserve the electron number"
        )));
    }
    let hint = if sum_rule < 1e-6 {
        "; the columns sum to zero (fixed-N sum rule): every electron-holding site is correlated, so U is undefined"
    } else {
        ""
    };
    let inverse0 = checked_inverse(&chi0, options.condition_limit, "chi0", hint)?;
    if inverse0.largest >= 0. {
        return Err(invalid(format!(
            "chi0 is not negative definite (largest eigenvalue {:.3e}): occupations must fall as the shift rises",
            inverse0.largest
        )));
    }
    let inverse = checked_inverse(&chi, options.condition_limit, "chi", hint)?;
    let u = &inverse0.matrix - &inverse.matrix;
    Ok(Response {
        shells,
        channels,
        u: symmetrize(&u),
        chi0,
        chi,
        nonlinearity,
        asymmetry: skew,
        sum_rule,
        condition_chi0: inverse0.condition,
        condition_chi: inverse.condition,
        evaluations: calls,
        limit: options.condition_limit,
    })
}
