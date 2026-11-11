//! The double-factorized two-body representation of an active-space Hamiltonian.
//!
//! The two-electron tensor of the active orbitals `c` (physicists' order) is
//! `<pq|rs> = (pr|qs) = sum_l s_l L^l_pr L^l_qs`. The leaves `L^l` are the first factorization,
//! taken as it comes: each Cholesky vector of the basis rotated into the active orbitals
//! (`s_l = +1`) and each signed augmentation factor of a PAW basis, scaled by the square root of
//! its weight (`s_l` the sign of the weight). The second factorization diagonalizes each leaf,
//! `L^l = sum_t lambda_t u_t u_t^T`. The truncation `tau` (Hartree) bounds what is dropped: the
//! truncated two-body operator is `(1/2) sum_l s_l sum_tu lambda_t lambda_u N_tu` over the kept
//! pairs, with `N_tu = sum_{sigma tau} a^dag_t,sigma a^dag_u,tau a_u,tau a_t,sigma` of norm at
//! most 4, so its change has norm at most `2 sum_l (S_l^2 - R_l^2)` (`S_l` the sum of the leaf's
//! `|lambda_t|`, `R_l` that of the kept ones). Eigenpairs are dropped smallest first, over all
//! leaves, while that bound stays at or below `tau`: no energy of the model moves by more than
//! `tau`, plus the roundoff of the Pauli coefficients at or below [`PAULI_FLOOR`] that the
//! operator drops. A leaf left with no eigenpair disappears. What is stored is the
//! kept `lambda` and `u`: at most `L n^2` numbers, never the `n^4` tensor.
//!
//! The qubit Hamiltonian is built leaf by leaf. The one-body operator
//! `O_l = sum_sigma sum_pr L~^l_pr a^dag_p,sigma a_r,sigma` of the truncated leaf `L~^l` is mapped
//! once, and its square is accumulated into the result one pair of Pauli strings at a time.
//! Since `O_l^2` is the normal-ordered two-body operator of the leaf plus
//! `sum_sigma sum_ps (L~^l L~^l)_ps a^dag_p,sigma a_s,sigma`, the two-body Hamiltonian is
//! `(1/2) sum_l s_l (O_l^2 - Q_l)` with `Q_l` that one-body piece. No list of `n^4` fermion
//! monomials is formed. With nothing truncated it equals the dense route's operator to roundoff.
use super::{integrals::Eri, invalid};
use crate::math::linalg::real_eigen;
use crate::quantum::{algebra::Term, Encoding, Operator, C64};
use crate::{product, Context, Error, Reservation, Result};
use nalgebra::DMatrix;
use std::collections::HashMap;

/// Eigenpair weights `lambda^2` at or below this (Hartree; `|lambda| <= 1e-12`) are roundoff and
/// are dropped whatever the truncation, so that `truncation = 0` keeps every eigenpair that
/// carries a number. They enter the error bound like any other dropped pair.
pub const WEIGHT_FLOOR: f64 = 1e-24;
/// Two-body Pauli coefficients at or below this (Hartree) are dropped from the operator: the
/// residue of strings the dense tensor has as exact zeros, at the level the dense route drops
/// tensor elements.
pub const PAULI_FLOOR: f64 = 1e-12;
/// One-body coefficients at or below this are not mapped, as on the dense route.
const ONE_BODY_FLOOR: f64 = 1e-12;

/// How an active-space model holds its two-electron integrals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Representation {
    /// The `n^4` tensor (the default).
    Dense,
    /// [`DoubleFactorized`]: the smallest eigenpairs dropped while the bound on the change of
    /// any energy stays at or below `truncation` Hartree.
    DoubleFactorized { truncation: f64 },
}

impl Representation {
    /// The representation named `name` (`"dense"` or `"double-factorized"`); `truncation`
    /// (Hartree) is required by the double-factorized one and refused by the dense one.
    pub fn parse(name: &str, truncation: Option<f64>) -> Result<Self> {
        match (name, truncation) {
            ("dense", None) => Ok(Self::Dense),
            ("dense", Some(_)) => Err(invalid(
                "a two_body truncation applies only to representation 'double-factorized'",
            )),
            ("double-factorized", Some(t)) if t.is_finite() && t >= 0. => {
                Ok(Self::DoubleFactorized { truncation: t })
            }
            ("double-factorized", Some(_)) => Err(invalid(
                "the double-factorized truncation must be a finite, nonnegative energy",
            )),
            ("double-factorized", None) => Err(invalid(
                "representation 'double-factorized' needs a truncation: it changes energies, \
                 so it is never implied",
            )),
            (other, _) => Err(invalid(format!(
                "unknown two_body representation {other:?}; choose 'dense' or 'double-factorized'"
            ))),
        }
    }
}

/// One leaf after its second factorization: the kept eigenvalues (square root of Hartree) and
/// their eigenvectors (`n x kept`, columns), with the sign of the leaf.
pub struct Leaf {
    pub sign: f64,
    pub values: Vec<f64>,
    pub vectors: DMatrix<f64>,
}
impl Leaf {
    /// `sum_t lambda_t u_t u_t^T`, the truncated leaf (`n x n`).
    pub fn matrix(&self) -> DMatrix<f64> {
        let scaled = DMatrix::from_fn(self.vectors.nrows(), self.vectors.ncols(), |i, t| {
            self.vectors[(i, t)] * self.values[t]
        });
        &scaled * self.vectors.transpose()
    }
}

/// The truncated double factorization of an active-space two-electron tensor.
pub struct DoubleFactorized {
    /// Active spatial orbitals.
    pub n: usize,
    /// The bound on the change of any energy, Hartree.
    pub truncation: f64,
    pub leaves: Vec<Leaf>,
    /// First-factorization vectors considered (Cholesky rank plus augmentation factors).
    pub candidates: usize,
    /// Eigenpairs before the truncation, `n * candidates`.
    pub eigenpairs: usize,
    /// Eigenpairs kept.
    pub kept: usize,
    /// `2 sum_l (S_l^2 - R_l^2)` Hartree, with `S_l` the sum of a leaf's `|lambda_t|` and `R_l`
    /// that of its kept ones: a bound on the operator norm of the change of the two-body
    /// Hamiltonian, hence on the change of any energy. At most the truncation, plus the roundoff
    /// below [`WEIGHT_FLOOR`].
    pub error_bound: f64,
    _memory: Reservation,
}

impl DoubleFactorized {
    /// The representation of `eri` in the active orbitals `orbitals` (`m x n`, columns over the
    /// orthonormal basis), truncated at `truncation` Hartree. Refuses complex orbitals and a
    /// packed (non-factorized) tensor by name rather than densifying it.
    pub fn from_eri(
        ctx: &Context,
        eri: &Eri,
        orbitals: &DMatrix<C64>,
        truncation: f64,
    ) -> Result<Self> {
        if !truncation.is_finite() || truncation < 0. {
            return Err(invalid(
                "the double-factorized truncation must be a finite, nonnegative energy",
            ));
        }
        let (m, n) = orbitals.shape();
        if m != eri.m() || n == 0 {
            return Err(invalid(
                "the active orbitals do not match the two-electron factors",
            ));
        }
        let scale = orbitals.iter().fold(1., |a: f64, v| a.max(v.norm()));
        if orbitals.iter().any(|v| v.im.abs() > 1e-10 * scale) {
            return Err(Error::Unsupported(
                "the double-factorized two-body representation needs real active orbitals; \
                 these are complex: use representation 'dense'"
                    .into(),
            ));
        }
        let c = orbitals.map(|v| v.re);
        // Every rotated leaf's eigen decomposition (`n^2 + n` numbers per leaf, the size of the
        // rotated factors), then one global truncation over all of them.
        // The verified solver: the tridiagonal QR iteration alone can stop at a wrong
        // decomposition of a clustered spectrum (a PAW leaf of H2O was off by 1e-4).
        let mut decomposed: Vec<(f64, Vec<f64>, DMatrix<f64>)> = Vec::new();
        let mut failure = None;
        visit(eri, &c, &mut |rotated, sign| {
            if failure.is_some() {
                return;
            }
            match real_eigen(0.5 * (&rotated + rotated.transpose())) {
                Ok((values, vectors)) => decomposed.push((sign, values, vectors)),
                Err(e) => failure = Some(e),
            }
        })?;
        if let Some(e) = failure {
            return Err(e);
        }
        let candidates = decomposed.len();
        let _work = ctx.reserve(product(&[candidates.max(1), n, n + 1, 8])?)?;
        let mut order: Vec<(f64, usize, usize)> = decomposed
            .iter()
            .enumerate()
            .flat_map(|(l, (_, values, _))| {
                values.iter().enumerate().map(move |(t, v)| (v * v, l, t))
            })
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        // The bound 2 sum_l (S_l^2 - R_l^2), S_l = sum_t |lambda_t| and R_l over the kept ones:
        // dropping `t` from leaf `l` raises it by 2 |lambda_t| (2 R_l - |lambda_t|).
        let mut retained: Vec<f64> = decomposed
            .iter()
            .map(|(_, values, _)| values.iter().map(|v| v.abs()).sum())
            .collect();
        let mut dropped = vec![vec![false; n]; candidates];
        let mut bound = 0f64;
        for &(w, l, t) in &order {
            let size = w.sqrt();
            let cost = 2. * size * (2. * retained[l] - size);
            if w <= WEIGHT_FLOOR || bound + cost <= truncation {
                bound += cost;
                retained[l] -= size;
                dropped[l][t] = true;
            }
        }
        let mut leaves = Vec::new();
        let mut kept = 0usize;
        for (l, (sign, values, vectors)) in decomposed.into_iter().enumerate() {
            let keep: Vec<usize> = (0..n).filter(|&t| !dropped[l][t]).collect();
            kept += keep.len();
            if !keep.is_empty() {
                leaves.push(Leaf {
                    sign,
                    values: keep.iter().map(|&t| values[t]).collect(),
                    vectors: DMatrix::from_fn(n, keep.len(), |i, j| vectors[(i, keep[j])]),
                });
            }
        }
        let stored = leaves
            .iter()
            .map(|l| l.values.len() * (n + 1))
            .sum::<usize>();
        let memory = ctx.reserve(product(&[stored.max(1), 8])?)?;
        Ok(Self {
            n,
            truncation,
            leaves,
            candidates,
            eigenpairs: candidates * n,
            kept,
            error_bound: bound,
            _memory: memory,
        })
    }

    /// The numbers the representation stores: each kept eigenvalue and its eigenvector.
    pub fn stored_numbers(&self) -> usize {
        self.kept * (self.n + 1)
    }

    /// The qubit operator `constant + sum h_pq E_pq + (1/2) sum_l s_l (O_l^2 - Q_l)` in
    /// `encoding` (spin-block modes `p + sigma n`), with the dense route's decomposition: the
    /// one-body terms, then the two-body terms.
    pub fn operator(
        &self,
        ctx: &Context,
        encoding: &Encoding,
        one: &[C64],
        constant: f64,
    ) -> Result<Operator> {
        let n = self.n;
        if one.len() != n * n || encoding.modes != 2 * n {
            return Err(invalid(
                "the double-factorized operator dimensions disagree",
            ));
        }
        // The mapped excitation E^sigma_pr for every pair, once.
        let _excitations = ctx.reserve(product(&[2 * n, 2 * n, 4 * 32])?)?;
        let mut excitations = Vec::with_capacity(2 * n * n);
        for spin in 0..2 {
            for p in 0..n {
                for r in 0..n {
                    let mapped =
                        encoding.monomial(ctx, &[(p + spin * n, true), (r + spin * n, false)])?;
                    excitations.push(mapped.terms.clone());
                }
            }
        }
        let excitation = |spin: usize, p: usize, r: usize| &excitations[(spin * n + p) * n + r];
        let one_body = |matrix: &dyn Fn(usize, usize) -> C64, floor: f64| -> Vec<Term> {
            let mut out = Vec::new();
            for spin in 0..2 {
                for p in 0..n {
                    for r in 0..n {
                        let h = matrix(p, r);
                        if h.norm() > floor {
                            out.extend(excitation(spin, p, r).iter().map(|t| Term {
                                coefficient: t.coefficient * h,
                                ..*t
                            }));
                        }
                    }
                }
            }
            out
        };
        let first = one_body(&|p, r| one[p * n + r], ONE_BODY_FLOOR);
        // The two-body accumulator: one entry per Pauli string, bounded by the dense route's
        // count of distinct spin-orbital quartets.
        let _accumulator = ctx.reserve(product(&[2 * n, 2 * n, 2 * n, 2 * n, 16])?)?;
        let mut two: HashMap<(u64, u64), C64> = HashMap::new();
        for leaf in &self.leaves {
            let l = leaf.matrix();
            let square = &l * &l;
            let mut strings: HashMap<(u64, u64), C64> = HashMap::new();
            for t in one_body(&|p, r| C64::new(l[(p, r)], 0.), 0.) {
                *strings.entry((t.x, t.z)).or_default() += t.coefficient;
            }
            let mut strings: Vec<Term> = strings
                .into_iter()
                .filter(|(_, c)| c.norm() > 1e-14)
                .map(|((x, z), coefficient)| Term { x, z, coefficient })
                .collect();
            strings.sort_unstable_by_key(|t| (t.x, t.z));
            let half = 0.5 * leaf.sign;
            // O^2 = sum_i c_i^2 + sum_{i<j} c_i c_j {S_i, S_j}; anticommuting pairs cancel.
            for (i, a) in strings.iter().enumerate() {
                *two.entry((0, 0)).or_default() += half * a.coefficient * a.coefficient;
                for b in &strings[i + 1..] {
                    if ((a.x & b.z).count_ones() + (a.z & b.x).count_ones()) % 2 == 1 {
                        continue;
                    }
                    let (x, z) = (a.x ^ b.x, a.z ^ b.z);
                    let exponent = (a.x & a.z).count_ones() as i32
                        + (b.x & b.z).count_ones() as i32
                        - (x & z).count_ones() as i32
                        + 2 * (a.z & b.x).count_ones() as i32;
                    *two.entry((x, z)).or_default() +=
                        2. * half * a.coefficient * b.coefficient * phase(exponent);
                }
            }
            for t in one_body(&|p, s| C64::new(square[(p, s)], 0.), 0.) {
                *two.entry((t.x, t.z)).or_default() -= half * t.coefficient;
            }
        }
        let second: Vec<Term> = two
            .into_iter()
            .filter(|(_, c)| c.norm() > PAULI_FLOOR)
            .map(|((x, z), coefficient)| Term { x, z, coefficient })
            .collect();
        let qubits = encoding.n_qubits();
        let identity = encoding.monomial(ctx, &[])?;
        let mut terms: Vec<Term> = identity
            .terms
            .iter()
            .map(|t| Term {
                coefficient: t.coefficient * constant,
                ..*t
            })
            .collect();
        terms.extend_from_slice(&first);
        terms.extend_from_slice(&second);
        let mut operator = Operator::from_terms(ctx, qubits, terms)?;
        for group in [first, second] {
            let group = Operator::from_terms(ctx, qubits, group)?;
            if !group.terms.is_empty() {
                operator.decomposition.push(group);
            }
        }
        Ok(operator)
    }
}

fn phase(n: i32) -> C64 {
    match n.rem_euclid(4) {
        0 => C64::new(1., 0.),
        1 => C64::new(0., 1.),
        2 => C64::new(-1., 0.),
        _ => C64::new(0., -1.),
    }
}

/// Every first-factorization leaf of `eri` rotated into `c` (`m x n`), with its sign.
fn visit(eri: &Eri, c: &DMatrix<f64>, push: &mut dyn FnMut(DMatrix<f64>, f64)) -> Result<()> {
    match eri {
        Eri::Cholesky {
            m, rank, factors, ..
        } => {
            let m = *m;
            for column in 0..*rank {
                let b = DMatrix::from_fn(m, m, |p, q| {
                    factors.as_slice()[super::integrals::pair(p, q) * rank + column]
                });
                push(c.transpose() * b * c, 1.);
            }
            Ok(())
        }
        Eri::Augmented {
            base, correction, ..
        } => {
            visit(base, c, push)?;
            for (factor, &weight) in correction.factors.iter().zip(&correction.weights) {
                if weight != 0. {
                    push(
                        c.transpose() * factor * c * weight.abs().sqrt(),
                        weight.signum(),
                    );
                }
            }
            Ok(())
        }
        Eri::Packed { .. } => Err(Error::Unsupported(
            "the double-factorized two-body representation needs Cholesky two-electron \
             factors; these integrals are a packed tensor: use representation 'dense'"
                .into(),
        )),
    }
}
