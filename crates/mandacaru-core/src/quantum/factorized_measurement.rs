//! The `O(M)` basis-rotation measurement of a molecular energy.
//!
//! Double factorization ([`super::factorization`]) writes the Hamiltonian as a constant, a one-body
//! term and `L` squares of one-body terms. Each of those `L + 1` terms is a function of
//! occupation numbers in an orbital basis of its own, so after the orbital rotation `U` it is a
//! **diagonal** observable (`I` and `Z` only): one computational-basis measurement of the rotated
//! register answers for the whole term. This module owns everything between the factorization and
//! the executor: the Givens network that realizes each `U` on adjacent modes, the diagonal
//! observable after it, and the energy assembled from the per-basis values.
//!
//! Conventions (Jordan-Wigner, spin-block order, qubit zero leftmost in a label):
//!
//! * mode `P = p + sigma * M` is qubit `P`; the factorization's rotations are block diagonal in
//!   spin, so the elimination never mixes the two blocks;
//! * a rotation of modes `p, p + 1` by `theta` is `exp(theta A)` with the anti-Hermitian
//!   `A = a^dag_p a_{p+1} - a^dag_{p+1} a_p = (i/2)(X_p Y_{p+1} - Y_p X_{p+1})`; the two Pauli
//!   strings commute, so the exponential is exactly two Pauli rotations of angle
//!   `theta * (+-1/2)`, the pair `(label, angle)` the circuit emitter takes for any generator;
//! * the circuit applies the rotations in the order the elimination found them. Two reversals
//!   cancel here: the identity `W = exp(A_{r_1}) ... exp(A_{r_K})` runs over the *reversed*
//!   elimination order, and a circuit given `[g_1, ..., g_K]` applies `g_1` first, so it realizes
//!   `exp(g_K) ... exp(g_1)`. Reversing the list once more is silently wrong (0.15 Ha low on LiH);
//!   `tests/factorized_measurement.rs` pins the single-particle matrix the circuit induces.
use super::factorization::{spatial_double_factorization, DoubleFactorizationData};
use super::{Operator, C64};
use crate::{product, Context, Error, Result};

/// Leaves whose weight is below this fraction of the largest are dropped.
pub const LEAF_TOLERANCE: f64 = 1e-8;
/// Orbital coefficients below this are dropped from a leaf's sum.
pub const COEFFICIENT_TOLERANCE: f64 = 1e-10;
/// Rotation angles below this are dropped: the gate would be the identity.
pub const GIVENS_TOLERANCE: f64 = 1e-12;
/// A coefficient below this is not written into an observable.
const ZERO: f64 = 1e-15;

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

/// Factor a real orthogonal `m x m` matrix (row major) into **adjacent** Givens rotations.
///
/// Returns `(rotations, signs)`: `rotations` is a list of `(p, theta)` -- a rotation mixing modes
/// `p` and `p + 1` by `theta`, in elimination order -- and `signs` the diagonal `+-1` left over,
/// so that `U = G_1(-theta_1) ... G_K(-theta_K) D` with `G_p(theta)` the rotation
/// `[[cos, sin], [-sin, cos]]` on rows `(p, p + 1)`.
///
/// The elimination is the ordinary Givens QR restricted to neighboring rows, so that every
/// rotation is a nearest-neighbor gate under Jordan-Wigner: column by column, each sub-diagonal
/// entry is zeroed against the row above it. What is left is orthogonal and upper triangular,
/// hence diagonal with entries `+-1`. The signs are not needed to measure occupations (`D` sends
/// `b_k -> +-b_k`, which leaves `b^dag_k b_k` unchanged); they are returned to check the
/// decomposition against its input.
pub fn givens_decomposition(
    unitary: &[f64],
    m: usize,
    tol: f64,
) -> Result<(Vec<(usize, f64)>, Vec<f64>)> {
    if unitary.len() != m * m {
        return Err(invalid(format!(
            "expected a square matrix, got {} elements for {m} rows",
            unitary.len()
        )));
    }
    if unitary.iter().any(|v| !v.is_finite()) {
        return Err(invalid("the Givens decomposition needs finite entries"));
    }
    let mut u = unitary.to_vec();
    for i in 0..m {
        for j in 0..m {
            let dot: f64 = (0..m).map(|k| u[i * m + k] * u[j * m + k]).sum();
            if (dot - f64::from(u8::from(i == j))).abs() > 1e-8 {
                return Err(invalid(
                    "the Givens decomposition needs a real orthogonal matrix; this one is not",
                ));
            }
        }
    }
    let mut rotations = Vec::new();
    for column in 0..m.saturating_sub(1) {
        for row in (column + 1..m).rev() {
            let below = u[row * m + column];
            let above = u[(row - 1) * m + column];
            if below.abs() <= tol {
                continue;
            }
            let theta = below.atan2(above);
            let (sin, cos) = theta.sin_cos();
            for col in 0..m {
                let first = u[(row - 1) * m + col];
                let second = u[row * m + col];
                u[(row - 1) * m + col] = cos * first + sin * second;
                u[row * m + col] = -sin * first + cos * second;
            }
            rotations.push((row - 1, theta));
        }
    }
    let signs = (0..m).map(|i| u[i * m + i]).collect();
    Ok((rotations, signs))
}

/// The anti-Hermitian generator `a^dag_p a_{p+1} - a^dag_{p+1} a_p` of a Givens rotation of
/// modes `p, p + 1` in an `n_modes` register, Jordan-Wigner: the two commuting Pauli strings with
/// their imaginary coefficients, `(X_p Y_{p+1}, +1/2)` and `(Y_p X_{p+1}, -1/2)`.
pub fn givens_generator(n_modes: usize, p: usize) -> Result<[(String, f64); 2]> {
    if p + 1 >= n_modes {
        return Err(invalid(format!(
            "a Givens rotation of modes {p} and {} needs more than {n_modes} modes",
            p + 1
        )));
    }
    let label = |first: char, second: char| {
        (0..n_modes)
            .map(|k| match k {
                k if k == p => first,
                k if k == p + 1 => second,
                _ => 'I',
            })
            .collect::<String>()
    };
    Ok([(label('X', 'Y'), 0.5), (label('Y', 'X'), -0.5)])
}

/// The circuit of one basis change: `(label, angle)` Pauli rotations in circuit order, from the
/// Givens decomposition of `orbitals` (`m x m`, row major). The angle is `theta` times the
/// generator's imaginary coefficient, exactly what the executor writes for an ansatz generator.
pub fn basis_rotation(orbitals: &[f64], m: usize) -> Result<Vec<(String, f64)>> {
    let (rotations, _signs) = givens_decomposition(orbitals, m, GIVENS_TOLERANCE)?;
    let mut circuit = Vec::with_capacity(2 * rotations.len());
    for (p, theta) in rotations {
        for (label, coefficient) in givens_generator(m, p)? {
            circuit.push((label, theta * coefficient));
        }
    }
    Ok(circuit)
}

/// `weight * (sum_k c_k n_k)` (`square = false`) or its square, as `(label, coefficient)` of
/// `I`/`Z` strings, with `n_k = (I - Z_k) / 2`.
///
/// `n_j n_k = (I - Z_j - Z_k + Z_j Z_k) / 4` for `j != k` and `n_k^2 = n_k`, so the square is
/// `sum_k c_k^2 n_k + 2 sum_{j<k} c_j c_k n_j n_k`. The result is diagonal by construction: the
/// whole claim of the scheme is that one computational-basis measurement answers for the leaf.
pub fn diagonal_observable(coefficients: &[f64], weight: f64, square: bool) -> Vec<(String, f64)> {
    let n = coefficients.len();
    let label = |sites: &[usize]| {
        (0..n)
            .map(|k| if sites.contains(&k) { 'Z' } else { 'I' })
            .collect::<String>()
    };
    let mut identity = 0.;
    let mut single = vec![0.; n];
    let mut pair = vec![0.; n * n];
    if square {
        for (k, &c) in coefficients.iter().enumerate() {
            // c_k^2 n_k
            identity += 0.5 * c * c;
            single[k] -= 0.5 * c * c;
        }
        for j in 0..n {
            for k in j + 1..n {
                let c = 2. * coefficients[j] * coefficients[k] / 4.;
                identity += c;
                single[j] -= c;
                single[k] -= c;
                pair[j * n + k] += c;
            }
        }
    } else {
        for (k, &c) in coefficients.iter().enumerate() {
            identity += 0.5 * c;
            single[k] -= 0.5 * c;
        }
    }
    let mut terms = vec![(label(&[]), weight * identity)];
    terms.extend((0..n).map(|k| (label(&[k]), weight * single[k])));
    for j in 0..n {
        for k in j + 1..n {
            terms.push((label(&[j, k]), weight * pair[j * n + k]));
        }
    }
    terms.retain(|(_, c)| c.abs() > ZERO);
    terms
}

/// One measurement basis: the Givens circuit that rotates into it and the diagonal observable
/// read off the rotated register.
pub struct FactorizedBasis {
    /// `(label, angle)` Pauli rotations, in circuit order, appended after the ansatz.
    pub rotation: Vec<(String, f64)>,
    /// `weight * (sum_k c_k n_k)^2` or the one-body `sum_k e_k n_k`, `I` and `Z` only.
    pub observable: Operator,
}

/// The factorized measurement plan of a molecular Hamiltonian: a constant and `L + 1` bases, the
/// one-body term first and then one per leaf. `<H> = constant + sum_b <observable_b>_b`.
pub struct FactorizedMeasurement {
    pub constant: f64,
    pub bases: Vec<FactorizedBasis>,
    /// Frobenius norm of the two-body tensor the leaf cutoff discarded.
    pub truncation_error: f64,
}

impl FactorizedMeasurement {
    /// Measurement plan of a spin-free, real Hamiltonian from its spatial integrals: the
    /// `m x m` one-body matrix `h`, the physicists' tensor `eri[p][q][r][s] = <pq|rs>` and the
    /// scalar `constant`, over `m` spatial orbitals (`2m` spin orbitals, spin-block order).
    ///
    /// Refuses what the scheme cannot express rather than measuring a different Hamiltonian: a
    /// wrong size, a non-symmetric one-body matrix, or a two-body tensor without the pair
    /// symmetry `<pq|rs> = <qp|sr>` the factorization assumes.
    pub fn from_integrals(
        ctx: &Context,
        m: usize,
        h: &[f64],
        eri: &[f64],
        constant: f64,
    ) -> Result<Self> {
        if h.len() != m * m || eri.len() != product(&[m, m, m, m])? {
            return Err(invalid("the integrals have the wrong size"));
        }
        let scale = h.iter().chain(eri).fold(1., |a: f64, v| a.max(v.abs()));
        // The pair matrix (m^2 x m^2) and its eigenvectors.
        let _memory = ctx.reserve(product(&[m, m, m, m, 8 * 3])?)?;
        for p in 0..m {
            for q in 0..m {
                if (h[p * m + q] - h[q * m + p]).abs() > 1e-9 * scale {
                    return Err(invalid("the one-body matrix is not symmetric"));
                }
                for r in 0..m {
                    for s in 0..m {
                        let a = eri[((p * m + q) * m + r) * m + s];
                        let b = eri[((q * m + p) * m + s) * m + r];
                        if (a - b).abs() > 1e-9 * scale {
                            return Err(invalid(
                                "the two-body tensor lacks the symmetry <pq|rs> = <qp|sr> the \
                                 factorization needs",
                            ));
                        }
                    }
                }
            }
        }
        let factorization = spatial_double_factorization(
            &h,
            m,
            Some((&eri, m)),
            None,
            None,
            None,
            LEAF_TOLERANCE,
            COEFFICIENT_TOLERANCE,
            None,
            constant,
        )
        .map_err(invalid)?;
        Self::from_factorization(ctx, &factorization)
    }

    /// The plan of an existing factorization (see [`Self::from_integrals`] for the conventions).
    pub fn from_factorization(
        ctx: &Context,
        factorization: &DoubleFactorizationData,
    ) -> Result<Self> {
        let n = factorization.n_modes;
        let row_major = |matrix: &nalgebra::DMatrix<f64>| {
            (0..n)
                .flat_map(|r| (0..n).map(move |c| (r, c)))
                .map(|(r, c)| matrix[(r, c)])
                .collect::<Vec<_>>()
        };
        let basis = |orbitals: &nalgebra::DMatrix<f64>,
                     coefficients: &[f64],
                     weight: f64,
                     square: bool|
         -> Result<FactorizedBasis> {
            let terms: Vec<_> = diagonal_observable(coefficients, weight, square)
                .into_iter()
                .map(|(label, c)| (label, C64::new(c, 0.)))
                .collect();
            Ok(FactorizedBasis {
                rotation: basis_rotation(&row_major(orbitals), n)?,
                observable: Operator::new(ctx, n, &terms)?,
            })
        };
        let mut bases = Vec::with_capacity(factorization.leaves.len() + 1);
        bases.push(basis(
            &factorization.one_body_orbitals,
            factorization.one_body_energies.as_slice(),
            1.,
            false,
        )?);
        for leaf in &factorization.leaves {
            bases.push(basis(
                &leaf.orbitals,
                leaf.coefficients.as_slice(),
                0.5 * leaf.weight,
                true,
            )?);
        }
        Ok(Self {
            constant: factorization.constant,
            bases,
            truncation_error: factorization.truncation_error,
        })
    }

    /// `L + 1`: one basis per leaf plus one for the one-body term. Each basis is a single
    /// qubit-wise commuting group (`I` and `Z` only), so this is also the number of measurement
    /// bases the energy costs.
    pub fn measurement_bases(&self) -> usize {
        self.bases.len()
    }

    /// `<H>` from the expectation value of each basis's observable, in basis order.
    pub fn energy(&self, values: &[f64]) -> Result<f64> {
        if values.len() != self.bases.len() || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid(format!(
                "the factorized energy needs {} finite basis values, got {}",
                self.bases.len(),
                values.len()
            )));
        }
        Ok(self.constant + values.iter().sum::<f64>())
    }
}
