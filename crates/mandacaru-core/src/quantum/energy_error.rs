//! The standard error of a shot-based energy, from the in-basis covariances of its strings.
//!
//! A shot run measures the Hamiltonian basis by basis: each qubit-wise commuting basis `b` is
//! rotated to `Z` (`exp(i pi/4 Y)` on its `X` qubits, `exp(-i pi/4 X)` on its `Y` qubits, both
//! sending the letter to `Z` with a plus sign) and read `N` times. The basis's part of the
//! energy, `H_b = sum_i c_i Z_i` after the rotation, is estimated by the mean of `N` per-shot
//! values, so its variance is `(<H_b^2> - <H_b>^2) / N`, and the bases are independent:
//!
//! `sigma_E^2 = sum_b ( sum_i c_i^2 + sum_{i != j} c_i c_j <Z_i Z_j> - E_b^2 ) / N`.
//!
//! `Z_i Z_j` is the `Z` string of the XOR of the two masks, read on the same shots, so the plan
//! adds the distinct products of each basis to that basis's request. The products cost no shot
//! and no basis, so no budget counts them. A basis whose products exceed [`MAX_PRODUCTS`] (a
//! classical limit) leaves the plan without products and its energies without an error, with
//! the reason in [`Plan::unavailable`]: the run itself measures what it measured before.
use super::{algebra::Term, Operator, C64};
use crate::{execution::qwc, Context, Error, Result};
use std::collections::HashMap;

/// Distinct product strings one measurement basis may need: a basis of `s` strings needs up to
/// `s (s - 1) / 2`, all `Z` strings read on the shots the basis already takes.
pub const MAX_PRODUCTS: usize = 20_000;

/// One basis of the plan: the rotations to append, its strings `(Z mask, c_i)` and products
/// `(Z mask, sum of c_i c_j over the ordered pairs i != j giving it)`.
pub struct Basis {
    pub tail: Vec<(String, f64)>,
    pub strings: Vec<(u64, f64)>,
    pub products: Vec<(u64, f64)>,
}

/// The energy's measurement with its error: a constant and the rotated bases.
pub struct Plan {
    pub n: usize,
    pub constant: f64,
    pub bases: Vec<Basis>,
    /// Why this plan reports no standard error (`None` when it does): a basis past the limit.
    pub unavailable: Option<String>,
}

fn label(n: usize, z: u64) -> String {
    Term {
        x: 0,
        z,
        coefficient: C64::new(1., 0.),
    }
    .label(n)
}

/// The distinct products of `strings` and their summed weights, or `None` past [`MAX_PRODUCTS`].
fn products(strings: &[(u64, f64)]) -> Option<Vec<(u64, f64)>> {
    let mut sums: HashMap<u64, f64> = HashMap::new();
    for (a, &(zi, ci)) in strings.iter().enumerate() {
        for &(zj, cj) in &strings[a + 1..] {
            *sums.entry(zi ^ zj).or_default() += 2. * ci * cj;
            if sums.len() > MAX_PRODUCTS {
                return None;
            }
        }
    }
    let mut out: Vec<_> = sums.into_iter().collect();
    out.sort_unstable_by_key(|(z, _)| *z);
    Some(out)
}

/// Why a plan reports no error: one of its bases needs more products than the limit.
fn over_limit(strings: usize) -> String {
    format!(
        "not reported: a measurement basis of {strings} strings needs more than {MAX_PRODUCTS} \
         covariance product strings, past the classical limit of the estimate"
    )
}

/// Every basis's products, or none with the reason when one basis is past the limit: an error
/// that misses one basis's covariance is not an error estimate.
fn all_products(bases: &mut [Basis]) -> Option<String> {
    let mut found = Vec::with_capacity(bases.len());
    for basis in bases.iter() {
        match products(&basis.strings) {
            Some(p) => found.push(p),
            None => {
                found.clear();
                let reason = over_limit(basis.strings.len());
                bases.iter_mut().for_each(|b| b.products.clear());
                return Some(reason);
            }
        }
    }
    for (basis, p) in bases.iter_mut().zip(found) {
        basis.products = p;
    }
    None
}

impl Plan {
    /// The plan of `hamiltonian` measured in its qubit-wise commuting bases
    /// (`execution::qwc`, the grouping the measurement cost already counts).
    pub fn commuting(ctx: &Context, hamiltonian: &Operator) -> Result<Self> {
        if !hamiltonian.is_hermitian() {
            return Err(Error::Invalid(
                "a measured Hamiltonian must be Hermitian".into(),
            ));
        }
        let n = hamiltonian.n;
        let mut constant = 0.;
        let terms: Vec<&Term> = hamiltonian
            .terms
            .iter()
            .filter(|t| {
                if (t.x, t.z) == (0, 0) {
                    constant += t.coefficient.re;
                    false
                } else {
                    true
                }
            })
            .collect();
        let labels: Vec<String> = terms.iter().map(|t| t.label(n)).collect();
        let groups = qwc(ctx, &labels, None)?;
        let mut bases = Vec::with_capacity(groups.groups.len());
        for (letters, members) in &groups.groups {
            let mut tail = Vec::new();
            for (j, c) in letters.bytes().enumerate() {
                let bit = 1u64 << (n - 1 - j);
                let rotation = |x: u64, z: u64| Term {
                    x,
                    z,
                    coefficient: C64::new(1., 0.),
                };
                match c {
                    b'X' => tail.push((rotation(bit, bit).label(n), std::f64::consts::FRAC_PI_4)),
                    b'Y' => tail.push((rotation(bit, 0).label(n), -std::f64::consts::FRAC_PI_4)),
                    _ => {}
                }
            }
            let strings: Vec<(u64, f64)> = members
                .iter()
                .map(|&i| (terms[i].x | terms[i].z, terms[i].coefficient.re))
                .collect();
            let products = Vec::new();
            bases.push(Basis {
                tail,
                strings,
                products,
            });
        }
        let unavailable = all_products(&mut bases);
        Ok(Self {
            n,
            constant,
            bases,
            unavailable,
        })
    }

    /// The plan of a factorized measurement: each basis is already a rotation and a diagonal
    /// observable.
    pub fn factorized(plan: &super::factorized_measurement::FactorizedMeasurement) -> Result<Self> {
        let mut bases = Vec::with_capacity(plan.bases.len());
        let mut constant = plan.constant;
        let mut n = 0;
        for basis in &plan.bases {
            n = basis.observable.n;
            let mut strings = Vec::new();
            for t in &basis.observable.terms {
                if t.x != 0 {
                    return Err(Error::Invalid(
                        "a factorized basis observable must be diagonal".into(),
                    ));
                }
                if t.z == 0 {
                    constant += t.coefficient.re;
                } else {
                    strings.push((t.z, t.coefficient.re));
                }
            }
            let products = Vec::new();
            bases.push(Basis {
                tail: basis.rotation.clone(),
                strings,
                products,
            });
        }
        let unavailable = all_products(&mut bases);
        Ok(Self {
            n,
            constant,
            bases,
            unavailable,
        })
    }

    /// The `Z` strings each basis reads: its strings, then its products not already among them.
    /// Returns, per basis, the labels and the position of each string and product among them.
    pub fn observables(&self) -> Vec<(Vec<String>, Vec<usize>, Vec<usize>)> {
        self.bases
            .iter()
            .map(|basis| {
                let mut labels = Vec::new();
                let mut at: HashMap<u64, usize> = HashMap::new();
                let mut place = |z: u64, labels: &mut Vec<String>| {
                    *at.entry(z).or_insert_with(|| {
                        labels.push(label(self.n, z));
                        labels.len() - 1
                    })
                };
                let strings = basis
                    .strings
                    .iter()
                    .map(|&(z, _)| place(z, &mut labels))
                    .collect();
                let products = basis
                    .products
                    .iter()
                    .map(|&(z, _)| place(z, &mut labels))
                    .collect();
                (labels, strings, products)
            })
            .collect()
    }

    /// `(E, sigma_E)` from each basis's read values (in [`Self::observables`] order) and `shots`.
    pub fn energy(
        &self,
        positions: &[(Vec<String>, Vec<usize>, Vec<usize>)],
        values: &[Vec<f64>],
        shots: u64,
    ) -> Result<(f64, Option<f64>)> {
        if shots == 0 || values.len() != self.bases.len() {
            return Err(Error::Invalid(
                "the energy error needs shots and one reading per basis".into(),
            ));
        }
        let (mut energy, mut variance) = (self.constant, 0.);
        for ((basis, (_, strings, products)), read) in self.bases.iter().zip(positions).zip(values)
        {
            let part: f64 = basis
                .strings
                .iter()
                .zip(strings)
                .map(|(&(_, c), &k)| c * read[k])
                .sum();
            let square: f64 = basis.strings.iter().map(|&(_, c)| c * c).sum::<f64>()
                + basis
                    .products
                    .iter()
                    .zip(products)
                    .map(|(&(_, w), &k)| w * read[k])
                    .sum::<f64>();
            energy += part;
            variance += (square - part * part).max(0.) / shots as f64;
        }
        Ok((energy, self.unavailable.is_none().then(|| variance.sqrt())))
    }
}
