//! Reduced density matrices measured on a circuit device, without a state vector.
//!
//! The spin-summed spatial density matrices of a state on `m` spatial orbitals (`2m` modes in
//! spin-block order, mode `p + sigma m`) are expectation values of ladder products:
//!
//! * `gamma_pq = sum_sigma <a^dag_p,sigma a_q,sigma>`;
//! * `Gamma_pqrs = sum_{sigma tau} <a^dag_p,sigma a^dag_q,tau a_s,tau a_r,sigma>`, the index order
//!   of the model's two-electron tensor, so `E = c + sum h_pq gamma_pq + (1/2) sum g_pqrs Gamma_pqrs`.
//!
//! A circuit device measures Hermitian observables, so each element is read as the real part of
//! its expectation, `<(A + A^dag)/2>`: the Hermitian part of the mapped operator, whose Pauli
//! coefficients are the real parts of the mapped ones. The states the variational circuits
//! prepare from the real reference are real, and so are their density matrices. One element is
//! measured per orbit of the symmetries of a real `Gamma` (`pqrs = qpsr = rspq = srqp`, about
//! `m^4 / 4` elements) and of `gamma` (`p <= q`), each a sum of at most 64 Pauli strings: the
//! classical cost is polynomial in the qubit count, with no `2^n` object anywhere.
use super::{algebra::Term, properties::SpatialRdms, Encoding, C64};
use crate::{product, Context, Error, Reservation, Result};
use std::collections::HashMap;

/// Pauli coefficients at or below this are roundoff of the mapping and are not measured.
const ZERO: f64 = 1e-14;

struct Element {
    index: [usize; 4],
    constant: f64,
    terms: Vec<(u32, f64)>,
    /// `(a, b, k)`: terms `a < b` of this element read in one basis, whose covariance needs the
    /// product string `k` (measured in that basis on the same shots).
    pairs: Vec<(u32, u32, u32)>,
}

/// The Pauli strings that determine the spatial 1- and 2-RDMs, and how each element is assembled
/// from their measured expectation values.
pub struct RdmMeasurement {
    pub m: usize,
    pub n_qubits: usize,
    strings: Vec<(u64, u64)>,
    one: Vec<Element>,
    two: Vec<Element>,
    bases: Vec<Group>,
    /// `(basis, Z mask)` of each product string the in-basis covariances need, indexed after
    /// the strings in [`Self::measured`].
    products: Vec<(u32, u64)>,
    _memory: Reservation,
}

/// One measurement basis: its letter on each qubit (`x`, `z` masks as in a Pauli term) and the
/// strings it measures.
struct Group {
    x: u64,
    z: u64,
    members: Vec<u32>,
}

/// Open bases a bucket is tried against, newest first: bounds the grouping at
/// `O(buckets * WINDOW)` compatibility checks of a few machine words each.
const WINDOW: usize = 1024;

/// Whether two Pauli patterns agree on every qubit both act on.
fn compatible(a: (u64, u64), b: (u64, u64)) -> bool {
    let common = (a.0 | a.1) & (b.0 | b.1);
    ((a.0 ^ b.0) | (a.1 ^ b.1)) & common == 0
}

/// A qubit-wise commuting partition of `strings`, polynomial in their number. Strings with the
/// same `X`/`Y` letters on the same qubits are pairwise qubit-wise commuting (everywhere else
/// they hold `Z` or `I`), so they form one bucket first; the buckets, in a deterministic order,
/// then join the first compatible basis among the last [`WINDOW`] opened, or open one.
fn group(strings: &[(u64, u64)]) -> Vec<Group> {
    let mut buckets: HashMap<(u64, u64), (u64, Vec<u32>)> = HashMap::new();
    for (i, &(x, z)) in strings.iter().enumerate() {
        let entry = buckets.entry((x, x & z)).or_insert((0, Vec::new()));
        entry.0 |= z & !x;
        entry.1.push(i as u32);
    }
    let mut keys: Vec<_> = buckets.into_iter().collect();
    // Largest buckets first (they constrain the most qubits), then by pattern.
    keys.sort_unstable_by(|a, b| {
        (b.1 .1.len(), (b.1 .0 | b.0 .0).count_ones())
            .cmp(&(a.1 .1.len(), (a.1 .0 | a.0 .0).count_ones()))
            .then(a.0.cmp(&b.0))
    });
    let mut bases: Vec<Group> = Vec::new();
    for ((x, y), (zs, members)) in keys {
        let pattern = (x, y | zs);
        let start = bases.len().saturating_sub(WINDOW);
        match bases[start..]
            .iter_mut()
            .find(|b| compatible((b.x, b.z), pattern))
        {
            Some(basis) => {
                basis.x |= pattern.0;
                basis.z |= pattern.1;
                basis.members.extend(members);
            }
            None => bases.push(Group {
                x: pattern.0,
                z: pattern.1,
                members,
            }),
        }
    }
    bases
}

/// The representative of the orbit of `(p, q, r, s)` under the symmetries of a real `Gamma`.
fn canonical(index: [usize; 4]) -> [usize; 4] {
    let [p, q, r, s] = index;
    *[[p, q, r, s], [q, p, s, r], [r, s, p, q], [s, r, q, p]]
        .iter()
        .min()
        .unwrap()
}

impl RdmMeasurement {
    /// The measurement of the RDMs of `encoding`'s register. `limit` caps the distinct Pauli
    /// strings (`measurement_budget["observables"]`): the plan stops and refuses as soon as it
    /// would need more, before it is complete.
    pub fn new(ctx: &Context, encoding: &Encoding, limit: Option<usize>) -> Result<Self> {
        if encoding.modes % 2 != 0 {
            return Err(Error::Invalid(
                "measured RDMs need an even spin-blocked register".into(),
            ));
        }
        let m = encoding.modes / 2;
        // The elements (at most m^4 / 4 + m^2) with their terms, and the string index.
        let memory = ctx.reserve(product(&[m.max(1), m.max(1), m.max(1), m.max(1), 160])?)?;
        let mut index: HashMap<(u64, u64), u32> = HashMap::new();
        let mut strings = Vec::new();
        let mut element =
            |ladders: &[Vec<(usize, bool)>], at: [usize; 4]| -> Result<Option<Element>> {
                let mut merged: HashMap<(u64, u64), f64> = HashMap::new();
                for product in ladders {
                    for t in &encoding.monomial(ctx, product)?.terms {
                        *merged.entry((t.x, t.z)).or_default() += t.coefficient.re;
                    }
                }
                let mut constant = 0.;
                let mut terms = Vec::new();
                let mut keys: Vec<_> = merged.into_iter().filter(|(_, c)| c.abs() > ZERO).collect();
                keys.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                for (key, c) in keys {
                    if key == (0, 0) {
                        constant += c;
                        continue;
                    }
                    let next = strings.len() as u32;
                    let i = *index.entry(key).or_insert_with(|| {
                        strings.push(key);
                        next
                    });
                    terms.push((i, c));
                }
                if let Some(limit) = limit {
                    if strings.len() > limit {
                        return Err(Error::Invalid(format!(
                            "measuring the reduced density matrices needs more than {limit} Pauli \
                         observables ({} distinct strings before the plan was complete): \
                         measurement observables budget exceeded; pass \
                         measurement_budget={{\"observables\": N}} sized for this run",
                            strings.len()
                        )));
                    }
                }
                Ok((constant != 0. || !terms.is_empty()).then_some(Element {
                    index: at,
                    constant,
                    terms,
                    pairs: Vec::new(),
                }))
            };
        let mut one = Vec::new();
        for p in 0..m {
            for q in p..m {
                let ladders: Vec<_> = (0..2)
                    .map(|s| vec![(p + s * m, true), (q + s * m, false)])
                    .collect();
                one.extend(element(&ladders, [p, q, 0, 0])?);
            }
        }
        let mut two = Vec::new();
        for p in 0..m {
            for q in 0..m {
                for r in 0..m {
                    for s in 0..m {
                        if canonical([p, q, r, s]) != [p, q, r, s] {
                            continue;
                        }
                        let mut ladders = Vec::with_capacity(4);
                        for sigma in 0..2 {
                            for tau in 0..2 {
                                ladders.push(vec![
                                    (p + sigma * m, true),
                                    (q + tau * m, true),
                                    (s + tau * m, false),
                                    (r + sigma * m, false),
                                ]);
                            }
                        }
                        two.extend(element(&ladders, [p, q, r, s])?);
                    }
                }
            }
        }
        let bases = group(&strings);
        // The basis of each string, and the product strings of every pair of an element's terms
        // that share one: in a basis rotated to Z, the product of two Z strings is the Z string
        // of the XOR of their masks, read on the same shots.
        let mut basis_of = vec![0u32; strings.len()];
        for (b, basis) in bases.iter().enumerate() {
            for &i in &basis.members {
                basis_of[i as usize] = b as u32;
            }
        }
        let image = |i: u32| {
            let (x, z) = strings[i as usize];
            x | z
        };
        let mut products: Vec<(u32, u64)> = Vec::new();
        let mut product_index: HashMap<(u32, u64), u32> = HashMap::new();
        for e in one.iter_mut().chain(two.iter_mut()) {
            for a in 0..e.terms.len() {
                for b in a + 1..e.terms.len() {
                    let (i, j) = (e.terms[a].0, e.terms[b].0);
                    let basis = basis_of[i as usize];
                    if basis != basis_of[j as usize] {
                        continue;
                    }
                    let key = (basis, image(i) ^ image(j));
                    let next = products.len() as u32;
                    let k = *product_index.entry(key).or_insert_with(|| {
                        products.push(key);
                        next
                    });
                    e.pairs.push((a as u32, b as u32, k));
                }
            }
        }
        Ok(Self {
            m,
            n_qubits: encoding.n_qubits(),
            strings,
            one,
            two,
            bases,
            products,
            _memory: memory,
        })
    }

    /// The number of values [`Self::assemble`] takes: the strings of [`Self::observables`],
    /// then the product strings the covariances need, in the indices [`Self::requests`] gives.
    pub fn measured(&self) -> usize {
        self.strings.len() + self.products.len()
    }

    /// The number of measurement bases of [`Self::requests`].
    pub fn bases(&self) -> usize {
        self.bases.len()
    }

    /// One request per measurement basis: the rotations to append after the ansatz, `(label,
    /// angle)` meaning `exp(i angle P)` as the executor takes them, and the `Z`-only strings to
    /// read after them, with the index of the value each one gives: a string of
    /// [`Self::observables`], or (from `observables().len()` on) a product string whose value,
    /// read on the same shots, gives the covariance of two strings of the basis.
    /// `exp(i pi/4 Y)` turns `X` into `Z` and `exp(-i pi/4 X)` turns `Y` into `Z`, both with a
    /// plus sign, so the value of a rotated string is the value of the string it stands for.
    #[allow(clippy::type_complexity)]
    pub fn requests(&self) -> Vec<(Vec<(String, f64)>, Vec<(String, u32)>)> {
        let n = self.n_qubits;
        let label = |x: u64, z: u64| {
            Term {
                x,
                z,
                coefficient: C64::new(1., 0.),
            }
            .label(n)
        };
        let mut extra: Vec<Vec<(u64, u32)>> = vec![Vec::new(); self.bases.len()];
        for (k, &(basis, mask)) in self.products.iter().enumerate() {
            extra[basis as usize].push((mask, (self.strings.len() + k) as u32));
        }
        self.bases
            .iter()
            .zip(extra)
            .map(|(basis, extra)| {
                let mut tail = Vec::new();
                for j in 0..n {
                    let bit = 1u64 << (n - 1 - j);
                    match (basis.x & bit != 0, basis.z & bit != 0) {
                        (true, false) => tail.push((label(bit, bit), std::f64::consts::FRAC_PI_4)),
                        (true, true) => tail.push((label(bit, 0), -std::f64::consts::FRAC_PI_4)),
                        _ => {}
                    }
                }
                let observables = basis
                    .members
                    .iter()
                    .map(|&i| {
                        let (x, z) = self.strings[i as usize];
                        (label(0, x | z), i)
                    })
                    .chain(extra.into_iter().map(|(mask, at)| (label(0, mask), at)))
                    .collect();
                (tail, observables)
            })
            .collect()
    }

    /// The measurement of `problem`'s RDMs on its register. A tapered problem is refused: its
    /// circuits live on the tapered register, and the RDM strings would need mapping there.
    pub fn of_problem(problem: &super::CachedProblem, limit: Option<usize>) -> Result<Self> {
        if problem.taper.is_some() {
            return Err(Error::Unsupported(
                "measured reduced density matrices of a tapered problem are not implemented: \
                 run with taper=False"
                    .into(),
            ));
        }
        Self::new(&problem.context, &problem.encoding, limit)
    }

    /// The Pauli strings to measure, qubit zero first; [`Self::assemble`] takes their values in
    /// this order.
    pub fn observables(&self) -> Vec<String> {
        self.strings
            .iter()
            .map(|&(x, z)| {
                Term {
                    x,
                    z,
                    coefficient: C64::new(1., 0.),
                }
                .label(self.n_qubits)
            })
            .collect()
    }

    /// The RDMs from the `values` of the [`Self::measured`] strings (in request index order),
    /// and, from their standard errors `errors` when given, the standard error of each element.
    /// Strings of one basis share its `shots`, so an element's variance is
    /// `sum c_i^2 s_i^2 + 2 sum c_i c_j (<P_i P_j> - <P_i><P_j>) / shots` over its pairs in one
    /// basis, `<P_i P_j>` the product string's value; without `shots` the strings are taken as
    /// independent.
    pub fn assemble(
        &self,
        ctx: &Context,
        values: &[f64],
        errors: Option<&[f64]>,
        shots: Option<u64>,
    ) -> Result<(SpatialRdms, Option<(Vec<f64>, Vec<f64>)>)> {
        let count = self.measured();
        if values.len() != count
            || errors.is_some_and(|e| e.len() != count)
            || shots == Some(0)
            || values
                .iter()
                .chain(errors.unwrap_or(&[]))
                .any(|v| !v.is_finite())
        {
            return Err(Error::Invalid(format!(
                "the reduced density matrices need {count} finite measured values (and errors)"
            )));
        }
        let m = self.m;
        let mut one = ctx.allocate::<C64>(product(&[m, m])?)?;
        let mut two = ctx.allocate::<C64>(product(&[m, m, m, m])?)?;
        let mut one_error = errors.map(|_| vec![0.; m * m]);
        let mut two_error = errors.map(|_| vec![0.; m * m * m * m]);
        let read = |e: &Element| {
            let value = e.constant
                + e.terms
                    .iter()
                    .map(|&(i, c)| c * values[i as usize])
                    .sum::<f64>();
            let error = errors.map(|s| {
                let diagonal: f64 = e
                    .terms
                    .iter()
                    .map(|&(i, c)| (c * s[i as usize]).powi(2))
                    .sum();
                let covariance: f64 = shots.map_or(0., |n| {
                    e.pairs
                        .iter()
                        .map(|&(a, b, k)| {
                            let (i, ci) = e.terms[a as usize];
                            let (j, cj) = e.terms[b as usize];
                            let product = values[self.strings.len() + k as usize];
                            2. * ci * cj * (product - values[i as usize] * values[j as usize])
                                / n as f64
                        })
                        .sum()
                });
                (diagonal + covariance).max(0.).sqrt()
            });
            (value, error)
        };
        for e in &self.one {
            let [p, q, ..] = e.index;
            let (value, error) = read(e);
            for (a, b) in [(p, q), (q, p)] {
                one.as_mut_slice()[a * m + b] = C64::new(value, 0.);
                if let (Some(out), Some(err)) = (one_error.as_mut(), error) {
                    out[a * m + b] = err;
                }
            }
        }
        for e in &self.two {
            let [p, q, r, s] = e.index;
            let (value, error) = read(e);
            for [a, b, c, d] in [[p, q, r, s], [q, p, s, r], [r, s, p, q], [s, r, q, p]] {
                let at = ((a * m + b) * m + c) * m + d;
                two.as_mut_slice()[at] = C64::new(value, 0.);
                if let (Some(out), Some(err)) = (two_error.as_mut(), error) {
                    out[at] = err;
                }
            }
        }
        Ok((SpatialRdms { m, one, two }, one_error.zip(two_error)))
    }
}
