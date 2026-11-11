//! The many-body problem of a fragment of Wannier functions: the Kohn-Sham one-body block, the
//! Kohn-Sham density matrix, the double counting and the two-body integrals, bare or screened,
//! and the same model in the natural orbitals of the density matrix.
//!
//! Notation (Hartree). `h` is the Kohn-Sham Hamiltonian between the fragment's functions,
//! `gamma_pq = <c_p^+ c_q> = <w_q|D|w_p>` the one-particle density matrix (a restricted crystal's
//! one holds both spins), `(pq|rs)` the Coulomb integrals of `rho_pq = w_p^* w_q` (chemists')
//! and `<pq|rs> = (pr|qs)` the same in physicists' order. The double counting is the Hartree and
//! exchange potential of the fragment's own electrons at the Kohn-Sham density,
//!
//! ```text
//! V^{dc,s}_pq = sum_t sum_rs gamma^t_rs (pq|rs)^{st}  -  sum_rs gamma^s_rs (ps|rq)^{ss}
//! ```
//!
//! (a restricted crystal: `sum_rs gamma_rs [(pq|rs) - 1/2 (ps|rq)]`), so that the model's one-body
//! part `h - V^dc` has the Fock matrix of `gamma` equal to `h`.
use super::pairs::Integrals;
use crate::{math::linalg::eigen, periodic::invalid, quantum::C64, Result};
use nalgebra::DMatrix;

pub type Matrix = DMatrix<C64>;

/// Two occupations closer than this share a natural orbital block.
pub const NATURAL_DEGENERACY: f64 = 1e-6;

/// A model's tensors in one orbital basis.
#[derive(Clone)]
pub struct Basis {
    /// `h - V^dc` per spin channel.
    pub one_body: Vec<Matrix>,
    /// `<pq|rs>` for `(s, t)` flattened as `s * spins + t`; each `n^4` in `[((p*n+q)*n+r)*n+s]`.
    pub two_body: Vec<Vec<C64>>,
}

pub struct Problem {
    /// The fragment's modes `(function, R)`.
    pub modes: Vec<(usize, [i64; 3])>,
    pub spins: usize,
    pub n: usize,
    pub kohn_sham: Vec<Matrix>,
    pub density_matrix: Vec<Matrix>,
    pub double_counting: Vec<Matrix>,
    pub wannier: Basis,
    pub natural: Basis,
    /// Occupations of the natural orbitals, most occupied first, per spin channel, and the
    /// orbitals as columns in the fragment's functions.
    pub occupations: Vec<Vec<f64>>,
    pub rotation: Vec<Matrix>,
    pub num_particles: [usize; 2],
    pub dielectric_head: Option<f64>,
}

impl Problem {
    /// The model in the Wannier functions (`"wannier"`) or in the natural orbitals of the density
    /// matrix (`"natural"`).
    pub fn basis(&self, name: &str) -> Result<&Basis> {
        match name {
            "wannier" => Ok(&self.wannier),
            "natural" => Ok(&self.natural),
            other => Err(invalid(format!(
                "basis must be 'wannier' or 'natural'; got {other:?}"
            ))),
        }
    }
}

/// Chemists' `(pq|rs)` to physicists' `<pq|rs> = (pr|qs)`.
fn physicists(chemist: &Integrals) -> Result<Vec<C64>> {
    let n = chemist.first;
    if chemist.second != n {
        return Err(invalid(
            "a fragment's integrals are square in its functions",
        ));
    }
    let mut out = vec![C64::default(); n * n * n * n];
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    out[((p * n + q) * n + r) * n + s] =
                        chemist.values[((p * n + r) * n + q) * n + s];
                }
            }
        }
    }
    Ok(out)
}

/// `tr(gamma)` of every channel must be within this of an integer for the electron count to be
/// taken from it.
const INTEGER_COUNT: f64 = 0.1;

/// Builds the problem from the chemists' integrals `chemist[s][t]` and the per-channel blocks.
pub fn assemble(
    modes: Vec<(usize, [i64; 3])>,
    chemist: &[Vec<Integrals>],
    kohn_sham: Vec<Matrix>,
    gamma: Vec<Matrix>,
    num_particles: Option<[usize; 2]>,
    head: Option<f64>,
) -> Result<Problem> {
    let spins = kohn_sham.len();
    let n = modes.len();
    if spins == 0
        || spins > 2
        || gamma.len() != spins
        || chemist.len() != spins
        || chemist.iter().any(|row| row.len() != spins)
        || kohn_sham.iter().chain(&gamma).any(|m| m.shape() != (n, n))
        || chemist
            .iter()
            .flatten()
            .any(|c| c.first != n || c.second != n)
    {
        return Err(invalid(
            "the blocks of a downfolded problem disagree in shape",
        ));
    }
    // V^dc per spin channel.
    let mut double = Vec::with_capacity(spins);
    for s in 0..spins {
        let mut d = Matrix::zeros(n, n);
        for t in 0..spins {
            let c = &chemist[s][t];
            for p in 0..n {
                for q in 0..n {
                    let mut hartree = C64::default();
                    for r in 0..n {
                        for u in 0..n {
                            hartree += gamma[t][(r, u)] * c.values[((p * n + q) * n + r) * n + u];
                        }
                    }
                    d[(p, q)] += hartree;
                }
            }
        }
        let c = &chemist[s][s];
        // restricted: half the exchange of the total density; polarized: the channel's own
        let weight = if spins == 1 { 0.5 } else { 1. };
        for p in 0..n {
            for q in 0..n {
                let mut exchange = C64::default();
                for r in 0..n {
                    for u in 0..n {
                        exchange += gamma[s][(r, u)] * c.values[((p * n + u) * n + r) * n + q];
                    }
                }
                d[(p, q)] -= exchange * weight;
            }
        }
        double.push((&d + d.adjoint()) * C64::new(0.5, 0.));
    }
    let traces: Vec<f64> = gamma.iter().map(|g| g.trace().re).collect();
    let particles = match num_particles {
        Some(counts) => counts,
        None => {
            let counts: Vec<usize> = traces.iter().map(|t| t.round().max(0.) as usize).collect();
            if traces
                .iter()
                .zip(&counts)
                .any(|(t, c)| (t - *c as f64).abs() > INTEGER_COUNT)
            {
                return Err(invalid(format!(
                    "the fragment holds {:.3} Kohn-Sham electrons ({} per channel), not near an \
                     integer: give num_particles",
                    traces.iter().sum::<f64>(),
                    traces
                        .iter()
                        .map(|t| format!("{t:.3}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
            if spins == 1 {
                [counts[0] - counts[0] / 2, counts[0] / 2]
            } else {
                [counts[0], counts[1]]
            }
        }
    };
    let two_body = chemist
        .iter()
        .flatten()
        .map(physicists)
        .collect::<Result<Vec<_>>>()?;
    let wannier = Basis {
        one_body: (0..spins).map(|s| &kohn_sham[s] - &double[s]).collect(),
        two_body,
    };
    let mut occupations = vec![];
    let mut rotation = vec![];
    for s in 0..spins {
        let (values, vectors) = natural_orbitals(&gamma[s], &kohn_sham[s])?;
        occupations.push(values);
        rotation.push(vectors);
    }
    let natural = rotated(&wannier, &rotation, n, spins);
    Ok(Problem {
        modes,
        spins,
        n,
        kohn_sham,
        density_matrix: gamma,
        double_counting: double,
        wannier,
        natural,
        occupations,
        rotation,
        num_particles: particles,
        dielectric_head: head,
    })
}

/// The column of each vector times the phase that makes its largest entry real and positive: an
/// eigenvector's arbitrary phase fixed, so the eigenvectors of a real matrix stay real.
fn real_columns(vectors: &mut Matrix) {
    for j in 0..vectors.ncols() {
        let mut at = 0;
        for i in 0..vectors.nrows() {
            if vectors[(i, j)].norm() > vectors[(at, j)].norm() + 1e-12 {
                at = i;
            }
        }
        let entry = vectors[(at, j)];
        if entry.norm() > 0. {
            let phase = C64::new(entry.norm(), 0.) / entry;
            for i in 0..vectors.nrows() {
                vectors[(i, j)] *= phase;
            }
        }
    }
}

/// `(occupations, C)`: the eigenvectors of `gamma` (columns, in the fragment's functions), most
/// occupied first; within a block of equal occupations (a full or an empty channel, say) the
/// eigenvectors of `h`, lowest first.
pub fn natural_orbitals(gamma: &Matrix, h: &Matrix) -> Result<(Vec<f64>, Matrix)> {
    let n = gamma.nrows();
    let (mut values, vectors) = eigen((gamma + gamma.adjoint()) * C64::new(0.5, 0.))?;
    values.reverse();
    let mut vectors = Matrix::from_fn(n, n, |i, j| vectors[(i, n - 1 - j)]);
    real_columns(&mut vectors);
    let mut start = 0;
    while start < n {
        let mut stop = start + 1;
        while stop < n && (values[stop] - values[start]).abs() < NATURAL_DEGENERACY {
            stop += 1;
        }
        if stop - start > 1 {
            let block = vectors.columns(start, stop - start).into_owned();
            let projected = block.adjoint() * h * &block;
            let (_, rotation) = eigen((&projected + projected.adjoint()) * C64::new(0.5, 0.))?;
            let mut mixed = &block * rotation;
            real_columns(&mut mixed);
            for j in 0..stop - start {
                vectors.set_column(start + j, &mixed.column(j));
            }
        }
        start = stop;
    }
    Ok((values, vectors))
}

/// The model in the orbitals `c[s]` (columns): `h' = C^+ h C` and
/// `<pq|rs>' = sum conj(C_ap) conj(C_bq) C_cr C_ds <ab|cd>` with the electron-1 orbitals of spin
/// `s` and the electron-2 orbitals of spin `t`.
pub fn rotated(basis: &Basis, c: &[Matrix], n: usize, spins: usize) -> Basis {
    let one_body = (0..spins)
        .map(|s| c[s].adjoint() * &basis.one_body[s] * &c[s])
        .collect();
    let mut two_body = Vec::with_capacity(spins * spins);
    for s in 0..spins {
        for t in 0..spins {
            let source = &basis.two_body[s * spins + t];
            let (cs, ct) = (&c[s], &c[t]);
            // contract one index at a time: a, b, c, d
            let mut x = source.clone();
            x = transform(&x, n, 0, |a, p| cs[(a, p)].conj());
            x = transform(&x, n, 1, |a, p| ct[(a, p)].conj());
            x = transform(&x, n, 2, |a, p| cs[(a, p)]);
            x = transform(&x, n, 3, |a, p| ct[(a, p)]);
            two_body.push(x);
        }
    }
    Basis { one_body, two_body }
}

/// Replaces index `axis` of an `n^4` tensor by `sum_a t[..a..] u(a, p)`.
fn transform(t: &[C64], n: usize, axis: usize, u: impl Fn(usize, usize) -> C64) -> Vec<C64> {
    let stride = n.pow(3 - axis as u32);
    let mut out = vec![C64::default(); t.len()];
    for (index, slot) in out.iter_mut().enumerate() {
        let p = (index / stride) % n;
        let base = index - p * stride;
        let mut sum = C64::default();
        for a in 0..n {
            sum += t[base + a * stride] * u(a, p);
        }
        *slot = sum;
    }
    out
}
