//! One-particle spectral function of a correlated state, by the Lehmann representation.
//!
//! For a normalized state `|psi>` of energy `E0 = <psi|H|psi>` and a creation operator
//! `c+ = sum_p w_p a+_{p sigma}`,
//!
//! `A(omega) = sum_n |<n, N+1| c+ |psi>|^2 delta(omega - (E_n - E0))
//!           + sum_n |<n, N-1| c |psi>|^2 delta(omega - (E0 - E_n))`,
//!
//! the addition and removal parts of `-Im G(omega) / pi`. The `N +- 1` eigenstates are those of
//! the same second-quantized Hamiltonian in the neighboring particle sectors; each part is
//! resolved by a Lanczos recursion started from `c+|psi>` (or `c|psi>`), with full
//! reorthogonalization. The poles are the Ritz values and the weights `|c+psi|^2 U_0j^2`: the
//! Gauss quadrature of the spectral measure, which reproduces its first `2K` moments for `K`
//! Krylov vectors and is the exact Lehmann sum once `K` reaches the sector dimension (the
//! recursion stops earlier only when the Krylov space closes, which is also exact). The total
//! weight is `<psi| {c, c+} |psi> = |w|^2` whatever `K` is: the sum rule.
//!
//! When `|psi>` is a variational state rather than an eigenstate, `E0` is its energy and the
//! result is the spectral function of that state, not of the exact ground state. Everything here
//! holds sector-sized vectors: it is an exact-state analysis, never part of a shot-based run.
use super::{sparse::SparseMatrix, Basis, Encoding, C64};
use crate::molecular::transform::Model;
use crate::{product, Context, Error, Result};
use nalgebra::{DMatrix, SymmetricEigen};

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
fn bit(n: usize, j: usize) -> u64 {
    1 << (n - 1 - j)
}
/// The poles (Hartree, excitation energies relative to `E0`, so removal poles are `E0 - E_n`)
/// and weights of one part of one spectral function.
#[derive(Clone, Debug, Default)]
pub struct Poles {
    pub energies: Vec<f64>,
    pub weights: Vec<f64>,
}
impl Poles {
    pub fn weight(&self) -> f64 {
        self.weights.iter().sum()
    }
}
/// The spectral functions of a set of creation operators, per spin.
#[derive(Clone, Debug)]
pub struct Spectral {
    /// `<psi|H|psi>`, including the model constant.
    pub ground_energy: f64,
    /// `addition[o][s]`: operator `o`, spin `s`.
    pub addition: Vec<[Poles; 2]>,
    pub removal: Vec<[Poles; 2]>,
}
/// A state in occupation (Jordan-Wigner) form: `(occupations, amplitude)`, mode `j` (spin-orbital
/// `p + sigma m`) at bit `2m - 1 - j`, the convention of [`Encoding`].
pub type OccupationState = [(u64, C64)];

/// Apply `sum_p w_p a+_{p sigma}` (`dagger`) or `sum_p conj(w_p) a_{p sigma}` to `state`,
/// into the sector `basis`.
fn ladder(
    m: usize,
    sigma: usize,
    w: &[C64],
    dagger: bool,
    state: &OccupationState,
    basis: &Basis,
    out: &mut [C64],
) -> Result<()> {
    let n = 2 * m;
    out.fill(C64::default());
    for &(ket, amplitude) in state {
        for (p, &coefficient) in w.iter().enumerate() {
            if coefficient == C64::default() {
                continue;
            }
            let j = p + sigma * m;
            let mask = bit(n, j);
            if (ket & mask != 0) == dagger {
                continue;
            }
            // Occupied modes before j sit at the bits above `mask`.
            let before = !(mask | (mask - 1)) & ((1u64 << n) - 1);
            let sign = if (ket & before).count_ones() % 2 == 0 {
                1.
            } else {
                -1.
            };
            let target = ket ^ mask;
            let row = basis
                .index(target)
                .ok_or_else(|| invalid("a ladder image left its particle sector"))?;
            let factor = if dagger {
                coefficient
            } else {
                coefficient.conj()
            };
            out[row] += factor * amplitude * sign;
        }
    }
    Ok(())
}

/// Lanczos recursion with full reorthogonalization from `start` (not normalized): returns the
/// Ritz values and weights `|start|^2 U_0j^2`.
fn lanczos(ctx: &Context, h: &SparseMatrix, start: &[C64], krylov: usize) -> Result<Poles> {
    let d = start.len();
    let norm2: f64 = start.iter().map(|v| v.norm_sqr()).sum();
    if norm2 <= 1e-28 {
        return Ok(Poles::default());
    }
    let steps = krylov.min(d).max(1);
    let _memory = ctx.reserve(product(&[steps + 2, d, 16])?)?;
    let mut basis: Vec<Vec<C64>> = Vec::with_capacity(steps);
    let scale = 1. / norm2.sqrt();
    basis.push(start.iter().map(|v| v * scale).collect());
    let mut alpha = Vec::with_capacity(steps);
    let mut beta: Vec<f64> = Vec::with_capacity(steps);
    let mut w = vec![C64::default(); d];
    for i in 0..steps {
        h.apply_into(&basis[i], &mut w);
        let a: f64 = basis[i]
            .iter()
            .zip(&w)
            .map(|(v, x)| (v.conj() * x).re)
            .sum();
        alpha.push(a);
        if i + 1 == steps {
            break;
        }
        // Full reorthogonalization, twice (classical Gram-Schmidt is stable when repeated).
        for _ in 0..2 {
            for v in &basis {
                let overlap: C64 = v.iter().zip(&w).map(|(v, x)| v.conj() * x).sum();
                for (x, v) in w.iter_mut().zip(v) {
                    *x -= overlap * v;
                }
            }
        }
        let b = w.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        let spread = alpha.iter().map(|v| v.abs()).fold(1f64, f64::max);
        if b <= 1e-12 * spread {
            break;
        }
        beta.push(b);
        basis.push(w.iter().map(|v| v / b).collect());
    }
    let k = alpha.len();
    let t = DMatrix::from_fn(k, k, |i, j| {
        if i == j {
            alpha[i]
        } else if i + 1 == j {
            beta[i]
        } else if j + 1 == i {
            beta[j]
        } else {
            0.
        }
    });
    let eigen = SymmetricEigen::new(t);
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
    Ok(Poles {
        energies: order.iter().map(|&j| eigen.eigenvalues[j]).collect(),
        weights: order
            .iter()
            .map(|&j| norm2 * eigen.eigenvectors[(0, j)].powi(2))
            .collect(),
    })
}

/// The Lehmann spectral functions of `operators` (each `m` coefficients `w_p` over the model's
/// spatial orbitals) in the state `state` of the restricted, collinear `model`, which holds
/// `particles` electrons. `krylov` bounds the Lanczos vectors per part.
pub fn lehmann(
    ctx: &Context,
    model: &Model,
    particles: [usize; 2],
    state: &OccupationState,
    operators: &[Vec<C64>],
    krylov: usize,
) -> Result<Spectral> {
    if model.spinor {
        return Err(invalid(
            "the spectral function needs a collinear (spin-blocked) model",
        ));
    }
    let m = model.m;
    let n = 2 * m;
    if krylov == 0 {
        return Err(invalid("the Krylov dimension must be positive"));
    }
    if operators
        .iter()
        .any(|w| w.len() != m || w.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()))
    {
        return Err(invalid(
            "every operator needs one finite coefficient per spatial orbital",
        ));
    }
    let hamiltonian = model.operator(ctx, "jordan_wigner")?;
    let sector = |p: [usize; 2]| -> Result<(Basis, SparseMatrix)> {
        let encoding = Encoding::new("jordan_wigner", n, p)?;
        let basis = Basis::new(ctx, &encoding, true)?;
        let matrix = SparseMatrix::new(ctx, &basis, &hamiltonian)?;
        Ok((basis, matrix))
    };
    // The state on its own sector, normalized, and its energy.
    let (basis, h) = sector(particles)?;
    let mut psi = ctx.allocate::<C64>(basis.len())?;
    for &(ket, amplitude) in state {
        if !amplitude.re.is_finite() || !amplitude.im.is_finite() {
            return Err(invalid("the state must be finite"));
        }
        let row = basis
            .index(ket)
            .ok_or_else(|| invalid("the state leaves its particle sector"))?;
        psi.as_mut_slice()[row] += amplitude;
    }
    let norm2: f64 = psi.as_slice().iter().map(|v| v.norm_sqr()).sum();
    if norm2 <= 1e-24 {
        return Err(invalid("the state is zero"));
    }
    let scale = 1. / norm2.sqrt();
    let normalized: Vec<(u64, C64)> = basis
        .indices
        .as_slice()
        .iter()
        .zip(psi.as_slice())
        .filter(|(_, v)| v.norm_sqr() > 0.)
        .map(|(&ket, v)| (ket, v * scale))
        .collect();
    let mut work = vec![C64::default(); basis.len()];
    let dense: Vec<C64> = psi.as_slice().iter().map(|v| v * scale).collect();
    h.apply_into(&dense, &mut work);
    let ground_energy: f64 = dense
        .iter()
        .zip(&work)
        .map(|(v, x)| (v.conj() * x).re)
        .sum();
    drop((psi, work, dense, h, basis));
    let mut addition = vec![[Poles::default(), Poles::default()]; operators.len()];
    let mut removal = vec![[Poles::default(), Poles::default()]; operators.len()];
    for sigma in 0..2 {
        for dagger in [true, false] {
            let mut target = particles;
            if dagger {
                if target[sigma] == m {
                    continue;
                }
                target[sigma] += 1;
            } else {
                if target[sigma] == 0 {
                    continue;
                }
                target[sigma] -= 1;
            }
            let (basis, h) = sector(target)?;
            let mut start = vec![C64::default(); basis.len()];
            for (o, w) in operators.iter().enumerate() {
                ladder(m, sigma, w, dagger, &normalized, &basis, &mut start)?;
                let mut poles = lanczos(ctx, &h, &start, krylov)?;
                for e in &mut poles.energies {
                    *e = if dagger {
                        *e - ground_energy
                    } else {
                        ground_energy - *e
                    };
                }
                if !dagger {
                    poles.energies.reverse();
                    poles.weights.reverse();
                }
                if dagger {
                    addition[o][sigma] = poles;
                } else {
                    removal[o][sigma] = poles;
                }
            }
        }
    }
    Ok(Spectral {
        ground_energy,
        addition,
        removal,
    })
}

/// `sum_j w_j (eta / pi) / ((omega - e_j)^2 + eta^2)` on `axis`: the Lorentzian broadening of a
/// set of poles, the spectral function at a finite distance `eta` above the real axis.
pub fn lorentzian(poles: &[&Poles], axis: &[f64], eta: f64) -> Result<Vec<f64>> {
    if !eta.is_finite() || eta <= 0. {
        return Err(invalid("the broadening must be positive and finite"));
    }
    crate::finite(axis, "energy axis")?;
    Ok(axis
        .iter()
        .map(|&omega| {
            poles
                .iter()
                .flat_map(|p| p.energies.iter().zip(&p.weights))
                .map(|(e, w)| w * eta / std::f64::consts::PI / ((omega - e).powi(2) + eta * eta))
                .sum()
        })
        .collect())
}
