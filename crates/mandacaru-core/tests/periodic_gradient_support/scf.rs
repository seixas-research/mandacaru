//! The self-consistent toy: for each spin and k, `H c = e S c` with the occupations of the model,
//! and the fixed point of the damped iteration `x -> f(x)` for `x = (n_shell, r, q)`.
use super::{c, d_eps, eps, shell_weights, shells, Prepared, ORBITALS};
use crate::hubbard_gradient_support::{Matrix, NP};
use mandacaru_core::{
    math::linalg::eigen,
    periodic::{
        gradient::{Bands, State},
        hubbard_gradient::operator,
    },
    quantum::C64,
};
use nalgebra::linalg::Cholesky;

pub const TOLERANCE: f64 = 1e-14;
const MAXIMUM_ITERATIONS: usize = 20000;
const MINIMUM_GAP: f64 = 0.05;

/// The state variables the iteration mixes.
#[derive(Clone)]
pub struct Mix {
    /// `n[spin][shell]`.
    pub n: Vec<Vec<Matrix>>,
    pub r: f64,
    pub q: Vec<f64>,
}

/// The potentials of a density: `v` (on `L`), `vxc` (for the core), the channel moments.
pub struct Fields {
    pub v: f64,
    pub vxc: f64,
    pub r: f64,
    pub moments: Vec<C64>,
}

pub struct Solution {
    /// What the last diagonalization was built from.
    pub input: Mix,
    /// What that diagonalization produced.
    pub output: Mix,
    pub state: State,
    pub energy: f64,
}

pub fn hermitize(m: &Matrix) -> Matrix {
    (m + m.adjoint()) * c(0.5)
}

pub fn fields(prep: &Prepared, r: f64, q: &[f64]) -> Fields {
    let x = r + prep.core;
    let hat: f64 = q.iter().zip(&prep.hat).map(|(q, h)| q * h).sum();
    let moments = (0..q.len())
        .map(|a| {
            let uq: f64 = (0..q.len()).map(|b| prep.u[(a, b)] * q[b]).sum();
            c(r * prep.hat[a] + uq + prep.uion[a] + prep.linear[a])
        })
        .collect();
    Fields {
        v: d_eps(x) + hat + prep.ion,
        vxc: d_eps(x),
        r,
        moments,
    }
}

/// The +U energy and the per-spin operator `O_s` of the occupations `n`.
pub fn hubbard(prep: &Prepared, n: &[Vec<Matrix>]) -> (f64, Vec<Matrix>) {
    let model = &prep.model;
    let p = model.toy.projectors();
    let shell_list = shells(&model.toy);
    let evaluations: Vec<_> = (0..shell_list.len())
        .map(|i| {
            let down = n.get(1).unwrap_or(&n[0]);
            model.correction.evaluate([&n[0][i], &down[i]])
        })
        .collect();
    let energy = evaluations.iter().map(|e| e.energy).sum();
    let operators = (0..model.spins())
        .map(|s| {
            let potentials: Vec<_> = evaluations.iter().map(|e| e.potential[s].clone()).collect();
            operator(p, &shell_list, &potentials).unwrap()
        })
        .collect();
    (energy, operators)
}

/// `generalized` eigenpairs of `H c = e S c`, the lowest `count`.
fn generalized(h: &Matrix, s: &Matrix, count: usize) -> (Vec<f64>, Matrix) {
    let l = Cholesky::new(s.clone())
        .expect("the toy overlap is positive definite")
        .l();
    let a = l.solve_lower_triangular(h).unwrap();
    let reduced = l.solve_lower_triangular(&a.adjoint()).unwrap().adjoint();
    let (e, y) = eigen(hermitize(&reduced)).unwrap();
    assert!(
        e[count] - e[count - 1] > MINIMUM_GAP,
        "the toy gap closed: {e:?}"
    );
    let vectors = l.adjoint().solve_upper_triangular(&y).unwrap();
    (e[..count].to_vec(), vectors.columns(0, count).into_owned())
}

/// `T + SR + onsite + C D_ion C^+`: the part of `H` no density enters.
pub fn fixed(prep: &Prepared, spin: usize, k: usize) -> Matrix {
    let sign = prep.model.signs[spin];
    let c_k = &prep.projection[k];
    &prep.hopping[k]
        + &prep.short[k]
        + prep.model.toy.onsite(sign)
        + c_k * &prep.d_ion * c_k.adjoint()
}

pub fn bands(
    prep: &Prepared,
    spin: usize,
    k: usize,
    fields: &Fields,
    hubbard: Option<&Matrix>,
) -> Bands {
    let c_k = &prep.projection[k];
    let mut channel = Matrix::zeros(prep.q_matrix.nrows(), prep.q_matrix.ncols());
    for (block, m) in prep.blocks.iter().zip(&fields.moments) {
        channel += block * *m;
    }
    if let Some(o) = hubbard {
        channel += o;
    }
    let h = fixed(prep, spin, k) + &prep.local[k] * c(fields.v) + c_k * channel * c_k.adjoint();
    let count = prep.model.occupations.len();
    let (e, vectors) = generalized(&hermitize(&h), &prep.overlap[k], count);
    Bands {
        coefficients: vectors,
        eigenvalues: e.iter().map(|x| x + prep.model.reference).collect(),
        occupations: prep.model.occupations.clone(),
    }
}

/// `x` of a state: `n_shell = W^T R W`, `r`, `q`.
pub fn observe(prep: &Prepared, state: &State) -> Mix {
    let model = &prep.model;
    let (p, h) = (model.toy.projectors(), model.toy.shell_atoms.len());
    let weights = shell_weights().map(c);
    let (mut r, mut q) = (0.0, vec![0.0; h]);
    let mut n = vec![];
    for s in 0..model.spins() {
        let mut moment = Matrix::zeros(p, p);
        for (k, &w) in model.toy.weights.iter().enumerate() {
            let d = state.bands[s][k].density();
            let c_k = &prep.projection[k];
            let projected = c_k.adjoint() * &d * c_k;
            moment += &projected * c(w);
            r += model.degeneracy * w * (&d * &prep.local[k]).trace().re;
            for (a, qa) in q.iter_mut().enumerate() {
                *qa += model.degeneracy * w * (&prep.blocks[a] * &projected).trace().re;
            }
        }
        n.push(
            (0..h)
                .map(|i| {
                    let block = moment.view((i * NP, i * NP), (NP, NP)).into_owned();
                    hermitize(&(weights.transpose() * block * &weights))
                })
                .collect(),
        );
    }
    Mix { n, r, q }
}

pub fn energy(prep: &Prepared, state: &State, out: &Mix) -> f64 {
    let model = &prep.model;
    let mut band = 0.0;
    for s in 0..model.spins() {
        for (k, &w) in model.toy.weights.iter().enumerate() {
            let d = state.bands[s][k].density();
            band += model.degeneracy * w * (&d * fixed(prep, s, k)).trace().re;
        }
    }
    let (hubbard_energy, _) = hubbard(prep, &out.n);
    let hat: f64 = out.q.iter().zip(&prep.hat).map(|(q, h)| q * h).sum();
    let h = out.q.len();
    let quadratic: f64 = (0..h)
        .flat_map(|a| (0..h).map(move |b| (a, b)))
        .map(|(a, b)| 0.5 * out.q[a] * prep.u[(a, b)] * out.q[b])
        .sum();
    let linear: f64 = (0..h)
        .map(|a| out.q[a] * (prep.uion[a] + prep.linear[a]))
        .sum();
    band + hubbard_energy
        + eps(out.r + prep.core)
        + out.r * (hat + prep.ion)
        + quadratic
        + linear
        + prep.constants
}

pub fn start(prep: &Prepared) -> Mix {
    let model = &prep.model;
    Mix {
        n: (0..model.spins())
            .map(|_| {
                (0..model.toy.shell_atoms.len())
                    .map(|_| Matrix::identity(ORBITALS, ORBITALS) * c(0.5))
                    .collect()
            })
            .collect(),
        r: 1.0,
        q: vec![0.0; model.toy.shell_atoms.len()],
    }
}

fn distance(a: &Mix, b: &Mix) -> f64 {
    let n = a.n.iter().flatten().zip(b.n.iter().flatten());
    let worst_n = n
        .flat_map(|(x, y)| (x - y).iter().map(|z| z.norm()).collect::<Vec<_>>())
        .fold(0.0, f64::max);
    let worst_q =
        a.q.iter()
            .zip(&b.q)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
    worst_n.max(worst_q).max((a.r - b.r).abs())
}

pub fn solve(prep: &Prepared, start_from: Option<&Mix>, tolerance: f64) -> Solution {
    let model = &prep.model;
    let mut x = start_from.cloned().unwrap_or_else(|| start(prep));
    for _ in 0..MAXIMUM_ITERATIONS {
        let f = fields(prep, x.r, &x.q);
        let (_, operators) = hubbard(prep, &x.n);
        let state = State {
            bands: (0..model.spins())
                .map(|s| {
                    (0..model.toy.weights.len())
                        .map(|k| bands(prep, s, k, &f, Some(&operators[s])))
                        .collect()
                })
                .collect(),
        };
        let out = observe(prep, &state);
        if distance(&x, &out) < tolerance {
            let energy = energy(prep, &state, &out);
            return Solution {
                input: x,
                output: out,
                state,
                energy,
            };
        }
        let mixed = |a: &Matrix, b: &Matrix| (a + b) * c(0.5);
        x = Mix {
            n: x.n
                .iter()
                .zip(&out.n)
                .map(|(xs, os)| xs.iter().zip(os).map(|(a, b)| mixed(a, b)).collect())
                .collect(),
            r: 0.5 * (x.r + out.r),
            q: x.q.iter().zip(&out.q).map(|(a, b)| 0.5 * (a + b)).collect(),
        };
    }
    panic!("the toy SCF did not converge");
}

pub fn solve_model(model: &super::Model, start_from: Option<&Mix>, tolerance: f64) -> Solution {
    solve(&Prepared::new(model), start_from, tolerance)
}
