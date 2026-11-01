//! The self-consistent toy: for each spin and k, `(H0 + C O C^dagger) c = e S c` with integer
//! occupations (the lowest `atoms` states), `n = W^T (sum_k w C^dagger D C) W` per shell, damped
//! fixed-point iteration to a density residual below `TOLERANCE`.
use super::{functional::Functional, shell_weights, Kind, Matrix, Toy, NM, NP};
use mandacaru_core::{math::linalg::eigen, quantum::C64};
use nalgebra::{linalg::Cholesky, DMatrix};

pub const TOLERANCE: f64 = 1e-14;
const MAXIMUM_ITERATIONS: usize = 20000;
const MINIMUM_GAP: f64 = 0.1;

pub struct State {
    /// `n[spin][shell]`.
    pub n: Vec<Vec<Matrix>>,
    /// `density[spin][k]`: `D = sum_occupied c c^dagger`.
    pub density: Vec<Vec<Matrix>>,
    /// `weighted[spin][k]`: `sum_occupied e c c^dagger`, with the eigenvalues of the full `H`.
    pub weighted: Vec<Vec<Matrix>>,
    /// The potentials of the output density, per spin and shell, and the shift they subtract.
    pub potentials: Vec<Vec<Matrix>>,
    pub double_counting: Vec<Vec<Matrix>>,
    pub energy: f64,
}

pub const SIGNS: [f64; 2] = [1.0, -1.0];

fn complex(m: &DMatrix<f64>) -> Matrix {
    m.map(|x| C64::new(x, 0.0))
}

fn hermitize(m: &Matrix) -> Matrix {
    (m + m.adjoint()) * C64::new(0.5, 0.0)
}

/// `O = W V W^T` on the projectors of every shell (written here, independent of the layer).
fn operator(toy: &Toy, potentials: &[Matrix]) -> Matrix {
    let w = complex(&shell_weights());
    let mut o = Matrix::zeros(toy.projectors(), toy.projectors());
    for (s, v) in potentials.iter().enumerate() {
        let block = &w * v * w.transpose();
        for i in 0..NP {
            for j in 0..NP {
                o[(s * NP + i, s * NP + j)] = block[(i, j)];
            }
        }
    }
    o
}

/// Occupied eigenpairs of `H c = e S c`.
fn generalized(h: &Matrix, s: &Matrix, occupied: usize) -> (Vec<f64>, Matrix) {
    let l = Cholesky::new(s.clone())
        .expect("the toy overlap is positive definite")
        .l();
    let a = l.solve_lower_triangular(h).unwrap();
    let reduced = l.solve_lower_triangular(&a.adjoint()).unwrap().adjoint();
    let (e, y) = eigen(hermitize(&reduced)).unwrap();
    assert!(
        e[occupied] - e[occupied - 1] > MINIMUM_GAP,
        "the toy gap closed: {e:?}"
    );
    let c = l.adjoint().solve_upper_triangular(&y).unwrap();
    (e[..occupied].to_vec(), c.columns(0, occupied).into_owned())
}

pub fn solve(toy: &Toy, functional: &Functional, start: Option<&[Vec<Matrix>]>) -> State {
    let ks = toy.kpoints();
    let overlap: Vec<_> = ks.iter().map(|&k| toy.bloch(Kind::Overlap, k)).collect();
    let hopping: Vec<_> = ks.iter().map(|&k| toy.bloch(Kind::Hopping, k)).collect();
    let projection: Vec<_> = ks.iter().map(|&k| toy.bloch(Kind::Projection, k)).collect();
    let bare: Vec<Vec<Matrix>> = SIGNS
        .iter()
        .map(|&sign| hopping.iter().map(|h| h + toy.onsite(sign)).collect())
        .collect();
    let w = complex(&shell_weights());
    let shells = toy.shell_atoms.len();
    let mut n: Vec<Vec<Matrix>> = match start {
        Some(n) => n.to_vec(),
        None => (0..2)
            .map(|_| {
                (0..shells)
                    .map(|_| Matrix::identity(NM, NM) * C64::new(0.5, 0.0))
                    .collect()
            })
            .collect(),
    };
    let occupied = toy.atoms();
    for _ in 0..MAXIMUM_ITERATIONS {
        let mut out: Vec<Vec<Matrix>> = vec![];
        let (mut density, mut weighted) = (vec![], vec![]);
        let evaluations: Vec<_> = (0..shells)
            .map(|i| functional.evaluate([&n[0][i], &n[1][i]]))
            .collect();
        for s in 0..2 {
            let potentials: Vec<_> = evaluations.iter().map(|e| e.potential[s].clone()).collect();
            let o = operator(toy, &potentials);
            let mut r = Matrix::zeros(toy.projectors(), toy.projectors());
            let (mut d_s, mut w_s) = (vec![], vec![]);
            for (k, &weight) in toy.weights.iter().enumerate() {
                let c = &projection[k];
                let h = &bare[s][k] + c * &o * c.adjoint();
                let (e, vectors) = generalized(&h, &overlap[k], occupied);
                let d = &vectors * vectors.adjoint();
                let scaled = Matrix::from_fn(toy.basis(), occupied, |i, j| vectors[(i, j)] * e[j]);
                w_s.push(&scaled * vectors.adjoint());
                r += c.adjoint() * &d * c * C64::new(weight, 0.0);
                d_s.push(d);
            }
            out.push(
                (0..shells)
                    .map(|i| {
                        let block = r.view((i * NP, i * NP), (NP, NP)).into_owned();
                        hermitize(&(w.transpose() * block * &w))
                    })
                    .collect(),
            );
            density.push(d_s);
            weighted.push(w_s);
        }
        let residual = (0..2)
            .flat_map(|s| (0..shells).map(move |i| (s, i)))
            .map(|(s, i)| {
                (&out[s][i] - &n[s][i])
                    .iter()
                    .map(|z| z.norm())
                    .fold(0.0, f64::max)
            })
            .fold(0.0, f64::max);
        if residual < TOLERANCE {
            let mut energy: f64 = (0..shells)
                .map(|i| functional.evaluate([&out[0][i], &out[1][i]]).energy)
                .sum();
            for s in 0..2 {
                for (k, &weight) in toy.weights.iter().enumerate() {
                    energy += weight * (&density[s][k] * &bare[s][k]).trace().re;
                }
            }
            let pick =
                |f: fn(&super::functional::Evaluation, usize) -> Matrix| -> Vec<Vec<Matrix>> {
                    (0..2)
                        .map(|s| evaluations.iter().map(|e| f(e, s)).collect())
                        .collect()
                };
            return State {
                n: out,
                density,
                weighted,
                potentials: pick(|e, s| e.potential[s].clone()),
                double_counting: pick(|e, s| e.double_counting[s].clone()),
                energy,
            };
        }
        for s in 0..2 {
            for i in 0..shells {
                n[s][i] = (&n[s][i] + &out[s][i]) * C64::new(0.5, 0.0);
            }
        }
    }
    panic!("the toy SCF did not converge");
}
