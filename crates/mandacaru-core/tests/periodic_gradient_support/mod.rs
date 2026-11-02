//! The toy of `periodic_gradient.rs`: the HU-10a toy (`hubbard_gradient_support`, one owner of
//! `C(X)`, `S(X)`, `H0(X)`) extended into a complete, self-consistent, translation-invariant
//! functional with the structure of the crystal energy and no mesh:
//!
//! * `S = S_grid + C q C^+` (P1, P5), `H = T + SR + onsite + v L + C (D_ion + sum_a m_a B_a + O) C^+`
//!   (P2, P4, P3, P5, +U), the pair functions `S_grid`, `T`, `SR`, `L` and `C` being the toy's
//!   Gaussian blocks (kinds Overlap, Hopping, Short, Local, Projection);
//! * one density channel `r = deg sum w Re tr[D L]` (the stand-in for `rho`) and the channel
//!   moments `q_a = deg sum w Re tr[B_a C^+ D C]` (one hat channel per shell);
//! * `E = band + E_U + eps(r + c) + r (sum_a q_a hat_a + ion) + (q U q)/2 + q.u_ion + q.linear
//!   + e_ion + c_ion` with `eps(x) = kappa x^2/2 + tau x^3/3`; `hat_a`, `ion`, `c` (core), `U`,
//!   `u_ion`, `linear`, `e_ion`, `c_ion` are image sums of Gaussians of interatomic distances.
//!   They stand for E1 (hat), E2 (ion, e_ion), E3 (U, u_ion, linear), E4 (c_ion) and E5 (core);
//!   like `functional.rs::Quartic` they are stand-ins, the real forms belong to CF-1.
//!
//! Every term depends on positions through differences only, so the toy is exactly translation
//! invariant (the real mesh terms carry the egg-box ripple; this toy has no mesh).
#![allow(dead_code)]

pub mod derivatives;
pub mod reference;
pub mod scf;

use crate::hubbard_gradient_support::{functional::Functional, Kind, Matrix, Toy, NP};
use mandacaru_core::{periodic::hubbard_gradient::Shell, quantum::C64};
use nalgebra::DMatrix;

/// Orbitals per Hubbard shell in this toy (a p shell, `l = 1`, so the production form applies).
pub const ORBITALS: usize = 3;
const KAPPA: f64 = 0.30;
const TAU: f64 = 0.15;
const U0: f64 = 1.0;

pub fn c(x: f64) -> C64 {
    C64::new(x, 0.0)
}

pub fn shell_weights() -> DMatrix<f64> {
    DMatrix::from_row_slice(NP, ORBITALS, &[0.8, 0.1, 0.2, 0.2, 0.7, 0.1, 0.1, 0.3, 0.6])
}

pub fn shells(toy: &Toy) -> Vec<Shell> {
    (0..toy.shell_atoms.len())
        .map(|s| Shell {
            projectors: (s * NP..(s + 1) * NP).collect(),
            weights: shell_weights(),
        })
        .collect()
}

#[derive(Clone)]
pub struct Model {
    pub toy: Toy,
    /// The spin splitting sign of each channel: `[1, -1]` (degeneracy 1) or `[0]` (degeneracy 2).
    pub signs: Vec<f64>,
    pub degeneracy: f64,
    pub occupations: Vec<f64>,
    pub correction: Functional,
    /// What the toy solver adds to the eigenvalues it returns.
    pub reference: f64,
}

impl Model {
    pub fn polarized(toy: Toy, correction: Functional) -> Self {
        let occupations = vec![1.0; toy.atoms()];
        Model {
            toy,
            signs: vec![1.0, -1.0],
            degeneracy: 1.0,
            occupations,
            correction,
            reference: 0.37,
        }
    }

    pub fn restricted(toy: Toy, correction: Functional) -> Self {
        Model {
            signs: vec![0.0],
            degeneracy: 2.0,
            ..Model::polarized(toy, correction)
        }
    }

    pub fn fractional(mut self, occupations: Vec<f64>) -> Self {
        self.occupations = occupations;
        self
    }

    pub fn spins(&self) -> usize {
        self.signs.len()
    }
}

/// `p x p` constants of the toy: the augmentation `q`, `D_ion` and the channel blocks.
pub fn augmentation(p: usize) -> Matrix {
    let a = DMatrix::from_fn(p, 3, |i, j| 0.6 * ((0.9 * i as f64 + 1.7 * j as f64).sin()));
    let q = &a * a.transpose() * 0.06 + DMatrix::identity(p, p) * 0.02;
    q.map(c)
}

pub fn ion_operator(p: usize) -> Matrix {
    DMatrix::from_fn(p, p, |i, j| {
        let (x, y) = (i as f64, j as f64);
        let off = 0.3 * (0.7 * x - 0.4 * y).cos() + 0.3 * (0.7 * y - 0.4 * x).cos();
        0.5 * off + if i == j { 0.2 } else { 0.0 }
    })
    .map(c)
}

pub fn channel_blocks(toy: &Toy) -> Vec<Matrix> {
    let p = toy.projectors();
    (0..toy.shell_atoms.len())
        .map(|a| {
            Matrix::from_fn(p, p, |i, j| {
                let inside = |x: usize| x / NP == a;
                if !(inside(i) && inside(j)) {
                    return c(0.0);
                }
                let entry = |x: usize, y: usize| {
                    (1.0 + 0.3 * a as f64) * (0.5 + 0.2 * ((x + 2 * y + a) as f64).sin())
                };
                c(0.5 * (entry(i, j) + entry(j, i)))
            })
        })
        .collect()
}

/// The image sum of `exp(-alpha |tau_i - tau_j + R|^2)` without the coincident term, and its
/// derivative with respect to `tau_i` (that with respect to `tau_j` is the negative).
fn pair(toy: &Toy, i: usize, j: usize, alpha: f64) -> (f64, [f64; 3]) {
    let (mut value, mut gradient) = (0.0, [0.0; 3]);
    for r in toy.images() {
        if i == j && r == [0.0; 3] {
            continue;
        }
        let d: [f64; 3] = std::array::from_fn(|a| toy.tau[i][a] - toy.tau[j][a] + r[a]);
        let e = (-alpha * d.iter().map(|x| x * x).sum::<f64>()).exp();
        value += e;
        for a in 0..3 {
            gradient[a] -= 2.0 * alpha * d[a] * e;
        }
    }
    (value, gradient)
}

/// `[i == x] - [j == x]`: how `tau_x` enters a function of `tau_i - tau_j`.
fn sign(i: usize, j: usize, x: usize) -> f64 {
    f64::from(u8::from(i == x)) - f64::from(u8::from(j == x))
}

/// One family of image-sum terms: `eta`, the exponent and the charges.
#[derive(Clone, Copy)]
struct Family {
    eta: f64,
    alpha: f64,
    offset: f64,
    slope: f64,
}

impl Family {
    fn z(&self, atom: usize) -> f64 {
        self.offset + self.slope * atom as f64
    }
}

const HAT: Family = Family {
    eta: 0.20,
    alpha: 0.09,
    offset: 1.0,
    slope: 0.3,
};
const ION: Family = Family {
    eta: 0.30,
    alpha: 0.11,
    offset: 1.0,
    slope: 0.2,
};
const ION_DENSE: Family = Family {
    eta: -0.20,
    alpha: 0.07,
    offset: 1.0,
    slope: -0.1,
};
const CONSTANT: Family = Family {
    eta: 0.40,
    alpha: 0.05,
    offset: 1.0,
    slope: 0.15,
};
const CORE: Family = Family {
    eta: 0.50,
    alpha: 0.10,
    offset: 1.0,
    slope: 0.25,
};
const UION: Family = Family {
    eta: 0.25,
    alpha: 0.08,
    offset: 1.0,
    slope: 0.1,
};
const LINEAR: Family = Family {
    eta: -0.20,
    alpha: 0.04,
    offset: 0.8,
    slope: 0.1,
};
const DENSE: Family = Family {
    eta: 0.30,
    alpha: 0.06,
    offset: 0.0,
    slope: 0.0,
};

/// The geometry-only functions of the toy and their derivatives with respect to one coordinate.
pub struct Geometry<'a> {
    pub toy: &'a Toy,
}

impl Geometry<'_> {
    /// `eta/2 sum_{ij} z_i z_j pair(i, j)`.
    fn bilinear(&self, f: Family) -> f64 {
        let n = self.toy.atoms();
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                sum += 0.5 * f.z(i) * f.z(j) * pair(self.toy, i, j, f.alpha).0;
            }
        }
        f.eta * sum
    }

    fn d_bilinear(&self, f: Family, x: usize, axis: usize) -> f64 {
        let n = self.toy.atoms();
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                let g = pair(self.toy, i, j, f.alpha).1[axis];
                sum += 0.5 * f.z(i) * f.z(j) * sign(i, j, x) * g;
            }
        }
        f.eta * sum
    }

    /// `eta sum_B z_B pair(atom, B)`.
    fn linear_sum(&self, f: Family, atom: usize) -> f64 {
        let sum: f64 = (0..self.toy.atoms())
            .map(|b| f.z(b) * pair(self.toy, atom, b, f.alpha).0)
            .sum();
        f.eta * sum
    }

    fn d_linear_sum(&self, f: Family, atom: usize, x: usize, axis: usize) -> f64 {
        let sum: f64 = (0..self.toy.atoms())
            .map(|b| f.z(b) * sign(atom, b, x) * pair(self.toy, atom, b, f.alpha).1[axis])
            .sum();
        f.eta * sum
    }

    fn channel_atom(&self, a: usize) -> usize {
        self.toy.shell_atoms[a]
    }

    pub fn hat(&self, a: usize) -> f64 {
        self.linear_sum(HAT, self.channel_atom(a))
    }
    pub fn d_hat(&self, a: usize, x: usize, axis: usize) -> f64 {
        self.d_linear_sum(HAT, self.channel_atom(a), x, axis)
    }
    pub fn ion(&self) -> f64 {
        self.bilinear(ION)
    }
    pub fn d_ion(&self, x: usize, axis: usize) -> f64 {
        self.d_bilinear(ION, x, axis)
    }
    pub fn ion_dense(&self) -> f64 {
        self.bilinear(ION_DENSE)
    }
    pub fn d_ion_dense(&self, x: usize, axis: usize) -> f64 {
        self.d_bilinear(ION_DENSE, x, axis)
    }
    pub fn constant(&self) -> f64 {
        self.bilinear(CONSTANT)
    }
    pub fn d_constant(&self, x: usize, axis: usize) -> f64 {
        self.d_bilinear(CONSTANT, x, axis)
    }
    pub fn core(&self) -> f64 {
        self.bilinear(CORE)
    }
    pub fn d_core(&self, x: usize, axis: usize) -> f64 {
        self.d_bilinear(CORE, x, axis)
    }
    pub fn uion(&self, a: usize) -> f64 {
        self.linear_sum(UION, self.channel_atom(a))
    }
    pub fn d_uion(&self, a: usize, x: usize, axis: usize) -> f64 {
        self.d_linear_sum(UION, self.channel_atom(a), x, axis)
    }
    pub fn linear(&self, a: usize) -> f64 {
        self.linear_sum(LINEAR, self.channel_atom(a))
    }
    pub fn d_linear(&self, a: usize, x: usize, axis: usize) -> f64 {
        self.d_linear_sum(LINEAR, self.channel_atom(a), x, axis)
    }

    pub fn u(&self) -> DMatrix<f64> {
        let h = self.toy.shell_atoms.len();
        DMatrix::from_fn(h, h, |a, b| {
            let (i, j) = (self.channel_atom(a), self.channel_atom(b));
            DENSE.eta * pair(self.toy, i, j, DENSE.alpha).0 + if a == b { U0 } else { 0.0 }
        })
    }
    pub fn d_u(&self, x: usize, axis: usize) -> DMatrix<f64> {
        let h = self.toy.shell_atoms.len();
        DMatrix::from_fn(h, h, |a, b| {
            let (i, j) = (self.channel_atom(a), self.channel_atom(b));
            DENSE.eta * sign(i, j, x) * pair(self.toy, i, j, DENSE.alpha).1[axis]
        })
    }
}

/// `eps(x)` and `eps'(x)` of the toy's local (xc-like) energy.
pub fn eps(x: f64) -> f64 {
    0.5 * KAPPA * x * x + TAU / 3.0 * x * x * x
}
pub fn d_eps(x: f64) -> f64 {
    KAPPA * x + TAU * x * x
}

/// Everything that depends on the geometry but not on the state, built once per solve.
pub struct Prepared {
    pub model: Model,
    pub overlap: Vec<Matrix>,
    pub hopping: Vec<Matrix>,
    pub short: Vec<Matrix>,
    pub local: Vec<Matrix>,
    pub projection: Vec<Matrix>,
    pub q_matrix: Matrix,
    pub d_ion: Matrix,
    pub blocks: Vec<Matrix>,
    pub hat: Vec<f64>,
    pub uion: Vec<f64>,
    pub linear: Vec<f64>,
    pub u: DMatrix<f64>,
    pub ion: f64,
    pub core: f64,
    /// `e_ion + c_ion`.
    pub constants: f64,
}

impl Prepared {
    pub fn new(model: &Model) -> Self {
        let toy = &model.toy;
        let ks = toy.kpoints();
        let all = |kind: Kind| -> Vec<Matrix> { ks.iter().map(|&k| toy.bloch(kind, k)).collect() };
        let g = Geometry { toy };
        let h = toy.shell_atoms.len();
        let q_matrix = augmentation(toy.projectors());
        let projection = all(Kind::Projection);
        let s1 = all(Kind::Overlap);
        let overlap = s1
            .iter()
            .zip(&projection)
            .map(|(s, cc)| s + cc * &q_matrix * cc.adjoint())
            .collect();
        Prepared {
            model: model.clone(),
            overlap,
            hopping: all(Kind::Hopping),
            short: all(Kind::Short),
            local: all(Kind::Local),
            projection,
            q_matrix,
            d_ion: ion_operator(toy.projectors()),
            blocks: channel_blocks(toy),
            hat: (0..h).map(|a| g.hat(a)).collect(),
            uion: (0..h).map(|a| g.uion(a)).collect(),
            linear: (0..h).map(|a| g.linear(a)).collect(),
            u: g.u(),
            ion: g.ion(),
            core: g.core(),
            constants: g.ion_dense() + g.constant(),
        }
    }
}
