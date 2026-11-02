//! Helpers shared by the `hubbard_spinor*` tests: the freeze, a deterministic generator,
//! random states, and thin wrappers over `models::hubbard_spinor`. Every reference is
//! computed here and not by the model under test.
#![allow(dead_code)]

use mandacaru_core::models::hubbard::Hubbard;
use mandacaru_core::models::hubbard_spinor::{self as spinor, Evaluation, Layout, Occupation};
use mandacaru_core::models::spin_orbit;
use mandacaru_core::Context;
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;
use serde_json::Value;

pub const ONE_MIB: usize = 1 << 20;

pub fn context() -> Context {
    Context::new(Some(1), 64 * ONE_MIB).unwrap()
}

pub fn freeze() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test/migration/reference/hubbard/spinor.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

pub fn fraction(v: &Value) -> f64 {
    let text = v.as_str().unwrap();
    let (a, b) = text.split_once('/').unwrap();
    a.parse::<f64>().unwrap() / b.parse::<f64>().unwrap()
}

pub fn json_matrix(v: &Value) -> DMatrix<C64> {
    let (re, im) = (v["re"].as_array().unwrap(), v["im"].as_array().unwrap());
    DMatrix::from_fn(re.len(), re.len(), |i, j| {
        C64::new(re[i][j].as_f64().unwrap(), im[i][j].as_f64().unwrap())
    })
}

/// Deterministic pseudo-random numbers in [-1, 1).
pub struct Lcg(pub u64);
impl Lcg {
    pub fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2. - 1.
    }
    pub fn unit(&mut self) -> f64 {
        0.5 * (self.next() + 1.)
    }
}

pub fn unitary(dim: usize, rng: &mut Lcg) -> DMatrix<C64> {
    DMatrix::from_fn(dim, dim, |_, _| C64::new(rng.next(), rng.next()))
        .qr()
        .q()
}

/// `Q diag(lambda) Q^dagger` with a random Q and eigenvalues in [0, 1].
pub fn random_state(dim: usize, rng: &mut Lcg) -> (DMatrix<C64>, DMatrix<C64>, Vec<f64>) {
    let q = unitary(dim, rng);
    let lambda: Vec<f64> = (0..dim).map(|_| rng.unit()).collect();
    let d = DMatrix::from_fn(dim, dim, |i, j| {
        C64::new(if i == j { lambda[i] } else { 0. }, 0.)
    });
    (&q * d * q.adjoint(), q, lambda)
}

pub fn hermitian_direction(dim: usize, rng: &mut Lcg) -> DMatrix<C64> {
    let a = DMatrix::from_fn(dim, dim, |_, _| C64::new(rng.next(), rng.next()));
    (&a + a.adjoint()) * C64::new(0.5, 0.)
}

pub fn flat(m: &DMatrix<C64>) -> Vec<C64> {
    let d = m.nrows();
    (0..d * d).map(|k| m[(k / d, k % d)]).collect()
}

pub fn dm(d: usize, v: &[C64]) -> DMatrix<C64> {
    DMatrix::from_fn(d, d, |i, j| v[i * d + j])
}

pub fn max_abs(m: &DMatrix<C64>) -> f64 {
    m.iter().map(|x| x.norm()).fold(0., f64::max)
}

pub fn hub(u: f64, j: f64) -> Hubbard {
    Hubbard::new(u, j).unwrap()
}

pub fn layout(l: usize, real: bool) -> Layout {
    Layout::new(l, real).unwrap()
}

pub fn occ(layout: Layout, m: &DMatrix<C64>) -> Occupation {
    Occupation::new(&context(), layout, &flat(m)).unwrap()
}

pub fn eval(h: &Hubbard, layout: Layout, m: &DMatrix<C64>) -> Evaluation {
    spinor::evaluate(&context(), h, &occ(layout, m)).unwrap()
}

pub fn potential(e: &Evaluation) -> DMatrix<C64> {
    let d = e.potential.layout().size();
    dm(d, e.potential.as_slice())
}

/// `Re tr[V H] = Re sum_ab V_ab H_ba`, computed here and not by the model.
pub fn tr_vh(v: &DMatrix<C64>, h: &DMatrix<C64>) -> f64 {
    let d = v.nrows();
    (0..d)
        .flat_map(|a| (0..d).map(move |b| (a, b)))
        .map(|(a, b)| (v[(a, b)] * h[(b, a)]).re)
        .sum()
}

/// Richardson derivative of the model's own energy along `h`.
pub fn finite_difference(hb: &Hubbard, lay: Layout, n: &DMatrix<C64>, h: &DMatrix<C64>) -> f64 {
    let f = |t: f64| eval(hb, lay, &(n + h * C64::new(t, 0.))).energy;
    let d = |s: f64| (f(s) - f(-s)) / (2. * s);
    (4. * d(0.04) - d(0.08)) / 3.
}

pub fn ls_matrix(l: usize, real: bool) -> DMatrix<C64> {
    let a = spin_orbit::angular(&context(), l, real).unwrap();
    dm(2 * (2 * l + 1), a.as_slice())
}

pub fn theta_matrix(lay: Layout) -> DMatrix<C64> {
    let u = spinor::time_reversal(&context(), lay).unwrap();
    dm(lay.size(), u.as_slice())
}

/// Pauli vector `tr[sigma n]`, test-local.
pub fn pauli(p: usize, n: &DMatrix<C64>) -> [f64; 3] {
    let mut m = [0.; 3];
    for a in 0..p {
        let (uu, ud, du, dd) = (n[(a, a)], n[(a, p + a)], n[(p + a, a)], n[(p + a, p + a)]);
        m[0] += (ud + du).re;
        m[1] += (C64::new(0., 1.) * (ud - du)).re;
        m[2] += (uu - dd).re;
    }
    m
}

pub fn spin_rotation(p: usize, rng: &mut Lcg) -> DMatrix<C64> {
    // exp(-i theta n.sigma / 2) = cos(theta/2) - i sin(theta/2) n.sigma
    let mut axis = [rng.next(), rng.next(), rng.next()];
    let norm = axis.iter().map(|x| x * x).sum::<f64>().sqrt();
    axis.iter_mut().for_each(|x| *x /= norm);
    let theta = 0.3 + 2.7 * rng.unit();
    let (c, s) = ((0.5 * theta).cos(), (0.5 * theta).sin());
    let u = [
        [
            C64::new(c, -s * axis[2]),
            C64::new(-s * axis[1], -s * axis[0]),
        ],
        [
            C64::new(s * axis[1], -s * axis[0]),
            C64::new(c, s * axis[2]),
        ],
    ];
    DMatrix::from_fn(2 * p, 2 * p, |i, j| {
        if i % p == j % p {
            u[i / p][j / p]
        } else {
            C64::default()
        }
    })
}

pub const CASES: [(usize, bool); 4] = [(2, false), (3, false), (2, true), (3, true)];
