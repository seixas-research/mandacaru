//! The Hermitian and symmetric eigensolvers return an accurate decomposition.
//!
//! The tridiagonal QR iteration underneath both can stop at a decomposition
//! whose vectors are orthonormal but whose residual is far from rounding, most
//! often on degenerate or clustered spectra. Both solvers check their result
//! and fall back to Jacobi rotations; these cases fail without that check.
use mandacaru_core::math::linalg::{eigen, real_eigen};
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

/// A 10x10 real symmetric matrix on which the unchecked QR iteration left a
/// relative residual of 5e-8.
const REPRODUCER: [[f64; 10]; 10] = [
    [-0.188848692099784526, 1.04552804093860763, -0.0274650038550962178, 0.0475603256816918701, 0.701871592247045983, 0.826012050417668342, 2.12165459624911534, 0.102834980130491499, -0.0394028326842824905, -1.64475148224720114],
    [1.04552804093860763, -0.178743556635501788, -0.682043531950027315, 0.0452094390206396746, -0.311354879968180898, 0.900571941057763503, -0.494074100908043157, 0.415099504903946481, -0.555976610391556547, 0.320319466753589133],
    [-0.0274650038550962178, -0.682043531950027315, 0.0229775648463958655, -0.241916125362163226, 0.226403428269946749, 0.668060156995863852, -0.00141694966278754775, -0.260230193633177231, -0.348021030646172180, -0.897108024362389012],
    [0.0475603256816918701, 0.0452094390206396746, -0.241916125362163226, 1.87470093001512317, 0.438797967690102653, -0.194544600790810318, -1.09140925558527124, 0.549490314498584764, 0.448999644949850452, 1.90429029357033830],
    [0.701871592247045983, -0.311354879968180898, 0.226403428269946749, 0.438797967690102653, 0.0618427102522051547, 0.603787982522788536, -0.270553827225303356, 0.930449323277430129, -0.967057521164093714, 0.366722870782862365],
    [0.826012050417668342, 0.900571941057763503, 0.668060156995863852, -0.194544600790810318, 0.603787982522788536, -0.838364726471482413, -0.793108973282918628, 0.0739160608176950423, -1.20507041645322333, 0.489913253173467012],
    [2.12165459624911534, -0.494074100908043157, -0.00141694966278754775, -1.09140925558527124, -0.270553827225303356, -0.793108973282918628, -0.588932217946592451, 0.317527590654003267, -1.34133402931250467, 1.17857039744321712],
    [0.102834980130491499, 0.415099504903946481, -0.260230193633177231, 0.549490314498584764, 0.930449323277430129, 0.0739160608176950423, 0.317527590654003267, 2.07325737482334116, 0.934794747013029226, -1.33117958573251194],
    [-0.0394028326842824905, -0.555976610391556547, -0.348021030646172180, 0.448999644949850452, -0.967057521164093714, -1.20507041645322333, -1.34133402931250467, 0.934794747013029226, 1.02547528325284665, -0.925776260623227043],
    [-1.64475148224720114, 0.320319466753589133, -0.897108024362389012, 1.90429029357033830, 0.366722870782862365, 0.489913253173467012, 1.17857039744321712, -1.33117958573251194, -0.925776260623227043, -0.617353322229558299],
];
const BOUND: f64 = 1e-11;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 2. - 1.
    }
}

/// Largest of the relative residual and the orthogonality error.
fn error(a: &DMatrix<C64>, w: &[f64], v: &DMatrix<C64>) -> f64 {
    let n = w.len();
    let d = DMatrix::from_fn(n, n, |i, j| if i == j { C64::new(w[i], 0.) } else { C64::default() });
    let residual = (a * v - v * d).norm() / a.norm().max(f64::MIN_POSITIVE);
    let orthogonality = (v.adjoint() * v - DMatrix::<C64>::identity(n, n)).norm();
    residual.max(orthogonality)
}
fn check_complex(a: &DMatrix<C64>) {
    let (w, v) = eigen(a.clone()).unwrap();
    let e = error(a, &w, &v);
    assert!(e < BOUND, "complex eigen error {e:e} for n={}", a.nrows());
    assert!(w.windows(2).all(|p| p[0] <= p[1]), "eigenvalues are ascending");
}
fn check_real(a: &DMatrix<f64>) {
    let (w, v) = real_eigen(a.clone()).unwrap();
    let (ac, vc) = (a.map(|x| C64::new(x, 0.)), v.map(|x| C64::new(x, 0.)));
    let e = error(&ac, &w, &vc);
    assert!(e < BOUND, "real eigen error {e:e} for n={}", a.nrows());
}
fn symmetric(rng: &mut Lcg, n: usize) -> DMatrix<f64> {
    let m = DMatrix::from_fn(n, n, |_, _| rng.next());
    (&m + m.transpose()) * 0.5
}
fn hermitian(rng: &mut Lcg, n: usize) -> DMatrix<C64> {
    let m = DMatrix::from_fn(n, n, |_, _| C64::new(rng.next(), rng.next()));
    (&m + m.adjoint()) * C64::new(0.5, 0.)
}

#[test]
fn the_reproducer_is_decomposed_accurately_by_both_solvers() {
    let a = DMatrix::from_fn(10, 10, |i, j| REPRODUCER[i][j]);
    check_real(&a);
    check_complex(&a.map(|x| C64::new(x, 0.)));
}

#[test]
fn random_matrices_of_many_sizes() {
    let mut rng = Lcg(12345);
    for &n in &[2usize, 3, 8, 16, 33, 48, 100] {
        for _ in 0..(if n > 50 { 10 } else { 60 }) {
            check_real(&symmetric(&mut rng, n));
            check_complex(&hermitian(&mut rng, n));
        }
    }
}

#[test]
fn degenerate_blocks_with_a_tiny_perturbation() {
    // I_k (x) M carries every eigenvalue of M k times; a 1e-13 perturbation
    // turns the degeneracy into a cluster. This family left a 4e-3 residual.
    let mut rng = Lcg(777);
    for &(k, m) in &[(2usize, 5usize), (4, 6), (6, 4), (3, 12)] {
        let base = hermitian(&mut rng, m);
        let n = k * m;
        let mut a = DMatrix::<C64>::zeros(n, n);
        for b in 0..k {
            a.view_mut((b * m, b * m), (m, m)).copy_from(&base);
        }
        let noise = hermitian(&mut rng, n) * C64::new(1e-13, 0.);
        check_complex(&(a.clone() + noise));
        check_complex(&a);
        let real = a.map(|x| x.re);
        check_real(&((&real + real.transpose()) * 0.5));
    }
}

#[test]
fn identity_plus_a_low_rank_term() {
    let mut rng = Lcg(99);
    for &n in &[12usize, 40] {
        let u = DMatrix::from_fn(n, 2, |_, _| C64::new(rng.next(), rng.next()));
        let a = DMatrix::<C64>::identity(n, n) + &u * u.adjoint();
        check_complex(&a);
    }
}

#[test]
fn the_acceptance_is_relative_to_the_matrix_scale() {
    // A matrix of norm 1e-8 must be as accurate, relatively, as one of norm 1.
    let mut rng = Lcg(2024);
    for _ in 0..40 {
        check_real(&(symmetric(&mut rng, 24) * 1e-8));
        check_complex(&(hermitian(&mut rng, 24) * C64::new(1e-8, 0.)));
    }
}
