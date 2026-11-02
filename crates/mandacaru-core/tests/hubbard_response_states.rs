//! `models::hubbard_response::states`: the bare response as a sum over states against the
//! finite difference of the frozen problem at fixed electron number.
mod hubbard_response_support;

use hubbard_response_support::*;
use mandacaru_core::models::hubbard_response::{sum_over_states, StateBlock};
use mandacaru_core::Error;
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

// ---- sum over states ---------------------------------------------------------------

/// Occupation of site `s` at fixed `electrons` and `kt`, Fermi level re-found.
fn site_occupation(h: &[f64], n: usize, electrons: f64, kt: f64, site: usize) -> f64 {
    let (e, v) = jacobi(h, n);
    let (f, _) = fermi_dirac(&e, electrons, kt);
    2. * (0..n).map(|k| f[k] * v[k * n + site].powi(2)).sum::<f64>()
}

fn frozen_difference(electrons: f64, kt: f64, sites: &[usize]) -> DMatrix<f64> {
    let n = 12;
    let a = 1e-4;
    let mut out = DMatrix::zeros(sites.len(), sites.len());
    for (j, &sj) in sites.iter().enumerate() {
        let shifted = |sign: f64| {
            let mut h = clean_ring(n);
            h[sj * n + sj] += sign * a;
            h
        };
        for (i, &si) in sites.iter().enumerate() {
            let plus = site_occupation(&shifted(1.), n, electrons, kt, si);
            let minus = site_occupation(&shifted(-1.), n, electrons, kt, si);
            out[(i, j)] = (plus - minus) / (2. * a);
        }
    }
    out
}

fn sum_for(f: &[f64], fp: &[f64], sites: &[usize]) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = 12;
    let (e, v) = jacobi(&clean_ring(n), n);
    let elements: Vec<Vec<C64>> = sites
        .iter()
        .map(|&s| {
            (0..n * n)
                .map(|x| C64::new(v[(x / n) * n + s] * v[(x % n) * n + s], 0.))
                .collect()
        })
        .collect();
    let slices: Vec<&[C64]> = elements.iter().map(|m| m.as_slice()).collect();
    let block = StateBlock {
        weight: 1.,
        levels: &e,
        occupations: f,
        slopes: fp,
        elements: &slices,
    };
    let sum = sum_over_states(&[block], 2.).unwrap();
    (sum.total(), sum.pairs)
}

fn states(electrons: f64, kt: f64, sites: &[usize]) -> (DMatrix<f64>, DMatrix<f64>) {
    let (e, _) = jacobi(&clean_ring(12), 12);
    let (f, fp) = fermi_dirac(&e, electrons, kt);
    sum_for(&f, &fp, sites)
}

#[test]
fn the_sum_over_states_matches_the_frozen_difference_and_the_fermi_term_is_needed() {
    let sites = [0, 1, 5];
    let reference = frozen_difference(12., 0.15, &sites);
    let (total, pairs) = states(12., 0.15, &sites);
    let with = worst(&total, &reference);
    let without = worst(&pairs, &reference);
    println!("metal: with Fermi term {with:.2e}, without {without:.3}");
    assert!(with < 1e-6, "sum over states vs frozen difference: {with}");
    // The negative control. Dropping the Fermi-level term leaves the response at fixed
    // potential, not at fixed electron number. Its size is not a tunable threshold: on a
    // translation-invariant ring every site has the same weight in every level (the diagonal
    // of the projector onto a degenerate subspace of dimension d is d / N_sites), so
    // s_I = S / 12 for all I and the dropped term is S / 144 in every entry, with
    // S = g sum_n f'_n the thermal density of states at the Fermi level. That closed form
    // is computed here from the occupations alone.
    let (e, _) = jacobi(&clean_ring(12), 12);
    let (_, slopes) = fermi_dirac(&e, 12., 0.15);
    let dropped = (2. * slopes.iter().sum::<f64>() / 144.).abs();
    println!("metal: closed-form dropped term {dropped:.6}");
    assert!(
        (without - dropped).abs() < 1e-6 && without > 1e4 * with,
        "dropping the Fermi-level term must err by exactly |S|/144 = {dropped}: {without}"
    );
    assert!(worst(&total, &total.transpose()) < 1e-14);
}

// ---- complex matrix elements ---------------------------------------------------------

/// Deterministic numbers in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2. - 1.
    }
    fn complex(&mut self) -> C64 {
        C64::new(self.next(), self.next())
    }
}

/// Eigenpairs of a Hermitian matrix by nalgebra, ascending: the model under test never
/// sees this routine, and the real Jacobi of the support module cannot take a complex matrix.
fn hermitian_eigen(h: &DMatrix<C64>) -> (Vec<f64>, DMatrix<C64>) {
    let eigen = nalgebra::linalg::SymmetricEigen::new(h.clone());
    let mut order: Vec<usize> = (0..h.nrows()).collect();
    order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
    let levels = order.iter().map(|&k| eigen.eigenvalues[k]).collect();
    let vectors = DMatrix::from_fn(h.nrows(), h.nrows(), |i, j| {
        eigen.eigenvectors[(i, order[j])]
    });
    (levels, vectors)
}

/// `<n|P|m>` row-major, the layout `StateBlock::elements` documents.
fn overlaps(vectors: &DMatrix<C64>, p: &DMatrix<C64>) -> Vec<C64> {
    let m = vectors.adjoint() * p * vectors;
    let bands = m.nrows();
    (0..bands * bands)
        .map(|x| m[(x / bands, x % bands)])
        .collect()
}

#[test]
fn complex_matrix_elements_fix_the_order_of_the_pair_product() {
    // A Bloch crystal at a general k has complex <n|P_I|m>; every other test here uses real
    // elements, for which M_J(m,n) = M_J(n,m) and a swapped pair product cannot be seen.
    // Here H0 is a random complex Hermitian matrix and P_I = |v_I><v_I| projects on a random
    // complex orbital, so Im <n|P_I|m> is of order one.
    let (bands, shells, electrons, kt, a) = (6, 3, 6., 0.4, 1e-4);
    let mut rng = Lcg(20261101);
    let mut h0 = DMatrix::from_fn(bands, bands, |_, _| rng.complex());
    h0 = (&h0 + h0.adjoint()) * C64::new(0.5, 0.);
    let projectors: Vec<DMatrix<C64>> = (0..shells)
        .map(|_| {
            let v = DMatrix::from_fn(bands, 1, |_, _| rng.complex());
            let v = &v / C64::new(v.norm(), 0.);
            &v * v.adjoint()
        })
        .collect();

    // Reference: thermal finite difference at fixed electron number, Fermi level re-found.
    let occupation = |h: &DMatrix<C64>, i: usize| {
        let (e, v) = hermitian_eigen(h);
        let (f, _) = fermi_dirac(&e, electrons, kt);
        let m = overlaps(&v, &projectors[i]);
        2. * (0..bands).map(|n| f[n] * m[n * bands + n].re).sum::<f64>()
    };
    let reference = DMatrix::from_fn(shells, shells, |i, j| {
        let shifted = |sign: f64| &h0 + &projectors[j] * C64::new(sign * a, 0.);
        (occupation(&shifted(1.), i) - occupation(&shifted(-1.), i)) / (2. * a)
    });

    let (e, v) = hermitian_eigen(&h0);
    let (f, fp) = fermi_dirac(&e, electrons, kt);
    let elements: Vec<Vec<C64>> = projectors.iter().map(|p| overlaps(&v, p)).collect();
    let imaginary = elements
        .iter()
        .flat_map(|m| m.iter().map(|x| x.im.abs()))
        .fold(0., f64::max);
    assert!(imaginary > 0.1, "the elements must be complex: {imaginary}");
    let slices: Vec<&[C64]> = elements.iter().map(|m| m.as_slice()).collect();
    let block = StateBlock {
        weight: 1.,
        levels: &e,
        occupations: &f,
        slopes: &fp,
        elements: &slices,
    };
    let sum = sum_over_states(&[block], 2.).unwrap();
    let agreement = worst(&sum.total(), &reference);
    println!("complex elements: agreement {agreement:.2e}, max |Im M| {imaginary:.2}");
    assert!(
        agreement < 1e-6,
        "complex sum over states vs frozen difference: {agreement}"
    );

    // The control: the same sum with M_J(m,n) replaced by M_J(n,m). It must miss the
    // reference by an amount that is not a rounding error, or the test above has no teeth.
    let swapped = DMatrix::from_fn(shells, shells, |i, j| {
        let mut total = 0.;
        for n in 0..bands {
            for m in 0..bands {
                let gap = e[n] - e[m];
                let c = if gap.abs() < 1e-10 {
                    fp[n]
                } else {
                    (f[n] - f[m]) / gap
                };
                total += 2. * c * (elements[i][n * bands + m] * elements[j][n * bands + m]).re;
            }
        }
        total
    }) + &sum.fermi;
    let miss = worst(&swapped, &reference);
    println!("complex elements: swapped M_J(n,m) misses by {miss:.2e}");
    assert!(
        miss > 1e3 * agreement.max(1e-9) && miss > 1e-2,
        "the swapped pair product must fail the reference: {miss}"
    );
}

#[test]
fn an_insulator_at_tiny_width_skips_the_fermi_term_instead_of_dividing_by_zero() {
    let sites = [0, 3];
    let reference = frozen_difference(10., 5e-4, &sites);
    // Zero-width limit: five doubly occupied orbitals, every slope exactly zero.
    let f: Vec<f64> = (0..12).map(|k| if k < 5 { 1. } else { 0. }).collect();
    let (total, pairs) = sum_for(&f, &[0.; 12], &sites);
    assert!(total.iter().all(|v| v.is_finite()));
    assert_eq!(
        total, pairs,
        "the sum of slopes underflows to zero: no Fermi term"
    );
    assert!(worst(&total, &reference) < 1e-6);
}

#[test]
fn inconsistent_state_blocks_are_refused() {
    let (levels, occupations, slopes) = ([0., 1.], [1., 0.], [0., 0.]);
    let m = [C64::new(1., 0.); 4];
    let one = [&m[..]];
    let base = StateBlock {
        weight: 1.,
        levels: &levels,
        occupations: &occupations,
        slopes: &slopes,
        elements: &one,
    };
    let short = [&m[..3]];
    let mut bad = [
        StateBlock {
            elements: &short,
            ..base
        },
        StateBlock {
            weight: -1.,
            ..base
        },
        StateBlock {
            occupations: &occupations[..1],
            ..base
        },
    ];
    for block in bad.iter_mut() {
        assert!(matches!(
            sum_over_states(&[*block], 2.),
            Err(Error::Invalid(_))
        ));
    }
    assert!(matches!(
        sum_over_states(&[base], 0.),
        Err(Error::Invalid(_))
    ));
}
