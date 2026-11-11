//! The ordered term-list algebra behind the Python operator builders keeps insertion order:
//! sums list a key where it first occurs, products visit left outer and right inner.
use mandacaru_core::quantum::{term_lists, C64};

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

#[test]
fn a_sum_adds_equal_keys_where_they_first_occur() {
    let terms = vec![("ZI", c(1., 0.)), ("XX", c(0., 2.)), ("ZI", c(0.5, 0.)), ("IY", c(-1., 0.))];
    let out = term_lists::sum(terms).unwrap();
    assert_eq!(out, vec![("ZI", c(1.5, 0.)), ("XX", c(0., 2.)), ("IY", c(-1., 0.))]);
    assert!(term_lists::sum(vec![("Z", c(f64::NAN, 0.))]).is_err());
    assert!(term_lists::sum(vec![("Z", c(f64::MAX, 0.)), ("Z", c(f64::MAX, 0.))]).is_err());
}

#[test]
fn scale_drop_and_hermiticity_keep_order() {
    let terms = vec![("ZZ", c(2., 0.)), ("XI", c(0., 1e-13)), ("IY", c(1., 1e-10))];
    assert_eq!(term_lists::scale(&terms, c(0., 1.)).unwrap()[0], ("ZZ", c(0., 2.)));
    assert_eq!(term_lists::drop_small(&terms, 1e-12).unwrap().iter().map(|t| t.0).collect::<Vec<_>>(), vec!["ZZ", "IY"]);
    // The 1e-13 imaginary string is below the drop threshold and does not count.
    assert!(term_lists::hermitian(&terms, 1e-9, 1e-12));
    assert!(!term_lists::hermitian(&[("Y", c(0., 1e-6))], 1e-9, 1e-12));
    assert!(term_lists::scale(&terms, c(f64::INFINITY, 0.)).is_err());
}

#[test]
fn a_fermion_product_concatenates_left_outer_and_dagger_reverses() {
    let a = vec![(vec![(0, true)], c(2., 0.)), (vec![(1, true)], c(0., 1.))];
    let b = vec![(vec![(1, false)], c(3., 0.))];
    let product = term_lists::fermion_product(&a, &b).unwrap();
    assert_eq!(
        product,
        vec![(vec![(0, true), (1, false)], c(6., 0.)), (vec![(1, true), (1, false)], c(0., 3.))]
    );
    let dagger = term_lists::fermion_dagger(&product).unwrap();
    assert_eq!(dagger[0], (vec![(1, true), (0, false)], c(6., 0.)));
    assert_eq!(dagger[1], (vec![(1, true), (1, false)], c(0., -3.)));
    // (a+_1 a_1 + h.c. with opposite phases) adds to the same product.
    let hermitian = term_lists::fermion_dagger(&[(vec![(1, true), (1, false)], c(0., 1.)), (vec![(1, true), (1, false)], c(0., 1.))]).unwrap();
    assert_eq!(hermitian, vec![(vec![(1, true), (1, false)], c(0., -2.))]);
    assert_eq!(term_lists::fermion_modes(&product), 2);
    assert_eq!(term_lists::fermion_modes(&[(vec![], c(1., 0.))]), 0);
}

#[test]
fn a_propagator_keeps_the_real_terms_and_sums_their_magnitudes() {
    let terms = vec![
        ("XX".to_string(), c(0.5, 0.)),
        ("ZI".to_string(), c(0., 0.)),
        ("IZ".to_string(), c(-0.25, 1e-13)),
    ];
    let (real, bound) = term_lists::real_pauli_terms(2, &terms).unwrap();
    assert_eq!(real, vec![("XX".to_string(), 0.5), ("IZ".to_string(), -0.25)]);
    assert_eq!(bound, 0.75);
    // The bound is added as Python's `sum` adds floats (compensated): 0.0005 + 0.5 + 1.25 + 0.5
    // is 2.2505, where a running sum gives 2.2504999999999997.
    let four = [("Z", 0.0005), ("Y", 0.5), ("I", -1.25), ("X", 0.5)]
        .map(|(label, value)| (label.to_string(), c(value, 0.)));
    assert_eq!(term_lists::real_pauli_terms(1, &four).unwrap().1, 2.2505);
    // No term left: the bound is a positive zero.
    let (none, zero) = term_lists::real_pauli_terms(1, &[("Z".to_string(), c(0., 0.))]).unwrap();
    assert!(none.is_empty() && zero == 0. && zero.is_sign_positive());
    let refused = |label: &str, value: C64| term_lists::real_pauli_terms(2, &[(label.to_string(), value)]);
    assert!(refused("XQ", c(1., 0.)).is_err());
    assert!(refused("X", c(1., 0.)).is_err());
    assert!(refused("XX", c(1., 1e-9)).is_err());
    assert!(refused("XX", c(f64::INFINITY, 0.)).is_err());
    let overflow = [("XX".to_string(), c(f64::MAX, 0.)), ("YY".to_string(), c(f64::MAX, 0.))];
    assert!(term_lists::real_pauli_terms(2, &overflow).is_err());
}
