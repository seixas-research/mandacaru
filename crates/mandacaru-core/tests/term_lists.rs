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
