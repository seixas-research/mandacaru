use mandacaru_core::math::lattice::{
    det3, input_lattice_rotations, selling_reduce, unimodular_inverse, validate_cell,
};
use nalgebra::Matrix3;

#[test]
fn test_unimodular_matrix_helpers() {
    let m = Matrix3::new(1, 2, 3, 0, 1, 4, 5, 6, 0);
    let d = det3(&m);
    assert_eq!(d, 1);
    let inv = unimodular_inverse(&m).unwrap();
    assert_eq!(det3(&inv), 1);
    let id = m * inv;
    assert_eq!(id, Matrix3::identity());
}

#[test]
fn test_selling_reduce_sc() {
    let a = 3.5;
    let cell = Matrix3::new(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
    let (b, u) = selling_reduce(&cell).unwrap();
    assert_eq!(det3(&u), 1);
    let diff = (b - cell).norm();
    assert!(diff < 1e-10);
}

#[test]
fn test_lattice_rotations_cubic() {
    let a = 3.8;
    let sc = Matrix3::new(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
    let rots = input_lattice_rotations(&sc, 1e-5).unwrap();
    assert_eq!(rots.len(), 48);
    // Identity must be first
    assert_eq!(rots[0], Matrix3::identity());

    // FCC
    let fcc = 0.5 * a * Matrix3::new(0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0);
    let rots_fcc = input_lattice_rotations(&fcc, 1e-5).unwrap();
    assert_eq!(rots_fcc.len(), 48);

    // BCC
    let bcc = 0.5 * a * Matrix3::new(-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0);
    let rots_bcc = input_lattice_rotations(&bcc, 1e-5).unwrap();
    assert_eq!(rots_bcc.len(), 48);
}

#[test]
fn test_lattice_rotations_hexagonal() {
    let a = 3.0;
    let c = 5.2;
    let hex = Matrix3::new(
        a, 0.0, 0.0,
        -a / 2.0, a * 3.0f64.sqrt() / 2.0, 0.0,
        0.0, 0.0, c,
    );
    let rots = input_lattice_rotations(&hex, 1e-5).unwrap();
    assert_eq!(rots.len(), 24);
}

#[test]
fn test_lattice_rotations_tetragonal() {
    let a = 3.0;
    let c = 4.5;
    let tet = Matrix3::new(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, c);
    let rots = input_lattice_rotations(&tet, 1e-5).unwrap();
    assert_eq!(rots.len(), 16);
}

#[test]
fn test_lattice_rotations_monoclinic() {
    let mono = Matrix3::new(3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.2, 0.0, 5.0);
    let rots = input_lattice_rotations(&mono, 1e-5).unwrap();
    assert_eq!(rots.len(), 4);
}

#[test]
fn test_lattice_rotations_triclinic() {
    let tri = Matrix3::new(3.1, 0.2, 0.3, 0.4, 4.2, 0.1, 0.5, 0.6, 5.3);
    let rots = input_lattice_rotations(&tri, 1e-5).unwrap();
    assert_eq!(rots.len(), 2); // {I, -I}
}

#[test]
fn test_lattice_rotations_skewed_supercells() {
    let a = 3.0;
    let base = Matrix3::new(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
    let skew = Matrix3::new(1, 0, 0, 3, 1, 0, 5, 2, 1);
    let skew_f64 = Matrix3::from_fn(|r, c| skew[(r, c)] as f64);
    let cell = skew_f64 * base;

    let rots = input_lattice_rotations(&cell, 1e-5).unwrap();
    assert_eq!(rots.len(), 48);

    // Large skew with entries up to 273
    let big_skew = Matrix3::new(1, 4, 0, 0, 1, 7, 0, 0, 1);
    let big_skew_f64 = Matrix3::from_fn(|r, c| big_skew[(r, c)] as f64);
    let cell_big = big_skew_f64 * (skew_f64 * base);
    let rots_big = input_lattice_rotations(&cell_big, 1e-5).unwrap();
    assert_eq!(rots_big.len(), 48);
}

#[test]
fn test_lattice_validation_errors() {
    let zero = Matrix3::zeros();
    assert!(validate_cell(&zero, 1e-5).is_err());
    assert!(input_lattice_rotations(&zero, 1e-5).is_err());

    let nan_cell = Matrix3::new(f64::NAN, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    assert!(validate_cell(&nan_cell, 1e-5).is_err());

    let good = Matrix3::new(3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0);
    assert!(validate_cell(&good, -1.0).is_err());
    assert!(validate_cell(&good, 0.0).is_err());
    assert!(validate_cell(&good, f64::NAN).is_err());

    // tol exceeding 0.1 x cell height
    assert!(input_lattice_rotations(&good, 1.0).is_err());
}
