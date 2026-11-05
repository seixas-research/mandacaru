use mandacaru_core::paw::basis_blocks::{
    atom_multipole_blocks, multipole_transform, projector_basis_transform, projector_sphere,
    ProjectorSpec,
};
use nalgebra::DMatrix;
use std::collections::BTreeMap;
use std::f64::consts::PI;

#[test]
fn projector_basis_transform_identity() {
    let b = vec![1.0, 0.0, 0.0, 1.0];
    let d = vec![2.0, 0.5, 0.5, 3.0];
    let q = vec![1.0, 0.1, 0.1, 1.0];
    let (d_out, q_out) = projector_basis_transform(&b, &d, &q, 2, "raw").unwrap();
    for (a, b) in d.iter().zip(d_out.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    for (a, b) in q.iter().zip(q_out.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn projector_basis_transform_scaled() {
    let b = vec![2.0, 0.0, 0.0, 2.0];
    let d = vec![4.0, 0.0, 0.0, 4.0];
    let q = vec![4.0, 0.0, 0.0, 4.0];
    // B^-1 = 0.5 * I -> B^-1 D B^-T = 0.25 * 4.0 = 1.0
    let (d_out, q_out) = projector_basis_transform(&b, &d, &q, 2, "raw").unwrap();
    assert!((d_out[0] - 1.0).abs() < 1e-12);
    assert!((d_out[3] - 1.0).abs() < 1e-12);
    assert!((q_out[0] - 1.0).abs() < 1e-12);
    assert!((q_out[3] - 1.0).abs() < 1e-12);
}

#[test]
fn multipole_transform_evaluates_inverse_sandwich() {
    let b1 = vec![2.0, 0.0, 0.0, 2.0];
    let b2 = vec![4.0, 0.0, 0.0, 4.0];
    let delta = vec![8.0, 0.0, 0.0, 8.0];
    // B1^-1 = 0.5, B2^-T = 0.25 -> 0.5 * 8.0 * 0.25 = 1.0
    let out = multipole_transform(&b1, 2, &b2, 2, &delta).unwrap();
    assert!((out[0] - 1.0).abs() < 1e-12);
    assert!((out[3] - 1.0).abs() < 1e-12);
}

#[test]
fn multipole_transform_preserves_rectangular_matrix_order() {
    let b1 = [2., 1., 0., 1.];
    let b2 = [1., 0., 1., 0., 2., 1., 0., 0., 3.];
    let delta = [1., 2., 4., 8., 16., 32.];
    let raw = multipole_transform(&b1, 2, &b2, 3, &delta).unwrap();
    let reconstructed = DMatrix::from_row_slice(2, 2, &b1)
        * DMatrix::from_row_slice(2, 3, &raw)
        * DMatrix::from_row_slice(3, 3, &b2).transpose();
    let expected = DMatrix::from_row_slice(2, 3, &delta);
    assert!(
        (&reconstructed - expected).norm() < 1e-12,
        "{reconstructed}"
    );
}

#[test]
fn projector_transform_preserves_general_matrix_order() {
    let b = [2., 1., 0., 1.];
    let d = [1., 2., 3., 5.];
    let q = [4., 6., 7., 9.];
    let (raw_d, raw_q) = projector_basis_transform(&b, &d, &q, 2, "raw").unwrap();
    let b_matrix = DMatrix::from_row_slice(2, 2, &b);
    for (raw, expected) in [(raw_d, d), (raw_q, q)] {
        let reconstructed = &b_matrix * DMatrix::from_row_slice(2, 2, &raw) * b_matrix.transpose();
        assert!((reconstructed - DMatrix::from_row_slice(2, 2, &expected)).norm() < 1e-12);
    }
}

#[test]
fn projector_sphere_integrates_volume() {
    let center = [0.0, 0.0, 0.0];
    let r_cut = 1.5;
    let (_points, weights) = projector_sphere(center, r_cut, 0.5, 8, 16, 32).unwrap();
    let total_volume: f64 = weights.iter().sum();
    let expected_volume = 4.0 / 3.0 * PI * r_cut.powi(3);
    assert!(
        (total_volume - expected_volume).abs() < 1e-4,
        "volume {total_volume} vs {expected_volume}"
    );
}

#[test]
fn atom_multipole_blocks_s_channel() {
    let projectors = vec![ProjectorSpec {
        l: 0,
        m: 0,
        index: 0,
    }];
    let mut deltas = BTreeMap::new();
    deltas.insert((0, 0, 0), vec![1.5]);
    let mut strides = BTreeMap::new();
    strides.insert((0, 0), (1, 1));

    let blocks = atom_multipole_blocks(&projectors, true, &deltas, &strides).unwrap();
    assert_eq!(blocks.len(), 1);
    let (l, m, ref mat) = blocks[0];
    assert_eq!(l, 0);
    assert_eq!(m, 0);
    assert_eq!(mat.len(), 1);
    // Y00 = 1 / sqrt(4 pi). Gaunt integral for L=0, l1=0, l2=0 is 1 / sqrt(4 pi)
    let expected = 1.5 / (4.0 * PI).sqrt();
    assert!((mat[0].re - expected).abs() < 1e-12);
}
