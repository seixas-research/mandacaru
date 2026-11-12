//! The Hall assembly of `periodic::berry::hall` without a crystal: the Chern vector
//! `K = sign(det G) sum_c C_c G_c` and `sigma_ab = eps_abd K_d / (4 pi^2)` (e^2/(hbar a0)), gated nonzero.
//! `K` carries `sign(det G)`, so a left-handed relabeling of one crystal gives the same tensor.
//! A stack of Chern-1 layers normal to `G_2` conducts `e^2/h` per layer: `sigma_xy` times the layer
//! spacing is `1/(2 pi)` in atomic units; in a skew cell `sigma_ab = eps_abd n_d / (2 pi V)` with
//! `n = a_0 x a_1`, computed here from the real-space lattice, independently of `G`.
use mandacaru_core::{periodic::berry::hall::conductivity, units::Quantity};
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;

fn reciprocal(lattice: &Matrix3<f64>) -> Matrix3<f64> {
    2. * PI * lattice.try_inverse().unwrap().transpose()
}

#[test]
fn a_stack_of_chern_layers_conducts_one_quantum_per_layer() {
    let (a, b, c) = (5., 6., 7.);
    let lattice = Matrix3::from_diagonal(&Vector3::new(a, b, c));
    let (k, sigma) = conductivity(&reciprocal(&lattice), [0, 0, 1]);
    assert!((k[2] - 2. * PI / c).abs() < 1e-14 && k[0] == 0. && k[1] == 0.);
    assert!(
        (sigma[0][1] * c - 1. / (2. * PI)).abs() < 1e-14,
        "{}",
        sigma[0][1]
    );
    assert_eq!(sigma[1][0], -sigma[0][1]);
    for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 2), (2, 1)] {
        assert!(sigma[i][j].abs() < 1e-15, "({i}, {j})");
    }
    // Chern -2 doubles it with the opposite sign; the planes normal to G_0 give sigma_yz.
    let (_, twice) = conductivity(&reciprocal(&lattice), [0, 0, -2]);
    assert!((twice[0][1] + 2. * sigma[0][1]).abs() < 1e-15);
    let (_, yz) = conductivity(&reciprocal(&lattice), [1, 0, 0]);
    assert!((yz[1][2] * a - 1. / (2. * PI)).abs() < 1e-14);
    // e^2/h per 7 Bohr layer spacing, in S/cm.
    let siemens = 3.874_045_86e-5 / (7. * 0.529_177_210_903e-8);
    assert!((sigma[0][1] * Quantity::Conductivity.factor(false) - siemens).abs() < 1e-3 * siemens);
}

#[test]
fn a_skew_cell_conducts_along_its_layer_normal() {
    let lattice = Matrix3::new(6., 1., 0.5, 0.3, 7., 1.2, 0., 0.4, 8.);
    let volume = lattice.determinant();
    let n = lattice.column(0).cross(&lattice.column(1));
    let (_, sigma) = conductivity(&reciprocal(&lattice), [0, 0, 1]);
    for (a, b, d) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
        let expected = n[d] / (2. * PI * volume);
        assert!((sigma[a][b] - expected).abs() < 1e-14, "({a}, {b})");
        assert_eq!(sigma[b][a], -sigma[a][b]);
    }
    assert!(sigma.iter().enumerate().all(|(i, row)| row[i] == 0.));
}

#[test]
fn relabeling_the_cell_into_a_left_handed_one_leaves_the_conductivity_unchanged() {
    // One crystal: Chern-1 layers in the xy plane, 7 Bohr apart. Right-handed labels
    // (a_0, a_1, a_2) = (5 x, 6 y, 7 z) measure C = +1 on the planes normal to G_2; swapping
    // a_0 <-> a_1 (det < 0) turns G_0 x G_1 to -z, so the same planes measure C = -1.
    let right = Matrix3::from_diagonal(&Vector3::new(5., 6., 7.));
    let left = Matrix3::new(0., 5., 0., 6., 0., 0., 0., 0., 7.);
    assert!(right.determinant() > 0. && left.determinant() < 0.);
    let (k_right, sigma_right) = conductivity(&reciprocal(&right), [0, 0, 1]);
    let (k_left, sigma_left) = conductivity(&reciprocal(&left), [0, 0, -1]);
    let layer = 1. / (2. * PI * 7.);
    assert!(
        (sigma_right[0][1] - layer).abs() < 1e-14,
        "{}",
        sigma_right[0][1]
    );
    assert!((sigma_right[0][1] - 0.022_736).abs() < 1e-6);
    for a in 0..3 {
        assert!((k_right[a] - k_left[a]).abs() < 1e-14, "K_{a}");
        for b in 0..3 {
            assert!(
                (sigma_right[a][b] - sigma_left[a][b]).abs() < 1e-15,
                "({a}, {b})"
            );
        }
    }
}
