//! B3 (`periodic::berry::perturbation`): the folding of polarization steps (an electronic branch
//! jump and an ion crossing the cell boundary carry whole quanta), the quarter-quantum refusal,
//! the assembly of Born charges and of the proper piezoelectric tensor from folded steps against
//! sums written out by hand, the Richardson combination, the acoustic sum rule (reported, imposed
//! on request), the Voigt strains, and the setup perturber's refusals. Model-free and fast, except
//! the last test, which builds a crystal (`MANDACARU_PAW_PATH`).
mod berry_support;
mod setup_shells_support;

use mandacaru_core::periodic::berry::{
    perturbation::{
        acoustic_sum_rule, born_charges, channel_steps, fold, folded_step, impose, piezoelectric,
        richardson, voigt_strain, Perturbation, Perturber, SetupPerturber, ASR_WARNING,
    },
    polarization::assemble,
};
use nalgebra::Matrix3;
use std::f64::consts::PI;

fn cell() -> Matrix3<f64> {
    Matrix3::new(6., 0.5, 0., 0., 7., 0.3, 0., 0., 8.)
}

#[test]
fn a_whole_quantum_from_a_branch_jump_or_an_ion_crossing_the_boundary_folds_away() {
    let lattice = cell();
    let charges = [1., 1.];
    // A branch jump: the same physical state with one string phase shifted by 2 pi.
    let minus = assemble(
        lattice,
        &[[0.5, 1., 1.], [3., 3., 4.]],
        &charges,
        2.,
        vec![[0.3, -0.2, 0.1]],
    )
    .unwrap();
    let plus = assemble(
        lattice,
        &[[0.5, 1., 1.], [3., 3., 4.]],
        &charges,
        2.,
        vec![[0.3 + 1e-3 - 2. * PI, -0.2, 0.1]],
    )
    .unwrap();
    let step = folded_step(&plus, &minus, "jump").unwrap();
    let expected = -2. * 1e-3 / (2. * PI); // f = 2, g = gcd(2, 1, 1) = 1
    assert!((step[0] - expected).abs() < 1e-14, "{step:?}");
    assert!(step[1].abs() < 1e-15 && step[2].abs() < 1e-15);
    // An ion crossing the boundary: atom 0 at s = -1e-4 against s = 1 - 1e-4 changes `raw` by
    // exactly Z_0 / g = 1 quantum along a_0, nothing after folding.
    let a0 = lattice.column(0);
    let near = |s: f64| [s * a0[0] + 1., s * a0[1] + 1., s * a0[2] + 1.];
    let inside = assemble(
        lattice,
        &[near(1. - 1e-4), [3., 3., 4.]],
        &charges,
        2.,
        vec![[0.3, -0.2, 0.1]],
    )
    .unwrap();
    let outside = assemble(
        lattice,
        &[near(-1e-4), [3., 3., 4.]],
        &charges,
        2.,
        vec![[0.3, -0.2, 0.1]],
    )
    .unwrap();
    assert!((inside.raw[0] - outside.raw[0] - 1.).abs() < 1e-9);
    let crossed = folded_step(&inside, &outside, "crossing").unwrap();
    assert!(crossed.iter().all(|d| d.abs() < 1e-9), "{crossed:?}");
    assert_eq!(fold(0.7), 0.7 - 1.);
}

#[test]
fn a_step_beyond_a_quarter_quantum_is_refused() {
    let lattice = cell();
    let a = assemble(lattice, &[[1., 1., 1.]], &[1.], 2., vec![[0., 0., 0.]]).unwrap();
    let b = assemble(
        lattice,
        &[[1., 1., 1.]],
        &[1.],
        2.,
        vec![[2. * PI * 0.15, 0., 0.]],
    )
    .unwrap();
    // f = 2: the electronic coordinate moves by -0.3 of a quantum.
    let message = folded_step(&b, &a, "atom 0").unwrap_err().to_string();
    assert!(
        message.contains("atom 0") && message.contains("more than a quarter"),
        "{message}"
    );
}

#[test]
fn channel_steps_fold_modulo_the_channel_period() {
    let lattice = cell();
    // Two spin channels (f = 1), g = gcd(1, 1) = 1: a channel's coordinate is continuous modulo 1.
    let minus = assemble(
        lattice,
        &[[1., 1., 1.]],
        &[1.],
        1.,
        vec![[0.1, 0., 0.], [0.2, 0., 0.]],
    )
    .unwrap();
    let plus = assemble(
        lattice,
        &[[1., 1., 1.]],
        &[1.],
        1.,
        vec![[0.1 + 2e-3, 0., 0.], [0.2 - 2. * PI + 1e-3, 0., 0.]],
    )
    .unwrap();
    let steps = channel_steps(&plus, &minus);
    assert!((steps[0][0] + 2e-3 / (2. * PI)).abs() < 1e-14);
    assert!((steps[1][0] + 1e-3 / (2. * PI)).abs() < 1e-14);
}

#[test]
fn born_charges_and_the_piezoelectric_tensor_assemble_as_written_by_hand() {
    let lattice = cell();
    let p = assemble(lattice, &[[1., 1., 1.]], &[1.], 2., vec![[0., 0., 0.]]).unwrap();
    let (quanta, volume) = (p.quanta, p.volume);
    let mut steps = vec![[[0.; 3]; 3]; 2];
    steps[0][1] = [1e-3, -2e-3, 5e-4];
    steps[1][2] = [-3e-4, 0., 7e-4];
    let delta = 0.02;
    let z = born_charges(&quanta, volume, &steps, delta);
    for (a, j) in [(0, 1), (1, 2)] {
        for i in 0..3 {
            let dp: f64 = (0..3)
                .map(|alpha| steps[a][j][alpha] * quanta[alpha][i])
                .sum();
            assert!((z[a][i][j] - volume * dp / (2. * delta)).abs() < 1e-15);
        }
    }
    assert_eq!(z[0][0][0], 0.);
    // The quanta are g a_j / Omega: one quantum along a_0 per unit displacement gives Z* = a_0.
    let mut unit = vec![[[0.; 3]; 3]; 1];
    unit[0][0] = [2. * delta, 0., 0.];
    let z = born_charges(&quanta, volume, &unit, delta);
    for i in 0..3 {
        assert!((z[0][i][0] - lattice[(i, 0)]).abs() < 1e-13);
    }
    let mut strain_steps = [[0.; 3]; 6];
    strain_steps[3] = [0., 4e-4, -1e-4];
    let e = piezoelectric(&quanta, &strain_steps, 0.005);
    for i in 0..3 {
        let dp: f64 = (0..3)
            .map(|alpha| strain_steps[3][alpha] * quanta[alpha][i])
            .sum();
        assert!((e[i][3] - dp / 0.01).abs() < 1e-15);
        assert_eq!(e[i][0], 0.);
    }
}

#[test]
fn richardson_removes_the_second_order_error_of_a_central_difference() {
    // A central difference of f(x) = x^3 + x^5 at 0: f'(0) + h^2 f'''/6 + O(h^4) = h^2 + h^4.
    let central = |h: f64| h * h + h.powi(4);
    let r = richardson(central(0.1), central(0.05));
    assert!(r.abs() < 1e-4 && r.abs() < central(0.05) / 10., "{r}");
}

#[test]
fn the_acoustic_sum_rule_is_reported_and_imposed_only_on_request() {
    let z = vec![
        [[1.1, 0.01, 0.], [0., 1.1, 0.], [0., 0., 1.1]],
        [[-1.0, 0., 0.], [0., -1.0, 0.], [0., 0.02, -1.0]],
    ];
    let rule = acoustic_sum_rule(&z);
    assert!((rule.residual - 0.1).abs() < 1e-14);
    assert!(rule.residual > ASR_WARNING && rule.warning.as_ref().unwrap().contains("0.100 e"));
    let fixed = impose(&z, &rule);
    assert!(acoustic_sum_rule(&fixed).residual < 1e-15);
    assert!(acoustic_sum_rule(&fixed).warning.is_none());
    // Imposing removes the mean from every atom and keeps their differences.
    assert!(((fixed[0][0][0] - fixed[1][0][0]) - (z[0][0][0] - z[1][0][0])).abs() < 1e-15);
}

#[test]
fn voigt_strains_are_symmetric_with_engineering_shear() {
    let xx = voigt_strain(0, 0.01).unwrap();
    assert_eq!(xx[(0, 0)], 0.01);
    let yz = voigt_strain(3, 0.01).unwrap();
    assert_eq!((yz[(1, 2)], yz[(2, 1)]), (0.005, 0.005));
    let xy = voigt_strain(5, 0.01).unwrap();
    assert_eq!((xy[(0, 1)], xy[(1, 0)]), (0.005, 0.005));
    assert!(voigt_strain(6, 0.01).is_err());
    assert!(Perturbation::Strain {
        voigt: 4,
        delta: -0.005
    }
    .name()
    .contains("xz"));
}

#[test]
fn the_setup_perturber_strains_a_cell_without_symmetry_and_refuses_a_displacement() {
    // The H2 chain of the Berry tests, built with symmetry=False.
    let solved = berry_support::solve(&berry_support::h2_chain("lda-sr", "lda"));
    let perturber = SetupPerturber {
        setup: &solved.setup,
    };
    let strained = perturber
        .setup(
            &solved.ctx,
            Perturbation::Strain {
                voigt: 0,
                delta: 0.005,
            },
        )
        .unwrap();
    assert!((strained.lattice[(0, 0)] - 1.005 * solved.setup.lattice[(0, 0)]).abs() < 1e-12);
    let error = perturber
        .setup(
            &solved.ctx,
            Perturbation::Displacement {
                atom: 0,
                axis: 0,
                delta: 0.01,
            },
        )
        .err()
        .expect("a displacement needs PeriodicSetup::with_positions");
    assert!(
        matches!(error, mandacaru_core::Error::Unsupported(_)),
        "{error:?}"
    );
}
