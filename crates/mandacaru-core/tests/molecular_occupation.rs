//! Occupations of a molecular Kohn-Sham channel (`molecular::occupation`): Fermi-Dirac with one
//! Fermi level per spin, exactly aufbau on a gapped molecule.
use mandacaru_core::molecular::occupation::{
    aufbau, density, entropy, fermi_dirac, is_integer, non_aufbau, weighted_density, DEFAULT_WIDTH,
};
use mandacaru_core::quantum::C64;
use nalgebra::DMatrix;

#[test]
fn a_gapped_frontier_is_exactly_aufbau() {
    // A gap of 0.008 Ha is 80 widths: every occupation is exactly 0 or 1, so the density keeps
    // the determinant's arithmetic and the entropy is exactly zero.
    let levels = [-1.0, -0.5, -0.304, -0.296, 0.1];
    let f = fermi_dirac(&levels, 3, DEFAULT_WIDTH).unwrap();
    assert_eq!(f, aufbau(5, 3));
    assert_eq!(entropy(&f), 0.);
    assert_eq!(fermi_dirac(&levels, 0, DEFAULT_WIDTH).unwrap(), vec![0.; 5]);
    assert_eq!(fermi_dirac(&levels, 5, DEFAULT_WIDTH).unwrap(), vec![1.; 5]);
}

#[test]
fn an_exactly_degenerate_shell_is_shared_equally() {
    // Restricted O2: five full levels, then the pi* pair holding one electron of each spin.
    let levels = [-1.2, -0.8, -0.5, -0.45, -0.45, -0.21, -0.21, 0.1];
    let f = fermi_dirac(&levels, 6, DEFAULT_WIDTH).unwrap();
    assert_eq!(&f[..5], &[1.; 5]);
    assert_eq!(f[5], f[6]);
    assert!((f[5] - 0.5).abs() < 1e-15);
    assert_eq!(f[7], 0.);
    assert!(!is_integer(&f));
    // The two half-filled orbitals carry ln 2 each.
    assert!((entropy(&f) - 2. * std::f64::consts::LN_2).abs() < 1e-14);
    // A threefold shell holding one electron of the spin: a third in each member.
    let p = fermi_dirac(&[-0.9, -0.3, -0.3, -0.3, 0.2], 2, DEFAULT_WIDTH).unwrap();
    for f in &p[1..4] {
        assert!((f - 1. / 3.).abs() < 1e-14);
    }
}

#[test]
fn a_split_shell_follows_fermi_dirac_and_holds_the_count_exactly() {
    // A pair split by two widths: the occupations are the Fermi-Dirac ones about the level that
    // holds exactly one electron, 1 / (1 + e^{+-1}).
    let levels = [
        -0.38,
        -0.2425 - DEFAULT_WIDTH,
        -0.2425 + DEFAULT_WIDTH,
        0.04,
    ];
    let f = fermi_dirac(&levels, 2, DEFAULT_WIDTH).unwrap();
    assert_eq!(f[0], 1.);
    assert_eq!(f[3], 0.);
    assert!((f[1] - 1. / (1. + (-1f64).exp())).abs() < 1e-9);
    assert!((f[1] + f[2] - 1.).abs() < 1e-15);
    assert!((f.iter().sum::<f64>() - 2.).abs() < 1e-15);
    // Levels in any order give the same occupation per level.
    let shuffled = [levels[2], levels[0], levels[3], levels[1]];
    let g = fermi_dirac(&shuffled, 2, DEFAULT_WIDTH).unwrap();
    assert!((g[0] - f[2]).abs() < 1e-15 && (g[3] - f[1]).abs() < 1e-15);
    assert!(fermi_dirac(&levels, 2, 0.).is_err());
    assert!(fermi_dirac(&levels, 5, DEFAULT_WIDTH).is_err());
}

#[test]
fn an_ensemble_density_is_the_weighted_sum_and_aufbau_keeps_the_projector() {
    let m = 4;
    let c = DMatrix::from_fn(m, m, |i, j| {
        C64::new(((i + 1) as f64 * (j + 2) as f64).sin(), 0.)
    });
    let q = c.qr().q();
    let projector = q.columns(0, 2) * q.columns(0, 2).adjoint();
    let aufbau_density = density(&q, &aufbau(m, 2));
    assert!((aufbau_density - &projector).norm() < 1e-14);
    let f = [1., 0.5, 0.5, 0.];
    let ensemble = density(&q, &f);
    assert!((&ensemble - weighted_density(&q, &f)).norm() < 1e-14);
    // The eigenvalues of the ensemble density are its occupations.
    let mut values: Vec<f64> = ensemble
        .map(|v| v.re)
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .collect();
    values.sort_by(f64::total_cmp);
    for (value, expected) in values.iter().zip([0., 0.5, 0.5, 1.]) {
        assert!((value - expected).abs() < 1e-12);
    }
    assert!((ensemble.trace().re - 2.).abs() < 1e-12);
}

#[test]
fn the_aufbau_test_flags_an_occupied_level_above_an_emptier_one() {
    let levels = [-1.0, 0.2, -0.1, 0.5];
    assert_eq!(non_aufbau(&levels, &aufbau(4, 2), 1e-6), Some((1, 2)));
    assert_eq!(
        non_aufbau(&[-1.0, -0.1, 0.2, 0.5], &aufbau(4, 2), 1e-6),
        None
    );
    // Fermi-Dirac occupations decrease with the level and never form a pair.
    let fd = [-1.0, -0.2425, -0.2423, 0.5];
    let f = fermi_dirac(&fd, 2, DEFAULT_WIDTH).unwrap();
    assert_eq!(non_aufbau(&fd, &f, 1e-6), None);
}
