//! Unit conversion of results (`units`) and the values the run log and the ASE results derive
//! (`derived`): each against a value computed by hand.
use mandacaru_core::{
    derived,
    units::{self, EnergyConversion, Quantity, BOHR_TO_ANGSTROM, HARTREE_TO_EV},
};

#[test]
fn results_convert_to_user_units_with_their_sign_cells_and_additions() {
    let energy = units::to_user(&[0.5], Quantity::Energy, false, 1).unwrap()[0];
    assert_eq!(energy, 0.5 * HARTREE_TO_EV);
    assert_eq!(units::to_user(&[0.5], Quantity::Energy, true, 1).unwrap()[0], 0.5);
    // A gradient becomes a force by its sign, as (-g) * factor.
    let force = units::to_user(&[0.1], Quantity::GradientToForce, false, 1).unwrap()[0];
    assert_eq!(force, (-0.1) * (HARTREE_TO_EV / BOHR_TO_ANGSTROM));
    assert_eq!(units::to_user(&[0.1], Quantity::GradientToForce, true, 1).unwrap()[0], -0.1);
    let stress = units::to_user(&[1.], Quantity::Stress, false, 1).unwrap()[0];
    assert_eq!(stress, HARTREE_TO_EV / BOHR_TO_ANGSTROM.powf(3.));
    // A supercell energy per primitive cell: (E * factor) * (1 / cells).
    let per_cell = units::to_user(&[-2.], Quantity::Energy, false, 4).unwrap()[0];
    assert_eq!(per_cell, (-2. * HARTREE_TO_EV) * (1. / 4.));
    let plus = units::to_user_values(&[0.1, 0.2], Quantity::Force, true, 1, Some(&[1., 2.])).unwrap();
    assert_eq!(plus, vec![1.1, 2.2]);
    assert_eq!(units::to_user_total(&[1., 2., 0.5], Quantity::Energy, true, 1).unwrap(), 3.5);
    assert!(units::to_user(&[1.], Quantity::Energy, false, 0).is_err());
    // A spread is a squared length; a response function takes one over the energy factor.
    assert_eq!(units::to_user(&[2.], Quantity::Area, false, 1).unwrap()[0], 2. * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM));
    assert_eq!(units::to_user(&[2.], Quantity::InverseEnergy, false, 1).unwrap()[0], 2. * (1. / HARTREE_TO_EV));
    assert_eq!(units::to_user(&[2.], Quantity::InverseEnergy, true, 1).unwrap()[0], 2.);
    assert!(Quantity::parse("volume").is_err());
}

#[test]
fn energy_conversions_keep_the_operation_of_each_direction() {
    let x = 0.123_456_789;
    assert_eq!(EnergyConversion::FromHartree(units::energy_factor("eV").unwrap()).apply(x), x * HARTREE_TO_EV);
    assert_eq!(EnergyConversion::ToHartree(units::energy_factor("eV").unwrap()).apply(x), x / HARTREE_TO_EV);
    assert_eq!(EnergyConversion::between("eV", "Ha").unwrap().apply(x), x * (1. / HARTREE_TO_EV));
    // Equal units are the identity, not a multiplication by a ratio near one.
    assert_eq!(EnergyConversion::between("Ha", "hartree").unwrap().apply(x), x);
    assert_eq!(units::length_factor_to_bohr("Angstrom").unwrap(), 1. / BOHR_TO_ANGSTROM);
    assert!(units::energy_factor("kcal").is_err());
    assert!(units::length_factor_to_bohr("nm").is_err());
}

#[test]
fn a_force_summary_reports_norms_their_maximum_rms_and_the_net_force() {
    let forces = [[3., 4., 0.], [0., 0., -1.], [-1., -2., 0.]];
    let summary = derived::force_summary(&forces).unwrap();
    assert_eq!(summary.norms, vec![5., 1., 5f64.sqrt()]);
    assert_eq!(summary.max, 5.);
    assert!((summary.rms - ((25. + 1. + 5.) / 3f64).sqrt()).abs() < 1e-15);
    // The sum is (2, 2, -1): its largest component.
    assert_eq!(summary.net, 2.);
    assert!(derived::force_summary(&[]).is_err());
    assert!(derived::force_summary(&[[f64::NAN, 0., 0.]]).is_err());
}

#[test]
fn an_iteration_row_applies_the_solver_strict_tests() {
    let first = derived::iteration_values(-1.0, None, -0.02, Some(0.01), Some(1e-3)).unwrap();
    assert_eq!(first.delta, None);
    assert_eq!(first.gradient_ev, 0.02 * HARTREE_TO_EV);
    assert_eq!(first.gradient_met, Some(false));
    assert_eq!(first.energy_met, Some(false));
    let later = derived::iteration_values(-1.0005, Some(-1.0), 0.005, Some(0.01), Some(1e-3)).unwrap();
    assert!((later.delta.unwrap() + 5e-4).abs() < 1e-15);
    assert_eq!(later.gradient_met, Some(true));
    assert_eq!(later.energy_met, Some(true));
    // At the threshold exactly the strict test fails; no threshold, no verdict.
    let edge = derived::iteration_values(-1.0, Some(-1.0), 0.01, Some(0.01), None).unwrap();
    assert_eq!(edge.gradient_met, Some(false));
    assert_eq!(edge.energy_met, None);
}

#[test]
fn cell_parameters_lengths_relative_energies_and_electron_counts() {
    let cubic = derived::cell_parameters([[2., 0., 0.], [0., 3., 0.], [0., 0., 4.]]).unwrap();
    assert_eq!(cubic, [2., 3., 4., 90., 90., 90.]);
    let hexagonal = derived::cell_parameters([[1., 0., 0.], [-0.5, 3f64.sqrt() / 2., 0.], [0., 0., 2.]]).unwrap();
    assert!((hexagonal[5] - 120.).abs() < 1e-12);
    assert_eq!(derived::cell_parameters([[0.; 3], [0., 1., 0.], [0., 0., 1.]]).unwrap()[4], 0.);
    assert_eq!(derived::relative_to_first(&[-3., -2.5, -1.]), vec![0., 0.5, 2.]);
    assert_eq!(derived::gaps(&[-3., -2.5, -1.]), vec![0.5, 1.5]);
    assert_eq!(derived::difference_to_user(-1.2, -1.0, Quantity::Energy, true).unwrap(), -1.2 - -1.0);
    assert_eq!(derived::total(&[1., 2., 3.]), 6.);
    // Python's `sum([1e16, 1.0, -1e16])` is 1.0 (compensated, Python 3.12+); added in order it is 0.0.
    assert_eq!(derived::total(&[1e16, 1., -1e16]), 1.);
    let mesh = derived::mesh_fractions([2, 1, 3]).unwrap();
    assert_eq!(mesh.len(), 6);
    assert_eq!(mesh[1], [0., 0., 1. / 3.]);
    assert_eq!(mesh[5], [0.5, 0., 2. / 3.]);
    assert!(derived::mesh_fractions([2, 0, 3]).is_err());
    assert_eq!(derived::norm(&[3., 4.]), 5.);
    assert_eq!(derived::distance(&[1., 1., 1.], &[1., 4., 5.]).unwrap(), 5.);
    // Two k-points of weight 1/2, two bands; restricted splits its total in halves.
    let restricted = derived::electron_counts(&[2., 2., 2., 0.], 1, 2, 2, &[0.5, 0.5]).unwrap();
    assert_eq!(restricted, [1.5, 1.5]);
    let polarized = derived::electron_counts(&[1., 1., 1., 0., 1., 0., 1., 0.], 2, 2, 2, &[0.5, 0.5]).unwrap();
    assert_eq!(polarized, [1.5, 1.]);
    assert!(derived::electron_counts(&[1.], 3, 1, 1, &[1.]).is_err());
}

#[test]
fn the_bloch_property_helpers_follow_their_definitions() {
    assert_eq!(derived::uniform_weights(4).unwrap(), vec![0.25; 4]);
    assert!(derived::uniform_weights(0).is_err());
    // (spins, bands) = (2, 3): spin 0 holds 2 electrons, spin 1 holds 1.
    let levels = [-0.5, -0.2, 0.1, -0.6, -0.1, 0.2];
    assert_eq!(derived::highest_occupied(&levels, 3, &[2, 1]).unwrap(), -0.2);
    assert_eq!(derived::highest_occupied(&levels, 3, &[0, 2]).unwrap(), -0.1);
    assert!(derived::highest_occupied(&levels, 3, &[0, 0]).is_err());
    // An even axis: start + i * step, the end exactly the last point.
    let axis = derived::energy_axis(&[-1. / HARTREE_TO_EV, 1. / HARTREE_TO_EV], 5, 0.5, 2.).unwrap();
    assert_eq!(axis.len(), 5);
    assert!((axis[0] + 2.).abs() < 1e-12 && axis[4] == (1. / HARTREE_TO_EV) * HARTREE_TO_EV + 1.);
    assert!((axis[2] - 0.).abs() < 1e-12);
    let labels: Vec<String> = ["s", "p", "s"].iter().map(|s| s.to_string()).collect();
    let (names, sums) = derived::column_groups(&[1., 2., 3., 4., 5., 6.], 2, &labels).unwrap();
    assert_eq!(names, vec!["s".to_string(), "p".to_string()]);
    assert_eq!(sums, vec![vec![4., 10.], vec![2., 5.]]);
    let moved = derived::displaced(vec![[0.; 3], [1., 1., 1.]], 1, 2, -1., 0.25).unwrap();
    assert_eq!(moved[1], [1., 1., 0.75]);
    assert!(derived::displaced(vec![[0.; 3]], 1, 0, 1., 0.1).is_err());
}

#[test]
fn grid_geometry_ratios_and_tolerance_digits() {
    let (dv, cell, volume) = derived::grid_cell([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.25]], [4, 2, 8]).unwrap();
    assert_eq!(dv, 0.0625);
    assert_eq!(cell, [[2., 0., 0.], [0., 1., 0.], [0., 0., 2.]]);
    assert_eq!(volume, 4.);
    assert!(derived::grid_cell([[1., 0., 0.], [1., 0., 0.], [0., 0., 1.]], [1, 1, 1]).is_err());
    let points = derived::grid_coordinates([1., 0., 0.], [[0.5, 0., 0.], [0., 1., 0.], [0., 0., 2.]], [2, 1, 2]);
    // (3, 4): x of (0,0,0), (0,0,1), (1,0,0), (1,0,1); then y; then z.
    assert_eq!(points, vec![1., 1., 1.5, 1.5, 0., 0., 0., 0., 0., 2., 0., 2.]);
    assert_eq!(derived::ratio(3., 4.).unwrap(), 0.75);
    assert!(derived::ratio(1., 0.).is_err());
    assert_eq!(derived::decimal_digits(1e-10).unwrap(), 10);
    assert_eq!(derived::decimal_digits(5.).unwrap(), 0);
    assert!(derived::decimal_digits(0.).is_err());
}

#[test]
fn free_particle_relations_invert_each_other() {
    // k^2 / 2 = E: one Hartree is k = sqrt(2) per Bohr.
    let k = derived::free_wavevector(HARTREE_TO_EV).unwrap();
    assert!((k - 2f64.sqrt()).abs() < 1e-15);
    assert!((derived::free_energy_ev(k).unwrap() - HARTREE_TO_EV).abs() < 1e-12);
    // The confinement radius is half the free wavelength at the shift: pi / k.
    let r = derived::energy_shift_radius(HARTREE_TO_EV / 2.).unwrap();
    assert!((r - std::f64::consts::PI).abs() < 1e-14);
    assert_eq!(derived::nyquist_wavevector(0.5, std::f64::consts::PI).unwrap(), 0.5);
    for bad in [0., -1., f64::NAN] {
        assert!(derived::free_wavevector(bad).is_err());
        assert!(derived::energy_shift_radius(bad).is_err());
        assert!(derived::nyquist_wavevector(0.5, bad).is_err());
    }
    assert!(derived::free_energy_ev(f64::INFINITY).is_err());
}

#[test]
fn workflow_values_are_the_numbers_a_spectrum_and_an_echo_report() {
    assert_eq!(derived::descending_weights(3).unwrap(), vec![3., 2., 1.]);
    assert!(derived::descending_weights(0).is_err());
    assert_eq!(derived::max_abs(&[0.5, -2., 1.]), 2.);
    assert_eq!(derived::max_abs(&[]), 0.);
    assert!(derived::max_abs(&[1., f64::NAN]).is_nan());
    assert_eq!(derived::sample_times(3, 0.5), vec![0., 0.5, 1.]);
    // 2 pi / (4 * 0.5) = pi and pi / 0.5 = 2 pi.
    assert_eq!(derived::fourier_bin_spacing(4, 0.5), std::f64::consts::PI);
    assert_eq!(derived::nyquist_frequency(0.5), 2. * std::f64::consts::PI);
    let amplitudes = [num_complex::Complex64::new(3., 4.), num_complex::Complex64::new(0., -2.)];
    assert_eq!(derived::magnitudes(&amplitudes), vec![5., 2.]);
    assert_eq!(derived::squared(3.), 9.);
    assert_eq!(derived::modulus_squared(3., 4.), 25.);
    assert_eq!(derived::pulse_bound(-0.5, 4.), 2.);
    // Python's sum is compensated: it recovers the 1.0 a running sum loses against 1e16.
    assert_eq!(derived::python_sum(&[1e16, 1., 1., -1e16]), 2.);
    assert_eq!([1e16, 1., 1., -1e16].iter().sum::<f64>(), 0.);
    assert_eq!(derived::python_sum(&[]), 0.);
}
