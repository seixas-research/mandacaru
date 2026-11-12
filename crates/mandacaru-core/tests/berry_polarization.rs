//! The polarization bookkeeping of `periodic::berry::polarization` without a crystal: the quantum
//! `gcd(f, Z_1, ...)`, the electronic and ionic coordinates, the fold onto the branch nearest zero
//! and the Cartesian vector, on hand-made phases whose answer is known.
use mandacaru_core::{
    periodic::berry::polarization::{assemble, quantum, spread_warnings, SPREAD_WARNING},
    units::{Quantity, BOHR_TO_METER, ELEMENTARY_CHARGE_COULOMB},
};
use nalgebra::Matrix3;
use std::f64::consts::PI;

#[test]
fn the_quantum_is_the_gcd_of_the_band_occupancy_and_the_valence_charges() {
    assert_eq!(quantum(2., &[4., 4.]).unwrap(), 2);
    assert_eq!(quantum(2., &[1., 3.]).unwrap(), 1);
    assert_eq!(quantum(1., &[4., 6.]).unwrap(), 1);
    assert_eq!(quantum(2., &[6., 8.]).unwrap(), 2);
    // A fractional valence leaves only the electrons' quantum.
    assert_eq!(quantum(2., &[1.5, 4.]).unwrap(), 2);
    assert!(quantum(3., &[1.]).is_err());
}

#[test]
fn a_neutral_atom_with_its_electron_on_the_ion_has_no_polarization() {
    // One ion Z = 1 at s = (0.3, 0.6, 0.1) of a skew cell, its electron (one spin channel, f = 1,
    // the other channel empty) centered on it: gamma_j = 2 pi s_j.
    let lattice = Matrix3::new(7., 1., 0.5, 0., 8., 1.5, 0., 0., 9.);
    let s = [0.3, 0.6, 0.1];
    let center: [f64; 3] = (lattice * nalgebra::Vector3::from(s)).into();
    let phases = vec![s.map(|x| 2. * PI * x), [0.; 3]];
    let p = assemble(lattice, &[center], &[1.], 1., phases).unwrap();
    assert_eq!(p.quantum, 1);
    for j in 0..3 {
        assert!(p.raw[j].abs() < 1e-12, "{:?}", p.raw);
        assert!(p.fractional[j].abs() < 1e-12);
        assert!(p.vector[j].abs() < 1e-14);
        assert!((p.ionic[j] - s[j]).abs() < 1e-12);
        assert!((p.electronic[j] + s[j]).abs() < 1e-12);
    }
}

#[test]
fn a_displaced_electron_gives_the_dipole_over_the_volume_on_the_branch_nearest_zero() {
    // Unpolarized (f = 2), Z = 2 at the origin, both electrons at s_x = 0.2: P = -2 e (0.2 a_1)/V.
    let lattice = Matrix3::from_diagonal(&nalgebra::Vector3::new(10., 11., 12.));
    let volume = 10. * 11. * 12.;
    let p = assemble(
        lattice,
        &[[0.; 3]],
        &[2.],
        2.,
        vec![[2. * PI * 0.2, 0., 0.]],
    )
    .unwrap();
    assert_eq!(p.quantum, 2);
    // In quanta of 2 e a_1 / V the coordinate is -0.2.
    assert!((p.fractional[0] + 0.2).abs() < 1e-12, "{:?}", p.fractional);
    assert!((p.vector[0] - (-2. * 0.2 * 10. / volume)).abs() < 1e-14);
    assert!((p.quanta[0][0] - 2. * 10. / volume).abs() < 1e-14);
    // Moving the electrons by one lattice vector changes the phase by 2 pi: one quantum in raw,
    // nothing once folded.
    let shifted = assemble(
        lattice,
        &[[0.; 3]],
        &[2.],
        2.,
        vec![[2. * PI * 1.2, 0., 0.]],
    )
    .unwrap();
    assert!((shifted.raw[0] - p.raw[0] + 1.).abs() < 1e-12);
    assert!((shifted.fractional[0] - p.fractional[0]).abs() < 1e-12);
    // Half a quantum folds to +1/2, never -1/2.
    let half = assemble(
        lattice,
        &[[0.; 3]],
        &[2.],
        2.,
        vec![[2. * PI * 0.5, 0., 0.]],
    )
    .unwrap();
    assert!(
        (half.fractional[0] - 0.5).abs() < 1e-12,
        "{:?}",
        half.fractional
    );
}

#[test]
fn spin_channels_add_and_the_unit_is_coulomb_per_square_meter() {
    let lattice = Matrix3::from_diagonal(&nalgebra::Vector3::new(6., 6., 6.));
    let both = assemble(
        lattice,
        &[[0.; 3]],
        &[2.],
        1.,
        vec![[0.4, 0., 0.], [0.4, 0., 0.]],
    )
    .unwrap();
    let paired = assemble(lattice, &[[0.; 3]], &[2.], 2., vec![[0.4, 0., 0.]]).unwrap();
    assert!((both.vector[0] - paired.vector[0]).abs() < 1e-15);
    let factor = Quantity::Polarization.factor(false);
    assert!((factor - ELEMENTARY_CHARGE_COULOMB / (BOHR_TO_METER * BOHR_TO_METER)).abs() < 1e-12);
    assert!((factor - 57.214766).abs() < 1e-5, "{factor}");
    assert_eq!(Quantity::Polarization.factor(true), 1.);
    assert_eq!(
        Quantity::parse("polarization").unwrap(),
        Quantity::Polarization
    );
    let conductance = Quantity::Conductivity.factor(false);
    // e^2/hbar = 2.434135e-4 S over a Bohr of 5.291772e-9 cm.
    assert!(
        (conductance - 2.434_134_8e-4 / 5.291_772_109e-9).abs() < 1.,
        "{conductance}"
    );
    assert_eq!(
        Quantity::parse("conductivity").unwrap(),
        Quantity::Conductivity
    );
}

#[test]
fn a_wide_string_phase_spread_warns_naming_the_direction_and_the_remedy() {
    assert!(spread_warnings([0.1, 0.2, SPREAD_WARNING], [4, 4, 4]).is_empty());
    let notes = spread_warnings([0.1, SPREAD_WARNING + 0.01, 2.5], [4, 2, 2]);
    assert_eq!(notes.len(), 2);
    assert!(
        notes[0].contains("G_1") && notes[0].contains("denser kpts"),
        "{}",
        notes[0]
    );
    assert!(
        notes[1].contains("G_2") && notes[1].contains("2.500"),
        "{}",
        notes[1]
    );
    // The assembly alone has no strings, so no spread and no warning.
    let lattice = Matrix3::identity() * 5.;
    assert!(assemble(lattice, &[[0.; 3]], &[2.], 2., vec![[0.; 3]])
        .unwrap()
        .warnings
        .is_empty());
}
