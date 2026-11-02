//! The range-separation parameter in the periodic screened exact exchange with compensation
//! charges (W-E4P / P1): the unit-charge sum rule of the omega dependence (between omega 0.005
//! and 0.01 only the Gygi-Baldereschi head moves, by -2 d_omega / sqrt(pi) per unit charge),
//! the smooth omega -> 0 limit to the bare kernel, and the split of the energy into the grid
//! pair, the cross term and the compensation charges.
mod periodic_screened_augmentation_support;
mod setup_support;

use mandacaru_core::periodic::{
    screened_augmentation::Augmentation,
    screened_exchange::{energy_parts, exchange_operators_with},
};
use num_complex::Complex64;
use periodic_screened_augmentation_support::*;
use std::f64::consts::PI;

#[test]
fn the_unit_charge_sum_rule_of_the_omega_dependence() {
    // Between omega = 0.005 and 0.01 only the Gygi-Baldereschi head changes (every other
    // exp(-Q^2 / 4 omega^2) underflows); per unit charge it moves by -2 d_omega / sqrt(pi).
    for (fixture, gamma, occupied) in CASES.iter() {
        let rig = Rig::new(fixture, *gamma, occupied);
        let c = &rig.crystal;
        let sites = rig.geometry.sites();
        let aug = Augmentation::new(c, sites, &rig.shapes).unwrap();
        let low = exchange_operators_with(c, &rig.d, screening(0.005), GB, Some(&aug)).unwrap();
        let high = exchange_operators_with(c, &rig.d, screening(0.01), GB, Some(&aug)).unwrap();
        let plain_low = exchange_operators_with(c, &rig.d, screening(0.005), GB, None).unwrap();
        let plain_high = exchange_operators_with(c, &rig.d, screening(0.01), GB, None).unwrap();
        let mut worst = 0f64;
        let mut miss = 0f64;
        let mut energy = 0.;
        for spin in 0..c.spins {
            for k in 0..c.points.len() {
                let at = spin * c.points.len() + k;
                let want =
                    overlap_square(&rig, spin, k) * Complex64::new(-2. * 0.005 / PI.sqrt(), 0.);
                worst = worst.max(
                    (&high.operators[at] - &low.operators[at] - &want)
                        .iter()
                        .map(|v| v.norm())
                        .fold(0., f64::max),
                );
                miss = miss.max(
                    (&plain_high.operators[at] - &plain_low.operators[at] - &want)
                        .iter()
                        .map(|v| v.norm())
                        .fold(0., f64::max),
                );
                let scale = if c.spins == 1 { 0.5 } else { 1.0 };
                let mult = if c.spins == 1 { 2. } else { 1. };
                energy += -0.5
                    * 0.25
                    * c.points[k].weight
                    * mult
                    * (&rig.d[at] * Complex64::new(scale, 0.) * &want).trace().re;
            }
        }
        let de = high.energy - low.energy;
        println!("{fixture}: K(0.01) - K(0.005) misses -2 d_omega S D S / sqrt(pi) by {worst:.2e} (grid pair alone {miss:.2e}); dE {de:.6e} against {energy:.6e}");
        assert!(worst < 1e-10, "{worst:e}");
        assert!(
            miss > 100. * worst.max(1e-10),
            "the grid pair alone must miss the sum rule"
        );
        assert!((de - energy).abs() < 1e-10 * energy.abs().max(1e-3));
    }
}

#[test]
fn omega_to_zero_is_the_bare_kernel_smoothly() {
    // omega <= 1e-12 is the bare Coulomb kernel with the bare Gygi-Baldereschi head; the head of
    // the screened one is linear in omega (erf = 1), -2 omega / sqrt(pi) per unit charge.
    for (fixture, gamma, occupied) in CASES.iter() {
        let rig = Rig::new(fixture, *gamma, occupied);
        let c = &rig.crystal;
        let sites = rig.geometry.sites();
        let aug = Augmentation::new(c, sites, &rig.shapes).unwrap();
        let bare = exchange_operators_with(c, &rig.d, screening(1e-13), GB, Some(&aug)).unwrap();
        for omega in [1e-6, 1e-5, 1e-4] {
            let screened =
                exchange_operators_with(c, &rig.d, screening(omega), GB, Some(&aug)).unwrap();
            let mut worst = 0f64;
            for spin in 0..c.spins {
                for k in 0..c.points.len() {
                    let at = spin * c.points.len() + k;
                    let want =
                        overlap_square(&rig, spin, k) * Complex64::new(-2. * omega / PI.sqrt(), 0.);
                    worst = worst.max(
                        (&screened.operators[at] - &bare.operators[at] - &want)
                            .iter()
                            .map(|v| v.norm())
                            .fold(0., f64::max),
                    );
                }
            }
            println!("{fixture}: omega {omega:e}: K - K_bare misses the linear law by {worst:.2e}");
            assert!(worst < 1e-10, "{fixture} {omega}: {worst:e}");
        }
        let bare_plain = exchange_operators_with(c, &rig.d, screening(1e-13), GB, None).unwrap();
        let gap = relative(&bare.operators[0], &bare_plain.operators[0]);
        assert!(gap > 1e-6 || rig.d[0].norm() == 0.);
    }
}

#[test]
fn the_energy_splits_into_grid_cross_and_compensation() {
    let rig = Rig::new("si_crystal.json", true, &[4]);
    let c = &rig.crystal;
    let sites = rig.geometry.sites();
    let aug = Augmentation::new(c, sites, &rig.shapes).unwrap();
    let parts = energy_parts(c, &rig.d, screening(0.11), GB, &aug).unwrap();
    let all = exchange_operators_with(c, &rig.d, screening(0.11), GB, Some(&aug)).unwrap();
    let pseudo = exchange_operators_with(c, &rig.d, screening(0.11), GB, None).unwrap();
    println!(
        "Si energy parts: grid {:.8}, cross {:.8}, compensation {:.8}; augmentation {:.8}",
        parts.pseudo,
        parts.cross,
        parts.compensation,
        parts.augmentation()
    );
    assert_eq!(parts.pseudo.to_bits(), pseudo.energy.to_bits());
    assert!((parts.total() - all.energy).abs() < 1e-12);
    assert!(parts.compensation < 0. && parts.cross.abs() > 1e-6);
}
