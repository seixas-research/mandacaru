//! The explicit terms of the hybrid crystal force (HX-1): the phase of the compensation shapes,
//! the response of the exchange to the stored projections (compensation charges and one-center
//! spheres), and the tie of the energy the derivative is taken of to the production entries.
//!
//! Every oracle is a central difference of the REAL energy entry (`exchange_operators_with`,
//! `Crystal::hybrid_exchange`) under a motion that needs no displaced crystal: the compensation
//! centers moved through `Sites`, or the stored projection arrays perturbed (the energy is
//! analytic in them), with the pair densities, states and everything else held fixed. The states
//! are the fixtures' mechanics densities. Needs `MANDACARU_PAW_PATH`.
mod periodic_hybrid_gradient_support;
mod periodic_screened_augmentation_support;
mod setup_support;

use mandacaru_core::periodic::hybrid_gradient::explicit;
use mandacaru_core::periodic::{
    screened_augmentation::Augmentation,
    screened_exchange::{exchange_operators_with, ExxDivTreatment},
};
use periodic_hybrid_gradient_support::*;
use periodic_screened_augmentation_support::{Geometry, Rig, GB};

/// (fixture, first point alone, occupied states): Si at Gamma, H2 with two points (`k != q`), a
/// unrestricted H with one electron per spin (the spin-scaling remainder is on the path; the
/// fully polarized limit `[1, 0]` is NOT smooth in the model, see the log).
pub const RIGS: [(&str, bool, &[usize]); 3] = [
    ("si_crystal.json", true, &[4]),
    ("h2_skew_pbe.json", false, &[1]),
    ("h_spin_lda.json", false, &[1, 1]),
];
/// The step of the centers (Bohr) and of the projection perturbation.
const STEP: f64 = 2e-3;
const PERTURBATION: f64 = 1e-3;

/// The rig and the geometry its compensation centers are evaluated at (Si off the symmetric
/// site; the molecules as they are).
fn setup(fixture: &str, gamma: bool, occupied: &[usize]) -> (Rig, Geometry) {
    let rig = Rig::new(fixture, gamma, occupied);
    let mut geometry = rig.geometry.translated([0.; 3]);
    if fixture.starts_with("si") {
        for axis in 0..3 {
            geometry = moved(&geometry, 1, axis, OFF_SITE[axis]);
        }
    }
    (rig, geometry)
}

#[test]
fn the_phase_derivative_is_the_energy_slope_of_moving_the_compensation_centers() {
    for (fixture, gamma, occupied) in RIGS {
        for exxdiv in [ExxDivTreatment::None, GB] {
            let (rig, geometry) = setup(fixture, gamma, occupied);
            let analytic = gradient(&rig, &geometry, exxdiv);
            let scale = analytic
                .phase
                .iter()
                .flatten()
                .fold(0f64, |m, v| m.max(v.abs()));
            assert!(
                scale > 1e-6,
                "{fixture}: a vacuous phase derivative {scale:e}"
            );
            let (mut worst, mut floor) = (0f64, 0f64);
            for atom in 0..geometry.atoms.len() {
                for axis in 0..3 {
                    let (slope, gap) = central(STEP, &mut |d| {
                        compensation_energy(&rig, &moved(&geometry, atom, axis, d), exxdiv)
                    });
                    worst = worst.max((slope - analytic.phase[atom][axis]).abs());
                    floor = floor.max(gap);
                }
            }
            println!(
                "{fixture} {exxdiv:?}: largest {scale:.3e}, deviation {worst:.2e}, floor {floor:.2e}"
            );
            assert!(worst < PHASE_GATE, "{fixture} {exxdiv:?}: {worst:e}");
            assert!(floor < PHASE_GATE / 10., "{fixture}: step floor {floor:e}");
        }
    }
}

#[test]
fn the_energy_the_gradient_differentiates_is_the_production_energy() {
    for (fixture, gamma, occupied) in RIGS {
        let (rig, geometry) = setup(fixture, gamma, occupied);
        for exxdiv in [ExxDivTreatment::None, GB] {
            let analytic = gradient(&rig, &geometry, exxdiv);
            let aug = Augmentation::new(&rig.crystal, geometry.sites(), &rig.shapes).unwrap();
            let production =
                exchange_operators_with(&rig.crystal, &rig.d, screening(&rig), exxdiv, Some(&aug))
                    .unwrap()
                    .energy;
            assert!(
                (analytic.energy - production).abs() < 1e-12 * production.abs().max(1.),
                "{fixture}: {:e} {:e}",
                analytic.energy,
                production
            );
            // The whole crystal: grid + compensation + one-center = Crystal::hybrid_exchange.
            let full = full_gradient(&rig, exxdiv);
            let total = full.energy + full.one_center_energy;
            let entry = full_energy(&rig, exxdiv);
            assert!(
                (total - entry).abs() < 1e-12 * entry.abs().max(1.),
                "{fixture}"
            );
            assert!(
                full.one_center_energy != 0.,
                "{fixture}: no one-center energy"
            );
        }
    }
}

#[test]
fn the_projection_response_is_the_energy_slope_of_the_stored_projections() {
    for (fixture, gamma, occupied) in RIGS {
        let mut rig = Rig::new(fixture, gamma, occupied);
        let (m, p) = (rig.crystal.m, rig.crystal.p);
        let nk = rig.crystal.points.len();
        let base: Vec<_> = rig
            .crystal
            .points
            .iter()
            .map(|x| x.projections.clone())
            .collect();
        let delta: Vec<_> = (0..nk)
            .map(|k| pseudo_random(m, p, 0.3 + k as f64))
            .collect();
        for exxdiv in [ExxDivTreatment::None, GB] {
            perturb(&mut rig, &base, &delta, 0.);
            let compensation = gradient(&rig, &rig.geometry, exxdiv).compensation;
            let full = full_gradient(&rig, exxdiv);
            let sphere = chain(&delta, &full.sphere);
            let (comp_fd, comp_floor) = central(PERTURBATION, &mut |e| {
                perturb(&mut rig, &base, &delta, e);
                compensation_energy(&rig, &rig.geometry, exxdiv)
            });
            let (full_fd, full_floor) = central(PERTURBATION, &mut |e| {
                perturb(&mut rig, &base, &delta, e);
                full_energy(&rig, exxdiv)
            });
            perturb(&mut rig, &base, &delta, 0.);
            let comp = chain(&delta, &compensation);
            let total = chain(&delta, &full.response());
            println!(
                "{fixture} {exxdiv:?}: compensation {comp:.6e} ({:.1e}), spheres {sphere:.6e}, \
                 total {total:.6e} ({:.1e})",
                (comp - comp_fd).abs(),
                (total - full_fd).abs()
            );
            assert!(
                comp.abs() > 1e-4 && sphere.abs() > 1e-4,
                "{fixture}: vacuous"
            );
            assert!(
                (comp - comp_fd).abs() < RESPONSE_GATE,
                "{fixture}: compensation"
            );
            assert!((total - full_fd).abs() < RESPONSE_GATE, "{fixture}: total");
            let floor = comp_floor.max(full_floor);
            println!("{fixture} {exxdiv:?}: step floor {floor:.2e}");
            assert!(
                floor < RESPONSE_GATE / 10.,
                "{fixture}: step floor {floor:e}"
            );
        }
    }
}

/// The atom each basis function sits on in the one-function-per-atom molecules and in Si (SZ,
/// four functions per atom): the owners only need to be consistent between the perturbation
/// and the motion.
fn owners(rig: &Rig) -> Vec<usize> {
    let atoms = rig.geometry.atoms.len();
    (0..rig.crystal.m)
        .map(|mu| mu * atoms / rig.crystal.m)
        .collect()
}

#[test]
fn the_function_derivative_is_the_energy_slope_of_perturbing_the_bloch_samples() {
    for (fixture, gamma, occupied) in RIGS {
        let mut rig = Rig::new(fixture, gamma, occupied);
        let owner = owners(&rig);
        let base: Vec<Vec<_>> = rig
            .crystal
            .points
            .iter()
            .map(|x| x.psi.read(&rig.crystal.context).unwrap().to_vec())
            .collect();
        let motion = Perturbation {
            owner: owner.clone(),
        };
        let aug = Augmentation::new(&rig.crystal, rig.geometry.sites(), &rig.shapes).unwrap();
        let g = explicit(
            &rig.crystal,
            &rig.d,
            screening(&rig),
            GB,
            &aug,
            Some(&motion),
        )
        .unwrap();
        drop(aug);
        let functions = g.functions.unwrap();
        let mut worst = 0f64;
        let mut floor = 0f64;
        let mut peak = 0f64;
        for atom in 0..rig.geometry.atoms.len() {
            let (slope, gap) = central(PERTURBATION, &mut |e| {
                perturb_samples(&mut rig, &base, &owner, atom, e);
                compensation_energy(&rig, &rig.geometry, GB)
            });
            perturb_samples(&mut rig, &base, &owner, atom, 0.);
            let (bra, ket) = (functions.bra[atom][0], functions.ket[atom][0]);
            println!(
                "{fixture} atom {atom}: slope {slope:+.9e}, bra {bra:+.9e} + ket {ket:+.9e} \
                 (dev {:.1e}, floor {gap:.1e})",
                (slope - bra - ket).abs()
            );
            worst = worst.max((slope - bra - ket).abs());
            floor = floor.max(gap);
            peak = peak.max(slope.abs());
        }
        assert!(peak > 1e3 * FUNCTION_GATE, "{fixture}: vacuous {peak:e}");
        assert!(worst < FUNCTION_GATE, "{fixture}: {worst:e}");
        assert!(
            floor < FUNCTION_GATE / 10.,
            "{fixture}: step floor {floor:e}"
        );
    }
}
