//! Negative controls of the explicit hybrid force terms (HX-1): each wrong variant is compared
//! with the central difference of the production energy and must miss it by more than 100 x the
//! gate of the matching positive test (`periodic_hybrid_gradient.rs`). Needs
//! `MANDACARU_PAW_PATH`.
mod periodic_hybrid_gradient_support;
mod periodic_screened_augmentation_support;
mod setup_support;

use periodic_hybrid_gradient_support::*;
use periodic_screened_augmentation_support::{Rig, GB};

/// (central difference, analytic) phase derivatives of every (atom, axis) of a rig.
fn phase_pairs(fixture: &str, gamma: bool, occupied: &[usize]) -> Vec<(f64, f64, usize)> {
    let rig = Rig::new(fixture, gamma, occupied);
    let mut geometry = rig.geometry.translated([0.; 3]);
    if fixture.starts_with("si") {
        for axis in 0..3 {
            geometry = moved(&geometry, 1, axis, OFF_SITE[axis]);
        }
    }
    let analytic = gradient(&rig, &geometry, GB);
    let mut out = Vec::new();
    for atom in 0..geometry.atoms.len() {
        for axis in 0..3 {
            let (slope, _) = central(2e-3, &mut |d| {
                compensation_energy(&rig, &moved(&geometry, atom, axis, d), GB)
            });
            out.push((slope, analytic.phase[atom][axis], atom));
        }
    }
    out
}

/// Worst miss of a wrong variant `wrong(analytic, atom, index_of_axis_in_list)`.
fn miss(pairs: &[(f64, f64, usize)], wrong: &dyn Fn(usize, &[(f64, f64, usize)]) -> f64) -> f64 {
    (0..pairs.len())
        .map(|i| (pairs[i].0 - wrong(i, pairs)).abs())
        .fold(0., f64::max)
}

#[test]
fn a_wrong_phase_derivative_misses_the_energy_slope_by_far_more_than_the_gate() {
    for (fixture, gamma, occupied) in [
        ("si_crystal.json", true, &[4usize][..]),
        ("h2_skew_pbe.json", false, &[1][..]),
    ] {
        let pairs = phase_pairs(fixture, gamma, occupied);
        let right = miss(&pairs, &|i, p| p[i].1);
        assert!(
            right < PHASE_GATE,
            "{fixture}: the right variant misses by {right:e}"
        );
        let dropped = miss(&pairs, &|_, _| 0.);
        let flipped = miss(&pairs, &|i, p| -p[i].1);
        // The other atom's derivative (owners swapped): atoms are listed 3 axes each.
        let swapped = miss(&pairs, &|i, p| p[(i + 3) % p.len()].1);
        println!("{fixture}: dropped {dropped:.2e}, sign {flipped:.2e}, owners {swapped:.2e}");
        for (name, value) in [("dropped", dropped), ("sign", flipped), ("owners", swapped)] {
            assert!(
                value > 100. * PHASE_GATE,
                "{fixture}: {name} misses by {value:e}"
            );
        }
    }
}

#[test]
fn a_wrong_projection_response_misses_the_energy_slope_by_far_more_than_the_gate() {
    for (fixture, gamma, occupied) in [
        ("si_crystal.json", true, &[4usize][..]),
        ("h2_skew_pbe.json", false, &[1][..]),
        ("h_spin_lda.json", false, &[1, 1][..]),
    ] {
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
        let full = full_gradient(&rig, GB);
        let (slope, _) = central(1e-3, &mut |e| {
            perturb(&mut rig, &base, &delta, e);
            full_energy(&rig, GB)
        });
        let right = chain(&delta, &full.response());
        assert!(
            (right - slope).abs() < RESPONSE_GATE,
            "{fixture}: {:e}",
            right - slope
        );
        let variants = [
            ("response dropped", 0.),
            ("sign", -right),
            ("spheres dropped", chain(&delta, &full.compensation)),
            ("compensation dropped", chain(&delta, &full.sphere)),
        ];
        for (name, value) in variants {
            let gap = (value - slope).abs();
            println!("{fixture}: {name} misses by {gap:.2e}");
            assert!(
                gap > 100. * RESPONSE_GATE,
                "{fixture}: {name} misses by {gap:e}"
            );
        }
    }
}
