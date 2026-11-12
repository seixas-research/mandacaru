//! `periodic::berry` needs no functional-specific code: the states of every functional, of DFT+U
//! and of a spinor crystal with spin-orbit coupling are ordinary Bloch states of their converged
//! Hamiltonian (`Frame::from_point` carries the +U operator, the hybrid `K(k)`, the meta-GGA and
//! spinor potentials). Each case runs the Berry quantities and checks the invariances its symmetry
//! fixes: a time-reversal crystal has odd curvature and zero Chern numbers, a centrosymmetric one
//! a polarization on 0 or half a quantum; with real spin-orbit coupling (HI on lda-fr) the
//! Wilson-loop centers of every time-reversal-invariant string come in Kramers pairs. The f = 1
//! spinor bookkeeping is gated in `berry_crystal.rs` (LiH on lda-fr against lda-sr).
//! Needs `MANDACARU_PAW_PATH`.
mod berry_support;
mod setup_shells_support;

use berry_support::*;
use mandacaru_core::periodic::berry::{curvature, hall, polarization, strings, Bands};

fn on_inversion_lattice(x: f64) -> bool {
    x.abs() < 1e-6 || (x.abs() - 0.5).abs() < 1e-6
}

/// Odd curvature and zero Chern numbers (time reversal); with `inversion`, P on 0 or 1/2.
fn invariances(solved: &Solved, inversion: bool) {
    let sampler = solved.sampler();
    let n = 3;
    let plane = curvature::berry_curvature(&sampler, 2, [n, n], 0., Bands::Occupied).unwrap();
    assert!(plane.chern.iter().all(|c| *c == 0), "{:?}", plane.chern);
    for flux in &plane.flux {
        let largest = flux.iter().fold(0f64, |m, v| m.max(v.abs()));
        for i in 0..n {
            for j in 0..n {
                let opposite = (n - 1 - i) * n + (n - 1 - j);
                assert!((flux[i * n + j] + flux[opposite]).abs() < 1e-6 * largest + 1e-10);
            }
        }
    }
    let p = polarization::polarization(&sampler, [2, 2, 2]).unwrap();
    assert!(p.minimum > 1e-3);
    if inversion {
        assert!(
            p.fractional.iter().all(|x| on_inversion_lattice(*x)),
            "{:?}",
            p.fractional
        );
    }
    let hall = hall::anomalous_hall(&sampler, [2, 2, 2]).unwrap();
    assert_eq!(hall.total, [0, 0, 0]);
}

#[test]
fn pbe_berry_quantities_satisfy_the_symmetries() {
    invariances(&solve(&h2_chain("pbe-sr", "pbe")), true);
}

#[test]
fn r2scan_berry_quantities_satisfy_the_symmetries() {
    invariances(&solve(&h2_chain("pbe-sr", "r2scan")), true);
}

#[test]
fn pbe_rvv10_berry_quantities_satisfy_the_symmetries() {
    invariances(&solve(&h2_chain("pbe-sr", "pbe+rvv10")), true);
}

#[test]
#[ignore = "slow: 15 s (measured 2026-11-11); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test berry_functionals -- --ignored hse06_berry_quantities_satisfy_the_symmetries"]
fn hse06_berry_quantities_satisfy_the_symmetries() {
    invariances(&solve(&h2_chain("pbe-sr", "hse06")), true);
}

/// A Zn atom (3d10 4s2) at the center of a cubic 8 Bohr cell and an H2 molecule (1.4 Bohr, along
/// x) at its corner, U = 3 eV on the Zn 3d shell: two closed shells with a clear gap, and inversion
/// through either. (Linear ZnH2, the first choice, is not an insulator in this SZ setup: its band
/// 10 holds 1.13 of 2 electrons at Gamma, measured 2026-11-11, and the occupation check refuses it.)
fn zn_and_h2() -> Cell {
    Cell {
        library: "lda-sr",
        symbols: vec!["Zn", "H", "H"],
        lattice: nalgebra::Matrix3::from_diagonal(&nalgebra::Vector3::new(8., 8., 8.)),
        grid: [32, 32, 32],
        fractional: vec![[0.5, 0.5, 0.5], [-0.7 / 8., 0., 0.], [0.7 / 8., 0., 0.]],
        moments: vec![0.; 3],
        spin: false,
        spinor: false,
        functional: "lda",
        mesh: [1, 1, 1],
        hubbard: Some(r#"{"Zn": {"l": "d", "U": 3.0}}"#),
        width: 1e-3,
    }
}

#[test]
#[ignore = "slow: 27 s (measured 2026-11-11); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test berry_functionals -- --ignored dft_plus_u_berry_quantities_satisfy_the_symmetries"]
fn dft_plus_u_berry_quantities_satisfy_the_symmetries() {
    let solved = solve(&zn_and_h2());
    assert!(solved.state.hubbard_energy.is_some(), "the +U term is on");
    invariances(&solved, true);
}

/// HI (3.04 Bohr bond along a_1) in a skew cell on lda-fr: the I 5p spin-orbit term is large.
fn hi_spinor() -> Cell {
    Cell {
        library: "lda-fr",
        symbols: vec!["I", "H"],
        // Columns a_1 = (8.5, 0, 0), a_2 = (0.8, 7, 0), a_3 = (0.4, 0.6, 7.5).
        lattice: nalgebra::Matrix3::new(8.5, 0.8, 0.4, 0., 7., 0.6, 0., 0., 7.5),
        grid: [23, 19, 20],
        fractional: vec![[0.1, 0.2, 0.15], [0.1 + 3.04 / 8.5, 0.2, 0.15]],
        moments: vec![0.; 2],
        spin: false,
        spinor: true,
        functional: "lda",
        mesh: [2, 1, 1],
        hubbard: None,
        width: 1e-3,
    }
}

#[test]
#[ignore = "slow: 70 s (measured 2026-11-11); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test berry_functionals -- --ignored spin_orbit_hi_has_kramers_degenerate_wilson_centers_on_time_reversal_invariant_strings"]
fn spin_orbit_hi_has_kramers_degenerate_wilson_centers_on_time_reversal_invariant_strings() {
    let solved = solve(&hi_spinor());
    assert!(solved.crystal.spinor && solved.crystal.time_reversal);
    invariances(&solved, false);
    // Every string of the (4, 2, 2) mesh has k_perp in {0, 1/2}^2, equal to -k_perp modulo G:
    // time reversal pairs its Wilson-loop eigenphases (Kramers), the root of a Z2 index.
    let set = strings::berry_phase(&solved.sampler(), 0, [4, 2, 2], Bands::Occupied).unwrap();
    for centers in &set.centers[0] {
        assert!(centers.len() % 2 == 0 && !centers.is_empty());
        for x in centers {
            let partners = centers
                .iter()
                .filter(|y| {
                    let d = *x - **y;
                    (d - d.round()).abs() < 1e-9
                })
                .count();
            assert!(
                partners % 2 == 0,
                "center {x} has no Kramers partner in {centers:?}"
            );
        }
    }
    // The spin-orbit term is not idle: the pairs are not all at one center.
    let spread = set.centers[0][0].last().unwrap() - set.centers[0][0][0];
    assert!(spread > 1e-3, "{:?}", set.centers[0][0]);
}

