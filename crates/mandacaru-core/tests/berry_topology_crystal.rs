//! The Z2 index of a crystal (`periodic::berry::flow`, `periodic::berry::parity`):
//! - the H2 chain (spin-degenerate, centrosymmetric) is zero by the flow of every plane and by the
//!   Fu-Kane product, and the two agree plane by plane;
//! - HI on `lda-fr` (large spin-orbit coupling, no inversion center): every plane has Z2 = 0, the
//!   parity route refuses for want of an inversion center;
//! - a spin-polarized crystal raises.
//!
//! Bulk Bi2Se3, the Z2 = 1 crystal, is `test/integrations/test_berry.py`. Needs
//! `MANDACARU_PAW_PATH`.
mod berry_support;
mod setup_shells_support;

use berry_support::*;
use mandacaru_core::periodic::berry::{flow, parity, Bands};

#[test]
fn a_spin_degenerate_centrosymmetric_crystal_has_z2_zero_by_parity_and_by_one_flow() {
    let solved = solve(&h2_chain("lda-sr", "lda"));
    let sampler = solved.sampler();
    let by_parity = parity::parity(&sampler).unwrap();
    assert_eq!((by_parity.strong, by_parity.weak), (0, [0, 0, 0]));
    assert_eq!(by_parity.trims.len(), 8);
    for t in &by_parity.trims {
        assert!(t.residual < 1e-6 && t.occupied == 1, "{t:?}");
        assert_eq!(t.delta, if t.minus % 2 == 0 { 1 } else { -1 });
    }
    // The cell center is an inversion center of the chain (modulo half lattice vectors).
    for c in by_parity.inversion.center {
        assert!((2. * c - (2. * c).round()).abs() < 1e-6, "{c}");
    }
    let plane = flow::plane_flow(&sampler, 2, 0, [3, 3, 3], 0.5, Bands::Occupied).unwrap();
    assert!(plane.half && plane.spin_degenerate && plane.winding.is_none());
    let z = plane.z2.as_ref().unwrap();
    // Every center is doubled, so every crossing is counted twice.
    assert_eq!((z.parity, z.crossings % 2), (0, 0));
    assert_eq!(plane.flow.k[0], 0.);
    assert!((plane.flow.k.last().unwrap() - 0.5).abs() < 1e-12);
    assert_eq!(z.parity, by_parity.planes[2][1]);
}

#[test]
#[ignore = "slow: 13 s (measured 2026-11-13, other agents running); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test berry_topology_crystal -- --ignored every_plane"]
fn a_spin_degenerate_centrosymmetric_crystal_has_z2_zero_by_flow_and_by_parity_on_every_plane() {
    let solved = solve(&h2_chain("lda-sr", "lda"));
    let sampler = solved.sampler();
    let size = [3, 3, 3];
    let found = flow::indices(&sampler, size).unwrap();
    assert_eq!((found.strong, found.weak), (0, [0, 0, 0]));
    for pair in &found.planes {
        for plane in pair {
            assert!(plane.half && plane.spin_degenerate && plane.winding.is_none());
            let z = plane.z2.as_ref().unwrap();
            assert_eq!(z.parity, 0);
            // Every center is doubled, so every crossing is counted twice.
            assert_eq!(z.crossings % 2, 0);
            assert_eq!(plane.flow.k[0], 0.);
            assert!((plane.flow.k.last().unwrap() - 0.5).abs() < 1e-12);
        }
    }
    let by_parity = parity::parity(&sampler).unwrap();
    assert_eq!((by_parity.strong, by_parity.weak), (0, [0, 0, 0]));
    assert_eq!(by_parity.trims.len(), 8);
    for t in &by_parity.trims {
        assert!(t.residual < 1e-6 && t.occupied == 1, "{t:?}");
        assert_eq!(t.delta, if t.minus % 2 == 0 { 1 } else { -1 });
    }
    // The cell center is an inversion center of the chain (modulo half lattice vectors).
    for c in by_parity.inversion.center {
        assert!((2. * c - (2. * c).round()).abs() < 1e-6, "{c}");
    }
    for c in 0..3 {
        for half in 0..2 {
            let flown = found.planes[c][half].z2.as_ref().unwrap().parity;
            assert_eq!(flown, by_parity.planes[c][half], "plane {c} {half}");
        }
    }
}

#[test]
fn a_time_reversal_plane_off_the_invariant_lines_has_a_whole_period_flow_and_a_zero_chern_number() {
    let solved = solve(&h2_chain("lda-sr", "lda"));
    let plane =
        flow::plane_flow(&solved.sampler(), 2, 0, [3, 3, 3], 0.25, Bands::Occupied).unwrap();
    assert!(!plane.half && plane.z2.is_none() && plane.winding == Some(0));
    assert!((plane.flow.k.last().unwrap() - 1.).abs() < 1e-12);
    assert_eq!(plane.flow.centers.first(), plane.flow.centers.last());
}

#[test]
fn a_spin_polarized_crystal_has_no_z2_index() {
    let mut cell = h2_chain("lda-sr", "lda");
    cell.spin = true;
    cell.moments = vec![0., 0.];
    let solved = solve(&cell);
    let sampler = solved.sampler();
    for message in [
        flow::indices(&sampler, [3, 3, 3]).unwrap_err().to_string(),
        parity::parity(&sampler).unwrap_err().to_string(),
    ] {
        assert!(message.contains("spin-polarized"), "{message}");
    }
}

#[test]
fn a_magnetic_spinor_crystal_has_no_z2_index() {
    // Two parallel moments on a spinor H2 chain break time reversal.
    let mut cell = h2_chain("lda-fr", "lda");
    cell.spinor = true;
    cell.moments = vec![1., 1.];
    let solved = solve(&cell);
    assert!(solved.crystal.spinor && !solved.crystal.time_reversal);
    let sampler = solved.sampler();
    for message in [
        flow::indices(&sampler, [3, 3, 3]).unwrap_err().to_string(),
        parity::parity(&sampler).unwrap_err().to_string(),
    ] {
        assert!(
            message.contains("time-reversal-invariant crystal"),
            "{message}"
        );
    }
}

#[test]
fn a_crystal_without_an_inversion_center_refuses_the_parity_route_and_keeps_the_flow() {
    // LiH is polar: no inversion center. Its flow still gives the spin-degenerate zero.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let message = parity::parity(&sampler).unwrap_err().to_string();
    assert!(message.contains("inversion center"), "{message}");
    let plane = flow::plane_flow(&sampler, 2, 0, [3, 3, 3], 0.5, Bands::Occupied).unwrap();
    assert_eq!(plane.z2.as_ref().unwrap().parity, 0);
}

/// HI (3.04 Bohr bond along a_1) in a skew cell on lda-fr, the cell of `berry_functionals`.
fn hi_spinor() -> Cell {
    Cell {
        library: "lda-fr",
        symbols: vec!["I", "H"],
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
#[ignore = "slow: 49 s (measured 2026-11-13, other agents running); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test berry_topology_crystal -- --ignored hi_on_lda_fr"]
fn hi_on_lda_fr_has_z2_zero_on_every_plane_and_no_inversion_center() {
    let solved = solve(&hi_spinor());
    assert!(solved.crystal.spinor && solved.crystal.time_reversal);
    let sampler = solved.sampler();
    let found = flow::indices(&sampler, [4, 3, 3]).unwrap();
    assert_eq!((found.strong, found.weak), (0, [0, 0, 0]));
    let mut split = 0f64;
    for pair in &found.planes {
        for plane in pair {
            assert!(plane.half && !plane.spin_degenerate);
            assert_eq!(plane.z2.as_ref().unwrap().parity, 0);
            // Kramers pairs split away from the lines.
            for centers in &plane.flow.centers {
                for j in 0..centers.len() / 2 {
                    split = split.max((centers[2 * j + 1] - centers[2 * j]).abs());
                }
            }
        }
    }
    // The spin-orbit coupling is not idle (by 1e-5 to 1e-4 of a cell in this light crystal).
    assert!(split > 1e-5, "{split}");
    let message = parity::parity(&sampler).unwrap_err().to_string();
    assert!(message.contains("inversion center"), "{message}");
}
