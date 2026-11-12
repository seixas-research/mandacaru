//! `periodic::berry` on DFT crystals (PAW-LCAO SZ, Rust setup), checked by physical identities:
//! - an exact nonzero gate: moving every atom by whole grid steps in a skew polar cell moves each
//!   channel's Berry phase along `G_j` by exactly `N_occ G_j . t` and leaves the total polarization
//!   unchanged modulo its quantum (it catches a sign, a transposed lattice, a wrong one-center
//!   phase `e^{-i b.R_a}`, spin factor or valence charge);
//! - the polarization converges with the string length on that polar cell;
//! - a centrosymmetric crystal sits on 0 or half a quantum;
//! - a spin-polarized crystal with zero moments equals the unpolarized one, channel by channel;
//! - time reversal: `Omega(-k) = -Omega(k)` and zero Chern numbers and Hall conductivity;
//! - Stokes on the DFT links, and random gauges on them;
//! - a metal is refused, its Hall conductivity as not implemented.
//! Needs `MANDACARU_PAW_PATH`.
mod berry_support;
mod setup_shells_support;

use berry_support::*;
use mandacaru_core::{
    periodic::berry::{
        curvature, hall, links, polarization, strings, Bands, Metal, LINK_TOLERANCE,
    },
    quantum::C64,
    Error,
};
use nalgebra::{DMatrix, Vector3};
use std::f64::consts::PI;

fn distance_to_lattice(x: f64) -> f64 {
    (x - x.round()).abs()
}

#[test]
fn a_rigid_shift_by_grid_steps_moves_each_phase_by_n_occ_g_dot_t_and_keeps_the_polarization() {
    let cell = lih("lda-sr");
    let steps = [1, 2, 3];
    let size = [2, 2, 2];
    let before = solve(&cell);
    let after = solve(&shifted(&cell, steps));
    let p = polarization::polarization(&before.sampler(), size).unwrap();
    let q = polarization::polarization(&after.sampler(), size).unwrap();
    let occupied = before
        .sampler()
        .manifold(Bands::Occupied, Metal::Undefined)
        .unwrap()
        .ranges[0]
        .1 as f64;
    assert!(occupied >= 1.);
    for j in 0..3 {
        // G_j . t = 2 pi steps_j / n_j.
        let expected = occupied * 2. * PI * steps[j] as f64 / cell.grid[j] as f64;
        let moved = links::wrap(q.phases[0][j] - p.phases[0][j] - expected);
        assert!(
            moved.abs() < 1e-6,
            "direction {j}: phase moved by {moved:e} off N_occ G.t"
        );
        assert!(
            distance_to_lattice(q.fractional[j] - p.fractional[j]) < 1e-6,
            "direction {j}: P moved by {} quanta",
            q.fractional[j] - p.fractional[j]
        );
        // Not vacuous: the electronic and ionic parts each moved by a fraction of a quantum.
        assert!(distance_to_lattice(q.electronic[j] - p.electronic[j]) > 1e-3);
    }
    assert!(p.minimum > 0.1, "{}", p.minimum);
}

#[test]
#[ignore = "slow: 26 s (measured 2026-11-11); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test berry_crystal -- --ignored the_polarization_of_polar_lih_converges_with_the_string_length"]
fn the_polarization_of_polar_lih_converges_with_the_string_length() {
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let along = |n: usize| polarization::polarization(&sampler, [n, 2, 2]).unwrap().raw[0];
    let (p2, p4, p6) = (along(2), along(4), along(6));
    let (first, second) = (distance_to_lattice(p4 - p2), distance_to_lattice(p6 - p4));
    eprintln!("raw P_1 in quanta: {p2:.9} {p4:.9} {p6:.9}");
    assert!(
        second < first,
        "the differences shrink: {first:e} then {second:e}"
    );
    assert!(second < 1e-3, "{second:e}");
}

#[test]
fn a_centrosymmetric_crystal_sits_on_zero_or_half_a_quantum() {
    let solved = solve(&h2_chain("lda-sr", "lda"));
    let p = polarization::polarization(&solved.sampler(), [3, 2, 2]).unwrap();
    for j in 0..3 {
        let x = p.fractional[j];
        assert!(
            x.abs() < 1e-6 || (x.abs() - 0.5).abs() < 1e-6,
            "direction {j}: {x} quanta"
        );
    }
    assert_eq!(p.quantum, 1);
    assert_eq!(p.degeneracy, 2.);
}

#[test]
fn a_spin_polarized_crystal_with_zero_moments_equals_the_unpolarized_one() {
    let unpolarized = solve(&h2_chain("lda-sr", "lda"));
    let mut cell = h2_chain("lda-sr", "lda");
    cell.spin = true;
    let polarized = solve(&cell);
    assert_eq!(polarized.crystal.spins, 2);
    let size = [3, 2, 2];
    let p = polarization::polarization(&unpolarized.sampler(), size).unwrap();
    let q = polarization::polarization(&polarized.sampler(), size).unwrap();
    for j in 0..3 {
        for s in 0..2 {
            assert!(links::wrap(q.phases[s][j] - p.phases[0][j]).abs() < 1e-6);
        }
        assert!(distance_to_lattice(q.raw[j] - p.raw[j]) < 1e-6);
        assert!((q.vector[j] - p.vector[j]).abs() < 1e-8);
    }
    let set = strings::berry_phase(&polarized.sampler(), 0, size, Bands::Occupied).unwrap();
    assert_eq!(set.ranges, vec![(0, 1), (0, 1)]);
}

#[test]
#[ignore = "slow: 13 s (measured 2026-11-11); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test berry_crystal -- --ignored time_reversal_makes_the_curvature_odd_and_every_chern_number_zero"]
fn time_reversal_makes_the_curvature_odd_and_every_chern_number_zero() {
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let n = 4;
    let plane = curvature::berry_curvature(&sampler, 2, [n, n], 0., Bands::Occupied).unwrap();
    assert_eq!(plane.chern, vec![0]);
    let flux = &plane.flux[0];
    let largest = flux.iter().fold(0f64, |m, v| m.max(v.abs()));
    assert!(
        largest > 1e-8,
        "the curvature is not zero to begin with: {largest:e}"
    );
    for i in 0..n {
        for j in 0..n {
            let opposite = (n - 1 - i) * n + (n - 1 - j);
            assert!(
                (flux[i * n + j] + flux[opposite]).abs() < 1e-6 * largest + 1e-10,
                "plaquette ({i}, {j})"
            );
        }
    }
    let hall = hall::anomalous_hall(&sampler, [2, 2, 2]).unwrap();
    assert_eq!(hall.total, [0, 0, 0]);
    assert!(hall.sigma.iter().flatten().all(|v| *v == 0.));
}

#[test]
fn strip_fluxes_of_dft_links_are_differences_of_adjacent_string_phases() {
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let manifold = sampler.manifold(Bands::Occupied, Metal::Undefined).unwrap();
    let lattice = sampler.lattice([3, 3, 1], [0.; 3], &manifold).unwrap();
    // Strings along G_0, one per index along G_1; the plane normal to G_2 is spanned by (0, 1).
    let set = strings::strings(&sampler, &lattice, 0).unwrap();
    let plane = curvature::plane(&sampler, &lattice, 2, 0).unwrap();
    assert_eq!(plane.axes, [0, 1]);
    for j in 0..3 {
        let strip: f64 = (0..3).map(|i| plane.flux[0][i * 3 + j]).sum();
        let difference = set.phases[0][j] - set.phases[0][(j + 1) % 3];
        assert!(links::wrap(strip - difference).abs() < 1e-9, "strip {j}");
    }
}

#[test]
fn random_gauges_on_dft_links_change_no_phase_center_or_flux() {
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let manifold = sampler.manifold(Bands::Occupied, Metal::Undefined).unwrap();
    let lattice = sampler
        .lattice([4, 1, 1], [0., 0.25, 0.], &manifold)
        .unwrap();
    let raw: Vec<DMatrix<C64>> = (0..4)
        .map(|s| sampler.link(&lattice, 0, s, 0).unwrap())
        .collect();
    let n = raw[0].nrows();
    // U(k_s) from a fixed rule: the Q of a QR of a deterministic complex matrix per point.
    let gauge = |s: usize| {
        DMatrix::<C64>::from_fn(n, n, |i, j| {
            let t = (1 + s * 7 + i * 3 + j * 5) as f64;
            C64::new((1.3 * t).sin(), (0.7 * t).cos())
        })
        .qr()
        .q()
    };
    let rotated: Vec<DMatrix<C64>> = (0..4)
        .map(|s| gauge(s).adjoint() * &raw[s] * gauge((s + 1) % 4))
        .collect();
    let (p, _) = links::string_phase(&raw, LINK_TOLERANCE).unwrap();
    let (q, _) = links::string_phase(&rotated, LINK_TOLERANCE).unwrap();
    assert!(links::wrap(p - q).abs() < 1e-10);
    let x = links::wannier_centers(&raw, LINK_TOLERANCE).unwrap();
    let y = links::wannier_centers(&rotated, LINK_TOLERANCE).unwrap();
    for (u, v) in x.iter().zip(&y) {
        assert!(distance_to_lattice(u - v) < 1e-9);
    }
    // The string's phase from the module equals the phase of its raw links.
    let set = strings::strings(&sampler, &lattice, 0).unwrap();
    assert!(links::wrap(set.phases[0][0] - p).abs() < 1e-12);
    let g = Vector3::from(sampler.reciprocal_vector(0)).norm();
    for (x, a) in set.centers[0][0].iter().zip(&set.connection[0][0]) {
        let centered = if *x > 0.5 { x - 1. } else { *x };
        assert!((a - 2. * PI * centered / g).abs() < 1e-12);
    }
}

/// One H per 3 Bohr along x: a half-filled band, a metal.
fn hydrogen_chain() -> Cell {
    let mut cell = h2_chain("lda-sr", "lda");
    cell.symbols = vec!["H"];
    cell.lattice = nalgebra::Matrix3::from_diagonal(&Vector3::new(3., 7., 7.));
    cell.grid = [9, 20, 20];
    cell.fractional = vec![[0.5, 0.5, 0.5]];
    cell.moments = vec![0.];
    cell.mesh = [6, 1, 1];
    cell.width = 1e-2;
    cell
}

#[test]
fn a_metal_is_refused_its_polarization_but_keeps_its_isolated_band_groups() {
    let solved = solve(&hydrogen_chain());
    let sampler = solved.sampler();
    match polarization::polarization(&sampler, [4, 2, 2]) {
        Err(Error::Invalid(message)) => assert!(message.contains("insulator"), "{message}"),
        other => panic!("a metal's polarization must be refused: {:?}", other.err()),
    }
    // An explicit band group is still defined for a metal: the chain's one band is a complete
    // group (nothing above it to touch), so its curvature runs and its Chern number is zero.
    // (Its Hall conductivity is B2's, in `berry_metals.rs`.)
    assert_eq!(solved.crystal.bands(), 1);
    let plane = curvature::berry_curvature(&sampler, 2, [4, 2], 0., Bands::Range(0, 1));
    assert_eq!(plane.map(|p| p.chern).unwrap(), vec![0]);
}

#[test]
fn bad_requests_are_refused() {
    let solved = solve(&h2_chain("lda-sr", "lda"));
    let sampler = solved.sampler();
    assert!(strings::berry_phase(&sampler, 3, [2, 2, 2], Bands::Occupied).is_err());
    assert!(strings::berry_phase(&sampler, 0, [1, 2, 2], Bands::Occupied).is_err());
    assert!(curvature::berry_curvature(&sampler, 2, [1, 4], 0., Bands::Occupied).is_err());
    assert!(curvature::berry_curvature(&sampler, 2, [4, 4], 1., Bands::Occupied).is_err());
    assert!(curvature::berry_curvature(&sampler, 2, [4, 4], 0., Bands::Range(1, 1)).is_err());
    let too_many = solved.crystal.bands() + 1;
    assert!(strings::berry_phase(&sampler, 0, [2, 1, 1], Bands::Range(0, too_many)).is_err());
    assert!(polarization::polarization(&sampler, [2, 2, 1]).is_err());
}

#[test]
#[ignore = "slow: 44 s (measured 2026-11-11); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test berry_crystal -- --ignored a_spinor_crystal_without_measurable_spin_orbit_equals_the_collinear_one"]
fn a_spinor_crystal_without_measurable_spin_orbit_equals_the_collinear_one() {
    // Li and H carry their spin-orbit term in the l = 1 channel, which the SZ basis (s only) does
    // not reach: LiH on lda-fr is the lda-sr crystal written as spinors (f = 1, Kramers pairs), a
    // clean gate of the spinor bookkeeping (vera, 2026-11-11: equal to 4e-14 eV).
    let collinear = solve(&lih("lda-sr"));
    let mut cell = lih("lda-fr");
    cell.spinor = true;
    let spinor = solve(&cell);
    assert!(spinor.crystal.spinor);
    assert!(
        (spinor.state.energy - collinear.state.energy).abs() < 1e-9,
        "{} against {} Ha",
        spinor.state.energy,
        collinear.state.energy
    );
    let size = [2, 2, 2];
    let (s, c) = (spinor.sampler(), collinear.sampler());
    let p = polarization::polarization(&s, size).unwrap();
    let q = polarization::polarization(&c, size).unwrap();
    assert_eq!((p.degeneracy, q.degeneracy), (1., 2.));
    for j in 0..3 {
        assert!(
            distance_to_lattice(p.fractional[j] - q.fractional[j]) < 1e-8,
            "P_{j}"
        );
    }
    // Each Kramers pair doubles the determinant phase of its collinear band.
    let spinor_strings = strings::berry_phase(&s, 0, size, Bands::Occupied).unwrap();
    let collinear_strings = strings::berry_phase(&c, 0, size, Bands::Occupied).unwrap();
    for (a, b) in spinor_strings.phases[0]
        .iter()
        .zip(&collinear_strings.phases[0])
    {
        assert!(links::wrap(a - 2. * b).abs() < 1e-8, "{a} against 2 x {b}");
    }
    let plane_s = curvature::berry_curvature(&s, 2, [3, 3], 0., Bands::Occupied).unwrap();
    let plane_c = curvature::berry_curvature(&c, 2, [3, 3], 0., Bands::Occupied).unwrap();
    for (a, b) in plane_s.flux[0].iter().zip(&plane_c.flux[0]) {
        assert!(
            links::wrap(a - 2. * b).abs() < 1e-8,
            "flux {a} against 2 x {b}"
        );
    }
}
