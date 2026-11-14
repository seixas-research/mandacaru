//! The Wannier-center flow invariants (`periodic::berry::topology`) on lattice models built here
//! (no crystal): the Chern number as the winding of the centers, the Z2 index as the parity of their
//! crossings of the largest-gap line.
//! - Qi-Wu-Zhang: the winding equals the plaquette Chern number of `links::lattice_field` for every
//!   mass (the sign convention is pinned here);
//! - Bernevig-Hughes-Zhang (two time-reversed Qi-Wu-Zhang blocks), and with its spin rotated by a
//!   random SU(2) so S_z is not conserved: Z2 = 1 for 0 < |m| < 2, else 0;
//! - Kane-Mele with and without Rashba coupling (S_z not conserved): Z2 = 1 at zero sublattice
//!   potential, 0 beyond `3 sqrt(3) lambda_so`;
//! - a random unitary gauge at every k leaves the centers and the invariants unchanged;
//! - refusals: a set that is not time-reversal invariant, a flow that moves too far for its depth.
mod berry_models_support;

use berry_models_support::*;
use mandacaru_core::{
    periodic::berry::{
        links,
        topology::{self, Adaptive, Flow},
    },
    quantum::C64,
};
use std::f64::consts::PI;

/// The sorted centers of the closed string along `k_x` (`n` points) of the `occupied` lowest
/// states of `h(k_x, k_y)` at `k_y = 2 pi t`, in a random gauge when `gauge` is given.
fn centers(
    h: &dyn Fn(f64, f64) -> Matrix,
    occupied: usize,
    n: usize,
    t: f64,
    gauge: &mut Option<Rng>,
) -> Vec<f64> {
    let states: Vec<Matrix> = (0..n)
        .map(|i| {
            let states = lowest(h(2. * PI * i as f64 / n as f64, 2. * PI * t), occupied);
            match gauge {
                Some(rng) => states * rng.unitary(occupied),
                None => states,
            }
        })
        .collect();
    links::wannier_centers(&string(&states), 1e-10).unwrap()
}

fn flow(
    h: &dyn Fn(f64, f64) -> Matrix,
    occupied: usize,
    end: f64,
    settings: Adaptive,
    gauge: &mut Option<Rng>,
) -> topology::Flow {
    topology::adaptive("a model plane", end, settings, |t| {
        Ok(centers(h, occupied, 24, t, gauge))
    })
    .unwrap()
}

const SETTINGS: Adaptive = Adaptive {
    intervals: 4,
    depth: 6,
};

fn z2(h: &dyn Fn(f64, f64) -> Matrix, occupied: usize, gauge: &mut Option<Rng>) -> u8 {
    topology::z2(&flow(h, occupied, 0.5, SETTINGS, gauge))
        .unwrap()
        .parity
}

/// Bernevig-Hughes-Zhang: the Qi-Wu-Zhang block and its time-reversed partner, spin rotated by `u`
/// (an SU(2) acting on the spin index of the 2 x 2 orbital blocks; `None` leaves S_z conserved).
fn bhz(m: f64, rotation: Option<Matrix>) -> impl Fn(f64, f64) -> Matrix {
    move |kx, ky| {
        let up = sigma(qwz(m, kx, ky));
        let down = sigma(qwz(m, -kx, -ky)).map(|v| v.conj());
        let mut h = Matrix::zeros(4, 4);
        h.view_mut((0, 0), (2, 2)).copy_from(&up);
        h.view_mut((2, 2), (2, 2)).copy_from(&down);
        match &rotation {
            // The basis is (spin, orbital): the rotation acts on the spin index.
            Some(u) => {
                let big = u.kronecker(&Matrix::identity(2, 2));
                &big * h * big.adjoint()
            }
            None => h,
        }
    }
}

/// An SU(2) rotation `exp(-i angle n.s / 2)`.
fn spin_rotation(angle: f64, n: [f64; 3]) -> Matrix {
    let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let s = sigma([n[0] / norm, n[1] / norm, n[2] / norm]);
    Matrix::identity(2, 2) * C64::new((angle / 2.).cos(), 0.) - s * C64::new(0., (angle / 2.).sin())
}

/// Kane-Mele on the honeycomb lattice (`a_1 = (1, 0)`, `a_2 = (1/2, sqrt(3)/2)`, B at
/// `(a_1 + a_2)/3`), nearest-neighbor hopping 1, the intrinsic spin-orbit term `i so nu s_z`
/// between second neighbors, the Rashba term `i rashba (s x d)_z` between nearest neighbors and a
/// sublattice potential `+-staggered`. Basis `(sublattice, spin)`; `H(k + G) = H(k)`.
fn kane_mele(so: f64, rashba: f64, staggered: f64) -> impl Fn(f64, f64) -> Matrix {
    let a = [[1., 0.], [0.5, 3f64.sqrt() / 2.]];
    let tau = [(a[0][0] + a[1][0]) / 3., (a[0][1] + a[1][1]) / 3.];
    let position = move |sub: usize, r: [i32; 2]| -> [f64; 2] {
        let base = if sub == 0 { [0., 0.] } else { tau };
        std::array::from_fn(|c| base[c] + r[0] as f64 * a[0][c] + r[1] as f64 * a[1][c])
    };
    let bond = 1. / 3f64.sqrt();
    let near =
        move |p: [f64; 2], q: [f64; 2]| ((p[0] - q[0]).hypot(p[1] - q[1]) - bond).abs() < 1e-9;
    move |k1, k2| {
        let phase = |r: [i32; 2]| C64::from_polar(1., k1 * r[0] as f64 + k2 * r[1] as f64);
        let sz = sigma([0., 0., 1.]);
        let mut h = Matrix::zeros(4, 4);
        for s in 0..2 {
            h[(s, s)] += C64::new(staggered, 0.);
            h[(2 + s, 2 + s)] -= C64::new(staggered, 0.);
        }
        // Nearest neighbors: B in cell r hops to A in cell 0.
        for r in [[0, 0], [-1, 0], [0, -1]] {
            let (to, from) = (position(0, [0, 0]), position(1, r));
            let d = [(to[0] - from[0]) / bond, (to[1] - from[1]) / bond];
            let sx = sigma([1., 0., 0.]);
            let sy = sigma([0., 1., 0.]);
            let block = Matrix::identity(2, 2) * C64::new(1., 0.)
                + (sx * C64::new(0., rashba * d[1]) - sy * C64::new(0., rashba * d[0]));
            let mut upper = h.view((0, 2), (2, 2)).into_owned();
            upper += block * phase(r);
            h.view_mut((0, 2), (2, 2)).copy_from(&upper);
        }
        let lower = h.view((0, 2), (2, 2)).adjoint();
        h.view_mut((2, 0), (2, 2)).copy_from(&lower);
        // Second neighbors: the site displaced by d hops to the site at the origin.
        for sub in 0..2 {
            for d in [[1, 0], [-1, 0], [0, 1], [0, -1], [-1, 1], [1, -1]] {
                let (to, from) = (position(sub, [0, 0]), position(sub, d));
                let middle = (-2..=2)
                    .flat_map(|x| (-2..=2).map(move |y| [x, y]))
                    .map(|r| position(1 - sub, r))
                    .find(|m| near(*m, to) && near(*m, from))
                    .expect("a second neighbor shares one neighbor");
                let (e1, e2) = (
                    [middle[0] - from[0], middle[1] - from[1]],
                    [to[0] - middle[0], to[1] - middle[1]],
                );
                let chirality = (e1[0] * e2[1] - e1[1] * e2[0]).signum();
                let block = &sz * C64::new(0., so * chirality) * phase(d);
                let mut diagonal = h.view((2 * sub, 2 * sub), (2, 2)).into_owned();
                diagonal += block;
                h.view_mut((2 * sub, 2 * sub), (2, 2)).copy_from(&diagonal);
            }
        }
        h
    }
}

#[test]
fn qwz_winding_equals_the_plaquette_chern_number_for_every_mass() {
    for m in [-3., -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.] {
        let h = |kx: f64, ky: f64| sigma(qwz(m, kx, ky));
        let identity = &mut || Matrix::identity(1, 1);
        // The plaquette Chern number on a 24 x 24 mesh.
        let n = 24;
        let k = |i: usize| 2. * PI * i as f64 / n as f64;
        let states: Vec<Matrix> = (0..n * n)
            .map(|p| lowest(h(k(p / n), k(p % n)), 1) * identity())
            .collect();
        let at = |i: usize, j: usize| &states[(i % n) * n + j % n];
        let dets = |step: (usize, usize)| -> Vec<C64> {
            (0..n * n)
                .map(|p| {
                    let (i, j) = (p / n, p % n);
                    links::link_phase(&link(at(i, j), at(i + step.0, j + step.1)), 1e-10)
                        .unwrap()
                        .0
                })
                .collect()
        };
        let chern = links::lattice_field(&dets((1, 0)), &dets((0, 1)), [n, n])
            .unwrap()
            .chern;
        let wound = topology::winding(&flow(&h, 1, 1., SETTINGS, &mut None)).unwrap();
        assert_eq!(
            wound, chern,
            "m {m}: winding {wound}, plaquette Chern number {chern}"
        );
        assert_eq!(chern.abs(), (m.abs() < 2.) as i64, "m {m}");
    }
}

#[test]
fn bhz_is_a_z2_insulator_for_a_mass_below_two_with_and_without_a_conserved_spin() {
    for (m, expected) in [
        (-3., 0),
        (-1., 1),
        (-0.5, 1),
        (0.5, 1),
        (1.5, 1),
        (2.5, 0),
        (3., 0),
    ] {
        assert_eq!(
            z2(&bhz(m, None), 2, &mut None),
            expected,
            "S_z conserved, m {m}"
        );
        let rotation = spin_rotation(0.9, [0.3, -0.5, 0.8]);
        assert_eq!(
            z2(&bhz(m, Some(rotation)), 2, &mut None),
            expected,
            "S_z mixed, m {m}"
        );
    }
    // A Chern number cancels between the two blocks.
    let h = bhz(1., None);
    let full = flow(&h, 2, 1., SETTINGS, &mut None);
    assert_eq!(topology::winding(&full).unwrap(), 0);
}

#[test]
fn kane_mele_is_a_z2_insulator_below_the_staggered_potential_threshold_with_rashba_coupling() {
    let so = 0.1;
    // The threshold of the Rashba-free model is 3 sqrt(3) so = 0.5196.
    for rashba in [0., 0.05, 0.15] {
        assert_eq!(
            z2(&kane_mele(so, rashba, 0.), 2, &mut None),
            1,
            "rashba {rashba}"
        );
        assert_eq!(
            z2(&kane_mele(so, rashba, 0.3), 2, &mut None),
            1,
            "rashba {rashba}"
        );
        assert_eq!(
            z2(&kane_mele(so, rashba, 0.8), 2, &mut None),
            0,
            "rashba {rashba}"
        );
    }
    // Without spin-orbit coupling a staggered potential alone is a trivial insulator, and without
    // the potential the model is gapless: it raises on the closed gap or counts nothing.
    assert_eq!(z2(&kane_mele(0., 0., 0.4), 2, &mut None), 0);
}

#[test]
fn a_random_gauge_at_every_k_changes_neither_the_centers_nor_the_invariants() {
    let h = kane_mele(0.1, 0.05, 0.);
    let plain = flow(&h, 2, 0.5, SETTINGS, &mut None);
    let mut gauge = Some(Rng(0x9e37_79b9_7f4a_7c15));
    let rotated = flow(&h, 2, 0.5, SETTINGS, &mut gauge);
    assert_eq!(plain.k, rotated.k);
    for (a, b) in plain
        .centers
        .iter()
        .flatten()
        .zip(rotated.centers.iter().flatten())
    {
        assert!((a - b).abs() < 1e-9, "{a} against {b}");
    }
    let (a, b) = (
        topology::z2(&plain).unwrap(),
        topology::z2(&rotated).unwrap(),
    );
    assert_eq!((a.parity, a.crossings), (b.parity, b.crossings));
    for (x, y) in a.gaps.iter().zip(&b.gaps) {
        assert!((x - y).abs() < 1e-9, "{x} against {y}");
    }
    for (x, y) in a.lines.iter().zip(&b.lines) {
        let d = x - y;
        assert!((d - d.round()).abs() < 1e-9, "{x} against {y}");
    }
    // And the winding of a Chern insulator.
    let h = |kx: f64, ky: f64| sigma(qwz(1., kx, ky));
    let wound = topology::winding(&flow(&h, 1, 1., SETTINGS, &mut gauge)).unwrap();
    assert_eq!(
        wound,
        topology::winding(&flow(&h, 1, 1., SETTINGS, &mut None)).unwrap()
    );
}

#[test]
fn the_flow_of_a_z2_odd_model_switches_kramers_partners_and_the_even_one_does_not() {
    // Kane-Mele, lambda_v = 0: the two centers split from a degenerate pair at k_perp = 0 and meet
    // again at 1/2 with the partners exchanged: the line in the gap is crossed once.
    let odd = topology::z2(&flow(
        &kane_mele(0.1, 0.05, 0.),
        2,
        0.5,
        SETTINGS,
        &mut None,
    ))
    .unwrap();
    assert_eq!((odd.parity, odd.crossings % 2), (1, 1));
    assert!(odd.lines.len() == odd.gaps.len() && odd.gaps.iter().all(|g| *g > 0.));
    let even = topology::z2(&flow(
        &kane_mele(0.1, 0.05, 0.8),
        2,
        0.5,
        SETTINGS,
        &mut None,
    ))
    .unwrap();
    assert_eq!(even.crossings % 2, 0);
}

#[test]
fn a_spin_degenerate_manifold_counts_every_crossing_twice() {
    // Each center of one Z2-odd block, duplicated: exact Kramers-like pairs everywhere.
    let h = bhz(1., None);
    let single = flow(&h, 2, 0.5, SETTINGS, &mut None);
    let doubled = Flow {
        k: single.k.clone(),
        centers: single
            .centers
            .iter()
            .map(|c| {
                let mut v: Vec<f64> = c.iter().chain(c).copied().collect();
                v.sort_by(f64::total_cmp);
                v
            })
            .collect(),
    };
    let z = topology::z2(&doubled).unwrap();
    assert_eq!((z.parity, z.crossings % 2), (0, 0));
}

#[test]
fn a_time_reversal_breaking_flow_raises_and_so_does_a_flow_that_moves_too_far_for_its_depth() {
    // Qi-Wu-Zhang alone (one center, a Chern insulator): no Kramers pairs.
    let h = |kx: f64, ky: f64| sigma(qwz(1., kx, ky));
    let one = flow(&h, 1, 0.5, SETTINGS, &mut None);
    let message = topology::z2(&one).unwrap_err().to_string();
    assert!(message.contains("not time-reversal invariant"), "{message}");
    // Pairs that split on a time-reversal-invariant line.
    let split = Flow {
        k: vec![0., 0.5],
        centers: vec![vec![0.1, 0.1], vec![0.1, 0.102]],
    };
    let message = topology::z2(&split).unwrap_err().to_string();
    assert!(message.contains("Kramers"), "{message}");
    // A flow to 1/2 only.
    assert!(topology::z2(&flow(&h, 1, 1., SETTINGS, &mut None)).is_err());
    // A center that moves a lot, with no halvings left: the error names the plane.
    let kane = kane_mele(0.1, 0.05, 0.);
    let err = topology::adaptive(
        "the k_2 = 1/2 plane",
        0.5,
        Adaptive {
            intervals: 1,
            depth: 0,
        },
        |t| Ok(centers(&kane, 2, 24, t, &mut None)),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("the k_2 = 1/2 plane"), "{err}");
}

#[test]
fn the_largest_gap_and_the_kramers_check_see_through_the_wrap_around() {
    // The widest gap is the one across 0: centers at 0.1 and 0.9 leave the wrap gap 0.2, the
    // inner gap 0.8.
    let (line, gap) = topology::largest_gap(&[0.1, 0.9]).unwrap();
    assert!(
        (gap - 0.8).abs() < 1e-12 && (line - 0.5).abs() < 1e-12,
        "{line} {gap}"
    );
    let (line, gap) = topology::largest_gap(&[0.1, 0.2, 0.3]).unwrap();
    assert!(
        (gap - 0.8).abs() < 1e-12 && (line - 0.7).abs() < 1e-12,
        "{line} {gap}"
    );
    // The first of equal gaps wins.
    let (line, _) = topology::largest_gap(&[0., 0.5]).unwrap();
    assert!((line - 0.25).abs() < 1e-12, "{line}");
    // A pair straddling 0 (1e-9 below and above the cell edge) is a pair.
    topology::check_kramers(&[1e-9, 0.3, 0.3, 1. - 1e-9], 0.).unwrap();
    assert!(topology::check_kramers(&[0.3, 0.3, 0.4], 0.).is_err());
}

/// The centers of `models` (decoupled, so the union of their centers) with `flat` Kramers pairs of
/// flat trivial bands (atomic limits, constant centers) at deterministic positions.
fn direct_sum(
    models: &[(&dyn Fn(f64, f64) -> Matrix, usize)],
    flat: usize,
    t: f64,
    gauge: &mut Option<Rng>,
) -> Vec<f64> {
    let mut all: Vec<f64> = models
        .iter()
        .flat_map(|(h, occupied)| centers(*h, *occupied, 24, t, gauge))
        .collect();
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for _ in 0..flat {
        let x = rng.next();
        all.extend([x, x]);
    }
    all.sort_by(f64::total_cmp);
    all
}

fn direct_sum_z2(models: &[(&dyn Fn(f64, f64) -> Matrix, usize)], flat: usize) -> u8 {
    let flow = topology::adaptive("a direct sum", 0.5, SETTINGS, |t| {
        Ok(direct_sum(models, flat, t, &mut None))
    })
    .unwrap();
    topology::z2(&flow).unwrap().parity
}

#[test]
fn direct_sums_add_their_indices_mod_two_with_many_centers_and_a_jumping_line() {
    let odd = kane_mele(0.1, 0.05, 0.);
    let even = kane_mele(0.1, 0.05, 0.8);
    let bhz_odd = bhz(1., None);
    let bhz_even = bhz(3., None);
    let (o, e) = (
        &odd as &dyn Fn(f64, f64) -> Matrix,
        &even as &dyn Fn(f64, f64) -> Matrix,
    );
    let (bo, be) = (
        &bhz_odd as &dyn Fn(f64, f64) -> Matrix,
        &bhz_even as &dyn Fn(f64, f64) -> Matrix,
    );
    for flat in [0, 20, 60] {
        assert_eq!(direct_sum_z2(&[(o, 2)], flat), 1, "odd, {flat} flat pairs");
        assert_eq!(direct_sum_z2(&[(e, 2)], flat), 0, "even, {flat} flat pairs");
        assert_eq!(
            direct_sum_z2(&[(o, 2), (bo, 2)], flat),
            0,
            "odd + odd, {flat}"
        );
        assert_eq!(
            direct_sum_z2(&[(o, 2), (be, 2)], flat),
            1,
            "odd + even, {flat}"
        );
        assert_eq!(
            direct_sum_z2(&[(e, 2), (be, 2)], flat),
            0,
            "even + even, {flat}"
        );
        assert_eq!(
            direct_sum_z2(&[(o, 2), (bo, 2), (o, 2)], flat),
            1,
            "three odd, {flat}"
        );
    }
}

#[test]
fn the_winding_of_strings_along_the_second_axis_is_minus_the_chern_number_of_the_plane() {
    // Strings along k_y with k_x transverse: the orientation of (a, b) = (y, x) is the reverse of
    // the plane's (x, y), so the winding flips; the caller of `flow::plane_flow` restores it.
    for m in [-1.5, 0.5, 1.5] {
        let h = |kx: f64, ky: f64| sigma(qwz(m, kx, ky));
        let swapped = |ky: f64, kx: f64| h(kx, ky);
        let along_x = topology::winding(&flow(&h, 1, 1., SETTINGS, &mut None)).unwrap();
        let along_y = topology::winding(&flow(&swapped, 1, 1., SETTINGS, &mut None)).unwrap();
        assert_eq!(along_y, -along_x, "m {m}");
        assert_eq!(along_x.abs(), 1, "m {m}");
    }
}
