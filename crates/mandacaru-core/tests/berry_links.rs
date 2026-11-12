//! The model-free Berry algebra (`periodic::berry::links`) on lattice models built here, so the
//! conventions are pinned independently of any crystal:
//! - a state localized at `x0` has the Berry phase `2 pi x0` and the Wannier center `x0`;
//! - the SSH chain has the phase 0 or pi, unchanged by a local gauge;
//! - the Qi-Wu-Zhang Chern number equals the continuum degree `(1/4 pi) int d.(d_x d x d_y d)/|d|^3`
//!   of its lower band, which fixes the curvature sign independently of the lattice formula;
//! - the Rice-Mele pump: the strip fluxes are the differences of adjacent string phases (Stokes),
//!   so the curvature sign is tied to the phase sign, and the Chern number is minus the winding of
//!   the phase;
//! - random unitary gauges at every k leave phases, Wilson-loop centers and fluxes unchanged.
mod berry_models_support;

use berry_models_support::*;
use mandacaru_core::{
    periodic::berry::links::{self, ALIASING},
    quantum::C64,
};
use std::f64::consts::PI;

#[test]
fn a_state_localized_at_x0_has_phase_two_pi_x0_and_its_wannier_center_at_x0() {
    // u_k = e^{-i k x0}: the link to k + b is e^{-i b x0} (a = 1).
    for x0 in [0.2, 0.37, -0.1] {
        let n = 16;
        let b = 2. * PI / n as f64;
        let links: Vec<Matrix> = (0..n)
            .map(|_| Matrix::from_element(1, 1, C64::from_polar(1., -b * x0)))
            .collect();
        let (phase, smallest) = links::string_phase(&links, 1e-10).unwrap();
        assert!(
            (links::wrap(phase - 2. * PI * x0)).abs() < 1e-12,
            "{x0}: {phase}"
        );
        assert!((smallest - 1.).abs() < 1e-12);
        let centers = links::wannier_centers(&links, 1e-10).unwrap();
        assert!(
            (centers[0] - x0.rem_euclid(1.)).abs() < 1e-12,
            "{x0}: {centers:?}"
        );
    }
}

#[test]
fn ssh_phase_is_zero_or_pi_invariant_under_local_gauges_and_singular_links_raise() {
    let n = 300;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for (hopping, topological) in [(0.5, true), (1.5, false)] {
        let states: Vec<Matrix> = (0..n)
            .map(|i| {
                let k = 2. * PI * i as f64 / n as f64;
                let phase = (C64::new(hopping, 0.) + C64::from_polar(1., k)).arg();
                Matrix::from_column_slice(
                    2,
                    1,
                    &[
                        -C64::from_polar(1. / 2f64.sqrt(), phase),
                        C64::new(1. / 2f64.sqrt(), 0.),
                    ],
                )
            })
            .collect();
        let (phase, smallest) = links::string_phase(&string(&states), 1e-10).unwrap();
        assert!((phase.abs() - if topological { PI } else { 0. }).abs() < 1e-10);
        assert!(smallest > 0.99);
        let gauged: Vec<Matrix> = states
            .iter()
            .map(|v| v * C64::from_polar(1., 2. * PI * rng.next()))
            .collect();
        let (again, _) = links::string_phase(&string(&gauged), 1e-10).unwrap();
        assert!(links::wrap(again - phase).abs() < 1e-10);
    }
    let zero = vec![Matrix::zeros(1, 1)];
    let message = links::string_phase(&zero, 1e-10).err().unwrap().to_string();
    assert!(message.contains("singular"), "{message}");
}

/// `(along_a, along_b)`: the unit-modulus determinants of the links of the `occupied` lowest
/// states of `hamiltonian` on an `n x n` mesh, with a gauge `gauge(k)` applied at every point.
fn plane_dets(
    n: usize,
    occupied: usize,
    hamiltonian: &dyn Fn(f64, f64) -> Matrix,
    gauge: &mut dyn FnMut() -> Matrix,
) -> (Vec<C64>, Vec<C64>) {
    let k = |i: usize| 2. * PI * i as f64 / n as f64;
    let states: Vec<Matrix> = (0..n * n)
        .map(|p| lowest(hamiltonian(k(p / n), k(p % n)), occupied) * gauge())
        .collect();
    let at = |i: usize, j: usize| &states[(i % n) * n + j % n];
    let mut a = vec![];
    let mut b = vec![];
    for i in 0..n {
        for j in 0..n {
            a.push(
                links::link_phase(&link(at(i, j), at(i + 1, j)), 1e-10)
                    .unwrap()
                    .0,
            );
            b.push(
                links::link_phase(&link(at(i, j), at(i, j + 1)), 1e-10)
                    .unwrap()
                    .0,
            );
        }
    }
    (a, b)
}

/// `(1/4 pi) int d.(d_x d x d_y d) / |d|^3`: the degree of `d / |d|` on the torus, by a midpoint
/// rule with analytic derivatives (independent of the link formula).
fn qwz_degree(m: f64) -> f64 {
    let n = 400;
    let h = 2. * PI / n as f64;
    let mut total = 0.;
    for i in 0..n {
        for j in 0..n {
            let (kx, ky) = ((i as f64 + 0.5) * h, (j as f64 + 0.5) * h);
            let d = qwz(m, kx, ky);
            let dx = [kx.cos(), 0., -kx.sin()];
            let dy = [0., ky.cos(), -ky.sin()];
            let cross = [
                dx[1] * dy[2] - dx[2] * dy[1],
                dx[2] * dy[0] - dx[0] * dy[2],
                dx[0] * dy[1] - dx[1] * dy[0],
            ];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            total += (d[0] * cross[0] + d[1] * cross[1] + d[2] * cross[2]) / r.powi(3);
        }
    }
    total * h * h / (4. * PI)
}

#[test]
fn qwz_chern_numbers_follow_the_continuum_degree_across_the_phase_diagram() {
    let mut seen = vec![];
    for m in [-3., -1., 1., 3.] {
        let identity = &mut || Matrix::identity(1, 1);
        let (a, b) = plane_dets(24, 1, &|kx, ky| sigma(qwz(m, kx, ky)), identity);
        let field = links::lattice_field(&a, &b, [24, 24]).unwrap();
        let degree = qwz_degree(m);
        assert!(
            (degree - degree.round()).abs() < 1e-3,
            "m {m}: degree {degree}"
        );
        assert_eq!(
            field.chern,
            degree.round() as i64,
            "m {m}: lattice against continuum"
        );
        assert!(field.max_flux < 0.5 * PI, "m {m}: {}", field.max_flux);
        seen.push(field.chern);
    }
    assert_eq!(seen[0], 0);
    assert_eq!(seen[3], 0);
    assert_eq!(seen[1].abs(), 1);
    assert_eq!(seen[1], -seen[2]);
}

#[test]
fn random_unitary_gauges_leave_phases_centers_fluxes_and_chern_numbers_unchanged() {
    // Two decoupled QWZ copies: two occupied bands, mixed by a random U(2) at every k.
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let copies = |m1: f64, m2: f64| {
        move |kx: f64, ky: f64| {
            let mut h = Matrix::zeros(4, 4);
            h.view_mut((0, 0), (2, 2))
                .copy_from(&sigma(qwz(m1, kx, ky)));
            h.view_mut((2, 2), (2, 2))
                .copy_from(&sigma(qwz(m2, kx, ky)));
            h
        }
    };
    let n = 20;
    for (m1, m2) in [(1., 1.), (1., -1.), (1., 3.)] {
        let model = copies(m1, m2);
        let (a, b) = plane_dets(n, 2, &model, &mut || Matrix::identity(2, 2));
        let plain = links::lattice_field(&a, &b, [n, n]).unwrap();
        let (a, b) = plane_dets(n, 2, &model, &mut || rng.unitary(2));
        let gauged = links::lattice_field(&a, &b, [n, n]).unwrap();
        assert_eq!(plain.chern, gauged.chern);
        for (x, y) in plain.flux.iter().zip(&gauged.flux) {
            assert!((x - y).abs() < 1e-10);
        }
        let single = |m: f64| {
            let (a, b) = plane_dets(n, 1, &|kx, ky| sigma(qwz(m, kx, ky)), &mut || {
                Matrix::identity(1, 1)
            });
            links::lattice_field(&a, &b, [n, n]).unwrap().chern
        };
        assert_eq!(
            plain.chern,
            single(m1) + single(m2),
            "the Chern number adds over bands"
        );
        // A string along kx at ky = 2 pi * 3 / n: phase and Wilson centers, plain and gauged.
        let along = |gauge: &mut dyn FnMut() -> Matrix| {
            let states: Vec<Matrix> = (0..n)
                .map(|i| {
                    let kx = 2. * PI * i as f64 / n as f64;
                    lowest(model(kx, 2. * PI * 3. / n as f64), 2) * gauge()
                })
                .collect();
            string(&states)
        };
        let reference = along(&mut || Matrix::identity(2, 2));
        let rotated = along(&mut || rng.unitary(2));
        let (p, _) = links::string_phase(&reference, 1e-10).unwrap();
        let (q, _) = links::string_phase(&rotated, 1e-10).unwrap();
        assert!(links::wrap(p - q).abs() < 1e-10);
        let x = links::wannier_centers(&reference, 1e-10).unwrap();
        let y = links::wannier_centers(&rotated, 1e-10).unwrap();
        for (u, v) in x.iter().zip(&y) {
            let d = u - v;
            assert!((d - d.round()).abs() < 1e-9, "{x:?} {y:?}");
        }
        let sum = x.iter().sum::<f64>() - p / (2. * PI);
        assert!(
            (sum - sum.round()).abs() < 1e-9,
            "centers sum to the phase over 2 pi"
        );
    }
}

#[test]
fn rice_mele_pump_strip_fluxes_are_phase_differences_and_chern_is_minus_the_winding() {
    // H(k, t) = (v + w cos k) sx + w sin k sy + D sz, v = 1 + cos(t)/2, w = 1 - cos(t)/2,
    // D = sin(t)/2: gapped for every t, a closed loop around the gap closing at v = w, D = 0.
    let (nk, nt) = (48, 40);
    let state = |i: usize, j: usize| {
        let k = 2. * PI * (i % nk) as f64 / nk as f64;
        let t = 2. * PI * (j % nt) as f64 / nt as f64;
        let (v, w, d) = (1. + 0.5 * t.cos(), 1. - 0.5 * t.cos(), 0.5 * t.sin());
        lowest(sigma([v + w * k.cos(), w * k.sin(), d]), 1)
    };
    let states: Vec<Vec<Matrix>> = (0..nk)
        .map(|i| (0..nt).map(|j| state(i, j)).collect())
        .collect();
    let at = |i: usize, j: usize| &states[i % nk][j % nt];
    let mut a = vec![];
    let mut b = vec![];
    for i in 0..nk {
        for j in 0..nt {
            a.push(
                links::link_phase(&link(at(i, j), at(i + 1, j)), 1e-10)
                    .unwrap()
                    .0,
            );
            b.push(
                links::link_phase(&link(at(i, j), at(i, j + 1)), 1e-10)
                    .unwrap()
                    .0,
            );
        }
    }
    let field = links::lattice_field(&a, &b, [nk, nt]).unwrap();
    assert_eq!(field.chern.abs(), 1, "one charge is pumped per cycle");
    let phases: Vec<f64> = (0..nt)
        .map(|j| {
            let column: Vec<Matrix> = (0..nk).map(|i| at(i, j).clone()).collect();
            links::string_phase(&string(&column), 1e-10).unwrap().0
        })
        .collect();
    let mut winding = 0.;
    for j in 0..nt {
        // The plaquettes (i, j) for all i form the strip between strings j and j + 1.
        let strip: f64 = (0..nk).map(|i| field.flux[i * nt + j]).sum();
        let difference = phases[j] - phases[(j + 1) % nt];
        assert!(
            links::wrap(strip - difference).abs() < 1e-9,
            "Stokes at t-step {j}"
        );
        winding += links::wrap(phases[(j + 1) % nt] - phases[j]);
    }
    assert!((winding / (2. * PI) + field.chern as f64).abs() < 1e-9);
}

#[test]
fn an_aliased_plaquette_raises_and_a_resolved_one_does_not() {
    let n = [4, 4];
    let ones = vec![C64::new(1., 0.); 16];
    for (theta, aliased) in [(0.5 * PI, false), (0.9 * PI, true)] {
        let mut a = ones.clone();
        a[0] = C64::from_polar(1., theta);
        let result = links::lattice_field(&a, &ones, n);
        assert_eq!(result.is_err(), aliased, "theta {theta}");
        if let Ok(field) = result {
            assert_eq!(field.chern, 0);
            assert!((field.max_flux - theta).abs() < 1e-12);
        }
    }
    assert!(ALIASING < PI);
}

#[test]
fn string_phases_average_on_the_branch_of_their_circular_mean() {
    let (mean, spread) = links::circular_average(&[PI - 0.01, -PI + 0.01]).unwrap();
    assert!(links::wrap(mean - PI).abs() < 1e-12, "{mean}");
    assert!((spread - 0.01).abs() < 1e-12);
    let (mean, _) = links::circular_average(&[0.2, 0.4]).unwrap();
    assert!((mean - 0.3).abs() < 1e-12);
    assert!((links::wrap(-PI) - PI).abs() < 1e-15);
    assert!(links::circular_average(&[]).is_err());
}

#[test]
fn a_degenerate_unitary_spectrum_is_resolved_by_hermitian_solves() {
    // Flat bands give Wilson loops with exactly repeated eigenphases (where a complex Schur form
    // may not converge), and two phases sharing a cosine.
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let phases = [0.3, 0.3, 0.3, -1.2, 1.2, PI, 2.0, 0., 0., 0., 0., 0.];
    let n = phases.len();
    let v = rng.unitary(n);
    let d = Matrix::from_diagonal(&nalgebra::DVector::from_iterator(
        n,
        phases.iter().map(|p| C64::from_polar(1., *p)),
    ));
    let w = &v * d * v.adjoint();
    let mut got: Vec<f64> = links::unitary_spectrum(&w)
        .unwrap()
        .iter()
        .map(|z| z.arg())
        .collect();
    let mut want: Vec<f64> = phases.iter().map(|p| links::wrap(*p)).collect();
    got.sort_by(f64::total_cmp);
    want.sort_by(f64::total_cmp);
    for (g, e) in got.iter().zip(&want) {
        assert!(links::wrap(g - e).abs() < 1e-9, "{got:?} against {want:?}");
    }
    assert!(links::unitary_spectrum(&(w * C64::new(2., 0.))).is_err());
}
