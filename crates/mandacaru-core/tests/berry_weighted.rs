//! The occupation-weighted lattice field of a metal (`periodic::berry::weighted`) on lattice
//! models built here, against independent continuum integrals:
//! - a doped Qi-Wu-Zhang model: the filling-weighted plane number
//!   `nu(mu) = (1/4 pi) int (f_- - f_+) d.(d_x d x d_y d)/|d|^3` (Fermi-Dirac at `mu`, analytic
//!   derivatives, midpoint rule), quantized in the gap, continuous through the band, with an error
//!   that falls as the mesh refines;
//! - two decoupled copies whose bands cross (the merge path) against the sum of their continua;
//! - the insulator limit equals the unweighted `links::lattice_field` plaquette by plaquette;
//! - random phases at every k change nothing; a time-reversal-symmetric (real) model gives zero.
mod berry_models_support;

use berry_models_support::*;
use mandacaru_core::{
    periodic::{
        berry::{links, weighted},
        spectrum::Smearing,
    },
    quantum::C64,
};
use nalgebra::SymmetricEigen;
use std::f64::consts::PI;

/// Sorted levels and the matching eigenvectors (columns) of a Hermitian matrix.
fn spectrum(h: Matrix) -> (Vec<f64>, Matrix) {
    let eig = SymmetricEigen::new(h);
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|a, b| eig.eigenvalues[*a].total_cmp(&eig.eigenvalues[*b]));
    let n = order.len();
    (
        order.iter().map(|i| eig.eigenvalues[*i]).collect(),
        Matrix::from_fn(n, n, |i, j| eig.eigenvectors[(i, order[j])]),
    )
}

struct Plane {
    levels: Vec<Vec<f64>>,
    occupations: Vec<Vec<f64>>,
    along_a: Vec<Matrix>,
    along_b: Vec<Matrix>,
}

/// Every band of `model` on an `n x n` mesh, Fermi-Dirac occupations at `mu`, width `width`
/// (one electron per band), a phase `gauge(p, band)` on every eigenvector.
fn plane(
    n: usize,
    model: &dyn Fn(f64, f64) -> Matrix,
    mu: f64,
    width: f64,
    gauge: &mut dyn FnMut() -> f64,
) -> Plane {
    let k = |i: usize| 2. * PI * i as f64 / n as f64;
    let mut levels = vec![];
    let mut states = vec![];
    for p in 0..n * n {
        let (e, mut v) = spectrum(model(k(p / n), k(p % n)));
        for mut column in v.column_iter_mut() {
            column *= C64::from_polar(1., gauge());
        }
        levels.push(e);
        states.push(v);
    }
    let at = |i: usize, j: usize| (i % n) * n + j % n;
    let mut along_a = vec![];
    let mut along_b = vec![];
    for i in 0..n {
        for j in 0..n {
            along_a.push(link(&states[at(i, j)], &states[at(i + 1, j)]));
            along_b.push(link(&states[at(i, j)], &states[at(i, j + 1)]));
        }
    }
    let occupations = levels
        .iter()
        .map(|e| {
            e.iter()
                .map(|x| Smearing::FermiDirac.occupation((x - mu) / width))
                .collect()
        })
        .collect();
    Plane {
        levels,
        occupations,
        along_a,
        along_b,
    }
}

fn field(p: &Plane, n: usize) -> weighted::WeightedField {
    weighted::weighted_field(
        &p.levels,
        &p.occupations,
        &p.along_a,
        &p.along_b,
        [n, n],
        1.,
    )
    .unwrap()
}

/// `(1/4 pi) int (f_- - f_+) d.(d_x d x d_y d)/|d|^3` of a QWZ model at `mu`: the lower band's
/// curvature is `+(1/2) d^.(d_x d^ x d_y d^)`, the upper band's its negative.
fn continuum(m: f64, mu: f64, width: f64) -> f64 {
    let n = 600;
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
            let f = |e: f64| Smearing::FermiDirac.occupation((e - mu) / width);
            total +=
                (f(-r) - f(r)) * (d[0] * cross[0] + d[1] * cross[1] + d[2] * cross[2]) / r.powi(3);
        }
    }
    total * h * h / (4. * PI)
}

fn qwz_model(m: f64) -> impl Fn(f64, f64) -> Matrix {
    move |kx, ky| sigma(qwz(m, kx, ky))
}

#[test]
fn a_doped_qwz_model_follows_the_continuum_filling_and_is_quantized_in_the_gap() {
    let (m, width) = (1., 0.1);
    let mut no_gauge = || 0.;
    // In the gap the plane number is the Chern number, exactly.
    let gap = field(&plane(24, &qwz_model(m), 0., 0.01, &mut no_gauge), 24);
    let chern = continuum(m, 0., 0.01);
    assert!(
        (chern - chern.round()).abs() < 1e-3 && chern.round().abs() == 1.,
        "{chern}"
    );
    assert!((gap.number - chern.round()).abs() < 1e-12, "{}", gap.number);
    assert_eq!(gap.merged, 0);
    // In the lower band, inside the upper band and above both: the continuum, approached as the
    // mesh refines; continuous in mu.
    let mut previous: Option<(f64, f64)> = None;
    for mu in [-2.0, -1.5, 1.5, 2.0, 3.5] {
        let exact = continuum(m, mu, width);
        let coarse = field(&plane(32, &qwz_model(m), mu, width, &mut no_gauge), 32).number;
        let fine = field(&plane(64, &qwz_model(m), mu, width, &mut no_gauge), 64).number;
        eprintln!("mu {mu}: continuum {exact:.6}, N = 32 {coarse:.6}, N = 64 {fine:.6}");
        assert!(
            (fine - exact).abs() < 5e-3,
            "mu {mu}: {fine} against {exact}"
        );
        assert!(
            (fine - exact).abs() <= (coarse - exact).abs() + 1e-9,
            "mu {mu}: refining"
        );
        if let Some((last_mu, last)) = previous {
            if mu - last_mu <= 0.5 {
                assert!(
                    (fine - last).abs() < 0.6,
                    "continuity from {last_mu} to {mu}"
                );
            }
        }
        previous = Some((mu, fine));
    }
    // Both bands full: no conductivity.
    let full = field(&plane(24, &qwz_model(m), 10., 0.01, &mut no_gauge), 24);
    assert!(full.number.abs() < 1e-10, "{}", full.number);
}

#[test]
fn crossing_bands_of_two_decoupled_copies_merge_and_add_their_continua() {
    // Copy 1 (m = 1) spans |e| in [1, 3], copy 2 (m = -1.6) |e| in [0.4, 3.6]: their bands cross.
    let (m1, m2, width) = (1., -1.6, 0.1);
    let model = move |kx: f64, ky: f64| {
        let mut h = Matrix::zeros(4, 4);
        h.view_mut((0, 0), (2, 2))
            .copy_from(&sigma(qwz(m1, kx, ky)));
        h.view_mut((2, 2), (2, 2))
            .copy_from(&sigma(qwz(m2, kx, ky)));
        h
    };
    let mut no_gauge = || 0.;
    for mu in [-1.8, 0.7] {
        let exact = continuum(m1, mu, width) + continuum(m2, mu, width);
        let coarse = field(&plane(32, &model, mu, width, &mut no_gauge), 32);
        let fine = field(&plane(64, &model, mu, width, &mut no_gauge), 64);
        eprintln!(
            "mu {mu}: continuum {exact:.6}, N = 32 {:.6} ({} merged), N = 64 {:.6} ({} merged)",
            coarse.number, coarse.merged, fine.number, fine.merged
        );
        assert!(fine.merged > 0, "the crossings take the merge path");
        assert!(
            (fine.number - exact).abs() < 2e-2,
            "mu {mu}: {} against {exact}",
            fine.number
        );
        assert!((fine.number - exact).abs() < (coarse.number - exact).abs() + 1e-9);
    }
}

#[test]
fn the_insulator_limit_is_the_unweighted_lattice_field() {
    let n = 20;
    let mut no_gauge = || 0.;
    let mut p = plane(n, &qwz_model(1.), 0., 0.01, &mut no_gauge);
    for row in &mut p.occupations {
        *row = vec![1., 0.];
    }
    let weighted = field(&p, n);
    let dets = |links: &[Matrix]| -> Vec<C64> {
        links
            .iter()
            .map(|m| {
                let block = m.view((0, 0), (1, 1)).into_owned();
                links::link_phase(&block, 1e-10).unwrap().0
            })
            .collect()
    };
    let plain = links::lattice_field(&dets(&p.along_a), &dets(&p.along_b), [n, n]).unwrap();
    for (a, b) in weighted.flux.iter().zip(&plain.flux) {
        assert!((a - b).abs() < 1e-14);
    }
    assert!((weighted.number - plain.chern as f64).abs() < 1e-12);
}

#[test]
fn random_phases_change_nothing_and_a_real_model_gives_zero() {
    let mut rng = Rng(0xdead_beef_cafe_f00d);
    let n = 24;
    let plain = field(&plane(n, &qwz_model(1.), -1.8, 0.1, &mut || 0.), n);
    let gauged = field(
        &plane(n, &qwz_model(1.), -1.8, 0.1, &mut || 2. * PI * rng.next()),
        n,
    );
    for (a, b) in plain.flux.iter().zip(&gauged.flux) {
        assert!((a - b).abs() < 1e-12);
    }
    // H(k) real symmetric, so H(-k) = H(k)*: the curvature is odd and every filling gives zero.
    // d_z > 0 everywhere: the bands never touch (a real model's touching points carry pi flux).
    let real = |kx: f64, ky: f64| sigma([kx.cos() + 0.3, 0., 2.5 + ky.cos() + 0.2 * kx.sin()]);
    for mu in [-2.0, -1.6, 2.0] {
        let zero = field(&plane(n, &real, mu, 0.1, &mut || 0.), n);
        assert!(zero.number.abs() < 1e-10, "mu {mu}: {}", zero.number);
    }
}

#[test]
fn the_occupation_window_bounds_every_smearing() {
    for scheme in [
        Smearing::FermiDirac,
        Smearing::Gaussian,
        Smearing::MethfesselPaxton,
    ] {
        let x = weighted::occupation_window(|x| scheme.occupation(x));
        assert!(x > 1. && x < 40., "{scheme:?}: {x}");
        for step in 0..200 {
            let y = x + step as f64 * 0.1;
            assert!(
                scheme.occupation(y).abs() <= weighted::EMPTY,
                "{scheme:?} at {y}"
            );
        }
    }
}

#[test]
fn a_real_point_node_merges_instead_of_aliasing_at_every_mesh_and_gives_zero() {
    // d = (sin(kx - x0), 0, 1/2 + cos(kx - x0) + cos(ky - y0)) is real: H(-k) = H(k)*, so every
    // filling gives zero. Its bands touch at the point nodes kx = x0 + pi, ky = y0 +- pi/3, which
    // carry a flux of pi with every link of the plaquette around them large: before B2's fix the
    // field raised there at any mesh (vera, 2026-11-11: 19 of 40 random offsets at N = 16).
    let node = |x0: f64, y0: f64| {
        move |kx: f64, ky: f64| {
            sigma([(kx - x0).sin(), 0., 0.5 + (kx - x0).cos() + (ky - y0).cos()])
        }
    };
    for n in [16, 32, 64] {
        // x0 = pi + h/2, y0 = pi/3 + h/2: one node at (h/2, h/2), the center of plaquette (0, 0).
        let h = 2. * PI / n as f64;
        let centered = node(PI + h / 2., PI / 3. + h / 2.);
        for mu in [0.0, 0.3] {
            let field = field(&plane(n, &centered, mu, 0.05, &mut || 0.), n);
            assert!(
                field.number.abs() < 1e-12,
                "N {n}, mu {mu}: {}",
                field.number
            );
            assert!(field.merged > 0, "N {n}: the node merged its two bands");
        }
    }
    let mut rng = Rng(0x0dd5_eed5_0f5e_7f5e);
    for _ in 0..40 {
        let model = node(2. * PI * rng.next(), 2. * PI * rng.next());
        let field = field(&plane(16, &model, 0.0, 0.05, &mut || 0.), 16);
        assert!(field.number.abs() < 1e-12, "{}", field.number);
    }
}
