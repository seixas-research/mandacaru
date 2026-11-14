//! Maximally localized Wannier functions of DFT crystals (PAW-LCAO SZ, Rust setup), checked by
//! identities that hold whatever the numbers are (dev_plan/WANNIER_PLAN.md, section 7):
//! - the center of the function of one isolated band is its Berry phase over 2 pi along every
//!   G_j, modulo a lattice vector: exactly on an orthogonal mesh (one shell per axis), up to the
//!   finite-difference error of the shells on a skew one;
//! - the gauge-invariant spread Omega_I does not change under any unitary gauge of the links;
//! - the minimization falls monotonically and reaches the same minimum from two guesses.
//! Needs `MANDACARU_PAW_PATH`.
mod berry_models_support;
mod berry_support;
mod setup_shells_support;

use berry_models_support::Rng;
use berry_support::{h2_chain, lih, solve, Solved};
use mandacaru_core::{
    periodic::{
        berry::{strings, Bands, Metal},
        bloch_mesh::MeshIndex,
        wannier::{self, sampling, setup::Wannier, Options},
    },
    quantum::C64,
};
use nalgebra::{DMatrix, Vector3};
use serde_json::json;
use std::f64::consts::PI;

type Matrix = DMatrix<C64>;

fn localize(solved: &Solved, size: [usize; 3], trials: serde_json::Value) -> Wannier {
    let request = json!({ "size": size, "trials": trials });
    Wannier::build(
        solved.crystal.clone(),
        solved.state.clone(),
        solved.setup.clone(),
        &request.to_string(),
        &Options {
            tolerance: 1e-12,
            gradient_tolerance: 1e-8,
            ..Options::default()
        },
    )
    .unwrap()
}

/// `max_j |x_j - gamma_j / 2 pi|` modulo 1: the center of the single function of `wannier` in
/// fractional coordinates against the string-averaged Berry phases of the occupied band.
fn center_against_phases(solved: &Solved, wannier: &Wannier, size: [usize; 3]) -> f64 {
    let center = Vector3::from(wannier.channels[0].localized.spread.centers.as_slice()[0]);
    let fractional = solved.setup.lattice.try_inverse().unwrap() * center;
    let sampler = solved.sampler();
    // One lattice of frames serves the strings along all three directions.
    let manifold = sampler.manifold(Bands::Occupied, Metal::Undefined).unwrap();
    let lattice = sampler.lattice(size, [0.; 3], &manifold).unwrap();
    (0..3)
        .map(|j| {
            let phase = strings::strings(&sampler, &lattice, j).unwrap().mean[0];
            let d = fractional[j] - phase / (2. * PI);
            (d - d.round()).abs()
        })
        .fold(0., f64::max)
}

#[test]
fn the_center_of_an_isolated_band_on_an_orthogonal_mesh_is_its_berry_phase_over_two_pi() {
    // Orthogonal: the shells are +-G_j/n_j, and the center is the string phase exactly.
    let chain = solve(&h2_chain("lda-sr", "lda"));
    let size = [4, 2, 2];
    let wannier = localize(
        &chain,
        size,
        json!([{"center": [3.0, 4., 4.], "orbital": "s"}]),
    );
    assert!(wannier.channels[0].localized.converged);
    let miss = center_against_phases(&chain, &wannier, size);
    eprintln!("orthogonal mesh: center off the Berry phase by {miss:e}");
    assert!(
        miss < 1e-8,
        "orthogonal mesh: center off the Berry phase by {miss:e}"
    );
}

#[test]
#[ignore = "slow: 11 s (measured 2026-11-12 after the frames were built in chunks; 84 s before); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test wannier_crystal -- --ignored the_center_of_an_isolated_band_on_a_skew_mesh_converges_to_its_berry_phase"]
fn the_center_of_an_isolated_band_on_a_skew_mesh_converges_to_its_berry_phase() {
    // Skew (polar LiH): equal up to the shells' finite-difference error, which falls with the mesh.
    let polar = solve(&lih("lda-sr"));
    let at = Vector3::from([0.1 + 3. / 7., 0.2, 0.15]);
    let h: [f64; 3] = (polar.setup.lattice * at).into();
    let coarse = center_against_phases(
        &polar,
        &localize(&polar, [4, 2, 2], json!([{"center": h, "orbital": "s"}])),
        [4, 2, 2],
    );
    let fine = center_against_phases(
        &polar,
        &localize(&polar, [4, 4, 4], json!([{"center": h, "orbital": "s"}])),
        [4, 4, 4],
    );
    eprintln!("skew mesh: center off the Berry phase by {coarse:e} (4x2x2), {fine:e} (4^3)");
    assert!(
        coarse < 2e-3 && fine < coarse,
        "skew mesh: {coarse:e} then {fine:e}"
    );
}

#[test]
fn the_invariant_spread_is_unchanged_by_random_gauges_of_dft_links() {
    let chain = solve(&h2_chain("lda-sr", "lda"));
    let (c, size) = (&*chain.crystal, [4, 2, 2]);
    let sampler = chain.sampler();
    let mesh_index = MeshIndex::new(size, [0.; 3]).unwrap();
    let frames = sampler
        .frames(&mesh_index, &[vec![0, 1]], |_, _| Ok(()))
        .unwrap()
        .remove(0);
    let mesh = sampling::overlap_mesh(c, &frames, size, sampler.augmentation()).unwrap();
    let identity = vec![Matrix::identity(2, 2); mesh.points];
    let reference = wannier::spread(&c.context, &mesh, &identity, None).unwrap();
    let mut rng = Rng(0x5eed_1234);
    let gauge: Vec<Matrix> = (0..mesh.points).map(|_| rng.unitary(2)).collect();
    let rotated = wannier::spread(&c.context, &mesh, &gauge, None).unwrap();
    assert!(
        (rotated.invariant - reference.invariant).abs() < 1e-10,
        "Omega_I moved by {:e}",
        rotated.invariant - reference.invariant
    );
    assert!(
        (rotated.total() - reference.total()).abs() > 1e-3,
        "the random gauge left the whole spread unchanged: the check is vacuous"
    );
}

#[test]
fn the_spread_falls_monotonically_to_one_minimum_from_two_guesses() {
    let chain = solve(&h2_chain("lda-sr", "lda"));
    let size = [4, 2, 2];
    let on_atoms = json!([
        {"center": [2.3, 4., 4.], "orbital": "s"},
        {"center": [3.7, 4., 4.], "orbital": "s"},
    ]);
    let displaced = json!([
        {"center": [2.96, 4.38, 3.81], "orbital": "s"},
        {"center": [3.23, 3.72, 4.57], "orbital": "s"},
    ]);
    let mut finals = vec![];
    for trials in [on_atoms, displaced] {
        let wannier = localize(&chain, size, trials);
        let localized = &wannier.channels[0].localized;
        assert!(localized.converged);
        let totals: Vec<f64> = localized.history.iter().map(|h| h.iter().sum()).collect();
        assert!(
            totals.windows(2).all(|w| w[1] <= w[0] + 1e-14),
            "{totals:?}"
        );
        let invariant = localized.history[0][0];
        assert!(localized
            .history
            .iter()
            .all(|h| (h[0] - invariant).abs() < 1e-10));
        finals.push(localized.spread.total());
    }
    assert!(
        (finals[0] - finals[1]).abs() < 1e-9,
        "two guesses end {:e} apart",
        finals[0] - finals[1]
    );
}

#[test]
fn converged_means_the_gradient_norm_is_below_its_threshold_from_a_rotated_start() {
    // `converged` must say what the options document: the gradient norm below
    // `gradient_tolerance` (and the last step below `tolerance`), not a stall of the line search.
    let chain = solve(&h2_chain("lda-sr", "lda"));
    let (c, size) = (&*chain.crystal, [4, 2, 2]);
    let sampler = chain.sampler();
    let mesh_index = MeshIndex::new(size, [0.; 3]).unwrap();
    let frames = sampler
        .frames(&mesh_index, &[vec![0, 1]], |_, _| Ok(()))
        .unwrap()
        .remove(0);
    let mesh = sampling::overlap_mesh(c, &frames, size, sampler.augmentation()).unwrap();
    // A smooth gauge (the Bloch states themselves) turned by a sizable random rotation
    // `exp(W_k)`, `|W_k| ~ 0.6`, at every k. (A Haar-random gauge leaves link diagonals near zero,
    // where Im ln M_nn and the spread's gradient are singular: no minimizer converges from there,
    // and the result must, and does, say converged = false.)
    let mut rng = Rng(0x0dd_ba11);
    let start: Vec<Matrix> = (0..mesh.points)
        .map(|_| {
            let a = Matrix::from_fn(2, 2, |_, _| C64::new(rng.normal(), rng.normal()));
            let generator = (&a - a.adjoint()) * C64::new(0.3, 0.);
            wannier::unitary_step(&generator).unwrap()
        })
        .collect();
    let options = Options::default();
    let result = wannier::localize(&c.context, &mesh, &start, None, &options).unwrap();
    let final_gradient = wannier::gradient(
        &c.context,
        &mesh,
        &result.gauge,
        result.spread.centers.as_slice(),
    )
    .unwrap();
    let norm = wannier::gradient_norm(&final_gradient.values);
    let tail: Vec<f64> = result
        .history
        .iter()
        .rev()
        .take(4)
        .map(|h| h.iter().sum())
        .collect();
    assert!(
        result.converged,
        "{} steps, gradient norm {norm:e}, last spreads {tail:?}",
        result.history.len() - 1
    );
    assert!(
        norm < options.gradient_tolerance,
        "converged at gradient norm {norm:e}, threshold {:e}",
        options.gradient_tolerance
    );
    let totals: Vec<f64> = result.history.iter().map(|h| h.iter().sum()).collect();
    assert!(
        totals.windows(2).all(|w| w[1] <= w[0] + 1e-12),
        "{totals:?}"
    );
    eprintln!(
        "rotated start: {} steps to gradient norm {norm:e}, spread {:.10} Bohr^2",
        result.history.len() - 1,
        result.spread.total()
    );
}

#[test]
fn each_function_carries_the_weight_of_its_own_atoms_trial() {
    let chain = solve(&h2_chain("lda-sr", "lda"));
    let wannier = localize(
        &chain,
        [4, 2, 2],
        json!([
            {"center": [2.3, 4., 4.], "orbital": "s", "label": "H0"},
            {"center": [3.7, 4., 4.], "orbital": "s", "label": "H1"},
        ]),
    );
    let w = &wannier.channels[0].trial_weights;
    assert_eq!(w.len(), 4);
    assert!(w.iter().all(|x| (0. ..=1. + 1e-12).contains(x)), "{w:?}");
    // The function nearer an atom has more of that atom's trial, and both are substantial.
    let centers = wannier.channels[0].localized.spread.centers.as_slice();
    for n in 0..2 {
        let own = if centers[n][0] < 3.0 { 0 } else { 1 };
        assert!(
            w[n * 2 + own] > w[n * 2 + 1 - own] && w[n * 2 + own] > 0.3,
            "{w:?}"
        );
    }
}
