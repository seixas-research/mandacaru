//! Which start the multi-start SCF returns when none converges (`molecular::scf`, `molecular::dft`):
//! the one closest to convergence (smallest density change of its last iteration), never one that
//! is merely lower in energy while still oscillating. An unreachable tolerance (1e-30) and an
//! iteration budget make every start unconverged on purpose; `guesses` adds starts one by one
//! (start 0 the core Hamiltonian, 1 its screened form, 2 a broken-symmetry rotation for UHF and
//! the UHF solution for GHF), so the residual of the returned state may never rise as starts are
//! added. A four-atom hydrogen chain (HAO basis, 2 A apart) for Hartree-Fock, a stretched H2
//! (2 A) for Kohn-Sham: the broken-symmetry start lies lower in energy and is still moving when
//! the budget ends, the others are nearly stationary.
use mandacaru_core::{
    math::SpatialGrid,
    molecular::{MeanField, MolecularIntegrals, ScfOptions},
    Context, Geometry, LengthUnit,
};

fn chain(ctx: &Context, atoms: usize, spacing: f64) -> MolecularIntegrals {
    let start = 2.5 - spacing * (atoms as f64 - 1.) / 2.;
    let positions = (0..atoms)
        .map(|i| [2.5, 2.5, start + spacing * i as f64])
        .collect();
    let geometry = Geometry::new(
        vec![1; atoms],
        positions,
        [
            [5., 0., 0.],
            [0., 5., 0.],
            [0., 0., 5. + spacing * atoms as f64],
        ],
        [false; 3],
        LengthUnit::Angstrom,
    )
    .unwrap();
    let grid =
        SpatialGrid::from_geometry(&geometry, [0.35 / 0.529177210903; 3], false, false).unwrap();
    MolecularIntegrals::new(
        ctx,
        geometry,
        grid,
        r#"{"name":"HAO"}"#,
        "spectral",
        None,
        Default::default(),
    )
    .unwrap()
}

fn budget(iterations: usize, guesses: usize) -> ScfOptions {
    ScfOptions {
        max_iterations: iterations,
        tolerance: 1e-30,
        guesses,
        ..Default::default()
    }
}

fn residual(state: &MeanField) -> f64 {
    assert!(!state.converged, "the tolerance is unreachable");
    state.iterations.last().unwrap().residual
}

/// The residual of the returned state with 1, 2 and 3 starts.
fn residuals(solve: impl Fn(usize) -> MeanField) -> [f64; 3] {
    let [a, b, c] = [1, 2, 3].map(|guesses| residual(&solve(guesses)));
    println!("residuals with 1, 2, 3 starts: {a:.3e} {b:.3e} {c:.3e}");
    [a, b, c]
}

#[test]
fn an_unconverged_hartree_fock_returns_the_start_closest_to_convergence() {
    let ctx = Context::new(Some(3), 1 << 30).unwrap();
    let integrals = chain(&ctx, 4, 2.);
    for (method, iterations) in [("uhf", 8), ("uhf", 15), ("ghf", 15)] {
        let [one, two, three] = residuals(|guesses| {
            integrals
                .mean_field(method, [2, 2], &budget(iterations, guesses))
                .unwrap()
        });
        assert!(
            three <= two && two <= one,
            "{method} after {iterations} iterations"
        );
    }
}

#[test]
fn an_unconverged_kohn_sham_returns_the_start_closest_to_convergence() {
    let ctx = Context::new(Some(3), 1 << 30).unwrap();
    let integrals = chain(&ctx, 2, 2.);
    for iterations in [8, 15] {
        let [one, two, three] = residuals(|guesses| {
            integrals
                .kohn_sham("lda", [1, 1], true, &budget(iterations, guesses))
                .unwrap()
        });
        assert!(three <= two && two <= one, "after {iterations} iterations");
    }
}
