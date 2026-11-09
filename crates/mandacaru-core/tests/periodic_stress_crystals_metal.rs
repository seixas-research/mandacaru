//! AS-6 on a metal: the one-atom fcc Al primitive cell (cubic, Fermi-Dirac occupations of
//! 0.3 eV). The stress of the Mermin free energy at FIXED occupations carries no explicit
//! smearing term (stationarity in the occupations), so the analytic stress must meet the
//! reconverged free-energy difference with partially occupied bands, which a molecule or an
//! insulator never exercises. `al_small` (a 2x1x1 mesh, two k-points) is the cheap metal of the
//! default run; `al` (a 3x3x3 Gamma-centered mesh, 14 k-points after time reversal, complex Bloch
//! phases on every translation, the cubic symmetry of the mesh intact) is the full-mesh case, about
//! 5 minutes, and the one whose stress must be proportional to the identity. Needs
//! `MANDACARU_PAW_PATH`.
//!
//! `cargo test --release -p mandacaru-core --test periodic_stress_crystals_metal -- --ignored --test-threads=1`
mod periodic_gradient_crystals_support;
mod periodic_stress_crystals_support;

use periodic_gradient_crystals_support::Data;
use periodic_stress_crystals_support::{
    gates::bites,
    run::{
        check, equation_of_state, isotropy, solve, stress_at, weak_terms, SCF_DENSITY_TOLERANCE,
        SCF_TOLERANCE,
    },
};

/// Terms of an Al stress the oracle cannot discriminate: the compensation hats and the
/// short-range ion-compensation sum are below 1e-6 Ha for a one-atom metal (2 gates), and the
/// gradient term is empty for LDA.
const SMALL: [&str; 3] = ["hats", "linear", "xc_gradient"];

/// The occupations of the converged bands are partial: a metal the test can tell from an insulator.
fn partial_bands(data: &Data) -> usize {
    let result = data.solve();
    let degeneracy = data.crystal.degeneracy();
    for (k, f) in result.occupations.iter().enumerate() {
        println!("  occupations of k-point {k}: {f:.4?} (degeneracy {degeneracy})");
    }
    result
        .occupations
        .iter()
        .flatten()
        .filter(|f| {
            let f = *f / degeneracy;
            f > 1e-3 && f < 1. - 1e-3
        })
        .count()
}

#[test]
#[ignore = "slow: 138 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_crystals_metal -- --ignored the_stress_of_a_converged_fermi_dirac_metal"]
fn the_stress_of_a_converged_fermi_dirac_metal() {
    let (mut data, stress, want) = check("al_small", true);
    let partial = partial_bands(&data);
    println!("  bands with an occupation strictly between 1e-3 and 1 - 1e-3: {partial}");
    assert!(
        partial >= 2,
        "the smearing check is vacuous: {partial} partially occupied bands"
    );
    equation_of_state(&data.manifest, &stress);
    let weak = weak_terms(&stress, &want);
    assert!(
        weak.iter().all(|w| SMALL.contains(w)),
        "terms the oracle cannot discriminate: {weak:?}"
    );
    // Wrong occupations: the same bands re-occupied with a quarter of the width are another
    // state of the free energy, and the stress must say so.
    let (_, state, _) = solve(&mut data, SCF_TOLERANCE, SCF_DENSITY_TOLERANCE);
    let mut options = data.options.clone();
    options.width *= 0.25;
    let narrow = stress_at(&data, &state, &options);
    let bite = bites(&narrow.gradient, &want);
    println!("  control: a quarter of the smearing width bites {bite:.2e} gates");
    assert!(bite > 100., "the occupations do not enter the stress");
}

#[test]
#[ignore = "about 5 minutes: 14 k-points at 22 s each; run with --ignored"]
fn the_stress_of_cubic_aluminum_on_a_full_mesh_is_isotropic() {
    let (data, stress, want) = check("al", true);
    let partial = partial_bands(&data);
    println!("  bands with an occupation strictly between 1e-3 and 1 - 1e-3: {partial}");
    assert!(
        partial >= 3,
        "the smearing check is vacuous: {partial} bands"
    );
    equation_of_state(&data.manifest, &stress);
    // Cubic: sigma = p I, up to the egg-box floor of the grid terms (the 6x6x6 grid and the mesh
    // keep the 48 operations).
    isotropy(&data.manifest, &stress);
    let weak = weak_terms(&stress, &want);
    assert!(
        weak.iter().all(|w| SMALL.contains(w)),
        "terms the oracle cannot discriminate: {weak:?}"
    );
}
