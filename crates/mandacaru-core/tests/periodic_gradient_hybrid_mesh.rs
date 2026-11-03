//! The explicit exact-exchange terms on a k MESH (`k != q` pairs, the Nyquist planes of an even
//! grid) against central differences of the production energy `Crystal::hybrid_exchange` of
//! crystals rebuilt at the moved geometry, with the density matrices of the converged SCF held
//! fixed: the displaced-crystal oracle of `periodic_hybrid_gradient_motion.rs` (Gamma only)
//! extended to the 2x2x2 Si mesh of the frozen HSE06 case. The pieces compared are the ones the
//! engine adds: the phase, the pair functions (bra and ket) and the projection chain. Needs
//! `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;

use mandacaru_core::periodic::{
    hybrid_gradient::{explicit, projection_forces},
    screened_exchange::Screening,
    setup::pulay::{projection_gradient, Mesh},
};
use periodic_gradient_crystals_support::{mesh::*, *};

/// The measured step floors are 1e-13 .. 1e-11 Ha/Bohr; the gate is 100 x the floor of each
/// coordinate, no tighter than 1e-10 (the SCF-free energies are good to about 1e-14 per build).
fn check(coordinates: &[(usize, usize)]) {
    let data = Data::load("si_hse06");
    let result = data.solve();
    let c = &data.crystal;
    let mesh = Mesh::new(&c.context, &c.grid).unwrap();
    let augmentation = c.hybrid_augmentation().unwrap();
    let motion = ShellMotion {
        data: &data,
        mesh: &mesh,
    };
    let gradient = explicit(
        c,
        &result.matrices,
        Screening::of(c).unwrap(),
        data.options.exxdiv_treatment,
        &augmentation,
        Some(&motion),
    )
    .unwrap();
    // The energy the derivative belongs to is the production one.
    let production = c
        .hybrid_exchange(&result.matrices, data.options.exxdiv_treatment)
        .unwrap()
        .energy;
    let tie = gradient.energy + gradient.one_center_energy - production;
    eprintln!("explicit energy - hybrid_exchange energy = {tie:e}");
    assert!(tie.abs() < 1e-12);
    let gradients: Vec<_> = c
        .points
        .iter()
        .map(|p| {
            projection_gradient(
                &c.context,
                &data.shells,
                &data.lattice,
                p.k,
                &data.projectors,
            )
            .unwrap()
        })
        .collect();
    let chain = projection_forces(
        &gradient.response(),
        &gradients,
        &data.owner,
        &data.projectors.iter().map(|p| p.atom).collect::<Vec<_>>(),
        data.atoms.len(),
    )
    .unwrap();
    let functions = gradient.functions.as_ref().unwrap().total();
    for &(atom, axis) in coordinates {
        let analytic = gradient.phase[atom][axis]
            + functions[atom][axis]
            + chain.rows[atom][axis]
            + chain.columns[atom][axis];
        let (slope, floor) = finite_difference(&data, &mesh, &result.matrices, atom, axis);
        eprintln!(
            "atom {atom} axis {axis}: analytic {analytic:+.9e}, displaced crystals {slope:+.9e}, \
             miss {:.2e}, step floor {floor:.1e}; phase {:+.3e}, functions {:+.3e}, rows {:+.3e}, columns {:+.3e}",
            (analytic - slope).abs(),
            gradient.phase[atom][axis],
            functions[atom][axis],
            chain.rows[atom][axis],
            chain.columns[atom][axis]
        );
        let gate = (100. * floor).max(1e-10);
        assert!(
            (analytic - slope).abs() < gate,
            "atom {atom} axis {axis}: gate {gate:e}"
        );
        // Not vacuous: every piece of the exchange force is far above the gate on this coordinate.
        assert!(
            [
                gradient.phase[atom][axis],
                functions[atom][axis],
                chain.rows[atom][axis],
                chain.columns[atom][axis]
            ]
            .iter()
            .all(|t| t.abs() > 100. * gate),
            "a term of atom {atom} axis {axis} is not above 100 gates"
        );
    }
}

/// One coordinate, the one the frozen Si oracle misses most (atom 1, x), in the default suite:
/// every build re-reads the datasets and the screened tables (about 15 s each).
#[test]
fn the_explicit_exchange_terms_on_the_k_mesh_are_the_displaced_crystal_derivative() {
    check(&[(1, 0)]);
}

#[test]
#[ignore = "three more coordinates of 15 s builds, six per coordinate: 383 s for four measured on 2026-11-02"]
fn the_other_coordinates_of_the_mesh_oracle() {
    check(&[(1, 1), (0, 1), (1, 2)]);
}
