//! Analytic PAW-LCAO forces against independently converged total energies.
//!
//! The PAW input is a synthetic hydrogen-like dataset built from smooth analytic
//! radial tables, so the test needs no dataset files (the real datasets are in
//! `paw_forces_dataset`; the per-term gates are in `paw_forces_terms`). The
//! reference is the derivative of reconverged total energies on the same fixed
//! grid, and the retired finite-difference expression (fixed orthonormal-frame
//! densities, Richardson steps) kept as a test oracle.
mod paw_forces_support;

use mandacaru_core::{
    math::storage::Precision,
    molecular::{MeanField, MolecularIntegrals},
    Context, Geometry, LengthUnit,
};
use paw_forces_support::{
    fixed_density_derivative,
    synthetic::{displaced, integrals, integrals_in, options, solve, Method, H2},
};

const HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM: f64 = 27.211386245988 / 0.529177210903;

/// Richardson-extrapolated derivative of independently converged total
/// energies on the same fixed grid, in Hartree per Bohr.
fn energy_derivative(
    base: &MolecularIntegrals,
    method: Method,
    particles: [usize; 2],
    atom: usize,
    axis: usize,
) -> f64 {
    let energy = |delta: f64| {
        let moved = displaced(base, atom, axis, delta);
        solve(&moved, method, particles).free_energy() + moved.nuclear
    };
    let h = 1e-3;
    let coarse = (energy(h) - energy(-h)) / (2. * h);
    let fine = (energy(0.5 * h) - energy(-0.5 * h)) / h;
    (4. * fine - coarse) / 3.
}
/// Largest |F_analytic - F_reference| in eV/Angstrom over the listed
/// components, for the reconverged energies and for the retired fixed-density
/// finite difference; both are printed for the record.
fn check(
    label: &str,
    base: &MolecularIntegrals,
    method: Method,
    particles: [usize; 2],
    components: &[(usize, usize)],
    tolerance_ev: f64,
) -> Vec<f64> {
    let state = solve(base, method, particles);
    let gradient = base.gradient(&state).unwrap();
    assert!(gradient.hellmann_feynman.is_none() && gradient.pulay.is_none());
    assert_eq!(gradient.scheme, "analytic");
    let to_ev = HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM;
    let (mut worst, mut worst_retired) = (0f64, 0f64);
    for &(atom, axis) in components {
        let reference = energy_derivative(base, method, particles, atom, axis);
        let (retired, _) = fixed_density_derivative(base, &state, atom, axis);
        let analytic = gradient.total.as_slice()[3 * atom + axis];
        worst = worst.max((analytic - reference).abs() * to_ev);
        worst_retired = worst_retired.max((analytic - retired).abs() * to_ev);
    }
    println!(
        "{label}: max |F - F_scf| = {worst:.3e} eV/A, max |F - F_retired| = {worst_retired:.3e} eV/A, \
         stationarity {:.2e}, max |F| = {:.4} eV/A",
        gradient.stationarity.unwrap(),
        gradient.total.as_slice().iter().fold(0f64, |a, v| a.max(v.abs())) * to_ev
    );
    assert!(
        worst.is_finite() && worst < tolerance_ev,
        "{label}: {worst:.3e} eV/A"
    );
    assert!(
        worst_retired.is_finite() && worst_retired < tolerance_ev,
        "{label} against the retired expression: {worst_retired:.3e} eV/A"
    );
    gradient.total.as_slice().to_vec()
}

#[test]
#[ignore = "slow: 29 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_forces -- --ignored closed_shell_h2_forces_follow_total_energies_for_hf_lda_and_pbe"]
fn closed_shell_h2_forces_follow_total_energies_for_hf_lda_and_pbe() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        for (functional, method) in [
            ("lda", Method::Hf("rhf")),
            ("lda", Method::Ks("lda")),
            ("pbe", Method::Ks("pbe")),
        ] {
            let base = integrals(&ctx, &H2, functional);
            let g = check(
                &format!(
                    "H2 {}",
                    match method {
                        Method::Hf(_) => "RHF".into(),
                        Method::Ks(n) => n.to_uppercase(),
                    }
                ),
                &base,
                method,
                [1, 1],
                &[(0, 2), (1, 2), (1, 0)],
                1e-5,
            );
            // The midpoint is a grid node, so the grid is mirror symmetric:
            // the bond force is antisymmetric and transverse forces vanish.
            let to_ev = HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM;
            assert!((g[2] + g[5]).abs() * to_ev < 1e-6, "net bond force");
            assert!([g[0], g[1], g[3], g[4]]
                .iter()
                .all(|v| v.abs() * to_ev < 1e-6));
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
#[ignore = "slow: 16 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_forces -- --ignored meta_gga_h2_forces_follow_total_energies"]
fn meta_gga_h2_forces_follow_total_energies() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H2, "pbe");
        check(
            "H2 r2SCAN",
            &base,
            Method::Ks("r2scan"),
            [1, 1],
            &[(0, 2), (1, 0)],
            1e-5,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
#[ignore = "slow: 42 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_forces -- --ignored open_shell_forces_follow_total_energies_for_uhf_and_uks"]
fn open_shell_forces_follow_total_energies_for_uhf_and_uks() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        // Stretched H2 triplet, and a bent H3 doublet that has net transverse
        // and bond-angle forces.
        let triplet = integrals(&ctx, &[[0., 0., -1.1], [0., 0., 1.1]], "lda");
        check(
            "H2 triplet UHF",
            &triplet,
            Method::Hf("uhf"),
            [2, 0],
            &[(0, 2), (1, 0)],
            1e-5,
        );
        check(
            "H2 triplet UKS-LDA",
            &triplet,
            Method::Ks("lda"),
            [2, 0],
            &[(1, 2)],
            1e-5,
        );
        let bent = integrals(
            &ctx,
            &[[0., 0., 0.], [1.7, 0., 0.3], [0.4, 1.6, -0.5]],
            "pbe",
        );
        check(
            "H3 UHF",
            &bent,
            Method::Hf("uhf"),
            [2, 1],
            &[(0, 0), (1, 2), (2, 1)],
            1e-5,
        );
        check(
            "H3 UKS-PBE",
            &bent,
            Method::Ks("pbe"),
            [2, 1],
            &[(1, 0), (2, 2)],
            1e-5,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
fn the_net_force_is_the_egg_box_ripple_the_fixed_grid_energy_has_under_a_rigid_translation() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        // Off the grid nodes, so the grid is not mirror symmetric: the sum of the
        // forces is not zero, it is the slope of the energy of the whole molecule
        // moved rigidly across the fixed grid (the egg-box ripple of this coarse
        // grid), and that slope is what the sum must reproduce.
        let positions = [[0.13, 0., 0.], [1.7, 0.21, 0.3], [0.4, 1.6, -0.5]];
        let bent = integrals(&ctx, &positions, "pbe");
        let state = solve(&bent, Method::Ks("pbe"), [2, 1]);
        let g = bent.gradient(&state).unwrap();
        let total = g.total.as_slice();
        let net: Vec<f64> = (0..3)
            .map(|a| (0..3).map(|i| total[3 * i + a]).sum())
            .collect();
        let largest = total.iter().fold(0f64, |a, v| a.max(v.abs()));
        let shifted = |t: f64| {
            let moved: Vec<[f64; 3]> = positions.iter().map(|p| [p[0] + t, p[1], p[2]]).collect();
            let geometry = Geometry::new(
                bent.geometry.numbers().to_vec(),
                moved,
                *bent.geometry.cell(),
                bent.geometry.pbc(),
                LengthUnit::Bohr,
            )
            .unwrap();
            let x = bent.at_geometry(geometry).unwrap();
            solve(&x, Method::Ks("pbe"), [2, 1]).free_energy() + x.nuclear
        };
        let h = 1e-3;
        let coarse = (shifted(h) - shifted(-h)) / (2. * h);
        let fine = (shifted(0.5 * h) - shifted(-0.5 * h)) / h;
        let ripple = (4. * fine - coarse) / 3.;
        let ev = HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM;
        println!(
            "egg-box ripple along x: net force {:.4e} Ha/Bohr, dE/dt {ripple:.4e} (largest force component {:.3e}), net y {:.3e}, net z {:.3e}  [eV/A: {:.4}]",
            net[0], largest, net[1], net[2], net[0] * ev
        );
        assert!((net[0] - ripple).abs() < 1e-7, "{:.3e}", net[0] - ripple);
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
fn bond_force_does_not_depend_on_the_cartesian_axis_of_the_bond() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        // The angular quadratures of the PAW spheres are not cubic-symmetric,
        // so a rotated molecule differs from the original by the (tiny)
        // quadrature error of the energy itself. The force must inherit
        // exactly that error, and no more: the axis bookkeeping is right when
        // the force difference stays at the size of the energy difference
        // scaled by the bond length, not at the size of the force.
        let along_z = integrals(&ctx, &H2, "pbe");
        let along_x = integrals(&ctx, &[[-0.8, 0., 0.], [0.8, 0., 0.]], "pbe");
        let z_state = solve(&along_z, Method::Ks("pbe"), [1, 1]);
        let x_state = solve(&along_x, Method::Ks("pbe"), [1, 1]);
        let z = along_z.gradient(&z_state).unwrap();
        let x = along_x.gradient(&x_state).unwrap();
        let to_ev = HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM;
        let force = (z.total.as_slice()[5] - x.total.as_slice()[3]).abs() * to_ev;
        let energy = (z_state.electronic_energy - x_state.electronic_energy).abs();
        println!("axis-permuted bond force difference: {force:.3e} eV/A, energy difference {energy:.3e} Ha");
        assert!(force < 1e-4 && energy < 1e-5);
        let (z, x) = (z.total.as_slice(), x.total.as_slice());
        let transverse = [z[0], z[1], z[3], z[4], x[1], x[2], x[4], x[5]]
            .iter()
            .fold(0f64, |a, v| a.max(v.abs()))
            * to_ev;
        assert!(transverse < 1e-6, "{transverse:.3e}");
    }
    assert_eq!(ctx.memory_used(), 0);
}
fn refusal(base: &MolecularIntegrals, state: &MeanField) -> String {
    base.gradient(state).err().unwrap().to_string()
}
#[test]
fn unsupported_states_raise_instead_of_returning_forces() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H2, "lda");
        let spinor = base.mean_field("ghf", [1, 1], &options()).unwrap();
        assert!(refusal(&base, &spinor).contains("spinor"));
        let mut state = solve(&base, Method::Hf("rhf"), [1, 1]);
        state.converged = false;
        assert!(refusal(&base, &state).contains("converged"));
        // rVV10 is differentiated (MG-F6, `paw_forces_dataset.rs`): no refusal to test here.
        // A PBE state relabeled as the screened hybrid is not the stationary
        // state of the hybrid operator: no gradient is returned for it.
        let mut relabeled = solve(&base, Method::Ks("pbe"), [1, 1]);
        relabeled.functional = Some("hse06".into());
        assert!(refusal(&base, &relabeled).contains("not computed with them"));
        // A state from other integrals is not stationary on these ones.
        let other = integrals(&ctx, &[[0., 0., -1.2], [0., 0., 1.2]], "lda");
        let foreign = solve(&other, Method::Hf("rhf"), [1, 1]);
        assert!(refusal(&base, &foreign).contains("not computed with them"));
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
fn a_float32_state_has_an_analytic_force_at_its_own_floor() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let reference = integrals(&ctx, &H2, "pbe");
        let float32 = integrals_in(&ctx, &H2, "pbe", Precision::Float32);
        let g64 = reference
            .gradient(&solve(&reference, Method::Ks("pbe"), [1, 1]))
            .unwrap();
        let state32 = {
            let mut o = options();
            o.tolerance = 1e-6;
            float32.kohn_sham("pbe", [1, 1], false, &o).unwrap()
        };
        assert!(state32.converged);
        let g32 = float32.gradient(&state32).unwrap();
        let to_ev = HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM;
        let worst = g64
            .total
            .as_slice()
            .iter()
            .zip(g32.total.as_slice())
            .fold(0f64, |a, (x, y)| a.max((x - y).abs()))
            * to_ev;
        println!(
            "float32 against float64 force: {worst:.3e} eV/A, float32 stationarity {:.2e}",
            g32.stationarity.unwrap()
        );
        // Measured 5.0e-6 eV/A with a float32 stationarity of 1.0e-7; the gate is 100 times the force.
        assert!(worst < 5e-4, "{worst:.3e} eV/A");
    }
    assert_eq!(ctx.memory_used(), 0);
}
