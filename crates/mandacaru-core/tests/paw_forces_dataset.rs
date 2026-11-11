//! Analytic PAW-LCAO forces on molecules of the shipped dataset library: water
//! (partial core of oxygen, a p shell with d polarization), lithium hydride (the
//! closed-1s core exchange of lithium) and the OH radical (an open shell).
//!
//! The references are the retired fixed-density Richardson difference
//! (`fixed_density_derivative`), the per-term oracles at fixed atomic-orbital
//! density, and reconverged total energies. The library is
//! `$MANDACARU_PAW_PATH`; a missing one FAILS the test.
mod paw_forces_support;
mod paw_molecular_support;

use mandacaru_core::{
    molecular::{MeanField, MolecularIntegrals, ScfOptions},
    paw::molecular::Arrays,
    Context, Geometry, LengthUnit,
};
use paw_forces_support::{ao_state, by_term, fixed_density_derivative, moved, term_derivatives};
use paw_molecular_support::{analytic_shells, dataset, molecule};

const TO_EV: f64 = 27.211386245988 / 0.529177210903;
const WATER: [[f64; 3]; 3] = [
    [0.11, 0.07, 0.12],
    [1.55, 0.04, -0.93],
    [-1.43, -0.05, -0.97],
];
const OH: [[f64; 3]; 2] = [[0.06, -0.04, 0.10], [0.13, 0.05, 1.88]];
const N2: [[f64; 3]; 2] = [[0.05, -0.03, -1.04], [-0.02, 0.06, 1.09]];
const O2: [[f64; 3]; 2] = [[0.04, -0.05, -1.15], [0.01, 0.03, 1.14]];
const CH4: [[f64; 3]; 5] = [
    [0.03, -0.02, 0.05],
    [1.45, 0.05, 1.12],
    [-1.43, -0.04, 1.14],
    [-0.02, 1.62, -1.05],
    [0.04, -1.58, -1.08],
];
const LIH: [[f64; 3]; 2] = [[0.04, -0.06, -0.08], [0.09, 0.03, 3.07]];

fn build(
    ctx: &Context,
    set: &str,
    elements: &[&str],
    numbers: &[u8],
    positions: &[[f64; 3]],
    shells: &[usize],
) -> MolecularIntegrals {
    build_on(ctx, set, elements, numbers, positions, shells, 0.5)
}
fn build_on(
    ctx: &Context,
    set: &str,
    elements: &[&str],
    numbers: &[u8],
    positions: &[[f64; 3]],
    shells: &[usize],
    spacing: f64,
) -> MolecularIntegrals {
    let mut arrays = Arrays::new();
    let mut species = Vec::new();
    let mut atoms = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for element in elements {
        let index = seen.iter().position(|s| s == element).unwrap_or_else(|| {
            let data = dataset::read(ctx, set, element);
            species.push(dataset::species(ctx, &mut arrays, &data));
            seen.push(element);
            seen.len() - 1
        });
        atoms.push(index);
    }
    molecule(
        ctx,
        numbers,
        positions,
        12.,
        spacing,
        (species, atoms, arrays),
        "raw",
        &analytic_shells(shells),
    )
}
fn options() -> ScfOptions {
    ScfOptions {
        tolerance: 1e-10,
        max_iterations: 400,
        ..Default::default()
    }
}
#[derive(Clone, Copy)]
enum Method {
    Hf(&'static str),
    Ks(&'static str),
}
fn solve(integrals: &MolecularIntegrals, method: Method, particles: [usize; 2]) -> MeanField {
    let state = match method {
        Method::Hf(kind) => integrals.mean_field(kind, particles, &options()).unwrap(),
        Method::Ks(name) => integrals
            .kohn_sham(name, particles, particles[0] != particles[1], &options())
            .unwrap(),
    };
    assert!(state.converged, "SCF did not converge");
    state
}
/// Largest differences (eV/A) from the retired expression and from reconverged
/// total energies over `components`; the reconverged check is made on `scf`.
fn check(
    label: &str,
    base: &MolecularIntegrals,
    method: Method,
    particles: [usize; 2],
    components: &[(usize, usize)],
    scf: &[(usize, usize)],
    tolerance_ev: f64,
) {
    let state = solve(base, method, particles);
    let gradient = base.gradient(&state).unwrap();
    let (mut retired_worst, mut scf_worst) = (0f64, 0f64);
    for &(atom, axis) in components {
        let (retired, _) = fixed_density_derivative(base, &state, atom, axis);
        let analytic = gradient.total.as_slice()[3 * atom + axis];
        retired_worst = retired_worst.max((analytic - retired).abs() * TO_EV);
    }
    for &(atom, axis) in scf {
        let energy = |delta: f64| {
            let x = base.at_geometry(moved(base, atom, axis, delta)).unwrap();
            solve(&x, method, particles).free_energy() + x.nuclear
        };
        let h = 1e-3;
        let coarse = (energy(h) - energy(-h)) / (2. * h);
        let fine = (energy(0.5 * h) - energy(-0.5 * h)) / h;
        let reference = (4. * fine - coarse) / 3.;
        scf_worst =
            scf_worst.max((gradient.total.as_slice()[3 * atom + axis] - reference).abs() * TO_EV);
    }
    let net: Vec<f64> = (0..3)
        .map(|a| {
            (0..base.geometry.numbers().len())
                .map(|i| gradient.total.as_slice()[3 * i + a])
                .sum::<f64>()
                * TO_EV
        })
        .collect();
    println!(
        "{label}: max |F - F_retired| = {retired_worst:.3e} eV/A, max |F - F_scf| = {scf_worst:.3e} eV/A, \
         net force {net:?} eV/A, stationarity {:.2e}, max |F| = {:.3} eV/A",
        gradient.stationarity.unwrap(),
        gradient.total.as_slice().iter().fold(0f64, |a, v| a.max(v.abs())) * TO_EV
    );
    assert!(
        retired_worst.is_finite() && retired_worst < tolerance_ev,
        "{label}: {retired_worst:.3e}"
    );
    assert!(
        scf_worst.is_finite() && scf_worst < tolerance_ev,
        "{label}: {scf_worst:.3e}"
    );
}

#[test]
#[ignore = "slow: 19 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_forces_dataset -- --ignored water_pbe_forces_follow_the_energy"]
fn water_pbe_forces_follow_the_energy() {
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let pbe = build(
            &ctx,
            "pbe-sr",
            &["O", "H", "H"],
            &[8, 1, 1],
            &WATER,
            &[0, 1, 2],
        );
        check(
            "H2O PBE",
            &pbe,
            Method::Ks("pbe"),
            [5, 5],
            &[(0, 2), (1, 0), (2, 1)],
            &[(1, 0)],
            2e-5,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}
#[test]
#[ignore = "slow: 39 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_forces_dataset -- --ignored water_hartree_fock_forces_follow_the_energy"]
fn water_hartree_fock_forces_follow_the_energy() {
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let lda = build(
            &ctx,
            "lda-sr",
            &["O", "H", "H"],
            &[8, 1, 1],
            &WATER,
            &[0, 1, 2],
        );
        check(
            "H2O RHF",
            &lda,
            Method::Hf("rhf"),
            [5, 5],
            &[(0, 2), (1, 0)],
            &[(2, 2)],
            2e-5,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}
#[test]
#[ignore = "measured 428 s on a loaded machine (over the 180 s limit); run: cargo test --release -p mandacaru-core --test paw_forces_dataset -- --ignored --test-threads=1 water_hse06_forces_follow_the_energy_through_the_screened_exchange"]
fn water_hse06_forces_follow_the_energy_through_the_screened_exchange() {
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let hybrid = build(
            &ctx,
            "pbe-sr",
            &["O", "H", "H"],
            &[8, 1, 1],
            &WATER,
            &[0, 1, 2],
        );
        check(
            "H2O HSE06",
            &hybrid,
            Method::Ks("hse06"),
            [5, 5],
            &[(0, 2), (1, 0), (2, 1)],
            &[(1, 0)],
            2e-5,
        );
        let state = solve(&hybrid, Method::Ks("hse06"), [5, 5]);
        let ao = ao_state(&hybrid, &state);
        let terms = by_term(&hybrid.gradient(&state).unwrap());
        let oracle = term_derivatives(&ctx, &hybrid, &state, &ao, 1, 0);
        for (term, (reference, estimate)) in &oracle {
            let analytic = terms[*term][3];
            println!(
                "H2O HSE06 {term:<20} analytic {analytic:+.9} oracle {reference:+.9} diff {:.2e} (oracle step {estimate:.1e})",
                (analytic - reference).abs()
            );
            assert!((analytic - reference).abs() < 1e-8, "{term}");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}
#[test]
#[ignore = "measured 327 s on a loaded machine (over the 180 s limit); run: cargo test --release -p mandacaru-core --test paw_forces_dataset -- --ignored --test-threads=1 the_hydroxyl_radical_hse06_force_has_the_spin_polarization_of_the_spheres"]
fn the_hydroxyl_radical_hse06_force_has_the_spin_polarization_of_the_spheres() {
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let hybrid = build(&ctx, "pbe-sr", &["O", "H"], &[8, 1], &OH, &[0, 1, 2]);
        check(
            "OH UKS-HSE06",
            &hybrid,
            Method::Ks("hse06"),
            [4, 3],
            &[(0, 2), (1, 1)],
            &[(1, 2)],
            2e-5,
        );
        let state = solve(&hybrid, Method::Ks("hse06"), [4, 3]);
        let ao = ao_state(&hybrid, &state);
        let terms = by_term(&hybrid.gradient(&state).unwrap());
        let oracle = term_derivatives(&ctx, &hybrid, &state, &ao, 0, 2);
        assert!(oracle.contains_key("hybrid_polarization"));
        for (term, (reference, estimate)) in &oracle {
            let analytic = terms[*term][2];
            println!(
                "OH UKS-HSE06 {term:<20} analytic {analytic:+.9} oracle {reference:+.9} diff {:.2e} (oracle step {estimate:.1e})",
                (analytic - reference).abs()
            );
            assert!((analytic - reference).abs() < 1e-8, "{term}");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}
#[test]
#[ignore = "measured 206 s on a loaded machine (over the 180 s limit); run: cargo test --release -p mandacaru-core --test paw_forces_dataset -- --ignored --test-threads=1 water_r2scan_forces_include_the_kinetic_density_response_and_the_core_tau"]
fn water_r2scan_forces_include_the_kinetic_density_response_and_the_core_tau() {
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let pbe = build(
            &ctx,
            "pbe-sr",
            &["O", "H", "H"],
            &[8, 1, 1],
            &WATER,
            &[0, 1, 2],
        );
        check(
            "H2O r2SCAN",
            &pbe,
            Method::Ks("r2scan"),
            [5, 5],
            &[(0, 2), (1, 0), (2, 2)],
            &[(1, 0)],
            2e-5,
        );
        let state = solve(&pbe, Method::Ks("r2scan"), [5, 5]);
        let ao = ao_state(&pbe, &state);
        let terms = by_term(&pbe.gradient(&state).unwrap());
        let oracle = term_derivatives(&ctx, &pbe, &state, &ao, 0, 2);
        for (term, (reference, _)) in &oracle {
            let analytic = terms[*term][2];
            println!(
                "H2O r2SCAN {term:<20} analytic {analytic:+.9} oracle {reference:+.9} diff {:.2e}",
                (analytic - reference).abs()
            );
            assert!((analytic - reference).abs() < 5e-7, "{term}");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}
#[test]
#[ignore = "slow: 25 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_forces_dataset -- --ignored lithium_hydride_forces_include_the_closed_core_exchange"]
fn lithium_hydride_forces_include_the_closed_core_exchange() {
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let lda = build(&ctx, "lda-sr", &["Li", "H"], &[3, 1], &LIH, &[0, 1]);
        check(
            "LiH RHF",
            &lda,
            Method::Hf("rhf"),
            [2, 2],
            &[(0, 2), (1, 0), (1, 2)],
            &[(1, 2)],
            2e-5,
        );
        let pbe = build(&ctx, "pbe-sr", &["Li", "H"], &[3, 1], &LIH, &[0, 1]);
        check(
            "LiH PBE",
            &pbe,
            Method::Ks("pbe"),
            [2, 2],
            &[(0, 2), (1, 0), (1, 2)],
            &[(1, 2)],
            2e-5,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}
#[test]
#[ignore = "slow: 69 s measured 2026-11-08 (a meta-GGA open shell, SCF reconvergence per displaced coordinate); run: cargo test --profile release-test -p mandacaru-core --test paw_forces_dataset -- --ignored --test-threads=1 the_hydroxyl_radical_r2scan_force_follows_the_spin_resolved_kinetic_density"]
fn the_hydroxyl_radical_r2scan_force_follows_the_spin_resolved_kinetic_density() {
    // MG-F7: the open-shell meta-GGA force. Each spin channel has its own kinetic density tau_s,
    // so the response -1/2 D(tau_pot,s D psi_s) runs per spin. The gates are those of OH UKS-PBE.
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let open = build(&ctx, "pbe-sr", &["O", "H"], &[8, 1], &OH, &[0, 1, 2]);
        check(
            "OH UKS-r2SCAN",
            &open,
            Method::Ks("r2scan"),
            [4, 3],
            &[(0, 2), (1, 0), (1, 1)],
            &[(1, 2)],
            2e-5,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}
#[test]
#[ignore = "slow: rVV10 molecules, SCF reconvergence per displaced coordinate; run: cargo test --profile release-test -p mandacaru-core --test paw_forces_dataset -- --ignored --test-threads=1 rvv10_forces_follow_the_energy_closed_and_open_shell"]
fn rvv10_forces_follow_the_energy_closed_and_open_shell() {
    // MG-F6: the nonlocal correlation adds no force term of its own; its partials join v_xc.
    // Closed-shell LiH (a partial core: the core term carries rVV10 too) for PBE+rVV10 and
    // r2SCAN+rVV10, and open-shell OH (UKS: rVV10 sees the total density) for PBE+rVV10.
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let lih = build(&ctx, "pbe-sr", &["Li", "H"], &[3, 1], &LIH, &[0, 1]);
        for name in ["pbe+rvv10", "r2scan+rvv10"] {
            check(
                &format!("LiH {name}"),
                &lih,
                Method::Ks(name),
                [2, 2],
                &[(0, 2), (1, 0)],
                &[(1, 2)],
                2e-5,
            );
        }
        let oh = build(&ctx, "pbe-sr", &["O", "H"], &[8, 1], &OH, &[0, 1, 2]);
        check(
            "OH UKS-PBE+rVV10",
            &oh,
            Method::Ks("pbe+rvv10"),
            [4, 3],
            &[(0, 2), (1, 0), (1, 1)],
            &[(1, 2)],
            2e-5,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}
#[test]
#[ignore = "slow: 45 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_forces_dataset -- --ignored the_hydroxyl_radical_has_an_open_shell_force"]
fn the_hydroxyl_radical_has_an_open_shell_force() {
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let pbe = build(&ctx, "pbe-sr", &["O", "H"], &[8, 1], &OH, &[0, 1, 2]);
        check(
            "OH UKS-PBE",
            &pbe,
            Method::Ks("pbe"),
            [4, 3],
            &[(0, 2), (1, 0), (1, 1)],
            &[(1, 2)],
            2e-5,
        );
        let lda = build(&ctx, "lda-sr", &["O", "H"], &[8, 1], &OH, &[0, 1, 2]);
        check(
            "OH UHF",
            &lda,
            Method::Hf("uhf"),
            [4, 3],
            &[(0, 2), (1, 1)],
            &[(1, 2)],
            2e-5,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}
#[test]
#[ignore = "slow: 32 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_forces_dataset -- --ignored every_term_on_water_agrees_with_its_oracle"]
fn every_term_on_water_agrees_with_its_oracle() {
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        for (set, method, label) in [
            ("pbe-sr", Method::Ks("pbe"), "H2O PBE"),
            ("lda-sr", Method::Hf("rhf"), "H2O RHF"),
        ] {
            let base = build(&ctx, set, &["O", "H", "H"], &[8, 1, 1], &WATER, &[0, 1, 2]);
            let state = solve(&base, method, [5, 5]);
            let ao = ao_state(&base, &state);
            let terms = by_term(&base.gradient(&state).unwrap());
            for (atom, axis) in [(0, 2), (1, 0)] {
                let oracle = term_derivatives(&ctx, &base, &state, &ao, atom, axis);
                for (term, (reference, estimate)) in &oracle {
                    let analytic = terms[*term][3 * atom + axis];
                    println!(
                        "{label} atom {atom} axis {axis} {term:<20} analytic {analytic:+.9} oracle {reference:+.9} diff {:.2e} (oracle step {estimate:.1e})",
                        (analytic - reference).abs()
                    );
                    assert!((analytic - reference).abs() < 5e-7, "{label} {term}");
                }
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}
#[test]
#[ignore = "slow: 34 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_forces_dataset -- --ignored nitrogen_and_triplet_oxygen_follow_the_energy"]
fn nitrogen_and_triplet_oxygen_follow_the_energy() {
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let pbe = build(&ctx, "pbe-sr", &["N", "N"], &[7, 7], &N2, &[0, 1, 2]);
        check(
            "N2 PBE",
            &pbe,
            Method::Ks("pbe"),
            [7, 7],
            &[(0, 2), (1, 0), (1, 2)],
            &[(1, 2)],
            2e-5,
        );
        let triplet = build(&ctx, "pbe-sr", &["O", "O"], &[8, 8], &O2, &[0, 1, 2]);
        check(
            "O2 triplet UKS-PBE",
            &triplet,
            Method::Ks("pbe"),
            [7, 5],
            &[(0, 2), (1, 0)],
            &[(1, 2)],
            2e-5,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}
#[test]
#[ignore = "slow: 26 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_forces_dataset -- --ignored methane_follows_the_energy"]
fn methane_follows_the_energy() {
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let methane = build(
            &ctx,
            "pbe-sr",
            &["C", "H", "H", "H", "H"],
            &[6, 1, 1, 1, 1],
            &CH4,
            &[0, 1],
        );
        check(
            "CH4 PBE",
            &methane,
            Method::Ks("pbe"),
            [5, 5],
            &[(0, 0), (1, 2), (3, 1)],
            &[(1, 0)],
            2e-5,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}
/// The net force and the torque of the forces `f` (Ha/Bohr) about the origin, eV/A and eV/A*Bohr.
fn net_and_torque(positions: &[[f64; 3]], f: &[f64]) -> ([f64; 3], [f64; 3]) {
    let mut net = [0.; 3];
    let mut torque = [0.; 3];
    for (i, x) in positions.iter().enumerate() {
        let y = &f[3 * i..3 * i + 3];
        for a in 0..3 {
            net[a] += y[a] * TO_EV;
        }
        torque[0] += (x[1] * y[2] - x[2] * y[1]) * TO_EV;
        torque[1] += (x[2] * y[0] - x[0] * y[2]) * TO_EV;
        torque[2] += (x[0] * y[1] - x[1] * y[0]) * TO_EV;
    }
    (net, torque)
}
#[test]
#[ignore = "slow: 41 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_forces_dataset -- --ignored the_net_force_of_water_is_the_egg_box_ripple_and_falls_with_the_grid_spacing"]
fn the_net_force_of_water_is_the_egg_box_ripple_and_falls_with_the_grid_spacing() {
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let mut ripples = Vec::new();
        for spacing in [0.5, 0.38] {
            let pbe = build_on(
                &ctx,
                "pbe-sr",
                &["O", "H", "H"],
                &[8, 1, 1],
                &WATER,
                &[0, 1, 2],
                spacing,
            );
            let state = solve(&pbe, Method::Ks("pbe"), [5, 5]);
            let g = pbe.gradient(&state).unwrap();
            let (net, torque) = net_and_torque(pbe.geometry.positions(), g.total.as_slice());
            let largest = g.total.as_slice().iter().fold(0f64, |a, v| a.max(v.abs())) * TO_EV;
            // Where the net force comes from: the sum over the atoms of every term.
            for (term, values) in by_term(&g) {
                let sum: Vec<f64> = (0..3)
                    .map(|a| (0..3).map(|i| values[3 * i + a]).sum::<f64>() * TO_EV)
                    .collect();
                println!("  spacing {spacing}: net {term:<20} {sum:.4?} eV/A");
            }
            // The energy of the molecule moved rigidly along z across the same grid.
            let shifted = |t: f64| {
                let moved: Vec<[f64; 3]> = pbe
                    .geometry
                    .positions()
                    .iter()
                    .map(|p| [p[0], p[1], p[2] + t])
                    .collect();
                let geometry = Geometry::new(
                    pbe.geometry.numbers().to_vec(),
                    moved,
                    *pbe.geometry.cell(),
                    pbe.geometry.pbc(),
                    LengthUnit::Bohr,
                )
                .unwrap();
                let x = pbe.at_geometry(geometry).unwrap();
                solve(&x, Method::Ks("pbe"), [5, 5]).free_energy() + x.nuclear
            };
            let h = 1e-3;
            let coarse = (shifted(h) - shifted(-h)) / (2. * h);
            let fine = (shifted(0.5 * h) - shifted(-0.5 * h)) / h;
            let slope = (4. * fine - coarse) / 3. * TO_EV;
            println!(
                "H2O PBE spacing {spacing}: net force {net:.3?} eV/A, dE/dz of the rigid shift {slope:.6} eV/A, torque {torque:.3?} eV/A*Bohr, largest force {largest:.3} eV/A"
            );
            assert!((net[2] - slope).abs() < 1e-5, "{:.3e}", net[2] - slope);
            ripples.push(net.iter().fold(0f64, |a, v| a.max(v.abs())));
        }
        assert!(ripples[1] < ripples[0], "{ripples:?}");
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}
#[test]
#[ignore = "slow: 81 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_forces_dataset -- --ignored lithium_hydride_hse06_force_follows_the_energy_with_a_closed_core"]
fn lithium_hydride_hse06_force_follows_the_energy_with_a_closed_core() {
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let lih = build(&ctx, "pbe-sr", &["Li", "H"], &[3, 1], &LIH, &[0, 1]);
        check(
            "LiH HSE06",
            &lih,
            Method::Ks("hse06"),
            [2, 2],
            &[(0, 2), (1, 0)],
            &[(1, 2)],
            2e-5,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}

/// Multi-shell frozen cores (aluminum's 10-electron 1s2s2p, gold's 60), `pbe-sr`, PBE: every
/// named term of the force against its own fixed-density oracle, and the total against the
/// fixed-density difference. Measured 2026-11-08: AlH and Au-Be (UKS) agree to <= 3e-9 term by
/// term. A reconverged finite difference of Au-Be had disagreed by 0.54 Ha/Bohr because the
/// displaced open-shell SCFs land on another branch, not because the force is wrong.
#[test]
#[ignore = "slow: 36 s (AlH 10 s, Au-Be 26 s; measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test paw_forces_dataset -- --ignored --nocapture multi_shell_core_terms"]
fn multi_shell_core_terms() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    for case in ["AlH", "AuBe"] {
        let (elements, numbers, particles): ([&str; 2], [u8; 2], [usize; 2]) = if case == "AuBe" {
            (["Au", "Be"], [79, 4], [11, 10])
        } else {
            (["Al", "H"], [13, 1], [2, 2])
        };
        let mut arrays = Arrays::new();
        let mut species = Vec::new();
        let mut valence = 0.;
        for element in elements {
            let data = dataset::read(&ctx, "pbe-sr", element);
            valence += data.valence_charge;
            let mut value = dataset::species(&ctx, &mut arrays, &data);
            value["allow_heavy_dft"] = serde_json::json!(true);

            species.push(value);
        }
        const ALH: [[f64; 3]; 2] = [[0.05, -0.04, 0.07], [0.11, 0.06, 3.10]];
        const AUBE: [[f64; 3]; 2] = [[0.07, -0.04, 0.05], [0.12, 0.09, 4.35]];
        // Two exponents per l: the one-exponent analytic shells leave the AlH SCF unconverged.
        let radii: Vec<f64> = (0..=350).map(|k| 0.02 * k as f64).collect();
        let shells: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..=2usize)
            .flat_map(|l| [1.1, 0.45].map(|e: f64| (l, e)))
            .map(|(l, e)| {
                let values = radii
                    .iter()
                    .map(|&r| r.powi(l as i32) * (-e * r).exp() * if r >= 7. { 0. } else { (1. - (r / 7.).powi(2)).powi(3) })
                    .collect();
                (l, radii.clone(), values)
            })
            .collect();
        let positions = if case == "AuBe" { AUBE } else { ALH };
        let base = molecule(&ctx, &numbers, &positions, 14., 0.5, (species, vec![0, 1], arrays), "raw", &shells);
        let electrons = valence.round() as usize;
        let particles = if case == "AuBe" { [electrons.div_ceil(2), electrons / 2] } else { particles };
        let state = solve(&base, Method::Ks("pbe"), particles);
        let gradient = base.gradient(&state).unwrap();
        let terms = by_term(&gradient);
        let ao = ao_state(&base, &state);
        for (atom, axis) in [(1, 2), (0, 0)] {
            let oracle = term_derivatives(&ctx, &base, &state, &ao, atom, axis);
            let (fd, _) = fixed_density_derivative(&base, &state, atom, axis);
            let total = gradient.total.as_slice()[3 * atom + axis];
            eprintln!("{case} atom {atom} axis {axis}: total analytic {total:+.8e}, fixed-density FD {fd:+.8e}");
            assert!((total - fd).abs() < 1e-7, "{case} atom {atom} axis {axis}: {total:e} vs {fd:e}");
            let mut names: Vec<_> = terms.keys().collect();
            names.sort();
            for name in names {
                let analytic = terms[name][3 * atom + axis];
                match oracle.get(name.as_str()) {
                    Some((reference, _)) => {
                        eprintln!("  {name:<34} analytic {analytic:+.8e} oracle {reference:+.8e} diff {:+.2e}", analytic - reference);
                        assert!((analytic - reference).abs() < 1e-7, "{case} {name}: {analytic:e} vs {reference:e}");
                    }
                    None => eprintln!("  {name:<34} analytic {analytic:+.8e} (no oracle)"),
                }
            }
        }
    }
}

