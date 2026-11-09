//! The SCF of the time-reversal-symmetric spinor hybrid on complex-k meshes (SH-1a, gates g4 and
//! g6, plan g-S1..g-S4): the zero-spin-orbit limit equals the scalar restricted hybrid on the
//! same whole mesh, Kramers pairs and `E(k) = E(-k)`, the bands rebuilt from the converged
//! density, a global spin rotation of the spin-orbit block, and the Hellmann-Feynman identity
//! `dF/dlambda = E_SO / lambda` of the hybrid free energy. Needs `MANDACARU_PAW_PATH`.
mod periodic_hybrid_spinor_support;
mod setup_support;

use mandacaru_core::periodic::{
    solver::diagonalize_with_exchange, Crystal, Options, PeriodicResult,
};
use num_complex::Complex64 as C64;
use periodic_hybrid_spinor_support::*;

const MESHES: [&str; 3] = ["si_so.json", "si_so_tr3.json", "si_so_tr4.json"];

fn options() -> Options {
    Options {
        tolerance: 1e-11,
        density_tolerance: 1e-9,
        max_iterations: 300,
        ..Options::default()
    }
}

fn solve(name: &str, edit: &dyn Fn(&mut serde_json::Value)) -> (Crystal, PeriodicResult) {
    let c = hybrid(&context(), name, edit);
    let r = c.solve(&options()).unwrap();
    assert!(r.converged, "{name}: not converged");
    (c, r)
}

/// The largest distance between the SCF eigenvalues and the bands of the converged density
/// matrices with the exchange rebuilt from them (Hartree).
fn band_mismatch(c: &Crystal, r: &PeriodicResult) -> f64 {
    let n = c.grid.index.points();
    let field = c.field_channels() * n;
    let mut worst = 0f64;
    for (k, point) in c.points.iter().enumerate() {
        let exchange = c
            .hybrid_exchange_at_k(&r.matrices, point, 0, options().exxdiv_treatment)
            .unwrap();
        let (bands, _, _) = diagonalize_with_exchange(
            c,
            point,
            &r.potential.as_slice()[..field],
            r.moment_potential.as_slice(),
            None,
            None,
            Some(&exchange),
        )
        .unwrap();
        for (band, scf) in bands.iter().zip(&r.eigenvalues[k]) {
            worst = worst.max((band + c.reference - scf).abs());
        }
    }
    worst
}

#[test]
#[ignore = "slow: 25 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_hybrid_spinor_scf -- --ignored without_spin_orbit_coupling_the_spinor_hybrid_is_the_scalar_hybrid_on_the_same_mesh"]
fn without_spin_orbit_coupling_the_spinor_hybrid_is_the_scalar_hybrid_on_the_same_mesh() {
    for name in MESHES {
        let zero = |v: &mut serde_json::Value| map_spin_orbit(v, &|h| h * C64::new(0., 0.));
        let (c, spinor) = solve(name, &zero);
        let (_, scalar) = solve(name, &|v| {
            scalar_twin(v);
        });
        let mut eig = 0f64;
        for (a, b) in spinor.eigenvalues.iter().zip(&scalar.eigenvalues) {
            for (i, e) in b.iter().enumerate() {
                eig = eig.max((a[2 * i] - e).abs()).max((a[2 * i + 1] - e).abs());
            }
        }
        let energy = (spinor.energy - scalar.energy).abs();
        println!(
            "{name} ({} points): spinor {:.12} scalar {:.12}, |dE| {energy:.2e}, bands {eig:.2e}; {} and {} iterations",
            c.points.len(), spinor.energy, scalar.energy, spinor.history.len(), scalar.history.len()
        );
        assert!(
            energy < 1e-9 && eig < 1e-9,
            "{name}: energy {energy:e}, bands {eig:e}"
        );
        assert!(
            scalar.energy.abs() > 1.
                && (scalar.eigenvalues[0][0] - scalar.eigenvalues[0][7]).abs() > 0.1
        );
    }
}

#[test]
#[ignore = "slow: 15 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_hybrid_spinor_scf -- --ignored the_bands_are_kramers_pairs_even_in_k_and_rebuilt_from_the_density"]
fn the_bands_are_kramers_pairs_even_in_k_and_rebuilt_from_the_density() {
    for name in MESHES {
        let (c, r) = solve(name, &|_| {});
        let (mut kramers, mut even, mut pairs) = (0f64, 0f64, 0);
        for k in 0..c.points.len() {
            let e = &r.eigenvalues[k];
            let j = partner(&c, k);
            for i in 0..e.len() / 2 {
                if j == k {
                    kramers = kramers.max((e[2 * i] - e[2 * i + 1]).abs());
                }
            }
            if j != k {
                pairs += 1;
                even = even.max(
                    e.iter()
                        .zip(&r.eigenvalues[j])
                        .map(|(a, b)| (a - b).abs())
                        .fold(0., f64::max),
                );
            }
        }
        let mismatch = band_mismatch(&c, &r);
        let last = r.history.last().unwrap().exchange_residual.unwrap();
        println!("{name}: Kramers at the time-reversal-invariant points {kramers:.2e}, E(k) - E(-k) {even:.2e} ({pairs} complex points), bands vs density {mismatch:.2e}, K residual {last:.2e}");
        assert!(kramers < 1e-8, "{name}: Kramers pairs split by {kramers:e}");
        assert!(even < 1e-8, "{name}: E(k) != E(-k) by {even:e}");
        assert!(mismatch <= last + 1e-12, "{name}: {mismatch:e} vs {last:e}");
        // Spin-orbit coupling is present and visible (the gates cannot pass on a scalar run).
        assert!(r.spin_orbit_energy.unwrap() < -1e-8);
    }
}

#[test]
#[ignore = "slow: 13 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_hybrid_spinor_scf -- --ignored a_global_spin_rotation_of_the_spin_orbit_block_leaves_the_hybrid_energy_unchanged"]
fn a_global_spin_rotation_of_the_spin_orbit_block_leaves_the_hybrid_energy_unchanged() {
    let name = "si_so_tr3.json";
    let (_, base) = solve(name, &|_| {});
    let r = spin_rotation(&su2(0.8), 8);
    let (_, rotated) = solve(name, &|v| map_spin_orbit(v, &|h| &r * h * r.adjoint()));
    let (_, doubled) = solve(name, &|v| map_spin_orbit(v, &|h| h * C64::new(2., 0.)));
    let change = (rotated.energy - base.energy).abs();
    let bands = base
        .eigenvalues
        .iter()
        .zip(&rotated.eigenvalues)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
        .fold(0., f64::max);
    let control = (doubled.energy - base.energy).abs();
    println!("spin rotation: dE {change:.2e}, bands {bands:.2e}; the control (spin-orbit block doubled) moves E by {control:.2e}");
    assert!(change < 1e-9 && bands < 1e-8, "{change:e} {bands:e}");
    assert!(
        control > 100. * 1e-9,
        "the control does not bite ({control:e})"
    );
}

#[test]
#[ignore = "slow: 26 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_hybrid_spinor_scf -- --ignored the_hybrid_free_energy_is_stationary_in_the_spin_orbit_strength"]
fn the_hybrid_free_energy_is_stationary_in_the_spin_orbit_strength() {
    // dF/dlambda = tr(D h_SO) = E_SO / lambda at self-consistency: a stationarity test of the whole
    // hybrid energy in D, with the compensation and one-center terms.
    let name = "si_so_tr3.json";
    let tight = Options {
        tolerance: 1e-13,
        density_tolerance: 1e-11,
        ..options()
    };
    let free = |lambda: f64| {
        let c = hybrid(&context(), name, &|v| {
            map_spin_orbit(v, &|h| h * C64::new(lambda, 0.))
        });
        let r = c.solve(&tight).unwrap();
        assert!(r.converged);
        (r.free_energy, r.spin_orbit_energy.unwrap())
    };
    let (_, e_so) = free(1.0);
    let central = |delta: f64| (free(1. + delta).0 - free(1. - delta).0) / (2. * delta);
    let (coarse, fine) = (central(0.2), central(0.1));
    let richardson = (4. * fine - coarse) / 3.;
    let miss = (richardson - e_so).abs();
    let floor = (fine - coarse).abs() / 3.;
    println!("dF/dlambda: central {coarse:.6e} (0.2), {fine:.6e} (0.1), Richardson {richardson:.6e}, E_SO {e_so:.6e}; miss {miss:.2e}, Richardson step {floor:.2e}");
    assert!(e_so.abs() > 1e-8);
    assert!(miss < 100. * floor.max(1e-12), "{miss:e} vs {floor:e}");
    // The wrong slope (twice E_SO, the second-order energy's lambda-derivative mistaken) is far.
    assert!((2. * e_so - richardson).abs() > 100. * floor.max(1e-12));
}
