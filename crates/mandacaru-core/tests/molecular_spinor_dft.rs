//! MG-F2a: spinor (noncollinear) Kohn-Sham for molecules (`molecular::spinor_dft`).
//!
//! - The Fock of a spinor density is the derivative of its energy, `dE = Re tr(F dD)`, in every
//!   Pauli channel, for a noncollinear state (two spinors with different spatial parts, so the
//!   spin direction varies in space).
//! - A block-diagonal spinor density `diag(D_up, D_dn)` has the unrestricted energy.
//! - r2SCAN (F2a part 2) in all of these, its spinor force included; HSE06 (F2c) in the energy,
//!   Fock, SCF and force (its exchange over the four spin blocks, the spheres' vector
//!   polarization).
mod paw_molecular_support;

use mandacaru_core::{molecular::spinor_dft::spinor_response, quantum::C64, Context};
use nalgebra::{DMatrix, DVector};
use paw_molecular_support::{integrals, H2};

type Matrix = DMatrix<C64>;

/// `sum |psi><psi|` of two spinors with different spatial parts plus an unpolarized background:
/// positive definite, so `n > |m|` pointwise, with a spin direction that varies in space.
fn noncollinear_density(m: usize) -> Matrix {
    let mut a = DVector::<C64>::zeros(2 * m);
    a[0] = C64::new(0.8, 0.);
    a[m + 1] = C64::from_polar(0.6, 0.7);
    let mut b = DVector::<C64>::zeros(2 * m);
    b[1] = C64::new(0.5, 0.);
    b[m] = C64::new(0., 0.75);
    b[m + 2.min(m - 1)] += C64::new(0.2, 0.1);
    // An unpolarized background 0.1 I (orthonormal frame): n - |m| >= 0.1 sum_a phi_a^2 everywhere,
    // and a perturbation e dD with e |dD| < 0.1 keeps every density positive, so no clamp of the
    // engine (a negative density is set to zero) enters the finite difference.
    &a * a.adjoint() + &b * b.adjoint() + Matrix::identity(2 * m, 2 * m) * C64::new(0.1, 0.)
}

fn hermitian_direction(m: usize) -> Matrix {
    let raw = Matrix::from_fn(2 * m, 2 * m, |i, j| {
        C64::new(((i * 7 + j * 3) % 11) as f64 / 11. - 0.5, ((i * 5 + j * 13) % 7) as f64 / 7. - 0.5)
    });
    (&raw + raw.adjoint()) * C64::new(0.5, 0.)
}

#[test]
fn the_spinor_fock_is_the_derivative_of_the_energy_in_every_channel() {
    fock_gate(&["lda", "pbe", "pbe+rvv10", "r2scan"]);
}

fn fock_gate(names: &[&str]) {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let base = integrals(&ctx, &H2, "dual");
        let m = base.m();
        let d = noncollinear_density(m);
        let direction = hermitian_direction(m);
        for &name in names {
            let at = spinor_response(&base, &d, name, None).unwrap();
            let energy = |e: f64| spinor_response(&base, &(&d + &direction * C64::new(e, 0.)), name, None).unwrap().energy;
            // 1e-5: PBE depends on grad |m|, whose kink where m crosses zero makes the third
            // derivative large (`pbe_step_scan`: the miss falls as h^2, 3.3e-7 at 1e-4, 3.3e-9 at 1e-5).
            let h = 1e-5;
            let fd = (energy(h) - energy(-h)) / (2. * h);
            let analytic = (&at.fock * &direction).trace().re;
            eprintln!("{name}: dE/de {fd:+.10e}, Re tr(F dD) {analytic:+.10e}, diff {:.2e}", fd - analytic);
            let combined = (fd - analytic).abs() < 1e-7 * (1. + analytic.abs());
            // Each Pauli channel alone: a perturbation of only D_a.
            for a in 0..4 {
                let single = pauli(m, a, &DMatrix::from_fn(m, m, |i, j| direction[(i, j)]));
                let energy = |e: f64| spinor_response(&base, &(&d + &single * C64::new(e, 0.)), name, None).unwrap().energy;
                let fd = (energy(h) - energy(-h)) / (2. * h);
                let analytic = (&at.fock * &single).trace().re;
                eprintln!("  {name} channel {a}: fd {fd:+.10e}, analytic {analytic:+.10e}, diff {:.2e}", fd - analytic);
                assert!((fd - analytic).abs() < 1e-7 * (1. + analytic.abs()), "{name}, channel {a}: {fd} vs {analytic}");
            }
            assert!(combined, "{name}: the combined direction");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// The 2m x 2m Hermitian matrix whose Pauli component `a` is `2 v` and the others zero.
fn pauli(m: usize, a: usize, v: &Matrix) -> Matrix {
    let v = (v + v.adjoint()) * C64::new(0.5, 0.);
    let mut out = Matrix::zeros(2 * m, 2 * m);
    for i in 0..m {
        for j in 0..m {
            let x = v[(i, j)];
            match a {
                0 => {
                    out[(i, j)] += x;
                    out[(m + i, m + j)] += x;
                }
                1 => {
                    out[(i, m + j)] += x;
                    out[(m + i, j)] += x;
                }
                2 => {
                    out[(i, m + j)] -= C64::i() * x;
                    out[(m + i, j)] += C64::i() * x;
                }
                _ => {
                    out[(i, j)] += x;
                    out[(m + i, m + j)] -= x;
                }
            }
        }
    }
    out
}

#[test]
fn a_collinear_spinor_density_has_the_unrestricted_energy() {
    collinear_energy_gate(&["lda", "pbe", "r2scan", "hse06"]);
}

fn collinear_energy_gate(names: &[&str]) {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let base = integrals(&ctx, &H2, "dual");
        let m = base.m();
        let up = Matrix::from_fn(m, m, |i, j| C64::new(if i == j && i < 2 { 0.9 - 0.3 * i as f64 } else if i + j == 1 { 0.1 } else { 0. }, 0.));
        let down = Matrix::from_fn(m, m, |i, j| C64::new(if i == 0 && j == 0 { 0.7 } else { 0. }, 0.));
        let mut d = Matrix::zeros(2 * m, 2 * m);
        d.view_mut((0, 0), (m, m)).copy_from(&up);
        d.view_mut((m, m), (m, m)).copy_from(&down);
        // m_z = 0.2 phi_0^2 + 0.2 phi_0 phi_1 + 0.6 phi_1^2 > 0 everywhere: the local frame is the
        // collinear one at every point, so a GGA or meta-GGA sees exactly the unrestricted fields.
        for &name in names {
            let spinor = spinor_response(&base, &d, name, None).unwrap().energy;
            let unrestricted = base.kohn_sham_energy(name, &[up.clone(), down.clone()]).unwrap();
            eprintln!("{name}: spinor {spinor:+.12e}, UKS {unrestricted:+.12e}, diff {:.2e}", spinor - unrestricted);
            assert!((spinor - unrestricted).abs() < 1e-9, "{name}: {spinor} vs {unrestricted}");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
#[ignore = "diagnostic: the step dependence of the combined PBE difference"]
fn pbe_step_scan() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let base = integrals(&ctx, &H2, "dual");
        let m = base.m();
        let d = noncollinear_density(m);
        let direction = hermitian_direction(m);
        let at = spinor_response(&base, &d, "pbe", None).unwrap();
        let analytic = (&at.fock * &direction).trace().re;
        for h in [1e-4, 3e-5, 1e-5] {
            let energy = |e: f64| spinor_response(&base, &(&d + &direction * C64::new(e, 0.)), "pbe", None).unwrap().energy;
            let fd = (energy(h) - energy(-h)) / (2. * h);
            eprintln!("pbe h {h:.0e}: diff {:+.3e}", fd - analytic);
        }
    }
}

/// Without spin-orbit coupling a spinor Kohn-Sham ground state is a collinear one: closed-shell H2
/// reaches the RKS energy, one electron (H2+) the UKS energy, through the SCF iteration shared
/// with restricted and unrestricted Kohn-Sham.
#[test]
fn the_spinor_scf_without_spin_orbit_reaches_the_collinear_ground_state() {
    scf_gate(&["lda", "pbe", "r2scan"]);
}

fn scf_gate(names: &[&str]) {
    use mandacaru_core::molecular::ScfOptions;
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let base = integrals(&ctx, &H2, "dual");
        let options = ScfOptions { tolerance: 1e-10, max_iterations: 300, ..Default::default() };
        for &name in names {
            let spinor = base.spinor_kohn_sham(name, 2, &options).unwrap();
            let rks = base.kohn_sham(name, [1, 1], false, &options).unwrap();
            assert!(spinor.converged && rks.converged, "{name}");
            eprintln!("{name} H2: spinor {:+.12e}, RKS {:+.12e}, diff {:.2e}, {} iterations",
                spinor.electronic_energy, rks.electronic_energy,
                spinor.electronic_energy - rks.electronic_energy, spinor.iterations.len());
            assert!((spinor.electronic_energy - rks.electronic_energy).abs() < 1e-8, "{name} H2");
            let open = base.spinor_kohn_sham(name, 1, &options).unwrap();
            let uks = base.kohn_sham(name, [1, 0], true, &options).unwrap();
            assert!(open.converged && uks.converged, "{name} H2+");
            eprintln!("{name} H2+: spinor {:+.12e}, UKS {:+.12e}, diff {:.2e}",
                open.electronic_energy, uks.electronic_energy, open.electronic_energy - uks.electronic_energy);
            assert!((open.electronic_energy - uks.electronic_energy).abs() < 1e-8, "{name} H2+");
            assert_eq!((spinor.method.as_str(), spinor.functional.as_deref()), ("ghf", Some(name)));
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// F2d increment 2: the force contraction keeps an orbital's density and W roles apart, so a
/// state whose orbitals are each carried as a density-only and a W-only entry has the same force.
#[test]
fn the_force_is_unchanged_when_density_and_w_roles_are_separate_entries() {
    use mandacaru_core::molecular::ScfOptions;
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let base = integrals(&ctx, &H2, "dual");
        let options = ScfOptions { tolerance: 1e-11, max_iterations: 300, ..Default::default() };
        for (name, unrestricted, particles) in [("lda", false, [1, 1]), ("pbe", true, [1, 0])] {
            let state = base.kohn_sham(name, particles, unrestricted, &options).unwrap();
            let (whole, split) = (base.gradient(&state).unwrap(), base.gradient_with_split_roles(&state).unwrap());
            let gap = whole.total.as_slice().iter().zip(split.total.as_slice()).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max);
            let size = whole.total.as_slice().iter().fold(0f64, |m, v| m.max(v.abs()));
            eprintln!("{name}: max |F - F(split roles)| = {gap:.2e} (force {size:.3e})");
            assert!(size > 1e-4 && gap < 1e-13, "{name}: {gap:e}");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// F2d increment 3b, the collinear limit: without spin-orbit coupling the spinor SCF of H2+ relaxes
/// to the UKS state, and its spinor gradient (charge pseudo-orbitals, the W of the spinor Fock, the
/// v_0 override, the magnetization term) is the UKS gradient.
#[test]
fn a_collinear_spinor_state_has_the_unrestricted_force() {
    use mandacaru_core::molecular::ScfOptions;
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let base = integrals(&ctx, &[[0., 0.1, -0.8], [0.05, 0., 0.8]], "dual");
        let options = ScfOptions { tolerance: 1e-11, max_iterations: 300, ..Default::default() };
        for name in ["lda", "pbe", "r2scan"] {
            let spinor = base.spinor_kohn_sham(name, 1, &options).unwrap();
            let uks = base.kohn_sham(name, [1, 0], true, &options).unwrap();
            let (gs, gu) = (base.spinor_gradient(&spinor).unwrap(), base.gradient(&uks).unwrap());
            let gap = gs.total.as_slice().iter().zip(gu.total.as_slice()).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max);
            let size = gu.total.as_slice().iter().fold(0f64, |m, v| m.max(v.abs()));
            let magnetization = gs.terms.iter().find(|(n, _)| *n == "xc/magnetization_basis").map(|(_, b)| b.as_slice().iter().fold(0f64, |m, v| m.max(v.abs()))).unwrap();
            eprintln!("{name} H2+: max |F(spinor) - F(UKS)| = {gap:.2e} (force {size:.3e}, magnetization term {magnetization:.3e})");
            assert!(size > 1e-3 && gap < 1e-8, "{name}: {gap:e}");
            assert!(magnetization > 1e-4, "{name}: the magnetization term is not exercised");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}


/// F2c part 2: the spinor force of a screened hybrid. Without spin-orbit coupling the spinor
/// functional is invariant under a global spin rotation, so a converged UKS HSE06 state turned to
/// a general axis is a stationary spinor state with the UKS energy and force. The turn fills the
/// spin-flip blocks and the imaginary parts, so every complex pair term of the exchange (grid
/// pairs, compensation moments, one-center tensor, the spheres' vector polarization) is
/// exercised. The two- and three-electron states are fully polarized, so their magnetization has
/// one sign and the local frame is the collinear one everywhere (a GGA's frame differs from UKS
/// where `m` changes sign); they have pairs of distinct spinors.
#[test]
#[ignore = "slow: 25 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test molecular_spinor_dft -- --ignored a_turned_hybrid"]
fn a_turned_hybrid_spinor_state_has_the_unrestricted_force() {
    use mandacaru_core::molecular::ScfOptions;
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let base = integrals(&ctx, &[[0., 0.1, -0.8], [0.05, 0., 0.8]], "dual");
        let m = base.m();
        let options = ScfOptions { tolerance: 1e-11, max_iterations: 300, ..Default::default() };
        let (theta, phi) = (1.1f64, 0.7f64);
        let (c, s) = ((theta / 2.).cos(), (theta / 2.).sin());
        let u = [[C64::new(c, 0.), -C64::from_polar(s, -phi)], [C64::from_polar(s, phi), C64::new(c, 0.)]];
        let mut turn = Matrix::zeros(2 * m, 2 * m);
        for a in 0..2 {
            for b in 0..2 {
                for k in 0..m {
                    turn[(a * m + k, b * m + k)] = u[a][b];
                }
            }
        }
        for electrons in [1usize, 2, 3] {
            let uks = base.kohn_sham("hse06", [electrons, 0], true, &options).unwrap();
            assert!(uks.converged, "UKS HSE06, {electrons} electrons");
            let mut d = Matrix::zeros(2 * m, 2 * m);
            d.view_mut((0, 0), (m, m)).copy_from(&uks.densities[0]);
            d.view_mut((m, m), (m, m)).copy_from(&uks.densities[1]);
            let turned = &turn * d * turn.adjoint();
            let energy = spinor_response(&base, &turned, "hse06", None).unwrap().energy;
            assert!((energy - uks.electronic_energy).abs() < 1e-9, "{electrons} e: {energy} vs {}", uks.electronic_energy);
            let spinor = base.spinor_gradient_of(&turned, "hse06", energy).unwrap();
            let collinear = base.gradient(&uks).unwrap();
            let gap = spinor.total.as_slice().iter().zip(collinear.total.as_slice()).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max);
            let size = collinear.total.as_slice().iter().fold(0f64, |m, v| m.max(v.abs()));
            for (name, values) in spinor.terms.iter().filter(|(n, _)| n.starts_with("hybrid")) {
                let scalar = collinear.terms.iter().find(|(n, _)| n == name).map(|(_, b)| b.as_slice().to_vec());
                eprintln!("  {electrons} e {name}: spinor {:?} UKS {:?}", &values.as_slice()[..3], scalar.as_ref().map(|v| &v[..3]));
            }
            eprintln!("HSE06 {electrons} e, turned: max |F(spinor) - F(UKS)| = {gap:.2e} (force {size:.3e})");
            assert!(size > 1e-3 && gap < 1e-8, "{electrons} electrons: {gap:e}");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
#[ignore = "slow: 14 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test molecular_spinor_dft -- --ignored a_hybrid_spinor_fock"]
fn a_hybrid_spinor_fock_is_the_derivative_of_the_energy_in_every_channel() {
    fock_gate(&["hse06"]);
}

#[test]
#[ignore = "slow: 25 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test molecular_spinor_dft -- --ignored a_hybrid_spinor_scf"]
fn a_hybrid_spinor_scf_without_spin_orbit_reaches_the_collinear_ground_state() {
    scf_gate(&["hse06"]);
}
