//! The DFT+U term of the molecular PAW-LCAO gradient against independently converged total
//! energies, and the refusal of a +U request the molecule cannot carry.
//!
//! The species is the synthetic hydrogen-like one of `paw_forces_support::synthetic` with a `d`
//! channel added (a test species: the shell is the projection on its `d` partial wave). The
//! reference is the Richardson derivative of reconverged total energies on the same fixed grid,
//! which includes the +U energy and the response of the orbitals to it. The synthetic `d` shell is
//! nearly empty (n_d ~ 1e-6), so this gate covers the projection and Pulay structure of the term
//! (the spin channel, the `m` mapping, the weights, the empty beta channel) in the low-occupation
//! limit, where `V = dE/dn` is nearly the constant `(U - J)/2`. U is 400 eV so that the +U term is
//! 1e-3 to 4e-3 eV/A, a thousand times the 1e-6 eV/A gate. The occupation dependence of `V`
//! (CuH, n_d = 4.9, spin-polarized) is gated in `test/algorithms/test_hubbard.py`.
mod paw_forces_support;

use mandacaru_core::{math::storage::Precision, Context};
use paw_forces_support::synthetic::{
    energy_derivative, integrals_with, solve, try_integrals_with, try_integrals_with_metadata,
    Method, H2,
};

const HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM: f64 = 27.211386245988 / 0.529177210903;

fn dudarev() -> String {
    r#"{"H": {"l": "d", "U": 400.0, "J": 0.0, "form": "dudarev"}}"#.into()
}
fn liechtenstein() -> String {
    r#"{"H": {"l": "d", "U": 400.0, "J": 60.0, "form": "liechtenstein"}}"#.into()
}

fn refusal(request: &str, lmax: usize) -> String {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    match try_integrals_with(&ctx, &H2, "lda", Precision::Float64, lmax, Some(request)) {
        Ok(_) => panic!("the build must refuse {request}"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn a_request_the_molecule_cannot_carry_raises_by_name_and_never_runs_as_plain_dft() {
    // An element that is not in the molecule.
    let ni = r#"{"Ni": {"l": "d", "U": 4.0}}"#;
    assert!(refusal(ni, 2).contains("hubbard_u names Ni, which is not in the system"));
    // One element present and one absent: the request is not partly applied.
    let both = r#"{"H": {"l": "d", "U": 4.0}, "Ni": {"l": "d", "U": 4.0}}"#;
    assert!(refusal(both, 2).contains("hubbard_u names Ni, which is not in the system"));
    // A shell the dataset has no channel for.
    let f = r#"{"H": {"l": "f", "U": 4.0}}"#;
    assert!(refusal(f, 2).contains("does not have channel l = 3"));
    // The same dataset without the d channel.
    assert!(refusal(&dudarev(), 1).contains("does not have channel l = 2"));
}

#[test]
fn a_hubbard_metadata_value_that_is_not_the_request_text_raises_and_null_means_none() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    for value in [
        serde_json::json!({"H": {"l": "d", "U": 4.0}}),
        serde_json::json!(true),
        serde_json::json!(4),
    ] {
        let built =
            try_integrals_with_metadata(&ctx, &H2, "lda", Precision::Float64, 2, Some(value));
        let error = built
            .err()
            .expect("a value that is not text must raise")
            .to_string();
        assert!(
            error.contains("'hubbard' metadata must be the JSON text"),
            "{error}"
        );
    }
    let null = try_integrals_with_metadata(
        &ctx,
        &H2,
        "lda",
        Precision::Float64,
        2,
        Some(serde_json::Value::Null),
    );
    assert!(null
        .unwrap()
        .corrections
        .as_ref()
        .unwrap()
        .hubbard
        .is_none());
}

#[test]
fn the_hartree_fock_gradient_of_a_molecule_with_hubbard_u_raises() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let base = integrals_with(&ctx, &H2, "lda", Precision::Float64, 2, Some(&dudarev()));
    let state = solve(&base, Method::Hf("rhf"), [1, 1]);
    let error = base
        .gradient(&state)
        .err()
        .expect("no +U force for an energy without +U")
        .to_string();
    assert!(
        error.contains("Hartree-Fock gradient of a molecule with hubbard_u"),
        "{error}"
    );
}

#[test]
fn a_request_the_molecule_carries_builds_its_shells() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let base = integrals_with(&ctx, &H2, "lda", Precision::Float64, 2, Some(&dudarev()));
    let hubbard = base.corrections.as_ref().unwrap().hubbard.as_ref().unwrap();
    assert_eq!(hubbard.shells.len(), 2);
    assert!(hubbard.shells.iter().all(|s| s.l == 2 && s.symbol == "H"));
}

/// Largest |F_analytic - F_reference| in eV/Angstrom over `components`, and the size of the +U
/// term itself (so that the gate is not passed by a term that is zero).
fn gate(
    label: &str,
    request: &str,
    method: Method,
    particles: [usize; 2],
    components: &[(usize, usize)],
) -> (f64, f64) {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let base = integrals_with(&ctx, &H2, "lda", Precision::Float64, 2, Some(request));
    let state = solve(&base, method, particles);
    let gradient = base.gradient(&state).unwrap();
    let to_ev = HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM;
    let term = gradient
        .terms
        .iter()
        .find(|(name, _)| *name == "hubbard/projection")
        .expect("a +U molecule has a hubbard/projection term");
    let (mut worst, mut size) = (0f64, 0f64);
    for &(atom, axis) in components {
        let reference = energy_derivative(&base, method, particles, atom, axis);
        let analytic = gradient.total.as_slice()[3 * atom + axis];
        worst = worst.max((analytic - reference).abs() * to_ev);
        size = size.max(term.1.as_slice()[3 * atom + axis].abs() * to_ev);
    }
    println!("{label}: max |F - F_scf| = {worst:.3e} eV/A, +U term {size:.3e} eV/A");
    (worst, size)
}

#[test]
fn direct_rust_molecular_dft_refuses_hubbard_u_with_screened_hybrid() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let base = integrals_with(&ctx, &H2, "lda", Precision::Float64, 2, Some(&dudarev()));
    let opts = Default::default();
    for msg in [
        base.kohn_sham("hse06", [1, 1], false, &opts)
            .err()
            .expect("must refuse")
            .to_string(),
        base.spinor_kohn_sham("hse06", 2, &opts)
            .err()
            .expect("must refuse")
            .to_string(),
        base.kohn_sham_energy("hse06", &[nalgebra::DMatrix::zeros(base.m(), base.m())])
            .unwrap_err()
            .to_string(),
    ] {
        assert!(msg.contains("hubbard_u") && msg.contains("hse06"), "{msg}");
    }
}

#[test]
fn spinor_noncollinear_magnetized_hubbard_force_matches_collinear_and_fd() {
    use mandacaru_core::{molecular::ScfOptions, Geometry, LengthUnit};
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let req = r#"{"H": {"l": "d", "U": 400.0, "J": 60.0, "form": "liechtenstein", "f4_f2": 0.7}}"#;
    let base = integrals_with(&ctx, &H2, "lda", Precision::Float64, 2, Some(req));
    let opts = ScfOptions {
        tolerance: 1e-10,
        max_iterations: 300,
        ..Default::default()
    };
    let uks = base.kohn_sham("lda", [7, 6], true, &opts).unwrap();
    assert!(uks.converged);
    let uks_grad = base.gradient(&uks).unwrap();

    // Tilted spin magnetization mixes all four spin blocks (uu, ud, du, dd).
    let moment = [0.3, -0.2, 0.33];
    let spinor = base
        .spinor_kohn_sham_noncollinear("lda", 0, &[moment, moment], Some(13), &opts)
        .unwrap();
    assert!(spinor.converged);
    assert!(
        (spinor.electronic_energy - uks.electronic_energy).abs() < 1e-8,
        "spinor vs UKS energy: {} vs {}",
        spinor.electronic_energy,
        uks.electronic_energy
    );
    let u_spinor = spinor.hubbard_energy.unwrap();
    let u_uks = uks.hubbard_energy.unwrap();
    assert!(
        (u_spinor - u_uks).abs() < 1e-8,
        "spinor vs UKS Hubbard energy: {u_spinor} vs {u_uks}"
    );
    assert!(
        u_spinor.abs() > 1e-5,
        "spinor Hubbard energy too small: {u_spinor}"
    );
    let m = base.m();
    let d_spinor = &spinor.densities[0];
    let off_diag = (0..m)
        .flat_map(|a| (0..m).map(move |b| d_spinor[(a, m + b)].norm()))
        .fold(0f64, f64::max);
    assert!(off_diag > 0.05, "tilted spinor must populate ud/du blocks");

    let spinor_grad = base.spinor_gradient(&spinor).unwrap();
    assert!(
        (spinor_grad.total.as_slice()[5] - uks_grad.total.as_slice()[5]).abs() < 1e-7,
        "spinor vs UKS force: {} vs {}",
        spinor_grad.total.as_slice()[5],
        uks_grad.total.as_slice()[5]
    );
    let h = 1e-3;
    let total = |dz: f64| {
        let mut pos = H2;
        pos[1][2] += dz;
        let geom = Geometry::new(
            base.geometry.numbers().to_vec(),
            pos.to_vec(),
            *base.geometry.cell(),
            base.geometry.pbc(),
            LengthUnit::Bohr,
        )
        .unwrap();
        let moved = base.at_geometry(geom).unwrap();
        let st = moved
            .spinor_kohn_sham_noncollinear("lda", 0, &[moment, moment], Some(13), &opts)
            .unwrap();
        assert!(st.converged);
        st.electronic_energy + moved.nuclear
    };
    let fd = (total(h) - total(-h)) / (2.0 * h);
    let err = (spinor_grad.total.as_slice()[5] - fd).abs();
    assert!(err < 1e-5, "tilted spinor +U force vs FD: diff {err:.3e}");

    // Wave F4-1: spinor molecular Casida +U (Liechtenstein) matches SU(2)-equivalent collinear UKS
    // excitations, spinor first-order Hubbard Fock kernel matches complex Fock FD to < 1e-9, spinor
    // excited-state gradients refuse hubbard_u, and weak-kick spinor RT-TDDFT reproduces spinor
    // Casida poles with energy conservation.
    let plain_spinor_base = integrals_with(&ctx, &H2, "lda", Precision::Float64, 2, None);
    let mut exc_req = mandacaru_core::molecular::excitations::Request::new(2);
    exc_req.spin = None;
    let exc_uks = mandacaru_core::molecular::excitations::excitations(&base, &uks, &exc_req)
        .unwrap();
    let exc_spinor = mandacaru_core::molecular::excitations::excitations(&base, &spinor, &exc_req)
        .unwrap();
    // Every spin-conserving UKS root appears among the Kramers/spinor excitation roots.
    for &w_uks in &exc_uks.energies {
        let best_diff = exc_spinor
            .energies
            .iter()
            .map(|w_sp| (w_sp - w_uks).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            best_diff < 1e-5,
            "UKS +U root {w_uks:.6} missing in spinor Casida {:?} (best diff {best_diff:.3e})",
            exc_spinor.energies
        );
    }

    // Complex Fock FD check of spinor_linear (real + imaginary + spin-flip perturbations).
    use mandacaru_core::quantum::C64;
    use nalgebra::DMatrix;
    let dim = 2 * m;
    let delta = DMatrix::from_fn(dim, dim, |a, b| {
        let s = ((a as f64 + 1.3) * 0.71 - (b as f64 + 0.4) * 0.37).sin() * 1e-2;
        let t = ((a as f64 + 0.5) * 0.43 + (b as f64 + 1.1) * 0.61).cos() * 1e-2;
        if a == b {
            C64::new(s, 0.0)
        } else if a < b {
            C64::new(s, t)
        } else {
            let s2 = ((b as f64 + 1.3) * 0.71 - (a as f64 + 0.4) * 0.37).sin() * 1e-2;
            let t2 = ((b as f64 + 0.5) * 0.43 + (a as f64 + 1.1) * 0.61).cos() * 1e-2;
            C64::new(s2, -t2)
        }
    });
    let hub_corr = base.corrections.as_ref().unwrap().hubbard.as_ref().unwrap();
    let analytic_du = hub_corr.spinor_linear(&ctx, &delta).unwrap();
    let fd_step = 1e-5;
    let d_plus = &spinor.densities[0] + &delta * C64::new(fd_step, 0.0);
    let d_minus = &spinor.densities[0] - &delta * C64::new(fd_step, 0.0);
    let f_plus = mandacaru_core::molecular::realtime::fock_matrices(
        &base,
        &spinor,
        std::slice::from_ref(&d_plus),
        None,
    )
    .unwrap()
    .0[0]
        .clone();
    let f_minus = mandacaru_core::molecular::realtime::fock_matrices(
        &base,
        &spinor,
        std::slice::from_ref(&d_minus),
        None,
    )
    .unwrap()
    .0[0]
        .clone();
    let p_plus = mandacaru_core::molecular::realtime::fock_matrices(
        &plain_spinor_base,
        &spinor,
        std::slice::from_ref(&d_plus),
        None,
    )
    .unwrap()
    .0[0]
        .clone();
    let p_minus = mandacaru_core::molecular::realtime::fock_matrices(
        &plain_spinor_base,
        &spinor,
        std::slice::from_ref(&d_minus),
        None,
    )
    .unwrap()
    .0[0]
        .clone();
    let fd_du = ((&f_plus - &f_minus) - (&p_plus - &p_minus)) / C64::new(2.0 * fd_step, 0.0);
    let du_err = (&fd_du - &analytic_du)
        .iter()
        .map(|z| z.norm())
        .fold(0.0f64, f64::max);
    assert!(du_err < 1e-9, "spinor +U linear Fock vs complex FD: {du_err:.3e}");

    let exc_grad_err =
        mandacaru_core::molecular::excitations::pick(&base, &spinor, &exc_req, 1, None)
            .unwrap()
            .gradient(&base, &spinor)
            .err()
            .expect("spinor excited-state gradient with hubbard_u must refuse")
            .to_string();
    assert!(exc_grad_err.contains("spinor DFT+U"), "{exc_grad_err}");

    // V9 weak-kick spinor RT-TDDFT dipole vs full spinor Casida poles with +U.
    let o = spinor.particles[0] + spinor.particles[1];
    let v = dim - o;
    let mut full_req =
        mandacaru_core::molecular::excitations::Request::new(o * v - 1);
    full_req.approximation = mandacaru_core::molecular::excitations::Approximation::Casida;
    full_req.spin = None;
    let casida =
        mandacaru_core::molecular::excitations::excitations(&base, &spinor, &full_req).unwrap();
    let dipoles = casida.spinor.as_ref().unwrap().transition_dipoles.clone();
    let alpha_casida = |t: f64| -> f64 {
        casida
            .energies
            .iter()
            .zip(&dipoles)
            .map(|(w, mu)| 2.0 * mu[2].norm_sqr() * (w * t).sin())
            .sum()
    };
    let kappa = 1e-4;
    let mut rt_pos =
        mandacaru_core::molecular::realtime::Request::new([0.0, 0.0, 1.0], kappa, 0.02, 50);
    rt_pos.scheme = mandacaru_core::math::propagator::Scheme::Aetrs;
    rt_pos.corrector_tolerance = 1e-10;
    let mut rt_neg = rt_pos.clone();
    rt_neg.strength = -kappa;
    let run_p = mandacaru_core::molecular::realtime::propagate(&base, &spinor, &rt_pos).unwrap();
    let run_m = mandacaru_core::molecular::realtime::propagate(&base, &spinor, &rt_neg).unwrap();
    assert!(run_p.energy_drift < 1e-3, "spinor +U RT drift: {}", run_p.energy_drift);
    let max_alpha = run_p
        .times
        .iter()
        .map(|&t| alpha_casida(t).abs())
        .fold(0.0f64, f64::max);
    let max_diff = run_p
        .times
        .iter()
        .enumerate()
        .map(|(i, &t)| ((run_p.along[i] - run_m.along[i]) / (2.0 * kappa) - alpha_casida(t)).abs())
        .fold(0.0f64, f64::max);
    let rel_v9 = max_diff / max_alpha.max(1e-12);
    assert!(rel_v9 < 2e-3, "spinor +U V9 error {rel_v9:.3e} (max_alpha {max_alpha:.3e})");
}

#[test]
#[ignore = "slow: 47 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test paw_forces_hubbard -- --ignored"]
fn the_hubbard_force_follows_total_energies_for_both_forms_restricted_and_unrestricted() {
    for (label, request, particles) in [
        ("Dudarev restricted", dudarev(), [6, 6]),
        ("Liechtenstein restricted", liechtenstein(), [6, 6]),
        ("Dudarev unrestricted", dudarev(), [7, 5]),
        ("Liechtenstein unrestricted", liechtenstein(), [7, 5]),
        ("Dudarev unrestricted, no beta electron", dudarev(), [2, 0]),
    ] {
        let (worst, size) = gate(
            label,
            &request,
            Method::Ks("lda"),
            particles,
            &[(0, 2), (1, 2)],
        );
        assert!(
            size > 1e-4,
            "{label}: the +U term is {size:.3e} eV/A, too small to test"
        );
        assert!(worst < 1e-6, "{label}: {worst:.3e} eV/A");
    }
}
