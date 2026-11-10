//! MG-Q7 at the problem level: a `CachedProblem` tapered by `taper()` against the untapered one
//! it came from, on the frozen H2, LiH and H2O Hamiltonians (`reference/tapering.json`).
//!
//! The pool generators that commute with the symmetries survive and act on the tapered register as
//! they did on the full one, so at any parameters the tapered energy, gradient and (lifted)
//! density matrices equal the untapered ones evaluated with the same generators.
use mandacaru_core::{
    quantum::{
        optimizer::{Method, Options},
        CachedProblem, Encoding, Operator, SolveOptions, C64,
    },
    Context,
};
use serde_json::Value;

fn frozen() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference/tapering.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// `(untapered, tapered)` problems of one frozen case, the same Hamiltonian and pool.
fn pair(
    ctx: &Context,
    molecule: &str,
    mapping: &str,
    pool: &str,
) -> (CachedProblem, CachedProblem) {
    let (m, particles) = match molecule {
        "H2" => (2, [1, 1]),
        "LiH" => (2, [1, 1]),
        _ => (6, [4, 4]),
    };
    let case = frozen()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["molecule"] == molecule && c["mapping"] == mapping)
        .unwrap()
        .clone();
    let n = case["n_qubits"].as_u64().unwrap() as usize;
    let terms: Vec<(String, C64)> = case["terms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            (
                t[0].as_str().unwrap().to_string(),
                C64::new(t[1].as_f64().unwrap(), t[2].as_f64().unwrap()),
            )
        })
        .collect();
    let build = || {
        let encoding = Encoding::new(mapping, 2 * m, particles).unwrap();
        assert_eq!(encoding.n_qubits(), n);
        let mut problem = CachedProblem::new(
            ctx,
            encoding,
            Operator::new(ctx, n, &terms).unwrap(),
            false,
            false,
            true,
        )
        .unwrap();
        if pool != "fermionic" {
            problem.use_pool(pool, true).unwrap();
        }
        problem
    };
    let (plain, mut tapered) = (build(), build());
    tapered.taper().unwrap();
    (plain, tapered)
}

fn max_difference(a: &[C64], b: &[C64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0., f64::max)
}

#[test]
fn a_tapered_problem_reproduces_the_untapered_energy_gradient_and_density_matrices() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    for (molecule, mapping, removed) in [
        ("H2", "jordan_wigner", 3),
        ("H2", "parity_reduced", 1),
        ("LiH", "bravyi_kitaev", 2),
        ("H2O", "jordan_wigner", 4),
        ("H2O", "parity_reduced", 2),
    ] {
        let (plain, tapered) = pair(&ctx, molecule, mapping, "fermionic");
        let taper = tapered.taper.as_ref().unwrap();
        let name = format!("{molecule} {mapping}");
        assert_eq!(taper.anchors.len(), removed, "{name}: the qubits removed");
        assert_eq!(
            tapered.n_qubits(),
            plain.n_qubits() - removed,
            "{name}: the register width"
        );
        assert_eq!(
            tapered.register("test").unwrap().len(),
            1 << tapered.n_qubits(),
            "{name}: the whole tapered register"
        );
        assert_eq!(
            tapered.pool.len() + tapered.dropped.len(),
            plain.pool.len(),
            "{name}: the pool is partitioned"
        );
        assert_eq!(tapered.pool_screenable.len(), tapered.pool.len());
        assert_eq!(tapered.growth_members.len(), tapered.pool.len());
        // The indices of the untapered pool the tapered one kept, in order.
        let dropped: Vec<usize> = tapered.dropped.iter().map(|(i, _)| *i).collect();
        let kept: Vec<usize> = (0..plain.pool.len())
            .filter(|i| !dropped.contains(i))
            .collect();
        assert!(!kept.is_empty(), "{name}: some generator survives");
        for (new, &old) in kept.iter().enumerate() {
            assert_eq!(
                tapered.pool[new].label, plain.pool[old].label,
                "{name}: labels follow the renumbering"
            );
        }
        // The reference energy and a point away from it.
        let theta: Vec<f64> = (0..kept.len())
            .map(|k| 0.11 * (k as f64 + 1.).sin() + 0.03)
            .collect();
        for trotter in [false, true] {
            let a = plain.evaluate(&kept, &theta, trotter).unwrap();
            let b = tapered
                .evaluate(&(0..kept.len()).collect::<Vec<_>>(), &theta, trotter)
                .unwrap();
            assert!(
                (a.energy - b.energy).abs() <= 1e-11,
                "{name}: energy {} against {}",
                a.energy,
                b.energy
            );
            let worst = a
                .gradient
                .iter()
                .zip(&b.gradient)
                .map(|(x, y)| (x - y).abs())
                .fold(0., f64::max);
            assert!(worst <= 1e-10, "{name}: gradient miss {worst:e}");
            // The tapered state lifted to the fermionic register reads the same density matrices.
            let want = plain.rdms(a.state.complex128().unwrap(), true).unwrap();
            let got = tapered.rdms(b.state.complex128().unwrap(), true).unwrap();
            let one = max_difference(want.one.as_slice(), got.one.as_slice());
            let two = max_difference(
                want.two.as_ref().unwrap().as_slice(),
                got.two.as_ref().unwrap().as_slice(),
            );
            println!("{name} trotter={trotter}: {} -> {} qubits, {} dropped, |dE| {:.1e}, |d1RDM| {one:.1e}, |d2RDM| {two:.1e}",
                plain.n_qubits(), tapered.n_qubits(), tapered.dropped.len(), (a.energy - b.energy).abs());
            assert!(one <= 1e-10 && two <= 1e-10, "{name}: RDMs {one:e} {two:e}");
        }
        // A fresh reference: the tapered register holds the Hartree-Fock determinant.
        let hf = tapered.evaluate(&[], &[], true).unwrap();
        assert!(
            (hf.energy - plain.evaluate(&[], &[], true).unwrap().energy).abs() <= 1e-11,
            "{name}: HF energy"
        );
    }
}

fn options(adapt: bool) -> SolveOptions {
    SolveOptions {
        adapt,
        gradient_tolerance: Some(1e-10),
        energy_tolerance: None,
        optimizer: Options {
            method: Method::Bfgs,
            tolerance: Some(1e-13),
            max_iterations: 500,
            max_evaluations: 20000,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn converged_energies_agree(cases: &[(&str, &str)]) {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    for &(molecule, mapping) in cases {
        let (plain, tapered) = pair(&ctx, molecule, mapping, "fermionic");
        for adapt in [false, true] {
            let a = plain.solve(&options(adapt), None).unwrap().energy;
            let b = tapered.solve(&options(adapt), None).unwrap().energy;
            println!(
                "{molecule} {mapping} adapt={adapt}: {a:+.12} {b:+.12} ({:.1e})",
                (a - b).abs()
            );
            assert!(
                (a - b).abs() <= 1e-8,
                "{molecule} {mapping} adapt={adapt}: {a} against {b}"
            );
        }
    }
}

#[test]
fn the_converged_energy_is_the_untapered_one_for_vqe_and_adapt() {
    converged_energies_agree(&[("H2", "parity_reduced"), ("LiH", "jordan_wigner")]);
}

#[test]
#[ignore = "slow: 21 s of VQE and ADAPT on twelve-qubit water, tapered and not; run: cargo test -p mandacaru-core --profile release-test --test tapered_problem -- --ignored"]
fn the_converged_energy_of_water_is_the_untapered_one() {
    converged_energies_agree(&[("H2O", "parity_reduced")]);
}

#[test]
fn the_pool_is_dropped_by_symmetry_and_a_register_pool_is_refused() {
    let ctx = Context::new(Some(1), 1 << 28).unwrap();
    let (plain, tapered) = pair(&ctx, "H2", "jordan_wigner", "fermionic");
    // The minimal-basis singles change the spatial parity; the double excitation keeps it.
    let dropped: Vec<&str> = tapered
        .dropped
        .iter()
        .map(|(_, label)| label.as_str())
        .collect();
    assert!(
        !dropped.is_empty() && dropped.iter().all(|l| l.starts_with('S')),
        "dropped {dropped:?}"
    );
    assert_eq!(tapered.pool.len() + dropped.len(), plain.pool.len());
    assert!(tapered.pool.iter().any(|g| g.label.starts_with('D')));
    // A pool that is register-based by construction cannot be tapered, with the wording the
    // parity-reduced register uses.
    for pool in ["qubit", "spin-orbit"] {
        let encoding = Encoding::new("jordan_wigner", 4, [1, 1]).unwrap();
        let h = plain.hamiltonian.copy_into(&ctx).unwrap();
        let mut problem = CachedProblem::new(&ctx, encoding, h, false, false, true).unwrap();
        problem.use_pool(pool, true).unwrap();
        let error = problem.taper().unwrap_err().to_string();
        assert!(error.contains("untapered register"), "{pool}: {error}");
        assert!(
            problem.taper.is_none(),
            "{pool}: a refusal leaves the problem as it was"
        );
    }
    // Tapering twice, and changing the pool afterwards, are refused.
    let (_, mut again) = pair(&ctx, "H2", "jordan_wigner", "fermionic");
    assert!(again
        .taper()
        .unwrap_err()
        .to_string()
        .contains("already tapered"));
    assert!(again
        .use_pool("qeb", true)
        .unwrap_err()
        .to_string()
        .contains("before tapering"));
}

#[test]
fn a_state_off_the_particle_sector_is_not_lifted_into_density_matrices() {
    let ctx = Context::new(Some(1), 1 << 28).unwrap();
    let (_, tapered) = pair(&ctx, "H2O", "jordan_wigner", "fermionic");
    let size = 1 << tapered.n_qubits();
    let (mut inside, mut outside) = (0, 0);
    for k in 0..size {
        let mut basis_state = vec![C64::default(); size];
        basis_state[k] = C64::new(1., 0.);
        match tapered.rdms(&basis_state, false) {
            Ok(rdms) => {
                // The trace of the one-body matrix is the particle number: the sector's.
                let n = tapered.encoding.modes;
                let trace: f64 = (0..n).map(|p| rdms.one.as_slice()[p * n + p].re).sum();
                assert!(
                    (trace - 8.).abs() < 1e-10,
                    "determinant {k}: {trace} particles"
                );
                inside += 1;
            }
            Err(e) => {
                assert!(e.to_string().contains("sector"), "{e}");
                outside += 1;
            }
        }
    }
    println!("{inside} tapered basis states lift into the (4, 4) sector, {outside} do not");
    assert!(inside > 0 && outside > 0);
    // The reference determinant is inside; a state of the wrong size is refused outright.
    let reference = tapered.evaluate(&[], &[], true).unwrap();
    assert!(tapered
        .rdms(reference.state.complex128().unwrap(), false)
        .is_ok());
    assert!(tapered.rdms(&vec![C64::default(); 3], false).is_err());
}

#[test]
fn the_subspace_references_stay_in_the_symmetry_sector() {
    let ctx = Context::new(Some(1), 1 << 28).unwrap();
    let options = SolveOptions {
        optimizer: Options {
            method: Method::Bfgs,
            tolerance: Some(1e-12),
            ..Default::default()
        },
        ..Default::default()
    };
    let (plain, tapered) = pair(&ctx, "H2", "jordan_wigner", "fermionic");
    let a = plain
        .solve_subspace(&options, &[1.0], None)
        .unwrap()
        .energies[0];
    let b = tapered
        .solve_subspace(&options, &[1.0], None)
        .unwrap()
        .energies[0];
    assert!((a - b).abs() <= 1e-9, "{a} against {b}");
    // Two states of the tapered register come out distinct and ordered, both in the sector.
    let two = tapered.solve_subspace(&options, &[2.0, 1.0], None).unwrap();
    assert_eq!(two.energies.len(), 2);
    assert!(two.energies[0] <= two.energies[1]);
}

/// The hardware rule (no `2^n` object on the path of a wide register) against the whole-register
/// decision: a tapered problem simulates the whole tapered register, so on fifty qubits it is
/// refused by the `Context` before anything is allocated, and the untapered problem (a particle
/// sector of 625 determinants) is untouched.
#[test]
fn a_fifty_qubit_register_refuses_to_taper_and_builds_no_exponential_object() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let n = 50;
    let string = |hot: &[(usize, char)]| -> String {
        (0..n)
            .map(|j| hot.iter().find(|h| h.0 == j).map_or('I', |h| h.1))
            .collect()
    };
    let mut terms = vec![];
    for j in 0..n {
        terms.push((string(&[(j, 'Z')]), C64::new(0.1 * (j + 1) as f64, 0.)));
    }
    // Hopping inside each spin block (modes 0..25 alpha, 25..50 beta) conserves both counts.
    for j in (0..n - 1).filter(|&j| j != n / 2 - 1) {
        terms.push((string(&[(j, 'X'), (j + 1, 'X')]), C64::new(0.3, 0.)));
        terms.push((string(&[(j, 'Y'), (j + 1, 'Y')]), C64::new(0.3, 0.)));
    }
    let encoding = Encoding::new("jordan_wigner", n, [1, 1]).unwrap();
    let mut problem = CachedProblem::new(
        &ctx,
        encoding,
        Operator::new(&ctx, n, &terms).unwrap(),
        true,
        false,
        true,
    )
    .unwrap();
    assert_eq!(problem.register("test").unwrap().len(), 625);
    let before = ctx.memory_used();
    let error = problem.taper().unwrap_err().to_string();
    println!("tapering 50 qubits: {error}");
    assert!(problem.taper.is_none() && problem.n_qubits() == n && problem.register("test").unwrap().len() == 625);
    assert!(
        ctx.memory_used() <= before + (8 << 20),
        "the refusal leaves nothing allocated"
    );
    assert!(
        problem.evaluate(&[], &[], true).is_ok(),
        "the untapered problem still runs"
    );
}

#[test]
fn a_tapered_checkpoint_is_written_on_the_tapered_register_and_resumes_only_there() {
    use mandacaru_core::persistence::checkpoint;
    let ctx = Context::new(Some(1), 1 << 28).unwrap();
    let (plain, tapered) = pair(&ctx, "H2", "jordan_wigner", "fermionic");
    let solved = tapered.solve(&options(true), None).unwrap();
    let parameters = solved.parameters.clone();
    let document = checkpoint(
        &ctx,
        &tapered,
        &solved.indices,
        &parameters,
        true,
        "adapt-vqe",
        serde_json::json!({}),
        solved.energy,
    )
    .unwrap();
    // The state and the energy the checkpoint prepares are those of the tapered register.
    let expectation = document.expectation(&ctx).unwrap();
    assert!(
        (expectation - solved.energy).abs() < 1e-10,
        "{expectation} against {}",
        solved.energy
    );
    let resumed = document.resume(&tapered, &options(true)).unwrap();
    assert!((resumed.energy - solved.energy).abs() < 1e-10);
    // The untapered problem is another register: the checkpoint is not its own.
    let error = document
        .resume(&plain, &options(true))
        .err()
        .expect("another register")
        .to_string();
    assert!(error.contains("register"), "{error}");
}
