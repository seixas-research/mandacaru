//! The analytic gradient of a variational state on a PAW-LCAO reference
//! (`molecular::variational_gradient`), gated step by step against independent derivatives on
//! the synthetic PAW hydrogen molecule of `paw_forces_support` (no dataset files).
mod paw_forces_support;
mod variational_forces_support;

use mandacaru_core::{
    molecular::{
        transform,
        variational_gradient::{extended_rdms, two_body_forms, FixedDensities},
        MeanField, MolecularIntegrals,
    },
    quantum::{properties, SolveOptions, C64},
    Context, Geometry, LengthUnit,
};
use nalgebra::DMatrix;
use paw_forces_support::synthetic::{integrals, solve, Method, H2};

fn real(matrix: &DMatrix<C64>) -> DMatrix<f64> {
    assert!(matrix.iter().all(|v| v.im.abs() < 1e-12));
    matrix.map(|v| v.re)
}

/// The raw-AO coefficients of the model orbitals of a mean-field state.
fn raw_orbitals(base: &MolecularIntegrals, state: &MeanField) -> DMatrix<f64> {
    &base.lowdin * real(&state.model_orbitals)
}

fn moved(base: &MolecularIntegrals, atom: usize, axis: usize, delta: f64) -> MolecularIntegrals {
    let mut positions = base.geometry.positions().to_vec();
    positions[atom][axis] += delta;
    let geometry = Geometry::new(
        base.geometry.numbers().to_vec(),
        positions,
        *base.geometry.cell(),
        base.geometry.pbc(),
        LengthUnit::Bohr,
    )
    .unwrap();
    base.at_geometry_with_fixed_pivots(geometry).unwrap()
}

/// `sum gamma h + 1/2 sum Gamma <pq|rs> + E_nn` of the raw-AO orbitals `a` on `integrals`.
fn energy_at_fixed_orbitals(
    integrals: &MolecularIntegrals,
    a: &DMatrix<f64>,
    one: &[f64],
    two: &[f64],
) -> f64 {
    let ctx = &integrals.context;
    let m = integrals.m();
    let t = a.ncols();
    let c = integrals.lowdin.clone().try_inverse().unwrap() * a;
    let c = c.map(|v| C64::new(v, 0.));
    let h = DMatrix::from_row_slice(m, m, integrals.one.as_slice()).map(|v| C64::new(v, 0.));
    let h = c.adjoint() * h * &c;
    let eri = integrals.eri().unwrap();
    let g = transform::four(ctx, &eri, [&c, &c, &c, &c]).unwrap();
    let mut energy = integrals.nuclear;
    for p in 0..t {
        for q in 0..t {
            energy += one[p * t + q] * h[(p, q)].re;
        }
    }
    for (d, v) in two.iter().zip(g.as_slice()) {
        energy += 0.5 * d * v.re;
    }
    energy
}

#[test]
fn the_fixed_density_gradient_of_the_hartree_fock_determinant_is_the_analytic_force() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H2, "lda");
        let state = solve(&base, Method::Hf("rhf"), [1, 1]);
        let reference = base.gradient(&state).unwrap();
        let all = raw_orbitals(&base, &state);
        let occupied = all.columns(0, 1).into_owned();
        // The determinant as one doubly occupied "frozen" orbital: gamma = 2, Gamma = 4 - 2.
        let (one, two) = extended_rdms(1, &[], &[]).unwrap();
        let p = &occupied * DMatrix::from_row_slice(1, 1, &one) * occupied.transpose();
        let w = &occupied * occupied.transpose() * (2. * state.energies[0][0]);
        let densities = FixedDensities {
            one: p,
            weighted: w,
            frame: occupied,
            two: two_body_forms(&two, 1).unwrap(),
            cluster_frame: None,
        };
        let gradient = base.fixed_density_gradient(&densities).unwrap();
        let worst = gradient
            .total
            .as_slice()
            .iter()
            .zip(reference.total.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        println!("HF determinant: max |dE - paw_gradient| = {worst:.3e} Ha/Bohr");
        assert!(worst < 1e-10, "{worst:.3e}");
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_fixed_density_gradient_of_an_adapt_state_is_the_derivative_at_fixed_orbitals() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let base = integrals(&ctx, &H2, "lda");
        let state = solve(&base, Method::Hf("rhf"), [1, 1]);
        let model = state.model(&base).unwrap();
        let problem = model
            .problem(&ctx, "jordan_wigner", true, false, true)
            .unwrap();
        let solved = problem
            .solve(
                &SolveOptions {
                    adapt: true,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        let rdm = properties::spatial_rdms(
            &ctx,
            &problem.encoding,
            problem.register("test").unwrap(),
            solved.state.complex128().unwrap(),
        )
        .unwrap();
        let t = rdm.m;
        let one: Vec<f64> = rdm.one.as_slice().iter().map(|v| v.re).collect();
        let two: Vec<f64> = rdm.two.as_slice().iter().map(|v| v.re).collect();
        let a = raw_orbitals(&base, &state);
        assert_eq!(a.ncols(), t);
        let p = &a * DMatrix::from_row_slice(t, t, &one) * a.transpose();
        let m = base.m();
        let densities = FixedDensities {
            one: p,
            weighted: DMatrix::zeros(m, m),
            frame: a.clone(),
            two: two_body_forms(&two, t).unwrap(),
            cluster_frame: None,
        };
        assert!((energy_at_fixed_orbitals(&base, &a, &one, &two) - solved.energy).abs() < 1e-9);
        let gradient = base.fixed_density_gradient(&densities).unwrap();
        let mut worst = 0f64;
        for atom in 0..2 {
            for axis in 0..3 {
                let e =
                    |d: f64| energy_at_fixed_orbitals(&moved(&base, atom, axis, d), &a, &one, &two);
                let h = 1e-3;
                let coarse = (e(h) - e(-h)) / (2. * h);
                let fine = (e(0.5 * h) - e(-0.5 * h)) / h;
                let reference = (4. * fine - coarse) / 3.;
                let analytic = gradient.total.as_slice()[3 * atom + axis];
                println!("atom {atom} axis {axis}: analytic {analytic:.10e} finite difference {reference:.10e}");
                worst = worst.max((analytic - reference).abs());
            }
        }
        println!("ADAPT state at fixed orbitals: max |dE - FD| = {worst:.3e} Ha/Bohr");
        assert!(worst < 1e-7, "{worst:.3e}");
    }
    assert_eq!(ctx.memory_used(), 0);
}

const HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM: f64 = 27.211386245988 / 0.529177210903;

/// The analytic force of an ADAPT state against the finite-difference oracle (displaced SCF,
/// selector and Hamiltonian, Richardson steps of 1e-3 Bohr) on every component. `shift` moves the
/// parameters off the optimum, so the Z-vector correction enters both routes.
fn parity(positions: &[[f64; 3]], particles: [usize; 2], active: Option<&str>, shift: f64) {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let base = integrals(&ctx, positions, "lda");
        parity_on(&base, particles, active, shift);
    }
    assert_eq!(ctx.memory_used(), 0);
}

fn parity_on(base: &MolecularIntegrals, particles: [usize; 2], active: Option<&str>, shift: f64) {
    parity_with(base, "rhf", particles, active, shift);
}

fn parity_with(
    base: &MolecularIntegrals,
    kind: &'static str,
    particles: [usize; 2],
    active: Option<&str>,
    shift: f64,
) {
    use mandacaru_core::molecular::response;
    use mandacaru_core::quantum::lagrangian::Ansatz;
    use variational_forces_support::oracle;
    let ctx = &base.context;
    {
        let state = solve(base, Method::Hf(kind), particles);
        let model = match active {
            Some(spec) => std::sync::Arc::new(state.active_model(base, spec).unwrap()),
            None => state.model(base).unwrap(),
        };
        let problem = model
            .problem(ctx, "jordan_wigner", true, false, true)
            .unwrap();
        let solved = problem
            .solve(
                &SolveOptions {
                    adapt: true,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        let theta: Vec<f64> = solved
            .parameters
            .iter()
            .enumerate()
            .map(|(k, t)| t + shift * ((k + 1) as f64).cos())
            .collect();
        let prepared = problem
            .prepare_state(&solved.indices, &theta, true)
            .unwrap();
        let psi = prepared.complex128().unwrap();
        let ansatz = Ansatz {
            indices: &solved.indices,
            parameters: &theta,
            product_form: true,
        };
        let reference = oracle::gradient(
            base,
            &state,
            &model,
            &problem,
            psi,
            &oracle::Options {
                active_spec: active,
                ansatz: Some(ansatz),
                ..Default::default()
            },
        )
        .unwrap();
        let analytic = response::gradient(
            base,
            &state,
            &model,
            &problem,
            psi,
            &response::Options {
                ansatz: Some(ansatz),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(analytic.scheme, "analytic");
        assert!((analytic.energy - reference.energy).abs() < 1e-10);
        let to_ev = HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM;
        let mut worst = 0f64;
        for (k, (a, b)) in analytic
            .gradient
            .as_slice()
            .iter()
            .zip(reference.gradient.as_slice())
            .enumerate()
        {
            println!("{active:?} component {k}: analytic {a:.10e} oracle {b:.10e} Ha/Bohr");
            worst = worst.max((a - b).abs() * to_ev);
        }
        let estimate = reference.estimate.unwrap() * to_ev;
        println!(
            "{active:?} shift {shift}: max |F - F_oracle| = {worst:.3e} eV/A \
             (oracle estimate {estimate:.3e} eV/A), reference stationarity {:.2e}, CPHF residual {:.2e}, forms {}",
            analytic.reference_stationarity, analytic.cphf_residual, analytic.forms
        );
        assert!(worst < estimate.max(1e-6), "{worst:.3e} eV/A");
    }
}

#[test]
fn the_analytic_force_of_a_full_space_adapt_state_matches_the_finite_difference_oracle() {
    parity(&H2, [1, 1], None, 0.);
}

#[test]
fn the_analytic_force_of_an_mp2_active_space_adapt_state_matches_the_finite_difference_oracle() {
    parity(
        &H2,
        [1, 1],
        Some(r#"{"orbitals": 4, "method": "mp2"}"#),
        0.02,
    );
}

/// Linear H4: two occupied orbitals, degenerate pi pairs, a frozen orbital.
const H4: [[f64; 3]; 4] = [[0., 0., -2.5], [0., 0., -0.8], [0., 0., 0.9], [0., 0., 2.6]];

#[test]
#[ignore = "slow: 95 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test variational_forces -- --ignored h4_analytic_forces"]
fn h4_analytic_forces_match_the_oracle_for_mp2_frozen_and_energy_active_spaces() {
    parity(
        &H4,
        [2, 2],
        Some(r#"{"orbitals": 5, "method": "mp2", "frozen": [0]}"#),
        0.02,
    );
    parity(&H4, [2, 2], Some(r#"{"orbitals": 6, "method": "mp2"}"#), 0.);
    parity(&H4, [2, 2], Some(r#"{"orbitals": 5}"#), 0.);
}

#[test]
fn a_degenerate_pair_is_aligned_by_rotation_and_a_split_pair_keeps_its_own() {
    // Four orbitals, one occupied; orbitals 1 and 2 are an exactly degenerate empty pair (as the
    // pi virtuals of LiH). The displaced SCF returns that pair rotated by 0.9 rad, which an
    // assignment cannot undo: the aligned frame must be the reference one. When the pair is split,
    // its rotation is physical and stays (only the order and phase are matched).
    let context = Context::new(Some(1), 32 << 20).unwrap();
    let identity = DMatrix::identity(4, 4).map(|v| C64::new(v, 0.));
    let (c, s) = (0.9f64.cos(), 0.9f64.sin());
    let mut rotated = identity.clone();
    rotated[(1, 1)] = C64::new(c, 0.);
    rotated[(2, 1)] = C64::new(s, 0.);
    rotated[(1, 2)] = C64::new(-s, 0.);
    rotated[(2, 2)] = C64::new(c, 0.);
    let aligned = |labels: &[f64]| {
        let alignment = variational_forces_support::oracle::align_orbitals(
            &context,
            &identity,
            &rotated,
            [1, 1],
            0.5,
            Some(labels),
            Some(labels),
        )
        .unwrap();
        // The aligned frame: column p is sum_a rotated_a W[a, p].
        DMatrix::from_fn(4, 4, |row, p| {
            alignment.columns[p]
                .iter()
                .map(|&(a, w)| rotated[(row, a)] * w)
                .sum::<C64>()
        })
    };
    let degenerate = aligned(&[-0.5, 0.26, 0.26, 1.2]);
    assert!((&degenerate - &identity).norm() < 1e-12, "{degenerate}");
    let split = aligned(&[-0.5, 0.20, 0.26, 1.2]);
    assert!(
        (&split - &identity).norm() > 0.5,
        "a split pair keeps its physical rotation"
    );
    // Without labels the frame is matched by order and phase only, as before.
    let unlabeled = variational_forces_support::oracle::align_orbitals(
        &context,
        &identity,
        &rotated,
        [1, 1],
        0.5,
        None,
        None,
    )
    .unwrap();
    assert!(unlabeled.columns.iter().all(|column| column.len() == 1));
    // An occupied and an empty orbital of equal energy are never mixed.
    let crossing = variational_forces_support::oracle::align_orbitals(
        &context,
        &identity,
        &identity,
        [1, 1],
        0.5,
        Some(&[0.3, 0.3, 0.5, 0.7]),
        Some(&[0.3, 0.3, 0.5, 0.7]),
    )
    .unwrap();
    assert!(crossing.columns.iter().all(|column| column.len() == 1));
    assert_eq!(context.memory_used(), 0);
}

mod paw_molecular_support;

/// Water on the shipped `lda-sr` set (oxygen with a core, analytic s, p, d shells, no symmetry):
/// an MP2 active space with a frozen orbital, off its optimum, against the oracle.
#[test]
#[ignore = "slow: 36 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test variational_forces -- --ignored water_analytic_force"]
fn water_analytic_force_matches_the_oracle_on_the_shipped_set() {
    use mandacaru_core::paw::molecular::Arrays;
    use paw_molecular_support::{analytic_shells, dataset, molecule};
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(4), 1536 << 20).unwrap();
    {
        let mut arrays = Arrays::new();
        let mut species = Vec::new();
        for element in ["O", "H"] {
            let data = dataset::read(&ctx, "lda-sr", element);
            species.push(dataset::species(&ctx, &mut arrays, &data));
        }
        let water = [
            [0.11, 0.07, 0.12],
            [1.55, 0.04, -0.93],
            [-1.43, -0.05, -0.97],
        ];
        let base = molecule(
            &ctx,
            &[8, 1, 1],
            &water,
            12.,
            0.5,
            (species, vec![0, 1, 1], arrays),
            "raw",
            &analytic_shells(&[0, 1, 2]),
        );
        parity_on(
            &base,
            [5, 5],
            Some(r#"{"orbitals": 6, "method": "mp2", "frozen": "auto"}"#),
            0.01,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}

/// A doublet H3 on an unrestricted reference: the natural orbitals of the total density, a CPHF
/// over both spins, and an active space that cuts the exactly degenerate empty cluster (its
/// Lowdin gauge), against the oracle.
#[test]
#[ignore = "slow: about 36 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test variational_forces -- --ignored an_unrestricted_doublet"]
fn an_unrestricted_doublet_matches_the_oracle() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let h3 = [[0., 0., -1.7], [0.3, 0., 0.1], [0., 0.2, 1.6]];
        let base = integrals(&ctx, &h3, "lda");
        parity_with(&base, "uhf", [2, 1], Some(r#"{"orbitals": 3}"#), 0.);
        parity_with(&base, "uhf", [2, 1], Some(r#"{"orbitals": 3}"#), 0.01);
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// A triplet H2: the two active natural orbitals are an exactly degenerate pair (both singly
/// occupied), so the whole response inside the active space is the Lowdin gauge.
#[test]
fn a_triplet_with_a_degenerate_active_pair_matches_the_oracle() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let base = integrals(&ctx, &H2, "lda");
        parity_with(&base, "uhf", [2, 0], Some(r#"{"orbitals": 2}"#), 0.);
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// A triplet H2 with an MP2 active space on its unrestricted reference: the open-shell MP2 has
/// alpha electrons only (singles and same-spin pairs), and the third and fourth active orbitals are
/// the most occupied virtual natural orbitals, an exactly degenerate pair (n = 3.5e-4 twice) kept
/// whole: an active space that cuts it has no continuous frame for the oracle to follow.
#[test]
fn a_triplet_with_an_mp2_active_space_matches_the_oracle() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let base = integrals(&ctx, &H2, "lda");
        parity_with(
            &base,
            "uhf",
            [2, 0],
            Some(r#"{"orbitals": 4, "method": "mp2"}"#),
            0.01,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// A doublet H3 with an MP2 active space: singles, same-spin and opposite-spin pairs of the
/// open-shell MP2, off the optimum (the Z-vector), against the oracle. Three orbitals: the
/// virtual natural orbital kept (n = 1.3e-2) is well apart from the next (3.3e-3), while the
/// following two (3.27e-3, 3.21e-3) nearly cross, so a fourth would put the cut at a crossing.
#[test]
#[ignore = "slow: 32 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test variational_forces -- --ignored an_unrestricted_doublet_with_an_mp2"]
fn an_unrestricted_doublet_with_an_mp2_active_space_matches_the_oracle() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let h3 = [[0., 0., -1.7], [0.3, 0., 0.1], [0., 0.2, 1.6]];
        let base = integrals(&ctx, &h3, "lda");
        parity_with(
            &base,
            "uhf",
            [2, 1],
            Some(r#"{"orbitals": 3, "method": "mp2"}"#),
            0.01,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// The hydroxyl radical on the shipped `lda-sr` set (oxygen with a core), unrestricted, off its
/// optimum, against the oracle.
#[test]
#[ignore = "slow: 17 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test variational_forces -- --ignored hydroxyl_analytic_force"]
fn hydroxyl_analytic_force_matches_the_oracle_on_the_shipped_set() {
    use mandacaru_core::paw::molecular::Arrays;
    use paw_molecular_support::{analytic_shells, dataset, molecule};
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(4), 1536 << 20).unwrap();
    {
        let mut arrays = Arrays::new();
        let mut species = Vec::new();
        for element in ["O", "H"] {
            let data = dataset::read(&ctx, "lda-sr", element);
            species.push(dataset::species(&ctx, &mut arrays, &data));
        }
        let oh = [[0.06, -0.04, 0.10], [0.13, 0.05, 1.88]];
        let base = molecule(
            &ctx,
            &[8, 1],
            &oh,
            12.,
            0.5,
            (species, vec![0, 1], arrays),
            "raw",
            &analytic_shells(&[0, 1, 2]),
        );
        parity_with(&base, "uhf", [4, 3], Some(r#"{"orbitals": 5}"#), 0.01);
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}

/// The hydroxyl radical with an MP2 active space on the shipped `lda-sr` set, off the optimum: three
/// doubly occupied orbitals, so the selector also rotates them by the occupied MP2 density's
/// eigenvectors (the case that found the missing occupied-block response: without it 5e-4 eV/A at
/// the stationary point, 8.8e-3 with the Z-vector), and two virtual natural orbitals, the second
/// 2.8e-4 in occupation from the first inactive one.
#[test]
#[ignore = "slow: 35 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test variational_forces -- --ignored hydroxyl_mp2"]
fn hydroxyl_mp2_analytic_force_matches_the_oracle_on_the_shipped_set() {
    use mandacaru_core::paw::molecular::Arrays;
    use paw_molecular_support::{analytic_shells, dataset, molecule};
    let started = std::time::Instant::now();
    let ctx = Context::new(Some(4), 1536 << 20).unwrap();
    {
        let mut arrays = Arrays::new();
        let mut species = Vec::new();
        for element in ["O", "H"] {
            let data = dataset::read(&ctx, "lda-sr", element);
            species.push(dataset::species(&ctx, &mut arrays, &data));
        }
        let oh = [[0.06, -0.04, 0.10], [0.13, 0.05, 1.88]];
        let base = molecule(
            &ctx,
            &[8, 1],
            &oh,
            12.,
            0.5,
            (species, vec![0, 1], arrays),
            "raw",
            &analytic_shells(&[0, 1, 2]),
        );
        parity_with(
            &base,
            "uhf",
            [4, 3],
            Some(r#"{"orbitals": 6, "method": "mp2", "frozen": "auto"}"#),
            0.01,
        );
        // A frozen doubly occupied orbital: the occupied natural orbitals of the rest rotate.
        parity_with(
            &base,
            "uhf",
            [4, 3],
            Some(r#"{"orbitals": 6, "method": "mp2", "frozen": [0]}"#),
            0.,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
    println!("elapsed {:.0} s", started.elapsed().as_secs_f64());
}

#[test]
fn subspace_vasqa_ground_and_excited_analytic_forces_match_finite_differences() {
    use mandacaru_core::{
        molecular::response::{self, SubspaceOptions},
        quantum::{lagrangian::Ansatz, optimizer, search::vasqa::SearchOptions},
    };

    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H2, "lda");
        let state = solve(&base, Method::Hf("rhf"), [1, 1]);
        let active = r#"{"orbitals": 2, "method": "energy"}"#;
        let model = std::sync::Arc::new(state.active_model(&base, active).unwrap());
        let problem = model
            .problem(&ctx, "jordan_wigner", true, false, true)
            .unwrap();
        // Four weights span all four determinants of H2's (1,1) active sector so both
        // ground (root 0) and doubly-excited (root 3) eigenstates couple to the ansatz.
        let weights = [0.6, 0.25, 0.1, 0.05];
        let search = SearchOptions {
            max_steps: 24,
            max_length: 2,
            move_weights: [1., 0.5, 0.5],
            temperatures: [0.02, 0.0005],
            length_penalty: 0.,
            proposal_temperature: 0.1,
            uniform: false,
            warm_start: true,
            ..Default::default()
        };
        let found = problem
            .search_subspace(&search, &weights, &[], &[])
            .unwrap();
        assert!(!found.solution.indices.is_empty());
        let indices = found.solution.indices.clone();
        let solve_opts = SolveOptions {
            adapt: true,
            max_growth: indices.len(),
            optimizer: optimizer::Options {
                tolerance: Some(1e-12),
                ..Default::default()
            },
            ..Default::default()
        };
        let solved = problem
            .solve_subspace_sequence(
                &solve_opts,
                &weights,
                Some(&found.solution.parameters),
                Some(&indices),
            )
            .unwrap();
        let states = solved.states.complex128().unwrap();
        let d = states.len() / weights.len();
        let ansatz = Ansatz {
            indices: &solved.indices,
            parameters: &solved.parameters,
            product_form: true,
        };
        let h = 2e-4;
        let mut excited_lagrangian_tested = false;
        for root in 0..weights.len() {
            let psi_root = &states[root * d..(root + 1) * d];
            let analytic = response::subspace_gradient(
                &base,
                &state,
                &model,
                &problem,
                psi_root,
                &SubspaceOptions {
                    weights: &weights,
                    ansatz: Some(ansatz),
                    lagrangian: Default::default(),
                    internal: false,
                },
            )
            .unwrap();
            assert_eq!(analytic.scheme, "analytic");
            assert!((analytic.energy - solved.energies[root]).abs() < 1e-10);
            let eval_at = |delta: f64| -> f64 {
                let disp = moved(&base, 1, 2, delta);
                let disp_scf = solve(&disp, Method::Hf("rhf"), [1, 1]);
                let disp_model = disp_scf.active_model(&disp, active).unwrap();
                let disp_problem = disp_model
                    .problem(&ctx, "jordan_wigner", true, false, true)
                    .unwrap();
                let disp_solved = disp_problem
                    .solve_subspace_sequence(
                        &solve_opts,
                        &weights,
                        Some(&solved.parameters),
                        Some(&solved.indices),
                    )
                    .unwrap();
                disp_solved.energies[root]
            };
            let fd_z = (eval_at(h) - eval_at(-h)) / (2. * h);
            let analytic_z = analytic.gradient.as_slice()[5];
            assert!(
                (analytic_z - fd_z).abs() < 5e-7,
                "root {root} converged dE/dz mismatch: analytic={analytic_z:.10} fd={fd_z:.10}"
            );

            let shifted_params: Vec<f64> = solved
                .parameters
                .iter()
                .enumerate()
                .map(|(k, t)| t + 0.04 * ((k + 1) as f64).cos())
                .collect();
            let shifted_ansatz = Ansatz {
                indices: &solved.indices,
                parameters: &shifted_params,
                product_form: true,
            };
            let shifted_res = problem
                .solve_subspace_sequence(
                    &SolveOptions {
                        adapt: true,
                        max_growth: solved.indices.len(),
                        optimizer: optimizer::Options {
                            max_iterations: 0,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    &weights,
                    Some(&shifted_params),
                    Some(&solved.indices),
                )
                .unwrap();
            assert_eq!(shifted_res.parameters, shifted_params);
            let shifted_states = shifted_res.states.complex128().unwrap();
            let shifted_root = &shifted_states[root * d..(root + 1) * d];
            let lagrangian = response::subspace_gradient(
                &base,
                &state,
                &model,
                &problem,
                shifted_root,
                &SubspaceOptions {
                    weights: &weights,
                    ansatz: Some(shifted_ansatz),
                    lagrangian: Default::default(),
                    internal: false,
                },
            )
            .unwrap();
            if let mandacaru_core::quantum::lagrangian::SubspaceResidual::Corrected {
                z: fixed_z,
                ..
            } = problem
                .subspace_lagrangian(&shifted_ansatz, &weights, shifted_root, &Default::default())
                .unwrap()
            {
                if root > 0 {
                    excited_lagrangian_tested = true;
                }
                let shifted_surface_at = |delta: f64| -> f64 {
                    let disp = moved(&base, 1, 2, delta);
                    let disp_scf = solve(&disp, Method::Hf("rhf"), [1, 1]);
                    let disp_model = disp_scf.active_model(&disp, active).unwrap();
                    let disp_problem = disp_model
                        .problem(&ctx, "jordan_wigner", true, false, true)
                        .unwrap();
                    let d_sub = disp_problem.register("sub").unwrap().len();
                    let disp_res = disp_problem
                        .solve_subspace_sequence(
                            &SolveOptions {
                                adapt: true,
                                max_growth: solved.indices.len(),
                                optimizer: optimizer::Options {
                                    max_iterations: 0,
                                    ..Default::default()
                                },
                                ..Default::default()
                            },
                            &weights,
                            Some(&shifted_params),
                            Some(&solved.indices),
                        )
                        .unwrap();
                    let ref_rows = disp_problem.subspace_ref_rows(weights.len()).unwrap();
                    let mut g_ens = vec![0.; solved.indices.len()];
                    for (&row, &w) in ref_rows.iter().zip(&weights) {
                        let mut seed = vec![C64::new(0., 0.); d_sub];
                        seed[row] = C64::new(1., 0.);
                        let ev = disp_problem
                            .evaluate_from_state(&solved.indices, &shifted_params, true, &seed)
                            .unwrap();
                        for (acc, &g) in g_ens.iter_mut().zip(&ev.gradient) {
                            *acc += w * g;
                        }
                    }
                    let z_dot_g: f64 = fixed_z.iter().zip(&g_ens).map(|(a, b)| a * b).sum();
                    disp_res.energies[root] - z_dot_g
                };
                let shifted_z = lagrangian.gradient.as_slice()[5];
                let shifted_fd_z = (shifted_surface_at(h) - shifted_surface_at(-h)) / (2. * h);
                assert!(
                    (shifted_z - shifted_fd_z).abs() < 5e-7,
                    "root {root} shifted Lagrangian d(E - z.g_ens)/dz mismatch: analytic={shifted_z:.10} fd={shifted_fd_z:.10}"
                );
            }
        }
        assert!(
            excited_lagrangian_tested,
            "at least one excited eigenstate should test the active Lagrangian Z-vector branch"
        );

        // Verify K=1 exact parity with single-state response::gradient
        let ss = problem
            .solve(
                &SolveOptions {
                    adapt: true,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        let ss_p: Vec<f64> = ss.parameters.iter().map(|t| t + 0.04).collect();
        let ss_eval = problem.prepare_state(&ss.indices, &ss_p, true).unwrap();
        let ss_psi = ss_eval.complex128().unwrap();
        let ss_ansatz = Ansatz {
            indices: &ss.indices,
            parameters: &ss_p,
            product_form: true,
        };
        let g_single = response::gradient(
            &base,
            &state,
            &model,
            &problem,
            ss_psi,
            &response::Options {
                ansatz: Some(ss_ansatz),
                ..Default::default()
            },
        )
        .unwrap();
        let g_sub1 = response::subspace_gradient(
            &base,
            &state,
            &model,
            &problem,
            ss_psi,
            &SubspaceOptions {
                weights: &[1.0],
                ansatz: Some(ss_ansatz),
                lagrangian: Default::default(),
                internal: false,
            },
        )
        .unwrap();
        assert!((g_single.gradient.as_slice()[5] - g_sub1.gradient.as_slice()[5]).abs() < 1e-13);
    }
    assert_eq!(ctx.memory_used(), 0);
}
