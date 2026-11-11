//! The factorized (`O(M)` basis-rotation) measurement against the frozen Python oracle
//! (`test/migration/reference/factorization.json`, written by `freeze_factorization.py`).
//!
//! H2 has identity rotations (the orbital rotation of its two spatial orbitals is trivial), so it
//! checks observables and energies only. LiH has six Givens rotations per basis and signs of
//! -1: it is the case that pins the generator order and phase.
use mandacaru_core::{
    molecular::transform::Model,
    quantum::{
        execution::{CircuitExecution, Request, Response},
        factorized_measurement::{
            basis_rotation, diagonal_observable, givens_decomposition, givens_generator,
            FactorizedMeasurement,
        },
        solver::Execution,
        Basis, CachedProblem, Operator, Space, C64,
    },
    Context,
};
use serde_json::Value;
use std::collections::BTreeMap;

fn context() -> Context {
    Context::new(Some(1), 256 << 20).unwrap()
}
fn frozen(name: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference/factorization.json");
    let text = std::fs::read_to_string(path).unwrap();
    let mut document: Value = serde_json::from_str(&text).unwrap();
    document["cases"][name].take()
}
fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|v| match v {
            Value::Array(_) => floats(v),
            other => vec![other.as_f64().unwrap()],
        })
        .collect()
}
/// `[[label, re, im], ...]` as a map.
fn terms(value: &Value) -> BTreeMap<String, (f64, f64)> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            (
                t[0].as_str().unwrap().to_string(),
                (t[1].as_f64().unwrap(), t[2].as_f64().unwrap()),
            )
        })
        .collect()
}
fn assert_terms(got: &BTreeMap<String, (f64, f64)>, want: &BTreeMap<String, (f64, f64)>, tol: f64) {
    for label in got.keys().chain(want.keys()) {
        let a = got.get(label).copied().unwrap_or_default();
        let b = want.get(label).copied().unwrap_or_default();
        assert!(
            (a.0 - b.0).abs() < tol && (a.1 - b.1).abs() < tol,
            "{label}: {a:?} against {b:?}"
        );
    }
}
fn model(ctx: &Context, case: &Value) -> Model {
    let m = case["n_spatial_orbitals"].as_u64().unwrap() as usize;
    let particles = case["state"]["occupied"].as_array().unwrap();
    let alpha = particles
        .iter()
        .filter(|q| (q.as_u64().unwrap() as usize) < m)
        .count();
    let fill = |values: Vec<f64>| {
        let mut buffer = ctx.allocate::<C64>(values.len()).unwrap();
        for (out, v) in buffer.as_mut_slice().iter_mut().zip(values) {
            *out = C64::new(v, 0.);
        }
        buffer
    };
    Model::new(
        ctx,
        m,
        [alpha, particles.len() - alpha],
        fill(floats(&case["one_body"])),
        fill(floats(&case["two_body"])),
        case["constant"].as_f64().unwrap(),
        false,
    )
    .unwrap()
}
fn operator_terms(operator: &Operator) -> BTreeMap<String, (f64, f64)> {
    operator
        .labels()
        .into_iter()
        .map(|(label, c)| (label, (c.re, c.im)))
        .collect()
}

#[test]
fn givens_generators_are_the_frozen_pairs_of_commuting_strings() {
    for name in ["H2", "LiH"] {
        let case = frozen(name);
        let n = case["n_modes"].as_u64().unwrap() as usize;
        for p in 0..n - 1 {
            let want = terms(&case["givens_generators"][p.to_string()]);
            let got: BTreeMap<_, _> = givens_generator(n, p)
                .unwrap()
                .into_iter()
                .map(|(label, im)| (label, (0., im)))
                .collect();
            assert_terms(&got, &want, 1e-12);
        }
    }
    assert!(givens_generator(4, 3).is_err());
}

#[test]
fn rotations_and_signs_are_the_frozen_decomposition_of_each_orbital_rotation() {
    for name in ["H2", "LiH"] {
        let case = frozen(name);
        let n = case["n_modes"].as_u64().unwrap() as usize;
        let reference = &case["factorization"];
        let mut keys = vec![(
            "one_body".to_string(),
            floats(&reference["one_body_orbitals"]),
        )];
        for (k, leaf) in reference["leaves"].as_array().unwrap().iter().enumerate() {
            keys.push((format!("leaf_{k}"), floats(&leaf["orbitals"])));
        }
        for (key, orbitals) in keys {
            let want = &case["basis_rotations"][&key];
            let (rotations, signs) = givens_decomposition(&orbitals, n, 1e-12).unwrap();
            let frozen_rotations = want["rotations"].as_array().unwrap();
            assert_eq!(rotations.len(), frozen_rotations.len(), "{name} {key}");
            for ((p, theta), frozen) in rotations.iter().zip(frozen_rotations) {
                assert_eq!(*p as u64, frozen[0].as_u64().unwrap());
                assert!((theta - frozen[1].as_f64().unwrap()).abs() < 1e-10);
            }
            for (s, f) in signs.iter().zip(floats(&want["signs"])) {
                assert!((s - f).abs() < 1e-10);
            }
            // The circuit is the elimination order as it comes: the frozen angles, twice each
            // (the two strings of a generator carry +-theta / 2).
            let circuit = basis_rotation(&orbitals, n).unwrap();
            assert_eq!(circuit.len(), 2 * rotations.len());
            for (k, (_p, theta)) in rotations.iter().enumerate() {
                assert!((circuit[2 * k].1 - 0.5 * theta).abs() < 1e-12);
                assert!((circuit[2 * k + 1].1 + 0.5 * theta).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn the_decomposition_refuses_a_matrix_that_is_not_orthogonal() {
    assert!(givens_decomposition(&[1., 1., 0., 1.], 2, 1e-12).is_err());
    assert!(givens_decomposition(&[1., 0., 0.], 2, 1e-12).is_err());
}

#[test]
fn diagonal_observables_are_the_squared_occupation_sums() {
    // (c . n)^2 on every occupation pattern, against the Z-string expansion.
    let c = [0.7, -0.4, 0.25];
    for (square, weight) in [(false, 1.3), (true, -0.6)] {
        let terms = diagonal_observable(&c, weight, square);
        for pattern in 0..8usize {
            let occupied: Vec<f64> = (0..3)
                .map(|k| f64::from(((pattern >> (2 - k)) & 1) as u8))
                .collect();
            let sum: f64 = c.iter().zip(&occupied).map(|(c, n)| c * n).sum();
            let want = weight * if square { sum * sum } else { sum };
            let got: f64 = terms
                .iter()
                .map(|(label, v)| {
                    v * label
                        .chars()
                        .zip(&occupied)
                        .map(|(ch, n)| if ch == 'Z' { 1. - 2. * n } else { 1. })
                        .product::<f64>()
                })
                .sum();
            assert!(
                (got - want).abs() < 1e-13,
                "{pattern} {square}: {got} against {want}"
            );
        }
        assert!(terms
            .iter()
            .all(|(label, _)| label.chars().all(|c| c == 'I' || c == 'Z')));
    }
}

#[test]
fn the_plan_of_a_model_is_the_frozen_measurement_problems() {
    for name in ["H2", "LiH"] {
        let ctx = context();
        let case = frozen(name);
        let plan = model(&ctx, &case).factorized_measurement(&ctx).unwrap();
        let reference = &case["factorization"];
        assert_eq!(
            plan.measurement_bases(),
            reference["measurement_bases"].as_u64().unwrap() as usize
        );
        assert!((plan.constant - reference["constant"].as_f64().unwrap()).abs() < 1e-12);
        assert!(
            (plan.truncation_error - reference["truncation_error"].as_f64().unwrap()).abs() < 1e-10
        );
        let problems = case["measurement_problems"].as_array().unwrap();
        let keys: Vec<String> = std::iter::once("one_body".to_string())
            .chain((0..plan.bases.len() - 1).map(|k| format!("leaf_{k}")))
            .collect();
        let state_generators = case["state"]["generators"].as_array().unwrap().len();
        let state_terms: usize = case["state"]["generators"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g.as_array().unwrap().len())
            .sum();
        assert!(state_terms >= state_generators);
        for ((basis, problem), key) in plan.bases.iter().zip(problems).zip(keys) {
            assert_terms(
                &operator_terms(&basis.observable),
                &terms(&problem["observable"]),
                1e-10,
            );
            // The rotation angles of the problem follow the state's, one per generator.
            let thetas = floats(&problem["thetas"]);
            let rotation_thetas = &thetas[case["state"]["thetas"].as_array().unwrap().len()..];
            let frozen_rotation = &case["basis_rotations"][&key];
            assert_eq!(
                rotation_thetas.len(),
                frozen_rotation["rotations"].as_array().unwrap().len()
            );
            assert_eq!(basis.rotation.len(), 2 * rotation_thetas.len());
            for (k, theta) in rotation_thetas.iter().enumerate() {
                assert!(
                    (basis.rotation[2 * k].1 - 0.5 * theta).abs() < 1e-10,
                    "{name} {key}"
                );
                assert!(
                    (basis.rotation[2 * k + 1].1 + 0.5 * theta).abs() < 1e-10,
                    "{name} {key}"
                );
            }
        }
    }
}

#[test]
fn a_spinor_model_and_complex_integrals_are_refused() {
    let ctx = context();
    let case = frozen("H2");
    let mut complex = model(&ctx, &case);
    complex.one.as_mut_slice()[0].im = 0.1;
    assert!(complex.factorized_measurement(&ctx).is_err());
    let mut asymmetric = model(&ctx, &case);
    asymmetric.two.as_mut_slice()[1].re += 0.3;
    assert!(asymmetric.factorized_measurement(&ctx).is_err());
}

/// What a device was asked: executor calls (one per measurement) and circuits (one per request).
#[derive(Default)]
struct Asked {
    calls: std::cell::Cell<usize>,
    circuits: std::cell::Cell<usize>,
}

/// A state-vector device: prepares `|reference>` and applies `exp(i angle P)` for each rotation,
/// once per request of a batch.
fn device<'a>(
    ctx: &'a Context,
    problem: &'a CachedProblem,
    asked: &'a Asked,
) -> impl FnMut(&[Request]) -> mandacaru_core::Result<Response> + 'a {
    let n = problem.hamiltonian.n;
    move |requests: &[Request]| {
        asked.calls.set(asked.calls.get() + 1);
        asked.circuits.set(asked.circuits.get() + requests.len());
        let basis = Basis::register(ctx, n)?;
        let mut states = Vec::new();
        for request in requests {
            let mut state = ctx.allocate::<C64>(basis.len())?;
            state.as_mut_slice()[problem.register_reference() as usize] = C64::new(1., 0.);
            for (label, angle) in request.labels.iter().zip(&request.angles) {
                let pauli = Operator::new(ctx, n, &[(label.clone(), C64::new(1., 0.))])?;
                let acted = pauli.apply(ctx, &basis, state.as_slice())?;
                let (cos, sin) = (angle.cos(), angle.sin());
                for (s, a) in state.as_mut_slice().iter_mut().zip(acted.as_slice()) {
                    *s = *s * cos + C64::new(0., sin) * a;
                }
            }
            states.push(state);
        }
        Ok(Response::States(states))
    }
}

/// `<observable_b>` of each basis on the frozen state, measured on the state-vector device.
fn basis_values(ctx: &Context, problem: &CachedProblem, case: &Value, drop_below: f64) -> Vec<f64> {
    let thetas = floats(&case["state"]["thetas"]);
    let mut labels = Vec::new();
    let mut angles = Vec::new();
    for (generator, theta) in case["state"]["generators"]
        .as_array()
        .unwrap()
        .iter()
        .zip(&thetas)
    {
        for (label, (_re, im)) in terms(generator) {
            labels.push(label);
            angles.push(theta * im);
        }
    }
    let asked = Asked::default();
    let mut call = device(ctx, problem, &asked);
    let basis = Basis::register(ctx, problem.hamiltonian.n).unwrap();
    problem
        .factorized
        .as_ref()
        .unwrap()
        .bases
        .iter()
        .map(|b| {
            let (mut labels, mut angles) = (labels.clone(), angles.clone());
            for (label, angle) in &b.rotation {
                labels.push(label.clone());
                angles.push(*angle);
            }
            let request = Request {
                labels,
                angles,
                observables: Vec::new(),
                priced: 0,
            };
            let Response::States(mut states) = call(&[request]).unwrap() else {
                panic!()
            };
            let state = states.remove(0);
            let kept: Vec<_> = b
                .observable
                .labels()
                .into_iter()
                .filter(|(label, c)| c.re.abs() > drop_below || label.chars().all(|ch| ch == 'I'))
                .collect();
            let observable = Operator::new(ctx, b.observable.n, &kept).unwrap();
            let acted = observable.apply(ctx, &basis, state.as_slice()).unwrap();
            state
                .as_slice()
                .iter()
                .zip(acted.as_slice())
                .map(|(s, a)| (s.conj() * a).re)
                .sum()
        })
        .collect()
}

fn problem_of(ctx: &Context, model: &Model, factorized: bool) -> CachedProblem {
    let mut problem = model
        .problem(ctx, "jordan_wigner", false, false, true)
        .unwrap();
    if factorized {
        problem
            .factorize(model.factorized_measurement(ctx).unwrap())
            .unwrap();
    }
    problem
}

#[test]
fn factorized_energy_equals_the_frozen_energies_and_the_commuting_path() {
    // Leaf truncation of the energy: 3e-14 for H2, 3e-12 for LiH (the frozen 1.65e-9 is the
    // estimator's, see below).
    for (name, truncation) in [("H2", 1e-12), ("LiH", 1e-10)] {
        let ctx = context();
        let case = frozen(name);
        let model = model(&ctx, &case);
        let thetas = floats(&case["state"]["thetas"]);
        let energies = &case["energies"];
        let qwc = problem_of(&ctx, &model, false);
        let plan = problem_of(&ctx, &model, true);
        // The frozen state is the first three pool generators.
        for (k, generator) in case["state"]["generators"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            assert_terms(
                &operator_terms(&plan.pool[k].operator),
                &terms(generator),
                1e-12,
            );
        }
        let indices = [0usize, 1, 2];

        let qwc_asked = Asked::default();
        let mut call = device(&ctx, &qwc, &qwc_asked);
        let mut execution = CircuitExecution::new(&mut call);
        let commuting = execution.evaluate(&qwc, &indices, &thetas, true).unwrap();
        assert!(
            (commuting.energy - energies["qwc"].as_f64().unwrap()).abs() < 1e-10,
            "{name}"
        );

        let plan_asked = Asked::default();
        let mut call = device(&ctx, &plan, &plan_asked);
        let mut execution = CircuitExecution::new(&mut call);
        let one_energy = execution
            .evaluate(&plan, &indices, &thetas, false)
            .unwrap()
            .energy;
        // One energy is one executor call carrying a circuit per basis.
        assert_eq!(plan_asked.calls.get(), 1);
        assert_eq!(
            plan_asked.circuits.get(),
            plan.factorized.as_ref().unwrap().measurement_bases()
        );
        // The frozen basis values came through Qiskit's `ObservablesArray`, which drops a
        // coefficient below 1e-8 (`ObservablesArray.coerce` of a 3e-9 term returns without it):
        // LiH's smallest leaf has Z terms of 1e-9 and 2e-9, which is the whole 1.65e-9 between
        // the frozen factorized and exact energies. The plan keeps them. Evidence: with the terms of 1e-8 or less dropped the plan reproduces
        // every frozen basis value to 1e-12, and with them kept it agrees with the exact energy
        // to 3e-12 (below).
        let per_basis = basis_values(&ctx, &plan, &case, 0.);
        let without_small_terms = basis_values(&ctx, &plan, &case, 1e-8);
        let frozen_values = floats(&energies["per_basis"]);
        for (b, frozen) in frozen_values.iter().enumerate() {
            assert!(
                (without_small_terms[b] - frozen).abs() < 1e-12,
                "{name} basis {b}: {} against {frozen}",
                without_small_terms[b]
            );
            assert!((per_basis[b] - frozen).abs() < 2e-9, "{name} basis {b}");
        }
        assert!(
            (one_energy
                - per_basis.iter().sum::<f64>()
                - plan.factorized.as_ref().unwrap().constant)
                .abs()
                < 1e-12
        );
        assert!(
            (one_energy - energies["exact"].as_f64().unwrap()).abs() < truncation,
            "{name}"
        );
        assert!((one_energy - commuting.energy).abs() < truncation, "{name}");

        // The parameter-shift gradient shifts the ansatz angles only, in the same measurement.
        let derived = execution.evaluate(&plan, &indices, &thetas, true).unwrap();
        for (a, b) in derived.gradient.iter().zip(&commuting.gradient) {
            assert!((a - b).abs() < 1e-6, "{name}: {a} against {b}");
        }
        assert!((derived.energy - one_energy).abs() < 1e-12);
    }
}

#[test]
fn a_register_free_problem_gets_expectations_per_basis_and_no_state() {
    let ctx = context();
    let case = frozen("LiH");
    let model = model(&ctx, &case);
    let mut plan = model
        .problem_in(
            &ctx,
            "jordan_wigner",
            Space::Pauli {
                number_conserving: true,
            },
            false,
            true,
        )
        .unwrap();
    plan.factorize(model.factorized_measurement(&ctx).unwrap())
        .unwrap();
    assert!(plan.is_register_free());
    let measured = plan.factorized.as_ref().unwrap();
    // Every basis observable is diagonal: the executor is asked for Z strings only.
    let mut asked = Vec::new();
    let mut calls = 0;
    let mut call = |requests: &[Request]| {
        calls += 1;
        for request in requests {
            asked.push((request.labels.len(), request.observables.clone()));
        }
        Ok(Response::Expectations(
            requests
                .iter()
                .map(|r| vec![0.25; r.observables.len()])
                .collect(),
        ))
    };
    let mut execution = CircuitExecution::new(&mut call);
    let energy = execution
        .evaluate(&plan, &[0], &[0.1], false)
        .unwrap()
        .energy;
    assert_eq!(asked.len(), measured.measurement_bases());
    assert_eq!(calls, 1, "the bases of one energy ride in one call");
    for ((rotation_and_ansatz, observables), basis) in asked.iter().zip(&measured.bases) {
        assert!(*rotation_and_ansatz >= basis.rotation.len());
        assert!(observables
            .iter()
            .all(|l| l.chars().all(|c| c == 'Z' || c == 'I')));
    }
    // A flat <Z...> = 0.25 landscape: the energy is the constant plus the weighted sums.
    let by_hand: f64 = measured.constant
        + measured
            .bases
            .iter()
            .map(|b| {
                b.observable
                    .labels()
                    .iter()
                    .map(|(label, c)| {
                        c.re * if label.chars().all(|c| c == 'I') {
                            1.
                        } else {
                            0.25
                        }
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();
    assert!((energy - by_hand).abs() < 1e-12);
}

#[test]
fn factorizing_a_tapered_or_non_jordan_wigner_problem_is_refused() {
    let ctx = context();
    let case = frozen("H2");
    let model = model(&ctx, &case);
    let plan = || model.factorized_measurement(&ctx).unwrap();
    let mut parity = model.problem(&ctx, "parity", false, false, true).unwrap();
    assert!(parity.factorize(plan()).is_err());
    let mut tapered = model
        .problem(&ctx, "jordan_wigner", false, false, true)
        .unwrap();
    tapered.taper().unwrap();
    assert!(tapered.factorize(plan()).is_err());
    let mut factorized = model
        .problem(&ctx, "jordan_wigner", false, false, true)
        .unwrap();
    factorized.factorize(plan()).unwrap();
    assert!(factorized.taper().is_err());
    // A plan of another register width.
    let two_orbitals = FactorizedMeasurement::from_integrals(&ctx, 1, &[0.3], &[0.1], 0.).unwrap();
    let mut wide = model
        .problem(&ctx, "jordan_wigner", false, false, true)
        .unwrap();
    assert!(wide.factorize(two_orbitals).is_err());
}

/// A real PAW leaf (`test/migration/reference/paw_factorization_leaf.json`) on which the bare
/// QR iteration reconstructs the matrix with a 1.1e-4 residual: the factorization's checked
/// solver decomposes it to rounding, so the measured leaf is the leaf of the Hamiltonian.
#[test]
fn a_paw_leaf_the_bare_qr_iteration_gets_wrong_is_decomposed_to_rounding() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference/paw_factorization_leaf.json");
    let case: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let rows = case["matrix"].as_array().unwrap();
    let n = rows.len();
    let matrix = nalgebra::DMatrix::from_fn(n, n, |i, j| rows[i][j].as_f64().unwrap());
    let residual = |values: &nalgebra::DVector<f64>, vectors: &nalgebra::DMatrix<f64>| {
        (vectors * nalgebra::DMatrix::from_diagonal(values) * vectors.transpose() - &matrix)
            .abs()
            .max()
    };
    // The failure this guards against is still in the bare solver.
    let bare = nalgebra::SymmetricEigen::new(matrix.clone());
    assert!(residual(&bare.eigenvalues, &bare.eigenvectors) > 1e-6);
    let (values, vectors) =
        mandacaru_core::math::linalg::checked_symmetric(matrix.clone()).unwrap();
    let error = residual(&values, &vectors);
    assert!(error < 1e-15, "{error:.3e}");
    // And the factorization itself: this leaf as the one Cholesky vector of a model, whose
    // double factorization must rebuild the tensor <pq|rs> = L_pr L_qs.
    let factors: Vec<f64> = (0..n * n).map(|k| matrix[(k / n, k % n)]).collect();
    let h = vec![0.; n * n];
    let data = mandacaru_core::quantum::factorization::spatial_double_factorization(
        &h, n, None, None, Some((&factors, 1, n)), None, 0., 0., None, 0.,
    )
    .unwrap();
    let mut worst = 0f64;
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    let rebuilt: f64 = data
                        .leaves
                        .iter()
                        .map(|leaf| {
                            let m = |a: usize, b: usize| {
                                (0..n)
                                    .map(|k| {
                                        leaf.orbitals[(a, k)]
                                            * leaf.coefficients[k]
                                            * leaf.orbitals[(b, k)]
                                    })
                                    .sum::<f64>()
                            };
                            leaf.weight * m(p, r) * m(q, s)
                        })
                        .sum();
                    worst = worst.max((rebuilt - matrix[(p, r)] * matrix[(q, s)]).abs());
                }
            }
        }
    }
    assert!(worst < 1e-14, "{worst:.3e}");
}
