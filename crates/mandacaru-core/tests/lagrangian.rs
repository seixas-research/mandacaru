//! The parameter-space half of the variational force correction (`quantum::lagrangian`): the
//! exact Hessian of a product-form ansatz, the Z vector it gives and the exact tangent state;
//! on a device, the Z vector of the shift-rule gradient and Hessian with its covariance from the
//! energies' shot noise (`measured_z`), checked against the scatter of repeated sampled runs, and
//! the refusal of a curvature that shot noise swamps.
use mandacaru_core::{
    math::{symmetric_solve, SpatialGrid},
    molecular::MolecularIntegrals,
    quantum::{
        lagrangian::{Ansatz, Options, Residual},
        CachedProblem, SolveOptions,
    },
    Context, Geometry, LengthUnit,
};
use nalgebra::DMatrix;

/// H2 on a coarse all-electron grid, with its ADAPT pool.
fn h2(context: &Context) -> CachedProblem {
    let geometry = Geometry::new(
        vec![1, 1],
        vec![[4., 4., 3.3], [4., 4., 4.7]],
        [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
        [false; 3],
        LengthUnit::Bohr,
    )
    .unwrap();
    let grid = SpatialGrid::from_geometry(&geometry, [0.5; 3], false, false).unwrap();
    let integrals = MolecularIntegrals::new(
        context,
        geometry,
        grid,
        r#"{"name":"HAO"}"#,
        "spectral",
        Some(1e-12),
        Default::default(),
    )
    .unwrap();
    let reference = integrals
        .mean_field("rhf", [1, 1], &Default::default())
        .unwrap();
    reference
        .model(&integrals)
        .unwrap()
        .problem(context, "jordan_wigner", true, false, true)
        .unwrap()
}

/// A product-form ansatz of every pool generator twice, off any optimum.
fn sequence(problem: &CachedProblem) -> (Vec<usize>, Vec<f64>) {
    let n = problem.pool.len();
    assert!(n >= 2, "the fixture needs at least two generators");
    let indices: Vec<usize> = (0..2 * n).map(|k| k % n).collect();
    let parameters = (0..2 * n).map(|k| 0.3 * ((k + 1) as f64).sin()).collect();
    (indices, parameters)
}

#[test]
fn the_exact_hessian_is_the_derivative_of_the_analytic_gradient() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let problem = h2(&context);
        let (indices, parameters) = sequence(&problem);
        let ansatz = Ansatz {
            indices: &indices,
            parameters: &parameters,
            product_form: true,
        };
        let n = parameters.len();
        let (hessian, gradient_evaluations, actions) = problem
            .parameter_hessian(&ansatz, &Options::default())
            .unwrap();
        assert_eq!((gradient_evaluations, actions), (0, n + 1));
        let h = 1e-4;
        for column in 0..n {
            let gradient = |sign: f64| {
                let mut theta = parameters.clone();
                theta[column] += sign * h;
                problem.evaluate(&indices, &theta, true).unwrap().gradient
            };
            let (up, down) = (gradient(1.), gradient(-1.));
            for row in 0..n {
                let difference = (up[row] - down[row]) / (2. * h);
                assert!(
                    (hessian[(row, column)] - difference).abs() < 1e-7,
                    "({row}, {column}): {} vs {difference}",
                    hessian[(row, column)]
                );
            }
        }
        // Exact to roundoff: the measured antisymmetry.
        assert!((&hessian - hessian.transpose()).norm() < 1e-12);
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn the_exact_z_vector_matches_the_differenced_hessian_and_its_tangent_gives_z_dot_g() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let problem = h2(&context);
        let solved = problem
            .solve(
                &SolveOptions {
                    adapt: true,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        let parameters: Vec<f64> = solved.parameters.iter().map(|t| t + 0.01).collect();
        let indices = solved.indices.clone();
        let ansatz = Ansatz {
            indices: &indices,
            parameters: &parameters,
            product_form: true,
        };
        let state = problem.prepare_state(&indices, &parameters, true).unwrap();
        let options = Options {
            floor: 0.,
            ..Default::default()
        };
        let Residual::Corrected { z, .. } = problem
            .lagrangian(&ansatz, state.complex128().unwrap(), &options)
            .unwrap()
        else {
            panic!("an off-optimum state is corrected");
        };
        // The truncated solve of the Hessian differenced from analytic gradients, as before.
        let n = parameters.len();
        let mut hessian = DMatrix::<f64>::zeros(n, n);
        for column in 0..n {
            let gradient = |sign: f64| {
                let mut theta = parameters.clone();
                theta[column] += sign * 1e-4;
                problem.evaluate(&indices, &theta, true).unwrap().gradient
            };
            let (up, down) = (gradient(1.), gradient(-1.));
            for row in 0..n {
                hessian[(row, column)] = (up[row] - down[row]) / 2e-4;
            }
        }
        let g = problem
            .evaluate(&indices, &parameters, true)
            .unwrap()
            .gradient;
        let differenced = symmetric_solve::solve(&hessian, &g, &options.solver).unwrap();
        let scale = differenced.x.iter().map(|v| v.abs()).fold(0., f64::max);
        for (a, b) in z.z.iter().zip(&differenced.x) {
            assert!((a - b).abs() < 1e-6 * scale, "{a} vs {b}");
        }
        assert!(z.solve.noise < 1e-12 && z.solve.exact && z.solve.gradient_evaluations == 0);
        // psi ± s psi' reproduce z . g exactly through the energy (a quadratic form).
        let energy = |s: &mandacaru_core::quantum::State| {
            let psi = s.complex128().unwrap();
            let basis = problem.register("test").unwrap();
            let action = problem.hamiltonian.apply(&context, basis, psi).unwrap();
            mandacaru_core::quantum::evolution::inner(psi, action.as_slice()).re
        };
        let through_states = (energy(&z.plus) - energy(&z.minus)) / (2. * z.scale);
        assert!(
            (through_states - z.z_dot_g).abs() < 1e-12 + 1e-9 * z.z_dot_g.abs(),
            "{through_states:.6e} vs {:.6e}",
            z.z_dot_g
        );
    }
    assert_eq!(context.memory_used(), 0);
}

/// The register-free H2 of the migration fixture (4 qubits, singles and doubles pool).
fn h2_register_free(context: &Context) -> CachedProblem {
    use mandacaru_core::persistence::{Document, HAMILTONIAN};
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference/h2_hamiltonian.json");
    let problem = Document::read(context, &fixture, HAMILTONIAN)
        .unwrap()
        .problem_in(
            context,
            mandacaru_core::quantum::Space::Pauli {
                number_conserving: true,
            },
            false,
            true,
        )
        .unwrap();
    assert!(problem.is_register_free() && problem.pool.len() >= 3);
    problem
}

/// The state after a request's circuit (its rotations applied to the reference on the full
/// register), its probabilities and the `Z` masks of its observables (`None` for a string that
/// is not diagonal), and the exact expectation of every observable.
#[allow(clippy::type_complexity)]
fn prepared(
    context: &Context,
    n: usize,
    reference: u64,
    request: &mandacaru_core::quantum::execution::Request,
) -> (Vec<f64>, Vec<u64>, Vec<u64>, Vec<f64>) {
    use mandacaru_core::quantum::{Basis, Operator, C64};
    let basis = Basis::register(context, n).unwrap();
    let mut state = vec![C64::default(); basis.len()];
    state[basis.index(reference).unwrap()] = C64::new(1., 0.);
    for (label, angle) in request.labels.iter().zip(&request.angles) {
        let p = Operator::new(context, n, &[(label.clone(), C64::new(1., 0.))]).unwrap();
        let action = p.apply(context, &basis, &state).unwrap();
        state = state
            .iter()
            .zip(action.as_slice())
            .map(|(a, b)| a * angle.cos() + C64::new(0., angle.sin()) * b)
            .collect();
    }
    let exact = request
        .observables
        .iter()
        .map(|label| {
            let p = Operator::new(context, n, &[(label.clone(), C64::new(1., 0.))]).unwrap();
            let action = p.apply(context, &basis, &state).unwrap();
            state.iter().zip(action.as_slice()).map(|(a, b)| (a.conj() * b).re).sum()
        })
        .collect();
    let masks = request
        .observables
        .iter()
        .map(|label| {
            label
                .bytes()
                .enumerate()
                .filter(|(_, c)| *c == b'Z')
                .map(|(j, _)| 1u64 << (n - 1 - j))
                .sum()
        })
        .collect();
    let kets = basis.indices.as_slice().to_vec();
    (state.iter().map(|a| a.norm_sqr()).collect(), kets, masks, exact)
}

fn parity(ket: u64, mask: u64) -> f64 {
    if (ket & mask).count_ones() % 2 == 0 { 1. } else { -1. }
}

/// The device's answer to `requests`: `shots` sampled bit strings per request (`Some`), or the
/// exact expectations (`None`).
fn answer(
    context: &Context,
    problem: &CachedProblem,
    requests: &[mandacaru_core::quantum::execution::Request],
    shots: Option<u64>,
    seed: &mut u64,
) -> Vec<Vec<f64>> {
    let (n, reference) = (problem.hamiltonian.n, problem.register_reference());
    let mut uniform = || {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 11) as f64 / (1u64 << 53) as f64
    };
    requests
        .iter()
        .map(|request| {
            let (probabilities, kets, masks, exact) = prepared(context, n, reference, request);
            match shots {
                None => exact,
                Some(shots) => {
                    assert!(
                        request.observables.iter().all(|l| l.chars().all(|c| c == 'I' || c == 'Z')),
                        "a sampled request reads Z strings"
                    );
                    let cumulative: Vec<f64> = probabilities
                        .iter()
                        .scan(0., |s, p| {
                            *s += p;
                            Some(*s)
                        })
                        .collect();
                    let mut sums = vec![0f64; masks.len()];
                    for _ in 0..shots {
                        let u = uniform() * cumulative[cumulative.len() - 1];
                        let k = cumulative.partition_point(|&c| c <= u).min(kets.len() - 1);
                        for (sum, &mask) in sums.iter_mut().zip(&masks) {
                            *sum += parity(kets[k], mask);
                        }
                    }
                    sums.into_iter().map(|s| s / shots as f64).collect()
                }
            }
        })
        .collect()
}

/// The device gradient, Hessian and Z vector at `parameters`, with `declared` shots per basis
/// and the readings sampled with that many shots (`sampled`) or exact.
fn device_z(
    context: &Context,
    problem: &CachedProblem,
    indices: &[usize],
    parameters: &[f64],
    declared: u64,
    sampled: bool,
    seed: &mut u64,
) -> mandacaru_core::Result<(mandacaru_core::quantum::execution::DeviceHessian, mandacaru_core::quantum::lagrangian::MeasuredZ)> {
    use mandacaru_core::quantum::execution::{CircuitExecution, Request, Response};
    let mut call = |requests: &[Request]| -> mandacaru_core::Result<Response> {
        Ok(Response::Expectations(answer(context, problem, requests, sampled.then_some(declared), seed)))
    };
    let mut execution = CircuitExecution::new(&mut call).with_shots(declared);
    let measured = execution.gradient_and_hessian(problem, indices, parameters)?;
    let options = Options { floor: 0., ..Default::default() };
    let solved = mandacaru_core::quantum::lagrangian::measured_z(
        &measured.gradient,
        &measured.hessian,
        &measured.energies,
        &options,
    )?;
    Ok((measured, solved))
}

#[test]
fn a_sampled_device_z_vector_scatters_as_its_propagated_covariance() {
    let context = Context::new(Some(1), 256 << 20).unwrap();
    {
        let problem = h2_register_free(&context);
        let (indices, parameters) = (vec![0usize, 1, 2], vec![0.05, -0.03, 0.08]);
        let (shots, runs) = (500u64, 160usize);
        let mut seed = 11u64;
        let (mut zs, mut covariances) = (Vec::new(), Vec::new());
        let mut significance = f64::INFINITY;
        for _ in 0..runs {
            let (measured, solved) =
                device_z(&context, &problem, &indices, &parameters, shots, true, &mut seed).unwrap();
            assert!(measured.energies.iter().all(|e| e.error > 0.));
            let solution = solved.solution.as_ref().unwrap();
            for (v, e) in solution.eigenvalues.iter().zip(&solved.curvature_errors) {
                significance = significance.min(v.abs() / e);
            }
            zs.push(solution.x.clone());
            covariances.push(solved.z_covariance.unwrap());
        }
        let n = parameters.len();
        let mean: Vec<f64> = (0..n).map(|k| zs.iter().map(|z| z[k]).sum::<f64>() / runs as f64).collect();
        let sample = DMatrix::from_fn(n, n, |k, l| {
            zs.iter().map(|z| (z[k] - mean[k]) * (z[l] - mean[l])).sum::<f64>() / (runs - 1) as f64
        });
        let propagated = covariances.iter().fold(DMatrix::zeros(n, n), |a, c| a + c) / runs as f64;
        eprintln!("smallest |lambda| / sigma {significance:.1}\nsample {sample:.3e}\npropagated {propagated:.3e}");
        // 160 runs estimate a standard deviation to about 6 percent: a 20 percent band is over
        // three of its own errors.
        for k in 0..n {
            let ratio = (sample[(k, k)] / propagated[(k, k)]).sqrt();
            assert!((ratio - 1.).abs() < 0.2, "z[{k}]: observed / propagated {ratio:.3}");
        }
        // The correlations too: the variance of a mixed direction.
        let u = nalgebra::DVector::from_vec(vec![1., -1., 1.]);
        let ratio = ((u.transpose() * &sample * &u)[0] / (u.transpose() * &propagated * &u)[0]).sqrt();
        assert!((ratio - 1.).abs() < 0.2, "mixed direction: observed / propagated {ratio:.3}");
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn a_curvature_within_five_of_its_shot_errors_is_refused_by_name() {
    let context = Context::new(Some(1), 256 << 20).unwrap();
    {
        let problem = h2_register_free(&context);
        let mut seed = 0u64;
        // The double excitation alone, exact readings: its curvature changes sign between the
        // Hartree-Fock point and a quarter turn; bisect to just inside the convex side.
        let double = (0..problem.pool.len())
            .find(|&k| {
                let (m, _) = device_z(&context, &problem, &[k], &[0.], 0, false, &mut seed).unwrap();
                m.gradient[0].abs() > 1e-6
            })
            .expect("a pool generator with a gradient at the reference");
        let curvature = |theta: f64, seed: &mut u64| {
            use mandacaru_core::quantum::execution::{CircuitExecution, Request, Response};
            let mut call = |requests: &[Request]| -> mandacaru_core::Result<Response> {
                Ok(Response::Expectations(answer(&context, &problem, requests, None, seed)))
            };
            let mut execution = CircuitExecution::new(&mut call);
            let m = execution.gradient_and_hessian(&problem, &[double], &[theta]).unwrap();
            m.hessian[(0, 0)]
        };
        let at_reference = curvature(0., &mut seed);
        assert!(at_reference > 0.);
        let direction = if curvature(0.05, &mut seed) < at_reference { 1. } else { -1. };
        let (mut convex, mut concave) = (0., 0.);
        for step in 1..40 {
            let theta = direction * 0.05 * step as f64;
            if curvature(theta, &mut seed) < 0. {
                concave = theta;
                break;
            }
            convex = theta;
        }
        assert!(concave != 0., "no inflection within the scan");
        while curvature(convex, &mut seed) > 1e-3 * at_reference {
            let middle = 0.5 * (convex + concave);
            if curvature(middle, &mut seed) > 0. { convex = middle } else { concave = middle }
        }
        let small = curvature(convex, &mut seed);
        assert!(small > 0. && small < 1e-3 * at_reference);
        // Exact readings with a declared 2000 shots per basis: the errors are the exact ones of
        // that many shots, and the curvature is deterministic.
        let refused = device_z(&context, &problem, &[double], &[convex], 2000, false, &mut seed)
            .err()
            .expect("a curvature under five sigma refuses")
            .to_string();
        eprintln!("{refused}");
        assert!(refused.contains("curvature") && refused.contains("standard error") && refused.contains("Raise shots"));
        // The same shots at the Hartree-Fock point, where the curvature is large, pass.
        let (_, solved) = device_z(&context, &problem, &[double], &[0.], 2000, false, &mut seed).unwrap();
        let solution = solved.solution.unwrap();
        let ratio = solution.eigenvalues[0] / solved.curvature_errors[0];
        assert!(ratio > mandacaru_core::quantum::lagrangian::CURVATURE_SIGNIFICANCE, "{ratio}");
    }
    assert_eq!(context.memory_used(), 0);
}

/// A measurement basis past the covariance-product limit (`energy_error::MAX_PRODUCTS`): 40 qubits,
/// every `Z_i` and `Z_i Z_j`, one diagonal basis of 820 strings. The Hessian batch is still
/// measured and solved (owner, 2026-11-10: "Run, error None"), its energies carry no error and the
/// reason, `measured_z` neither checks the curvature nor forms a covariance, and no product string
/// is priced. The device answers every string with a value set by the first rotation's angle, so
/// the energy has a gradient and a positive curvature there.
#[test]
fn a_basis_past_the_product_limit_measures_the_hessian_without_an_error() {
    use mandacaru_core::persistence::{Document, HAMILTONIAN};
    use mandacaru_core::quantum::execution::{CircuitExecution, Request, Response};
    let context = Context::new(Some(1), 1 << 30).unwrap();
    {
        let n = 40;
        let mut terms = Vec::new();
        for i in 0..n {
            for j in i..n {
                let label: String = (0..n).map(|k| if k == i || k == j { 'Z' } else { 'I' }).collect();
                terms.push(serde_json::json!([label, 0.01 * (i + j + 1) as f64, 0.0]));
            }
        }
        let payload = serde_json::json!({
            "format": "mandacaru-qubit-hamiltonian", "version": 1, "num_qubits": n,
            "mapping": "jordan_wigner", "num_particles": [1, 1], "n_spatial_orbitals": n / 2,
            "metadata": {"energy_unit": "Ha"}, "terms": terms,
        });
        let problem = Document::from_value(&context, payload, HAMILTONIAN)
            .unwrap()
            .problem_in(
                &context,
                mandacaru_core::quantum::Space::Pauli { number_conserving: true },
                false,
                true,
            )
            .unwrap();
        assert!(problem.is_register_free());
        let mut call = |requests: &[Request]| -> mandacaru_core::Result<Response> {
            Ok(Response::Expectations(
                requests
                    .iter()
                    .map(|r| {
                        assert_eq!(r.priced, r.observables.len(), "no product string is measured");
                        vec![-0.2 * r.angles[0].cos(); r.observables.len()]
                    })
                    .collect(),
            ))
        };
        let mut execution = CircuitExecution::new(&mut call).with_shots(1000);
        let measured = execution.gradient_and_hessian(&problem, &[0], &[0.3]).unwrap();
        let reason = measured.error_unavailable.as_deref().expect("the reason is reported");
        assert!(reason.contains("not reported") && reason.contains("20000"), "{reason}");
        assert!(measured.energies.iter().all(|e| e.error == 0.));
        let options = Options { floor: 0., ..Default::default() };
        let solved = mandacaru_core::quantum::lagrangian::measured_z(
            &measured.gradient,
            &measured.hessian,
            &measured.energies,
            &options,
        )
        .unwrap();
        assert!(solved.solution.is_some() && solved.z_covariance.is_none());
        assert!(solved.curvature_errors.iter().all(|e| *e == 0.));
    }
    assert_eq!(context.memory_used(), 0);
}
