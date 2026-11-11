//! Mid-run checkpoints (`SolveOptions::checkpoint`): ADAPT writes one after every `every` accepted
//! growth steps, a fixed ansatz the best point so far after every `every` cost evaluations.
mod solver_support;

use mandacaru_core::{
    quantum::{solver::Checkpointing, SolveOptions},
    Context,
};
use serde_json::Value;

fn read(path: &std::path::Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn plan(path: &std::path::Path, every: usize, method: &str) -> Option<Checkpointing> {
    Some(Checkpointing {
        path: path.to_path_buf(),
        every,
        method: method.into(),
    })
}

#[test]
fn adapt_writes_after_every_second_growth_step_what_that_step_reached() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    let problem = solver_support::problem(&ctx, "fermionic");
    let dir = std::env::temp_dir().join(format!("mandacaru-checkpoint-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("adapt.json");
    let options = SolveOptions {
        adapt: true,
        max_growth: 6,
        gradient_tolerance: Some(1e-6),
        checkpoint: plan(&path, 2, "adapt-vqe"),
        ..Default::default()
    };
    let mut seen: Vec<(usize, Vec<f64>, f64, Option<Value>)> = Vec::new();
    let mut observer = |indices: &[usize], parameters: &[f64], energy: f64| {
        // The checkpoint is written before the observer sees the step.
        let file = path.exists().then(|| read(&path));
        seen.push((indices.len(), parameters.to_vec(), energy, file));
        Ok(())
    };
    let result = problem
        .solve_observed(&options, None, None, &mut observer)
        .unwrap();
    assert!(seen.len() >= 4);
    for (step, (count, parameters, energy, file)) in seen.iter().enumerate() {
        let step = step + 1;
        if step == 1 {
            assert!(file.is_none(), "nothing is written before the second step");
            continue;
        }
        let file = file.as_ref().expect("a checkpoint exists from step 2 on");
        let status = &file["status"];
        // The file holds the last even step reached.
        assert_eq!(
            status["iteration"].as_u64().unwrap() as usize,
            step - step % 2,
            "step {step}"
        );
        assert_eq!(status["complete"], Value::Bool(false));
        if step % 2 == 0 {
            let operators = file["operators"].as_array().unwrap();
            assert_eq!(operators.len(), *count);
            for (operator, theta) in operators.iter().zip(parameters) {
                assert_eq!(operator["parameter"].as_f64().unwrap(), *theta);
            }
            assert_eq!(file["energy"].as_f64().unwrap(), *energy);
            assert_eq!(file["preparation"], "product");
        }
    }
    drop(result);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_fixed_ansatz_writes_its_best_point_every_few_evaluations() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    let problem = solver_support::problem(&ctx, "fermionic");
    let dir = std::env::temp_dir().join(format!("mandacaru-checkpoint-vqe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vqe.json");
    let options = SolveOptions {
        checkpoint: plan(&path, 5, "vqe"),
        ..Default::default()
    };
    let result = problem.solve(&options, None).unwrap();
    let file = read(&path);
    let evaluations = file["status"]["num_evaluations"].as_u64().unwrap() as usize;
    assert_eq!(evaluations % 5, 0);
    assert!(evaluations >= 5 && evaluations <= result.history.len());
    // The best of the first `evaluations` energies, at its own parameters.
    let best = result.history[..evaluations]
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert_eq!(file["energy"].as_f64().unwrap(), best);
    let parameters: Vec<f64> = file["operators"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["parameter"].as_f64().unwrap())
        .collect();
    let value = problem
        .evaluate(&result.indices, &parameters, false)
        .unwrap()
        .energy;
    assert!((value - best).abs() < 1e-12);
    assert_eq!(file["preparation"], "sum");
    // A zero cadence or another method is refused.
    let wrong = SolveOptions {
        checkpoint: plan(&path, 0, "vqe"),
        ..Default::default()
    };
    assert!(problem.solve(&wrong, None).is_err());
    drop(result);
    std::fs::remove_dir_all(&dir).unwrap();
}

fn subspace_options(adapt: bool, max_growth: usize) -> SolveOptions {
    SolveOptions {
        adapt,
        max_growth,
        gradient_tolerance: Some(1e-6),
        ..Default::default()
    }
}

#[test]
fn a_subspace_adapt_run_resumed_from_its_checkpoint_grows_as_if_never_stopped() {
    use mandacaru_core::persistence::{self, Document, CHECKPOINT};
    use mandacaru_core::quantum::optimizer::Method;
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    let problem = solver_support::problem(&ctx, "fermionic");
    let weights = [2., 1.];
    let dir = std::env::temp_dir().join(format!("mandacaru-subspace-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("subspace.json");
    // SLSQP and BFGS store their curvature estimate, and the resumed run is the uninterrupted
    // one, bit for bit. L-BFGS carries none: the stored sequence is relaxed and the growth
    // follows to the same operators and energies.
    for method in [Method::Lbfgs, Method::Bfgs, Method::Slsqp] {
        let options = |max_growth| {
            let mut options = subspace_options(true, max_growth);
            options.optimizer.method = method;
            options
        };
        let straight = problem.solve_subspace(&options(6), &weights, None).unwrap();
        let partial = problem.solve_subspace(&options(2), &weights, None).unwrap();
        let carries = method != Method::Lbfgs;
        assert_eq!(partial.hessian.is_some(), carries);
        let status = persistence::status_with_hessian(
            serde_json::json!({"complete": true, "converged": false}),
            method.name(),
            partial.hessian.as_deref(),
        )
        .unwrap();
        persistence::subspace_checkpoint(
            &ctx,
            &problem,
            &partial.indices,
            &partial.parameters,
            true,
            true,
            &weights,
            &partial.energies,
            status,
        )
        .unwrap()
        .save(&path)
        .unwrap();
        let file = read(&path);
        assert_eq!(file["method"], "subspace-adapt-vqe");
        assert_eq!(file["subspace"]["weights"], serde_json::json!(weights));
        assert_eq!(file["status"].get("hessian").is_some(), carries);
        let doc = Document::read(&ctx, &path, CHECKPOINT).unwrap();
        let resumed = doc.resume_subspace(&problem, &options(6), &weights).unwrap();
        eprintln!(
            "{method:?}: straight {:?} {:?}; resumed {:?} {:?}",
            straight.indices, straight.energies, resumed.indices, resumed.energies
        );
        assert!(resumed.notes.is_empty());
        assert_eq!(resumed.indices, straight.indices);
        if carries {
            assert_eq!(resumed.energies, straight.energies);
            assert_eq!(resumed.parameters, straight.parameters);
            assert_eq!(resumed.hessian, straight.hessian);
        } else {
            for (a, b) in resumed.energies.iter().zip(&straight.energies) {
                assert!((a - b).abs() < 1e-8);
            }
        }
        // Other weights, or the fixed-ansatz method, are refused.
        assert!(doc.resume_subspace(&problem, &options(6), &[3., 1.]).is_err());
        let mut fixed = options(6);
        fixed.adapt = false;
        assert!(doc.resume_subspace(&problem, &fixed, &weights).is_err());
        // A version-2 file has no Hessian: SLSQP and BFGS resume from the identity and say so.
        if carries {
            let mut old = file.clone();
            old["status"].as_object_mut().unwrap().remove("hessian");
            old["version"] = 2.into();
            let old_path = dir.join("old.json");
            std::fs::write(&old_path, old.to_string()).unwrap();
            let old = Document::read(&ctx, &old_path, CHECKPOINT).unwrap();
            let cold = old.resume_subspace(&problem, &options(6), &weights).unwrap();
            assert_eq!(cold.notes.len(), 1);
            assert_eq!(cold.indices, straight.indices);
        }
    }
    drop(problem);
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_subspace_vqe_resumed_at_its_optimum_stays_there() {
    use mandacaru_core::persistence::{self, Document, CHECKPOINT};
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    let problem = solver_support::problem(&ctx, "fermionic");
    let weights = [2., 1.];
    let dir = std::env::temp_dir().join(format!("mandacaru-subspace-vqe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("subspace.json");
    let first = problem
        .solve_subspace(&subspace_options(false, 1), &weights, None)
        .unwrap();
    persistence::subspace_checkpoint(
        &ctx,
        &problem,
        &first.indices,
        &first.parameters,
        false,
        false,
        &weights,
        &first.energies,
        serde_json::json!({"complete": true, "converged": first.success}),
    )
    .unwrap()
    .save(&path)
    .unwrap();
    let doc = Document::read(&ctx, &path, CHECKPOINT).unwrap();
    let resumed = doc
        .resume_subspace(&problem, &subspace_options(false, 1), &weights)
        .unwrap();
    for (a, b) in resumed.energies.iter().zip(&first.energies) {
        assert!((a - b).abs() < 1e-10);
    }
    assert!(resumed.history.len() < first.history.len());
    drop((first, resumed, doc, problem));
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(ctx.memory_used(), 0);
}
