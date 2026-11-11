//! The curvature estimate of SLSQP (a Hessian approximation) and of BFGS (an inverse Hessian
//! approximation) across ADAPT growth: the result carries it, a checkpoint stores it with the
//! optimizer that kept it, a resumed run starts from it, and every other optimizer or growth
//! mode carries none.
mod solver_support;

use mandacaru_core::{
    persistence::{self, Document, CHECKPOINT},
    quantum::{optimizer::Method, solver::Checkpointing, SolveOptions},
    Context,
};
use serde_json::Value;

const CARRIERS: [Method; 2] = [Method::Slsqp, Method::Bfgs];

fn options(method: Method, max_growth: usize) -> SolveOptions {
    let mut options = SolveOptions {
        adapt: true,
        max_growth,
        gradient_tolerance: Some(1e-6),
        ..Default::default()
    };
    options.optimizer.method = method;
    options.optimizer.tolerance = Some(1e-12);
    options
}

fn read(path: &std::path::Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn slsqp_and_bfgs_growth_return_a_symmetric_estimate_of_the_final_ansatz_and_nothing_else_does() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    let problem = solver_support::problem(&ctx, "fermionic");
    for method in CARRIERS {
        let result = problem.solve(&options(method, 4), None).unwrap();
        let n = result.indices.len();
        assert!(n >= 3);
        let hessian = result.hessian.as_ref().expect("the growth keeps a curvature estimate");
        assert_eq!(hessian.len(), n * n);
        for i in 0..n {
            for j in 0..i {
                // SLSQP's update is exactly symmetric, BFGS's only to roundoff.
                assert!((hessian[i * n + j] - hessian[j * n + i]).abs() < 1e-12, "{method:?}");
            }
        }
        // A sweep that holds the earlier parameters and a fixed ansatz carry none.
        let sweep = SolveOptions { quenching: false, ..options(method, 4) };
        assert!(problem.solve(&sweep, None).unwrap().hessian.is_none());
        let fixed = SolveOptions { adapt: false, ..options(method, 4) };
        assert!(problem.solve(&fixed, None).unwrap().hessian.is_none());
        // Pruning drops the removed operator's parameter from the estimate with it.
        let pruned = SolveOptions { prune: true, ..options(method, 8) };
        let result = problem.solve(&pruned, None).unwrap();
        let n = result.indices.len();
        assert_eq!(result.hessian.as_ref().map(Vec::len), Some(n * n));
    }
    // L-BFGS keeps correction pairs, not a dense estimate: it starts every step cold.
    assert!(problem.solve(&options(Method::Lbfgs, 4), None).unwrap().hessian.is_none());
}

#[test]
fn a_carried_estimate_is_refused_unless_it_fits_a_carried_slsqp_or_bfgs_sequence() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    let problem = solver_support::problem(&ctx, "fermionic");
    for method in CARRIERS {
        let grown = problem.solve(&options(method, 3), None).unwrap();
        let n = grown.indices.len();
        let carried = |method, hessian: Vec<f64>| SolveOptions { hessian: Some(hessian), ..options(method, 6) };
        let hessian = grown.hessian.clone().unwrap();
        let relaxed = problem
            .solve_sequence(&carried(method, hessian.clone()), Some(&grown.parameters), Some(&grown.indices))
            .unwrap();
        assert!(relaxed.seed.as_ref().unwrap().warm);
        assert!(relaxed.hessian.is_some());
        let cold = problem
            .solve_sequence(&options(method, 6), Some(&grown.parameters), Some(&grown.indices))
            .unwrap();
        assert!(!cold.seed.as_ref().unwrap().warm);
        // A fresh run, a method that keeps none and a wrong size all refuse it.
        assert!(problem.solve(&carried(method, hessian.clone()), None).is_err());
        assert!(problem
            .solve_sequence(&carried(Method::Lbfgs, hessian.clone()), Some(&grown.parameters), Some(&grown.indices))
            .is_err());
        assert!(problem
            .solve_sequence(&carried(method, hessian[..n * n - 1].to_vec()), Some(&grown.parameters), Some(&grown.indices))
            .is_err());
    }
}

#[test]
fn a_resumed_adapt_run_grows_from_the_stored_estimate_as_if_never_stopped() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    let problem = solver_support::problem(&ctx, "fermionic");
    let dir = std::env::temp_dir().join(format!("mandacaru-hessian-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("adapt.json");
    for method in CARRIERS {
        let name = method.name();
        let straight = problem.solve(&options(method, 6), None).unwrap();
        let partial_options = SolveOptions {
            checkpoint: Some(Checkpointing { path: path.clone(), every: 1, method: "adapt-vqe".into() }),
            ..options(method, 3)
        };
        let partial = problem.solve(&partial_options, None).unwrap();
        // The file written after the third growth step stores that step's estimate and the
        // optimizer that kept it, schema version 3.
        let file = read(&path);
        assert_eq!(file["version"], 3);
        assert_eq!(file["status"]["iteration"], 3);
        let n = file["operators"].as_array().unwrap().len();
        assert_eq!(file["status"]["hessian"]["optimizer"], name);
        assert_eq!(file["status"]["hessian"]["dimension"].as_u64().unwrap() as usize, n);
        assert_eq!(file["status"]["hessian"]["values"].as_array().unwrap().len(), n * n);
        let stored = Document::read(&ctx, &path, CHECKPOINT).unwrap();
        let resumed = stored.resume(&problem, &options(method, 6)).unwrap();
        assert!(resumed.notes.is_empty());
        // A checkpoint with its estimate continues the growth: no relaxation of the stored
        // sequence, and every later step is the uninterrupted run's, bit for bit.
        assert!(resumed.seed.is_none());
        assert_eq!(resumed.indices, straight.indices);
        assert_eq!(resumed.parameters, straight.parameters);
        assert_eq!(resumed.energy, straight.energy);
        assert_eq!(resumed.hessian, straight.hessian);
        assert_eq!(resumed.iterations.len(), straight.iterations.len() - 3);
        for (later, earlier) in resumed.iterations.iter().zip(&straight.iterations[3..]) {
            assert_eq!((later.pool_index, later.energy, later.num_evaluations), (earlier.pool_index, earlier.energy, earlier.num_evaluations));
        }
        // The other carrier's estimate is not converted and not used: the identity, with a note
        // naming both optimizers, then the relaxation of the stored sequence.
        let other = if method == Method::Slsqp { Method::Bfgs } else { Method::Slsqp };
        let crossed = stored.resume(&problem, &options(other, 6)).unwrap();
        assert_eq!(crossed.notes.len(), 1, "{name} file, {} run", other.name());
        assert!(crossed.notes[0].contains(name) && crossed.notes[0].contains(other.name()));
        assert!(!crossed.seed.as_ref().unwrap().warm);
        assert_eq!(crossed.indices, straight.indices);
        // L-BFGS carries none and neither says anything nor uses it.
        let lbfgs = stored.resume(&problem, &options(Method::Lbfgs, 6)).unwrap();
        assert!(lbfgs.notes.is_empty() && lbfgs.hessian.is_none());
        // A version-3 entry without `optimizer` was written when only SLSQP carried one.
        let mut unnamed = file.clone();
        unnamed["status"]["hessian"].as_object_mut().unwrap().remove("optimizer");
        let unnamed_path = dir.join("unnamed.json");
        std::fs::write(&unnamed_path, unnamed.to_string()).unwrap();
        let unnamed = Document::read(&ctx, &unnamed_path, CHECKPOINT).unwrap();
        let from_slsqp = unnamed.resume(&problem, &options(Method::Slsqp, 6)).unwrap();
        assert!(from_slsqp.notes.is_empty());
        assert_eq!(unnamed.resume(&problem, &options(Method::Bfgs, 6)).unwrap().notes.len(), 1);
        // The same file without the estimate resumes from the identity and says so (version 2),
        // or says nothing (version 3: another optimizer wrote it).
        let mut legacy = file.clone();
        legacy["status"].as_object_mut().unwrap().remove("hessian");
        legacy["version"] = 2.into();
        let legacy_path = dir.join("legacy.json");
        std::fs::write(&legacy_path, legacy.to_string()).unwrap();
        let old = Document::read(&ctx, &legacy_path, CHECKPOINT).unwrap();
        let cold = old.resume(&problem, &options(method, 6)).unwrap();
        assert_eq!(cold.notes.len(), 1);
        assert!(cold.notes[0].contains("Hessian"));
        assert!(!cold.seed.as_ref().unwrap().warm);
        // From the identity the growth need not take the uninterrupted run's operators (it
        // says so): it keeps the stored ones and does not end above them.
        assert_eq!(cold.indices[..n], straight.indices[..n]);
        assert!(cold.energy <= partial.energy + 1e-9);
        let mut none = file.clone();
        none["status"].as_object_mut().unwrap().remove("hessian");
        let none_path = dir.join("none.json");
        std::fs::write(&none_path, none.to_string()).unwrap();
        let unrecorded = Document::read(&ctx, &none_path, CHECKPOINT).unwrap();
        assert!(unrecorded.resume(&problem, &options(method, 6)).unwrap().notes.is_empty());
        // An estimate that does not match the operator count is refused.
        let mut wrong = file.clone();
        wrong["status"]["hessian"]["dimension"] = (n + 1).into();
        let wrong_path = dir.join("wrong.json");
        std::fs::write(&wrong_path, wrong.to_string()).unwrap();
        let wrong_doc = Document::read(&ctx, &wrong_path, CHECKPOINT).unwrap();
        assert!(wrong_doc.resume(&problem, &options(method, 6)).is_err());
        // The final checkpoint a caller writes carries the result's estimate the same way.
        let status = persistence::status_with_hessian(
            serde_json::json!({"complete": true, "converged": partial.success}),
            name,
            partial.hessian.as_deref(),
        )
        .unwrap();
        let done = persistence::checkpoint(
            &ctx, &problem, &partial.indices, &partial.parameters, true, "adapt-vqe", status, partial.energy,
        )
        .unwrap();
        done.save(&path).unwrap();
        let final_file = read(&path);
        assert_eq!(final_file["status"]["hessian"]["optimizer"], name);
        assert_eq!(final_file["status"]["hessian"]["dimension"].as_u64().unwrap() as usize, partial.indices.len());
        drop((straight, partial, resumed, crossed, lbfgs, from_slsqp, cold, stored, old, unrecorded, unnamed_path, wrong_doc, done));
    }
    assert!(persistence::status_with_hessian(serde_json::json!({}), "SLSQP", Some(&[1., 2., 3.])).is_err());
    drop(problem);
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(ctx.memory_used(), 0);
}
