//! MG-Q8: the pairwise ranker of VALSQA's decision A (`quantum::search::learning::ranker`) against
//! the deleted Python ranker it restores (`test/migration/reference/ranker.json`,
//! `freeze_ranker.py`, which runs the Python functions verbatim from commit 241d688's parent):
//! the standardization exactly, the pair count, and the weights at the accuracy of SciPy's
//! L-BFGS-B, whose stopping tolerance is looser than the Newton solve here. The Rust weights are
//! checked to be the minimizer to round-off independently: the objective's gradient vanishes.
use mandacaru_core::quantum::search::learning::ranker::{Ranker, Sample, DESCRIPTORS};
use serde_json::Value;

fn frozen() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test/migration/reference/ranker.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn the_ranker_is_the_deleted_python_one() {
    let reference = frozen();
    let states: Vec<(Vec<[f64; DESCRIPTORS]>, Vec<f64>)> = reference["states"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            let rows = s["descriptors"].as_array().unwrap().iter().map(|r| floats(r).try_into().unwrap()).collect();
            (rows, floats(&s["targets"]))
        })
        .collect();
    let samples: Vec<Sample> = states.iter().map(|(d, t)| Sample { descriptors: d, targets: t }).collect();
    let l2 = reference["l2"].as_f64().unwrap();
    let ranker = Ranker::fit(&samples, l2).unwrap().unwrap();
    assert_eq!(ranker.pairs, reference["pairs"].as_u64().unwrap() as usize);
    let (mean, std, weights) = (floats(&reference["mean"]), floats(&reference["std"]), floats(&reference["weights"]));
    let worst = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f64, f64::max);
    assert!(worst(&ranker.mean, &mean) <= 1e-14 && worst(&ranker.std, &std) <= 1e-14);
    let miss = worst(&ranker.weights, &weights);
    println!("weights {:?}\n  miss against the Python L-BFGS-B weights {miss:.1e}", ranker.weights);
    assert!(miss <= 1e-4, "{miss:e}");
    // The constant column carries no weight.
    assert_eq!(ranker.weights[4], 0.);
    // The objective's gradient at the Rust weights vanishes: the minimizer, independently of SciPy.
    let standard = |row: &[f64; DESCRIPTORS]| -> Vec<f64> { (0..DESCRIPTORS).map(|c| (row[c] - ranker.mean[c]) / ranker.std[c]).collect() };
    let ridge = 2. * l2 * ranker.pairs as f64;
    let mut gradient: Vec<f64> = ranker.weights.iter().map(|w| ridge * w).collect();
    for (rows, targets) in &states {
        for a in 0..targets.len() {
            for b in a + 1..targets.len() {
                if targets[a] == targets[b] {
                    continue;
                }
                let (i, j) = if targets[a] < targets[b] { (a, b) } else { (b, a) };
                let d: Vec<f64> = standard(&rows[i]).iter().zip(standard(&rows[j])).map(|(x, y)| x - y).collect();
                let margin: f64 = d.iter().zip(&ranker.weights).map(|(x, w)| x * w).sum();
                for c in 0..DESCRIPTORS {
                    gradient[c] -= d[c] / (1. + margin.exp());
                }
            }
        }
    }
    let norm = gradient.iter().fold(0f64, |m, g| m.max(g.abs()));
    println!("  gradient at the Rust weights {norm:.1e}");
    assert!(norm <= 1e-9, "{norm:e}");
    // A higher score is a larger predicted decrease: the best candidate of the first state ranks first.
    let (rows, targets) = &states[0];
    let best = (0..targets.len()).min_by(|&a, &b| targets[a].total_cmp(&targets[b])).unwrap();
    let top = (0..targets.len()).max_by(|&a, &b| ranker.score(&rows[a]).total_cmp(&ranker.score(&rows[b]))).unwrap();
    println!("  first state: best observed {best}, top scored {top}");
}

#[test]
fn a_ranker_needs_a_pair() {
    let rows = [[0.; DESCRIPTORS]; 2];
    let tied = [Sample { descriptors: &rows, targets: &[1., 1.] }];
    assert!(Ranker::fit(&tied, 1e-2).unwrap().is_none());
    assert!(Ranker::fit(&[], 1e-2).unwrap().is_none());
    assert!(Ranker::fit(&tied, 0.).is_err());
}
