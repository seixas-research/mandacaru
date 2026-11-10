use mandacaru_core::{
    quantum::search::learning::{
        data::Dataset,
        training::{self, Model, Options},
    },
    Context,
};
use serde_json::json;

fn observations() -> String {
    let mut problems = serde_json::Map::new();
    let mut records = Vec::new();
    for group in 0..3 {
        let identity = format!("group{group}");
        problems.insert(identity.clone(),json!({"format":"mandacaru-search-problem","version":1,
            "num_qubits":4,"spatial_orbitals":2,"num_particles":[1,0],"mapping":"jordan_wigner",
            "hamiltonian":[["ZIII",0.3+group as f64*0.01,0.],["XXII",0.2,0.]],
            "pool":[[["YIII",0.,1.]],[["IYII",0.,1.]],[["IIYI",0.,1.]]],"composition":[[group+1,2]]}));
        for state in 0..2 {
            for operator in 0..3 {
                let energy = -0.01 * (operator as f64 + 1.);
                records.push(json!({"format":"mandacaru-search-edit","version":1,"move":"insert",
                "problem_id":identity,"source":[],"source_parameters":[],"source_energy":state as f64*0.01,
                "source_gradients":[0.1,0.2,0.3],"operator":operator,"delta_energy":energy,
                "energy":state as f64*0.01+energy}));
            }
        }
    }
    json!({"problems":problems,"records":records}).to_string()
}

#[test]
fn likelihood_adjoint_and_rank_ties_match_independent_values() {
    let scores = [0.2, -0.4, 1.2];
    let targets = [-1., 0.1, -0.5];
    let (_, gradient) = training::first_choice(&scores, &targets, 0.3).unwrap();
    for index in 0..3 {
        let mut plus = scores;
        let mut minus = scores;
        plus[index] += 1e-6;
        minus[index] -= 1e-6;
        let expected = (training::first_choice(&plus, &targets, 0.3).unwrap().0
            - training::first_choice(&minus, &targets, 0.3).unwrap().0)
            / 2e-6;
        assert!((gradient[index] - expected).abs() < 1e-9);
    }
    assert!(
        (training::spearman(&[1., 1., 3., 4.], &[4., 3., 2., 1.]).unwrap() + 0.9486832980505138)
            .abs()
            < 1e-12
    );
    assert!(training::spearman(&[1.; 3], &[1., 2., 3.]).is_none());
    assert!(training::first_choice(&[f64::NAN], &[1.], 1.).is_err());
}

#[test]
fn native_training_lowers_observed_loss_preserves_archives_and_releases_every_budget() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let data = Dataset::from_json(&context, &observations(), "molecule").unwrap();
        assert_eq!(data.groups.len(), 3);
        assert_eq!(data.states.len(), 6);
        let states: Vec<_> = data.states.iter().collect();
        let mut options = Options {
            epochs: 1,
            min_steps: 0,
            batch: 6,
            ..Default::default()
        };
        let initial = training::train(&context, &[], &options).unwrap();
        let targets: Vec<_> = states
            .iter()
            .flat_map(|s| s.targets.iter().copied())
            .collect();
        let mean = targets.iter().sum::<f64>() / targets.len() as f64;
        let target_scale =
            (targets.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / targets.len() as f64).sqrt()
                * options.label_temperature;
        let loss = |model: &Model| {
            states
                .iter()
                .map(|s| {
                    let scores = model.scores(&context, &s.graph, &s.rows).unwrap();
                    training::first_choice(&scores, &s.targets, target_scale)
                        .unwrap()
                        .0
                })
                .sum::<f64>()
                / states.len() as f64
        };
        options.min_steps = 30;
        let model = training::train(&context, &states, &options).unwrap();
        let mut baseline = model.as_json();
        baseline["weights"] = initial.parameters.as_ref().unwrap().as_json();
        let baseline = Model::from_json(&context, &baseline.to_string()).unwrap();
        let before = loss(&baseline);
        assert!(loss(&model) < before - 0.05, "{} vs {before}", loss(&model));
        let restored = Model::from_json(&context, &model.as_json().to_string()).unwrap();
        assert!((loss(&model) - loss(&restored)).abs() < 1e-14);
        let mut corrupt = model.as_json();
        corrupt["std"][0] = json!(0.);
        assert!(Model::from_json(&context, &corrupt.to_string()).is_err());
        options.min_steps = 1;
        options.fractions = vec![1.];
        let validated = training::fit(&context, &data, &options).unwrap();
        // The synthetic gradient already ranks every observed decrease
        // perfectly. A learned model must not claim it beat that baseline.
        assert!(!validated.ready);
        assert_eq!(validated.report["held_out_groups"], 3);
        assert_eq!(
            validated.report["curve"][0]["folds"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            validated.report["curve"][0]["within_state"]["gradient"]["spearman"],
            1.
        );
        let empty =
            Dataset::from_json(&context, r#"{"problems":{},"records":[]}"#, "molecule").unwrap();
        let model = training::fit(&context, &empty, &options).unwrap();
        assert!(!model.ready);
        assert!(model.report["reason"]
            .as_str()
            .unwrap()
            .contains("0 groups"));
    }
    assert_eq!(context.memory_used(), 0);
    // Construction also fails safely before an oversized training tape.
    let small = Context::new(Some(1), 1024).unwrap();
    assert!(Dataset::from_json(&small, &observations(), "molecule").is_err());
    assert_eq!(small.memory_used(), 0);
}

#[test]
fn both_kinds_are_assessed_and_the_ranker_draws_from_its_own_scores() {
    use mandacaru_core::quantum::search::learning::training::Kind;
    let context = Context::new(Some(1), 256 << 20).unwrap();
    let data = Dataset::from_json(&context, &observations(), "molecule").unwrap();
    let options = Options {
        kinds: vec![Kind::Ranker, Kind::Graph],
        min_steps: 1,
        epochs: 1,
        fractions: vec![1.],
        ..Default::default()
    };
    let model = training::fit(&context, &data, &options).unwrap();
    // The synthetic gradient ranks every decrease perfectly: neither kind can beat it, so the model
    // keeps the last kind named, holds both predictors, and is not ready.
    assert!(!model.ready);
    assert_eq!(model.kind, Kind::Graph);
    assert!(model.ranker.is_some() && model.parameters.is_some());
    let within = &model.report["curve"][0]["within_state"];
    assert!(within["ranker"]["spearman"].is_number() && within["graph"]["spearman"].is_number(), "{within}");
    assert_eq!(model.report["kinds"], json!(["ranker", "graph"]));
    // A ranker model round-trips and scores with the ranker alone.
    let mut ranker = model;
    ranker.kind = Kind::Ranker;
    let restored = Model::from_json(&context, &ranker.as_json().to_string()).unwrap();
    assert_eq!(restored.kind, Kind::Ranker);
    assert_eq!(restored.ranker, ranker.ranker);
    let state = &data.states[0];
    let scores = restored.scores(&context, &state.graph, &state.rows).unwrap();
    let r = restored.ranker.as_ref().unwrap();
    for (row, score) in (0..state.rows.descriptors.nrows()).zip(&scores) {
        let x: [f64; 9] = std::array::from_fn(|c| state.rows.descriptors[(row, c)]);
        assert_eq!(*score, r.score(&x));
    }
    // A ranker-only fit holds no network, and its file says so.
    let only = training::fit(&context, &data, &Options { kinds: vec![Kind::Ranker], ..options.clone() }).unwrap();
    assert!(only.parameters.is_none() && only.as_json().get("weights").is_none());
    let mut missing = only.as_json();
    missing["kind"] = json!("graph");
    assert!(Model::from_json(&context, &missing.to_string()).is_err());
    assert!(training::fit(&context, &data, &Options { kinds: vec![], ..options.clone() }).is_err());
    drop((ranker, restored, only, data));
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn the_gradient_teaches_only_the_first_stage_of_training() {
    use mandacaru_core::quantum::search::learning::training::{first_choice, taught_choice, teacher_weight, softmax};
    // The weight starts at `mixing` and reaches zero at the fraction of the steps, then stays there.
    assert_eq!(teacher_weight(0.3, 0.2, 0, 100), 0.3);
    assert!((teacher_weight(0.3, 0.2, 10, 100) - 0.15).abs() < 1e-15);
    assert_eq!(teacher_weight(0.3, 0.2, 20, 100), 0.);
    assert_eq!(teacher_weight(0.3, 0.2, 99, 100), 0.);
    let (scores, targets, teacher) = ([0.2, -0.4, 1.1], [0.5, -0.2, 0.1], [0.6, 0.3, 0.1]);
    // Weight zero is the observed likelihood; weight one has the teacher as its target.
    assert_eq!(taught_choice(&scores, &targets, 0.7, &teacher, 0.).unwrap(), first_choice(&scores, &targets, 0.7).unwrap());
    let (_, adjoint) = taught_choice(&scores, &targets, 0.7, &teacher, 1.).unwrap();
    let p = softmax(&scores);
    for i in 0..3 {
        assert!((adjoint[i] - (p[i] - teacher[i])).abs() < 1e-15);
    }
    // The adjoint is the derivative of the blended loss.
    let loss = |s: &[f64]| taught_choice(s, &targets, 0.7, &teacher, 0.4).unwrap().0;
    let (_, adjoint) = taught_choice(&scores, &targets, 0.7, &teacher, 0.4).unwrap();
    for i in 0..3 {
        let (mut up, mut down) = (scores, scores);
        up[i] += 1e-6;
        down[i] -= 1e-6;
        assert!(((loss(&up) - loss(&down)) / 2e-6 - adjoint[i]).abs() < 1e-8);
    }
}
