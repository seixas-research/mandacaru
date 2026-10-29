//! State-wise first-choice likelihood, native Adam, and held-out ranking.
use super::{
    data::{Dataset, State},
    graph::{Candidates, Graph, Matrix},
    network::{self, Parameters},
};
use crate::{
    math::{autodiff::Tape, random::Rng},
    product, Context, Error, Result,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

#[derive(Clone, Debug)]
pub struct Options {
    pub seed: u64,
    pub epochs: usize,
    pub min_steps: usize,
    pub batch: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub label_temperature: f64,
    pub fractions: Vec<f64>,
    pub folds: Option<usize>,
    pub mixing: f64,
    pub temperature: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            seed: 0,
            epochs: 100,
            min_steps: 1000,
            batch: 16,
            learning_rate: 3e-3,
            weight_decay: 1e-4,
            label_temperature: 0.3,
            fractions: vec![0.25, 0.5, 1.],
            folds: None,
            mixing: 0.3,
            temperature: 1.,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        if self.batch == 0
            || self.epochs == 0
            || self.folds.is_some_and(|n| n < 2)
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.
            || !self.weight_decay.is_finite()
            || self.weight_decay < 0.
            || !self.label_temperature.is_finite()
            || self.label_temperature <= 0.
            || !self.temperature.is_finite()
            || self.temperature <= 0.
            || !self.mixing.is_finite()
            || self.mixing <= 0.
            || self.mixing > 1.
            || self.fractions.is_empty()
            || self
                .fractions
                .iter()
                .any(|v| !v.is_finite() || *v <= 0. || *v > 1.)
            || self.fractions.last() != Some(&1.)
        {
            return Err(invalid("training needs positive epochs, batch and temperatures, at least two folds, fractions in (0,1] ending at one, and mixing in (0,1]"));
        }
        Ok(())
    }
}
pub struct Model {
    pub parameters: Parameters,
    pub mean: [f64; 9],
    pub std: [f64; 9],
    pub ready: bool,
    pub report: Value,
    pub mixing: f64,
    pub temperature: f64,
}
impl Model {
    pub fn scores(&self, context: &Context, graph: &Graph, rows: &Candidates) -> Result<Vec<f64>> {
        let mut tape = Tape::new(context);
        let variables = self.parameters.variables(&mut tape)?;
        let scores = network::scores(&mut tape, &variables, graph, rows, &self.mean, &self.std)?;
        let values = tape.value(scores).as_slice().to_vec();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("learned scores overflow"));
        }
        Ok(values)
    }
    pub fn as_json(&self) -> Value {
        json!({"format":"mandacaru-proposal-model","version":1,"kind":"graph",
            "weights":self.parameters.as_json(),"mean":self.mean,"std":self.std,
            "ready":self.ready,"report":self.report,"mixing":self.mixing,"temperature":self.temperature})
    }
    pub fn from_json(context: &Context, source: &str) -> Result<Self> {
        let _parse = context.reserve(product(&[source.len(), 32])?)?;
        let v: Value = serde_json::from_str(source).map_err(|e| invalid(&e.to_string()))?;
        if v["format"] != "mandacaru-proposal-model" || v["version"] != 1 || v["kind"] != "graph" {
            return Err(invalid("unsupported native proposal model schema"));
        }
        let array = |name: &str| -> Result<[f64; 9]> {
            let mut out = [0.; 9];
            let entries = v[name]
                .as_array()
                .filter(|a| a.len() == 9)
                .ok_or_else(|| invalid("proposal normalization requires nine descriptors"))?;
            for (target, value) in out.iter_mut().zip(entries) {
                *target = value
                    .as_f64()
                    .filter(|x| x.is_finite())
                    .ok_or_else(|| invalid("proposal normalization must be finite"))?;
            }
            Ok(out)
        };
        let mean = array("mean")?;
        let std = array("std")?;
        let mixing = v["mixing"].as_f64().unwrap_or(f64::NAN);
        let temperature = v["temperature"].as_f64().unwrap_or(f64::NAN);
        if std.iter().any(|s| *s <= 0.)
            || !mixing.is_finite()
            || mixing <= 0.
            || mixing > 1.
            || !temperature.is_finite()
            || temperature <= 0.
        {
            return Err(invalid(
                "proposal model has invalid scales, mixing or temperature",
            ));
        }
        Ok(Self {
            parameters: Parameters::from_json(context, &v["weights"])?,
            mean,
            std,
            ready: v["ready"]
                .as_bool()
                .ok_or_else(|| invalid("proposal readiness must be a boolean"))?,
            report: v["report"].clone(),
            mixing,
            temperature,
        })
    }
}
pub fn softmax(values: &[f64]) -> Vec<f64> {
    let top = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out: Vec<_> = values.iter().map(|v| (v - top).exp()).collect();
    let sum = out.iter().sum::<f64>();
    for v in &mut out {
        *v /= sum;
    }
    out
}
/// Returns the exact score adjoint of the state-wise cross entropy.
pub fn first_choice(scores: &[f64], targets: &[f64], scale: f64) -> Result<(f64, Vec<f64>)> {
    if scores.len() != targets.len()
        || scores.is_empty()
        || !scale.is_finite()
        || scale <= 0.
        || scores.iter().chain(targets).any(|v| !v.is_finite())
    {
        return Err(invalid(
            "choice likelihood needs equally sized finite observations and a positive label scale",
        ));
    }
    let p = softmax(scores);
    let labels: Vec<_> = targets.iter().map(|v| -v / scale).collect();
    if labels.iter().any(|v| !v.is_finite()) {
        return Err(invalid("choice labels overflow"));
    }
    let q = softmax(&labels);
    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_sum = top + scores.iter().map(|v| (v - top).exp()).sum::<f64>().ln();
    Ok((
        q.iter().zip(scores).map(|(q, s)| q * (log_sum - s)).sum(),
        p.iter().zip(q).map(|(p, q)| p - q).collect(),
    ))
}
fn normalized(states: &[&State]) -> ([f64; 9], [f64; 9], f64) {
    let count = states.iter().map(|s| s.targets.len()).sum::<usize>() as f64;
    if count == 0. {
        return ([0.; 9], [1.; 9], 1.);
    }
    let mut mean = [0.; 9];
    for state in states {
        for r in 0..state.targets.len() {
            for (c, v) in mean.iter_mut().enumerate() {
                *v += state.rows.descriptors[(r, c)] / count;
            }
        }
    }
    let target_mean = states.iter().flat_map(|s| &s.targets).sum::<f64>() / count;
    let mut std = [0.; 9];
    let mut variance = 0.;
    for state in states {
        for r in 0..state.targets.len() {
            variance += (state.targets[r] - target_mean).powi(2) / count;
            for (c, v) in std.iter_mut().enumerate() {
                *v += (state.rows.descriptors[(r, c)] - mean[c]).powi(2) / count;
            }
        }
    }
    for v in &mut std {
        *v = v.sqrt();
        if *v <= 1e-12 {
            *v = 1.;
        }
    }
    (mean, std, if variance > 0. { variance.sqrt() } else { 1. })
}
fn decay(name: &str) -> bool {
    name == "w1"
        || name == "w2"
        || name == "qubit_in"
        || name == "term_in"
        || [
            "to_term.",
            "to_qubit.",
            "term_update.",
            "qubit_update.",
            "readout.",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}
fn shuffle<T>(values: &mut [T], random: &mut Rng) {
    for i in (1..values.len()).rev() {
        let j = random.index(i + 1);
        values.swap(i, j);
    }
}
pub fn train(context: &Context, states: &[&State], options: &Options) -> Result<Model> {
    options.validate()?;
    let mut parameters = Parameters::new(context, options.seed)?;
    let (mean, std, target_scale) = normalized(states);
    if !states.is_empty() {
        let cells = parameters.values.values().map(|m| m.len()).sum::<usize>();
        let _adam_memory = context.reserve(product(&[cells, 24])?)?;
        let zeros = || {
            parameters
                .values
                .iter()
                .map(|(k, v)| (k.clone(), Matrix::zeros(v.nrows(), v.ncols())))
                .collect::<BTreeMap<_, _>>()
        };
        let mut first = zeros();
        let mut second = zeros();
        let mut gradient = zeros();
        let mut random = Rng(options.seed);
        let mut order: Vec<_> = (0..states.len()).collect();
        let steps = options
            .min_steps
            .max(product(&[options.epochs, states.len()])?.div_ceil(options.batch));
        for step in 1..=steps {
            shuffle(&mut order, &mut random);
            let selected = &order[..options.batch.min(order.len())];
            let mut tape = Tape::new(context);
            let variables = parameters.variables(&mut tape)?;
            let mut embeddings = BTreeMap::new();
            let mut outputs = Vec::new();
            for &index in selected {
                let state = states[index];
                let identity = std::sync::Arc::as_ptr(&state.graph) as usize;
                let (zh, za) = if let Some(pair) = embeddings.get(&identity) {
                    *pair
                } else {
                    let pair = network::embed(&mut tape, &variables, &state.graph)?;
                    embeddings.insert(identity, pair);
                    pair
                };
                let output =
                    network::head(&mut tape, &variables, zh, za, &state.rows, &mean, &std)?;
                let (_, adjoint) = first_choice(
                    tape.value(output).as_slice(),
                    &state.targets,
                    target_scale * options.label_temperature,
                )?;
                outputs.push((
                    output,
                    Matrix::from_column_slice(adjoint.len(), 1, &adjoint) / selected.len() as f64,
                ));
            }
            tape.backward_many(&outputs)?;
            for (name, sum) in &mut gradient {
                sum.copy_from(tape.gradient(variables[name]));
            }
            drop(tape);
            for (name, weights) in &mut parameters.values {
                let first = first.get_mut(name).unwrap();
                let second = second.get_mut(name).unwrap();
                let gradient = &gradient[name];
                for i in 0..weights.len() {
                    let g = gradient[i]
                        + if decay(name) {
                            2. * options.weight_decay * weights[i]
                        } else {
                            0.
                        };
                    first[i] = 0.9 * first[i] + 0.1 * g;
                    second[i] = 0.999 * second[i] + 0.001 * g * g;
                    weights[i] -= options.learning_rate
                        * (first[i] / (1. - 0.9f64.powf(step as f64)))
                        / ((second[i] / (1. - 0.999f64.powf(step as f64))).sqrt() + 1e-8);
                    if !weights[i].is_finite() {
                        return Err(invalid("proposal training produced nonfinite weights"));
                    }
                }
            }
        }
    }
    Ok(Model {
        parameters,
        mean,
        std,
        ready: false,
        report: Value::Null,
        mixing: options.mixing,
        temperature: options.temperature,
    })
}
fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<_> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.; values.len()];
    let mut first = 0;
    while first < order.len() {
        let mut last = first + 1;
        while last < order.len() && values[order[last]] == values[order[first]] {
            last += 1;
        }
        let average = (first + last - 1) as f64 / 2.;
        for &index in &order[first..last] {
            ranks[index] = average;
        }
        first = last;
    }
    ranks
}
pub fn spearman(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.len() < 3 || a.iter().chain(b).any(|v| !v.is_finite()) {
        return None;
    }
    let a = ranks(a);
    let b = ranks(b);
    let mean = (a.len() - 1) as f64 / 2.;
    let covariance = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - mean) * (b - mean))
        .sum::<f64>();
    let variance = a.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
        * b.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    if variance > 0. {
        Some(covariance / variance.sqrt())
    } else {
        None
    }
}
fn average(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}
pub fn fit(context: &Context, dataset: &Dataset, options: &Options) -> Result<Model> {
    options.validate()?;
    let mut random = Rng(options.seed);
    let mut groups = dataset.groups.clone();
    shuffle(&mut groups, &mut random);
    let nfolds = options.folds.unwrap_or(groups.len()).min(groups.len());
    let mut curve = Vec::new();
    let mut full_graph = Vec::new();
    let mut full_gradient = Vec::new();
    let mut valid_groups = 0;
    let mut wins = 0;
    if groups.len() >= 3 {
        for &fraction in &options.fractions {
            let mut folds = Vec::new();
            full_graph.clear();
            full_gradient.clear();
            valid_groups = 0;
            wins = 0;
            for fold in 0..nfolds {
                let held_out: Vec<_> = groups
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % nfolds == fold)
                    .map(|(_, g)| g.as_str())
                    .collect();
                let mut training: Vec<_> = dataset
                    .states
                    .iter()
                    .filter(|s| !held_out.contains(&s.group.as_str()))
                    .collect();
                shuffle(&mut training, &mut random);
                training.truncate(((training.len() as f64 * fraction).ceil() as usize).max(1));
                let model = train(context, &training, options)?;
                let mut by_group = BTreeMap::<String, (Vec<f64>, Vec<f64>)>::new();
                for state in dataset
                    .states
                    .iter()
                    .filter(|s| held_out.contains(&s.group.as_str()) && s.targets.len() >= 3)
                {
                    let scores = model.scores(context, &state.graph, &state.rows)?;
                    let observed: Vec<_> = state.changes.iter().map(|v| -v).collect();
                    let entry = by_group.entry(state.group.clone()).or_default();
                    if let Some(value) = spearman(&scores, &observed) {
                        entry.0.push(value);
                        full_graph.push(value);
                    }
                    if let Some(value) = spearman(&state.gradients, &observed) {
                        entry.1.push(value);
                        full_gradient.push(value);
                    }
                }
                let mut metrics = Vec::new();
                for (group, (learned, gradient)) in by_group {
                    let learned = average(&learned);
                    let gradient = average(&gradient);
                    if let Some(baseline) = gradient {
                        valid_groups += 1;
                        if learned.is_some_and(|v| v > baseline) {
                            wins += 1;
                        }
                    }
                    metrics.push(json!({"group":group,"states":gradient.map(|_|true),"graph":learned,"gradient":gradient}));
                }
                folds.push(
                    json!({"held_out":held_out,"train_states":training.len(),"groups":metrics}),
                );
            }
            curve.push(json!({"fraction":fraction,"folds":folds,"within_state":{"graph":{"spearman":average(&full_graph)},"gradient":{"spearman":average(&full_gradient)}}}));
        }
    }
    let learned = average(&full_graph);
    let gradient = average(&full_gradient);
    let ready = valid_groups >= 3
        && wins * 2 > valid_groups
        && learned.zip(gradient).is_some_and(|(g, b)| g >= b + 0.05);
    let reason = if groups.len() < 3 {
        format!(
            "{} groups; at least three are needed to hold one out and train on two",
            groups.len()
        )
    } else if valid_groups < 3 {
        format!("only {valid_groups} held-out groups have nonconstant states with three or more insertions; record chains with screen_insertions")
    } else {
        let metric =
            |value: Option<f64>| value.map_or_else(|| "undefined".into(), |v| format!("{v:.3}"));
        format!("within-state Spearman: graph {}, gradient {}; improved in {wins} of {valid_groups} groups; readiness requires a 0.05 margin and a strict majority",metric(learned),metric(gradient))
    };
    let states: Vec<_> = dataset.states.iter().collect();
    let mut model = train(context, &states, options)?;
    model.ready = ready;
    model.report = json!({"ready":ready,"kind":"graph","reason":reason,"groups":groups.len(),"states":states.len(),
        "insertions":dataset.observations,"curve":curve,"kinds":["graph"],"held_out_groups":valid_groups,"wins":wins});
    Ok(model)
}

pub fn fit_json(
    context: &Context,
    data: &str,
    group_by: &str,
    options: &Options,
) -> Result<String> {
    context.run(|| {
        let dataset = Dataset::from_json(context, data, group_by)?;
        let model = fit(context, &dataset, options)?;
        serde_json::to_string(&model.as_json()).map_err(|e| invalid(&e.to_string()))
    })
}
