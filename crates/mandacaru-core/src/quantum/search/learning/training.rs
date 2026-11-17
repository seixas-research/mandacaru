//! State-wise first-choice likelihood, native Adam, and held-out ranking of the two proposal
//! kinds of VALSQA's decision A: the pairwise ranker ([`super::ranker`]) and the graph network.
//!
//! Readiness is judged within states: a kind's pooled within-state Spearman correlation must exceed
//! the gradient's by 0.05, and it must beat the gradient in most held-out groups. Among the kinds
//! that pass, the highest correlation is chosen; a tie goes to the simpler kind (the ranker). With
//! none passing, the model keeps the last kind requested and is not ready.
use super::{
    data::{Dataset, EditDataset, EditKind, EditState, State},
    graph::{compress, BipartiteGraph, Candidates, Matrix},
    network::{self, Parameters},
    ranker::{PairwiseRanker, PairwiseSample, Ranker, Sample, DESCRIPTORS},
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

/// A proposal predictor, the simpler first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Ranker,
    Graph,
}
impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Ranker => "ranker",
            Kind::Graph => "graph",
        }
    }
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "ranker" => Ok(Kind::Ranker),
            "graph" => Ok(Kind::Graph),
            _ => Err(invalid("proposal kinds are \"ranker\" and \"graph\"")),
        }
    }
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
    /// The initial weight of the gradient softmax in the graph model's training targets.
    pub mixing: f64,
    /// The share of the training steps over which that weight falls linearly to zero; the rest of
    /// the training, and every draw of a trained model, uses the model alone.
    pub mixing_fraction: f64,
    pub temperature: f64,
    /// The predictors to assess and train (at least one, no repeat).
    pub kinds: Vec<Kind>,
    /// The ranker's ridge: the mean pairwise loss plus `l2 |w|^2`.
    pub l2: f64,
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
            mixing_fraction: 0.2,
            temperature: 1.,
            kinds: vec![Kind::Graph],
            l2: 1e-2,
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
            || self.kinds.is_empty()
            || (1..self.kinds.len()).any(|i| self.kinds[..i].contains(&self.kinds[i]))
            || !self.l2.is_finite()
            || self.l2 <= 0.
            || !(self.mixing_fraction > 0. && self.mixing_fraction <= 1.)
        {
            return Err(invalid("training needs positive epochs, batch and temperatures, at least two folds, fractions in (0,1] ending at one, mixing in (0,1], at least one proposal kind without repeats, and a positive ranker l2"));
        }
        Ok(())
    }
}
fn parse_fixed_array<const F: usize>(from: &Value, name: &str) -> Result<[f64; F]> {
    let mut out = [0.; F];
    let entries = from[name]
        .as_array()
        .filter(|a| a.len() == F)
        .ok_or_else(|| invalid("proposal feature vector length mismatch"))?;
    for (target, value) in out.iter_mut().zip(entries) {
        *target = value
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or_else(|| invalid("proposal feature values must be finite"))?;
    }
    Ok(out)
}

fn parse_ranker_json<const F: usize>(value: &Value) -> Result<PairwiseRanker<F>> {
    let ranker = PairwiseRanker {
        weights: parse_fixed_array(value, "weights")?,
        mean: parse_fixed_array(value, "mean")?,
        std: parse_fixed_array(value, "std")?,
        pairs: value["pairs"]
            .as_u64()
            .ok_or_else(|| invalid("a ranker needs its pair count"))? as usize,
    };
    if ranker.std.iter().any(|s| *s <= 0.) {
        return Err(invalid("a ranker's scales must be positive"));
    }
    Ok(ranker)
}

pub struct Model {
    /// The kind the proposal draws from.
    pub kind: Kind,
    /// The graph network, when trained.
    pub parameters: Option<Parameters>,
    /// The pairwise ranker, when trained (`None` also when no state holds a pair).
    pub ranker: Option<Ranker>,
    /// Optional multi-move pairwise edit ranker (`insert`, `delete`, `replace`) with sequence
    /// overlap (`BG3`), curvature (`BG4`), and edit position (`BG1`) features.
    pub edit: Option<EditModel>,
    pub mean: [f64; 9],
    pub std: [f64; 9],
    pub ready: bool,
    pub report: Value,
    pub mixing: f64,
    pub temperature: f64,
}
impl Model {
    /// Wraps a standalone trained [`EditModel`] as a ready `VALSQA` proposal [`Model`].
    pub fn from_edit(edit: EditModel, temperature: f64) -> Result<Self> {
        if !temperature.is_finite() || temperature <= 0. {
            return Err(invalid(
                "proposal model temperature must be positive and finite",
            ));
        }
        Ok(Self {
            kind: Kind::Ranker,
            parameters: None,
            ranker: None,
            edit: Some(edit),
            mean: [0.; 9],
            std: [1.; 9],
            ready: true,
            report: json!({"ready": true, "kind": "edit", "reason": "multi-move edit model"}),
            mixing: 0.3,
            temperature,
        })
    }

    /// Attaches a multi-move [`EditModel`] (`BG1`/`BG3`/`BG4`) to this proposal [`Model`].
    pub fn with_edit(mut self, edit: EditModel) -> Self {
        self.edit = Some(edit);
        self
    }

    /// True if an attached [`EditModel`] places nonzero weight on diagonal pool curvatures (`row[10]`).
    pub fn uses_curvatures(&self) -> bool {
        let Some(edit) = &self.edit else {
            return false;
        };
        [&edit.insert, &edit.delete, &edit.replace]
            .into_iter()
            .flatten()
            .any(|head| head.weights[10] != 0.)
    }

    pub fn scores(
        &self,
        context: &Context,
        graph: &BipartiteGraph,
        rows: &Candidates,
    ) -> Result<Vec<f64>> {
        self.scores_with_curvatures(context, graph, rows, None)
    }

    /// Scores candidate insertion/replacement operators, optionally folding in exact pool
    /// curvatures (`BG4`) and multi-move `EditModel` sequence-overlap / position signals.
    pub fn scores_with_curvatures(
        &self,
        context: &Context,
        graph: &BipartiteGraph,
        rows: &Candidates,
        curvatures: Option<&[f64]>,
    ) -> Result<Vec<f64>> {
        let n_rows = rows.operators.len();
        let mut values = if self.kind == Kind::Ranker {
            match &self.ranker {
                Some(ranker) => descriptor_rows(rows)
                    .iter()
                    .map(|row| ranker.score(row))
                    .collect(),
                None if self.edit.as_ref().and_then(|e| e.insert.as_ref()).is_some() => {
                    vec![0.; n_rows]
                }
                None => return Err(invalid("a ranker model holds no ranker")),
            }
        } else if let Some(parameters) = &self.parameters {
            let mut tape = Tape::new(context);
            let variables = parameters.variables(&mut tape)?;
            let scores =
                network::scores(&mut tape, &variables, graph, rows, &self.mean, &self.std)?;
            tape.value(scores).as_slice().to_vec()
        } else if self.edit.as_ref().and_then(|e| e.insert.as_ref()).is_some() {
            vec![0.; n_rows]
        } else {
            return Err(invalid("a graph model holds no network"));
        };
        if let Some(insert_head) = self.edit.as_ref().and_then(|e| e.insert.as_ref()) {
            let positions =
                vec![rows.descriptors.get((0, 8)).copied().unwrap_or(0.) as usize; n_rows];
            for (slot, edit_row) in values
                .iter_mut()
                .zip(edit_rows_for(rows, &positions, curvatures))
            {
                *slot += insert_head.score(&edit_row);
            }
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("learned scores overflow"));
        }
        Ok(values)
    }

    /// Scores each occupied position `0..sequence.len()` for removal (`EditKind::Delete`),
    /// returning `Ok(None)` when no `delete` edit head is present on this model.
    pub fn removal_scores(
        &self,
        context: &Context,
        graph: &BipartiteGraph,
        sequence: &[usize],
        gradients: &[f64],
        curvatures: Option<&[f64]>,
        energy_above_reference: f64,
    ) -> Result<Option<Vec<f64>>> {
        let Some(delete_head) = self.edit.as_ref().and_then(|e| e.delete.as_ref()) else {
            return Ok(None);
        };
        if sequence.is_empty() {
            return Ok(None);
        }
        let rows = Candidates::new(
            context,
            graph,
            sequence,
            sequence,
            gradients,
            energy_above_reference,
        )?;
        let positions: Vec<usize> = (0..sequence.len()).collect();
        let scores: Vec<f64> = edit_rows_for(&rows, &positions, curvatures)
            .iter()
            .map(|row| delete_head.score(row))
            .collect();
        if scores.iter().any(|v| !v.is_finite()) {
            return Err(invalid("learned removal scores overflow"));
        }
        Ok(Some(scores))
    }

    pub fn as_json(&self) -> Value {
        let mut out = json!({"format":"mandacaru-proposal-model","version":1,"kind":self.kind.name(),
            "mean":self.mean,"std":self.std,
            "ready":self.ready,"report":self.report,"mixing":self.mixing,"temperature":self.temperature});
        if let Some(parameters) = &self.parameters {
            out["weights"] = parameters.as_json();
        }
        if let Some(r) = &self.ranker {
            out["ranker"] = json!({"weights":r.weights,"mean":r.mean,"std":r.std,"pairs":r.pairs});
        }
        if let Some(edit) = &self.edit {
            out["edit"] = edit.as_json();
        }
        out
    }
    pub fn from_json(context: &Context, source: &str) -> Result<Self> {
        let _parse = context.reserve(product(&[source.len(), 32])?)?;
        let v: Value = serde_json::from_str(source).map_err(|e| invalid(&e.to_string()))?;
        if v["format"] != "mandacaru-proposal-model" || v["version"] != 1 {
            return Err(invalid("unsupported native proposal model schema"));
        }
        let kind = Kind::parse(v["kind"].as_str().unwrap_or(""))?;
        let mean: [f64; 9] = parse_fixed_array(&v, "mean")?;
        let std: [f64; 9] = parse_fixed_array(&v, "std")?;
        let ranker = match v.get("ranker") {
            Some(r) => Some(parse_ranker_json(r)?),
            None => None,
        };
        let parameters = match v.get("weights") {
            Some(w) => Some(Parameters::from_json(context, w)?),
            None => None,
        };
        let edit = match v.get("edit") {
            Some(e) if !e.is_null() => Some(EditModel::from_json(e)?),
            _ => None,
        };
        if (kind == Kind::Graph && parameters.is_none() && edit.is_none())
            || (kind == Kind::Ranker && ranker.is_none() && edit.is_none())
        {
            return Err(invalid(
                "a proposal model must hold the predictor of its kind",
            ));
        }
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
            kind,
            parameters,
            ranker,
            edit,
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
/// The gradient's weight in the training targets at `step` of `steps`: `mixing`, falling linearly
/// to zero at `fraction` of the steps.
pub fn teacher_weight(mixing: f64, fraction: f64, step: usize, steps: usize) -> f64 {
    let end = fraction * steps as f64;
    mixing * (1. - step as f64 / end).max(0.)
}

/// [`first_choice`] against the observed distribution blended with a teacher's: `(1 - w) q + w t`.
pub fn taught_choice(
    scores: &[f64],
    targets: &[f64],
    scale: f64,
    teacher: &[f64],
    weight: f64,
) -> Result<(f64, Vec<f64>)> {
    if weight == 0. {
        return first_choice(scores, targets, scale);
    }
    if teacher.len() != scores.len() || !(0. ..=1.).contains(&weight) {
        return Err(invalid(
            "a teacher needs one probability per candidate and a weight in [0, 1]",
        ));
    }
    let (_, adjoint) = first_choice(scores, targets, scale)?;
    let p = softmax(scores);
    // q = p - adjoint; the blended target and its cross-entropy gradient.
    let q: Vec<f64> = p
        .iter()
        .zip(&adjoint)
        .zip(teacher)
        .map(|((p, a), t)| (1. - weight) * (p - a) + weight * t)
        .collect();
    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_sum = top + scores.iter().map(|v| (v - top).exp()).sum::<f64>().ln();
    Ok((
        q.iter().zip(scores).map(|(q, s)| q * (log_sum - s)).sum(),
        p.iter().zip(&q).map(|(p, q)| p - q).collect(),
    ))
}

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
                // The gradient softmax aids the first stage of the training only.
                let weight =
                    teacher_weight(options.mixing, options.mixing_fraction, step - 1, steps);
                let teacher = crate::quantum::search::kernel::softmax(&state.gradients, 1.)?;
                let (_, adjoint) = taught_choice(
                    tape.value(output).as_slice(),
                    &state.targets,
                    target_scale * options.label_temperature,
                    &teacher,
                    weight,
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
        kind: Kind::Graph,
        parameters: Some(parameters),
        ranker: None,
        edit: None,
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
/// The descriptor rows of a state's candidates.
fn descriptor_rows(rows: &Candidates) -> Vec<[f64; DESCRIPTORS]> {
    let d = &rows.descriptors;
    (0..d.nrows())
        .map(|r| std::array::from_fn(|c| d[(r, c)]))
        .collect()
}

/// The pairwise ranker of `states` (their descriptor rows and targets).
pub fn ranker(states: &[&State], l2: f64) -> Result<Option<Ranker>> {
    let rows: Vec<Vec<[f64; DESCRIPTORS]>> =
        states.iter().map(|s| descriptor_rows(&s.rows)).collect();
    let samples: Vec<Sample> = states
        .iter()
        .zip(&rows)
        .map(|(s, d)| Sample {
            descriptors: d,
            targets: &s.targets,
        })
        .collect();
    Ranker::fit(&samples, l2)
}

/// Every kind of `options.kinds` trained on `states`, the proposal kind being the last one.
fn trained(context: &Context, states: &[&State], options: &Options) -> Result<Model> {
    let mut model = if options.kinds.contains(&Kind::Graph) {
        train(context, states, options)?
    } else {
        let (mean, std, _) = normalized(states);
        Model {
            kind: Kind::Ranker,
            parameters: None,
            ranker: None,
            edit: None,
            mean,
            std,
            ready: false,
            report: Value::Null,
            mixing: options.mixing,
            temperature: options.temperature,
        }
    };
    if options.kinds.contains(&Kind::Ranker) {
        model.ranker = ranker(states, options.l2)?;
    }
    model.kind = *options.kinds.last().expect("validated: at least one kind");
    Ok(model)
}

pub fn fit(context: &Context, dataset: &Dataset, options: &Options) -> Result<Model> {
    options.validate()?;
    let mut random = Rng(options.seed);
    let mut groups = dataset.groups.clone();
    shuffle(&mut groups, &mut random);
    let nfolds = options.folds.unwrap_or(groups.len()).min(groups.len());
    let mut curve = Vec::new();
    // Of the full-data point: the pooled within-state correlations per kind and of the gradient,
    // and per held-out group the averages that decide the wins.
    let mut pooled = BTreeMap::<Kind, Vec<f64>>::new();
    let mut full_gradient = Vec::new();
    let mut per_group: Vec<(BTreeMap<Kind, Option<f64>>, f64)> = Vec::new();
    if groups.len() >= 3 {
        for &fraction in &options.fractions {
            let mut folds = Vec::new();
            pooled.clear();
            full_gradient.clear();
            per_group.clear();
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
                let mut model = trained(context, &training, options)?;
                let mut by_group = BTreeMap::<String, (BTreeMap<Kind, Vec<f64>>, Vec<f64>)>::new();
                for state in dataset
                    .states
                    .iter()
                    .filter(|s| held_out.contains(&s.group.as_str()) && s.targets.len() >= 3)
                {
                    let observed: Vec<_> = state.changes.iter().map(|v| -v).collect();
                    let entry = by_group.entry(state.group.clone()).or_default();
                    for &kind in &options.kinds {
                        if kind == Kind::Ranker && model.ranker.is_none() {
                            continue;
                        }
                        model.kind = kind;
                        let scores = model.scores(context, &state.graph, &state.rows)?;
                        if let Some(value) = spearman(&scores, &observed) {
                            entry.0.entry(kind).or_default().push(value);
                            pooled.entry(kind).or_default().push(value);
                        }
                    }
                    if let Some(value) = spearman(&state.gradients, &observed) {
                        entry.1.push(value);
                        full_gradient.push(value);
                    }
                }
                let mut metrics = Vec::new();
                for (group, (learned, gradient)) in by_group {
                    let learned: BTreeMap<Kind, Option<f64>> = options
                        .kinds
                        .iter()
                        .map(|&k| (k, learned.get(&k).and_then(|v| average(v))))
                        .collect();
                    let gradient = average(&gradient);
                    let mut entry = json!({"group":group,"gradient":gradient});
                    for (kind, value) in &learned {
                        entry[kind.name()] = json!(value);
                    }
                    metrics.push(entry);
                    if let Some(baseline) = gradient {
                        per_group.push((learned, baseline));
                    }
                }
                folds.push(
                    json!({"held_out":held_out,"train_states":training.len(),"groups":metrics}),
                );
            }
            let mut within = json!({"gradient":{"spearman":average(&full_gradient)}});
            for kind in &options.kinds {
                within[kind.name()] =
                    json!({"spearman": pooled.get(kind).and_then(|v| average(v))});
            }
            curve.push(json!({"fraction":fraction,"folds":folds,"within_state":within}));
        }
    }
    let gradient = average(&full_gradient);
    let valid_groups = per_group.len();
    let metric =
        |value: Option<f64>| value.map_or_else(|| "undefined".into(), |v| format!("{v:.3}"));
    let mut passed: Vec<(f64, Kind, usize)> = Vec::new();
    let mut verdicts = Vec::new();
    for &kind in &options.kinds {
        let value = pooled.get(&kind).and_then(|v| average(v));
        let wins = per_group
            .iter()
            .filter(|(learned, baseline)| {
                learned
                    .get(&kind)
                    .copied()
                    .flatten()
                    .is_some_and(|v| v > *baseline)
            })
            .count();
        if valid_groups >= 3
            && wins * 2 > valid_groups
            && value.zip(gradient).is_some_and(|(v, b)| v >= b + 0.05)
        {
            passed.push((value.expect("checked"), kind, wins));
        }
        verdicts.push(format!(
            "{} {} ({wins} of {valid_groups} groups)",
            kind.name(),
            metric(value)
        ));
    }
    // The highest correlation; a tie goes to the simpler kind (the ranker).
    let chosen = passed
        .iter()
        .copied()
        .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));
    let reason = if groups.len() < 3 {
        format!(
            "{} groups; at least three are needed to hold one out and train on two",
            groups.len()
        )
    } else if valid_groups < 3 {
        format!("only {valid_groups} held-out groups have nonconstant states with three or more insertions; record chains with screen_insertions")
    } else if let Some((value, kind, wins)) = chosen {
        format!("{}: within-state Spearman {value:.3} against the gradient's {}, better in {wins} of {valid_groups} groups", kind.name(), metric(gradient))
    } else {
        format!("within-state Spearman: gradient {}, {}; a model needs the gradient's plus 0.05 and most groups", metric(gradient), verdicts.join(", "))
    };
    let states: Vec<_> = dataset.states.iter().collect();
    let mut model = trained(context, &states, options)?;
    let ready = chosen.is_some();
    if let Some((_, kind, _)) = chosen {
        model.kind = kind;
    }
    if model.kind == Kind::Ranker && model.ranker.is_none() {
        // No state holds a pair: the ranker cannot draw, and the model is not ready.
        model.kind = *options
            .kinds
            .iter()
            .find(|&&k| k == Kind::Graph)
            .unwrap_or(&Kind::Ranker);
    }
    model.ready = ready && (model.kind != Kind::Ranker || model.ranker.is_some());
    let kinds: Vec<_> = options.kinds.iter().map(|k| k.name()).collect();
    model.report = json!({"ready":model.ready,"kind":model.kind.name(),"reason":reason,"groups":groups.len(),"states":states.len(),
        "insertions":dataset.observations,"curve":curve,"kinds":kinds,"held_out_groups":valid_groups,
        "wins":chosen.map_or(0, |c| c.2)});
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

/// Feature width for multi-move edit ranking: 9 base descriptors (`DESCRIPTORS`), sequence
/// commutator overlap (`BG3`), compressed diagonal curvature (`BG4`), and normalized edit
/// position (`BG1`).
pub const EDIT_FEATURES: usize = DESCRIPTORS + 3;

/// Builds `EDIT_FEATURES`-column feature rows from `Candidates`, edit `positions`, and
/// optional second-order diagonal pool `curvatures`.
pub fn edit_rows_for(
    rows: &Candidates,
    positions: &[usize],
    curvatures: Option<&[f64]>,
) -> Vec<[f64; EDIT_FEATURES]> {
    let n_rows = rows.operators.len();
    let mut out = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let mut row = [0.; EDIT_FEATURES];
        for c in 0..DESCRIPTORS {
            row[c] = rows.descriptors[(r, c)];
        }
        row[9] = rows.sequence_overlap[r];
        let op = rows.operators[r];
        row[10] = curvatures
            .and_then(|curv| curv.get(op).copied())
            .map(compress)
            .unwrap_or(0.);
        let length = rows.descriptors[(r, 8)].max(1.);
        let pos = positions.get(r).copied().unwrap_or(0) as f64;
        row[11] = pos / length;
        out.push(row);
    }
    out
}

/// Builds `EDIT_FEATURES`-column feature rows for one multi-move [`EditState`].
pub fn edit_feature_rows(state: &EditState) -> Vec<[f64; EDIT_FEATURES]> {
    edit_rows_for(&state.rows, &state.positions, state.curvatures.as_deref())
}

/// Multi-move pairwise proposal heads trained over [`EditDataset`] (`BG1` + `BG3` + `BG4`).
#[derive(Clone, Debug, PartialEq)]
pub struct EditModel {
    pub insert: Option<PairwiseRanker<EDIT_FEATURES>>,
    pub delete: Option<PairwiseRanker<EDIT_FEATURES>>,
    pub replace: Option<PairwiseRanker<EDIT_FEATURES>>,
    pub l2: f64,
}

impl EditModel {
    /// Returns the fitted ranker for `kind`, if one had distinguishing training pairs.
    pub fn for_kind(&self, kind: EditKind) -> Option<&PairwiseRanker<EDIT_FEATURES>> {
        match kind {
            EditKind::Insert => self.insert.as_ref(),
            EditKind::Delete => self.delete.as_ref(),
            EditKind::Replace => self.replace.as_ref(),
        }
    }

    /// Scores every candidate move in `state` (higher means larger predicted energy decrease).
    pub fn score(&self, state: &EditState) -> Result<Vec<f64>> {
        let ranker = self.for_kind(state.kind).ok_or_else(|| {
            invalid(&format!(
                "edit model has no trained head for `{}` moves",
                state.kind.name()
            ))
        })?;
        Ok(edit_feature_rows(state)
            .iter()
            .map(|row| ranker.score(row))
            .collect())
    }
}

fn fit_edit_head(
    states: &[EditState],
    kind: EditKind,
    l2: f64,
) -> Result<Option<PairwiseRanker<EDIT_FEATURES>>> {
    let subset: Vec<(&EditState, Vec<[f64; EDIT_FEATURES]>)> = states
        .iter()
        .filter(|s| s.kind == kind)
        .map(|s| (s, edit_feature_rows(s)))
        .collect();
    let samples: Vec<PairwiseSample<'_, EDIT_FEATURES>> = subset
        .iter()
        .map(|(s, rows)| PairwiseSample {
            descriptors: rows,
            targets: &s.targets,
        })
        .collect();
    PairwiseRanker::fit(&samples, l2)
}

/// Fits independent per-move-kind `PairwiseRanker<EDIT_FEATURES>` heads over an [`EditDataset`].
pub fn fit_edit_model(dataset: &EditDataset, l2: f64) -> Result<EditModel> {
    Ok(EditModel {
        insert: fit_edit_head(&dataset.states, EditKind::Insert, l2)?,
        delete: fit_edit_head(&dataset.states, EditKind::Delete, l2)?,
        replace: fit_edit_head(&dataset.states, EditKind::Replace, l2)?,
        l2,
    })
}

fn edit_head_json(ranker: Option<&PairwiseRanker<EDIT_FEATURES>>) -> Value {
    ranker.map_or(Value::Null, |r| {
        json!({
            "weights": r.weights,
            "mean": r.mean,
            "std": r.std,
            "pairs": r.pairs,
        })
    })
}

impl EditModel {
    /// Parses a schema-v1 serialized [`EditModel`] from JSON.
    pub fn from_json(v: &Value) -> Result<Self> {
        if v["format"] != "mandacaru-edit-proposal-model"
            || v["version"] != 1
            || v["features"].as_u64() != Some(EDIT_FEATURES as u64)
        {
            return Err(invalid("unsupported edit proposal model schema"));
        }
        let l2 = v["l2"]
            .as_f64()
            .filter(|x| x.is_finite() && *x > 0.)
            .ok_or_else(|| invalid("edit proposal model needs positive finite l2"))?;
        let head = |key: &str| -> Result<Option<PairwiseRanker<EDIT_FEATURES>>> {
            match v.get(key) {
                Some(h) if !h.is_null() => Ok(Some(parse_ranker_json(h)?)),
                _ => Ok(None),
            }
        };
        Ok(Self {
            insert: head("insert")?,
            delete: head("delete")?,
            replace: head("replace")?,
            l2,
        })
    }

    /// Serializes this multi-move edit ranker to schema-v1 JSON.
    pub fn as_json(&self) -> Value {
        json!({
            "format": "mandacaru-edit-proposal-model",
            "version": 1,
            "features": EDIT_FEATURES,
            "l2": self.l2,
            "insert": edit_head_json(self.insert.as_ref()),
            "delete": edit_head_json(self.delete.as_ref()),
            "replace": edit_head_json(self.replace.as_ref()),
        })
    }
}

/// Parses a multi-move [`EditDataset`] from JSON and fits an [`EditModel`], returning JSON with
/// trained per-move heads alongside dataset group and move counts.
pub fn fit_edit_json(
    context: &Context,
    data: &str,
    group_by: &str,
    l2: f64,
    moves: &[EditKind],
) -> Result<String> {
    context.run(|| {
        let dataset = EditDataset::from_json(context, data, group_by, moves)?;
        let model = fit_edit_model(&dataset, l2)?;
        let payload = json!({
            "format": "mandacaru-edit-proposal-model",
            "version": 1,
            "features": EDIT_FEATURES,
            "l2": model.l2,
            "groups": dataset.groups,
            "counts": {
                "insert": dataset.counts[0],
                "delete": dataset.counts[1],
                "replace": dataset.counts[2],
            },
            "states": dataset.states.len(),
            "insert": edit_head_json(model.insert.as_ref()),
            "delete": edit_head_json(model.delete.as_ref()),
            "replace": edit_head_json(model.replace.as_ref()),
        });
        serde_json::to_string(&payload).map_err(|e| invalid(&e.to_string()))
    })
}
