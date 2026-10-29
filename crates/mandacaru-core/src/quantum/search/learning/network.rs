//! Two term/qubit message-passing layers and a first-choice scoring head.
use super::graph::{Candidates, Graph, Matrix};
use crate::{
    math::{
        autodiff::{Id, Tape},
        random::Rng,
    },
    product, Context, Error, Reservation, Result,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
pub const HIDDEN: usize = 32;
pub const LAYERS: usize = 2;
pub const HEAD: usize = 64;
pub const EMBEDDING: usize = 4 * HIDDEN + 3;
pub const GRAPH_WIDTH: usize = EMBEDDING + 3 * HIDDEN;
fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
#[derive(Debug)]
pub struct Parameters {
    pub values: BTreeMap<String, Matrix>,
    _memory: Reservation,
}
fn layout() -> Vec<(String, usize, usize, bool)> {
    let mut result = vec![
        ("qubit_in".into(), 4, HIDDEN, false),
        ("qubit_bias".into(), 1, HIDDEN, true),
        ("term_in".into(), 6, HIDDEN, false),
        ("term_bias".into(), 1, HIDDEN, true),
        ("w1".into(), GRAPH_WIDTH + 9, HEAD, false),
        ("b1".into(), 1, HEAD, true),
        ("w2".into(), HEAD, 1, false),
        ("w_desc".into(), 9, 1, true),
    ];
    for layer in 0..LAYERS {
        for letter in 0..3 {
            for name in ["to_term", "to_qubit"] {
                result.push((format!("{name}.{layer}.{letter}"), HIDDEN, HIDDEN, false));
            }
        }
        for name in ["term_update", "qubit_update"] {
            result.push((format!("{name}.{layer}"), 2 * HIDDEN, HIDDEN, false));
            result.push((format!("{name}_bias.{layer}"), 1, HIDDEN, true));
        }
    }
    for letter in 0..3 {
        result.push((format!("readout.{letter}"), HIDDEN, HIDDEN, false));
    }
    result
}
impl Parameters {
    pub fn new(context: &Context, seed: u64) -> Result<Self> {
        let shapes = layout();
        let cells = shapes.iter().try_fold(0usize, |n, (_, r, c, _)| {
            n.checked_add(product(&[*r, *c])?).ok_or(Error::Overflow)
        })?;
        let memory = context.reserve(
            product(&[cells, 8])?
                .checked_add(shapes.len() * 256)
                .ok_or(Error::Overflow)?,
        )?;
        let mut random = Rng(seed);
        let mut values = BTreeMap::new();
        for (name, rows, columns, zero) in shapes {
            let value = if zero {
                Matrix::zeros(rows, columns)
            } else {
                Matrix::from_fn(rows, columns, |_, _| random.normal() / (rows as f64).sqrt())
            };
            values.insert(name, value);
        }
        Ok(Self {
            values,
            _memory: memory,
        })
    }
    pub fn from_json(context: &Context, value: &Value) -> Result<Self> {
        let mut result = Self::new(context, 0)?;
        let entries = value
            .as_object()
            .ok_or_else(|| invalid("network weights must be an object"))?;
        if entries.len() != result.values.len() {
            return Err(invalid(
                "network weight layout differs from this model schema",
            ));
        }
        for (name, matrix) in &mut result.values {
            let row = &value[name];
            if row["rows"].as_u64() != Some(matrix.nrows() as u64)
                || row["columns"].as_u64() != Some(matrix.ncols() as u64)
            {
                return Err(invalid("network weight dimensions disagree"));
            }
            let data = row["values"]
                .as_array()
                .filter(|a| a.len() == matrix.len())
                .ok_or_else(|| invalid("network weight storage dimensions disagree"))?;
            for (target, value) in matrix.iter_mut().zip(data) {
                *target = value
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| invalid("network weights must be finite"))?;
            }
        }
        Ok(result)
    }
    pub fn as_json(&self) -> Value {
        Value::Object(self.values.iter().map(|(name,value)| (name.clone(),json!({"rows":value.nrows(),"columns":value.ncols(),"values":value.as_slice()}))).collect())
    }
    pub fn variables(&self, tape: &mut Tape) -> Result<BTreeMap<String, Id>> {
        self.values
            .iter()
            .map(|(name, value)| Ok((name.clone(), tape.leaf(value)?)))
            .collect()
    }
}
pub fn embed(tape: &mut Tape, p: &BTreeMap<String, Id>, graph: &Graph) -> Result<(Id, Id)> {
    let q = tape.constant(&graph.qubit)?;
    let h = tape.constant(&graph.term)?;
    let mut hq = tape.matmul(q, p["qubit_in"])?;
    hq = tape.add(hq, p["qubit_bias"])?;
    hq = tape.tanh(hq)?;
    let mut hp = tape.matmul(h, p["term_in"])?;
    hp = tape.add(hp, p["term_bias"])?;
    hp = tape.tanh(hp)?;
    let incidence = graph
        .incidence
        .iter()
        .map(|m| tape.constant(m))
        .collect::<Result<Vec<_>>>()?;
    let weighted = graph
        .weighted
        .iter()
        .map(|m| tape.constant(&m.transpose()))
        .collect::<Result<Vec<_>>>()?;
    for layer in 0..LAYERS {
        let mut message = None;
        for (letter, &edge) in incidence.iter().enumerate() {
            let transformed = tape.matmul(hq, p[&format!("to_term.{layer}.{letter}")])?;
            let part = tape.matmul(edge, transformed)?;
            message = Some(if let Some(prior) = message {
                tape.add(prior, part)?
            } else {
                part
            });
        }
        let message = tape.scale_rows(message.unwrap(), &graph.support)?;
        let combined = tape.concat(&[hp, message])?;
        let update = tape.matmul(combined, p[&format!("term_update.{layer}")])?;
        let update = tape.add(update, p[&format!("term_update_bias.{layer}")])?;
        let update = tape.tanh(update)?;
        hp = tape.add(hp, update)?;
        let mut message = None;
        for (letter, &edge) in weighted.iter().enumerate() {
            let transformed = tape.matmul(hp, p[&format!("to_qubit.{layer}.{letter}")])?;
            let part = tape.matmul(edge, transformed)?;
            message = Some(if let Some(prior) = message {
                tape.add(prior, part)?
            } else {
                part
            });
        }
        let message = tape.scale_rows(message.unwrap(), &graph.reach)?;
        let combined = tape.concat(&[hq, message])?;
        let update = tape.matmul(combined, p[&format!("qubit_update.{layer}")])?;
        let update = tape.add(update, p[&format!("qubit_update_bias.{layer}")])?;
        let update = tape.tanh(update)?;
        hq = tape.add(hq, update)?;
    }
    let mut za = None;
    for letter in 0..3 {
        let transformed = tape.matmul(hq, p[&format!("readout.{letter}")])?;
        let edge = tape.constant(&graph.pool[letter])?;
        let part = tape.matmul(edge, transformed)?;
        za = Some(if let Some(prior) = za {
            tape.add(prior, part)?
        } else {
            part
        });
    }
    let nq = tape.value(hq).nrows();
    let average = tape.constant(&Matrix::from_element(1, nq, 1. / nq as f64))?;
    let ones = tape.constant(&Matrix::from_element(1, nq, 1.))?;
    let mean_q = tape.matmul(average, hq)?;
    let sum_q = tape.matmul(ones, hq)?;
    let total = graph.term_weights.iter().sum::<f64>() + 1e-12;
    let weights = tape.constant(&Matrix::from_row_slice(
        1,
        graph.term_weights.len(),
        &graph.term_weights,
    ))?;
    let weights_normalized =
        tape.constant(&Matrix::from_fn(1, graph.term_weights.len(), |_, c| {
            graph.term_weights[c] / total
        }))?;
    let mean_p = tape.matmul(weights_normalized, hp)?;
    let sum_p = tape.matmul(weights, hp)?;
    let global = tape.constant(&graph.global)?;
    Ok((
        tape.concat(&[mean_q, sum_q, mean_p, sum_p, global])?,
        za.unwrap(),
    ))
}
pub fn scores(
    tape: &mut Tape,
    p: &BTreeMap<String, Id>,
    graph: &Graph,
    rows: &Candidates,
    mean: &[f64; 9],
    std: &[f64; 9],
) -> Result<Id> {
    let (zh, za) = embed(tape, p, graph)?;
    head(tape, p, zh, za, rows, mean, std)
}
pub fn head(
    tape: &mut Tape,
    p: &BTreeMap<String, Id>,
    zh: Id,
    za: Id,
    rows: &Candidates,
    mean: &[f64; 9],
    std: &[f64; 9],
) -> Result<Id> {
    if std.iter().any(|v| !v.is_finite() || *v <= 0.) || mean.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "descriptor normalization requires finite means and positive scales",
        ));
    }
    let count = rows.operators.len();
    let zh = tape.repeat(zh, count)?;
    let occupancy = tape.constant(&rows.occupancy)?;
    let positional = tape.constant(&rows.positional)?;
    let mean_c = tape.matmul(occupancy, za)?;
    let mean_pos = tape.matmul(positional, za)?;
    let mean_c = tape.repeat(mean_c, count)?;
    let mean_pos = tape.repeat(mean_pos, count)?;
    let candidate = tape.gather(za, &rows.operators)?;
    let features = tape.concat(&[zh, mean_c, mean_pos, candidate])?;
    let normalized = tape.normalize_rows(features, 1e-6)?;
    let descriptors = tape.constant(&Matrix::from_fn(count, 9, |r, c| {
        (rows.descriptors[(r, c)] - mean[c]) / std[c]
    }))?;
    let input = tape.concat(&[normalized, descriptors])?;
    let hidden = tape.matmul(input, p["w1"])?;
    let hidden = tape.add(hidden, p["b1"])?;
    let hidden = tape.tanh(hidden)?;
    let nonlinear = tape.matmul(hidden, p["w2"])?;
    let linear = tape.matmul(descriptors, p["w_desc"])?;
    tape.add(linear, nonlinear)
}
