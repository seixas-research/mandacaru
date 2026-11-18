//! Validated insertion observations grouped by physical source state.
use super::graph::{compress, BipartiteGraph, Candidates};
use crate::{product, Context, Error, Reservation, Result};
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc};
fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
pub struct State {
    pub graph: Arc<BipartiteGraph>,
    pub rows: Candidates,
    pub targets: Vec<f64>,
    pub changes: Vec<f64>,
    pub gradients: Vec<f64>,
    pub group: String,
    _memory: Reservation,
}
pub struct Dataset {
    pub states: Vec<State>,
    pub groups: Vec<String>,
    pub observations: usize,
    _memory: Reservation,
}
/// Edit move kind recorded in VASQA/VALSQA simulation stores (`BG1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EditKind {
    Insert,
    Delete,
    Replace,
}
impl EditKind {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "insert" => Ok(EditKind::Insert),
            "delete" => Ok(EditKind::Delete),
            "replace" => Ok(EditKind::Replace),
            _ => Err(invalid("move kind must be insert, delete, or replace")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            EditKind::Insert => "insert",
            EditKind::Delete => "delete",
            EditKind::Replace => "replace",
        }
    }
}
/// Multi-move validated observation group (`insert`, `delete`, `replace`) with optional
/// second-order diagonal curvature descriptors (`BG1` + `BG4`).
pub struct EditState {
    pub kind: EditKind,
    pub graph: Arc<BipartiteGraph>,
    pub rows: Candidates,
    pub positions: Vec<usize>,
    pub targets: Vec<f64>,
    pub changes: Vec<f64>,
    pub gradients: Vec<f64>,
    pub curvatures: Option<Vec<f64>>,
    pub group: String,
    _memory: Reservation,
}
pub struct EditDataset {
    pub states: Vec<EditState>,
    pub groups: Vec<String>,
    pub counts: [usize; 3],
    _memory: Reservation,
}
impl EditDataset {
    pub fn from_json(
        context: &Context,
        source: &str,
        group_by: &str,
        moves: &[EditKind],
    ) -> Result<Self> {
        if !["molecule", "hamiltonian", "trajectory"].contains(&group_by) {
            return Err(invalid(
                "training groups are molecule, hamiltonian or trajectory",
            ));
        }
        if moves.is_empty() {
            return Err(invalid("edit dataset requires at least one move kind"));
        }
        let _parse = context.reserve(product(&[source.len(), 32])?)?;
        let input: Value = serde_json::from_str(source).map_err(|e| invalid(&e.to_string()))?;
        let problems = input["problems"]
            .as_object()
            .ok_or_else(|| invalid("training data needs problem metadata"))?;
        let records = input["records"]
            .as_array()
            .ok_or_else(|| invalid("training data needs recorded edits"))?;
        let memory = context.reserve(product(&[records.len(), 128])?)?;
        let mut grouped = BTreeMap::<(String, String, EditKind, String), Vec<&Value>>::new();
        for record in records {
            if record["format"] != "mandacaru-search-edit" || record["version"] != 1 {
                return Err(invalid("unsupported search edit observation format"));
            }
            let kind = EditKind::parse(record["move"].as_str().unwrap_or(""))?;
            if !moves.contains(&kind) {
                continue;
            }
            let identity = record["problem_id"]
                .as_str()
                .ok_or_else(|| invalid("edit observations need a problem identifier"))?;
            let metadata = problems
                .get(identity)
                .ok_or_else(|| invalid("an edit observation refers to a missing Hamiltonian"))?;
            let group = match group_by {
                "molecule" => {
                    if metadata["composition"].is_array() {
                        serde_json::to_string(&metadata["composition"])
                            .map_err(|_| invalid("edit dataset composition must serialize"))?
                    } else {
                        identity.to_owned()
                    }
                }
                "trajectory" => record["trajectory"].as_str().unwrap_or(identity).to_owned(),
                _ => identity.to_owned(),
            };
            let key = serde_json::to_string(&(
                &record["source"],
                &record["source_parameters"],
                &record["source_energy"],
            ))
            .map_err(|_| invalid("edit state key must serialize"))?;
            grouped
                .entry((identity.to_owned(), group, kind, key))
                .or_default()
                .push(record);
        }
        let mut graphs = BTreeMap::<String, Arc<BipartiteGraph>>::new();
        let mut states = Vec::new();
        let mut counts = [0usize; 3];
        for ((identity, group, kind, _), records) in grouped {
            if records.len() < 2 {
                continue;
            }
            let graph = if let Some(graph) = graphs.get(&identity) {
                graph.clone()
            } else {
                let graph = Arc::new(BipartiteGraph::from_metadata(
                    context,
                    &problems[&identity],
                )?);
                graphs.insert(identity, graph.clone());
                graph
            };
            let first = records[0];
            let sequence = indices(&first["source"])?;
            let parameters = first["source_parameters"]
                .as_array()
                .filter(|a| a.len() == sequence.len())
                .ok_or_else(|| invalid("source parameters must match the recorded sequence"))?;
            for value in parameters {
                number(value, "source parameter")?;
            }
            let gradients = first["source_gradients"]
                .as_array()
                .ok_or_else(|| invalid("edit observations need source-state pool gradients"))?
                .iter()
                .map(|v| number(v, "pool gradient"))
                .collect::<Result<Vec<_>>>()?;
            let curvatures = first
                .get("source_curvatures")
                .and_then(|c| c.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|v| number(v, "pool curvature"))
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?;
            let energy = number(&first["source_energy"], "source energy")?;
            let mut positions = Vec::with_capacity(records.len());
            let mut operators = Vec::with_capacity(records.len());
            let mut changes = Vec::with_capacity(records.len());
            for row in &records {
                if row["source_gradients"] != first["source_gradients"] {
                    return Err(invalid(
                        "one physical source state has inconsistent gradient observations",
                    ));
                }
                let pos = match (kind, row.get("position").and_then(|v| v.as_u64())) {
                    (_, Some(v)) => usize::try_from(v)
                        .map_err(|_| invalid("edit observations need a valid position"))?,
                    (EditKind::Insert, None) if row["position"].is_null() => sequence.len(),
                    _ => return Err(invalid("edit observations need a valid position")),
                };
                let op = match kind {
                    EditKind::Delete => {
                        if pos >= sequence.len() {
                            return Err(invalid(
                                "delete observation position exceeds source length",
                            ));
                        }
                        sequence[pos]
                    }
                    EditKind::Insert | EditKind::Replace => row["operator"]
                        .as_u64()
                        .and_then(|v| usize::try_from(v).ok())
                        .ok_or_else(|| {
                            invalid("insert/replace observations need an operator index")
                        })?,
                };
                let change = number(&row["delta_energy"], "edit energy change")?;
                let proposed = number(&row["energy"], "proposed energy")?;
                if (proposed - energy - change).abs()
                    > 1e-12 * proposed.abs().max(energy.abs()).max(1.)
                {
                    return Err(invalid(
                        "recorded edit energy change disagrees with source/proposed",
                    ));
                }
                positions.push(pos);
                operators.push(op);
                changes.push(change);
            }
            let rows = Candidates::new(
                context,
                &graph,
                &sequence,
                &operators,
                &gradients,
                energy - graph.reference_energy,
            )?;
            let targets: Vec<_> = changes.iter().map(|&v| compress(v)).collect();
            if targets.iter().any(|v| !v.is_finite()) {
                return Err(invalid("compressed edit targets overflow"));
            }
            let observed_gradients = operators.iter().map(|&i| gradients[i].abs()).collect();
            let state_memory = context.reserve(product(&[records.len(), 32])?)?;
            counts[match kind {
                EditKind::Insert => 0,
                EditKind::Delete => 1,
                EditKind::Replace => 2,
            }] += records.len();
            states.push(EditState {
                kind,
                graph,
                rows,
                positions,
                targets,
                changes,
                gradients: observed_gradients,
                curvatures,
                group,
                _memory: state_memory,
            });
        }
        let groups = states
            .iter()
            .map(|s| (s.group.clone(), ()))
            .collect::<BTreeMap<_, _>>()
            .into_keys()
            .collect();
        Ok(Self {
            states,
            groups,
            counts,
            _memory: memory,
        })
    }
}
fn number(value: &Value, name: &str) -> Result<f64> {
    value
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| invalid(&format!("{name} must be finite")))
}
fn indices(value: &Value) -> Result<Vec<usize>> {
    value
        .as_array()
        .ok_or_else(|| invalid("source sequence must be an array"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| invalid("source indices must be nonnegative"))
        })
        .collect()
}
impl Dataset {
    pub fn from_json(context: &Context, source: &str, group_by: &str) -> Result<Self> {
        let parsed = EditDataset::from_json(context, source, group_by, &[EditKind::Insert])?;
        let states: Vec<State> = parsed
            .states
            .into_iter()
            .map(|s| State {
                graph: s.graph,
                rows: s.rows,
                targets: s.targets,
                changes: s.changes,
                gradients: s.gradients,
                group: s.group,
                _memory: s._memory,
            })
            .collect();
        Ok(Self {
            states,
            groups: parsed.groups,
            observations: parsed.counts[0],
            _memory: parsed._memory,
        })
    }
}
