//! Validated insertion observations grouped by physical source state.
use super::graph::{compress, Candidates, Graph};
use crate::{product, Context, Error, Reservation, Result};
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc};
fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
pub struct State {
    pub graph: Arc<Graph>,
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
        if !["molecule", "hamiltonian", "trajectory"].contains(&group_by) {
            return Err(invalid(
                "training groups are molecule, hamiltonian or trajectory",
            ));
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
        let mut grouped = BTreeMap::<(String, String, String), Vec<&Value>>::new();
        for record in records {
            if record["format"] != "mandacaru-search-edit" || record["version"] != 1 {
                return Err(invalid("unsupported insertion observation format"));
            }
            if record["move"] != "insert" {
                continue;
            }
            let identity = record["problem_id"]
                .as_str()
                .ok_or_else(|| invalid("insertion observations need a problem identifier"))?;
            let metadata = problems.get(identity).ok_or_else(|| {
                invalid("an insertion observation refers to a missing Hamiltonian")
            })?;
            let group = match group_by {
                "molecule" => {
                    if metadata["composition"].is_array() {
                        serde_json::to_string(&metadata["composition"]).unwrap()
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
            .unwrap();
            grouped
                .entry((identity.to_owned(), group, key))
                .or_default()
                .push(record);
        }
        let mut graphs = BTreeMap::<String, Arc<Graph>>::new();
        let mut states = Vec::new();
        let mut observations = 0;
        for ((identity, group, _), records) in grouped {
            if records.len() < 2 {
                continue;
            }
            let graph = if let Some(graph) = graphs.get(&identity) {
                graph.clone()
            } else {
                let graph = Arc::new(Graph::from_metadata(context, &problems[&identity])?);
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
                .ok_or_else(|| invalid("insertions need source-state pool gradients"))?
                .iter()
                .map(|v| number(v, "pool gradient"))
                .collect::<Result<Vec<_>>>()?;
            let energy = number(&first["source_energy"], "source energy")?;
            let operators = records
                .iter()
                .map(|row| {
                    row["operator"]
                        .as_u64()
                        .and_then(|v| usize::try_from(v).ok())
                        .ok_or_else(|| invalid("insertions need an operator index"))
                })
                .collect::<Result<Vec<_>>>()?;
            let changes = records
                .iter()
                .map(|row| number(&row["delta_energy"], "insertion energy change"))
                .collect::<Result<Vec<_>>>()?;
            for (record, &change) in records.iter().zip(&changes) {
                let proposed = number(&record["energy"], "proposed energy")?;
                if (proposed - energy - change).abs()
                    > 1e-12 * proposed.abs().max(energy.abs()).max(1.)
                {
                    return Err(invalid("recorded insertion energy change disagrees with its source and proposed energies"));
                }
            }
            if records
                .iter()
                .any(|row| row["source_gradients"] != first["source_gradients"])
            {
                return Err(invalid(
                    "one physical source state has inconsistent gradient observations",
                ));
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
                return Err(invalid("compressed insertion targets overflow"));
            }
            let observed_gradients = operators.iter().map(|&i| gradients[i].abs()).collect();
            let state_memory = context.reserve(product(&[records.len(), 32])?)?;
            observations += records.len();
            states.push(State {
                graph,
                rows,
                targets,
                changes,
                gradients: observed_gradients,
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
            observations,
            _memory: memory,
        })
    }
}
