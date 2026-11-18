use super::invalid;
use crate::quantum::{amplitude::State, evolution, solver::SolveResult};
use crate::{Reservation, Result};
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct ReplicaState {
    pub sequence: Vec<usize>,
    pub parameters: Vec<f64>,
    pub rng: u64,
}
#[derive(Clone, Debug)]
pub struct SearchCheckpoint {
    pub step: usize,
    pub total_steps: usize,
    pub seeded_kept: bool,
    pub replicas: Vec<ReplicaState>,
    pub best_sequence: Vec<usize>,
    pub best_parameters: Vec<f64>,
    pub best_energy_sequence: Vec<usize>,
    pub best_energy_parameters: Vec<f64>,
    pub swap_rng: u64,
    pub screening_rng: u64,
    pub acceptance: [(usize, usize); 3],
    pub swap_acceptance: (usize, usize),
}
impl SearchCheckpoint {
    pub fn to_json(&self) -> Value {
        let replicas: Vec<Value> = self
            .replicas
            .iter()
            .map(|r| {
                json!({
                    "sequence": r.sequence,
                    "parameters": r.parameters,
                    "rng": r.rng.to_string(),
                })
            })
            .collect();
        json!({
            "version": 1,
            "step": self.step,
            "total_steps": self.total_steps,
            "seeded_kept": self.seeded_kept,
            "replicas": replicas,
            "best_sequence": self.best_sequence,
            "best_parameters": self.best_parameters,
            "best_energy_sequence": self.best_energy_sequence,
            "best_energy_parameters": self.best_energy_parameters,
            "swap_rng": self.swap_rng.to_string(),
            "screening_rng": self.screening_rng.to_string(),
            "acceptance": self.acceptance,
            "swap_acceptance": self.swap_acceptance,
        })
    }
    pub fn from_json(value: &Value, pool_len: usize, max_length: usize) -> Result<Self> {
        if value["version"] != 1 {
            return Err(invalid("unsupported search checkpoint version"));
        }
        let u64_field = |v: &Value, name: &str| -> Result<u64> {
            match v {
                Value::String(s) => s
                    .parse::<u64>()
                    .map_err(|_| invalid(&format!("invalid {name} rng seed"))),
                Value::Number(n) => n
                    .as_u64()
                    .ok_or_else(|| invalid(&format!("invalid {name} rng seed"))),
                _ => Err(invalid(&format!("missing {name} rng seed"))),
            }
        };
        let check_seq = |seq: &Value, params: &Value| -> Result<(Vec<usize>, Vec<f64>)> {
            let seq: Vec<usize> = serde_json::from_value(seq.clone())
                .map_err(|_| invalid("invalid sequence in search checkpoint"))?;
            let params: Vec<f64> = serde_json::from_value(params.clone())
                .map_err(|_| invalid("invalid parameters in search checkpoint"))?;
            if seq.len() != params.len()
                || seq.len() > max_length
                || seq.iter().any(|&i| i >= pool_len)
                || params.iter().any(|p| !p.is_finite())
            {
                return Err(invalid(
                    "search checkpoint sequence or parameters disagree with pool or max_length",
                ));
            }
            Ok((seq, params))
        };
        let step = value["step"]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| invalid("missing search checkpoint step"))?;
        let total_steps = value["total_steps"]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| invalid("missing search checkpoint total_steps"))?;
        let seeded_kept = value["seeded_kept"].as_bool().unwrap_or(false);
        let raw_replicas = value["replicas"]
            .as_array()
            .filter(|a| !a.is_empty())
            .ok_or_else(|| invalid("search checkpoint needs at least one replica"))?;
        let mut replicas = Vec::with_capacity(raw_replicas.len());
        for entry in raw_replicas {
            let (sequence, parameters) = check_seq(&entry["sequence"], &entry["parameters"])?;
            let rng = u64_field(&entry["rng"], "replica")?;
            replicas.push(ReplicaState {
                sequence,
                parameters,
                rng,
            });
        }
        let (best_sequence, best_parameters) =
            check_seq(&value["best_sequence"], &value["best_parameters"])?;
        let (best_energy_sequence, best_energy_parameters) = check_seq(
            &value["best_energy_sequence"],
            &value["best_energy_parameters"],
        )?;
        let swap_rng = u64_field(&value["swap_rng"], "swap")?;
        let screening_rng = u64_field(&value["screening_rng"], "screening")?;
        let acceptance: [(usize, usize); 3] =
            serde_json::from_value(value["acceptance"].clone())
                .map_err(|_| invalid("invalid acceptance counters in search checkpoint"))?;
        let swap_acceptance: (usize, usize) =
            serde_json::from_value(value["swap_acceptance"].clone())
                .map_err(|_| invalid("invalid swap_acceptance in search checkpoint"))?;
        Ok(Self {
            step,
            total_steps,
            seeded_kept,
            replicas,
            best_sequence,
            best_parameters,
            best_energy_sequence,
            best_energy_parameters,
            swap_rng,
            screening_rng,
            acceptance,
            swap_acceptance,
        })
    }
}
pub struct SearchResult {
    pub solution: SolveResult,
    pub records: Vec<Value>,
    pub steps: Vec<Value>,
    pub acceptance: [(usize, usize); 3],
    pub optimal_cost: f64,
    pub best_energy: f64,
    pub best_energy_indices: Vec<usize>,
    pub best_energy_parameters: Vec<f64>,
    pub final_energy: f64,
    pub final_indices: Vec<usize>,
    pub final_parameters: Vec<f64>,
    pub num_architectures: usize,
    pub num_screenings: usize,
    /// Operator importances computed (one per position of a weighed state), not in `history`.
    /// Each is the gradient entry of one evaluation with that angle zeroed, taken from the
    /// shared prefix and the truncated sweep of [`evolution::importances`], not from a full
    /// evaluation.
    pub num_importance_evaluations: usize,
    /// What the importances did in whole-state actions, against
    /// [`evolution::Work::separate_evaluations`] per weighed state (one full evaluation each).
    pub importance_work: evolution::Work,
    /// Gradients the optimizer asked for: the others of the `history` entries (one energy per
    /// objective call) were trial points its line search rejected on the energy alone.
    pub num_gradient_evaluations: usize,
    pub failures: Vec<(usize, String)>,
    /// One row per attempted replica swap (empty with one replica): the step, the pair, their
    /// temperatures and costs, the log acceptance and the decision.
    pub swaps: Vec<Value>,
    /// Accepted and attempted replica swaps.
    pub swap_acceptance: (usize, usize),
    /// Resumable Markov chain state at completion (`step == max_steps`).
    pub checkpoint: SearchCheckpoint,
    pub(super) _memory: Reservation,
}
/// Output of a Subspace-VASQA [`QuantumProblem::search_subspace`] run.
///
/// The Markov chain minimizes the weighted ensemble cost
/// $F(C) = \sum_{j=0}^{K-1} w_j \langle \psi_j(C) | H | \psi_j(C) \rangle + \lambda L$,
/// and finishes with a Rayleigh–Ritz diagonalization of $H$ inside the span of the $K$ prepared
/// states to return orthogonal variational eigenstates ($E_0 \le E_1 \le \dots \le E_{K-1}$).
/// `solution.energy` holds $E_0$ and `solution.state` holds the corresponding rotated ground
/// eigenstate; `best_energy` and `optimal_cost` record the weighted ensemble energy and cost
/// optimized by the search.
pub struct SubspaceSearchResult {
    pub solution: SolveResult,
    /// Rayleigh–Ritz sorted eigenstate energies $E_0 \le E_1 \le \dots \le E_{K-1}$.
    pub energies: Vec<f64>,
    /// Bare expectation energies $\langle \psi_j | H | \psi_j \rangle$ prior to subspace rotation.
    pub ensemble_energies: Vec<f64>,
    /// Orthonormal rotated eigenstates stacked row-major (`k x d`).
    pub states: State,
    /// Ordering of the reference determinants sorted by bare expectation energy.
    pub reference_indices: Vec<usize>,
    pub records: Vec<Value>,
    pub steps: Vec<Value>,
    pub acceptance: [(usize, usize); 3],
    pub optimal_cost: f64,
    pub best_energy: f64,
    pub best_energy_indices: Vec<usize>,
    pub best_energy_parameters: Vec<f64>,
    pub final_energy: f64,
    pub final_indices: Vec<usize>,
    pub final_parameters: Vec<f64>,
    pub num_architectures: usize,
    pub num_screenings: usize,
    pub num_importance_evaluations: usize,
    pub importance_work: evolution::Work,
    pub num_gradient_evaluations: usize,
    pub failures: Vec<(usize, String)>,
    pub swaps: Vec<Value>,
    pub swap_acceptance: (usize, usize),
    pub checkpoint: SearchCheckpoint,
    pub(super) _memory: Reservation,
}
