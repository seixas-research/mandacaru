//! Quenched Metropolis–Hastings search over variational operator sequences (VASQA).
//!
//! Each Markov chain step proposes an architecture edit (`Insert`, `Delete`, or `Replace`) over a
//! bounded operator sequence $C = (\mu_1, \dots, \mu_L)$ with $0 \le L \le L_{\max}$, relaxes the
//! continuous parameters $\theta$ to obtain $E_{\mathrm{opt}}(C)$, forms the regularized cost
//! $F(C) = E_{\mathrm{opt}}(C) + \lambda L$, and accepts the edit with Metropolis–Hastings
//! probability:
//!
//! $$\alpha(C \to C') = \min\!\left(1, \exp\!\left(-\frac{F(C') - F(C)}{T}\right) \frac{q(C \mid C')}{q(C' \mid C)}\right)$$
use super::kernel::{self, Action, Kernel, Move};
use super::learning::{
    graph::{BipartiteGraph, Candidates},
    training::{self, Model},
};
use crate::quantum::{
    amplitude::State,
    evolution, parallel,
    solver::{self, Execution},
    subspace, Operator,
};
use crate::{
    math::random::Rng,
    product,
    quantum::{optimizer, QuantumProblem, SolveResult, C64},
    Buffer, Context, Error, Reservation, Result,
};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    time::Instant,
};
fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
/// Configuration for a VASQA [`QuantumProblem::search`] run.
///
/// # Examples
///
/// ```
/// use mandacaru_core::{
///     quantum::{search::vasqa::SearchOptions, QuantumProblem, Encoding, Operator, C64},
///     Context,
/// };
///
/// let ctx = Context::new(Some(1), 64 << 20).unwrap();
/// let h = Operator::new(
///     &ctx,
///     4,
///     &[
///         ("ZIII".into(), C64::new(0.3, 0.)),
///         ("XXII".into(), C64::new(0.2, 0.)),
///         ("YYII".into(), C64::new(0.2, 0.)),
///     ],
/// )
/// .unwrap();
/// let problem = QuantumProblem::new(
///     &ctx,
///     Encoding::new("jordan_wigner", 4, [1, 0]).unwrap(),
///     h,
///     true,
///     false,
///     true,
/// )
/// .unwrap();
/// let options = SearchOptions {
///     max_steps: 20,
///     max_length: 3,
///     seed: 7,
///     ..Default::default()
/// };
/// let result = problem.search(&options, &[], &[]).unwrap();
/// assert!((result.solution.energy + 0.5).abs() < 1e-8);
/// ```
#[derive(Clone, Debug)]
pub struct SearchOptions {
    pub max_steps: usize,
    pub transfer_steps: Option<usize>,
    pub max_length: usize,
    pub move_weights: [f64; 3],
    pub temperatures: [f64; 2],
    pub length_penalty: f64,
    pub proposal_temperature: f64,
    pub uniform: bool,
    pub warm_start: bool,
    pub replace_inherit: bool,
    pub screen_insertions: usize,
    pub record: bool,
    pub seed: u64,
    pub optimizer: optimizer::Options,
    /// Chains run side by side (parallel tempering, Hukushima and Nemoto 1996). Replica `r` runs at
    /// `replica_spacing^r` times the schedule's temperature, so replica 0 is the chain of
    /// `replicas = 1` at the same seed; after every step one adjacent pair, drawn uniformly, swaps
    /// its states with probability `min(1, exp((1/T_r - 1/T_{r+1}) (F_r - F_{r+1})))`. The product
    /// of the replicas' targets is invariant under the swap, so with `warm_start = false` and a
    /// fixed temperature every replica keeps its own target; otherwise the exchange is a heuristic
    /// of the optimizer, like the annealed chain itself. The returned state is the lowest cost any
    /// replica evaluated; `final` is replica 0's.
    pub replicas: usize,
    /// The ratio of adjacent replica temperatures (at least 1).
    pub replica_spacing: f64,
    /// Carry the SLSQP Hessian or BFGS inverse-Hessian approximation across warm-started
    /// architecture proposals (`Insert` inserts a unit diagonal row/col, `Delete` drops or
    /// Schur-complements the removed coordinate, `Replace` resets the replaced coordinate unless
    /// `replace_inherit` is set). Default `false` preserves uncoupled identity starts.
    pub carry_hessian: bool,
    /// Memoize repeated `(sequence, initial_parameter_bits)` relaxations in warm-start chains
    /// (without curvature carry), skipping redundant optimizer invocations when a proposal
    /// revisits the exact same sequence and starting angles. Default `true`.
    pub memoize_warm: bool,
    /// Enable zero-`D`-allocation sparse energy evaluation and sparse tangent-vector adjoint
    /// gradients on large registers (`true`, new default). Set `false` for dense buffers.
    pub sparse_kernels: bool,
    /// Two-stage delayed-rejection screening using the exact upper bound `B = -dF/T - ln q_fwd`
    /// (`false`, default, so recorded steps always include exact `log_q_reverse`). Set `true` to
    /// skip reverse `weigh`/`screen` computation on proposals already rejected against `B`,
    /// preserving exact detailed balance while cutting proposal overhead on high-rejection chains.
    pub early_reject: bool,
    /// Drop near-zero operators (`|theta| <= 1e-14`) from the returned best/final circuits after
    /// search completes (`false`, default). Preserves state and energy up to roundoff while
    /// removing redundant zero-angle identity gates from stochastic search sequences.
    pub strip_zero_operators: bool,
    /// Optional colder annealing schedule `[initial, final]` (in Hartree) used when a transferred
    /// or seeded starting ansatz is kept (`Plan row B2`, default `None` uses [`Self::temperatures`]).
    /// Prevents reheating an already relaxed seed while exploring local edits.
    pub seeded_temperatures: Option<[f64; 2]>,
    /// Optional cap on the number of largest-magnitude Hamiltonian Pauli terms used to screen
    /// candidate pool gradients for operator proposals (`None`, default, uses the exact
    /// Hamiltonian). Exact energies are always used during relaxation and Metropolis acceptance.
    pub screen_hamiltonian_terms: Option<usize>,
    /// Record exact diagonal pool curvatures $\langle\psi|[[H,G_\mu],G_\mu]|\psi\rangle$ as
    /// `"source_curvatures"` on each recorded edit step for `BG4` training (`false`, default).
    pub record_curvatures: bool,
}
impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            max_steps: 200,
            transfer_steps: None,
            max_length: 20,
            move_weights: [1.; 3],
            temperatures: [0.1 / 27.211386245988, 0.001 / 27.211386245988],
            length_penalty: 0.,
            proposal_temperature: 0.1,
            uniform: false,
            warm_start: true,
            replace_inherit: false,
            screen_insertions: 0,
            record: false,
            seed: 0,
            optimizer: optimizer::Options::default(),
            replicas: 1,
            replica_spacing: 2.,
            carry_hessian: false,
            memoize_warm: true,
            sparse_kernels: true,
            early_reject: false,
            strip_zero_operators: false,
            seeded_temperatures: None,
            screen_hamiltonian_terms: None,
            record_curvatures: false,
        }
    }
}
impl SearchOptions {
    fn validate_schedule(&self, temperatures: [f64; 2]) -> Result<()> {
        if temperatures.iter().any(|v| !v.is_finite() || *v < 0.)
            || (temperatures[0] != temperatures[1] && temperatures.contains(&0.))
        {
            return Err(invalid("search needs finite nonnegative temperatures and length penalty, positive proposal temperature, and recording for screened insertions"));
        }
        if self.replicas > 1
            && (temperatures.contains(&0.)
                || temperatures.iter().any(|t| {
                    !(t * self.replica_spacing.powi(self.replicas as i32 - 1)).is_finite()
                }))
        {
            return Err(invalid("replica exchange needs at least one replica, a finite temperature spacing of at least 1, positive temperatures and no screened insertions"));
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        self.optimizer.validate()?;
        Kernel::new(2, self.max_length, self.move_weights)?;
        if !self.length_penalty.is_finite()
            || self.length_penalty < 0.
            || !self.proposal_temperature.is_finite()
            || self.proposal_temperature <= 0.
            || (self.screen_insertions > 0 && !self.record)
        {
            return Err(invalid("search needs finite nonnegative temperatures and length penalty, positive proposal temperature, and recording for screened insertions"));
        }
        if self.replicas == 0
            || !self.replica_spacing.is_finite()
            || self.replica_spacing < 1.
            || (self.replicas > 1 && self.screen_insertions > 0)
        {
            return Err(invalid("replica exchange needs at least one replica, a finite temperature spacing of at least 1, positive temperatures and no screened insertions"));
        }
        self.validate_schedule(self.temperatures)?;
        if let Some(seeded) = self.seeded_temperatures {
            self.validate_schedule(seeded)?;
        }
        if self.screen_hamiltonian_terms == Some(0) {
            return Err(invalid(
                "screen_hamiltonian_terms must be positive when set",
            ));
        }
        Ok(())
    }
    fn active_temperatures(&self, seeded_kept: bool) -> [f64; 2] {
        self.seeded_temperatures
            .filter(|_| seeded_kept)
            .unwrap_or(self.temperatures)
    }
    fn replica_temperature_on(
        &self,
        schedule: [f64; 2],
        step: usize,
        steps: usize,
        replica: usize,
    ) -> f64 {
        let base = if steps <= 1 || schedule[0] == schedule[1] {
            schedule[0]
        } else {
            schedule[0] * (schedule[1] / schedule[0]).powf(step as f64 / (steps - 1) as f64)
        };
        if replica == 0 {
            base
        } else {
            base * self.replica_spacing.powi(replica as i32)
        }
    }
}
/// The pool screen of a state and the operator proposal drawn from it. `gradients` is empty when
/// nothing but the proposal reads them (a uniform chain that records nothing).
#[derive(Clone)]
struct Screen {
    gradients: Vec<f64>,
    probabilities: Vec<f64>,
    curvatures: Option<Vec<f64>>,
}
/// The position weights of a state's deletions and replacements.
#[derive(Clone)]
struct Weights {
    /// Each operator's importance: the |gradient| it would have if re-inserted at its own position
    /// (its angle zero, the other angles of this state).
    importances: Vec<f64>,
    /// The deletion and replacement position weights of [`kernel::removal`].
    removal: Vec<f64>,
}
/// A relaxed state and, once something needs them, its two analyses. The relaxation is what the
/// Metropolis test reads (the cost); the screen and the weights are the proposal distribution
/// drawn from the state. A proposal's reverse probability reads only the half of the distribution
/// that its move kind uses, and a rejected proposal's state is never drawn from, so each half is
/// computed when something first needs it ([`Evaluator::screen`], [`Evaluator::weigh`]) and a state
/// becomes the chain's current state only complete ([`Evaluator::complete`]).
#[derive(Clone)]
struct Evaluated {
    sequence: Vec<usize>,
    parameters: Vec<f64>,
    energy: f64,
    cost: f64,
    screen: Option<Screen>,
    weights: Option<Weights>,
    hessian: Option<Vec<f64>>,
    success: bool,
    message: String,
}
impl Evaluated {
    fn screened(&self) -> Result<&Screen> {
        self.screen
            .as_ref()
            .ok_or_else(|| invalid("a search state was used before its pool screen"))
    }
    fn weighed(&self) -> Result<&Weights> {
        self.weights
            .as_ref()
            .ok_or_else(|| invalid("a search state was used before its removal weights"))
    }
}
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
    _memory: Reservation,
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
    _memory: Reservation,
}
/// One replica: its current state and its own random stream.
struct Replica {
    current: Evaluated,
    random: Rng,
}
/// Replica `replica`'s stream: replica 0 draws the stream of a single chain at `seed`.
fn replica_stream(seed: u64, replica: usize) -> Rng {
    if replica == 0 {
        Rng(seed)
    } else {
        let mut mixer = Rng(seed ^ (replica as u64).wrapping_mul(0xa0761d6478bd642f));
        Rng(mixer.next_u64())
    }
}
#[derive(Clone, Copy)]
struct Ensemble<'a> {
    weights: &'a [f64],
    ref_rows: &'a [usize],
}

struct ScreenCache {
    states: Vec<State>,
    hpsis: Vec<Option<Buffer<C64>>>,
}

fn parameter_bits(parameters: &[f64]) -> Vec<u64> {
    parameters.iter().map(|&p| p.to_bits()).collect()
}

fn adapt_hessian(
    action: Action,
    hessian: &[f64],
    old_len: usize,
    method: optimizer::Method,
    replace_inherit: bool,
) -> Option<Vec<f64>> {
    if hessian.len() != old_len * old_len {
        return None;
    }
    match action.kind {
        Move::Insert => Some(solver::insert_hessian(hessian, old_len, action.position)),
        Move::Delete => {
            if old_len <= 1 {
                Some(vec![])
            } else if method == optimizer::Method::Bfgs {
                Some(solver::drop_parameter_of_inverse(
                    hessian,
                    old_len,
                    action.position,
                ))
            } else {
                Some(solver::drop_parameter(hessian, old_len, action.position))
            }
        }
        Move::Replace => {
            if old_len == 0 {
                None
            } else if replace_inherit {
                Some(hessian.to_vec())
            } else {
                Some(solver::reset_hessian_parameter(
                    hessian,
                    old_len,
                    action.position,
                ))
            }
        }
    }
}

struct WorkerDelta {
    cache: BTreeMap<Vec<usize>, Evaluated>,
    warm_cache: BTreeMap<(Vec<usize>, Vec<u64>), Evaluated>,
    seen: BTreeSet<Vec<usize>>,
    history: Vec<f64>,
    optimizer_steps: usize,
    screenings: usize,
    importance_evaluations: usize,
    importance_work: evolution::Work,
    gradient_evaluations: usize,
    failures: Vec<(usize, String)>,
}

struct Evaluator<'a> {
    problem: &'a QuantumProblem,
    options: &'a SearchOptions,
    learned: Option<&'a Model>,
    graph: Option<&'a BipartiteGraph>,
    screen_hamiltonian: Option<&'a Operator>,
    ensemble: Option<Ensemble<'a>>,
    prior_cache: Option<&'a BTreeMap<Vec<usize>, Evaluated>>,
    prior_warm: Option<&'a BTreeMap<(Vec<usize>, Vec<u64>), Evaluated>>,
    cache: BTreeMap<Vec<usize>, Evaluated>,
    warm_cache: BTreeMap<(Vec<usize>, Vec<u64>), Evaluated>,
    seen: BTreeSet<Vec<usize>>,
    history: Vec<f64>,
    optimizer_steps: usize,
    screenings: usize,
    importance_evaluations: usize,
    importance_work: evolution::Work,
    gradient_evaluations: usize,
    failures: Vec<(usize, String)>,
}
impl<'a> Evaluator<'a> {
    fn new(
        problem: &'a QuantumProblem,
        options: &'a SearchOptions,
        learned: Option<&'a Model>,
        graph: Option<&'a BipartiteGraph>,
        screen_hamiltonian: Option<&'a Operator>,
        ensemble: Option<Ensemble<'a>>,
    ) -> Self {
        Self {
            problem,
            options,
            learned,
            graph,
            screen_hamiltonian,
            ensemble,
            prior_cache: None,
            prior_warm: None,
            cache: BTreeMap::new(),
            warm_cache: BTreeMap::new(),
            seen: BTreeSet::new(),
            history: vec![],
            optimizer_steps: 0,
            screenings: 0,
            importance_evaluations: 0,
            importance_work: evolution::Work::default(),
            gradient_evaluations: 0,
            failures: vec![],
        }
    }
    fn fork_readonly(&self) -> Evaluator<'_> {
        Evaluator {
            problem: self.problem,
            options: self.options,
            learned: self.learned,
            graph: self.graph,
            screen_hamiltonian: self.screen_hamiltonian,
            ensemble: self.ensemble,
            prior_cache: Some(&self.cache),
            prior_warm: Some(&self.warm_cache),
            cache: BTreeMap::new(),
            warm_cache: BTreeMap::new(),
            seen: BTreeSet::new(),
            history: vec![],
            optimizer_steps: 0,
            screenings: 0,
            importance_evaluations: 0,
            importance_work: evolution::Work::default(),
            gradient_evaluations: 0,
            failures: vec![],
        }
    }
    fn into_delta(self) -> WorkerDelta {
        WorkerDelta {
            cache: self.cache,
            warm_cache: self.warm_cache,
            seen: self.seen,
            history: self.history,
            optimizer_steps: self.optimizer_steps,
            screenings: self.screenings,
            importance_evaluations: self.importance_evaluations,
            importance_work: self.importance_work,
            gradient_evaluations: self.gradient_evaluations,
            failures: self.failures,
        }
    }
    fn merge_delta(&mut self, worker: WorkerDelta) {
        for (k, v) in worker.cache {
            self.cache.entry(k).or_insert(v);
        }
        for (k, v) in worker.warm_cache {
            self.warm_cache.entry(k).or_insert(v);
        }
        self.seen.extend(worker.seen);
        self.history.extend(worker.history);
        self.optimizer_steps += worker.optimizer_steps;
        self.screenings += worker.screenings;
        self.importance_evaluations += worker.importance_evaluations;
        self.importance_work += worker.importance_work;
        self.gradient_evaluations += worker.gradient_evaluations;
        self.failures.extend(worker.failures);
    }
    /// The relaxed state of `sequence`, without its analyses unless the memo holds them, plus
    /// the converged `(state, centered_hpsi)` pair when available for direct screening reuse.
    fn evaluate(
        &mut self,
        sequence: Vec<usize>,
        initial: Vec<f64>,
        start_hessian: Option<&[f64]>,
        step: usize,
        execution: Option<&mut (dyn Execution + '_)>,
    ) -> Result<(Evaluated, Option<ScreenCache>)> {
        self.seen.insert(sequence.clone());
        if !self.options.warm_start {
            if let Some(cached) = self
                .cache
                .get(&sequence)
                .or_else(|| self.prior_cache.and_then(|c| c.get(&sequence)))
            {
                return Ok((cached.clone(), None));
            }
        } else if self.options.memoize_warm && start_hessian.is_none() {
            let key = (sequence.clone(), parameter_bits(&initial));
            if let Some(cached) = self
                .warm_cache
                .get(&key)
                .or_else(|| self.prior_warm.and_then(|c| c.get(&key)))
            {
                return Ok((cached.clone(), None));
            }
        }
        let warm_key =
            (self.options.warm_start && self.options.memoize_warm && start_hessian.is_none())
                .then(|| (sequence.clone(), parameter_bits(&initial)));
        let (result, screen_cache) =
            self.relax(sequence, initial, start_hessian, step, execution)?;
        if !self.options.warm_start {
            self.cache.insert(result.sequence.clone(), result.clone());
        } else if let Some(key) = warm_key {
            self.warm_cache.insert(key, result.clone());
        }
        Ok((result, screen_cache))
    }
    /// Write the analyses of `state` into the memo entry of its sequence, when that entry is this
    /// very state (an uncached transferred start shares the sequence, not the angles).
    fn store(&mut self, state: &Evaluated, initial_bits: Option<&[u64]>) {
        if !self.options.warm_start {
            if let Some(entry) = self.cache.get_mut(&state.sequence) {
                if entry.parameters == state.parameters {
                    entry.screen = state.screen.clone();
                    entry.weights = state.weights.clone();
                }
            }
        } else if self.options.memoize_warm {
            if let Some(bits) = initial_bits {
                let key = (state.sequence.clone(), bits.to_vec());
                if let Some(entry) = self.warm_cache.get_mut(&key) {
                    if entry.parameters == state.parameters {
                        entry.screen = state.screen.clone();
                        entry.weights = state.weights.clone();
                    }
                }
            }
        }
    }
    /// Relax `sequence` from `initial` without reading or writing the cache: a start that is not
    /// the zero start is not a function of the sequence, so it must never be stored as one.
    fn evaluate_uncached(
        &mut self,
        sequence: Vec<usize>,
        initial: Vec<f64>,
        step: usize,
        execution: Option<&mut (dyn Execution + '_)>,
    ) -> Result<(Evaluated, Option<ScreenCache>)> {
        self.seen.insert(sequence.clone());
        self.relax(sequence, initial, None, step, execution)
    }
    fn relax(
        &mut self,
        sequence: Vec<usize>,
        initial: Vec<f64>,
        start_hessian: Option<&[f64]>,
        step: usize,
        execution: Option<&mut (dyn Execution + '_)>,
    ) -> Result<(Evaluated, Option<ScreenCache>)> {
        let mut screen_cache = None;
        let (parameters, energy, hessian, success, message) = if sequence.is_empty() {
            let energy = match (execution, &self.ensemble) {
                (Some(exec), None) => exec.evaluate(self.problem, &[], &[], false)?.energy,
                (Some(_), Some(_)) => {
                    return Err(invalid(
                        "subspace search uses local state-vector simulation",
                    ));
                }
                (None, Some(ens)) => {
                    self.problem
                        .subspace_probe_capturing(
                            &[],
                            &[],
                            true,
                            ens.weights,
                            ens.ref_rows,
                            None,
                            self.options.sparse_kernels,
                        )?
                        .energy
                }
                (None, None) => {
                    self.problem
                        .probe_with_mode(&[], &[], true, self.options.sparse_kernels)?
                        .energy
                }
            };
            (vec![], energy, None, true, "empty reference".into())
        } else if let Some(exec) = execution {
            let carries = self.options.carry_hessian
                && self.options.warm_start
                && self.options.optimizer.method.carries_curvature();
            let mut gradient_count = 0usize;
            let result = optimizer::minimize_deferred_from(
                &self.problem.context,
                &initial,
                &self.options.optimizer,
                start_hessian.filter(|_| carries),
                |p| {
                    let eval = exec.evaluate(self.problem, &sequence, p, true)?;
                    gradient_count += 1;
                    Ok(optimizer::Probe::ready(eval.energy, eval.gradient))
                },
            )?;
            self.gradient_evaluations += gradient_count;
            self.history.extend(result.history);
            self.optimizer_steps += result.steps;
            if !result.success {
                self.failures.push((step, result.message.clone()));
            }
            (
                result.parameters,
                result.energy,
                result.hessian.filter(|_| carries),
                result.success,
                result.message,
            )
        } else {
            let gradients = std::cell::Cell::new(0);
            let single_capture: solver::CapturedScreen = Rc::new(RefCell::new(None));
            let subspace_capture: subspace::CapturedSubspace = Rc::new(RefCell::new(None));
            let carries = self.options.carry_hessian
                && self.options.warm_start
                && self.options.optimizer.method.carries_curvature();
            let sparse_kernels = self.options.sparse_kernels;
            let result = optimizer::minimize_deferred_from(
                &self.problem.context,
                &initial,
                &self.options.optimizer,
                start_hessian.filter(|_| carries),
                |p| {
                    // A rejected trial of the line search never asks for its gradient.
                    let probe = match &self.ensemble {
                        Some(ens) => self.problem.subspace_probe_capturing(
                            &sequence,
                            p,
                            true,
                            ens.weights,
                            ens.ref_rows,
                            Some(Rc::clone(&subspace_capture)),
                            sparse_kernels,
                        )?,
                        None => self.problem.probe_capturing(
                            &sequence,
                            p,
                            true,
                            Rc::clone(&single_capture),
                            sparse_kernels,
                        )?,
                    };
                    let counter = &gradients;
                    Ok(optimizer::Probe::deferred(probe.energy, move || {
                        counter.set(counter.get() + 1);
                        probe.into_gradient()
                    }))
                },
            )?;
            self.gradient_evaluations += gradients.get();
            self.history.extend(result.history);
            self.optimizer_steps += result.steps;
            if !result.success {
                self.failures.push((step, result.message.clone()));
            }
            if self.ensemble.is_some() {
                if let Some(point) = subspace_capture.borrow_mut().take() {
                    if point.parameters == result.parameters {
                        screen_cache = Some(ScreenCache {
                            states: point.states,
                            hpsis: point.hpsis,
                        });
                    }
                }
            } else if let Some(point) = single_capture.borrow_mut().take() {
                if point.parameters == result.parameters {
                    if let Some(hpsi) = point.centered_hpsi {
                        screen_cache = Some(ScreenCache {
                            states: vec![point.state],
                            hpsis: vec![Some(hpsi)],
                        });
                    }
                }
            }
            (
                result.parameters,
                result.energy,
                result.hessian.filter(|_| carries),
                result.success,
                result.message,
            )
        };
        Ok((
            Evaluated {
                cost: energy + self.options.length_penalty * sequence.len() as f64,
                sequence,
                parameters,
                energy,
                screen: None,
                weights: None,
                hessian,
                success,
                message,
            },
            screen_cache,
        ))
    }
    /// The pool screen and the operator proposal of `state` (a no-op once computed), optionally
    /// reusing the `(state, centered_hpsi)` captured from the relaxation's final accepted step.
    fn screen(
        &mut self,
        state: &mut Evaluated,
        precomputed: Option<&ScreenCache>,
        initial_bits: Option<&[u64]>,
        execution: Option<&mut (dyn Execution + '_)>,
    ) -> Result<()> {
        if state.screen.is_some() {
            return Ok(());
        }
        // The pool screen feeds the gradient proposal, a ready learned model, the recorded rows and
        // the final gradient (computed once at the end when no state screens it): a uniform chain
        // that records nothing needs none of it per proposal.
        let screen = !self.options.uniform
            || self.options.record
            || (self.learned.is_some_and(|m| m.ready) && self.graph.is_some());
        let wants_curvatures = self.ensemble.is_none()
            && (self.options.record_curvatures
                || self.learned.is_some_and(|m| m.ready && m.uses_curvatures()));
        let is_local = execution.is_none();
        let gradients = if screen {
            self.screenings += 1;
            if let Some(exec) = execution {
                exec.pool_gradients(self.problem, &state.sequence, &state.parameters)?
            } else {
                match (&self.ensemble, self.screen_hamiltonian) {
                    (Some(ens), Some(h_surr)) => {
                        let mut acc = vec![0.; self.problem.pool.len()];
                        for (&w, &ref_row) in ens.weights.iter().zip(ens.ref_rows) {
                            let prepared = self.problem.subspace_prepare_state(
                                &state.sequence,
                                &state.parameters,
                                true,
                                ref_row,
                                self.options.sparse_kernels,
                            )?;
                            for (slot, g) in acc
                                .iter_mut()
                                .zip(self.problem.pool_gradients_for(&prepared, true, h_surr)?)
                            {
                                *slot += w * g;
                            }
                        }
                        acc
                    }
                    (Some(ens), None) => {
                        let mut acc = vec![0.; self.problem.pool.len()];
                        for (j, (&w, &ref_row)) in ens.weights.iter().zip(ens.ref_rows).enumerate()
                        {
                            let cached_pair = precomputed
                                .and_then(|c| Some((c.states.get(j)?, c.hpsis.get(j)?.as_ref())));
                            let g_vec = match cached_pair {
                                Some((st, Some(hpsi))) => self
                                    .problem
                                    .pool_gradients_with_centered_hpsi(st, hpsi.as_slice())?,
                                Some((st, None)) => self.problem.pool_gradients(st, true)?,
                                None => {
                                    let prepared = self.problem.subspace_prepare_state(
                                        &state.sequence,
                                        &state.parameters,
                                        true,
                                        ref_row,
                                        self.options.sparse_kernels,
                                    )?;
                                    self.problem.pool_gradients(&prepared, true)?
                                }
                            };
                            for (slot, g) in acc.iter_mut().zip(g_vec) {
                                *slot += w * g;
                            }
                        }
                        acc
                    }
                    (None, Some(h_surr)) => {
                        let prepared =
                            self.problem
                                .prepare_state(&state.sequence, &state.parameters, true)?;
                        self.problem.pool_gradients_for(&prepared, true, h_surr)?
                    }
                    (None, None) => {
                        let cached = precomputed
                            .and_then(|c| Some((c.states.first()?, c.hpsis.first()?.as_ref()?)));
                        if let Some((cached_state, cached_hpsi)) = cached {
                            self.problem.pool_gradients_with_centered_hpsi(
                                cached_state,
                                cached_hpsi.as_slice(),
                            )?
                        } else {
                            let prepared = self.problem.prepare_state(
                                &state.sequence,
                                &state.parameters,
                                true,
                            )?;
                            self.problem.pool_gradients(&prepared, true)?
                        }
                    }
                }
            }
        } else {
            vec![]
        };
        let curvatures = if wants_curvatures && is_local {
            let prepared = self
                .problem
                .prepare_state(&state.sequence, &state.parameters, true)?;
            Some(self.problem.pool_curvatures(&prepared)?)
        } else {
            None
        };
        let mut probabilities = if self.options.uniform {
            let pool = self.problem.pool.len();
            vec![1. / pool as f64; pool]
        } else {
            kernel::softmax(&gradients, self.options.proposal_temperature)?
        };
        if let (Some(model), Some(graph)) = (self.learned, &self.graph) {
            if model.ready && !gradients.is_empty() {
                let candidates: Vec<_> = (0..gradients.len()).collect();
                let rows = Candidates::new(
                    &self.problem.context,
                    graph,
                    &state.sequence,
                    &candidates,
                    &gradients,
                    state.energy - graph.reference_energy,
                )?;
                let scores = model.scores_with_curvatures(
                    &self.problem.context,
                    graph,
                    &rows,
                    curvatures.as_deref(),
                )?;
                let scores: Vec<_> = scores.iter().map(|v| v / model.temperature).collect();
                if scores.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("learned proposal temperature overflows scores"));
                }
                // A ready model draws from its own proposal alone: the gradient enters only the
                // first stage of its training (owner, 2026-11-09).
                probabilities = training::softmax(&scores);
                // Every operator must keep a positive probability, or an insertion loses its reverse
                // deletion and the Metropolis-Hastings ratio is no longer exact.
                if probabilities.iter().any(|p| *p <= 0.) {
                    return Err(invalid(
                        "the learned proposal gives an operator zero probability (its scores span more \
                         than the float range at this temperature): raise the model's temperature",
                    ));
                }
            }
        }
        state.screen = Some(Screen {
            gradients,
            probabilities,
            curvatures,
        });
        self.store(state, initial_bits);
        Ok(())
    }
    /// The importances and removal weights of `state` (a no-op once computed).
    fn weigh(
        &mut self,
        state: &mut Evaluated,
        initial_bits: Option<&[u64]>,
        mut execution: Option<&mut (dyn Execution + '_)>,
    ) -> Result<()> {
        if state.weights.is_some() {
            return Ok(());
        }
        if self
            .learned
            .is_some_and(|m| m.ready && m.edit.as_ref().is_some_and(|e| e.delete.is_some()))
            && state.screen.is_none()
            && !state.sequence.is_empty()
        {
            self.screen(state, None, initial_bits, execution.as_deref_mut())?;
        }
        let (importances, work, count) = if let Some(exec) = execution {
            let mut importances = vec![0.; state.sequence.len()];
            for i in 0..state.sequence.len() {
                let mut zeroed = state.parameters.clone();
                zeroed[i] = 0.;
                let eval = exec.evaluate(self.problem, &state.sequence, &zeroed, true)?;
                importances[i] = eval.gradient.get(i).copied().unwrap_or(0.).abs();
            }
            (
                importances,
                evolution::Work::default(),
                state.sequence.len(),
            )
        } else {
            match &self.ensemble {
                Some(ens) => {
                    let mut acc = vec![0.; state.sequence.len()];
                    let mut total_work = evolution::Work::default();
                    for (&w, &ref_row) in ens.weights.iter().zip(ens.ref_rows) {
                        let measured = self.problem.importances_from_row(
                            &state.sequence,
                            &state.parameters,
                            ref_row,
                            self.options.sparse_kernels,
                        )?;
                        for (slot, v) in acc.iter_mut().zip(measured.values) {
                            *slot += w * v;
                        }
                        total_work += measured.work;
                    }
                    (acc, total_work, ens.weights.len() * state.sequence.len())
                }
                None => {
                    let measured = self.problem.importances_with_mode(
                        &state.sequence,
                        &state.parameters,
                        self.options.sparse_kernels,
                    )?;
                    (measured.values, measured.work, state.sequence.len())
                }
            }
        };
        self.importance_evaluations += count;
        self.importance_work += work;
        let mut removal = kernel::removal(&importances, self.options.proposal_temperature)?;
        if let (Some(model), Some(graph), Some(screen)) =
            (self.learned, &self.graph, state.screen.as_ref())
        {
            if model.ready && !state.sequence.is_empty() && !screen.gradients.is_empty() {
                if let Some(rem_scores) = model.removal_scores(
                    &self.problem.context,
                    graph,
                    &state.sequence,
                    &screen.gradients,
                    screen.curvatures.as_deref(),
                    state.energy - graph.reference_energy,
                )? {
                    let scaled: Vec<_> = rem_scores.iter().map(|v| v / model.temperature).collect();
                    if scaled.iter().any(|v| !v.is_finite()) {
                        return Err(invalid("learned removal temperature overflows scores"));
                    }
                    let learned_removal = training::softmax(&scaled);
                    if learned_removal.iter().any(|p| *p <= 0.) {
                        return Err(invalid(
                            "the learned deletion proposal gives a position zero probability: raise the model's temperature",
                        ));
                    }
                    removal = learned_removal;
                }
            }
        }
        state.weights = Some(Weights {
            importances,
            removal,
        });
        self.store(state, initial_bits);
        Ok(())
    }
    /// Both analyses: what a state needs to be drawn from, so to become the current state.
    fn complete(
        &mut self,
        state: &mut Evaluated,
        precomputed: Option<&ScreenCache>,
        initial_bits: Option<&[u64]>,
        mut execution: Option<&mut (dyn Execution + '_)>,
    ) -> Result<()> {
        self.screen(state, precomputed, initial_bits, execution.as_deref_mut())?;
        self.weigh(state, initial_bits, execution)
    }
}
type StepOutcome = (f64, Option<(f64, f64)>, bool);

fn record(
    step: usize,
    action: Action,
    source: &Evaluated,
    proposed: &Evaluated,
    temperature: f64,
    outcome: Option<StepOutcome>,
    vectors: bool,
) -> Result<Value> {
    let mut row = json!({"step":step,"move":action.kind.name(),"position":action.position,
        "operator":if matches!(action.kind,Move::Insert|Move::Replace) {Some(action.operator)} else {None},
        "source":source.sequence,"proposed":proposed.sequence,"source_parameters":source.parameters,"parameters":proposed.parameters,
        "source_energy":source.energy,"energy":proposed.energy,"source_cost":source.cost,"cost":proposed.cost,
        "delta_energy":proposed.energy-source.energy,"delta_cost":proposed.cost-source.cost,
        "temperature":temperature,"log_q_forward":outcome.as_ref().map(|v|v.0),
        "log_q_reverse":outcome.as_ref().and_then(|v|v.1.map(|r|r.0)),
        "log_acceptance":outcome.as_ref().and_then(|v|v.1.map(|r|r.1)),
        "accepted":outcome.as_ref().map(|v|v.2),"screened":outcome.is_none(),
        "optimizer_success":proposed.success,"optimizer_message":proposed.message});
    // The four source vectors (pool and sequence long) are the recorded edit's training data: a
    // step row of a chain that records nothing does not copy them.
    if vectors {
        let (screen, weights) = (source.screened()?, source.weighed()?);
        row["source_gradients"] = json!(screen.gradients);
        row["source_probabilities"] = json!(screen.probabilities);
        row["source_importances"] = json!(weights.importances);
        row["source_removal"] = json!(weights.removal);
        if let Some(curvs) = &screen.curvatures {
            row["source_curvatures"] = json!(curvs);
        }
    }
    Ok(row)
}
fn strip_zeros_from(
    problem: &QuantumProblem,
    state: &mut Evaluated,
    length_penalty: f64,
    sparse_kernels: bool,
    ensemble: Option<&Ensemble<'_>>,
    execution: Option<&mut (dyn Execution + '_)>,
) -> Result<()> {
    let keep: Vec<usize> = state
        .parameters
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p.abs() > 1e-14)
        .map(|(i, _)| i)
        .collect();
    if keep.len() == state.sequence.len() {
        return Ok(());
    }
    state.sequence = keep.iter().map(|&i| state.sequence[i]).collect();
    state.parameters = keep.iter().map(|&i| state.parameters[i]).collect();
    let energy = if let Some(exec) = execution {
        exec.evaluate(problem, &state.sequence, &state.parameters, false)?
            .energy
    } else {
        match ensemble {
            Some(ens) => {
                problem
                    .subspace_probe_capturing(
                        &state.sequence,
                        &state.parameters,
                        true,
                        ens.weights,
                        ens.ref_rows,
                        None,
                        sparse_kernels,
                    )?
                    .energy
            }
            None => {
                problem
                    .evaluate_with_mode(&state.sequence, &state.parameters, true, sparse_kernels)?
                    .energy
            }
        }
    };
    state.energy = energy;
    state.cost = energy + length_penalty * state.sequence.len() as f64;
    state.screen = None;
    state.weights = None;
    state.hessian = None;
    Ok(())
}
struct ReplicaStepData {
    row: Value,
    move_index: usize,
    accepted: bool,
    proposed: Evaluated,
}
fn step_replica(
    evaluator: &mut Evaluator<'_>,
    chain: &mut Replica,
    kernel: &Kernel,
    step: usize,
    replica: usize,
    replicas: usize,
    temperature: f64,
    mut execution: Option<&mut (dyn Execution + '_)>,
) -> Result<ReplicaStepData> {
    let ctx = &evaluator.problem.context;
    let options = evaluator.options;
    let current = &mut chain.current;
    let random = &mut chain.random;
    let (screen, weights) = (current.screened()?, current.weighed()?);
    let action = kernel.draw(
        ctx,
        &current.sequence,
        &screen.probabilities,
        &weights.removal,
        random,
    )?;
    let sequence = action.apply(&current.sequence);
    let initial = if options.warm_start {
        action.transfer(&current.parameters, options.replace_inherit)
    } else {
        vec![0.; sequence.len()]
    };
    let init_bits = (options.warm_start && options.memoize_warm).then(|| parameter_bits(&initial));
    let start_hessian = if options.carry_hessian && options.warm_start {
        current.hessian.as_deref().and_then(|h| {
            adapt_hessian(
                action,
                h,
                current.sequence.len(),
                options.optimizer.method,
                options.replace_inherit,
            )
        })
    } else {
        None
    };
    let (mut proposed, proposed_cache) = evaluator.evaluate(
        sequence,
        initial,
        start_hessian.as_deref(),
        step + 1,
        execution.as_deref_mut(),
    )?;
    let forward = kernel.log_q(
        ctx,
        &proposed.sequence,
        &current.sequence,
        &screen.probabilities,
        &weights.removal,
    )?;
    let mut drawn_log_u = None;
    let mut early_rejected = false;
    if options.early_reject {
        let bound =
            kernel::log_acceptance(temperature, current.cost, proposed.cost, forward, 1e-12);
        if bound < 0. {
            let log_u = random.uniform().ln();
            if log_u >= bound {
                early_rejected = true;
            } else {
                drawn_log_u = Some(log_u);
            }
        }
    }
    let (record_outcome, accepted) = if early_rejected {
        (Some((forward, None, false)), false)
    } else {
        if matches!(action.kind, Move::Insert | Move::Replace) {
            evaluator.weigh(
                &mut proposed,
                init_bits.as_deref(),
                execution.as_deref_mut(),
            )?;
        }
        if matches!(action.kind, Move::Delete | Move::Replace) {
            evaluator.screen(
                &mut proposed,
                proposed_cache.as_ref(),
                init_bits.as_deref(),
                execution.as_deref_mut(),
            )?;
        }
        let reverse = kernel.log_q_matched(
            &current.sequence,
            &proposed.sequence,
            proposed.screen.as_ref().map(|s| &s.probabilities[..]),
            proposed.weights.as_ref().map(|w| &w.removal[..]),
        )?;
        let ratio =
            kernel::log_acceptance(temperature, current.cost, proposed.cost, forward, reverse);
        let accepted = if ratio >= 0. {
            true
        } else {
            let log_u = drawn_log_u.unwrap_or_else(|| random.uniform().ln());
            log_u < ratio
        };
        (Some((forward, Some((reverse, ratio)), accepted)), accepted)
    };
    if accepted {
        evaluator.complete(
            &mut proposed,
            proposed_cache.as_ref(),
            init_bits.as_deref(),
            execution,
        )?;
    }
    let mut row = record(
        step + 1,
        action,
        current,
        &proposed,
        temperature,
        record_outcome,
        options.record,
    )?;
    if replicas > 1 {
        row["replica"] = json!(replica);
    }
    let move_index = match action.kind {
        Move::Insert => 0,
        Move::Delete => 1,
        Move::Replace => 2,
    };
    let recorded_proposed = proposed.clone();
    if accepted {
        *current = proposed;
    }
    Ok(ReplicaStepData {
        row,
        move_index,
        accepted,
        proposed: recorded_proposed,
    })
}
impl QuantumProblem {
    pub fn search(
        &self,
        options: &SearchOptions,
        initial_indices: &[usize],
        initial_parameters: &[f64],
    ) -> Result<SearchResult> {
        self.search_with_model(options, initial_indices, initial_parameters, None)
    }
    pub fn search_with_model(
        &self,
        options: &SearchOptions,
        initial_indices: &[usize],
        initial_parameters: &[f64],
        learned: Option<&Model>,
    ) -> Result<SearchResult> {
        self.search_with_execution(options, initial_indices, initial_parameters, learned, None)
    }
    /// Run VASQA/VALSQA using an optional external [`Execution`] circuit/shot backend.
    pub fn search_with_execution(
        &self,
        options: &SearchOptions,
        initial_indices: &[usize],
        initial_parameters: &[f64],
        learned: Option<&Model>,
        execution: Option<&mut dyn Execution>,
    ) -> Result<SearchResult> {
        let (result, _) = self.search_inner(
            options,
            initial_indices,
            initial_parameters,
            learned,
            None,
            execution,
            None,
        )?;
        Ok(result)
    }
    /// Resume a previously checkpointed VASQA/VALSQA chain up to `options.max_steps`.
    pub fn resume_search(
        &self,
        options: &SearchOptions,
        checkpoint: &SearchCheckpoint,
        learned: Option<&Model>,
        execution: Option<&mut dyn Execution>,
    ) -> Result<SearchResult> {
        let (result, _) = self.search_inner(
            options,
            &[],
            &[],
            learned,
            None,
            execution,
            Some(checkpoint),
        )?;
        Ok(result)
    }
    /// Subspace-VASQA: Markov-chain architecture search minimizing weighted ensemble energy
    /// followed by Rayleigh–Ritz subspace diagonalization for ground and excited eigenstates.
    pub fn search_subspace(
        &self,
        options: &SearchOptions,
        weights: &[f64],
        initial_indices: &[usize],
        initial_parameters: &[f64],
    ) -> Result<SubspaceSearchResult> {
        self.search_subspace_with_model(options, weights, initial_indices, initial_parameters, None)
    }
    /// Like [`Self::search_subspace`], accepting an optional learned graph proposal [`Model`].
    pub fn search_subspace_with_model(
        &self,
        options: &SearchOptions,
        weights: &[f64],
        initial_indices: &[usize],
        initial_parameters: &[f64],
        learned: Option<&Model>,
    ) -> Result<SubspaceSearchResult> {
        self.search_subspace_inner(
            options,
            weights,
            initial_indices,
            initial_parameters,
            learned,
            None,
        )
    }
    /// Resume a previously checkpointed Subspace-VASQA/VALSQA chain up to `options.max_steps`.
    pub fn resume_search_subspace(
        &self,
        options: &SearchOptions,
        weights: &[f64],
        checkpoint: &SearchCheckpoint,
        learned: Option<&Model>,
    ) -> Result<SubspaceSearchResult> {
        self.search_subspace_inner(options, weights, &[], &[], learned, Some(checkpoint))
    }
    fn search_subspace_inner(
        &self,
        options: &SearchOptions,
        weights: &[f64],
        initial_indices: &[usize],
        initial_parameters: &[f64],
        learned: Option<&Model>,
        checkpoint: Option<&SearchCheckpoint>,
    ) -> Result<SubspaceSearchResult> {
        subspace::validate_subspace_weights(weights)?;
        let ref_rows = self.subspace_ref_rows(weights.len())?;
        let ensemble = Ensemble {
            weights,
            ref_rows: &ref_rows,
        };
        let (core, diag) = self.search_inner(
            options,
            initial_indices,
            initial_parameters,
            learned,
            Some(ensemble),
            None,
            checkpoint,
        )?;
        let (ensemble_energies, energies, states) = diag.ok_or_else(|| {
            Error::Kernel("ensemble search did not produce subspace eigenstates".into())
        })?;
        let mut reference_indices: Vec<usize> = (0..ensemble_energies.len()).collect();
        reference_indices.sort_by(|&a, &b| ensemble_energies[a].total_cmp(&ensemble_energies[b]));
        let SearchResult {
            solution,
            records,
            steps,
            acceptance,
            optimal_cost,
            best_energy,
            best_energy_indices,
            best_energy_parameters,
            final_energy,
            final_indices,
            final_parameters,
            num_architectures,
            num_screenings,
            num_importance_evaluations,
            importance_work,
            num_gradient_evaluations,
            failures,
            swaps,
            swap_acceptance,
            checkpoint,
            _memory,
        } = core;
        Ok(SubspaceSearchResult {
            solution,
            energies,
            ensemble_energies,
            states,
            reference_indices,
            records,
            steps,
            acceptance,
            optimal_cost,
            best_energy,
            best_energy_indices,
            best_energy_parameters,
            final_energy,
            final_indices,
            final_parameters,
            num_architectures,
            num_screenings,
            num_importance_evaluations,
            importance_work,
            num_gradient_evaluations,
            failures,
            swaps,
            swap_acceptance,
            checkpoint,
            _memory,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn search_inner(
        &self,
        options: &SearchOptions,
        initial_indices: &[usize],
        initial_parameters: &[f64],
        learned: Option<&Model>,
        ensemble: Option<Ensemble<'_>>,
        execution: Option<&mut dyn Execution>,
        resume: Option<&SearchCheckpoint>,
    ) -> Result<(SearchResult, Option<(Vec<f64>, Vec<f64>, State)>)> {
        options.validate()?;
        if self.hva.is_some()
            || initial_indices.len() != initial_parameters.len()
            || initial_indices.len() > options.max_length
            || initial_indices.iter().any(|i| *i >= self.pool.len())
            || initial_parameters.iter().any(|p| !p.is_finite())
        {
            return Err(invalid(
                "search needs a valid excitation sequence and finite initial parameters",
            ));
        }
        if ensemble.is_some() && execution.is_some() {
            return Err(invalid(
                "subspace search uses local state-vector simulation",
            ));
        }
        let mut execution = execution;
        self.context.run(|| {
            let started = Instant::now();
            let ctx: &Context = &self.context;
            let replicas = options.replicas;
            // Every replica proposes once per step.
            let evaluations = options.max_steps.max(options.transfer_steps.unwrap_or(0)).checked_mul(options.screen_insertions.min(self.pool.len())+1).and_then(|v|v.checked_add(3)).and_then(|v|v.checked_mul(replicas)).ok_or(Error::Overflow)?;
            let memory = ctx.reserve(product(&[evaluations,self.pool.len()+options.max_length+32,256])?)?;
            let result_memory = ctx.reserve(product(&[evaluations,options.optimizer.max_evaluations,8])?)?;
            let graph = if learned.is_some_and(|m| m.ready) {Some(BipartiteGraph::from_problem(self)?)} else {None};
            let screen_hamiltonian = options
                .screen_hamiltonian_terms
                .filter(|&k| k < self.hamiltonian.terms.len())
                .map(|k| self.hamiltonian.top_terms(ctx, k))
                .transpose()?;
            let mut evaluator = Evaluator::new(
                self,
                options,
                learned,
                graph.as_ref(),
                screen_hamiltonian.as_ref(),
                ensemble,
            );
            let (mut reference, ref_cache) =
                evaluator.evaluate(vec![], vec![], None, 0, execution.as_deref_mut())?;
            evaluator.complete(
                &mut reference,
                ref_cache.as_ref(),
                None,
                execution.as_deref_mut(),
            )?;
            let reference_energy = reference.energy;
            let mut seed = None;
            let (start_step, seeded_kept, chain_steps, mut best, mut best_energy, mut chains, mut swap_random, mut screening_random, mut swap_acceptance, mut acceptance) = if let Some(ckpt) = resume {
                if ckpt.replicas.len() != replicas {
                    return Err(invalid("search checkpoint replica count disagrees with options.replicas"));
                }
                let chain_steps = if ckpt.seeded_kept {
                    options.transfer_steps.unwrap_or(options.max_steps)
                } else {
                    options.max_steps
                };
                if ckpt.step > chain_steps {
                    return Err(invalid("search checkpoint step already exceeds max_steps"));
                }
                let probe_state = |seq: &[usize], params: &[f64], exec: Option<&mut (dyn Execution + '_)>| -> Result<Evaluated> {
                    let energy = if let Some(e) = exec {
                        e.evaluate(self, seq, params, false)?.energy
                    } else if let Some(ens) = &ensemble {
                        self.subspace_probe_capturing(seq, params, true, ens.weights, ens.ref_rows, None, options.sparse_kernels)?.energy
                    } else {
                        self.probe_with_mode(seq, params, true, options.sparse_kernels)?.energy
                    };
                    Ok(Evaluated {
                        cost: energy + options.length_penalty * seq.len() as f64,
                        sequence: seq.to_vec(),
                        parameters: params.to_vec(),
                        energy,
                        screen: None,
                        weights: None,
                        hessian: None,
                        success: true,
                        message: "resumed state".into(),
                    })
                };
                let best = probe_state(&ckpt.best_sequence, &ckpt.best_parameters, execution.as_deref_mut())?;
                let best_energy = probe_state(&ckpt.best_energy_sequence, &ckpt.best_energy_parameters, execution.as_deref_mut())?;
                let mut chains = Vec::with_capacity(replicas);
                for r in &ckpt.replicas {
                    let mut cur = probe_state(&r.sequence, &r.parameters, execution.as_deref_mut())?;
                    evaluator.seen.insert(cur.sequence.clone());
                    evaluator.complete(&mut cur, None, None, execution.as_deref_mut())?;
                    chains.push(Replica { current: cur, random: Rng(r.rng) });
                }
                (
                    ckpt.step,
                    ckpt.seeded_kept,
                    chain_steps,
                    best,
                    best_energy,
                    chains,
                    Rng(ckpt.swap_rng),
                    Rng(ckpt.screening_rng),
                    ckpt.swap_acceptance,
                    ckpt.acceptance,
                )
            } else {
                let mut current = reference;
                let mut transferred_start=None;
                if !initial_indices.is_empty() {
                    let before=evaluator.history.len();let before_steps=evaluator.optimizer_steps;
                    let init_bits = parameter_bits(initial_parameters);
                    let (mut transferred, tr_cache) = if options.warm_start {
                        evaluator.evaluate(
                            initial_indices.to_vec(),
                            initial_parameters.to_vec(),
                            None,
                            0,
                            execution.as_deref_mut(),
                        )?
                    } else {
                        evaluator.evaluate_uncached(
                            initial_indices.to_vec(),
                            initial_parameters.to_vec(),
                            0,
                            execution.as_deref_mut(),
                        )?
                    };
                    let kept=transferred.cost<current.cost;
                    seed=Some(crate::quantum::solver::SeedRelaxation {kept,energy:transferred.energy,success:transferred.success,
                        message:transferred.message.clone(),evaluations:evaluator.history.len()-before,steps:evaluator.optimizer_steps-before_steps,warm:false});
                    if kept {
                        if options.warm_start {
                            evaluator.complete(
                                &mut transferred,
                                tr_cache.as_ref(),
                                Some(&init_bits),
                                execution.as_deref_mut(),
                            )?;
                            current=transferred;
                        } else {
                            let (mut zero_cur, zero_cache) = evaluator.evaluate(
                                initial_indices.to_vec(),
                                vec![0.; initial_indices.len()],
                                None,
                                0,
                                execution.as_deref_mut(),
                            )?;
                            evaluator.complete(
                                &mut zero_cur,
                                zero_cache.as_ref(),
                                None,
                                execution.as_deref_mut(),
                            )?;
                            current=zero_cur;
                            transferred_start=Some(transferred);
                        }
                    }
                }
                let seeded_kept = seed.as_ref().is_some_and(|s| s.kept);
                let chain_steps=if seeded_kept {options.transfer_steps.unwrap_or(options.max_steps)} else {options.max_steps};
                let mut best = current.clone();
                let mut best_energy = current.clone();
                if let Some(start) = transferred_start {
                    if start.cost<best.cost {best=start.clone();}
                    if start.energy<best_energy.energy {best_energy=start;}
                }
                // Replica 0 is the single chain: the same start, the same stream, the same draws.
                let mut chains: Vec<Replica> = (1..replicas).map(|r| Replica {current: current.clone(), random: replica_stream(options.seed, r)}).collect();
                chains.insert(0, Replica {current, random: replica_stream(options.seed, 0)});
                (
                    0,
                    seeded_kept,
                    chain_steps,
                    best,
                    best_energy,
                    chains,
                    Rng(options.seed ^ 0x5851f42d4c957f2d),
                    Rng(options.seed ^ 0xd1b54a32d192ed03),
                    (0, 0),
                    [(0,0);3],
                )
            };
            let schedule = options.active_temperatures(seeded_kept);
            let kernel = if self.pool.is_empty() {None} else {Some(Kernel::new(self.pool.len(),options.max_length,options.move_weights)?)};
            let mut records = Vec::new();
            let mut steps = Vec::new();
            let mut swaps = Vec::new();
            let mut screened = BTreeSet::new();
            let parallel_replicas = replicas > 1
                && execution.is_none()
                && ctx.width() > 1
                && ctx.pool().is_some()
                && !parallel::is_inside();
            for step in start_step..if kernel.is_some() {chain_steps} else {start_step} {
                let kernel = kernel.as_ref().unwrap();
                if parallel_replicas {
                    let pool = ctx.pool().expect("checked");
                    let mut workers: Vec<_> = (0..replicas).map(|_| evaluator.fork_readonly()).collect();
                    let outcomes: Vec<Result<ReplicaStepData>> = pool.install(|| {
                        chains
                            .par_iter_mut()
                            .zip(workers.par_iter_mut())
                            .enumerate()
                            .map(|(replica, (chain, worker))| {
                                let temperature = options.replica_temperature_on(schedule, step, chain_steps, replica);
                                parallel::inside(|| {
                                    step_replica(worker, chain, kernel, step, replica, replicas, temperature, None)
                                })
                            })
                            .collect()
                    });
                    let deltas: Vec<_> = workers.into_iter().map(Evaluator::into_delta).collect();
                    for (delta, outcome) in deltas.into_iter().zip(outcomes) {
                        evaluator.merge_delta(delta);
                        let data = outcome?;
                        if options.record {records.push(data.row.clone());}
                        steps.push(data.row);
                        acceptance[data.move_index].1 += 1;
                        if data.accepted {acceptance[data.move_index].0 += 1;}
                        if data.proposed.energy < best_energy.energy {best_energy = data.proposed.clone();}
                        if data.proposed.cost < best.cost {best = data.proposed;}
                    }
                } else {
                    for (replica, chain) in chains.iter_mut().enumerate() {
                        let temperature = options.replica_temperature_on(schedule, step, chain_steps, replica);
                        if options.screen_insertions > 0 && chain.current.sequence.len() < options.max_length && screened.insert(chain.current.sequence.clone()) {
                            let mut remaining: Vec<_> = (0..self.pool.len()).collect();
                            let probabilities = chain.current.screened()?.probabilities.clone();
                            for _ in 0..options.screen_insertions.min(remaining.len()) {
                                let mut draw = screening_random.uniform()*remaining.iter().map(|&i|probabilities[i]).sum::<f64>();
                                let mut selected = remaining.len()-1;
                                for (j,&i) in remaining.iter().enumerate() {draw-=probabilities[i];if draw<0. {selected=j;break;}}
                                let operator = remaining.remove(selected);
                                let action = Action {kind:Move::Insert,position:screening_random.index(chain.current.sequence.len()+1),operator};
                                let sequence = action.apply(&chain.current.sequence);
                                let initial = if options.warm_start {action.transfer(&chain.current.parameters,options.replace_inherit)} else {vec![0.;sequence.len()]};
                                let (proposed, _) = evaluator.evaluate(
                                    sequence,
                                    initial,
                                    None,
                                    step + 1,
                                    execution.as_deref_mut(),
                                )?;
                                // A screened insertion is only recorded: its relaxation is all the row reads.
                                records.push(record(step+1,action,&chain.current,&proposed,temperature,None,true)?);
                            }
                        }
                        let data = step_replica(
                            &mut evaluator,
                            chain,
                            kernel,
                            step,
                            replica,
                            replicas,
                            temperature,
                            execution.as_deref_mut(),
                        )?;
                        if options.record {records.push(data.row.clone());}
                        steps.push(data.row);
                        acceptance[data.move_index].1 += 1;
                        if data.accepted {acceptance[data.move_index].0 += 1;}
                        if data.proposed.energy < best_energy.energy {best_energy = data.proposed.clone();}
                        if data.proposed.cost < best.cost {best = data.proposed;}
                    }
                }
                if replicas > 1 {
                    // One adjacent pair exchanges its states; the streams stay with their replicas.
                    let pair = swap_random.index(replicas - 1);
                    let (cold, hot) = (
                        options.replica_temperature_on(schedule, step, chain_steps, pair),
                        options.replica_temperature_on(schedule, step, chain_steps, pair + 1),
                    );
                    let (cold_cost, hot_cost) = (chains[pair].current.cost, chains[pair+1].current.cost);
                    let sequences = json!([chains[pair].current.sequence, chains[pair+1].current.sequence]);
                    let ratio = (1./cold - 1./hot)*(cold_cost - hot_cost);
                    let accepted = ratio>=0. || swap_random.uniform().ln()<ratio;
                    swap_acceptance.1 += 1;
                    if accepted {
                        swap_acceptance.0 += 1;
                        let (lower, upper) = chains.split_at_mut(pair+1);
                        std::mem::swap(&mut lower[pair].current, &mut upper[0].current);
                    }
                    swaps.push(json!({"step":step+1,"replicas":[pair,pair+1],"temperatures":[cold,hot],"costs":[cold_cost,hot_cost],"sequences":sequences,"log_acceptance":ratio,"accepted":accepted}));
                }
            }
            let checkpoint = SearchCheckpoint {
                step: if kernel.is_some() { chain_steps } else { start_step },
                total_steps: chain_steps,
                seeded_kept,
                replicas: chains
                    .iter()
                    .map(|c| ReplicaState {
                        sequence: c.current.sequence.clone(),
                        parameters: c.current.parameters.clone(),
                        rng: c.random.0,
                    })
                    .collect(),
                best_sequence: best.sequence.clone(),
                best_parameters: best.parameters.clone(),
                best_energy_sequence: best_energy.sequence.clone(),
                best_energy_parameters: best_energy.parameters.clone(),
                swap_rng: swap_random.0,
                screening_rng: screening_random.0,
                acceptance,
                swap_acceptance,
            };
            let mut current = chains.swap_remove(0).current;
            let Evaluator {
                mut history,
                mut optimizer_steps,
                mut screenings,
                mut importance_evaluations,
                mut importance_work,
                mut gradient_evaluations,
                mut failures,
                seen,
                ..
            } = evaluator;
            if options.strip_zero_operators {
                strip_zeros_from(self, &mut best, options.length_penalty, options.sparse_kernels, ensemble.as_ref(), execution.as_deref_mut())?;
                strip_zeros_from(self, &mut best_energy, options.length_penalty, options.sparse_kernels, ensemble.as_ref(), execution.as_deref_mut())?;
                strip_zeros_from(self, &mut current, options.length_penalty, options.sparse_kernels, ensemble.as_ref(), execution.as_deref_mut())?;
            }
            let exact_screen_available = screen_hamiltonian.is_none();
            let (solution_energy, final_state, final_gradients, diag) = if let Some(ens) = ensemble {
                let diag = self.subspace_prepare_and_diagonalize(
                    &best.sequence,
                    &best.parameters,
                    true,
                    ens.ref_rows,
                    options.sparse_kernels,
                )?;
                let final_gradients = match best.screen.as_ref().filter(|s| exact_screen_available && s.gradients.len() == self.pool.len()) {
                    Some(s) => s.gradients.clone(),
                    None => {
                        screenings += 1;
                        let mut acc = vec![0.; self.pool.len()];
                        for (&w, &ref_row) in ens.weights.iter().zip(ens.ref_rows) {
                            let st = self.subspace_prepare_state(
                                &best.sequence,
                                &best.parameters,
                                true,
                                ref_row,
                                options.sparse_kernels,
                            )?;
                            for (slot, g) in acc.iter_mut().zip(self.pool_gradients(&st, true)?) {
                                *slot += w * g;
                            }
                        }
                        acc
                    }
                };
                (
                    diag.energies[0],
                    diag.ground_state,
                    final_gradients,
                    Some((diag.ensemble_energies, diag.energies, diag.states)),
                )
            } else {
                let final_state = self.result_state(&best.sequence, &best.parameters, true)?;
                // A state that was not screened (a uniform chain that records nothing, or screened on a
                // surrogate Hamiltonian) is screened once against the exact Hamiltonian at completion.
                let final_gradients = match best.screen.as_ref().filter(|s| exact_screen_available && s.gradients.len() == self.pool.len()) {
                    Some(s) => s.gradients.clone(),
                    None => {
                        screenings += 1;
                        if let Some(exec) = execution.as_deref_mut() {
                            exec.pool_gradients(self, &best.sequence, &best.parameters)?
                        } else {
                            self.pool_gradients(&final_state, true)?
                        }
                    }
                };
                (best.energy, final_state, final_gradients, None)
            };
            let _ = (&mut history, &mut optimizer_steps, &mut importance_evaluations, &mut importance_work, &mut gradient_evaluations, &mut failures);
            Ok((SearchResult {
                solution: SolveResult {seed,energy:solution_energy,reference:reference_energy,parameters:best.parameters,
                    state:final_state,labels:best.sequence.iter().map(|&i|self.pool[i].label.clone()).collect(),indices:best.sequence,
                    iterations:vec![],history,optimizer_steps,success:false,
                    message:"architecture search budget completed; no eigenstate convergence certificate".into(),
                    final_gradient:final_gradients.iter().map(|g|g.abs()).fold(0.,f64::max),budget_stopped:false,elapsed_seconds:started.elapsed().as_secs_f64(),pruned:vec![],hessian:best.hessian,notes:vec![],_memory:result_memory},
                records,steps,acceptance,optimal_cost:best.cost,best_energy:best_energy.energy,
                best_energy_indices:best_energy.sequence,best_energy_parameters:best_energy.parameters,final_energy:current.energy,
                final_indices:current.sequence,final_parameters:current.parameters,num_architectures:seen.len(),
                num_screenings:screenings,num_importance_evaluations:importance_evaluations,importance_work,num_gradient_evaluations:gradient_evaluations,failures,swaps,swap_acceptance,checkpoint,_memory:memory,
            }, diag))
        })
    }
}
