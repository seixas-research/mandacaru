use super::{invalid, State};
use crate::{
    math::storage::Precision,
    quantum::{amplitude, optimizer},
    Reservation, Result,
};

/// Where and how often a solve writes its mid-run checkpoint (`SolveOptions::checkpoint`).
#[derive(Clone, Debug)]
pub struct Checkpointing {
    pub path: std::path::PathBuf,
    /// Accepted growth steps (ADAPT) or cost evaluations (fixed ansatz) between two writes.
    pub every: usize,
    /// The checkpoint's method name (`vqe` or `adapt-vqe`).
    pub method: String,
}

#[derive(Clone, Debug)]
pub struct SolveOptions {
    pub adapt: bool,
    pub trotter: bool,
    /// Re-optimize every parameter after each change (`true`, standard ADAPT-VQE and VQE).
    /// `false` optimizes only what changed: ADAPT varies the newly appended parameters with every
    /// earlier one frozen at its optimum, and a fixed ansatz is swept once, one parameter at a
    /// time in order, each with the earlier ones at their optimized values and the later ones at
    /// their starting values. Cheaper per step; the energy is an upper bound to the joint one.
    pub quenching: bool,
    /// TETRIS-ADAPT-VQE (Anastasiou et al., Phys. Rev. Research 6, 013254, 2024): a growth step
    /// appends, after the largest gradient, every further operator in descending gradient order
    /// whose qubit support is disjoint from those already taken this step. One growth step, one
    /// circuit layer; never worse than the single operator.
    pub tetris: bool,
    /// Pruned-ADAPT-VQE (J. Chem. Theory Comput. 21, 8720, 2025): after each growth step's
    /// optimization, the operator ranked first by `exp(-alpha x) / theta^2` (`x` its relative
    /// position, the newest never) is removed when `|theta|` is under a fraction of the mean
    /// amplitude of the most recent ones ([`PRUNE_ALPHA`], [`PRUNE_FRACTION`], [`PRUNE_RECENT`],
    /// [`PRUNE_MIN_OPERATORS`]); the parameters are not re-optimized.
    pub prune: bool,
    /// Mid-run checkpoints: ADAPT writes one after every `every` accepted growth steps, a fixed
    /// ansatz the best point so far after every `every` cost evaluations. The caller writes the
    /// final, complete checkpoint.
    pub checkpoint: Option<Checkpointing>,
    pub max_growth: usize,
    pub gradient_tolerance: Option<f64>,
    pub energy_tolerance: Option<f64>,
    pub optimizer: optimizer::Options,
    /// The curvature estimate (row-major, `n x n`) of a carried ADAPT sequence of `n`
    /// parameters, from the run that produced it (a checkpoint, or the previous geometry's
    /// handover): the relaxation of the sequence starts from it instead of the identity. It is
    /// the Hessian approximation of SLSQP or the inverse Hessian approximation of BFGS, and
    /// belongs to the method that produced it: the caller hands a method its own. Any other
    /// method, or no carried sequence, refuses it.
    pub hessian: Option<Vec<f64>>,
    /// The carried sequence and its `hessian` are the stored end of an earlier growth step of
    /// this very problem (a checkpoint of the same Hamiltonian): the run continues growing from
    /// that point as the uninterrupted run did, without relaxing the sequence first (a relaxation
    /// would move it by roundoff and could reorder operators of equal gradient). Needs `hessian`.
    pub continuation: bool,
    /// Enable sparse-state energy expectation and sparse tangent-vector adjoint gradients on large
    /// registers (`true`, new default): avoids `O(D)` allocations on rejected trials and reduces
    /// gradient complexity from `O(L D)` to `O(L^2 nnz)` while staying bit-identical within roundoff.
    pub sparse_kernels: bool,
    /// Reuse converged `(energy, gradient, state, centered_hpsi)` across VQE diagnostics and
    /// ADAPT-VQE pool screening, and evaluate ADAPT pruning with energy-only `probe` (`true`,
    /// new default). Set `false` to re-run full evaluations from scratch.
    pub reuse_evaluations: bool,
    /// Carry BFGS/SLSQP curvature estimates across ADAPT-VQE growth steps (`true`, default).
    /// Set `false` to restart each growth step from the identity curvature estimate.
    pub carry_hessian: bool,
}
impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            adapt: false,
            trotter: false,
            quenching: true,
            tetris: false,
            prune: false,
            checkpoint: None,
            max_growth: 50,
            gradient_tolerance: Some(1e-7),
            energy_tolerance: None,
            optimizer: optimizer::Options::default(),
            hessian: None,
            continuation: false,
            sparse_kernels: true,
            reuse_evaluations: true,
            carry_hessian: true,
        }
    }
}
impl SolveOptions {
    /// Reject thresholds the state precision cannot resolve. The float32
    /// floors are the measured values of `amplitude::FLOAT32_FLOORS`.
    pub fn validate(&self, precision: Precision) -> Result<()> {
        self.optimizer.validate()?;
        if self
            .gradient_tolerance
            .into_iter()
            .chain(self.energy_tolerance)
            .any(|t| !t.is_finite() || t <= 0.)
            || (self.adapt && self.gradient_tolerance.is_none() && self.energy_tolerance.is_none())
        {
            return Err(invalid(
                "ADAPT needs positive finite convergence thresholds and at least one criterion",
            ));
        }
        if let Some(floors) = amplitude::floors(precision) {
            let name = precision.name();
            // `None` is the optimizer default (1e-9), below the floor; SPSA
            // without a tolerance runs its whole budget and tests nothing.
            let tolerance = self
                .optimizer
                .tolerance
                .or((self.optimizer.method != optimizer::Method::Spsa)
                    .then_some(optimizer::DEFAULT_TOLERANCE));
            if let Some(tolerance) = tolerance.filter(|&t| t < floors.energy) {
                return Err(invalid(&format!(
                    "optimizer tolerance {:e} Ha is below the {name} state floor {:e} Ha; \
                     raise it or use precision='float64'",
                    tolerance, floors.energy
                )));
            }
            if self.adapt {
                if let Some(t) = self.energy_tolerance.filter(|&t| t < floors.energy) {
                    return Err(invalid(&format!(
                        "ADAPT energy threshold {t:e} Ha is below the {name} state floor {:e} Ha; \
                         raise it or use precision='float64'",
                        floors.energy
                    )));
                }
                if let Some(t) = self.gradient_tolerance.filter(|&t| t < floors.gradient) {
                    return Err(invalid(&format!(
                        "ADAPT gradient threshold {t:e} Ha/rad is below the {name} state floor {:e} Ha/rad; \
                         raise it or use precision='float64'",
                        floors.gradient
                    )));
                }
            }
        }
        Ok(())
    }
}
pub struct Iteration {
    pub pool_index: usize,
    pub energy: f64,
    pub max_gradient: f64,
    pub optimizer_steps: usize,
    pub num_evaluations: usize,
    pub success: bool,
    pub message: String,
    pub num_parameters: usize,
}
pub struct SolveResult {
    pub seed: Option<SeedRelaxation>,
    pub energy: f64,
    pub reference: f64,
    pub parameters: Vec<f64>,
    pub state: State,
    pub indices: Vec<usize>,
    pub labels: Vec<String>,
    pub iterations: Vec<Iteration>,
    pub history: Vec<f64>,
    pub optimizer_steps: usize,
    pub success: bool,
    pub message: String,
    pub final_gradient: f64,
    /// The executor refused further circuits (its run budget is exhausted)
    /// and the result is the best point reached, `success == false`.
    pub budget_stopped: bool,
    pub elapsed_seconds: f64,
    /// `(growth step, label)` of every operator pruning removed (`SolveOptions::prune`).
    pub pruned: Vec<(usize, String)>,
    /// The curvature estimate (row-major, `n x n`) of the final ADAPT ansatz of `n` parameters,
    /// for a SLSQP (Hessian) or BFGS (inverse Hessian) run that re-optimized every parameter
    /// after each growth step (`quenching`): the next growth step, a resume or a handover to
    /// another geometry starts from it. `None` otherwise (the optimizer that produced it is
    /// `SolveOptions::optimizer.method`).
    pub hessian: Option<Vec<f64>>,
    /// What the caller should know about how the run started (a checkpoint that recorded no
    /// Hessian resumes from the identity).
    pub notes: Vec<String>,
    pub(crate) _memory: Reservation,
}
pub struct SeedRelaxation {
    pub kept: bool,
    pub energy: f64,
    pub success: bool,
    pub message: String,
    pub evaluations: usize,
    pub steps: usize,
    /// Whether the relaxation started from a carried Hessian (`SolveOptions::hessian`) rather
    /// than the identity.
    pub warm: bool,
}
