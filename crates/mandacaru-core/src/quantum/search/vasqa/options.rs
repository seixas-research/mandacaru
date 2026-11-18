use super::invalid;
use crate::quantum::{optimizer, search::kernel::Kernel};
use crate::Result;

/// Configuration for a VASQA [`crate::quantum::QuantumProblem::search`] run.
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
    pub(super) fn active_temperatures(&self, seeded_kept: bool) -> [f64; 2] {
        self.seeded_temperatures
            .filter(|_| seeded_kept)
            .unwrap_or(self.temperatures)
    }
    pub(super) fn replica_temperature_on(
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
