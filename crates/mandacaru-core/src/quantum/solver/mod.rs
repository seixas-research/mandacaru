mod adapt;
mod hessian;
mod options;
mod probes;

pub use adapt::{prune_candidate, PRUNE_ALPHA, PRUNE_FRACTION, PRUNE_MIN_OPERATORS, PRUNE_RECENT};
pub(in crate::quantum) use hessian::{
    drop_parameter, drop_parameter_of_inverse, extend_hessian, insert_hessian,
    reset_hessian_parameter,
};
pub use options::{Checkpointing, Iteration, SeedRelaxation, SolveOptions, SolveResult};

use super::amplitude::State;
use super::factorized_measurement::FactorizedMeasurement;
use super::{algebra::Generator, evolution, tapering::Taper, Basis, Encoding, Operator, C64};
use crate::{math::storage::Precision, Buffer, Context, Error, Reservation, Result};
use std::path::Path;

/// The pool screen takes the support-restricted route when fewer than one amplitude in this many
/// is nonzero. Only a cost threshold: the restricted route gives the dense bits
/// (`Operator::apply_support_into` refuses whenever it could not), so no value depends on it.
const RESTRICTED_SUPPORT: usize = 32;

/// Run `body` once with `A` bound to the stored amplitude of `precision`.
/// The one dispatch point of every quantum entry: below it the code is
/// generic, so there is a single numerical definition per precision.
macro_rules! by_precision {
    ($precision:expr, $a:ident => $body:expr) => {
        match $precision {
            $crate::math::storage::Precision::Float64 => {
                type $a = $crate::quantum::C64;
                $body
            }
            $crate::math::storage::Precision::Float32 => {
                type $a = num_complex::Complex32;
                $body
            }
        }
    };
}
pub(crate) use by_precision;

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
/// What a problem holds its states in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    /// The particle-number sector: a basis of the determinants with the requested counts.
    Sector,
    /// The whole `2^n` register.
    Register,
    /// No register: the Hamiltonian, the pool and the measured observables are Pauli operators
    /// and nothing else, so no object of size `2^n` or of the sector's size exists. This is the
    /// shot-based mode: only a circuit executor can evaluate it. With `number_conserving`, the
    /// Hamiltonian and every generator are checked algebraically to commute with the alpha and
    /// beta number operators.
    Pauli { number_conserving: bool },
}

/// The refusal of a state-vector operation on a register-free problem, naming the operation.
pub fn register_free_refusal(what: &str) -> Error {
    Error::Invalid(format!(
        "{what} needs a state-vector register, but this problem is register-free (a shot-based \
         run holds Pauli operators only and builds no 2^n or sector-sized object); run with \
         shots=0 on a register that fits in memory"
    ))
}

/// `Basis::check_invariant` when there is a basis; otherwise the commutation with the particle
/// number operators when the register-free problem asked for it (nothing to check without them).
fn check_invariant(
    basis: Option<&Basis>,
    conserved: &[Operator],
    ctx: &Context,
    operator: &Operator,
) -> Result<()> {
    if let Some(basis) = basis {
        return basis.check_invariant(operator);
    }
    for number in conserved {
        if !operator.commutes_with(ctx, number)? {
            return Err(invalid(
                "operator does not preserve the requested particle sector",
            ));
        }
    }
    Ok(())
}
pub type GrowthObserver<'a> = dyn FnMut(&[usize], &[f64], f64) -> Result<()> + 'a;

/// External circuit execution. Optimization and operator construction stay native.
pub trait Execution {
    fn evaluate(
        &mut self,
        problem: &QuantumProblem,
        indices: &[usize],
        parameters: &[f64],
        gradients: bool,
    ) -> Result<evolution::Evaluation>;
    fn pool_gradients(
        &mut self,
        problem: &QuantumProblem,
        indices: &[usize],
        parameters: &[f64],
    ) -> Result<Vec<f64>>;
}

pub struct QuantumProblem {
    pub context: Context,
    pub encoding: Encoding,
    pub hamiltonian: Operator,
    /// The state-vector register (the particle sector or the whole register), or `None` for a
    /// register-free problem: Pauli operators only, built for shot-based execution, where no
    /// `2^n` or sector-sized object may exist. Read it through [`QuantumProblem::register`].
    pub basis: Option<Basis>,
    /// The alpha and beta number operators a register-free problem checks its Hamiltonian and
    /// pool generators against (algebraically, in place of `Basis::check_invariant`); empty when
    /// a basis checks them, or when the register-free problem asked for no check.
    pub conserved: Vec<Operator>,
    pub pool: Vec<Generator>,
    pub dense: Option<Buffer<C64>>,
    /// Storage precision of every state this problem prepares. The
    /// Hamiltonian, the pool, the parameters and all reductions are float64.
    pub precision: Precision,
    pub sparse: Option<crate::quantum::sparse::SparseMatrix>,
    pub pool_screenable: Vec<bool>,
    pub growth_members: Vec<Vec<usize>>,
    pub hva: Option<crate::quantum::hva::HVA>,
    /// The name of the pool `pool` was built from (`pools::kind` describes it).
    pub pool_name: String,
    /// The Z2 symmetries removed from the register, when [`QuantumProblem::taper`] found any. Then
    /// `hamiltonian`, `pool` and `basis` live on the tapered register while `encoding` still
    /// describes the fermionic problem.
    pub taper: Option<Taper>,
    /// The pool generators dropped as symmetry-changing by `taper`: their index in the pool as
    /// built, and their label.
    pub dropped: Vec<(usize, String)>,
    /// The `O(M)` basis-rotation measurement of the energy, when the problem was built from a
    /// molecular model and asked for it (`QuantumProblem::factorize`). Circuit execution then
    /// measures the energy in its `L + 1` bases instead of the qubit-wise commuting groups of
    /// `hamiltonian`; pool gradients stay on the commuting groups.
    pub factorized: Option<FactorizedMeasurement>,
    pub momentum: Option<crate::quantum::CrystalMomentum>,
    pub point_symmetry: Option<crate::quantum::CrystalPointSymmetry>,
    pub(in crate::quantum) _pool_metadata: Reservation,
}
/// Shared slot capturing `(parameters, energy, gradient, state, centered_hpsi)` at an optimizer's
/// evaluated step (`centered_hpsi` is `Some` when the gradient pass left `(H - I) psi` intact).
pub(in crate::quantum) struct CapturedPoint {
    pub parameters: Vec<f64>,
    pub energy: f64,
    pub gradient: Vec<f64>,
    pub state: State,
    pub centered_hpsi: Option<Buffer<C64>>,
}

/// Shared capture slot filled whenever an optimizer requests the deferred gradient of a trial.
pub(in crate::quantum) type CapturedScreen = std::rc::Rc<std::cell::RefCell<Option<CapturedPoint>>>;

impl QuantumProblem {
    /// Read a bounded schema-v1 JSON cache; integrals and quantum SDKs are absent.
    pub fn load(
        ctx: &Context,
        path: &Path,
        sector: bool,
        dense: bool,
        singles: bool,
    ) -> Result<Self> {
        crate::persistence::Document::read(ctx, path, crate::persistence::HAMILTONIAN)?
            .problem(ctx, sector, dense, singles)
    }
    pub fn new(
        ctx: &Context,
        encoding: Encoding,
        hamiltonian: Operator,
        sector: bool,
        dense: bool,
        singles: bool,
    ) -> Result<Self> {
        let space = if sector {
            Space::Sector
        } else {
            Space::Register
        };
        Self::new_in(ctx, encoding, hamiltonian, space, dense, singles)
    }

    /// A problem in the given [`Space`]: the particle sector, the whole register, or no register
    /// at all (Pauli operators only, for shot-based execution).
    pub fn new_in(
        ctx: &Context,
        encoding: Encoding,
        hamiltonian: Operator,
        space: Space,
        dense: bool,
        singles: bool,
    ) -> Result<Self> {
        if hamiltonian.n != encoding.n_qubits() || !hamiltonian.is_hermitian() {
            return Err(invalid(
                "Hamiltonian must be Hermitian and match its encoding",
            ));
        }
        let (basis, conserved) = match space {
            Space::Sector => (Some(Basis::new(ctx, &encoding, true)?), Vec::new()),
            Space::Register => (Some(Basis::new(ctx, &encoding, false)?), Vec::new()),
            Space::Pauli { number_conserving } => {
                if dense {
                    return Err(register_free_refusal("a dense Hamiltonian matrix"));
                }
                let conserved = if number_conserving {
                    encoding.number_operators(ctx)?.into()
                } else {
                    Vec::new()
                };
                (None, conserved)
            }
        };
        Self::build_with_pool(
            ctx,
            encoding,
            hamiltonian,
            basis,
            conserved,
            "fermionic",
            dense,
            singles,
        )
    }

    /// Construct a relativistic two-component spinor / GHF problem conserving total particle number
    /// with the `"spin-orbit"` excitation pool (allowing spin-flipping SOC couplings).
    pub fn spinor_in(
        ctx: &Context,
        encoding: Encoding,
        hamiltonian: Operator,
        space: Space,
        dense: bool,
        singles: bool,
    ) -> Result<Self> {
        if hamiltonian.n != encoding.n_qubits() || !hamiltonian.is_hermitian() {
            return Err(invalid(
                "Hamiltonian must be Hermitian and match its encoding",
            ));
        }
        let (basis, conserved) = match space {
            Space::Sector => (Some(Basis::total_number(ctx, &encoding)?), Vec::new()),
            Space::Register => (Some(Basis::new(ctx, &encoding, false)?), Vec::new()),
            Space::Pauli { number_conserving } => {
                if dense {
                    return Err(register_free_refusal("a dense Hamiltonian matrix"));
                }
                let mut ops = Vec::new();
                if number_conserving {
                    let [na, nb] = encoding.number_operators(ctx)?;
                    ops.push(na.add(ctx, &nb, C64::new(1., 0.))?);
                }
                (None, ops)
            }
        };
        Self::build_with_pool(
            ctx,
            encoding,
            hamiltonian,
            basis,
            conserved,
            "spin-orbit",
            dense,
            singles,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_with_pool(
        ctx: &Context,
        encoding: Encoding,
        hamiltonian: Operator,
        basis: Option<Basis>,
        conserved: Vec<Operator>,
        pool_name: &str,
        dense: bool,
        singles: bool,
    ) -> Result<Self> {
        check_invariant(basis.as_ref(), &conserved, ctx, &hamiltonian)?;
        let built = crate::quantum::pools::build(ctx, &encoding, pool_name, singles)?;
        let pool = built.operators;
        for g in &pool {
            check_invariant(basis.as_ref(), &conserved, ctx, &g.operator)?;
        }
        let dense = match (&basis, dense) {
            (Some(basis), true) => Some(hamiltonian.dense(ctx, basis)?),
            _ => None,
        };
        Ok(Self {
            context: ctx.clone(),
            encoding,
            hamiltonian,
            basis,
            conserved,
            pool,
            dense,
            precision: Precision::Float64,
            sparse: None,
            pool_screenable: built.screenable,
            growth_members: built.growth_members,
            hva: None,
            pool_name: pool_name.into(),
            taper: None,
            dropped: Vec::new(),
            factorized: None,
            momentum: None,
            point_symmetry: None,
            _pool_metadata: built.memory,
        })
    }
    /// Whether this problem holds no state-vector register (see [`Space::Pauli`]).
    pub fn is_register_free(&self) -> bool {
        self.basis.is_none()
    }
    /// The state-vector register, or a refusal that names `what` needed it. Every state-vector
    /// method reaches the basis through here, so a register-free problem can never compute one
    /// silently.
    pub fn register(&self, what: &str) -> Result<&Basis> {
        self.basis
            .as_ref()
            .ok_or_else(|| register_free_refusal(what))
    }
    /// Check that `operator` preserves the particle sector: with the basis when there is one,
    /// with the number operators when the problem is register-free and asked for the check.
    pub fn check_invariant(&self, operator: &Operator) -> Result<()> {
        check_invariant(
            self.basis.as_ref(),
            &self.conserved,
            &self.context,
            operator,
        )
    }
    /// The state a result carries: the prepared one, or the empty state of a register-free problem.
    pub(in crate::quantum) fn result_state(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
    ) -> Result<State> {
        if self.is_register_free() {
            return State::zeros(&self.context, self.precision, 0);
        }
        self.prepare_state(indices, parameters, trotter)
    }
    /// `<r| [H, G_k] |r>` for every pool generator at the reference determinant `r`: the first
    /// ADAPT screening of a register-free problem, exact and `O(pool * terms^2)`, with no state.
    pub(in crate::quantum) fn reference_gradients(&self) -> Result<Vec<f64>> {
        let reference = self.register_reference();
        self.pool
            .iter()
            .map(|g| {
                Ok(self
                    .hamiltonian
                    .commutator(&self.context, &g.operator)?
                    .expectation_at(reference)
                    .re)
            })
            .collect()
    }
    /// The register width the Hamiltonian and the states live on: the encoding's, or the tapered
    /// one.
    pub fn n_qubits(&self) -> usize {
        self.hamiltonian.n
    }
    /// The reference determinant on that register (occupation bits, qubit zero the most
    /// significant).
    pub fn register_reference(&self) -> u64 {
        let reference = self.encoding.reference();
        self.taper.as_ref().map_or(reference, |t| t.bits(reference))
    }
    /// State storage: complex128 (`Float64`, the default) or complex64.
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }
    /// Measure the energy of circuit executions in the `L + 1` bases of `plan`, the double
    /// factorization of the molecular model this problem was built from. Refuses what it cannot
    /// express, naming it: a tapered register, a Hamiltonian-variational ansatz, an encoding that
    /// is not Jordan-Wigner, a plan of another register width.
    pub fn factorize(&mut self, plan: FactorizedMeasurement) -> Result<()> {
        if self.encoding.mapping != "jordan_wigner" {
            return Err(invalid(
                "the factorized measurement reads occupations off the computational basis, \
                 which is the Jordan-Wigner picture; measure other mappings on the commuting \
                 groups",
            ));
        }
        if self.taper.is_some() || self.hva.is_some() {
            return Err(invalid(
                "the factorized measurement does not apply to a tapered register or a \
                 Hamiltonian-variational ansatz",
            ));
        }
        if plan
            .bases
            .iter()
            .any(|b| b.observable.n != self.hamiltonian.n)
        {
            return Err(invalid("the measurement plan is not for this register"));
        }
        self.factorized = Some(plan);
        Ok(())
    }
    pub fn use_pool(&mut self, name: &str, singles: bool) -> Result<()> {
        if self.taper.is_some() {
            return Err(invalid("choose the pool before tapering the register"));
        }
        let pool = crate::quantum::pools::build_with_crystal_symmetry(
            &self.context,
            &self.encoding,
            name,
            singles,
            self.momentum.as_ref(),
            self.point_symmetry.as_ref(),
        )?;
        for generator in &pool.operators {
            self.check_invariant(&generator.operator)?;
        }
        self.pool = pool.operators;
        self.pool_screenable = pool.screenable;
        self.growth_members = pool.growth_members;
        self._pool_metadata = pool.memory;
        self.pool_name = name.into();
        self.hva = None;
        Ok(())
    }
    pub(in crate::quantum) fn growth_steps(&self, indices: &[usize]) -> Result<usize> {
        let mut count = 0;
        let mut i = 0;
        while i < indices.len() {
            let index = indices[i];
            if index >= self.pool.len() {
                return Err(invalid("generator index exceeds pool"));
            }
            if self.pool_screenable[index] {
                i += 1;
            } else {
                if i + 1 >= indices.len()
                    || !self
                        .growth_members
                        .iter()
                        .any(|m| m.as_slice() == &indices[i..i + 2])
                {
                    return Err(invalid("checkpoint splits a coupled-exchange growth step"));
                }
                i += 2;
            }
            count += 1;
        }
        Ok(count)
    }
}
