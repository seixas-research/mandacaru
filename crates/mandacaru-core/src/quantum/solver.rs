use super::amplitude::{self, inner_wide, Amplitude, State};
use super::factorized_measurement::FactorizedMeasurement;
use super::{
    algebra::Generator, evolution, optimizer, tapering::Taper, Basis, Encoding, Operator, C64,
};
use crate::{math::storage::Precision, product, Buffer, Context, Error, Reservation, Result};
use num_complex::Complex32;
use std::{path::Path, time::Instant};

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
    pub sparse: Option<super::sparse::SparseMatrix>,
    pub pool_screenable: Vec<bool>,
    pub growth_members: Vec<Vec<usize>>,
    pub hva: Option<super::hva::HVA>,
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
    pub(super) _pool_metadata: Reservation,
}
/// Shared slot capturing `(parameters, energy, gradient, state, centered_hpsi)` at an optimizer's
/// evaluated step (`centered_hpsi` is `Some` when the gradient pass left `(H - I) psi` intact).
pub(super) struct CapturedPoint {
    pub parameters: Vec<f64>,
    pub energy: f64,
    pub gradient: Vec<f64>,
    pub state: State,
    pub centered_hpsi: Option<Buffer<C64>>,
}

/// Shared capture slot filled whenever an optimizer requests the deferred gradient of a trial.
pub(super) type CapturedScreen = std::rc::Rc<std::cell::RefCell<Option<CapturedPoint>>>;

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
        check_invariant(basis.as_ref(), &conserved, ctx, &hamiltonian)?;
        let built = super::pools::build(ctx, &encoding, "fermionic", singles)?;
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
            pool_name: "fermionic".into(),
            taper: None,
            dropped: Vec::new(),
            factorized: None,
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
    pub(super) fn result_state(
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
    pub(super) fn reference_gradients(&self) -> Result<Vec<f64>> {
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
        let pool = super::pools::build(&self.context, &self.encoding, name, singles)?;
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
    pub(super) fn growth_steps(&self, indices: &[usize]) -> Result<usize> {
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
    pub fn evaluate(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
    ) -> Result<evolution::Evaluation> {
        self.evaluate_with_mode(indices, parameters, trotter, true)
    }
    /// Like [`Self::evaluate`], letting the caller enable or disable sparse forward/tangent kernels.
    pub fn evaluate_with_mode(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
        sparse_kernels: bool,
    ) -> Result<evolution::Evaluation> {
        by_precision!(
            self.precision,
            A => self.evaluate_as::<A>(indices, parameters, trotter, None, sparse_kernels)
        )
    }
    /// Evaluates energy and parameter gradients starting from an arbitrary float64 state `initial`.
    pub fn evaluate_from_state(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
        initial: &[C64],
    ) -> Result<evolution::Evaluation> {
        if self.precision != Precision::Float64 {
            return Err(invalid("state-seeded evaluations require float64"));
        }
        self.evaluate_as::<C64>(indices, parameters, trotter, Some(initial), true)
    }
    fn evaluate_as<A: Amplitude>(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
        initial: Option<&[A]>,
        sparse_kernels: bool,
    ) -> Result<evolution::Evaluation> {
        let basis = self.register("a state-vector energy evaluation")?;
        if let Some(hva) = &self.hva {
            let expanded = hva.expand(&self.pool, indices, parameters)?;
            let mut out = evolution::evaluate_from_with_mode::<A>(
                &self.context,
                basis,
                &self.hamiltonian,
                self.dense.as_ref().map(|d| d.as_slice()),
                self.sparse.as_ref(),
                &expanded.operators,
                &expanded.parameters,
                true,
                initial,
                sparse_kernels,
            )?;
            let mut gradient = vec![0.; parameters.len()];
            for (g, (i, w)) in out.gradient.iter().zip(expanded.links) {
                gradient[i] += w * g;
            }
            out.gradient = gradient;
            return Ok(out);
        }
        let ops = self.product_operators(indices)?;
        evolution::evaluate_from_with_mode::<A>(
            &self.context,
            basis,
            &self.hamiltonian,
            self.dense.as_ref().map(|d| d.as_slice()),
            self.sparse.as_ref(),
            &ops,
            parameters,
            trotter,
            initial,
            sparse_kernels,
        )
    }
    /// The energy at `parameters` with the gradient deferred: what an optimizer's line search
    /// needs to test a trial point. The gradient is computed from the state this call prepared,
    /// by the code [`QuantumProblem::evaluate`] runs, so the energy and the gradient are its bits;
    /// a point whose gradient is never read costs the forward half only. A summed (non-product)
    /// ansatz keeps its derivative rows in the forward pass, so it defers only the final
    /// contractions; an HVA ansatz, whose gradient is folded onto its shared angles, is evaluated
    /// in full.
    pub fn probe<'p>(
        &'p self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
    ) -> Result<optimizer::Probe<'p>> {
        self.probe_with_mode(indices, parameters, trotter, true)
    }
    /// Like [`Self::probe`], letting the caller enable or disable sparse forward/tangent kernels.
    pub fn probe_with_mode<'p>(
        &'p self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
        sparse_kernels: bool,
    ) -> Result<optimizer::Probe<'p>> {
        by_precision!(
            self.precision,
            A => self.probe_as::<A>(indices, parameters, trotter, None, sparse_kernels)
        )
    }
    /// Like [`Self::probe_with_mode`], capturing `(parameters, energy, gradient, state,
    /// centered_hpsi)` whenever the optimizer accepts a point and evaluates its gradient.
    pub(super) fn probe_capturing<'p>(
        &'p self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
        capture: CapturedScreen,
        sparse_kernels: bool,
    ) -> Result<optimizer::Probe<'p>> {
        by_precision!(
            self.precision,
            A => self.probe_as::<A>(indices, parameters, trotter, Some(capture), sparse_kernels)
        )
    }
    fn probe_as<'p, A: Amplitude>(
        &'p self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
        capture: Option<CapturedScreen>,
        sparse_kernels: bool,
    ) -> Result<optimizer::Probe<'p>> {
        if self.hva.is_some() {
            let full = self.evaluate_as::<A>(indices, parameters, trotter, None, sparse_kernels)?;
            if let Some(slot) = capture {
                let gradient = full.gradient.clone();
                *slot.borrow_mut() = Some(CapturedPoint {
                    parameters: parameters.to_vec(),
                    energy: full.energy,
                    gradient: gradient.clone(),
                    state: full.state,
                    centered_hpsi: None,
                });
                return Ok(optimizer::Probe::ready(full.energy, gradient));
            }
            return Ok(optimizer::Probe::ready(full.energy, full.gradient));
        }
        let basis = self.register("a state-vector energy evaluation")?;
        let ops = self.product_operators(indices)?;
        let theta = parameters.to_vec();
        let context = &self.context;
        let forward = evolution::forward_from_with_mode::<A>(
            context,
            basis,
            &self.hamiltonian,
            self.dense.as_ref().map(|d| d.as_slice()),
            self.sparse.as_ref(),
            &ops,
            &theta,
            trotter,
            None,
            sparse_kernels,
        )?;
        let energy = forward.energy();
        Ok(optimizer::Probe::deferred(energy, move || {
            if let Some(slot) = capture {
                let (eval, hpsi) =
                    forward.finish_with_hpsi(context, basis, &ops, &theta, trotter)?;
                let gradient = eval.gradient.clone();
                *slot.borrow_mut() = Some(CapturedPoint {
                    parameters: theta,
                    energy: eval.energy,
                    gradient: gradient.clone(),
                    state: eval.state,
                    centered_hpsi: hpsi,
                });
                return Ok(gradient);
            }
            Ok(forward
                .finish(context, basis, &ops, &theta, trotter)?
                .gradient)
        }))
    }
    /// The pool generators of `indices`, in order.
    fn product_operators(&self, indices: &[usize]) -> Result<Vec<&Operator>> {
        indices
            .iter()
            .map(|&i| {
                self.pool
                    .get(i)
                    .map(|g| &g.operator)
                    .ok_or_else(|| invalid("generator index exceeds pool"))
            })
            .collect()
    }
    /// `|dE/dtheta_i|` at the angles with `theta_i` set to zero, for every position `i` of a
    /// product ansatz: each equals the `i`-th gradient entry of [`QuantumProblem::evaluate`] at
    /// that point, in the same bits, from one shared forward prefix and a sweep that stops at
    /// `i` (see [`evolution::importances`]).
    pub fn importances(
        &self,
        indices: &[usize],
        parameters: &[f64],
    ) -> Result<evolution::Importances> {
        self.importances_with_mode(indices, parameters, true)
    }
    /// Like [`Self::importances`], letting the caller enable or disable sparse prefix tracking.
    pub fn importances_with_mode(
        &self,
        indices: &[usize],
        parameters: &[f64],
        sparse_kernels: bool,
    ) -> Result<evolution::Importances> {
        if self.hva.is_some() {
            return Err(invalid(
                "operator importances belong to a product ansatz; an HVA ties its parameters",
            ));
        }
        by_precision!(
            self.precision,
            A => self.importances_as::<A>(indices, parameters, sparse_kernels)
        )
    }
    fn importances_as<A: Amplitude>(
        &self,
        indices: &[usize],
        parameters: &[f64],
        sparse_kernels: bool,
    ) -> Result<evolution::Importances> {
        let basis = self.register("an importance evaluation")?;
        evolution::importances_with_mode::<A>(
            &self.context,
            basis,
            &self.hamiltonian,
            self.dense.as_ref().map(|d| d.as_slice()),
            self.sparse.as_ref(),
            &self.product_operators(indices)?,
            parameters,
            sparse_kernels,
        )
    }
    /// Importances starting from an explicit computational basis `start_row` (for subspace ensembles).
    pub(super) fn importances_from_row(
        &self,
        indices: &[usize],
        parameters: &[f64],
        start_row: usize,
        sparse_kernels: bool,
    ) -> Result<evolution::Importances> {
        if self.hva.is_some() {
            return Err(invalid(
                "operator importances belong to a product ansatz; an HVA ties its parameters",
            ));
        }
        let basis = self.register("an importance evaluation")?;
        let ops = self.product_operators(indices)?;
        by_precision!(
            self.precision,
            A => evolution::importances_from_row::<A>(
                &self.context,
                basis,
                &self.hamiltonian,
                self.dense.as_ref().map(|d| d.as_slice()),
                self.sparse.as_ref(),
                &ops,
                parameters,
                start_row,
                sparse_kernels,
            )
        )
    }
    pub fn prepare_state(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
    ) -> Result<State> {
        by_precision!(self.precision, A => self.prepare_state_as::<A>(indices, parameters, trotter))
    }
    fn prepare_state_as<A: Amplitude>(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
    ) -> Result<State> {
        Ok(A::wrap(
            self.prepare_buffer::<A>(indices, parameters, trotter)?,
        ))
    }
    pub(super) fn prepare_buffer<A: Amplitude>(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
    ) -> Result<Buffer<A>> {
        let basis = self.register("prepare_state")?;
        let expanded = self.sequence(indices, parameters)?;
        evolution::state_from::<A>(
            &self.context,
            basis,
            &expanded.operators,
            &expanded.parameters,
            trotter || self.hva.is_some(),
            false,
            None,
        )
    }
    /// `2 Re <H psi | G_i psi>` for every pool generator (the Hamiltonian
    /// without its identity term when `centered`). `H psi` is stored in the
    /// state precision, `G_i psi` is a float64 row, the sums are float64.
    pub fn pool_gradients(&self, state: &State, centered: bool) -> Result<Vec<f64>> {
        self.pool_gradients_for(state, centered, &self.hamiltonian)
    }
    /// Like [`Self::pool_gradients`], evaluated against an explicit Pauli `hamiltonian` (for
    /// example, a top-term surrogate operator used during proposal screening).
    pub fn pool_gradients_for(
        &self,
        state: &State,
        centered: bool,
        hamiltonian: &Operator,
    ) -> Result<Vec<f64>> {
        match state {
            State::Float64(buffer) => {
                self.pool_gradients_of::<C64>(hamiltonian, buffer.as_slice(), centered, None)
            }
            State::Float32(buffer) => {
                self.pool_gradients_of::<Complex32>(hamiltonian, buffer.as_slice(), centered, None)
            }
        }
    }
    /// Exact diagonal second-order curvature `<psi | [[H, G_i], G_i] | psi>` (= `d^2 E / d theta_i^2 |_{0}`)
    /// for every generator `G_i` in the pool on `state`.
    pub fn pool_curvatures(&self, state: &State) -> Result<Vec<f64>> {
        match state {
            State::Float64(buffer) => self.pool_curvatures_of::<C64>(buffer.as_slice()),
            State::Float32(buffer) => self.pool_curvatures_of::<Complex32>(buffer.as_slice()),
        }
    }
    /// Like [`Self::pool_gradients`] with `centered = true`, reusing a precomputed `(H - I) psi`
    /// float64 buffer (such as the one retained from [`evolution::Forward::finish_with_hpsi`]).
    pub(super) fn pool_gradients_with_centered_hpsi(
        &self,
        state: &State,
        hpsi_wide: &[C64],
    ) -> Result<Vec<f64>> {
        let basis = self.register("pool gradients on a local state")?;
        match state {
            State::Float64(buffer) => {
                // Zero-copy reuse of `(H - I) psi` directly in Float64 without allocating `D`.
                self.pool_gradients_with_slice::<C64, C64>(basis, buffer.as_slice(), hpsi_wide)
            }
            State::Float32(buffer) => self.pool_gradients_of::<Complex32>(
                &self.hamiltonian,
                buffer.as_slice(),
                true,
                Some(hpsi_wide),
            ),
        }
    }
    pub fn pool_curvatures_of<A: Amplitude>(&self, state: &[A]) -> Result<Vec<f64>> {
        let ctx = &self.context;
        let basis = self.register("pool curvatures on a local state")?;
        let d = basis.len();
        if state.len() != d || !amplitude::finite(state) {
            return Err(invalid("state must be finite and match the operator basis"));
        }
        let mut hpsi = ctx.allocate::<C64>(d)?;
        self.hamiltonian
            .add_centered_into(basis, state, C64::new(1., 0.), hpsi.as_mut_slice());
        let hpsi_slice = hpsi.as_slice();
        let work = self
            .pool
            .len()
            .saturating_mul(d)
            .saturating_mul(self.hamiltonian.terms.len().max(1));
        let threads = super::parallel::threads_for(ctx, work, self.pool.len());
        let mut lanes = (0..threads)
            .map(|_| Ok((ctx.allocate::<C64>(d)?, ctx.allocate::<C64>(d)?)))
            .collect::<Result<Vec<_>>>()?;
        let mut out = vec![0.; self.pool.len()];
        super::parallel::parallel_tasks(
            ctx,
            &mut out,
            &mut lanes,
            |first, values, (g_buf, work)| {
                let g_slice = g_buf.as_mut_slice();
                let w_slice = work.as_mut_slice();
                for (k, value) in values.iter_mut().enumerate() {
                    let op = &self.pool[first + k].operator;
                    op.apply_into(basis, state, g_slice);
                    w_slice.fill(C64::default());
                    self.hamiltonian
                        .add_centered_into(basis, g_slice, C64::new(1., 0.), w_slice);
                    let ghg = inner_wide(g_slice, w_slice).re;
                    op.apply_into(basis, g_slice, w_slice);
                    let hgg = inner_wide(hpsi_slice, w_slice).re;
                    *value = 2. * (ghg + hgg);
                }
            },
        );
        Ok(out)
    }
    fn pool_gradients_of<A: Amplitude>(
        &self,
        hamiltonian: &Operator,
        state: &[A],
        centered: bool,
        precomputed_hpsi_wide: Option<&[C64]>,
    ) -> Result<Vec<f64>> {
        let ctx = &self.context;
        let basis = self.register("pool gradients on a local state")?;
        let d = basis.len();
        if let Some(pre) = precomputed_hpsi_wide {
            if pre.len() != d {
                return Err(invalid(
                    "precomputed H psi length must match the operator basis",
                ));
            }
            let mut buf = ctx.allocate::<A>(d)?;
            for (dst, &src) in buf.as_mut_slice().iter_mut().zip(pre) {
                *dst = A::narrow(src);
            }
            return self.pool_gradients_with_slice(basis, state, buf.as_slice());
        }
        let mut buf = ctx.allocate::<A>(d)?;
        let mut acc = ctx.allocate::<C64>(A::accumulator_len(d))?;
        A::with_wide(buf.as_mut_slice(), acc.as_mut_slice(), |wide| {
            if centered {
                hamiltonian.add_centered_into(basis, state, C64::new(1., 0.), wide)
            } else {
                hamiltonian.add_into(basis, state, C64::new(1., 0.), wide)
            }
        });
        self.pool_gradients_with_slice(basis, state, buf.as_slice())
    }
    fn pool_gradients_with_slice<A: Amplitude, H: Amplitude>(
        &self,
        basis: &Basis,
        state: &[A],
        hpsi: &[H],
    ) -> Result<Vec<f64>> {
        let ctx = &self.context;
        let d = basis.len();
        if state.len() != d || hpsi.len() != d || !amplitude::finite(state) {
            return Err(invalid("state must be finite and match the operator basis"));
        }
        // A sparse state restricts each generator's work to the rows it can write. The route is
        // chosen once from the data (the support of the state and the finiteness of `H psi`),
        // never from the thread count, and gives the dense bits: see `Operator::apply_support_into`
        // and `amplitude::inner_wide_rows`. A non-finite `H psi` would turn a skipped `0 * inf`
        // into a NaN the dense dot has, so it takes the dense route.
        let nonzero = state.iter().filter(|a| **a != A::default()).count();
        let restricted = nonzero.saturating_mul(RESTRICTED_SUPPORT) < d && amplitude::finite(hpsi);
        let _support_bytes = ctx.reserve(if restricted {
            nonzero.saturating_mul(size_of::<usize>())
        } else {
            0
        })?;
        let support = restricted.then(|| {
            state
                .iter()
                .enumerate()
                .filter(|(_, a)| **a != A::default())
                .map(|(row, _)| row)
                .collect::<Vec<_>>()
        });
        let support = support.as_deref();
        // Every generator is independent: the pool splits across threads, each with its own row
        // (clean, `+0` throughout, between generators on the restricted route) and its own list
        // of the rows a generator wrote.
        let work = match support {
            Some(support) => self
                .pool
                .len()
                .saturating_mul(support.len())
                .saturating_mul(16),
            None => self.pool.len().saturating_mul(d),
        };
        let threads = super::parallel::threads_for(ctx, work, self.pool.len());
        // The rows written are fewer than an eighth of the basis (`apply_support_into` refuses
        // otherwise), each an eight-byte index, and a growing list may hold twice that.
        let _written_bytes = ctx.reserve(match support {
            Some(_) => threads.saturating_mul(d).saturating_mul(2),
            None => 0,
        })?;
        let mut lanes = (0..threads)
            .map(|_| Ok((ctx.allocate::<C64>(d)?, Vec::<usize>::new())))
            .collect::<Result<Vec<_>>>()?;
        let mut out = vec![0.; self.pool.len()];
        super::parallel::parallel_tasks(
            ctx,
            &mut out,
            &mut lanes,
            |first, values, (action, written)| {
                let action = action.as_mut_slice();
                for (k, value) in values.iter_mut().enumerate() {
                    let operator = &self.pool[first + k].operator;
                    let restricted = support.is_some_and(|support| {
                        operator.apply_support_into(basis, state, support, action, written)
                    });
                    if restricted {
                        *value = 2. * amplitude::inner_wide_rows(hpsi, action, written).re;
                        for &row in written.iter() {
                            action[row] = C64::default();
                        }
                    } else {
                        operator.apply_into(basis, state, action);
                        *value = 2. * inner_wide(hpsi, action).re;
                        if support.is_some() {
                            action.fill(C64::default());
                        }
                    }
                }
            },
        );
        Ok(out)
    }
    pub fn sequence<'a>(
        &'a self,
        indices: &[usize],
        parameters: &[f64],
    ) -> Result<super::hva::Expanded<'a>> {
        if let Some(hva) = &self.hva {
            return hva.expand(&self.pool, indices, parameters);
        }
        if indices.len() != parameters.len() {
            return Err(invalid("generator indices must match parameters"));
        }
        let operators = indices
            .iter()
            .map(|&i| {
                self.pool
                    .get(i)
                    .map(|g| &g.operator)
                    .ok_or_else(|| invalid("generator index exceeds pool"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(super::hva::Expanded {
            operators,
            parameters: parameters.to_vec(),
            links: (0..parameters.len()).map(|i| (i, 1.)).collect(),
        })
    }
    pub fn solve(&self, options: &SolveOptions, initial: Option<&[f64]>) -> Result<SolveResult> {
        self.solve_sequence(options, initial, None)
    }
    pub fn solve_sequence(
        &self,
        options: &SolveOptions,
        initial: Option<&[f64]>,
        sequence: Option<&[usize]>,
    ) -> Result<SolveResult> {
        self.context
            .run(|| self.solve_inner(options, initial, sequence, None, None))
    }
    pub fn solve_observed(
        &self,
        options: &SolveOptions,
        initial: Option<&[f64]>,
        sequence: Option<&[usize]>,
        observer: &mut GrowthObserver<'_>,
    ) -> Result<SolveResult> {
        self.context
            .run(|| self.solve_inner(options, initial, sequence, Some(observer), None))
    }
    pub fn solve_executed(
        &self,
        options: &SolveOptions,
        initial: Option<&[f64]>,
        observer: Option<&mut GrowthObserver<'_>>,
        execution: &mut dyn Execution,
    ) -> Result<SolveResult> {
        self.context
            .run(|| self.solve_inner(options, initial, None, observer, Some(execution)))
    }
    fn solve_inner(
        &self,
        options: &SolveOptions,
        initial: Option<&[f64]>,
        sequence: Option<&[usize]>,
        mut observer: Option<&mut GrowthObserver<'_>>,
        mut execution: Option<&mut dyn Execution>,
    ) -> Result<SolveResult> {
        if self.is_register_free() && execution.is_none() {
            return Err(register_free_refusal("a native simulation"));
        }
        // A circuit executor measures energies: the state floors do not apply.
        options.validate(if execution.is_none() {
            self.precision
        } else {
            Precision::Float64
        })?;
        if options.adapt && self.hva.is_some() {
            return Err(invalid("HVA is a fixed ansatz"));
        }
        if options
            .checkpoint
            .as_ref()
            .is_some_and(|c| c.every == 0 || !["vqe", "adapt-vqe"].contains(&c.method.as_str()))
        {
            return Err(invalid(
                "a mid-run checkpoint needs a positive cadence and the vqe or adapt-vqe method",
            ));
        }
        if (options.tetris || options.prune) && (!options.adapt || sequence.is_some()) {
            return Err(invalid(
                "tetris and prune shape the growth of a fresh ADAPT run; a fixed ansatz, a resumed \
                 or a transferred sequence has none",
            ));
        }
        if sequence.is_some_and(|s| {
            !options.adapt
                || self.growth_steps(s).is_err()
                || self.growth_steps(s).is_ok_and(|n| n > options.max_growth)
                || s.iter().any(|&i| i >= self.pool.len())
                || initial.is_none_or(|p| p.len() != s.len())
        }) {
            return Err(invalid(
                "resumed ADAPT sequence must fit the pool, parameters and total growth budget",
            ));
        }
        if options.continuation && options.hessian.is_none() {
            return Err(invalid(
                "a continued growth needs the Hessian it stopped with",
            ));
        }
        if let Some(hessian) = &options.hessian {
            let n = sequence.map_or(0, <[usize]>::len);
            if !options.optimizer.method.carries_curvature()
                || sequence.is_none()
                || hessian.len() != n * n
            {
                return Err(invalid(
                    "a carried Hessian needs a carried SLSQP or BFGS ADAPT sequence of as many parameters",
                ));
            }
        }
        let start = Instant::now();
        let ctx = &self.context;
        // SLSQP keeps a Hessian approximation of the parameters, BFGS an inverse Hessian
        // approximation; a growth step that re-optimizes all of them starts from the previous
        // step's, extended by the identity for the appended ones (the identity is the cold start
        // of either), instead of relearning the curvature from the identity.
        let carries_curvature = options.carry_hessian
            && options.adapt
            && options.quenching
            && options.optimizer.method.carries_curvature();
        let mut curvature: Option<Vec<f64>> = None;
        let max_parameters = if options.adapt {
            product(&[
                options.max_growth,
                self.growth_members
                    .iter()
                    .map(Vec::len)
                    .max()
                    .unwrap_or(1)
                    .max(1),
            ])?
        } else {
            self.pool.len()
        };
        // Hold output/history reservations for the entire lifetime of the result.
        let _memory = ctx.reserve(
            product(&[
                options.optimizer.max_evaluations,
                if options.adapt {
                    options
                        .max_growth
                        .checked_add(usize::from(sequence.is_some_and(|s| !s.is_empty())))
                        .ok_or(Error::Overflow)?
                        .max(1)
                } else {
                    1
                },
                8,
            ])?
            .checked_add(product(&[max_parameters, 256])?)
            .ok_or(Error::Overflow)?,
        )?;
        // The energy of the reference determinant. A register-free problem reads it off the
        // Hamiltonian's diagonal Pauli terms (exact, no state); otherwise it is the native
        // evaluation of the reference state.
        let reference = if self.is_register_free() {
            self.hamiltonian
                .expectation_at(self.register_reference())
                .re
        } else {
            self.probe_with_mode(&[], &[], true, options.sparse_kernels)?
                .energy
        };
        let derivatives = !matches!(
            options.optimizer.method,
            optimizer::Method::Cobyla | optimizer::Method::NelderMead | optimizer::Method::Spsa
        );
        let mut indices = Vec::new();
        let mut parameters = Vec::new();
        let mut iterations = Vec::new();
        let mut history = Vec::new();
        let mut steps = 0;
        let mut success = false;
        let mut message = "growth budget exhausted".to_owned();
        let mut final_gradient = 0.;
        let mut seed = None;
        // A run budget exhausted mid-solve (the executor answered with a stop)
        // ends the solve at the best point reached. `measured` holds the last
        // measured energy of the accepted ansatz; without one there is no
        // point to keep and the interruption propagates.
        let mut stopped: Option<String> = None;
        let mut measured: Option<f64> = None;
        let mut pruned: Vec<(usize, String)> = Vec::new();
        let mut reused_eval: Option<CapturedPoint> = None;
        let mut screened_final = false;
        if !options.adapt {
            indices = (0..self.pool.len())
                .filter(|&i| self.pool_screenable[i])
                .collect();
            parameters = initial.map(|x| x.to_vec()).unwrap_or_else(|| {
                vec![self.hva.as_ref().map_or(0., |h| h.seed_angle); indices.len()]
            });
            if parameters.len() != indices.len() {
                return Err(invalid(
                    "initial parameter count differs from the UCCSD pool",
                ));
            }
            // The best point seen, written every `every` evaluations when checkpointing: the
            // optimizer's own iterate may be a trial step, the best is not.
            let mut best: Option<(Vec<f64>, f64)> = None;
            let mut evaluations = 0usize;
            let capture: Option<CapturedScreen> = (execution.is_none()
                && options.reuse_evaluations)
                .then(|| std::rc::Rc::new(std::cell::RefCell::new(None)));
            let mut objective = |t: &[f64]| -> Result<optimizer::Probe<'_>> {
                let v = probe_of(
                    self,
                    execution.as_deref_mut(),
                    &indices,
                    t,
                    options.trotter,
                    derivatives,
                    capture.as_ref().cloned(),
                    options.sparse_kernels,
                )?;
                if let Some(plan) = &options.checkpoint {
                    evaluations += 1;
                    if best.as_ref().is_none_or(|(_, e)| v.energy < *e) {
                        best = Some((t.to_vec(), v.energy));
                    }
                    if evaluations % plan.every == 0 {
                        let (x, e) = best.as_ref().expect("an evaluation was made");
                        write_checkpoint(
                            self,
                            plan,
                            &indices,
                            x,
                            options.trotter,
                            *e,
                            false,
                            evaluations,
                            options.optimizer.method.name(),
                            None,
                        )?;
                    }
                }
                Ok(v)
            };
            let outcome = if options.quenching {
                optimizer::minimize_interruptible_deferred_from(
                    ctx,
                    &parameters,
                    &options.optimizer,
                    None,
                    &mut objective,
                )?
            } else {
                sweep(ctx, &parameters, &options.optimizer, &mut objective)?
            };
            parameters = outcome.parameters;
            history = outcome.history;
            steps = outcome.steps;
            success = outcome.success;
            message = outcome.message;
            measured = Some(outcome.energy);
            stopped = outcome.interrupted;
            reused_eval = capture
                .as_ref()
                .and_then(|c| c.borrow_mut().take())
                .filter(|p| p.parameters == parameters);
            final_gradient = f64::NAN;
            // A measured gradient is two device evaluations per rotation: only a gradient-based
            // optimizer, which already pays for them, reports one. A derivative-free shot-based
            // run (register-free) leaves it NaN rather than spending thousands of jobs on a
            // diagnostic; a simulated one is free and keeps it.
            if stopped.is_none() && (!self.is_register_free() || derivatives) {
                if let Some(point) = reused_eval.as_ref() {
                    final_gradient = point.gradient.iter().fold(0f64, |a, g| a.max(g.abs()));
                } else {
                    let post_capture: Option<CapturedScreen> = (execution.is_none()
                        && options.reuse_evaluations)
                        .then(|| std::rc::Rc::new(std::cell::RefCell::new(None)));
                    let measured_gradient = probe_of(
                        self,
                        execution.as_deref_mut(),
                        &indices,
                        &parameters,
                        options.trotter,
                        true,
                        post_capture.as_ref().cloned(),
                        options.sparse_kernels,
                    )
                    .and_then(optimizer::Probe::into_gradient);
                    match measured_gradient {
                        Ok(gradient) => {
                            final_gradient = gradient.iter().fold(0f64, |a, g| a.max(g.abs()));
                            reused_eval = post_capture
                                .as_ref()
                                .and_then(|c| c.borrow_mut().take())
                                .filter(|p| p.parameters == parameters);
                        }
                        Err(Error::Interrupted(m)) => stopped = Some(m),
                        Err(e) => return Err(e),
                    }
                }
            }
        } else {
            if sequence.is_none() && initial.is_some_and(|p| !p.is_empty()) {
                return Err(invalid("basic ADAPT starts with an empty ansatz"));
            }
            if let Some(sequence) = sequence {
                indices = sequence.to_vec();
                parameters = initial.unwrap().to_vec();
                if options.continuation {
                    curvature = options.hessian.clone().filter(|_| carries_curvature);
                } else if !indices.is_empty() {
                    let warm = options.hessian.is_some();
                    let seed_capture: Option<CapturedScreen> = (execution.is_none()
                        && options.reuse_evaluations)
                        .then(|| std::rc::Rc::new(std::cell::RefCell::new(None)));
                    let mut outcome = optimizer::minimize_deferred_from(
                        ctx,
                        &parameters,
                        &options.optimizer,
                        options.hessian.as_deref(),
                        |t| {
                            probe_of(
                                self,
                                execution.as_deref_mut(),
                                &indices,
                                t,
                                true,
                                derivatives,
                                seed_capture.as_ref().cloned(),
                                options.sparse_kernels,
                            )
                        },
                    )?;
                    let kept = outcome.energy < reference;
                    if kept && carries_curvature {
                        curvature = outcome.hessian.take();
                    }
                    seed = Some(SeedRelaxation {
                        kept,
                        energy: outcome.energy,
                        success: outcome.success,
                        message: outcome.message.clone(),
                        evaluations: outcome.history.len(),
                        steps: outcome.steps,
                        warm,
                    });
                    steps += outcome.steps;
                    history.extend(outcome.history);
                    parameters = outcome.parameters;
                    if kept {
                        reused_eval = seed_capture
                            .as_ref()
                            .and_then(|c| c.borrow_mut().take())
                            .filter(|p| p.parameters == parameters);
                    } else {
                        indices.clear();
                        parameters.clear();
                    }
                }
            }
            let mut delta = None;
            for growth in self.growth_steps(&indices)?..=options.max_growth {
                // Release the objective closure before measured pool screening.
                let (energy, reused_state, reused_hpsi) = match execution.as_mut() {
                    Some(device) => {
                        reused_eval = None;
                        match device.evaluate(self, &indices, &parameters, false) {
                            Err(Error::Interrupted(m)) if measured.is_some() => {
                                stopped = Some(m);
                                break;
                            }
                            other => {
                                let eval = other?;
                                (eval.energy, Some(eval.state), None)
                            }
                        }
                    }
                    None => {
                        if let Some(point) = reused_eval.take() {
                            (point.energy, Some(point.state), point.centered_hpsi)
                        } else {
                            let eval = self.evaluate_with_mode(
                                &indices,
                                &parameters,
                                true,
                                options.sparse_kernels,
                            )?;
                            (eval.energy, Some(eval.state), None)
                        }
                    }
                };
                measured = Some(energy);
                let mut selected = 0;
                final_gradient = 0.;
                let mut gradients = vec![0.; self.pool.len()];
                // Local screening is the first step only (the reference determinant, where the
                // gradients are exact); from the second step on the device measures them all.
                // A register-free problem reads the first step off Pauli terms, never a state.
                if execution.is_none() || indices.is_empty() {
                    let signed = if self.is_register_free() {
                        self.reference_gradients()?
                    } else {
                        let state = reused_state
                            .as_ref()
                            .expect("native screening holds a prepared state");
                        match reused_hpsi.as_ref() {
                            Some(hpsi) => {
                                self.pool_gradients_with_centered_hpsi(state, hpsi.as_slice())?
                            }
                            None => self.pool_gradients(state, true)?,
                        }
                    };
                    for (i, signed) in signed.into_iter().enumerate() {
                        let magnitude = signed.abs();
                        gradients[i] = magnitude;
                        if self.pool_screenable[i] && magnitude > final_gradient {
                            final_gradient = magnitude;
                            selected = i;
                        }
                    }
                }
                if !indices.is_empty() {
                    if let Some(device) = execution.as_mut() {
                        gradients = match device.pool_gradients(self, &indices, &parameters) {
                            Err(Error::Interrupted(m)) => {
                                stopped = Some(m);
                                break;
                            }
                            other => other?,
                        };
                        if gradients.len() != self.pool.len()
                            || gradients.iter().any(|g| !g.is_finite())
                        {
                            return Err(invalid(
                                "external pool gradients have invalid size or values",
                            ));
                        }
                        final_gradient = 0.;
                        for (i, gradient) in gradients.iter_mut().enumerate() {
                            *gradient = gradient.abs();
                            if self.pool_screenable[i] && *gradient > final_gradient {
                                final_gradient = *gradient;
                                selected = i;
                            }
                        }
                    }
                }
                let gradient_ok = options
                    .gradient_tolerance
                    .is_none_or(|t| final_gradient < t);
                let energy_ok = options
                    .energy_tolerance
                    .is_none_or(|t| delta.is_some_and(|d: f64| d.abs() < t));
                if final_gradient < 1e-12 || (gradient_ok && energy_ok) {
                    screened_final = execution.is_none();
                    if execution.is_none() && options.reuse_evaluations {
                        if let Some(state) = reused_state {
                            reused_eval = Some(CapturedPoint {
                                parameters: parameters.clone(),
                                energy,
                                gradient: Vec::new(),
                                state,
                                centered_hpsi: reused_hpsi,
                            });
                        }
                    }
                    success = true;
                    message = "pool gradients converged".into();
                    break;
                }
                if growth == options.max_growth {
                    screened_final = execution.is_none();
                    if execution.is_none() && options.reuse_evaluations {
                        if let Some(state) = reused_state {
                            reused_eval = Some(CapturedPoint {
                                parameters: parameters.clone(),
                                energy,
                                gradient: Vec::new(),
                                state,
                                centered_hpsi: reused_hpsi,
                            });
                        }
                    }
                    break;
                }
                let picks = if options.tetris {
                    disjoint_selection(
                        &self.pool,
                        &self.pool_screenable,
                        &gradients,
                        self.n_qubits(),
                    )
                } else {
                    vec![selected]
                };
                let accepted = indices.len();
                for pick in picks {
                    let members: Vec<_> = self.growth_members[pick]
                        .iter()
                        .copied()
                        .filter(|&i| gradients[i] > 1e-12)
                        .collect();
                    let grown = if members.len() >= 2 {
                        members
                    } else {
                        vec![pick]
                    };
                    for index in grown {
                        indices.push(index);
                        parameters.push(0.);
                    }
                }

                let step_capture: Option<CapturedScreen> = (execution.is_none()
                    && options.reuse_evaluations)
                    .then(|| std::rc::Rc::new(std::cell::RefCell::new(None)));
                let objective = |t: &[f64]| {
                    probe_of(
                        self,
                        execution.as_deref_mut(),
                        &indices,
                        t,
                        true,
                        derivatives,
                        step_capture.as_ref().cloned(),
                        options.sparse_kernels,
                    )
                };
                let start_curvature = curvature
                    .as_deref()
                    .filter(|h| carries_curvature && h.len() == accepted * accepted)
                    .map(|h| extend_hessian(h, accepted, parameters.len()));
                let optimized = if options.quenching {
                    optimizer::minimize_interruptible_deferred_from(
                        ctx,
                        &parameters,
                        &options.optimizer,
                        start_curvature.as_deref(),
                        objective,
                    )
                } else {
                    // Only the parameters this step appended; the earlier ones stay at their optimum.
                    minimize_block(
                        ctx,
                        &parameters,
                        accepted..parameters.len(),
                        &options.optimizer,
                        objective,
                    )
                };
                let mut outcome = match optimized {
                    // Stopped before the grown ansatz was measured once:
                    // growth ends at the last accepted ansatz.
                    Err(Error::Interrupted(m)) if measured.is_some() => {
                        indices.truncate(accepted);
                        parameters.truncate(accepted);
                        stopped = Some(m);
                        break;
                    }
                    other => other?,
                };
                let mut step_energy = outcome.energy;
                parameters = std::mem::take(&mut outcome.parameters);
                curvature = if carries_curvature {
                    outcome.hessian.take()
                } else {
                    None
                };
                reused_eval = if outcome.interrupted.is_none() {
                    step_capture
                        .as_ref()
                        .and_then(|c| c.borrow_mut().take())
                        .filter(|p| p.parameters == parameters)
                } else {
                    None
                };
                if options.prune && outcome.interrupted.is_none() {
                    if let Some(at) = prune_candidate(&parameters) {
                        reused_eval = None;
                        let (mut shorter, mut angles) = (indices.clone(), parameters.clone());
                        shorter.remove(at);
                        angles.remove(at);
                        curvature = curvature.map(|h| {
                            if options.optimizer.method == optimizer::Method::Bfgs {
                                drop_parameter_of_inverse(&h, parameters.len(), at)
                            } else {
                                drop_parameter(&h, parameters.len(), at)
                            }
                        });
                        // The energy of the shorter ansatz at the same angles: not re-optimized.
                        // A stop before it is measured keeps the unpruned ansatz and its energy.
                        let evaluated = match execution.as_mut() {
                            Some(device) => device
                                .evaluate(self, &shorter, &angles, false)
                                .map(|eval| eval.energy),
                            None if options.reuse_evaluations => self
                                .probe_with_mode(&shorter, &angles, true, options.sparse_kernels)
                                .map(|probe| probe.energy),
                            None => self
                                .evaluate_with_mode(&shorter, &angles, true, options.sparse_kernels)
                                .map(|eval| eval.energy),
                        };
                        match evaluated {
                            Err(Error::Interrupted(m)) => {
                                measured = Some(step_energy);
                                stopped = Some(m);
                                break;
                            }
                            other => step_energy = other?,
                        }
                        pruned.push((iterations.len() + 1, self.pool[indices[at]].label.clone()));
                        indices = shorter;
                        parameters = angles;
                    }
                }
                delta = Some(step_energy - energy);
                measured = Some(step_energy);
                steps += outcome.steps;
                iterations.push(Iteration {
                    pool_index: selected,
                    energy: step_energy,
                    max_gradient: final_gradient,
                    optimizer_steps: outcome.steps,
                    num_evaluations: outcome.history.len(),
                    success: outcome.success,
                    message: outcome.message.clone(),
                    num_parameters: parameters.len(),
                });
                history.extend(outcome.history);
                if let Some(plan) = &options.checkpoint {
                    if iterations.len() % plan.every == 0 {
                        write_checkpoint(
                            self,
                            plan,
                            &indices,
                            &parameters,
                            true,
                            step_energy,
                            true,
                            iterations.len(),
                            options.optimizer.method.name(),
                            curvature.as_deref(),
                        )?;
                    }
                }
                if let Some(observer) = observer.as_deref_mut() {
                    observer(&indices, &parameters, step_energy)?;
                }
                if outcome.interrupted.is_some() {
                    // The partly optimized ansatz at its best point is kept.
                    stopped = outcome.interrupted;
                    break;
                }
                // Preserve an inner optimizer's best point and failure status.
                // Growth continues; only the outer pool criteria certify ADAPT
                // convergence, even when the inner tolerance hits roundoff.
            }
        }
        let trotter = options.adapt || options.trotter;
        let final_evaluation = match (execution.as_mut(), stopped.is_some()) {
            (_, true) => None,
            (Some(device), false) => match device.evaluate(self, &indices, &parameters, false) {
                Err(Error::Interrupted(m)) if measured.is_some() => {
                    stopped = Some(m);
                    None
                }
                other => {
                    let eval = other?;
                    Some((eval.energy, eval.state, None))
                }
            },
            (None, false) => {
                if let Some(point) = reused_eval.take() {
                    Some((point.energy, point.state, point.centered_hpsi))
                } else {
                    let eval = self.evaluate_with_mode(
                        &indices,
                        &parameters,
                        trotter,
                        options.sparse_kernels,
                    )?;
                    Some((eval.energy, eval.state, None))
                }
            }
        };
        // After a stop no circuit is submitted: the state comes from the
        // native simulator and the energy is the last one measured.
        let (energy, state, final_hpsi) = match final_evaluation {
            Some(evaluation) => evaluation,
            None => (
                measured.expect("a stop keeps a measured point"),
                self.result_state(&indices, &parameters, trotter)?,
                None,
            ),
        };
        if let Some(reason) = &stopped {
            success = false;
            message = format!("run budget exhausted: {reason}");
        }
        if options.adapt
            && execution.is_none()
            && (!options.reuse_evaluations || !screened_final || stopped.is_some())
        {
            final_gradient = 0.;
            let signed = match final_hpsi.as_ref() {
                Some(hpsi) if options.reuse_evaluations => {
                    self.pool_gradients_with_centered_hpsi(&state, hpsi.as_slice())?
                }
                _ => self.pool_gradients(&state, true)?,
            };
            for (i, signed) in signed.into_iter().enumerate() {
                if self.pool_screenable[i] {
                    final_gradient = final_gradient.max(signed.abs());
                }
            }
        }
        let labels = indices
            .iter()
            .map(|&i| self.pool[i].label.clone())
            .collect();
        let indices_len = indices.len();
        Ok(SolveResult {
            seed,
            energy,
            reference,
            parameters,
            state,
            indices,
            labels,
            iterations,
            history,
            optimizer_steps: steps,
            success,
            message,
            final_gradient,
            budget_stopped: stopped.is_some(),
            elapsed_seconds: start.elapsed().as_secs_f64(),
            pruned,
            hessian: curvature.filter(|h| h.len() == indices_len * indices_len),
            notes: Vec::new(),
            _memory,
        })
    }
}
/// One mid-run checkpoint: the ansatz and parameters so far, marked incomplete. `growth` says
/// whether `count` counts growth steps (ADAPT) or cost evaluations (a fixed ansatz).
#[allow(clippy::too_many_arguments)]
/// The `n x n` Hessian of `m` parameters (row-major) as the start for `n >= m`: the old block,
/// the identity for each appended parameter, no coupling between old and new.
pub(super) fn extend_hessian(old: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut grown = vec![0.; n * n];
    for i in 0..n {
        if i < m {
            grown[i * n..i * n + m].copy_from_slice(&old[i * m..(i + 1) * m]);
        } else {
            grown[i * n + i] = 1.;
        }
    }
    grown
}
/// The INVERSE Hessian of `n` parameters without the one at `at`: the inverse of the reduced
/// Hessian, `H_ij - H_ia H_aj / H_aa` (a Schur complement, not the sub-block).
pub(super) fn drop_parameter_of_inverse(old: &[f64], n: usize, at: usize) -> Vec<f64> {
    let keep: Vec<usize> = (0..n).filter(|&i| i != at).collect();
    let pivot = old[at * n + at];
    keep.iter()
        .flat_map(|&i| keep.iter().map(move |&j| (i, j)))
        .map(|(i, j)| {
            if pivot.abs() > f64::MIN_POSITIVE {
                old[i * n + j] - old[i * n + at] * old[at * n + j] / pivot
            } else {
                old[i * n + j]
            }
        })
        .collect()
}
/// The Hessian of `n` parameters without the one at `at`.
pub(super) fn drop_parameter(old: &[f64], n: usize, at: usize) -> Vec<f64> {
    let keep: Vec<usize> = (0..n).filter(|&i| i != at).collect();
    keep.iter()
        .flat_map(|&i| keep.iter().map(move |&j| (i, j)))
        .map(|(i, j)| old[i * n + j])
        .collect()
}
/// Insert a new uncoupled parameter at index `at` into an `n x n` row-major curvature matrix,
/// yielding an `(n + 1) x (n + 1)` matrix with `1.0` on the new diagonal entry.
pub(super) fn insert_hessian(old: &[f64], n: usize, at: usize) -> Vec<f64> {
    let m = n + 1;
    let mut grown = vec![0.; m * m];
    for i in 0..m {
        if i == at {
            grown[i * m + i] = 1.;
            continue;
        }
        let oi = if i > at { i - 1 } else { i };
        for j in 0..m {
            if j == at {
                continue;
            }
            let oj = if j > at { j - 1 } else { j };
            grown[i * m + j] = old[oi * n + oj];
        }
    }
    grown
}
/// Reset row and column `at` of an `n x n` row-major curvature matrix to the unit vector `e_at`,
/// keeping the remaining `(n - 1) x (n - 1)` block positive definite when replacing an operator.
pub(super) fn reset_hessian_parameter(old: &[f64], n: usize, at: usize) -> Vec<f64> {
    let mut out = old.to_vec();
    for k in 0..n {
        out[at * n + k] = 0.;
        out[k * n + at] = 0.;
    }
    out[at * n + at] = 1.;
    out
}

#[allow(clippy::too_many_arguments)]
fn write_checkpoint(
    problem: &QuantumProblem,
    plan: &Checkpointing,
    indices: &[usize],
    parameters: &[f64],
    product_form: bool,
    energy: f64,
    growth: bool,
    count: usize,
    optimizer: &str,
    hessian: Option<&[f64]>,
) -> Result<()> {
    let status = crate::persistence::status_with_hessian(
        serde_json::json!({
            "complete": false,
            "converged": false,
            (if growth { "iteration" } else { "num_evaluations" }): count,
        }),
        optimizer,
        hessian,
    )?;
    crate::persistence::checkpoint(
        &problem.context,
        problem,
        indices,
        parameters,
        product_form,
        &plan.method,
        status,
        energy,
    )?
    .save(&plan.path)
}

/// Pruning's position weight `alpha` in `exp(-alpha x)`.
pub const PRUNE_ALPHA: f64 = 10.;
/// Fraction of the recent mean amplitude below which the top-ranked operator is removed.
pub const PRUNE_FRACTION: f64 = 0.1;
/// How many of the most recently added operators define that mean.
pub const PRUNE_RECENT: usize = 4;
/// No pruning at or below this many parameters: the ranking needs a position scale.
pub const PRUNE_MIN_OPERATORS: usize = 2;

/// The qubits a generator acts on, as a bit mask over its Pauli terms.
fn support(operator: &crate::quantum::Operator) -> u64 {
    operator
        .terms
        .iter()
        .filter(|t| t.coefficient.norm() > 0.)
        .fold(0, |mask, t| mask | t.x | t.z)
}

/// TETRIS selection: the screenable pool indices of one growth step, the largest gradient
/// first, then each next largest whose support touches no qubit taken so far, until the
/// register is covered or nothing disjoint with a gradient above 1e-12 is left.
fn disjoint_selection(
    pool: &[crate::quantum::algebra::Generator],
    screenable: &[bool],
    gradients: &[f64],
    n_qubits: usize,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..pool.len()).filter(|&i| screenable[i]).collect();
    order.sort_by(|&a, &b| gradients[b].total_cmp(&gradients[a]));
    let full = if n_qubits >= 64 {
        u64::MAX
    } else {
        (1u64 << n_qubits) - 1
    };
    let (mut chosen, mut covered) = (Vec::new(), 0u64);
    for i in order {
        if gradients[i] <= 1e-12 {
            break;
        }
        let mask = support(&pool[i].operator);
        if !chosen.is_empty() && mask & covered != 0 {
            continue;
        }
        chosen.push(i);
        covered |= mask;
        if covered & full == full {
            break;
        }
    }
    chosen
}

/// Pruning's candidate among optimized `parameters` (ansatz order): the position of the one to
/// remove, or `None`.
pub fn prune_candidate(parameters: &[f64]) -> Option<usize> {
    let n = parameters.len();
    if n <= PRUNE_MIN_OPERATORS {
        return None;
    }
    let recent = PRUNE_RECENT.min(n);
    let threshold = PRUNE_FRACTION
        * parameters[n - recent..]
            .iter()
            .map(|t| t.abs())
            .sum::<f64>()
        / recent as f64;
    // The newest operator is never a candidate: it was just selected on the largest gradient.
    let (candidate, _) = parameters[..n - 1]
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let x = (i + 1) as f64 / n as f64;
            (i, (-PRUNE_ALPHA * x).exp() / (t * t))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    (parameters[candidate].abs() < threshold).then_some(candidate)
}

/// One objective evaluation of a solve: a circuit executor's measurement when there is one (its
/// gradient is measured with the energy, as the executor decides, so it is ready), else the
/// native evaluation with its gradient deferred ([`QuantumProblem::probe_with_mode`]).
#[allow(clippy::too_many_arguments)]
fn probe_of<'p>(
    problem: &'p QuantumProblem,
    execution: Option<&mut (dyn Execution + '_)>,
    indices: &[usize],
    parameters: &[f64],
    trotter: bool,
    gradients: bool,
    capture: Option<CapturedScreen>,
    sparse_kernels: bool,
) -> Result<optimizer::Probe<'p>> {
    match execution {
        Some(device) => {
            let value = device.evaluate(problem, indices, parameters, gradients)?;
            Ok(optimizer::Probe::ready(value.energy, value.gradient))
        }
        None => match capture {
            Some(slot) => {
                problem.probe_capturing(indices, parameters, trotter, slot, sparse_kernels)
            }
            None => problem.probe_with_mode(indices, parameters, trotter, sparse_kernels),
        },
    }
}

/// Minimize `objective` over the coordinates `free` of `full`, the others held: the objective and
/// the outcome see the full vector, the optimizer only the free block (and its gradient slice).
fn minimize_block<'p>(
    ctx: &crate::Context,
    full: &[f64],
    free: std::ops::Range<usize>,
    options: &optimizer::Options,
    mut objective: impl FnMut(&[f64]) -> Result<optimizer::Probe<'p>>,
) -> Result<optimizer::Outcome> {
    let mut point = full.to_vec();
    let start = full[free.clone()].to_vec();
    let mut outcome =
        optimizer::minimize_interruptible_deferred_from(ctx, &start, options, None, |block| {
            point[free.clone()].copy_from_slice(block);
            let probe = objective(&point)?;
            let free = free.clone();
            let length = point.len();
            Ok(probe.map_gradient(move |gradient| {
                if gradient.len() == length {
                    gradient[free].to_vec()
                } else {
                    gradient
                }
            }))
        })?;
    let mut parameters = full.to_vec();
    parameters[free].copy_from_slice(&outcome.parameters);
    outcome.parameters = parameters;
    Ok(outcome)
}

/// One sequential sweep (`quenching = false` on a fixed ansatz): parameter `k` alone, `0..k` at
/// their optimized values and `k+1..` at their starting values. The outcome's steps and history
/// are the sums over the one-parameter searches; a stop ends the sweep at the best point.
fn sweep<'p>(
    ctx: &crate::Context,
    initial: &[f64],
    options: &optimizer::Options,
    mut objective: impl FnMut(&[f64]) -> Result<optimizer::Probe<'p>>,
) -> Result<optimizer::Outcome> {
    if initial.len() <= 1 {
        return optimizer::minimize_interruptible_deferred_from(
            ctx, initial, options, None, objective,
        );
    }
    let mut parameters = initial.to_vec();
    let (mut history, mut steps, mut success) = (Vec::new(), 0, true);
    let mut energy = f64::NAN;
    for k in 0..parameters.len() {
        let step = minimize_block(ctx, &parameters, k..k + 1, options, &mut objective)?;
        parameters = step.parameters;
        energy = step.energy;
        history.extend(step.history);
        steps += step.steps;
        success &= step.success;
        if step.interrupted.is_some() {
            return Ok(optimizer::Outcome {
                parameters,
                energy,
                history,
                steps,
                success: false,
                message: step.message,
                hessian: None,
                interrupted: step.interrupted,
            });
        }
    }
    Ok(optimizer::Outcome {
        parameters,
        energy,
        history,
        steps,
        success,
        message: "sequential sweep (quenching=False)".into(),
        hessian: None,
        interrupted: None,
    })
}

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
            if tolerance.is_some_and(|t| t < floors.energy) {
                return Err(invalid(&format!(
                    "optimizer tolerance {:e} Ha is below the {name} state floor {:e} Ha; \
                     raise it or use precision='float64'",
                    tolerance.unwrap(),
                    floors.energy
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
    pub(super) _memory: Reservation,
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

#[cfg(test)]
mod hessian_tests {
    use super::{drop_parameter, drop_parameter_of_inverse, extend_hessian};
    #[test]
    fn dropping_a_parameter_from_an_inverse_gives_the_inverse_of_the_reduced_hessian() {
        // B = [[4, 1, 0.5], [1, 3, -0.4], [0.5, -0.4, 2]]; its inverse, then the inverse of B
        // without parameter 1, from the Schur complement.
        let b =
            nalgebra::DMatrix::from_row_slice(3, 3, &[4., 1., 0.5, 1., 3., -0.4, 0.5, -0.4, 2.]);
        let inverse = b.clone().try_inverse().unwrap();
        let row_major: Vec<f64> = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|ij| inverse[ij])
            .collect();
        let reduced = nalgebra::DMatrix::from_row_slice(2, 2, &[4., 0.5, 0.5, 2.])
            .try_inverse()
            .unwrap();
        let dropped = drop_parameter_of_inverse(&row_major, 3, 1);
        for (k, value) in dropped.iter().enumerate() {
            assert!((value - reduced[(k / 2, k % 2)]).abs() < 1e-12);
        }
    }
    #[test]
    fn a_grown_hessian_keeps_the_old_block_and_adds_the_identity() {
        let old = [2., 0.5, 0.5, 3.];
        assert_eq!(
            extend_hessian(&old, 2, 4),
            [2., 0.5, 0., 0., 0.5, 3., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]
        );
        assert_eq!(extend_hessian(&old, 2, 2), old);
    }
    #[test]
    fn dropping_a_parameter_removes_its_row_and_column() {
        let h = [1., 2., 3., 2., 4., 5., 3., 5., 6.];
        assert_eq!(drop_parameter(&h, 3, 1), [1., 3., 3., 6.]);
        assert_eq!(drop_parameter(&h, 3, 0), [4., 5., 5., 6.]);
    }
}
