use super::{
    by_precision, invalid, CapturedPoint, CapturedScreen, Operator, QuantumProblem,
    RESTRICTED_SUPPORT,
};
use crate::{
    math::storage::Precision,
    quantum::{
        amplitude::{self, inner_wide, Amplitude, State},
        evolution, optimizer, parallel, Basis, C64,
    },
    Buffer, Result,
};
use num_complex::Complex32;

impl QuantumProblem {
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
    pub(in crate::quantum) fn probe_capturing<'p>(
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
    pub(in crate::quantum) fn importances_from_row(
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
    pub(in crate::quantum) fn prepare_buffer<A: Amplitude>(
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
    pub(in crate::quantum) fn pool_gradients_with_centered_hpsi(
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
        let threads = parallel::threads_for(ctx, work, self.pool.len());
        let mut lanes = (0..threads)
            .map(|_| Ok((ctx.allocate::<C64>(d)?, ctx.allocate::<C64>(d)?)))
            .collect::<Result<Vec<_>>>()?;
        let mut out = vec![0.; self.pool.len()];
        parallel::parallel_tasks(ctx, &mut out, &mut lanes, |first, values, (g_buf, work)| {
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
        });
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
        let threads = parallel::threads_for(ctx, work, self.pool.len());
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
        parallel::parallel_tasks(
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
    ) -> Result<crate::quantum::hva::Expanded<'a>> {
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
        Ok(crate::quantum::hva::Expanded {
            operators,
            parameters: parameters.to_vec(),
            links: (0..parameters.len()).map(|i| (i, 1.)).collect(),
        })
    }
}
