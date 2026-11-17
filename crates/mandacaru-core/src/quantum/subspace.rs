//! Weighted shared-unitary VQE/ADAPT; bare expectations, sorted with their states.
use super::amplitude::{inner_wide, stack, Amplitude, State};
use super::solver::by_precision;
use super::{evolution, optimizer, QuantumProblem, SolveOptions};
use crate::quantum::C64;
use crate::{product, Buffer, Error, Reservation, Result};
use nalgebra::DMatrix;
use std::time::Instant;

/// Captured multi-state evaluation from `subspace_probe_capturing` for reuse in screening/final steps.
pub(super) struct CapturedSubspacePoint {
    pub parameters: Vec<f64>,
    pub gradient: Vec<f64>,
    pub energies: Vec<f64>,
    pub states: Vec<State>,
    pub hpsis: Vec<Option<Buffer<C64>>>,
}

/// Output of preparing `k` subspace ansatz states and Rayleigh-Ritz diagonalizing `H`.
pub(super) struct SubspaceDiagonalization {
    pub ensemble_energies: Vec<f64>,
    pub energies: Vec<f64>,
    pub ground_state: State,
    pub states: State,
}

pub(super) type CapturedSubspace = std::rc::Rc<std::cell::RefCell<Option<CapturedSubspacePoint>>>;

pub struct SubspaceResult {
    pub energies: Vec<f64>,
    pub parameters: Vec<f64>,
    pub states: State,
    pub reference: f64,
    pub indices: Vec<usize>,
    pub reference_indices: Vec<usize>,
    pub history: Vec<f64>,
    pub optimizer_failures: Vec<(usize, String)>,
    pub success: bool,
    pub message: String,
    pub final_gradient: f64,
    pub elapsed_seconds: f64,
    /// The Hessian approximation (row-major, `n x n`) of the final ansatz of `n` parameters,
    /// for SLSQP growth (see `SolveResult::hessian`); `None` otherwise.
    pub hessian: Option<Vec<f64>>,
    /// What the caller should know about how the run started (see `SolveResult::notes`).
    pub notes: Vec<String>,
    _memory: Reservation,
}
pub(super) fn validate_subspace_weights(weights: &[f64]) -> Result<()> {
    if weights.is_empty()
        || weights.iter().any(|w| !w.is_finite() || *w <= 0.)
        || weights.windows(2).any(|w| w[1] >= w[0])
    {
        return Err(Error::Invalid(
            "subspace weights must be finite, positive and strictly decreasing".into(),
        ));
    }
    Ok(())
}

impl QuantumProblem {
    pub fn subspace_ref_rows(&self, count: usize) -> Result<Vec<usize>> {
        let states = self.sector_determinants()?;
        if count == 0 || count > states.len() {
            return Err(Error::Invalid(
                "number of subspace states exceeds the particle sector or is zero".into(),
            ));
        }
        Ok(states
            .into_iter()
            .take(count)
            .map(|(_, _, row)| row)
            .collect())
    }
    pub(super) fn subspace_evaluate<A: Amplitude>(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
        ref_row: usize,
        sparse_kernels: bool,
    ) -> Result<evolution::Evaluation> {
        let ops: Vec<_> = indices.iter().map(|&i| &self.pool[i].operator).collect();
        evolution::evaluate_from_row::<A>(
            &self.context,
            self.register("a subspace calculation")?,
            &self.hamiltonian,
            self.dense.as_ref().map(|x| x.as_slice()),
            self.sparse.as_ref(),
            &ops,
            parameters,
            trotter,
            None,
            ref_row,
            sparse_kernels,
        )
    }
    pub(super) fn subspace_probe_with<'p, A: Amplitude>(
        &'p self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
        weights: &[f64],
        ref_rows: &[usize],
        capture: Option<CapturedSubspace>,
        sparse_kernels: bool,
    ) -> Result<optimizer::Probe<'p>> {
        let basis = self.register("a subspace calculation")?;
        let ops: Vec<_> = indices.iter().map(|&i| &self.pool[i].operator).collect();
        let theta = parameters.to_vec();
        let mut forwards = Vec::with_capacity(weights.len());
        let mut weighted_energy = 0.;
        for (w, &ref_row) in weights.iter().zip(ref_rows) {
            let forward = evolution::forward_from_row::<A>(
                &self.context,
                basis,
                &self.hamiltonian,
                self.dense.as_ref().map(|x| x.as_slice()),
                self.sparse.as_ref(),
                &ops,
                &theta,
                trotter,
                None,
                ref_row,
                sparse_kernels,
            )?;
            weighted_energy += w * forward.energy();
            forwards.push((*w, forward));
        }
        let ctx = &self.context;
        Ok(optimizer::Probe::deferred(weighted_energy, move || {
            let mut gradient = vec![0.; theta.len()];
            let capturing = capture.is_some();
            let mut energies = Vec::with_capacity(if capturing { forwards.len() } else { 0 });
            let mut states = Vec::with_capacity(if capturing { forwards.len() } else { 0 });
            let mut hpsis = Vec::with_capacity(if capturing { forwards.len() } else { 0 });
            for (w, forward) in forwards {
                let (eval, hpsi) = if capturing {
                    forward.finish_with_hpsi(ctx, basis, &ops, &theta, trotter)?
                } else {
                    (forward.finish(ctx, basis, &ops, &theta, trotter)?, None)
                };
                for (g, x) in gradient.iter_mut().zip(eval.gradient) {
                    *g += w * x;
                }
                if capturing {
                    energies.push(eval.energy);
                    states.push(eval.state);
                    hpsis.push(hpsi);
                }
            }
            if let Some(slot) = capture {
                *slot.borrow_mut() = Some(CapturedSubspacePoint {
                    parameters: theta,
                    gradient: gradient.clone(),
                    energies,
                    states,
                    hpsis,
                });
            }
            Ok(gradient)
        }))
    }
    /// Precision-dispatched wrapper over [`Self::subspace_probe_with`] for ensemble objectives.
    pub(super) fn subspace_probe_capturing<'p>(
        &'p self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
        weights: &[f64],
        ref_rows: &[usize],
        capture: Option<CapturedSubspace>,
        sparse_kernels: bool,
    ) -> Result<optimizer::Probe<'p>> {
        by_precision!(
            self.precision,
            A => self.subspace_probe_with::<A>(
                indices,
                parameters,
                trotter,
                weights,
                ref_rows,
                capture,
                sparse_kernels,
            )
        )
    }
    /// Prepare a single computational basis `ref_row` under the circuit parameters.
    pub(super) fn subspace_prepare_state(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
        ref_row: usize,
        sparse_kernels: bool,
    ) -> Result<State> {
        by_precision!(
            self.precision,
            A => Ok(self
                .subspace_evaluate::<A>(indices, parameters, trotter, ref_row, sparse_kernels)?
                .state)
        )
    }
    /// Prepare each reference state under the shared circuit, compute bare ensemble expectations,
    /// and Rayleigh-Ritz diagonalize `H` inside their span.
    pub(super) fn subspace_prepare_and_diagonalize(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
        ref_rows: &[usize],
        sparse_kernels: bool,
    ) -> Result<SubspaceDiagonalization> {
        let basis = self.register("a subspace diagonalization")?;
        let d = basis.len();
        let mut ensemble_energies = Vec::with_capacity(ref_rows.len());
        let mut bare_states = Vec::with_capacity(ref_rows.len());
        by_precision!(self.precision, A => {
            for &ref_row in ref_rows {
                let eval = self.subspace_evaluate::<A>(
                    indices,
                    parameters,
                    trotter,
                    ref_row,
                    sparse_kernels,
                )?;
                ensemble_energies.push(eval.energy);
                bare_states.push(eval.state);
            }
        });
        let (energies, states) = self.diagonalize_subspace(&bare_states)?;
        let ground_state = by_precision!(self.precision, A => {
            let mut single = stack::<A>(&self.context, 1, d)?;
            let src = A::view(&states).expect("eigenstates have the problem's precision");
            single.as_mut_slice().copy_from_slice(&src[..d]);
            A::wrap(single)
        });
        Ok(SubspaceDiagonalization {
            ensemble_energies,
            energies,
            ground_state,
            states,
        })
    }
    /// Rayleigh-Ritz diagonalization of `<psi_a | H | psi_b>` inside `span(states)`, returning
    /// sorted variational eigenvalues and rotated orthogonal eigenvectors (`k x d` stacked [`State`]).
    pub(super) fn diagonalize_subspace(&self, states: &[State]) -> Result<(Vec<f64>, State)> {
        let basis = self.register("a subspace diagonalization")?;
        let d = basis.len();
        let k = states.len();
        let ctx = &self.context;
        let mut h_mat = DMatrix::<C64>::zeros(k, k);
        let mut wide = ctx.allocate::<C64>(d)?;
        for b in 0..k {
            match &states[b] {
                State::Float64(buf) => {
                    self.hamiltonian
                        .apply_into(basis, buf.as_slice(), wide.as_mut_slice());
                }
                State::Float32(buf) => {
                    self.hamiltonian
                        .apply_into(basis, buf.as_slice(), wide.as_mut_slice());
                }
            }
            for a in 0..=b {
                let val = match &states[a] {
                    State::Float64(buf) => inner_wide(buf.as_slice(), wide.as_slice()),
                    State::Float32(buf) => inner_wide(buf.as_slice(), wide.as_slice()),
                };
                h_mat[(a, b)] = val;
                if a != b {
                    h_mat[(b, a)] = val.conj();
                } else {
                    h_mat[(a, a)] = C64::new(val.re, 0.);
                }
            }
        }
        let eigen = h_mat.symmetric_eigen();
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
        let energies = order.iter().map(|&i| eigen.eigenvalues[i]).collect();

        let rotated = match self.precision {
            crate::math::storage::Precision::Float64 => {
                let mut out = stack::<C64>(ctx, k, d)?;
                let dst = out.as_mut_slice();
                for (out_idx, &eig_idx) in order.iter().enumerate() {
                    let row_dst = &mut dst[out_idx * d..(out_idx + 1) * d];
                    for (src_idx, state) in states.iter().enumerate() {
                        let coeff = eigen.eigenvectors[(src_idx, eig_idx)];
                        let src = state.complex128()?;
                        for (acc, &amp) in row_dst.iter_mut().zip(src) {
                            *acc += coeff * amp;
                        }
                    }
                }
                State::Float64(out)
            }
            crate::math::storage::Precision::Float32 => {
                let mut out = stack::<num_complex::Complex32>(ctx, k, d)?;
                let mut row_wide = ctx.allocate::<C64>(d)?;
                for (out_idx, &eig_idx) in order.iter().enumerate() {
                    row_wide.as_mut_slice().fill(C64::default());
                    for (src_idx, state) in states.iter().enumerate() {
                        let coeff = eigen.eigenvectors[(src_idx, eig_idx)];
                        let src = state.complex64()?;
                        for (acc, &amp) in row_wide.as_mut_slice().iter_mut().zip(src) {
                            *acc += coeff * amp.widen();
                        }
                    }
                    let row_dst = &mut out.as_mut_slice()[out_idx * d..(out_idx + 1) * d];
                    for (d_val, &w) in row_dst.iter_mut().zip(row_wide.as_slice()) {
                        *d_val = num_complex::Complex32::narrow(w);
                    }
                }
                State::Float32(out)
            }
        };
        Ok((energies, rotated))
    }
    pub fn solve_subspace(
        &self,
        options: &SolveOptions,
        weights: &[f64],
        initial: Option<&[f64]>,
    ) -> Result<SubspaceResult> {
        self.solve_subspace_sequence(options, weights, initial, None)
    }
    /// As [`Self::solve_subspace`], an ADAPT run continuing from `sequence` (a resumed
    /// checkpoint): the stored ansatz is relaxed first, its growth steps count toward
    /// `max_growth`, and growth continues from it.
    pub fn solve_subspace_sequence(
        &self,
        options: &SolveOptions,
        weights: &[f64],
        initial: Option<&[f64]>,
        sequence: Option<&[usize]>,
    ) -> Result<SubspaceResult> {
        by_precision!(
            self.precision,
            A => self.solve_subspace_as::<A>(options, weights, initial, sequence)
        )
    }
    fn solve_subspace_as<A: Amplitude>(
        &self,
        options: &SolveOptions,
        weights: &[f64],
        initial: Option<&[f64]>,
        sequence: Option<&[usize]>,
    ) -> Result<SubspaceResult> {
        self.context.run(|| {
            options.validate(self.precision)?;
            validate_subspace_weights(weights)?;
            let start = Instant::now();
            let ctx = &self.context;
            let k = weights.len();
            let d = self.register("a subspace calculation")?.len();
            let ref_rows = self.subspace_ref_rows(k)?;
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
            let _workspace = ctx.reserve(product(&[
                max_parameters.checked_add(k).ok_or(Error::Overflow)?,
                64,
            ])?)?;
            let runs = if options.adapt { options.max_growth } else { 1 };
            let memory = ctx.reserve(
                product(&[runs, options.optimizer.max_evaluations, 8])?
                    .checked_add(product(&[runs, 128])?)
                    .ok_or(Error::Overflow)?,
            )?;
            if options.continuation && options.hessian.is_none() {
                return Err(Error::Invalid(
                    "a continued growth needs the Hessian it stopped with".into(),
                ));
            }
            if let Some(hessian) = &options.hessian {
                let n = sequence.map_or(0, <[usize]>::len);
                if !options.optimizer.method.carries_curvature()
                    || !options.adapt
                    || sequence.is_none()
                    || hessian.len() != n * n
                {
                    return Err(Error::Invalid(
                        "a carried Hessian needs a carried SLSQP or BFGS ADAPT sequence of as many \
                         parameters"
                            .into(),
                    ));
                }
            }
            if sequence.is_some_and(|s| {
                !options.adapt
                    || self.growth_steps(s).is_err()
                    || self.growth_steps(s).is_ok_and(|n| n > options.max_growth)
                    || initial.is_none_or(|p| p.len() != s.len())
            }) {
                return Err(Error::Invalid(
                    "a resumed subspace ADAPT sequence must fit the pool, its parameters and the \
                     growth budget"
                        .into(),
                ));
            }
            let mut indices: Vec<usize> = if let Some(sequence) = sequence {
                sequence.to_vec()
            } else if options.adapt {
                vec![]
            } else {
                (0..self.pool.len())
                    .filter(|&i| self.pool_screenable[i])
                    .collect()
            };
            let mut parameters = initial
                .map(|x| x.to_vec())
                .unwrap_or_else(|| vec![0.; indices.len()]);
            if parameters.len() != indices.len() {
                return Err(Error::Invalid(
                    "subspace initial parameter count differs from the ansatz".into(),
                ));
            }
            let mut reused: Option<CapturedSubspacePoint> = None;
            let objective = |indices: &[usize],
                             theta: &[f64],
                             capture: Option<CapturedSubspace>|
             -> Result<optimizer::Probe<'_>> {
                self.subspace_probe_with::<A>(
                    indices,
                    theta,
                    if options.adapt { true } else { options.trotter },
                    weights,
                    &ref_rows,
                    capture,
                    options.sparse_kernels,
                )
            };
            let reference = self
                .subspace_evaluate::<A>(&[], &[], true, ref_rows[0], options.sparse_kernels)?
                .energy;
            let mut history = Vec::new();
            let mut optimizer_failures = Vec::new();
            let mut success = false;
            let mut message = "growth budget exhausted".to_owned();
            let mut delta = None;
            let mut final_gradient = 0.;
            let mut screened_final = false;
            // The SLSQP Hessian or BFGS inverse Hessian approximation travels from one growth
            // step to the next, extended by the identity for the appended parameters (as in
            // `solve`).
            let carries_curvature = options.carry_hessian
                && options.adapt
                && options.optimizer.method.carries_curvature();
            let mut curvature: Option<Vec<f64>> = None;
            if !options.adapt {
                let capture: Option<CapturedSubspace> = options
                    .reuse_evaluations
                    .then(|| std::rc::Rc::new(std::cell::RefCell::new(None)));
                let out = optimizer::minimize_deferred_from(
                    ctx,
                    &parameters,
                    &options.optimizer,
                    None,
                    |t| objective(&indices, t, capture.as_ref().cloned()),
                )?;
                parameters = out.parameters;
                history = out.history;
                success = out.success;
                message = out.message;
                reused = capture
                    .as_ref()
                    .and_then(|c| c.borrow_mut().take())
                    .filter(|p| p.parameters == parameters);
                if let Some(ref p) = reused {
                    final_gradient = p.gradient.iter().fold(0f64, |a, g| a.max(g.abs()));
                } else {
                    let post_capture: Option<CapturedSubspace> = options
                        .reuse_evaluations
                        .then(|| std::rc::Rc::new(std::cell::RefCell::new(None)));
                    final_gradient =
                        objective(&indices, &parameters, post_capture.as_ref().cloned())?
                            .into_gradient()?
                            .iter()
                            .fold(0f64, |a, g| a.max(g.abs()));
                    reused = post_capture
                        .as_ref()
                        .and_then(|c| c.borrow_mut().take())
                        .filter(|p| p.parameters == parameters);
                }
            } else {
                if options.continuation {
                    // The stored end of an earlier growth step of this very problem: growth
                    // continues from it without a relaxation, as the uninterrupted run did.
                    curvature = options.hessian.clone().filter(|_| carries_curvature);
                } else if !indices.is_empty() {
                    // A resumed ansatz is relaxed before it grows.
                    let seed_capture: Option<CapturedSubspace> = options
                        .reuse_evaluations
                        .then(|| std::rc::Rc::new(std::cell::RefCell::new(None)));
                    let mut out = optimizer::minimize_deferred_from(
                        ctx,
                        &parameters,
                        &options.optimizer,
                        options.hessian.as_deref(),
                        |t| objective(&indices, t, seed_capture.as_ref().cloned()),
                    )?;
                    parameters = std::mem::take(&mut out.parameters);
                    history.append(&mut out.history);
                    curvature = out.hessian.take().filter(|_| carries_curvature);
                    reused = seed_capture
                        .as_ref()
                        .and_then(|c| c.borrow_mut().take())
                        .filter(|p| p.parameters == parameters);
                    if !out.success {
                        optimizer_failures.push((0, out.message));
                    }
                }
                for growth in self.growth_steps(&indices)?..=options.max_growth {
                    let mut gradients = ctx.allocate::<f64>(self.pool.len())?;
                    let (step_energies, step_states, step_hpsis) = match reused.take() {
                        Some(point) => (point.energies, point.states, point.hpsis),
                        None => {
                            let mut es = Vec::with_capacity(k);
                            let mut ss = Vec::with_capacity(k);
                            let mut hs = Vec::with_capacity(k);
                            for &ref_row in &ref_rows {
                                let evaluation = self.subspace_evaluate::<A>(
                                    &indices,
                                    &parameters,
                                    true,
                                    ref_row,
                                    options.sparse_kernels,
                                )?;
                                es.push(evaluation.energy);
                                ss.push(evaluation.state);
                                hs.push(None);
                            }
                            (es, ss, hs)
                        }
                    };
                    let mut lowest = f64::INFINITY;
                    for ((&e, state), (hpsi_opt, &w)) in step_energies
                        .iter()
                        .zip(&step_states)
                        .zip(step_hpsis.iter().zip(weights))
                    {
                        lowest = lowest.min(e);
                        let g_vec = match hpsi_opt.as_ref() {
                            Some(hpsi) if options.reuse_evaluations => {
                                self.pool_gradients_with_centered_hpsi(state, hpsi.as_slice())?
                            }
                            _ => self.pool_gradients(state, false)?,
                        };
                        for (slot, g) in gradients.as_mut_slice().iter_mut().zip(g_vec) {
                            *slot += w * g;
                        }
                    }
                    let selected = gradients
                        .as_slice()
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| self.pool_screenable[*i])
                        .max_by(|(i, a), (j, b)| a.abs().total_cmp(&b.abs()).then_with(|| j.cmp(i)))
                        .map(|(i, _)| i);
                    final_gradient = selected
                        .map(|i| gradients.as_slice()[i].abs())
                        .unwrap_or(0.);
                    if gradients.as_slice().iter().any(|g| !g.is_finite()) {
                        return Err(Error::Kernel("nonfinite weighted pool gradient".into()));
                    }
                    let gradient_ok = options
                        .gradient_tolerance
                        .is_none_or(|t| final_gradient < t);
                    let energy_ok = options
                        .energy_tolerance
                        .is_none_or(|t| delta.is_some_and(|change: f64| change.abs() < t));
                    if final_gradient < 1e-12 || (gradient_ok && energy_ok) {
                        screened_final = true;
                        if options.reuse_evaluations {
                            reused = Some(CapturedSubspacePoint {
                                parameters: parameters.clone(),
                                gradient: gradients.as_slice().to_vec(),
                                energies: step_energies,
                                states: step_states,
                                hpsis: step_hpsis,
                            });
                        }
                        success = true;
                        message = "weighted pool gradients converged".into();
                        break;
                    }
                    if growth == options.max_growth {
                        screened_final = true;
                        if options.reuse_evaluations {
                            reused = Some(CapturedSubspacePoint {
                                parameters: parameters.clone(),
                                gradient: gradients.as_slice().to_vec(),
                                energies: step_energies,
                                states: step_states,
                                hpsis: step_hpsis,
                            });
                        }
                        break;
                    }
                    let selected = selected.unwrap();
                    let members: Vec<_> = self.growth_members[selected]
                        .iter()
                        .copied()
                        .filter(|&i| gradients.as_slice()[i].abs() > 1e-12)
                        .collect();
                    let grown = if members.len() >= 2 {
                        members
                    } else {
                        vec![selected]
                    };
                    let accepted = indices.len();
                    for index in grown {
                        indices.push(index);
                        parameters.push(0.);
                    }
                    let start_curvature = curvature
                        .as_deref()
                        .filter(|h| carries_curvature && h.len() == accepted * accepted)
                        .map(|h| super::solver::extend_hessian(h, accepted, parameters.len()));
                    let step_capture: Option<CapturedSubspace> = options
                        .reuse_evaluations
                        .then(|| std::rc::Rc::new(std::cell::RefCell::new(None)));
                    let mut out = optimizer::minimize_deferred_from(
                        ctx,
                        &parameters,
                        &options.optimizer,
                        start_curvature.as_deref(),
                        |t| objective(&indices, t, step_capture.as_ref().cloned()),
                    )?;
                    parameters = std::mem::take(&mut out.parameters);
                    history.append(&mut out.history);
                    curvature = out.hessian.take().filter(|_| carries_curvature);
                    reused = step_capture
                        .as_ref()
                        .and_then(|c| c.borrow_mut().take())
                        .filter(|p| p.parameters == parameters);
                    let mut next_lowest = f64::INFINITY;
                    if let Some(point) = reused.as_ref() {
                        for &e in &point.energies {
                            next_lowest = next_lowest.min(e);
                        }
                    } else {
                        for &ref_row in &ref_rows {
                            next_lowest = next_lowest.min(
                                self.subspace_evaluate::<A>(
                                    &indices,
                                    &parameters,
                                    true,
                                    ref_row,
                                    options.sparse_kernels,
                                )?
                                .energy,
                            );
                        }
                    }
                    delta = Some(next_lowest - lowest);
                    if !out.success {
                        optimizer_failures.push((growth + 1, out.message));
                    }
                }
            }
            let (energies, final_states, final_hpsis) = match reused.take() {
                Some(point) => (point.energies, point.states, point.hpsis),
                None => {
                    let mut es = Vec::with_capacity(k);
                    let mut ss = Vec::with_capacity(k);
                    let mut hs = Vec::with_capacity(k);
                    let trotter = if options.adapt { true } else { options.trotter };
                    for &ref_row in &ref_rows {
                        let result = self.subspace_evaluate::<A>(
                            &indices,
                            &parameters,
                            trotter,
                            ref_row,
                            options.sparse_kernels,
                        )?;
                        es.push(result.energy);
                        ss.push(result.state);
                        hs.push(None);
                    }
                    (es, ss, hs)
                }
            };
            let mut states = stack::<A>(ctx, k, d)?;
            for (j, st) in final_states.iter().enumerate() {
                states.as_mut_slice()[j * d..(j + 1) * d]
                    .copy_from_slice(A::view(st).expect("state has the problem's precision"));
            }
            if options.adapt && (!options.reuse_evaluations || !screened_final) {
                let mut gradients = ctx.allocate::<f64>(self.pool.len())?;
                for ((state, hpsi_opt), &w) in final_states.iter().zip(&final_hpsis).zip(weights) {
                    let g_vec = match hpsi_opt.as_ref() {
                        Some(hpsi) if options.reuse_evaluations => {
                            self.pool_gradients_with_centered_hpsi(state, hpsi.as_slice())?
                        }
                        _ => self.pool_gradients(state, false)?,
                    };
                    for (slot, g) in gradients.as_mut_slice().iter_mut().zip(g_vec) {
                        *slot += w * g;
                    }
                }
                if gradients.as_slice().iter().any(|g| !g.is_finite()) {
                    return Err(Error::Kernel("nonfinite weighted pool gradient".into()));
                }
                final_gradient = gradients
                    .as_slice()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| self.pool_screenable[*i])
                    .fold(0f64, |a, (_, g)| a.max(g.abs()));
            }
            let mut order: Vec<_> = (0..k).collect();
            order.sort_by(|&a, &b| energies[a].total_cmp(&energies[b]));
            let mut sorted = stack::<A>(ctx, k, d)?;
            for (j, &source) in order.iter().enumerate() {
                sorted.as_mut_slice()[j * d..(j + 1) * d]
                    .copy_from_slice(&states.as_slice()[source * d..(source + 1) * d]);
            }
            let sorted_energies = order.iter().map(|&i| energies[i]).collect();
            let indices_len = indices.len();
            Ok(SubspaceResult {
                energies: sorted_energies,
                parameters,
                states: A::wrap(sorted),
                reference,
                indices,
                reference_indices: order,
                history,
                optimizer_failures,
                success,
                message,
                final_gradient,
                elapsed_seconds: start.elapsed().as_secs_f64(),
                hessian: curvature.filter(|h| h.len() == indices_len * indices_len),
                notes: Vec::new(),
                _memory: memory,
            })
        })
    }
}
