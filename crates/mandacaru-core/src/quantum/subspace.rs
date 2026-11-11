//! Weighted shared-unitary VQE/ADAPT; bare expectations, sorted with their states.
use super::amplitude::{stack, Amplitude, State};
use super::solver::by_precision;
use super::{evolution, optimizer, CachedProblem, SolveOptions};
use crate::quantum::C64;
use crate::{product, Error, Reservation, Result};
use std::time::Instant;

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
impl CachedProblem {
    fn subspace_references<A: Amplitude>(&self, count: usize) -> Result<crate::Buffer<A>> {
        let ctx = &self.context;
        let states = self.sector_determinants()?;
        if count == 0 || count > states.len() {
            return Err(Error::Invalid(
                "number of subspace states exceeds the particle sector or is zero".into(),
            ));
        }
        let d = self.register("a subspace calculation")?.len();
        let mut references = stack::<A>(ctx, count, d)?;
        for (j, (_, _, row)) in states.into_iter().take(count).enumerate() {
            references.as_mut_slice()[j * d + row] = A::narrow(C64::new(1., 0.));
        }
        Ok(references)
    }
    fn subspace_evaluate<A: Amplitude>(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
        reference: &[A],
    ) -> Result<evolution::Evaluation> {
        let ops: Vec<_> = indices.iter().map(|&i| &self.pool[i].operator).collect();
        evolution::evaluate_from::<A>(
            &self.context,
            self.register("a subspace calculation")?,
            &self.hamiltonian,
            self.dense.as_ref().map(|x| x.as_slice()),
            self.sparse.as_ref(),
            &ops,
            parameters,
            trotter,
            Some(reference),
        )
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
            if weights.is_empty()
                || weights.iter().any(|w| !w.is_finite() || *w <= 0.)
                || weights.windows(2).any(|w| w[1] >= w[0])
            {
                return Err(Error::Invalid(
                    "subspace weights must be finite, positive and strictly decreasing".into(),
                ));
            }
            let start = Instant::now();
            let ctx = &self.context;
            let k = weights.len();
            let d = self.register("a subspace calculation")?.len();
            let references = self.subspace_references::<A>(k)?;
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
            let objective = |indices: &[usize], theta: &[f64]| -> Result<(f64, Vec<f64>)> {
                let mut energy = 0.;
                let mut gradient = vec![0.; theta.len()];
                for (j, w) in weights.iter().enumerate() {
                    let result = self.subspace_evaluate::<A>(
                        indices,
                        theta,
                        if options.adapt { true } else { options.trotter },
                        &references.as_slice()[j * d..(j + 1) * d],
                    )?;
                    energy += w * result.energy;
                    for (g, x) in gradient.iter_mut().zip(result.gradient) {
                        *g += w * x;
                    }
                }
                Ok((energy, gradient))
            };
            let reference = self
                .subspace_evaluate(&[], &[], true, &references.as_slice()[..d])?
                .energy;
            let mut history = Vec::new();
            let mut optimizer_failures = Vec::new();
            let mut success = false;
            let mut message = "growth budget exhausted".to_owned();
            let mut delta = None;
            let mut final_gradient = 0.;
            // The SLSQP Hessian or BFGS inverse Hessian approximation travels from one growth
            // step to the next, extended by the identity for the appended parameters (as in
            // `solve`).
            let carries_curvature = options.adapt && options.optimizer.method.carries_curvature();
            let mut curvature: Option<Vec<f64>> = None;
            if !options.adapt {
                let out = optimizer::minimize(ctx, &parameters, &options.optimizer, |t| {
                    objective(&indices, t)
                })?;
                parameters = out.parameters;
                history = out.history;
                success = out.success;
                message = out.message;
                final_gradient = objective(&indices, &parameters)?
                    .1
                    .iter()
                    .fold(0f64, |a, g| a.max(g.abs()));
            } else {
                if options.continuation {
                    // The stored end of an earlier growth step of this very problem: growth
                    // continues from it without a relaxation, as the uninterrupted run did.
                    curvature = options.hessian.clone();
                } else if !indices.is_empty() {
                    // A resumed ansatz is relaxed before it grows.
                    let mut out = optimizer::minimize_from(
                        ctx,
                        &parameters,
                        &options.optimizer,
                        options.hessian.as_deref(),
                        |t| objective(&indices, t),
                    )?;
                    parameters = std::mem::take(&mut out.parameters);
                    history.append(&mut out.history);
                    curvature = out.hessian.take().filter(|_| carries_curvature);
                    if !out.success {
                        optimizer_failures.push((0, out.message));
                    }
                }
                for growth in self.growth_steps(&indices)?..=options.max_growth {
                    let mut gradients = ctx.allocate::<f64>(self.pool.len())?;
                    let mut lowest = f64::INFINITY;
                    for (j, w) in weights.iter().enumerate() {
                        let evaluation = self.subspace_evaluate::<A>(
                            &indices,
                            &parameters,
                            true,
                            &references.as_slice()[j * d..(j + 1) * d],
                        )?;
                        lowest = lowest.min(evaluation.energy);
                        for (slot, g) in gradients
                            .as_mut_slice()
                            .iter_mut()
                            .zip(self.pool_gradients(&evaluation.state, false)?)
                        {
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
                        success = true;
                        message = "weighted pool gradients converged".into();
                        break;
                    }
                    if growth == options.max_growth {
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
                    let mut out = optimizer::minimize_from(
                        ctx,
                        &parameters,
                        &options.optimizer,
                        start_curvature.as_deref(),
                        |t| objective(&indices, t),
                    )?;
                    parameters = std::mem::take(&mut out.parameters);
                    history.append(&mut out.history);
                    curvature = out.hessian.take().filter(|_| carries_curvature);
                    let mut next_lowest = f64::INFINITY;
                    for j in 0..k {
                        next_lowest = next_lowest.min(
                            self.subspace_evaluate::<A>(
                                &indices,
                                &parameters,
                                true,
                                &references.as_slice()[j * d..(j + 1) * d],
                            )?
                            .energy,
                        );
                    }
                    delta = Some(next_lowest - lowest);
                    if !out.success {
                        optimizer_failures.push((growth + 1, out.message));
                    }
                }
            }
            let mut energies = Vec::new();
            let mut states = stack::<A>(ctx, k, d)?;
            for j in 0..k {
                let result = self.subspace_evaluate::<A>(
                    &indices,
                    &parameters,
                    if options.adapt { true } else { options.trotter },
                    &references.as_slice()[j * d..(j + 1) * d],
                )?;
                energies.push(result.energy);
                states.as_mut_slice()[j * d..(j + 1) * d].copy_from_slice(
                    A::view(&result.state).expect("state has the problem's precision"),
                );
            }
            if options.adapt {
                let mut gradients = ctx.allocate::<f64>(self.pool.len())?;
                for (j, w) in weights.iter().enumerate() {
                    let state = A::wrap(copy_row(ctx, &states.as_slice()[j * d..(j + 1) * d])?);
                    for (slot, g) in gradients
                        .as_mut_slice()
                        .iter_mut()
                        .zip(self.pool_gradients(&state, false)?)
                    {
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

/// A stored row copied into its own buffer (a state to screen the pool with).
fn copy_row<A: Amplitude>(ctx: &crate::Context, row: &[A]) -> Result<crate::Buffer<A>> {
    let mut out = ctx.allocate::<A>(row.len())?;
    out.as_mut_slice().copy_from_slice(row);
    Ok(out)
}
