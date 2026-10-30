//! Sequential variational deflation with analytic overlap-penalty gradients.
use super::amplitude::{inner_wide, stack, Amplitude, State};
use super::solver::by_precision;
use super::{evolution, optimizer, CachedProblem, SolveOptions, C64};
use crate::{Buffer, Error, Reservation, Result};
use std::time::Instant;
pub struct Levels {
    pub energies: Vec<f64>,
    pub states: State,
    pub parameters: Vec<Vec<f64>>,
    pub success: Vec<bool>,
    pub messages: Vec<String>,
    pub evaluations: usize,
    pub reference: f64,
    pub elapsed_seconds: f64,
    _memory: Reservation,
}
impl CachedProblem {
    pub fn energy_levels(
        &self,
        options: &SolveOptions,
        count: usize,
        restarts: usize,
        beta: Option<f64>,
        seed: u64,
    ) -> Result<Levels> {
        by_precision!(
            self.precision,
            A => self.energy_levels_as::<A>(options, count, restarts, beta, seed)
        )
    }
    fn energy_levels_as<A: Amplitude>(
        &self,
        options: &SolveOptions,
        count: usize,
        restarts: usize,
        beta: Option<f64>,
        seed: u64,
    ) -> Result<Levels> {
        options.validate(self.precision)?;
        if options.adapt {
            return Err(Error::Invalid(
                "sequential deflation currently uses a fixed ansatz".into(),
            ));
        }
        if count == 0 || count > self.basis.len() || restarts == 0 {
            return Err(Error::Invalid(
                "deflation needs a positive state/restart count fitting the simulation basis"
                    .into(),
            ));
        }
        let beta = beta.unwrap_or_else(|| 2. * self.hamiltonian.norm_bound());
        if !beta.is_finite() || beta <= 0. {
            return Err(Error::Invalid(
                "deflation penalty must be positive and finite".into(),
            ));
        }
        self.context.run(|| {
            let start = Instant::now();
            let ctx = &self.context;
            let d = self.basis.len();
            let indices: Vec<_> = (0..self.pool.len())
                .filter(|&i| self.pool_screenable[i])
                .collect();
            let n = indices.len();
            let memory = ctx.reserve(
                crate::product(&[count, n.max(1), 16])?
                    .checked_add(crate::product(&[count, 128])?)
                    .ok_or(Error::Overflow)?,
            )?;
            let mut states = stack::<A>(ctx, count, d)?;
            let mut energies = Vec::new();
            let mut parameters = Vec::new();
            let mut success = Vec::new();
            let mut messages = Vec::new();
            let mut evaluations = 0;
            let reference = self.evaluate(&[], &[], true)?.energy;
            let mut random = seed;
            for level in 0..count {
                let mut best: Option<(optimizer::Outcome, Buffer<A>)> = None;
                for restart in 0..restarts {
                    let initial: Vec<_> = (0..n)
                        .map(|_| {
                            if level == 0 && restart == 0 {
                                0.
                            } else {
                                random = random.wrapping_add(0x9e3779b97f4a7c15);
                                let mut z = random;
                                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                                z ^= z >> 31;
                                2. * std::f64::consts::PI
                                    * ((z >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
                            }
                        })
                        .collect();
                    let out = optimizer::minimize(ctx, &initial, &options.optimizer, |theta| {
                        let sequence = self.sequence(&indices, theta)?;
                        let data = evolution::state_from::<A>(
                            ctx,
                            &self.basis,
                            &sequence.operators,
                            &sequence.parameters,
                            options.trotter || self.hva.is_some(),
                            true,
                            None,
                        )?;
                        let psi = &data.as_slice()[..d];
                        let mut action = ctx.allocate::<C64>(d)?;
                        self.hamiltonian
                            .apply_into(&self.basis, psi, action.as_mut_slice());
                        for previous in states.as_slice()[..level * d].chunks_exact(d) {
                            let overlap = evolution::inner(previous, psi);
                            for (h, s) in action.as_mut_slice().iter_mut().zip(previous) {
                                *h += beta * overlap * s.widen();
                            }
                        }
                        let energy = inner_wide(psi, action.as_slice()).re;
                        let mut gradient = vec![0.; n];
                        for (i, (logical, weight)) in sequence.links.iter().enumerate() {
                            gradient[*logical] += weight
                                * 2.
                                * inner_wide(
                                    &data.as_slice()[(i + 1) * d..(i + 2) * d],
                                    action.as_slice(),
                                )
                                .re
                        }
                        Ok((energy, gradient))
                    })?;
                    evaluations += out.history.len();
                    if best.as_ref().is_none_or(|(b, _)| out.energy < b.energy) {
                        let state =
                            self.prepare_buffer::<A>(&indices, &out.parameters, options.trotter)?;
                        best = Some((out, state));
                    }
                }
                let (out, state) = best.unwrap();
                let mut action = ctx.allocate::<C64>(d)?;
                self.hamiltonian
                    .apply_into(&self.basis, state.as_slice(), action.as_mut_slice());
                energies.push(inner_wide(state.as_slice(), action.as_slice()).re);
                parameters.push(out.parameters);
                success.push(out.success);
                messages.push(out.message);
                states.as_mut_slice()[level * d..(level + 1) * d].copy_from_slice(state.as_slice());
            }
            let mut order: Vec<_> = (0..count).collect();
            order.sort_by(|&i, &j| energies[i].total_cmp(&energies[j]));
            let mut sorted = stack::<A>(ctx, count, d)?;
            for (i, &j) in order.iter().enumerate() {
                sorted.as_mut_slice()[i * d..(i + 1) * d]
                    .copy_from_slice(&states.as_slice()[j * d..(j + 1) * d]);
            }
            Ok(Levels {
                energies: order.iter().map(|&j| energies[j]).collect(),
                parameters: order.iter().map(|&j| parameters[j].clone()).collect(),
                success: order.iter().map(|&j| success[j]).collect(),
                messages: order.iter().map(|&j| messages[j].clone()).collect(),
                states: A::wrap(sorted),
                evaluations,
                reference,
                elapsed_seconds: start.elapsed().as_secs_f64(),
                _memory: memory,
            })
        })
    }
}
