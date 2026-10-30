//! Ordered Pauli product formulas and finite-register phase estimation.
use super::{
    algebra::Term,
    amplitude::{finite, inner, stack, Amplitude, State},
    Basis, Operator, C64,
};
use crate::{product, Buffer, Context, Error, Reservation, Result};
use rustfft::FftPlanner;

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
pub fn check_state<A: Amplitude>(state: &[A], dimension: usize, normalized: bool) -> Result<()> {
    if state.len() != dimension || !finite(state) {
        return Err(invalid(
            "state must have finite amplitudes and match the basis",
        ));
    }
    if normalized && (inner(state, state).re - 1.).abs() > A::INPUT_NORM_TOLERANCE {
        return Err(invalid("initial state must be normalized"));
    }
    Ok(())
}
pub struct Propagator {
    basis: Basis,
    h: Operator,
    rotations: Vec<(Term, f64)>,
    constant: f64,
    _memory: Reservation,
}
impl Propagator {
    /// The same ordered Pauli rotations used by propagation, for SDK export.
    pub fn schedule(&self) -> (f64, Vec<(String, f64)>) {
        let n = self.h.n;
        let rotations = self
            .rotations
            .iter()
            .map(|(term, weight)| {
                let label = (0..n)
                    .map(|j| {
                        let bit = 1u64 << (n - 1 - j);
                        match (term.x & bit != 0, term.z & bit != 0) {
                            (false, false) => 'I',
                            (true, false) => 'X',
                            (false, true) => 'Z',
                            (true, true) => 'Y',
                        }
                    })
                    .collect();
                (label, *weight)
            })
            .collect();
        (self.constant, rotations)
    }

    pub fn new(ctx: &Context, n: usize, terms: &[(String, C64)], order: usize) -> Result<Self> {
        if order != 1 && (order == 0 || order % 2 != 0) {
            return Err(invalid("order must be 1 or a positive even integer"));
        }
        // Refuse exponential schedule growth before allocating or recursing.
        let stage_count = if order <= 2 {
            1
        } else {
            5usize
                .checked_pow(u32::try_from(order / 2 - 1).map_err(|_| Error::Overflow)?)
                .ok_or(Error::Overflow)?
        };
        let count = product(&[terms.len(), stage_count, 2])?;
        if count > 1_000_000 {
            return Err(invalid(
                "Suzuki rotation schedule exceeds one million entries",
            ));
        }
        let memory = ctx.reserve(product(&[count, std::mem::size_of::<(Term, f64)>()])?)?;
        let h = Operator::new(ctx, n, terms)?;
        if !h.is_hermitian() || terms.iter().any(|(_, c)| c.im.abs() > 1e-12) {
            return Err(invalid("propagation requires a Hermitian Hamiltonian"));
        }
        let mut active = Vec::new();
        let mut constant = 0.;
        for (label, coefficient) in terms {
            let op = Operator::new(ctx, n, &[(label.clone(), *coefficient)])?;
            for t in op.terms {
                if t.x == 0 && t.z == 0 {
                    constant += t.coefficient.re;
                } else {
                    active.push(t);
                }
            }
        }
        let mut rotations = Vec::new();
        fn stages(order: usize, scale: f64, active: &[Term], out: &mut Vec<(Term, f64)>) {
            if order > 2 {
                let p = 1. / (4. - 4f64.powf(1. / (order - 1) as f64));
                for w in [p, p, 1. - 4. * p, p, p] {
                    stages(order - 2, scale * w, active, out);
                }
            } else {
                for t in active.iter().chain(active.iter().rev()) {
                    let weight = t.coefficient.re * scale / 2.;
                    if let Some((last, w)) = out.last_mut() {
                        if last.x == t.x && last.z == t.z {
                            *w += weight;
                            continue;
                        }
                    }
                    out.push((
                        Term {
                            x: t.x,
                            z: t.z,
                            coefficient: C64::new(1., 0.),
                        },
                        weight,
                    ));
                }
            }
        }
        if order == 1 || active.len() <= 1 {
            for t in active {
                let weight = t.coefficient.re;
                rotations.push((
                    Term {
                        coefficient: C64::new(1., 0.),
                        ..t
                    },
                    weight,
                ));
            }
        } else {
            stages(order, 1., &active, &mut rotations);
        }
        Ok(Self {
            basis: Basis::register(ctx, n)?,
            h,
            rotations,
            constant,
            _memory: memory,
        })
    }
    /// Ordered Pauli rotations on a stored state of either precision; each
    /// rotation is evaluated in float64 and rounded once into the stored state.
    pub fn evolve<A: Amplitude>(
        &self,
        ctx: &Context,
        state: &[A],
        time: f64,
        steps: usize,
        inverse: bool,
        normalized: bool,
    ) -> Result<Buffer<A>> {
        ctx.run(|| {
            check_state(state, self.basis.len(), normalized)?;
            if !time.is_finite() || steps == 0 {
                return Err(invalid("finite time and positive steps required"));
            }
            let mut out = ctx.allocate::<A>(state.len())?;
            out.as_mut_slice().copy_from_slice(state);
            let signed_time = if inverse { -time } else { time };
            let dt = signed_time / steps as f64;
            if self.rotations.iter().any(|(_, w)| !(dt * w).is_finite())
                || !(signed_time * self.constant).is_finite()
            {
                return Err(invalid("propagation angle overflow"));
            }
            if time != 0. {
                let mut scratch = ctx.allocate::<A>(state.len())?;
                for _ in 0..steps {
                    for k in 0..self.rotations.len() {
                        let (term, w) = &self.rotations[if inverse {
                            self.rotations.len() - 1 - k
                        } else {
                            k
                        }];
                        let (sine, cosine) = (dt * w).sin_cos();
                        for ket in 0..state.len() {
                            let target = ket ^ term.x as usize;
                            scratch.as_mut_slice()[target] = A::narrow(
                                cosine * out.as_slice()[target].widen()
                                    - C64::new(0., sine)
                                        * term.amplitude(ket as u64)
                                        * out.as_slice()[ket].widen(),
                            );
                        }
                        std::mem::swap(&mut out, &mut scratch);
                    }
                }
                let phase = C64::from_polar(1., -signed_time * self.constant);
                for c in out.as_mut_slice() {
                    *c = A::narrow(c.widen() * phase);
                }
            }
            if !finite(out.as_slice()) {
                return Err(Error::Kernel(
                    "propagation produced nonfinite amplitudes".into(),
                ));
            }
            Ok(out)
        })
    }
    pub fn apply<A: Amplitude>(&self, ctx: &Context, state: &[A]) -> Result<Buffer<A>> {
        ctx.run(|| {
            if self.h.n != self.basis.n || state.len() != self.basis.len() || !finite(state) {
                return Err(invalid("state must be finite and match the operator basis"));
            }
            let mut out = ctx.allocate::<A>(state.len())?;
            let mut acc = ctx.allocate::<C64>(A::accumulator_len(state.len()))?;
            A::with_wide(out.as_mut_slice(), acc.as_mut_slice(), |wide| {
                self.h.add_into(&self.basis, state, C64::new(1., 0.), wide)
            });
            Ok(out)
        })
    }
}

pub struct QpeResult {
    pub probabilities: Buffer<f64>,
    pub amplitudes: Option<State>,
    pub exact_energies: Option<Vec<f64>>,
    pub overlaps: Option<Vec<f64>>,
    _memory: Reservation,
}
/// Exact finite-register simulation: powers of exp(2πi(H-lo)/width), then DFT.
pub fn qpe<A: Amplitude>(
    ctx: &Context,
    h: &Operator,
    state: &[A],
    evaluation_qubits: usize,
    window: [f64; 2],
    keep: bool,
) -> Result<QpeResult> {
    ctx.run(|| {
        if !h.is_hermitian()
            || !window.iter().all(|x| x.is_finite())
            || window[1] <= window[0]
            || !(window[1] - window[0]).is_finite()
        {
            return Err(invalid(
                "QPE needs a Hermitian Hamiltonian and a finite increasing energy window",
            ));
        }
        if evaluation_qubits == 0 {
            return Err(invalid("evaluation qubits must be positive"));
        }
        let bins = 1usize
            .checked_shl(u32::try_from(evaluation_qubits).map_err(|_| Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        let basis = Basis::register(ctx, h.n)?;
        let d = basis.len();
        check_state(state, d, true)?;
        let memory = ctx.reserve(if h.n <= 12 { product(&[d, 16])? } else { 0 })?;
        let mut columns = stack::<A>(ctx, bins, d)?;
        let mut exact_energies = None;
        let mut overlaps = None;
        let width = window[1] - window[0];
        if h.n <= 12 {
            let _workspace = ctx.reserve(product(&[d, d, 16, 10])?)?;
            let dense = h.dense(ctx, &basis)?;
            let (energies, vectors) = crate::math::linalg::eigen(
                nalgebra::DMatrix::from_row_slice(d, d, dense.as_slice()),
            )?;
            let c: Vec<C64> = (0..d)
                .map(|j| {
                    (0..d)
                        .map(|i| vectors[(i, j)].conj() * state[i].widen())
                        .sum()
                })
                .collect();
            let mut row = ctx.allocate::<C64>(d)?;
            for k in 0..bins {
                row.as_mut_slice().fill(C64::default());
                for j in 0..d {
                    let phase = C64::from_polar(
                        1.,
                        std::f64::consts::TAU * k as f64 * (energies[j] - window[0]) / width,
                    ) * c[j];
                    for i in 0..d {
                        row.as_mut_slice()[i] += vectors[(i, j)] * phase;
                    }
                }
                for (to, &from) in columns.as_mut_slice()[k * d..(k + 1) * d]
                    .iter_mut()
                    .zip(row.as_slice())
                {
                    *to = A::narrow(from);
                }
            }
            overlaps = Some(c.iter().map(|x| x.norm_sqr()).collect());
            exact_energies = Some(energies);
        } else {
            let mut terms: Vec<Term> = h
                .terms
                .iter()
                .map(|t| Term {
                    x: t.x,
                    z: t.z,
                    coefficient: t.coefficient * C64::new(0., 1.),
                })
                .collect();
            terms.push(Term {
                x: 0,
                z: 0,
                coefficient: C64::new(0., -window[0]),
            });
            let generator = Operator::from_terms(ctx, h.n, terms)?;
            columns.as_mut_slice()[..d].copy_from_slice(state);
            let mut acc = ctx.allocate::<C64>(A::accumulator_len(d))?;
            for k in 1..bins {
                let (previous, next) = columns.as_mut_slice().split_at_mut(k * d);
                next[..d].copy_from_slice(&previous[(k - 1) * d..k * d]);
                super::evolution::exponential(
                    ctx,
                    &basis,
                    &[&generator],
                    &[std::f64::consts::TAU / width],
                    &mut next[..d],
                    1,
                    &[],
                    acc.as_mut_slice(),
                )?;
            }
        }
        // Cover planner tables and its temporary construction workspace as well.
        let _fft_plan = ctx.reserve(product(&[bins, 128])?)?;
        let fft = FftPlanner::new().plan_fft_forward(bins);
        let mut line = ctx.allocate::<C64>(bins)?;
        let mut scratch = ctx.allocate::<C64>(fft.get_inplace_scratch_len())?;
        for i in 0..d {
            for k in 0..bins {
                line.as_mut_slice()[k] = columns.as_slice()[k * d + i].widen();
            }
            fft.process_with_scratch(line.as_mut_slice(), scratch.as_mut_slice());
            for k in 0..bins {
                columns.as_mut_slice()[k * d + i] = A::narrow(line.as_slice()[k] / bins as f64);
            }
        }
        let mut probabilities = ctx.allocate::<f64>(bins)?;
        for (k, p) in probabilities.as_mut_slice().iter_mut().enumerate() {
            *p = inner(
                &columns.as_slice()[k * d..(k + 1) * d],
                &columns.as_slice()[k * d..(k + 1) * d],
            )
            .re;
        }
        if probabilities.as_slice().iter().any(|p| !p.is_finite())
            || (probabilities.as_slice().iter().sum::<f64>() - 1.).abs()
                > A::norm_tolerance(bins).max(1e-9)
        {
            return Err(Error::Kernel("QPE lost probability normalization".into()));
        }
        Ok(QpeResult {
            probabilities,
            amplitudes: if keep { Some(A::wrap(columns)) } else { None },
            exact_energies,
            overlaps,
            _memory: memory,
        })
    })
}
