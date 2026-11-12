//! Matrix-free exponential actions and analytic derivatives. Product ansatz
//! energies use an adjoint sweep; the summed UCC exponential uses a
//! differentiated Taylor recurrence (Frechet action).
//!
//! Every kernel is generic over the stored amplitude (`Amplitude`): complex128
//! or complex64 rows. Many-term sums (a generator applied to a row, the Taylor
//! recurrence of one row) accumulate in a float64 row and are rounded once into
//! the stored row; every reduction is float64.
use super::amplitude::{finite, inner_wide, norm_squared, stack, Amplitude, State};
use super::{Basis, Operator, C64};
use crate::{Buffer, Context, Error, Result};

pub use super::amplitude::inner;

fn norm<A: Amplitude>(a: &[A]) -> f64 {
    norm_squared(a).sqrt()
}
/// `out = narrow(G input)`: the many-term sum accumulates in `acc` (float64).
fn apply_stored<A: Amplitude>(
    generator: &Operator,
    basis: &Basis,
    input: &[A],
    out: &mut [A],
    acc: &mut [C64],
) {
    A::with_wide(out, acc, |wide| {
        generator.add_into(basis, input, C64::new(1., 0.), wide)
    });
}

/// The closed-form rotation of a one-generator exponential, `exp(theta G) = 1 + linear G +
/// quadratic G^2`, as `(linear, quadratic)`, for a generator whose strings have the eigenvalue
/// spacing `frequency`.
fn rotation(frequency: f64, theta: f64) -> Result<(f64, f64)> {
    let angle = theta * frequency;
    if !angle.is_finite() {
        return Err(Error::Invalid("excitation rotation angle overflow".into()));
    }
    Ok((
        angle.sin() / frequency,
        2. * (0.5 * angle).sin().powi(2) / frequency.powi(2),
    ))
}
/// The flip mask and rotation of a generator the closed form rotates in one pass over row pairs
/// (an excitation: one flip mask and several strings), or `None`.
fn pair_rotation(generator: &Operator, theta: f64) -> Result<Option<(u64, f64, f64)>> {
    let (Some(frequency), Some(x)) = (generator.rotation_frequency, generator.single_mask()) else {
        return Ok(None);
    };
    if generator.terms.len() <= 1 {
        return Ok(None);
    }
    let (linear, quadratic) = rotation(frequency, theta)?;
    Ok(Some((x, linear, quadratic)))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn exponential<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    data: &mut [A],
    rows: usize,
    directions: &[Option<usize>],
    acc: &mut [C64],
) -> Result<()> {
    let d = basis.len();
    if let [generator] = ops {
        if generator.terms.len() > 1 {
            if let Some(frequency) = generator.rotation_frequency {
                let (linear, quadratic) = rotation(frequency, theta[0])?;
                let mut first = ctx.allocate::<A>(d)?;
                if let Some(x) = generator.single_mask() {
                    // An excitation: one pass over the pairs it connects.
                    for row in data.chunks_exact_mut(d).take(rows) {
                        generator.rotate_pairs(basis, x, linear, quadratic, row, first.as_mut_slice());
                    }
                } else {
                let mut own = ctx.allocate::<C64>(A::own_wide_len(d))?;
                for row in data.chunks_exact_mut(d).take(rows) {
                    apply_stored(generator, basis, row, first.as_mut_slice(), acc);
                    // G(G row) stays a float64 row: it is read, never stored.
                    let second = A::wide_row(own.as_mut_slice(), acc);
                    generator.apply_into(basis, first.as_slice(), second);
                    for ((value, &a), &b) in row.iter_mut().zip(first.as_slice()).zip(second.iter())
                    {
                        *value = A::narrow(value.widen() + (linear * a.widen() + quadratic * b));
                    }
                }
                }
                if directions.iter().any(Option::is_some) {
                    apply_stored(generator, basis, &data[..d], first.as_mut_slice(), acc);
                    for (j, direction) in directions.iter().enumerate() {
                        if direction == &Some(0) {
                            for (value, &derivative) in data[(j + 1) * d..(j + 2) * d]
                                .iter_mut()
                                .zip(first.as_slice())
                            {
                                *value = A::narrow(value.widen() + derivative.widen());
                            }
                        }
                    }
                }
                return Ok(());
            }
        }
    }
    if ops.len() == 1
        && ops[0].terms.len() == 1
        && ops[0].terms[0].coefficient.re == 0.
        && ops[0].terms[0].coefficient.im != 0.
    {
        let coefficient = ops[0].terms[0].coefficient.im;
        let angle = theta[0] * coefficient;
        if !angle.is_finite() {
            return Err(Error::Invalid("Pauli rotation angle overflow".into()));
        }
        let mut action = ctx.allocate::<A>(d)?;
        let (sin, cos) = angle.sin_cos();
        for row in data.chunks_exact_mut(d).take(rows) {
            apply_stored(ops[0], basis, row, action.as_mut_slice(), acc);
            for (out, &applied) in row.iter_mut().zip(action.as_slice()) {
                *out = A::narrow(out.widen() * cos + applied.widen() * (sin / coefficient));
            }
        }
        if directions.iter().any(Option::is_some) {
            apply_stored(ops[0], basis, &data[..d], action.as_mut_slice(), acc);
            for (j, direction) in directions.iter().enumerate() {
                if direction == &Some(0) {
                    for (out, &applied) in data[(j + 1) * d..(j + 2) * d]
                        .iter_mut()
                        .zip(action.as_slice())
                    {
                        *out = A::narrow(out.widen() + applied.widen());
                    }
                }
            }
        }
        return Ok(());
    }
    let bound: f64 = ops
        .iter()
        .zip(theta)
        .map(|(g, t)| g.norm_bound() * t.abs())
        .sum();
    if !bound.is_finite() || bound > 2048. {
        return Err(Error::Invalid(
            "exponential action exceeds the scaling budget".into(),
        ));
    }
    let steps = (bound / 0.5).ceil().max(1.) as usize;
    let scale = 1. / steps as f64;
    let mut term = stack::<A>(ctx, rows, d)?;
    let mut next = stack::<A>(ctx, rows, d)?;
    for _ in 0..steps {
        term.as_mut_slice().copy_from_slice(data);
        let mut converged = false;
        for order in 1..=80 {
            for row in 0..rows {
                let source = &term.as_slice()[row * d..(row + 1) * d];
                A::with_wide(
                    &mut next.as_mut_slice()[row * d..(row + 1) * d],
                    acc,
                    |wide| {
                        for (g, t) in ops.iter().zip(theta) {
                            g.add_into(basis, source, C64::new(*t, 0.), wide);
                        }
                        if row > 0 {
                            if let Some(g) = directions[row - 1] {
                                ops[g].add_into(
                                    basis,
                                    &term.as_slice()[..d],
                                    C64::new(1., 0.),
                                    wide,
                                );
                            }
                        }
                    },
                );
            }
            let factor = scale / order as f64;
            for (v, delta) in data.iter_mut().zip(next.as_mut_slice()) {
                *delta = A::narrow(delta.widen() * factor);
                *v = A::narrow(v.widen() + delta.widen());
            }
            if norm(next.as_slice()) < A::TAYLOR_STOP * norm(data).max(1.) {
                converged = true;
                break;
            }
            std::mem::swap(&mut term, &mut next);
        }
        if !converged || !finite(data) {
            return Err(Error::Kernel(
                "matrix-free exponential failed to converge".into(),
            ));
        }
    }
    Ok(())
}

/// The ordered product `exp(theta_k G_k) ... exp(theta_0 G_0)` applied to `data` (`rows` rows: the
/// state and any derivative rows) one generator at a time. `offset` is the position of `ops[0]` in
/// the full ansatz (the derivative row `j` follows generator `j`). A large one-row product ansatz
/// rotates by gathering into a second row and swapping the two, one parallel region per operator
/// and no copy back; the same bits as the in-place pair update the other path runs. `gathered`
/// keeps that second row between calls.
#[allow(clippy::too_many_arguments)]
fn rotate_product<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    offset: usize,
    rows: usize,
    data: &mut Buffer<A>,
    acc: &mut [C64],
    gathered: &mut Option<Buffer<A>>,
) -> Result<()> {
    let d = basis.len();
    for (k, (g, t)) in ops.iter().zip(theta).enumerate() {
        if rows == 1 && g.lockstep(basis) {
            if let Some((x, linear, quadratic)) = pair_rotation(g, *t)? {
                let other = match gathered.as_mut() {
                    Some(other) => other,
                    None => gathered.insert(ctx.allocate::<A>(d)?),
                };
                g.rotate_gather(basis, x, linear, quadratic, data.as_slice(), other.as_mut_slice());
                std::mem::swap(data, other);
                continue;
            }
        }
        let directions: Vec<_> = (0..rows - 1)
            .map(|j| if j == offset + k { Some(0) } else { None })
            .collect();
        exponential(ctx, basis, &[*g], &[*t], data.as_mut_slice(), rows, &directions, acc)?;
    }
    Ok(())
}

pub struct Evaluation {
    pub energy: f64,
    pub gradient: Vec<f64>,
    pub state: State,
}
pub fn state<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
    derivatives: bool,
) -> Result<Buffer<A>> {
    state_from(ctx, basis, ops, theta, trotter, derivatives, None)
}
#[allow(clippy::too_many_arguments)]
pub fn state_from<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
    derivatives: bool,
    initial: Option<&[A]>,
) -> Result<Buffer<A>> {
    if ops.len() != theta.len()
        || theta.iter().any(|t| !t.is_finite())
        || ops.iter().any(|g| g.n != basis.n || !g.is_antihermitian())
    {
        return Err(Error::Invalid(
            "finite parameters must match anti-Hermitian generators".into(),
        ));
    }
    let rows = if derivatives {
        theta.len().checked_add(1).ok_or(Error::Overflow)?
    } else {
        1
    };
    let d = basis.len();
    let mut data = stack::<A>(ctx, rows, d)?;
    let mut acc = ctx.allocate::<C64>(A::accumulator_len(d))?;
    if let Some(initial) = initial {
        super::dynamics::check_state(initial, d, true)?;
        data.as_mut_slice()[..d].copy_from_slice(initial);
    } else {
        data.as_mut_slice()[basis.reference] = A::narrow(C64::new(1., 0.));
    }
    if trotter {
        let mut gathered: Option<Buffer<A>> = None;
        rotate_product(ctx, basis, ops, theta, 0, rows, &mut data, acc.as_mut_slice(), &mut gathered)?;
    } else {
        let directions: Vec<_> = (0..rows - 1).map(Some).collect();
        exponential(
            ctx,
            basis,
            ops,
            theta,
            data.as_mut_slice(),
            rows,
            &directions,
            acc.as_mut_slice(),
        )?;
    }
    let state_norm = norm(&data.as_slice()[..d]);
    if (state_norm - 1.).abs() > A::norm_tolerance(ops.len()) {
        return Err(Error::Kernel(
            "exponential action lost state normalization".into(),
        ));
    }
    Ok(data)
}
#[allow(clippy::too_many_arguments)]
pub fn evaluate<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    h: &Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
) -> Result<Evaluation> {
    evaluate_from::<A>(ctx, basis, h, dense, sparse, ops, theta, trotter, None)
}
#[allow(clippy::too_many_arguments)]
pub fn evaluate_from<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    h: &Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
    initial: Option<&[A]>,
) -> Result<Evaluation> {
    forward_from::<A>(ctx, basis, h, dense, sparse, ops, theta, trotter, initial)?
        .finish(ctx, basis, ops, theta, trotter)
}

/// `(H - identity) psi` into `wide` (float64) and the energy `<psi|(H - identity)|psi> / <psi|psi>`
/// of the state, as `(norm squared, energy)`. `identity` is the Hamiltonian's identity term, which
/// the Hermiticity check adds back.
fn centered_expectation<A: Amplitude>(
    basis: &Basis,
    h: &Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    identity: C64,
    psi: &[A],
    wide: &mut [C64],
) -> Result<(f64, C64)> {
    let d = basis.len();
    let normalization = inner(psi, psi).re;
    if let Some(matrix) = dense {
        for (row, out) in wide.iter_mut().enumerate() {
            *out = matrix[row * d..(row + 1) * d]
                .iter()
                .zip(psi)
                .enumerate()
                .filter(|(column, _)| *column != row)
                .map(|(_, (a, b))| a * b.widen())
                .sum::<C64>()
                + h.centered_diagonal(basis.indices.as_slice()[row]) * psi[row].widen();
        }
    } else if let Some(sparse) = sparse {
        sparse.apply_centered_into(psi, wide);
    } else {
        h.apply_centered_into(basis, psi, wide);
    }
    let energy = inner_wide(psi, wide) / normalization;
    if !energy.re.is_finite() || (energy.im + identity.im).abs() > A::HERMITIAN_TOLERANCE {
        return Err(Error::Kernel("invalid Hermitian expectation value".into()));
    }
    Ok((normalization, energy))
}

/// The forward half of an evaluation: the prepared state (and its derivative rows when the
/// ansatz is not a product), `(H - identity) psi` and the energy. The gradient is the other half
/// ([`Forward::finish`]): an optimizer's rejected trial point needs only [`Forward::energy`], so
/// the half it does not use is never spent. Both halves are one code path, so an energy and a
/// gradient taken from one `Forward` are the bits [`evaluate`] returns.
pub struct Forward<A: Amplitude> {
    data: Buffer<A>,
    wide: Buffer<C64>,
    normalization: f64,
    energy: C64,
    identity: C64,
}
impl<A: Amplitude> Forward<A> {
    /// The energy, with the Hamiltonian's identity term.
    pub fn energy(&self) -> f64 {
        self.energy.re + self.identity.re
    }
    /// The energy, the gradient and the state. `ops`, `theta` and `trotter` are the ones this
    /// forward half was prepared with.
    pub fn finish(
        self,
        ctx: &Context,
        basis: &Basis,
        ops: &[&Operator],
        theta: &[f64],
        trotter: bool,
    ) -> Result<Evaluation> {
        let Self {
            data,
            mut wide,
            normalization,
            energy,
            identity,
        } = self;
        let d = basis.len();
        let rows = data.as_slice().len() / d;
        let psi = &data.as_slice()[..d];
        let gradient: Vec<f64> = if trotter {
            adjoint_sweep::<A>(ctx, basis, ops, theta, 0, psi, &mut wide, energy.re, normalization)?
        } else {
            // The residual (H - E) psi replaces H psi in place.
            for (out, &p) in wide.as_mut_slice().iter_mut().zip(psi) {
                *out -= energy.re * p.widen();
            }
            (0..theta.len())
                .map(|k| {
                    2. * inner_wide(&data.as_slice()[(k + 1) * d..(k + 2) * d], wide.as_slice()).re
                        / normalization
                })
                .collect()
        };
        if gradient.iter().any(|g| !g.is_finite()) {
            return Err(Error::Kernel("nonfinite energy derivative".into()));
        }
        drop(wide);
        let state = if rows == 1 {
            data
        } else {
            let mut state = ctx.allocate::<A>(d)?;
            state.as_mut_slice().copy_from_slice(&data.as_slice()[..d]);
            state
        };
        Ok(Evaluation {
            energy: energy.re + identity.re,
            gradient,
            state: A::wrap(state),
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn forward_from<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    h: &Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
    initial: Option<&[A]>,
) -> Result<Forward<A>> {
    if h.n != basis.n
        || dense.is_some_and(|matrix| basis.len().checked_mul(basis.len()) != Some(matrix.len()))
    {
        return Err(Error::Invalid(
            "Hamiltonian representation must match the basis".into(),
        ));
    }
    // A non-product ansatz keeps its derivative rows in the stack its Taylor series stops on, so
    // they are part of the energy's bits and are never dropped here.
    let data = state_from(ctx, basis, ops, theta, trotter, !trotter, initial)?;
    let d = basis.len();
    let identity = h.identity();
    // (H - identity) psi, accumulated in float64.
    let mut wide = ctx.allocate::<C64>(d)?;
    let (normalization, energy) = centered_expectation(
        basis,
        h,
        dense,
        sparse,
        identity,
        &data.as_slice()[..d],
        wide.as_mut_slice(),
    )?;
    Ok(Forward {
        data,
        wide,
        normalization,
        energy,
        identity,
    })
}

/// The adjoint sweep of a product ansatz: `dE/dtheta_k` for `k` from the last generator down to
/// `lowest` (the entries below it stay zero). `psi` is the final state and `wide` holds
/// `(H - identity) psi` and is overwritten. At step `k` the two rows are psi_k and its energy
/// adjoint lambda_k; unitary inversion recovers both preceding rows without retaining every
/// intermediate state or propagating all parameter derivatives. The step at `lowest` reads its
/// overlap and stops: nothing below it is wanted, so the rows are not unwound past it (with
/// `lowest = 0` this is the full gradient, whose last step never unwound either).
#[allow(clippy::too_many_arguments)]
fn adjoint_sweep<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    lowest: usize,
    psi: &[A],
    wide: &mut Buffer<C64>,
    energy: f64,
    normalization: f64,
) -> Result<Vec<f64>> {
    let d = basis.len();
    let mut work = stack::<A>(ctx, 2, d)?;
    work.as_mut_slice()[..d].copy_from_slice(psi);
    for ((out, &h), &p) in work.as_mut_slice()[d..]
        .iter_mut()
        .zip(wide.as_slice())
        .zip(psi)
    {
        // Removing E|psi> leaves the real gradient unchanged and avoids
        // cancellation from large constant shifts in the Hamiltonian.
        *out = A::narrow(h - energy * p.widen());
    }
    let mut gradient = vec![0.; theta.len()];
    // The adjoint step of a large excitation is one fused pass into a second pair of rows
    // (the two swap), not an overlap and two rotations; the same bits as they give.
    let mut gathered: Option<Buffer<A>> = None;
    for k in (lowest..theta.len()).rev() {
        if k > lowest && ops[k].lockstep(basis) {
            if let Some((x, linear, quadratic)) = pair_rotation(ops[k], -theta[k])? {
                let other = match gathered.as_mut() {
                    Some(other) => other,
                    None => gathered.insert(stack::<A>(ctx, 2, d)?),
                };
                let (psi_out, lambda_out) = other.as_mut_slice().split_at_mut(d);
                let (psi_now, lambda_now) = work.as_slice().split_at(d);
                let overlap = ops[k].adjoint_step(basis, x, linear, quadratic, psi_now, lambda_now, psi_out, lambda_out);
                std::mem::swap(&mut work, other);
                gradient[k] = 2. * overlap.re / normalization;
                continue;
            }
        }
        let overlap = match ops[k].single_mask() {
            Some(x) => ops[k].overlap_single(basis, x, &work.as_slice()[d..], &work.as_slice()[..d]),
            None => {
                ops[k].apply_into(basis, &work.as_slice()[..d], wide.as_mut_slice());
                inner_wide(&work.as_slice()[d..], wide.as_slice())
            }
        };
        gradient[k] = 2. * overlap.re / normalization;
        if k > lowest {
            exponential(
                ctx,
                basis,
                &[ops[k]],
                &[-theta[k]],
                work.as_mut_slice(),
                2,
                &[None],
                wide.as_mut_slice(),
            )?;
        }
    }
    Ok(gradient)
}

/// The work an [`importances`] call did, in whole-state actions: generator rotations of one row,
/// adjoint steps (the overlap, and for all but the last of a sweep the unwinding of two rows), and
/// energy expectations (`H psi`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Work {
    pub rotations: usize,
    pub adjoint_steps: usize,
    pub expectations: usize,
}
impl Work {
    /// What `positions` separate evaluations of a product ansatz of that many generators cost,
    /// each a full forward pass and a full sweep: the work [`importances`] replaces.
    pub fn separate_evaluations(length: usize) -> Self {
        Self {
            rotations: length * length,
            adjoint_steps: length * length,
            expectations: length,
        }
    }
}
impl std::ops::AddAssign for Work {
    fn add_assign(&mut self, other: Self) {
        self.rotations += other.rotations;
        self.adjoint_steps += other.adjoint_steps;
        self.expectations += other.expectations;
    }
}

pub struct Importances {
    /// `|dE/dtheta_i|` with `theta_i` set to zero, for every position `i`.
    pub values: Vec<f64>,
    pub work: Work,
}

/// The importance of every generator of a product ansatz: `|dE/dtheta_i|` evaluated with `theta_i`
/// set to zero and the other angles as given, which is what [`evaluate`] returns at that point.
///
/// Position `i` is the same product with generator `i` removed, so what the positions share is
/// computed once. Every evaluation starts with the same prefix `exp(theta_{i-1} G_{i-1}) ...
/// exp(theta_0 G_0)|reference>` (the code path of a rotation does not depend on what follows it),
/// so one running prefix state is advanced a generator at a time and each position copies it and
/// applies only the generators after `i`. Position `i` reads one gradient entry, so its sweep
/// stops at `i` ([`adjoint_sweep`]). The sweep runs from the final state of that position and
/// unwinds by inversion, so it shares nothing with the other positions: it is not shared.
///
/// The zero-angle rotation of generator `i` is skipped. It adds `0 * x` terms to a row, which
/// leaves every magnitude unchanged and can only turn a `-0` component into `+0`; every later
/// operation (products, sums, the zero tests, which treat `+-0` alike, and the `abs` the
/// importance takes) maps signed zeros to values that differ at most in the sign of an exact
/// zero, never in a nonzero value, so each importance is bit-for-bit what the separate
/// evaluation gives.
#[allow(clippy::too_many_arguments)]
pub fn importances<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    h: &Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    ops: &[&Operator],
    theta: &[f64],
) -> Result<Importances> {
    if h.n != basis.n
        || dense.is_some_and(|matrix| basis.len().checked_mul(basis.len()) != Some(matrix.len()))
    {
        return Err(Error::Invalid(
            "Hamiltonian representation must match the basis".into(),
        ));
    }
    if ops.len() != theta.len()
        || theta.iter().any(|t| !t.is_finite())
        || ops.iter().any(|g| g.n != basis.n || !g.is_antihermitian())
    {
        return Err(Error::Invalid(
            "finite parameters must match anti-Hermitian generators".into(),
        ));
    }
    let length = ops.len();
    let d = basis.len();
    let identity = h.identity();
    let mut prefix = stack::<A>(ctx, 1, d)?;
    prefix.as_mut_slice()[basis.reference] = A::narrow(C64::new(1., 0.));
    let mut data = stack::<A>(ctx, 1, d)?;
    let mut acc = ctx.allocate::<C64>(A::accumulator_len(d))?;
    let mut wide = ctx.allocate::<C64>(d)?;
    let mut gathered: Option<Buffer<A>> = None;
    let mut values = Vec::with_capacity(length);
    let mut work = Work::default();
    for i in 0..length {
        data.as_mut_slice().copy_from_slice(prefix.as_slice());
        rotate_product(
            ctx,
            basis,
            &ops[i + 1..],
            &theta[i + 1..],
            i + 1,
            1,
            &mut data,
            acc.as_mut_slice(),
            &mut gathered,
        )?;
        work.rotations += length - i - 1;
        if (norm(data.as_slice()) - 1.).abs() > A::norm_tolerance(length) {
            return Err(Error::Kernel(
                "exponential action lost state normalization".into(),
            ));
        }
        let (normalization, energy) = centered_expectation(
            basis,
            h,
            dense,
            sparse,
            identity,
            data.as_slice(),
            wide.as_mut_slice(),
        )?;
        work.expectations += 1;
        let gradient = adjoint_sweep::<A>(
            ctx,
            basis,
            ops,
            theta,
            i,
            data.as_slice(),
            &mut wide,
            energy.re,
            normalization,
        )?;
        work.adjoint_steps += length - i;
        if gradient[i..].iter().any(|g| !g.is_finite()) {
            return Err(Error::Kernel("nonfinite energy derivative".into()));
        }
        values.push(gradient[i].abs());
        if i + 1 < length {
            rotate_product(
                ctx,
                basis,
                &ops[i..=i],
                &theta[i..=i],
                i,
                1,
                &mut prefix,
                acc.as_mut_slice(),
                &mut gathered,
            )?;
            work.rotations += 1;
        }
    }
    Ok(Importances { values, work })
}
