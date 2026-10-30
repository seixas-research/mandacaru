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
                let angle = theta[0] * frequency;
                if !angle.is_finite() {
                    return Err(Error::Invalid("excitation rotation angle overflow".into()));
                }
                let linear = angle.sin() / frequency;
                let quadratic = 2. * (0.5 * angle).sin().powi(2) / frequency.powi(2);
                let mut first = ctx.allocate::<A>(d)?;
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
        for (k, (g, t)) in ops.iter().zip(theta).enumerate() {
            let directions: Vec<_> = (0..rows - 1)
                .map(|j| if j == k { Some(0) } else { None })
                .collect();
            exponential(
                ctx,
                basis,
                &[*g],
                &[*t],
                data.as_mut_slice(),
                rows,
                &directions,
                acc.as_mut_slice(),
            )?;
        }
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
    if h.n != basis.n
        || dense.is_some_and(|matrix| basis.len().checked_mul(basis.len()) != Some(matrix.len()))
    {
        return Err(Error::Invalid(
            "Hamiltonian representation must match the basis".into(),
        ));
    }
    let data = state_from(ctx, basis, ops, theta, trotter, !trotter, initial)?;
    let d = basis.len();
    let rows = data.as_slice().len() / d;
    let psi = &data.as_slice()[..d];
    let normalization = inner(psi, psi).re;
    let identity = h.identity();
    // (H - identity) psi, accumulated in float64.
    let mut wide = ctx.allocate::<C64>(d)?;
    if let Some(matrix) = dense {
        for (row, out) in wide.as_mut_slice().iter_mut().enumerate() {
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
        sparse.apply_centered_into(psi, wide.as_mut_slice());
    } else {
        h.apply_centered_into(basis, psi, wide.as_mut_slice());
    }
    let energy = inner_wide(psi, wide.as_slice()) / normalization;
    if !energy.re.is_finite() || (energy.im + identity.im).abs() > A::HERMITIAN_TOLERANCE {
        return Err(Error::Kernel("invalid Hermitian expectation value".into()));
    }
    let gradient: Vec<f64> = if trotter {
        // At step k the two rows are psi_k and its energy adjoint lambda_k.
        // Unitary inversion recovers both preceding rows without retaining
        // every intermediate state or propagating all parameter derivatives.
        let mut work = stack::<A>(ctx, 2, d)?;
        work.as_mut_slice()[..d].copy_from_slice(psi);
        for ((out, &h), &p) in work.as_mut_slice()[d..]
            .iter_mut()
            .zip(wide.as_slice())
            .zip(psi)
        {
            // Removing E|psi> leaves the real gradient unchanged and avoids
            // cancellation from large constant shifts in the Hamiltonian.
            *out = A::narrow(h - energy.re * p.widen());
        }
        let mut gradient = vec![0.; theta.len()];
        for k in (0..theta.len()).rev() {
            ops[k].apply_into(basis, &work.as_slice()[..d], wide.as_mut_slice());
            gradient[k] =
                2. * inner_wide(&work.as_slice()[d..], wide.as_slice()).re / normalization;
            if k > 0 {
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
        gradient
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
