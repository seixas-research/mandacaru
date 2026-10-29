//! Matrix-free exponential actions and analytic derivatives. Product ansatz
//! energies use an adjoint sweep; the summed UCC exponential uses a
//! differentiated Taylor recurrence (Frechet action).
use super::{Basis, Operator, C64};
use crate::{product, Buffer, Context, Error, Result};

pub fn inner(a: &[C64], b: &[C64]) -> C64 {
    a.iter().zip(b).map(|(x, y)| x.conj() * y).sum()
}
fn norm(a: &[C64]) -> f64 {
    a.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

pub(super) fn exponential(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    data: &mut [C64],
    rows: usize,
    directions: &[Option<usize>],
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
                let mut first = ctx.allocate::<C64>(d)?;
                let mut second = ctx.allocate::<C64>(d)?;
                for row in data.chunks_exact_mut(d).take(rows) {
                    generator.apply_into(basis, row, first.as_mut_slice());
                    generator.apply_into(basis, first.as_slice(), second.as_mut_slice());
                    for ((value, &a), &b) in
                        row.iter_mut().zip(first.as_slice()).zip(second.as_slice())
                    {
                        *value += linear * a + quadratic * b;
                    }
                }
                if directions.iter().any(Option::is_some) {
                    generator.apply_into(basis, &data[..d], first.as_mut_slice());
                    for (j, direction) in directions.iter().enumerate() {
                        if direction == &Some(0) {
                            for (value, &derivative) in data[(j + 1) * d..(j + 2) * d]
                                .iter_mut()
                                .zip(first.as_slice())
                            {
                                *value += derivative;
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
        let mut action = ctx.allocate::<C64>(d)?;
        let (sin, cos) = angle.sin_cos();
        for row in data.chunks_exact_mut(d).take(rows) {
            ops[0].apply_into(basis, row, action.as_mut_slice());
            for (out, &applied) in row.iter_mut().zip(action.as_slice()) {
                *out = *out * cos + applied * (sin / coefficient);
            }
        }
        if directions.iter().any(Option::is_some) {
            ops[0].apply_into(basis, &data[..d], action.as_mut_slice());
            for (j, direction) in directions.iter().enumerate() {
                if direction == &Some(0) {
                    for (out, &applied) in data[(j + 1) * d..(j + 2) * d]
                        .iter_mut()
                        .zip(action.as_slice())
                    {
                        *out += applied;
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
    let size = product(&[rows, d])?;
    let mut term = ctx.allocate::<C64>(size)?;
    let mut next = ctx.allocate::<C64>(size)?;
    let mut applied = ctx.allocate::<C64>(d)?;
    for _ in 0..steps {
        term.as_mut_slice().copy_from_slice(data);
        let mut converged = false;
        for order in 1..=80 {
            next.as_mut_slice().fill(C64::default());
            for row in 0..rows {
                for (g, t) in ops.iter().zip(theta) {
                    g.apply_into(
                        basis,
                        &term.as_slice()[row * d..(row + 1) * d],
                        applied.as_mut_slice(),
                    );
                    for (v, a) in next.as_mut_slice()[row * d..(row + 1) * d]
                        .iter_mut()
                        .zip(applied.as_slice())
                    {
                        *v += a * t;
                    }
                }
                if row > 0 {
                    if let Some(g) = directions[row - 1] {
                        ops[g].apply_into(basis, &term.as_slice()[..d], applied.as_mut_slice());
                        for (v, a) in next.as_mut_slice()[row * d..(row + 1) * d]
                            .iter_mut()
                            .zip(applied.as_slice())
                        {
                            *v += a;
                        }
                    }
                }
            }
            for (v, delta) in data.iter_mut().zip(next.as_mut_slice()) {
                *delta *= scale / order as f64;
                *v += *delta;
            }
            if norm(next.as_slice()) < 2e-16 * norm(data).max(1.) {
                converged = true;
                break;
            }
            std::mem::swap(&mut term, &mut next);
        }
        if !converged || data.iter().any(|c| !c.re.is_finite() || !c.im.is_finite()) {
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
    pub state: Buffer<C64>,
}
pub fn state(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
    derivatives: bool,
) -> Result<Buffer<C64>> {
    state_from(ctx, basis, ops, theta, trotter, derivatives, None)
}
#[allow(clippy::too_many_arguments)]
pub fn state_from(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
    derivatives: bool,
    initial: Option<&[C64]>,
) -> Result<Buffer<C64>> {
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
    let mut data = ctx.allocate::<C64>(product(&[rows, basis.len()])?)?;
    if let Some(initial) = initial {
        super::dynamics::check_state(initial, basis.len(), true)?;
        data.as_mut_slice()[..basis.len()].copy_from_slice(initial);
    } else {
        data.as_mut_slice()[basis.reference] = C64::new(1., 0.);
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
        )?;
    }
    let state_norm = norm(&data.as_slice()[..basis.len()]);
    if (state_norm - 1.).abs() > 1e-10 {
        return Err(Error::Kernel(
            "exponential action lost state normalization".into(),
        ));
    }
    Ok(data)
}
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    ctx: &Context,
    basis: &Basis,
    h: &Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
) -> Result<Evaluation> {
    evaluate_from(ctx, basis, h, dense, sparse, ops, theta, trotter, None)
}
#[allow(clippy::too_many_arguments)]
pub fn evaluate_from(
    ctx: &Context,
    basis: &Basis,
    h: &Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
    initial: Option<&[C64]>,
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
    let psi = &data.as_slice()[..d];
    let normalization = inner(psi, psi).re;
    let identity = h.identity();
    let mut hpsi = ctx.allocate::<C64>(d)?;
    if let Some(matrix) = dense {
        for (row, out) in hpsi.as_mut_slice().iter_mut().enumerate() {
            *out = matrix[row * d..(row + 1) * d]
                .iter()
                .zip(psi)
                .enumerate()
                .filter(|(column, _)| *column != row)
                .map(|(_, (a, b))| a * b)
                .sum::<C64>()
                + h.centered_diagonal(basis.indices.as_slice()[row]) * psi[row];
        }
    } else if let Some(sparse) = sparse {
        sparse.apply_centered_into(psi, hpsi.as_mut_slice());
    } else {
        h.apply_centered_into(basis, psi, hpsi.as_mut_slice());
    }
    let energy = inner(psi, hpsi.as_slice()) / normalization;
    if !energy.re.is_finite() || (energy.im + identity.im).abs() > 1e-10 {
        return Err(Error::Kernel("invalid Hermitian expectation value".into()));
    }
    let gradient: Vec<f64> = if trotter {
        // At step k the two rows are psi_k and its energy adjoint lambda_k.
        // Unitary inversion recovers both preceding rows without retaining
        // every intermediate state or propagating all parameter derivatives.
        let mut work = ctx.allocate::<C64>(product(&[2, d])?)?;
        work.as_mut_slice()[..d].copy_from_slice(psi);
        for (out, (&h, &p)) in work.as_mut_slice()[d..]
            .iter_mut()
            .zip(hpsi.as_slice().iter().zip(psi))
        {
            // Removing E|psi> leaves the real gradient unchanged and avoids
            // cancellation from large constant shifts in the Hamiltonian.
            *out = h - energy.re * p;
        }
        let mut action = ctx.allocate::<C64>(d)?;
        let mut gradient = vec![0.; theta.len()];
        for k in (0..theta.len()).rev() {
            ops[k].apply_into(basis, &work.as_slice()[..d], action.as_mut_slice());
            gradient[k] = 2. * inner(&work.as_slice()[d..], action.as_slice()).re / normalization;
            if k > 0 {
                exponential(
                    ctx,
                    basis,
                    &[ops[k]],
                    &[-theta[k]],
                    work.as_mut_slice(),
                    2,
                    &[None],
                )?;
            }
        }
        gradient
    } else {
        let mut residual = ctx.allocate::<C64>(d)?;
        for ((out, &h), &p) in residual
            .as_mut_slice()
            .iter_mut()
            .zip(hpsi.as_slice())
            .zip(psi)
        {
            *out = h - energy.re * p;
        }
        (0..theta.len())
            .map(|k| {
                2. * inner(
                    &data.as_slice()[(k + 1) * d..(k + 2) * d],
                    residual.as_slice(),
                )
                .re / normalization
            })
            .collect()
    };
    if gradient.iter().any(|g| !g.is_finite()) {
        return Err(Error::Kernel("nonfinite energy derivative".into()));
    }
    let mut state = ctx.allocate(d)?;
    state.as_mut_slice().copy_from_slice(psi);
    Ok(Evaluation {
        energy: energy.re + identity.re,
        gradient,
        state,
    })
}
