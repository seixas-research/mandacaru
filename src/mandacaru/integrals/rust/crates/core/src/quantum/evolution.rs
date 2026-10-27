//! Matrix-free exponential actions and exact forward derivatives. The summed
//! UCC exponential uses a differentiated Taylor recurrence (Frechet action),
//! rather than treating noncommuting generators as a product.
use super::{Basis, Operator, C64};
use crate::{product, Buffer, Context, Error, Result};

pub fn inner(a: &[C64], b: &[C64]) -> C64 {
    a.iter().zip(b).map(|(x, y)| x.conj() * y).sum()
}
fn norm(a: &[C64]) -> f64 {
    a.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

fn exponential(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    data: &mut [C64],
    rows: usize,
    directions: &[Option<usize>],
) -> Result<()> {
    let d = basis.len();
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
    data.as_mut_slice()[basis.reference] = C64::new(1., 0.);
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
    if h.n != basis.n
        || dense.is_some_and(|matrix| basis.len().checked_mul(basis.len()) != Some(matrix.len()))
    {
        return Err(Error::Invalid(
            "Hamiltonian representation must match the basis".into(),
        ));
    }
    let data = state(ctx, basis, ops, theta, trotter, true)?;
    let d = basis.len();
    let psi = &data.as_slice()[..d];
    let mut hpsi = ctx.allocate::<C64>(d)?;
    if let Some(matrix) = dense {
        for (row, out) in hpsi.as_mut_slice().iter_mut().enumerate() {
            *out = matrix[row * d..(row + 1) * d]
                .iter()
                .zip(psi)
                .map(|(a, b)| a * b)
                .sum();
        }
    } else if let Some(sparse) = sparse {
        sparse.apply_into(psi, hpsi.as_mut_slice());
    } else {
        h.apply_into(basis, psi, hpsi.as_mut_slice());
    }
    let energy = inner(psi, hpsi.as_slice());
    if !energy.re.is_finite() || energy.im.abs() > 1e-10 {
        return Err(Error::Kernel("invalid Hermitian expectation value".into()));
    }
    let gradient: Vec<f64> = (0..theta.len())
        .map(|k| 2. * inner(&data.as_slice()[(k + 1) * d..(k + 2) * d], hpsi.as_slice()).re)
        .collect();
    if gradient.iter().any(|g| !g.is_finite()) {
        return Err(Error::Kernel("nonfinite energy derivative".into()));
    }
    let mut state = ctx.allocate(d)?;
    state.as_mut_slice().copy_from_slice(psi);
    Ok(Evaluation {
        energy: energy.re,
        gradient,
        state,
    })
}
