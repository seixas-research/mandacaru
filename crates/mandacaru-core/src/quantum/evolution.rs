//! Matrix-free exponential actions and analytic derivatives. Product ansatz
//! energies use an adjoint sweep; the summed UCC exponential uses a
//! differentiated Taylor recurrence (Frechet action).
//!
//! Every kernel is generic over the stored amplitude (`Amplitude`): complex128
//! or complex64 rows. Many-term sums (a generator applied to a row, the Taylor
//! recurrence of one row) accumulate in a float64 row and are rounded once into
//! the stored row; every reduction is float64.
use super::amplitude::{
    finite, inner_wide, inner_wide_rows, norm_squared, stack, Amplitude, State,
};
use super::{parallel, Basis, Operator, C64};
use crate::{math::storage::Precision, Buffer, Context, Error, Result};
use rayon::prelude::*;

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
                        generator.rotate_pairs(
                            basis,
                            x,
                            linear,
                            quadratic,
                            row,
                            first.as_mut_slice(),
                        );
                    }
                } else {
                    let mut own = ctx.allocate::<C64>(A::own_wide_len(d))?;
                    for row in data.chunks_exact_mut(d).take(rows) {
                        apply_stored(generator, basis, row, first.as_mut_slice(), acc);
                        // G(G row) stays a float64 row: it is read, never stored.
                        let second = A::wide_row(own.as_mut_slice(), acc);
                        generator.apply_into(basis, first.as_slice(), second);
                        for ((value, &a), &b) in
                            row.iter_mut().zip(first.as_slice()).zip(second.iter())
                        {
                            *value =
                                A::narrow(value.widen() + (linear * a.widen() + quadratic * b));
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
/// and no copy back; the same bits as the in-place pair update the other path runs. When `rows == 1`
/// and `support` is tracked, sparse single-mask excitation steps touch only reachable pairs `(r, c)`.
#[allow(clippy::too_many_arguments)]
fn rotate_product_support<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    offset: usize,
    rows: usize,
    data: &mut Buffer<A>,
    acc: &mut [C64],
    gathered: &mut Option<Buffer<A>>,
    support: &mut Option<Vec<usize>>,
) -> Result<()> {
    let d = basis.len();
    for (k, (g, t)) in ops.iter().zip(theta).enumerate() {
        if rows == 1 {
            if let Some(sup) = support.as_mut() {
                if sup.len().saturating_mul(8) < d {
                    if let Some((x, linear, quadratic)) = pair_rotation(g, *t)? {
                        *sup = g.rotate_pairs_support(
                            basis,
                            x,
                            linear,
                            quadratic,
                            data.as_mut_slice(),
                            sup,
                        );
                        continue;
                    }
                }
                *support = None;
            }
            if g.lockstep(basis) {
                if let Some((x, linear, quadratic)) = pair_rotation(g, *t)? {
                    let other = match gathered.as_mut() {
                        Some(other) => other,
                        None => gathered.insert(ctx.allocate::<A>(d)?),
                    };
                    g.rotate_gather(
                        basis,
                        x,
                        linear,
                        quadratic,
                        data.as_slice(),
                        other.as_mut_slice(),
                    );
                    std::mem::swap(data, other);
                    continue;
                }
            }
        } else {
            *support = None;
        }
        let directions: Vec<_> = (0..rows - 1)
            .map(|j| if j == offset + k { Some(0) } else { None })
            .collect();
        exponential(
            ctx,
            basis,
            &[*g],
            &[*t],
            data.as_mut_slice(),
            rows,
            &directions,
            acc,
        )?;
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
    Ok(state_from_with_support(
        ctx,
        basis,
        ops,
        theta,
        trotter,
        derivatives,
        initial,
        basis.reference,
        true,
    )?
    .0)
}

#[allow(clippy::too_many_arguments)]
fn state_from_with_support<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
    derivatives: bool,
    initial: Option<&[A]>,
    start_row: usize,
    sparse_kernels: bool,
) -> Result<(Buffer<A>, Option<Vec<usize>>)> {
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
    if start_row >= d {
        return Err(Error::Invalid(
            "initial computational basis index exceeds basis dimension".into(),
        ));
    }
    let mut data = stack::<A>(ctx, rows, d)?;
    let mut acc = ctx.allocate::<C64>(A::accumulator_len(d))?;
    let mut support = if let Some(initial) = initial {
        super::dynamics::check_state(initial, d, true)?;
        data.as_mut_slice()[..d].copy_from_slice(initial);
        None
    } else {
        data.as_mut_slice()[start_row] = A::narrow(C64::new(1., 0.));
        (sparse_kernels && rows == 1 && d >= 1024).then(|| vec![start_row])
    };
    if trotter {
        let mut gathered: Option<Buffer<A>> = None;
        rotate_product_support(
            ctx,
            basis,
            ops,
            theta,
            0,
            rows,
            &mut data,
            acc.as_mut_slice(),
            &mut gathered,
            &mut support,
        )?;
    } else {
        support = None;
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
    let state_norm = match support.as_deref() {
        Some(sup) => sup
            .iter()
            .map(|&r| data.as_slice()[r].widen().norm_sqr())
            .sum::<f64>()
            .sqrt(),
        None => norm(&data.as_slice()[..d]),
    };
    if (state_norm - 1.).abs() > A::norm_tolerance(ops.len()) {
        return Err(Error::Kernel(
            "exponential action lost state normalization".into(),
        ));
    }
    Ok((data, support))
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
    evaluate_from_with_mode::<A>(
        ctx, basis, h, dense, sparse, ops, theta, trotter, initial, true,
    )
}
#[allow(clippy::too_many_arguments)]
pub fn evaluate_from_with_mode<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    h: &Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
    initial: Option<&[A]>,
    sparse_kernels: bool,
) -> Result<Evaluation> {
    evaluate_from_row::<A>(
        ctx,
        basis,
        h,
        dense,
        sparse,
        ops,
        theta,
        trotter,
        initial,
        basis.reference,
        sparse_kernels,
    )
}
/// Like [`evaluate_from_with_mode`], preparing from basis index `start_row` when `initial` is `None`.
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_from_row<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    h: &Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
    initial: Option<&[A]>,
    start_row: usize,
    sparse_kernels: bool,
) -> Result<Evaluation> {
    forward_from_row::<A>(
        ctx,
        basis,
        h,
        dense,
        sparse,
        ops,
        theta,
        trotter,
        initial,
        start_row,
        sparse_kernels,
    )?
    .finish(ctx, basis, ops, theta, trotter)
}

/// `(H - identity) psi` into `wide` (float64) and the energy `<psi|(H - identity)|psi> / <psi|psi>`
/// of the state, as `(norm squared, energy)`. `identity` is the Hamiltonian's identity term, which
/// the Hermiticity check adds back.
#[allow(clippy::too_many_arguments)]
fn centered_expectation<A: Amplitude>(
    basis: &Basis,
    h: &Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    identity: C64,
    psi: &[A],
    support: Option<&[usize]>,
    wide: &mut [C64],
) -> Result<(f64, C64)> {
    let d = basis.len();
    let normalization = match support {
        Some(sup) => sup.iter().map(|&r| psi[r].widen().norm_sqr()).sum::<f64>(),
        None => inner(psi, psi).re,
    };
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
    } else if let Some(sup) = support {
        h.apply_centered_support_into(basis, psi, sup, wide);
    } else {
        h.apply_centered_into(basis, psi, wide);
    }
    let expect = match support {
        Some(sup)
            if h.norm_bound() < 1e100 && normalization.is_finite() && normalization <= 1e100 =>
        {
            inner_wide_rows(psi, wide, sup)
        }
        _ => inner_wide(psi, wide),
    };
    let energy = expect / normalization;
    if !energy.re.is_finite() || (energy.im + identity.im).abs() > A::HERMITIAN_TOLERANCE {
        return Err(Error::Kernel("invalid Hermitian expectation value".into()));
    }
    Ok((normalization, energy))
}

/// Whether every operator in `ops` is a single-mask excitation with a certified rotation frequency,
/// eligible for the sparse tangent-vector gradient formulation.
fn can_tangent(ops: &[&Operator]) -> bool {
    ops.iter()
        .all(|g| g.single_mask().is_some() && g.terms.len() > 1 && g.rotation_frequency.is_some())
}

/// Threshold condition under which [`Forward::finish`] and [`importances`] switch from dense
/// `O(L D)` adjoint unwinding to the sparse `O(L^2 nnz)` tangent-vector formulation.
fn use_sparse_tangent<A: Amplitude>(
    d: usize,
    ops: &[&Operator],
    support: Option<&[usize]>,
) -> bool {
    A::PRECISION == Precision::Float64
        && d >= 1024
        && support.is_some_and(|s| s.len().saturating_mul(ops.len().max(1)).saturating_mul(4) < d)
        && can_tangent(ops)
}

/// Sorted 3-way sparse inner product `<tk | (hpsi - energy * psi)>` in `O(|tk| + |hpsi| + |psi|)`.
/// Computes `(hpsi_r - energy * psi_r)` elementwise before accumulating to avoid cancellation
/// when `energy` is large and to match the dense `(wide[r] - energy * psi[r])` residual order.
fn sparse_residual_inner(
    tk: &[(usize, C64)],
    hpsi: &[(usize, C64)],
    psi: &[(usize, C64)],
    energy: f64,
) -> C64 {
    let (mut j, mut m) = (0usize, 0usize);
    let mut sum = C64::default();
    for &(r, amp) in tk {
        while j < hpsi.len() && hpsi[j].0 < r {
            j += 1;
        }
        while m < psi.len() && psi[m].0 < r {
            m += 1;
        }
        let h_val = if j < hpsi.len() && hpsi[j].0 == r {
            hpsi[j].1
        } else {
            C64::default()
        };
        let p_val = if m < psi.len() && psi[m].0 == r {
            psi[m].1
        } else {
            C64::default()
        };
        sum += amp.conj() * (h_val - energy * p_val);
    }
    sum
}

/// Propagate `ops` with `theta` using sparse pair rotations from `|start_row>`, returning
/// `(rotations, states)` where `states[k] = U_k ... U_0 |start_row>`.
fn sparse_forward_states(
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    start_row: usize,
) -> Result<(Vec<(u64, f64, f64)>, Vec<Vec<(usize, C64)>>)> {
    let length = ops.len();
    let mut rotations = Vec::with_capacity(length);
    for (g, &t) in ops.iter().zip(theta) {
        let (x, linear, quadratic) = pair_rotation(g, t)?
            .ok_or_else(|| Error::Invalid("operator lacks single-mask pair rotation".into()))?;
        rotations.push((x, linear, quadratic));
    }
    let mut states = Vec::with_capacity(length);
    let mut cur = vec![(start_row, C64::new(1., 0.))];
    for (k, g) in ops.iter().enumerate() {
        let (x, linear, quadratic) = rotations[k];
        cur = g.rotate_sparse_state(basis, x, linear, quadratic, &cur);
        if k + 1 < length {
            states.push(cur.clone());
        } else {
            states.push(cur);
            break;
        }
    }
    Ok((rotations, states))
}

/// Evaluate `|d_k psi>` for each `k` via sparse tangent propagation and dot with a residual closure.
fn tangent_gradient_with(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    rotations: &[(u64, f64, f64)],
    states: &[Vec<(usize, C64)>],
    normalization: f64,
    overlap_with: impl Fn(&[(usize, C64)]) -> C64 + Sync,
) -> Vec<f64> {
    let length = ops.len();
    let work: usize = states
        .iter()
        .map(Vec::len)
        .sum::<usize>()
        .saturating_mul(length);
    let compute_k = |k: usize| -> f64 {
        let mut tk = ops[k].apply_sparse_state(basis, &states[k]);
        for j in k + 1..length {
            let (x, linear, quadratic) = rotations[j];
            tk = ops[j].rotate_sparse_state(basis, x, linear, quadratic, &tk);
        }
        2. * overlap_with(&tk).re / normalization
    };
    let threads = parallel::threads_for(ctx, work.saturating_mul(64), length);
    if threads > 1 {
        if let Some(pool) = ctx.pool() {
            return pool.install(|| {
                (0..length)
                    .into_par_iter()
                    .map(|k| parallel::inside(|| compute_k(k)))
                    .collect()
            });
        }
    }
    (0..length).map(compute_k).collect()
}

/// Analytic gradient `dE/dtheta_k = 2 Re <d_k psi | (H - E) | psi>` of a product ansatz via
/// sparse tangent-vector propagation:
/// `|d_k psi> = U_{L-1} ... U_{k+1} G_k (U_k ... U_0 |ref>)`.
///
/// Each tangent vector stays sparse (`O(nnz)` nonzeros), avoiding `O(L D)` dense adjoint
/// unwinding of `(H - E) |psi>` when `nnz << D`.
#[allow(clippy::too_many_arguments)]
pub fn tangent_gradient<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    psi: &[A],
    wide: &[C64],
    energy: f64,
    normalization: f64,
) -> Result<Vec<f64>> {
    tangent_gradient_row(
        ctx,
        basis,
        ops,
        theta,
        psi,
        wide,
        energy,
        normalization,
        basis.reference,
    )
}

#[allow(clippy::too_many_arguments)]
fn tangent_gradient_row<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    psi: &[A],
    wide: &[C64],
    energy: f64,
    normalization: f64,
    start_row: usize,
) -> Result<Vec<f64>> {
    if ops.len() != theta.len() || !can_tangent(ops) {
        return Err(Error::Invalid(
            "tangent gradient requires matching single-mask excitation operators".into(),
        ));
    }
    if ops.is_empty() {
        return Ok(vec![]);
    }
    let (rotations, states) = sparse_forward_states(basis, ops, theta, start_row)?;
    Ok(tangent_gradient_with(
        ctx,
        basis,
        ops,
        &rotations,
        &states,
        normalization,
        |tk| {
            tk.iter()
                .map(|&(r, amp)| amp.conj() * (wide[r] - energy * psi[r].widen()))
                .sum()
        },
    ))
}

/// Zero-`D` sparse tangent gradient from sparse `psi` and sparse `(H - I) psi`.
#[allow(clippy::too_many_arguments)]
fn tangent_gradient_sparse(
    ctx: &Context,
    basis: &Basis,
    ops: &[&Operator],
    theta: &[f64],
    psi_sparse: &[(usize, C64)],
    hpsi_sparse: &[(usize, C64)],
    energy: f64,
    normalization: f64,
    start_row: usize,
) -> Result<Vec<f64>> {
    if ops.is_empty() {
        return Ok(vec![]);
    }
    let (rotations, states) = sparse_forward_states(basis, ops, theta, start_row)?;
    Ok(tangent_gradient_with(
        ctx,
        basis,
        ops,
        &rotations,
        &states,
        normalization,
        |tk| sparse_residual_inner(tk, hpsi_sparse, psi_sparse, energy),
    ))
}

enum ForwardInner<'h, A: Amplitude> {
    Dense {
        data: Buffer<A>,
        wide: Buffer<C64>,
        support: Option<Vec<usize>>,
    },
    Sparse {
        entries: Vec<(usize, C64)>,
        kets: Vec<u64>,
        h: &'h Operator,
    },
}

/// The forward half of an evaluation: the prepared state (and its derivative rows when the
/// ansatz is not a product), `(H - identity) psi` and the energy. The gradient is the other half
/// ([`Forward::finish`]): an optimizer's rejected trial point needs only [`Forward::energy`], so
/// the half it does not use is never spent. When the circuit stays sparse on large registers, even
/// the state and `(H - identity) psi` stay sparse until [`Forward::finish`] materializes them.
pub struct Forward<'h, A: Amplitude> {
    inner: ForwardInner<'h, A>,
    start_row: usize,
    normalization: f64,
    energy: C64,
    identity: C64,
}
impl<'h, A: Amplitude> Forward<'h, A> {
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
        Ok(self.finish_inner(ctx, basis, ops, theta, trotter, false)?.0)
    }
    /// Like [`Self::finish`], also returning the intact `(H - identity) psi` buffer when the
    /// gradient pass left `wide` unmodified (for reuse in subsequent pool screening).
    pub fn finish_with_hpsi(
        self,
        ctx: &Context,
        basis: &Basis,
        ops: &[&Operator],
        theta: &[f64],
        trotter: bool,
    ) -> Result<(Evaluation, Option<Buffer<C64>>)> {
        self.finish_inner(ctx, basis, ops, theta, trotter, true)
    }
    fn finish_inner(
        self,
        ctx: &Context,
        basis: &Basis,
        ops: &[&Operator],
        theta: &[f64],
        trotter: bool,
        need_hpsi: bool,
    ) -> Result<(Evaluation, Option<Buffer<C64>>)> {
        let Self {
            inner,
            start_row,
            normalization,
            energy,
            identity,
        } = self;
        let d = basis.len();
        let (data, mut wide, support) = match inner {
            ForwardInner::Dense {
                data,
                wide,
                support,
            } => (data, wide, support),
            ForwardInner::Sparse { entries, kets, h } => {
                if !need_hpsi && entries.len().saturating_mul(ops.len()).saturating_mul(4) < d {
                    let hpsi_sparse = h.centered_action_sparse_on_kets(basis, &entries, &kets);
                    let gradient = tangent_gradient_sparse(
                        ctx,
                        basis,
                        ops,
                        theta,
                        &entries,
                        &hpsi_sparse,
                        energy.re,
                        normalization,
                        start_row,
                    )?;
                    if gradient.iter().any(|g| !g.is_finite()) {
                        return Err(Error::Kernel("nonfinite energy derivative".into()));
                    }
                    let mut state = stack::<A>(ctx, 1, d)?;
                    {
                        let slice = state.as_mut_slice();
                        for (r, amp) in entries {
                            slice[r] = A::narrow(amp);
                        }
                    }
                    return Ok((
                        Evaluation {
                            energy: energy.re + identity.re,
                            gradient,
                            state: A::wrap(state),
                        },
                        None,
                    ));
                }
                let mut data = stack::<A>(ctx, 1, d)?;
                let mut support = Vec::with_capacity(entries.len());
                {
                    let slice = data.as_mut_slice();
                    for (r, amp) in entries {
                        slice[r] = A::narrow(amp);
                        support.push(r);
                    }
                }
                let mut wide = ctx.allocate::<C64>(d)?;
                h.apply_centered_support_into(
                    basis,
                    data.as_slice(),
                    &support,
                    wide.as_mut_slice(),
                );
                (data, wide, Some(support))
            }
        };
        let rows = data.as_slice().len() / d;
        let psi = &data.as_slice()[..d];
        let preserves_wide = trotter && (can_tangent(ops) || ops.len() <= 1);
        let gradient: Vec<f64> = if trotter {
            if use_sparse_tangent::<A>(d, ops, support.as_deref()) {
                tangent_gradient_row(
                    ctx,
                    basis,
                    ops,
                    theta,
                    psi,
                    wide.as_slice(),
                    energy.re,
                    normalization,
                    start_row,
                )?
            } else {
                adjoint_sweep::<A>(
                    ctx,
                    basis,
                    ops,
                    theta,
                    0,
                    psi,
                    &mut wide,
                    energy.re,
                    normalization,
                )?
            }
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
        let hpsi = if preserves_wide { Some(wide) } else { None };
        let state = if rows == 1 {
            data
        } else {
            let mut state = ctx.allocate::<A>(d)?;
            state.as_mut_slice().copy_from_slice(&data.as_slice()[..d]);
            state
        };
        Ok((
            Evaluation {
                energy: energy.re + identity.re,
                gradient,
                state: A::wrap(state),
            },
            hpsi,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn forward_from<'h, A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    h: &'h Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
    initial: Option<&[A]>,
) -> Result<Forward<'h, A>> {
    forward_from_with_mode::<A>(
        ctx, basis, h, dense, sparse, ops, theta, trotter, initial, true,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_from_with_mode<'h, A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    h: &'h Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
    initial: Option<&[A]>,
    sparse_kernels: bool,
) -> Result<Forward<'h, A>> {
    forward_from_row::<A>(
        ctx,
        basis,
        h,
        dense,
        sparse,
        ops,
        theta,
        trotter,
        initial,
        basis.reference,
        sparse_kernels,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_from_row<'h, A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    h: &'h Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    ops: &[&Operator],
    theta: &[f64],
    trotter: bool,
    initial: Option<&[A]>,
    start_row: usize,
    sparse_kernels: bool,
) -> Result<Forward<'h, A>> {
    if h.n != basis.n
        || dense.is_some_and(|matrix| basis.len().checked_mul(basis.len()) != Some(matrix.len()))
    {
        return Err(Error::Invalid(
            "Hamiltonian representation must match the basis".into(),
        ));
    }
    let d = basis.len();
    if start_row >= d {
        return Err(Error::Invalid(
            "initial computational basis index exceeds basis dimension".into(),
        ));
    }
    let identity = h.identity();
    if sparse_kernels
        && trotter
        && initial.is_none()
        && dense.is_none()
        && A::PRECISION == Precision::Float64
        && d >= 1024
        && can_tangent(ops)
    {
        if ops.len() != theta.len()
            || theta.iter().any(|t| !t.is_finite())
            || ops.iter().any(|g| g.n != basis.n || !g.is_antihermitian())
        {
            return Err(Error::Invalid(
                "finite parameters must match anti-Hermitian generators".into(),
            ));
        }
        let sparse_limit = d / 8;
        let mut entries = vec![(start_row, C64::new(1., 0.))];
        let mut fell_back = None;
        for (k, (&g, &t)) in ops.iter().zip(theta).enumerate() {
            let (x, linear, quadratic) = pair_rotation(g, t)?
                .ok_or_else(|| Error::Invalid("operator lacks single-mask pair rotation".into()))?;
            entries = g.rotate_sparse_state(basis, x, linear, quadratic, &entries);
            if entries.len() >= sparse_limit {
                fell_back = Some(k + 1);
                break;
            }
        }
        if let Some(next_k) = fell_back {
            let mut data = stack::<A>(ctx, 1, d)?;
            for (r, amp) in entries {
                data.as_mut_slice()[r] = A::narrow(amp);
            }
            let mut acc = ctx.allocate::<C64>(A::accumulator_len(d))?;
            let mut gathered = None;
            let mut support = None;
            rotate_product_support(
                ctx,
                basis,
                &ops[next_k..],
                &theta[next_k..],
                0,
                1,
                &mut data,
                acc.as_mut_slice(),
                &mut gathered,
                &mut support,
            )?;
            let state_norm = norm(&data.as_slice()[..d]);
            if (state_norm - 1.).abs() > A::norm_tolerance(ops.len()) {
                return Err(Error::Kernel(
                    "exponential action lost state normalization".into(),
                ));
            }
            let mut wide = ctx.allocate::<C64>(d)?;
            let (normalization, energy) = centered_expectation(
                basis,
                h,
                dense,
                sparse,
                identity,
                &data.as_slice()[..d],
                None,
                wide.as_mut_slice(),
            )?;
            return Ok(Forward {
                inner: ForwardInner::Dense {
                    data,
                    wide,
                    support: None,
                },
                start_row,
                normalization,
                energy,
                identity,
            });
        }
        let normalization: f64 = entries.iter().map(|&(_, a)| a.norm_sqr()).sum();
        if (normalization.sqrt() - 1.).abs() > A::norm_tolerance(ops.len()) {
            return Err(Error::Kernel(
                "exponential action lost state normalization".into(),
            ));
        }
        let kets: Vec<u64> = entries
            .iter()
            .map(|&(r, _)| Operator::ket(basis, r))
            .collect();
        let expect = h.centered_expectation_sparse(&entries, &kets);
        let energy = expect / normalization;
        if !energy.re.is_finite() || (energy.im + identity.im).abs() > A::HERMITIAN_TOLERANCE {
            return Err(Error::Kernel("invalid Hermitian expectation value".into()));
        }
        return Ok(Forward {
            inner: ForwardInner::Sparse { entries, kets, h },
            start_row,
            normalization,
            energy,
            identity,
        });
    }
    // A non-product ansatz keeps its derivative rows in the stack its Taylor series stops on, so
    // they are part of the energy's bits and are never dropped here.
    let (data, support) = state_from_with_support(
        ctx,
        basis,
        ops,
        theta,
        trotter,
        !trotter,
        initial,
        start_row,
        sparse_kernels,
    )?;
    let mut wide = ctx.allocate::<C64>(d)?;
    let (normalization, energy) = centered_expectation(
        basis,
        h,
        dense,
        sparse,
        identity,
        &data.as_slice()[..d],
        support.as_deref(),
        wide.as_mut_slice(),
    )?;
    Ok(Forward {
        inner: ForwardInner::Dense {
            data,
            wide,
            support,
        },
        start_row,
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
                let overlap = ops[k].adjoint_step(
                    basis, x, linear, quadratic, psi_now, lambda_now, psi_out, lambda_out,
                );
                std::mem::swap(&mut work, other);
                gradient[k] = 2. * overlap.re / normalization;
                continue;
            }
        }
        let overlap = match ops[k].single_mask() {
            Some(x) => {
                ops[k].overlap_single(basis, x, &work.as_slice()[d..], &work.as_slice()[..d])
            }
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

struct ImportanceScratch<A: Amplitude> {
    data: Buffer<A>,
    acc: Buffer<C64>,
    wide: Buffer<C64>,
    gathered: Option<Buffer<A>>,
    dirty_support: Option<Vec<usize>>,
}

impl<A: Amplitude> ImportanceScratch<A> {
    fn new(ctx: &Context, d: usize) -> Result<Self> {
        Ok(Self {
            data: stack::<A>(ctx, 1, d)?,
            acc: ctx.allocate::<C64>(A::accumulator_len(d))?,
            wide: ctx.allocate::<C64>(d)?,
            gathered: None,
            dirty_support: Some(Vec::new()),
        })
    }

    fn load_prefix(&mut self, prefix: &[A], prefix_support: Option<&[usize]>) {
        let dst = self.data.as_mut_slice();
        match (self.dirty_support.take(), prefix_support) {
            (Some(dirty), Some(sup)) => {
                for r in dirty {
                    dst[r] = A::default();
                }
                for &r in sup {
                    dst[r] = prefix[r];
                }
            }
            _ => dst.copy_from_slice(prefix),
        }
    }
}

/// Evaluate `|dE/dtheta_i|` at `theta_i = 0` for a single position `i`, given `prefix` =
/// `U_{i-1} ... U_0 |ref>` and its optional nonzero support `prefix_support`.
#[allow(clippy::too_many_arguments)]
fn importance_at<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    h: &Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    identity: C64,
    ops: &[&Operator],
    theta: &[f64],
    i: usize,
    prefix: &[A],
    prefix_support: Option<&[usize]>,
    scratch: &mut ImportanceScratch<A>,
) -> Result<(f64, Work)> {
    let length = ops.len();
    let d = basis.len();
    scratch.load_prefix(prefix, prefix_support);
    let mut cur_support = prefix_support.map(|s| s.to_vec());
    rotate_product_support(
        ctx,
        basis,
        &ops[i + 1..],
        &theta[i + 1..],
        i + 1,
        1,
        &mut scratch.data,
        scratch.acc.as_mut_slice(),
        &mut scratch.gathered,
        &mut cur_support,
    )?;
    let state_norm = match cur_support.as_deref() {
        Some(sup) => sup
            .iter()
            .map(|&r| scratch.data.as_slice()[r].widen().norm_sqr())
            .sum::<f64>()
            .sqrt(),
        None => norm(scratch.data.as_slice()),
    };
    if (state_norm - 1.).abs() > A::norm_tolerance(length) {
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
        scratch.data.as_slice(),
        cur_support.as_deref(),
        scratch.wide.as_mut_slice(),
    )?;
    let importance = if use_sparse_tangent::<A>(d, ops, cur_support.as_deref()) {
        if let Some(p_sup) = prefix_support {
            let p_sparse: Vec<(usize, C64)> =
                p_sup.iter().map(|&r| (r, prefix[r].widen())).collect();
            let mut ti = ops[i].apply_sparse_state(basis, &p_sparse);
            for j in i + 1..length {
                let (x, linear, quadratic) = pair_rotation(ops[j], theta[j])?.ok_or_else(|| {
                    Error::Invalid("operator lacks single-mask pair rotation".into())
                })?;
                ti = ops[j].rotate_sparse_state(basis, x, linear, quadratic, &ti);
            }
            let psi = scratch.data.as_slice();
            let wide = scratch.wide.as_slice();
            let overlap: C64 = ti
                .iter()
                .map(|&(r, amp)| amp.conj() * (wide[r] - energy.re * psi[r].widen()))
                .sum();
            let g_i = 2. * overlap.re / normalization;
            if !g_i.is_finite() {
                return Err(Error::Kernel("nonfinite energy derivative".into()));
            }
            g_i.abs()
        } else {
            let gradient = adjoint_sweep::<A>(
                ctx,
                basis,
                ops,
                theta,
                i,
                scratch.data.as_slice(),
                &mut scratch.wide,
                energy.re,
                normalization,
            )?;
            if gradient[i..].iter().any(|g| !g.is_finite()) {
                return Err(Error::Kernel("nonfinite energy derivative".into()));
            }
            gradient[i].abs()
        }
    } else {
        let gradient = adjoint_sweep::<A>(
            ctx,
            basis,
            ops,
            theta,
            i,
            scratch.data.as_slice(),
            &mut scratch.wide,
            energy.re,
            normalization,
        )?;
        if gradient[i..].iter().any(|g| !g.is_finite()) {
            return Err(Error::Kernel("nonfinite energy derivative".into()));
        }
        gradient[i].abs()
    };
    scratch.dirty_support = cur_support;
    Ok((
        importance,
        Work {
            rotations: length - i - 1,
            adjoint_steps: length - i,
            expectations: 1,
        },
    ))
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
/// When multiple worker threads are available on [`Context::pool`] and the memory budget allows
/// storing the `L` prefix states, positions `0..L` are evaluated in parallel using interleaved
/// load balancing, producing bit-identical results across thread counts.
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
    importances_with_mode::<A>(ctx, basis, h, dense, sparse, ops, theta, true)
}

/// Zero-`D` sparse importance sweep: keeps prefix, suffix, `(H - I) psi`, and tangent states as
/// sorted sparse `(usize, C64)` vectors when all `L` evaluated circuits stay sparse, parallelized
/// cleanly via Rayon across positions and bit-identical across thread counts. Returns `Ok(None)` if
/// any state crosses the sparse crossover so [`importances_with_mode`] falls back to dense cleanly.
#[allow(clippy::too_many_arguments)]
fn importances_sparse(
    ctx: &Context,
    basis: &Basis,
    h: &Operator,
    identity: C64,
    ops: &[&Operator],
    theta: &[f64],
    start_row: usize,
) -> Result<Option<Importances>> {
    let length = ops.len();
    let d = basis.len();
    let sparse_limit = d / 8;
    let is_sparse =
        |nnz: usize| nnz < sparse_limit && nnz.saturating_mul(length).saturating_mul(4) < d;

    let mut rotations = Vec::with_capacity(length);
    for (g, &t) in ops.iter().zip(theta) {
        let Some(rot) = pair_rotation(g, t)? else {
            return Ok(None);
        };
        rotations.push(rot);
    }
    let mut prefixes = Vec::with_capacity(length);
    let mut cur = vec![(start_row, C64::new(1., 0.))];
    for i in 0..length {
        if !is_sparse(cur.len()) {
            return Ok(None);
        }
        if i + 1 < length {
            prefixes.push(cur.clone());
            let (x, linear, quadratic) = rotations[i];
            cur = ops[i].rotate_sparse_state(basis, x, linear, quadratic, &cur);
        } else {
            prefixes.push(cur);
            break;
        }
    }

    let eval_position = |i: usize| -> Result<Option<(f64, Work)>> {
        let mut psi = prefixes[i].clone();
        for j in i + 1..length {
            let (x, linear, quadratic) = rotations[j];
            psi = ops[j].rotate_sparse_state(basis, x, linear, quadratic, &psi);
            if !is_sparse(psi.len()) {
                return Ok(None);
            }
        }
        let norm_sqr: f64 = psi.iter().map(|&(_, a)| a.norm_sqr()).sum();
        if (norm_sqr.sqrt() - 1.).abs() > C64::norm_tolerance(length) {
            return Err(Error::Kernel(
                "exponential action lost state normalization".into(),
            ));
        }
        let kets: Vec<u64> = psi.iter().map(|&(r, _)| Operator::ket(basis, r)).collect();
        let expect = h.centered_expectation_sparse(&psi, &kets);
        let energy = expect / norm_sqr;
        if !energy.re.is_finite() || (energy.im + identity.im).abs() > C64::HERMITIAN_TOLERANCE {
            return Err(Error::Kernel("invalid Hermitian expectation value".into()));
        }
        let hpsi = h.centered_action_sparse_on_kets(basis, &psi, &kets);
        let mut ti = ops[i].apply_sparse_state(basis, &prefixes[i]);
        for j in i + 1..length {
            let (x, linear, quadratic) = rotations[j];
            ti = ops[j].rotate_sparse_state(basis, x, linear, quadratic, &ti);
        }
        let overlap = sparse_residual_inner(&ti, &hpsi, &psi, energy.re);
        let g_i = 2. * overlap.re / norm_sqr;
        if !g_i.is_finite() {
            return Err(Error::Kernel("nonfinite energy derivative".into()));
        }
        Ok(Some((
            g_i.abs(),
            Work {
                rotations: length - i - 1,
                adjoint_steps: length - i,
                expectations: 1,
            },
        )))
    };

    let logical_work: usize = prefixes
        .iter()
        .map(Vec::len)
        .sum::<usize>()
        .saturating_mul(length)
        .saturating_mul(64);
    let lanes = parallel::threads_for(ctx, logical_work, length);
    let outcomes: Vec<Result<Option<(f64, Work)>>> = if lanes > 1 {
        if let Some(pool) = ctx.pool() {
            pool.install(|| {
                (0..length)
                    .into_par_iter()
                    .map(|i| parallel::inside(|| eval_position(i)))
                    .collect()
            })
        } else {
            (0..length).map(eval_position).collect()
        }
    } else {
        (0..length).map(eval_position).collect()
    };

    let mut values = Vec::with_capacity(length);
    let mut work = Work {
        rotations: length - 1,
        ..Work::default()
    };
    for item in outcomes {
        let Some((val, item_work)) = item? else {
            return Ok(None);
        };
        values.push(val);
        work += item_work;
    }
    Ok(Some(Importances { values, work }))
}

#[allow(clippy::too_many_arguments)]
pub fn importances_with_mode<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    h: &Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    ops: &[&Operator],
    theta: &[f64],
    sparse_kernels: bool,
) -> Result<Importances> {
    importances_from_row::<A>(
        ctx,
        basis,
        h,
        dense,
        sparse,
        ops,
        theta,
        basis.reference,
        sparse_kernels,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn importances_from_row<A: Amplitude>(
    ctx: &Context,
    basis: &Basis,
    h: &Operator,
    dense: Option<&[C64]>,
    sparse: Option<&super::sparse::SparseMatrix>,
    ops: &[&Operator],
    theta: &[f64],
    start_row: usize,
    sparse_kernels: bool,
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
    if length == 0 {
        return Ok(Importances {
            values: vec![],
            work: Work::default(),
        });
    }
    let d = basis.len();
    if start_row >= d {
        return Err(Error::Invalid(
            "initial computational basis index exceeds basis dimension".into(),
        ));
    }
    let identity = h.identity();
    if sparse_kernels
        && dense.is_none()
        && A::PRECISION == Precision::Float64
        && d >= 1024
        && can_tangent(ops)
    {
        if let Some(sparse_result) =
            importances_sparse(ctx, basis, h, identity, ops, theta, start_row)?
        {
            return Ok(sparse_result);
        }
    }
    let logical_work = length
        .saturating_mul(length + 1)
        .saturating_div(2)
        .saturating_mul(d);
    let lanes = parallel::threads_for(ctx, logical_work, length);
    let prefix_bytes = length
        .saturating_mul(d)
        .saturating_mul(std::mem::size_of::<A>());
    let lane_bytes = lanes.saturating_mul(
        d.saturating_mul(std::mem::size_of::<A>() * 3 + std::mem::size_of::<C64>()),
    );
    let available = ctx.memory_limit().saturating_sub(ctx.memory_used());
    if lanes > 1 && prefix_bytes.saturating_add(lane_bytes) <= available / 2 {
        if let Some(pool) = ctx.pool() {
            let mut prefix = stack::<A>(ctx, 1, d)?;
            prefix.as_mut_slice()[start_row] = A::narrow(C64::new(1., 0.));
            let mut prefix_support: Option<Vec<usize>> =
                (sparse_kernels && d >= 1024).then(|| vec![start_row]);
            let mut acc = ctx.allocate::<C64>(A::accumulator_len(d))?;
            let mut gathered: Option<Buffer<A>> = None;
            let mut prefixes: Vec<(Buffer<A>, Option<Vec<usize>>)> = Vec::with_capacity(length);
            for i in 0..length {
                let mut snap = stack::<A>(ctx, 1, d)?;
                match prefix_support.as_deref() {
                    Some(sup) => {
                        for &r in sup {
                            snap.as_mut_slice()[r] = prefix.as_slice()[r];
                        }
                    }
                    None => snap.as_mut_slice().copy_from_slice(prefix.as_slice()),
                }
                prefixes.push((snap, prefix_support.clone()));
                if i + 1 < length {
                    rotate_product_support(
                        ctx,
                        basis,
                        &ops[i..=i],
                        &theta[i..=i],
                        i,
                        1,
                        &mut prefix,
                        acc.as_mut_slice(),
                        &mut gathered,
                        &mut prefix_support,
                    )?;
                }
            }
            drop(prefix);
            drop(acc);
            drop(gathered);
            let mut scratches: Vec<ImportanceScratch<A>> = (0..lanes)
                .map(|_| ImportanceScratch::new(ctx, d))
                .collect::<Result<Vec<_>>>()?;
            let results: Vec<Result<Vec<(usize, f64, Work)>>> = pool.install(|| {
                scratches
                    .par_iter_mut()
                    .enumerate()
                    .map(|(w, scratch)| {
                        parallel::inside(|| {
                            let mut local = Vec::new();
                            for i in (w..length).step_by(lanes) {
                                let (ref p_buf, ref p_sup) = prefixes[i];
                                let (val, item_work) = importance_at(
                                    ctx,
                                    basis,
                                    h,
                                    dense,
                                    sparse,
                                    identity,
                                    ops,
                                    theta,
                                    i,
                                    p_buf.as_slice(),
                                    p_sup.as_deref(),
                                    scratch,
                                )?;
                                local.push((i, val, item_work));
                            }
                            Ok(local)
                        })
                    })
                    .collect()
            });
            let mut values = vec![0.; length];
            let mut item_works = vec![Work::default(); length];
            for bucket in results {
                for (i, val, item_work) in bucket? {
                    values[i] = val;
                    item_works[i] = item_work;
                }
            }
            let mut work = Work {
                rotations: length - 1,
                ..Work::default()
            };
            for item_work in item_works {
                work += item_work;
            }
            return Ok(Importances { values, work });
        }
    }
    let mut prefix = stack::<A>(ctx, 1, d)?;
    prefix.as_mut_slice()[start_row] = A::narrow(C64::new(1., 0.));
    let mut prefix_support: Option<Vec<usize>> =
        (sparse_kernels && d >= 1024).then(|| vec![start_row]);
    let mut prefix_acc = ctx.allocate::<C64>(A::accumulator_len(d))?;
    let mut prefix_gathered: Option<Buffer<A>> = None;
    let mut scratch = ImportanceScratch::new(ctx, d)?;
    let mut values = Vec::with_capacity(length);
    let mut work = Work::default();
    for i in 0..length {
        let (val, item_work) = importance_at(
            ctx,
            basis,
            h,
            dense,
            sparse,
            identity,
            ops,
            theta,
            i,
            prefix.as_slice(),
            prefix_support.as_deref(),
            &mut scratch,
        )?;
        values.push(val);
        work += item_work;
        if i + 1 < length {
            rotate_product_support(
                ctx,
                basis,
                &ops[i..=i],
                &theta[i..=i],
                i,
                1,
                &mut prefix,
                prefix_acc.as_mut_slice(),
                &mut prefix_gathered,
                &mut prefix_support,
            )?;
            work.rotations += 1;
        }
    }
    Ok(Importances { values, work })
}
