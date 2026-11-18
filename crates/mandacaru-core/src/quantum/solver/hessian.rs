use super::{Checkpointing, QuantumProblem};
use crate::Result;

/// The `n x n` Hessian of `m` parameters (row-major) as the start for `n >= m`: the old block,
/// the identity for each appended parameter, no coupling between old and new.
pub(in crate::quantum) fn extend_hessian(old: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut grown = vec![0.; n * n];
    for i in 0..n {
        if i < m {
            grown[i * n..i * n + m].copy_from_slice(&old[i * m..(i + 1) * m]);
        } else {
            grown[i * n + i] = 1.;
        }
    }
    grown
}
/// The INVERSE Hessian of `n` parameters without the one at `at`: the inverse of the reduced
/// Hessian, `H_ij - H_ia H_aj / H_aa` (a Schur complement, not the sub-block).
pub(in crate::quantum) fn drop_parameter_of_inverse(old: &[f64], n: usize, at: usize) -> Vec<f64> {
    let keep: Vec<usize> = (0..n).filter(|&i| i != at).collect();
    let pivot = old[at * n + at];
    keep.iter()
        .flat_map(|&i| keep.iter().map(move |&j| (i, j)))
        .map(|(i, j)| {
            if pivot.abs() > f64::MIN_POSITIVE {
                old[i * n + j] - old[i * n + at] * old[at * n + j] / pivot
            } else {
                old[i * n + j]
            }
        })
        .collect()
}
/// The Hessian of `n` parameters without the one at `at`.
pub(in crate::quantum) fn drop_parameter(old: &[f64], n: usize, at: usize) -> Vec<f64> {
    let keep: Vec<usize> = (0..n).filter(|&i| i != at).collect();
    keep.iter()
        .flat_map(|&i| keep.iter().map(move |&j| (i, j)))
        .map(|(i, j)| old[i * n + j])
        .collect()
}
/// Insert a new uncoupled parameter at index `at` into an `n x n` row-major curvature matrix,
/// yielding an `(n + 1) x (n + 1)` matrix with `1.0` on the new diagonal entry.
pub(in crate::quantum) fn insert_hessian(old: &[f64], n: usize, at: usize) -> Vec<f64> {
    let m = n + 1;
    let mut grown = vec![0.; m * m];
    for i in 0..m {
        if i == at {
            grown[i * m + i] = 1.;
            continue;
        }
        let oi = if i > at { i - 1 } else { i };
        for j in 0..m {
            if j == at {
                continue;
            }
            let oj = if j > at { j - 1 } else { j };
            grown[i * m + j] = old[oi * n + oj];
        }
    }
    grown
}
/// Reset row and column `at` of an `n x n` row-major curvature matrix to the unit vector `e_at`,
/// keeping the remaining `(n - 1) x (n - 1)` block positive definite when replacing an operator.
pub(in crate::quantum) fn reset_hessian_parameter(old: &[f64], n: usize, at: usize) -> Vec<f64> {
    let mut out = old.to_vec();
    for k in 0..n {
        out[at * n + k] = 0.;
        out[k * n + at] = 0.;
    }
    out[at * n + at] = 1.;
    out
}

/// One mid-run checkpoint: the ansatz and parameters so far, marked incomplete. `growth` says
/// whether `count` counts growth steps (ADAPT) or cost evaluations (a fixed ansatz).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(super) fn write_checkpoint(
    problem: &QuantumProblem,
    plan: &Checkpointing,
    indices: &[usize],
    parameters: &[f64],
    product_form: bool,
    energy: f64,
    growth: bool,
    count: usize,
    optimizer: &str,
    hessian: Option<&[f64]>,
) -> Result<()> {
    let status = crate::persistence::status_with_hessian(
        serde_json::json!({
            "complete": false,
            "converged": false,
            (if growth { "iteration" } else { "num_evaluations" }): count,
        }),
        optimizer,
        hessian,
    )?;
    crate::persistence::checkpoint(
        &problem.context,
        problem,
        indices,
        parameters,
        product_form,
        &plan.method,
        status,
        energy,
    )?
    .save(&plan.path)
}

#[cfg(test)]
mod hessian_tests {
    use super::{drop_parameter, drop_parameter_of_inverse, extend_hessian};
    #[test]
    fn dropping_a_parameter_from_an_inverse_gives_the_inverse_of_the_reduced_hessian() {
        // B = [[4, 1, 0.5], [1, 3, -0.4], [0.5, -0.4, 2]]; its inverse, then the inverse of B
        // without parameter 1, from the Schur complement.
        let b =
            nalgebra::DMatrix::from_row_slice(3, 3, &[4., 1., 0.5, 1., 3., -0.4, 0.5, -0.4, 2.]);
        let inverse = b.clone().try_inverse().unwrap();
        let row_major: Vec<f64> = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|ij| inverse[ij])
            .collect();
        let reduced = nalgebra::DMatrix::from_row_slice(2, 2, &[4., 0.5, 0.5, 2.])
            .try_inverse()
            .unwrap();
        let dropped = drop_parameter_of_inverse(&row_major, 3, 1);
        for (k, value) in dropped.iter().enumerate() {
            assert!((value - reduced[(k / 2, k % 2)]).abs() < 1e-12);
        }
    }
    #[test]
    fn a_grown_hessian_keeps_the_old_block_and_adds_the_identity() {
        let old = [2., 0.5, 0.5, 3.];
        assert_eq!(
            extend_hessian(&old, 2, 4),
            [2., 0.5, 0., 0., 0.5, 3., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]
        );
        assert_eq!(extend_hessian(&old, 2, 2), old);
    }
    #[test]
    fn dropping_a_parameter_removes_its_row_and_column() {
        let h = [1., 2., 3., 2., 4., 5., 3., 5., 6.];
        assert_eq!(drop_parameter(&h, 3, 1), [1., 3., 3., 6.]);
        assert_eq!(drop_parameter(&h, 3, 0), [4., 5., 5., 6.]);
    }
}
