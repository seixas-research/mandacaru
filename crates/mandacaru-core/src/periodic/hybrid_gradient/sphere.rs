//! The one-center terms' response to the projections: `E_oc` is a function of the projected
//! density `P_s = sum_k w_k C_k^+ D_s(k) C_k` alone (the sphere model is rigid with its atom),
//! so `dE_oc = sum_s tr[(-a O_s) dP_s]` and, per k, `Z = -a w_k O_s C_k^+ D_s(k)` (`p x m`): the
//! engine's `H_C C^+ D` with `H_C = -a O`. `O_s` is what the model returns for `P`; nothing of
//! the model is re-derived here.
use super::Matrix;
use crate::{
    math::storage::MatrixSource,
    periodic::{
        invalid,
        screened_augmentation::Augmentation,
        screened_exchange::{one_center, Screening},
        Crystal,
    },
    quantum::C64,
    Result,
};

/// `(Z per (spin, k), E_oc)`; empty and zero for an augmentation without a model.
pub(super) fn response<M: MatrixSource>(
    c: &Crystal,
    matrices: &[M],
    screening: Screening,
    augmentation: &Augmentation,
) -> Result<(Vec<Matrix>, f64)> {
    let nk = c.points.len();
    let zero = || vec![Matrix::zeros(c.p, c.m); c.spins * nk];
    let Some(model) = augmentation.one_center() else {
        return Ok((zero(), 0.));
    };
    let terms = one_center(c, matrices, model)?;
    if terms.operators.len() != c.spins {
        return Err(invalid(
            "the one-center model returned the wrong number of operators",
        ));
    }
    let mut out = zero();
    for (s, operator) in terms.operators.iter().enumerate() {
        for (k, point) in c.points.iter().enumerate() {
            let d = matrices[s * nk + k].matrix(&c.context)?;
            out[s * nk + k] = operator
                * point.projections.adjoint()
                * &*d
                * C64::new(-screening.fraction * point.weight, 0.);
        }
    }
    Ok((out, terms.energy))
}
