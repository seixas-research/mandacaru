//! The one-center terms' response to the projections: `E_oc` is a function of the projected
//! density `P_s = sum_k w_k C_k^+ D_s(k) C_k` alone (the sphere model is rigid with its atom),
//! so `dE_oc = sum_s tr[(-a O_s) dP_s]` and, per k, `Z = -a w_k O_s C_k^+ D_s(k)` (`p x m`): the
//! engine's `H_C C^+ D` with `H_C = -a O`. `O_s` is what the model returns for `P`; nothing of
//! the model is re-derived here.
//!
//! A spinor crystal has one `2p x 2p` operator `O` (blocks `O_st`, the full spin-structure-aware
//! sphere exchange, the cross blocks included) and `2m x 2m` density matrices `D_st`. The
//! projections act identically on both components, `P_st = sum_k w_k C^+ D_st C`, so
//! `dE_oc = -a sum_{s,t} tr(O_ts dP_st)` and `Z = -a w_k sum_{s,t} O_st C^+ D_ts` (`p x m`).
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
    // The matrices this response holds outside the `Context`'s buffers: `Z` per (spin, k) and one
    // scratch, the projected density and the operator of the model (`2p x 2p` for a spinor).
    let _held = c.context.reserve(crate::product(&[
        16,
        2 * c.spins * nk * c.p * c.m + 16 * c.p * c.p,
    ])?)?;
    let zero = || vec![Matrix::zeros(c.p, c.m); c.spins * nk];
    let Some(model) = augmentation.one_center() else {
        return Ok((zero(), 0.));
    };
    let terms = one_center(c, matrices, model)?;
    if terms.operators.len() != if c.spinor { 1 } else { c.spins } {
        return Err(invalid(
            "the one-center model returned the wrong number of operators",
        ));
    }
    let mut out = zero();
    if c.spinor {
        let (m, p) = (c.m, c.p);
        let operator = &terms.operators[0];
        if operator.shape() != (2 * p, 2 * p) {
            return Err(invalid("the spinor one-center operator must be 2p x 2p"));
        }
        for (k, point) in c.points.iter().enumerate() {
            let d = matrices[k].matrix(&c.context)?;
            let mut z = Matrix::zeros(p, m);
            for s in 0..2 {
                for t in 0..2 {
                    z += operator.view((s * p, t * p), (p, p))
                        * point.projections.adjoint()
                        * d.view((t * m, s * m), (m, m));
                }
            }
            out[k] = z * C64::new(-screening.fraction * point.weight, 0.);
        }
        return Ok((out, terms.energy));
    }
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
