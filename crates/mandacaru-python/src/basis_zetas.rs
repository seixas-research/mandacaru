//! Binding of the split-valence multiple-zeta construction of the PAW-LCAO
//! basis. Arrays are copied in and returned owned; the numerics are in
//! `mandacaru_core::basis::radial::zeta_values`.
use super::{array, copy, error, Array, PyContext};
use mandacaru_core::basis::radial;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// The radial functions of `count` zetas on the grid `r`: the first-zeta
/// `values` followed by up to `count - 1` split-valence refinements (fewer when
/// the split leaves nothing). `tail_norms` selects the first-zeta scheme; without
/// it each zeta splits the previous one at `split_norm` of its squared norm.
/// With `fractions` the tails are the squared-norm fractions left outside the
/// split radius (the polarization convention) instead of the norms.
#[pyfunction]
#[pyo3(signature = (context, r, values, l, count, split_norm, tail_norms=None, fractions=false))]
#[allow(clippy::too_many_arguments)]
fn radial_zetas<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    l: usize,
    count: usize,
    split_norm: f64,
    tail_norms: Option<PyReadonlyArray1<'py, f64>>,
    fractions: bool,
) -> PyResult<Vec<Array<'py, f64>>> {
    let ctx = context.inner.clone();
    let (r, _a) = copy(&ctx, &r)?;
    let (values, _b) = copy(&ctx, &values)?;
    let tails = tail_norms.as_ref().map(|v| copy(&ctx, v)).transpose()?;
    let zetas = py
        .detach(|| {
            let split = if fractions {
                radial::zeta_values_by_fraction
            } else {
                radial::zeta_values
            };
            split(
                &ctx,
                &r,
                &values,
                l,
                count,
                split_norm,
                tails.as_ref().map(|(v, _)| v.as_slice()),
            )
        })
        .map_err(error)?;
    zetas
        .into_iter()
        .map(|table| {
            let (data, _held) = table.into_parts();
            array(py, &[data.len()], data)
        })
        .collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(radial_zetas, m)?)?;
    Ok(())
}
