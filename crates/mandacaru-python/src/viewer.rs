//! The 3D viewer's geometry (`math::isosurface`): isosurface triangles, the enclosing level and
//! the sampled cloud. Validated copies in, owned arrays out; no numerics here.
use super::{array, error, layout};
use mandacaru_core::math::isosurface;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};

fn owned(input: &PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    layout(input)?;
    Ok(input.as_slice()?.to_vec())
}

/// `(triangles, 3, 3)`: the surface `values == level` on a structured grid of `shape` (row-major
/// `values`, the Cartesian coordinates `x`, `y`, `z` of every node).
#[pyfunction]
fn isosurface_triangles<'py>(
    py: Python<'py>,
    shape: (usize, usize, usize),
    values: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    z: PyReadonlyArray1<'_, f64>,
    level: f64,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f64>>> {
    let (values, x, y, z) = (owned(&values)?, owned(&x)?, owned(&y)?, owned(&z)?);
    let triangles = py
        .detach(|| isosurface::marching_tetrahedra([shape.0, shape.1, shape.2], &values, [&x, &y, &z], level))
        .map_err(error)?;
    let count = triangles.len();
    array(py, &[count, 3, 3], triangles.into_iter().flatten().flatten().collect())
}

/// The smallest weight `w0` such that the weights `>= w0` hold `fraction` of the total.
#[pyfunction]
fn enclosing_level(weights: PyReadonlyArray1<'_, f64>, fraction: f64) -> PyResult<f64> {
    isosurface::enclosing_level(&owned(&weights)?, fraction).map_err(error)
}

/// `(picked, positions)`: `count` nodes drawn with probability proportional to `weights` and the
/// points placed uniformly inside their cells (`nodes` `(n, 3)`, `step` `(3, 3)` with one axis
/// vector per column), from the stream of `seed`.
#[pyfunction]
fn weighted_cloud<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<'_, f64>,
    nodes: PyReadonlyArray2<'_, f64>,
    step: PyReadonlyArray2<'_, f64>,
    count: usize,
    seed: u64,
) -> PyResult<(Bound<'py, numpy::PyArrayDyn<u64>>, Bound<'py, numpy::PyArrayDyn<f64>>)> {
    let weights = owned(&weights)?;
    layout(&nodes)?;
    layout(&step)?;
    if nodes.shape() != [weights.len(), 3] || step.shape() != [3, 3] {
        return Err(PyValueError::new_err("a cloud needs (n, 3) nodes for n weights and a (3, 3) step"));
    }
    let rows: Vec<[f64; 3]> = nodes.as_slice()?.chunks(3).map(|r| [r[0], r[1], r[2]]).collect();
    let s = step.as_slice()?;
    let step: [[f64; 3]; 3] = std::array::from_fn(|b| [s[3 * b], s[3 * b + 1], s[3 * b + 2]]);
    let (picked, points) = py
        .detach(|| isosurface::weighted_cloud(&weights, &rows, &step, count, seed))
        .map_err(error)?;
    Ok((
        array(py, &[count], picked.into_iter().map(|p| p as u64).collect())?,
        array(py, &[count, 3], points.into_iter().flatten().collect())?,
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(isosurface_triangles, m)?)?;
    m.add_function(wrap_pyfunction!(enclosing_level, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_cloud, m)?)?;
    Ok(())
}
