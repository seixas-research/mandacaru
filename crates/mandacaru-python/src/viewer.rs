//! The 3D viewer's geometry (`math::isosurface`) and the values it draws from (`math::viewer`):
//! isosurface triangles, the enclosing level, the sampled cloud, the weights, shading, bonds,
//! extent and camera path. Validated copies in, owned arrays out; no numerics here.
use super::{array, error, layout};
use mandacaru_core::math::{isosurface, viewer};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
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

/// The rows of an `(n, width)` array.
fn rows<const W: usize>(input: &PyReadonlyArray2<'_, f64>, what: &str) -> PyResult<Vec<[f64; W]>> {
    layout(input)?;
    if input.shape()[1] != W {
        return Err(PyValueError::new_err(format!("{what} must be (n, {W})")));
    }
    Ok(input.as_slice()?.chunks(W).map(|r| std::array::from_fn(|i| r[i])).collect())
}

fn flat<const W: usize>(rows: Vec<[f64; W]>) -> Vec<f64> {
    rows.into_iter().flatten().collect()
}

/// `|f|^power` of every value of a field (`viewer::weights`).
#[pyfunction]
fn field_weights<'py>(py: Python<'py>, values: PyReadonlyArray1<'_, f64>, power: u32) -> PyResult<Bound<'py, numpy::PyArrayDyn<f64>>> {
    let out = viewer::weights(&owned(&values)?, power).map_err(error)?;
    let n = out.len();
    array(py, &[n], out)
}

/// The isovalue enclosing `fraction` of the weight of a field (`viewer::isovalue`).
#[pyfunction]
fn field_isovalue(py: Python<'_>, values: PyReadonlyArray1<'_, f64>, power: u32, fraction: f64) -> PyResult<f64> {
    let values = owned(&values)?;
    py.detach(|| viewer::isovalue(&values, power, fraction)).map_err(error)
}

/// `(triangles, signs)` of the lobes `f = level` and, when `signed`, `-f = level` (`viewer::signed_surface`).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn signed_isosurface<'py>(
    py: Python<'py>,
    shape: (usize, usize, usize),
    values: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    z: PyReadonlyArray1<'_, f64>,
    level: f64,
    signed: bool,
) -> PyResult<(Bound<'py, numpy::PyArrayDyn<f64>>, Bound<'py, numpy::PyArrayDyn<i64>>)> {
    let (values, x, y, z) = (owned(&values)?, owned(&x)?, owned(&y)?, owned(&z)?);
    let (triangles, signs) = py
        .detach(|| viewer::signed_surface([shape.0, shape.1, shape.2], &values, [&x, &y, &z], level, signed))
        .map_err(error)?;
    let count = triangles.len();
    Ok((
        array(py, &[count, 3, 3], triangles.into_iter().flatten().flatten().collect())?,
        array(py, &[count], signs)?,
    ))
}

/// `(brightness, positive)` of a cloud's field values (`viewer::cloud_shading`).
#[pyfunction]
fn cloud_shading<'py>(py: Python<'py>, values: PyReadonlyArray1<'_, f64>, power: u32) -> PyResult<(Bound<'py, numpy::PyArrayDyn<f64>>, Bound<'py, numpy::PyArrayDyn<bool>>)> {
    let (brightness, positive) = viewer::cloud_shading(&owned(&values)?, power).map_err(error)?;
    let n = brightness.len();
    Ok((array(py, &[n], brightness)?, array(py, &[n], positive)?))
}

fn triangles(input: &PyReadonlyArray3<'_, f64>) -> PyResult<Vec<isosurface::Triangle>> {
    layout(input)?;
    if input.shape()[1..] != [3, 3] {
        return Err(PyValueError::new_err("triangles must be (n, 3, 3)"));
    }
    Ok(input.as_slice()?.chunks(9).map(|t| std::array::from_fn(|v| [t[3 * v], t[3 * v + 1], t[3 * v + 2]])).collect())
}

/// The unit normal of every triangle, `(n, 3)` (`viewer::surface_normals`).
#[pyfunction]
fn surface_normals<'py>(py: Python<'py>, triangles_in: PyReadonlyArray3<'_, f64>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f64>>> {
    let normals = viewer::surface_normals(&triangles(&triangles_in)?);
    let n = normals.len();
    array(py, &[n, 3], flat(normals))
}

/// The flat color of every triangle by the sign of its lobe, `(n, 3)` (`viewer::surface_colors`).
#[pyfunction]
fn surface_colors<'py>(py: Python<'py>, signs: PyReadonlyArray1<'_, i64>, positive: [f64; 3], negative: [f64; 3]) -> PyResult<Bound<'py, numpy::PyArrayDyn<f64>>> {
    layout(&signs)?;
    let colors = viewer::surface_colors(signs.as_slice()?, positive, negative);
    let n = colors.len();
    array(py, &[n, 3], flat(colors))
}

/// The `(n, 4)` face colors of a lit surface (`viewer::surface_rgba`).
#[pyfunction]
fn surface_rgba<'py>(py: Python<'py>, normals: PyReadonlyArray2<'_, f64>, base: PyReadonlyArray2<'_, f64>, elevation: f64, azimuth: f64, alpha: f64) -> PyResult<Bound<'py, numpy::PyArrayDyn<f64>>> {
    let out = viewer::surface_rgba(&rows::<3>(&normals, "normals")?, &rows::<3>(&base, "base colors")?, elevation, azimuth, alpha).map_err(error)?;
    let n = out.len();
    array(py, &[n, 4], flat(out))
}

/// The bonded pairs of nuclei `(n, 3)` with covalent radii (`viewer::bonded_pairs`).
#[pyfunction]
fn bonded_pairs(nuclei: PyReadonlyArray2<'_, f64>, radii: PyReadonlyArray1<'_, f64>) -> PyResult<Vec<(usize, usize)>> {
    viewer::bonded_pairs(&rows::<3>(&nuclei, "nuclei")?, &owned(&radii)?).map_err(error)
}

/// The marker area of every nucleus (`viewer::marker_sizes`).
#[pyfunction]
fn marker_sizes<'py>(py: Python<'py>, radii: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f64>>> {
    let out = viewer::marker_sizes(&owned(&radii)?).map_err(error)?;
    let n = out.len();
    array(py, &[n], out)
}

/// `points.mean(axis=0)` of `(n, 3)` points (`viewer::mean_point`).
#[pyfunction]
fn mean_point(points: PyReadonlyArray2<'_, f64>) -> PyResult<[f64; 3]> {
    viewer::mean_point(&rows::<3>(&points, "points")?).map_err(error)
}

/// The half-width of the view at zoom one (`viewer::view_extent`).
#[pyfunction]
fn view_extent(py: Python<'_>, nuclei: PyReadonlyArray2<'_, f64>, center: [f64; 3], shown: PyReadonlyArray2<'_, f64>, percentile: f64) -> PyResult<f64> {
    let (nuclei, shown) = (rows::<3>(&nuclei, "nuclei")?, rows::<3>(&shown, "shown points")?);
    py.detach(|| viewer::view_extent(&nuclei, center, &shown, percentile)).map_err(error)
}

/// The axis limits `[[lo, hi]; 3]` of the view at `zoom` (`viewer::view_limits`).
#[pyfunction]
fn view_limits(center: [f64; 3], extent: f64, zoom: f64) -> [[f64; 2]; 3] {
    viewer::view_limits(center, extent, zoom)
}

/// The delay of one animation frame in milliseconds (`viewer::frame_duration`).
#[pyfunction]
fn frame_duration(fps: f64) -> PyResult<u64> {
    viewer::frame_duration(fps).map_err(error)
}

/// `(frames, 6)` rows `(elevation, azimuth, zoom, fx, fy, fz)` of an animation's camera path from
/// `(k, 6)` keyframes (`viewer::camera_path`); `rotate` is `"azimuth"`, `"elevation"` or `None`.
#[pyfunction]
#[pyo3(signature = (poses, frames, zoom=None, rotate=None, turns=1.0))]
fn camera_path<'py>(
    py: Python<'py>,
    poses: PyReadonlyArray2<'_, f64>,
    frames: usize,
    zoom: Option<PyReadonlyArray1<'_, f64>>,
    rotate: Option<&str>,
    turns: f64,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f64>>> {
    let poses: Vec<viewer::Pose> = rows::<6>(&poses, "poses")?
        .into_iter()
        .map(|p| viewer::Pose { elevation: p[0], azimuth: p[1], zoom: p[2], focus: [p[3], p[4], p[5]] })
        .collect();
    let zoom = zoom.as_ref().map(owned).transpose()?;
    let rotation = match rotate {
        None => viewer::Rotation::None,
        Some("azimuth") => viewer::Rotation::Azimuth,
        Some("elevation") => viewer::Rotation::Elevation,
        Some(other) => return Err(PyValueError::new_err(format!("rotate must be 'azimuth', 'elevation' or None, got {other:?}"))),
    };
    let path = viewer::camera_path(&poses, frames, zoom.as_deref(), rotation, turns).map_err(error)?;
    let rows: Vec<[f64; 6]> = path
        .into_iter()
        .map(|p| [p.elevation, p.azimuth, p.zoom, p.focus[0], p.focus[1], p.focus[2]])
        .collect();
    let n = rows.len();
    array(py, &[n, 6], flat(rows))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(field_weights, m)?)?;
    m.add_function(wrap_pyfunction!(field_isovalue, m)?)?;
    m.add_function(wrap_pyfunction!(signed_isosurface, m)?)?;
    m.add_function(wrap_pyfunction!(cloud_shading, m)?)?;
    m.add_function(wrap_pyfunction!(surface_normals, m)?)?;
    m.add_function(wrap_pyfunction!(surface_colors, m)?)?;
    m.add_function(wrap_pyfunction!(surface_rgba, m)?)?;
    m.add_function(wrap_pyfunction!(bonded_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(marker_sizes, m)?)?;
    m.add_function(wrap_pyfunction!(mean_point, m)?)?;
    m.add_function(wrap_pyfunction!(view_extent, m)?)?;
    m.add_function(wrap_pyfunction!(camera_path, m)?)?;
    m.add_function(wrap_pyfunction!(view_limits, m)?)?;
    m.add_function(wrap_pyfunction!(frame_duration, m)?)?;
    m.add_function(wrap_pyfunction!(isosurface_triangles, m)?)?;
    m.add_function(wrap_pyfunction!(enclosing_level, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_cloud, m)?)?;
    Ok(())
}
