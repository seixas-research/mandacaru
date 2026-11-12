//! Bindings of unit conversion (`mandacaru_core::units::to_user`) and of the values the run log
//! and the ASE results derive (`mandacaru_core::derived`). Python formats what these return.
use crate::arrays::error;
use mandacaru_core::{derived, units};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyDict};

/// `values` (a float or a float64 array, atomic units) in user units: `quantity` is `energy`,
/// `length`, `force`, `gradient_to_force`, `stress`, `area` or `inverse_energy`; `atomic_units` keeps atomic units; `cells`
/// divides a supercell value per primitive cell; `plus` (user units, the shape of `values`) is
/// added after the conversion; `total` sums `values` first and returns one float
/// (`units::to_user_values`, `units::to_user_total`).
#[pyfunction]
#[pyo3(signature=(values, quantity, atomic_units=false, cells=1, total=false, plus=None))]
pub fn to_user_units<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    quantity: &str,
    atomic_units: bool,
    cells: u64,
    total: bool,
    plus: Option<PyReadonlyArrayDyn<'py, f64>>,
) -> PyResult<Bound<'py, PyAny>> {
    let quantity = units::Quantity::parse(quantity).map_err(error)?;
    let plus: Option<Vec<f64>> = plus.map(|p| p.as_array().iter().copied().collect());
    if let Ok(value) = values.extract::<f64>() {
        let out = units::to_user_values(&[value], quantity, atomic_units, cells, plus.as_deref())
            .map_err(error)?;
        return Ok(out[0].into_pyobject(py)?.into_any());
    }
    let array: PyReadonlyArrayDyn<'py, f64> = values.extract()?;
    let data = array.as_array().iter().copied().collect::<Vec<_>>();
    if total {
        let sum = units::to_user_total(&data, quantity, atomic_units, cells).map_err(error)?;
        return Ok(sum.into_pyobject(py)?.into_any());
    }
    let shape = array.shape().to_vec();
    let out = units::to_user_values(&data, quantity, atomic_units, cells, plus.as_deref()).map_err(error)?;
    let result: Bound<'py, PyArrayDyn<f64>> = out.into_pyarray(py).reshape(shape)?;
    Ok(result.into_any())
}

/// `values - values[0]` (`derived::relative_to_first`).
#[pyfunction]
pub fn relative_to_first<'py>(py: Python<'py>, values: Vec<f64>) -> Bound<'py, numpy::PyArray1<f64>> {
    derived::relative_to_first(&values).into_pyarray(py)
}

/// `(a - b)` in user units (`derived::difference_to_user`).
#[pyfunction]
#[pyo3(signature=(a, b, quantity, atomic_units=false))]
pub fn difference_to_user(a: f64, b: f64, quantity: &str, atomic_units: bool) -> PyResult<f64> {
    derived::difference_to_user(a, b, units::Quantity::parse(quantity).map_err(error)?, atomic_units).map_err(error)
}

/// The electrons of each spin of a crystal from its `(spins, kpoints, bands)` occupations and
/// k-point weights (`derived::electron_counts`).
#[pyfunction]
pub fn electron_counts(occupations: numpy::PyReadonlyArray3<'_, f64>, weights: Vec<f64>) -> PyResult<(f64, f64)> {
    let shape = occupations.shape().to_vec();
    let data: Vec<f64> = occupations.as_array().iter().copied().collect();
    let [a, b] = derived::electron_counts(&data, shape[0], shape[1], shape[2], &weights).map_err(error)?;
    Ok((a, b))
}

/// `{"norms", "max", "rms", "net"}` of `(atoms, 3)` forces (`derived::force_summary`).
#[pyfunction]
pub fn force_summary<'py>(py: Python<'py>, forces: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyDict>> {
    let array = forces.as_array();
    if array.ncols() != 3 {
        return Err(error(mandacaru_core::Error::Invalid("forces must be (atoms, 3)".into())));
    }
    let rows: Vec<[f64; 3]> = array.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
    let summary = derived::force_summary(&rows).map_err(error)?;
    let out = PyDict::new(py);
    out.set_item("norms", summary.norms.into_pyarray(py))?;
    out.set_item("max", summary.max)?;
    out.set_item("rms", summary.rms)?;
    out.set_item("net", summary.net)?;
    Ok(out)
}

/// `{"delta", "gradient_ev", "gradient_met", "energy_met"}` of one ADAPT iteration row
/// (`derived::iteration_values`).
#[pyfunction]
#[pyo3(signature=(energy, previous, gradient_ha, gradient_threshold=None, energy_threshold=None))]
pub fn iteration_values<'py>(
    py: Python<'py>,
    energy: f64,
    previous: Option<f64>,
    gradient_ha: f64,
    gradient_threshold: Option<f64>,
    energy_threshold: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let row = derived::iteration_values(energy, previous, gradient_ha, gradient_threshold, energy_threshold)
        .map_err(error)?;
    let out = PyDict::new(py);
    out.set_item("delta", row.delta)?;
    out.set_item("gradient_ev", row.gradient_ev)?;
    out.set_item("gradient_met", row.gradient_met)?;
    out.set_item("energy_met", row.energy_met)?;
    Ok(out)
}

/// `(a, b, c, alpha, beta, gamma)` of a cell given by its row vectors (`derived::cell_parameters`).
#[pyfunction]
pub fn cell_parameters(cell: [[f64; 3]; 3]) -> PyResult<[f64; 6]> {
    derived::cell_parameters(cell).map_err(error)
}

/// Apply `conversion` to a float, a complex, or a float64 / complex128 array (a complex value
/// part by part), returning the same kind.
fn convert_any<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    conversion: units::EnergyConversion,
) -> PyResult<Bound<'py, PyAny>> {
    use mandacaru_core::quantum::C64;
    if let Ok(value) = values.extract::<f64>() {
        return Ok(conversion.apply(value).into_pyobject(py)?.into_any());
    }
    if let Ok(value) = values.cast::<pyo3::types::PyComplex>() {
        return Ok(pyo3::types::PyComplex::from_doubles(
            py,
            conversion.apply(value.real()),
            conversion.apply(value.imag()),
        )
        .into_any());
    }
    if let Ok(array) = values.extract::<PyReadonlyArrayDyn<'py, f64>>() {
        let shape = array.shape().to_vec();
        let out: Vec<f64> = array.as_array().iter().map(|&v| conversion.apply(v)).collect();
        let result: Bound<'py, PyArrayDyn<f64>> = out.into_pyarray(py).reshape(shape)?;
        return Ok(result.into_any());
    }
    let array: PyReadonlyArrayDyn<'py, C64> = values.extract()?;
    let shape = array.shape().to_vec();
    let out: Vec<C64> =
        array.as_array().iter().map(|v| C64::new(conversion.apply(v.re), conversion.apply(v.im))).collect();
    let result: Bound<'py, PyArrayDyn<C64>> = out.into_pyarray(py).reshape(shape)?;
    Ok(result.into_any())
}

/// An energy in Hartree (float, complex or array) in `unit` (`units::energy_factor`).
#[pyfunction]
pub fn energy_from_hartree<'py>(py: Python<'py>, values: &Bound<'py, PyAny>, unit: &str) -> PyResult<Bound<'py, PyAny>> {
    convert_any(py, values, units::EnergyConversion::FromHartree(units::energy_factor(unit).map_err(error)?))
}
/// An energy in `unit` (float, complex or array) in Hartree.
#[pyfunction]
pub fn energy_to_hartree<'py>(py: Python<'py>, values: &Bound<'py, PyAny>, unit: &str) -> PyResult<Bound<'py, PyAny>> {
    convert_any(py, values, units::EnergyConversion::ToHartree(units::energy_factor(unit).map_err(error)?))
}
/// An energy from `from_unit` to `to_unit` (the value itself when the units agree).
#[pyfunction]
pub fn convert_energy<'py>(py: Python<'py>, values: &Bound<'py, PyAny>, from_unit: &str, to_unit: &str) -> PyResult<Bound<'py, PyAny>> {
    convert_any(py, values, units::EnergyConversion::between(from_unit, to_unit).map_err(error)?)
}
/// A length in `unit` (float or array) in Bohr (`units::length_factor_to_bohr`).
#[pyfunction]
pub fn length_to_bohr<'py>(py: Python<'py>, values: &Bound<'py, PyAny>, unit: &str) -> PyResult<Bound<'py, PyAny>> {
    convert_any(py, values, units::EnergyConversion::FromHartree(units::length_factor_to_bohr(unit).map_err(error)?))
}
/// The Hartree-to-`unit` factor of an accepted energy spelling (`units::energy_factor`).
#[pyfunction]
pub fn energy_unit_factor(unit: &str) -> PyResult<f64> {
    units::energy_factor(unit).map_err(error)
}

/// The sum of `values` as Python's `sum` computes it (`derived::total`).
#[pyfunction]
pub fn total(values: Vec<f64>) -> f64 {
    derived::total(&values)
}
/// The fractional points of a Gamma-centered mesh, `k` fastest (`derived::mesh_fractions`).
#[pyfunction]
pub fn mesh_fractions(shape: [usize; 3]) -> PyResult<Vec<[f64; 3]>> {
    derived::mesh_fractions(shape).map_err(error)
}
/// The Euclidean norm of `vector` (`derived::norm`).
#[pyfunction]
pub fn vector_norm(vector: Vec<f64>) -> f64 {
    derived::norm(&vector)
}
/// The distance between two points (`derived::distance`).
#[pyfunction]
pub fn distance(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    derived::distance(&a, &b).map_err(error)
}

/// `values[i + 1] - values[i]` (`derived::gaps`).
#[pyfunction]
pub fn gaps<'py>(py: Python<'py>, values: Vec<f64>) -> Bound<'py, numpy::PyArray1<f64>> {
    derived::gaps(&values).into_pyarray(py)
}

/// `n` weights `1 / n` (`derived::uniform_weights`).
#[pyfunction]
pub fn uniform_weights<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    Ok(derived::uniform_weights(n).map_err(error)?.into_pyarray(py))
}
/// The highest occupied level (Hartree) of `(spins, bands)` levels (`derived::highest_occupied`).
#[pyfunction]
pub fn highest_occupied(levels: PyReadonlyArray2<'_, f64>, particles: Vec<usize>) -> PyResult<f64> {
    let bands = levels.shape()[1];
    let data: Vec<f64> = levels.as_array().iter().copied().collect();
    derived::highest_occupied(&data, bands, &particles).map_err(error)
}
/// An even energy axis (eV) over the levels (Hartree) widened by `margin` eV (`derived::energy_axis`).
#[pyfunction]
#[pyo3(signature=(levels_ha, npoints, margin, scale=1.0))]
pub fn energy_axis<'py>(py: Python<'py>, levels_ha: PyReadonlyArrayDyn<'py, f64>, npoints: usize, margin: f64, scale: f64) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let data: Vec<f64> = levels_ha.as_array().iter().copied().collect();
    Ok(derived::energy_axis(&data, npoints, margin, scale).map_err(error)?.into_pyarray(py))
}
/// `(labels, sums)`: the columns of `(rows, labels)` values summed per label (`derived::column_groups`).
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn column_groups<'py>(py: Python<'py>, values: PyReadonlyArray2<'py, f64>, labels: Vec<String>) -> PyResult<(Vec<String>, Vec<Bound<'py, numpy::PyArray1<f64>>>)> {
    let rows = values.shape()[0];
    let data: Vec<f64> = values.as_array().iter().copied().collect();
    let (names, sums) = derived::column_groups(&data, rows, &labels).map_err(error)?;
    Ok((names, sums.into_iter().map(|v| v.into_pyarray(py)).collect()))
}
/// `positions` with atom `index` moved by `sign * step` along `axis` (`derived::displaced`).
#[pyfunction]
pub fn displaced<'py>(py: Python<'py>, positions: PyReadonlyArray2<'py, f64>, index: usize, axis: usize, sign: f64, step: f64) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let rows: Vec<[f64; 3]> = positions.as_array().rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
    let n = rows.len();
    let out = derived::displaced(rows, index, axis, sign, step).map_err(error)?;
    let flat: Vec<f64> = out.into_iter().flatten().collect();
    Ok(flat.into_pyarray(py).reshape([n, 3])?)
}

/// `a / b` (`derived::ratio`).
#[pyfunction]
pub fn ratio(a: f64, b: f64) -> PyResult<f64> {
    derived::ratio(a, b).map_err(error)
}
/// `(voxel volume, cell, cell volume)` of a grid (`derived::grid_cell`).
#[pyfunction]
pub fn grid_cell<'py>(py: Python<'py>, step: [[f64; 3]; 3], shape: [usize; 3]) -> PyResult<(f64, Bound<'py, numpy::PyArray2<f64>>, f64)> {
    let (dv, cell, volume) = derived::grid_cell(step, shape).map_err(error)?;
    let flat: Vec<f64> = cell.iter().flatten().copied().collect();
    Ok((dv, flat.into_pyarray(py).reshape([3, 3])?, volume))
}
/// The points of a grid, `(3, nx, ny, nz)` (`derived::grid_coordinates`).
#[pyfunction]
pub fn grid_coordinates<'py>(py: Python<'py>, origin: [f64; 3], step: [[f64; 3]; 3], shape: [usize; 3]) -> PyResult<Bound<'py, numpy::PyArray4<f64>>> {
    Ok(derived::grid_coordinates(origin, step, shape).into_pyarray(py).reshape([3, shape[0], shape[1], shape[2]])?)
}
/// The decimal digits a tolerance resolves (`derived::decimal_digits`).
#[pyfunction]
pub fn decimal_digits(atol: f64) -> PyResult<u32> {
    derived::decimal_digits(atol).map_err(error)
}

/// `values` rounded to `decimals` places, ties to even (`derived::rounded`).
#[pyfunction]
pub fn rounded(values: Vec<f64>, decimals: i32) -> Vec<f64> {
    derived::rounded(&values, decimals)
}

/// `sqrt(2 E)` in 1/Bohr for `E` in eV (`derived::free_wavevector`).
#[pyfunction]
pub fn free_wavevector(energy_ev: f64) -> PyResult<f64> {
    derived::free_wavevector(energy_ev).map_err(error)
}

/// `k^2 / 2` in eV for `k` in 1/Bohr (`derived::free_energy_ev`).
#[pyfunction]
pub fn free_energy_ev(k: f64) -> PyResult<f64> {
    derived::free_energy_ev(k).map_err(error)
}

/// The confinement radius (Bohr) of an energy shift in eV (`derived::energy_shift_radius`).
#[pyfunction]
pub fn energy_shift_radius(shift_ev: f64) -> PyResult<f64> {
    derived::energy_shift_radius(shift_ev).map_err(error)
}

/// `fraction * pi / spacing` (1/Bohr) (`derived::nyquist_wavevector`).
#[pyfunction]
pub fn nyquist_wavevector(fraction: f64, spacing_bohr: f64) -> PyResult<f64> {
    derived::nyquist_wavevector(fraction, spacing_bohr).map_err(error)
}

/// The weights `n, n - 1, ..., 1` (`derived::descending_weights`).
#[pyfunction]
pub fn descending_weights<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    Ok(derived::descending_weights(n).map_err(error)?.into_pyarray(py))
}

/// `max |v|` of an array of any shape, zero when empty (`derived::max_abs`).
#[pyfunction]
pub fn max_abs(values: PyReadonlyArrayDyn<'_, f64>) -> f64 {
    derived::max_abs(&values.as_array().iter().copied().collect::<Vec<_>>())
}

/// `np.min(values)` of a nonempty array (`numpy_rules::minimum`).
#[pyfunction]
pub fn array_minimum(values: PyReadonlyArrayDyn<'_, f64>) -> PyResult<f64> {
    let data: Vec<f64> = values.as_array().iter().copied().collect();
    mandacaru_core::math::numpy_rules::minimum(&data)
        .ok_or_else(|| error(mandacaru_core::Error::Invalid("the minimum of no values".into())))
}

/// `np.argmax(values)` of a nonempty one-dimensional array (`numpy_rules::argmax`).
#[pyfunction]
pub fn array_argmax(values: PyReadonlyArrayDyn<'_, f64>) -> PyResult<usize> {
    let data: Vec<f64> = values.as_array().iter().copied().collect();
    mandacaru_core::math::numpy_rules::argmax(&data)
        .ok_or_else(|| error(mandacaru_core::Error::Invalid("the argmax of no values".into())))
}

/// `i dt` for `i` in `0..n` (`derived::sample_times`).
#[pyfunction]
pub fn sample_times<'py>(py: Python<'py>, n: usize, dt: f64) -> Bound<'py, numpy::PyArray1<f64>> {
    derived::sample_times(n, dt).into_pyarray(py)
}

/// `2 pi / (n dt)` (`derived::fourier_bin_spacing`).
#[pyfunction]
pub fn fourier_bin_spacing(samples: usize, dt: f64) -> f64 {
    derived::fourier_bin_spacing(samples, dt)
}

/// `pi / dt` (`derived::nyquist_frequency`).
#[pyfunction]
pub fn nyquist_frequency(dt: f64) -> f64 {
    derived::nyquist_frequency(dt)
}

/// `x ** 2` through the C library's `pow`, as Python's float power (`derived::squared`).
#[pyfunction]
pub fn squared(x: f64) -> f64 {
    derived::squared(x)
}

/// `abs(complex(re, im)) ** 2` (`derived::modulus_squared`).
#[pyfunction]
pub fn modulus_squared(re: f64, im: f64) -> f64 {
    derived::modulus_squared(re, im)
}

/// `|tau| * norm_bound` (`derived::pulse_bound`).
#[pyfunction]
pub fn pulse_bound(tau: f64, norm_bound: f64) -> f64 {
    derived::pulse_bound(tau, norm_bound)
}

/// `numerators / denominators`, element by element (`derived::quotients`).
#[pyfunction]
pub fn quotients<'py>(
    py: Python<'py>,
    numerators: numpy::PyReadonlyArray1<'py, f64>,
    denominators: numpy::PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let out = derived::quotients(numerators.as_slice()?, denominators.as_slice()?).map_err(error)?;
    Ok(out.into_pyarray(py))
}

/// `round(sum(values))` as Python computes it (`derived::rounded_total`).
#[pyfunction]
pub fn rounded_total(values: Vec<f64>) -> i64 {
    derived::rounded_total(&values)
}
