//! Bindings of the PAW-LCAO basis construction: confined orbitals, the
//! polarization shells, the bound state of an empty channel and the diffuse
//! shells. Arrays are copied in and returned owned; the numerics are in
//! `mandacaru_core::basis::{confinement, pao, diffuse}`.
use super::{array, copy, error, Array, PyContext};
use mandacaru_core::{
    basis::{confinement::Potential, diffuse, pao as radial},
    paw::{
        confinement::{self, ChannelData, Prepared, Radius},
        pao,
    },
    Reservation,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};

/// A PAW channel's arrays, copied out of Python and accounted in the context.
struct Owned {
    table: Vec<f64>,
    local: Vec<f64>,
    projectors: Vec<f64>,
    rank: usize,
    coupling: Vec<f64>,
    overlap: Vec<f64>,
    wave: Vec<f64>,
    l: usize,
    r_cut: f64,
    _held: Vec<Reservation>,
}
impl Owned {
    #[allow(clippy::too_many_arguments)]
    fn take(
        context: &PyContext,
        table: &PyReadonlyArray1<'_, f64>,
        local: &PyReadonlyArray1<'_, f64>,
        projectors: &PyReadonlyArray2<'_, f64>,
        coupling: &PyReadonlyArray2<'_, f64>,
        overlap: &PyReadonlyArray2<'_, f64>,
        wave: &PyReadonlyArray1<'_, f64>,
        l: usize,
        r_cut: f64,
    ) -> PyResult<Self> {
        let ctx = &context.inner;
        let rank = projectors.shape()[0];
        let (table, a) = copy(ctx, table)?;
        let (local, b) = copy(ctx, local)?;
        let (projectors, c) = copy(ctx, projectors)?;
        let (coupling, d) = copy(ctx, coupling)?;
        let (overlap, e) = copy(ctx, overlap)?;
        let (wave, f) = copy(ctx, wave)?;
        Ok(Self {
            table,
            local,
            projectors,
            rank,
            coupling,
            overlap,
            wave,
            l,
            r_cut,
            _held: vec![a, b, c, d, e, f],
        })
    }
    fn data(&self) -> ChannelData<'_> {
        ChannelData {
            table: &self.table,
            local: &self.local,
            projectors: &self.projectors,
            rank: self.rank,
            coupling: &self.coupling,
            overlap: &self.overlap,
            wave: &self.wave,
            l: self.l,
            r_cut: self.r_cut,
        }
    }
}

fn potential_of(amplitude: f64, inner_fraction: f64) -> Potential {
    Potential {
        amplitude,
        inner_fraction,
    }
}

#[pyfunction]
fn confinement_potential<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    r_c: f64,
    amplitude: f64,
    inner_fraction: f64,
) -> PyResult<Array<'py, f64>> {
    let out = mandacaru_core::basis::confinement::potential(
        r.as_slice()?,
        r_c,
        amplitude,
        inner_fraction,
    );
    array(py, &[out.len()], out)
}

#[pyfunction]
#[pyo3(signature=(context, r, local, projectors, coupling, overlap, wave, l, r_cut, r_c, amplitude, inner_fraction, state=0))]
#[allow(clippy::too_many_arguments)]
fn confinement_energy(
    context: PyRef<'_, PyContext>,
    py: Python<'_>,
    r: PyReadonlyArray1<'_, f64>,
    local: PyReadonlyArray1<'_, f64>,
    projectors: PyReadonlyArray2<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
    overlap: PyReadonlyArray2<'_, f64>,
    wave: PyReadonlyArray1<'_, f64>,
    l: usize,
    r_cut: f64,
    r_c: f64,
    amplitude: f64,
    inner_fraction: f64,
    state: usize,
) -> PyResult<f64> {
    let channel = Owned::take(
        &context,
        &r,
        &local,
        &projectors,
        &coupling,
        &overlap,
        &wave,
        l,
        r_cut,
    )?;
    let ctx = &context.inner;
    py.detach(|| {
        let prepared = Prepared::new(ctx, channel.data())?;
        confinement::energy(
            ctx,
            &prepared,
            r_c,
            &potential_of(amplitude, inner_fraction),
            state,
        )
    })
    .map_err(error)
}

#[pyfunction]
#[pyo3(signature=(context, r, local, projectors, coupling, overlap, wave, l, r_cut, r_c, amplitude, inner_fraction, state=0))]
#[allow(clippy::too_many_arguments)]
fn confinement_eigenstate<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    local: PyReadonlyArray1<'py, f64>,
    projectors: PyReadonlyArray2<'py, f64>,
    coupling: PyReadonlyArray2<'py, f64>,
    overlap: PyReadonlyArray2<'py, f64>,
    wave: PyReadonlyArray1<'py, f64>,
    l: usize,
    r_cut: f64,
    r_c: f64,
    amplitude: f64,
    inner_fraction: f64,
    state: usize,
) -> PyResult<(Array<'py, f64>, Array<'py, f64>, f64)> {
    let channel = Owned::take(
        &context,
        &r,
        &local,
        &projectors,
        &coupling,
        &overlap,
        &wave,
        l,
        r_cut,
    )?;
    let ctx = &context.inner;
    let (nodes, u, energy) = py
        .detach(|| {
            let prepared = Prepared::new(ctx, channel.data())?;
            confinement::eigenstate(
                ctx,
                &prepared,
                r_c,
                &potential_of(amplitude, inner_fraction),
                state,
            )
        })
        .map_err(error)?;
    Ok((
        array(py, &[nodes.len()], nodes)?,
        array(py, &[u.len()], u)?,
        energy,
    ))
}

#[pyfunction]
#[pyo3(signature=(context, r, local, projectors, coupling, overlap, wave, l, r_cut, state=0))]
#[allow(clippy::too_many_arguments)]
fn confinement_free_energy(
    context: PyRef<'_, PyContext>,
    py: Python<'_>,
    r: PyReadonlyArray1<'_, f64>,
    local: PyReadonlyArray1<'_, f64>,
    projectors: PyReadonlyArray2<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
    overlap: PyReadonlyArray2<'_, f64>,
    wave: PyReadonlyArray1<'_, f64>,
    l: usize,
    r_cut: f64,
    state: usize,
) -> PyResult<f64> {
    let channel = Owned::take(
        &context,
        &r,
        &local,
        &projectors,
        &coupling,
        &overlap,
        &wave,
        l,
        r_cut,
    )?;
    let ctx = &context.inner;
    py.detach(|| {
        let prepared = Prepared::new(ctx, channel.data())?;
        confinement::free_energy(ctx, &prepared, state)
    })
    .map_err(error)
}

/// Status codes of the radius search, shared with the Python wrapper:
/// 0 found, 1 the tightest wall (the shift would cut into the sphere),
/// 2 the shift is too large (value: the largest usable, eV), 3 the shift is
/// too small (value: the reach of the radial table, Bohr).
#[pyfunction]
#[pyo3(signature=(context, r, local, projectors, coupling, overlap, wave, l, r_cut, shift_ev, amplitude, inner_fraction, tightest=false, state=0, free_energy=None))]
#[allow(clippy::too_many_arguments)]
fn confinement_radius(
    context: PyRef<'_, PyContext>,
    py: Python<'_>,
    r: PyReadonlyArray1<'_, f64>,
    local: PyReadonlyArray1<'_, f64>,
    projectors: PyReadonlyArray2<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
    overlap: PyReadonlyArray2<'_, f64>,
    wave: PyReadonlyArray1<'_, f64>,
    l: usize,
    r_cut: f64,
    shift_ev: f64,
    amplitude: f64,
    inner_fraction: f64,
    tightest: bool,
    state: usize,
    free_energy: Option<f64>,
) -> PyResult<(u8, f64)> {
    let channel = Owned::take(
        &context,
        &r,
        &local,
        &projectors,
        &coupling,
        &overlap,
        &wave,
        l,
        r_cut,
    )?;
    let ctx = &context.inner;
    let found = py
        .detach(|| {
            let prepared = Prepared::new(ctx, channel.data())?;
            let free = match free_energy {
                Some(value) => value,
                None => confinement::free_energy(ctx, &prepared, state)?,
            };
            confinement::radius(
                ctx,
                &prepared,
                free,
                shift_ev,
                &potential_of(amplitude, inner_fraction),
                tightest,
                state,
            )
        })
        .map_err(error)?;
    Ok(match found {
        Radius::Found(value) => (0, value),
        Radius::Tightest(value) => (1, value),
        Radius::TooLarge { largest_ev } => (2, largest_ev),
        Radius::TooSmall { reach } => (3, reach),
    })
}

/// `(status, r_c or reach, energy, free_energy, R on the table)`; status 0 is
/// an orbital, 3 a shift too small to reach inside the table.
#[pyfunction]
#[pyo3(signature=(context, r, local, projectors, coupling, overlap, wave, l, r_cut, shift_ev, amplitude, inner_fraction, state=0, free_energy=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn confinement_orbital<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    local: PyReadonlyArray1<'py, f64>,
    projectors: PyReadonlyArray2<'py, f64>,
    coupling: PyReadonlyArray2<'py, f64>,
    overlap: PyReadonlyArray2<'py, f64>,
    wave: PyReadonlyArray1<'py, f64>,
    l: usize,
    r_cut: f64,
    shift_ev: f64,
    amplitude: f64,
    inner_fraction: f64,
    state: usize,
    free_energy: Option<f64>,
) -> PyResult<(u8, f64, f64, f64, Array<'py, f64>)> {
    let channel = Owned::take(
        &context,
        &r,
        &local,
        &projectors,
        &coupling,
        &overlap,
        &wave,
        l,
        r_cut,
    )?;
    let ctx = &context.inner;
    let out = py
        .detach(|| {
            let prepared = Prepared::new(ctx, channel.data())?;
            confinement::orbital(
                ctx,
                &prepared,
                shift_ev,
                &potential_of(amplitude, inner_fraction),
                state,
                free_energy,
            )
        })
        .map_err(error)?;
    match out {
        Ok(orbital) => {
            let n = orbital.radial.len();
            Ok((
                0,
                orbital.r_c,
                orbital.energy,
                orbital.free_energy,
                array(py, &[n], orbital.radial)?,
            ))
        }
        Err(reach) => Ok((3, reach, f64::NAN, f64::NAN, array(py, &[0], Vec::new())?)),
    }
}

#[pyfunction]
fn pao_quasi_gaussian<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    r_cut: f64,
) -> PyResult<Array<'py, f64>> {
    let out = radial::quasi_gaussian(r.as_slice()?, alpha, r_cut);
    array(py, &[out.len()], out)
}

#[pyfunction]
fn pao_polarization_radial<'py>(
    py: Python<'py>,
    table: PyReadonlyArray1<'py, f64>,
    l: usize,
    r_cut: f64,
    r_char: f64,
) -> PyResult<Array<'py, f64>> {
    let out = radial::polarization_radial(table.as_slice()?, l, r_cut, r_char).map_err(error)?;
    array(py, &[out.len()], out)
}

#[pyfunction]
fn pao_power_shape<'py>(
    py: Python<'py>,
    table: PyReadonlyArray1<'py, f64>,
    outermost: PyReadonlyArray1<'py, f64>,
    power: usize,
) -> PyResult<Array<'py, f64>> {
    let out =
        radial::power_shape(table.as_slice()?, outermost.as_slice()?, power).map_err(error)?;
    array(py, &[out.len()], out)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pao_bound_level(
    context: PyRef<'_, PyContext>,
    py: Python<'_>,
    r: PyReadonlyArray1<'_, f64>,
    local: PyReadonlyArray1<'_, f64>,
    projectors: PyReadonlyArray2<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
    overlap: PyReadonlyArray2<'_, f64>,
    wave: PyReadonlyArray1<'_, f64>,
    l: usize,
    r_cut: f64,
    r_max: f64,
) -> PyResult<f64> {
    let channel = Owned::take(
        &context,
        &r,
        &local,
        &projectors,
        &coupling,
        &overlap,
        &wave,
        l,
        r_cut,
    )?;
    let ctx = &context.inner;
    py.detach(|| pao::bound_level(ctx, channel.data(), r_max))
        .map_err(error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pao_bound_radial<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    local: PyReadonlyArray1<'py, f64>,
    projectors: PyReadonlyArray2<'py, f64>,
    coupling: PyReadonlyArray2<'py, f64>,
    overlap: PyReadonlyArray2<'py, f64>,
    wave: PyReadonlyArray1<'py, f64>,
    l: usize,
    r_cut: f64,
    r_max: f64,
) -> PyResult<(Array<'py, f64>, f64)> {
    let channel = Owned::take(
        &context,
        &r,
        &local,
        &projectors,
        &coupling,
        &overlap,
        &wave,
        l,
        r_cut,
    )?;
    let ctx = &context.inner;
    let (radial, level) = py
        .detach(|| pao::bound_radial(ctx, channel.data(), r_max))
        .map_err(error)?;
    Ok((array(py, &[radial.len()], radial)?, level))
}

#[pyfunction]
fn diffuse_parameters(l: usize, rms: f64, cutoff_norm: f64) -> PyResult<(f64, f64)> {
    diffuse::parameters(l, rms, cutoff_norm).map_err(error)
}

#[pyfunction]
fn diffuse_radial<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    l: usize,
    rms: f64,
    cutoff_norm: f64,
) -> PyResult<(Array<'py, f64>, f64, f64)> {
    let (values, alpha, r_cut) =
        diffuse::radial(r.as_slice()?, l, rms, cutoff_norm).map_err(error)?;
    Ok((array(py, &[values.len()], values)?, alpha, r_cut))
}

#[pyfunction]
fn diffuse_rms(r: PyReadonlyArray1<'_, f64>, values: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let (r, values) = (r.as_slice()?, values.as_slice()?);
    if r.len() != values.len() || r.len() < 2 {
        return Err(PyValueError::new_err(
            "rms needs a table of matching arrays",
        ));
    }
    Ok(diffuse::rms(r, values))
}

#[pyfunction]
fn diffuse_radial_norm(
    r: PyReadonlyArray1<'_, f64>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let (r, values) = (r.as_slice()?, values.as_slice()?);
    if r.len() != values.len() || r.len() < 2 {
        return Err(PyValueError::new_err(
            "a norm needs a table of matching arrays",
        ));
    }
    Ok(radial::radial_norm(r, values))
}

#[pyfunction]
fn diffuse_extended_grid<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    reach: f64,
) -> PyResult<Array<'py, f64>> {
    let out = diffuse::extended_grid(r.as_slice()?, reach).map_err(error)?;
    array(py, &[out.len()], out)
}

#[pyfunction]
fn diffuse_on_grid<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r_new: PyReadonlyArray1<'py, f64>,
    r_old: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    let (r_new, _a) = copy(&context.inner, &r_new)?;
    let (r_old, _b) = copy(&context.inner, &r_old)?;
    let (values, _c) = copy(&context.inner, &values)?;
    let ctx = &context.inner;
    let out = py
        .detach(|| diffuse::on_grid(ctx, &r_new, &r_old, &values))
        .map_err(error)?;
    array(py, &[out.len()], out)
}

#[pyfunction]
fn diffuse_orthogonalize<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    same: Vec<PyReadonlyArray1<'py, f64>>,
    values: PyReadonlyArray1<'py, f64>,
    renormalize: bool,
    tolerance: f64,
) -> PyResult<(f64, bool, Array<'py, f64>)> {
    let (r, _a) = copy(&context.inner, &r)?;
    let (values, _b) = copy(&context.inner, &values)?;
    let mut held = Vec::with_capacity(same.len());
    let mut keep = Vec::with_capacity(same.len());
    for g in &same {
        let (g, reservation) = copy(&context.inner, g)?;
        held.push(g);
        keep.push(reservation);
    }
    let shell = py
        .detach(|| {
            let views: Vec<&[f64]> = held.iter().map(Vec::as_slice).collect();
            diffuse::orthogonalize(&r, &views, &values, renormalize, tolerance)
        })
        .map_err(error)?;
    let n = shell.values.len();
    Ok((
        shell.residual,
        shell.accepted,
        array(py, &[n], shell.values)?,
    ))
}

#[pyfunction]
fn diffuse_face_distance(
    origin: [f64; 3],
    step: [[f64; 3]; 3],
    shape: [usize; 3],
    center: [f64; 3],
) -> PyResult<f64> {
    diffuse::face_distance(origin, step, shape, center).map_err(error)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(confinement_potential, m)?)?;
    m.add_function(wrap_pyfunction!(confinement_energy, m)?)?;
    m.add_function(wrap_pyfunction!(confinement_eigenstate, m)?)?;
    m.add_function(wrap_pyfunction!(confinement_free_energy, m)?)?;
    m.add_function(wrap_pyfunction!(confinement_radius, m)?)?;
    m.add_function(wrap_pyfunction!(confinement_orbital, m)?)?;
    m.add_function(wrap_pyfunction!(pao_quasi_gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(pao_polarization_radial, m)?)?;
    m.add_function(wrap_pyfunction!(pao_power_shape, m)?)?;
    m.add_function(wrap_pyfunction!(pao_bound_level, m)?)?;
    m.add_function(wrap_pyfunction!(pao_bound_radial, m)?)?;
    m.add_function(wrap_pyfunction!(diffuse_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(diffuse_radial, m)?)?;
    m.add_function(wrap_pyfunction!(diffuse_rms, m)?)?;
    m.add_function(wrap_pyfunction!(diffuse_radial_norm, m)?)?;
    m.add_function(wrap_pyfunction!(diffuse_extended_grid, m)?)?;
    m.add_function(wrap_pyfunction!(diffuse_on_grid, m)?)?;
    m.add_function(wrap_pyfunction!(diffuse_orthogonalize, m)?)?;
    m.add_function(wrap_pyfunction!(diffuse_face_distance, m)?)?;
    Ok(())
}
