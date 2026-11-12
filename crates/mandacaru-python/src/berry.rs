//! Berry-phase bindings of a converged crystal: owned arrays out, Python detached for the work.
//! No numerics here; `mandacaru_core::periodic::berry` computes every value (atomic units).
use super::{array, error, periodic::PyPeriodicResult};
use mandacaru_core::periodic::{
    berry::{curvature, hall, polarization, strings, Bands, Sampler},
    setup::PeriodicSetup,
};
use pyo3::{exceptions::PyNotImplementedError, prelude::*, types::PyDict};

fn setup(result: &PyPeriodicResult) -> PyResult<&PeriodicSetup> {
    result.setup.as_deref().ok_or_else(|| {
        PyNotImplementedError::new_err(
            "Berry phases need states off the SCF mesh, which only the native periodic setup \
             makes; a crystal loaded from a snapshot has none",
        )
    })
}

fn bands(range: Option<(usize, usize)>) -> Bands {
    range.map_or(Bands::Occupied, |(lo, hi)| Bands::Range(lo, hi))
}

pub(crate) fn berry_phase<'py>(
    result: &PyPeriodicResult,
    py: Python<'py>,
    direction: usize,
    size: [usize; 3],
    range: Option<(usize, usize)>,
) -> PyResult<Bound<'py, PyDict>> {
    let setup = setup(result)?;
    let (crystal, state) = (&*result.crystal, &*result.inner);
    let set = py
        .detach(|| {
            let sampler = Sampler::new(crystal, state, setup)?;
            strings::berry_phase(&sampler, direction, size, bands(range))
        })
        .map_err(error)?;
    let [t0, t1] = mandacaru_core::periodic::berry::transverse_axes(direction).map_err(error)?;
    let (n0, n1) = (size[t0], size[t1]);
    let channels = set.phases.len();
    let out = PyDict::new(py);
    out.set_item("direction", direction)?;
    out.set_item("size", size.to_vec())?;
    out.set_item("transverse_axes", vec![t0, t1])?;
    out.set_item("bands", set.ranges.clone())?;
    out.set_item(
        "starts",
        array(
            py,
            &[n0, n1, 3],
            set.starts.iter().flatten().copied().collect(),
        )?,
    )?;
    out.set_item(
        "phases",
        array(
            py,
            &[channels, n0, n1],
            set.phases.iter().flatten().copied().collect(),
        )?,
    )?;
    out.set_item("mean", set.mean.clone())?;
    out.set_item("spread", set.spread.clone())?;
    let mut centers = Vec::with_capacity(channels);
    let mut connection = Vec::with_capacity(channels);
    for s in 0..channels {
        let width = set.centers[s].first().map_or(0, |c| c.len());
        centers.push(array(
            py,
            &[n0, n1, width],
            set.centers[s].iter().flatten().copied().collect(),
        )?);
        connection.push(array(
            py,
            &[n0, n1, width],
            set.connection[s].iter().flatten().copied().collect(),
        )?);
    }
    out.set_item("centers", centers)?;
    out.set_item("connection", connection)?;
    out.set_item("minimum_singular_value", set.minimum)?;
    Ok(out)
}

pub(crate) fn berry_curvature<'py>(
    result: &PyPeriodicResult,
    py: Python<'py>,
    normal: usize,
    size: [usize; 3],
    offset: f64,
    range: Option<(usize, usize)>,
) -> PyResult<Bound<'py, PyDict>> {
    let setup = setup(result)?;
    let (crystal, state) = (&*result.crystal, &*result.inner);
    let [a, b] = mandacaru_core::periodic::berry::plane_axes(normal).map_err(error)?;
    let plane = py
        .detach(|| {
            let sampler = Sampler::new(crystal, state, setup)?;
            curvature::berry_curvature(&sampler, normal, [size[a], size[b]], offset, bands(range))
        })
        .map_err(error)?;
    let [na, nb] = plane.size;
    let channels = plane.flux.len();
    let out = PyDict::new(py);
    out.set_item("normal", plane.normal)?;
    out.set_item("axes", plane.axes.to_vec())?;
    out.set_item("size", plane.size.to_vec())?;
    out.set_item("offset", plane.offset)?;
    out.set_item("orientation", plane.orientation.to_vec())?;
    out.set_item(
        "flux",
        array(
            py,
            &[channels, na, nb],
            plane.flux.iter().flatten().copied().collect(),
        )?,
    )?;
    out.set_item(
        "curvature",
        array(
            py,
            &[channels, na, nb],
            plane.curvature.iter().flatten().copied().collect(),
        )?,
    )?;
    out.set_item(
        "centers",
        array(
            py,
            &[na, nb, 3],
            plane.centers.iter().flatten().copied().collect(),
        )?,
    )?;
    out.set_item("bands", plane.ranges.clone())?;
    out.set_item("chern", plane.chern.clone())?;
    out.set_item("chern_total", plane.total_chern())?;
    out.set_item("max_flux", plane.max_flux)?;
    out.set_item("minimum_singular_value", plane.minimum)?;
    Ok(out)
}

pub(crate) fn polarization<'py>(
    result: &PyPeriodicResult,
    py: Python<'py>,
    size: [usize; 3],
) -> PyResult<Bound<'py, PyDict>> {
    let setup = setup(result)?;
    let (crystal, state) = (&*result.crystal, &*result.inner);
    let p = py
        .detach(|| {
            let sampler = Sampler::new(crystal, state, setup)?;
            polarization::polarization(&sampler, size)
        })
        .map_err(error)?;
    let out = PyDict::new(py);
    out.set_item("size", p.size.to_vec())?;
    out.set_item("degeneracy", p.degeneracy)?;
    out.set_item("quantum", p.quantum)?;
    out.set_item(
        "phases",
        array(
            py,
            &[p.phases.len(), 3],
            p.phases.iter().flatten().copied().collect(),
        )?,
    )?;
    out.set_item("spread", p.spread.to_vec())?;
    for (name, values) in [
        ("electronic", p.electronic),
        ("ionic", p.ionic),
        ("raw", p.raw),
        ("fractional", p.fractional),
        ("vector", p.vector),
        ("electronic_vector", p.cartesian(p.electronic)),
        ("ionic_vector", p.cartesian(p.ionic)),
    ] {
        out.set_item(name, array(py, &[3], values.to_vec())?)?;
    }
    out.set_item(
        "quanta",
        array(py, &[3, 3], p.quanta.iter().flatten().copied().collect())?,
    )?;
    out.set_item("volume", p.volume)?;
    out.set_item("warnings", p.warnings.clone())?;
    out.set_item("minimum_singular_value", p.minimum)?;
    Ok(out)
}

pub(crate) fn anomalous_hall<'py>(
    result: &PyPeriodicResult,
    py: Python<'py>,
    size: [usize; 3],
) -> PyResult<Bound<'py, PyDict>> {
    let setup = setup(result)?;
    let (crystal, state) = (&*result.crystal, &*result.inner);
    let h = py
        .detach(|| {
            let sampler = Sampler::new(crystal, state, setup)?;
            hall::anomalous_hall(&sampler, size)
        })
        .map_err(error)?;
    let out = PyDict::new(py);
    out.set_item("size", h.size.to_vec())?;
    out.set_item("chern", h.chern.iter().cloned().collect::<Vec<_>>())?;
    out.set_item("total", h.total.to_vec())?;
    out.set_item("vector", array(py, &[3], h.vector.to_vec())?)?;
    out.set_item(
        "sigma",
        array(py, &[3, 3], h.sigma.iter().flatten().copied().collect())?,
    )?;
    out.set_item("max_flux", h.max_flux)?;
    out.set_item("minimum_singular_value", h.minimum)?;
    out.set_item("metallic", h.metallic)?;
    out.set_item("numbers", h.numbers.to_vec())?;
    out.set_item("layers", h.layers.iter().cloned().collect::<Vec<_>>())?;
    out.set_item(
        "sigma_coarse",
        h.sigma_coarse
            .map(|sigma| array(py, &[3, 3], sigma.iter().flatten().copied().collect()))
            .transpose()?,
    )?;
    out.set_item("cap", h.cap)?;
    out.set_item("merged", h.merged)?;
    out.set_item("plaquettes", h.plaquettes)?;
    out.set_item("fermi", h.fermi)?;
    out.set_item("scf_electron_error", h.scf_electron_error)?;
    out.set_item("warnings", h.warnings.clone())?;
    Ok(out)
}
