//! Owned Wannier result bindings. Numerical work detaches from Python.
use super::{array, copy, downfold::PyDownfolded, error};
use mandacaru_core::{
    periodic::wannier::{
        downfold::{CoulombBox, Request, Screening},
        setup::Wannier,
        Localization,
    },
    quantum::C64,
};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyDict};
use std::sync::Arc;

#[pyclass(name = "WannierResult", module = "mandacaru._native", frozen)]
pub struct PyWannierResult {
    pub inner: Arc<Wannier>,
}
#[pymethods]
impl PyWannierResult {
    #[getter]
    fn n_spins(&self) -> usize {
        self.inner.channels.len()
    }
    #[pyo3(signature=(*,spin=0))]
    fn snapshot<'py>(&self, py: Python<'py>, spin: usize) -> PyResult<Bound<'py, PyDict>> {
        let owner = &self.inner;
        let channel = owner.channels.get(spin).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("invalid Wannier spin channel")
        })?;
        let localized = &channel.localized;
        let n = owner.trials.len();
        let out = PyDict::new(py);
        out.set_item(
            "centers",
            array(
                py,
                &[n, 3],
                localized
                    .spread
                    .centers
                    .as_slice()
                    .iter()
                    .flatten()
                    .copied()
                    .collect(),
            )?,
        )?;
        out.set_item("spreads", array(py, &[n], localized.spread.spreads())?)?;
        out.set_item("omega_invariant", localized.spread.invariant)?;
        out.set_item("omega_offdiagonal", localized.spread.off_diagonal)?;
        out.set_item("omega_diagonal", localized.spread.diagonal)?;
        out.set_item("converged", localized.converged)?;
        out.set_item(
            "disentanglement_converged",
            channel.disentanglement_converged,
        )?;
        out.set_item(
            "history",
            array(
                py,
                &[localized.history.len(), 3],
                localized.history.iter().flatten().copied().collect(),
            )?,
        )?;
        out.set_item("disentanglement", channel.disentanglement.clone())?;
        out.set_item(
            "unitary",
            array(
                py,
                &[localized.gauge.len(), owner.bands.len(), n],
                localized
                    .gauge
                    .iter()
                    .flat_map(|m| {
                        (0..m.nrows()).flat_map(move |i| (0..m.ncols()).map(move |j| m[(i, j)]))
                    })
                    .collect::<Vec<C64>>(),
            )?,
        )?;
        out.set_item(
            "kpoints",
            array(
                py,
                &[channel.frames.len(), 3],
                channel.frames.iter().flat_map(|f| f.fractional).collect(),
            )?,
        )?;
        out.set_item(
            "lattice_vectors",
            array(
                py,
                &[channel.hoppings.vectors.as_slice().len(), 3],
                channel
                    .hoppings
                    .vectors
                    .as_slice()
                    .iter()
                    .flatten()
                    .copied()
                    .collect(),
            )?,
        )?;
        out.set_item(
            "hamiltonian_R",
            array(
                py,
                &[channel.hoppings.vectors.as_slice().len(), n, n],
                channel.hoppings.values.as_slice().to_vec(),
            )?,
        )?;
        let nr = channel.hoppings.vectors.as_slice().len();
        let pos_flat: Vec<C64> = channel
            .positions_r
            .iter()
            .flat_map(|b| b.as_slice().iter().copied())
            .collect();
        out.set_item("positions_R", array(py, &[3, nr, n, n], pos_flat)?)?;
        out.set_item(
            "position_asymmetry",
            channel.interpolation.position_asymmetry,
        )?;
        let lat = channel.interpolation.lattice;
        let cell_rows: Vec<f64> = (0..3)
            .flat_map(|a| (0..3).map(move |x| lat[(x, a)]))
            .collect();
        out.set_item("cell_bohr", array(py, &[3, 3], cell_rows)?)?;
        out.set_item("fermi_ha", owner.state.fermi)?;
        match owner.localization {
            Localization::Spread => {
                out.set_item("localization", "spread")?;
                out.set_item("populations", py.None())?;
                out.set_item("exponent", py.None())?;
            }
            Localization::Populations { scheme, exponent } => {
                out.set_item("localization", "populations")?;
                out.set_item("populations", scheme.name())?;
                out.set_item("exponent", exponent)?;
            }
        }
        out.set_item(
            "labels",
            owner
                .trials
                .iter()
                .map(|t| t.label.clone())
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "trial_weights",
            array(py, &[n, owner.trials.len()], channel.trial_weights.clone())?,
        )?;
        out.set_item("bands", owner.bands.clone())?;
        out.set_item("size", owner.size)?;
        Ok(out)
    }
    /// The many-body problem of a fragment of the functions (bare interaction, in Hartree).
    #[pyo3(signature=(functions=None,num_particles=None,*,screening=None,screening_cutoff=None,screening_bands=None,coulomb="supercell"))]
    fn downfold(
        &self,
        py: Python<'_>,
        functions: Option<Vec<(usize, [i64; 3])>>,
        num_particles: Option<[usize; 2]>,
        screening: Option<&str>,
        screening_cutoff: Option<f64>,
        screening_bands: Option<usize>,
        coulomb: &str,
    ) -> PyResult<PyDownfolded> {
        let request = Request {
            functions,
            num_particles,
            screening: screening.map(Screening::parse).transpose().map_err(error)?,
            cutoff: screening_cutoff,
            bands: screening_bands,
            coulomb: CoulombBox::parse(coulomb).map_err(error)?,
            workers: None,
        };
        let owner = self.inner.clone();
        let inner = py.detach(|| owner.downfold(&request)).map_err(error)?;
        Ok(PyDownfolded {
            inner,
            screening: screening.map(str::to_owned),
            coulomb: coulomb.to_owned(),
        })
    }
    #[pyo3(signature=(fractional_kpoints,*,spin=0,eigenvectors=false))]
    fn interpolate<'py>(
        &self,
        py: Python<'py>,
        fractional_kpoints: PyReadonlyArray2<'py, f64>,
        spin: usize,
        eigenvectors: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if fractional_kpoints.shape()[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fractional k-points need shape (n,3)",
            ));
        }
        let channel = self.inner.channels.get(spin).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("invalid Wannier spin channel")
        })?;
        let ctx = &self.inner.crystal.context;
        let (values, _memory) = copy(ctx, &fractional_kpoints)?;
        let points: Vec<[f64; 3]> = values.chunks_exact(3).map(|r| [r[0], r[1], r[2]]).collect();
        let n_k = points.len();
        let j = channel.hoppings.functions;
        if !eigenvectors {
            let out = py
                .detach(|| channel.hoppings.interpolate(ctx, &points))
                .map_err(error)?;
            return Ok(array(py, &[n_k, j], out.as_slice().to_vec())?.into_any());
        }
        let (e_buf, vecs) = py
            .detach(|| channel.interpolation.interpolate_with_vectors(ctx, &points))
            .map_err(error)?;
        let e_arr = array(py, &[n_k, j], e_buf.as_slice().to_vec())?;
        let mut flat_vecs = Vec::with_capacity(n_k * j * j);
        for m in &vecs {
            for r in 0..m.nrows() {
                for c in 0..m.ncols() {
                    flat_vecs.push(m[(r, c)]);
                }
            }
        }
        let v_arr = array(py, &[n_k, j, j], flat_vecs)?;
        Ok((e_arr, v_arr).into_pyobject(py)?.into_any())
    }
    /// Wannier-interpolated anomalous Hall conductivity on mesh `size` (`periodic::wannier::setup::Wannier::anomalous_hall`), e^2/(hbar Bohr).
    #[pyo3(signature=(size,*,refine=true))]
    fn anomalous_hall<'py>(
        &self,
        py: Python<'py>,
        size: [usize; 3],
        refine: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let owner = self.inner.clone();
        let h = py
            .detach(|| owner.anomalous_hall(size, refine))
            .map_err(error)?;
        super::berry::hall_dict(py, &h, "wannier")
    }
}
