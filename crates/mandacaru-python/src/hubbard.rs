//! Bindings of the DFT+U option and shell data. Arrays are copied in and returned owned.
//! The option is read, validated and resolved by `mandacaru_core::models::hubbard_request`, the
//! table by `mandacaru_core::periodic::hubbard_table`, the weights by `paw::shell`; the views of
//! a converged run are read from the owned `Report`. Nothing is decided or computed here.
use super::{array, copy, error, PyContext};
use mandacaru_core::{
    models::{
        hubbard::{shell_letter, Form},
        hubbard_response::{Kernel, Response},
        hubbard_request::Request,
    },
    periodic::{
        hubbard_response::ResponseResult as PeriodicResponseResult,
        hubbard_table::{
            build, check_datasets, check_loaded as check_table, ChannelData, Dataset, Layout,
            Projector, Species,
        },
        Crystal, PeriodicResult,
    },
    molecular::hubbard_response::ResponseResult as MolecularResponseResult,
    product, Error, Reservation,
};
use numpy::{Complex64, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{
    exceptions::{PyKeyError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList},
};
use std::collections::{BTreeMap, HashMap};

/// `(ae, vanderbilt, r_cut, occupations)` of one channel: `ae` is `waves x grid`, `vanderbilt`
/// is `waves x waves`, `occupations` holds one number per wave (a library entry that stores only
/// the scalar channel occupation is passed as `[occupation, 0, ...]`).
type ChannelArrays<'py> = (
    PyReadonlyArray2<'py, f64>,
    PyReadonlyArray2<'py, f64>,
    f64,
    PyReadonlyArray1<'py, f64>,
);
/// The radial grid of a species, its channels by `l` and the file the dataset came from.
type SpeciesArrays<'py> = (
    PyReadonlyArray1<'py, f64>,
    HashMap<usize, ChannelArrays<'py>>,
    String,
);

/// Validates the `hubbard_u` option (its JSON text, energies in eV) against the functional and
/// the precision of the calculation; raises with the reason. Nothing is returned: the text is
/// read again where it is used, by the same parser.
#[pyfunction]
fn hubbard_validate(spec: &str, xc: &str, precision: &str) -> PyResult<()> {
    Request::parse(spec)
        .and_then(|request| request.require_scope(xc, precision))
        .map_err(error)
}

/// Refuses, before any crystal exists, what the elements and their datasets already rule out:
/// a spin-orbit dataset, an element the option names that the system lacks, a dataset without
/// the requested channel, a channel with no occupied reference wave. `symbols`, `sources` (the
/// dataset files) and `occupations` (by `l`, one number per partial wave) hold one entry per
/// distinct element.
#[pyfunction]
fn hubbard_check_atoms<'py>(
    context: PyRef<'_, PyContext>,
    spec: &str,
    spin_orbit: bool,
    symbols: Vec<String>,
    sources: Vec<String>,
    occupations: Vec<HashMap<usize, PyReadonlyArray1<'py, f64>>>,
) -> PyResult<()> {
    let ctx = &context.inner;
    let request = Request::parse(spec).map_err(error)?;
    if sources.len() != symbols.len() || occupations.len() != symbols.len() {
        return Err(PyValueError::new_err(
            "hubbard_u: one dataset file and one occupation table per element are needed",
        ));
    }
    let mut holds: Vec<Reservation> = Vec::new();
    let mut owned: Vec<BTreeMap<usize, Vec<f64>>> = Vec::new();
    owned
        .try_reserve_exact(occupations.len())
        .map_err(|_| error(Error::Allocation))?;
    for channels in &occupations {
        let mut by_l = BTreeMap::new();
        for (l, values) in channels {
            let (values, hold) = copy(ctx, values)?;
            holds.push(hold);
            by_l.insert(*l, values);
        }
        owned.push(by_l);
    }
    let datasets: Vec<Dataset<'_>> = symbols
        .iter()
        .zip(&sources)
        .zip(&owned)
        .map(|((symbol, source), by_l)| Dataset {
            symbol,
            source,
            occupations: by_l.iter().map(|(l, v)| (*l, v.as_slice())).collect(),
        })
        .collect();
    check_datasets(&request, spin_orbit, &datasets).map_err(error)
}

/// The `hubbard` object of a crystal snapshot, as JSON text, for the option `spec`.
///
/// `labels` is `projectors x 4` (atom, `l`, `m`, radial index per global projector position)
/// with `symbols` and `bases` (`"raw"` or `"dual"`) beside it; `species_of_atom` indexes
/// `species` for each atom, and `moments` holds each atom's signed initial moment.
#[pyfunction]
fn hubbard_table<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    spec: &str,
    spin_orbit: bool,
    symbols: Vec<String>,
    bases: Vec<String>,
    labels: PyReadonlyArray2<'py, i64>,
    species_of_atom: Vec<usize>,
    species: Vec<SpeciesArrays<'py>>,
    moments: PyReadonlyArray1<'py, f64>,
) -> PyResult<String> {
    let ctx = &context.inner;
    let request = Request::parse(spec).map_err(error)?;
    let count = symbols.len();
    if labels.shape() != [count, 4] || bases.len() != count {
        return Err(PyValueError::new_err(
            "hubbard_u: the projector labels must hold one row of (atom, l, m, index), one \
             symbol and one basis per projector",
        ));
    }
    let index = |value: i64| {
        usize::try_from(value)
            .map_err(|_| PyValueError::new_err("hubbard_u: a projector label is negative"))
    };
    let (labels, _labels) = copy(ctx, &labels)?;
    let projectors = labels
        .chunks_exact(4)
        .zip(symbols.into_iter().zip(bases))
        .map(|(row, (symbol, basis))| {
            Ok(Projector {
                symbol,
                basis,
                atom: index(row[0])?,
                l: index(row[1])?,
                m: row[2],
                index: index(row[3])?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut holds: Vec<Reservation> = Vec::new();
    let mut owned = Vec::with_capacity(species.len());
    for (r, channels, source) in &species {
        let (grid, hold) = copy(ctx, r)?;
        holds.push(hold);
        let mut by_l = BTreeMap::new();
        for (l, (ae, vanderbilt, r_cut, occupations)) in channels {
            let waves = ae.shape()[0];
            if vanderbilt.shape() != [waves, waves] {
                return Err(PyValueError::new_err(
                    "hubbard_u: the Vanderbilt matrix must be waves x waves",
                ));
            }
            let (ae, a) = copy(ctx, ae)?;
            let (vanderbilt, v) = copy(ctx, vanderbilt)?;
            let (occupations, o) = copy(ctx, occupations)?;
            holds.extend([a, v, o]);
            by_l.insert(
                *l,
                ChannelData {
                    waves,
                    ae,
                    vanderbilt,
                    r_cut: *r_cut,
                    occupations,
                },
            );
        }
        owned.push(Species {
            r: grid,
            channels: by_l,
            source: source.clone(),
        });
    }
    let (moments, _moments) = copy(ctx, &moments)?;
    let table = py
        .detach(|| {
            let layout = Layout {
                projectors: &projectors,
                species_of_atom: &species_of_atom,
                species: &owned,
                moments: &moments,
                spin_orbit,
            };
            build(ctx, &request, &layout)
        })
        .map_err(error)?;
    serde_json::to_string(&table.to_json()).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Compares the option `spec` with the DFT+U table of a loaded crystal; raises on a mismatch.
pub(crate) fn check_loaded(spec: &str, crystal: &Crystal) -> PyResult<()> {
    let request = Request::parse(spec).map_err(error)?;
    check_table(&request, crystal.hubbard.as_ref()).map_err(error)
}

/// The DFT+U shells of a crystal as `(atom, symbol, l, u_ha, j_ha)` rows, or `None` when it has
/// no `hubbard` object. Read from the crystal that owns the definition.
pub(crate) fn definition(crystal: &Crystal) -> Option<Vec<(usize, String, usize, f64, f64)>> {
    crystal.hubbard.as_ref().map(|spec| {
        spec.shells
            .iter()
            .map(|s| (s.atom, s.symbol.clone(), s.l, s.u, s.j))
            .collect()
    })
}

/// One dictionary per shell of a converged DFT+U run: the definition (`atom`, `symbol`, `l`,
/// `shell` letter, `u_ha`, `j_ha`, `u_eff_ha`), the occupation matrices `n` (complex,
/// `spins x (2l+1) x (2l+1)`), their traces, and `energy_ha`, the shell's `E_U`. `None` for a
/// run without `hubbard`.
pub(crate) fn view<'py>(
    py: Python<'py>,
    crystal: &Crystal,
    result: &PeriodicResult,
) -> PyResult<Option<Bound<'py, PyList>>> {
    let (Some(spec), Some(report)) = (crystal.hubbard.as_ref(), result.hubbard.as_ref()) else {
        return Ok(None);
    };
    let energies = spec
        .shell_energies(crystal, &report.occupations)
        .map_err(error)?;
    let out = PyList::empty(py);
    for (index, shell) in spec.shells.iter().enumerate() {
        let p = shell.size();
        let d = if crystal.spinor { 2 * p } else { p };
        let count = product(&[crystal.spins, d, d]).map_err(error)?;
        // The matrices (16 bytes per entry) and the traces are reserved before they are built.
        let matrices = product(&[count, 16]).map_err(error)?;
        let trace_bytes = product(&[crystal.spins, 8]).map_err(error)?;
        let bytes = matrices.checked_add(trace_bytes).ok_or(Error::Overflow);
        let _hold = crystal
            .context
            .reserve(bytes.map_err(error)?)
            .map_err(error)?;
        let mut values = Vec::with_capacity(count);
        let mut traces = Vec::with_capacity(crystal.spins);
        for spin in 0..crystal.spins {
            let n = report.occupation(spin, index);
            traces.push(report.occupation_trace(spin, index, p));
            values.extend(n.iter().map(|z| Complex64::new(z.re, z.im)));
        }
        let item = PyDict::new(py);
        item.set_item("atom", shell.atom)?;
        item.set_item("symbol", &shell.symbol)?;
        item.set_item("l", shell.l)?;
        item.set_item("shell", shell_letter(shell.l).map(String::from))?;
        item.set_item("u_ha", shell.u)?;
        item.set_item("j_ha", shell.j)?;
        item.set_item("u_eff_ha", shell.u_eff())?;
        item.set_item("energy_ha", energies[index])?;
        item.set_item("trace", traces)?;
        item.set_item("n", array(py, &[crystal.spins, d, d], values)?)?;
        if let Some(source) = &shell.source {
            item.set_item("U_source", source)?;
        }
        out.append(item)?;
    }
    Ok(Some(out))
}

pub(crate) fn parse_kernel(name: &str) -> PyResult<Kernel> {
    match name.to_ascii_lowercase().as_str() {
        "spin-summed" | "spin_summed" => Ok(Kernel::SpinSummed),
        "charge" => Ok(Kernel::Charge),
        "same-spin" | "same_spin" => Ok(Kernel::SameSpin),
        other => Err(PyValueError::new_err(format!(
            "unknown response kernel {other:?}; choose 'spin-summed', 'charge', or 'same-spin'"
        ))),
    }
}

fn parse_form(name: &str) -> PyResult<Form> {
    Form::parse(name).map_err(error)
}

fn matrix_to_py<'py>(
    py: Python<'py>,
    m: &nalgebra::DMatrix<f64>,
    scale: f64,
) -> PyResult<super::Array<'py, f64>> {
    let rows = m.nrows();
    let cols = m.ncols();
    let data: Vec<f64> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| m[(r, c)] * scale))
        .collect();
    array(py, &[rows, cols], data)
}

/// Native linear-response Hubbard U view (for crystals and molecules).
#[pyclass(name = "HubbardResponse", module = "mandacaru._native", frozen)]
pub struct PyHubbardResponse {
    u_matrix: nalgebra::DMatrix<f64>,
    u_charge: nalgebra::DMatrix<f64>,
    u_cross: Option<nalgebra::DMatrix<f64>>,
    chi0: nalgebra::DMatrix<f64>,
    chi: nalgebra::DMatrix<f64>,
    alphas: Vec<f64>,
    j_response: Option<Vec<f64>>,
    symbols: Vec<String>,
    self_consistent: bool,
    response: Response,
    energy_scale: f64,
    unit_name: String,
}

impl PyHubbardResponse {
    pub(crate) fn from_periodic(
        res: PeriodicResponseResult,
        energy_scale: f64,
        unit_name: String,
    ) -> Self {
        let u_cross = res.u_cross();
        Self {
            u_matrix: res.u_matrix,
            u_charge: res.u_charge,
            u_cross,
            chi0: res.chi0,
            chi: res.chi,
            alphas: res.alphas,
            j_response: res.j_response,
            symbols: res.symbols,
            self_consistent: res.self_consistent,
            response: res.response,
            energy_scale,
            unit_name,
        }
    }

    pub(crate) fn from_molecular(
        res: MolecularResponseResult,
        energy_scale: f64,
        unit_name: String,
    ) -> Self {
        let u_cross = res.u_cross();
        Self {
            u_matrix: res.u_matrix,
            u_charge: res.u_charge,
            u_cross,
            chi0: res.chi0,
            chi: res.chi,
            alphas: res.alphas,
            j_response: res.j_response,
            symbols: res.symbols,
            self_consistent: false,
            response: res.response,
            energy_scale,
            unit_name,
        }
    }
}

#[pymethods]
impl PyHubbardResponse {
    #[getter]
    fn u_matrix<'py>(&self, py: Python<'py>) -> PyResult<super::Array<'py, f64>> {
        matrix_to_py(py, &self.u_matrix, self.energy_scale)
    }
    #[getter]
    fn u_charge<'py>(&self, py: Python<'py>) -> PyResult<super::Array<'py, f64>> {
        matrix_to_py(py, &self.u_charge, self.energy_scale)
    }
    #[getter]
    fn u_cross<'py>(&self, py: Python<'py>) -> PyResult<Option<super::Array<'py, f64>>> {
        self.u_cross
            .as_ref()
            .map(|m| matrix_to_py(py, m, self.energy_scale))
            .transpose()
    }
    #[getter]
    fn chi0<'py>(&self, py: Python<'py>) -> PyResult<super::Array<'py, f64>> {
        matrix_to_py(py, &self.chi0, 1.0 / self.energy_scale)
    }
    #[getter]
    fn chi<'py>(&self, py: Python<'py>) -> PyResult<super::Array<'py, f64>> {
        matrix_to_py(py, &self.chi, 1.0 / self.energy_scale)
    }
    #[getter]
    fn alphas(&self) -> Vec<f64> {
        self.alphas.iter().map(|a| a * self.energy_scale).collect()
    }
    #[getter]
    fn j_response(&self) -> Option<Vec<f64>> {
        self.j_response
            .as_ref()
            .map(|jv| jv.iter().map(|j| j * self.energy_scale).collect())
    }
    #[getter]
    fn shells(&self) -> usize {
        self.response.shells
    }
    #[getter]
    fn channels(&self) -> usize {
        self.response.channels
    }
    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.symbols.clone()
    }
    #[getter]
    fn self_consistent(&self) -> bool {
        self.self_consistent
    }
    #[getter]
    fn nonlinearity(&self) -> Option<f64> {
        self.response.nonlinearity
    }
    #[getter]
    fn asymmetry(&self) -> f64 {
        self.response.asymmetry
    }
    #[getter]
    fn floor(&self) -> f64 {
        self.response.sum_rule
    }
    #[getter]
    fn condition_chi0(&self) -> f64 {
        self.response.condition_chi0
    }
    #[getter]
    fn condition_chi(&self) -> f64 {
        self.response.condition_chi
    }
    #[getter]
    fn condition(&self) -> f64 {
        self.response.condition_chi0.max(self.response.condition_chi)
    }
    #[getter]
    fn energy_unit(&self) -> String {
        self.unit_name.clone()
    }

    /// Groups shells by element and validates spread tolerance in native Rust.
    #[pyo3(signature=(*, form="dudarev", kernel="spin-summed", j=None, tolerance=0.5))]
    fn hubbard_u<'py>(
        &self,
        py: Python<'py>,
        form: &str,
        kernel: &str,
        j: Option<f64>,
        tolerance: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let form = parse_form(form)?;
        let kernel = parse_kernel(kernel)?;
        let tol_ha = tolerance / self.energy_scale;
        let j_ha = j.map(|v| v / self.energy_scale);
        if let Some(jv) = j_ha {
            if !jv.is_finite() || jv < 0. {
                return Err(PyValueError::new_err("Hubbard J must be finite and non-negative"));
            }
        }
        let keys: Vec<&str> = self.symbols.iter().map(String::as_str).collect();
        let groups = self
            .response
            .hubbard_u(kernel, &keys, tol_ha)
            .map_err(error)?;
        let j_vec = self.j_response.as_deref();
        let out = PyDict::new(py);
        for g in groups {
            let j_val = match (form, j_ha) {
                (Form::Dudarev, _) => j_ha.unwrap_or(0.0),
                (Form::Liechtenstein, Some(jv)) => jv,
                (Form::Liechtenstein, None) => match j_vec {
                    Some(jv) => {
                        let vals: Vec<f64> = g.shells.iter().map(|&s| jv[s].max(0.0)).collect();
                        vals.iter().sum::<f64>() / vals.len() as f64
                    }
                    None => 0.0,
                },
            };
            if j_val > g.u.max(0.0) {
                return Err(PyValueError::new_err(format!(
                    "Hubbard J ({:.4} {}) exceeds extracted U ({:.4} {}) for {:?}",
                    j_val * self.energy_scale,
                    self.unit_name,
                    g.u * self.energy_scale,
                    self.unit_name,
                    g.key
                )));
            }
            let entry = PyDict::new(py);
            entry.set_item("U", g.u * self.energy_scale)?;
            entry.set_item("J", j_val * self.energy_scale)?;
            entry.set_item("U_eff", (g.u - j_val) * self.energy_scale)?;
            entry.set_item("spread", g.spread * self.energy_scale)?;
            entry.set_item("shells", g.shells)?;
            entry.set_item("form", form.name())?;
            entry.set_item("U_source", if self.self_consistent { "lr-scf" } else { "linear-response" })?;
            out.set_item(g.key, entry)?;
        }
        Ok(out)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("u_matrix", self.u_matrix(py)?)?;
        out.set_item("u_charge", self.u_charge(py)?)?;
        if let Some(uc) = self.u_cross(py)? {
            out.set_item("u_cross", uc)?;
        }
        out.set_item("chi0", self.chi0(py)?)?;
        out.set_item("chi", self.chi(py)?)?;
        out.set_item("alphas", self.alphas())?;
        if let Some(j) = self.j_response() {
            out.set_item("j_response", j)?;
        }
        out.set_item("shells", self.shells())?;
        out.set_item("channels", self.channels())?;
        out.set_item("symbols", self.symbols())?;
        out.set_item("self_consistent", self.self_consistent())?;
        out.set_item("nonlinearity", self.nonlinearity())?;
        out.set_item("asymmetry", self.asymmetry())?;
        out.set_item("floor", self.floor())?;
        out.set_item("condition_chi0", self.condition_chi0())?;
        out.set_item("condition_chi", self.condition_chi())?;
        out.set_item("condition", self.condition())?;
        out.set_item("energy_unit", self.energy_unit())?;
        Ok(out)
    }

    fn to_json(&self) -> PyResult<String> {
        let mat_vec = |m: &nalgebra::DMatrix<f64>, scale: f64| -> Vec<Vec<f64>> {
            (0..m.nrows())
                .map(|r| (0..m.ncols()).map(|c| m[(r, c)] * scale).collect())
                .collect()
        };
        let payload = serde_json::json!({
            "energy_unit": self.unit_name,
            "shells": self.response.shells,
            "channels": self.response.channels,
            "symbols": self.symbols,
            "self_consistent": self.self_consistent,
            "alphas": self.alphas(),
            "u_matrix": mat_vec(&self.u_matrix, self.energy_scale),
            "u_charge": mat_vec(&self.u_charge, self.energy_scale),
            "u_cross": self.u_cross.as_ref().map(|m| mat_vec(m, self.energy_scale)),
            "j_response": self.j_response(),
            "chi0": mat_vec(&self.chi0, 1.0 / self.energy_scale),
            "chi": mat_vec(&self.chi, 1.0 / self.energy_scale),
            "nonlinearity": self.response.nonlinearity,
            "asymmetry": self.response.asymmetry,
            "floor": self.response.sum_rule,
            "condition_chi0": self.response.condition_chi0,
            "condition_chi": self.response.condition_chi,
            "condition": self.condition(),
        });
        serde_json::to_string(&payload).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn keys(&self) -> Vec<&'static str> {
        let mut keys = vec![
            "u_matrix",
            "u_charge",
            "chi0",
            "chi",
            "alphas",
            "shells",
            "channels",
            "symbols",
            "self_consistent",
            "nonlinearity",
            "asymmetry",
            "floor",
            "condition_chi0",
            "condition_chi",
            "condition",
            "energy_unit",
        ];
        if self.u_cross.is_some() {
            keys.push("u_cross");
        }
        if self.j_response.is_some() {
            keys.push("j_response");
        }
        keys
    }

    fn __contains__(&self, key: &str) -> bool {
        match key {
            "u_matrix" | "u_charge" | "chi0" | "chi" | "alphas" | "shells" | "channels"
            | "symbols" | "self_consistent" | "nonlinearity" | "asymmetry" | "floor"
            | "condition_chi0" | "condition_chi" | "condition" | "energy_unit" => true,
            "u_cross" => self.u_cross.is_some(),
            "j_response" => self.j_response.is_some(),
            _ => false,
        }
    }

    #[pyo3(signature=(key, default=None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        key: &str,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        match self.__getitem__(py, key) {
            Ok(v) => Ok(v),
            Err(_) => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Py<PyAny>> {
        let dict = self.to_dict(py)?;
        match dict.get_item(key)? {
            Some(item) => Ok(item.unbind()),
            None => Err(PyKeyError::new_err(format!("{key:?}"))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HubbardResponse(shells={}, channels={}, self_consistent={}, unit={:?})",
            self.response.shells, self.response.channels, self.self_consistent, self.unit_name
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHubbardResponse>()?;
    m.add_function(wrap_pyfunction!(hubbard_validate, m)?)?;
    m.add_function(wrap_pyfunction!(hubbard_check_atoms, m)?)?;
    m.add_function(wrap_pyfunction!(hubbard_table, m)?)?;
    Ok(())
}
