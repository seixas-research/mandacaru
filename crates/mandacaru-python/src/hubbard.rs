//! Bindings of the DFT+U option and shell data. Arrays are copied in and returned owned.
//! The option is read, validated and resolved by `mandacaru_core::models::hubbard_request`, the
//! table by `mandacaru_core::periodic::hubbard_table`, the weights by `paw::shell`; the views of
//! a converged run are read from the owned `Report`. Nothing is decided or computed here.
use super::{array, copy, error, PyContext};
use mandacaru_core::{
    models::{hubbard::shell_letter, hubbard_request::Request},
    periodic::{
        hubbard_table::{
            build, check_loaded as check_table, ChannelData, Layout, Projector, Species,
        },
        Crystal, PeriodicResult,
    },
    product, Reservation,
};
use numpy::{Complex64, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
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
/// The radial grid of a species and its channels by `l`.
type SpeciesArrays<'py> = (
    PyReadonlyArray1<'py, f64>,
    HashMap<usize, ChannelArrays<'py>>,
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
    for (r, channels) in &species {
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
/// `shell` letter, `u_ha`, `j_ha`, `u_eff_ha`), the occupation matrices `n` (`spins x (2l+1) x (2l+1)`, complex), their
/// traces, and `energy_ha`, the shell's `E_U`. `None` for a run without `hubbard`.
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
        let mut values = Vec::with_capacity(crystal.spins * p * p);
        let mut traces = Vec::with_capacity(crystal.spins);
        for spin in 0..crystal.spins {
            let n = report.occupation(spin, index);
            traces.push((0..p).map(|a| n[a * p + a].re).sum::<f64>());
            values.extend_from_slice(n);
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
        let _hold = crystal
            .context
            .reserve(product(&[values.len(), 16]).map_err(error)?)
            .map_err(error)?;
        let n = values.iter().map(|z| Complex64::new(z.re, z.im)).collect();
        item.set_item("n", array(py, &[crystal.spins, p, p], n)?)?;
        out.append(item)?;
    }
    Ok(Some(out))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hubbard_validate, m)?)?;
    m.add_function(wrap_pyfunction!(hubbard_table, m)?)?;
    Ok(())
}
