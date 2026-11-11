//! Bindings of the Bloch methods: the Born-von Karman supercell Hamiltonian, its unfolded
//! mean-field bands, the Lehmann spectral function and the tetrahedron density of states.
//! Validated inputs are copied in, owned arrays come back; the numerics are in
//! `mandacaru_core::{molecular::periodic, quantum::spectral, periodic::tetrahedron}`.
use crate::arrays::{array, copy, error};
use crate::context::PyContext;
use crate::geometry::PyGeometry;
use crate::molecular::{PyIntegrals, PyMeanField, PyModel};
use crate::quantum::PyProblem;
use mandacaru_core::math::storage::Precision;
use mandacaru_core::molecular::bloch_path::bloch_path_spectral;
use mandacaru_core::molecular::periodic::{bloch_spectral, mean_field_bands};
use mandacaru_core::molecular::MolecularIntegrals;
use mandacaru_core::periodic::tetrahedron::TetrahedronMesh;
use mandacaru_core::quantum::spectral::Poles;
use nalgebra::Matrix3;
use numpy::{
    Complex64, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

/// The supercell integrals of the Gamma-centered `mesh` of `primitive`.
#[pyfunction]
#[pyo3(signature=(context, primitive, mesh, *, basis_spec, spacing_bohr, cholesky_tolerance=None, precision="float64"))]
pub fn supercell_integrals(
    py: Python<'_>,
    context: &PyContext,
    primitive: &PyGeometry,
    mesh: [usize; 3],
    basis_spec: &str,
    spacing_bohr: f64,
    cholesky_tolerance: Option<f64>,
    precision: &str,
) -> PyResult<PyIntegrals> {
    let precision = Precision::parse(precision).map_err(error)?;
    let geometry = primitive.inner.clone();
    let inner = py
        .detach(|| {
            MolecularIntegrals::supercell(
                &context.inner,
                &geometry,
                mesh,
                basis_spec,
                spacing_bohr,
                cholesky_tolerance,
                precision,
            )
        })
        .map_err(error)?;
    Ok(PyIntegrals {
        inner: Arc::new(inner),
    })
}

/// `{"mesh", "n_cells", "kpoints", "madelung_ha", "n_primitive", "orbital_cell",
/// "primitive_orbital", "primitive_atom", "primitive_l"}` of a supercell calculation.
#[pyfunction]
pub fn bloch_supercell<'py>(
    py: Python<'py>,
    integrals: &PyIntegrals,
) -> PyResult<Bound<'py, PyDict>> {
    let inner = &integrals.inner;
    let supercell = inner
        .supercell
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("these integrals are not a periodic supercell"))?;
    let cells = supercell.orbital_cells(&inner.basis).map_err(error)?;
    let k = supercell.kpoints();
    let out = PyDict::new(py);
    out.set_item("mesh", supercell.mesh.to_vec())?;
    out.set_item("n_cells", supercell.n_cells())?;
    out.set_item("kpoints", array(py, &[k.len(), 3], k.concat())?)?;
    out.set_item("madelung_ha", inner.madelung)?;
    out.set_item("n_primitive", cells.n_primitive)?;
    let mut primitive_atom = vec![0usize; cells.n_primitive];
    let mut primitive_l = vec![0usize; cells.n_primitive];
    for (index, orbital) in inner.basis.orbitals.iter().enumerate() {
        primitive_atom[cells.orbital[index]] = supercell.atoms[orbital.atom];
        primitive_l[cells.orbital[index]] = orbital.l;
    }
    out.set_item("orbital_cell", cells.cell)?;
    out.set_item("primitive_orbital", cells.orbital)?;
    out.set_item("primitive_atom", primitive_atom)?;
    out.set_item("primitive_l", primitive_l)?;
    Ok(out)
}

/// Mean-field bands of a supercell reference unfolded onto the primitive cell at `kpoints`
/// (fractional, primitive reciprocal lattice): `{"eigenvalues": (spins, nk, n),
/// "weights": (spins, nk, n bands, n orbitals)}`, Hartree.
#[pyfunction]
pub fn bloch_bands<'py>(
    py: Python<'py>,
    mean_field: &PyMeanField,
    kpoints: Vec<[f64; 3]>,
) -> PyResult<Bound<'py, PyDict>> {
    if kpoints.is_empty() {
        return Err(PyValueError::new_err("kpoints cannot be empty"));
    }
    let integrals = &mean_field.integrals;
    let bands = py
        .detach(|| mean_field_bands(integrals, &mean_field.inner, &kpoints))
        .map_err(error)?;
    let (s, nk, n) = (bands.spins, kpoints.len(), bands.n_primitive);
    let out = PyDict::new(py);
    out.set_item("eigenvalues", array(py, &[s, nk, n], bands.eigenvalues)?)?;
    out.set_item("weights", array(py, &[s, nk, n, n], bands.weights)?)?;
    Ok(out)
}

/// The state of `problem` in occupation form, for the Lehmann representation.
fn occupation_state(
    ctx: &mandacaru_core::Context,
    problem: &PyProblem,
    state: &PyReadonlyArray1<'_, Complex64>,
) -> PyResult<Vec<(u64, mandacaru_core::quantum::C64)>> {
    let cached = &problem.inner;
    if cached.taper.is_some() {
        return Err(PyValueError::new_err(
            "the spectral function needs the untapered register: use taper=False",
        ));
    }
    let basis = cached.basis.as_ref().ok_or_else(|| {
        PyValueError::new_err(
            "the spectral function needs a state vector; a register-free (shot-based) problem has none",
        )
    })?;
    let (values, _memory) = copy(ctx, state)?;
    if values.len() != basis.len() {
        return Err(PyValueError::new_err(
            "the state does not match the problem's simulation basis",
        ));
    }
    Ok(basis
        .indices
        .as_slice()
        .iter()
        .zip(&values)
        .filter(|(_, v)| v.norm_sqr() > 0.)
        .map(|(&ket, &v)| (cached.encoding.decode(ket), v))
        .collect())
}

/// The Lehmann spectral function of the correlated state `state` (in the problem's register)
/// at mesh `kpoints`. Every pole is one entry of the flat arrays `energies`, `weights`,
/// `k`, `orbital`, `spin`, `addition`; energies are excitation energies in Hartree (removal
/// poles at `E0 - E_n`).
#[pyfunction]
#[pyo3(signature=(mean_field, model, problem, state, kpoints, *, eta_ha, axis_ha=None, points=2001, krylov=200))]
#[allow(clippy::too_many_arguments)]
pub fn bloch_spectral_function<'py>(
    py: Python<'py>,
    mean_field: &PyMeanField,
    model: &PyModel,
    problem: &PyProblem,
    state: PyReadonlyArray1<'py, Complex64>,
    kpoints: Vec<[f64; 3]>,
    eta_ha: f64,
    axis_ha: Option<PyReadonlyArray1<'py, f64>>,
    points: usize,
    krylov: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = &mean_field.integrals.context;
    let occupations = occupation_state(ctx, problem, &state)?;
    let integrals = &mean_field.integrals;
    let given = match &axis_ha {
        Some(a) => Some(copy(ctx, a)?),
        None => None,
    };
    let (spectral, axis, values) = py
        .detach(|| -> mandacaru_core::Result<_> {
            let spectral = bloch_spectral(
                ctx,
                integrals,
                &mean_field.inner,
                &model.source,
                &occupations,
                &kpoints,
                krylov,
            )?;
            let axis = match &given {
                Some((a, _)) => a.clone(),
                None => spectral.default_axis(eta_ha, points)?,
            };
            let values = spectral.broadened(&axis, eta_ha)?;
            Ok((spectral, axis, values))
        })
        .map_err(error)?;
    spectral_dict(py, &spectral, axis, values)
}

/// The Lehmann spectral function along a band path: the mesh points of the calculation's
/// `kpts` that lie on the path (named points, or explicit `path_points`), one row each, in path
/// order. With `irreducible` every row is resolved at the irreducible representative of its
/// star and the symmetry of the state is audited. The result is the dictionary of
/// `bloch_spectral_function` over the path's k-points, plus `kpath` (the selection) and `audit`.
#[pyfunction]
#[pyo3(signature=(mean_field, model, problem, state, cell, pbc, *, eta_ha, path=None, path_points=None, irreducible=false, positions=None, numbers=None, symprec=1e-5, axis_ha=None, points=2001, krylov=200))]
#[allow(clippy::too_many_arguments)]
pub fn bloch_path_spectral_function<'py>(
    py: Python<'py>,
    mean_field: &PyMeanField,
    model: &PyModel,
    problem: &PyProblem,
    state: PyReadonlyArray1<'py, Complex64>,
    cell: PyReadonlyArray2<'py, f64>,
    pbc: [bool; 3],
    eta_ha: f64,
    path: Option<&str>,
    path_points: Option<Vec<(String, [f64; 3])>>,
    irreducible: bool,
    positions: Option<PyReadonlyArray2<'py, f64>>,
    numbers: Option<PyReadonlyArray1<'py, i64>>,
    symprec: f64,
    axis_ha: Option<PyReadonlyArray1<'py, f64>>,
    points: usize,
    krylov: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let integrals = &mean_field.integrals;
    let ctx = &integrals.context;
    let supercell = integrals.supercell.as_ref().ok_or_else(|| {
        PyValueError::new_err("a spectral function along a path needs a periodic supercell calculation")
    })?;
    let occupations = occupation_state(ctx, problem, &state)?;
    let lattice = crate::kpath::lattice(&cell)?;
    let (band, metric) = crate::kpath::band_path_of(&lattice, pbc, path, path_points)?;
    let given = match &axis_ha {
        Some(a) => Some(copy(ctx, a)?),
        None => None,
    };
    let reduction = if irreducible {
        let (Some(positions), Some(numbers)) = (&positions, &numbers) else {
            return Err(PyValueError::new_err(
                "an irreducible reduction needs the atoms' positions and numbers",
            ));
        };
        let (flat, _flat) = copy(ctx, positions)?;
        let (numbers, _numbers) = copy(ctx, numbers)?;
        let labels = numbers
            .iter()
            .map(|&z| {
                u32::try_from(z)
                    .map_err(|_| PyValueError::new_err("atomic numbers must be positive"))
            })
            .collect::<PyResult<Vec<u32>>>()?;
        Some((
            flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect::<Vec<_>>(),
            labels,
        ))
    } else {
        None
    };
    let (selected, spectral, axis, values, audit) = py
        .detach(|| -> mandacaru_core::Result<_> {
            let mesh = supercell.kpoints();
            let selected = mandacaru_core::periodic::kpath::select(&band, &metric, &mesh)?;
            let zone = match &reduction {
                Some((fractional, labels)) => Some(mandacaru_core::periodic::kpath::irreducible_zone(
                    ctx,
                    &lattice,
                    fractional,
                    labels,
                    supercell.mesh,
                    true,
                    symprec,
                )?),
                None => None,
            };
            let result = bloch_path_spectral(
                ctx,
                integrals,
                &mean_field.inner,
                &model.source,
                &occupations,
                &selected,
                zone.as_ref(),
                eta_ha,
                given.as_ref().map(|(a, _)| a.as_slice()),
                points,
                krylov,
            )?;
            let values = result.spectral.broadened(&result.axis, eta_ha)?;
            Ok((selected, result.spectral, result.axis, values, result.audit))
        })
        .map_err(error)?;
    let out = spectral_dict(py, &spectral, axis, values)?;
    out.set_item("kpath", crate::kpath::path_dict(py, &selected)?)?;
    match audit {
        Some(audit) => {
            let item = PyDict::new(py);
            item.set_item("representative", audit.representative.to_vec())?;
            item.set_item("image", audit.image.to_vec())?;
            item.set_item("star_size", audit.star_size)?;
            item.set_item("residual", audit.residual)?;
            item.set_item("symmetric", audit.symmetric)?;
            out.set_item("audit", item)?;
        }
        None => out.set_item("audit", py.None())?,
    }
    Ok(out)
}

/// The arrays of a Lehmann spectral function at `spectral.kpoints`.
fn spectral_dict<'py>(
    py: Python<'py>,
    spectral: &mandacaru_core::molecular::periodic::BlochSpectral,
    axis: Vec<f64>,
    values: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let n = spectral.n_primitive;
    let nk = spectral.kpoints.len();
    let (dominant_energy, dominant_weight, sum_rule) = spectral.summary();
    let (mut energies, mut weights) = (Vec::new(), Vec::new());
    let (mut k_index, mut orbital, mut spin, mut addition) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut push = |poles: &Poles, row: usize, s: usize, add: bool| {
        for (e, w) in poles.energies.iter().zip(&poles.weights) {
            energies.push(*e);
            weights.push(*w);
            k_index.push(row / n);
            orbital.push(row % n);
            spin.push(s);
            addition.push(add);
        }
    };
    for (row, (add, remove)) in spectral
        .parts
        .addition
        .iter()
        .zip(&spectral.parts.removal)
        .enumerate()
    {
        for s in 0..2 {
            push(&add[s], row, s, true);
            push(&remove[s], row, s, false);
        }
    }
    let p = energies.len();
    let out = PyDict::new(py);
    out.set_item("ground_energy_ha", spectral.parts.ground_energy)?;
    out.set_item("axis_ha", array(py, &[axis.len()], axis.clone())?)?;
    out.set_item("values", array(py, &[nk, axis.len()], values)?)?;
    out.set_item("kpoints", array(py, &[nk, 3], spectral.kpoints.concat())?)?;
    out.set_item("n_primitive", n)?;
    out.set_item("energies", array(py, &[p], energies)?)?;
    out.set_item("weights", array(py, &[p], weights)?)?;
    out.set_item("k", array(py, &[p], k_index)?)?;
    out.set_item("orbital", array(py, &[p], orbital)?)?;
    out.set_item("spin", array(py, &[p], spin)?)?;
    out.set_item("addition", array(py, &[p], addition)?)?;
    out.set_item("dominant_energy", array(py, &[nk, n, 2], dominant_energy)?)?;
    out.set_item("dominant_weight", array(py, &[nk, n, 2], dominant_weight)?)?;
    out.set_item("sum_rule", array(py, &[nk, n, 2], sum_rule)?)?;
    let (removal_edge, addition_edge) = spectral.edges(1e-8);
    out.set_item("removal_edge_ha", removal_edge)?;
    out.set_item("addition_edge_ha", addition_edge)?;
    Ok(out)
}

/// Linear-tetrahedron density of states on the full Gamma-centered `shape` mesh.
/// `eigenvalues`: `(spins, nk, bands)` at `kpoints` (any order); `projections`: optional
/// `(spins, nk, bands, channels)`. Returns `{"dos", "integrated"}` and, with projections,
/// `"projected"` `(axis, channels)`, each summed over spins and times `degeneracy`.
#[pyfunction]
#[pyo3(signature=(context, shape, reciprocal, kpoints, eigenvalues, axis, *, projections=None, degeneracy=1.0))]
#[allow(clippy::too_many_arguments)]
pub fn tetrahedron_dos<'py>(
    py: Python<'py>,
    context: &PyContext,
    shape: [usize; 3],
    reciprocal: PyReadonlyArray2<'py, f64>,
    kpoints: Vec<[f64; 3]>,
    eigenvalues: PyReadonlyArray3<'py, f64>,
    axis: PyReadonlyArray1<'py, f64>,
    projections: Option<PyReadonlyArray4<'py, f64>>,
    degeneracy: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if reciprocal.shape() != [3, 3] {
        return Err(PyValueError::new_err(
            "reciprocal must be 3 x 3 (lattice vectors as rows)",
        ));
    }
    let [spins, nk, bands]: [usize; 3] = eigenvalues.shape().try_into().unwrap();
    let ctx = &context.inner;
    let (b, _b) = copy(ctx, &reciprocal)?;
    let (levels, _levels) = copy(ctx, &eigenvalues)?;
    let (x, _x) = copy(ctx, &axis)?;
    let projected = match &projections {
        Some(p) => {
            let s = p.shape();
            if s[0] != spins || s[1] != nk || s[2] != bands {
                return Err(PyValueError::new_err(
                    "projections must be (spins, nk, bands, channels) like the eigenvalues",
                ));
            }
            Some((copy(ctx, p)?, s[3]))
        }
        None => None,
    };
    // Rows of `reciprocal` are the reciprocal vectors; the mesh takes them as columns.
    let lattice = Matrix3::from_row_slice(&b).transpose();
    let result = py
        .detach(|| {
            TetrahedronMesh::new(shape, &lattice)?.spin_summed(
                &kpoints,
                &levels,
                spins,
                bands,
                projected.as_ref().map(|((weights, _), channels)| (&weights[..], *channels)),
                degeneracy,
                &x,
            )
        })
        .map_err(error)?;
    let out = PyDict::new(py);
    out.set_item("dos", array(py, &[x.len()], result.0)?)?;
    out.set_item("integrated", array(py, &[x.len()], result.1)?)?;
    if let (Some(p), Some((_, c))) = (result.2, projected) {
        out.set_item("projected", array(py, &[x.len(), c], p)?)?;
    }
    Ok(out)
}
