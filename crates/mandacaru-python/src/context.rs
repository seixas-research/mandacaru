//! Execution context and dispatch to Python-independent scientific operations.
use super::{
    advanced, array, atom, complex, complex_array, copy, error, models, molecular, paw, periodic,
    persistence, quantum, Array, PyGeometry, PyGrid, PySpatialGrid, Workspace,
};
use mandacaru_core::periodic::{
    ewald::Ewald,
    spectrum::{self, Smearing},
    Crystal,
};
use mandacaru_core::{kernels, product, Context, Error};
use numpy::{
    Complex64, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use periodic::PyCrystal;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use std::sync::Arc;
type GateRecord = (String, usize, Option<usize>, Option<f64>);
#[pyclass(name = "Context", module = "mandacaru._native", frozen)]
pub(crate) struct PyContext {
    pub(crate) inner: Context,
}
#[pymethods]
impl PyContext {
    #[pyo3(signature=(l,*,real=false))]
    fn spin_orbit_angular<'py>(
        &self,
        py: Python<'py>,
        l: usize,
        real: bool,
    ) -> PyResult<Array<'py, Complex64>> {
        let out = py
            .detach(|| mandacaru_core::models::spin_orbit::angular(&self.inner, l, real))
            .map_err(error)?;
        let size = 2 * (2 * l + 1);
        array(py, &[size, size], out.as_slice().to_vec())
    }
    #[pyo3(signature=(left,right,channels,*,hermitian=false))]
    fn spin_orbit_pair<'py>(
        &self,
        py: Python<'py>,
        left: PyReadonlyArray2<'py, Complex64>,
        right: PyReadonlyArray2<'py, Complex64>,
        channels: Vec<(usize, bool, Vec<usize>, PyReadonlyArray2<'py, Complex64>)>,
        hermitian: bool,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::models::spin_orbit::{self, Channel, Projectors};
        let rows = left.shape()[0];
        let columns = right.shape()[0];
        let count = left.shape()[1];
        if right.shape()[1] != count {
            return Err(PyValueError::new_err(
                "left and right spin-orbit projections need matching projector counts",
            ));
        }
        let (left, _left) = copy(&self.inner, &left)?;
        let (right, _right) = copy(&self.inner, &right)?;
        let mut owned = Vec::with_capacity(channels.len());
        let _channels = self
            .inner
            .reserve(product(&[channels.len(), 128]).map_err(error)?)
            .map_err(error)?;
        for (l, real, indices, coupling) in channels {
            super::layout(&coupling)?;
            if coupling.shape()[0] != coupling.shape()[1] {
                return Err(PyValueError::new_err(
                    "spin-orbit radial coupling must be square",
                ));
            }
            owned.push(Channel {
                l,
                real,
                indices: self.inner.copy_slice(&indices).map_err(error)?,
                radial: coupling.shape()[0],
                coupling: self.inner.copy_slice(coupling.as_slice()?).map_err(error)?,
            });
        }
        let out = py
            .detach(|| {
                spin_orbit::pair(
                    &self.inner,
                    Projectors {
                        left: &left,
                        right: &right,
                        rows,
                        columns,
                        count,
                    },
                    &owned,
                    hermitian,
                )
            })
            .map_err(error)?;
        array(py, &[2 * rows, 2 * columns], out.as_slice().to_vec())
    }
    fn remove_force_translation<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let atoms = values.shape()[0];
        if values.shape()[1] != 3 {
            return Err(PyValueError::new_err("forces must have shape (atoms,3)"));
        }
        let (values, _memory) = copy(&self.inner, &values)?;
        let projected = py
            .detach(|| mandacaru_core::math::rigid::remove_translation(&self.inner, &values))
            .map_err(error)?;
        array(py, &[atoms, 3], projected.as_slice().to_vec())
    }
    #[pyo3(signature=(data,*,group_by="molecule",fractions=vec![0.25,0.5,1.0],seed=0,folds=None,gnn_epochs=100,mixing=0.3,temperature=1.0))]
    fn fit_proposal(
        &self,
        py: Python<'_>,
        data: &str,
        group_by: &str,
        fractions: Vec<f64>,
        seed: u64,
        folds: Option<usize>,
        gnn_epochs: usize,
        mixing: f64,
        temperature: f64,
    ) -> PyResult<String> {
        use mandacaru_core::quantum::search::learning::training::{self, Options};
        py.detach(|| {
            let options = Options {
                fractions,
                seed,
                folds,
                epochs: gnn_epochs,
                mixing,
                temperature,
                ..Default::default()
            };
            training::fit_json(&self.inner, data, group_by, &options)
        })
        .map_err(error)
    }
    #[pyo3(signature=(qubits,num_particles=None))]
    fn active_space_dimension(
        &self,
        qubits: usize,
        num_particles: Option<[usize; 2]>,
    ) -> PyResult<usize> {
        mandacaru_core::quantum::expressibility::dimension(qubits, num_particles).map_err(error)
    }
    fn expressibility_kl(
        &self,
        py: Python<'_>,
        fidelities: PyReadonlyArray1<'_, f64>,
        bins: usize,
        dimension: usize,
    ) -> PyResult<f64> {
        let (values, _memory) = super::copy(&self.inner, &fidelities)?;
        py.detach(|| {
            mandacaru_core::quantum::expressibility::kl(&self.inner, &values, bins, dimension)
        })
        .map_err(error)
    }
    #[pyo3(signature=(geometry,*,basis_spec="{\"name\":\"HAO\"}"))]
    fn basis_size(
        &self,
        py: Python<'_>,
        geometry: &PyGeometry,
        basis_spec: &str,
    ) -> PyResult<usize> {
        py.detach(|| {
            mandacaru_core::basis::Basis::build(&self.inner, &geometry.inner, basis_spec)
                .map(|basis| basis.orbitals.len())
        })
        .map_err(error)
    }
    #[pyo3(signature=(levels,weights,axis,*,width_ha=0.01,degeneracy=2.))]
    fn density_of_states<'py>(
        &self,
        py: Python<'py>,
        levels: PyReadonlyArray3<'py, f64>,
        weights: PyReadonlyArray1<'py, f64>,
        axis: PyReadonlyArray1<'py, f64>,
        width_ha: f64,
        degeneracy: f64,
    ) -> PyResult<Array<'py, f64>> {
        let shape: [usize; 3] = levels.shape().try_into().unwrap();
        let (levels, _levels) = copy(&self.inner, &levels)?;
        let (weights, _weights) = copy(&self.inner, &weights)?;
        let (axis, _axis) = copy(&self.inner, &axis)?;
        let out = py
            .detach(|| {
                mandacaru_core::models::spectrum::gaussian_dos(
                    &self.inner,
                    mandacaru_core::models::spectrum::DosInput {
                        shape,
                        levels: &levels,
                        weights: &weights,
                    },
                    &axis,
                    width_ha,
                    degeneracy,
                )
            })
            .map_err(error)?;
        array(py, &[axis.len()], out.as_slice().to_vec())
    }
    #[pyo3(signature=(operator,n_spatial_orbitals,num_particles,*,mapping="jordan_wigner",sector=true,include_singles=true,sparse=false,pool="fermionic",precision="float64"))]
    fn quantum_problem(
        &self,
        py: Python<'_>,
        operator: &quantum::PyOperator,
        n_spatial_orbitals: usize,
        num_particles: [usize; 2],
        mapping: &str,
        sector: bool,
        include_singles: bool,
        sparse: bool,
        pool: &str,
        precision: &str,
    ) -> PyResult<quantum::PyProblem> {
        let precision = quantum::parse_precision(precision)?;
        let inner = py
            .detach(|| {
                let modes = n_spatial_orbitals.checked_mul(2).ok_or(Error::Overflow)?;
                let encoding =
                    mandacaru_core::quantum::Encoding::new(mapping, modes, num_particles)?;
                let operator = operator.inner.copy_into(&self.inner)?;
                let mut problem = mandacaru_core::quantum::CachedProblem::new(
                    &self.inner,
                    encoding,
                    operator,
                    sector,
                    false,
                    include_singles,
                )?
                .with_precision(precision);
                if pool != "fermionic" {
                    problem.use_pool(pool, include_singles)?;
                }
                if sparse {
                    problem.sparse = Some(mandacaru_core::quantum::sparse::SparseMatrix::new(
                        &self.inner,
                        &problem.basis,
                        &problem.hamiltonian,
                    )?);
                }
                Ok::<_, Error>(problem)
            })
            .map_err(error)?;
        Ok(quantum::PyProblem { inner })
    }
    #[pyo3(signature=(variables,*,omega=0.11))]
    fn exchange_hole<'py>(
        &self,
        py: Python<'py>,
        variables: PyReadonlyArray2<'py, f64>,
        omega: f64,
    ) -> PyResult<Array<'py, f64>> {
        if variables.shape()[1] != 2 {
            return Err(PyValueError::new_err("exchange hole requires shape (n,2)"));
        }
        let rows = variables.shape()[0];
        let (values, _memory) = copy(&self.inner, &variables)?;
        let out = py
            .detach(|| mandacaru_core::models::xc::exchange_hole_batch(&self.inner, &values, omega))
            .map_err(error)?;
        array(py, &[rows, 3], out.as_slice().to_vec())
    }
    #[pyo3(signature=(variables,*,omega=0.11,exact_fraction=0.25))]
    fn hybrid_points<'py>(
        &self,
        py: Python<'py>,
        variables: PyReadonlyArray2<'py, f64>,
        omega: f64,
        exact_fraction: f64,
    ) -> PyResult<Array<'py, f64>> {
        if variables.shape()[1] != 7 {
            return Err(PyValueError::new_err("hybrid points require shape (n,7)"));
        }
        let rows = variables.shape()[0];
        let (values, _memory) = copy(&self.inner, &variables)?;
        let out = py
            .detach(|| {
                mandacaru_core::models::xc::hybrid_batch(
                    &self.inner,
                    &values,
                    omega,
                    exact_fraction,
                )
            })
            .map_err(error)?;
        array(py, &[rows, 8], out.as_slice().to_vec())
    }
    #[pyo3(signature=(grid,density,sigma,*,b=11.95,c=0.0093,q_min=1e-4,q_cut=0.5,nodes=16,padding=2))]
    fn nonlocal_correlation<'py>(
        &self,
        py: Python<'py>,
        grid: &PySpatialGrid,
        density: PyReadonlyArray1<'py, f64>,
        sigma: PyReadonlyArray1<'py, f64>,
        b: f64,
        c: f64,
        q_min: f64,
        q_cut: f64,
        nodes: usize,
        padding: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (rho, _rho_memory) = copy(&self.inner, &density)?;
        let (sigma, _sigma_memory) = copy(&self.inner, &sigma)?;
        let options = mandacaru_core::models::nonlocal::Options {
            b,
            c,
            q_min,
            q_cut,
            nodes,
            padding,
        };
        let terms = py
            .detach(|| {
                mandacaru_core::models::nonlocal::calculate(
                    &self.inner,
                    &grid.inner,
                    &rho,
                    &sigma,
                    options,
                )
            })
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item("energy", terms.energy)?;
        out.set_item(
            "d_density",
            array(py, &[rho.len()], terms.density.as_slice().to_vec())?,
        )?;
        out.set_item(
            "d_sigma",
            array(py, &[rho.len()], terms.sigma.as_slice().to_vec())?,
        )?;
        Ok(out)
    }
    #[pyo3(signature=(functional,variables,*,relativistic=false))]
    fn xc_points<'py>(
        &self,
        py: Python<'py>,
        functional: &str,
        variables: PyReadonlyArray2<'py, f64>,
        relativistic: bool,
    ) -> PyResult<Array<'py, f64>> {
        if variables.shape()[1] != 7 {
            return Err(PyValueError::new_err("point XC requires shape (n,7)"));
        }
        let rows = variables.shape()[0];
        let (variables, _memory) = copy(&self.inner, &variables)?;
        let out = py
            .detach(|| {
                mandacaru_core::models::xc::point_batch(
                    &self.inner,
                    functional,
                    &variables,
                    relativistic,
                )
            })
            .map_err(error)?;
        array(py, &[rows, 8], out.as_slice().to_vec())
    }
    #[pyo3(signature=(path,data,origin,step,numbers,positions,*,format="cube",charges=None,cell=None,periodic=false,comment=""))]
    fn write_field(
        &self,
        py: Python<'_>,
        path: &str,
        data: PyReadonlyArrayDyn<'_, f64>,
        origin: [f64; 3],
        step: [[f64; 3]; 3],
        numbers: Vec<u8>,
        positions: Vec<[f64; 3]>,
        format: &str,
        charges: Option<Vec<f64>>,
        cell: Option<[[f64; 3]; 3]>,
        periodic: bool,
        comment: &str,
    ) -> PyResult<()> {
        let shape: [usize; 3] = data
            .shape()
            .try_into()
            .map_err(|_| PyValueError::new_err("field must be three-dimensional"))?;
        let (data, _memory) = copy(&self.inner, &data)?;
        let _metadata = self
            .inner
            .reserve(product(&[numbers.len(), 64]).map_err(error)?)
            .map_err(error)?;
        let charges = charges.unwrap_or_else(|| numbers.iter().map(|&z| z as f64).collect());
        py.detach(|| {
            mandacaru_core::volumetric::Field {
                shape,
                data: &data,
                origin,
                step,
                numbers: &numbers,
                positions: &positions,
                charges: &charges,
            }
            .write(
                &self.inner,
                std::path::Path::new(path),
                format,
                comment,
                cell,
                periodic,
            )
        })
        .map_err(error)
    }
    #[pyo3(signature=(path,energies,reciprocal_cell,fermi_level,*,comment="Mandacaru band energies"))]
    fn write_bands(
        &self,
        py: Python<'_>,
        path: &str,
        energies: PyReadonlyArrayDyn<'_, f64>,
        reciprocal_cell: [[f64; 3]; 3],
        fermi_level: f64,
        comment: &str,
    ) -> PyResult<()> {
        let shape: [usize; 4] = energies
            .shape()
            .try_into()
            .map_err(|_| PyValueError::new_err("band grid must be four-dimensional"))?;
        let (data, _memory) = copy(&self.inner, &energies)?;
        py.detach(|| {
            mandacaru_core::volumetric::bxsf(
                &self.inner,
                std::path::Path::new(path),
                shape,
                &data,
                reciprocal_cell,
                fermi_level,
                comment,
            )
        })
        .map_err(error)
    }
    fn read_snapshot(
        &self,
        py: Python<'_>,
        path: &str,
        kind: &str,
    ) -> PyResult<persistence::PySnapshot> {
        let inner = py
            .detach(|| {
                mandacaru_core::persistence::Document::read(
                    &self.inner,
                    std::path::Path::new(path),
                    kind,
                )
            })
            .map_err(error)?;
        Ok(persistence::PySnapshot {
            context: self.inner.clone(),
            inner,
        })
    }
    fn parse_snapshot(
        &self,
        py: Python<'_>,
        text: &str,
        kind: &str,
    ) -> PyResult<persistence::PySnapshot> {
        let inner = py
            .detach(|| mandacaru_core::persistence::Document::parse(&self.inner, text, kind))
            .map_err(error)?;
        Ok(persistence::PySnapshot {
            context: self.inner.clone(),
            inner,
        })
    }
    #[pyo3(signature=(labels,weights=None))]
    fn qwc_groups(
        &self,
        py: Python<'_>,
        labels: Vec<String>,
        weights: Option<Vec<f64>>,
    ) -> PyResult<Vec<(String, Vec<usize>)>> {
        let out = py
            .detach(|| mandacaru_core::execution::qwc(&self.inner, &labels, weights.as_deref()))
            .map_err(error)?;
        Ok(out.groups)
    }
    fn count_expectations(
        &self,
        py: Python<'_>,
        labels: Vec<String>,
        counts: Vec<(String, f64)>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| mandacaru_core::execution::expectations(&self.inner, &labels, &counts))
            .map_err(error)
    }
    fn compile_circuit(
        &self,
        py: Python<'_>,
        num_qubits: usize,
        occupied: Vec<usize>,
        generators: Vec<Vec<(String, f64, f64)>>,
        parameters: Vec<f64>,
    ) -> PyResult<Vec<GateRecord>> {
        let out = py
            .detach(|| {
                mandacaru_core::execution::compile(
                    &self.inner,
                    num_qubits,
                    &occupied,
                    &generators,
                    &parameters,
                )
            })
            .map_err(error)?;
        Ok(out
            .gates
            .into_iter()
            .map(|g| (g.name.to_owned(), g.first, g.second, g.angle))
            .collect())
    }
    #[pyo3(signature=(*,jobs=None,total_shots=None))]
    fn run_ledger(&self, jobs: Option<u64>, total_shots: Option<u64>) -> persistence::PyLedger {
        persistence::PyLedger {
            inner: std::sync::Mutex::new(mandacaru_core::execution::RunLedger::new(
                &self.inner,
                jobs,
                total_shots,
            )),
        }
    }
    #[pyo3(signature=(links,*,tolerance=1e-10))]
    fn wilson_loop<'py>(
        &self,
        py: Python<'py>,
        links: PyReadonlyArrayDyn<'py, Complex64>,
        tolerance: f64,
    ) -> PyResult<(f64, f64)> {
        let shape = links.shape();
        if shape.len() != 3 || shape[1] != shape[2] {
            return Err(PyValueError::new_err(
                "Wilson links need shape (n_links,n_occupied,n_occupied)",
            ));
        }
        let count = shape[0];
        let occupied = shape[1];
        let (values, _memory) = copy(&self.inner, &links)?;
        py.detach(|| {
            mandacaru_core::periodic::analysis::wilson_loop(
                &self.inner,
                &values,
                count,
                occupied,
                tolerance,
            )
        })
        .map_err(error)
    }

    fn periodic_problem(
        &self,
        py: Python<'_>,
        metadata_json: &str,
        arrays: &Bound<'_, PyDict>,
    ) -> PyResult<PyCrystal> {
        let snapshot_json = metadata_json;
        let arrays = periodic::input_arrays(&self.inner, arrays, metadata_json, None)?;
        let _input = self.inner.reserve(snapshot_json.len()).map_err(error)?;
        let owned = snapshot_json.to_owned();
        let inner = py
            .detach(|| Crystal::from_arrays(&self.inner, &owned, arrays))
            .map_err(error)?;
        Ok(PyCrystal {
            inner: Arc::new(inner),
        })
    }
    fn load_crystal(&self, py: Python<'_>, path: &str) -> PyResult<PyCrystal> {
        let inner = py
            .detach(|| Crystal::load(&self.inner, path))
            .map_err(error)?;
        Ok(PyCrystal {
            inner: Arc::new(inner),
        })
    }
    #[pyo3(signature=(size,*,gamma=false,rotations=Vec::new(),time_reversal=true))]
    fn kmesh<'py>(
        &self,
        py: Python<'py>,
        size: [usize; 3],
        gamma: bool,
        rotations: Vec<[i32; 9]>,
        time_reversal: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mesh = py
            .detach(|| spectrum::mesh(&self.inner, size, gamma, &rotations, time_reversal))
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item(
            "full",
            array(
                py,
                &[mesh.full.len(), 3],
                mesh.full.iter().flatten().copied().collect(),
            )?,
        )?;
        out.set_item(
            "points",
            array(
                py,
                &[mesh.points.len(), 3],
                mesh.points.iter().flatten().copied().collect(),
            )?,
        )?;
        out.set_item("weights", mesh.weights.clone())?;
        out.set_item("mapping", mesh.mapping.clone())?;
        Ok(out)
    }
    #[pyo3(signature=(eigenvalues,weights,electrons,*,method="fermi-dirac",width_ha=0.003674932217565499,degeneracy=2.))]
    fn fermi_distribution<'py>(
        &self,
        py: Python<'py>,
        eigenvalues: Vec<Vec<f64>>,
        weights: Vec<f64>,
        electrons: f64,
        method: &str,
        width_ha: f64,
        degeneracy: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let method = Smearing::parse(method).map_err(error)?;
        let (mu, occupations, entropy) = py
            .detach(|| {
                let mu = spectrum::fermi(
                    &eigenvalues,
                    &weights,
                    electrons,
                    method,
                    width_ha,
                    degeneracy,
                )?;
                let occupations = eigenvalues
                    .iter()
                    .map(|e| {
                        e.iter()
                            .map(|e| degeneracy * method.occupation((e - mu) / width_ha))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let entropy = degeneracy
                    * eigenvalues
                        .iter()
                        .zip(&weights)
                        .map(|(e, w)| {
                            w * e
                                .iter()
                                .map(|e| method.entropy((e - mu) / width_ha))
                                .sum::<f64>()
                        })
                        .sum::<f64>();
                Ok::<_, mandacaru_core::Error>((mu, occupations, entropy))
            })
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item("fermi_level", mu)?;
        out.set_item("occupations", occupations)?;
        out.set_item("entropy", entropy)?;
        Ok(out)
    }
    #[pyo3(signature=(positions_bohr,charges,cell_bohr,*,accuracy=1e-12,eta=None,points_bohr=None,softening_bohr=0.))]
    fn ewald<'py>(
        &self,
        py: Python<'py>,
        positions_bohr: PyReadonlyArray2<'py, f64>,
        charges: PyReadonlyArray1<'py, f64>,
        cell_bohr: PyReadonlyArray2<'py, f64>,
        accuracy: f64,
        eta: Option<f64>,
        points_bohr: Option<PyReadonlyArray2<'py, f64>>,
        softening_bohr: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        if positions_bohr.shape()[1] != 3
            || positions_bohr.shape()[0] != charges.len()
            || cell_bohr.shape() != [3, 3]
            || points_bohr.as_ref().is_some_and(|p| p.shape()[1] != 3)
        {
            return Err(PyValueError::new_err(
                "Ewald positions/points need (N,3), charges (N,) and cell (3,3)",
            ));
        }
        let (positions, _p) = copy(&self.inner, &positions_bohr)?;
        let (charges, _q) = copy(&self.inner, &charges)?;
        let (cell, _c) = copy(&self.inner, &cell_bohr)?;
        let points = points_bohr
            .as_ref()
            .map(|p| copy(&self.inner, p))
            .transpose()?;
        let (result, potential) = py
            .detach(|| {
                let _coordinates = self.inner.reserve(product(&[
                    positions
                        .len()
                        .checked_add(points.as_ref().map_or(0, |v| v.0.len()))
                        .ok_or(mandacaru_core::Error::Overflow)?,
                    8,
                ])?)?;
                let positions: Vec<[f64; 3]> = positions
                    .chunks_exact(3)
                    .map(|r| r.try_into().unwrap())
                    .collect();
                let system = Ewald::new(
                    &self.inner,
                    &positions,
                    &charges,
                    cell.as_slice().try_into().unwrap(),
                    accuracy,
                    eta,
                )?;
                let result = system.evaluate(&self.inner)?;
                let potential = points
                    .as_ref()
                    .map(|(v, _)| {
                        let points: Vec<[f64; 3]> =
                            v.chunks_exact(3).map(|r| r.try_into().unwrap()).collect();
                        system.potential(&self.inner, &points, softening_bohr)
                    })
                    .transpose()?;
                Ok::<_, mandacaru_core::Error>((result, potential))
            })
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item("energy", result.energy)?;
        out.set_item(
            "forces",
            array(py, &[charges.len(), 3], result.forces.as_slice().to_vec())?,
        )?;
        out.set_item("stress", array(py, &[3, 3], result.stress.to_vec())?)?;
        if let Some(potential) = potential {
            out.set_item(
                "potential",
                array(
                    py,
                    &[potential.as_slice().len()],
                    potential.as_slice().to_vec(),
                )?,
            )?;
        }
        Ok(out)
    }

    #[pyo3(signature=(atomic_number,*,points=4000,r_max=25.,max_iterations=200,tolerance=1e-6,mixing=0.3,configuration=None,confinement=None,functional="lda",relativity="none",polish=12))]
    fn solve_atom<'py>(
        &self,
        py: Python<'py>,
        atomic_number: u8,
        points: usize,
        r_max: f64,
        max_iterations: usize,
        tolerance: f64,
        mixing: f64,
        configuration: Option<Vec<(usize, usize, f64)>>,
        confinement: Option<PyReadonlyArray1<'py, f64>>,
        functional: &str,
        relativity: &str,
        polish: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let model = mandacaru_core::models::atom::AtomModel {
            functional: functional.to_string(),
            relativity: mandacaru_core::models::relativity::Treatment::parse(relativity)
                .map_err(error)?,
            polish,
        };
        let options = mandacaru_core::models::atom::AtomOptions {
            points,
            r_max,
            max_iterations,
            tolerance,
            mixing,
        };
        atom::solve(
            self,
            py,
            atomic_number,
            options,
            model,
            configuration,
            confinement,
        )
    }
    #[pyo3(signature=(atomic_number,*,functional="lda",relativity="none",r_max=25.,points=None))]
    fn relaxed_configuration(
        &self,
        py: Python<'_>,
        atomic_number: u8,
        functional: &str,
        relativity: &str,
        r_max: f64,
        points: Option<usize>,
    ) -> PyResult<Vec<(usize, usize, f64)>> {
        atom::relaxed_configuration(
            self,
            py,
            atomic_number,
            functional,
            relativity,
            r_max,
            points,
        )
    }
    #[pyo3(signature=(r,potential,l,n_nodes,*,relativity="none",kappa=None,atomic_number=0.,energy_guess=None))]
    fn solve_radial<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray1<'py, f64>,
        potential: PyReadonlyArray1<'py, f64>,
        l: usize,
        n_nodes: usize,
        relativity: &str,
        kappa: Option<i32>,
        atomic_number: f64,
        energy_guess: Option<f64>,
    ) -> PyResult<(Array<'py, f64>, f64)> {
        atom::solve_radial(
            self,
            py,
            r,
            potential,
            l,
            n_nodes,
            relativity,
            kappa,
            atomic_number,
            energy_guess,
        )
    }
    fn lda_terms<'py>(
        &self,
        py: Python<'py>,
        density: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        atom::lda_terms(self, py, density)
    }
    #[pyo3(signature=(n,l,charge,radius_bohr,*,intervals=2000))]
    fn confined_radial<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        l: usize,
        charge: f64,
        radius_bohr: f64,
        intervals: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let table = py
            .detach(|| {
                mandacaru_core::basis::radial::confined(
                    &self.inner,
                    n,
                    l,
                    charge,
                    radius_bohr,
                    intervals,
                )
            })
            .map_err(error)?;
        let out = PyDict::new(py);
        let size = table.values.as_slice().len();
        out.set_item(
            "r",
            array(
                py,
                &[size],
                (0..size).map(|i| i as f64 * table.step).collect::<Vec<_>>(),
            )?,
        )?;
        out.set_item(
            "radial",
            array(py, &[size], table.values.as_slice().to_vec())?,
        )?;
        out.set_item("energy", table.energy)?;
        Ok(out)
    }
    #[pyo3(signature=(grid,densities,*,functional="lda",kinetic_densities=None))]
    fn exchange_correlation<'py>(
        &self,
        py: Python<'py>,
        grid: &PySpatialGrid,
        densities: PyReadonlyArray2<'py, f64>,
        functional: &str,
        kinetic_densities: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if densities.shape() != [2, grid.inner.index.points()] {
            return Err(PyValueError::new_err("densities must have shape (2,ngrid)"));
        }
        let (rho, _memory) = copy(&self.inner, &densities)?;
        if kinetic_densities
            .as_ref()
            .is_some_and(|v| v.shape() != densities.shape())
        {
            return Err(PyValueError::new_err(
                "kinetic densities must match spin densities",
            ));
        }
        let kinetic = kinetic_densities
            .as_ref()
            .map(|v| copy(&self.inner, v))
            .transpose()?;
        let value = py
            .detach(|| {
                mandacaru_core::models::xc::evaluate_fields(
                    &self.inner,
                    &grid.inner,
                    &rho,
                    kinetic.as_ref().map(|v| v.0.as_slice()),
                    functional,
                )
            })
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item("energy", value.energy)?;
        if let Some(tau) = value.tau_potentials {
            out.set_item(
                "tau_potentials",
                array(py, &[2, grid.inner.index.points()], tau.as_slice().to_vec())?,
            )?;
        }
        out.set_item(
            "potentials",
            array(
                py,
                &[2, grid.inner.index.points()],
                value.potentials.as_slice().to_vec(),
            )?,
        )?;
        Ok(out)
    }
    #[pyo3(signature=(geometry,coordinates,*,basis_spec="{\"name\":\"HAO\"}"))]
    fn sample_basis<'py>(
        &self,
        py: Python<'py>,
        geometry: &PyGeometry,
        coordinates: PyReadonlyArray2<'py, f64>,
        basis_spec: &str,
    ) -> PyResult<Array<'py, f64>> {
        if coordinates.shape()[1] != 3 {
            return Err(PyValueError::new_err(
                "coordinates must have shape (N,3) in Bohr",
            ));
        }
        let (points, _memory) = copy(&self.inner, &coordinates)?;
        let (m, out) = py
            .detach(|| {
                mandacaru_core::finite(&points, "basis coordinates")?;
                let basis =
                    mandacaru_core::basis::Basis::build(&self.inner, &geometry.inner, basis_spec)?;
                let m = basis.orbitals.len();
                let n = points.len() / 3;
                let mut out = self.inner.allocate(mandacaru_core::product(&[m, n])?)?;
                for (a, orbital) in basis.orbitals.iter().enumerate() {
                    for k in 0..n {
                        out.as_mut_slice()[a * n + k] =
                            orbital.evaluate(points[k * 3..k * 3 + 3].try_into().unwrap());
                    }
                }
                Ok::<_, Error>((m, out))
            })
            .map_err(error)?;
        array(py, &[m, points.len() / 3], out.as_slice().to_vec())
    }
    #[pyo3(signature=(geometry,*,basis_spec="{\"name\":\"HAO\"}",spacing_bohr=0.37794522492515404,grid=None,kinetic="spectral",cholesky_tolerance=None,precision="float64"))]
    #[allow(clippy::too_many_arguments)]
    fn molecular_integrals(
        &self,
        py: Python<'_>,
        geometry: &PyGeometry,
        basis_spec: &str,
        spacing_bohr: f64,
        grid: Option<&PySpatialGrid>,
        kinetic: &str,
        cholesky_tolerance: Option<f64>,
        precision: &str,
    ) -> PyResult<molecular::PyIntegrals> {
        let precision =
            mandacaru_core::math::storage::Precision::parse(precision).map_err(error)?;
        let geometry = geometry.inner.clone();
        let grid = match grid {
            Some(grid) => grid.inner.clone(),
            None => mandacaru_core::math::SpatialGrid::from_geometry(
                &geometry,
                [spacing_bohr; 3],
                false,
                false,
            )
            .map_err(error)?,
        };
        let inner = py
            .detach(|| {
                mandacaru_core::molecular::MolecularIntegrals::new(
                    &self.inner,
                    geometry,
                    grid,
                    basis_spec,
                    kinetic,
                    cholesky_tolerance,
                    precision,
                )
            })
            .map_err(error)?;
        Ok(molecular::PyIntegrals {
            inner: std::sync::Arc::new(inner),
        })
    }
    #[pyo3(signature=(geometry,grid,metadata,arrays,*,kinetic="spectral",cholesky_tolerance=1e-10,precision="float64"))]
    #[allow(clippy::too_many_arguments)]
    fn paw_molecular_integrals(
        &self,
        py: Python<'_>,
        geometry: &PyGeometry,
        grid: &PySpatialGrid,
        metadata: &str,
        arrays: &Bound<'_, PyDict>,
        kinetic: &str,
        cholesky_tolerance: f64,
        precision: &str,
    ) -> PyResult<molecular::PyIntegrals> {
        let precision =
            mandacaru_core::math::storage::Precision::parse(precision).map_err(error)?;
        let mut tables = mandacaru_core::paw::molecular::Arrays::new();
        for (key, value) in arrays.iter() {
            let values: PyReadonlyArrayDyn<'_, f64> = value.extract()?;
            super::layout(&values)?;
            tables.insert(
                key.extract::<String>()?,
                self.inner.copy_slice(values.as_slice()?).map_err(error)?,
            );
        }
        let _metadata = self.inner.reserve(metadata.len()).map_err(error)?;
        let inner = py
            .detach(|| {
                mandacaru_core::paw::molecular::build(
                    &self.inner,
                    geometry.inner.clone(),
                    grid.inner.clone(),
                    metadata,
                    tables,
                    kinetic,
                    cholesky_tolerance,
                    precision,
                )
            })
            .map_err(error)?;
        Ok(molecular::PyIntegrals {
            inner: Arc::new(inner),
        })
    }
    #[pyo3(signature=(grid,density,*,periodic=false,slab_axis=None))]
    fn poisson<'py>(
        &self,
        py: Python<'py>,
        grid: &PySpatialGrid,
        density: PyReadonlyArray1<'py, Complex64>,
        periodic: bool,
        slab_axis: Option<usize>,
    ) -> PyResult<Array<'py, Complex64>> {
        molecular::poisson(py, &self.inner, grid, density, periodic, slab_axis)
    }
    fn pauli_operator(
        &self,
        py: Python<'_>,
        num_qubits: usize,
        terms: Vec<(String, f64, f64)>,
    ) -> PyResult<quantum::PyOperator> {
        py.detach(|| quantum::operator(&self.inner, num_qubits, terms))
    }
    #[pyo3(signature=(num_qubits,terms,*,order=2,precision="float64"))]
    fn propagator(
        &self,
        py: Python<'_>,
        num_qubits: usize,
        terms: Vec<(String, f64, f64)>,
        order: usize,
        precision: &str,
    ) -> PyResult<advanced::PyPropagator> {
        py.detach(|| advanced::PyPropagator::new(&self.inner, num_qubits, terms, order, precision))
    }
    fn mp2<'py>(
        &self,
        py: Python<'py>,
        one: PyReadonlyArray2<'py, Complex64>,
        two: PyReadonlyArrayDyn<'py, Complex64>,
        occupied: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        advanced::mp2(py, &self.inner, one, two, occupied)
    }
    fn mp2_from_block<'py>(
        &self,
        py: Python<'py>,
        oovv: PyReadonlyArrayDyn<'py, f64>,
        energies: PyReadonlyArray1<'py, f64>,
        occupied: usize,
        canonical: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let m = energies.len();
        if occupied == 0
            || occupied >= m
            || canonical.shape() != [m, m]
            || oovv.shape() != [occupied, occupied, m - occupied, m - occupied]
        {
            return Err(PyValueError::new_err("MP2 block dimensions disagree"));
        }
        let (oovv, _b) = copy(&self.inner, &oovv)?;
        let (energies, _e) = copy(&self.inner, &energies)?;
        let (canonical, _c) = copy(&self.inner, &canonical)?;
        let out = py
            .detach(|| {
                mandacaru_core::molecular::correlation::from_block(
                    &self.inner,
                    &oovv,
                    &energies,
                    occupied,
                    &canonical,
                )
            })
            .map_err(error)?;
        advanced::mp2_result(py, out)
    }
    #[pyo3(signature=(n_modes,num_particles,terms,*,mapping="jordan_wigner"))]
    fn fermion_operator(
        &self,
        py: Python<'_>,
        n_modes: usize,
        num_particles: [usize; 2],
        terms: quantum::FermionTerms,
        mapping: &str,
    ) -> PyResult<quantum::PyOperator> {
        py.detach(|| quantum::fermion(&self.inner, n_modes, num_particles, mapping, terms))
    }
    #[pyo3(signature=(path,*,sector=true,dense=false,include_singles=true,sparse=false,precision="float64"))]
    fn load_hamiltonian(
        &self,
        py: Python<'_>,
        path: String,
        sector: bool,
        dense: bool,
        include_singles: bool,
        sparse: bool,
        precision: &str,
    ) -> PyResult<quantum::PyProblem> {
        let precision = quantum::parse_precision(precision)?;
        if sparse && dense {
            return Err(PyValueError::new_err(
                "choose either dense or sparse storage",
            ));
        }
        let mut inner = py
            .detach(|| {
                mandacaru_core::quantum::CachedProblem::load(
                    &self.inner,
                    std::path::Path::new(&path),
                    sector,
                    dense,
                    include_singles,
                )
                .map(|problem| problem.with_precision(precision))
            })
            .map_err(error)?;
        if sparse {
            inner.sparse = Some(
                py.detach(|| {
                    mandacaru_core::quantum::sparse::SparseMatrix::new(
                        &self.inner,
                        &inner.basis,
                        &inner.hamiltonian,
                    )
                })
                .map_err(error)?,
            );
        }
        Ok(quantum::PyProblem { inner })
    }
    #[new]
    #[pyo3(signature = (threads=None, memory_limit_bytes=536870912))]
    fn new(threads: Option<usize>, memory_limit_bytes: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Context::new(threads, memory_limit_bytes).map_err(error)?,
        })
    }
    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads()
    }
    #[getter]
    fn memory_limit_bytes(&self) -> usize {
        self.inner.memory_limit()
    }
    #[getter]
    fn memory_used_bytes(&self) -> usize {
        self.inner.memory_used()
    }
    #[getter]
    fn memory_peak_bytes(&self) -> usize {
        self.inner.memory_peak()
    }
    #[getter]
    fn worker_peak(&self) -> usize {
        self.inner.workers_peak()
    }
    #[getter]
    fn workers_launched(&self) -> usize {
        self.inner.workers_total()
    }

    fn workspace(&self, elements: usize) -> PyResult<Workspace> {
        Ok(Workspace {
            buffer: self.inner.allocate(elements).map_err(error)?,
        })
    }

    fn laplacian<'py>(
        &self,
        py: Python<'py>,
        grid: &PyGrid,
        fields: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let shape = fields.shape().to_vec();
        if shape.len() != 4 || shape[1..] != grid.inner.shape() {
            return Err(PyValueError::new_err("fields must have shape (N,nx,ny,nz)"));
        }
        let (input, _input) = copy(&self.inner, &fields)?;
        let result = py
            .detach(|| kernels::laplacian(&self.inner, &grid.inner, shape[0], &input))
            .map_err(error)?;
        array(py, &shape, result.as_slice().to_vec())
    }
    fn laplacian_f32<'py>(
        &self,
        py: Python<'py>,
        grid: &PyGrid,
        fields: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Array<'py, f32>> {
        let shape = fields.shape().to_vec();
        if shape.len() != 4 || shape[1..] != grid.inner.shape() {
            return Err(PyValueError::new_err("fields must have shape (N,nx,ny,nz)"));
        }
        let (input, _input) = copy(&self.inner, &fields)?;
        let result = py
            .detach(|| kernels::laplacian_f32(&self.inner, &grid.inner, shape[0], &input))
            .map_err(error)?;
        array(py, &shape, result.as_slice().to_vec())
    }
    fn one_body<'py>(
        &self,
        py: Python<'py>,
        grid: &PyGrid,
        orbitals: PyReadonlyArray2<'py, Complex64>,
        potential: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Array<'py, Complex64>, Array<'py, Complex64>)> {
        let m = orbitals.shape()[0];
        let (psi, _psi) = complex(&self.inner, &orbitals)?;
        let (potential, _potential) = copy(&self.inner, &potential)?;
        let (kinetic, local) = py
            .detach(|| kernels::one_body(&self.inner, &grid.inner, m, &psi, &potential))
            .map_err(error)?;
        Ok((
            complex_array(py, &[m, m], &kinetic)?,
            complex_array(py, &[m, m], &local)?,
        ))
    }
    fn project<'py>(
        &self,
        py: Python<'py>,
        dv_bohr3: f64,
        orbitals: PyReadonlyArray2<'py, Complex64>,
        projectors: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<Array<'py, Complex64>> {
        let (m, g, p) = (
            orbitals.shape()[0],
            orbitals.shape()[1],
            projectors.shape()[0],
        );
        if projectors.shape()[1] != g {
            return Err(PyValueError::new_err(
                "projector and orbital grid lengths differ",
            ));
        }
        let (psi, _psi) = complex(&self.inner, &orbitals)?;
        let (chi, _chi) = complex(&self.inner, &projectors)?;
        let result = py
            .detach(|| kernels::project(&self.inner, g, m, p, dv_bohr3, &psi, &chi))
            .map_err(error)?;
        complex_array(py, &[m, p], &result)
    }
    fn bessel<'py>(
        &self,
        py: Python<'py>,
        l: i32,
        radii: PyReadonlyArray1<'py, f64>,
        wave_numbers: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let (r, _r) = copy(&self.inner, &radii)?;
        let (k, _k) = copy(&self.inner, &wave_numbers)?;
        let result = py
            .detach(|| kernels::bessel(&self.inner, l, &r, &k))
            .map_err(error)?;
        array(py, &[r.len(), k.len()], result.as_slice().to_vec())
    }
    fn fourier_bessel<'py>(
        &self,
        py: Python<'py>,
        l: i32,
        radii: PyReadonlyArray1<'py, f64>,
        weighted: PyReadonlyArray1<'py, f64>,
        wave_numbers: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let (r, _r) = copy(&self.inner, &radii)?;
        let (w, _w) = copy(&self.inner, &weighted)?;
        let (k, _k) = copy(&self.inner, &wave_numbers)?;
        let result = py
            .detach(|| kernels::fourier_bessel(&self.inner, l, &r, &w, &k))
            .map_err(error)?;
        array(py, &[k.len()], result.as_slice().to_vec())
    }
    #[pyo3(signature=(r, potential, radius, *, shift=0., fit_window=None))]
    fn paw_local_potential<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray1<'py, f64>,
        potential: PyReadonlyArray1<'py, f64>,
        radius: f64,
        shift: f64,
        fit_window: Option<f64>,
    ) -> PyResult<Array<'py, f64>> {
        paw::local(self, py, r, potential, radius, shift, fit_window)
    }
    fn radial_hartree<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray1<'py, f64>,
        density: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        models::radial_hartree(self, py, r, density)
    }
    #[pyo3(signature=(grid,tables,*,kinetic_density=false))]
    #[allow(clippy::type_complexity)]
    fn radial_fields<'py>(
        &self,
        py: Python<'py>,
        grid: &PySpatialGrid,
        tables: Vec<(
            PyReadonlyArray1<'py, f64>,
            PyReadonlyArray1<'py, f64>,
            PyReadonlyArray2<'py, f64>,
        )>,
        kinetic_density: bool,
    ) -> PyResult<Array<'py, f64>> {
        use mandacaru_core::paw::fields::{self, RadialSource};
        let _metadata = self
            .inner
            .reserve(mandacaru_core::product(&[tables.len(), 128]).map_err(error)?)
            .map_err(error)?;
        let mut sources = Vec::with_capacity(tables.len());
        for (r, values, centers) in tables {
            super::layout(&r)?;
            super::layout(&values)?;
            super::layout(&centers)?;
            if centers.shape()[1] != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "radial field centers need shape (n,3)",
                ));
            }
            let mut positions = self
                .inner
                .allocate::<[f64; 3]>(centers.shape()[0])
                .map_err(error)?;
            for (to, from) in positions
                .as_mut_slice()
                .iter_mut()
                .zip(centers.as_slice()?.chunks_exact(3))
            {
                to.copy_from_slice(from);
            }
            sources.push(RadialSource {
                knots: self.inner.copy_slice(r.as_slice()?).map_err(error)?,
                values: self.inner.copy_slice(values.as_slice()?).map_err(error)?,
                centers: positions,
            });
        }
        let out = py
            .detach(|| fields::sample_many(&self.inner, &grid.inner, &sources, kinetic_density))
            .map_err(error)?;
        array(py, &[grid.inner.index.points()], out.as_slice().to_vec())
    }
    #[pyo3(signature=(r, density, *, functional="lda", relativistic=false))]
    fn radial_xc<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray1<'py, f64>,
        density: PyReadonlyArray1<'py, f64>,
        functional: &str,
        relativistic: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        models::radial_xc(self, py, r, density, functional, relativistic)
    }
    #[pyo3(signature=(r, ae, potential, local, l, radius, local_radius, energies, wavevectors, coefficients))]
    fn paw_channel<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray1<'py, f64>,
        ae: PyReadonlyArray2<'py, f64>,
        potential: PyReadonlyArray1<'py, f64>,
        local: PyReadonlyArray1<'py, f64>,
        l: usize,
        radius: f64,
        local_radius: f64,
        energies: Vec<f64>,
        wavevectors: Vec<f64>,
        coefficients: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        paw::channel(
            self,
            py,
            r,
            ae,
            potential,
            local,
            l,
            radius,
            local_radius,
            energies,
            wavevectors,
            coefficients,
        )
    }
    fn paw_compensation<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray1<'py, f64>,
        radius: f64,
    ) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
        let (r, _memory) = copy(&self.inner, &r)?;
        if !radius.is_finite() || radius <= 0. || r.iter().any(|v| !v.is_finite() || *v < 0.) {
            return Err(PyValueError::new_err(
                "compensation needs a positive radius and nonnegative coordinates",
            ));
        }
        let (shape, potential) = py
            .detach(|| {
                let mut shape = self.inner.allocate::<f64>(r.len())?;
                let mut potential = self.inner.allocate::<f64>(r.len())?;
                for (i, _) in r.iter().enumerate() {
                    shape.as_mut_slice()[i] = mandacaru_core::paw::radial::shape(r[i], radius);
                    potential.as_mut_slice()[i] =
                        mandacaru_core::paw::radial::compensation_potential(r[i], radius);
                }
                Ok::<_, Error>((shape, potential))
            })
            .map_err(error)?;
        Ok((
            array(py, &[r.len()], shape.as_slice().to_vec())?,
            array(py, &[r.len()], potential.as_slice().to_vec())?,
        ))
    }
    #[pyo3(signature=(radius_a,radius_b,distance,*,points=160,angles=80))]
    fn paw_compensation_coulomb(
        &self,
        py: Python<'_>,
        radius_a: f64,
        radius_b: f64,
        distance: f64,
        points: usize,
        angles: usize,
    ) -> PyResult<f64> {
        let _memory = self
            .inner
            .reserve(
                product(&[
                    points
                        .checked_add(angles)
                        .ok_or_else(|| error(Error::Overflow))?,
                    16,
                ])
                .map_err(error)?,
            )
            .map_err(error)?;
        py.detach(|| {
            mandacaru_core::paw::radial::compensation_coulomb(
                radius_a, radius_b, distance, points, angles,
            )
        })
        .map_err(error)
    }
    fn eigenpair<'py>(
        &self,
        py: Python<'py>,
        diagonal: PyReadonlyArray1<'py, f64>,
        off_diagonal: PyReadonlyArray1<'py, f64>,
        index: usize,
    ) -> PyResult<(f64, Array<'py, f64>)> {
        let (d, _d) = copy(&self.inner, &diagonal)?;
        let (e, _e) = copy(&self.inner, &off_diagonal)?;
        let (value, vector) = py
            .detach(|| kernels::eigenpair(&self.inner, &d, &e, index))
            .map_err(error)?;
        Ok((value, array(py, &[d.len()], vector.as_slice().to_vec())?))
    }
    #[pyo3(signature=(radii,potential,projectors,coupling,overlap,*,state=0,vectors=true))]
    fn paw_eigenpair<'py>(
        &self,
        py: Python<'py>,
        radii: PyReadonlyArray1<'py, f64>,
        potential: PyReadonlyArray1<'py, f64>,
        projectors: PyReadonlyArray2<'py, f64>,
        coupling: PyReadonlyArray2<'py, f64>,
        overlap: PyReadonlyArray2<'py, f64>,
        state: usize,
        vectors: bool,
    ) -> PyResult<(f64, Option<Array<'py, f64>>)> {
        let n = radii.len();
        let rank = projectors.shape()[0];
        if projectors.shape()[1] != n
            || coupling.shape() != [rank, rank]
            || overlap.shape() != [rank, rank]
        {
            return Err(PyValueError::new_err(
                "PAW projectors must be (rank,nodes) and coupling/overlap (rank,rank)",
            ));
        }
        let (radii, _r) = copy(&self.inner, &radii)?;
        let (potential, _v) = copy(&self.inner, &potential)?;
        let (projectors, _p) = copy(&self.inner, &projectors)?;
        let (coupling, _d) = copy(&self.inner, &coupling)?;
        let (overlap, _q) = copy(&self.inner, &overlap)?;
        let result = py
            .detach(|| {
                mandacaru_core::paw::spectrum::eigenpair(
                    &self.inner,
                    mandacaru_core::paw::spectrum::Channel {
                        radii: &radii,
                        potential: &potential,
                        projectors: &projectors,
                        coupling: &coupling,
                        overlap: &overlap,
                        rank,
                    },
                    state,
                    vectors,
                )
            })
            .map_err(error)?;
        let vector = result
            .vector
            .as_ref()
            .map(|v| array(py, &[n], v.as_slice().to_vec()))
            .transpose()?;
        Ok((result.energy, vector))
    }
    #[pyo3(signature=(coefficient, source, initial, step_squared, boundary, inward=false))]
    fn numerov<'py>(
        &self,
        py: Python<'py>,
        coefficient: PyReadonlyArray1<'py, f64>,
        source: PyReadonlyArray1<'py, f64>,
        initial: PyReadonlyArray1<'py, f64>,
        step_squared: f64,
        boundary: usize,
        inward: bool,
    ) -> PyResult<Array<'py, f64>> {
        let (f, _f) = copy(&self.inner, &coefficient)?;
        let (s, _s) = copy(&self.inner, &source)?;
        let (u, _u) = copy(&self.inner, &initial)?;
        let result = py
            .detach(|| kernels::numerov(&self.inner, &f, &s, &u, step_squared, boundary, inward))
            .map_err(error)?;
        array(py, &[f.len()], result.as_slice().to_vec())
    }
    #[pyo3(signature=(coordinates, center, support, knots, coefficients, cutoff, l, ms, derivative=-1))]
    fn real_shell<'py>(
        &self,
        py: Python<'py>,
        coordinates: PyReadonlyArray2<'py, f64>,
        center: [f64; 3],
        support: f64,
        knots: PyReadonlyArray1<'py, f64>,
        coefficients: PyReadonlyArray1<'py, f64>,
        cutoff: f64,
        l: i32,
        ms: PyReadonlyArray1<'py, i32>,
        derivative: i32,
    ) -> PyResult<Array<'py, f64>> {
        if coordinates.shape()[0] != 3 {
            return Err(PyValueError::new_err("coordinates must have shape (3,G)"));
        }
        let points = coordinates.shape()[1];
        let (xyz, _xyz) = copy(&self.inner, &coordinates)?;
        let (b, _b) = copy(&self.inner, &knots)?;
        let (c, _c) = copy(&self.inner, &coefficients)?;
        let (ms, _ms) = copy(&self.inner, &ms)?;
        let spline = kernels::Spline::new(&b, &c, cutoff, l).map_err(error)?;
        let result = py
            .detach(|| {
                kernels::real_shell(
                    &self.inner,
                    [&xyz[..points], &xyz[points..2 * points], &xyz[2 * points..]],
                    center,
                    support,
                    &spline,
                    &ms,
                    derivative,
                )
            })
            .map_err(error)?;
        array(py, &[ms.len(), points], result.as_slice().to_vec())
    }
}
