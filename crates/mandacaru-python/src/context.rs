//! Execution context and dispatch to Python-independent scientific operations.
use super::{
    advanced, array, atom, complex, complex_array, copy, error, models, molecular, periodic,
    persistence, quantum, Array, PyGeometry, PyGrid, PySpatialGrid, Workspace,
};
use mandacaru_core::periodic::{
    ewald::Ewald,
    spectrum::{self, Smearing},
    Crystal,
};
use mandacaru_core::{kernels, product, Context, Error};
use numpy::{
    Complex32, Complex64, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn,
    PyReadwriteArray3, PyUntypedArrayMethods,
};
use periodic::PyCrystal;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use nalgebra::Matrix3;
use std::sync::Arc;
type GateRecord = (String, usize, Option<usize>, Option<f64>);
type ValenceSource<'py> = (PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, f64>, [f64; 3]);

fn valence_inputs<'a, 'py>(
    sources: &'a [ValenceSource<'py>],
) -> PyResult<(Vec<mandacaru_core::periodic::setup::fields::Tables<'a>>, Vec<mandacaru_core::periodic::setup::fields::Placed>)> {
    use mandacaru_core::periodic::setup::fields::{Placed, Tables};
    let mut tables = Vec::with_capacity(sources.len());
    let mut atoms = Vec::with_capacity(sources.len());
    for (species, (r, values, center)) in sources.iter().enumerate() {
        super::layout(r)?;
        super::layout(values)?;
        tables.push(Tables { radius: r.as_slice()?, valence_density: values.as_slice()?, core: None });
        atoms.push(Placed { species, center: *center });
    }
    Ok((tables, atoms))
}

enum PeriodicOperator<'py> {
    Real(PyReadonlyArray2<'py, f64>),
    Complex(PyReadonlyArray2<'py, Complex64>),
}
enum PeriodicOperatorView<'a> {
    Real(&'a [f64]),
    Complex(&'a [Complex64]),
}
impl PeriodicOperatorView<'_> {
    fn get(&self, i: usize) -> Complex64 {
        match self {
            Self::Real(values) => Complex64::new(values[i], 0.),
            Self::Complex(values) => values[i],
        }
    }
}
impl<'py> PeriodicOperator<'py> {
    fn extract(value: &Bound<'py, PyAny>, p: usize) -> PyResult<Self> {
        let result = if let Ok(a) = value.extract::<PyReadonlyArray2<'py, f64>>() {
            super::layout(&a)?;
            if a.shape() != [p, p] { return Err(PyValueError::new_err("operator shape must be (projectors, projectors)")); }
            Self::Real(a)
        } else {
            let a = value.extract::<PyReadonlyArray2<'py, Complex64>>()?;
            super::layout(&a)?;
            if a.shape() != [p, p] { return Err(PyValueError::new_err("operator shape must be (projectors, projectors)")); }
            Self::Complex(a)
        };
        Ok(result)
    }
    fn view(&self) -> PyResult<PeriodicOperatorView<'_>> {
        Ok(match self {
            Self::Real(a) => PeriodicOperatorView::Real(a.as_slice()?),
            Self::Complex(a) => PeriodicOperatorView::Complex(a.as_slice()?),
        })
    }
}
#[pyclass(name = "Context", module = "mandacaru._native", frozen)]
pub(crate) struct PyContext {
    pub(crate) inner: Context,
}
#[pymethods]
impl PyContext {
    #[pyo3(signature = (
        method,
        x0,
        cost,
        gradient = None,
        max_iterations = 1000,
        max_evaluations = 10000,
        tolerance = None,
        seed = 0,
        track_best = true,
        patience = 5,
        eps = 1.4901161193847656e-8
    ))]
    fn minimize<'py>(
        &self,
        py: Python<'py>,
        method: &str,
        x0: PyReadonlyArray1<'py, f64>,
        cost: Bound<'py, PyAny>,
        gradient: Option<Bound<'py, PyAny>>,
        max_iterations: usize,
        max_evaluations: usize,
        tolerance: Option<f64>,
        seed: u64,
        track_best: bool,
        patience: usize,
        eps: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        crate::optimizers::native_minimize(
            py,
            self,
            method,
            x0,
            cost,
            gradient,
            max_iterations,
            max_evaluations,
            tolerance,
            seed,
            track_best,
            patience,
            eps,
        )
    }

    /// Spectral Bloch gradients, borrowing contiguous complex128 samples.
    fn periodic_bloch_gradients<'py>(
        &self, py: Python<'py>, grid: &PySpatialGrid, k: [f64; 3],
        orbitals: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<Array<'py, Complex64>> {
        super::layout(&orbitals)?;
        let m = orbitals.shape()[0];
        if orbitals.shape()[1] != grid.inner.index.points() {
            return Err(PyValueError::new_err("orbitals must have shape (functions, grid points)"));
        }
        let out = mandacaru_core::periodic::fields::bloch_gradients(
            &self.inner, &grid.inner, k, orbitals.as_slice()?, m).map_err(error)?;
        let (values, _reservation) = out.into_parts();
        array(py, &[3, m, grid.inner.index.points()], values)
    }

    /// Spectral Bloch gradients with complex64 input and complex128 output.
    fn periodic_bloch_gradients_f32<'py>(
        &self, py: Python<'py>, grid: &PySpatialGrid, k: [f64; 3],
        orbitals: PyReadonlyArray2<'py, Complex32>,
    ) -> PyResult<Array<'py, Complex64>> {
        super::layout(&orbitals)?;
        let m = orbitals.shape()[0];
        if orbitals.shape()[1] != grid.inner.index.points() {
            return Err(PyValueError::new_err("orbitals must have shape (functions, grid points)"));
        }
        let out = mandacaru_core::periodic::fields::bloch_gradients_f32(
            &self.inner, &grid.inner, k, orbitals.as_slice()?, m).map_err(error)?;
        let (values, _reservation) = out.into_parts();
        array(py, &[3, m, grid.inner.index.points()], values)
    }

    /// Smooth valence density `rho` and projected density matrix `R = sum_k w_k C(k)^+ P^k C(k)`.
    #[pyo3(signature=(m, p_proj, weights, matrices, psi_list, projections))]
    fn periodic_density_and_projected<'py>(
        &self,
        py: Python<'py>,
        m: usize,
        p_proj: usize,
        weights: PyReadonlyArray1<'py, f64>,
        matrices: Vec<PyReadonlyArray2<'py, Complex64>>,
        psi_list: Vec<PyReadonlyArray2<'py, Complex64>>,
        projections: Vec<PyReadonlyArray2<'py, Complex64>>,
    ) -> PyResult<(Array<'py, f64>, Array<'py, Complex64>)> {
        use mandacaru_core::periodic::setup::spinor_tau::density_and_projected;
        let w = weights.as_slice()?;
        let nk = w.len();
        if matrices.len() != nk || psi_list.len() != nk || projections.len() != nk {
            return Err(PyValueError::new_err(
                "weights, matrices, psi_list and projections must have matching lengths",
            ));
        }
        let points = if nk > 0 { psi_list[0].shape()[1] } else { 0 };
        let mut mat_slices = Vec::with_capacity(nk);
        let mut psi_slices = Vec::with_capacity(nk);
        let mut proj_slices = Vec::with_capacity(nk);
        for i in 0..nk {
            super::layout(&matrices[i])?;
            super::layout(&psi_list[i])?;
            super::layout(&projections[i])?;
            mat_slices.push(matrices[i].as_slice()?);
            psi_slices.push(psi_list[i].as_slice()?);
            proj_slices.push(projections[i].as_slice()?);
        }
        let res = py
            .detach(|| {
                density_and_projected(
                    &self.inner,
                    m,
                    points,
                    p_proj,
                    w,
                    &mat_slices,
                    &psi_slices,
                    &proj_slices,
                )
            })
            .map_err(error)?;
        Ok((
            array(py, &[points], res.rho)?,
            array(py, &[p_proj, p_proj], res.r_matrix)?,
        ))
    }

    /// Scalar kinetic-energy density from k-dependent density matrices and Bloch gradients.
    #[pyo3(signature=(m, weights, matrices, gradients))]
    fn periodic_kinetic_density<'py>(
        &self,
        py: Python<'py>,
        m: usize,
        weights: PyReadonlyArray1<'py, f64>,
        matrices: Vec<PyReadonlyArray2<'py, Complex64>>,
        gradients: Vec<PyReadonlyArray3<'py, Complex64>>,
    ) -> PyResult<Array<'py, f64>> {
        use mandacaru_core::periodic::setup::spinor_tau::scalar_kinetic_density;
        let w = weights.as_slice()?;
        let nk = w.len();
        if matrices.len() != nk || gradients.len() != nk {
            return Err(PyValueError::new_err(
                "weights, matrices and gradients must have matching lengths",
            ));
        }
        let points = if nk > 0 { gradients[0].shape()[2] } else { 0 };
        let mut mat_slices = Vec::with_capacity(nk);
        let mut grad_slices = Vec::with_capacity(nk);
        for i in 0..nk {
            super::layout(&matrices[i])?;
            super::layout(&gradients[i])?;
            mat_slices.push(matrices[i].as_slice()?);
            grad_slices.push(gradients[i].as_slice()?);
        }
        let tau = py
            .detach(|| scalar_kinetic_density(&self.inner, m, points, w, &mat_slices, &grad_slices))
            .map_err(error)?;
        array(py, &[points], tau)
    }

    /// Smooth magnetization vector of spinor density matrices on the grid.
    #[pyo3(signature=(m, weights, matrices, psi_list))]
    fn periodic_spinor_magnetization<'py>(
        &self,
        py: Python<'py>,
        m: usize,
        weights: PyReadonlyArray1<'py, f64>,
        matrices: Vec<PyReadonlyArray2<'py, Complex64>>,
        psi_list: Vec<PyReadonlyArray2<'py, Complex64>>,
    ) -> PyResult<Array<'py, f64>> {
        use mandacaru_core::periodic::setup::spinor_tau::spinor_magnetization;
        let w = weights.as_slice()?;
        let nk = w.len();
        if matrices.len() != nk || psi_list.len() != nk {
            return Err(PyValueError::new_err(
                "weights, matrices and psi_list must have matching lengths",
            ));
        }
        let points = if nk > 0 { psi_list[0].shape()[1] } else { 0 };
        let mut mat_slices = Vec::with_capacity(nk);
        let mut psi_slices = Vec::with_capacity(nk);
        for i in 0..nk {
            super::layout(&matrices[i])?;
            super::layout(&psi_list[i])?;
            mat_slices.push(matrices[i].as_slice()?);
            psi_slices.push(psi_list[i].as_slice()?);
        }
        let m_vec = py
            .detach(|| spinor_magnetization(&self.inner, m, points, w, &mat_slices, &psi_slices))
            .map_err(error)?;
        array(py, &[3, points], m_vec)
    }

    /// Spinor kinetic-energy density `(tau, tau_m)` from 2M x 2M density matrices and gradients.
    #[pyo3(signature=(m, weights, matrices, gradients))]
    fn periodic_spinor_kinetic_density<'py>(
        &self,
        py: Python<'py>,
        m: usize,
        weights: PyReadonlyArray1<'py, f64>,
        matrices: Vec<PyReadonlyArray2<'py, Complex64>>,
        gradients: Vec<PyReadonlyArray3<'py, Complex64>>,
    ) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
        use mandacaru_core::periodic::setup::spinor_tau::spinor_kinetic_density;
        let w = weights.as_slice()?;
        let nk = w.len();
        if matrices.len() != nk || gradients.len() != nk {
            return Err(PyValueError::new_err(
                "weights, matrices and gradients must have matching lengths",
            ));
        }
        let points = if nk > 0 { gradients[0].shape()[2] } else { 0 };
        let mut mat_slices = Vec::with_capacity(nk);
        let mut grad_slices = Vec::with_capacity(nk);
        for i in 0..nk {
            super::layout(&matrices[i])?;
            super::layout(&gradients[i])?;
            mat_slices.push(matrices[i].as_slice()?);
            grad_slices.push(gradients[i].as_slice()?);
        }
        let res = py
            .detach(|| spinor_kinetic_density(&self.inner, m, points, w, &mat_slices, &grad_slices))
            .map_err(error)?;
        Ok((
            array(py, &[points], res.tau)?,
            array(py, &[3, points], res.tau_m)?,
        ))
    }

    /// Point-ion energy minus Gaussian ions' reciprocal energy.
    #[pyo3(signature=(lattice, centers, charges, radii, sigma))]
    fn periodic_ion_constants(
        &self,
        lattice: [[f64; 3]; 3],
        centers: Vec<[f64; 3]>,
        charges: Vec<f64>,
        radii: Vec<f64>,
        sigma: f64,
    ) -> PyResult<f64> {
        use mandacaru_core::periodic::setup::{ions, shapes};
        if centers.len() != charges.len() || centers.len() != radii.len() {
            return Err(PyValueError::new_err(
                "centers, charges and radii must have equal lengths",
            ));
        }
        let atoms: Vec<_> = centers
            .into_iter()
            .zip(charges)
            .zip(radii)
            .map(|((center, charge), compensation_radius)| shapes::Atom {
                center,
                charge,
                compensation_radius,
            })
            .collect();
        let cell = nalgebra::Matrix3::from_fn(|i, j| lattice[i][j]);
        let channels = Vec::new();
        let sites = shapes::Sites {
            atoms: &atoms,
            channels: &channels,
            lattice: cell,
            sigma,
        };
        ions::ion_constants(&sites).map_err(error)
    }

    /// Analytic gradient of point-ion energy with respect to atom positions.
    #[pyo3(signature=(lattice, centers, charges, radii, sigma))]
    fn periodic_ion_constants_gradient<'py>(
        &self,
        py: Python<'py>,
        lattice: [[f64; 3]; 3],
        centers: Vec<[f64; 3]>,
        charges: Vec<f64>,
        radii: Vec<f64>,
        sigma: f64,
    ) -> PyResult<Array<'py, f64>> {
        use mandacaru_core::periodic::setup::{gradients, shapes};
        if centers.len() != charges.len() || centers.len() != radii.len() {
            return Err(PyValueError::new_err(
                "centers, charges and radii must have equal lengths",
            ));
        }
        let atoms: Vec<_> = centers
            .into_iter()
            .zip(charges)
            .zip(radii)
            .map(|((center, charge), compensation_radius)| shapes::Atom {
                center,
                charge,
                compensation_radius,
            })
            .collect();
        let cell = nalgebra::Matrix3::from_fn(|i, j| lattice[i][j]);
        let channels = Vec::new();
        let sites = shapes::Sites {
            atoms: &atoms,
            channels: &channels,
            lattice: cell,
            sigma,
        };
        let grad = gradients::ion_constants_gradient(&sites).map_err(error)?;
        let n = grad.len();
        let mut flat = Vec::with_capacity(n * 3);
        for row in grad {
            flat.extend_from_slice(&row);
        }
        array(py, &[n, 3], flat)
    }

    /// Onsite long-range ion compensation term per channel.
    #[pyo3(signature=(lattice, centers, charges, radii, channels, sigma))]
    fn periodic_onsite_ion_compensation(
        &self,
        lattice: [[f64; 3]; 3],
        centers: Vec<[f64; 3]>,
        charges: Vec<f64>,
        radii: Vec<f64>,
        channels: Vec<(usize, usize, i32)>,
        sigma: f64,
    ) -> PyResult<Vec<f64>> {
        use mandacaru_core::periodic::setup::{ions, shapes};
        if centers.len() != charges.len() || centers.len() != radii.len() {
            return Err(PyValueError::new_err(
                "centers, charges and radii must have equal lengths",
            ));
        }
        let atoms: Vec<_> = centers
            .into_iter()
            .zip(charges)
            .zip(radii)
            .map(|((center, charge), compensation_radius)| shapes::Atom {
                center,
                charge,
                compensation_radius,
            })
            .collect();
        let cell = nalgebra::Matrix3::from_fn(|i, j| lattice[i][j]);
        let sites = shapes::Sites {
            atoms: &atoms,
            channels: &channels,
            lattice: cell,
            sigma,
        };
        ions::onsite_ion_compensation(&sites).map_err(error)
    }

    /// Short-range half of the ions' potential on the compensation charges.
    #[pyo3(signature=(lattice, centers, charges, radii, channels, sigma, r_tables, v_tables, cutoffs))]
    fn periodic_short_range_ion_compensation<'py>(
        &self,
        py: Python<'py>,
        lattice: [[f64; 3]; 3],
        centers: Vec<[f64; 3]>,
        charges: Vec<f64>,
        radii: Vec<f64>,
        channels: Vec<(usize, usize, i32)>,
        sigma: f64,
        r_tables: Vec<PyReadonlyArray1<'py, f64>>,
        v_tables: Vec<PyReadonlyArray1<'py, f64>>,
        cutoffs: Vec<f64>,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::periodic::setup::{ions, shapes};
        let n_atoms = centers.len();
        if n_atoms != charges.len()
            || n_atoms != radii.len()
            || n_atoms != r_tables.len()
            || n_atoms != v_tables.len()
            || n_atoms != cutoffs.len()
        {
            return Err(PyValueError::new_err(
                "centers, charges, radii, r_tables, v_tables, and cutoffs must have equal lengths",
            ));
        }
        for i in 0..n_atoms {
            super::layout(&r_tables[i])?;
            super::layout(&v_tables[i])?;
        }
        let mut potentials = Vec::with_capacity(n_atoms);
        for i in 0..n_atoms {
            potentials.push(
                ions::LocalPotential::new(
                    &self.inner,
                    r_tables[i].as_slice()?,
                    v_tables[i].as_slice()?,
                    charges[i],
                    cutoffs[i],
                )
                .map_err(error)?,
            );
        }
        let atoms: Vec<_> = centers
            .into_iter()
            .zip(charges)
            .zip(radii)
            .map(|((center, charge), compensation_radius)| shapes::Atom {
                center,
                charge,
                compensation_radius,
            })
            .collect();
        let cell = nalgebra::Matrix3::from_fn(|i, j| lattice[i][j]);
        let sites = shapes::Sites {
            atoms: &atoms,
            channels: &channels,
            lattice: cell,
            sigma,
        };
        let res = py
            .detach(|| ions::short_range_ion_compensation(&self.inner, &sites, &potentials))
            .map_err(error)?;
        let n = res.len();
        array(py, &[n], res)
    }

    /// Analytic derivative of short_range_ion_compensation with respect to atom positions.
    #[pyo3(signature=(lattice, centers, charges, radii, channels, sigma, r_tables, v_tables, cutoffs))]
    fn periodic_short_range_ion_compensation_gradient<'py>(
        &self,
        py: Python<'py>,
        lattice: [[f64; 3]; 3],
        centers: Vec<[f64; 3]>,
        charges: Vec<f64>,
        radii: Vec<f64>,
        channels: Vec<(usize, usize, i32)>,
        sigma: f64,
        r_tables: Vec<PyReadonlyArray1<'py, f64>>,
        v_tables: Vec<PyReadonlyArray1<'py, f64>>,
        cutoffs: Vec<f64>,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::periodic::setup::{gradients, ions, shapes};
        let n_atoms = centers.len();
        if n_atoms != charges.len()
            || n_atoms != radii.len()
            || n_atoms != r_tables.len()
            || n_atoms != v_tables.len()
            || n_atoms != cutoffs.len()
        {
            return Err(PyValueError::new_err(
                "centers, charges, radii, r_tables, v_tables, and cutoffs must have equal lengths",
            ));
        }
        for i in 0..n_atoms {
            super::layout(&r_tables[i])?;
            super::layout(&v_tables[i])?;
        }
        let mut potentials = Vec::with_capacity(n_atoms);
        for i in 0..n_atoms {
            potentials.push(
                ions::LocalPotential::new(
                    &self.inner,
                    r_tables[i].as_slice()?,
                    v_tables[i].as_slice()?,
                    charges[i],
                    cutoffs[i],
                )
                .map_err(error)?,
            );
        }
        let atoms: Vec<_> = centers
            .into_iter()
            .zip(charges)
            .zip(radii)
            .map(|((center, charge), compensation_radius)| shapes::Atom {
                center,
                charge,
                compensation_radius,
            })
            .collect();
        let cell = nalgebra::Matrix3::from_fn(|i, j| lattice[i][j]);
        let sites = shapes::Sites {
            atoms: &atoms,
            channels: &channels,
            lattice: cell,
            sigma,
        };
        let res = py
            .detach(|| gradients::short_range_ion_compensation_gradient(&self.inner, &sites, &potentials))
            .map_err(error)?;
        let n_channels = channels.len();
        let mut flat = Vec::with_capacity(n_channels * n_atoms * 3);
        for atom_vec in res {
            for xyz in atom_vec {
                flat.extend_from_slice(&xyz);
            }
        }
        array(py, &[n_channels, n_atoms, 3], flat)
    }

    /// Contraction of projected density matrix R with multipole blocks for each channel.
    #[pyo3(signature=(r, atom_positions, channels, blocks))]
    fn periodic_moment_traces<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray2<'py, Complex64>,
        atom_positions: Vec<Vec<usize>>,
        channels: Vec<(usize, usize, i32)>,
        blocks: Vec<PyReadonlyArray2<'py, Complex64>>,
    ) -> PyResult<Array<'py, Complex64>> {
        super::layout(&r)?;
        let p = r.shape()[0];
        if r.shape()[1] != p {
            return Err(PyValueError::new_err("R matrix must be square"));
        }
        if channels.len() != blocks.len() {
            return Err(PyValueError::new_err(
                "channels and blocks must have matching lengths",
            ));
        }
        let r_slice = r.as_slice()?;
        let mut q_out = Vec::with_capacity(channels.len());
        for (c, (atom, _, _)) in channels.iter().enumerate() {
            if *atom >= atom_positions.len() {
                return Err(PyValueError::new_err("atom index out of range in channel"));
            }
            let own = &atom_positions[*atom];
            let blk = &blocks[c];
            super::layout(blk)?;
            let blk_shape = blk.shape();
            let na = own.len();
            if blk_shape[0] != na || blk_shape[1] != na {
                return Err(PyValueError::new_err("multipole block dimension mismatch"));
            }
            let blk_slice = blk.as_slice()?;
            let mut sum = Complex64::default();
            for i in 0..na {
                let pi = own[i];
                if pi >= p {
                    return Err(PyValueError::new_err("projector index out of range"));
                }
                for j in 0..na {
                    let pj = own[j];
                    if pj >= p {
                        return Err(PyValueError::new_err("projector index out of range"));
                    }
                    sum += r_slice[pi * p + pj] * blk_slice[j * na + i];
                }
            }
            q_out.push(sum);
        }
        array(py, &[channels.len()], q_out)
    }

    /// Assembly of moment operator D = sum_c w_c blk_c on each atom's projector block.
    #[pyo3(signature=(total_p, atom_positions, weights, channel_atoms, blocks))]
    fn periodic_moment_operator<'py>(
        &self,
        py: Python<'py>,
        total_p: usize,
        atom_positions: Vec<Vec<usize>>,
        weights: PyReadonlyArray1<'py, Complex64>,
        channel_atoms: Vec<usize>,
        blocks: Vec<PyReadonlyArray2<'py, Complex64>>,
    ) -> PyResult<Array<'py, Complex64>> {
        super::layout(&weights)?;
        let nc = channel_atoms.len();
        if weights.len() != nc || blocks.len() != nc {
            return Err(PyValueError::new_err(
                "weights, channel_atoms and blocks must have matching lengths",
            ));
        }
        let w_slice = weights.as_slice()?;
        let mut d = vec![Complex64::default(); total_p * total_p];
        for (c, &atom) in channel_atoms.iter().enumerate() {
            if atom >= atom_positions.len() {
                return Err(PyValueError::new_err("atom index out of range"));
            }
            let own = &atom_positions[atom];
            let blk = &blocks[c];
            super::layout(blk)?;
            let na = own.len();
            let blk_shape = blk.shape();
            if blk_shape[0] != na || blk_shape[1] != na {
                return Err(PyValueError::new_err("multipole block dimension mismatch"));
            }
            let blk_slice = blk.as_slice()?;
            let w = w_slice[c];
            for i in 0..na {
                let pi = own[i];
                if pi >= total_p {
                    return Err(PyValueError::new_err("projector index out of range"));
                }
                for j in 0..na {
                    let pj = own[j];
                    if pj >= total_p {
                        return Err(PyValueError::new_err("projector index out of range"));
                    }
                    d[pi * total_p + pj] += w * blk_slice[i * na + j];
                }
            }
        }
        array(py, &[total_p, total_p], d)
    }

    /// Evaluates one shell's Bloch sums: discovers lattice translation images reaching
    /// the region, computes the k-point phase factors, and calls the native bloch_shell kernel.
    #[pyo3(signature=(points, center, lattice, kpoints, region_center, region_radius, support, breaks, coeffs, rc, l, ms, rows, out, derivative=-1))]
    fn periodic_bloch_shell<'py>(
        &self,
        points: (PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, f64>),
        center: [f64; 3],
        lattice: PyReadonlyArray2<'py, f64>,
        kpoints: PyReadonlyArray2<'py, f64>,
        region_center: [f64; 3],
        region_radius: f64,
        support: f64,
        breaks: PyReadonlyArrayDyn<'py, f64>,
        coeffs: PyReadonlyArrayDyn<'py, f64>,
        rc: f64,
        l: i32,
        ms: Vec<i32>,
        rows: Vec<i64>,
        mut out: PyReadwriteArray3<'py, Complex64>,
        derivative: i32,
    ) -> PyResult<bool> {
        let pts = [points.0.as_slice()?, points.1.as_slice()?, points.2.as_slice()?];
        let npts = pts[0].len();
        if pts[1].len() != npts || pts[2].len() != npts {
            return Err(PyValueError::new_err("points x, y, z must have the same length"));
        }
        let lat_slice = lattice.as_slice()?;
        if lat_slice.len() != 9 {
            return Err(PyValueError::new_err("lattice must be 3x3"));
        }
        let lat_mat = Matrix3::from_row_slice(lat_slice);
        let k_shape = kpoints.shape();
        if k_shape.len() != 2 || k_shape[1] != 3 {
            return Err(PyValueError::new_err("kpoints must have shape (nk, 3)"));
        }
        let nk = k_shape[0];
        let k_slice = kpoints.as_slice()?;
        let k_vecs: Vec<[f64; 3]> = (0..nk)
            .map(|i| [k_slice[3 * i], k_slice[3 * i + 1], k_slice[3 * i + 2]])
            .collect();
        let region = mandacaru_core::periodic::setup::images::Region {
            center: region_center,
            radius: region_radius,
        };
        let images = mandacaru_core::periodic::setup::images::Images::around(
            &self.inner,
            &lat_mat,
            &region,
            center,
            support,
        )
        .map_err(error)?;
        if images.is_empty() {
            return Ok(true);
        }
        let phases = images.phases(&self.inner, &k_vecs, 1.0).map_err(error)?;
        let spline = mandacaru_core::kernels::Spline::new(
            breaks.as_slice()?,
            coeffs.as_slice()?,
            rc,
            l,
        )
        .map_err(error)?;
        let call = mandacaru_core::kernels::ShellImages {
            xyz: [&pts[0], &pts[1], &pts[2]],
            center,
            images: images.vectors(),
            phases: phases.as_slice(),
            support,
            spline: &spline,
            ms: &ms,
            derivative,
        };
        let m_dim = out.shape()[1];
        let out_slice = out.as_slice_mut()?;
        mandacaru_core::kernels::bloch_shell(
            &self.inner,
            &call,
            &rows,
            m_dim,
            out_slice,
        )
        .map_err(error)?;
        Ok(true)
    }

    /// Evaluates one shell's Bloch projections onto columns: discovers lattice images,
    /// calls bloch_project, computes phases, and contracts directly into out.
    #[pyo3(signature=(points, center, lattice, kpoints, region_center, region_radius, support, breaks, coeffs, rc, l, ms, members, columns, out, derivative=-1))]
    fn periodic_bloch_project<'py>(
        &self,
        points: (PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, f64>),
        center: [f64; 3],
        lattice: PyReadonlyArray2<'py, f64>,
        kpoints: PyReadonlyArray2<'py, f64>,
        region_center: [f64; 3],
        region_radius: f64,
        support: f64,
        breaks: PyReadonlyArrayDyn<'py, f64>,
        coeffs: PyReadonlyArrayDyn<'py, f64>,
        rc: f64,
        l: i32,
        ms: Vec<i32>,
        members: Vec<usize>,
        columns: PyReadonlyArray2<'py, Complex64>,
        mut out: PyReadwriteArray3<'py, Complex64>,
        derivative: i32,
    ) -> PyResult<bool> {
        let pts = [points.0.as_slice()?, points.1.as_slice()?, points.2.as_slice()?];
        let npts = pts[0].len();
        if pts[1].len() != npts || pts[2].len() != npts {
            return Err(PyValueError::new_err("points x, y, z must have the same length"));
        }
        let lat_slice = lattice.as_slice()?;
        if lat_slice.len() != 9 {
            return Err(PyValueError::new_err("lattice must be 3x3"));
        }
        let lat_mat = Matrix3::from_row_slice(lat_slice);
        let k_shape = kpoints.shape();
        if k_shape.len() != 2 || k_shape[1] != 3 {
            return Err(PyValueError::new_err("kpoints must have shape (nk, 3)"));
        }
        let nk = k_shape[0];
        let k_slice = kpoints.as_slice()?;
        let k_vecs: Vec<[f64; 3]> = (0..nk)
            .map(|i| [k_slice[3 * i], k_slice[3 * i + 1], k_slice[3 * i + 2]])
            .collect();
        let region = mandacaru_core::periodic::setup::images::Region {
            center: region_center,
            radius: region_radius,
        };
        let images = mandacaru_core::periodic::setup::images::Images::around(
            &self.inner,
            &lat_mat,
            &region,
            center,
            support,
        )
        .map_err(error)?;
        if images.is_empty() {
            return Ok(true);
        }
        let spline = mandacaru_core::kernels::Spline::new(
            breaks.as_slice()?,
            coeffs.as_slice()?,
            rc,
            l,
        )
        .map_err(error)?;
        let unit_phases = vec![mandacaru_core::quantum::C64::new(1.0, 0.0); images.len()];
        let call = mandacaru_core::kernels::ShellImages {
            xyz: [&pts[0], &pts[1], &pts[2]],
            center,
            images: images.vectors(),
            phases: &unit_phases,
            support,
            spline: &spline,
            ms: &ms,
            derivative,
        };
        let col_slice = columns.as_slice()?;
        let p_cols = columns.shape()[0];
        if columns.shape()[1] != npts {
            return Err(PyValueError::new_err("columns must have shape (P, npts)"));
        }
        let blocks = mandacaru_core::kernels::bloch_project(
            &self.inner,
            &call,
            col_slice,
        )
        .map_err(error)?;
        let phases = images.phases(&self.inner, &k_vecs, -1.0).map_err(error)?;
        let nm = ms.len();
        let m_dim = out.shape()[1];
        let out_slice = out.as_slice_mut()?;
        for k in 0..nk {
            for (i, &mu) in members.iter().enumerate() {
                let target_offset = (k * m_dim + mu) * p_cols;
                let target = &mut out_slice[target_offset..target_offset + p_cols];
                for r in 0..images.len() {
                    let phase = phases.as_slice()[k * images.len() + r];
                    let block_offset = (r * nm + i) * p_cols;
                    let block = &blocks.as_slice()[block_offset..block_offset + p_cols];
                    for (to, from) in target.iter_mut().zip(block) {
                        *to += phase * from;
                    }
                }
            }
        }
        Ok(true)
    }

    /// Initial smooth density or collinear magnetization from radial datasets.
    #[pyo3(signature=(grid, sources, *, charges=None, moments=None, magnetization=false))]
    #[allow(clippy::type_complexity)]
    fn periodic_initial_valence<'py>(
        &self, py: Python<'py>, grid: &PySpatialGrid,
        sources: Vec<ValenceSource<'py>>,
        charges: Option<Vec<f64>>, moments: Option<Vec<f64>>, magnetization: bool,
    ) -> PyResult<Array<'py, f64>> {
        use mandacaru_core::periodic::setup::fields;
        let _metadata = self.inner.reserve(product(&[sources.len(), 128]).map_err(error)?).map_err(error)?;
        let (tables, atoms) = valence_inputs(&sources)?;
        let field = if magnetization {
            let charges = charges.ok_or_else(|| PyValueError::new_err("magnetization needs atomic charges"))?;
            let moments = moments.unwrap_or_else(|| vec![0.; sources.len()]);
            fields::initial_magnetization(&self.inner, &grid.inner, &tables, &atoms, &charges, &moments)
        } else {
            if charges.is_some() || moments.is_some() {
                return Err(PyValueError::new_err("density does not take charges or moments"));
            }
            fields::initial_density(&self.inner, &grid.inner, &tables, &atoms)
        }.map_err(error)?;
        let (data, _reservation) = field.into_parts();
        array(py, &[grid.inner.index.points()], data)
    }

    /// Per-atom Hirshfeld shares of the superposed valence density.
    fn periodic_valence_shares<'py>(
        &self, py: Python<'py>, grid: &PySpatialGrid, sources: Vec<ValenceSource<'py>>,
    ) -> PyResult<Array<'py, f64>> {
        if sources.is_empty() { return Err(PyValueError::new_err("valence shares need at least one atom")); }
        let _metadata = self.inner.reserve(product(&[sources.len(), 128]).map_err(error)?).map_err(error)?;
        let (tables, atoms) = valence_inputs(&sources)?;
        let field = mandacaru_core::periodic::setup::fields::valence_shares(
            &self.inner, &grid.inner, &tables, &atoms,
        ).map_err(error)?;
        let (data, _reservation) = field.into_parts();
        array(py, &[sources.len(), grid.inner.index.points()], data)
    }

    /// Initial compensation moments in channel order.
    fn periodic_initial_monopoles<'py>(
        &self, py: Python<'py>, channels: Vec<(usize, usize, i32)>, charges: Vec<f64>,
    ) -> PyResult<Array<'py, Complex64>> {
        let values = mandacaru_core::periodic::setup::fields::initial_monopoles(&channels, &charges)
            .map_err(error)?;
        array(py, &[values.len()], values)
    }

    /// One periodic smooth-valence field using the complete radial spline.
    fn periodic_valence_field<'py>(
        &self, py: Python<'py>, grid: &PySpatialGrid,
        radius: PyReadonlyArray1<'py, f64>, values: PyReadonlyArray1<'py, f64>,
        center: [f64; 3],
    ) -> PyResult<Array<'py, f64>> {
        super::layout(&radius)?;
        super::layout(&values)?;
        let field = mandacaru_core::periodic::setup::fields::valence_field(
            &self.inner, &grid.inner, radius.as_slice()?, values.as_slice()?, center,
        ).map_err(error)?;
        let (data, _reservation) = field.into_parts();
        array(py, &[grid.inner.index.points()], data)
    }

    /// Complete the PAW overlap and fixed matrices without Python products.
    fn periodic_paw_matrices<'py>(
        &self, py: Python<'py>, projections: PyReadonlyArray2<'py, Complex64>,
        overlap: Bound<'py, PyAny>, ionic: Bound<'py, PyAny>,
        grid_overlap: PyReadonlyArray2<'py, Complex64>,
        kinetic: PyReadonlyArray2<'py, Complex64>, local: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<(Array<'py, Complex64>, Array<'py, Complex64>)> {
        for a in [&projections, &grid_overlap, &kinetic, &local] { super::layout(a)?; }
        let (m, p) = (projections.shape()[0], projections.shape()[1]);
        if [grid_overlap.shape(), kinetic.shape(), local.shape()].iter().any(|shape| *shape != [m, m]) {
            return Err(PyValueError::new_err("grid, kinetic and local matrices must have shape (functions, functions)"));
        }
        let q = PeriodicOperator::extract(&overlap, p)?;
        let d = PeriodicOperator::extract(&ionic, p)?;
        let (q, d, c) = (q.view()?, d.view()?, projections.as_slice()?);
        let (s, h) = mandacaru_core::periodic::setup::matrices::assemble_paw_matrices(
            &self.inner, m, p, grid_overlap.as_slice()?, kinetic.as_slice()?, local.as_slice()?,
            |i, j| c[i * p + j], |i, j| q.get(i * p + j), |i, j| d.get(i * p + j),
        ).map_err(error)?;
        let rows = |matrix: &nalgebra::DMatrix<Complex64>| (0..m)
            .flat_map(|i| (0..m).map(move |j| (i, j)))
            .map(|(i, j)| matrix[(i, j)]).collect();
        Ok((array(py, &[m, m], rows(&s))?, array(py, &[m, m], rows(&h))?))
    }

    /// Grid overlap, borrowing aligned contiguous orbital samples.
    fn periodic_overlap<'py>(
        &self, py: Python<'py>, orbitals: PyReadonlyArray2<'py, Complex64>, dv: f64,
    ) -> PyResult<Array<'py, Complex64>> {
        super::layout(&orbitals)?;
        let (m, n) = (orbitals.shape()[0], orbitals.shape()[1]);
        let matrix = mandacaru_core::periodic::setup::matrices::grid_overlap(
            orbitals.as_slice()?, m, n, dv).map_err(error)?;
        let rows = (0..m).flat_map(|i| (0..m).map(move |j| (i, j)))
            .map(|(i, j)| matrix[(i, j)]).collect();
        array(py, &[m, m], rows)
    }

    /// Spectral kinetic matrix, borrowing aligned contiguous orbital samples.
    fn periodic_kinetic<'py>(
        &self, py: Python<'py>, grid: &PySpatialGrid, k: [f64; 3],
        orbitals: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::periodic::setup::pulay::kinetic_grid;
        super::layout(&orbitals)?;
        let m = orbitals.shape()[0];
        if m == 0 || orbitals.shape()[1] != grid.inner.index.points() {
            return Err(PyValueError::new_err("orbitals must have shape (functions, grid points)"));
        }
        let matrix = kinetic_grid(&self.inner, &grid.inner, k, orbitals.as_slice()?, m)
            .map_err(error)?;
        let rows = (0..m).flat_map(|i| (0..m).map(move |j| (i, j)))
            .map(|(i, j)| matrix[(i, j)]).collect();
        array(py, &[m, m], rows)
    }

    /// As above, with single-precision samples promoted while forming FFT input.
    fn periodic_kinetic_f32<'py>(
        &self, py: Python<'py>, grid: &PySpatialGrid, k: [f64; 3],
        orbitals: PyReadonlyArray2<'py, Complex32>,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::periodic::setup::pulay::kinetic_grid_f32;
        super::layout(&orbitals)?;
        let m = orbitals.shape()[0];
        if m == 0 || orbitals.shape()[1] != grid.inner.index.points() {
            return Err(PyValueError::new_err("orbitals must have shape (functions, grid points)"));
        }
        let matrix = kinetic_grid_f32(&self.inner, &grid.inner, k, orbitals.as_slice()?, m)
            .map_err(error)?;
        let rows = (0..m).flat_map(|i| (0..m).map(move |j| (i, j)))
            .map(|(i, j)| matrix[(i, j)]).collect();
        array(py, &[m, m], rows)
    }

    /// Dense reciprocal-space compensation and ion couplings for a periodic cell.
    #[pyo3(signature=(lattice, centers, charges, radii, channels, tables=None, sigma=0.0, cutoff=15.0))]
    fn periodic_compensation_couplings<'py>(
        &self,
        py: Python<'py>,
        lattice: [[f64; 3]; 3],
        centers: Vec<[f64; 3]>,
        charges: Vec<f64>,
        radii: Vec<f64>,
        channels: Vec<(usize, usize, i32)>,
        tables: Option<Vec<(
            f64,
            usize,
            PyReadonlyArray1<'py, f64>,
            PyReadonlyArray1<'py, f64>,
        )>>,
        sigma: f64,
        cutoff: f64,
    ) -> PyResult<(Array<'py, Complex64>, Array<'py, Complex64>, f64)> {
        use mandacaru_core::periodic::setup::{electrostatics, shapes};
        if centers.len() != charges.len() || centers.len() != radii.len() {
            return Err(PyValueError::new_err(
                "centers, charges and radii must have equal lengths",
            ));
        }
        let atoms: Vec<_> = centers
            .into_iter()
            .zip(charges)
            .zip(radii)
            .map(|((center, charge), compensation_radius)| shapes::Atom {
                center,
                charge,
                compensation_radius,
            })
            .collect();
        let cell = nalgebra::Matrix3::from_fn(|i, j| lattice[i][j]);
        let count = channels.len();
        let owned_tables = if let Some(tables) = tables {
            let mut table_bytes = 0usize;
            for (_, _, q, f) in &tables {
                let elements = q
                    .len()
                    .checked_add(f.len())
                    .ok_or_else(|| error(Error::Overflow))?;
                table_bytes = table_bytes
                    .checked_add(product(&[elements, 8]).map_err(error)?)
                    .ok_or_else(|| error(Error::Overflow))?;
            }
            let table_reservation = self.inner.reserve(table_bytes).map_err(error)?;
            let mut owned = Vec::with_capacity(tables.len());
            for (radius, l, q, f) in &tables {
                super::layout(q)?;
                super::layout(f)?;
                owned.push((*radius, *l, q.as_slice()?.to_vec(), f.as_slice()?.to_vec()));
            }
            drop(table_reservation);
            Some(owned)
        } else {
            None
        };
        let result = py
            .detach(|| {
                let sites = shapes::Sites {
                    atoms: &atoms,
                    channels: &channels,
                    lattice: cell,
                    sigma,
                };
                let shapes = match owned_tables {
                    Some(tbls) => {
                        let shapes = shapes::Shapes::from_tables(&self.inner, &sites, tbls)?;
                        if shapes.top() < cutoff {
                            return Err(mandacaru_core::Error::Invalid(
                                "compensation table does not reach the cutoff".into(),
                            ));
                        }
                        shapes
                    }
                    None => shapes::Shapes::new(&self.inner, &sites, cutoff)?,
                };
                electrostatics::couplings(&self.inner, &shapes, &sites, cutoff)
            })
            .map_err(error)?;
        Ok((
            array(py, &[count, count], result.u.as_slice().to_vec())?,
            array(py, &[count], result.u_ion.as_slice().to_vec())?,
            result.e_ion,
        ))
    }

    /// Gaussian ion transform at given reciprocal vectors: shape `(3, ...) -> (...)`.
    #[pyo3(signature=(lattice, centers, charges, radii, sigma, vectors))]
    fn periodic_ion_transform<'py>(
        &self,
        py: Python<'py>,
        lattice: [[f64; 3]; 3],
        centers: Vec<[f64; 3]>,
        charges: Vec<f64>,
        radii: Vec<f64>,
        sigma: f64,
        vectors: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::periodic::setup::{electrostatics, shapes};
        super::layout(&vectors)?;
        let shape = vectors.shape();
        if shape.is_empty() || shape[0] != 3 {
            return Err(PyValueError::new_err(
                "vectors shape must be (3, ...)",
            ));
        }
        let total_vectors: usize = shape[1..].iter().product();
        if centers.len() != charges.len() || centers.len() != radii.len() {
            return Err(PyValueError::new_err(
                "centers, charges and radii must have equal lengths",
            ));
        }
        let atoms: Vec<_> = centers
            .into_iter()
            .zip(charges)
            .zip(radii)
            .map(|((center, charge), compensation_radius)| shapes::Atom {
                center,
                charge,
                compensation_radius,
            })
            .collect();
        let cell = nalgebra::Matrix3::from_fn(|i, j| lattice[i][j]);
        let channels = Vec::new();
        let sites = shapes::Sites {
            atoms: &atoms,
            channels: &channels,
            lattice: cell,
            sigma,
        };
        let slice = vectors.as_slice()?;
        let res: Vec<Complex64> = py.detach(|| {
            let x = &slice[0..total_vectors];
            let y = &slice[total_vectors..2 * total_vectors];
            let z = &slice[2 * total_vectors..3 * total_vectors];
            (0..total_vectors)
                .map(|i| electrostatics::ion_transform(&sites, [x[i], y[i], z[i]]))
                .collect()
        });
        array(py, &shape[1..], res)
    }

    /// Gaussian ion charge on spatial grid, alias-averaged.
    #[pyo3(signature=(grid, lattice, centers, charges, radii, sigma))]
    fn periodic_ion_charge<'py>(
        &self,
        py: Python<'py>,
        grid: &PySpatialGrid,
        lattice: [[f64; 3]; 3],
        centers: Vec<[f64; 3]>,
        charges: Vec<f64>,
        radii: Vec<f64>,
        sigma: f64,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::periodic::setup::{electrostatics, shapes};
        if centers.len() != charges.len() || centers.len() != radii.len() {
            return Err(PyValueError::new_err(
                "centers, charges and radii must have equal lengths",
            ));
        }
        let atoms: Vec<_> = centers
            .into_iter()
            .zip(charges)
            .zip(radii)
            .map(|((center, charge), compensation_radius)| shapes::Atom {
                center,
                charge,
                compensation_radius,
            })
            .collect();
        let cell = nalgebra::Matrix3::from_fn(|i, j| lattice[i][j]);
        let channels = Vec::new();
        let sites = shapes::Sites {
            atoms: &atoms,
            channels: &channels,
            lattice: cell,
            sigma,
        };
        let buf = py
            .detach(|| electrostatics::ion_charge(&self.inner, &grid.inner, &sites))
            .map_err(error)?;
        let n = grid.inner.index.points();
        array(py, &[n], buf.as_slice().to_vec())
    }

    /// Dense compensation couplings at Bloch shift under kernel.
    #[pyo3(signature=(lattice, centers, charges, radii, channels, sigma, cutoff, shift, kernel))]
    fn periodic_dense_compensation<'py>(
        &self,
        py: Python<'py>,
        lattice: [[f64; 3]; 3],
        centers: Vec<[f64; 3]>,
        charges: Vec<f64>,
        radii: Vec<f64>,
        channels: Vec<(usize, usize, i32)>,
        sigma: f64,
        cutoff: f64,
        shift: [f64; 3],
        kernel: Bound<'py, PyAny>,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::periodic::setup::{dense, shapes};
        if centers.len() != charges.len() || centers.len() != radii.len() {
            return Err(PyValueError::new_err(
                "centers, charges and radii must have equal lengths",
            ));
        }
        let atoms: Vec<_> = centers
            .into_iter()
            .zip(charges)
            .zip(radii)
            .map(|((center, charge), compensation_radius)| shapes::Atom {
                center,
                charge,
                compensation_radius,
            })
            .collect();
        let cell = nalgebra::Matrix3::from_fn(|i, j| lattice[i][j]);
        let n_channels = channels.len();
        let k_fn = |q2: f64| -> f64 {
            let val = match kernel.call1((q2,)) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            if let Ok(f) = val.extract::<f64>() {
                return f;
            }
            if let Ok(f) = val.call_method0("__float__").and_then(|v| v.extract::<f64>()) {
                return f;
            }
            0.0
        };
        let sites = shapes::Sites {
            atoms: &atoms,
            channels: &channels,
            lattice: cell,
            sigma,
        };
        let reach = cutoff + shift.iter().map(|x| x * x).sum::<f64>().sqrt();
        let shapes = shapes::Shapes::new(&self.inner, &sites, reach).map_err(error)?;
        let res = dense::dense_compensation(&self.inner, &shapes, &sites, cutoff, shift, &k_fn).map_err(error)?;
        array(py, &[n_channels, n_channels], res)
    }

    /// Dense compensation gradient at Bloch shift under kernel: (3, n_channels, n_channels).
    #[pyo3(signature=(lattice, centers, charges, radii, channels, sigma, cutoff, shift, kernel))]
    fn periodic_dense_compensation_gradient<'py>(
        &self,
        py: Python<'py>,
        lattice: [[f64; 3]; 3],
        centers: Vec<[f64; 3]>,
        charges: Vec<f64>,
        radii: Vec<f64>,
        channels: Vec<(usize, usize, i32)>,
        sigma: f64,
        cutoff: f64,
        shift: [f64; 3],
        kernel: Bound<'py, PyAny>,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::periodic::setup::{dense, shapes};
        if centers.len() != charges.len() || centers.len() != radii.len() {
            return Err(PyValueError::new_err(
                "centers, charges and radii must have equal lengths",
            ));
        }
        let atoms: Vec<_> = centers
            .into_iter()
            .zip(charges)
            .zip(radii)
            .map(|((center, charge), compensation_radius)| shapes::Atom {
                center,
                charge,
                compensation_radius,
            })
            .collect();
        let cell = nalgebra::Matrix3::from_fn(|i, j| lattice[i][j]);
        let n_channels = channels.len();
        let k_fn = |q2: f64| -> f64 {
            let val = match kernel.call1((q2,)) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            if let Ok(f) = val.extract::<f64>() {
                return f;
            }
            if let Ok(f) = val.call_method0("__float__").and_then(|v| v.extract::<f64>()) {
                return f;
            }
            0.0
        };
        let sites = shapes::Sites {
            atoms: &atoms,
            channels: &channels,
            lattice: cell,
            sigma,
        };
        let reach = cutoff + shift.iter().map(|x| x * x).sum::<f64>().sqrt();
        let shapes = shapes::Shapes::new(&self.inner, &sites, reach).map_err(error)?;
        let [gx, gy, gz] = dense::dense_compensation_gradient(&self.inner, &shapes, &sites, cutoff, shift, &k_fn).map_err(error)?;
        let mut flat = Vec::with_capacity(3 * n_channels * n_channels);
        flat.extend(gx);
        flat.extend(gy);
        flat.extend(gz);
        array(py, &[3, n_channels, n_channels], flat)
    }

    /// Alias-averaged reciprocal Coulomb kernel on the spatial grid.
    #[pyo3(signature=(grid))]
    fn periodic_coulomb_kernel<'py>(
        &self,
        py: Python<'py>,
        grid: &PySpatialGrid,
    ) -> PyResult<Array<'py, f64>> {
        use mandacaru_core::periodic::setup::electrostatics;
        let buf = py
            .detach(|| electrostatics::kernel(&self.inner, &grid.inner))
            .map_err(error)?;
        let n = grid.inner.index.points();
        array(py, &[n], buf.as_slice().to_vec())
    }

    /// Cartesian k-points from fractional coordinates and lattice vectors: (n, 3).
    #[pyo3(signature=(lattice, fractional))]
    fn periodic_cartesian_kpoints<'py>(
        &self,
        py: Python<'py>,
        lattice: [[f64; 3]; 3],
        fractional: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        super::layout(&fractional)?;
        let shape = fractional.shape();
        let (n, k_slice) = if shape.len() == 1 && shape[0] == 3 {
            (1, fractional.as_slice()?)
        } else if shape.len() == 2 && shape[1] == 3 {
            (shape[0], fractional.as_slice()?)
        } else {
            return Err(PyValueError::new_err(
                "fractional coordinates must have shape (3,) or (n, 3)",
            ));
        };
        let cell = nalgebra::Matrix3::from_fn(|i, j| lattice[i][j]);
        let inv = cell
            .try_inverse()
            .ok_or_else(|| PyValueError::new_err("singular lattice"))?;
        let b = (2.0 * std::f64::consts::PI) * inv.transpose();
        let mut flat = Vec::with_capacity(n * 3);
        for i in 0..n {
            let kf = nalgebra::Vector3::new(k_slice[i * 3], k_slice[i * 3 + 1], k_slice[i * 3 + 2]);
            let kc = b * kf;
            flat.extend_from_slice(kc.as_slice());
        }
        array(py, &[n, 3], flat)
    }

    /// Time-reversal reduction of arbitrary fractional k-points: (representatives, weights).
    #[pyo3(signature=(kpoints, tolerance=1e-8))]
    fn periodic_time_reversal_reduce<'py>(
        &self,
        py: Python<'py>,
        kpoints: PyReadonlyArray2<'py, f64>,
        tolerance: f64,
    ) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
        super::layout(&kpoints)?;
        if kpoints.shape()[1] != 3 {
            return Err(PyValueError::new_err("kpoints must have shape (n, 3)"));
        }
        let n = kpoints.shape()[0];
        let slice = kpoints.as_slice()?;
        let mut used = vec![false; n];
        let mut keep = Vec::new();
        let mut counts = Vec::new();
        for i in 0..n {
            if used[i] {
                continue;
            }
            used[i] = true;
            let mut count = 1.0;
            let p = [slice[i * 3], slice[i * 3 + 1], slice[i * 3 + 2]];
            let partner = [-p[0], -p[1], -p[2]];
            for j in (i + 1)..n {
                if used[j] {
                    continue;
                }
                let q = [slice[j * 3], slice[j * 3 + 1], slice[j * 3 + 2]];
                let d0 = q[0] - partner[0];
                let d1 = q[1] - partner[1];
                let d2 = q[2] - partner[2];
                if (d0 - d0.round()).abs() < tolerance
                    && (d1 - d1.round()).abs() < tolerance
                    && (d2 - d2.round()).abs() < tolerance
                {
                    used[j] = true;
                    count += 1.0;
                    break;
                }
            }
            keep.extend_from_slice(&p);
            counts.push(count);
        }
        let total: f64 = counts.iter().sum();
        let weights: Vec<f64> = counts.into_iter().map(|c| c / total).collect();
        let m = weights.len();
        Ok((array(py, &[m, 3], keep)?, array(py, &[m], weights)?))
    }

    /// Indices of basis functions whose support reaches within region_radius of region_center.
    #[pyo3(signature=(lattice, centers, supports, region_center, region_radius))]
    fn periodic_reaching(
        &self,
        lattice: [[f64; 3]; 3],
        centers: Vec<[f64; 3]>,
        supports: Vec<f64>,
        region_center: [f64; 3],
        region_radius: f64,
    ) -> PyResult<Vec<usize>> {
        use mandacaru_core::periodic::setup::shapes::visit_lattice;
        if centers.len() != supports.len() {
            return Err(PyValueError::new_err(
                "centers and supports must have equal lengths",
            ));
        }
        let cell = nalgebra::Matrix3::from_fn(|i, j| lattice[i][j]);
        let mut keep = Vec::new();
        let mut verdicts: std::collections::HashMap<(i64, i64, i64, i64), bool> =
            std::collections::HashMap::new();
        for (mu, (&center, &support)) in centers.iter().zip(&supports).enumerate() {
            let key = (
                (center[0] * 1e12).round() as i64,
                (center[1] * 1e12).round() as i64,
                (center[2] * 1e12).round() as i64,
                (support * 1e12).round() as i64,
            );
            let reaches = if let Some(&v) = verdicts.get(&key) {
                v
            } else {
                let reach = support + region_radius;
                let offset = ((center[0] - region_center[0]).powi(2)
                    + (center[1] - region_center[1]).powi(2)
                    + (center[2] - region_center[2]).powi(2))
                .sqrt();
                let radius = reach + offset;
                let mut found = false;
                visit_lattice(&cell, radius, false, 256, &mut |vectors| {
                    if !found {
                        for v in vectors {
                            let dx = center[0] + v[0] - region_center[0];
                            let dy = center[1] + v[1] - region_center[1];
                            let dz = center[2] + v[2] - region_center[2];
                            if dx * dx + dy * dy + dz * dz <= reach * reach + 1e-12 {
                                found = true;
                                break;
                            }
                        }
                    }
                    Ok(())
                })
                .map_err(error)?;
                verdicts.insert(key, found);
                found
            };
            if reaches {
                keep.push(mu);
            }
        }
        Ok(keep)
    }

    /// Lexicographical sort of 3D points by 1-Bohr spatial bins: floor(x), floor(y), floor(z).
    #[pyo3(signature=(x, y, z))]
    fn periodic_spatial_bins_order<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        z: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Array<'py, usize>> {
        let xs = x.as_slice()?;
        let ys = y.as_slice()?;
        let zs = z.as_slice()?;
        let n = xs.len();
        if ys.len() != n || zs.len() != n {
            return Err(PyValueError::new_err("x, y, z must have matching lengths"));
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let kx_a = xs[a].floor() as i64;
            let kx_b = xs[b].floor() as i64;
            if kx_a != kx_b {
                return kx_a.cmp(&kx_b);
            }
            let ky_a = ys[a].floor() as i64;
            let ky_b = ys[b].floor() as i64;
            if ky_a != ky_b {
                return ky_a.cmp(&ky_b);
            }
            let kz_a = zs[a].floor() as i64;
            let kz_b = zs[b].floor() as i64;
            kz_a.cmp(&kz_b)
        });
        array(py, &[n], order)
    }

    /// Invert spatial bin order in-place for every row of a 3D (nk, M, npts) array:
    /// out[k, mu, order[i]] = out_sorted[k, mu, i]
    #[pyo3(signature=(array, order))]
    fn periodic_unpermute_rows<'py>(
        &self,
        mut array: PyReadwriteArray3<'py, Complex64>,
        order: PyReadonlyArray1<'py, usize>,
    ) -> PyResult<()> {
        let order_slice = order.as_slice()?;
        let npts = order_slice.len();
        if (0..npts).all(|i| order_slice[i] == i) {
            return Ok(());
        }
        let arr_slice = array.as_slice_mut()?;
        let total = arr_slice.len();
        if total % npts != 0 {
            return Err(PyValueError::new_err("array length must be a multiple of order length"));
        }
        let n_rows = total / npts;
        let mut temp = vec![Complex64::default(); npts];
        for r in 0..n_rows {
            let row = &mut arr_slice[r * npts..(r + 1) * npts];
            temp.copy_from_slice(row);
            for (sorted_idx, &orig_idx) in order_slice.iter().enumerate() {
                if orig_idx < npts {
                    row[orig_idx] = temp[sorted_idx];
                }
            }
        }
        Ok(())
    }

    /// Time-reversal reduction of a uniform k-point mesh on integer lattice 2 * size.
    #[pyo3(signature=(mesh, size))]
    fn periodic_mesh_wedge_time_reversal<'py>(
        &self,
        py: Python<'py>,
        mesh: PyReadonlyArray2<'py, f64>,
        size: [usize; 3],
    ) -> PyResult<(Array<'py, f64>, Array<'py, usize>)> {
        let n_mesh = mesh.shape()[0];
        if mesh.shape()[1] != 3 {
            return Err(PyValueError::new_err("mesh must have shape (nk, 3)"));
        }
        let mesh_slice = mesh.as_slice()?;
        let double = [2 * size[0] as i64, 2 * size[1] as i64, 2 * size[2] as i64];
        let total_table = (double[0] * double[1] * double[2]) as usize;
        let mut where_table = vec![-1isize; total_table];
        let mut flat_indices = Vec::with_capacity(n_mesh);
        let mut inv_flat_indices = Vec::with_capacity(n_mesh);
        for i in 0..n_mesh {
            let k0 = mesh_slice[i * 3];
            let k1 = mesh_slice[i * 3 + 1];
            let k2 = mesh_slice[i * 3 + 2];
            let c0 = (k0 * double[0] as f64).round() as i64;
            let c1 = (k1 * double[1] as f64).round() as i64;
            let c2 = (k2 * double[2] as f64).round() as i64;
            let m0 = c0.rem_euclid(double[0]);
            let m1 = c1.rem_euclid(double[1]);
            let m2 = c2.rem_euclid(double[2]);
            let flat = ((m0 * double[1] + m1) * double[2] + m2) as usize;
            where_table[flat] = i as isize;
            flat_indices.push(flat);

            let inv_m0 = (-c0).rem_euclid(double[0]);
            let inv_m1 = (-c1).rem_euclid(double[1]);
            let inv_m2 = (-c2).rem_euclid(double[2]);
            let inv_flat = ((inv_m0 * double[1] + inv_m1) * double[2] + inv_flat_m2(inv_m2)) as usize;
            fn inv_flat_m2(m: i64) -> i64 { m }
            inv_flat_indices.push(inv_flat);
        }
        let mut partner = Vec::with_capacity(n_mesh);
        for &inv_flat in &inv_flat_indices {
            let p = where_table[inv_flat];
            if p < 0 {
                return Err(PyValueError::new_err("the k-point mesh is not symmetric under k -> -k"));
            }
            partner.push(p as usize);
        }
        let mut keep = Vec::new();
        for i in 0..n_mesh {
            if i <= partner[i] {
                keep.push(i);
            }
        }
        let mut slot = vec![0usize; n_mesh];
        for (rep_idx, &orig_idx) in keep.iter().enumerate() {
            slot[orig_idx] = rep_idx;
            slot[partner[orig_idx]] = rep_idx;
        }
        let mut red_mesh = Vec::with_capacity(keep.len() * 3);
        for &orig_idx in &keep {
            red_mesh.push(mesh_slice[orig_idx * 3]);
            red_mesh.push(mesh_slice[orig_idx * 3 + 1]);
            red_mesh.push(mesh_slice[orig_idx * 3 + 2]);
        }
        let red_array = array(py, &[keep.len(), 3], red_mesh)?;
        let slot_array = array(py, &[n_mesh], slot)?;
        Ok((red_array, slot_array))
    }

    /// Space group operations that map the real-space grid onto itself.
    #[pyo3(signature=(grid_shape, lattice, frac_positions, numbers, rotations, translations, max_l))]
    fn periodic_grid_symmetry<'py>(
        &self,
        py: Python<'py>,
        grid_shape: [usize; 3],
        lattice: [[f64; 3]; 3],
        frac_positions: Vec<[f64; 3]>,
        numbers: Vec<i64>,
        rotations: Vec<[[f64; 3]; 3]>,
        translations: Vec<[f64; 3]>,
        max_l: usize,
    ) -> PyResult<Option<(Vec<usize>, Vec<Array<'py, i64>>, Vec<Array<'py, i64>>, Vec<Vec<Array<'py, Complex64>>>)>> {
        use mandacaru_core::periodic::setup::symmetry;
        let n_atoms = frac_positions.len();
        if numbers.len() != n_atoms {
            return Err(PyValueError::new_err("frac_positions and numbers must have matching lengths"));
        }
        let n_grid = grid_shape.map(|v| v as f64);
        let n_int = grid_shape.map(|v| v as i64);
        let total_nodes = grid_shape[0] * grid_shape[1] * grid_shape[2];
        let lat = nalgebra::Matrix3::from_fn(|i, j| lattice[i][j]);
        let inv_lat = lat.try_inverse().ok_or_else(|| PyValueError::new_err("singular lattice matrix"))?;
        let lat_t = lat.transpose();

        let mut keep = Vec::new();
        let mut permutations = Vec::new();
        let mut atom_maps = Vec::new();
        let mut rotations_out = Vec::new();

        'op_loop: for (s, (w, t)) in rotations.iter().zip(&translations).enumerate() {
            // Check scale: W_ij * N_i / N_j
            let mut scale = [[0i64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let sc = w[i][j] * n_grid[i] / n_grid[j];
                    if (sc - sc.round()).abs() > 1e-9 {
                        continue 'op_loop;
                    }
                    scale[i][j] = sc.round() as i64;
                }
            }
            // Check shift: t_i * N_i
            let mut shift = [0i64; 3];
            for i in 0..3 {
                let sh = t[i] * n_grid[i];
                if (sh - sh.round()).abs() > 1e-6 {
                    continue 'op_loop;
                }
                shift[i] = sh.round() as i64;
            }

            // Grid permutation
            let mut perm = vec![0i64; total_nodes];
            let mut flat = 0;
            for a in 0..n_int[0] {
                for b in 0..n_int[1] {
                    for c in 0..n_int[2] {
                        let im0 = (scale[0][0] * a + scale[0][1] * b + scale[0][2] * c + shift[0]).rem_euclid(n_int[0]);
                        let im1 = (scale[1][0] * a + scale[1][1] * b + scale[1][2] * c + shift[1]).rem_euclid(n_int[1]);
                        let im2 = (scale[2][0] * a + scale[2][1] * b + scale[2][2] * c + shift[2]).rem_euclid(n_int[2]);
                        perm[flat] = (im0 * n_int[1] + im1) * n_int[2] + im2;
                        flat += 1;
                    }
                }
            }

            // Atom mapping: mapped = (frac @ W.T + t) % 1.0
            let mut atom_map = vec![0i64; n_atoms];
            for a in 0..n_atoms {
                let fa = &frac_positions[a];
                let mx0 = (fa[0] * w[0][0] + fa[1] * w[0][1] + fa[2] * w[0][2] + t[0]).rem_euclid(1.0);
                let mx1 = (fa[0] * w[1][0] + fa[1] * w[1][1] + fa[2] * w[1][2] + t[1]).rem_euclid(1.0);
                let mx2 = (fa[0] * w[2][0] + fa[1] * w[2][1] + fa[2] * w[2][2] + t[2]).rem_euclid(1.0);
                let mut matches = Vec::new();
                for b in 0..n_atoms {
                    if numbers[b] != numbers[a] {
                        continue;
                    }
                    let fb = &frac_positions[b];
                    let dx = fb[0] - mx0;
                    let dy = fb[1] - mx1;
                    let dz = fb[2] - mx2;
                    let d0 = dx - dx.round();
                    let d1 = dy - dy.round();
                    let d2 = dz - dz.round();
                    let cx = d0 * lat_t[(0, 0)] + d1 * lat_t[(1, 0)] + d2 * lat_t[(2, 0)];
                    let cy = d0 * lat_t[(0, 1)] + d1 * lat_t[(1, 1)] + d2 * lat_t[(2, 1)];
                    let cz = d0 * lat_t[(0, 2)] + d1 * lat_t[(1, 2)] + d2 * lat_t[(2, 2)];
                    let dist = (cx * cx + cy * cy + cz * cz).sqrt();
                    if dist < 1e-4 {
                        matches.push(b);
                    }
                }
                if matches.len() != 1 {
                    continue 'op_loop;
                }
                atom_map[a] = matches[0] as i64;
            }

            // Cartesian rotation: R = lattice @ W @ inverse
            let w_mat = nalgebra::Matrix3::from_fn(|i, j| w[i][j]);
            let r_cart = lat * w_mat * inv_lat;
            let tables = symmetry::rotation_tables(&r_cart, max_l, 1e-6).map_err(error)?;
            let mut l_arrays = Vec::with_capacity(max_l + 1);
            for l in 0..=max_l {
                let t_l = &tables[l];
                let width = 2 * l + 1;
                let mut flat_t = Vec::with_capacity(width * width);
                for row in 0..width {
                    for col in 0..width {
                        flat_t.push(t_l[(row, col)]);
                    }
                }
                l_arrays.push(array(py, &[width, width], flat_t)?);
            }

            keep.push(s);
            permutations.push(array(py, &[total_nodes], perm)?);
            atom_maps.push(array(py, &[n_atoms], atom_map)?);
            rotations_out.push(l_arrays);
        }

        if keep.len() <= 1 {
            return Ok(None);
        }
        Ok(Some((keep, permutations, atom_maps, rotations_out)))
    }

    /// Remainder transforms for one atom's projector pairs at reciprocal vectors.
    #[pyo3(signature=(levels, center, vectors, step=0.025))]
    fn periodic_remainder_transforms<'py>(
        &self,
        py: Python<'py>,
        levels: Vec<(usize, PyReadonlyArray3<'py, f64>, Vec<(i32, PyReadonlyArray2<'py, Complex64>)>)>,
        center: [f64; 3],
        vectors: PyReadonlyArray2<'py, f64>,
        step: f64,
    ) -> PyResult<Array<'py, Complex64>> {
        use std::f64::consts::PI;
        let v_shape = vectors.shape();
        if v_shape.len() != 2 || v_shape[1] != 3 {
            return Err(PyValueError::new_err("vectors must have shape (count, 3)"));
        }
        let count = v_shape[0];
        let vec_slice = vectors.as_slice()?;
        if levels.is_empty() {
            return array(py, &[0, 0, count], vec![]);
        }
        let n = levels[0].1.shape()[1];
        let mut out = vec![Complex64::default(); n * n * count];

        for (l, table_arr, couplings) in &levels {
            let l = *l;
            let table_slice = table_arr.as_slice()?;
            let points = table_arr.shape()[0];
            let unit = [
                Complex64::new(1., 0.),
                Complex64::new(0., -1.),
                Complex64::new(-1., 0.),
                Complex64::new(0., 1.),
            ][l % 4];

            let mut c_slices = Vec::with_capacity(couplings.len());
            for (m, ang_arr) in couplings {
                c_slices.push((*m, ang_arr.as_slice()?));
            }

            for v in 0..count {
                let gx = vec_slice[v * 3];
                let gy = vec_slice[v * 3 + 1];
                let gz = vec_slice[v * 3 + 2];
                let norm = (gx * gx + gy * gy + gz * gz).sqrt();
                let scale = (norm / step).min(points as f64 - 1.000001);
                let low = scale.floor() as usize;
                let frac = scale - scale.floor();

                let direction = if norm > 0. {
                    [gx / norm, gy / norm, gz / norm]
                } else {
                    [1., 0., 0.]
                };
                let angle = -(gx * center[0] + gy * center[1] + gz * center[2]);
                let phase = Complex64::new(angle.cos(), angle.sin());

                for &(m, ang_slice) in &c_slices {
                    let harmonic = unit
                        * phase
                        * mandacaru_core::math::angular::complex(l, m, direction)
                        * (4. * PI);
                    for a in 0..n {
                        for b in 0..n {
                            let ab = a * n + b;
                            let r0 = table_slice[(low * n + a) * n + b];
                            let r1 = table_slice[((low + 1) * n + a) * n + b];
                            let radial = r0 * (1. - frac) + r1 * frac;
                            out[ab * count + v] += ang_slice[ab] * radial * harmonic;
                        }
                    }
                }
            }
        }
        array(py, &[n, n, count], out)
    }

    /// Assemble the radial table (q_size, n, n) and angular couplings [(m, angular (n, n))]
    /// for remainder plan level L.
    #[pyo3(signature=(l, q_size, projectors, radial))]
    fn periodic_remainder_plan_level<'py>(
        &self,
        py: Python<'py>,
        l: usize,
        q_size: usize,
        projectors: Vec<(usize, i32, usize, bool)>,
        radial: std::collections::HashMap<(usize, usize), PyReadonlyArray3<'py, f64>>,
    ) -> PyResult<(Array<'py, f64>, Vec<(i32, Array<'py, Complex64>)>)> {
        use mandacaru_core::paw::multipole::{gaunt, gaunt_grid, multipole_range};
        let n = projectors.len();
        let mut table = vec![0.0f64; q_size * n * n];
        for (a, pa) in projectors.iter().enumerate() {
            for (b, pb) in projectors.iter().enumerate() {
                if let Some(r_arr) = radial.get(&(pa.0, pb.0)) {
                    let r_slice = r_arr.as_slice()?;
                    let shape = r_arr.shape();
                    let (dim_i, dim_j) = (shape[1], shape[2]);
                    let (idx_i, idx_j) = (pa.2, pb.2);
                    for q in 0..q_size {
                        table[(q * n + a) * n + b] = r_slice[(q * dim_i + idx_i) * dim_j + idx_j];
                    }
                }
            }
        }

        let grid = gaunt_grid().map_err(error)?;
        let mut couplings = Vec::new();
        for m in -(l as i32)..=(l as i32) {
            let mut angular = vec![Complex64::default(); n * n];
            let mut any_nonzero = false;
            for (a, pa) in projectors.iter().enumerate() {
                for (b, pb) in projectors.iter().enumerate() {
                    if multipole_range(pa.0, pb.0).contains(&l) {
                        let g = gaunt([l, pa.0, pb.0], [m, pa.1, pb.1], pa.3, &grid);
                        if g.norm() != 0.0 {
                            any_nonzero = true;
                        }
                        angular[a * n + b] = g;
                    }
                }
            }
            if any_nonzero {
                couplings.push((m, array(py, &[n, n], angular)?));
            }
        }
        let py_table = array(py, &[q_size, n, n], table)?;
        Ok((py_table, couplings))
    }

    /// Invert v1 and v2 Vanderbilt matrices and transform radial remainder E:
    /// E[q, a, b] = sum_{i, j} b1[a, i] * E[q, i, j] * b2[b, j]
    #[pyo3(signature=(v1, v2, e))]
    fn periodic_raw_vanderbilt_transform<'py>(
        &self,
        py: Python<'py>,
        v1: PyReadonlyArray2<'py, f64>,
        v2: PyReadonlyArray2<'py, f64>,
        e: PyReadonlyArray3<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        use nalgebra::DMatrix;
        let s1 = v1.shape();
        let s2 = v2.shape();
        let (w1, w2) = (s1[0], s2[0]);
        let m1 = DMatrix::from_row_slice(w1, w1, v1.as_slice()?);
        let m2 = DMatrix::from_row_slice(w2, w2, v2.as_slice()?);
        let b1 = m1.try_inverse().ok_or_else(|| PyValueError::new_err("singular Vanderbilt matrix"))?;
        let b2 = m2.try_inverse().ok_or_else(|| PyValueError::new_err("singular Vanderbilt matrix"))?;
        let b2_t = b2.transpose();

        let e_shape = e.shape();
        let (q_size, e_w1, e_w2) = (e_shape[0], e_shape[1], e_shape[2]);
        if e_w1 != w1 || e_w2 != w2 {
            return Err(PyValueError::new_err("e shape must match v1 and v2 dimensions"));
        }
        let e_slice = e.as_slice()?;
        let mut out = vec![0.0f64; q_size * w1 * w2];
        for q in 0..q_size {
            let block = &e_slice[q * w1 * w2..(q + 1) * w1 * w2];
            let em = DMatrix::from_row_slice(w1, w2, block);
            let t = &b1 * em * &b2_t;
            for a in 0..w1 {
                for b in 0..w2 {
                    out[(q * w1 + a) * w2 + b] = t[(a, b)];
                }
            }
        }
        array(py, &[q_size, w1, w2], out)
    }

    /// Round 3D coordinates to 1e-12 Bohr precision for hashing/keying.
    #[pyo3(signature=(coords))]
    fn periodic_round_coords(&self, coords: [f64; 3]) -> (f64, f64, f64) {
        (
            (coords[0] * 1e12).round() / 1e12,
            (coords[1] * 1e12).round() / 1e12,
            (coords[2] * 1e12).round() / 1e12,
        )
    }

    /// Wigner D-matrix / spherical harmonic rotation matrix for shell L: (2L+1, 2L+1).
    #[pyo3(signature=(r, l))]
    fn periodic_harmonic_rotation<'py>(
        &self,
        py: Python<'py>,
        r: [[f64; 3]; 3],
        l: usize,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::periodic::setup::symmetry;
        let mat = nalgebra::Matrix3::from_fn(|i, j| r[i][j]);
        let tables = symmetry::rotation_tables(&mat, l, 1e-6).map_err(error)?;
        let t = &tables[l];
        let width = 2 * l + 1;
        let mut flat = Vec::with_capacity(width * width);
        for row in 0..width {
            for col in 0..width {
                flat.push(t[(row, col)]);
            }
        }
        array(py, &[width, width], flat)
    }

    /// Center and bounding radius of the cell the grid spans.
    #[pyo3(signature=(lattice, origin))]
    fn periodic_cell_region(
        &self,
        lattice: [[f64; 3]; 3],
        origin: [f64; 3],
    ) -> ([f64; 3], f64) {
        use mandacaru_core::periodic::setup::images::Region;
        let cell = nalgebra::Matrix3::from_fn(|i, j| lattice[i][j]);
        let region = Region::of_cell(&cell, origin);
        (region.center, region.radius)
    }

    /// Every channel's compensation shape at arbitrary vectors: (n_channels, ...).
    #[pyo3(signature=(centers, radii, channels, vectors))]
    fn periodic_compensation_transforms<'py>(
        &self,
        py: Python<'py>,
        centers: Vec<[f64; 3]>,
        radii: Vec<f64>,
        channels: Vec<(usize, usize, i32)>,
        vectors: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::periodic::setup::shapes::{Atom, Shapes, Sites};
        let shape = vectors.shape();
        if shape.is_empty() || shape[0] != 3 {
            return Err(PyValueError::new_err("vectors must have shape (3, ...)"));
        }
        let total_vectors: usize = shape[1..].iter().product();
        let vec_slice = vectors.as_slice()?;
        let mut flat_vectors = Vec::with_capacity(total_vectors);
        for v in 0..total_vectors {
            flat_vectors.push([
                vec_slice[v],
                vec_slice[total_vectors + v],
                vec_slice[2 * total_vectors + v],
            ]);
        }
        let mut atoms = Vec::with_capacity(centers.len());
        for (c, &r) in centers.iter().zip(&radii) {
            atoms.push(Atom {
                center: *c,
                charge: 1.0,
                compensation_radius: r,
            });
        }
        let sites = Sites {
            lattice: nalgebra::Matrix3::identity(),
            sigma: 1.0,
            atoms: &atoms,
            channels: &channels,
        };
        let q_max = flat_vectors
            .iter()
            .map(|g| (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt())
            .fold(0.0f64, f64::max);
        let shapes = Shapes::new(&self.inner, &sites, q_max).map_err(error)?;
        let n_channels = channels.len();
        let mut out = vec![Complex64::default(); n_channels * total_vectors];
        shapes
            .transforms(&sites, &flat_vectors, &mut out, total_vectors)
            .map_err(error)?;
        let mut out_shape = vec![n_channels];
        out_shape.extend_from_slice(&shape[1..]);
        array(py, &out_shape, out)
    }

    /// On-site self-energy tail of a compensation shape beyond cutoff.
    #[pyo3(signature=(radius, l, cutoff=15.0))]
    fn periodic_shape_tail(&self, radius: f64, l: usize, cutoff: f64) -> PyResult<f64> {
        use mandacaru_core::periodic::setup::shapes::{Atom, Shapes, Sites};
        let sites = Sites {
            lattice: nalgebra::Matrix3::identity(),
            sigma: 1.0,
            atoms: &[Atom {
                center: [0.0; 3],
                charge: 1.0,
                compensation_radius: radius,
            }],
            channels: &[(0, l, 0)],
        };
        let shapes = Shapes::new(&self.inner, &sites, cutoff.max(120.0)).map_err(error)?;
        shapes.tail(radius, l, cutoff).map_err(error)
    }

    /// Return (q, F) shape table for compensation shape of given radius and angular momentum L.
    #[pyo3(signature=(radius, l, q_max))]
    fn periodic_shape_table<'py>(
        &self,
        py: Python<'py>,
        radius: f64,
        l: usize,
        q_max: f64,
    ) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
        use mandacaru_core::periodic::setup::shapes::{Atom, Shapes, Sites};
        let sites = Sites {
            lattice: nalgebra::Matrix3::identity(),
            sigma: 1.0,
            atoms: &[Atom {
                center: [0.0; 3],
                charge: 1.0,
                compensation_radius: radius,
            }],
            channels: &[(0, l, 0)],
        };
        let shapes = Shapes::new(&self.inner, &sites, q_max).map_err(error)?;
        let table = shapes.table(radius, l).map_err(error)?;
        let q_arr = array(py, &[table.q.len()], table.q.clone())?;
        let f_arr = array(py, &[table.f.len()], table.f.clone())?;
        Ok((q_arr, f_arr))
    }

    /// Interpolate F_L(q) of a compensation shape of radius and angular momentum L.
    #[pyo3(signature=(radius, l, q))]
    fn periodic_shape_transform<'py>(
        &self,
        py: Python<'py>,
        radius: f64,
        l: usize,
        q: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        use mandacaru_core::periodic::setup::shapes::{Atom, Shapes, Sites};
        let q_slice = q.as_slice()?;
        let q_max = q_slice.iter().copied().fold(0.0f64, f64::max);
        let sites = Sites {
            lattice: nalgebra::Matrix3::identity(),
            sigma: 1.0,
            atoms: &[Atom {
                center: [0.0; 3],
                charge: 1.0,
                compensation_radius: radius,
            }],
            channels: &[(0, l, 0)],
        };
        let shapes = Shapes::new(&self.inner, &sites, q_max).map_err(error)?;
        let table = shapes.table(radius, l).map_err(error)?;
        let mut out = Vec::with_capacity(q_slice.len());
        for &x in q_slice {
            out.push(table.at(x));
        }
        array(py, q.shape(), out)
    }

    /// Clear the global compensation shape table cache.
    fn periodic_clear_shape_cache(&self) {
        mandacaru_core::periodic::setup::shapes::clear_shape_cache();
    }

    /// Accumulate conj(chi) * weighted * chi_ket into block for each k-point.
    #[pyo3(signature=(block, chi, chi_ket, weighted))]
    fn periodic_accumulate_zgemm<'py>(
        &self,
        _py: Python<'py>,
        mut block: PyReadwriteArray3<'py, Complex64>,
        chi: PyReadonlyArray3<'py, Complex64>,
        chi_ket: PyReadonlyArray3<'py, Complex64>,
        weighted: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        let mut block_view = block.as_array_mut();
        let chi_view = chi.as_array();
        let chi_ket_view = chi_ket.as_array();
        let w = weighted.as_slice()?;
        let chi_shape = chi_view.shape();
        let chi_ket_shape = chi_ket_view.shape();
        let nk = chi_shape[0];
        let n_bra = chi_shape[1];
        let n_pts = chi_shape[2];
        let n_ket = chi_ket_shape[1];
        if chi_ket_shape[0] != nk || chi_ket_shape[2] != n_pts || n_pts != w.len() {
            return Err(PyValueError::new_err("dimension mismatch in accumulate_zgemm"));
        }
        for k in 0..nk {
            for b in 0..n_bra {
                for k_idx in 0..n_ket {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for p in 0..n_pts {
                        sum += chi_view[[k, b, p]].conj() * w[p] * chi_ket_view[[k, k_idx, p]];
                    }
                    block_view[[k, b, k_idx]] += sum;
                }
            }
        }
        Ok(())
    }

    /// Sample single basis gradient at Cartesian points (x, y, z) in Bohr.
    #[pyo3(signature=(x, y, z, center, breaks, coeffs, rc, l, m, axis))]
    fn periodic_basis_gradient<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        z: PyReadonlyArray1<'py, f64>,
        center: [f64; 3],
        breaks: PyReadonlyArray1<'py, f64>,
        coeffs: PyReadonlyArrayDyn<'py, f64>,
        rc: f64,
        l: i32,
        m: i32,
        axis: usize,
    ) -> PyResult<Array<'py, Complex64>> {
        let pts = [x.as_slice()?, y.as_slice()?, z.as_slice()?];
        let npts = pts[0].len();
        if pts[1].len() != npts || pts[2].len() != npts {
            return Err(PyValueError::new_err("x, y, z must have matching lengths"));
        }
        let spline = mandacaru_core::kernels::Spline::new(
            breaks.as_slice()?,
            coeffs.as_slice()?,
            rc,
            l,
        )
        .map_err(error)?;
        let support = f64::from_bits(rc.to_bits() + 1);
        let ms = [m];
        let phases = [Complex64::new(1.0, 0.0)];
        let images = [[0.0; 3]];
        let call = mandacaru_core::kernels::ShellImages {
            xyz: [&pts[0], &pts[1], &pts[2]],
            center,
            images: &images,
            phases: &phases,
            support,
            spline: &spline,
            ms: &ms,
            derivative: axis as i32,
        };
        let mut out = vec![Complex64::new(0.0, 0.0); npts];
        mandacaru_core::kernels::bloch_shell(
            &self.inner,
            &call,
            &[0],
            1,
            &mut out,
        )
        .map_err(error)?;
        array(py, &[npts], out)
    }

    /// Pack multipole 2D blocks into 3D (n_channels, p, p) tensor.
    #[pyo3(signature=(positions, blocks, p))]
    fn periodic_multipole_blocks<'py>(
        &self,
        py: Python<'py>,
        positions: Vec<Vec<usize>>,
        blocks: Vec<PyReadonlyArray2<'py, Complex64>>,
        p: usize,
    ) -> PyResult<Array<'py, Complex64>> {
        let n_ch = positions.len();
        if blocks.len() != n_ch {
            return Err(PyValueError::new_err("positions and blocks must have same length"));
        }
        let mut out = vec![Complex64::new(0.0, 0.0); n_ch * p * p];
        for a in 0..n_ch {
            let own = &positions[a];
            let blk = blocks[a].as_array();
            let n = own.len();
            let offset = a * p * p;
            for i in 0..n {
                let row = own[i];
                for j in 0..n {
                    let col = own[j];
                    out[offset + row * p + col] = blk[[i, j]];
                }
            }
        }
        array(py, &[n_ch, p, p], out)
    }

    /// Build space-group channel symmetrizer matrix.
    #[pyo3(signature=(channels, atom_maps=None, rotations=None, n_operations=0))]
    fn periodic_channel_symmetry<'py>(
        &self,
        py: Python<'py>,
        channels: Vec<(usize, usize, i64)>,
        atom_maps: Option<Vec<PyReadonlyArray1<'py, i64>>>,
        rotations: Option<Vec<std::collections::HashMap<usize, PyReadonlyArray2<'py, Complex64>>>>,
        n_operations: usize,
    ) -> PyResult<Array<'py, Complex64>> {
        use std::collections::HashMap;
        let n_ch = channels.len();
        let mut out = vec![Complex64::new(0.0, 0.0); n_ch * n_ch];
        if atom_maps.is_none() || rotations.is_none() || n_operations == 0 {
            for i in 0..n_ch {
                out[i * n_ch + i] = Complex64::new(1.0, 0.0);
            }
            return array(py, &[n_ch, n_ch], out);
        }
        let maps = atom_maps.unwrap();
        let rots = rotations.unwrap();
        let mut index = HashMap::with_capacity(n_ch);
        for (i, &ch) in channels.iter().enumerate() {
            index.insert(ch, i);
        }
        let inv_n_ops = 1.0 / (n_operations as f64);
        for (atom_map, rot_dict) in maps.iter().zip(&rots) {
            let map_slice = atom_map.as_slice()?;
            for (a, &(atom, degree, order)) in channels.iter().enumerate() {
                if let Some(rot_mat) = rot_dict.get(&degree) {
                    let rot_arr = rot_mat.as_array();
                    let target_atom = map_slice[atom] as usize;
                    let d_i64 = degree as i64;
                    for other in -d_i64..=d_i64 {
                        if let Some(&source) = index.get(&(target_atom, degree, other)) {
                            let row_idx = (order + d_i64) as usize;
                            let col_idx = (other + d_i64) as usize;
                            let rot_val = rot_arr[[row_idx, col_idx]];
                            out[a * n_ch + source] += rot_val * inv_n_ops;
                        }
                    }
                }
            }
        }
        array(py, &[n_ch, n_ch], out)
    }

    /// Kerker preconditioner weights on the FFT grid.
    #[pyo3(signature=(shape, origin, step))]
    fn periodic_kerker<'py>(
        &self,
        py: Python<'py>,
        shape: [usize; 3],
        origin: [f64; 3],
        step: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        use mandacaru_core::math::SpatialGrid;
        let s_slice = step.as_slice()?;
        if s_slice.len() != 9 {
            return Err(PyValueError::new_err("step must have 9 elements"));
        }
        let mut s = [0.0; 9];
        s.copy_from_slice(s_slice);
        let grid = SpatialGrid::new(shape, origin, s, true).map_err(error)?;
        let buf = mandacaru_core::periodic::setup::fields::kerker(&self.inner, &grid).map_err(error)?;
        array(py, &[shape[0], shape[1], shape[2]], buf.as_slice().to_vec())
    }

    /// Origin phase factor on the FFT grid.
    #[pyo3(signature=(shape, origin, step))]
    fn periodic_phase<'py>(
        &self,
        py: Python<'py>,
        shape: [usize; 3],
        origin: [f64; 3],
        step: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::math::SpatialGrid;
        let s_slice = step.as_slice()?;
        if s_slice.len() != 9 {
            return Err(PyValueError::new_err("step must have 9 elements"));
        }
        let mut s = [0.0; 9];
        s.copy_from_slice(s_slice);
        let grid = SpatialGrid::new(shape, origin, s, true).map_err(error)?;
        let buf = mandacaru_core::periodic::setup::fields::phase(&self.inner, &grid).map_err(error)?;
        array(py, &[shape[0], shape[1], shape[2]], buf.as_slice().to_vec())
    }

    /// Reference electrostatic energy shift.
    #[pyo3(signature=(sigma, charges, volume))]
    fn periodic_reference(&self, sigma: f64, charges: Vec<f64>, volume: f64) -> f64 {
        use std::f64::consts::PI;
        2.0 * PI * sigma * sigma * charges.iter().sum::<f64>() / volume
    }

    /// Spinor magnetization array with initial magnetization placed along z.
    #[pyo3(signature=(m, n))]
    fn periodic_spinor_initial_magnetization<'py>(
        &self,
        py: Python<'py>,
        m: PyReadonlyArray1<'py, f64>,
        n: usize,
    ) -> PyResult<Array<'py, f64>> {
        let mut out = vec![0.0; 3 * n];
        let m_slice = m.as_slice()?;
        for (i, &v) in m_slice.iter().enumerate() {
            out[2 * n + i] = v;
        }
        array(py, &[3, n], out)
    }

    /// Convert Cartesian k-points to fractional coordinates: k @ lattice / (2*pi).
    #[pyo3(signature=(cartesian_k, lattice))]
    fn periodic_fractional_kpoints<'py>(
        &self,
        py: Python<'py>,
        cartesian_k: PyReadonlyArray2<'py, f64>,
        lattice: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let k_view = cartesian_k.as_array();
        let lat_view = lattice.as_array();
        let nk = k_view.shape()[0];
        let mut out = vec![0.0; nk * 3];
        let inv_two_pi = 1.0 / (2.0 * std::f64::consts::PI);
        for i in 0..nk {
            for j in 0..3 {
                let mut sum = 0.0;
                for c in 0..3 {
                    sum += k_view[[i, c]] * lat_view[[c, j]];
                }
                out[i * 3 + j] = sum * inv_two_pi;
            }
        }
        array(py, &[nk, 3], out)
    }

    /// Box wave vectors and norms averaged over the grid's lattice operations.
    #[pyo3(signature=(shape, origin, step))]
    fn periodic_symmetric_aliases<'py>(
        &self,
        py: Python<'py>,
        shape: [usize; 3],
        origin: [f64; 3],
        step: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
        use mandacaru_core::math::{reciprocal, SpatialGrid};
        let s_slice = step.as_slice()?;
        if s_slice.len() != 9 {
            return Err(PyValueError::new_err("step must have 9 elements"));
        }
        let mut s = [0.0; 9];
        s.copy_from_slice(s_slice);
        let grid = SpatialGrid::new(shape, origin, s, true).map_err(error)?;
        let vecs = reciprocal::symmetric_wavevectors(&self.inner, &grid).map_err(error)?;
        let nrms = reciprocal::norms(&self.inner, &grid).map_err(error)?;
        let (n1, n2, n3) = (shape[0], shape[1], shape[2]);
        let vecs_arr = array(py, &[3, n1, n2, n3], vecs.as_slice().to_vec())?;
        let nrms_arr = array(py, &[n1, n2, n3], nrms.as_slice().to_vec())?;
        Ok((vecs_arr, nrms_arr))
    }

    /// Wave vectors and kinetic multipliers of a Bloch function averaged over lattice operations.
    #[pyo3(signature=(shape, origin, step, k))]
    fn periodic_bloch_aliases<'py>(
        &self,
        py: Python<'py>,
        shape: [usize; 3],
        origin: [f64; 3],
        step: PyReadonlyArray2<'py, f64>,
        k: [f64; 3],
    ) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
        use mandacaru_core::math::{reciprocal, SpatialGrid};
        let s_slice = step.as_slice()?;
        if s_slice.len() != 9 {
            return Err(PyValueError::new_err("step must have 9 elements"));
        }
        let mut s = [0.0; 9];
        s.copy_from_slice(s_slice);
        let grid = SpatialGrid::new(shape, origin, s, true).map_err(error)?;
        let symbols = reciprocal::bloch(&self.inner, &grid, k).map_err(error)?;
        let (n1, n2, n3) = (shape[0], shape[1], shape[2]);
        let vecs_arr = array(py, &[3, n1, n2, n3], symbols.vectors.as_slice().to_vec())?;
        let sqrs_arr = array(py, &[n1, n2, n3], symbols.squares.as_slice().to_vec())?;
        Ok((vecs_arr, sqrs_arr))
    }

    /// Lattice point operations V acting on FFT frequencies.
    #[pyo3(signature=(shape, origin, step))]
    fn periodic_lattice_operations<'py>(
        &self,
        py: Python<'py>,
        shape: [usize; 3],
        origin: [f64; 3],
        step: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Vec<Array<'py, i64>>> {
        use mandacaru_core::math::{reciprocal, SpatialGrid};
        let s_slice = step.as_slice()?;
        if s_slice.len() != 9 {
            return Err(PyValueError::new_err("step must have 9 elements"));
        }
        let mut s = [0.0; 9];
        s.copy_from_slice(s_slice);
        let grid = SpatialGrid::new(shape, origin, s, true).map_err(error)?;
        let ops = reciprocal::lattice_operations(&self.inner, &grid).map_err(error)?;
        let mut out = Vec::with_capacity(ops.len());
        for v in &ops {
            let mut data = Vec::with_capacity(9);
            for i in 0..3 {
                for j in 0..3 {
                    data.push(v[(i, j)].round() as i64);
                }
            }
            out.push(array(py, &[3, 3], data)?);
        }
        Ok(out)
    }

    /// 4pi / G^2 with G=0 set to 0.
    #[pyo3(signature=(g))]
    fn periodic_coulomb_wavevector_kernel<'py>(
        &self,
        py: Python<'py>,
        g: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let g_arr = g.as_array();
        let shape = g_arr.shape();
        if shape.is_empty() || shape[0] != 3 {
            return Err(PyValueError::new_err("g must have shape (3, ...)"));
        }
        let rest_len: usize = shape[1..].iter().product();
        let mut out = Vec::with_capacity(rest_len);
        let four_pi = 4.0 * std::f64::consts::PI;
        let gx_view = g_arr.index_axis(numpy::ndarray::Axis(0), 0);
        let gy_view = g_arr.index_axis(numpy::ndarray::Axis(0), 1);
        let gz_view = g_arr.index_axis(numpy::ndarray::Axis(0), 2);
        for ((&gx, &gy), &gz) in gx_view.iter().zip(gy_view.iter()).zip(gz_view.iter()) {
            let g2 = gx * gx + gy * gy + gz * gz;
            if g2 > 0.0 {
                out.push(four_pi / g2);
            } else {
                out.push(0.0);
            }
        }
        array(py, &shape[1..], out)
    }

    /// exp(-i G . center) for given center in Bohr.
    #[pyo3(signature=(g, center))]
    fn periodic_structure_factor<'py>(
        &self,
        py: Python<'py>,
        g: PyReadonlyArrayDyn<'py, f64>,
        center: [f64; 3],
    ) -> PyResult<Array<'py, Complex64>> {
        let g_arr = g.as_array();
        let shape = g_arr.shape();
        if shape.is_empty() || shape[0] != 3 {
            return Err(PyValueError::new_err("g must have shape (3, ...)"));
        }
        let rest_len: usize = shape[1..].iter().product();
        let mut out = Vec::with_capacity(rest_len);
        let gx_view = g_arr.index_axis(numpy::ndarray::Axis(0), 0);
        let gy_view = g_arr.index_axis(numpy::ndarray::Axis(0), 1);
        let gz_view = g_arr.index_axis(numpy::ndarray::Axis(0), 2);
        for ((&gx, &gy), &gz) in gx_view.iter().zip(gy_view.iter()).zip(gz_view.iter()) {
            let dot = gx * center[0] + gy * center[1] + gz * center[2];
            out.push(Complex64::new((-dot).cos(), (-dot).sin()));
        }
        array(py, &shape[1..], out)
    }

    /// Spherical coordinates (|G|, theta, phi) of G vectors.
    #[pyo3(signature=(g))]
    fn periodic_spherical<'py>(
        &self,
        py: Python<'py>,
        g: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<(Array<'py, f64>, Array<'py, f64>, Array<'py, f64>)> {
        let g_arr = g.as_array();
        let shape = g_arr.shape();
        if shape.is_empty() || shape[0] != 3 {
            return Err(PyValueError::new_err("g must have shape (3, ...)"));
        }
        let rest_len: usize = shape[1..].iter().product();
        let mut norms = Vec::with_capacity(rest_len);
        let mut thetas = Vec::with_capacity(rest_len);
        let mut phis = Vec::with_capacity(rest_len);
        let gx_view = g_arr.index_axis(numpy::ndarray::Axis(0), 0);
        let gy_view = g_arr.index_axis(numpy::ndarray::Axis(0), 1);
        let gz_view = g_arr.index_axis(numpy::ndarray::Axis(0), 2);
        for ((&gx, &gy), &gz) in gx_view.iter().zip(gy_view.iter()).zip(gz_view.iter()) {
            let r2 = gx * gx + gy * gy + gz * gz;
            let r = r2.sqrt();
            norms.push(r);
            let safe = if r > 0.0 { r } else { 1.0 };
            let cos_t = (gz / safe).clamp(-1.0, 1.0);
            thetas.push(cos_t.acos());
            phis.push(gy.atan2(gx));
        }
        let n_arr = array(py, &shape[1..], norms)?;
        let t_arr = array(py, &shape[1..], thetas)?;
        let p_arr = array(py, &shape[1..], phis)?;
        Ok((n_arr, t_arr, p_arr))
    }

    /// All lattice translations R with |R| <= radius (rows, Bohr).
    #[pyo3(signature=(lattice, radius))]
    fn periodic_lattice_translations<'py>(
        &self,
        py: Python<'py>,
        lattice: PyReadonlyArray2<'py, f64>,
        radius: f64,
    ) -> PyResult<Array<'py, f64>> {
        let lat = lattice.as_array();
        if lat.shape() != [3, 3] {
            return Err(PyValueError::new_err("lattice must be 3x3"));
        }
        let a = Matrix3::from_fn(|i, j| lat[[i, j]]);
        let inv_t = a
            .try_inverse()
            .ok_or_else(|| PyValueError::new_err("singular lattice"))?
            .transpose();
        let b = 2.0 * std::f64::consts::PI * inv_t;
        let mut reach = [0i64; 3];
        for i in 0..3 {
            let col_norm = (b[(0, i)].powi(2) + b[(1, i)].powi(2) + b[(2, i)].powi(2)).sqrt();
            reach[i] = (radius * col_norm / (2.0 * std::f64::consts::PI)).ceil() as i64;
        }
        let rad_sq = (radius + 1e-12).powi(2);
        let mut out = Vec::new();
        for i in -reach[0]..=reach[0] {
            for j in -reach[1]..=reach[1] {
                for k in -reach[2]..=reach[2] {
                    let rx = a[(0, 0)] * i as f64 + a[(0, 1)] * j as f64 + a[(0, 2)] * k as f64;
                    let ry = a[(1, 0)] * i as f64 + a[(1, 1)] * j as f64 + a[(1, 2)] * k as f64;
                    let rz = a[(2, 0)] * i as f64 + a[(2, 1)] * j as f64 + a[(2, 2)] * k as f64;
                    if rx * rx + ry * ry + rz * rz <= rad_sq {
                        out.push(rx);
                        out.push(ry);
                        out.push(rz);
                    }
                }
            }
        }
        let n = out.len() / 3;
        array(py, &[n, 3], out)
    }

    /// Wavevectors G on the FFT grid: shape (3, n1, n2, n3).
    #[pyo3(signature=(shape, step))]
    fn periodic_wavevectors<'py>(
        &self,
        py: Python<'py>,
        shape: [usize; 3],
        step: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        use mandacaru_core::math::{reciprocal, SpatialGrid};
        let s_slice = step.as_slice()?;
        if s_slice.len() != 9 {
            return Err(PyValueError::new_err("step must have 9 elements"));
        }
        let mut s = [0.0; 9];
        s.copy_from_slice(s_slice);
        let grid = SpatialGrid::new(shape, [0.0; 3], s, true).map_err(error)?;
        let b = reciprocal::reciprocal_cell(&grid).map_err(error)?;
        let n = grid.index.points();
        let mut out = vec![0.0; 3 * n];
        for at in 0..n {
            let vec = reciprocal::box_vector(&b, shape, at);
            out[at] = vec[0];
            out[n + at] = vec[1];
            out[2 * n + at] = vec[2];
        }
        array(py, &[3, shape[0], shape[1], shape[2]], out)
    }

    /// (index, vectors, weights) of the grid alias table.
    #[pyo3(signature=(shape, origin, step))]
    fn periodic_alias_table<'py>(
        &self,
        py: Python<'py>,
        shape: [usize; 3],
        origin: [f64; 3],
        step: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Array<'py, i64>, Array<'py, f64>, Array<'py, f64>)> {
        use mandacaru_core::math::{reciprocal, SpatialGrid};
        let s_slice = step.as_slice()?;
        if s_slice.len() != 9 {
            return Err(PyValueError::new_err("step must have 9 elements"));
        }
        let mut s = [0.0; 9];
        s.copy_from_slice(s_slice);
        let grid = SpatialGrid::new(shape, origin, s, true).map_err(error)?;
        let table = reciprocal::alias_table(&self.inner, &grid).map_err(error)?;
        let n = table.index.len();
        let idx_arr = array(py, &[n], table.index.iter().map(|&i| i as i64).collect())?;
        let mut vecs_flat = vec![0.0; 3 * n];
        for i in 0..n {
            vecs_flat[i] = table.vectors[i][0];
            vecs_flat[n + i] = table.vectors[i][1];
            vecs_flat[2 * n + i] = table.vectors[i][2];
        }
        let vecs_arr = array(py, &[3, n], vecs_flat)?;
        let w_arr = array(py, &[n], table.weights.clone())?;
        Ok((idx_arr, vecs_arr, w_arr))
    }

    /// Spherical Bessel transform of tabulated radial function using trapezoidal rule.
    #[pyo3(signature=(r, values, l, q))]
    fn periodic_radial_transform<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray1<'py, f64>,
        values: PyReadonlyArray1<'py, f64>,
        l: i32,
        q: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let r_slice = r.as_slice()?;
        let v_slice = values.as_slice()?;
        let nr = r_slice.len();
        if v_slice.len() != nr || nr < 2 {
            return Err(PyValueError::new_err("r and values must have matching length >= 2"));
        }
        let mut fw = vec![0.0; nr];
        for i in 0..nr {
            let dr = if i == 0 {
                0.5 * (r_slice[1] - r_slice[0])
            } else if i == nr - 1 {
                0.5 * (r_slice[nr - 1] - r_slice[nr - 2])
            } else {
                0.5 * (r_slice[i + 1] - r_slice[i - 1])
            };
            let rad = r_slice[i];
            fw[i] = v_slice[i] * rad * rad * dr;
        }
        let q_view = q.as_array();
        let flat_q: Vec<f64> = q_view.iter().copied().collect();
        let (r_buf, _r) = copy(&self.inner, &r)?;
        let res = kernels::fourier_bessel(&self.inner, l, &r_buf, &fw, &flat_q).map_err(error)?;
        array(py, q_view.shape(), res.as_slice().to_vec())
    }

    /// Block size for k-point Bloch sums fitting the memory budget.
    #[pyo3(signature=(grid_size, m, block_max, block_bytes))]
    fn periodic_kpoint_block(&self, grid_size: usize, m: usize, block_max: usize, block_bytes: usize) -> usize {
        let largest = grid_size.max(131072);
        let per_k = 16 * m * largest;
        if per_k == 0 {
            return block_max;
        }
        block_max.min(1.max(block_bytes / per_k))
    }

    /// Wide block size for projector quadrature across k-points.
    #[pyo3(signature=(m, n_projectors, has_spin_orbit, block, block_bytes))]
    fn periodic_wide_kpoint_block(
        &self,
        m: usize,
        n_projectors: usize,
        has_spin_orbit: bool,
        block: usize,
        block_bytes: usize,
    ) -> usize {
        let so_term = if has_spin_orbit { 4 * m } else { 0 };
        let per_k = 16 * m * (m + n_projectors + so_term);
        if per_k == 0 {
            return block;
        }
        block.max(block_bytes / per_k)
    }

    /// Zero spin-orbit matrix of shape (nk, 2M, 2M).
    #[pyo3(signature=(nk, m))]
    fn periodic_empty_spin_orbit<'py>(
        &self,
        py: Python<'py>,
        nk: usize,
        m: usize,
    ) -> PyResult<Array<'py, Complex64>> {
        let dim = 2 * m;
        let total = nk * dim * dim;
        array(py, &[nk, dim, dim], vec![Complex64::default(); total])
    }

    /// Batched spin-orbit matrices across k-points: (nk, 2M, 2M).
    #[pyo3(signature=(projections, channels))]
    fn periodic_spin_orbit_matrices<'py>(
        &self,
        py: Python<'py>,
        projections: PyReadonlyArray3<'py, Complex64>,
        channels: Vec<(usize, bool, Vec<usize>, PyReadonlyArray2<'py, Complex64>)>,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::models::spin_orbit::{self, Channel, Projectors};
        let nk = projections.shape()[0];
        let m = projections.shape()[1];
        let count = projections.shape()[2];
        if channels.is_empty() {
            let dim = 2 * m;
            return array(py, &[nk, dim, dim], vec![Complex64::default(); nk * dim * dim]);
        }
        let mut owned = Vec::with_capacity(channels.len());
        let _channels_res = self
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
        let proj_slice = projections.as_slice()?;
        let stride = m * count;
        let dim = 2 * m;
        let mut out = vec![Complex64::default(); nk * dim * dim];
        py.detach(|| -> mandacaru_core::Result<()> {
            for (k, out_k) in out.chunks_exact_mut(dim * dim).enumerate() {
                let left_k = &proj_slice[k * stride..(k + 1) * stride];
                let mut left_buf = self.inner.allocate::<Complex64>(left_k.len())?;
                left_buf.as_mut_slice().copy_from_slice(left_k);
                let h_k = spin_orbit::pair(
                    &self.inner,
                    Projectors {
                        left: left_buf.as_slice(),
                        right: left_buf.as_slice(),
                        rows: m,
                        columns: m,
                        count,
                    },
                    &owned,
                    true,
                )?;
                out_k.copy_from_slice(h_k.as_slice());
            }
            Ok(())
        })
        .map_err(error)?;
        array(py, &[nk, dim, dim], out)
    }

    /// Average a flat grid field over space group operations.
    #[pyo3(signature=(permutations, values))]
    fn periodic_symmetrize_field<'py>(
        &self,
        py: Python<'py>,
        permutations: Vec<PyReadonlyArray1<'py, i64>>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let val_slice = values.as_slice()?;
        let n = val_slice.len();
        let n_ops = permutations.len();
        if n_ops == 0 {
            return array(py, &[n], val_slice.to_vec());
        }
        let inv_ops = 1.0 / n_ops as f64;
        let mut out = vec![0.0f64; n];
        for perm in &permutations {
            let perm_slice = perm.as_slice()?;
            for (i, &p) in perm_slice.iter().enumerate() {
                let p = p as usize;
                if i < n && p < n {
                    out[i] += val_slice[p];
                }
            }
        }
        for v in out.iter_mut() {
            *v *= inv_ops;
        }
        array(py, &[n], out)
    }

    /// Average compensation multipole moments over space group operations.
    #[pyo3(signature=(atom_maps, rotations, keys, values))]
    fn periodic_symmetrize_moments<'py>(
        &self,
        py: Python<'py>,
        atom_maps: Vec<PyReadonlyArray1<'py, i64>>,
        rotations: Vec<std::collections::HashMap<usize, PyReadonlyArray2<'py, Complex64>>>,
        keys: Vec<(usize, usize, i32)>,
        values: PyReadonlyArray1<'py, Complex64>,
    ) -> PyResult<Array<'py, Complex64>> {
        use std::collections::HashMap;
        let val_slice = values.as_slice()?;
        let n_ops = atom_maps.len();
        if n_ops == 0 {
            return array(py, &[keys.len()], val_slice.to_vec());
        }
        let inv_ops = 1.0 / n_ops as f64;
        let q_map: HashMap<(usize, usize, i32), Complex64> = keys
            .iter()
            .cloned()
            .zip(val_slice.iter().cloned())
            .collect();
        let mut out = vec![Complex64::default(); keys.len()];
        for (atom_map, rot_map) in atom_maps.iter().zip(&rotations) {
            let map_slice = atom_map.as_slice()?;
            for (idx, &(atom, l, m)) in keys.iter().enumerate() {
                if let Some(t_arr) = rot_map.get(&l) {
                    let source = map_slice.get(atom).copied().unwrap_or(atom as i64) as usize;
                    let t_slice = t_arr.as_slice()?;
                    let dim = 2 * l + 1;
                    let row = (m + l as i32) as usize;
                    let mut total = Complex64::default();
                    for mp in -(l as i32)..=(l as i32) {
                        if let Some(&val) = q_map.get(&(source, l, mp)) {
                            let col = (mp + l as i32) as usize;
                            total += t_slice[row * dim + col] * val;
                        }
                    }
                    out[idx] += total;
                }
            }
        }
        for v in out.iter_mut() {
            *v *= inv_ops;
        }
        array(py, &[keys.len()], out)
    }

    /// Radius beyond which a radial table vanishes within tolerance * peak.
    #[pyo3(signature=(r, values, tolerance=1e-12))]
    fn periodic_radial_support(
        &self,
        r: PyReadonlyArray1<'_, f64>,
        values: PyReadonlyArray1<'_, f64>,
        tolerance: f64,
    ) -> PyResult<f64> {
        let r_slice = r.as_slice()?;
        let val_slice = values.as_slice()?;
        if r_slice.is_empty() || r_slice.len() != val_slice.len() {
            return Err(PyValueError::new_err("mismatched or empty radial table"));
        }
        let peak = val_slice.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let thresh = tolerance * peak;
        let edge = match val_slice.iter().rposition(|v| v.abs() > thresh) {
            Some(last) => r_slice[(last + 1).min(r_slice.len() - 1)],
            None => r_slice[0],
        };
        Ok(edge)
    }

    /// Sizing of sphere quadrature chunks.
    #[pyo3(signature=(n_kpoints, n_functions, sphere_bytes, sphere_chunk))]
    fn periodic_sphere_chunk_size(
        &self,
        n_kpoints: usize,
        n_functions: usize,
        sphere_bytes: usize,
        sphere_chunk: usize,
    ) -> usize {
        let nk = n_kpoints.max(1);
        let nf = n_functions.max(1);
        let per_pt = 32 * nk * nf;
        let fit = if per_pt == 0 { sphere_chunk } else { sphere_bytes / per_pt };
        let clamped = fit.min(sphere_chunk);
        clamped.max(512)
    }

    /// Subgroup of rotations mapping a k-point mesh onto itself (mod 1).
    #[pyo3(signature=(rotations, mesh, tolerance=1e-8))]
    fn periodic_filter_mesh_rotations<'py>(
        &self,
        rotations: Vec<[[f64; 3]; 3]>,
        mesh: PyReadonlyArray2<'py, f64>,
        tolerance: f64,
    ) -> PyResult<Vec<usize>> {
        let shape = mesh.shape();
        if shape[1] != 3 {
            return Err(PyValueError::new_err("mesh must have shape (nk, 3)"));
        }
        let nk = shape[0];
        let mesh_slice = mesh.as_slice()?;
        let mut k_points = Vec::with_capacity(nk);
        for i in 0..nk {
            k_points.push([
                mesh_slice[i * 3],
                mesh_slice[i * 3 + 1],
                mesh_slice[i * 3 + 2],
            ]);
        }
        let mut keep = Vec::new();
        for (s, w) in rotations.iter().enumerate() {
            let mut all_on_mesh = true;
            for k in &k_points {
                let kx = k[0] * w[0][0] + k[1] * w[1][0] + k[2] * w[2][0];
                let ky = k[0] * w[0][1] + k[1] * w[1][1] + k[2] * w[2][1];
                let kz = k[0] * w[0][2] + k[1] * w[1][2] + k[2] * w[2][2];
                let matches = k_points.iter().any(|m| {
                    let dx = kx - m[0];
                    let dy = ky - m[1];
                    let dz = kz - m[2];
                    (dx - dx.round()).abs() < tolerance
                        && (dy - dy.round()).abs() < tolerance
                        && (dz - dz.round()).abs() < tolerance
                });
                if !matches {
                    all_on_mesh = false;
                    break;
                }
            }
            if all_on_mesh {
                keep.push(s);
            }
        }
        Ok(keep)
    }
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
    #[pyo3(signature=(variables,*,omega=mandacaru_core::models::HSE06_OMEGA))]
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
    #[pyo3(signature=(variables,*,omega=mandacaru_core::models::HSE06_OMEGA,exact_fraction=mandacaru_core::models::HSE06_EXACT_FRACTION))]
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
    fn read_snapshot_header(
        &self,
        py: Python<'_>,
        path: &str,
        kind: &str,
    ) -> PyResult<persistence::PySnapshot> {
        let inner = py
            .detach(|| {
                mandacaru_core::persistence::Document::read_header(
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
    fn read_parquet<'py>(&self, py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
        super::parquet::read_parquet_with_ctx(&self.inner, py, path)
    }
    fn read_parquet_metadata<'py>(
        &self,
        py: Python<'py>,
        path: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        super::parquet::read_parquet_metadata_with_ctx(&self.inner, py, path)
    }
    #[pyo3(signature = (path, columns, metadata, compression = "zstd"))]
    fn write_parquet(
        &self,
        py: Python<'_>,
        path: &str,
        columns: &Bound<'_, PyDict>,
        metadata: &Bound<'_, PyDict>,
        compression: &str,
    ) -> PyResult<()> {
        super::parquet::write_parquet_with_ctx(
            &self.inner,
            py,
            path,
            columns,
            metadata,
            compression,
        )
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

    /// With consume=True, remove input arrays as they are copied, including if
    /// validation of a later array fails. The default leaves the dictionary intact.
    #[pyo3(signature = (metadata_json, arrays, *, consume=false))]
    fn periodic_problem(
        &self,
        py: Python<'_>,
        metadata_json: &str,
        arrays: &Bound<'_, PyDict>,
        consume: bool,
    ) -> PyResult<PyCrystal> {
        let snapshot_json = metadata_json;
        let arrays = periodic::input_arrays(&self.inner, arrays, metadata_json, None, consume)?;
        let _input = self.inner.reserve(snapshot_json.len()).map_err(error)?;
        let owned = snapshot_json.to_owned();
        let inner = py
            .detach(|| Crystal::from_arrays(&self.inner, &owned, arrays))
            .map_err(error)?;
        Ok(PyCrystal {
            inner: Arc::new(inner),
            setup: None,
        })
    }
    #[pyo3(signature = (request_json, dataset_paths))]
    fn periodic_problem_from_request(
        &self,
        py: Python<'_>,
        request_json: &str,
        dataset_paths: Vec<String>,
    ) -> PyResult<periodic::PyCrystal> {
        self._periodic_problem_from_request(py, request_json, dataset_paths)
    }
    #[doc(hidden)]
    fn _periodic_problem_from_request(
        &self,
        py: Python<'_>,
        request_json: &str,
        dataset_paths: Vec<String>,
    ) -> PyResult<periodic::PyCrystal> {
        let v: serde_json::Value = serde_json::from_str(request_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let mut input = mandacaru_core::periodic::setup::parse_input_json(&v).map_err(error)?;
        if input.dataset_files.is_none() && !dataset_paths.is_empty() {
            let mut files = Vec::with_capacity(dataset_paths.len());
            for p in &dataset_paths {
                let digest = mandacaru_core::paw::digest::sha256_file(std::path::Path::new(p)).map_err(error)?;
                files.push((p.clone(), digest));
            }
            // The supplied paths are indexed by species. Store them in atom
            // order, including when the species and atom counts happen to match.
            input.dataset_files = Some(input.atoms.iter().map(|atom| {
                files.get(atom.species).cloned().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("atom dataset index is out of range")
                })
            }).collect::<PyResult<Vec<_>>>()?);
        }
        let mut datasets = Vec::with_capacity(dataset_paths.len());
        for p in &dataset_paths {
            datasets.push(std::sync::Arc::new(
                mandacaru_core::paw::record::PawDataset::read(&self.inner, std::path::Path::new(p))
                    .map_err(error)?,
            ));
        }
        let setup = mandacaru_core::periodic::setup::create_setup(&self.inner, input, datasets)
            .map_err(error)?;
        let setup_arc = std::sync::Arc::new(setup);
        let setup_for_build = setup_arc.clone();
        let crystal = py.detach(|| setup_for_build.build(&self.inner)).map_err(error)?;
        Ok(periodic::PyCrystal {
            inner: std::sync::Arc::new(crystal),
            setup: Some(setup_arc),
        })
    }
    fn load_crystal(&self, py: Python<'_>, path: &str) -> PyResult<PyCrystal> {
        let inner = py
            .detach(|| Crystal::load(&self.inner, path))
            .map_err(error)?;
        Ok(PyCrystal {
            inner: Arc::new(inner),
            setup: None,
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
    #[pyo3(signature=(grid,densities,*,functional="lda",kinetic_densities=None,gradient_densities=None,relativistic=false))]
    fn exchange_correlation<'py>(
        &self,
        py: Python<'py>,
        grid: &PySpatialGrid,
        densities: PyReadonlyArray2<'py, f64>,
        functional: &str,
        kinetic_densities: Option<PyReadonlyArray2<'py, f64>>,
        gradient_densities: Option<PyReadonlyArray2<'py, f64>>,
        relativistic: bool,
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
        if gradient_densities
            .as_ref()
            .is_some_and(|v| v.shape() != [6, grid.inner.index.points()])
        {
            return Err(PyValueError::new_err(
                "gradient densities must have shape (6,ngrid)",
            ));
        }
        let grad = gradient_densities
            .as_ref()
            .map(|v| copy(&self.inner, v))
            .transpose()?;
        let value = py
            .detach(|| {
                mandacaru_core::models::xc::evaluate_fields_relativistic(
                    &self.inner,
                    &grid.inner,
                    &rho,
                    kinetic.as_ref().map(|v| v.0.as_slice()),
                    grad.as_ref().map(|v| v.0.as_slice()),
                    functional,
                    relativistic,
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
        if let Some(gp) = value.gradient_potentials {
            out.set_item(
                "gradient_potentials",
                array(py, &[6, grid.inner.index.points()], gp.as_slice().to_vec())?,
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
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(geometry, grid, metadata, arrays, *, kinetic="spectral", cholesky_tolerance=1e-10, precision="float64", consume=false))]
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
        consume: bool,
    ) -> PyResult<molecular::PyIntegrals> {
        let precision =
            mandacaru_core::math::storage::Precision::parse(precision).map_err(error)?;
        let mut tables = mandacaru_core::paw::molecular::Arrays::new();
        for py_key in arrays.keys().iter() {
            let key: String = py_key.extract()?;
            let value = arrays.get_item(&py_key)?.ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err("molecular array disappeared during transfer")
            })?;
            let values: PyReadonlyArrayDyn<'_, f64> = value.extract()?;
            super::layout(&values)?;
            tables.insert(
                key,
                self.inner.copy_slice(values.as_slice()?).map_err(error)?,
            );
            if consume {
                arrays.del_item(&py_key)?;
            }
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
    #[pyo3(signature = (threads=None, memory_limit_bytes=34359738368, disk_offload_bytes=None, scratch_dir=None))]
    fn new(
        threads: Option<usize>,
        memory_limit_bytes: usize,
        disk_offload_bytes: Option<usize>,
        scratch_dir: Option<std::path::PathBuf>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Context::with_offload_and_scratch(
                threads,
                memory_limit_bytes,
                disk_offload_bytes.unwrap_or(mandacaru_core::context::DEFAULT_OFFLOAD_BYTES),
                scratch_dir,
            )
            .map_err(error)?,
        })
    }
    #[getter]
    fn disk_offload_bytes(&self) -> usize {
        self.inner.offload_threshold()
    }
    /// Hands the pages the allocator holds free back to the operating system; `False` on a
    /// platform without such a call. A best-effort hygiene step between calculations.
    fn release_freed_memory(&self) -> bool {
        self.inner.release_freed_memory()
    }
    /// Bytes this context's archives have written to the scratch folder so far.
    #[getter]
    fn disk_archived_bytes(&self) -> usize {
        self.inner.archived_bytes()
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
    #[pyo3(signature=(geometry, functional, volume_ratios=None, s_r=None, r_max_angstrom=None, l_decay_angstrom=None))]
    fn tkatchenko_scheffler<'py>(
        &self,
        py: Python<'py>,
        geometry: &PyGeometry,
        functional: &str,
        volume_ratios: Option<PyReadonlyArray1<'py, f64>>,
        s_r: Option<f64>,
        r_max_angstrom: Option<f64>,
        l_decay_angstrom: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ratios = match volume_ratios {
            Some(ref v) => Some(copy(&self.inner, v)?),
            None => None,
        };
        let ratios_slice = ratios.as_ref().map(|(vec, _res)| vec.as_slice());
        let out = py
            .detach(|| {
                mandacaru_core::models::ts::tkatchenko_scheffler(
                    &geometry.inner,
                    functional,
                    ratios_slice,
                    s_r,
                    r_max_angstrom,
                    l_decay_angstrom,
                )
            })
            .map_err(error)?;
        let n = geometry.inner.numbers().len();
        let dict = PyDict::new(py);
        dict.set_item("energy_ha", out.energy_ha)?;
        dict.set_item("energy_ev", out.energy_ev)?;
        let mut forces_ha_flat = Vec::with_capacity(n * 3);
        let mut forces_ev_flat = Vec::with_capacity(n * 3);
        for i in 0..n {
            for a in 0..3 {
                forces_ha_flat.push(out.forces_ha_bohr[i][a]);
                forces_ev_flat.push(out.forces_ev_angstrom[i][a]);
            }
        }
        dict.set_item("forces_ha_bohr", array(py, &[n, 3], forces_ha_flat)?)?;
        dict.set_item("forces_ev_angstrom", array(py, &[n, 3], forces_ev_flat)?)?;
        Ok(dict)
    }

    #[pyo3(signature=(weights, pot_r))]
    fn periodic_short_range_weights<'py>(
        &self,
        py: Python<'py>,
        weights: PyReadonlyArray1<'py, f64>,
        pot_r: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let w = weights.as_slice()?;
        let p = pot_r.as_slice()?;
        let n_r = p.len();
        if n_r == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("pot_r must not be empty"));
        }
        let n_dirs = w.len() / n_r;
        let mut out = Vec::with_capacity(w.len());
        for i in 0..w.len() {
            let r_idx = i / n_dirs;
            out.push(w[i] * p[r_idx]);
        }
        array(py, &[w.len()], out)
    }

    #[pyo3(signature=(max_val, step))]
    fn periodic_remainder_q<'py>(
        &self,
        py: Python<'py>,
        max_val: f64,
        step: f64,
    ) -> PyResult<Array<'py, f64>> {
        let n = (max_val / step) as usize + 2;
        let mut q = Vec::with_capacity(n);
        for i in 0..n {
            q.push(i as f64 * step);
        }
        array(py, &[n], q)
    }

    #[pyo3(signature=(c, rows, members, block))]
    fn periodic_accumulate_projection_block<'py>(
        &self,
        mut c: PyReadwriteArray3<'py, Complex64>,
        rows: Vec<usize>,
        members: Vec<usize>,
        block: PyReadonlyArray3<'py, Complex64>,
    ) -> PyResult<()> {
        let b_arr = block.as_array();
        let mut c_arr = c.as_array_mut();
        let nk = b_arr.shape()[0];
        for k in 0..nk {
            for (bi, &r) in rows.iter().enumerate() {
                for (bj, &m) in members.iter().enumerate() {
                    c_arr[[k, r, m]] += b_arr[[k, bi, bj]];
                }
            }
        }
        Ok(())
    }

    #[pyo3(signature=(columns, weights))]
    fn periodic_weighted_columns<'py>(
        &self,
        py: Python<'py>,
        columns: Vec<PyReadonlyArray1<'py, Complex64>>,
        weights: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Array<'py, Complex64>> {
        let w = weights.as_slice()?;
        let p = columns.len();
        let npts = w.len();
        let mut out = Vec::with_capacity(p * npts);
        for col in &columns {
            let s = col.as_slice()?;
            for i in 0..npts {
                out.push(s[i] * w[i]);
            }
        }
        array(py, &[p, npts], out)
    }

    #[pyo3(signature=(n))]
    fn periodic_uniform_weights<'py>(
        &self,
        py: Python<'py>,
        n: usize,
    ) -> PyResult<Array<'py, f64>> {
        let val = if n > 0 { 1.0 / n as f64 } else { 0.0 };
        array(py, &[n], vec![val; n])
    }

    #[pyo3(signature=(weights))]
    fn periodic_normalize_weights<'py>(
        &self,
        py: Python<'py>,
        weights: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let w = weights.as_slice()?;
        let sum: f64 = w.iter().sum();
        let inv = if sum.abs() > 1e-15 { 1.0 / sum } else { 1.0 };
        let out: Vec<f64> = w.iter().map(|&x| x * inv).collect();
        array(py, &[w.len()], out)
    }

    #[pyo3(signature=(n))]
    fn periodic_range_indices<'py>(
        &self,
        py: Python<'py>,
        n: usize,
    ) -> PyResult<Array<'py, i64>> {
        let out: Vec<i64> = (0..n as i64).collect();
        array(py, &[n], out)
    }

    #[pyo3(signature=(shape, step, cell, scale))]
    fn periodic_grid_spans_cell<'py>(
        &self,
        shape: [usize; 3],
        step: PyReadonlyArray2<'py, f64>,
        cell: PyReadonlyArray2<'py, f64>,
        scale: f64,
    ) -> PyResult<bool> {
        let step_arr = step.as_slice()?;
        let cell_arr = cell.as_slice()?;
        if step_arr.len() != 9 || cell_arr.len() != 9 {
            return Err(PyValueError::new_err("step and cell must be 3x3"));
        }
        for i in 0..3 {
            for j in 0..3 {
                let spanned = shape[i] as f64 * step_arr[j * 3 + i];
                let target = cell_arr[i * 3 + j] * scale;
                if (spanned - target).abs() > 1e-6 {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    #[pyo3(signature=(shape, step))]
    fn periodic_lattice_vectors<'py>(
        &self,
        py: Python<'py>,
        shape: [usize; 3],
        step: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let step_arr = step.as_slice()?;
        if step_arr.len() != 9 {
            return Err(PyValueError::new_err("step must be 3x3"));
        }
        let mut out = vec![0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                out[i * 3 + j] = step_arr[i * 3 + j] * shape[j] as f64;
            }
        }
        array(py, &[3, 3], out)
    }

    #[pyo3(signature=(lattice_columns))]
    fn periodic_reciprocal_vectors<'py>(
        &self,
        py: Python<'py>,
        lattice_columns: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let cell_slice = lattice_columns.as_slice()?;
        if cell_slice.len() != 9 {
            return Err(PyValueError::new_err("lattice_columns must be 3x3"));
        }
        let a = nalgebra::Matrix3::from_row_slice(cell_slice);
        let inv = a.try_inverse().ok_or_else(|| PyValueError::new_err("singular lattice matrix"))?;
        let b = inv.transpose() * (2.0 * std::f64::consts::PI);
        let mut out = vec![0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                out[i * 3 + j] = b[(i, j)];
            }
        }
        array(py, &[3, 3], out)
    }

    #[pyo3(signature=(shape, step))]
    fn periodic_cell_volume(
        &self,
        shape: [usize; 3],
        step: PyReadonlyArray2<f64>,
    ) -> PyResult<f64> {
        let step_arr = step.as_slice()?;
        if step_arr.len() != 9 {
            return Err(PyValueError::new_err("step must be 3x3"));
        }
        let mut cell = nalgebra::Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                cell[(i, j)] = step_arr[i * 3 + j] * shape[j] as f64;
            }
        }
        Ok(cell.determinant().abs())
    }

    #[pyo3(signature=(shape, origin, step))]
    fn periodic_grid_origin<'py>(
        &self,
        py: Python<'py>,
        shape: [usize; 3],
        origin: [f64; 3],
        step: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        use mandacaru_core::math::SpatialGrid;
        let s_slice = step.as_slice()?;
        if s_slice.len() != 9 {
            return Err(PyValueError::new_err("step must have 9 elements"));
        }
        let mut s = [0.0; 9];
        s.copy_from_slice(s_slice);
        let grid = SpatialGrid::new(shape, origin, s, true).map_err(error)?;
        let pt0 = grid.point(0);
        array(py, &[3], vec![pt0[0], pt0[1], pt0[2]])
    }

    #[pyo3(signature=(g, center, radial, l, m))]
    fn periodic_multipole_transform<'py>(
        &self,
        py: Python<'py>,
        g: PyReadonlyArrayDyn<'py, f64>,
        center: [f64; 3],
        radial: PyReadonlyArrayDyn<'py, f64>,
        l: i32,
        m: i32,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::math::angular;
        let g_shape = g.shape();
        if g_shape.is_empty() || g_shape[0] != 3 {
            return Err(PyValueError::new_err("G must have shape (3, ...)"));
        }
        let n_pts: usize = g_shape[1..].iter().product();
        let g_slice = g.as_slice()?;
        let rad_slice = radial.as_slice()?;
        if rad_slice.len() != n_pts {
            return Err(PyValueError::new_err(format!(
                "radial size ({}) does not match G spatial size ({})",
                rad_slice.len(),
                n_pts
            )));
        }
        let prefactor = 4.0 * std::f64::consts::PI * Complex64::new(0.0, -1.0).powi(l);
        let mut out = Vec::with_capacity(n_pts);
        for i in 0..n_pts {
            let gx = g_slice[i];
            let gy = g_slice[n_pts + i];
            let gz = g_slice[2 * n_pts + i];
            let y_lm = angular::complex(l as usize, m, [gx, gy, gz]);
            let phase = Complex64::new(0.0, -(gx * center[0] + gy * center[1] + gz * center[2])).exp();
            let val = prefactor * y_lm * rad_slice[i] * phase;
            out.push(val);
        }
        array(py, &g_shape[1..], out)
    }

    #[pyo3(signature=(shape, origin, step, dv, values))]
    fn periodic_to_reciprocal<'py>(
        &self,
        py: Python<'py>,
        shape: [usize; 3],
        origin: [f64; 3],
        step: PyReadonlyArray2<'py, f64>,
        dv: f64,
        values: &Bound<'py, PyAny>,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::math::{fft::Fft3, reciprocal, SpatialGrid};
        let s_slice = step.as_slice()?;
        if s_slice.len() != 9 {
            return Err(PyValueError::new_err("step must have 9 elements"));
        }
        let mut s = [0.0; 9];
        s.copy_from_slice(s_slice);
        let grid = SpatialGrid::new(shape, origin, s, true).map_err(error)?;
        let n = grid.index.points();
        let mut buf = if let Ok(c_arr) = values.extract::<PyReadonlyArrayDyn<'py, Complex64>>() {
            let sl = c_arr.as_slice()?;
            if sl.len() != n {
                return Err(PyValueError::new_err(format!(
                    "values length ({}) does not match grid points ({})",
                    sl.len(),
                    n
                )));
            }
            sl.to_vec()
        } else if let Ok(f_arr) = values.extract::<PyReadonlyArrayDyn<'py, f64>>() {
            let sl = f_arr.as_slice()?;
            if sl.len() != n {
                return Err(PyValueError::new_err(format!(
                    "values length ({}) does not match grid points ({})",
                    sl.len(),
                    n
                )));
            }
            sl.iter().map(|&x| Complex64::new(x, 0.0)).collect()
        } else {
            return Err(PyValueError::new_err("values must be float or complex array"));
        };

        let mut fft = Fft3::new(&self.inner, shape).map_err(error)?;
        fft.transform(&mut buf, false).map_err(error)?;
        let b = reciprocal::reciprocal_cell(&grid).map_err(error)?;
        for at in 0..n {
            let g = reciprocal::box_vector(&b, shape, at);
            let phase = Complex64::new(0.0, -(origin[0] * g[0] + origin[1] * g[1] + origin[2] * g[2])).exp();
            buf[at] = buf[at] * phase * dv;
        }
        array(py, &[shape[0], shape[1], shape[2]], buf)
    }

    #[pyo3(signature=(shape, origin, step, transform))]
    fn periodic_to_real<'py>(
        &self,
        py: Python<'py>,
        shape: [usize; 3],
        origin: [f64; 3],
        step: PyReadonlyArray2<'py, f64>,
        transform: PyReadonlyArrayDyn<'py, Complex64>,
    ) -> PyResult<Array<'py, Complex64>> {
        use mandacaru_core::math::{fft::Fft3, reciprocal, SpatialGrid};
        let s_slice = step.as_slice()?;
        if s_slice.len() != 9 {
            return Err(PyValueError::new_err("step must have 9 elements"));
        }
        let mut s = [0.0; 9];
        s.copy_from_slice(s_slice);
        let grid = SpatialGrid::new(shape, origin, s, true).map_err(error)?;
        let n = grid.index.points();
        let t_slice = transform.as_slice()?;
        if t_slice.len() != n {
            return Err(PyValueError::new_err(format!(
                "transform length ({}) does not match grid points ({})",
                t_slice.len(),
                n
            )));
        }
        let b = reciprocal::reciprocal_cell(&grid).map_err(error)?;
        let mut buf = vec![Complex64::default(); n];
        for at in 0..n {
            let g = reciprocal::box_vector(&b, shape, at);
            let phase = Complex64::new(0.0, origin[0] * g[0] + origin[1] * g[1] + origin[2] * g[2]).exp();
            buf[at] = t_slice[at] * phase;
        }
        let mut fft = Fft3::new(&self.inner, shape).map_err(error)?;
        fft.transform(&mut buf, true).map_err(error)?;
        let shape_vec = grid.index.shape();
        let cell = grid.step * nalgebra::Matrix3::from_diagonal(&nalgebra::Vector3::from_fn(|i, _| shape_vec[i] as f64));
        let cell_vol = cell.determinant().abs();
        let scale = n as f64 / cell_vol;
        for at in 0..n {
            buf[at] = buf[at] * scale;
        }
        array(py, &[n], buf)
    }

    #[pyo3(signature=(index, weights, box_values, alias_values, shape))]
    fn periodic_alias_combine<'py>(
        &self,
        py: Python<'py>,
        index: PyReadonlyArray1<'py, i64>,
        weights: PyReadonlyArray1<'py, f64>,
        box_values: &Bound<'py, PyAny>,
        alias_values: &Bound<'py, PyAny>,
        shape: Vec<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let idx_slice = index.as_slice()?;
        let w_slice = weights.as_slice()?;
        let n_alias = idx_slice.len();
        if w_slice.len() != n_alias {
            return Err(PyValueError::new_err("index and weights must have matching lengths"));
        }

        if let (Ok(box_c), Ok(alias_c)) = (
            box_values.extract::<PyReadonlyArrayDyn<'py, Complex64>>(),
            alias_values.extract::<PyReadonlyArrayDyn<'py, Complex64>>(),
        ) {
            let b_slice = box_c.as_slice()?;
            let a_slice = alias_c.as_slice()?;
            let mut out = b_slice.to_vec();
            if n_alias > 0 && !out.is_empty() {
                let n_box = shape.iter().rev().take(3).product::<usize>();
                let n_batch = if n_box > 0 { b_slice.len() / n_box } else { 1 };
                let a_lead = if n_alias > 0 { a_slice.len() / n_alias } else { 1 };
                if n_batch == a_lead {
                    for b in 0..n_batch {
                        for k in 0..n_alias {
                            let i = idx_slice[k] as usize;
                            if i < n_box {
                                let base = b_slice[b * n_box + i];
                                out[b * n_box + i] += (a_slice[b * n_alias + k] - base) * w_slice[k];
                            }
                        }
                    }
                }
            }
            Ok(array(py, &shape, out)?.into_any())
        } else if let (Ok(box_f), Ok(alias_f)) = (
            box_values.extract::<PyReadonlyArrayDyn<'py, f64>>(),
            alias_values.extract::<PyReadonlyArrayDyn<'py, f64>>(),
        ) {
            let b_slice = box_f.as_slice()?;
            let a_slice = alias_f.as_slice()?;
            let mut out = b_slice.to_vec();
            if n_alias > 0 && !out.is_empty() {
                let n_box = shape.iter().rev().take(3).product::<usize>();
                let n_batch = if n_box > 0 { b_slice.len() / n_box } else { 1 };
                let a_lead = if n_alias > 0 { a_slice.len() / n_alias } else { 1 };
                if n_batch == a_lead {
                    for b in 0..n_batch {
                        for k in 0..n_alias {
                            let i = idx_slice[k] as usize;
                            if i < n_box {
                                let base = b_slice[b * n_box + i];
                                out[b * n_box + i] += (a_slice[b * n_alias + k] - base) * w_slice[k];
                            }
                        }
                    }
                }
            }
            Ok(array(py, &shape, out)?.into_any())
        } else {
            Err(PyValueError::new_err("box_values and alias_values must be arrays of float or complex"))
        }
    }
}
