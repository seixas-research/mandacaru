//! PyO3 bindings for quantum chemistry tensor factorization.

use crate::arrays::{array, layout, Array};
use mandacaru_core::quantum::factorization as native;
use num_complex::Complex64 as NumComplex64;
use numpy::{
    Complex64, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn dmatrix_to_row_major(mat: &nalgebra::DMatrix<f64>) -> Vec<f64> {
    let (nrows, ncols) = (mat.nrows(), mat.ncols());
    let mut data = Vec::with_capacity(nrows * ncols);
    for r in 0..nrows {
        for c in 0..ncols {
            data.push(mat[(r, c)]);
        }
    }
    data
}

fn dmatrix_complex_to_row_major(mat: &nalgebra::DMatrix<NumComplex64>) -> Vec<Complex64> {
    let (nrows, ncols) = (mat.nrows(), mat.ncols());
    let mut data = Vec::with_capacity(nrows * ncols);
    for r in 0..nrows {
        for c in 0..ncols {
            let z = mat[(r, c)];
            data.push(Complex64::new(z.re, z.im));
        }
    }
    data
}

#[pyfunction]
#[pyo3(signature = (h_pq, g_pqrs, tol=1e-8, coefficient_tol=1e-10, max_leaves=None, constant=0.0))]
pub fn double_factorization<'py>(
    py: Python<'py>,
    h_pq: PyReadonlyArray2<'py, f64>,
    g_pqrs: PyReadonlyArray4<'py, f64>,
    tol: f64,
    coefficient_tol: f64,
    max_leaves: Option<usize>,
    constant: f64,
) -> PyResult<(
    usize,
    f64,
    Array<'py, Complex64>,
    Array<'py, f64>,
    Array<'py, f64>,
    Vec<(f64, Array<'py, f64>, Array<'py, f64>)>,
    f64,
)> {
    layout(&h_pq)?;
    layout(&g_pqrs)?;

    let h_shape = h_pq.shape();
    if h_shape[0] != h_shape[1] {
        return Err(PyValueError::new_err(format!(
            "expected square one-body matrix, got {:?}",
            h_shape
        )));
    }
    let m = h_shape[0];
    let g_shape = g_pqrs.shape();
    if g_shape != [m, m, m, m] {
        return Err(PyValueError::new_err(format!(
            "two-body tensor ({}, {}, {}, {}) does not match a {}-mode one-body tensor",
            g_shape[0], g_shape[1], g_shape[2], g_shape[3], m
        )));
    }

    let h_slice = h_pq.as_slice()?;
    let g_slice = g_pqrs.as_slice()?;

    let res = native::double_factorization(
        h_slice,
        g_slice,
        m,
        tol,
        coefficient_tol,
        max_leaves,
        constant,
    )
    .map_err(PyValueError::new_err)?;

    let one_body_data = dmatrix_complex_to_row_major(&res.one_body);
    let one_body_arr = array(py, &[m, m], one_body_data)?;

    let ob_rot_data = dmatrix_to_row_major(&res.one_body_orbitals);
    let ob_rot_arr = array(py, &[m, m], ob_rot_data)?;

    let ob_energies = res.one_body_energies.as_slice().to_vec();
    let ob_energies_arr = array(py, &[m], ob_energies)?;

    let mut py_leaves = Vec::with_capacity(res.leaves.len());
    for leaf in res.leaves {
        let orb_data = dmatrix_to_row_major(&leaf.orbitals);
        let orb_arr = array(py, &[m, m], orb_data)?;
        let coeff_data = leaf.coefficients.as_slice().to_vec();
        let coeff_arr = array(py, &[m], coeff_data)?;
        py_leaves.push((leaf.weight, orb_arr, coeff_arr));
    }

    Ok((
        res.n_modes,
        res.constant,
        one_body_arr,
        ob_rot_arr,
        ob_energies_arr,
        py_leaves,
        res.truncation_error,
    ))
}

#[pyfunction]
#[pyo3(signature = (h, eri=None, packed_eri=None, factors=None, coupling=None, tol=1e-8, coefficient_tol=1e-10, max_leaves=None, constant=0.0))]
pub fn spatial_double_factorization<'py>(
    py: Python<'py>,
    h: PyReadonlyArray2<'py, f64>,
    eri: Option<PyReadonlyArray4<'py, f64>>,
    packed_eri: Option<PyReadonlyArray1<'py, f64>>,
    factors: Option<PyReadonlyArray3<'py, f64>>,
    coupling: Option<PyReadonlyArray2<'py, f64>>,
    tol: f64,
    coefficient_tol: f64,
    max_leaves: Option<usize>,
    constant: f64,
) -> PyResult<(
    usize,
    f64,
    Array<'py, Complex64>,
    Array<'py, f64>,
    Array<'py, f64>,
    Vec<(f64, Array<'py, f64>, Array<'py, f64>)>,
    f64,
)> {
    layout(&h)?;
    let h_shape = h.shape();
    if h_shape[0] != h_shape[1] {
        return Err(PyValueError::new_err(format!(
            "expected square one-body matrix, got {:?}",
            h_shape
        )));
    }
    let m = h_shape[0];
    let h_slice = h.as_slice()?;

    let mut dense_holder = None;
    if let Some(ref e) = eri {
        layout(e)?;
        let e_shape = e.shape();
        if e_shape != [m, m, m, m] {
            return Err(PyValueError::new_err(format!(
                "two-body tensor ({}, {}, {}, {}) does not match a {}-orbital one-body matrix",
                e_shape[0], e_shape[1], e_shape[2], e_shape[3], m
            )));
        }
        dense_holder = Some((e.as_slice()?, m));
    }

    let mut packed_holder = None;
    if let Some(ref p) = packed_eri {
        layout(p)?;
        packed_holder = Some((p.as_slice()?, m));
    }

    let mut factors_holder = None;
    if let Some(ref f) = factors {
        layout(f)?;
        let f_shape = f.shape();
        if f_shape.len() != 3 || f_shape[1] != m || f_shape[2] != m {
            return Err(PyValueError::new_err(format!(
                "pair factors ({}, {}, {}) do not match a {}-orbital one-body matrix",
                f_shape[0], f_shape[1], f_shape[2], m
            )));
        }
        factors_holder = Some((f.as_slice()?, f_shape[0], m));
    }

    let mut coupling_holder = None;
    if let Some(ref c) = coupling {
        layout(c)?;
        let c_shape = c.shape();
        coupling_holder = Some((c.as_slice()?, c_shape[0]));
    }

    let res = native::spatial_double_factorization(
        h_slice,
        m,
        dense_holder,
        packed_holder,
        factors_holder,
        coupling_holder,
        tol,
        coefficient_tol,
        max_leaves,
        constant,
    )
    .map_err(PyValueError::new_err)?;

    let n = res.n_modes;
    let one_body_data = dmatrix_complex_to_row_major(&res.one_body);
    let one_body_arr = array(py, &[n, n], one_body_data)?;

    let ob_rot_data = dmatrix_to_row_major(&res.one_body_orbitals);
    let ob_rot_arr = array(py, &[n, n], ob_rot_data)?;

    let ob_energies = res.one_body_energies.as_slice().to_vec();
    let ob_energies_arr = array(py, &[n], ob_energies)?;

    let mut py_leaves = Vec::with_capacity(res.leaves.len());
    for leaf in res.leaves {
        let orb_data = dmatrix_to_row_major(&leaf.orbitals);
        let orb_arr = array(py, &[n, n], orb_data)?;
        let coeff_data = leaf.coefficients.as_slice().to_vec();
        let coeff_arr = array(py, &[n], coeff_data)?;
        py_leaves.push((leaf.weight, orb_arr, coeff_arr));
    }

    Ok((
        res.n_modes,
        res.constant,
        one_body_arr,
        ob_rot_arr,
        ob_energies_arr,
        py_leaves,
        res.truncation_error,
    ))
}

#[pyfunction]
#[pyo3(signature = (eri=None, packed_eri=None, packed_m=None, factors=None, coupling=None))]
pub fn pair_spectrum<'py>(
    py: Python<'py>,
    eri: Option<PyReadonlyArray4<'py, f64>>,
    packed_eri: Option<PyReadonlyArray1<'py, f64>>,
    packed_m: Option<usize>,
    factors: Option<PyReadonlyArray3<'py, f64>>,
    coupling: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
    let mut dense_holder = None;
    if let Some(ref e) = eri {
        layout(e)?;
        let m = e.shape()[0];
        dense_holder = Some((e.as_slice()?, m));
    }

    let mut packed_holder = None;
    if let Some(ref p) = packed_eri {
        layout(p)?;
        let m = packed_m.ok_or_else(|| PyValueError::new_err("packed_m required for packed eri"))?;
        packed_holder = Some((p.as_slice()?, m));
    }

    let mut factors_holder = None;
    if let Some(ref f) = factors {
        layout(f)?;
        let f_shape = f.shape();
        factors_holder = Some((f.as_slice()?, f_shape[0], f_shape[1]));
    }

    let mut coupling_holder = None;
    if let Some(ref c) = coupling {
        layout(c)?;
        let c_shape = c.shape();
        coupling_holder = Some((c.as_slice()?, c_shape[0]));
    }

    let (weights, vectors) = native::pair_spectrum(
        dense_holder,
        packed_holder,
        factors_holder,
        coupling_holder,
    )
    .map_err(PyValueError::new_err)?;

    let w_len = weights.len();
    let weights_arr = array(py, &[w_len], weights.as_slice().to_vec())?;

    let v_data = dmatrix_to_row_major(&vectors);
    let vectors_arr = array(py, &[vectors.nrows(), vectors.ncols()], v_data)?;

    Ok((weights_arr, vectors_arr))
}

#[pyfunction]
#[pyo3(signature = (constant, one_body, leaves_weights, leaves_orbitals, leaves_coefficients, one_rdm, two_rdm))]
pub fn factorized_energy_rdm<'py>(
    _py: Python<'py>,
    constant: f64,
    one_body: PyReadonlyArray2<'py, Complex64>,
    leaves_weights: Vec<f64>,
    leaves_orbitals: Vec<PyReadonlyArray2<'py, f64>>,
    leaves_coefficients: Vec<PyReadonlyArray1<'py, f64>>,
    one_rdm: PyReadonlyArray2<'py, f64>,
    two_rdm: PyReadonlyArray4<'py, f64>,
) -> PyResult<f64> {
    layout(&one_body)?;
    layout(&one_rdm)?;
    layout(&two_rdm)?;

    let n_modes = one_body.shape()[0];
    let ob_slice = one_body.as_slice()?;
    let one_body_native: Vec<NumComplex64> = ob_slice
        .iter()
        .map(|z| NumComplex64::new(z.re, z.im))
        .collect();

    let one_rdm_slice = one_rdm.as_slice()?;
    let two_rdm_slice = two_rdm.as_slice()?;

    let mut orbitals_vec = Vec::with_capacity(leaves_orbitals.len());
    for orb in &leaves_orbitals {
        layout(orb)?;
        orbitals_vec.push(orb.as_slice()?.to_vec());
    }

    let mut coeffs_vec = Vec::with_capacity(leaves_coefficients.len());
    for c in &leaves_coefficients {
        layout(c)?;
        coeffs_vec.push(c.as_slice()?.to_vec());
    }

    let energy = native::factorized_energy_rdm(
        constant,
        &one_body_native,
        &leaves_weights,
        &orbitals_vec,
        &coeffs_vec,
        n_modes,
        one_rdm_slice,
        two_rdm_slice,
    );

    Ok(energy)
}
