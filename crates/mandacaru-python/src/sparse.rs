//! Sparse matrix operations: CSR construction, matvec, and matrix exponential.
use crate::arrays::array;
use numpy::{Complex64, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::{
    exceptions::{PyIndexError, PyValueError},
    prelude::*,
    types::{PyComplex, PyDict, PySlice, PyTuple},
};

fn to_i64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<i64>>() {
        Ok(arr.as_slice()?.to_vec())
    } else if let Ok(arr) = obj.extract::<PyReadonlyArray1<i32>>() {
        Ok(arr.as_slice()?.iter().map(|&x| x as i64).collect())
    } else {
        Err(PyValueError::new_err("expected 1D int64 or int32 array"))
    }
}

/// Convert COO coordinates and data to CSR format: (data, indices, indptr).
/// If dimension fits in 31 bits, indices and indptr are int32, matching scipy.sparse.
#[pyfunction]
pub fn csr_from_coo<'py>(
    py: Python<'py>,
    num_rows: usize,
    num_cols: usize,
    rows: Bound<'py, PyAny>,
    cols: Bound<'py, PyAny>,
    data: PyReadonlyArray1<'_, Complex64>,
) -> PyResult<(
    Bound<'py, numpy::PyArrayDyn<Complex64>>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
)> {
    let row_vec = to_i64_vec(&rows)?;
    let col_vec = to_i64_vec(&cols)?;
    let data_slice = data.as_slice()?;
    let nnz = data_slice.len();
    if row_vec.len() != nnz || col_vec.len() != nnz {
        return Err(PyValueError::new_err(
            "rows, cols, and data must have the same length",
        ));
    }

    let mut indptr = vec![0i64; num_rows + 1];
    for &r in &row_vec {
        if r >= 0 && (r as usize) < num_rows {
            indptr[r as usize + 1] += 1;
        }
    }
    for i in 0..num_rows {
        indptr[i + 1] += indptr[i];
    }

    let total_valid = indptr[num_rows] as usize;
    let mut head = indptr[..num_rows].to_vec();
    let mut out_data = vec![Complex64::new(0.0, 0.0); total_valid];
    let mut out_indices = vec![0i64; total_valid];

    for i in 0..nnz {
        let r = row_vec[i];
        if r >= 0 && (r as usize) < num_rows {
            let r_u = r as usize;
            let slot = head[r_u] as usize;
            head[r_u] += 1;
            out_indices[slot] = col_vec[i];
            out_data[slot] = data_slice[i];
        }
    }

    // Sort column indices within each row for canonical CSR order
    for r in 0..num_rows {
        let start = indptr[r] as usize;
        let end = indptr[r + 1] as usize;
        if end > start + 1 {
            let mut row_entries: Vec<(i64, Complex64)> = (start..end)
                .map(|k| (out_indices[k], out_data[k]))
                .collect();
            row_entries.sort_by_key(|&(c, _)| c);
            for (k, (c, v)) in (start..end).zip(row_entries) {
                out_indices[k] = c;
                out_data[k] = v;
            }
        }
    }

    let use_int32 = num_rows < (i32::MAX as usize)
        && num_cols < (i32::MAX as usize)
        && total_valid < (i32::MAX as usize);

    let py_data = array(py, &[total_valid], out_data)?;
    if use_int32 {
        let idx_32: Vec<i32> = out_indices.into_iter().map(|c| c as i32).collect();
        let ptr_32: Vec<i32> = indptr.into_iter().map(|p| p as i32).collect();
        let py_idx = array(py, &[total_valid], idx_32)?;
        let py_ptr = array(py, &[num_rows + 1], ptr_32)?;
        Ok((py_data, py_idx.into_any(), py_ptr.into_any()))
    } else {
        let py_idx = array(py, &[total_valid], out_indices)?;
        let py_ptr = array(py, &[num_rows + 1], indptr)?;
        Ok((py_data, py_idx.into_any(), py_ptr.into_any()))
    }
}

/// Compute matrix-vector product y = A @ x for CSR matrix A and 1D or 2D vector x.
#[pyfunction]
pub fn csr_matvec<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'_, Complex64>,
    indices: Bound<'py, PyAny>,
    indptr: Bound<'py, PyAny>,
    vector: PyReadonlyArrayDyn<'_, Complex64>,
    num_rows: usize,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<Complex64>>> {
    let data_slice = data.as_slice()?;
    let idx_vec = to_i64_vec(&indices)?;
    let ptr_vec = to_i64_vec(&indptr)?;
    let vec_view = vector.as_array();
    let vec_shape = vec_view.shape();

    if ptr_vec.len() != num_rows + 1 {
        return Err(PyValueError::new_err(
            "indptr length must equal num_rows + 1",
        ));
    }

    if vec_shape.len() == 1 {
        let in_dim = vec_shape[0];
        let mut out = vec![Complex64::new(0.0, 0.0); num_rows];
        for r in 0..num_rows {
            let start = ptr_vec[r] as usize;
            let end = ptr_vec[r + 1] as usize;
            let mut sum = Complex64::new(0.0, 0.0);
            for k in start..end {
                let c = idx_vec[k] as usize;
                if c < in_dim {
                    sum += data_slice[k] * vec_view[c];
                }
            }
            out[r] = sum;
        }
        Ok(array(py, &[num_rows], out)?)
    } else if vec_shape.len() == 2 {
        let in_dim = vec_shape[0];
        let k_cols = vec_shape[1];
        let mut out = vec![Complex64::new(0.0, 0.0); num_rows * k_cols];
        for r in 0..num_rows {
            let start = ptr_vec[r] as usize;
            let end = ptr_vec[r + 1] as usize;
            for col_idx in 0..k_cols {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in start..end {
                    let c = idx_vec[k] as usize;
                    if c < in_dim {
                        sum += data_slice[k] * vec_view[[c, col_idx]];
                    }
                }
                out[r * k_cols + col_idx] = sum;
            }
        }
        Ok(array(py, &[num_rows, k_cols], out)?)
    } else {
        Err(PyValueError::new_err("vector must be 1D or 2D"))
    }
}

/// Compute exp(angle * A) @ vector for sparse CSR matrix A using Taylor scaling and squaring.
#[pyfunction]
pub fn sparse_expm_multiply<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'_, Complex64>,
    indices: Bound<'py, PyAny>,
    indptr: Bound<'py, PyAny>,
    angle: f64,
    vector: PyReadonlyArrayDyn<'_, Complex64>,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<Complex64>>> {
    let data_slice = data.as_slice()?;
    let idx_vec = to_i64_vec(&indices)?;
    let ptr_vec = to_i64_vec(&indptr)?;
    let vec_view = vector.as_array();
    let n = vec_view.len();

    if ptr_vec.len() != n + 1 {
        return Err(PyValueError::new_err(
            "matrix dimension must match vector length",
        ));
    }

    if angle == 0.0 || data_slice.is_empty() {
        return Ok(array(py, &[n], vec_view.iter().copied().collect())?);
    }

    // 1-norm upper bound: max column sum
    let mut col_sums = vec![0.0f64; n];
    for (&idx, &val) in idx_vec.iter().zip(data_slice.iter()) {
        let c = idx as usize;
        if c < n {
            col_sums[c] += (val.re * val.re + val.im * val.im).sqrt();
        }
    }
    let norm_a = col_sums.into_iter().fold(0.0f64, f64::max);
    let bound = angle.abs() * norm_a;
    if bound == 0.0 {
        return Ok(array(py, &[n], vec_view.iter().copied().collect())?);
    }

    let steps = (bound / 0.5).ceil().max(1.0) as usize;
    let scale = angle / (steps as f64);
    let mut curr: Vec<Complex64> = vec_view.iter().copied().collect();
    let mut next = vec![Complex64::new(0.0, 0.0); n];
    let mut term = vec![Complex64::new(0.0, 0.0); n];
    let mut accum = vec![Complex64::new(0.0, 0.0); n];

    for _ in 0..steps {
        accum.copy_from_slice(&curr);
        term.copy_from_slice(&curr);
        for order in 1..=80 {
            let factor = scale / (order as f64);
            let mut term_norm_sq = 0.0;
            for r in 0..n {
                let start = ptr_vec[r] as usize;
                let end = ptr_vec[r + 1] as usize;
                let mut row_sum = Complex64::new(0.0, 0.0);
                for k in start..end {
                    let c = idx_vec[k] as usize;
                    row_sum += data_slice[k] * term[c];
                }
                let val = row_sum * factor;
                next[r] = val;
                term_norm_sq += val.re * val.re + val.im * val.im;
            }
            term.copy_from_slice(&next);
            let mut accum_norm_sq = 0.0;
            for r in 0..n {
                accum[r] += term[r];
                accum_norm_sq += accum[r].re * accum[r].re + accum[r].im * accum[r].im;
            }
            if term_norm_sq < 1e-30 * accum_norm_sq {
                break;
            }
        }
        curr.copy_from_slice(&accum);
    }

    Ok(array(py, &[n], curr)?)
}

#[derive(Clone)]
#[pyclass(name = "CSRMatrix")]
pub struct PyCSRMatrix {
    pub shape: (usize, usize),
    pub rust_indices: Vec<usize>,
    pub rust_indptr: Vec<usize>,
    pub is_complex: bool,
    pub real_data: Vec<f64>,
    pub complex_data: Vec<Complex64>,
    pub has_sorted_indices: bool,
    pub use_int32: bool,
}

impl PyCSRMatrix {
    fn from_coo_internal(
        num_rows: usize,
        num_cols: usize,
        rows: &[i64],
        cols: &[i64],
        real_vals: Option<&[f64]>,
        cplx_vals: Option<&[Complex64]>,
    ) -> PyResult<Self> {
        let nnz = rows.len();
        let mut indptr = vec![0usize; num_rows + 1];
        for &r in rows {
            if r >= 0 && (r as usize) < num_rows {
                indptr[r as usize + 1] += 1;
            }
        }
        for i in 0..num_rows {
            indptr[i + 1] += indptr[i];
        }

        let total_valid = indptr[num_rows];
        let mut head = indptr[..num_rows].to_vec();
        let mut out_indices = vec![0usize; total_valid];
        let is_cplx = cplx_vals.is_some();
        let mut out_real = if is_cplx {
            Vec::new()
        } else {
            vec![0.0f64; total_valid]
        };
        let mut out_cplx = if is_cplx {
            vec![Complex64::new(0.0, 0.0); total_valid]
        } else {
            Vec::new()
        };

        for i in 0..nnz {
            let r = rows[i];
            if r >= 0 && (r as usize) < num_rows {
                let r_u = r as usize;
                let slot = head[r_u];
                head[r_u] += 1;
                out_indices[slot] = cols[i] as usize;
                if is_cplx {
                    out_cplx[slot] = cplx_vals.unwrap()[i];
                } else {
                    out_real[slot] = real_vals.unwrap()[i];
                }
            }
        }

        let mut final_indices = Vec::with_capacity(total_valid);
        let mut final_real = if is_cplx {
            Vec::new()
        } else {
            Vec::with_capacity(total_valid)
        };
        let mut final_cplx = if is_cplx {
            Vec::with_capacity(total_valid)
        } else {
            Vec::new()
        };
        let mut final_indptr = vec![0usize; num_rows + 1];

        for r in 0..num_rows {
            let start = indptr[r];
            let end = indptr[r + 1];
            if end > start {
                if is_cplx {
                    let mut row_entries: Vec<(usize, Complex64)> = (start..end)
                        .map(|k| (out_indices[k], out_cplx[k]))
                        .collect();
                    row_entries.sort_by_key(|&(c, _)| c);
                    for (c, v) in row_entries {
                        if let Some(last_c) = final_indices.last_mut() {
                            if *last_c == c && final_cplx.len() > final_indptr[r] {
                                *final_cplx.last_mut().unwrap() += v;
                                continue;
                            }
                        }
                        final_indices.push(c);
                        final_cplx.push(v);
                    }
                } else {
                    let mut row_entries: Vec<(usize, f64)> = (start..end)
                        .map(|k| (out_indices[k], out_real[k]))
                        .collect();
                    row_entries.sort_by_key(|&(c, _)| c);
                    for (c, v) in row_entries {
                        if let Some(last_c) = final_indices.last_mut() {
                            if *last_c == c && final_real.len() > final_indptr[r] {
                                *final_real.last_mut().unwrap() += v;
                                continue;
                            }
                        }
                        final_indices.push(c);
                        final_real.push(v);
                    }
                }
            }
            final_indptr[r + 1] = final_indices.len();
        }

        let nnz_final = final_indices.len();
        let use_int32 = num_rows < (i32::MAX as usize)
            && num_cols < (i32::MAX as usize)
            && nnz_final < (i32::MAX as usize);

        Ok(Self {
            shape: (num_rows, num_cols),
            rust_indices: final_indices,
            rust_indptr: final_indptr,
            is_complex: is_cplx,
            real_data: final_real,
            complex_data: final_cplx,
            has_sorted_indices: true,
            use_int32,
        })
    }
}

#[pymethods]
impl PyCSRMatrix {
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn new(
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let _ = py;
        let n_args = args.len();
        if n_args == 5 {
            let num_rows = args.get_item(0)?.extract::<usize>()?;
            let num_cols = args.get_item(1)?.extract::<usize>()?;
            let rows = to_i64_vec(&args.get_item(2)?)?;
            let cols = to_i64_vec(&args.get_item(3)?)?;
            let data_obj = args.get_item(4)?;
            if let Ok(cplx_arr) = data_obj.extract::<PyReadonlyArray1<'_, Complex64>>() {
                return Self::from_coo_internal(
                    num_rows,
                    num_cols,
                    &rows,
                    &cols,
                    None,
                    Some(cplx_arr.as_slice()?),
                );
            } else if let Ok(real_arr) = data_obj.extract::<PyReadonlyArray1<'_, f64>>() {
                return Self::from_coo_internal(
                    num_rows,
                    num_cols,
                    &rows,
                    &cols,
                    Some(real_arr.as_slice()?),
                    None,
                );
            } else {
                return Err(PyValueError::new_err(
                    "expected float64 or complex128 data array",
                ));
            }
        } else if n_args == 1 || n_args == 2 {
            let first = args.get_item(0)?;
            if let Ok(shape_tuple) = first.extract::<(usize, usize)>() {
                let (r, c) = shape_tuple;
                let use_int32 = r < (i32::MAX as usize) && c < (i32::MAX as usize);
                return Ok(Self {
                    shape: (r, c),
                    rust_indices: Vec::new(),
                    rust_indptr: vec![0; r + 1],
                    is_complex: true,
                    real_data: Vec::new(),
                    complex_data: Vec::new(),
                    has_sorted_indices: true,
                    use_int32,
                });
            }
            if let Ok(trip) = first.cast::<PyTuple>() {
                if trip.len() == 3 {
                    let data_obj = trip.get_item(0)?;
                    let idx_obj = trip.get_item(1)?;
                    let ptr_obj = trip.get_item(2)?;
                    let indices_vec: Vec<usize> = to_i64_vec(&idx_obj)?
                        .into_iter()
                        .map(|x| x as usize)
                        .collect();
                    let indptr_vec: Vec<usize> = to_i64_vec(&ptr_obj)?
                        .into_iter()
                        .map(|x| x as usize)
                        .collect();
                    let num_rows = if indptr_vec.is_empty() {
                        0
                    } else {
                        indptr_vec.len() - 1
                    };
                    let shape = if let Some(kw) = kwargs {
                        if let Some(s) = kw.get_item("shape")? {
                            s.extract::<(usize, usize)>()?
                        } else {
                            (
                                num_rows,
                                indices_vec.iter().max().map(|&m| m + 1).unwrap_or(0),
                            )
                        }
                    } else if n_args == 2 {
                        args.get_item(1)?.extract::<(usize, usize)>()?
                    } else {
                        (
                            num_rows,
                            indices_vec.iter().max().map(|&m| m + 1).unwrap_or(0),
                        )
                    };
                    let use_int32 = shape.0 < (i32::MAX as usize)
                        && shape.1 < (i32::MAX as usize)
                        && indices_vec.len() < (i32::MAX as usize);
                    if let Ok(cplx_arr) = data_obj.extract::<PyReadonlyArray1<'_, Complex64>>() {
                        return Ok(Self {
                            shape,
                            rust_indices: indices_vec,
                            rust_indptr: indptr_vec,
                            is_complex: true,
                            real_data: Vec::new(),
                            complex_data: cplx_arr.as_slice()?.to_vec(),
                            has_sorted_indices: false,
                            use_int32,
                        });
                    } else if let Ok(real_arr) = data_obj.extract::<PyReadonlyArray1<'_, f64>>() {
                        return Ok(Self {
                            shape,
                            rust_indices: indices_vec,
                            rust_indptr: indptr_vec,
                            is_complex: false,
                            real_data: real_arr.as_slice()?.to_vec(),
                            complex_data: Vec::new(),
                            has_sorted_indices: false,
                            use_int32,
                        });
                    } else {
                        return Err(PyValueError::new_err(
                            "data array must be float64 or complex128",
                        ));
                    }
                }
            }
        }
        Err(PyValueError::new_err(
            "unsupported arguments for CSRMatrix constructor",
        ))
    }

    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    #[getter]
    pub fn nnz(&self) -> usize {
        self.rust_indices.len()
    }

    #[getter]
    pub fn format(&self) -> &'static str {
        "csr"
    }

    #[getter]
    pub fn has_sorted_indices(&self) -> bool {
        self.has_sorted_indices
    }

    #[setter]
    pub fn set_has_sorted_indices(&mut self, val: bool) {
        self.has_sorted_indices = val;
    }

    #[getter]
    pub fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.is_complex {
            array(py, &[self.complex_data.len()], self.complex_data.clone())
                .map(|a| a.into_any())
        } else {
            array(py, &[self.real_data.len()], self.real_data.clone()).map(|a| a.into_any())
        }
    }

    #[getter]
    pub fn indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.use_int32 {
            let idx: Vec<i32> = self.rust_indices.iter().map(|&x| x as i32).collect();
            array(py, &[idx.len()], idx).map(|a| a.into_any())
        } else {
            let idx: Vec<i64> = self.rust_indices.iter().map(|&x| x as i64).collect();
            array(py, &[idx.len()], idx).map(|a| a.into_any())
        }
    }

    #[getter]
    pub fn indptr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.use_int32 {
            let ptr: Vec<i32> = self.rust_indptr.iter().map(|&x| x as i32).collect();
            array(py, &[ptr.len()], ptr).map(|a| a.into_any())
        } else {
            let ptr: Vec<i64> = self.rust_indptr.iter().map(|&x| x as i64).collect();
            array(py, &[ptr.len()], ptr).map(|a| a.into_any())
        }
    }

    #[getter]
    pub fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        if self.is_complex {
            np.getattr("complex128")
        } else {
            np.getattr("float64")
        }
    }

    #[getter]
    #[allow(non_snake_case)]
    pub fn T(&self) -> PyResult<Self> {
        self.transpose()
    }

    pub fn sort_indices(&mut self) -> PyResult<()> {
        if self.has_sorted_indices {
            return Ok(());
        }
        for r in 0..self.shape.0 {
            let start = self.rust_indptr[r];
            let end = self.rust_indptr[r + 1];
            if end > start + 1 {
                if self.is_complex {
                    let mut pairs: Vec<(usize, Complex64)> = (start..end)
                        .map(|k| (self.rust_indices[k], self.complex_data[k]))
                        .collect();
                    pairs.sort_by_key(|&(c, _)| c);
                    for (k, (c, v)) in (start..end).zip(pairs) {
                        self.rust_indices[k] = c;
                        self.complex_data[k] = v;
                    }
                } else {
                    let mut pairs: Vec<(usize, f64)> = (start..end)
                        .map(|k| (self.rust_indices[k], self.real_data[k]))
                        .collect();
                    pairs.sort_by_key(|&(c, _)| c);
                    for (k, (c, v)) in (start..end).zip(pairs) {
                        self.rust_indices[k] = c;
                        self.real_data[k] = v;
                    }
                }
            }
        }
        self.has_sorted_indices = true;
        Ok(())
    }

    pub fn eliminate_zeros(&mut self) -> PyResult<()> {
        let mut new_indices = Vec::new();
        let mut new_indptr = vec![0usize; self.shape.0 + 1];
        if self.is_complex {
            let mut new_cplx = Vec::new();
            for r in 0..self.shape.0 {
                let start = self.rust_indptr[r];
                let end = self.rust_indptr[r + 1];
                for k in start..end {
                    let val = self.complex_data[k];
                    if val.re != 0.0 || val.im != 0.0 {
                        new_indices.push(self.rust_indices[k]);
                        new_cplx.push(val);
                    }
                }
                new_indptr[r + 1] = new_indices.len();
            }
            self.complex_data = new_cplx;
        } else {
            let mut new_real = Vec::new();
            for r in 0..self.shape.0 {
                let start = self.rust_indptr[r];
                let end = self.rust_indptr[r + 1];
                for k in start..end {
                    let val = self.real_data[k];
                    if val != 0.0 {
                        new_indices.push(self.rust_indices[k]);
                        new_real.push(val);
                    }
                }
                new_indptr[r + 1] = new_indices.len();
            }
            self.real_data = new_real;
        }
        self.rust_indices = new_indices;
        self.rust_indptr = new_indptr;
        Ok(())
    }

    pub fn tocsr(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn tocoo(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }

    pub fn conj(&self) -> Self {
        if self.is_complex {
            let conj_data: Vec<Complex64> =
                self.complex_data.iter().map(|c| c.conj()).collect();
            Self {
                complex_data: conj_data,
                ..self.clone()
            }
        } else {
            self.clone()
        }
    }

    pub fn conjugate(&self) -> Self {
        self.conj()
    }

    pub fn transpose(&self) -> PyResult<Self> {
        let (num_rows, num_cols) = self.shape;
        let mut counts = vec![0usize; num_cols];
        for &c in &self.rust_indices {
            if c < num_cols {
                counts[c] += 1;
            }
        }
        let mut new_indptr = vec![0usize; num_cols + 1];
        for c in 0..num_cols {
            new_indptr[c + 1] = new_indptr[c] + counts[c];
        }
        let mut head = new_indptr[..num_cols].to_vec();
        let total = self.rust_indices.len();
        let mut new_indices = vec![0usize; total];
        let mut new_real = if self.is_complex {
            Vec::new()
        } else {
            vec![0.0f64; total]
        };
        let mut new_cplx = if self.is_complex {
            vec![Complex64::new(0.0, 0.0); total]
        } else {
            Vec::new()
        };

        for r in 0..num_rows {
            let start = self.rust_indptr[r];
            let end = self.rust_indptr[r + 1];
            for k in start..end {
                let c = self.rust_indices[k];
                if c < num_cols {
                    let slot = head[c];
                    head[c] += 1;
                    new_indices[slot] = r;
                    if self.is_complex {
                        new_cplx[slot] = self.complex_data[k];
                    } else {
                        new_real[slot] = self.real_data[k];
                    }
                }
            }
        }

        let mut res = Self {
            shape: (num_cols, num_rows),
            rust_indices: new_indices,
            rust_indptr: new_indptr,
            is_complex: self.is_complex,
            real_data: new_real,
            complex_data: new_cplx,
            has_sorted_indices: false,
            use_int32: self.use_int32,
        };
        res.sort_indices()?;
        Ok(res)
    }

    pub fn toarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (num_rows, num_cols) = self.shape;
        if self.is_complex {
            let mut dense = vec![Complex64::new(0.0, 0.0); num_rows * num_cols];
            for r in 0..num_rows {
                let start = self.rust_indptr[r];
                let end = self.rust_indptr[r + 1];
                for k in start..end {
                    let c = self.rust_indices[k];
                    if c < num_cols {
                        dense[r * num_cols + c] = self.complex_data[k];
                    }
                }
            }
            array(py, &[num_rows, num_cols], dense).map(|a| a.into_any())
        } else {
            let mut dense = vec![0.0f64; num_rows * num_cols];
            for r in 0..num_rows {
                let start = self.rust_indptr[r];
                let end = self.rust_indptr[r + 1];
                for k in start..end {
                    let c = self.rust_indices[k];
                    if c < num_cols {
                        dense[r * num_cols + c] = self.real_data[k];
                    }
                }
            }
            array(py, &[num_rows, num_cols], dense).map(|a| a.into_any())
        }
    }

    pub fn power(&self, p: i32) -> PyResult<Self> {
        if self.is_complex {
            let new_cplx: Vec<Complex64> = self
                .complex_data
                .iter()
                .map(|&c| {
                    if p == 2 {
                        c * c
                    } else if p == 1 {
                        c
                    } else if p == 0 {
                        Complex64::new(1.0, 0.0)
                    } else {
                        let r = (c.re * c.re + c.im * c.im).sqrt();
                        let theta = c.im.atan2(c.re);
                        let r_p = r.powi(p);
                        let theta_p = theta * (p as f64);
                        Complex64::new(r_p * theta_p.cos(), r_p * theta_p.sin())
                    }
                })
                .collect();
            Ok(Self {
                complex_data: new_cplx,
                ..self.clone()
            })
        } else {
            let new_real: Vec<f64> = self.real_data.iter().map(|&x| x.powi(p)).collect();
            Ok(Self {
                real_data: new_real,
                ..self.clone()
            })
        }
    }

    #[pyo3(signature = (axis=None))]
    pub fn sum<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        let (num_rows, num_cols) = self.shape;
        match axis {
            None => {
                if self.is_complex {
                    let total: Complex64 = self.complex_data.iter().sum();
                    Ok(PyComplex::from_doubles(py, total.re, total.im).into_any())
                } else {
                    let total: f64 = self.real_data.iter().sum();
                    Ok(total.into_pyobject(py)?.into_any())
                }
            }
            Some(1) | Some(-1) => {
                if self.is_complex {
                    let mut row_sums = vec![Complex64::new(0.0, 0.0); num_rows];
                    for r in 0..num_rows {
                        let start = self.rust_indptr[r];
                        let end = self.rust_indptr[r + 1];
                        row_sums[r] = self.complex_data[start..end].iter().sum();
                    }
                    array(py, &[num_rows, 1], row_sums).map(|a| a.into_any())
                } else {
                    let mut row_sums = vec![0.0f64; num_rows];
                    for r in 0..num_rows {
                        let start = self.rust_indptr[r];
                        let end = self.rust_indptr[r + 1];
                        row_sums[r] = self.real_data[start..end].iter().sum();
                    }
                    array(py, &[num_rows, 1], row_sums).map(|a| a.into_any())
                }
            }
            Some(0) => {
                if self.is_complex {
                    let mut col_sums = vec![Complex64::new(0.0, 0.0); num_cols];
                    for r in 0..num_rows {
                        let start = self.rust_indptr[r];
                        let end = self.rust_indptr[r + 1];
                        for k in start..end {
                            let c = self.rust_indices[k];
                            if c < num_cols {
                                col_sums[c] += self.complex_data[k];
                            }
                        }
                    }
                    array(py, &[1, num_cols], col_sums).map(|a| a.into_any())
                } else {
                    let mut col_sums = vec![0.0f64; num_cols];
                    for r in 0..num_rows {
                        let start = self.rust_indptr[r];
                        let end = self.rust_indptr[r + 1];
                        for k in start..end {
                            let c = self.rust_indices[k];
                            if c < num_cols {
                                col_sums[c] += self.real_data[k];
                            }
                        }
                    }
                    array(py, &[1, num_cols], col_sums).map(|a| a.into_any())
                }
            }
            _ => Err(PyValueError::new_err("axis out of bounds")),
        }
    }

    pub fn multiply(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_mat) = other.extract::<PyRef<'_, PyCSRMatrix>>() {
            if self.shape != other_mat.shape
                || self.rust_indices.len() != other_mat.rust_indices.len()
            {
                return Err(PyValueError::new_err(
                    "matrix shapes and sparsity must match for elementwise multiply",
                ));
            }
            if self.is_complex || other_mat.is_complex {
                let mult_data: Vec<Complex64> = (0..self.rust_indices.len())
                    .map(|i| {
                        let a = if self.is_complex {
                            self.complex_data[i]
                        } else {
                            Complex64::new(self.real_data[i], 0.0)
                        };
                        let b = if other_mat.is_complex {
                            other_mat.complex_data[i]
                        } else {
                            Complex64::new(other_mat.real_data[i], 0.0)
                        };
                        a * b
                    })
                    .collect();
                Ok(Self {
                    shape: self.shape,
                    rust_indices: self.rust_indices.clone(),
                    rust_indptr: self.rust_indptr.clone(),
                    is_complex: true,
                    real_data: Vec::new(),
                    complex_data: mult_data,
                    has_sorted_indices: self.has_sorted_indices,
                    use_int32: self.use_int32,
                })
            } else {
                let mult_data: Vec<f64> = self
                    .real_data
                    .iter()
                    .zip(&other_mat.real_data)
                    .map(|(&a, &b)| a * b)
                    .collect();
                Ok(Self {
                    shape: self.shape,
                    rust_indices: self.rust_indices.clone(),
                    rust_indptr: self.rust_indptr.clone(),
                    is_complex: false,
                    real_data: mult_data,
                    complex_data: Vec::new(),
                    has_sorted_indices: self.has_sorted_indices,
                    use_int32: self.use_int32,
                })
            }
        } else if let Ok(py_c) = other.cast::<PyComplex>() {
            self.__mul__(py_c.as_any())
        } else if let Ok(scalar_f) = other.extract::<f64>() {
            let py = other.py();
            self.__mul__(&scalar_f.into_pyobject(py)?.into_any())
        } else {
            Err(PyValueError::new_err("unsupported operand for multiply"))
        }
    }

    pub fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(py_c) = other.cast::<PyComplex>() {
            let c = Complex64::new(py_c.real(), py_c.imag());
            let new_cplx: Vec<Complex64> = (0..self.rust_indices.len())
                .map(|i| {
                    let val = if self.is_complex {
                        self.complex_data[i]
                    } else {
                        Complex64::new(self.real_data[i], 0.0)
                    };
                    val * c
                })
                .collect();
            Ok(Self {
                complex_data: new_cplx,
                real_data: Vec::new(),
                is_complex: true,
                ..self.clone()
            })
        } else if let Ok(x) = other.extract::<f64>() {
            if self.is_complex {
                let new_cplx: Vec<Complex64> =
                    self.complex_data.iter().map(|&c| c * x).collect();
                Ok(Self {
                    complex_data: new_cplx,
                    ..self.clone()
                })
            } else {
                let new_real: Vec<f64> = self.real_data.iter().map(|&r| r * x).collect();
                Ok(Self {
                    real_data: new_real,
                    ..self.clone()
                })
            }
        } else {
            Err(PyValueError::new_err(
                "scalar multiplication requires float or complex",
            ))
        }
    }

    pub fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    pub fn __neg__(&self) -> PyResult<Self> {
        if self.is_complex {
            let new_cplx: Vec<Complex64> = self.complex_data.iter().map(|&c| -c).collect();
            Ok(Self {
                complex_data: new_cplx,
                ..self.clone()
            })
        } else {
            let new_real: Vec<f64> = self.real_data.iter().map(|&x| -x).collect();
            Ok(Self {
                real_data: new_real,
                ..self.clone()
            })
        }
    }

    pub fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other_mat = other.extract::<PyRef<'_, PyCSRMatrix>>()?;
        self.add_sub_internal(&other_mat, 1.0)
    }

    pub fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other_mat = other.extract::<PyRef<'_, PyCSRMatrix>>()?;
        self.add_sub_internal(&other_mat, -1.0)
    }

    pub fn __matmul__<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (num_rows, num_cols) = self.shape;
        if let Ok(other_mat) = other.extract::<PyRef<'_, PyCSRMatrix>>() {
            if num_cols != other_mat.shape.0 {
                return Err(PyValueError::new_err(format!(
                    "dimension mismatch for matmul: ({}, {}) x ({}, {})",
                    num_rows, num_cols, other_mat.shape.0, other_mat.shape.1
                )));
            }
            let out_cols = other_mat.shape.1;
            let is_cplx = self.is_complex || other_mat.is_complex;
            let mut out_indices = Vec::new();
            let mut out_indptr = vec![0usize; num_rows + 1];
            let mut out_real = if is_cplx { Vec::new() } else { Vec::new() };
            let mut out_cplx = if is_cplx { Vec::new() } else { Vec::new() };

            for r in 0..num_rows {
                let start_a = self.rust_indptr[r];
                let end_a = self.rust_indptr[r + 1];
                if is_cplx {
                    let mut row_acc: Vec<(usize, Complex64)> = Vec::new();
                    for ka in start_a..end_a {
                        let k = self.rust_indices[ka];
                        let va = if self.is_complex {
                            self.complex_data[ka]
                        } else {
                            Complex64::new(self.real_data[ka], 0.0)
                        };
                        let start_b = other_mat.rust_indptr[k];
                        let end_b = other_mat.rust_indptr[k + 1];
                        for kb in start_b..end_b {
                            let j = other_mat.rust_indices[kb];
                            let vb = if other_mat.is_complex {
                                other_mat.complex_data[kb]
                            } else {
                                Complex64::new(other_mat.real_data[kb], 0.0)
                            };
                            row_acc.push((j, va * vb));
                        }
                    }
                    if !row_acc.is_empty() {
                        row_acc.sort_by_key(|&(c, _)| c);
                        let mut last_c = row_acc[0].0;
                        let mut last_v = row_acc[0].1;
                        for &(c, v) in &row_acc[1..] {
                            if c == last_c {
                                last_v += v;
                            } else {
                                if last_v.re.abs() > 1e-15 || last_v.im.abs() > 1e-15 {
                                    out_indices.push(last_c);
                                    out_cplx.push(last_v);
                                }
                                last_c = c;
                                last_v = v;
                            }
                        }
                        if last_v.re.abs() > 1e-15 || last_v.im.abs() > 1e-15 {
                            out_indices.push(last_c);
                            out_cplx.push(last_v);
                        }
                    }
                } else {
                    let mut row_acc: Vec<(usize, f64)> = Vec::new();
                    for ka in start_a..end_a {
                        let k = self.rust_indices[ka];
                        let va = self.real_data[ka];
                        let start_b = other_mat.rust_indptr[k];
                        let end_b = other_mat.rust_indptr[k + 1];
                        for kb in start_b..end_b {
                            let j = other_mat.rust_indices[kb];
                            let vb = other_mat.real_data[kb];
                            row_acc.push((j, va * vb));
                        }
                    }
                    if !row_acc.is_empty() {
                        row_acc.sort_by_key(|&(c, _)| c);
                        let mut last_c = row_acc[0].0;
                        let mut last_v = row_acc[0].1;
                        for &(c, v) in &row_acc[1..] {
                            if c == last_c {
                                last_v += v;
                            } else {
                                if last_v.abs() > 1e-15 {
                                    out_indices.push(last_c);
                                    out_real.push(last_v);
                                }
                                last_c = c;
                                last_v = v;
                            }
                        }
                        if last_v.abs() > 1e-15 {
                            out_indices.push(last_c);
                            out_real.push(last_v);
                        }
                    }
                }
                out_indptr[r + 1] = out_indices.len();
            }

            let use_int32 = num_rows < (i32::MAX as usize)
                && out_cols < (i32::MAX as usize)
                && out_indices.len() < (i32::MAX as usize);

            let res = Self {
                shape: (num_rows, out_cols),
                rust_indices: out_indices,
                rust_indptr: out_indptr,
                is_complex: is_cplx,
                real_data: out_real,
                complex_data: out_cplx,
                has_sorted_indices: true,
                use_int32,
            };
            return Ok(Bound::new(py, res)?.into_any());
        }

        if let Ok(cplx_arr) = other.extract::<PyReadonlyArray1<'_, Complex64>>() {
            let v = cplx_arr.as_array();
            let mut out = vec![Complex64::new(0.0, 0.0); num_rows];
            for r in 0..num_rows {
                let start = self.rust_indptr[r];
                let end = self.rust_indptr[r + 1];
                let mut sum = Complex64::new(0.0, 0.0);
                for k in start..end {
                    let c = self.rust_indices[k];
                    if c < v.len() {
                        let va = if self.is_complex {
                            self.complex_data[k]
                        } else {
                            Complex64::new(self.real_data[k], 0.0)
                        };
                        sum += va * v[c];
                    }
                }
                out[r] = sum;
            }
            return array(py, &[num_rows], out).map(|a| a.into_any());
        }

        if let Ok(real_arr) = other.extract::<PyReadonlyArray1<'_, f64>>() {
            let v = real_arr.as_array();
            if self.is_complex {
                let mut out = vec![Complex64::new(0.0, 0.0); num_rows];
                for r in 0..num_rows {
                    let start = self.rust_indptr[r];
                    let end = self.rust_indptr[r + 1];
                    let mut sum = Complex64::new(0.0, 0.0);
                    for k in start..end {
                        let c = self.rust_indices[k];
                        if c < v.len() {
                            sum += self.complex_data[k] * v[c];
                        }
                    }
                    out[r] = sum;
                }
                return array(py, &[num_rows], out).map(|a| a.into_any());
            } else {
                let mut out = vec![0.0f64; num_rows];
                for r in 0..num_rows {
                    let start = self.rust_indptr[r];
                    let end = self.rust_indptr[r + 1];
                    let mut sum = 0.0;
                    for k in start..end {
                        let c = self.rust_indices[k];
                        if c < v.len() {
                            sum += self.real_data[k] * v[c];
                        }
                    }
                    out[r] = sum;
                }
                return array(py, &[num_rows], out).map(|a| a.into_any());
            }
        }

        if let Ok(cplx_2d) = other.extract::<PyReadonlyArray2<'_, Complex64>>() {
            let arr2 = cplx_2d.as_array();
            let (other_r, other_c) = (arr2.shape()[0], arr2.shape()[1]);
            let mut out = vec![Complex64::new(0.0, 0.0); num_rows * other_c];
            for r in 0..num_rows {
                let start = self.rust_indptr[r];
                let end = self.rust_indptr[r + 1];
                for col in 0..other_c {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for k in start..end {
                        let c = self.rust_indices[k];
                        if c < other_r {
                            let va = if self.is_complex {
                                self.complex_data[k]
                            } else {
                                Complex64::new(self.real_data[k], 0.0)
                            };
                            sum += va * arr2[[c, col]];
                        }
                    }
                    out[r * other_c + col] = sum;
                }
            }
            return array(py, &[num_rows, other_c], out).map(|a| a.into_any());
        }

        if let Ok(real_2d) = other.extract::<PyReadonlyArray2<'_, f64>>() {
            let arr2 = real_2d.as_array();
            let (other_r, other_c) = (arr2.shape()[0], arr2.shape()[1]);
            if self.is_complex {
                let mut out = vec![Complex64::new(0.0, 0.0); num_rows * other_c];
                for r in 0..num_rows {
                    let start = self.rust_indptr[r];
                    let end = self.rust_indptr[r + 1];
                    for col in 0..other_c {
                        let mut sum = Complex64::new(0.0, 0.0);
                        for k in start..end {
                            let c = self.rust_indices[k];
                            if c < other_r {
                                sum += self.complex_data[k] * arr2[[c, col]];
                            }
                        }
                        out[r * other_c + col] = sum;
                    }
                }
                return array(py, &[num_rows, other_c], out).map(|a| a.into_any());
            } else {
                let mut out = vec![0.0f64; num_rows * other_c];
                for r in 0..num_rows {
                    let start = self.rust_indptr[r];
                    let end = self.rust_indptr[r + 1];
                    for col in 0..other_c {
                        let mut sum = 0.0;
                        for k in start..end {
                            let c = self.rust_indices[k];
                            if c < other_r {
                                sum += self.real_data[k] * arr2[[c, col]];
                            }
                        }
                        out[r * other_c + col] = sum;
                    }
                }
                return array(py, &[num_rows, other_c], out).map(|a| a.into_any());
            }
        }

        Err(PyValueError::new_err(
            "unsupported operand for matmul with CSRMatrix",
        ))
    }

    pub fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        item: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (num_rows, num_cols) = self.shape;
        if let Ok(tup) = item.cast::<PyTuple>() {
            if tup.len() == 2 {
                let item0 = tup.get_item(0)?;
                let item1 = tup.get_item(1)?;
                let is_slice0 = item0.is_instance_of::<PySlice>();
                let is_slice1 = item1.is_instance_of::<PySlice>();

                if is_slice0 && !is_slice1 {
                    let cols: Vec<usize> = to_i64_vec(&item1)?
                        .into_iter()
                        .map(|x| x as usize)
                        .collect();
                    let mut col_map = vec![-1i64; num_cols];
                    for (new_c, &old_c) in cols.iter().enumerate() {
                        if old_c < num_cols {
                            col_map[old_c] = new_c as i64;
                        }
                    }
                    let mut out_indices = Vec::new();
                    let mut out_indptr = vec![0usize; num_rows + 1];
                    let mut out_real = if self.is_complex {
                        Vec::new()
                    } else {
                        Vec::new()
                    };
                    let mut out_cplx = if self.is_complex {
                        Vec::new()
                    } else {
                        Vec::new()
                    };

                    for r in 0..num_rows {
                        let start = self.rust_indptr[r];
                        let end = self.rust_indptr[r + 1];
                        for k in start..end {
                            let c = self.rust_indices[k];
                            if c < num_cols && col_map[c] >= 0 {
                                out_indices.push(col_map[c] as usize);
                                if self.is_complex {
                                    out_cplx.push(self.complex_data[k]);
                                } else {
                                    out_real.push(self.real_data[k]);
                                }
                            }
                        }
                        out_indptr[r + 1] = out_indices.len();
                    }
                    let res = Self {
                        shape: (num_rows, cols.len()),
                        rust_indices: out_indices,
                        rust_indptr: out_indptr,
                        is_complex: self.is_complex,
                        real_data: out_real,
                        complex_data: out_cplx,
                        has_sorted_indices: false,
                        use_int32: self.use_int32,
                    };
                    return Ok(Bound::new(py, res)?.into_any());
                } else if !is_slice0 && is_slice1 {
                    let rows: Vec<usize> = to_i64_vec(&item0)?
                        .into_iter()
                        .map(|x| x as usize)
                        .collect();
                    let mut out_indices = Vec::new();
                    let mut out_indptr = vec![0usize; rows.len() + 1];
                    let mut out_real = if self.is_complex {
                        Vec::new()
                    } else {
                        Vec::new()
                    };
                    let mut out_cplx = if self.is_complex {
                        Vec::new()
                    } else {
                        Vec::new()
                    };

                    for (new_r, &r) in rows.iter().enumerate() {
                        if r < num_rows {
                            let start = self.rust_indptr[r];
                            let end = self.rust_indptr[r + 1];
                            for k in start..end {
                                out_indices.push(self.rust_indices[k]);
                                if self.is_complex {
                                    out_cplx.push(self.complex_data[k]);
                                } else {
                                    out_real.push(self.real_data[k]);
                                }
                            }
                        }
                        out_indptr[new_r + 1] = out_indices.len();
                    }
                    let res = Self {
                        shape: (rows.len(), num_cols),
                        rust_indices: out_indices,
                        rust_indptr: out_indptr,
                        is_complex: self.is_complex,
                        real_data: out_real,
                        complex_data: out_cplx,
                        has_sorted_indices: self.has_sorted_indices,
                        use_int32: self.use_int32,
                    };
                    return Ok(Bound::new(py, res)?.into_any());
                } else if !is_slice0 && !is_slice1 {
                    let rows: Vec<usize> = to_i64_vec(&item0)?
                        .into_iter()
                        .map(|x| x as usize)
                        .collect();
                    let cols: Vec<usize> = to_i64_vec(&item1)?
                        .into_iter()
                        .map(|x| x as usize)
                        .collect();
                    let mut col_map = vec![-1i64; num_cols];
                    for (new_c, &old_c) in cols.iter().enumerate() {
                        if old_c < num_cols {
                            col_map[old_c] = new_c as i64;
                        }
                    }
                    let mut out_indices = Vec::new();
                    let mut out_indptr = vec![0usize; rows.len() + 1];
                    let mut out_real = if self.is_complex {
                        Vec::new()
                    } else {
                        Vec::new()
                    };
                    let mut out_cplx = if self.is_complex {
                        Vec::new()
                    } else {
                        Vec::new()
                    };

                    for (new_r, &r) in rows.iter().enumerate() {
                        if r < num_rows {
                            let start = self.rust_indptr[r];
                            let end = self.rust_indptr[r + 1];
                            for k in start..end {
                                let c = self.rust_indices[k];
                                if c < num_cols && col_map[c] >= 0 {
                                    out_indices.push(col_map[c] as usize);
                                    if self.is_complex {
                                        out_cplx.push(self.complex_data[k]);
                                    } else {
                                        out_real.push(self.real_data[k]);
                                    }
                                }
                            }
                        }
                        out_indptr[new_r + 1] = out_indices.len();
                    }
                    let res = Self {
                        shape: (rows.len(), cols.len()),
                        rust_indices: out_indices,
                        rust_indptr: out_indptr,
                        is_complex: self.is_complex,
                        real_data: out_real,
                        complex_data: out_cplx,
                        has_sorted_indices: false,
                        use_int32: self.use_int32,
                    };
                    return Ok(Bound::new(py, res)?.into_any());
                }
            }
        }

        if let Ok(rows) = to_i64_vec(&item) {
            let row_indices: Vec<usize> = rows.into_iter().map(|x| x as usize).collect();
            let mut out_indices = Vec::new();
            let mut out_indptr = vec![0usize; row_indices.len() + 1];
            let mut out_real = if self.is_complex {
                Vec::new()
            } else {
                Vec::new()
            };
            let mut out_cplx = if self.is_complex {
                Vec::new()
            } else {
                Vec::new()
            };

            for (new_r, &r) in row_indices.iter().enumerate() {
                if r < num_rows {
                    let start = self.rust_indptr[r];
                    let end = self.rust_indptr[r + 1];
                    for k in start..end {
                        out_indices.push(self.rust_indices[k]);
                        if self.is_complex {
                            out_cplx.push(self.complex_data[k]);
                        } else {
                            out_real.push(self.real_data[k]);
                        }
                    }
                }
                out_indptr[new_r + 1] = out_indices.len();
            }
            let res = Self {
                shape: (row_indices.len(), num_cols),
                rust_indices: out_indices,
                rust_indptr: out_indptr,
                is_complex: self.is_complex,
                real_data: out_real,
                complex_data: out_cplx,
                has_sorted_indices: self.has_sorted_indices,
                use_int32: self.use_int32,
            };
            return Ok(Bound::new(py, res)?.into_any());
        }

        if let Ok(single_r) = item.extract::<usize>() {
            let mut out_indices = Vec::new();
            let mut out_indptr = vec![0usize; 2];
            let mut out_real = if self.is_complex {
                Vec::new()
            } else {
                Vec::new()
            };
            let mut out_cplx = if self.is_complex {
                Vec::new()
            } else {
                Vec::new()
            };
            if single_r < num_rows {
                let start = self.rust_indptr[single_r];
                let end = self.rust_indptr[single_r + 1];
                for k in start..end {
                    out_indices.push(self.rust_indices[k]);
                    if self.is_complex {
                        out_cplx.push(self.complex_data[k]);
                    } else {
                        out_real.push(self.real_data[k]);
                    }
                }
            }
            out_indptr[1] = out_indices.len();
            let res = Self {
                shape: (1, num_cols),
                rust_indices: out_indices,
                rust_indptr: out_indptr,
                is_complex: self.is_complex,
                real_data: out_real,
                complex_data: out_cplx,
                has_sorted_indices: self.has_sorted_indices,
                use_int32: self.use_int32,
            };
            return Ok(Bound::new(py, res)?.into_any());
        }

        Err(PyIndexError::new_err("unsupported indexing on CSRMatrix"))
    }
}

impl PyCSRMatrix {
    fn add_sub_internal(&self, other: &PyCSRMatrix, sign: f64) -> PyResult<Self> {
        if self.shape != other.shape {
            return Err(PyValueError::new_err(
                "shape mismatch for addition/subtraction",
            ));
        }
        let (num_rows, num_cols) = self.shape;
        let is_cplx = self.is_complex || other.is_complex;
        let mut out_indices = Vec::new();
        let mut out_indptr = vec![0usize; num_rows + 1];
        let mut out_real = if is_cplx { Vec::new() } else { Vec::new() };
        let mut out_cplx = if is_cplx { Vec::new() } else { Vec::new() };

        for r in 0..num_rows {
            let mut pa = self.rust_indptr[r];
            let end_a = self.rust_indptr[r + 1];
            let mut pb = other.rust_indptr[r];
            let end_b = other.rust_indptr[r + 1];

            while pa < end_a || pb < end_b {
                if pa < end_a && (pb == end_b || self.rust_indices[pa] < other.rust_indices[pb]) {
                    out_indices.push(self.rust_indices[pa]);
                    if is_cplx {
                        let va = if self.is_complex {
                            self.complex_data[pa]
                        } else {
                            Complex64::new(self.real_data[pa], 0.0)
                        };
                        out_cplx.push(va);
                    } else {
                        out_real.push(self.real_data[pa]);
                    }
                    pa += 1;
                } else if pb < end_b && (pa == end_a || other.rust_indices[pb] < self.rust_indices[pa]) {
                    out_indices.push(other.rust_indices[pb]);
                    if is_cplx {
                        let vb = if other.is_complex {
                            other.complex_data[pb]
                        } else {
                            Complex64::new(other.real_data[pb], 0.0)
                        };
                        out_cplx.push(vb * sign);
                    } else {
                        out_real.push(other.real_data[pb] * sign);
                    }
                    pb += 1;
                } else {
                    let col = self.rust_indices[pa];
                    if is_cplx {
                        let va = if self.is_complex {
                            self.complex_data[pa]
                        } else {
                            Complex64::new(self.real_data[pa], 0.0)
                        };
                        let vb = if other.is_complex {
                            other.complex_data[pb]
                        } else {
                            Complex64::new(other.real_data[pb], 0.0)
                        };
                        let val = va + vb * sign;
                        if val.re.abs() > 1e-15 || val.im.abs() > 1e-15 {
                            out_indices.push(col);
                            out_cplx.push(val);
                        }
                    } else {
                        let val = self.real_data[pa] + other.real_data[pb] * sign;
                        if val.abs() > 1e-15 {
                            out_indices.push(col);
                            out_real.push(val);
                        }
                    }
                    pa += 1;
                    pb += 1;
                }
            }
            out_indptr[r + 1] = out_indices.len();
        }

        let use_int32 = num_rows < (i32::MAX as usize)
            && num_cols < (i32::MAX as usize)
            && out_indices.len() < (i32::MAX as usize);

        Ok(Self {
            shape: (num_rows, num_cols),
            rust_indices: out_indices,
            rust_indptr: out_indptr,
            is_complex: is_cplx,
            real_data: out_real,
            complex_data: out_cplx,
            has_sorted_indices: true,
            use_int32,
        })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCSRMatrix>()?;
    m.add_function(wrap_pyfunction!(csr_from_coo, m)?)?;
    m.add_function(wrap_pyfunction!(csr_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_expm_multiply, m)?)?;
    Ok(())
}
