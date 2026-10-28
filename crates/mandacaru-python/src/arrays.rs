//! Input validation and owned NumPy transfers; no scientific computation.
use mandacaru_core::{product, Buffer, Context, Error, Reservation};
use numpy::{
    ndarray::{ArrayD, Dimension, IxDyn},
    Complex64, IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArray, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyMemoryError, PyOverflowError, PyRuntimeError, PyValueError},
    prelude::*,
};
pub(crate) type Array<'py, T> = Bound<'py, PyArrayDyn<T>>;
pub(crate) type NativeComplex = mandacaru_core::Complex64;
pub(crate) fn error(error: Error) -> PyErr {
    match error {
        Error::Invalid(_) => PyValueError::new_err(error.to_string()),
        Error::Overflow => PyOverflowError::new_err(error.to_string()),
        Error::Budget { .. } | Error::Allocation => PyMemoryError::new_err(error.to_string()),
        Error::Kernel(_) => PyRuntimeError::new_err(error.to_string()),
    }
}

pub(crate) fn array<'py, T: numpy::Element>(
    py: Python<'py>,
    shape: &[usize],
    data: Vec<T>,
) -> PyResult<Array<'py, T>> {
    Ok(ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .into_pyarray(py))
}

pub(crate) fn layout<T: numpy::Element, D: Dimension>(
    input: &PyReadonlyArray<'_, T, D>,
) -> PyResult<()> {
    if !input.is_c_contiguous() || (input.data() as usize) % std::mem::align_of::<T>() != 0 {
        return Err(PyValueError::new_err(
            "array must be aligned and C-contiguous",
        ));
    }
    Ok(())
}

pub(crate) fn copy<T: numpy::Element + Copy, D: Dimension>(
    ctx: &Context,
    input: &PyReadonlyArray<'_, T, D>,
) -> PyResult<(Vec<T>, Reservation)> {
    layout(input)?;
    let reservation = ctx
        .reserve(product(&[input.len(), std::mem::size_of::<T>()]).map_err(error)?)
        .map_err(error)?;
    Ok((input.as_slice()?.to_vec(), reservation))
}

pub(crate) fn complex<D: Dimension>(
    ctx: &Context,
    input: &PyReadonlyArray<'_, Complex64, D>,
) -> PyResult<(Vec<NativeComplex>, Reservation)> {
    layout(input)?;
    let reservation = ctx
        .reserve(product(&[input.len(), 16]).map_err(error)?)
        .map_err(error)?;
    Ok((
        input
            .as_slice()?
            .iter()
            .map(|v| NativeComplex::new(v.re, v.im))
            .collect(),
        reservation,
    ))
}

pub(crate) fn complex_array<'py>(
    py: Python<'py>,
    shape: &[usize],
    data: &Buffer<NativeComplex>,
) -> PyResult<Array<'py, Complex64>> {
    array(
        py,
        shape,
        data.as_slice()
            .iter()
            .map(|v| Complex64::new(v.re, v.im))
            .collect(),
    )
}
