//! Checked column-major multiplication, including padded grid tiles.
//! Platform math kernels execute on the calling thread; they never change
//! a process-wide thread setting or introduce a second scientific engine.
use crate::{Error, Result};
use nalgebra::{DMatrixView, DMatrixViewMut};

pub struct MatrixRef<'a> {
    pub values: &'a [f64],
    pub rows: usize,
    pub columns: usize,
    pub leading: usize,
}
pub struct MatrixMut<'a> {
    pub values: &'a mut [f64],
    pub rows: usize,
    pub columns: usize,
    pub leading: usize,
}
fn validate(length: usize, rows: usize, columns: usize, leading: usize) -> Result<()> {
    let size = if rows == 0 || columns == 0 {
        0
    } else {
        (columns - 1)
            .checked_mul(leading)
            .and_then(|n| n.checked_add(rows))
            .ok_or(Error::Overflow)?
    };
    if leading < rows || size > length {
        return Err(Error::Invalid(
            "matrix multiplication storage disagrees with its dimensions and leading stride".into(),
        ));
    }
    Ok(())
}
pub fn multiply(
    a: MatrixRef<'_>,
    b: MatrixRef<'_>,
    c: MatrixMut<'_>,
    alpha: f64,
    beta: f64,
) -> Result<()> {
    #[cfg(target_os = "macos")]
    let mut c = c;
    validate(a.values.len(), a.rows, a.columns, a.leading)?;
    validate(b.values.len(), b.rows, b.columns, b.leading)?;
    validate(c.values.len(), c.rows, c.columns, c.leading)?;
    if a.columns != b.rows
        || c.rows != a.rows
        || c.columns != b.columns
        || !alpha.is_finite()
        || !beta.is_finite()
    {
        return Err(Error::Invalid(
            "matrix multiplication dimensions or scale factors disagree".into(),
        ));
    }
    if c.rows == 0 || c.columns == 0 {
        return Ok(());
    }
    if a.columns == 0 {
        for column in 0..c.columns {
            for row in 0..c.rows {
                let value = &mut c.values[row + column * c.leading];
                *value = if beta == 0. { 0. } else { *value * beta };
            }
        }
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    if accelerate::multiply(&a, &b, &mut c, alpha, beta) {
        return Ok(());
    }
    let left = DMatrixView::from_slice_with_strides(a.values, a.rows, a.columns, 1, a.leading);
    let right = DMatrixView::from_slice_with_strides(b.values, b.rows, b.columns, 1, b.leading);
    let mut result =
        DMatrixViewMut::from_slice_with_strides_mut(c.values, c.rows, c.columns, 1, c.leading);
    result.gemm(alpha, &left, &right, beta);
    Ok(())
}

#[cfg(target_os = "macos")]
mod accelerate {
    use super::*;
    use std::{
        ffi::{c_char, c_int, c_uint, c_void},
        sync::OnceLock,
    };
    type Gemm = unsafe extern "C" fn(
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        f64,
        *const f64,
        c_int,
        *const f64,
        c_int,
        f64,
        *mut f64,
        c_int,
    );
    type SetThreads = unsafe extern "C" fn(c_uint) -> c_int;
    type GetThreads = unsafe extern "C" fn() -> c_uint;
    struct Functions {
        gemm: Gemm,
        set: SetThreads,
        get: GetThreads,
    }
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {}
    unsafe extern "C" {
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    static FUNCTIONS: OnceLock<Option<Functions>> = OnceLock::new();
    fn functions() -> Option<&'static Functions> {
        FUNCTIONS
            .get_or_init(|| {
                // Darwin's RTLD_DEFAULT searches the linked system framework.
                // Runtime lookup preserves the package's deployment target on
                // systems predating the per-thread BLAS policy API.
                let handle = (-2isize) as *mut c_void;
                // SAFETY: fixed, NUL-terminated SDK symbol names; no library is
                // unloaded. The NEWLAPACK symbol has the 32-bit integer ABI.
                let (gemm, set, get) = unsafe {
                    (
                        dlsym(handle, c"cblas_dgemm$NEWLAPACK".as_ptr()),
                        dlsym(handle, c"BLASSetThreading".as_ptr()),
                        dlsym(handle, c"BLASGetThreading".as_ptr()),
                    )
                };
                if gemm.is_null() || set.is_null() || get.is_null() {
                    return None;
                }
                // SAFETY: signatures match cblas_new.h and thread_api.h in the
                // platform SDK. All pointers were checked and remain valid.
                Some(unsafe {
                    Functions {
                        gemm: std::mem::transmute::<*mut c_void, Gemm>(gemm),
                        set: std::mem::transmute::<*mut c_void, SetThreads>(set),
                        get: std::mem::transmute::<*mut c_void, GetThreads>(get),
                    }
                })
            })
            .as_ref()
    }
    struct Restore<'a> {
        functions: &'a Functions,
        previous: c_uint,
    }
    impl Drop for Restore<'_> {
        fn drop(&mut self) {
            // SAFETY: restores the model returned for this same calling thread.
            unsafe {
                (self.functions.set)(self.previous);
            }
        }
    }
    pub(super) fn multiply(
        a: &MatrixRef<'_>,
        b: &MatrixRef<'_>,
        c: &mut MatrixMut<'_>,
        alpha: f64,
        beta: f64,
    ) -> bool {
        let Some(functions) = functions() else {
            return false;
        };
        let sizes = [
            a.rows, b.columns, a.columns, a.leading, b.leading, c.leading,
        ];
        let mut abi = [0i32; 6];
        for (to, &from) in abi.iter_mut().zip(&sizes) {
            let Ok(size) = i32::try_from(from) else {
                return false;
            };
            *to = size;
        }
        // SAFETY: the API reads and changes only this calling thread's model.
        let previous = unsafe { (functions.get)() };
        if unsafe { (functions.set)(1) } != 0 {
            return false;
        }
        let _restore = Restore {
            functions,
            previous,
        };
        // SAFETY: the public wrapper checked every storage extent and matrix
        // dimension. Column-major views use unit row stride and the supplied
        // leading dimensions; Rust borrows guarantee disjoint mutable output.
        // BLAS runs single-threaded and the prior policy is restored on return.
        unsafe {
            (functions.gemm)(
                102,
                111,
                111,
                abi[0],
                abi[1],
                abi[2],
                alpha,
                a.values.as_ptr(),
                abi[3],
                b.values.as_ptr(),
                abi[4],
                beta,
                c.values.as_mut_ptr(),
                abi[5],
            );
        }
        true
    }
    #[cfg(test)]
    #[test]
    fn calling_thread_policy_is_restored() {
        if let Some(f) = functions() {
            eprintln!("Using the system GEMM with a calling-thread BLAS policy");
            let previous = unsafe { (f.get)() };
            let mut output = [0.];
            super::multiply(
                MatrixRef {
                    values: &[2.],
                    rows: 1,
                    columns: 1,
                    leading: 1,
                },
                MatrixRef {
                    values: &[3.],
                    rows: 1,
                    columns: 1,
                    leading: 1,
                },
                MatrixMut {
                    values: &mut output,
                    rows: 1,
                    columns: 1,
                    leading: 1,
                },
                1.,
                0.,
            )
            .unwrap();
            assert_eq!(output, [6.]);
            assert_eq!(unsafe { (f.get)() }, previous);
        }
    }
}
