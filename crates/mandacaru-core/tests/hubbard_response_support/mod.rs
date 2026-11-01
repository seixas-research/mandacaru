//! Exactly solvable models for `tests/hubbard_response.rs`. Every reference is solved
//! here by a routine that shares no code with the response kernel: a cyclic Jacobi
//! diagonalization, a Gauss-Jordan inverse, a Sturm-count bisection, closed forms.
#![allow(dead_code)]

pub mod exact;
pub mod models;

pub use exact::*;
pub use models::*;

use mandacaru_core::models::hubbard_response::{measure, Fit, Options, Probe, Response};
use mandacaru_core::{Context, Error};
use nalgebra::DMatrix;

pub fn context() -> Context {
    Context::new(Some(1), 64 << 20).unwrap()
}

pub fn run(probe: &mut dyn Probe, alphas: &[f64], fit: Fit) -> Response {
    measure(&context(), probe, &Options::new(alphas, fit).unwrap()).unwrap()
}

pub fn refusal(probe: &mut dyn Probe, options: &Options) -> String {
    match measure(&context(), probe, options) {
        Err(Error::Invalid(m)) => m,
        Err(other) => panic!("expected an Invalid refusal, got {other:?}"),
        Ok(_) => panic!("expected a refusal"),
    }
}

pub fn mat(rows: usize, values: &[f64]) -> DMatrix<f64> {
    DMatrix::from_row_slice(rows, values.len() / rows, values)
}

pub fn worst(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    (a - b).iter().fold(0., |m, v| f64::max(m, v.abs()))
}

pub fn two_level(vu: Vu) -> TwoLevel {
    TwoLevel {
        d0: 0.3,
        t: 0.5,
        w: 0.9,
        kappa: 0.4,
        u_in: 1.2,
        vu,
    }
}

