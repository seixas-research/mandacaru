//! Scientific definitions shared by solvers and language bindings.
pub mod atom;
pub mod dispersion;
pub mod elements;
pub mod nonlocal;
pub mod spherical;
pub mod xc;
fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::Invalid(message.into())
}

use crate::{Error, Result};

pub struct Functional {
    pub name: &'static str,
    pub semilocal: &'static str,
    pub requires_tau: bool,
    pub screened_exchange: Option<(f64, f64)>,
    pub nonlocal_b: Option<f64>,
}

pub const FUNCTIONALS: &[Functional] = &[
    Functional {
        name: "lda",
        semilocal: "lda",
        requires_tau: false,
        screened_exchange: None,
        nonlocal_b: None,
    },
    Functional {
        name: "pbe",
        semilocal: "pbe",
        requires_tau: false,
        screened_exchange: None,
        nonlocal_b: None,
    },
    Functional {
        name: "r2scan",
        semilocal: "r2scan",
        requires_tau: true,
        screened_exchange: None,
        nonlocal_b: None,
    },
    Functional {
        name: "hse06",
        semilocal: "hse06",
        requires_tau: false,
        screened_exchange: Some((0.11, 0.25)),
        nonlocal_b: None,
    },
    Functional {
        name: "pbe+rvv10",
        semilocal: "pbe",
        requires_tau: false,
        screened_exchange: None,
        nonlocal_b: Some(6.3),
    },
    Functional {
        name: "r2scan+rvv10",
        semilocal: "r2scan",
        requires_tau: true,
        screened_exchange: None,
        nonlocal_b: Some(11.95),
    },
];

pub fn functional(name: &str) -> Result<&'static Functional> {
    FUNCTIONALS
        .iter()
        .find(|f| f.name == name)
        .ok_or_else(|| Error::Invalid(format!("unknown canonical XC functional {name:?}")))
}
