//! Scientific definitions shared by solvers and language bindings.
pub mod atom;
pub mod configuration;
pub mod dispersion;
pub mod elements;
pub mod hubbard;
pub mod hubbard_liechtenstein;
pub mod hubbard_request;
pub mod hubbard_response;
pub mod hubbard_slater;
pub mod hubbard_spinor;
pub mod mixing;
pub mod nonlocal;
pub mod noncollinear;
pub mod numerov;
pub mod radial_xc;
pub mod relativity;
pub mod selection;
pub mod spectrum;
pub mod spherical;
pub mod spin_orbit;
pub mod ts;
pub mod xc;
pub mod xc_scheme;
fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::Invalid(message.into())
}

use crate::{Error, Result};

/// Range-separation parameter of HSE06 (Bohr^-1): the one definition every screened-exchange
/// path reads, through the `screened_exchange` entry of the functional registry.
pub const HSE06_OMEGA: f64 = 0.11;
/// Fraction of short-range exact exchange in HSE06, which the semilocal part gives up.
pub const HSE06_EXACT_FRACTION: f64 = 0.25;

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
        screened_exchange: Some((HSE06_OMEGA, HSE06_EXACT_FRACTION)),
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
