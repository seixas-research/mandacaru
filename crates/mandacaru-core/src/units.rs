use crate::{Error, Result};

pub const BOHR_TO_ANGSTROM: f64 = 0.529177210903;
pub const HARTREE_TO_EV: f64 = 27.211386245988;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LengthUnit {
    Bohr,
    Angstrom,
}

impl LengthUnit {
    pub fn parse(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "bohr" | "au" | "a0" => Ok(Self::Bohr),
            "angstrom" | "a" => Ok(Self::Angstrom),
            _ => Err(Error::Invalid(
                "length unit must be 'angstrom' or 'bohr'".into(),
            )),
        }
    }
    pub fn to_bohr(self, value: f64) -> f64 {
        match self {
            Self::Bohr => value,
            Self::Angstrom => value * (1.0 / BOHR_TO_ANGSTROM),
        }
    }
}
