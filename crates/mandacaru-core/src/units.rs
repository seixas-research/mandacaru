use crate::{Error, Result};

pub const BOHR_TO_ANGSTROM: f64 = 0.529177210903;
pub const HARTREE_TO_EV: f64 = 27.211386245988;
/// `1 / BOHR_TO_ANGSTROM`.
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / BOHR_TO_ANGSTROM;
/// `1 / HARTREE_TO_EV`.
pub const EV_TO_HARTREE: f64 = 1.0 / HARTREE_TO_EV;
/// The elementary charge in coulomb (exact in the SI since 2019).
pub const ELEMENTARY_CHARGE_COULOMB: f64 = 1.602176634e-19;
/// The Planck constant in joule seconds (exact in the SI since 2019).
pub const PLANCK_JOULE_SECOND: f64 = 6.62607015e-34;
/// The Bohr radius in meters.
pub const BOHR_TO_METER: f64 = BOHR_TO_ANGSTROM * 1e-10;
/// The speed of light in vacuum, meters per second (exact in the SI).
pub const SPEED_OF_LIGHT_METER_PER_SECOND: f64 = 299_792_458.;

/// The vacuum wavelength in nanometers of a photon of `energy` Hartree, `h c / E`.
pub fn wavelength_nm(energy: f64) -> Result<f64> {
    if !energy.is_finite() || energy <= 0. {
        return Err(Error::Invalid(
            "a wavelength needs a positive finite photon energy".into(),
        ));
    }
    let joule = energy * HARTREE_TO_EV * ELEMENTARY_CHARGE_COULOMB;
    Ok(PLANCK_JOULE_SECOND * SPEED_OF_LIGHT_METER_PER_SECOND / joule * 1e9)
}

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

/// What a result value measures, for its conversion from the engine's atomic units to the units
/// a user sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantity {
    /// Hartree to eV.
    Energy,
    /// Bohr to Angstrom.
    Length,
    /// Hartree/Bohr to eV/Angstrom.
    Force,
    /// An energy gradient `dE/dR` (Hartree/Bohr) to the force `-dE/dR` (eV/Angstrom).
    GradientToForce,
    /// Hartree/Bohr^3 to eV/Angstrom^3.
    Stress,
    /// Bohr^2 to Angstrom^2 (a spread).
    Area,
    /// 1/Hartree to 1/eV (a response function), as the value times `1 / HARTREE_TO_EV`.
    InverseEnergy,
    /// e/Bohr^2 to C/m^2 (an electric polarization).
    Polarization,
    /// e^2/(hbar Bohr) to S/cm (a conductivity).
    Conductivity,
}

impl Quantity {
    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "energy" => Self::Energy,
            "length" => Self::Length,
            "force" => Self::Force,
            "gradient_to_force" => Self::GradientToForce,
            "stress" => Self::Stress,
            "area" => Self::Area,
            "inverse_energy" => Self::InverseEnergy,
            "polarization" => Self::Polarization,
            "conductivity" => Self::Conductivity,
            _ => {
                return Err(Error::Invalid(
                    "quantity must be 'energy', 'length', 'force', 'gradient_to_force', 'stress', \
                     'area', 'inverse_energy', 'polarization' or 'conductivity'"
                        .into(),
                ))
            }
        })
    }
    /// The factor a value in atomic units is multiplied by: one in atomic units (minus one for a
    /// gradient turned into a force).
    pub fn factor(self, atomic: bool) -> f64 {
        let sign = if self == Self::GradientToForce { -1. } else { 1. };
        if atomic {
            return sign;
        }
        sign * match self {
            Self::Energy => HARTREE_TO_EV,
            Self::Length => BOHR_TO_ANGSTROM,
            Self::Force | Self::GradientToForce => HARTREE_TO_EV / BOHR_TO_ANGSTROM,
            Self::Stress => HARTREE_TO_EV / BOHR_TO_ANGSTROM.powf(3.),
            Self::Area => BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM,
            Self::InverseEnergy => 1.0 / HARTREE_TO_EV,
            Self::Polarization => ELEMENTARY_CHARGE_COULOMB / (BOHR_TO_METER * BOHR_TO_METER),
            // e^2/hbar = 2 pi e^2/h siemens, per Bohr in centimeters.
            Self::Conductivity => {
                2. * std::f64::consts::PI * ELEMENTARY_CHARGE_COULOMB * ELEMENTARY_CHARGE_COULOMB
                    / PLANCK_JOULE_SECOND
                    / (BOHR_TO_METER * 100.)
            }
        }
    }
}

/// `values` (atomic units) in the units a user sees (`atomic`: kept in atomic units), each
/// divided by `cells` when the value is per primitive cell of a supercell of that many cells.
/// A gradient becomes a force by its sign: `(-g) * factor`.
pub fn to_user(values: &[f64], quantity: Quantity, atomic: bool, cells: u64) -> Result<Vec<f64>> {
    if cells == 0 {
        return Err(Error::Invalid("a supercell holds at least one cell".into()));
    }
    let magnitude = quantity.factor(atomic).abs();
    let flip = quantity == Quantity::GradientToForce;
    let per_cell = 1.0 / cells as f64;
    Ok(values
        .iter()
        .map(|&v| {
            let value = if flip { -v } else { v };
            let user = value * magnitude;
            if cells == 1 { user } else { user * per_cell }
        })
        .collect())
}

/// [`to_user`] of `values`, then `plus` (already in user units, one per value) added: the
/// dispersion forces of a result, for one.
pub fn to_user_values(
    values: &[f64],
    quantity: Quantity,
    atomic: bool,
    cells: u64,
    plus: Option<&[f64]>,
) -> Result<Vec<f64>> {
    let mut out = to_user(values, quantity, atomic, cells)?;
    if let Some(extra) = plus {
        if extra.len() != out.len() {
            return Err(Error::Invalid("the added values must match the converted ones".into()));
        }
        for (value, add) in out.iter_mut().zip(extra) {
            *value += add;
        }
    }
    Ok(out)
}

/// The sum of `values` (atomic units, added in order) in user units: an energy and the terms
/// added to it.
pub fn to_user_total(values: &[f64], quantity: Quantity, atomic: bool, cells: u64) -> Result<f64> {
    let total: f64 = values.iter().sum();
    Ok(to_user(&[total], quantity, atomic, cells)?[0])
}

/// The Hartree-to-`unit` factor of an accepted energy spelling: `eV`, or `Ha`, `hartree`,
/// `au` (case-insensitive).
pub fn energy_factor(unit: &str) -> Result<f64> {
    match unit.to_ascii_lowercase().as_str() {
        "ev" => Ok(HARTREE_TO_EV),
        "ha" | "hartree" | "au" => Ok(1.),
        _ => Err(Error::Invalid(format!("unknown energy unit {unit:?}; use 'eV' or 'Ha'"))),
    }
}

/// The `unit`-to-Bohr factor of an accepted length spelling: `angstrom`, `a`, or `bohr`, `au`,
/// `a0` (case-insensitive).
pub fn length_factor_to_bohr(unit: &str) -> Result<f64> {
    match unit.to_ascii_lowercase().as_str() {
        "angstrom" | "a" => Ok(ANGSTROM_TO_BOHR),
        "bohr" | "au" | "a0" => Ok(1.),
        _ => Err(Error::Invalid(format!("unknown length unit {unit:?}; use 'angstrom' or 'bohr'"))),
    }
}

/// How an energy is converted: from Hartree (`value * factor`), to Hartree (`value / factor`),
/// or between two units (`value * (to / from)`, the value itself when the ratio is one).
#[derive(Debug, Clone, Copy)]
pub enum EnergyConversion {
    FromHartree(f64),
    ToHartree(f64),
    Ratio(f64),
}

impl EnergyConversion {
    pub fn between(from: &str, to: &str) -> Result<Self> {
        Ok(Self::Ratio(energy_factor(to)? / energy_factor(from)?))
    }
    pub fn apply(self, value: f64) -> f64 {
        match self {
            Self::FromHartree(f) => value * f,
            Self::ToHartree(f) => value / f,
            Self::Ratio(f) => if f == 1. { value } else { value * f },
        }
    }
}
