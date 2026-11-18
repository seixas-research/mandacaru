//! Electronic smearing distributions and chemical-potential root finding.

use crate::{Error, Result};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Smearing {
    FermiDirac,
    Gaussian,
    MethfesselPaxton,
}

impl Smearing {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "fermi-dirac" | "fd" | "fermi" => Ok(Self::FermiDirac),
            "gaussian" => Ok(Self::Gaussian),
            "methfessel-paxton" | "mp" => Ok(Self::MethfesselPaxton),
            _ => Err(Error::Invalid("unknown smearing method".into())),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::FermiDirac => "fermi-dirac",
            Self::Gaussian => "gaussian",
            Self::MethfesselPaxton => "methfessel-paxton",
        }
    }

    pub fn occupation(self, x: f64) -> f64 {
        match self {
            Self::FermiDirac => 0.5 * (1. - (0.5 * x).tanh()),
            Self::Gaussian => 0.5 * libm::erfc(x),
            Self::MethfesselPaxton => {
                0.5 * libm::erfc(x) - x * (-x * x).exp() / (2. * PI.sqrt())
            }
        }
    }

    /// Derivative with respect to the dimensionless energy argument.
    pub fn slope(self, x: f64) -> f64 {
        match self {
            Self::FermiDirac => 0.25 * ((0.5 * x).tanh().powi(2) - 1.),
            Self::Gaussian if x.abs() <= 40. => -(-x * x).exp() / PI.sqrt(),
            Self::MethfesselPaxton if x.abs() <= 40. => (x * x - 1.5) * (-x * x).exp() / PI.sqrt(),
            _ => 0.,
        }
    }

    pub fn entropy(self, x: f64) -> f64 {
        match self {
            Self::FermiDirac => {
                let f = self.occupation(x);
                let term = |v: f64| if v > 0. { v * v.ln() } else { 0. };
                -term(f) - term(1. - f)
            }
            Self::Gaussian => (-x * x).exp() / (2. * PI.sqrt()),
            Self::MethfesselPaxton => -(2. * x * x - 1.) * (-x * x).exp() / (4. * PI.sqrt()),
        }
    }
}

// Pairwise eight-lane reduction avoids moving a saturated insulating Fermi
// bracket merely by changing the nesting of k-point and band summations.
fn occupation_sum(mut values: impl Iterator<Item = f64>, length: usize) -> f64 {
    if length < 8 {
        return values.sum();
    }
    let mut lanes = std::array::from_fn::<_, 8, _>(|_| values.next().unwrap_or(0.));
    for _ in 0..(length / 8 - 1) {
        for lane in &mut lanes {
            *lane += values.next().unwrap_or(0.);
        }
    }
    let sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    values.fold(sum, |a, b| a + b)
}

pub fn fermi(
    eps: &[Vec<f64>],
    weights: &[f64],
    electrons: f64,
    method: Smearing,
    width: f64,
    degeneracy: f64,
) -> Result<f64> {
    if eps.is_empty()
        || eps.len() != weights.len()
        || eps
            .iter()
            .any(|v| v.is_empty() || v.iter().any(|x| !x.is_finite()))
        || weights.iter().any(|w| !w.is_finite() || *w < 0.)
        || !width.is_finite()
        || width <= 0.
        || !electrons.is_finite()
        || electrons < 0.
        || !degeneracy.is_finite()
        || degeneracy <= 0.
    {
        return Err(Error::Invalid(
            "invalid Fermi-level dimensions, energies, electron count or smearing width".into(),
        ));
    }
    let capacity = degeneracy
        * eps
            .iter()
            .zip(weights)
            .map(|(e, w)| e.len() as f64 * w)
            .sum::<f64>();
    if electrons > capacity + 1e-9 || capacity <= 0. {
        return Err(Error::Invalid("electron count exceeds the band capacity".into()));
    }
    let mut lo = eps.iter().flatten().copied().fold(f64::INFINITY, f64::min) - 60. * width;
    let mut hi = eps
        .iter()
        .flatten()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        + 60. * width;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let length = eps.iter().map(Vec::len).sum();
        let count = degeneracy
            * occupation_sum(
                eps.iter().zip(weights).flat_map(|(e, w)| {
                    e.iter()
                        .map(move |e| w * method.occupation((e - mid) / width))
                }),
                length,
            );
        if count < electrons {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-13 {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}
