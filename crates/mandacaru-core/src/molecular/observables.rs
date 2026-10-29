//! Molecular density matrices, natural orbitals and sampled scalar fields.
use super::{invalid, MeanField, MolecularIntegrals};
use crate::{
    math::{linalg::eigen, SpatialGrid},
    product, Buffer, Context, Reservation, Result,
};
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;
type Matrix = DMatrix<C64>;

pub struct ActiveDensity<'a> {
    /// <a_p^dagger a_q>, row-major, alpha then beta active modes.
    pub gamma: &'a [C64],
    pub active: &'a [usize],
    pub frozen: &'a [usize],
    pub rotation: Option<&'a Matrix>,
}
pub struct Analysis {
    pub occupations: Vec<f64>,
    /// Columns in the original nonorthogonal atomic basis.
    pub coefficients: Matrix,
    pub mo_coefficients: Option<Matrix>,
    pub electrons: f64,
    density: Matrix,
    reference: Matrix,
    orbitals: Matrix,
    _memory: Reservation,
}

pub fn analyze(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    active: Option<ActiveDensity<'_>>,
) -> Result<Analysis> {
    let m = integrals.m();
    let memory = integrals.context.reserve(product(&[m, m, 16, 24])?)?;
    let reference_density = if reference.method == "ghf" {
        Matrix::from_fn(m, m, |a, b| {
            reference.densities[0][(a, b)] + reference.densities[0][(m + a, m + b)]
        })
    } else if reference.densities.len() == 1 {
        &reference.densities[0] * C64::new(2., 0.)
    } else {
        &reference.densities[0] + &reference.densities[1]
    };
    let density = if let Some(input) = active {
        if reference.method == "ghf" {
            return Err(invalid(
                "spatial active densities cannot be applied to GHF spinors",
            ));
        }
        let a = input.active.len();
        let n = a.checked_mul(2).ok_or(crate::Error::Overflow)?;
        if a == 0 || input.gamma.len() != product(&[n, n])? {
            return Err(invalid(
                "active one-particle RDM has inconsistent dimensions",
            ));
        }
        let mut seen = vec![false; m];
        for &index in input.active.iter().chain(input.frozen) {
            if index >= m || seen[index] {
                return Err(invalid(
                    "active and frozen orbitals must be disjoint valid indices",
                ));
            }
            seen[index] = true;
        }
        if input
            .gamma
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(invalid("one-particle RDM must be finite"));
        }
        for p in 0..n {
            for q in 0..n {
                if (input.gamma[p * n + q] - input.gamma[q * n + p].conj()).norm() > 1e-9 {
                    return Err(invalid("one-particle RDM must be Hermitian"));
                }
            }
        }
        let mut mo = Matrix::zeros(m, m);
        for &i in input.frozen {
            mo[(i, i)] = C64::new(2., 0.);
        }
        for (p, &i) in input.active.iter().enumerate() {
            for (q, &j) in input.active.iter().enumerate() {
                // The orbital density D_pq is <a_q^dagger a_p>.
                mo[(i, j)] = input.gamma[q * n + p] + input.gamma[(a + q) * n + a + p];
            }
        }
        let mo = if let Some(rotation) = input.rotation {
            if rotation.shape() != (m, m)
                || rotation
                    .iter()
                    .any(|c| !c.re.is_finite() || !c.im.is_finite())
                || (rotation.adjoint() * rotation - Matrix::identity(m, m))
                    .iter()
                    .any(|c| c.norm() > 1e-8)
            {
                return Err(invalid(
                    "orbital rotation must be a finite unitary matrix matching the source orbitals",
                ));
            }
            rotation * mo * rotation.adjoint()
        } else {
            mo
        };
        &reference.model_orbitals * mo * reference.model_orbitals.adjoint()
    } else {
        reference_density.clone()
    };
    let density = (&density + density.adjoint()) * C64::new(0.5, 0.);
    let electrons = density.trace().re;
    let (values, vectors) = eigen(density.clone())?;
    if values.iter().any(|&v| !(-1e-8..=2. + 1e-8).contains(&v)) {
        return Err(invalid(
            "spatial natural occupations must be between zero and two",
        ));
    }
    let occupations = values.iter().rev().map(|v| v.clamp(0., 2.)).collect();
    let orbitals = Matrix::from_fn(m, m, |i, j| vectors[(i, m - 1 - j)]);
    let coefficients = integrals.lowdin.map(|v| C64::new(v, 0.)) * &orbitals;
    let mo_coefficients = if reference.method == "ghf" {
        None
    } else {
        Some(reference.model_orbitals.adjoint() * &orbitals)
    };
    Ok(Analysis {
        occupations,
        coefficients,
        mo_coefficients,
        electrons,
        density,
        reference: reference_density,
        orbitals,
        _memory: memory,
    })
}

pub struct Field {
    pub values: Buffer<f64>,
    pub integral: f64,
    pub norm: f64,
    pub component: String,
    pub augmentation_charge: f64,
    pub electrons: f64,
}
pub fn sample(
    integrals: &MolecularIntegrals,
    state: &Analysis,
    grid: &SpatialGrid,
    quantity: &str,
    index: usize,
    component: &str,
) -> Result<Field> {
    if grid.periodic {
        return Err(invalid("molecular fields need an isolated grid"));
    }
    if ![
        "density",
        "reference_density",
        "difference_density",
        "natural_orbital",
    ]
    .contains(&quantity)
    {
        return Err(invalid("unknown native molecular field quantity"));
    }
    if !["auto", "real", "imag", "modulus"].contains(&component) {
        return Err(invalid(
            "field component must be auto, real, imag or modulus",
        ));
    }
    let m = integrals.m();
    if quantity == "natural_orbital" && index >= m {
        return Err(invalid("natural orbital index outside the basis"));
    }
    let ctx: &Context = &integrals.context;
    let samples = integrals.basis.sample(ctx, grid)?;
    let g = grid.index.points();
    let _work = ctx.reserve(
        product(&[m, m, 16, 4])?
            .checked_add(product(&[m, 16, 2])?)
            .ok_or(crate::Error::Overflow)?,
    )?;
    let mut values = ctx.allocate::<f64>(g)?;
    let mut amplitudes = ctx.allocate::<C64>(g)?;
    let density = match quantity {
        "reference_density" => state.reference.clone(),
        "difference_density" => &state.density - &state.reference,
        _ => state.density.clone(),
    };
    let electrons = if quantity == "natural_orbital" {
        state.electrons
    } else {
        density.trace().re
    };
    // The displayed PAW field is the smooth valence density. Account for the
    // missing on-site charge separately, using the same metric as the solve.
    let augmentation_charge = if quantity == "natural_orbital" {
        0.
    } else {
        integrals.corrections.as_ref().map_or(0., |c| {
            let metric = integrals.lowdin.transpose() * &c.overlap * &integrals.lowdin;
            (0..m)
                .map(|a| {
                    (0..m)
                        .map(|b| density[(a, b)].re * metric[(b, a)])
                        .sum::<f64>()
                })
                .sum()
        })
    };
    let mut phi = vec![0.; m];
    for k in 0..g {
        for (p, value) in phi.iter_mut().enumerate() {
            *value = (0..m)
                .map(|a| integrals.lowdin[(a, p)] * samples.as_slice()[a * g + k])
                .sum();
        }
        let value = if quantity == "natural_orbital" {
            (0..m)
                .map(|p| phi[p] * state.orbitals[(p, index)])
                .sum::<C64>()
        } else {
            (0..m)
                .map(|p| {
                    (0..m)
                        .map(|q| phi[p] * density[(p, q)] * phi[q])
                        .sum::<C64>()
                })
                .sum()
        };
        amplitudes.as_mut_slice()[k] = value;
    }
    let pivot = amplitudes
        .as_slice()
        .iter()
        .max_by(|a, b| a.norm_sqr().total_cmp(&b.norm_sqr()))
        .copied()
        .unwrap_or_default();
    let phase = if quantity == "natural_orbital" && pivot.norm() > 0. {
        pivot.conj() / pivot.norm()
    } else {
        C64::new(1., 0.)
    };
    let component = if component == "auto" {
        let mut real = 0f64;
        let mut imaginary = 0f64;
        for value in amplitudes.as_slice() {
            let value = value * phase;
            real = real.max(value.re.abs());
            imaginary = imaginary.max(value.im.abs());
        }
        if quantity == "natural_orbital" && imaginary > 1e-10 * real.max(1e-30) {
            return Err(invalid("the orbital is complex after phase alignment; choose component real, imag or modulus"));
        }
        "real"
    } else {
        component
    };
    for (out, value) in values.as_mut_slice().iter_mut().zip(amplitudes.as_slice()) {
        let value = value * phase;
        let real = match component {
            "real" => value.re,
            "imag" => value.im,
            _ => value.norm(),
        };
        if !real.is_finite() {
            return Err(invalid("nonfinite molecular field"));
        }
        *out = real;
    }
    let integral = values.as_slice().iter().sum::<f64>() * grid.index.dv();
    let norm = values.as_slice().iter().map(|v| v * v).sum::<f64>() * grid.index.dv();
    crate::finite(&[integral, norm], "field integrals")?;
    Ok(Field {
        values,
        integral,
        norm,
        component: component.to_owned(),
        augmentation_charge,
        electrons,
    })
}
