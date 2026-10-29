//! Periodic meta-GGA fields, Bloch kinetic densities and their matrix action.
use super::{
    invalid,
    solver::{derivative, sym_field},
    Crystal, KPoint,
};
use crate::{
    math::{
        fft::Fft3,
        projection::{hermitian_density, ComplexProjector},
        reciprocal::{self, BlochSymbols},
        storage::MatrixSource,
    },
    models::{self, nonlocal, xc},
    product,
    quantum::C64,
    Buffer, Result,
};
use nalgebra::DMatrix;

/// A kinetic-density guess from actual core-Hamiltonian Bloch orbitals.
/// Each SCF iteration replaces it with the new occupied orbital density.
pub(super) fn initial_kinetic_density(
    c: &Crystal,
    options: &super::Options,
) -> Result<Buffer<f64>> {
    let m = c.m;
    let mut energies = Vec::new();
    let mut coefficients = Vec::new();
    let mut weights = Vec::new();
    for _ in 0..c.spins {
        for point in &c.points {
            let (eps, vectors) =
                crate::math::linalg::eigen(point.lowdin.adjoint() * &point.fixed * &point.lowdin)?;
            energies.push(eps);
            coefficients.push(&point.lowdin * vectors);
            weights.push(point.weight);
        }
    }
    let degeneracy = if c.spins == 1 { 2. } else { 1. };
    let mu = super::spectrum::fermi(
        &energies,
        &weights,
        c.electrons,
        options.smearing,
        options.width,
        degeneracy,
    )?;
    let matrices: Vec<_> = coefficients
        .iter()
        .zip(&energies)
        .map(|(vectors, eps)| {
            vectors
                * DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
                    m,
                    eps.iter().map(|e| {
                        C64::new(
                            degeneracy * options.smearing.occupation((e - mu) / options.width),
                            0.,
                        )
                    }),
                ))
                * vectors.adjoint()
        })
        .collect();
    kinetic_density(c, &matrices)
}

pub struct Terms {
    pub energy: f64,
    pub potential: Buffer<f64>,
    pub kinetic_potential: Option<Buffer<f64>>,
}

pub fn evaluate(c: &Crystal, fft: &mut Fft3, rho: &[f64], tau: Option<&[f64]>) -> Result<Terms> {
    let ctx = &c.context;
    let definition = models::functional(&c.functional)?;
    let n = c.grid.index.points();
    if rho.len() != product(&[c.spins, n])?
        || rho.iter().any(|v| !v.is_finite())
        || tau.is_some_and(|v| v.len() != rho.len() || v.iter().any(|v| !v.is_finite()))
        || (definition.requires_tau && (tau.is_none() || c.core_tau.is_none()))
    {
        return Err(invalid(
            "periodic XC needs matching spin fields and actual kinetic densities for a meta-GGA",
        ));
    }
    let mut densities = ctx.allocate::<f64>(2 * n)?;
    let mut kinetic = ctx.allocate::<f64>(2 * n)?;
    for s in 0..2 {
        for k in 0..n {
            densities.as_mut_slice()[s * n + k] = (if c.spins == 1 {
                0.5 * rho[k]
            } else {
                rho[s * n + k]
            } + 0.5 * c.core.as_slice()[k])
                .max(0.);
            kinetic.as_mut_slice()[s * n + k] = tau
                .map_or(0., |t| {
                    if c.spins == 1 {
                        0.5 * t[k]
                    } else {
                        t[s * n + k]
                    }
                })
                .max(0.)
                + 0.5 * c.core_tau.as_ref().map_or(0., |t| t.as_slice()[k]);
        }
    }
    let mut gradients = ctx.allocate::<f64>(6 * n)?;
    let mut sigma = ctx.allocate::<f64>(3 * n)?;
    for s in 0..2 {
        for a in 0..3 {
            let d = derivative(c, fft, &densities.as_slice()[s * n..(s + 1) * n], a)?;
            gradients.as_mut_slice()[(3 * s + a) * n..(3 * s + a + 1) * n]
                .copy_from_slice(d.as_slice());
        }
    }
    for k in 0..n {
        if densities.as_slice()[k] + densities.as_slice()[n + k] > 1e-10 {
            for a in 0..3 {
                let u = gradients.as_slice()[a * n + k];
                let d = gradients.as_slice()[(3 + a) * n + k];
                sigma.as_mut_slice()[k] += u * u;
                sigma.as_mut_slice()[n + k] += u * d;
                sigma.as_mut_slice()[2 * n + k] += d * d;
            }
        }
    }
    let nonlocal = if let Some(b) = definition.nonlocal_b {
        let mut total = ctx.allocate::<f64>(n)?;
        let mut total_sigma = ctx.allocate::<f64>(n)?;
        for k in 0..n {
            total.as_mut_slice()[k] = densities.as_slice()[k] + densities.as_slice()[n + k];
            total_sigma.as_mut_slice()[k] =
                (sigma.as_slice()[k] + 2. * sigma.as_slice()[n + k] + sigma.as_slice()[2 * n + k])
                    .max(0.);
        }
        Some(nonlocal::evaluate(
            ctx,
            &c.grid,
            total.as_slice(),
            total_sigma.as_slice(),
            nonlocal::Options {
                b,
                ..Default::default()
            },
        )?)
    } else {
        None
    };
    let mut local = ctx.allocate::<f64>(2 * n)?;
    let mut vtau = definition
        .requires_tau
        .then(|| ctx.allocate::<f64>(c.spins * n))
        .transpose()?;
    let mut flux = ctx.allocate::<f64>(6 * n)?;
    let mut energy = nonlocal.as_ref().map_or(0., |t| t.energy);
    for k in 0..n {
        let (value, mut partial) = xc::point_partials(
            definition.semilocal,
            [
                densities.as_slice()[k],
                densities.as_slice()[n + k],
                sigma.as_slice()[k],
                sigma.as_slice()[n + k],
                sigma.as_slice()[2 * n + k],
                kinetic.as_slice()[k],
                kinetic.as_slice()[n + k],
            ],
            c.relativistic && matches!(definition.semilocal, "lda" | "pbe"),
        )?;
        energy += value * c.grid.index.dv();
        if let Some(nl) = &nonlocal {
            partial[0] += nl.density.as_slice()[k];
            partial[1] += nl.density.as_slice()[k];
            partial[2] += nl.sigma.as_slice()[k];
            partial[3] += 2. * nl.sigma.as_slice()[k];
            partial[4] += nl.sigma.as_slice()[k];
        }
        local.as_mut_slice()[k] = partial[0];
        local.as_mut_slice()[n + k] = partial[1];
        if let Some(v) = &mut vtau {
            for s in 0..c.spins {
                v.as_mut_slice()[s * n + k] = if tau.unwrap()[s * n + k] < 0. {
                    0.
                } else {
                    partial[5 + s]
                };
            }
        }
        if densities.as_slice()[k] + densities.as_slice()[n + k] > 1e-10 {
            for a in 0..3 {
                let u = gradients.as_slice()[a * n + k];
                let d = gradients.as_slice()[(3 + a) * n + k];
                flux.as_mut_slice()[a * n + k] = 2. * partial[2] * u + partial[3] * d;
                flux.as_mut_slice()[(3 + a) * n + k] = 2. * partial[4] * d + partial[3] * u;
            }
        }
    }
    for s in 0..2 {
        for a in 0..3 {
            let divergence = derivative(
                c,
                fft,
                &flux.as_slice()[(3 * s + a) * n..(3 * s + a + 1) * n],
                a,
            )?;
            for k in 0..n {
                local.as_mut_slice()[s * n + k] -= divergence.as_slice()[k];
            }
        }
    }
    let mut potential = ctx.allocate::<f64>(c.spins * n)?;
    for s in 0..c.spins {
        for k in 0..n {
            let raw = if c.spins == 1 {
                rho[k] + c.core.as_slice()[k]
            } else {
                rho[s * n + k] + 0.5 * c.core.as_slice()[k]
            };
            potential.as_mut_slice()[s * n + k] = if raw < 0. {
                0.
            } else {
                local.as_slice()[s * n + k]
            };
        }
    }
    Ok(Terms {
        energy,
        potential,
        kinetic_potential: vtau,
    })
}

fn orbital_gradient(
    c: &Crystal,
    data: &KPoint,
    axis: usize,
    symbols: &BlochSymbols,
    fft: &mut Fft3,
) -> Result<Buffer<C64>> {
    let n = c.grid.index.points();
    let mut gradient = c.context.allocate::<C64>(product(&[c.m, n])?)?;
    let mut phase = c.context.allocate::<C64>(n)?;
    let samples = data.psi.read(&c.context)?;
    for (k, value) in phase.as_mut_slice().iter_mut().enumerate() {
        let r = c.grid.point(k);
        *value = C64::from_polar(1., (0..3).map(|a| data.k[a] * r[a]).sum());
    }
    for a in 0..c.m {
        let row = &mut gradient.as_mut_slice()[a * n..(a + 1) * n];
        for (k, value) in row.iter_mut().enumerate() {
            *value = samples[a * n + k] * phase.as_slice()[k].conj();
        }
        fft.transform(row, false)?;
        for (k, value) in row.iter_mut().enumerate() {
            *value *= C64::new(0., symbols.vectors.as_slice()[axis * n + k]);
        }
        fft.transform(row, true)?;
        for (k, value) in row.iter_mut().enumerate() {
            *value *= phase.as_slice()[k];
        }
    }
    Ok(gradient)
}

pub fn kinetic_density<M: MatrixSource>(c: &Crystal, matrices: &[M]) -> Result<Buffer<f64>> {
    if matrices.len() != c.spins * c.points.len() {
        return Err(invalid(
            "kinetic density needs a matrix for each spin and k-point",
        ));
    }
    let n = c.grid.index.points();
    let mut raw = c.context.allocate::<f64>(c.spins * n)?;
    let mut fft = Fft3::new(&c.context, c.grid.index.shape())?;
    for (k, data) in c.points.iter().enumerate() {
        let symbols = reciprocal::bloch(&c.context, &c.grid, data.k)?;
        for axis in 0..3 {
            let gradient = orbital_gradient(c, data, axis, &symbols, &mut fft)?;
            for s in 0..c.spins {
                let matrix = matrices[s * c.points.len() + k].matrix(&c.context)?;
                let density = hermitian_density(&c.context, gradient.as_slice(), &matrix, n)?;
                for i in 0..n {
                    raw.as_mut_slice()[s * n + i] += 0.5 * data.weight * density.as_slice()[i];
                }
            }
        }
    }
    let mut result = c.context.allocate::<f64>(c.spins * n)?;
    for s in 0..c.spins {
        sym_field(
            c,
            &raw.as_slice()[s * n..(s + 1) * n],
            &mut result.as_mut_slice()[s * n..(s + 1) * n],
        );
    }
    for value in result.as_mut_slice() {
        *value = value.max(0.);
    }
    Ok(result)
}

pub fn kinetic_operator(c: &Crystal, data: &KPoint, potential: &[f64]) -> Result<Buffer<C64>> {
    let n = c.grid.index.points();
    if potential.len() != n {
        return Err(invalid("kinetic potential must match the periodic grid"));
    }
    let mut result = c.context.allocate::<C64>(product(&[c.m, c.m])?)?;
    let mut field = c.context.allocate::<f64>(n)?;
    for (to, from) in field.as_mut_slice().iter_mut().zip(potential) {
        *to = 0.5 * c.grid.index.dv() * from;
    }
    let symbols = reciprocal::bloch(&c.context, &c.grid, data.k)?;
    let mut fft = Fft3::new(&c.context, c.grid.index.shape())?;
    for axis in 0..3 {
        let gradient = orbital_gradient(c, data, axis, &symbols, &mut fft)?;
        let mut projector = ComplexProjector::new(&c.context, gradient.as_slice(), c.m, n)?;
        let matrix = projector.project(field.as_slice())?;
        for (to, from) in result.as_mut_slice().iter_mut().zip(matrix) {
            *to += from;
        }
    }
    Ok(result)
}
