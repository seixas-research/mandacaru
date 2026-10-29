//! Two-component Bloch states with a vector magnetization and separable SOC.
//! Spin-major ordering: alpha orbitals followed by beta orbitals.
use super::{invalid, solver, Crystal, KPoint};
use crate::{
    math::{
        fft::Fft3,
        linalg::eigen,
        projection::{hermitian_density, ComplexProjector},
        storage::MatrixSource,
    },
    quantum::C64,
    Buffer, Result,
};
use nalgebra::DMatrix;
type Matrix = DMatrix<C64>;

/// Rotate the local density into its eigenframe, differentiate the collinear
/// functional there, and rotate the functional derivative back. In a GGA the
/// derivatives are taken on the eigenvalue fields before this final rotation.
pub fn exchange_correlation(
    c: &Crystal,
    fft: &mut Fft3,
    fields: &[f64],
) -> Result<super::fields::Terms> {
    if c.time_reversal {
        return solver::collinear_xc(c, fft, fields, None, 1);
    }
    let n = c.grid.index.points();
    if fields.len() != 4 * n || fields.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "spinor XC needs charge and three finite magnetization fields",
        ));
    }
    let mut rotated = c.context.allocate::<f64>(2 * n)?;
    let mut norms = c.context.allocate::<f64>(n)?;
    for k in 0..n {
        let norm = (1..4)
            .map(|a| fields[a * n + k].powi(2))
            .sum::<f64>()
            .sqrt();
        norms.as_mut_slice()[k] = norm;
        rotated.as_mut_slice()[k] = 0.5 * (fields[k] + norm);
        rotated.as_mut_slice()[n + k] = 0.5 * (fields[k] - norm);
    }
    let terms = solver::collinear_xc(c, fft, rotated.as_slice(), None, 2)?;
    let mut potential = c.context.allocate::<f64>(4 * n)?;
    for k in 0..n {
        let up = terms.potential.as_slice()[k];
        let down = terms.potential.as_slice()[n + k];
        potential.as_mut_slice()[k] = 0.5 * (up + down);
        if norms.as_slice()[k] > 1e-14 {
            let radial = 0.5 * (up - down) / norms.as_slice()[k];
            for a in 1..4 {
                potential.as_mut_slice()[a * n + k] = radial * fields[a * n + k];
            }
        }
    }
    Ok(super::fields::Terms {
        energy: terms.energy,
        potential,
        kinetic_potential: None,
    })
}

pub fn diagonalize(
    c: &Crystal,
    point: &KPoint,
    fields: &[f64],
    moments: &[C64],
) -> Result<(Vec<f64>, Matrix, Matrix)> {
    let m = c.m;
    let n = c.grid.index.points();
    if fields.len() != c.field_channels() * n
        || moments.len() != c.channels
        || point.spin_orbit.is_none()
    {
        return Err(invalid(
            "spinor Hamiltonian needs four local potential fields and all moment potentials",
        ));
    }
    let operator = Matrix::from_fn(c.p, c.p, |i, j| {
        (0..c.channels)
            .map(|a| moments[a] * c.blocks.as_slice()[(a * c.p + i) * c.p + j])
            .sum()
    });
    let fixed = &point.fixed + &point.projections * operator * point.projections.adjoint();
    let mut h = Matrix::zeros(2 * m, 2 * m);
    for s in 0..2 {
        h.view_mut((s * m, s * m), (m, m)).copy_from(&fixed);
    }
    if let Some(soc) = &point.spin_orbit {
        h += soc;
    }
    let samples = point.psi.read(&c.context)?;
    let mut projector = ComplexProjector::new(&c.context, &samples, m, n)?;
    let mut scaled = c.context.allocate::<f64>(n)?;
    // Each projected Pauli component acts on its corresponding spin blocks.
    for a in 0..c.field_channels() {
        for k in 0..n {
            scaled.as_mut_slice()[k] = fields[a * n + k] * c.grid.index.dv();
        }
        let values = projector.project(scaled.as_slice())?;
        for i in 0..m {
            for j in 0..m {
                let value = values[i + j * m];
                match a {
                    0 => {
                        h[(i, j)] += value;
                        h[(m + i, m + j)] += value;
                    }
                    1 => {
                        h[(i, m + j)] += value;
                        h[(m + i, j)] += value;
                    }
                    2 => {
                        h[(i, m + j)] -= C64::i() * value;
                        h[(m + i, j)] += C64::i() * value;
                    }
                    _ => {
                        h[(i, j)] += value;
                        h[(m + i, m + j)] -= value;
                    }
                }
            }
        }
    }
    drop(projector);
    drop(samples);
    let lowdin = Matrix::from_fn(2 * m, 2 * m, |i, j| {
        if i / m == j / m {
            point.lowdin[(i % m, j % m)]
        } else {
            C64::default()
        }
    });
    let h = (&h + h.adjoint()) * C64::new(0.5, 0.);
    let (eps, y) = eigen(lowdin.adjoint() * h * &lowdin)?;
    Ok((eps, &lowdin * &y, y))
}

/// Spatial Hermitian matrices for (n, mx, my, mz). This retains transverse
/// coherences; dropping the off-diagonal spin blocks would discard SOC.
fn component(d: &Matrix, m: usize, a: usize) -> Matrix {
    Matrix::from_fn(m, m, |i, j| match a {
        0 => d[(i, j)] + d[(m + i, m + j)],
        1 => d[(i, m + j)] + d[(m + i, j)],
        2 => C64::i() * (d[(i, m + j)] - d[(m + i, j)]),
        _ => d[(i, j)] - d[(m + i, m + j)],
    })
}
pub fn density<M: MatrixSource>(c: &Crystal, matrices: &[M]) -> Result<(Buffer<f64>, Buffer<C64>)> {
    if matrices.len() != c.points.len() {
        return Err(invalid(
            "spinor density matrices must match the Bloch basis and k-point mesh",
        ));
    }
    let n = c.grid.index.points();
    let p = c.p;
    let mut rho = c.context.allocate::<f64>(c.field_channels() * n)?;
    let mut projected = Matrix::zeros(p, p);
    for (point, d) in c.points.iter().zip(matrices) {
        let d = d.matrix(&c.context)?;
        if d.shape() != (2 * c.m, 2 * c.m) {
            return Err(invalid("spinor matrix dimensions disagree with the basis"));
        }
        let samples = point.psi.read(&c.context)?;
        for a in 0..c.field_channels() {
            let d = component(&d, c.m, a);
            if a == 0 {
                projected += point.projections.adjoint()
                    * &d
                    * &point.projections
                    * C64::new(point.weight, 0.);
            }
            let field = hermitian_density(&c.context, &samples, &d, n)?;
            for k in 0..n {
                rho.as_mut_slice()[a * n + k] += point.weight * field.as_slice()[k];
            }
        }
    }
    let mut q = c.context.allocate::<C64>(c.channels)?;
    for a in 0..c.channels {
        q.as_mut_slice()[a] = (0..p)
            .flat_map(|i| (0..p).map(move |j| (i, j)))
            .map(|(i, j)| projected[(i, j)] * c.blocks.as_slice()[(a * p + j) * p + i])
            .sum();
    }
    if c.time_reversal {
        let mut charge = c.context.allocate::<f64>(n)?;
        solver::sym_field(c, rho.as_slice(), charge.as_mut_slice());
        rho = charge;
        let mut moments = c.context.allocate::<C64>(c.channels)?;
        for a in 0..c.channels {
            moments.as_mut_slice()[a] = (0..c.channels)
                .map(|b| c.qsym.as_slice()[a * c.channels + b] * q.as_slice()[b])
                .sum();
        }
        q = moments;
    }
    Ok((rho, q))
}
pub fn fixed_energy(point: &KPoint, d: &Matrix, m: usize) -> f64 {
    (component(d, m, 0) * &point.fixed).trace().re
        + point
            .spin_orbit
            .as_ref()
            .map_or(0., |soc| (d * soc).trace().re)
}
pub fn spin_orbit_energy<M: MatrixSource>(c: &Crystal, matrices: &[M]) -> Result<f64> {
    c.points
        .iter()
        .zip(matrices)
        .map(|(point, d)| {
            let d = d.matrix(&c.context)?;
            Ok(point.weight
                * point
                    .spin_orbit
                    .as_ref()
                    .map_or(0., |soc| (&*d * soc).trace().re))
        })
        .sum()
}
/// Total moment uses the PAW overlap metric, including augmentation charge.
pub fn magnetic_moment<M: MatrixSource>(c: &Crystal, matrices: &[M]) -> Result<f64> {
    if c.time_reversal {
        return Ok(0.);
    }
    let squared = (1..4)
        .map(|a| {
            c.points
                .iter()
                .zip(matrices)
                .map(|(point, d)| {
                    let d = d.matrix(&c.context)?;
                    Ok(point.weight * (component(&d, c.m, a) * &point.overlap).trace().re)
                })
                .sum::<Result<f64>>()
                .map(|v| v.powi(2))
        })
        .sum::<Result<f64>>()?;
    Ok(squared.sqrt())
}
