//! Jellium-convention Ewald sums with analytic forces and strain derivatives.
use super::invalid;
use crate::{finite, product, quantum::C64, Buffer, Context, Reservation, Result};
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;
struct Vectors {
    values: Vec<Vector3<f64>>,
    _memory: Reservation,
}
fn vectors(ctx: &Context, a: &Matrix3<f64>, cutoff: f64) -> Result<Vectors> {
    let b = a
        .try_inverse()
        .ok_or_else(|| invalid("singular Ewald lattice"))?
        .transpose();
    let mut limits = [0i32; 3];
    for (i, limit) in limits.iter_mut().enumerate() {
        let count = (cutoff * b.column(i).norm()).ceil() + 1.;
        if count > i32::MAX as f64 / 2. {
            return Err(crate::Error::Overflow);
        }
        *limit = count as i32;
    }
    let n = product(&limits.map(|n| 2 * n as usize + 1))?;
    let memory = ctx.reserve(product(&[n, 24])?)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(n)
        .map_err(|_| crate::Error::Allocation)?;
    for i in -limits[0]..=limits[0] {
        for j in -limits[1]..=limits[1] {
            for k in -limits[2]..=limits[2] {
                values.push(a * Vector3::new(i as f64, j as f64, k as f64));
            }
        }
    }
    Ok(Vectors {
        values,
        _memory: memory,
    })
}
pub struct Ewald {
    positions: Vec<Vector3<f64>>,
    charges: Vec<f64>,
    eta: f64,
    volume: f64,
    real: Vectors,
    reciprocal: Vectors,
    rcut: f64,
    gcut: f64,
    a: Matrix3<f64>,
    _memory: Reservation,
}
pub struct EwaldResult {
    pub energy: f64,
    pub forces: Buffer<f64>,
    pub stress: [f64; 9],
}
impl Ewald {
    pub fn new(
        ctx: &Context,
        positions: &[[f64; 3]],
        charges: &[f64],
        cell: [f64; 9],
        accuracy: f64,
        eta: Option<f64>,
    ) -> Result<Self> {
        finite(&cell, "Ewald cell")?;
        finite(charges, "Ewald charges")?;
        for r in positions {
            finite(r, "Ewald positions")?;
        }
        if positions.len() != charges.len()
            || !accuracy.is_finite()
            || accuracy <= 0.
            || accuracy >= 1.
        {
            return Err(invalid("invalid Ewald dimensions or accuracy"));
        }
        let a = Matrix3::from_row_slice(&cell).transpose();
        let volume = a.determinant().abs();
        let inverse = a
            .try_inverse()
            .ok_or_else(|| invalid("Ewald cell must be nonsingular"))?;
        if !volume.is_finite() || volume <= 0. {
            return Err(invalid("invalid Ewald cell volume"));
        }
        let eta = eta
            .unwrap_or(PI.sqrt() * (charges.len().max(1) as f64 / (volume * volume)).powf(1. / 6.));
        if !eta.is_finite() || eta <= 0. {
            return Err(invalid("Ewald eta must be finite and positive"));
        }
        let scale = charges
            .iter()
            .map(|q| q * q)
            .fold(0f64, f64::max)
            .max(f64::MIN_POSITIVE);
        let tolerance = accuracy
            / if scale < f64::MIN_POSITIVE * 2. {
                1.
            } else {
                scale
            };
        let mut rcut = 1. / eta;
        let mut gcut = 2. * eta;
        for _ in 0..200 {
            if libm::erfc(eta * rcut) / rcut < tolerance {
                break;
            }
            rcut *= 1.25;
        }
        for _ in 0..200 {
            if (-gcut * gcut / (4. * eta * eta)).exp() / (gcut * gcut) < tolerance * volume {
                break;
            }
            gcut *= 1.25;
        }
        let memory = ctx.reserve(product(&[positions.len(), 32])?)?;
        let positions = positions
            .iter()
            .map(|r| {
                let f = inverse * Vector3::from(*r);
                a * f.map(|v| v - v.floor())
            })
            .collect();
        Ok(Self {
            positions,
            charges: charges.to_vec(),
            eta,
            volume,
            real: vectors(ctx, &a, rcut)?,
            reciprocal: vectors(ctx, &(inverse.transpose() * (2. * PI)), gcut)?,
            rcut,
            gcut,
            a,
            _memory: memory,
        })
    }
    pub fn evaluate(&self, ctx: &Context) -> Result<EwaldResult> {
        let mut energy = 0.;
        let mut forces = ctx.allocate::<f64>(3 * self.charges.len())?;
        let mut stress = Matrix3::<f64>::zeros();
        let eta = self.eta;
        for (i, (ri, &qi)) in self.positions.iter().zip(&self.charges).enumerate() {
            for (j, (rj, &qj)) in self.positions.iter().zip(&self.charges).enumerate() {
                for shift in &self.real.values {
                    let delta = ri - rj + shift;
                    let r = delta.norm();
                    if r <= 1e-12 {
                        if i != j {
                            return Err(invalid(
                                "Ewald charges must not coincide modulo the lattice",
                            ));
                        }
                        continue;
                    }
                    if r >= self.rcut {
                        continue;
                    }
                    let erfc = libm::erfc(eta * r);
                    energy += 0.5 * qi * qj * erfc / r;
                    let weight = qi
                        * qj
                        * (erfc / r.powi(3)
                            + 2. * eta / PI.sqrt() * (-(eta * r).powi(2)).exp() / (r * r));
                    for axis in 0..3 {
                        forces.as_mut_slice()[3 * i + axis] += weight * delta[axis];
                    }
                    stress -= delta * delta.transpose() * (0.5 * weight);
                }
            }
        }
        for g in &self.reciprocal.values {
            let g2 = g.norm_squared();
            if g2 <= 1e-12 || g2 >= self.gcut * self.gcut {
                continue;
            }
            let structure = self
                .positions
                .iter()
                .zip(&self.charges)
                .map(|(r, q)| C64::from_polar(*q, g.dot(r)))
                .sum::<C64>();
            let weight = (-g2 / (4. * eta * eta)).exp() / g2;
            let term = 2. * PI / self.volume * weight * structure.norm_sqr();
            energy += term;
            stress += ((g * g.transpose()) * (2. * (1. / g2 + 1. / (4. * eta * eta)))
                - Matrix3::identity())
                * term;
            for (i, (r, &q)) in self.positions.iter().zip(&self.charges).enumerate() {
                let f = 4. * PI / self.volume
                    * q
                    * weight
                    * (C64::from_polar(1., g.dot(r)) * structure.conj()).im;
                for axis in 0..3 {
                    forces.as_mut_slice()[3 * i + axis] += f * g[axis];
                }
            }
        }
        let background =
            PI / (2. * eta * eta * self.volume) * self.charges.iter().sum::<f64>().powi(2);
        energy -= eta / PI.sqrt() * self.charges.iter().map(|q| q * q).sum::<f64>() + background;
        stress = (stress + Matrix3::identity() * background) / self.volume;
        let values = std::array::from_fn(|i| stress[(i / 3, i % 3)]);
        finite(&[energy], "Ewald energy")?;
        finite(forces.as_slice(), "Ewald forces")?;
        finite(&values, "Ewald stress")?;
        Ok(EwaldResult {
            energy,
            forces,
            stress: values,
        })
    }
    pub fn potential(
        &self,
        ctx: &Context,
        points: &[[f64; 3]],
        softening: f64,
    ) -> Result<Buffer<f64>> {
        if !softening.is_finite() || softening < 0. {
            return Err(invalid("invalid Ewald softening"));
        }
        let mut out = ctx.allocate::<f64>(points.len())?;
        let inverse = self.a.try_inverse().unwrap();
        for (i, point) in points.iter().enumerate() {
            finite(point, "Ewald points")?;
            let f = inverse * Vector3::from(*point);
            let point = self.a * f.map(|v| v - v.floor());
            let mut value = 0.;
            for (r, q) in self.positions.iter().zip(&self.charges) {
                for shift in &self.real.values {
                    let r = (point - r - shift).norm();
                    if r < self.rcut {
                        let r = r.max(softening);
                        if r <= 1e-12 {
                            return Err(invalid(
                                "Ewald potential diverges at a charge; supply positive softening",
                            ));
                        }
                        value += q * libm::erfc(self.eta * r) / r;
                    }
                }
            }
            for g in &self.reciprocal.values {
                let g2 = g.norm_squared();
                if g2 <= 1e-12 || g2 >= self.gcut * self.gcut {
                    continue;
                }
                let structure = self
                    .positions
                    .iter()
                    .zip(&self.charges)
                    .map(|(r, q)| C64::from_polar(*q, -g.dot(r)))
                    .sum::<C64>();
                value += 4. * PI / self.volume * (-g2 / (4. * self.eta * self.eta)).exp() / g2
                    * (C64::from_polar(1., g.dot(&point)) * structure).re;
            }
            out.as_mut_slice()[i] =
                value - PI * self.charges.iter().sum::<f64>() / (self.eta * self.eta * self.volume);
        }
        finite(out.as_slice(), "Ewald potential")?;
        Ok(out)
    }
}
