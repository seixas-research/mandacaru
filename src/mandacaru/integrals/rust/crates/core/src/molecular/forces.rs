//! Stationary molecular gradients. No displaced SCF or Python AD callback.
use super::{
    dft,
    fft::{spectral_laplacian, Poisson},
    invalid, xc, MeanField, MolecularIntegrals,
};
use crate::{product, quantum::C64, Buffer, Result};
use nalgebra::DMatrix;

pub struct Gradient {
    pub hellmann_feynman: Buffer<f64>,
    pub pulay: Buffer<f64>,
    pub total: Buffer<f64>,
}
impl MolecularIntegrals {
    pub fn gradient(&self, state: &MeanField) -> Result<Gradient> {
        if !state.converged {
            return Err(invalid("forces require a converged SCF state"));
        }
        if state.method == "ghf" {
            return Err(invalid("native spinor gradients are not implemented"));
        }
        if self.kinetic_method != "spectral" {
            return Err(invalid(
                "native stationary gradients currently require spectral kinetic energy",
            ));
        }
        if state.functional.is_none() && matches!(self.eri, super::integrals::Eri::Cholesky { .. })
        {
            return Err(invalid("native HF gradients require packed integrals; Cholesky derivatives are not implemented"));
        }
        let m = self.m();
        let channels = if state.method == "rhf" { 1 } else { 2 };
        if state.densities.len() != channels
            || state.coefficients.len() != channels
            || state.energies.len() != channels
            || state.particles.iter().any(|&n| n > m)
            || state
                .densities
                .iter()
                .chain(&state.coefficients)
                .any(|d| d.shape() != (m, m))
            || state.energies.iter().any(|e| e.len() != m)
        {
            return Err(invalid(
                "SCF state dimensions do not match molecular integrals",
            ));
        }
        self.context.run(|| {
            let ctx = &self.context;
            let m = self.m();
            let n = self.grid.index.points();
            let dv = self.grid.index.dv();
            let atoms = self.geometry.numbers().len();
            let _matrix_work = ctx.reserve(product(&[m, m, 8, 16])?)?;
            let mut hf = ctx.allocate::<f64>(3 * atoms)?;
            let mut pulay = ctx.allocate::<f64>(3 * atoms)?;
            let mut total_gradient = ctx.allocate::<f64>(3 * atoms)?;
            let rho = dft::grid_densities(self, &state.densities)?;
            let mut source = ctx.allocate::<C64>(n)?;
            let mut potential = ctx.allocate::<C64>(n)?;
            for k in 0..n {
                source.as_mut_slice()[k] = C64::new(rho.as_slice()[k] + rho.as_slice()[n + k], 0.);
            }
            let mut poisson = Poisson::isolated(ctx, &self.grid)?;
            poisson.solve_into(source.as_slice(), potential.as_mut_slice())?;
            let xc = state
                .functional
                .as_ref()
                .map(|name| xc::evaluate(ctx, &self.grid, rho.as_slice(), name))
                .transpose()?;
            let mut external = ctx.allocate::<f64>(n)?;
            let softening = 0.5
                * self
                    .grid
                    .spacings()
                    .into_iter()
                    .fold(f64::INFINITY, f64::min);
            for a in 0..atoms {
                let position = self.geometry.positions()[a];
                let z = self.geometry.numbers()[a] as f64;
                for k in 0..n {
                    let point = self.grid.point(k);
                    let delta = std::array::from_fn::<_, 3, _>(|i| point[i] - position[i]);
                    let r = delta.iter().map(|x| x * x).sum::<f64>().sqrt();
                    external.as_mut_slice()[k] -= z / r.max(softening);
                    if r > softening {
                        for (axis, delta) in delta.iter().enumerate() {
                            hf.as_mut_slice()[3 * a + axis] -=
                                z * delta / r.powi(3) * source.as_slice()[k].re * dv;
                        }
                    }
                }
                for b in 0..a {
                    let delta = std::array::from_fn::<_, 3, _>(|i| {
                        position[i] - self.geometry.positions()[b][i]
                    });
                    let r = delta.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let factor = z * self.geometry.numbers()[b] as f64 / r.powi(3);
                    for (axis, delta) in delta.iter().enumerate() {
                        hf.as_mut_slice()[3 * a + axis] -= factor * delta;
                        hf.as_mut_slice()[3 * b + axis] += factor * delta;
                    }
                }
            }
            let mut dphi = ctx.allocate::<f64>(product(&[m, 3, n])?)?;
            for a in 0..m {
                for k in 0..n {
                    let gradient = self.basis.orbitals[a].gradient(self.grid.point(k));
                    for (axis, component) in gradient.iter().enumerate() {
                        dphi.as_mut_slice()[(a * 3 + axis) * n + k] = -component;
                    }
                }
            }
            for spin in 0..2 {
                let index = if state.method == "rhf" { 0 } else { spin };
                let occupied = state.particles[spin];
                let coefficients = &state.coefficients[index];
                if coefficients.iter().any(|v| v.im.abs() > 1e-12) {
                    return Err(invalid(
                        "native stationary gradients currently require real collinear orbitals",
                    ));
                }
                let ao =
                    &self.lowdin * DMatrix::from_fn(m, occupied, |a, i| coefficients[(a, i)].re);
                let mut orbitals = ctx.allocate::<f64>(product(&[occupied, n])?)?;
                for i in 0..occupied {
                    for a in 0..m {
                        for k in 0..n {
                            orbitals.as_mut_slice()[i * n + k] +=
                                ao[(a, i)] * self.samples.as_slice()[a * n + k];
                        }
                    }
                }
                let mut exchange = ctx.allocate::<C64>(n)?;
                for i in 0..occupied {
                    let psi = &orbitals.as_slice()[i * n..(i + 1) * n];
                    let lap = spectral_laplacian(ctx, &self.grid, psi)?;
                    let mut action = ctx.allocate::<f64>(n)?;
                    for (k, psi) in psi.iter().enumerate() {
                        action.as_mut_slice()[k] = -0.5 * lap.as_slice()[k]
                            + (external.as_slice()[k]
                                + potential.as_slice()[k].re
                                + xc.as_ref()
                                    .map_or(0., |v| v.potentials.as_slice()[spin * n + k]))
                                * psi;
                    }
                    if xc.is_none() {
                        for j in 0..occupied {
                            for (k, psi) in psi.iter().enumerate() {
                                source.as_mut_slice()[k] =
                                    C64::new(psi * orbitals.as_slice()[j * n + k], 0.);
                            }
                            poisson.solve_into(source.as_slice(), exchange.as_mut_slice())?;
                            for k in 0..n {
                                action.as_mut_slice()[k] -=
                                    orbitals.as_slice()[j * n + k] * exchange.as_slice()[k].re;
                            }
                        }
                    }
                    let eps = state.energies[index][i];
                    for a in 0..m {
                        for axis in 0..3 {
                            let value = (0..n)
                                .map(|k| {
                                    dphi.as_slice()[(a * 3 + axis) * n + k]
                                        * (action.as_slice()[k] - eps * psi[k])
                                })
                                .sum::<f64>()
                                * 2.
                                * ao[(a, i)]
                                * dv;
                            pulay.as_mut_slice()[3 * self.basis.orbitals[a].atom + axis] += value;
                        }
                    }
                }
            }
            for i in 0..3 * atoms {
                total_gradient.as_mut_slice()[i] = hf.as_slice()[i] + pulay.as_slice()[i];
            }
            if total_gradient.as_slice().iter().any(|v| !v.is_finite()) {
                return Err(invalid("nonfinite stationary gradient"));
            }
            Ok(Gradient {
                hellmann_feynman: hf,
                pulay,
                total: total_gradient,
            })
        })
    }
}
