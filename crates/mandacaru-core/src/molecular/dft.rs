//! Restricted and collinear unrestricted Kohn–Sham with native XC and Poisson.
use super::{
    invalid,
    scf::{self, Candidate, Diis, Iteration},
    MeanField, MolecularIntegrals, ScfOptions,
};
use crate::math::fft::Poisson;
use crate::models::xc;
use crate::{product, quantum::C64, Buffer, Result};
use nalgebra::DMatrix;
type Matrix = DMatrix<C64>;

pub(super) fn grid_densities(integrals: &MolecularIntegrals, d: &[Matrix]) -> Result<Buffer<f64>> {
    let m = integrals.m();
    let n = integrals.grid.index.points();
    let mut rho = integrals.context.allocate::<f64>(2 * n)?;
    let phi = integrals.orthonormal_samples.as_slice();
    for s in 0..2 {
        let density = &d[if d.len() == 1 { 0 } else { s }];
        for k in 0..n {
            let value = (0..m)
                .map(|a| {
                    (0..m)
                        .map(|b| density[(a, b)].re * phi[a * n + k] * phi[b * n + k])
                        .sum::<f64>()
                })
                .sum::<f64>();
            // Roundoff in a positive semidefinite density may be slightly negative.
            rho.as_mut_slice()[s * n + k] = value.max(0.);
        }
    }
    Ok(rho)
}
struct Evaluation {
    f: Vec<Matrix>,
    energy: f64,
    hartree: f64,
    xc: f64,
}
pub(super) fn orbital_gradients(integrals: &MolecularIntegrals) -> Result<Buffer<f64>> {
    let m = integrals.m();
    let n = integrals.grid.index.points();
    let mut gradients = integrals.context.allocate::<f64>(product(&[3, m, n])?)?;
    for axis in 0..3 {
        for a in 0..m {
            xc::derivative(
                &integrals.grid,
                &integrals.orthonormal_samples.as_slice()[a * n..(a + 1) * n],
                axis,
                &mut gradients.as_mut_slice()[(axis * m + a) * n..(axis * m + a + 1) * n],
            )?;
        }
    }
    Ok(gradients)
}
pub(super) fn grid_kinetic(
    integrals: &MolecularIntegrals,
    d: &[Matrix],
    gradients: &[f64],
) -> Result<Buffer<f64>> {
    let m = integrals.m();
    let n = integrals.grid.index.points();
    if gradients.len() != product(&[3, m, n])? {
        return Err(invalid("orbital gradients have inconsistent dimensions"));
    }
    let mut tau = integrals.context.allocate::<f64>(2 * n)?;
    for spin in 0..2 {
        let density = &d[if d.len() == 1 { 0 } else { spin }];
        for k in 0..n {
            let mut value = 0.;
            for axis in 0..3 {
                for a in 0..m {
                    for b in 0..m {
                        value += 0.5
                            * density[(a, b)].re
                            * gradients[(axis * m + a) * n + k]
                            * gradients[(axis * m + b) * n + k];
                    }
                }
            }
            tau.as_mut_slice()[spin * n + k] = value.max(0.);
        }
    }
    Ok(tau)
}
fn evaluate(
    integrals: &MolecularIntegrals,
    h: &Matrix,
    d: &[Matrix],
    name: &str,
    poisson: &mut Poisson,
    orbital_gradients: Option<&[f64]>,
    screened: Option<&super::integrals::Eri>,
) -> Result<Evaluation> {
    let ctx = &integrals.context;
    let m = integrals.m();
    let n = integrals.grid.index.points();
    let dv = integrals.grid.index.dv();
    let rho = grid_densities(integrals, d)?;
    let mut total = ctx.allocate::<C64>(n)?;
    for k in 0..n {
        total.as_mut_slice()[k] = C64::new(rho.as_slice()[k] + rho.as_slice()[n + k], 0.);
    }
    let mut hartree = ctx.allocate::<C64>(n)?;
    poisson.solve_into(total.as_slice(), hartree.as_mut_slice())?;
    let kinetic = orbital_gradients
        .map(|gradient| grid_kinetic(integrals, d, gradient))
        .transpose()?;
    let terms = xc::evaluate_fields(
        ctx,
        &integrals.grid,
        rho.as_slice(),
        kinetic.as_ref().map(|v| v.as_slice()),
        name,
    )?;
    let eh = 0.5
        * total
            .as_slice()
            .iter()
            .zip(hartree.as_slice())
            .map(|(r, v)| (r * v).re)
            .sum::<f64>()
        * dv;
    let mut f = vec![];
    let mut exact_energy = 0.;
    let phi = integrals.orthonormal_samples.as_slice();
    for s in 0..d.len() {
        let v = Matrix::from_fn(m, m, |a, b| {
            C64::new(
                (0..n)
                    .map(|k| {
                        let local = phi[a * n + k]
                            * phi[b * n + k]
                            * (hartree.as_slice()[k].re + terms.potentials.as_slice()[s * n + k]);
                        let meta = if let (Some(gradient), Some(tau)) =
                            (orbital_gradients, terms.tau_potentials.as_ref())
                        {
                            0.5 * tau.as_slice()[s * n + k]
                                * (0..3)
                                    .map(|axis| {
                                        gradient[(axis * m + a) * n + k]
                                            * gradient[(axis * m + b) * n + k]
                                    })
                                    .sum::<f64>()
                        } else {
                            0.
                        };
                        local + meta
                    })
                    .sum::<f64>()
                    * dv,
                0.,
            )
        });
        let mut fock = h + v;
        if let Some(eri) = screened {
            let (_, exchange) = scf::jk(eri, m, &d[s]);
            exact_energy -=
                0.125 * (d[s].clone() * &exchange).trace().re * if d.len() == 1 { 2. } else { 1. };
            fock -= exchange * C64::new(0.25, 0.);
        }
        f.push(scf::hermitian(fock));
    }
    let eone =
        d.iter().map(|d| (d * h).trace().re).sum::<f64>() * if d.len() == 1 { 2. } else { 1. };
    Ok(Evaluation {
        f,
        energy: eone + eh + terms.energy + exact_energy,
        hartree: eh,
        xc: terms.energy + exact_energy,
    })
}
impl MolecularIntegrals {
    pub fn kohn_sham(
        &self,
        functional: &str,
        particles: [usize; 2],
        unrestricted: bool,
        options: &ScfOptions,
    ) -> Result<MeanField> {
        let definition = crate::models::functional(functional)?;
        self.context.run(|| {
            let m = self.m();
            let ctx = &self.context;
            let screened = if let Some((omega, _)) = definition.screened_exchange {
                let mut solver = Poisson::screened(ctx, &self.grid, omega)?;
                Some(super::integrals::cholesky_from_grid(
                    ctx,
                    &self.grid,
                    self.orthonormal_samples.as_slice(),
                    m,
                    1e-12,
                    &mut solver,
                )?)
            } else {
                None
            };
            let orbital_gradients = if definition.requires_tau {
                Some(orbital_gradients(self)?)
            } else {
                None
            };
            if particles.iter().any(|&v| v > m)
                || (!unrestricted && particles[0] != particles[1])
                || options.max_iterations == 0
                || options.guesses == 0
                || !options.tolerance.is_finite()
                || options.tolerance <= 0.
            {
                return Err(invalid(
                    "invalid Kohn-Sham occupation, iteration budget or tolerance",
                ));
            }
            let workspace = ctx.reserve(
                product(&[2 * m, 2 * m, 16, 160])?
                    .checked_add(product(&[options.max_iterations, 128, 4])?)
                    .ok_or(crate::Error::Overflow)?,
            )?;
            let h = Matrix::from_fn(m, m, |a, b| C64::new(self.one.as_slice()[a * m + b], 0.));
            let (_, core) = crate::math::linalg::eigen(h.clone())?;
            let mut starts = vec![if unrestricted {
                vec![core.clone(), core.clone()]
            } else {
                vec![core.clone()]
            }];
            let occupations = if unrestricted {
                particles.to_vec()
            } else {
                vec![particles[0]]
            };
            let mut poisson = Poisson::isolated(ctx, &self.grid)?;
            if options.guesses >= 2 {
                let d = starts[0]
                    .iter()
                    .zip(&occupations)
                    .map(|(c, &n)| scf::density(c, n))
                    .collect::<Vec<_>>();
                let f = evaluate(
                    self,
                    &h,
                    &d,
                    functional,
                    &mut poisson,
                    orbital_gradients.as_ref().map(|v| v.as_slice()),
                    screened.as_ref(),
                )?
                .f;
                starts.push(
                    f.into_iter()
                        .map(|f| {
                            crate::math::linalg::eigen((&h + f) * C64::new(0.5, 0.)).map(|(_, c)| c)
                        })
                        .collect::<Result<_>>()?,
                );
            }
            if unrestricted
                && options.guesses >= 3
                && particles[0] == particles[1]
                && particles[0] > 0
                && particles[0] < m
            {
                let i = particles[0] - 1;
                let a = particles[0];
                let mut pair = vec![core.clone(), core.clone()];
                for (spin, c) in pair.iter_mut().enumerate() {
                    let theta = if spin == 0 { 0.3f64 } else { -0.3f64 };
                    for k in 0..m {
                        c[(k, i)] = core[(k, i)] * theta.cos() + core[(k, a)] * theta.sin();
                        c[(k, a)] = -core[(k, i)] * theta.sin() + core[(k, a)] * theta.cos();
                    }
                }
                starts.push(pair);
            }
            let mut best: Option<(Candidate, f64, f64)> = None;
            for mut c in starts {
                let mut d = c
                    .iter()
                    .zip(&occupations)
                    .map(|(c, &n)| scf::density(c, n))
                    .collect::<Vec<_>>();
                let mut previous = None;
                let mut history = vec![];
                let mut mixer = Diis::new();
                let mut converged = false;
                for iteration in 1..=options.max_iterations {
                    let value = evaluate(
                        self,
                        &h,
                        &d,
                        functional,
                        &mut poisson,
                        orbital_gradients.as_ref().map(|v| v.as_slice()),
                        screened.as_ref(),
                    )?;
                    let effective = if options.diis {
                        mixer.extrapolate(&value.f, &d)
                    } else {
                        value.f
                    };
                    c = effective
                        .into_iter()
                        .map(|f| crate::math::linalg::eigen(f).map(|(_, c)| c))
                        .collect::<Result<_>>()?;
                    let next = c
                        .iter()
                        .zip(&occupations)
                        .map(|(c, &n)| scf::density(c, n))
                        .collect::<Vec<_>>();
                    let residual = d
                        .iter()
                        .zip(&next)
                        .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).norm()))
                        .fold(0f64, f64::max);
                    let delta = previous.map(|p| value.energy - p);
                    history.push(Iteration {
                        iteration,
                        energy: value.energy,
                        delta,
                        residual,
                    });
                    d = next;
                    previous = Some(value.energy);
                    if delta.is_some_and(|de: f64| de.abs() < options.tolerance)
                        && residual < options.tolerance
                    {
                        converged = true;
                        break;
                    }
                }
                let value = evaluate(
                    self,
                    &h,
                    &d,
                    functional,
                    &mut poisson,
                    orbital_gradients.as_ref().map(|v| v.as_slice()),
                    screened.as_ref(),
                )?;
                let energies = value
                    .f
                    .into_iter()
                    .map(|f| crate::math::linalg::eigen(f).map(|(e, _)| e))
                    .collect::<Result<_>>()?;
                let candidate = Candidate {
                    energy: value.energy,
                    converged,
                    history,
                    coefficients: c,
                    densities: d,
                    energies,
                };
                if best.as_ref().is_none_or(|(b, _, _)| {
                    (candidate.converged && !b.converged)
                        || (candidate.converged == b.converged
                            && candidate.energy < b.energy - 1e-12)
                }) {
                    best = Some((candidate, value.hartree, value.xc));
                }
            }
            let (candidate, eh, exc) = best.unwrap();
            self.finish(
                if unrestricted { "uhf" } else { "rhf" },
                particles,
                options,
                candidate,
                workspace,
                Some((functional, eh, exc)),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, Geometry, LengthUnit};
    #[test]
    fn meta_gga_fock_is_the_derivative_of_its_discrete_density_energy() {
        let ctx = Context::new(Some(1), 64 << 20).unwrap();
        let geometry = Geometry::new(
            vec![1, 1],
            vec![[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]],
            [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
            [false; 3],
            LengthUnit::Angstrom,
        )
        .unwrap();
        let grid = super::super::SpatialGrid::from_geometry(
            &geometry,
            [0.5 / 0.529177210903; 3],
            false,
            false,
        )
        .unwrap();
        let integrals =
            MolecularIntegrals::new(&ctx, geometry, grid, r#"{"name":"HAO"}"#, "spectral", None)
                .unwrap();
        let m = integrals.m();
        let n = integrals.grid.index.points();
        let mut gradients = ctx.allocate::<f64>(3 * m * n).unwrap();
        for axis in 0..3 {
            for a in 0..m {
                xc::derivative(
                    &integrals.grid,
                    &integrals.orthonormal_samples.as_slice()[a * n..(a + 1) * n],
                    axis,
                    &mut gradients.as_mut_slice()[(axis * m + a) * n..(axis * m + a + 1) * n],
                )
                .unwrap();
            }
        }
        let h = Matrix::from_fn(m, m, |a, b| {
            C64::new(integrals.one.as_slice()[a * m + b], 0.)
        });
        let density = Matrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
            C64::new(0.45, 0.),
            C64::new(0.55, 0.),
        ]));
        let direction = Matrix::from_row_slice(
            2,
            2,
            &[
                C64::new(0.3, 0.),
                C64::new(0.2, 0.),
                C64::new(0.2, 0.),
                C64::new(-0.3, 0.),
            ],
        );
        let mut poisson = Poisson::isolated(&ctx, &integrals.grid).unwrap();
        let evaluate_density = |d: Matrix, p: &mut Poisson| {
            evaluate(
                &integrals,
                &h,
                &[d],
                "r2scan",
                p,
                Some(gradients.as_slice()),
                None,
            )
            .unwrap()
        };
        let base = evaluate_density(density.clone(), &mut poisson);
        let step = 1e-5;
        let plus = evaluate_density(&density + &direction * C64::new(step, 0.), &mut poisson);
        let minus = evaluate_density(&density - &direction * C64::new(step, 0.), &mut poisson);
        let finite = (plus.energy - minus.energy) / (2. * step);
        let analytic = 2. * (&direction * &base.f[0]).trace().re;
        assert!((finite - analytic).abs() < 2e-7, "{finite} vs {analytic}");
    }
}
