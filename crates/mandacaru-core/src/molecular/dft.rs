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
    let mut rho = grid_densities(integrals, d)?;
    let paw = integrals.corrections.as_ref().and_then(|c| c.dft.as_ref());
    let mut total = ctx.allocate::<C64>(n)?;
    for k in 0..n {
        total.as_mut_slice()[k] = C64::new(rho.as_slice()[k] + rho.as_slice()[n + k], 0.);
    }
    let mut hartree = ctx.allocate::<C64>(n)?;
    poisson.solve_into(total.as_slice(), hartree.as_mut_slice())?;
    let mut kinetic = orbital_gradients
        .map(|gradient| grid_kinetic(integrals, d, gradient))
        .transpose()?;
    if let Some(paw) = paw {
        for s in 0..2 {
            for k in 0..n {
                rho.as_mut_slice()[s * n + k] += 0.5 * paw.core.as_slice()[k];
                if let Some(tau) = kinetic.as_mut() {
                    tau.as_mut_slice()[s * n + k] += 0.5 * paw.core_tau.as_slice()[k];
                }
            }
        }
    }
    let terms = xc::evaluate_fields_relativistic(
        ctx,
        &integrals.grid,
        rho.as_slice(),
        kinetic.as_ref().map(|v| v.as_slice()),
        name,
        paw.is_some_and(|p| p.relativistic_exchange)
            && matches!(crate::models::functional(name)?.semilocal, "lda" | "pbe"),
    )?;
    let mut eh = 0.5
        * total
            .as_slice()
            .iter()
            .zip(hartree.as_slice())
            .map(|(r, v)| (r * v).re)
            .sum::<f64>()
        * dv;
    let mut augmentation = Matrix::zeros(m, m);
    if let Some(paw) = paw {
        for (factor, weight) in paw.pairs.factors.iter().zip(&paw.pairs.weights) {
            let moment = d
                .iter()
                .map(|density| {
                    (0..m)
                        .map(|a| {
                            (0..m)
                                .map(|b| density[(a, b)].re * factor[(b, a)])
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
                * if d.len() == 1 { 2. } else { 1. };
            eh += 0.5 * weight * moment * moment;
            for a in 0..m {
                for b in 0..m {
                    augmentation[(a, b)] += weight * moment * factor[(a, b)];
                }
            }
        }
    }
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
        let mut fock = h + v + &augmentation;
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
        energy: eone + eh + terms.energy + exact_energy + paw.map_or(0., |p| p.constant),
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
        if self.corrections.as_ref().is_some_and(|c| c.dft.is_none()) {
            return Err(invalid(
                "native molecular PAW Kohn-Sham augmentation is not yet implemented",
            ));
        }
        let definition = crate::models::functional(functional)?;
        if self.corrections.is_some() && definition.screened_exchange.is_some() {
            return Err(invalid(
                "PAW screened exact exchange requires augmented short-range pair integrals",
            ));
        }
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
            let h = Matrix::from_fn(m, m, |a, b| {
                C64::new(
                    self.one.as_slice()[a * m + b]
                        + self
                            .corrections
                            .as_ref()
                            .and_then(|c| c.dft.as_ref())
                            .map_or(0., |p| p.one[(a, b)]),
                    0.,
                )
            });
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
    use crate::{math::SpatialGrid, Context, Geometry, LengthUnit};
    #[test]
    fn paw_core_and_signed_compensation_fock_differentiate_the_same_energy() {
        use crate::molecular::integrals::{MolecularCorrections, PairCorrection, PawKohnSham};
        use std::sync::Arc;
        let ctx = Context::new(Some(1), 64 << 20).unwrap();
        {
            let geometry = Geometry::new(
                vec![1, 1],
                vec![[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]],
                [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
                [false; 3],
                LengthUnit::Angstrom,
            )
            .unwrap();
            let grid = SpatialGrid::from_geometry(&geometry, [1.; 3], false, false).unwrap();
            let n = grid.index.points();
            let mut integrals = MolecularIntegrals::new(
                &ctx,
                geometry,
                grid,
                r#"{"name":"HAO"}"#,
                "spectral",
                None,
            )
            .unwrap();
            let m = integrals.m();
            let mut core = ctx.allocate::<f64>(n).unwrap();
            let mut tau = ctx.allocate::<f64>(n).unwrap();
            for k in 0..n {
                let point = integrals.grid.point(k);
                let center = integrals.geometry.positions()[0];
                let r2 = (0..3).map(|a| (point[a] - center[a]).powi(2)).sum::<f64>();
                core.as_mut_slice()[k] = 0.2 * (-r2).exp();
                tau.as_mut_slice()[k] = 0.5 * r2 * core.as_slice()[k];
            }
            let one = DMatrix::from_row_slice(m, m, &[0.1, 0.03, 0.03, -0.2]);
            integrals.corrections = Some(MolecularCorrections {
                external: ctx.allocate(n).unwrap(),
                overlap: DMatrix::zeros(m, m),
                potential: DMatrix::zeros(m, m),
                charges: vec![1., 1.],
                electrons: 2,
                pairs: Arc::new(PairCorrection {
                    factors: vec![],
                    weights: vec![],
                    memory: ctx.reserve(0).unwrap(),
                }),
                dft: Some(PawKohnSham {
                    one: one.clone(),
                    pairs: PairCorrection {
                        factors: vec![DMatrix::from_row_slice(m, m, &[0.2, 0.1, 0.1, -0.3])],
                        weights: vec![-0.4],
                        memory: ctx.reserve(32).unwrap(),
                    },
                    core,
                    core_tau: tau,
                    constant: 0.17,
                    relativistic_exchange: true,
                    memory: ctx.reserve(32).unwrap(),
                }),
            });
            let h = Matrix::from_fn(m, m, |a, b| {
                C64::new(integrals.one.as_slice()[a * m + b] + one[(a, b)], 0.)
            });
            let density = Matrix::from_row_slice(
                m,
                m,
                &[
                    C64::new(0.45, 0.),
                    C64::new(0.05, 0.01),
                    C64::new(0.05, -0.01),
                    C64::new(0.55, 0.),
                ],
            );
            let direction = Matrix::from_row_slice(
                m,
                m,
                &[
                    C64::new(0.3, 0.),
                    C64::new(0.2, 0.1),
                    C64::new(0.2, -0.1),
                    C64::new(-0.3, 0.),
                ],
            );
            let gradients = orbital_gradients(&integrals).unwrap();
            let mut poisson = Poisson::isolated(&ctx, &integrals.grid).unwrap();
            for functional in ["lda", "pbe", "r2scan"] {
                for unrestricted in [false, true] {
                    let densities = if unrestricted {
                        vec![density.clone(), &density * C64::new(0.7, 0.)]
                    } else {
                        vec![density.clone()]
                    };
                    let evaluated = evaluate(
                        &integrals,
                        &h,
                        &densities,
                        functional,
                        &mut poisson,
                        Some(gradients.as_slice()),
                        None,
                    )
                    .unwrap();
                    let expected = (&evaluated.f[0] * &direction).trace().re
                        * if unrestricted { 1. } else { 2. };
                    let step = 1e-6;
                    let mut plus = densities.clone();
                    let mut minus = densities;
                    plus[0] += &direction * C64::new(step, 0.);
                    minus[0] -= &direction * C64::new(step, 0.);
                    let ep = evaluate(
                        &integrals,
                        &h,
                        &plus,
                        functional,
                        &mut poisson,
                        Some(gradients.as_slice()),
                        None,
                    )
                    .unwrap()
                    .energy;
                    let em = evaluate(
                        &integrals,
                        &h,
                        &minus,
                        functional,
                        &mut poisson,
                        Some(gradients.as_slice()),
                        None,
                    )
                    .unwrap()
                    .energy;
                    assert!(
                        ((ep - em) / (2. * step) - expected).abs() < 2e-8,
                        "{functional} unrestricted={unrestricted}"
                    );
                }
            }
        }
        assert_eq!(ctx.memory_used(), 0);
    }
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
