//! Restricted and collinear unrestricted Kohn–Sham with native XC and Poisson.
use super::{
    fft::Poisson,
    invalid,
    scf::{self, Candidate, Diis, Iteration},
    xc, MeanField, MolecularIntegrals, ScfOptions,
};
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
fn evaluate(
    integrals: &MolecularIntegrals,
    h: &Matrix,
    d: &[Matrix],
    name: &str,
    poisson: &mut Poisson,
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
    let terms = xc::evaluate(ctx, &integrals.grid, rho.as_slice(), name)?;
    let eh = 0.5
        * total
            .as_slice()
            .iter()
            .zip(hartree.as_slice())
            .map(|(r, v)| (r * v).re)
            .sum::<f64>()
        * dv;
    let mut f = vec![];
    let phi = integrals.orthonormal_samples.as_slice();
    for s in 0..d.len() {
        let v = Matrix::from_fn(m, m, |a, b| {
            C64::new(
                (0..n)
                    .map(|k| {
                        phi[a * n + k]
                            * phi[b * n + k]
                            * (hartree.as_slice()[k].re + terms.potentials.as_slice()[s * n + k])
                    })
                    .sum::<f64>()
                    * dv,
                0.,
            )
        });
        f.push(scf::hermitian(h + v));
    }
    let eone =
        d.iter().map(|d| (d * h).trace().re).sum::<f64>() * if d.len() == 1 { 2. } else { 1. };
    Ok(Evaluation {
        f,
        energy: eone + eh + terms.energy,
        hartree: eh,
        xc: terms.energy,
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
        let functional = xc::functional(functional)?;
        self.context.run(|| {
            let m = self.m();
            let ctx = &self.context;
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
            let (_, core) = scf::eigen(h.clone())?;
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
                let f = evaluate(self, &h, &d, functional, &mut poisson)?.f;
                starts.push(
                    f.into_iter()
                        .map(|f| scf::eigen((&h + f) * C64::new(0.5, 0.)).map(|(_, c)| c))
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
                    let value = evaluate(self, &h, &d, functional, &mut poisson)?;
                    let effective = if options.diis {
                        mixer.extrapolate(&value.f, &d)
                    } else {
                        value.f
                    };
                    c = effective
                        .into_iter()
                        .map(|f| scf::eigen(f).map(|(_, c)| c))
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
                let value = evaluate(self, &h, &d, functional, &mut poisson)?;
                let energies = value
                    .f
                    .into_iter()
                    .map(|f| scf::eigen(f).map(|(e, _)| e))
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
