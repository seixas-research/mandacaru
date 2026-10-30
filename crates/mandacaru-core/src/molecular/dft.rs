//! Restricted and collinear unrestricted Kohn–Sham with native XC and Poisson.
use super::{
    invalid,
    scf::{self, Candidate, Diis, Iteration},
    MeanField, MolecularIntegrals, ScfOptions,
};
use crate::math::fft::Poisson;
use crate::math::storage::{with_rows, RealSamples, SampleBlocks};
use crate::models::xc;
use crate::{product, quantum::C64, Buffer, Result};
use nalgebra::DMatrix;
type Matrix = DMatrix<C64>;

pub(super) fn grid_densities(integrals: &MolecularIntegrals, d: &[Matrix]) -> Result<Buffer<f64>> {
    let m = integrals.m();
    let n = integrals.grid.index.points();
    let mut rho = integrals.context.allocate::<f64>(2 * n)?;
    let mut blocks = SampleBlocks::new(&integrals.context, &integrals.orthonormal_samples, m, n)?;
    for start in (0..n).step_by(blocks.block()) {
        let count = blocks.load(start)?;
        let (phi, w) = (blocks.values(), blocks.stride());
        for s in 0..2 {
            let density = &d[if d.len() == 1 { 0 } else { s }];
            for i in 0..count {
                let value = (0..m)
                    .map(|a| {
                        (0..m)
                            .map(|b| density[(a, b)].re * phi[a * w + i] * phi[b * w + i])
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                // Roundoff in a positive semidefinite density may be slightly negative.
                rho.as_mut_slice()[s * n + start + i] = value.max(0.);
            }
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
/// Orbital gradients `[axis][orbital][point]`, stored in the calculation's
/// precision. Each row is differentiated in float64 from the widened stored
/// orbital and rounded once.
pub(super) fn orbital_gradients(integrals: &MolecularIntegrals) -> Result<RealSamples> {
    let m = integrals.m();
    let n = integrals.grid.index.points();
    let ctx = &integrals.context;
    let mut gradients = RealSamples::zeros(ctx, integrals.precision(), product(&[3, m, n])?)?;
    let mut phi = ctx.allocate::<f64>(n)?;
    let mut row = ctx.allocate::<f64>(n)?;
    for a in 0..m {
        integrals
            .orthonormal_samples
            .widen_range(a * n, phi.as_mut_slice())?;
        for axis in 0..3 {
            xc::derivative(&integrals.grid, phi.as_slice(), axis, row.as_mut_slice())?;
            gradients.fill_row(axis * m + a, n, row.as_slice())?;
        }
    }
    Ok(gradients)
}
pub(super) fn grid_kinetic(
    integrals: &MolecularIntegrals,
    d: &[Matrix],
    gradients: &RealSamples,
) -> Result<Buffer<f64>> {
    let m = integrals.m();
    let n = integrals.grid.index.points();
    if gradients.len() != product(&[3, m, n])? {
        return Err(invalid("orbital gradients have inconsistent dimensions"));
    }
    let mut tau = integrals.context.allocate::<f64>(2 * n)?;
    let mut blocks = SampleBlocks::new(&integrals.context, gradients, 3 * m, n)?;
    for start in (0..n).step_by(blocks.block()) {
        let count = blocks.load(start)?;
        let (gradients, w) = (blocks.values(), blocks.stride());
        for spin in 0..2 {
            let density = &d[if d.len() == 1 { 0 } else { spin }];
            for i in 0..count {
                let mut value = 0.;
                for axis in 0..3 {
                    for a in 0..m {
                        for b in 0..m {
                            value += 0.5
                                * density[(a, b)].re
                                * gradients[(axis * m + a) * w + i]
                                * gradients[(axis * m + b) * w + i];
                        }
                    }
                }
                tau.as_mut_slice()[spin * n + start + i] = value.max(0.);
            }
        }
    }
    Ok(tau)
}
/// Valence density gradient `2 sum_ab D_ab phi_a grad phi_b` of both spin
/// channels (`[spin][axis][point]`) from the orbitals' own gradients: the
/// product rule that makes `|grad rho|^2 / 8 rho` of one orbital equal the
/// `tau` of `grid_kinetic`. The grid derivative of the sampled density does not
/// obey the product rule where the density is not resolved, and the mismatch
/// drives the iso-orbital indicator of a meta-GGA negative.
pub(super) fn grid_density_gradients(
    integrals: &MolecularIntegrals,
    d: &[Matrix],
    gradients: &RealSamples,
) -> Result<Buffer<f64>> {
    let m = integrals.m();
    let n = integrals.grid.index.points();
    if gradients.len() != product(&[3, m, n])? {
        return Err(invalid("orbital gradients have inconsistent dimensions"));
    }
    let ctx = &integrals.context;
    let mut field = ctx.allocate::<f64>(product(&[6, n])?)?;
    let mut orbitals = SampleBlocks::new(ctx, &integrals.orthonormal_samples, m, n)?;
    let mut slopes = SampleBlocks::new(ctx, gradients, 3 * m, n)?;
    for start in (0..n).step_by(orbitals.block()) {
        let count = orbitals.load(start)?;
        slopes.load(start)?;
        let (phi, gradients, w) = (orbitals.values(), slopes.values(), orbitals.stride());
        for spin in 0..2 {
            let density = &d[if d.len() == 1 { 0 } else { spin }];
            for axis in 0..3 {
                let out = &mut field.as_mut_slice()
                    [(spin * 3 + axis) * n + start..(spin * 3 + axis) * n + start + count];
                for (i, out) in out.iter_mut().enumerate() {
                    let mut value = 0.;
                    for a in 0..m {
                        for b in 0..m {
                            value += density[(a, b)].re
                                * phi[a * w + i]
                                * gradients[(axis * m + b) * w + i];
                        }
                    }
                    *out = 2. * value;
                }
            }
        }
    }
    Ok(field)
}
fn evaluate(
    integrals: &MolecularIntegrals,
    h: &Matrix,
    d: &[Matrix],
    name: &str,
    poisson: &mut Poisson,
    orbital_gradients: Option<&RealSamples>,
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
    // A meta-GGA sees the orbitals' own density gradient, valence by the
    // product rule and the partial core (half per spin) by the grid.
    let mut gradient = orbital_gradients
        .filter(|_| crate::models::functional(name).is_ok_and(|f| f.requires_tau))
        .map(|orbital| grid_density_gradients(integrals, d, orbital))
        .transpose()?;
    if let (Some(gradient), Some(paw)) = (gradient.as_mut(), paw) {
        let mut core = ctx.allocate::<f64>(n)?;
        for axis in 0..3 {
            xc::derivative(
                &integrals.grid,
                paw.core.as_slice(),
                axis,
                core.as_mut_slice(),
            )?;
            for s in 0..2 {
                for k in 0..n {
                    gradient.as_mut_slice()[(s * 3 + axis) * n + k] += 0.5 * core.as_slice()[k];
                }
            }
        }
    }
    let terms = xc::evaluate_fields_relativistic(
        ctx,
        &integrals.grid,
        rho.as_slice(),
        kinetic.as_ref().map(|v| v.as_slice()),
        gradient.as_ref().map(|v| v.as_slice()),
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
    // The local matrix elements accumulate over point blocks widened to
    // float64; each (a, b) sum runs over the points in ascending order.
    let mut sums = vec![DMatrix::<f64>::zeros(m, m); d.len()];
    {
        let mut orbitals = SampleBlocks::new(ctx, &integrals.orthonormal_samples, m, n)?;
        let mut slopes = orbital_gradients
            .map(|gradients| SampleBlocks::new(ctx, gradients, 3 * m, n))
            .transpose()?;
        for start in (0..n).step_by(orbitals.block()) {
            let count = orbitals.load(start)?;
            if let Some(slopes) = slopes.as_mut() {
                slopes.load(start)?;
            }
            let (phi, w) = (orbitals.values(), orbitals.stride());
            let gradient = slopes.as_ref().map(|s| s.values());
            for (s, sum) in sums.iter_mut().enumerate() {
                for a in 0..m {
                    for b in 0..m {
                        let mut value = sum[(a, b)];
                        for i in 0..count {
                            let k = start + i;
                            let local = phi[a * w + i]
                                * phi[b * w + i]
                                * (hartree.as_slice()[k].re
                                    + terms.potentials.as_slice()[s * n + k]);
                            let meta = if let (Some(gradient), Some(tau)) =
                                (gradient, terms.tau_potentials.as_ref())
                            {
                                0.5 * tau.as_slice()[s * n + k]
                                    * (0..3)
                                        .map(|axis| {
                                            gradient[(axis * m + a) * w + i]
                                                * gradient[(axis * m + b) * w + i]
                                        })
                                        .sum::<f64>()
                            } else {
                                0.
                            };
                            // dE/dD_ab through the product-rule density gradient.
                            let flux = if let (Some(gradient), Some(flux)) =
                                (gradient, terms.gradient_potentials.as_ref())
                            {
                                (0..3)
                                    .map(|axis| {
                                        flux.as_slice()[(s * 3 + axis) * n + k]
                                            * (phi[a * w + i] * gradient[(axis * m + b) * w + i]
                                                + gradient[(axis * m + a) * w + i] * phi[b * w + i])
                                    })
                                    .sum::<f64>()
                            } else {
                                0.
                            };
                            value += local + meta + flux;
                        }
                        sum[(a, b)] = value;
                    }
                }
            }
        }
    }
    for s in 0..d.len() {
        let v = Matrix::from_fn(m, m, |a, b| C64::new(sums[s][(a, b)] * dv, 0.));
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
    /// One-electron matrix of the Kohn-Sham model: the shared core operator
    /// plus the frozen atomic one-center Kohn-Sham difference of a PAW setup.
    fn kohn_sham_core(&self) -> Matrix {
        let m = self.m();
        Matrix::from_fn(m, m, |a, b| {
            C64::new(
                self.one.as_slice()[a * m + b]
                    + self
                        .corrections
                        .as_ref()
                        .and_then(|c| c.dft.as_ref())
                        .map_or(0., |p| p.one[(a, b)]),
                0.,
            )
        })
    }
    /// Electronic Kohn-Sham energy of orthonormal-frame spin densities, which
    /// need not be idempotent. Nuclear repulsion is not included. Screened
    /// exact exchange has no PAW augmentation, so it is not evaluated here.
    pub(super) fn kohn_sham_energy(&self, functional: &str, d: &[Matrix]) -> Result<f64> {
        let definition = crate::models::functional(functional)?;
        let m = self.m();
        if definition.screened_exchange.is_some()
            || d.is_empty()
            || d.len() > 2
            || d.iter().any(|d| d.shape() != (m, m))
        {
            return Err(invalid(
                "Kohn-Sham energy needs one or two spin densities and no screened exchange",
            ));
        }
        let mut poisson = Poisson::isolated(&self.context, &self.grid)?;
        let gradients = if definition.requires_tau {
            Some(orbital_gradients(self)?)
        } else {
            None
        };
        Ok(evaluate(
            self,
            &self.kohn_sham_core(),
            d,
            functional,
            &mut poisson,
            gradients.as_ref(),
            None,
        )?
        .energy)
    }
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
                Some(with_rows!(self.orthonormal_samples.rows(), |samples| {
                    super::integrals::cholesky_from_grid(
                        ctx,
                        &self.grid,
                        samples,
                        m,
                        1e-12,
                        &mut solver,
                        None,
                    )?
                }))
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
            let h = self.kohn_sham_core();
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
                    orbital_gradients.as_ref(),
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
            for start in starts {
                let density = |c: &Vec<Matrix>| -> Vec<Matrix> {
                    c.iter()
                        .zip(&occupations)
                        .map(|(c, &n)| scf::density(c, n))
                        .collect()
                };
                let mut c = start.clone();
                let mut d = density(&c);
                let mut history = vec![];
                let mut converged = false;
                // The plain DIIS iteration decides every start that converges with it. A
                // start that does not converge is retried with a level shift: a degenerate
                // open shell (a diatomic radical's pi electron) lets the occupied orbital
                // flip between iterations, and the first rise in energy then switches on a
                // shift, as in the Hartree-Fock loop, that damps the flip. The shift is not
                // used for a start that converges without it, because a shifted iteration
                // can settle into a different stationary point than the plain one.
                for retry in [false, true] {
                    if retry {
                        c = start.clone();
                        d = density(&c);
                        history = vec![];
                    }
                    let mut previous = None;
                    let mut mixer = Diis::new();
                    let mut shift = 0.;
                    for iteration in 1..=options.max_iterations {
                        let value = evaluate(
                            self,
                            &h,
                            &d,
                            functional,
                            &mut poisson,
                            orbital_gradients.as_ref(),
                            screened.as_ref(),
                        )?;
                        if retry
                            && previous.is_some_and(|p: f64| value.energy > p + 1e-10)
                            && shift == 0.
                        {
                            shift = 0.5;
                        }
                        let mut effective = if options.diis {
                            mixer.extrapolate(&value.f, &d)
                        } else {
                            value.f
                        };
                        if shift > 0. {
                            for (f, d) in effective.iter_mut().zip(&d) {
                                *f += (Matrix::identity(f.nrows(), f.nrows()) - d)
                                    * C64::new(shift, 0.);
                            }
                        }
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
                    if converged {
                        break;
                    }
                }
                let value = evaluate(
                    self,
                    &h,
                    &d,
                    functional,
                    &mut poisson,
                    orbital_gradients.as_ref(),
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
                Default::default(),
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
                        Some(&gradients),
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
                        Some(&gradients),
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
                        Some(&gradients),
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
        let integrals = MolecularIntegrals::new(
            &ctx,
            geometry,
            grid,
            r#"{"name":"HAO"}"#,
            "spectral",
            None,
            Default::default(),
        )
        .unwrap();
        let m = integrals.m();
        let gradients = orbital_gradients(&integrals).unwrap();
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
            evaluate(&integrals, &h, &[d], "r2scan", p, Some(&gradients), None).unwrap()
        };
        let base = evaluate_density(density.clone(), &mut poisson);
        let step = 1e-5;
        let plus = evaluate_density(&density + &direction * C64::new(step, 0.), &mut poisson);
        let minus = evaluate_density(&density - &direction * C64::new(step, 0.), &mut poisson);
        let finite = (plus.energy - minus.energy) / (2. * step);
        let analytic = 2. * (&direction * &base.f[0]).trace().re;
        assert!((finite - analytic).abs() < 2e-7, "{finite} vs {analytic}");
    }
    #[test]
    fn a_one_orbital_density_is_iso_orbital_with_the_orbital_product_rule_gradient() {
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
        let integrals = MolecularIntegrals::new(
            &ctx,
            geometry,
            grid,
            r#"{"name":"HAO"}"#,
            "spectral",
            None,
            Default::default(),
        )
        .unwrap();
        let n = integrals.grid.index.points();
        // One orbital per spin: rho = psi^2, so tau = |grad psi|^2 / 2 and the
        // Weizsaecker kinetic density |grad rho|^2 / 8 rho must equal it.
        let orbital = [0.6, 0.8];
        let density = Matrix::from_fn(2, 2, |a, b| C64::new(orbital[a] * orbital[b], 0.));
        let d = [density];
        let gradients = orbital_gradients(&integrals).unwrap();
        let tau = grid_kinetic(&integrals, &d, &gradients).unwrap();
        let product_rule = grid_density_gradients(&integrals, &d, &gradients).unwrap();
        let rho = grid_densities(&integrals, &d).unwrap();
        let mut stencil = vec![0.; 3 * n];
        for axis in 0..3 {
            xc::derivative(
                &integrals.grid,
                &rho.as_slice()[..n],
                axis,
                &mut stencil[axis * n..(axis + 1) * n],
            )
            .unwrap();
        }
        let weizsaecker = |gradient: &dyn Fn(usize) -> f64, k: usize| {
            (0..3)
                .map(|axis| gradient(axis * n + k).powi(2))
                .sum::<f64>()
                / (8. * rho.as_slice()[k])
        };
        let (mut worst, mut violated, mut counted) = (0f64, 0, 0);
        for k in 0..n {
            if rho.as_slice()[k] < 1e-12 {
                continue;
            }
            counted += 1;
            let t = tau.as_slice()[k];
            worst = worst.max((weizsaecker(&|i| product_rule.as_slice()[i], k) - t).abs() / t);
            if t - weizsaecker(&|i| stencil[i], k) < -1e-3 * t {
                violated += 1;
            }
        }
        assert!(counted > 1000);
        assert!(worst < 1e-9, "product rule: |tau_W - tau| / tau = {worst}");
        // The grid derivative of the sampled density breaks that identity
        // where the density is not resolved: the indicator would be negative.
        assert!(violated > 0, "the stencil should violate tau_W <= tau");
    }
}
