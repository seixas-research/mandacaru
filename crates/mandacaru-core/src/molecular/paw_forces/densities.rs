//! The nuclear gradient of a general fixed density in the PAW-LCAO Hartree-Fock model
//! ([`FixedDensities`]): the grid parts (kinetic, external and overlap motion of the basis
//! functions, the Hartree and compensation potentials of every Coulomb form) here, the sphere and
//! projection parts in `paw::molecular::gradient::densities`.
use super::{grid, nuclear_gradient};
use crate::math::fft::Poisson;
use crate::math::linalg::real_eigen;
use crate::molecular::forces::Gradient;
use crate::molecular::{
    integrals::laplacian_of, invalid, variational_gradient::FixedDensities, MolecularIntegrals,
};
use crate::paw::molecular::gradient::{Forms, Occupied};
use crate::{product, quantum::C64, Result};
use nalgebra::DMatrix;

/// Eigen-decomposition of a symmetric matrix into pseudo-orbitals: `weight` (or `w_weight`) the
/// eigenvalue, those below `1e-14` of the largest dropped.
fn pseudo(matrix: &DMatrix<f64>, energy_role: bool) -> Result<Vec<Occupied>> {
    let (values, vectors) = real_eigen((matrix + matrix.transpose()) * 0.5)?;
    let largest = values.iter().fold(0f64, |a, v| a.max(v.abs()));
    Ok(values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.abs() > 1e-14 * largest)
        .map(|(j, &v)| Occupied {
            spin: 0,
            weight: if energy_role { 0. } else { v },
            energy: 0.,
            w_weight: if energy_role { v } else { 0. },
            ao: vectors.column(j).iter().copied().collect(),
        })
        .collect())
}

impl MolecularIntegrals {
    /// `d/dR [sum P h + 1/2 sum_k lambda_k B(X_k, X_k) - (W . S)] + d E_nn / dR` at fixed raw-AO
    /// densities, Hartree per Bohr: the explicit derivative of the PAW-LCAO Hartree-Fock model's
    /// energy of these densities, the overlap term `-tr[W dS]` included.
    pub fn fixed_density_gradient(&self, densities: &FixedDensities) -> Result<Gradient> {
        let setup = self.paw_setup.as_ref().ok_or_else(|| {
            invalid("the analytic variational force needs a PAW-LCAO setup; the all-electron basis has none")
        })?;
        let corrections = self
            .corrections
            .as_ref()
            .ok_or_else(|| invalid("the analytic variational force needs the PAW corrections"))?;
        let m = self.m();
        let r = densities.frame.ncols();
        if densities.one.shape() != (m, m)
            || densities.weighted.shape() != (m, m)
            || densities.frame.nrows() != m
            || densities.two.iter().any(|(_, x)| x.shape() != (r, r))
        {
            return Err(invalid("fixed densities must match the basis"));
        }
        self.context.run(|| {
            let ctx = &self.context;
            let n = self.grid.index.points();
            let atoms = self.geometry.numbers().len();
            let mut occupied = pseudo(&densities.one, false)?;
            let count = occupied.len();
            occupied.extend(pseudo(&densities.weighted, true)?);
            let nocc = occupied.len();
            let _memory = ctx.reserve(product(&[nocc.max(1) + r, n, 8, 3])?)?;
            let psi = self.on_grid(&occupied)?;
            let mut rho_one = ctx.allocate::<f64>(n)?;
            let mut kinetic = ctx.allocate::<f64>(product(&[nocc.max(1), n])?)?;
            for (i, o) in occupied.iter().enumerate().take(count) {
                let row = &psi.as_slice()[i * n..(i + 1) * n];
                for (x, p) in rho_one.as_mut_slice().iter_mut().zip(row) {
                    *x += o.weight * p * p;
                }
                let lap = laplacian_of(ctx, &self.grid, &self.kinetic_method, row)?;
                for (k, v) in lap.as_slice().iter().enumerate() {
                    kinetic.as_mut_slice()[i * n + k] = -0.5 * v;
                }
            }
            let zeros = ctx.allocate::<f64>(n)?;
            let points = grid::points(self)?;
            let combined = grid::contract_combined(
                self,
                &occupied,
                points.as_slice(),
                grid::CombinedInputs {
                    psi: psi.as_slice(),
                    external: corrections.external.as_slice(),
                    hartree: zeros.as_slice(),
                    xc: None,
                    coulomb: zeros.as_slice(),
                    kinetic: Some(kinetic.as_slice()),
                    xc_gradient: None,
                    xc_tau: None,
                },
            );
            drop(kinetic);
            let mut terms: Vec<(&'static str, Vec<f64>)> = vec![
                (
                    "kinetic/basis",
                    combined.kinetic.unwrap_or_else(|| vec![0.; 3 * atoms]),
                ),
                ("external/basis", combined.external),
                ("pulay_grid/basis", combined.pulay),
            ];
            // The forms: the frame orbitals on the grid and each form's density.
            let frame: Vec<Occupied> = (0..r)
                .map(|p| Occupied {
                    spin: 0,
                    weight: 1.,
                    energy: 0.,
                    w_weight: 0.,
                    ao: densities.frame.column(p).iter().copied().collect(),
                })
                .collect();
            let phi = self.on_grid(&frame)?;
            let nforms = densities.two.len();
            // Pair products phi_p phi_q (p <= q) by point blocks, so the form densities and the
            // actions below are two matrix products, never a per-form loop over the grid.
            let pairs: Vec<(usize, usize)> =
                (0..r).flat_map(|p| (p..r).map(move |q| (p, q))).collect();
            let npairs = pairs.len();
            let stride = n.min(4096).max(1);
            let _blocks = ctx.reserve(product(&[npairs.max(1), stride, 8, 2])?)?;
            let symmetric = |x: &DMatrix<f64>, p: usize, q: usize| 0.5 * (x[(p, q)] + x[(q, p)]);
            // rho_k = sum_{p<=q} w_pq x_pq phi_p phi_q, w = 2 off the diagonal.
            let weights = DMatrix::from_fn(nforms.max(1), npairs, |k, a| {
                if k >= nforms {
                    return 0.;
                }
                let (p, q) = pairs[a];
                let x = &densities.two[k].1;
                if p == q {
                    x[(p, p)]
                } else {
                    2. * symmetric(x, p, q)
                }
            });
            let pair_block = |start: usize, count: usize| {
                DMatrix::from_fn(npairs, count, |a, i| {
                    let (p, q) = pairs[a];
                    phi.as_slice()[p * n + start + i] * phi.as_slice()[q * n + start + i]
                })
            };
            let mut rho = ctx.allocate::<f64>(product(&[nforms.max(1), n])?)?;
            for start in (0..n).step_by(stride) {
                let count = stride.min(n - start);
                let block = &weights * pair_block(start, count);
                for k in 0..nforms {
                    rho.as_mut_slice()[k * n + start..k * n + start + count]
                        .copy_from_slice(block.row(k).transpose().as_slice());
                }
            }
            let paw = setup.density_terms(
                self,
                &occupied[..count],
                rho_one.as_slice(),
                &densities.weighted,
                &Forms {
                    frame: &densities.frame,
                    terms: &densities.two,
                    rho: &rho.as_slice()[..nforms * n],
                },
            )?;
            // tau_p = sum_k lambda_k U_k sum_q x_pq phi_q, U_k = V_H[rho_k] + the compensation potential.
            let mut poisson = Poisson::isolated(ctx, &self.grid)?;
            let mut source = ctx.allocate::<C64>(n)?;
            let mut solved = ctx.allocate::<C64>(n)?;
            // U_k = lambda_k (V_H[rho_k] + the compensation potential), then the actions
            // tau_p = sum_q Y_pq phi_q with Y_pq = sum_k x_k,pq U_k (one product per point block).
            let mut potentials = ctx.allocate::<f64>(product(&[nforms.max(1), n])?)?;
            for (k, (lambda, _)) in densities.two.iter().enumerate() {
                for i in 0..n {
                    source.as_mut_slice()[i] = C64::new(rho.as_slice()[k * n + i], 0.);
                }
                poisson.solve_into(source.as_slice(), solved.as_mut_slice())?;
                for i in 0..n {
                    potentials.as_mut_slice()[k * n + i] =
                        lambda * (solved.as_slice()[i].re + paw.compensation.as_slice()[k * n + i]);
                }
            }
            drop(rho);
            let coefficients = DMatrix::from_fn(npairs, nforms.max(1), |a, k| {
                if k >= nforms {
                    return 0.;
                }
                let (p, q) = pairs[a];
                symmetric(&densities.two[k].1, p, q)
            });
            let mut tau = ctx.allocate::<f64>(product(&[r.max(1), n])?)?;
            for start in (0..n).step_by(stride) {
                let count = stride.min(n - start);
                let block = DMatrix::from_fn(nforms.max(1), count, |k, i| {
                    if k < nforms {
                        potentials.as_slice()[k * n + start + i]
                    } else {
                        0.
                    }
                });
                let y = &coefficients * block;
                let t = tau.as_mut_slice();
                let f = phi.as_slice();
                for (a, &(p, q)) in pairs.iter().enumerate() {
                    for i in 0..count {
                        let value = y[(a, i)];
                        t[p * n + start + i] += value * f[q * n + start + i];
                        if p != q {
                            t[q * n + start + i] += value * f[p * n + start + i];
                        }
                    }
                }
            }
            drop(potentials);
            terms.push((
                "two_body/basis",
                grid::contract(self, &frame, points.as_slice(), tau.as_slice()),
            ));
            for part in paw.contributions {
                terms.push((part.name, part.gradient));
            }
            terms.push(("nuclear/ion_ion", nuclear_gradient(self, corrections)));
            let mut total = ctx.allocate::<f64>(3 * atoms)?;
            let mut named = Vec::new();
            for (name, values) in terms {
                for (t, v) in total.as_mut_slice().iter_mut().zip(&values) {
                    *t += v;
                }
                let mut buffer = ctx.allocate::<f64>(3 * atoms)?;
                buffer.as_mut_slice().copy_from_slice(&values);
                named.push((name, buffer));
            }
            if total.as_slice().iter().any(|v| !v.is_finite()) {
                return Err(invalid("nonfinite fixed-density gradient"));
            }
            Ok(Gradient {
                hellmann_feynman: None,
                pulay: None,
                total,
                scheme: "analytic",
                stationarity: None,
                terms: named,
            })
        })
    }
}
