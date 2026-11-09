//! The screened exact exchange of a spinor state in the gradient (MG-F2c).
//!
//! A spinor state of unit occupations, `D = sum_i psi_i psi_i^+` with two-component spinors
//! `psi_i = (psi_i,up, psi_i,dn)`, has the exchange form of the scalar hybrid
//! ([`super::screened`]) with every pair quantity summed over the two components:
//!
//! ```text
//! E_x = -(a/2) sum_ij (rho_ij | rho_ji),   rho_ij = sum_s conj(psi_is) psi_js + sum_A Q^A_ij g_A,
//! Q^A_ij = sum_s x_is^+ B_A x_js,          X_k,ij = sum_s x_is^+ F_k x_js
//! ```
//!
//! (`x_is` the projections of component `s`, `B_A` the compensation blocks, `F_k` and `w_k` the
//! screened one-center tensor), plus the frozen short-range semilocal exchange of the spheres and
//! its non-collinear polarization `P(Re Q^0..3)` (`screened_polarization::vector`). For a real
//! spinor with one component every term here is the scalar term of that spin.
//!
//! The derivatives are taken with respect to `conj(x_is)` and `conj(psi_is)`: `pi_is = 2 dE /
//! d conj(x_is)` and the grid action `A_is = dE / d conj(psi_is)` (the exchange operator on the
//! component). Since `dE = Re sum (pi^+ dx) = sum [Re pi . d Re x + Im pi . d Im x]`, the caller
//! carries each spinor as four real pseudo-orbitals `(i, s, Re)`, `(i, s, Im)` of unit weight, and
//! the real contractions (`projections::force`, the grid contraction of the basis derivative)
//! apply unchanged.
use super::{compensation, projections, Contribution, Occupied};
use crate::{
    math::fft::Poisson,
    molecular::MolecularIntegrals,
    paw::molecular::{invalid, Setup},
    quantum::C64,
    Buffer, Result,
};
use nalgebra::{DMatrix, DVector};

type CVector = DVector<C64>;
type CMatrix = DMatrix<C64>;

/// The index of pseudo-orbital `(spinor, component, part)`, `part` 0 the real and 1 the
/// imaginary part.
pub fn pseudo_index(spinor: usize, component: usize, part: usize) -> usize {
    (2 * spinor + component) * 2 + part
}

/// What [`Setup::spinor_hybrid`] returns: the two grid actions on the pseudo-orbitals (`fraction`
/// included), for the caller's contraction with the basis derivative, and every other part.
pub struct SpinorExchange {
    /// The grid pairs' exchange action, `-a sum_j psi_js v_ji` (Re and Im per component).
    pub grid: Buffer<f64>,
    /// The compensation charges' part, `-a sum_j psi_js sum_A Q^A_ji g_A`.
    pub compensation: Buffer<f64>,
    pub contributions: Vec<Contribution>,
}

fn sym(m: &DMatrix<f64>) -> DMatrix<f64> {
    (m + m.transpose()) * 0.5
}

fn complex(m: &DMatrix<f64>) -> CMatrix {
    m.map(|v| C64::new(v, 0.))
}

impl Setup {
    /// The explicit position derivative of the screened exchange (`omega`, `fraction`) of the
    /// spinor state whose pseudo-orbitals are `pseudo` (layout [`pseudo_index`], unit weights)
    /// with grid values `psi` (`pseudo.len() x n`).
    pub fn spinor_hybrid(
        &self,
        integrals: &MolecularIntegrals,
        omega: f64,
        fraction: f64,
        pseudo: &[Occupied],
        psi: &[f64],
    ) -> Result<SpinorExchange> {
        let ctx = &integrals.context;
        let (n, dv) = (integrals.grid.index.points(), integrals.grid.index.dv());
        if pseudo.len() % 4 != 0 || psi.len() != pseudo.len() * n {
            return Err(invalid("spinor pseudo-orbitals come in fours, each sampled on the grid"));
        }
        let count = pseudo.len() / 4;
        let e = pseudo_index;
        let wave = |i: usize, s: usize, k: usize| C64::new(psi[e(i, s, 0) * n + k], psi[e(i, s, 1) * n + k]);
        let _memory = ctx.reserve(crate::product(&[2 * count.max(1), n, 16, 3])?)?;

        // The grid pairs under the screened kernel: one solve per unordered pair.
        let mut grid = ctx.allocate::<f64>(crate::product(&[pseudo.len().max(1), n])?)?;
        {
            let mut action = vec![C64::new(0., 0.); 2 * count * n];
            let mut poisson = Poisson::screened(ctx, &integrals.grid, omega)?;
            let mut source = ctx.allocate::<C64>(n)?;
            let mut solved = ctx.allocate::<C64>(n)?;
            for i in 0..count {
                for j in 0..=i {
                    for k in 0..n {
                        source.as_mut_slice()[k] = (0..2).map(|s| wave(j, s, k).conj() * wave(i, s, k)).sum();
                    }
                    poisson.solve_into(source.as_slice(), solved.as_mut_slice())?;
                    for s in 0..2 {
                        for k in 0..n {
                            let v = solved.as_slice()[k];
                            action[(2 * i + s) * n + k] -= wave(j, s, k) * v;
                            if i != j {
                                action[(2 * j + s) * n + k] -= wave(i, s, k) * v.conj();
                            }
                        }
                    }
                }
            }
            scatter(&action, count, n, fraction, grid.as_mut_slice());
        }

        let atoms = self.screened_atoms(ctx, omega)?;
        let channels = compensation::screened_channels(self, integrals, omega)?;
        let nch = channels.len();
        let (_, projected) = projections::project(self, integrals, pseudo)?;
        let x: Vec<Vec<[CVector; 2]>> = projected
            .iter()
            .map(|rows| {
                (0..count)
                    .map(|i| {
                        std::array::from_fn(|s| {
                            rows[e(i, s, 0)].zip_map(&rows[e(i, s, 1)], |re, im| C64::new(re, im))
                        })
                    })
                    .collect()
            })
            .collect();
        // The screened Coulomb coupling of two multipoles (the moments' own build).
        let mut nothing = ctx.allocate::<f64>(n)?;
        nothing.as_mut_slice().fill(0.);
        let moments = compensation::moments(
            self,
            integrals,
            &channels,
            None,
            pseudo,
            &projected,
            psi,
            nothing.as_slice(),
            false,
        )?;
        let usym = &moments.usym;
        let pair = |a: usize, i: usize, j: usize| (a * count + i) * count + j;
        let blocks: Vec<CMatrix> = channels.iter().map(|c| complex(&sym(c.block))).collect();
        let mut q = vec![C64::new(0., 0.); nch * count * count];
        let mut xw = vec![C64::new(0., 0.); nch * count * count];
        // `-sum_ij Re(Q^A_ji rho_ij(k))`, the weight of the compensation centers' field.
        let mut center = vec![0.; nch * n];
        let mut compensation_action = vec![C64::new(0., 0.); 2 * count * n];
        let mut field = ctx.allocate::<f64>(n)?;
        let mut t = vec![C64::new(0., 0.); 2 * count * n];
        for (a, channel) in channels.iter().enumerate() {
            let xa = &x[channel.atom];
            for i in 0..count {
                for j in 0..count {
                    q[pair(a, i, j)] = (0..2).map(|s| xa[i][s].dotc(&(&blocks[a] * &xa[j][s]))).sum();
                }
            }
            compensation::grid_field(integrals, channel, field.as_mut_slice());
            let f = field.as_slice();
            for i in 0..count {
                for j in 0..count {
                    let mut value = C64::new(0., 0.);
                    for k in 0..n {
                        value += (wave(i, 0, k).conj() * wave(j, 0, k) + wave(i, 1, k).conj() * wave(j, 1, k)) * f[k];
                    }
                    xw[pair(a, i, j)] = value * dv;
                }
            }
            // t_is(k) = sum_j Q^A_ji psi_js(k).
            t.fill(C64::new(0., 0.));
            for i in 0..count {
                for j in 0..count {
                    let element = q[pair(a, j, i)];
                    if element.norm() == 0. {
                        continue;
                    }
                    for s in 0..2 {
                        for k in 0..n {
                            t[(2 * i + s) * n + k] += element * wave(j, s, k);
                        }
                    }
                }
            }
            for i in 0..count {
                for s in 0..2 {
                    for k in 0..n {
                        let value = t[(2 * i + s) * n + k];
                        compensation_action[(2 * i + s) * n + k] -= value * f[k];
                        center[a * n + k] -= (wave(i, s, k).conj() * value).re;
                    }
                }
            }
        }
        drop(t);
        let mut compensation = ctx.allocate::<f64>(crate::product(&[pseudo.len().max(1), n])?)?;
        scatter(&compensation_action, count, n, fraction, compensation.as_mut_slice());
        drop(compensation_action);

        let samples = projections::derivative_samples(self, integrals)?;
        let force = |pi: Vec<Vec<[CVector; 2]>>| -> Vec<f64> {
            let real: Vec<Vec<DVector<f64>>> = pi
                .iter()
                .map(|rows| {
                    let mut out = vec![DVector::zeros(0); 4 * count];
                    for (i, row) in rows.iter().enumerate() {
                        for s in 0..2 {
                            out[e(i, s, 0)] = row[s].map(|v| v.re);
                            out[e(i, s, 1)] = row[s].map(|v| v.im);
                        }
                    }
                    out
                })
                .collect();
            projections::force(integrals, pseudo, &samples, &real).into_iter().map(|v| v * fraction).collect()
        };
        let mut contributions = Vec::new();

        // The pair integrals' exchange form in the projections: compensation-compensation,
        // grid-compensation and the one-center tensor.
        let mut pairs = Vec::new();
        for (atom, &species) in self.atoms.iter().enumerate() {
            let own: Vec<usize> = (0..nch).filter(|&a| channels[a].atom == atom).collect();
            let factors: Vec<CMatrix> = atoms[species].factors.iter().map(complex).collect();
            let weights = &atoms[species].weights;
            let xa = &x[atom];
            let p = xa.first().map_or(0, |v| v[0].len());
            let mut rows = Vec::with_capacity(count);
            for o in 0..count {
                let mut pi: [CVector; 2] = std::array::from_fn(|_| CVector::zeros(p));
                for j in 0..count {
                    for &a in &own {
                        let coefficient = (0..nch).map(|b| q[pair(b, j, o)] * usym[(a, b)]).sum::<C64>() + xw[pair(a, j, o)];
                        for s in 0..2 {
                            pi[s] += &blocks[a] * &xa[j][s] * coefficient;
                        }
                    }
                    for (factor, &w) in factors.iter().zip(weights) {
                        let element: C64 = (0..2).map(|s| xa[j][s].dotc(&(factor * &xa[o][s]))).sum();
                        for s in 0..2 {
                            pi[s] += factor * &xa[j][s] * (element * w);
                        }
                    }
                }
                rows.push(pi.map(|v| v * C64::new(-2., 0.)));
            }
            pairs.push(rows);
        }
        contributions.push(Contribution { name: "hybrid_pairs/projection", gradient: force(pairs) });

        let mut exchange = DMatrix::<f64>::zeros(nch, nch);
        for a in 0..nch {
            for b in 0..nch {
                let mut sum = 0.;
                for i in 0..count {
                    for j in 0..count {
                        sum += (q[pair(a, i, j)] * q[pair(b, j, i)]).re;
                    }
                }
                exchange[(a, b)] = -0.5 * sum;
            }
        }
        let electrostatic = compensation::electrostatic_force(self, integrals, &channels, &exchange)?;
        contributions.push(Contribution {
            name: "hybrid_pairs/electrostatic",
            gradient: electrostatic.into_iter().map(|v| v * fraction).collect(),
        });
        let centers = compensation::field_force(integrals, &channels, |a, k| center[a * n + k])?;
        contributions.push(Contribution {
            name: "hybrid_pairs/compensation_center",
            gradient: centers.into_iter().map(|v| v * fraction).collect(),
        });

        // The frozen short-range semilocal exchange of the spheres, linear in their density.
        let one_body = self
            .atoms
            .iter()
            .enumerate()
            .map(|(atom, &species)| {
                let v = complex(&sym(&atoms[species].v));
                x[atom].iter().map(|xi| std::array::from_fn(|s| &v * &xi[s] * C64::new(-2., 0.))).collect()
            })
            .collect();
        contributions.push(Contribution { name: "hybrid_one_body/projection", gradient: force(one_body) });

        // Its non-collinear polarization: `O^a = dP/dQ^a` of the sphere's Pauli components,
        // lifted to the spin blocks `H_uu = O^0 + O^3`, `H_dd = O^0 - O^3`, `H_ud = O^1 - i O^2`,
        // `H_du = O^1 + i O^2`, and `pi_is = -2 sum_t H_st x_it`.
        let mut polarization = Vec::new();
        for (atom, &species) in self.atoms.iter().enumerate() {
            let xa = &x[atom];
            let p = xa.first().map_or(0, |v| v[0].len());
            let mut gamma: [[CMatrix; 2]; 2] = std::array::from_fn(|_| std::array::from_fn(|_| CMatrix::zeros(p, p)));
            for xi in xa {
                for s in 0..2 {
                    for t in 0..2 {
                        gamma[s][t] += &xi[s] * xi[t].adjoint();
                    }
                }
            }
            let i = C64::i();
            let pauli: [DMatrix<f64>; 4] = [
                (&gamma[0][0] + &gamma[1][1]).map(|v| v.re),
                (&gamma[0][1] + &gamma[1][0]).map(|v| v.re),
                ((&gamma[0][1] - &gamma[1][0]) * i).map(|v| v.re),
                (&gamma[0][0] - &gamma[1][1]).map(|v| v.re),
            ];
            let terms = atoms[species].polarization.evaluate_vector(ctx, &pauli)?;
            let [o0, o1, o2, o3] = terms.operators.map(|o| complex(&sym(&o)));
            let h = [[&o0 + &o3, &o1 - &o2 * i], [&o1 + &o2 * i, &o0 - &o3]];
            polarization.push(
                xa.iter()
                    .map(|xi| std::array::from_fn(|s| (&h[s][0] * &xi[0] + &h[s][1] * &xi[1]) * C64::new(-2., 0.)))
                    .collect(),
            );
        }
        contributions.push(Contribution { name: "hybrid_polarization/projection", gradient: force(polarization) });

        Ok(SpinorExchange { grid, compensation, contributions })
    }
}

/// `fraction` times a complex component action (`[spinor][component][point]`) as the real
/// pseudo-orbital rows of [`pseudo_index`].
fn scatter(action: &[C64], count: usize, n: usize, fraction: f64, out: &mut [f64]) {
    for i in 0..count {
        for s in 0..2 {
            for k in 0..n {
                let value = action[(2 * i + s) * n + k] * fraction;
                out[pseudo_index(i, s, 0) * n + k] = value.re;
                out[pseudo_index(i, s, 1) * n + k] = value.im;
            }
        }
    }
}
