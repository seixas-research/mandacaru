//! The screened hybrid's pieces of an excited-state gradient (`dev_plan/TDDFT_PLAN.md`, T7): the
//! explicit derivatives of its quadratic exact exchange and of its spheres along the directions
//! of the Lagrangian, at fixed raw-AO densities.
//!
//! The exchange part of the Lagrangian is a sum of exchange forms of different densities (the
//! reference with the relaxed difference, the symmetric and the imaginary transition densities),
//! none of them idempotent. Each is a Hermitian matrix `X = sum_k lambda_k phi_k phi_k^+` (signed
//! weights; complex orbitals for the imaginary direction), and every product of two of them is
//!
//! ```text
//! E = -(a/2) sum_kl c_kl (rho_kl | rho_lk),   rho_kl = conj(phi_k) phi_l + sum_A Q^A_kl g_A,
//! ```
//!
//! with real symmetric couplings `c_kl` (`lambda_k mu_l` between the orbitals of two densities,
//! zero between spins) and the screened one-center tensor in the projections. This is the spinor
//! exchange of [`super::spinor_screened`] with one component and the couplings in every pair
//! sum; the frozen one-body term and the polarization of the spheres, which are not exchange
//! forms, are [`Setup::hybrid_sphere_derivative`].
use super::{compensation, projections, Contribution, Occupied, SpinorExchange};
use crate::{
    math::fft::Poisson,
    math::linalg::real_eigen,
    molecular::MolecularIntegrals,
    paw::molecular::{invalid, Setup},
    quantum::C64,
    Result,
};
use nalgebra::{DMatrix, DVector};

type CVector = DVector<C64>;
type CMatrix = DMatrix<C64>;

fn sym(m: &DMatrix<f64>) -> DMatrix<f64> {
    (m + m.transpose()) * 0.5
}

fn complex(m: &DMatrix<f64>) -> CMatrix {
    m.map(|v| C64::new(v, 0.))
}

impl Setup {
    /// The explicit position derivative of the coupled screened exchange `E` (module
    /// documentation; `omega`, `fraction`) of the complex orbitals whose real and imaginary parts
    /// are the pseudo-orbitals `2 k` and `2 k + 1` of `pseudo` (unit weights), with grid values
    /// `psi` (`pseudo.len() x n`) and the real symmetric `couplings` (`count x count`). The two
    /// grid actions (`fraction` included) are for the caller's contraction with the basis
    /// derivative.
    pub fn coupled_exchange(
        &self,
        integrals: &MolecularIntegrals,
        omega: f64,
        fraction: f64,
        pseudo: &[Occupied],
        psi: &[f64],
        couplings: &DMatrix<f64>,
    ) -> Result<SpinorExchange> {
        let ctx = &integrals.context;
        let (n, dv) = (integrals.grid.index.points(), integrals.grid.index.dv());
        let count = pseudo.len() / 2;
        if pseudo.len() % 2 != 0
            || psi.len() != pseudo.len() * n
            || couplings.shape() != (count, count)
            || (couplings - couplings.transpose()).amax() > 1e-12 * couplings.amax().max(1.)
        {
            return Err(invalid(
                "coupled exchange needs pairs of pseudo-orbitals on the grid and symmetric couplings",
            ));
        }
        let c = |i: usize, j: usize| couplings[(i, j)];
        let wave = |i: usize, k: usize| C64::new(psi[2 * i * n + k], psi[(2 * i + 1) * n + k]);
        let _memory = ctx.reserve(crate::product(&[3 * count.max(1), n, 16])?)?;
        let scatter = |action: &[C64], out: &mut [f64]| {
            for i in 0..count {
                for k in 0..n {
                    let value = action[i * n + k] * fraction;
                    out[2 * i * n + k] = value.re;
                    out[(2 * i + 1) * n + k] = value.im;
                }
            }
        };

        // The grid pairs under the screened kernel: one solve per coupled unordered pair.
        let mut grid = ctx.allocate::<f64>(crate::product(&[pseudo.len().max(1), n])?)?;
        {
            let mut action = vec![C64::new(0., 0.); count * n];
            let mut poisson = Poisson::screened(ctx, &integrals.grid, omega)?;
            let mut source = ctx.allocate::<C64>(n)?;
            let mut solved = ctx.allocate::<C64>(n)?;
            for i in 0..count {
                for j in 0..=i {
                    let w = c(i, j);
                    if w == 0. {
                        continue;
                    }
                    for k in 0..n {
                        source.as_mut_slice()[k] = wave(j, k).conj() * wave(i, k);
                    }
                    poisson.solve_into(source.as_slice(), solved.as_mut_slice())?;
                    for k in 0..n {
                        let v = solved.as_slice()[k] * w;
                        action[i * n + k] -= wave(j, k) * v;
                        if i != j {
                            action[j * n + k] -= wave(i, k) * v.conj();
                        }
                    }
                }
            }
            scatter(&action, grid.as_mut_slice());
        }

        let atoms = self.screened_atoms(ctx, omega)?;
        let channels = compensation::screened_channels(self, integrals, omega)?;
        let nch = channels.len();
        let (_, projected) = projections::project(self, integrals, pseudo)?;
        let x: Vec<Vec<CVector>> = projected
            .iter()
            .map(|rows| {
                (0..count)
                    .map(|i| rows[2 * i].zip_map(&rows[2 * i + 1], C64::new))
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
        // `-sum_ij c_ij Re(Q^A_ji conj(phi_i) phi_j)`, the weight of the compensation centers' field.
        let mut center = vec![0.; nch * n];
        let mut compensation_action = vec![C64::new(0., 0.); count * n];
        let mut field = ctx.allocate::<f64>(n)?;
        let mut t = vec![C64::new(0., 0.); count * n];
        for (a, channel) in channels.iter().enumerate() {
            let xa = &x[channel.atom];
            for i in 0..count {
                for j in 0..count {
                    q[pair(a, i, j)] = xa[i].dotc(&(&blocks[a] * &xa[j]));
                }
            }
            compensation::grid_field(integrals, channel, field.as_mut_slice());
            let f = field.as_slice();
            for i in 0..count {
                for j in 0..count {
                    if c(i, j) == 0. {
                        continue;
                    }
                    let mut value = C64::new(0., 0.);
                    for k in 0..n {
                        value += wave(i, k).conj() * wave(j, k) * f[k];
                    }
                    xw[pair(a, i, j)] = value * dv;
                }
            }
            // t_i(k) = sum_j c_ij Q^A_ji phi_j(k).
            t.fill(C64::new(0., 0.));
            for i in 0..count {
                for j in 0..count {
                    let element = q[pair(a, j, i)] * c(i, j);
                    if element.norm() == 0. {
                        continue;
                    }
                    for k in 0..n {
                        t[i * n + k] += element * wave(j, k);
                    }
                }
            }
            for i in 0..count {
                for k in 0..n {
                    let value = t[i * n + k];
                    compensation_action[i * n + k] -= value * f[k];
                    center[a * n + k] -= (wave(i, k).conj() * value).re;
                }
            }
        }
        drop(t);
        let mut compensation = ctx.allocate::<f64>(crate::product(&[pseudo.len().max(1), n])?)?;
        scatter(&compensation_action, compensation.as_mut_slice());
        drop(compensation_action);

        let samples = projections::derivative_samples(self, integrals)?;
        let force = |pi: Vec<Vec<CVector>>| -> Vec<f64> {
            let real: Vec<Vec<DVector<f64>>> = pi
                .iter()
                .map(|rows| {
                    let mut out = vec![DVector::zeros(0); 2 * count];
                    for (i, row) in rows.iter().enumerate() {
                        out[2 * i] = row.map(|v| v.re);
                        out[2 * i + 1] = row.map(|v| v.im);
                    }
                    out
                })
                .collect();
            projections::force(integrals, pseudo, &samples, &real)
                .into_iter()
                .map(|v| v * fraction)
                .collect()
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
            let p = xa.first().map_or(0, |v| v.len());
            let mut rows = Vec::with_capacity(count);
            for o in 0..count {
                let mut pi = CVector::zeros(p);
                for j in 0..count {
                    let coupling = C64::new(c(j, o), 0.);
                    if coupling.re == 0. {
                        continue;
                    }
                    for &a in &own {
                        let coefficient = (0..nch)
                            .map(|b| q[pair(b, j, o)] * usym[(a, b)])
                            .sum::<C64>()
                            + xw[pair(a, j, o)];
                        pi += &blocks[a] * &xa[j] * (coefficient * coupling);
                    }
                    for (factor, &w) in factors.iter().zip(weights) {
                        let element = xa[j].dotc(&(factor * &xa[o]));
                        pi += factor * &xa[j] * (element * w * coupling);
                    }
                }
                rows.push(pi * C64::new(-2., 0.));
            }
            pairs.push(rows);
        }
        contributions.push(Contribution {
            name: "hybrid_pairs/projection",
            gradient: force(pairs),
        });

        let mut exchange = DMatrix::<f64>::zeros(nch, nch);
        for a in 0..nch {
            for b in 0..nch {
                let mut sum = 0.;
                for i in 0..count {
                    for j in 0..count {
                        let w = c(i, j);
                        if w != 0. {
                            sum += w * (q[pair(a, i, j)] * q[pair(b, j, i)]).re;
                        }
                    }
                }
                exchange[(a, b)] = -0.5 * sum;
            }
        }
        let electrostatic =
            compensation::electrostatic_force(self, integrals, &channels, &exchange)?;
        contributions.push(Contribution {
            name: "hybrid_pairs/electrostatic",
            gradient: electrostatic.into_iter().map(|v| v * fraction).collect(),
        });
        let centers = compensation::field_force(integrals, &channels, |a, k| center[a * n + k])?;
        contributions.push(Contribution {
            name: "hybrid_pairs/compensation_center",
            gradient: centers.into_iter().map(|v| v * fraction).collect(),
        });
        Ok(SpinorExchange {
            grid,
            compensation,
            contributions,
        })
    }

    /// The explicit derivative of `-fraction` times the spheres' frozen short-range semilocal
    /// exchange `tr(V n)` and its polarization `P(n_up, n_dn)` (`omega`) along an excited
    /// state's directions, at fixed raw-AO spin matrices: the reference `d0`, the relaxed
    /// difference `p` (first derivative) and the transition density `t` (second derivative),
    ///
    /// ```text
    /// -a sum_atoms,s [ tr(V dn_s[P]) + tr(O0_s dn_s[P]) + tr((O1_s[P] + O2_s[T, T]) dn_s[D0])
    ///                  + 2 tr(O1_s[T] dn_s[T]) ],
    /// ```
    ///
    /// with `O0 = dP/dn_s` at the reference, `O1` its first change, `O2` its second derivative
    /// along `T` (projector layout) and `dn_s[M]` the position derivative of the projections of
    /// `M` at fixed `M`. Three per atom.
    pub fn hybrid_sphere_derivative(
        &self,
        integrals: &MolecularIntegrals,
        omega: f64,
        fraction: f64,
        d0: [&DMatrix<f64>; 2],
        p: [&DMatrix<f64>; 2],
        t: [&DMatrix<f64>; 2],
    ) -> Result<Vec<f64>> {
        let ctx = &integrals.context;
        let m = integrals.m();
        if [d0, p, t].iter().flatten().any(|x| x.shape() != (m, m)) {
            return Err(invalid("the sphere directions must match the basis"));
        }
        let atoms = self.screened_atoms(ctx, omega)?;
        // Pseudo-orbitals of each matrix and spin (signed weights), tagged by their matrix.
        let mut orbitals = Vec::new();
        let mut role = Vec::new();
        for (r, matrix) in [d0, p, t].iter().enumerate() {
            for (spin, x) in matrix.iter().enumerate() {
                let (values, vectors) = real_eigen(sym(x))?;
                let largest = values.iter().fold(0f64, |a, v| a.max(v.abs()));
                for (j, &value) in values.iter().enumerate() {
                    if value.abs() <= 1e-14 * largest {
                        continue;
                    }
                    orbitals.push(Occupied {
                        spin,
                        weight: value,
                        energy: 0.,
                        w_weight: 0.,
                        ao: vectors.column(j).iter().copied().collect(),
                    });
                    role.push(r);
                }
            }
        }
        let (_, projected) = projections::project(self, integrals, &orbitals)?;
        let mut pi: Vec<Vec<DVector<f64>>> = Vec::new();
        for (a, &species) in self.atoms.iter().enumerate() {
            let size = self.species_terms[species].projectors.len();
            // `n[matrix][spin]` in the projector layout.
            let mut n = [[(); 2]; 3].map(|row| row.map(|_| DMatrix::<f64>::zeros(size, size)));
            for (o, orbital) in orbitals.iter().enumerate() {
                let x = &projected[a][o];
                n[role[o]][orbital.spin] += x * x.transpose() * orbital.weight;
            }
            let polarization = &atoms[species].polarization;
            let ground = polarization.evaluate(ctx, &n[0][0], &n[0][1])?;
            let kernel = polarization.kernel(ctx, &n[0][0], &n[0][1])?;
            let along_p = polarization.linear_response(&kernel, &n[1][0], &n[1][1])?;
            let along_t = polarization.linear_response(&kernel, &n[2][0], &n[2][1])?;
            let second = polarization.second_change(&n[0][0], &n[0][1], &n[2][0], &n[2][1])?;
            let v = sym(&atoms[species].v);
            let ground = [&ground.up, &ground.down];
            let operator = |r: usize, s: usize| match r {
                0 => sym(&(&along_p[s] + &second[s] * 2.)),
                1 => &v + sym(ground[s]),
                _ => sym(&along_t[s]) * 2.,
            };
            let operators = [0, 1, 2].map(|r| [0, 1].map(|s| operator(r, s)));
            pi.push(
                orbitals
                    .iter()
                    .enumerate()
                    .map(|(o, orbital)| {
                        &operators[role[o]][orbital.spin]
                            * &projected[a][o]
                            * (-2. * orbital.weight * fraction)
                    })
                    .collect(),
            );
        }
        let samples = projections::derivative_samples(self, integrals)?;
        Ok(projections::force(integrals, &orbitals, &samples, &pi))
    }
}
