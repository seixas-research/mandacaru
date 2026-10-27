//! RHF, UHF and complex spinor GHF with bounded Pulay DIIS and level shifting.
use super::{
    invalid,
    transform::{self, Model},
    MolecularIntegrals,
};
use crate::quantum::C64;
use crate::{product, Reservation, Result};
use nalgebra::{DMatrix, DVector, SymmetricEigen};

type Matrix = DMatrix<C64>;
#[derive(Clone, Debug)]
pub struct ScfOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub diis: bool,
    pub guesses: usize,
}
impl Default for ScfOptions {
    fn default() -> Self {
        Self {
            max_iterations: 300,
            tolerance: 1e-9,
            diis: true,
            guesses: 3,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Iteration {
    pub iteration: usize,
    pub energy: f64,
    pub delta: Option<f64>,
    pub residual: f64,
}
pub struct MeanField {
    pub method: String,
    pub electronic_energy: f64,
    pub reference_energy: f64,
    pub converged: bool,
    pub iterations: Vec<Iteration>,
    pub particles: [usize; 2],
    pub energies: Vec<Vec<f64>>,
    pub coefficients: Vec<Matrix>,
    pub densities: Vec<Matrix>,
    pub natural_occupations: Vec<f64>,
    pub model_orbitals: Matrix,
    pub model: Model,
    pub kramers_pairing: f64,
    _memory: Reservation,
}
pub fn eigen(matrix: Matrix) -> Result<(Vec<f64>, Matrix)> {
    if matrix
        .iter()
        .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(invalid("nonfinite Hermitian matrix"));
    }
    let spectrum = SymmetricEigen::try_new(matrix, 1e-14, 10000)
        .ok_or_else(|| crate::Error::Kernel("Hermitian eigensolver did not converge".into()))?;
    let n = spectrum.eigenvalues.len();
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&a, &b| spectrum.eigenvalues[a].total_cmp(&spectrum.eigenvalues[b]));
    let energies = order.iter().map(|&j| spectrum.eigenvalues[j]).collect();
    let mut c = Matrix::from_fn(n, n, |i, j| spectrum.eigenvectors[(i, order[j])]);
    for j in 0..n {
        let pivot = (0..n)
            .max_by(|&a, &b| c[(a, j)].norm().total_cmp(&c[(b, j)].norm()))
            .unwrap();
        let phase = c[(pivot, j)].conj() / c[(pivot, j)].norm();
        for i in 0..n {
            c[(i, j)] *= phase;
        }
    }
    Ok((energies, c))
}
fn density(c: &Matrix, n: usize) -> Matrix {
    let occupied = c.columns(0, n);
    occupied * occupied.adjoint()
}
fn hermitian(m: Matrix) -> Matrix {
    (&m + m.adjoint()) * C64::new(0.5, 0.)
}
fn energy(d: &[Matrix], f: &[Matrix], h: &Matrix, method: &str) -> f64 {
    let multiplier = if method == "rhf" { 1. } else { 0.5 };
    d.iter()
        .zip(f)
        .map(|(d, f)| {
            d.iter()
                .enumerate()
                .map(|(k, d)| {
                    let i = k % h.nrows();
                    let j = k / h.nrows();
                    (d * (h[(j, i)] + f[(j, i)])).re
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        * multiplier
}
fn jk(integrals: &MolecularIntegrals, d: &Matrix) -> (Matrix, Matrix) {
    let m = integrals.m();
    let mut j = Matrix::zeros(m, m);
    let mut k = Matrix::zeros(m, m);
    if let super::integrals::Eri::Cholesky { rank, factors, .. } = &integrals.eri {
        for column in 0..*rank {
            let b = Matrix::from_fn(m, m, |p, q| {
                C64::new(
                    factors.as_slice()[super::integrals::pair(p, q) * rank + column],
                    0.,
                )
            });
            let weight = (d * &b).trace();
            j += &b * weight;
            k += &b * d * &b;
        }
        return (j, k);
    }
    for p in 0..m {
        for q in 0..m {
            for r in 0..m {
                for s in 0..m {
                    j[(p, q)] += d[(s, r)] * integrals.eri.value(p, r, q, s);
                    k[(p, q)] += d[(s, r)] * integrals.eri.value(p, r, s, q);
                }
            }
        }
    }
    (j, k)
}
fn fock(integrals: &MolecularIntegrals, h: &Matrix, d: &[Matrix], method: &str) -> Vec<Matrix> {
    let m = integrals.m();
    if method == "rhf" {
        let (j, k) = jk(integrals, &d[0]);
        return vec![hermitian(h + j * C64::new(2., 0.) - k)];
    }
    if method == "uhf" {
        let (j, k_a) = jk(integrals, &d[0]);
        let (j_b, k_b) = jk(integrals, &d[1]);
        return vec![hermitian(h + &j + &j_b - k_a), hermitian(h + j + j_b - k_b)];
    }
    let d = &d[0];
    let total = d.view((0, 0), (m, m)).into_owned() + d.view((m, m), (m, m));
    let (j, _) = jk(integrals, &total);
    let mut f = h.clone();
    for alpha in 0..2 {
        for beta in 0..2 {
            let (_, exchange) = jk(
                integrals,
                &d.view((alpha * m, beta * m), (m, m)).into_owned(),
            );
            for p in 0..m {
                for q in 0..m {
                    f[(alpha * m + p, beta * m + q)] -= exchange[(p, q)];
                    if alpha == beta {
                        f[(alpha * m + p, beta * m + q)] += j[(p, q)];
                    }
                }
            }
        }
    }
    vec![hermitian(f)]
}
struct Diis {
    focks: Vec<Vec<Matrix>>,
    errors: Vec<Vec<Matrix>>,
}
impl Diis {
    fn new() -> Self {
        Self {
            focks: vec![],
            errors: vec![],
        }
    }
    fn extrapolate(&mut self, f: &[Matrix], d: &[Matrix]) -> Vec<Matrix> {
        let errors: Vec<_> = f.iter().zip(d).map(|(f, d)| f * d - d * f).collect();
        self.errors.push(errors);
        self.focks.push(f.to_vec());
        if self.focks.len() > 8 {
            self.focks.remove(0);
            self.errors.remove(0);
        }
        let n = self.focks.len();
        if n < 2 {
            return f.to_vec();
        }
        let mut b = DMatrix::<f64>::zeros(n + 1, n + 1);
        for i in 0..n {
            for j in 0..n {
                b[(i, j)] = self.errors[i]
                    .iter()
                    .zip(&self.errors[j])
                    .map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a.conj() * b).re).sum::<f64>())
                    .sum();
            }
            b[(i, n)] = -1.;
            b[(n, i)] = -1.;
        }
        let mut rhs = DVector::zeros(n + 1);
        rhs[n] = -1.;
        let Some(weights) = b.lu().solve(&rhs) else {
            return f.to_vec();
        };
        if weights.iter().any(|w| !w.is_finite() || w.abs() > 1e12) {
            return f.to_vec();
        }
        (0..f.len())
            .map(|spin| {
                self.focks
                    .iter()
                    .zip(weights.iter())
                    .map(|(f, w)| &f[spin] * C64::new(*w, 0.))
                    .fold(Matrix::zeros(f[spin].nrows(), f[spin].ncols()), |a, b| {
                        a + b
                    })
            })
            .collect()
    }
}
struct Candidate {
    energy: f64,
    converged: bool,
    history: Vec<Iteration>,
    coefficients: Vec<Matrix>,
    densities: Vec<Matrix>,
    energies: Vec<Vec<f64>>,
}
fn iterate(
    integrals: &MolecularIntegrals,
    method: &str,
    h: &Matrix,
    particles: [usize; 2],
    mut c: Vec<Matrix>,
    options: &ScfOptions,
) -> Result<Candidate> {
    let occupations = if method == "ghf" {
        vec![particles.iter().sum()]
    } else if method == "rhf" {
        vec![particles[0]]
    } else {
        particles.to_vec()
    };
    let mut d: Vec<_> = c
        .iter()
        .zip(&occupations)
        .map(|(c, &n)| density(c, n))
        .collect();
    let mut previous = None;
    let mut converged = false;
    let mut history = Vec::new();
    let mut mixer = Diis::new();
    let mut shift = 0.;
    for iteration in 1..=options.max_iterations {
        let f = fock(integrals, h, &d, method);
        let current = energy(&d, &f, h, method);
        if !current.is_finite() {
            return Err(crate::Error::Kernel("nonfinite SCF energy".into()));
        }
        if previous.is_some_and(|e| current > e + 1e-10) && shift == 0. {
            shift = 0.5;
        }
        let mut effective = if options.diis {
            mixer.extrapolate(&f, &d)
        } else {
            f
        };
        for (f, d) in effective.iter_mut().zip(&d) {
            if shift > 0. {
                *f += (Matrix::identity(h.nrows(), h.nrows()) - d) * C64::new(shift, 0.);
            }
        }
        c = effective
            .into_iter()
            .map(|f| eigen(hermitian(f)).map(|(_, c)| c))
            .collect::<Result<_>>()?;
        let next: Vec<_> = c
            .iter()
            .zip(&occupations)
            .map(|(c, &n)| density(c, n))
            .collect();
        let residual = d
            .iter()
            .zip(&next)
            .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).norm()))
            .fold(0f64, f64::max);
        let delta = previous.map(|energy| current - energy);
        history.push(Iteration {
            iteration,
            energy: current,
            delta,
            residual,
        });
        d = next;
        previous = Some(current);
        if delta.is_some_and(|de: f64| de.abs() < options.tolerance) && residual < options.tolerance
        {
            converged = true;
            break;
        }
    }
    let final_f = fock(integrals, h, &d, method);
    let final_energy = energy(&d, &final_f, h, method);
    let mut energies = Vec::new();
    for (spin, f) in final_f.into_iter().enumerate() {
        let (eps, canonical) = eigen(f)?;
        energies.push(eps);
        if method != "rhf" {
            c[spin] = canonical;
        }
    }
    Ok(Candidate {
        energy: final_energy,
        converged,
        history,
        coefficients: c,
        densities: d,
        energies,
    })
}
fn collinear(ca: &Matrix, cb: &Matrix, particles: [usize; 2]) -> Matrix {
    let m = ca.nrows();
    let mut out = Matrix::zeros(2 * m, 2 * m);
    let mut column = 0;
    for occupied in [true, false] {
        for (spin, (c, &n)) in [ca, cb].into_iter().zip(&particles).enumerate() {
            let range = if occupied { 0..n } else { n..m };
            for j in range {
                for i in 0..m {
                    out[(spin * m + i, column)] = c[(i, j)];
                }
                column += 1;
            }
        }
    }
    out
}
impl MolecularIntegrals {
    pub fn mean_field(
        &self,
        method: &str,
        particles: [usize; 2],
        options: &ScfOptions,
    ) -> Result<MeanField> {
        self.context
            .run(|| self.mean_field_inner(method, particles, options))
    }
    fn mean_field_inner(
        &self,
        method: &str,
        particles: [usize; 2],
        options: &ScfOptions,
    ) -> Result<MeanField> {
        let m = self.m();
        let ctx = &self.context;
        if !["rhf", "uhf", "ghf"].contains(&method)
            || particles.iter().any(|n| *n > m)
            || (method == "rhf" && particles[0] != particles[1])
            || options.max_iterations == 0
            || !options.tolerance.is_finite()
            || options.tolerance <= 0.
        {
            return Err(invalid(
                "invalid SCF method, occupation, iteration budget or tolerance",
            ));
        }
        let mut workspace = ctx.reserve(
            product(&[2 * m, 2 * m, 16, 160])?
                .checked_add(product(&[options.max_iterations, 128, 4])?)
                .ok_or(crate::Error::Overflow)?,
        )?;
        let spatial = Matrix::from_fn(m, m, |i, j| C64::new(self.one.as_slice()[i * m + j], 0.));
        let h = if method == "ghf" {
            Matrix::from_fn(2 * m, 2 * m, |i, j| {
                if i / m == j / m {
                    spatial[(i % m, j % m)]
                } else {
                    C64::default()
                }
            })
        } else {
            spatial.clone()
        };
        let (_, core) = eigen(h.clone())?;
        let mut starts = vec![if method == "uhf" {
            vec![core.clone(), core.clone()]
        } else {
            vec![core.clone()]
        }];
        if options.guesses >= 2 {
            let screened = if method == "uhf" {
                let d = density(&core, particles[0]) + density(&core, particles[1]);
                let (j, _) = jk(self, &d);
                &h + j * C64::new(0.5, 0.)
            } else {
                let d = density(
                    &core,
                    if method == "rhf" {
                        particles[0]
                    } else {
                        particles.iter().sum()
                    },
                );
                let f = fock(self, &h, &[d], method).remove(0);
                (&h + f) * C64::new(0.5, 0.)
            };
            let (_, c) = eigen(hermitian(screened))?;
            starts.push(if method == "uhf" {
                vec![c.clone(), c]
            } else {
                vec![c]
            });
        }
        if method == "uhf"
            && options.guesses >= 3
            && particles[0] == particles[1]
            && particles[0] > 0
            && particles[0] < m
        {
            let i = particles[0] - 1;
            let a = particles[0];
            let mut ca = core.clone();
            let mut cb = core.clone();
            for k in 0..m {
                ca[(k, i)] = core[(k, i)] * 0.3f64.cos() + core[(k, a)] * 0.3f64.sin();
                ca[(k, a)] = -core[(k, i)] * 0.3f64.sin() + core[(k, a)] * 0.3f64.cos();
                cb[(k, i)] = core[(k, i)] * 0.3f64.cos() - core[(k, a)] * 0.3f64.sin();
                cb[(k, a)] = core[(k, i)] * 0.3f64.sin() + core[(k, a)] * 0.3f64.cos();
            }
            starts.push(vec![ca, cb]);
        }
        if method == "ghf" && options.guesses >= 3 {
            let uhf = self.mean_field_inner("uhf", particles, options)?;
            starts.push(vec![collinear(
                &uhf.coefficients[0],
                &uhf.coefficients[1],
                particles,
            )]);
        }
        let mut best: Option<Candidate> = None;
        for start in starts {
            let candidate = iterate(self, method, &h, particles, start, options)?;
            if best.as_ref().is_none_or(|best| {
                (candidate.converged && !best.converged)
                    || (candidate.converged == best.converged
                        && candidate.energy < best.energy - 1e-12)
            }) {
                best = Some(candidate);
            }
        }
        let best = best.unwrap();
        let mut natural_occupations = vec![];
        let model_orbitals = if method == "uhf" {
            let (mut occupations, c) = eigen(&best.densities[0] + &best.densities[1])?;
            occupations.reverse();
            natural_occupations = occupations;
            Matrix::from_fn(m, m, |i, j| c[(i, m - 1 - j)])
        } else if method == "ghf" {
            Matrix::from_fn(2 * m, 2 * m, |i, j| {
                best.coefficients[0][(i, if j < m { 2 * j } else { 2 * (j - m) + 1 })]
            })
        } else {
            best.coefficients[0].clone()
        };
        let n = model_orbitals.ncols();
        let transformed_h = model_orbitals.adjoint() * &h * &model_orbitals;
        let mut one = ctx.allocate::<C64>(product(&[n, n])?)?;
        for i in 0..n {
            for j in 0..n {
                one.as_mut_slice()[i * n + j] = transformed_h[(i, j)];
            }
        }
        let two = if method == "ghf" {
            let mut result = ctx.allocate::<C64>(product(&[n, n, n, n])?)?;
            for alpha in 0..2 {
                for beta in 0..2 {
                    let ca = model_orbitals.rows(alpha * m, m).into_owned();
                    let cb = model_orbitals.rows(beta * m, m).into_owned();
                    let block = transform::four(ctx, &self.eri, [&ca, &cb, &ca, &cb])?;
                    for (a, b) in result.as_mut_slice().iter_mut().zip(block.as_slice()) {
                        *a += b;
                    }
                }
            }
            result
        } else {
            transform::four(ctx, &self.eri, [&model_orbitals; 4])?
        };
        let reference_energy = if method == "uhf" {
            let d = vec![
                density(&model_orbitals, particles[0]),
                density(&model_orbitals, particles[1]),
            ];
            energy(&d, &fock(self, &h, &d, method), &h, method)
        } else {
            best.energy
        };
        let model_particles = if method == "ghf" {
            let n: usize = particles.iter().sum();
            [n.div_ceil(2), n / 2]
        } else {
            particles
        };
        let kramers_pairing = if method == "ghf" {
            (0..particles.iter().sum::<usize>() / 2)
                .map(|i| best.energies[0][2 * i + 1] - best.energies[0][2 * i])
                .fold(0f64, f64::max)
        } else {
            0.
        };
        let model = Model::new(
            ctx,
            m,
            model_particles,
            one,
            two,
            self.nuclear,
            method == "ghf",
        )?;
        workspace.shrink_to(
            product(&[2 * m, 2 * m, 16, 24])?
                .checked_add(product(&[options.max_iterations, 128])?)
                .ok_or(crate::Error::Overflow)?,
        )?;
        Ok(MeanField {
            method: method.into(),
            electronic_energy: best.energy,
            reference_energy,
            converged: best.converged,
            iterations: best.history,
            particles: model_particles,
            energies: best.energies,
            coefficients: best.coefficients,
            densities: best.densities,
            natural_occupations,
            model_orbitals,
            model,
            kramers_pairing,
            _memory: workspace,
        })
    }
}
