//! RHF, UHF and complex spinor GHF with bounded Pulay DIIS and level shifting.
use super::{
    invalid,
    transform::{self, Model},
    MolecularIntegrals,
};
use crate::math::linalg::eigen;
use crate::quantum::C64;
use crate::{product, Reservation, Result};
use nalgebra::{DMatrix, DVector};
use std::sync::{Arc, Mutex};

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
    model: Mutex<Option<Arc<Model>>>,
    pub kramers_pairing: f64,
    pub functional: Option<String>,
    pub hartree_energy: f64,
    pub xc_energy: f64,
    _memory: Reservation,
}
pub(super) fn density(c: &Matrix, n: usize) -> Matrix {
    let occupied = c.columns(0, n);
    occupied * occupied.adjoint()
}
pub(super) fn hermitian(m: Matrix) -> Matrix {
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
pub(super) fn jk(eri: &super::integrals::Eri, m: usize, d: &Matrix) -> (Matrix, Matrix) {
    // Collinear real densities stay in real arithmetic. No imaginary part
    // is discarded: genuinely complex and spinor densities use the complex
    // path below. Pair factors are intrinsically real in this engine.
    if d.iter().all(|v| v.im == 0.) {
        let (j, k) = real_jk(eri, m, &d.map(|v| v.re));
        return (j.map(|v| C64::new(v, 0.)), k.map(|v| C64::new(v, 0.)));
    }
    if let super::integrals::Eri::Augmented {
        base, correction, ..
    } = eri
    {
        let (mut j, mut k) = jk(base, m, d);
        for (factor, &weight) in correction.factors.iter().zip(&correction.weights) {
            let b = factor.map(|v| C64::new(v, 0.));
            let charge = (0..m)
                .flat_map(|p| (0..m).map(move |q| (p, q)))
                .map(|(p, q)| d[(p, q)] * b[(q, p)])
                .sum::<C64>();
            j += &b * (charge * weight);
            k += &b * d * &b * C64::new(weight, 0.);
        }
        return (j, k);
    }
    let mut j = Matrix::zeros(m, m);
    let mut k = Matrix::zeros(m, m);
    if let super::integrals::Eri::Cholesky { rank, factors, .. } = eri {
        for column in 0..*rank {
            let b = Matrix::from_fn(m, m, |p, q| {
                C64::new(
                    factors.as_slice()[super::integrals::pair(p, q) * rank + column],
                    0.,
                )
            });
            let weight = (0..m)
                .flat_map(|p| (0..m).map(move |q| (p, q)))
                .map(|(p, q)| d[(p, q)] * b[(q, p)])
                .sum::<C64>();
            j += &b * weight;
            k += &b * d * &b;
        }
        return (j, k);
    }
    for p in 0..m {
        for q in 0..m {
            for r in 0..m {
                for s in 0..m {
                    j[(p, q)] += d[(s, r)] * eri.value(p, r, q, s);
                    k[(p, q)] += d[(s, r)] * eri.value(p, r, s, q);
                }
            }
        }
    }
    (j, k)
}
fn real_jk(
    eri: &super::integrals::Eri,
    m: usize,
    d: &DMatrix<f64>,
) -> (DMatrix<f64>, DMatrix<f64>) {
    use super::integrals::{pair, Eri};
    if let Eri::Augmented {
        base, correction, ..
    } = eri
    {
        let (mut j, mut k) = real_jk(base, m, d);
        for (b, &weight) in correction.factors.iter().zip(&correction.weights) {
            let charge = (0..m)
                .flat_map(|p| (0..m).map(move |q| (p, q)))
                .map(|(p, q)| d[(p, q)] * b[(q, p)])
                .sum::<f64>();
            j += b * (charge * weight);
            k += b * d * b * weight;
        }
        return (j, k);
    }
    let mut j = DMatrix::zeros(m, m);
    let mut k = DMatrix::zeros(m, m);
    if let Eri::Cholesky { rank, factors, .. } = eri {
        for column in 0..*rank {
            let b = DMatrix::from_fn(m, m, |p, q| factors.as_slice()[pair(p, q) * rank + column]);
            let charge = (0..m)
                .flat_map(|p| (0..m).map(move |q| (p, q)))
                .map(|(p, q)| d[(p, q)] * b[(q, p)])
                .sum::<f64>();
            j += &b * charge;
            k += &b * d * &b;
        }
        return (j, k);
    }
    for p in 0..m {
        for q in 0..m {
            for r in 0..m {
                for s in 0..m {
                    j[(p, q)] += d[(s, r)] * eri.value(p, r, q, s);
                    k[(p, q)] += d[(s, r)] * eri.value(p, r, s, q);
                }
            }
        }
    }
    (j, k)
}
fn fock(eri: &super::integrals::Eri, h: &Matrix, d: &[Matrix], method: &str) -> Vec<Matrix> {
    let m = eri.m();
    if method == "rhf" {
        let (j, k) = jk(eri, m, &d[0]);
        return vec![hermitian(h + j * C64::new(2., 0.) - k)];
    }
    if method == "uhf" {
        let (j, k_a) = jk(eri, m, &d[0]);
        let (j_b, k_b) = jk(eri, m, &d[1]);
        return vec![hermitian(h + &j + &j_b - k_a), hermitian(h + j + j_b - k_b)];
    }
    let d = &d[0];
    let total = d.view((0, 0), (m, m)).into_owned() + d.view((m, m), (m, m));
    let (j, _) = jk(eri, m, &total);
    let mut f = h.clone();
    for alpha in 0..2 {
        for beta in 0..2 {
            let (_, exchange) = jk(eri, m, &d.view((alpha * m, beta * m), (m, m)).into_owned());
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
pub(super) struct Diis {
    focks: Vec<Vec<Matrix>>,
    errors: Vec<Vec<Matrix>>,
}
impl Diis {
    pub(super) fn new() -> Self {
        Self {
            focks: vec![],
            errors: vec![],
        }
    }
    pub(super) fn extrapolate(&mut self, f: &[Matrix], d: &[Matrix]) -> Vec<Matrix> {
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
pub(super) struct Candidate {
    pub(super) energy: f64,
    pub(super) converged: bool,
    pub(super) history: Vec<Iteration>,
    pub(super) coefficients: Vec<Matrix>,
    pub(super) densities: Vec<Matrix>,
    pub(super) energies: Vec<Vec<f64>>,
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
        let f = fock(integrals.eri()?.as_ref(), h, &d, method);
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
    let final_f = fock(integrals.eri()?.as_ref(), h, &d, method);
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
        let workspace = ctx.reserve(
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
                let (j, _) = jk(self.eri()?.as_ref(), m, &d);
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
                let f = fock(self.eri()?.as_ref(), &h, &[d], method).remove(0);
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
        self.finish(method, particles, options, best, workspace, None)
    }
    pub(super) fn finish(
        &self,
        method: &str,
        particles: [usize; 2],
        options: &ScfOptions,
        best: Candidate,
        mut workspace: Reservation,
        xc: Option<(&str, f64, f64)>,
    ) -> Result<MeanField> {
        let m = self.m();
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
            spatial
        };
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
        let reference_energy = if method == "uhf" {
            let d = vec![
                density(&model_orbitals, particles[0]),
                density(&model_orbitals, particles[1]),
            ];
            energy(&d, &fock(self.eri()?.as_ref(), &h, &d, method), &h, method)
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
            model: Mutex::new(None),
            kramers_pairing,
            functional: xc.map(|v| v.0.to_string()),
            hartree_energy: xc.map_or(0., |v| v.1),
            xc_energy: xc.map_or(0., |v| v.2),
            _memory: workspace,
        })
    }
}

impl MeanField {
    pub fn has_model(&self) -> bool {
        self.model
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }
    pub fn model(&self, integrals: &MolecularIntegrals) -> Result<Arc<Model>> {
        let mut cache = self.model.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(model) = cache.as_ref() {
            return Ok(model.clone());
        }
        let ctx = &integrals.context;
        let m = integrals.m();
        let c = &self.model_orbitals;
        let n = c.ncols();
        let eri = integrals.eri()?;
        let spatial = Matrix::from_fn(m, m, |i, j| {
            C64::new(integrals.one.as_slice()[i * m + j], 0.)
        });
        let h = if self.method == "ghf" {
            Matrix::from_fn(2 * m, 2 * m, |i, j| {
                if i / m == j / m {
                    spatial[(i % m, j % m)]
                } else {
                    C64::default()
                }
            })
        } else {
            spatial
        };
        let transformed = c.adjoint() * h * c;
        let mut one = ctx.allocate::<C64>(product(&[n, n])?)?;
        for i in 0..n {
            for j in 0..n {
                one.as_mut_slice()[i * n + j] = transformed[(i, j)];
            }
        }
        let two = if self.method == "ghf" {
            let mut out = ctx.allocate::<C64>(product(&[n, n, n, n])?)?;
            for alpha in 0..2 {
                for beta in 0..2 {
                    let ca = c.rows(alpha * m, m).into_owned();
                    let cb = c.rows(beta * m, m).into_owned();
                    let block = transform::four(ctx, &eri, [&ca, &cb, &ca, &cb])?;
                    for (a, b) in out.as_mut_slice().iter_mut().zip(block.as_slice()) {
                        *a += b;
                    }
                }
            }
            out
        } else {
            transform::four(ctx, &eri, [c; 4])?
        };
        let model = Arc::new(Model::new(
            ctx,
            m,
            self.particles,
            one,
            two,
            integrals.nuclear,
            self.method == "ghf",
        )?);
        *cache = Some(model.clone());
        Ok(model)
    }
}
