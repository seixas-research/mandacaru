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
    /// Occupations (0 to 1 per spin orbital) of each channel's orbitals, in the order of
    /// `energies`: aufbau for Hartree-Fock, Fermi-Dirac for Kohn-Sham
    /// ([`super::occupation::fermi_dirac`]), which is exactly 0 or 1 on a gapped molecule.
    pub occupations: Vec<Vec<f64>>,
    /// Kohn-Sham: `-width S` of the occupations, in Hartree (0 when they are integer). The
    /// Mermin free energy, whose derivative the analytic forces are, is
    /// `electronic_energy + entropy_energy`.
    pub entropy_energy: f64,
    /// Kohn-Sham: the Fermi-Dirac width of the occupations, in Hartree.
    pub smearing_width: Option<f64>,
    /// Hartree by which an occupied level lies above an empty one when the Kohn-Sham state
    /// is not aufbau (a near-degenerate frontier off a symmetry axis); `None` when it is.
    pub aufbau_inversion: Option<f64>,
    pub natural_occupations: Vec<f64>,
    pub model_orbitals: Matrix,
    /// Why the degenerate clusters of `model_orbitals` could not be put into a deterministic
    /// frame ([`super::frame`]); a model or active space built on them refuses.
    pub(super) frame_error: Option<String>,
    model: Mutex<Option<Arc<Model>>>,
    pub kramers_pairing: f64,
    pub functional: Option<String>,
    pub hartree_energy: f64,
    pub xc_energy: f64,
    pub hubbard_energy: Option<f64>,
    pub hubbard: Option<Vec<crate::molecular::hubbard::MolecularHubbardReport>>,
    _memory: Reservation,
}
pub(super) fn density(c: &Matrix, n: usize) -> Matrix {
    let occupied = c.columns(0, n);
    occupied * occupied.adjoint()
}
pub(super) fn hermitian(m: Matrix) -> Matrix {
    (&m + m.adjoint()) * C64::new(0.5, 0.)
}
pub(super) fn energy(d: &[Matrix], f: &[Matrix], h: &Matrix, method: &str) -> f64 {
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
    /// For RHF, the (occupied, empty) columns of an occupation that is not aufbau: an occupied
    /// orbital whose unshifted Fock energy lies above an empty one's by more than
    /// [`AUFBAU_TOLERANCE`]. Such a stationary point is not a minimum (a level shift can hold
    /// one), so the candidate is not converged.
    pub(super) unstable_pair: Option<(usize, usize)>,
    /// Occupations of each channel's columns when they are not aufbau (a Kohn-Sham ensemble of
    /// a degenerate shell); `None` for the aufbau determinant.
    pub(super) occupations: Option<Vec<Vec<f64>>>,
    /// Kohn-Sham: Hartree by which an occupied level lies above an empty one, when the state
    /// is not aufbau (kept and labeled, see `kohn_sham`); `None` for an aufbau state.
    pub(super) aufbau_inversion: Option<f64>,
    /// Kohn-Sham: `-width S` of the Fermi-Dirac occupations (0 for integer ones); the Mermin
    /// free energy is `energy + entropy`.
    pub(super) entropy: f64,
}
/// The largest entry of the commutators `[F, D]` of each spin channel (orthonormal basis): the
/// orbital gradient, zero exactly at a stationary point. A small step (energy change and density
/// change) does not bound it, and an analytic force is first order in it, so convergence also
/// asks it to fall below the tolerance (owner decision 2026-11-09, "Add a stationarity test").
pub(super) fn stationarity(f: &[Matrix], d: &[Matrix]) -> f64 {
    f.iter()
        .zip(d)
        .flat_map(|(f, d)| (f * d - d * f).iter().map(|v| v.norm()).collect::<Vec<_>>())
        .fold(0f64, f64::max)
}
/// Hartree by which an occupied RHF orbital may lie above an empty one (a degenerate shell at
/// the Fermi level passes).
pub const AUFBAU_TOLERANCE: f64 = 1e-6;
/// The highest occupied and the lowest empty column of `c` by the diagonal of `f`, when the
/// first lies above the second by more than [`AUFBAU_TOLERANCE`].
fn non_aufbau(f: &Matrix, c: &Matrix, occupied: usize) -> Option<(usize, usize)> {
    let level = |j: usize| (c.column(j).adjoint() * f * c.column(j))[(0, 0)].re;
    let n = c.ncols();
    if occupied == 0 || occupied >= n {
        return None;
    }
    let (i, high) = (0..occupied)
        .map(|j| (j, level(j)))
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    let (a, low) = (occupied..n)
        .map(|j| (j, level(j)))
        .min_by(|a, b| a.1.total_cmp(&b.1))?;
    (high > low + AUFBAU_TOLERANCE).then_some((i, a))
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
        let gradient = stationarity(&f, &d);
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
        if delta.is_some_and(|de: f64| de.abs() < options.tolerance)
            && residual < options.tolerance
            && gradient < options.tolerance
        {
            converged = true;
            break;
        }
    }
    let final_f = fock(integrals.eri()?.as_ref(), h, &d, method);
    let final_energy = energy(&d, &final_f, h, method);
    let unstable_pair = if method == "rhf" {
        non_aufbau(&final_f[0], &c[0], occupations[0])
    } else {
        None
    };
    converged &= unstable_pair.is_none();
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
        unstable_pair,
        occupations: None,
        aufbau_inversion: None,
        entropy: 0.,
    })
}
/// `c` with columns `i` and `a` rotated into each other by `angle` (radians).
fn rotated(c: &Matrix, i: usize, a: usize, angle: f64) -> Matrix {
    let mut out = c.clone();
    let (cos, sin) = (angle.cos(), angle.sin());
    for k in 0..c.nrows() {
        out[(k, i)] = c[(k, i)] * cos + c[(k, a)] * sin;
        out[(k, a)] = -c[(k, i)] * sin + c[(k, a)] * cos;
    }
    out
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
    /// Follow the same collinear stationary state under a small geometry
    /// displacement. The old orthonormal orbitals are a starting guess;
    /// Fock matrices and convergence tests use the new integrals throughout.
    pub fn continue_mean_field(
        &self,
        previous: &MeanField,
        options: &ScfOptions,
    ) -> Result<MeanField> {
        let m = self.m();
        let method = previous.method.as_str();
        let channels = if method == "rhf" { 1 } else { 2 };
        if !["rhf", "uhf"].contains(&method)
            || previous.functional.is_some()
            || !previous.converged
            || previous.coefficients.len() != channels
            || previous.particles.iter().any(|p| *p > m)
            || options.max_iterations == 0
            || !options.tolerance.is_finite()
            || options.tolerance <= 0.
            || previous.coefficients.iter().any(|c| {
                c.shape() != (m, m)
                    || c.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
                    || (c.adjoint() * c - Matrix::identity(m, m))
                        .iter()
                        .any(|v| v.norm() > 1e-8)
            })
        {
            return Err(invalid("SCF continuation needs a converged collinear HF frame with matching orthonormal orbitals and valid controls"));
        }
        self.context.run(|| {
            // Factorization and SCF mixing do not run simultaneously. Release
            // the factorization's FFT/column scratch before reserving DIIS
            // workspace, rather than charging both peaks at the same time.
            self.eri()?;
            let workspace = self.context.reserve(
                product(&[2 * m, 2 * m, 16, 160])?
                    .checked_add(product(&[options.max_iterations, 128, 4])?)
                    .ok_or(crate::Error::Overflow)?,
            )?;
            let h = Matrix::from_fn(m, m, |i, j| C64::new(self.one.as_slice()[i * m + j], 0.));
            let candidate = iterate(
                self,
                method,
                &h,
                previous.particles,
                previous.coefficients.clone(),
                options,
            )?;
            self.finish(
                method,
                previous.particles,
                options,
                candidate,
                workspace,
                None,
                None,
            )
        })
    }
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
        if self.paw_setup.as_ref().is_some_and(|s| s.has_spin_orbit()) {
            return Err(crate::Error::Unsupported(
                "Hartree-Fock of a molecule with spin-orbit datasets is not available: the spin-orbit \
                 term is carried by spinor Kohn-Sham (spinor_kohn_sham) only"
                    .into(),
            ));
        }
        self.eri()?;
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
        // A converged candidate beats an unconverged one; then the lower energy wins.
        let better = |candidate: &Candidate, best: Option<&Candidate>| {
            best.is_none_or(|best| {
                (candidate.converged && !best.converged)
                    || (candidate.converged == best.converged
                        && candidate.energy < best.energy - 1e-12)
            })
        };
        let mut best: Option<Candidate> = None;
        let mut restarts = Vec::new();
        for start in starts {
            let candidate = iterate(self, method, &h, particles, start, options)?;
            // A non-aufbau RHF stationary point (held by the level shift) is a saddle: restart
            // once from it with the offending pair mixed, the same 0.3 rad the UHF start uses,
            // and the shift off. The selector keeps whichever is lower and converged.
            if let Some((i, a)) = candidate.unstable_pair {
                restarts.push(vec![rotated(&candidate.coefficients[0], i, a, 0.3)]);
            }
            if better(&candidate, best.as_ref()) {
                best = Some(candidate);
            }
        }
        for start in restarts {
            let candidate = iterate(self, method, &h, particles, start, options)?;
            if better(&candidate, best.as_ref()) {
                best = Some(candidate);
            }
        }
        let best = best.unwrap();
        self.finish(method, particles, options, best, workspace, None, None)
    }
    pub(super) fn finish(
        &self,
        method: &str,
        particles: [usize; 2],
        options: &ScfOptions,
        best: Candidate,
        mut workspace: Reservation,
        xc: Option<(&str, f64, f64)>,
        hubbard: Option<(f64, Vec<crate::molecular::hubbard::MolecularHubbardReport>)>,
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
        let reference_energy = if method == "uhf" && xc.is_none() {
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
        let occupations = match &best.occupations {
            Some(occupations) => occupations.clone(),
            None => {
                let filled: Vec<usize> = match method {
                    "ghf" => vec![particles.iter().sum()],
                    "rhf" => vec![particles[0]],
                    _ => particles.to_vec(),
                };
                best.energies
                    .iter()
                    .zip(filled)
                    .map(|(levels, n)| super::occupation::aufbau(levels.len(), n))
                    .collect()
            }
        };
        workspace.shrink_to(
            product(&[2 * m, 2 * m, 16, 24])?
                .checked_add(product(&[options.max_iterations, 128])?)
                .ok_or(crate::Error::Overflow)?,
        )?;
        let mut state = MeanField {
            method: method.into(),
            electronic_energy: best.energy,
            reference_energy,
            converged: best.converged,
            iterations: best.history,
            particles: model_particles,
            energies: best.energies,
            coefficients: best.coefficients,
            densities: best.densities,
            occupations,
            aufbau_inversion: best.aufbau_inversion,
            entropy_energy: best.entropy,
            smearing_width: None,
            natural_occupations,
            model_orbitals,
            frame_error: None,
            model: Mutex::new(None),
            kramers_pairing,
            functional: xc.map(|v| v.0.to_string()),
            hartree_energy: xc.map_or(0., |v| v.1),
            xc_energy: xc.map_or(0., |v| v.2),
            hubbard_energy: hubbard.as_ref().map(|v| v.0),
            hubbard: hubbard.map(|v| v.1),
            _memory: workspace,
        };
        state.fix_frame(self)?;
        Ok(state)
    }
}

impl MolecularIntegrals {
    /// Electronic Hartree-Fock energy of orthonormal-frame spin densities,
    /// which need not be idempotent. Nuclear repulsion is not included.
    pub fn hartree_fock_energy(&self, method: &str, d: &[Matrix]) -> Result<f64> {
        let (h, f) = self.hartree_fock_operators(method, d)?;
        Ok(energy(d, &f, &h, method))
    }
    /// The core operator and the Fock matrices (one per spin channel) of
    /// orthonormal-frame densities, for the collinear methods.
    pub(super) fn hartree_fock_operators(
        &self,
        method: &str,
        d: &[Matrix],
    ) -> Result<(Matrix, Vec<Matrix>)> {
        let m = self.m();
        let channels = if method == "rhf" { 1 } else { 2 };
        if !["rhf", "uhf"].contains(&method)
            || d.len() != channels
            || d.iter().any(|d| d.shape() != (m, m))
        {
            return Err(invalid(
                "Hartree-Fock energy needs collinear spin densities in the molecular basis",
            ));
        }
        let h = Matrix::from_fn(m, m, |i, j| C64::new(self.one.as_slice()[i * m + j], 0.));
        let f = fock(self.eri()?.as_ref(), &h, d, method);
        Ok((h, f))
    }
}
impl MeanField {
    /// How the state occupies its orbitals, as the run log writes it: `aufbau` (every
    /// occupation 0 or 1), `fractional (Fermi-Dirac, width W Ha, -TS X Ha)` (a Kohn-Sham state
    /// with a degenerate or nearly degenerate frontier), or `not aufbau (inversion X Ha)` when
    /// a level holds more than one below it, `X` Hartree lower (a state a level shift held).
    pub fn occupation_label(&self) -> String {
        match (self.is_ensemble(), self.aufbau_inversion) {
            (_, Some(inversion)) => format!("not aufbau (inversion {inversion:.3e} Ha)"),
            (true, None) => format!(
                "fractional (Fermi-Dirac, width {:.1e} Ha, -TS {:.3e} Ha)",
                self.smearing_width.unwrap_or(0.),
                self.entropy_energy
            ),
            (false, None) => "aufbau".into(),
        }
    }
    /// The electronic Mermin free energy `E - width S` (Hartree; `electronic_energy` when the
    /// occupations are integer).
    pub fn free_energy(&self) -> f64 {
        self.electronic_energy + self.entropy_energy
    }
    /// The total Mermin free energy, nuclear repulsion included (Hartree).
    pub fn total_free_energy(&self, integrals: &MolecularIntegrals) -> f64 {
        self.free_energy() + integrals.nuclear
    }
    /// Whether some occupation is fractional (Fermi-Dirac occupations of a degenerate or nearly
    /// degenerate frontier) rather than a determinant's 0 or 1.
    pub fn is_ensemble(&self) -> bool {
        !self
            .occupations
            .iter()
            .all(|o| super::occupation::is_integer(o))
    }
    /// An error for a consumer that needs a determinant (`what` names it) when the state is an
    /// ensemble; `Ok` otherwise.
    pub fn require_determinant(&self, what: &str) -> Result<()> {
        if self.is_ensemble() {
            return Err(crate::Error::Unsupported(format!(
                "{what} needs an integer-occupation reference, and this Kohn-Sham state has \
                 fractional occupations (a degenerate or nearly degenerate frontier shell, \
                 Fermi-Dirac with one Fermi level per spin); request the spin state explicitly \
                 instead: initial magnetic moments or unrestricted=True"
            )));
        }
        Ok(())
    }
    /// Electrons per orbital of each channel, in the order of `energies`: 0 to 2 for a
    /// spin-restricted channel, 0 to 1 otherwise (the ASE convention).
    pub fn occupation_numbers(&self) -> Vec<Vec<f64>> {
        let degeneracy = if self.method == "rhf" { 2. } else { 1. };
        self.occupations
            .iter()
            .map(|o| o.iter().map(|f| degeneracy * f).collect())
            .collect()
    }
    pub fn has_model(&self) -> bool {
        self.model
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }
    pub fn model(&self, integrals: &MolecularIntegrals) -> Result<Arc<Model>> {
        self.require_frame()?;
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
