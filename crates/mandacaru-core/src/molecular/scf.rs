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
    if let Some(rank) = eri.factor_count() {
        for column in 0..rank {
            let b = eri.factor(column).map(|v| C64::new(v, 0.));
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
    use super::integrals::Eri;
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
    if let Some(rank) = eri.factor_count() {
        for column in 0..rank {
            let b = eri.factor(column);
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
/// The density change below which two unconverged candidates are equally stationary, whatever
/// the tolerance asked: the rounding of a last SCF step.
const RESIDUAL_FLOOR: f64 = 1e-12;
impl Candidate {
    /// The largest density change of the last iteration: how far the run was from a fixed point
    /// when it stopped (infinite for a run that made no iteration).
    pub(super) fn final_residual(&self) -> f64 {
        self.history.last().map_or(f64::INFINITY, |i| i.residual)
    }
    /// Whether this candidate beats `best` among candidates of the same standing (the callers
    /// rank the standing first, converged above unconverged): converged ones by the lower free
    /// energy, `energy + entropy`, with a 1e-12 margin; unconverged ones, whose energy is that
    /// of an iterate still moving (one that oscillates can sit below a nearly stationary one),
    /// by the smaller final residual, a residual below `tolerance` (and below
    /// [`RESIDUAL_FLOOR`], the rounding of a last step) counting as that floor: two stationary
    /// points are not told apart by it, and the lower energy decides between them.
    pub(super) fn beats_at_equal_standing(&self, best: &Candidate, tolerance: f64) -> bool {
        if self.converged {
            self.energy + self.entropy < best.energy + best.entropy - 1e-12
        } else {
            let floor = tolerance.max(RESIDUAL_FLOOR);
            let (own, other) = (
                self.final_residual().max(floor),
                best.final_residual().max(floor),
            );
            own < other
                || (own == other && self.energy + self.entropy < best.energy + best.entropy - 1e-12)
        }
    }
    /// Whether this candidate replaces `best`: a converged candidate beats an unconverged one,
    /// then [`Self::beats_at_equal_standing`].
    pub(super) fn beats(&self, best: &Candidate, tolerance: f64) -> bool {
        (self.converged && !best.converged)
            || (self.converged == best.converged && self.beats_at_equal_standing(best, tolerance))
    }
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
/// The SCF from the orbitals `c`. The first `settle` iterations take the level-shifted Fock
/// without DIIS ([`SETTLE_SHIFT`]; a restart off a saddle); DIIS then starts from an empty history.
fn iterate(
    integrals: &MolecularIntegrals,
    method: &str,
    h: &Matrix,
    particles: [usize; 2],
    mut c: Vec<Matrix>,
    options: &ScfOptions,
    settle: usize,
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
        let settling = iteration <= settle;
        if settling {
            shift = SETTLE_SHIFT;
        } else if iteration == settle + 1 {
            shift = 0.;
        }
        if previous.is_some_and(|e| current > e + 1e-10) && shift == 0. {
            shift = 0.5;
        }
        let gradient = stationarity(&f, &d);
        let mut effective = if options.diis && !settling {
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
/// Orbital-Hessian sizes (real rotation parameters) up to which the lowest root is found by a
/// dense diagonalization of the operator (every symmetry block seen). Each column is one J/K
/// build, the cost of one SCF iteration, so the dense build costs at most this many SCF
/// iterations (measured, benzene PAW-LCAO SZ in an 8 A box, 225 rotations: 1.23 s, 5.5 ms a
/// column, against 3.5 s for the SCF; seeded Davidson 0.25 s, 45 applications, the same root);
/// above it, seeded Davidson ([`crate::math::davidson::lowest_root`]).
pub const DENSE_HESSIAN_LIMIT: usize = 256;
/// The residual norm a Davidson root of the orbital Hessian must reach.
const HESSIAN_TOLERANCE: f64 = 1e-7;

/// The orbital Hessian of a Hartree-Fock determinant (RHF, UHF or GHF) on its occupied-virtual
/// rotations, at the semicanonical orbitals of its density (the occupied and the empty space of
/// each channel's density, each diagonalizing the Fock there). A rotation `x_ai` of channel `s`
/// acts as `(eps_a - eps_i) x_ai + (c_a^+ G_s[D1] c_i)`, `D1_s = c_v x c_o^+ + h.c.`, `G` the
/// method's two-electron response (`2 J - K` for RHF, `J[D1_a + D1_b] - K[D1_s]` for UHF, the
/// spinor `J - K` for GHF). Real orbitals rotate by real `x` (real to real); complex orbitals and
/// GHF by complex `x`, whose real and imaginary parts are separate coordinates. A GHF determinant
/// is invariant under global spin rotations: those exact zero modes are projected out, so they
/// can be neither followed nor mistaken for a saddle. The energy's curvature along a unit mode is
/// `4 root` (RHF) or `2 root` (UHF, GHF).
pub struct OrbitalHessian {
    eri: Arc<super::integrals::Eri>,
    method: String,
    /// Semicanonical orbitals (occupied first), their levels, the occupied count, per channel.
    channels: Vec<(Matrix, Vec<f64>, usize)>,
    complex: bool,
    /// Orthonormal columns projected out of the operator.
    zero_modes: DMatrix<f64>,
    /// The occupied-virtual Fock blocks `c_a^+ F c_i` in rotation coordinates: the orbital
    /// gradient (zero at a stationary point); the energy's change along a rotation `x` is
    /// `4 g . x` (RHF) or `2 g . x` (UHF, GHF), so `H x = -g` is the Newton step.
    gradient: DVector<f64>,
}

impl OrbitalHessian {
    /// The Hessian of the determinant of `densities` (one per channel of `method`, orthonormal
    /// frame) under the core operator `h`; `None` when it has no rotation.
    fn new(
        eri: Arc<super::integrals::Eri>,
        h: &Matrix,
        method: &str,
        densities: &[Matrix],
        particles: [usize; 2],
    ) -> Result<Option<Self>> {
        let filled: Vec<usize> = match method {
            "rhf" => vec![particles[0]],
            "uhf" => particles.to_vec(),
            "ghf" => vec![particles.iter().sum()],
            _ => {
                return Err(invalid(
                    "an orbital Hessian needs an RHF, UHF or GHF determinant",
                ))
            }
        };
        if densities.len() != filled.len() {
            return Err(invalid("an orbital Hessian needs one density per channel"));
        }
        let f = fock(eri.as_ref(), h, densities, method);
        let complex = method == "ghf"
            || f.iter()
                .chain(densities)
                .any(|m| m.iter().any(|v| v.im != 0.));
        let eigenpairs = |m: &Matrix| -> Result<(Vec<f64>, Matrix)> {
            if m.nrows() == 0 {
                // A channel with no occupied or no empty orbital has nothing to diagonalize there.
                return Ok((Vec::new(), m.clone()));
            }
            if complex {
                eigen(hermitian(m.clone()))
            } else {
                let (values, vectors) = crate::math::linalg::real_eigen(m.map(|v| v.re))?;
                Ok((values, vectors.map(|v| C64::new(v, 0.))))
            }
        };
        let mut channels = Vec::with_capacity(filled.len());
        let mut blocks = Vec::with_capacity(filled.len());
        for ((d, f), &n) in densities.iter().zip(&f).zip(&filled) {
            let size = d.nrows();
            // The density's occupied space: its eigenvectors of the n largest occupations.
            let (_, natural) = eigenpairs(d)?;
            let occupied = Matrix::from_fn(size, n, |p, i| natural[(p, size - 1 - i)]);
            let empty = Matrix::from_fn(size, size - n, |p, a| natural[(p, size - 1 - n - a)]);
            let (eps_o, u_o) = eigenpairs(&(occupied.adjoint() * f * &occupied))?;
            let (eps_v, u_v) = eigenpairs(&(empty.adjoint() * f * &empty))?;
            let (occupied, empty) = (occupied * u_o, empty * u_v);
            let c = Matrix::from_fn(size, size, |p, q| {
                if q < n {
                    occupied[(p, q)]
                } else {
                    empty[(p, q - n)]
                }
            });
            blocks.push(c.columns(n, size - n).adjoint() * f * c.columns(0, n));
            channels.push((c, eps_o.into_iter().chain(eps_v).collect(), n));
        }
        let mut hessian = Self {
            eri,
            method: method.into(),
            channels,
            complex,
            zero_modes: DMatrix::zeros(0, 0),
            gradient: DVector::zeros(0),
        };
        hessian.gradient = hessian.pack(&blocks);
        let dimension = hessian.size();
        if dimension == 0 {
            return Ok(None);
        }
        hessian.zero_modes = DMatrix::zeros(dimension, 0);
        let spin_conserving = method == "ghf" && {
            let m = h.nrows() / 2;
            (0..m).all(|p| {
                (0..m).all(|q| {
                    (h[(p, q)] - h[(m + p, m + q)]).norm() <= 1e-12
                        && h[(p, m + q)].norm() <= 1e-12
                        && h[(m + p, q)].norm() <= 1e-12
                })
            })
        };
        if spin_conserving {
            // The generators -i/2 Sigma_k of the global spin rotations, in the rotation coordinates.
            let (c, _, n) = &hessian.channels[0];
            let m = c.nrows() / 2;
            let one = C64::new(1., 0.);
            let i = C64::new(0., 1.);
            let sigma = |k: usize| {
                Matrix::from_fn(2 * m, 2 * m, |p, q| {
                    let (sp, sq, same) = (p / m, q / m, p % m == q % m);
                    if !same {
                        return C64::default();
                    }
                    match (k, sp, sq) {
                        (0, 0, 1) | (0, 1, 0) => one,
                        (1, 0, 1) => -i,
                        (1, 1, 0) => i,
                        (2, 0, 0) => one,
                        (2, 1, 1) => -one,
                        _ => C64::default(),
                    }
                })
            };
            let candidates: Vec<DVector<f64>> = (0..3)
                .map(|k| {
                    let block = c.columns(*n, 2 * m - n).adjoint()
                        * sigma(k)
                        * c.columns(0, *n)
                        * C64::new(0., -0.5);
                    hessian.pack(&[block])
                })
                .collect();
            let mut columns: Vec<DVector<f64>> = Vec::new();
            for mut t in candidates {
                for _ in 0..2 {
                    for z in &columns {
                        let overlap = z.dot(&t);
                        t -= z * overlap;
                    }
                }
                let norm = t.norm();
                if norm > 1e-8 {
                    columns.push(t / norm);
                }
            }
            if !columns.is_empty() {
                hessian.zero_modes = DMatrix::from_columns(&columns);
            }
        }
        Ok(Some(hessian))
    }
    /// The number of real rotation coordinates.
    fn size(&self) -> usize {
        let factor = if self.complex { 2 } else { 1 };
        self.channels
            .iter()
            .map(|(c, _, n)| factor * n * (c.ncols() - n))
            .sum()
    }
    /// The occupied-virtual blocks `(m - n) x n` of each channel as rotation coordinates: per
    /// channel the real parts row-major (`a * n + i`), then (complex) the imaginary parts.
    fn pack(&self, blocks: &[Matrix]) -> DVector<f64> {
        let mut out = Vec::with_capacity(self.size());
        for block in blocks {
            let rows = (0..block.nrows()).flat_map(|a| (0..block.ncols()).map(move |i| (a, i)));
            out.extend(rows.clone().map(|(a, i)| block[(a, i)].re));
            if self.complex {
                out.extend(rows.map(|(a, i)| block[(a, i)].im));
            }
        }
        DVector::from_vec(out)
    }
    fn unpack(&self, x: &[f64]) -> Vec<Matrix> {
        let mut offset = 0;
        self.channels
            .iter()
            .map(|(c, _, n)| {
                let (n, v) = (*n, c.ncols() - n);
                let block = Matrix::from_fn(v, n, |a, i| {
                    let re = x[offset + a * n + i];
                    let im = if self.complex {
                        x[offset + v * n + a * n + i]
                    } else {
                        0.
                    };
                    C64::new(re, im)
                });
                offset += if self.complex { 2 * v * n } else { v * n };
                block
            })
            .collect()
    }
    fn project(&self, block: DMatrix<f64>) -> DMatrix<f64> {
        if self.zero_modes.ncols() == 0 {
            return block;
        }
        let overlap = self.zero_modes.transpose() * &block;
        block - &self.zero_modes * overlap
    }
    /// The method of the determinant.
    pub fn method(&self) -> &str {
        &self.method
    }
    /// Whether the rotations carry imaginary parts (complex orbitals or GHF).
    pub fn is_complex(&self) -> bool {
        self.complex
    }
    /// The number of exact zero modes projected out (GHF global spin rotations).
    pub fn zero_modes(&self) -> usize {
        self.zero_modes.ncols()
    }
    /// The orbitals of each channel rotated by `exp(angle K)`, `K` the anti-Hermitian generator
    /// whose occupied-virtual block is the rotation `mode` (coordinates of this operator).
    pub fn rotated(&self, mode: &DVector<f64>, angle: f64) -> Result<Vec<Matrix>> {
        if mode.len() != self.size() || !angle.is_finite() {
            return Err(invalid(
                "a rotation needs a mode of the Hessian's size and a finite angle",
            ));
        }
        self.unpack(mode.as_slice())
            .into_iter()
            .zip(&self.channels)
            .map(|(x, (c, _, n))| {
                let (m, n) = (c.ncols(), *n);
                let mut k = Matrix::zeros(m, m);
                for a in n..m {
                    for i in 0..n {
                        k[(a, i)] = x[(a - n, i)];
                        k[(i, a)] = -x[(a - n, i)].conj();
                    }
                }
                Ok(c * exponential(&k, angle, self.complex)?)
            })
            .collect()
    }
    /// The densities of [`Self::rotated`].
    pub fn rotated_densities(&self, mode: &DVector<f64>, angle: f64) -> Result<Vec<Matrix>> {
        Ok(self
            .rotated(mode, angle)?
            .iter()
            .zip(&self.channels)
            .map(|(c, (_, _, n))| density(c, *n))
            .collect())
    }
}

/// `exp(angle K)` of an anti-Hermitian `K`: real `K` from the eigenpairs `w^2, W` of `-K^2`,
/// `W cos(angle w) W^T + K W (sin(angle w) / w) W^T` (real to rounding, no imaginary part made);
/// complex `K` from the eigenpairs `w, V` of the Hermitian `i K`, `V exp(-i angle w) V^+`.
fn exponential(k: &Matrix, angle: f64, complex: bool) -> Result<Matrix> {
    let m = k.nrows();
    if complex {
        let (w, v) = eigen(hermitian(k * C64::new(0., 1.)))?;
        let phases = DVector::from_iterator(m, w.iter().map(|w| C64::from_polar(1., -angle * w)));
        return Ok(&v * Matrix::from_diagonal(&phases) * v.adjoint());
    }
    let k = k.map(|v| v.re);
    let (squares, w) = crate::math::linalg::real_eigen(-(&k * &k))?;
    let frequencies: Vec<f64> = squares.iter().map(|s| s.max(0.).sqrt()).collect();
    let cos = DVector::from_iterator(m, frequencies.iter().map(|w| (angle * w).cos()));
    let sinc = DVector::from_iterator(
        m,
        frequencies.iter().map(|&w| {
            if w < 1e-12 {
                angle
            } else {
                (angle * w).sin() / w
            }
        }),
    );
    let u = &w * DMatrix::from_diagonal(&cos) * w.transpose()
        + &k * &w * DMatrix::from_diagonal(&sinc) * w.transpose();
    Ok(u.map(|v| C64::new(v, 0.)))
}

impl crate::math::davidson::Symmetric for OrbitalHessian {
    fn dimension(&self) -> usize {
        self.size()
    }
    fn diagonal(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.size());
        for (_, eps, n) in &self.channels {
            let n = *n;
            let gaps = (n..eps.len()).flat_map(|a| (0..n).map(move |i| eps[a] - eps[i]));
            out.extend(gaps.clone());
            if self.complex {
                out.extend(gaps);
            }
        }
        out
    }
    fn apply(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let block = self.project(block.clone());
        let mut out = DMatrix::<f64>::zeros(block.nrows(), block.ncols());
        let size = self.channels[0].0.nrows();
        let zero = Matrix::zeros(size, size);
        for column in 0..block.ncols() {
            let x = self.unpack(block.column(column).as_slice());
            let d1: Vec<Matrix> = x
                .iter()
                .zip(&self.channels)
                .map(|(x, (c, _, n))| {
                    let half = c.columns(*n, c.ncols() - n) * x * c.columns(0, *n).adjoint();
                    &half + half.adjoint()
                })
                .collect();
            let response = fock(self.eri.as_ref(), &zero, &d1, &self.method);
            let y: Vec<Matrix> = x
                .iter()
                .zip(&self.channels)
                .zip(&response)
                .map(|((x, (c, eps, n)), g)| {
                    let n = *n;
                    let coupled = c.columns(n, c.ncols() - n).adjoint() * g * c.columns(0, n);
                    Matrix::from_fn(x.nrows(), n, |a, i| {
                        x[(a, i)] * C64::new(eps[n + a] - eps[i], 0.) + coupled[(a, i)]
                    })
                })
                .collect();
            out.set_column(column, &self.pack(&y));
        }
        Ok(self.project(out))
    }
}

/// The lowest root of the orbital Hessian of `densities`: `Ok(None)` without a rotation;
/// `converged` false when the root solve is not shown converged (the state is then not shown
/// stable).
fn lowest_mode(
    eri: Arc<super::integrals::Eri>,
    h: &Matrix,
    method: &str,
    densities: &[Matrix],
    particles: [usize; 2],
) -> Result<Option<(OrbitalHessian, crate::math::davidson::LowestRoot)>> {
    let Some(hessian) = OrbitalHessian::new(eri, h, method, densities, particles)? else {
        return Ok(None);
    };
    let root =
        crate::math::davidson::lowest_root(&hessian, DENSE_HESSIAN_LIMIT, HESSIAN_TOLERANCE)?;
    Ok(root.map(|root| (hessian, root)))
}

/// A converged Hartree-Fock determinant whose orbital Hessian has a root below minus this
/// (Hartree) is a saddle. Degenerate shells held in one determinant keep rotational modes the
/// grid pins at 1e-5 to 3e-4 Ha of either sign; one below minus this is followed.
pub const STABILITY_TOLERANCE: f64 = 1e-5;
/// How many saddles one SCF follows downhill before it reports the state as not converged.
const STABILITY_FOLLOWS: usize = 4;
/// The rotation angles (radians, along the unit mode) of the line search off a saddle.
const STABILITY_ANGLES: [f64; 6] = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2];
/// The first SCF iterations from a rotated determinant take the undamped Fock with the level
/// shift on and no DIIS, so the restart descends from the line search's point instead of
/// extrapolating back to the saddle; DIIS then starts with an empty history.
const SETTLE_ITERATIONS: usize = 12;
/// The level shift (Hartree) of those iterations.
const SETTLE_SHIFT: f64 = 0.5;
/// The most second-order steps of [`second_order_descent`].
const NEWTON_STEPS: usize = 200;
/// The longest second-order step (radians).
const NEWTON_RADIUS: f64 = 0.2;
/// The lowest eigenvalue (Hartree) of the shifted Hessian of a second-order step.
const NEWTON_MARGIN: f64 = 0.02;

/// Stability following of a converged Hartree-Fock candidate (RHF, UHF, GHF): an aufbau
/// determinant can still be a saddle, which the SCF holds once it reaches it (N2 PAW-LCAO SZ in a
/// box, root -0.32 Ha, 0.67 Ha above the minimum). While the lowest root of the orbital Hessian
/// is below `-STABILITY_TOLERANCE`, the determinant is rotated along that mode to the lowest of
/// [`STABILITY_ANGLES`] and the SCF restarts from it as a descent ([`SETTLE_ITERATIONS`]); if
/// that restart does not end converged and lower, it restarts from the same point by
/// [`second_order_descent`]. A lower restart is kept, converged or not as it is (an
/// unconverged one ends the following, reported as not converged); a restart that is not lower,
/// a root solve not shown converged, or a saddle left after [`STABILITY_FOLLOWS`] restarts makes
/// the state not converged: a saddle is never returned as a converged state.
fn follow_instabilities(
    integrals: &MolecularIntegrals,
    method: &str,
    h: &Matrix,
    particles: [usize; 2],
    options: &ScfOptions,
    mut best: Candidate,
) -> Result<Candidate> {
    for follow in 0..=STABILITY_FOLLOWS {
        if !best.converged {
            break;
        }
        let eri = integrals.eri()?;
        let Some((hessian, root)) =
            lowest_mode(eri.clone(), h, method, &best.densities, particles)?
        else {
            break;
        };
        if !root.converged {
            best.converged = false;
            break;
        }
        if root.value >= -STABILITY_TOLERANCE {
            break;
        }
        if follow == STABILITY_FOLLOWS {
            best.converged = false;
            break;
        }
        let start = descent(eri.as_ref(), h, method, &hessian, &root.vector)?;
        let below = |c: &Candidate| c.energy < best.energy - 1e-10;
        let mut candidate = iterate(
            integrals,
            method,
            h,
            particles,
            start.clone(),
            options,
            SETTLE_ITERATIONS,
        )?;
        if !(candidate.converged && below(&candidate)) {
            let second = second_order_descent(integrals, method, h, particles, options, start)?;
            if (second.converged && below(&second)) || second.energy < candidate.energy - 1e-10 {
                candidate = second;
            }
        }
        if below(&candidate) {
            best = candidate;
        } else {
            best.converged = false;
            break;
        }
    }
    Ok(best)
}

/// Second-order steps of a restart off a saddle or out of a soft/oscillating valley:
/// at the semicanonical orbitals of each point, the step `x = -(H + mu)^-1 g` with `mu` holding
/// the shifted Hessian [`NEWTON_MARGIN`] above zero (or zero on a strictly positive Hessian so
/// soft atomic/shell valleys move at full Newton step), at most [`NEWTON_RADIUS`] long, accepted
/// when the energy falls (or, at the float rounding floor of a positive Hessian, when `[F, D]`
/// falls strictly); then the SCF from the last point.
fn second_order_descent(
    integrals: &MolecularIntegrals,
    method: &str,
    h: &Matrix,
    particles: [usize; 2],
    options: &ScfOptions,
    start: Vec<Matrix>,
) -> Result<Candidate> {
    let eri = integrals.eri()?;
    let filled: Vec<usize> = match method {
        "rhf" => vec![particles[0]],
        "uhf" => particles.to_vec(),
        _ => vec![particles.iter().sum()],
    };
    let densities = |c: &[Matrix]| -> Vec<Matrix> {
        c.iter().zip(&filled).map(|(c, &n)| density(c, n)).collect()
    };
    let mut c = start;
    let mut d = densities(&c);
    let mut f_now = fock(eri.as_ref(), h, &d, method);
    let mut value = energy(&d, &f_now, h, method);
    let mut stat_now = stationarity(&f_now, &d);
    let mut radius = NEWTON_RADIUS;
    let mut missed = false;
    for _ in 0..NEWTON_STEPS {
        let Some(hessian) = OrbitalHessian::new(eri.clone(), h, method, &d, particles)? else {
            break;
        };
        let target = (options.tolerance * 1e-2).max(1e-13);
        if hessian.gradient.amax() < target && stat_now < target {
            break;
        }
        let Some(root) =
            crate::math::davidson::lowest_root(&hessian, DENSE_HESSIAN_LIMIT, HESSIAN_TOLERANCE)?
        else {
            break;
        };
        if !root.converged {
            break;
        }
        let cg_tol = 1e-8;
        let cg_iters = hessian.size().max(200);
        let shift = if root.value > 1e-6 {
            0.
        } else if root.value >= -STABILITY_TOLERANCE {
            (2. * (-root.value).max(0.)).max(1e-7)
        } else {
            (NEWTON_MARGIN - root.value).max(0.)
        };
        let mut solve = crate::math::davidson::solve_shifted(
            &hessian,
            shift,
            &-&hessian.gradient,
            cg_tol,
            cg_iters,
        )?;
        if !solve.positive && shift < NEWTON_MARGIN - root.value {
            let safe_shift = (NEWTON_MARGIN - root.value).max(NEWTON_MARGIN);
            solve = crate::math::davidson::solve_shifted(
                &hessian,
                safe_shift,
                &-&hessian.gradient,
                cg_tol,
                cg_iters,
            )?;
        }
        if !solve.positive {
            // The shifted Hessian has a direction of non-positive curvature: the root search
            // missed a lower root, so no point reached from here is shown to be a minimum.
            missed = true;
            break;
        }
        let mut x = solve.x;
        if x.norm() > radius {
            x *= radius / x.norm();
        }
        if root.value < -STABILITY_TOLERANCE {
            // Negative curvature: the trust region's step reaches its boundary along the mode,
            // downhill to first order.
            let sign = if root.vector.dot(&hessian.gradient) > 0. {
                -1.
            } else {
                1.
            };
            let along = (radius * radius - x.norm_squared()).max(0.).sqrt();
            x += &root.vector * (sign * along);
        }
        let mut accepted = false;
        let step_floor = (options.tolerance * 1e-3).min(1e-12);
        while x.norm() > step_floor {
            let trial = hessian.rotated(&x, 1.)?;
            let td = densities(&trial);
            let tf = fock(eri.as_ref(), h, &td, method);
            let trial_value = energy(&td, &tf, h, method);
            let trial_stat = stationarity(&tf, &td);
            let floor_tol = 1e-12 * (1. + value.abs());
            if trial_value < value
                || (root.value >= -STABILITY_TOLERANCE
                    && trial_value <= value + floor_tol
                    && trial_stat < 0.9 * stat_now)
            {
                (c, d, f_now, value, stat_now) = (trial, td, tf, trial_value, trial_stat);
                radius = (2. * radius).min(NEWTON_RADIUS);
                accepted = true;
                break;
            }
            x *= 0.5;
            radius *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    let _ = f_now;
    let mut out = iterate(integrals, method, h, particles, c, options, 0)?;
    out.converged &= !missed;
    Ok(out)
}

/// The orbitals of `hessian` rotated along the unit `mode` to the angle of [`STABILITY_ANGLES`]
/// with the lowest energy.
fn descent(
    eri: &super::integrals::Eri,
    h: &Matrix,
    method: &str,
    hessian: &OrbitalHessian,
    mode: &DVector<f64>,
) -> Result<Vec<Matrix>> {
    let mut lowest = (f64::INFINITY, STABILITY_ANGLES[0]);
    for angle in STABILITY_ANGLES {
        let d = hessian.rotated_densities(mode, angle)?;
        let value = energy(&d, &fock(eri, h, &d, method), h, method);
        if value < lowest.0 {
            lowest = (value, angle);
        }
    }
    hessian.rotated(mode, lowest.1)
}

impl MolecularIntegrals {
    /// The orbital Hessian ([`OrbitalHessian`]) of a Hartree-Fock `state` (RHF, UHF or GHF);
    /// `None` for a Kohn-Sham state or a determinant without rotations.
    pub fn orbital_hessian(&self, state: &MeanField) -> Result<Option<OrbitalHessian>> {
        if !["rhf", "uhf", "ghf"].contains(&state.method.as_str()) || state.functional.is_some() {
            return Ok(None);
        }
        OrbitalHessian::new(
            self.eri()?,
            &self.core_operator(&state.method)?,
            &state.method,
            &state.densities,
            state.particles,
        )
    }
    /// The lowest root (Hartree) of the orbital Hessian of a Hartree-Fock `state`: negative for a
    /// saddle; `None` for a Kohn-Sham state or no rotation. Raises when the root solve is not
    /// shown converged.
    pub fn stability(&self, state: &MeanField) -> Result<Option<f64>> {
        let Some(hessian) = self.orbital_hessian(state)? else {
            return Ok(None);
        };
        let Some(root) =
            crate::math::davidson::lowest_root(&hessian, DENSE_HESSIAN_LIMIT, HESSIAN_TOLERANCE)?
        else {
            return Ok(None);
        };
        if !root.converged {
            return Err(crate::Error::Kernel(
                "the lowest root of the orbital Hessian did not converge: the state's stability is not shown".into(),
            ));
        }
        Ok(Some(root.value))
    }
    /// The core operator of `method` in the orthonormal frame (two-component spin-orbit included
    /// for GHF on a spin-orbit PAW dataset).
    fn core_operator(&self, method: &str) -> Result<Matrix> {
        let m = self.m();
        let spatial = Matrix::from_fn(m, m, |i, j| C64::new(self.one.as_slice()[i * m + j], 0.));
        if method != "ghf" {
            return Ok(spatial);
        }
        let mut h = Matrix::from_fn(2 * m, 2 * m, |i, j| {
            if i / m == j / m {
                spatial[(i % m, j % m)]
            } else {
                C64::default()
            }
        });
        if let Some(spin_orbit) = super::spinor_dft::spin_orbit_term(self)? {
            if spin_orbit.shape() != (2 * m, 2 * m) {
                return Err(invalid("spin-orbit matrix must be (2m, 2m)"));
            }
            h += spin_orbit;
        }
        Ok(h)
    }
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
        self.require_hartree_fock_core()?;
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
            let mut candidate = iterate(
                self,
                method,
                &h,
                previous.particles,
                previous.coefficients.clone(),
                options,
                0,
            )?;
            // The same state is followed, never left for another: a continuation that lands on a
            // saddle, or whose stability is not shown, is reported not converged.
            if candidate.converged {
                let shown = lowest_mode(
                    self.eri()?,
                    &h,
                    method,
                    &candidate.densities,
                    previous.particles,
                )?
                .is_none_or(|(_, root)| root.converged && root.value >= -STABILITY_TOLERANCE);
                candidate.converged = shown;
            }
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
        self.require_hartree_fock_core()?;
        let has_spin_orbit = self.paw_setup.as_ref().is_some_and(|s| s.has_spin_orbit());
        if method != "ghf" && has_spin_orbit {
            return Err(crate::Error::Unsupported(
                "spin-collinear Hartree-Fock of a molecule with spin-orbit datasets is not \
                 available: use generalized Hartree-Fock (ghf), a variational solver, or \
                 spinor Kohn-Sham (spinor_kohn_sham)"
                    .into(),
            ));
        }
        self.eri()?;
        let workspace = ctx.reserve(
            product(&[2 * m, 2 * m, 16, 160])?
                .checked_add(product(&[options.max_iterations, 128, 4])?)
                .ok_or(crate::Error::Overflow)?,
        )?;
        let h = self.core_operator(method)?;
        let (_, core) = eigen(h.clone())?;
        let mut starts = vec![if method == "uhf" {
            vec![core.clone(), core.clone()]
        } else {
            vec![core.clone()]
        }];
        if options.guesses >= 2 {
            if method == "uhf" {
                let da = density(&core, particles[0]);
                let db = density(&core, particles[1]);
                let f = fock(self.eri()?.as_ref(), &h, &[da, db], "uhf");
                let mut spin_starts = Vec::with_capacity(2);
                for f_s in f {
                    let (_, c) = eigen(hermitian((&h + f_s) * C64::new(0.5, 0.)))?;
                    spin_starts.push(c);
                }
                starts.push(spin_starts);
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
                let (_, c) = eigen(hermitian((&h + f) * C64::new(0.5, 0.)))?;
                starts.push(vec![c]);
            }
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
        if method == "ghf" && options.guesses >= 3 && !has_spin_orbit {
            let uhf = self.mean_field_inner("uhf", particles, options)?;
            starts.push(vec![collinear(
                &uhf.coefficients[0],
                &uhf.coefficients[1],
                particles,
            )]);
        }
        // A converged candidate beats an unconverged one; then the lower energy wins among the
        // converged and the smaller final residual among the unconverged (`Candidate::beats`).
        let better = |candidate: &Candidate, best: Option<&Candidate>| {
            best.is_none_or(|best| candidate.beats(best, options.tolerance))
        };
        let mut best: Option<Candidate> = None;
        let mut restarts = Vec::new();
        for start in &starts {
            let candidate = iterate(self, method, &h, particles, start.clone(), options, 0)?;
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
            let candidate = iterate(self, method, &h, particles, start, options, 0)?;
            if better(&candidate, best.as_ref()) {
                best = Some(candidate);
            }
        }
        if !best.as_ref().is_some_and(|b| b.converged) {
            for start in starts.into_iter().rev() {
                let candidate = iterate(
                    self,
                    method,
                    &h,
                    particles,
                    start,
                    options,
                    SETTLE_ITERATIONS,
                )?;
                if better(&candidate, best.as_ref()) {
                    best = Some(candidate);
                }
            }
            let current = best.take().unwrap();
            if !current.converged {
                let second = second_order_descent(
                    self,
                    method,
                    &h,
                    particles,
                    options,
                    current.coefficients.clone(),
                )?;
                best = Some(if better(&second, Some(&current)) {
                    second
                } else {
                    current
                });
            } else {
                best = Some(current);
            }
        }
        let best = follow_instabilities(self, method, &h, particles, options, best.unwrap())?;
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
        let h = self.core_operator(method)?;
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
    pub(super) fn build_model_from_orbitals(
        &self,
        integrals: &MolecularIntegrals,
        orbitals: &Matrix,
    ) -> Result<Model> {
        let ctx = &integrals.context;
        let m = integrals.m();
        let n = orbitals.ncols();
        let eri = integrals.eri()?;
        let h = integrals.core_operator(&self.method)?;
        let transformed = orbitals.adjoint() * h * orbitals;
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
                    let ca = orbitals.rows(alpha * m, m).into_owned();
                    let cb = orbitals.rows(beta * m, m).into_owned();
                    let block = transform::four(ctx, &eri, [&ca, &cb, &ca, &cb])?;
                    for (a, b) in out.as_mut_slice().iter_mut().zip(block.as_slice()) {
                        *a += b;
                    }
                }
            }
            out
        } else {
            transform::four(ctx, &eri, [orbitals; 4])?
        };
        let n_spatial = if self.method == "ghf" { n / 2 } else { n };
        let mut model = Model::new(
            ctx,
            n_spatial,
            self.particles,
            one,
            two,
            integrals.nuclear,
            self.method == "ghf",
        )?;
        model.exchange_madelung = integrals.exchange_madelung()?;
        Ok(model)
    }

    pub fn model(&self, integrals: &MolecularIntegrals) -> Result<Arc<Model>> {
        integrals.require_hartree_fock_core()?;
        self.require_frame()?;
        let mut cache = self.model.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(model) = cache.as_ref() {
            return Ok(model.clone());
        }
        let model = self.build_model_from_orbitals(integrals, &self.model_orbitals)?;
        let model = Arc::new(model);
        *cache = Some(model.clone());
        Ok(model)
    }
}

impl MeanField {
    /// The state vector of this reference in its model frame under `mapping`: the reference
    /// determinant of a restricted state; for an unrestricted one, every determinant of the
    /// `2m` model spin orbitals weighted by the product of the two spins' overlap determinants
    /// with the occupied spin orbitals.
    pub fn reference_state(
        &self,
        integrals: &MolecularIntegrals,
        mapping: &str,
    ) -> Result<crate::Buffer<C64>> {
        let m = integrals.m();
        let particles = self.particles;
        let encoding = crate::quantum::Encoding::new(mapping, 2 * m, particles)?;
        let mut state = integrals.context.allocate::<C64>(
            1usize
                .checked_shl(encoding.n_qubits() as u32)
                .ok_or(crate::Error::Overflow)?,
        )?;
        if self.method != "uhf" {
            state.as_mut_slice()[encoding.reference() as usize] = C64::new(1., 0.);
            return Ok(state);
        }
        let occupation_encoding = crate::quantum::Encoding::new("jordan_wigner", 2 * m, particles)?;
        let basis = crate::quantum::Basis::new(&integrals.context, &occupation_encoding, true)?;
        let rotations: Vec<_> = self
            .coefficients
            .iter()
            .map(|c| self.model_orbitals.adjoint() * c)
            .collect();
        for &occupation in basis.indices.as_slice() {
            let mut amplitude = C64::new(1., 0.);
            for spin in 0..2 {
                let occupied: Vec<_> = (0..m)
                    .filter(|&i| occupation & (1u64 << (2 * m - 1 - spin * m - i)) != 0)
                    .collect();
                let determinant =
                    nalgebra::DMatrix::from_fn(particles[spin], particles[spin], |i, j| {
                        rotations[spin][(occupied[i], j)]
                    })
                    .determinant();
                amplitude *= determinant;
            }
            state.as_mut_slice()[encoding.encode(occupation) as usize] = amplitude;
        }
        Ok(state)
    }

    /// The one-body density of each spin in the model frame, `(2m, 2m)` row-major.
    pub fn model_rdm(&self, integrals: &MolecularIntegrals) -> Result<crate::Buffer<C64>> {
        let m = integrals.m();
        let _work = integrals.context.reserve(crate::product(&[m, m, 16, 8])?)?;
        let mut out = integrals
            .context
            .allocate::<C64>(crate::product(&[2 * m, 2 * m])?)?;
        if self.densities.len() == 1 && self.densities[0].nrows() == 2 * m {
            let transformed =
                self.model_orbitals.adjoint() * &self.densities[0] * &self.model_orbitals;
            for p in 0..2 * m {
                for q in 0..2 * m {
                    out.as_mut_slice()[p * 2 * m + q] = transformed[(p, q)];
                }
            }
            return Ok(out);
        }
        for spin in 0..2 {
            let density = &self.densities[if self.densities.len() == 1 { 0 } else { spin }];
            let transformed = self.model_orbitals.adjoint() * density * &self.model_orbitals;
            for p in 0..m {
                for q in 0..m {
                    out.as_mut_slice()[(spin * m + p) * 2 * m + spin * m + q] = transformed[(p, q)];
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A candidate that stopped with `residual`, at `energy`, converged or not; no orbitals (the
    /// selection reads only these).
    fn candidate(energy: f64, residual: Option<f64>, converged: bool) -> Candidate {
        Candidate {
            energy,
            converged,
            history: residual
                .into_iter()
                .map(|residual| Iteration {
                    iteration: 1,
                    energy,
                    delta: None,
                    residual,
                })
                .collect(),
            coefficients: vec![],
            densities: vec![],
            energies: vec![],
            unstable_pair: None,
            occupations: None,
            aufbau_inversion: None,
            entropy: 0.,
        }
    }

    /// The selection that the Hartree-Fock, Kohn-Sham and spinor Kohn-Sham multi-start SCFs share
    /// (`Candidate::beats`): a converged candidate first, then the lower energy among converged
    /// ones (1e-12 margin) and the smaller final residual among unconverged ones, residuals
    /// below the tolerance counting as equal (then the lower energy).
    #[test]
    fn selection_prefers_converged_then_energy_then_residual() {
        let tolerance = 1e-9;
        let oscillating = candidate(-2.74, Some(5e-3), false);
        let stationary = candidate(-2.70, Some(1e-12), false);
        assert!(
            stationary.beats(&oscillating, tolerance) && !oscillating.beats(&stationary, tolerance)
        );
        let converged = candidate(-2.60, Some(1e-13), true);
        assert!(
            converged.beats(&oscillating, tolerance) && !oscillating.beats(&converged, tolerance)
        );
        let lower = candidate(-2.65, Some(1e-14), true);
        assert!(lower.beats(&converged, tolerance) && !converged.beats(&lower, tolerance));
        let tied = candidate(-2.60 - 1e-13, Some(1e-14), true);
        assert!(!tied.beats(&converged, tolerance) && !converged.beats(&tied, tolerance));
        let none = candidate(-3., None, false);
        assert!(oscillating.beats(&none, tolerance) && !none.beats(&oscillating, tolerance));
        // Two stationary points below the tolerance: the rounding of the last step does not
        // decide, the energy does (a symmetric saddle at 1e-16 against the minimum at 2e-15).
        let saddle = candidate(-1.0355, Some(1.7e-16), false);
        let minimum = candidate(-1.1136, Some(2.2e-15), false);
        assert!(minimum.beats(&saddle, tolerance) && !saddle.beats(&minimum, tolerance));
        // An unreachable tolerance does not make the rounding of the last step a criterion.
        assert!(minimum.beats(&saddle, 1e-30) && !saddle.beats(&minimum, 1e-30));
    }
}
