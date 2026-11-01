//! Toy probes. Each is a small Hubbard-corrected Kohn-Sham model whose response
//! kernel is known in closed form or measured by differencing its own potential.
use super::exact::*;
use mandacaru_core::models::hubbard_response::Probe;
use mandacaru_core::{Error, Result};
use nalgebra::DMatrix;

/// Response given by matrices: `N(a) = c + M a + even a^2 + cubic a^3` (elementwise),
/// with `M = frozen` or `scf`. The constant and the even term are common mode.
pub struct Linear {
    pub frozen: DMatrix<f64>,
    pub scf: DMatrix<f64>,
    pub channels: usize,
    pub even: f64,
    pub cubic: f64,
}

impl Linear {
    pub fn new(frozen: DMatrix<f64>, scf: DMatrix<f64>, channels: usize) -> Self {
        Self {
            frozen,
            scf,
            channels,
            even: 0.,
            cubic: 0.,
        }
    }
    fn evaluate(&self, m: &DMatrix<f64>, s: &[f64]) -> Vec<f64> {
        let linear = m * DMatrix::from_column_slice(s.len(), 1, s);
        (0..s.len())
            .map(|i| 0.7 + linear[i] + self.even * s[i] * s[i] + self.cubic * s[i].powi(3))
            .collect()
    }
}

impl Probe for Linear {
    fn shells(&self) -> usize {
        self.frozen.nrows() / self.channels
    }
    fn channels(&self) -> usize {
        self.channels
    }
    fn frozen(&mut self, shifts: &[f64]) -> Result<Vec<f64>> {
        Ok(self.evaluate(&self.frozen, shifts))
    }
    fn scf(&mut self, shifts: &[f64]) -> Result<Vec<f64>> {
        Ok(self.evaluate(&self.scf, shifts))
    }
}

/// A probe that misbehaves on purpose.
pub enum Fault {
    ShortVector,
    NotFinite,
    Fails,
}

pub struct Faulty(pub Fault);

impl Faulty {
    fn answer(&self) -> Result<Vec<f64>> {
        match self.0 {
            Fault::ShortVector => Ok(vec![0.]),
            Fault::NotFinite => Ok(vec![f64::NAN, 0.]),
            Fault::Fails => Err(Error::Kernel("the SCF diverged".into())),
        }
    }
}

impl Probe for Faulty {
    fn shells(&self) -> usize {
        2
    }
    fn channels(&self) -> usize {
        1
    }
    fn frozen(&mut self, _: &[f64]) -> Result<Vec<f64>> {
        self.answer()
    }
    fn scf(&mut self, _: &[f64]) -> Result<Vec<f64>> {
        self.answer()
    }
}

/// How the Dudarev potential `u_in (1/2 - N/2)` enters the perturbed SCF.
#[derive(Clone, Copy, PartialEq)]
pub enum Vu {
    Absent,
    Frozen,
    Responds,
}

/// Two sites, one doubly occupied orbital (restricted, `N = 2`). Site `a` is the
/// correlated shell; the potential on it is `w N + kappa N^3` (Hartree-like) plus the
/// Dudarev term. The lower orbital of `[[h_a, -t], [-t, h_b]]` has weight
/// `N_a = 1 - d / sqrt(d^2 + 4 t^2)`, `d = h_a - h_b`, in closed form.
pub struct TwoLevel {
    pub d0: f64,
    pub t: f64,
    pub w: f64,
    pub kappa: f64,
    pub u_in: f64,
    pub vu: Vu,
}

impl TwoLevel {
    pub fn occupation(&self, d: f64) -> f64 {
        1. - d / (d * d + 4. * self.t * self.t).sqrt()
    }
    /// Closed-form `dN/dd`.
    pub fn slope(&self, d: f64) -> f64 {
        -4. * self.t * self.t / (d * d + 4. * self.t * self.t).powf(1.5)
    }
    pub fn hxc(&self, n: f64) -> f64 {
        self.w * n + self.kappa * n.powi(3)
    }
    pub fn dudarev(&self, n: f64) -> f64 {
        self.u_in * (0.5 - 0.5 * n)
    }
    /// Self-consistent `N`, bisection. `fixed_vu` freezes the Dudarev term.
    fn solve(&self, alpha: f64, fixed_vu: Option<f64>) -> f64 {
        let residual = |n: f64| {
            let vu = match (self.vu, fixed_vu) {
                (Vu::Absent, _) => 0.,
                (_, Some(v)) => v,
                (_, None) => self.dudarev(n),
            };
            n - self.occupation(self.d0 + alpha + self.hxc(n) + vu)
        };
        bisect(0., 2., residual)
    }
    pub fn ground(&self) -> f64 {
        self.solve(0., None)
    }
    fn ground_vu(&self) -> f64 {
        if self.vu == Vu::Absent {
            0.
        } else {
            self.dudarev(self.ground())
        }
    }
    /// The kernel the probe realizes, derived by hand (not by the code under test).
    pub fn exact_kernel(&self) -> f64 {
        let n0 = self.ground();
        let hartree = self.w + 3. * self.kappa * n0 * n0;
        if self.vu == Vu::Responds {
            hartree - 0.5 * self.u_in
        } else {
            hartree
        }
    }
    /// The same kernel measured by shifting the model's own potential.
    pub fn measured_kernel(&self) -> f64 {
        let (n0, h) = (self.ground(), 1e-4);
        let hxc = (self.hxc(n0 + h) - self.hxc(n0 - h)) / (2. * h);
        if self.vu == Vu::Responds {
            hxc + (self.dudarev(n0 + h) - self.dudarev(n0 - h)) / (2. * h)
        } else {
            hxc
        }
    }
}

impl Probe for TwoLevel {
    fn shells(&self) -> usize {
        1
    }
    fn channels(&self) -> usize {
        1
    }
    fn frozen(&mut self, s: &[f64]) -> Result<Vec<f64>> {
        let n0 = self.ground();
        Ok(vec![self.occupation(
            self.d0 + s[0] + self.hxc(n0) + self.ground_vu(),
        )])
    }
    fn scf(&mut self, s: &[f64]) -> Result<Vec<f64>> {
        let fixed = (self.vu == Vu::Frozen).then(|| self.ground_vu());
        Ok(vec![self.solve(s[0], fixed)])
    }
}

/// The two-electron Hubbard dimer with a shift on site 1. `scf` is the exact
/// interacting singlet ground state, `frozen` the same problem at `U = 0`.
pub struct Dimer {
    pub t: f64,
    pub u: f64,
}

impl Dimer {
    /// `<n_1>` of the exact ground state in the basis `(|11>, |22>, covalent)`.
    fn occupation(&self, u: f64, alpha: f64) -> f64 {
        let r = std::f64::consts::SQRT_2 * self.t;
        let h = [u + 2. * alpha, 0., -r, 0., u, -r, -r, -r, alpha];
        let (_, v) = lowest_3x3(&h);
        (2. * v[0] * v[0] + v[2] * v[2]) / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    }
    /// `U_resp = 1/chi0 - 1/chi` from second-order perturbation theory of the
    /// symmetric dimer: one odd excited state at energy `U`, matrix element `b`.
    pub fn analytic_u(&self) -> f64 {
        let e0 = 0.5 * self.u - (0.25 * self.u * self.u + 4. * self.t * self.t).sqrt();
        let ion = -e0 / (2. * self.t);
        let b2 = ion * ion / (1. + ion * ion);
        -2. * self.t + (self.u - e0) / (2. * b2)
    }
}

impl Probe for Dimer {
    fn shells(&self) -> usize {
        1
    }
    fn channels(&self) -> usize {
        1
    }
    fn frozen(&mut self, s: &[f64]) -> Result<Vec<f64>> {
        Ok(vec![self.occupation(0., s[0])])
    }
    fn scf(&mut self, s: &[f64]) -> Result<Vec<f64>> {
        Ok(vec![self.occupation(self.u, s[0])])
    }
}

/// One correlated site per spin with a ligand each, one electron per spin. The
/// mean-field potential `V_s = (U - J) n_s + U n_s'` has the kernel
/// `[[U - J, U], [U, U - J]]`; `delta` splits the two channels' levels.
pub struct Hund {
    pub t: f64,
    pub u: f64,
    pub j: f64,
    pub delta: f64,
}

impl Hund {
    pub fn occupation(&self, d: f64) -> f64 {
        0.5 * (1. - d / (d * d + 4. * self.t * self.t).sqrt())
    }
    pub fn slope(&self, d: f64) -> f64 {
        -2. * self.t * self.t / (d * d + 4. * self.t * self.t).powf(1.5)
    }
    fn levels(&self) -> [f64; 2] {
        [self.delta, -self.delta]
    }
    fn potential(&self, n: [f64; 2]) -> [f64; 2] {
        [
            (self.u - self.j) * n[0] + self.u * n[1],
            (self.u - self.j) * n[1] + self.u * n[0],
        ]
    }
    pub fn ground(&self) -> [f64; 2] {
        self.solve([0., 0.])
    }
    fn solve(&self, s: [f64; 2]) -> [f64; 2] {
        let mut n = [0.5, 0.5];
        for _ in 0..5000 {
            let v = self.potential(n);
            let l = self.levels();
            let next = [
                self.occupation(l[0] + s[0] + v[0]),
                self.occupation(l[1] + s[1] + v[1]),
            ];
            let change = (next[0] - n[0]).abs().max((next[1] - n[1]).abs());
            n = [0.5 * (n[0] + next[0]), 0.5 * (n[1] + next[1])];
            if change < 1e-16 {
                break;
            }
        }
        n
    }
    /// Per-channel bare response at the ground state, closed form.
    pub fn chi0(&self) -> [f64; 2] {
        let n0 = self.ground();
        let (v, l) = (self.potential(n0), self.levels());
        [self.slope(l[0] + v[0]), self.slope(l[1] + v[1])]
    }
}

impl Probe for Hund {
    fn shells(&self) -> usize {
        1
    }
    fn channels(&self) -> usize {
        2
    }
    fn frozen(&mut self, s: &[f64]) -> Result<Vec<f64>> {
        let (v, l) = (self.potential(self.ground()), self.levels());
        Ok((0..2)
            .map(|c| self.occupation(l[c] + s[c] + v[c]))
            .collect())
    }
    fn scf(&mut self, s: &[f64]) -> Result<Vec<f64>> {
        Ok(self.solve([s[0], s[1]]).to_vec())
    }
}

/// A periodic ring of `sites` sites, `electrons` electrons, hopping `-t`, on-site
/// disorder `eps`. Shell `i` is site `corr[i]`. The feedback on shell `I` is
/// `sum_J w_IJ N_J` (Hartree-like, so the kernel is `w` exactly) plus the Dudarev term.
pub struct Ring {
    pub eps: Vec<f64>,
    pub t: f64,
    pub electrons: usize,
    pub corr: Vec<usize>,
    pub w: DMatrix<f64>,
    pub u_in: f64,
    pub vu: Vu,
    /// Failing control: the frozen pass uses the bare Hamiltonian, no feedback.
    pub bare_chi0: bool,
}

impl Ring {
    pub fn new(sites: usize, electrons: usize, corr: Vec<usize>, w: DMatrix<f64>) -> Self {
        let eps = (0..sites)
            .map(|i| 0.15 * (1.7 * i as f64 + 0.4).sin())
            .collect();
        Self {
            eps,
            t: 1.,
            electrons,
            corr,
            w,
            u_in: 0.,
            vu: Vu::Absent,
            bare_chi0: false,
        }
    }
    fn sites(&self) -> usize {
        self.eps.len()
    }
    /// Shell occupations (spin-summed) of the lowest `electrons / 2` orbitals of
    /// `eps + extra`, where `extra` has one entry per shell.
    fn occupations(&self, extra: &[f64]) -> Vec<f64> {
        let n = self.sites();
        let mut h = vec![0.; n * n];
        for i in 0..n {
            h[i * n + i] = self.eps[i];
            let j = (i + 1) % n;
            h[i * n + j] = -self.t;
            h[j * n + i] = -self.t;
        }
        for (k, &site) in self.corr.iter().enumerate() {
            h[site * n + site] += extra[k];
        }
        let (_, v) = jacobi(&h, n);
        self.corr
            .iter()
            .map(|&site| {
                2. * (0..self.electrons / 2)
                    .map(|k| v[k * n + site].powi(2))
                    .sum::<f64>()
            })
            .collect()
    }
    fn potential(&self, n: &[f64], shifts: &[f64], vu: Option<&[f64]>) -> Vec<f64> {
        let w = &self.w * DMatrix::from_column_slice(n.len(), 1, n);
        (0..n.len())
            .map(|i| {
                let v = match (self.vu, vu) {
                    (Vu::Absent, _) => 0.,
                    (_, Some(fixed)) => fixed[i],
                    (_, None) => self.u_in * (0.5 - 0.5 * n[i]),
                };
                w[i] + v + shifts[i]
            })
            .collect()
    }
    fn solve(&self, shifts: &[f64], vu: Option<&[f64]>) -> Vec<f64> {
        let mut n = vec![1.; self.corr.len()];
        for _ in 0..2000 {
            let next = self.occupations(&self.potential(&n, shifts, vu));
            let change = (0..n.len())
                .map(|i| (next[i] - n[i]).abs())
                .fold(0., f64::max);
            for i in 0..n.len() {
                n[i] = 0.5 * (n[i] + next[i]);
            }
            if change < 1e-15 {
                break;
            }
        }
        n
    }
    pub fn ground(&self) -> Vec<f64> {
        self.solve(&vec![0.; self.corr.len()], None)
    }
    fn ground_vu(&self) -> Vec<f64> {
        self.ground()
            .iter()
            .map(|n| self.u_in * (0.5 - 0.5 * n))
            .collect()
    }
    /// The kernel the probe realizes: `w`, minus `u_in / 2` on the diagonal when the
    /// Dudarev term responds.
    pub fn exact_kernel(&self) -> DMatrix<f64> {
        let mut f = self.w.clone();
        if self.vu == Vu::Responds {
            for i in 0..f.nrows() {
                f[(i, i)] -= 0.5 * self.u_in;
            }
        }
        f
    }
}

impl Probe for Ring {
    fn shells(&self) -> usize {
        self.corr.len()
    }
    fn channels(&self) -> usize {
        1
    }
    fn frozen(&mut self, s: &[f64]) -> Result<Vec<f64>> {
        if self.bare_chi0 {
            return Ok(self.occupations(s));
        }
        let vu = self.ground_vu();
        let v = self.potential(&self.ground(), s, Some(&vu));
        Ok(self.occupations(&v))
    }
    fn scf(&mut self, s: &[f64]) -> Result<Vec<f64>> {
        let vu = self.ground_vu();
        let fixed = (self.vu == Vu::Frozen).then_some(vu.as_slice());
        Ok(self.solve(s, fixed))
    }
}

/// Clean ring Hamiltonian for the sum-over-states reference (shells are sites).
pub fn clean_ring(sites: usize) -> Vec<f64> {
    let mut h = vec![0.; sites * sites];
    for i in 0..sites {
        let j = (i + 1) % sites;
        h[i * sites + j] = -1.;
        h[j * sites + i] = -1.;
    }
    h
}
