//! The linear-response operator of an unrestricted (collinear, spin-polarized) molecular
//! reference: spin-conserving excitations of both spins in the spin-orbital form.
//!
//! A trial vector holds an alpha block `y_a` (occupied x empty alpha orbitals, index
//! `i * v_a + a`) followed by a beta block `y_b`. It stands for the change of each spin's density
//! matrix `dD_s = sum_ia y_s,ia (|i_s><a_s| + |a_s><i_s|)`, and [`Unrestricted::sum`] returns
//! `(A + B) y = (e_a - e_i) y + <i_s| F1_s[dD_a, dD_b] |a_s>`, with `F1_s` the first-order change
//! of the reference's own Fock matrix of spin `s`: the exact second derivative of the energy the
//! unrestricted SCF minimizes, piece by piece, as for the closed shell ([`super::Response`]):
//!
//! - Kohn-Sham: the Hartree potential of the total transition density by the SCF's isolated
//!   Poisson solver, the PAW compensation pairs (quadratic in the total density matrix), the
//!   spin-resolved adiabatic XC kernel at the polarized ground density, and for HSE06 the
//!   short-range exact exchange of each spin and the second derivative of the spheres' spin
//!   polarization at the polarized reference (`paw::molecular::SpinTerms::polarization_kernel`);
//! - Hartree-Fock: the Coulomb kernel of both spins and each spin's exchange, on the SCF's pair
//!   factors transformed to each spin's canonical orbitals.
//!
//! [`Unrestricted::difference`] is `(A - B) y`: the gaps and the exchange part. A spin channel
//! with no occupied or no empty orbital contributes no transitions.
use super::{
    invalid,
    response::{check_reference, local_elements, mo_samples, pair_density, Factors},
    Coupling, Operator, Spin,
};
use crate::{
    math::{fft::Poisson, linalg::real_eigen},
    models::xc_kernel::{self, Kernel},
    molecular::{dft, MeanField, MolecularIntegrals},
    product,
    quantum::C64,
    Buffer, Level, Result,
};
use nalgebra::DMatrix;

/// One spin's orbitals and the pieces of the kernel transformed to them.
pub(super) struct SpinChannel {
    pub(super) o: usize,
    pub(super) v: usize,
    pub(super) orbitals: DMatrix<f64>,
    pub(super) energies: Vec<f64>,
    /// Canonical orbitals on the grid, `[orbital][point]` (Kohn-Sham with a grid kernel only).
    samples: Option<Buffer<f64>>,
    /// The PAW compensation pair factors' occupied-empty blocks (their weights are shared).
    pairs: Vec<DMatrix<f64>>,
    factors: Option<Factors>,
}
impl SpinChannel {
    fn size(&self) -> usize {
        self.o * self.v
    }
    fn active(&self) -> bool {
        self.o > 0 && self.v > 0
    }
}

/// The work arrays of one lane.
struct Lane {
    poisson: Option<Poisson>,
    xc: Option<xc_kernel::Scratch>,
    densities: Buffer<f64>,
    potentials: Buffer<f64>,
    total: Buffer<C64>,
    hartree: Buffer<C64>,
}

/// The spin-conserving response operator of an unrestricted reference.
pub struct Unrestricted<'a> {
    integrals: &'a MolecularIntegrals,
    pub(super) channels: [SpinChannel; 2],
    delta: Vec<f64>,
    poisson: bool,
    pair_weights: Vec<f64>,
    xc: Option<Kernel>,
    factor_coulomb: bool,
    exchange: f64,
    polarization: Option<crate::paw::molecular::SphereKernels>,
}

impl<'a> Unrestricted<'a> {
    /// The operator of `reference` (a converged unrestricted Hartree-Fock or Kohn-Sham state of
    /// `integrals`, open shell or not) for `coupling`.
    pub fn new(
        integrals: &'a MolecularIntegrals,
        reference: &MeanField,
        coupling: Coupling,
    ) -> Result<Self> {
        check_reference(integrals, reference)?;
        if reference.method != "uhf" || reference.densities.len() != 2 {
            return Err(invalid(
                "the unrestricted response needs an unrestricted reference with two spin densities",
            ));
        }
        let ctx = &integrals.context;
        let m = integrals.m();
        let densities: Vec<DMatrix<C64>> = reference.densities.clone();
        if densities
            .iter()
            .any(|d| d.shape() != (m, m) || d.iter().any(|z| z.im.abs() > 1e-12))
        {
            return Err(invalid("linear response needs real spin density matrices"));
        }
        // The reference's own Fock matrices; a screened hybrid's screening is built once.
        let functional = reference.functional.clone();
        let mut screening = match functional.as_deref() {
            Some(name) => integrals.screening(name)?,
            None => None,
        };
        let (focks, fields) = match functional.as_deref() {
            Some(name) => {
                let (focks, fields, _, _) = dft::response_with_shared(
                    integrals,
                    &densities,
                    name,
                    None,
                    screening.as_ref(),
                )?;
                (focks, Some(fields))
            }
            None => (
                super::reference_fock(integrals, reference, &densities)?,
                None,
            ),
        };
        let mut channels = Vec::with_capacity(2);
        for spin in 0..2 {
            let o = reference.particles[spin];
            if o > m {
                return Err(invalid("a spin channel holds more electrons than orbitals"));
            }
            let fock = focks[spin].map(|z| z.re);
            let (energies, orbitals) = real_eigen((&fock + fock.transpose()) * 0.5)?;
            let projector = orbitals.columns(0, o) * orbitals.columns(0, o).transpose();
            let stationarity = (projector - densities[spin].map(|z| z.re)).amax();
            if stationarity > 1e-6 {
                return Err(invalid(format!(
                    "the reference is not stationary: the occupied orbitals of spin {spin} differ \
                     from the lowest eigenvectors of its Fock matrix by {stationarity:.2e} (above 1e-6)"
                )));
            }
            if o > 0 && o < m && energies[o] - energies[o - 1] <= 1e-6 {
                return Err(invalid(format!(
                    "linear response needs a gap between the highest occupied and lowest empty \
                     orbital of spin {spin}"
                )));
            }
            channels.push(SpinChannel {
                o,
                v: m - o,
                orbitals,
                energies,
                samples: None,
                pairs: vec![],
                factors: None,
            });
        }
        let delta: Vec<f64> = channels
            .iter()
            .flat_map(|c| {
                (0..c.o)
                    .flat_map(move |i| (c.o..c.o + c.v).map(move |a| c.energies[a] - c.energies[i]))
            })
            .collect();
        if delta.is_empty() {
            return Err(invalid(
                "linear response needs at least one occupied and one empty orbital",
            ));
        }
        let [alpha, beta]: [SpinChannel; 2] = channels
            .try_into()
            .map_err(|_| invalid("two spin channels expected"))?;
        let mut operator = Self {
            integrals,
            channels: [alpha, beta],
            delta,
            poisson: false,
            pair_weights: vec![],
            xc: None,
            factor_coulomb: false,
            exchange: 0.,
            polarization: None,
        };
        if coupling == Coupling::Independent {
            return Ok(operator);
        }
        let real: Vec<DMatrix<f64>> = densities.iter().map(|d| d.map(|z| z.re)).collect();
        match (functional.as_deref(), fields) {
            (Some(name), Some(fields)) => {
                let paw = integrals.corrections.as_ref().and_then(|c| c.dft.as_ref());
                operator.poisson = true;
                if let Some(paw) = paw {
                    operator.pair_weights = paw.pairs.weights.clone();
                    for channel in &mut operator.channels {
                        let (co, cv) = (
                            channel.orbitals.columns(0, channel.o),
                            channel.orbitals.columns(channel.o, channel.v),
                        );
                        channel.pairs = paw
                            .pairs
                            .factors
                            .iter()
                            .map(|b| co.transpose() * b * cv)
                            .collect();
                    }
                }
                if coupling == Coupling::Adiabatic {
                    let semilocal = crate::models::functional(name)?.semilocal;
                    let n = integrals.grid.index.points();
                    let mut rho = ctx.copy_slice(fields.valence.as_slice())?;
                    if let Some(paw) = paw {
                        for s in 0..2 {
                            for k in 0..n {
                                rho.as_mut_slice()[s * n + k] += 0.5 * paw.core.as_slice()[k];
                            }
                        }
                    }
                    let relativistic = paw.is_some_and(|p| p.relativistic_exchange)
                        && matches!(semilocal, "lda" | "pbe");
                    operator.xc = Some(Kernel::new(
                        ctx,
                        &integrals.grid,
                        rho.as_slice(),
                        semilocal,
                        relativistic,
                    )?);
                    if let Some(screening) = screening.take() {
                        let _memory = ctx.reserve(product(&[m, m, 8, 8])?)?;
                        for channel in &mut operator.channels {
                            channel.factors = Some(Factors::new(
                                &screening.eri,
                                &channel.orbitals,
                                channel.o,
                                true,
                            )?);
                        }
                        operator.exchange = screening.fraction;
                        if let Some(terms) = screening.spin.as_ref() {
                            operator.polarization =
                                Some(terms.polarization_kernel(ctx, &real[0], &real[1])?);
                        }
                    }
                }
                for channel in &mut operator.channels {
                    if channel.active() {
                        channel.samples = Some(mo_samples(integrals, &channel.orbitals)?);
                    }
                }
            }
            (None, _) => {
                let exchange = coupling == Coupling::Adiabatic;
                let eri = integrals.eri()?;
                let _memory = ctx.reserve(product(&[eri.m(), eri.m(), 8, 8])?)?;
                for channel in &mut operator.channels {
                    channel.factors =
                        Some(Factors::new(&eri, &channel.orbitals, channel.o, exchange)?);
                }
                operator.factor_coulomb = true;
                operator.exchange = if exchange { 1. } else { 0. };
            }
            _ => return Err(invalid("a Kohn-Sham reference returned no grid fields")),
        }
        Ok(operator)
    }
    pub fn dimension(&self) -> usize {
        self.delta.len()
    }
    /// `e_a - e_i` of every transition, alpha then beta.
    pub fn delta(&self) -> &[f64] {
        &self.delta
    }
    /// The occupied and empty orbital counts of spin `spin` (0 alpha, 1 beta).
    pub fn channel(&self, spin: usize) -> (usize, usize) {
        (self.channels[spin].o, self.channels[spin].v)
    }
    /// The canonical orbitals of spin `spin` in the orthonormal frame (columns, ascending energy).
    pub fn orbitals(&self, spin: usize) -> &DMatrix<f64> {
        &self.channels[spin].orbitals
    }
    pub fn orbital_energies(&self, spin: usize) -> &[f64] {
        &self.channels[spin].energies
    }
    /// The two spin blocks of the trial vector `y` (alpha then beta).
    pub fn matrices(&self, y: &[f64]) -> [DMatrix<f64>; 2] {
        let (a, b) = (&self.channels[0], &self.channels[1]);
        [
            DMatrix::from_row_slice(a.o, a.v, &y[..a.size()]),
            DMatrix::from_row_slice(b.o, b.v, &y[a.size()..a.size() + b.size()]),
        ]
    }
    fn flat(&self, k: &[DMatrix<f64>; 2]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.dimension());
        for (channel, k) in self.channels.iter().zip(k) {
            for i in 0..channel.o {
                for a in 0..channel.v {
                    out.push(k[(i, a)]);
                }
            }
        }
        out
    }
    fn lane(&self) -> Result<Lane> {
        let ctx = &self.integrals.context;
        let n = self.integrals.grid.index.points();
        Ok(Lane {
            poisson: if self.poisson {
                Some(Poisson::isolated(ctx, &self.integrals.grid)?)
            } else {
                None
            },
            xc: self.xc.as_ref().map(|k| k.scratch(ctx)).transpose()?,
            densities: ctx.allocate(2 * n)?,
            potentials: ctx.allocate(2 * n)?,
            total: ctx.allocate(if self.poisson { n } else { 0 })?,
            hartree: ctx.allocate(if self.poisson { n } else { 0 })?,
        })
    }
    /// The grid part of the kernel of one trial vector: Hartree and XC, both spins.
    fn grid_kernel(&self, y: &[DMatrix<f64>; 2], lane: &mut Lane) -> Result<[DMatrix<f64>; 2]> {
        let grid = &self.integrals.grid;
        let n = grid.index.points();
        let densities = lane.densities.as_mut_slice();
        densities.fill(0.);
        for (s, channel) in self.channels.iter().enumerate() {
            if let Some(samples) = channel.samples.as_ref() {
                pair_density(
                    samples.as_slice(),
                    n,
                    channel.o,
                    channel.v,
                    &y[s],
                    &mut densities[s * n..(s + 1) * n],
                )?;
            }
        }
        let potential = lane.potentials.as_mut_slice();
        potential.fill(0.);
        if let Some(kernel) = &self.xc {
            let scratch = lane
                .xc
                .as_mut()
                .ok_or_else(|| invalid("an XC lane has no scratch"))?;
            kernel.apply(grid, lane.densities.as_slice(), potential, scratch)?;
        }
        if let Some(poisson) = lane.poisson.as_mut() {
            let (d, total) = (lane.densities.as_slice(), lane.total.as_mut_slice());
            for k in 0..n {
                total[k] = C64::new(d[k] + d[n + k], 0.);
            }
            poisson.solve_into(lane.total.as_slice(), lane.hartree.as_mut_slice())?;
            for s in 0..2 {
                for (p, h) in potential[s * n..(s + 1) * n]
                    .iter_mut()
                    .zip(lane.hartree.as_slice())
                {
                    *p += h.re;
                }
            }
        }
        let dv = grid.index.dv();
        let mut out = [DMatrix::zeros(0, 0), DMatrix::zeros(0, 0)];
        for (s, channel) in self.channels.iter().enumerate() {
            out[s] = match channel.samples.as_ref() {
                Some(samples) => local_elements(
                    samples.as_slice(),
                    n,
                    channel.o,
                    channel.v,
                    &potential[s * n..(s + 1) * n],
                    dv,
                )?,
                None => DMatrix::zeros(channel.o, channel.v),
            };
        }
        Ok(out)
    }
    /// `<i_s| F1_s[dD_sym(y)] |a_s>` of every column: `(A + B) y - (e_a - e_i) y`.
    pub fn symmetric_kernel(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        if block.nrows() != self.dimension() {
            return Err(invalid(
                "a trial block does not match the response dimension",
            ));
        }
        let count = block.ncols();
        let ys: Vec<[DMatrix<f64>; 2]> = (0..count)
            .map(|c| self.matrices(block.column(c).as_slice()))
            .collect();
        let mut kernels: Vec<[DMatrix<f64>; 2]> = ys
            .iter()
            .map(|y| {
                [
                    DMatrix::zeros(y[0].nrows(), y[0].ncols()),
                    DMatrix::zeros(y[1].nrows(), y[1].ncols()),
                ]
            })
            .collect();
        if self.channels.iter().any(|c| c.samples.is_some()) {
            let ctx = &self.integrals.context;
            let lanes = ctx.lanes(Level::Threads, count);
            let mut states = (0..lanes)
                .map(|_| self.lane())
                .collect::<Result<Vec<_>>>()?;
            let grid = ctx.map_tasks(Level::Threads, &mut states, count, |lane, c| {
                self.grid_kernel(&ys[c], lane)
            })?;
            for (k, g) in kernels.iter_mut().zip(grid) {
                k[0] += &g[0];
                k[1] += &g[1];
            }
        }
        for (k, y) in kernels.iter_mut().zip(&ys) {
            // tr(dD_total B) = 2 sum_s B_s . y_s for each pair factor.
            for (p, &w) in self.pair_weights.iter().enumerate() {
                let moment: f64 = (0..2)
                    .map(|s| 2. * self.channels[s].pairs[p].dot(&y[s]))
                    .sum();
                for s in 0..2 {
                    k[s] += &self.channels[s].pairs[p] * (w * moment);
                }
            }
            if self.factor_coulomb {
                let (fa, fb) = match (&self.channels[0].factors, &self.channels[1].factors) {
                    (Some(a), Some(b)) => (a, b),
                    _ => return Err(invalid("the Coulomb kernel lost its factors")),
                };
                for (p, &w) in fa.weights.iter().enumerate() {
                    let charge = 2. * (fa.ov[p].dot(&y[0]) + fb.ov[p].dot(&y[1]));
                    k[0] += &fa.ov[p] * (w * charge);
                    k[1] += &fb.ov[p] * (w * charge);
                }
            }
            if self.exchange != 0. {
                for s in 0..2 {
                    if let Some(factors) = &self.channels[s].factors {
                        k[s] -= factors.exchange(&y[s], 1.) * self.exchange;
                    }
                }
            }
            if let Some(kernels) = &self.polarization {
                let change: Vec<DMatrix<f64>> = (0..2)
                    .map(|s| {
                        let c = &self.channels[s];
                        let (co, cv) = (c.orbitals.columns(0, c.o), c.orbitals.columns(c.o, c.v));
                        let d = co * &y[s] * cv.transpose();
                        &d + d.transpose()
                    })
                    .collect();
                let focks = kernels.apply(&change[0], &change[1])?;
                for s in 0..2 {
                    let c = &self.channels[s];
                    let (co, cv) = (c.orbitals.columns(0, c.o), c.orbitals.columns(c.o, c.v));
                    k[s] -= co.transpose() * &focks[s] * cv * self.exchange;
                }
            }
        }
        let mut out = DMatrix::zeros(self.dimension(), count);
        for (c, k) in kernels.iter().enumerate() {
            out.set_column(c, &nalgebra::DVector::from_vec(self.flat(k)));
        }
        Ok(out)
    }
    /// `<i_s| F1_s[dD_anti(y)] |a_s>` of every column: `(A - B) y - (e_a - e_i) y` (exchange only).
    pub fn antisymmetric_kernel(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        if block.nrows() != self.dimension() {
            return Err(invalid(
                "a trial block does not match the response dimension",
            ));
        }
        let mut out = DMatrix::zeros(self.dimension(), block.ncols());
        if self.exchange == 0. {
            return Ok(out);
        }
        for c in 0..block.ncols() {
            let y = self.matrices(block.column(c).as_slice());
            let mut k = [
                DMatrix::zeros(y[0].nrows(), y[0].ncols()),
                DMatrix::zeros(y[1].nrows(), y[1].ncols()),
            ];
            for s in 0..2 {
                if let Some(factors) = &self.channels[s].factors {
                    k[s] = factors.exchange(&y[s], -1.) * -self.exchange;
                }
            }
            out.set_column(c, &nalgebra::DVector::from_vec(self.flat(&k)));
        }
        Ok(out)
    }
    fn diagonal_times(&self, block: &DMatrix<f64>) -> DMatrix<f64> {
        DMatrix::from_fn(block.nrows(), block.ncols(), |r, c| {
            self.delta[r] * block[(r, c)]
        })
    }
    /// `(A + B) block`.
    pub fn sum(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        Ok(self.diagonal_times(block) + self.symmetric_kernel(block)?)
    }
    /// `(A - B) block`.
    pub fn difference(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        Ok(self.diagonal_times(block) + self.antisymmetric_kernel(block)?)
    }
    /// `A block` (the Tamm-Dancoff operator).
    pub fn resonant(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let symmetric = self.symmetric_kernel(block)?;
        let antisymmetric = self.antisymmetric_kernel(block)?;
        Ok(self.diagonal_times(block) + (symmetric + antisymmetric) * 0.5)
    }
}

impl Operator for Unrestricted<'_> {
    fn dimension(&self) -> usize {
        Unrestricted::dimension(self)
    }
    fn delta(&self) -> &[f64] {
        Unrestricted::delta(self)
    }
    fn sum(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        Unrestricted::sum(self, block)
    }
    fn difference(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        Unrestricted::difference(self, block)
    }
    fn resonant(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        Unrestricted::resonant(self, block)
    }
    fn spin(&self) -> Spin {
        Spin::Unrestricted
    }
}
