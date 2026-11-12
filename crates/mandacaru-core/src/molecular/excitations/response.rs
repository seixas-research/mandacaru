//! The linear-response operator of a closed-shell molecular reference.
//!
//! A trial vector `y` (occupied x virtual, index `i * v + a`) stands for the change of each
//! spin's density matrix `dD = sum_ia y_ia (|i><a| + |a><i|)` (`dD_dn = dD_up` for a singlet,
//! `-dD_up` for a triplet). [`Response::sum`] returns `(A + B) y = (e_a - e_i) y + <i| F1 |a>`,
//! where `F1` is the first-order change of the reference's own Fock matrix: so the kernel is the
//! exact second derivative of the energy the SCF minimizes, piece by piece.
//!
//! - Kohn-Sham: the Hartree potential of the transition density by the SCF's isolated Poisson
//!   solver, the PAW compensation pairs `PawKohnSham::pairs` (quadratic in D), and the
//!   adiabatic XC kernel of the smooth density ([`models::xc_kernel`](crate::models::xc_kernel)).
//!   The linearized PAW one-center energy has no second derivative, so it adds nothing.
//! - Hartree-Fock: the Coulomb and exchange terms of the SCF's own pair factors (`eri()`, the
//!   one-center `DeltaW` included), transformed to the canonical orbitals once.
//!
//! [`Response::difference`] is `(A - B) y`: the orbital-energy differences plus the exchange
//! part, which is the only piece of these kernels that sees the antisymmetric part of `dD`.
//! The grid passes run on the calculator's lanes, one trial vector per task, each serial and in
//! a fixed order, so the result is the same for every thread count.
use super::{invalid, Coupling, Spin};
use crate::{
    math::{
        fft::Poisson,
        linalg::real_eigen,
        storage::{Precision, SampleBlocks},
    },
    models::xc_kernel::{self, Kernel},
    molecular::{dft, integrals::Eri, MeanField, MolecularIntegrals},
    product,
    quantum::C64,
    Buffer, Level, Result,
};
use nalgebra::{Const, DMatrix, DMatrixView, Dyn};

/// Points per block of the grid passes.
const BLOCK: usize = 2048;

/// Pair factors `(pq|rs) = sum_P w_P L^P_pq L^P_rs` in the canonical orbitals.
pub(super) struct Factors {
    pub(super) weights: Vec<f64>,
    pub(super) ov: Vec<DMatrix<f64>>,
    pub(super) oo: Vec<DMatrix<f64>>,
    pub(super) vv: Vec<DMatrix<f64>>,
}
impl Factors {
    /// The factors of `eri` (and their `oo`, `vv` blocks when `exchange`), in `orbitals`.
    pub(super) fn new(eri: &Eri, orbitals: &DMatrix<f64>, o: usize, exchange: bool) -> Result<Self> {
        let m = orbitals.nrows();
        let (co, cv) = (orbitals.columns(0, o), orbitals.columns(o, m - o));
        let mut out = Self {
            weights: vec![],
            ov: vec![],
            oo: vec![],
            vv: vec![],
        };
        let mut push = |weight: f64, factor: &DMatrix<f64>| {
            out.weights.push(weight);
            out.ov.push(co.transpose() * factor * cv);
            if exchange {
                out.oo.push(co.transpose() * factor * co);
                out.vv.push(cv.transpose() * factor * cv);
            }
        };
        let mut stack = vec![eri];
        while let Some(eri) = stack.pop() {
            match eri {
                Eri::Augmented {
                    base, correction, ..
                } => {
                    for (factor, &w) in correction.factors.iter().zip(&correction.weights) {
                        push(w, factor);
                    }
                    stack.push(base);
                }
                Eri::Cholesky { rank, factors, .. } => {
                    for column in 0..*rank {
                        let b = DMatrix::from_fn(m, m, |p, q| {
                            factors.as_slice()
                                [crate::molecular::integrals::pair(p, q) * rank + column]
                        });
                        push(1., &b);
                    }
                }
                Eri::Packed { .. } => {
                    return Err(crate::Error::Unsupported(
                        "linear response needs factored Coulomb integrals, not a packed table"
                            .into(),
                    ))
                }
            }
        }
        Ok(out)
    }
    /// `sum_P w_P (L^P . y) L^P` over the occupied-virtual blocks: `sum_jb (ia|jb) y_jb`.
    pub(super) fn coulomb(&self, y: &DMatrix<f64>) -> DMatrix<f64> {
        let mut out = DMatrix::zeros(y.nrows(), y.ncols());
        for (l, &w) in self.ov.iter().zip(&self.weights) {
            out += l * (w * l.dot(y));
        }
        out
    }
    /// `sum_jb [(ij|ab) + sign (ib|ja)] y_jb`.
    pub(super) fn exchange(&self, y: &DMatrix<f64>, sign: f64) -> DMatrix<f64> {
        let mut out = DMatrix::zeros(y.nrows(), y.ncols());
        for (((oo, vv), ov), &w) in self
            .oo
            .iter()
            .zip(&self.vv)
            .zip(&self.ov)
            .zip(&self.weights)
        {
            out += (oo * y * vv + (ov * y.transpose() * ov) * sign) * w;
        }
        out
    }
}

/// The work arrays of one lane.
struct Lane {
    poisson: Option<Poisson>,
    xc: Option<xc_kernel::Scratch>,
    density: Buffer<f64>,
    spins: Buffer<f64>,
    potentials: Buffer<f64>,
    total: Buffer<C64>,
    hartree: Buffer<C64>,
}

/// The closed-shell response operator of one spin symmetry and kernel.
pub struct Response<'a> {
    integrals: &'a MolecularIntegrals,
    spin: Spin,
    o: usize,
    v: usize,
    orbitals: DMatrix<f64>,
    energies: Vec<f64>,
    delta: Vec<f64>,
    /// Canonical orbitals on the grid, `[orbital][point]` (Kohn-Sham with a kernel only).
    samples: Option<Buffer<f64>>,
    poisson: bool,
    pairs: Vec<(f64, DMatrix<f64>)>,
    xc: Option<Kernel>,
    factors: Option<Factors>,
    /// Whether `factors` also carry the Coulomb kernel (Hartree-Fock; Kohn-Sham takes it from the
    /// grid).
    factor_coulomb: bool,
    /// The exact-exchange fraction of the factors' exchange kernel (1 for Hartree-Fock, the
    /// screened fraction for HSE06, 0 without).
    exchange: f64,
    /// A screened hybrid's spheres and the ground-state density, for the spin-polarizing
    /// (triplet) part of their semilocal short-range exchange.
    polarization: Option<crate::paw::molecular::SphereKernels>,
}

/// What a reference needs to be for this module, or the reason it is refused.
pub(super) fn check_reference(integrals: &MolecularIntegrals, reference: &MeanField) -> Result<()> {
    let unsupported = |what: String| Err(crate::Error::Unsupported(what));
    if reference.method == "ghf" {
        if reference.functional.is_some() {
            return unsupported(
                "linear response of spinor Kohn-Sham (a spin-orbit dataset) is not implemented \
                 yet; a scalar-relativistic dataset runs restricted Kohn-Sham"
                    .into(),
            );
        }
        return Err(invalid(
            "linear response needs a restricted (rhf or dft) reference; method='ghf' gives a \
             two-component state it does not take",
        ));
    }
    if integrals.grid.periodic || integrals.supercell.is_some() {
        return Err(invalid(
            "molecular linear response needs an isolated molecule, not a periodic supercell",
        ));
    }
    if integrals.precision() == Precision::Float32 {
        return Err(invalid(
            "linear response runs in float64 only; precision='float32' is not supported",
        ));
    }
    if !reference.converged {
        return Err(invalid(
            "linear response needs a converged reference; the SCF did not converge",
        ));
    }
    reference.require_determinant("linear response")?;
    if reference.aufbau_inversion.is_some() {
        return unsupported(
            "linear response needs an aufbau reference; this state holds an occupied level above \
             an empty one"
                .into(),
        );
    }
    if integrals
        .corrections
        .as_ref()
        .is_some_and(|c| c.hubbard.is_some())
    {
        return unsupported("linear response with hubbard_u is not implemented yet".into());
    }
    if let Some(name) = reference.functional.as_deref() {
        let definition = crate::models::functional(name)?;
        if definition.nonlocal_b.is_some() {
            return unsupported(format!(
                "linear response with xc='{name}' needs the nonlocal (rVV10) kernel, which is not \
                 implemented; neglecting it would run a different calculation"
            ));
        }
        if definition.requires_tau {
            return unsupported(format!(
                "linear response with the meta-GGA xc='{name}' is not implemented yet; \
                 xc='lda', xc='pbe' and xc='hse06' are"
            ));
        }
    }
    Ok(())
}

impl<'a> Response<'a> {
    /// The operator of `reference` (a converged closed-shell RHF or restricted Kohn-Sham state of
    /// `integrals`) for `spin` and `coupling`.
    pub fn new(
        integrals: &'a MolecularIntegrals,
        reference: &MeanField,
        spin: Spin,
        coupling: Coupling,
    ) -> Result<Self> {
        check_reference(integrals, reference)?;
        if reference.method == "uhf" || reference.particles[0] != reference.particles[1] {
            return Err(invalid(
                "the spin-adapted response needs a closed-shell restricted reference; an \
                 unrestricted one runs the unrestricted response",
            ));
        }
        if spin == Spin::Unrestricted {
            return Err(invalid(
                "a closed-shell reference is spin-adapted: ask for singlets or triplets",
            ));
        }
        let ctx = &integrals.context;
        let m = integrals.m();
        let o = reference.particles[0];
        if o == 0 || o >= m {
            return Err(invalid(
                "linear response needs at least one occupied and one empty orbital",
            ));
        }
        let v = m - o;
        let density = &reference.densities[0];
        if density.shape() != (m, m) || density.iter().any(|z| z.im.abs() > 1e-12) {
            return Err(invalid(
                "linear response needs a real restricted density matrix",
            ));
        }
        // The reference's own Fock matrix, from the same builder the SCF used; a screened
        // hybrid's screening is built once and serves the Fock matrix and the kernel.
        let functional = reference.functional.clone();
        let mut screening = match functional.as_deref() {
            Some(name) => integrals.screening(name)?,
            None => None,
        };
        let (fock, fields) = match functional.as_deref() {
            Some(name) => {
                let (fock, fields, _, _) = dft::response_with_shared(
                    integrals,
                    std::slice::from_ref(density),
                    name,
                    None,
                    screening.as_ref(),
                )?;
                (fock[0].map(|z| z.re), Some(fields))
            }
            None => (
                super::reference_fock(integrals, reference, std::slice::from_ref(density))?[0]
                    .map(|z| z.re),
                None,
            ),
        };
        let (energies, orbitals) = real_eigen((&fock + fock.transpose()) * 0.5)?;
        let projector = orbitals.columns(0, o) * orbitals.columns(0, o).transpose();
        let stationarity = (projector - density.map(|z| z.re)).amax();
        if stationarity > 1e-6 {
            return Err(invalid(format!(
                "the reference is not stationary: its occupied orbitals differ from the lowest \
                 eigenvectors of its Fock matrix by {stationarity:.2e} (above 1e-6)"
            )));
        }
        if energies[o] - energies[o - 1] <= 1e-6 {
            return Err(invalid(
                "linear response needs a gap between the highest occupied and lowest empty \
                 orbital of the reference",
            ));
        }
        let delta = (0..o)
            .flat_map(|i| (o..m).map(move |a| (i, a)))
            .map(|(i, a)| energies[a] - energies[i])
            .collect();
        let mut response = Self {
            integrals,
            spin,
            o,
            v,
            orbitals,
            energies,
            delta,
            samples: None,
            poisson: false,
            pairs: vec![],
            xc: None,
            factors: None,
            factor_coulomb: false,
            exchange: 0.,
            polarization: None,
        };
        if coupling == Coupling::Independent {
            return Ok(response);
        }
        let singlet = spin == Spin::Singlet;
        match (functional.as_deref(), fields) {
            (Some(name), Some(fields)) => {
                let paw = integrals.corrections.as_ref().and_then(|c| c.dft.as_ref());
                response.poisson = singlet;
                if singlet {
                    if let Some(paw) = paw {
                        let co = response.orbitals.columns(0, o);
                        let cv = response.orbitals.columns(o, v);
                        response.pairs = paw
                            .pairs
                            .factors
                            .iter()
                            .zip(&paw.pairs.weights)
                            .map(|(b, &w)| (w, co.transpose() * b * cv))
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
                    response.xc = Some(Kernel::new(
                        ctx,
                        &integrals.grid,
                        rho.as_slice(),
                        semilocal,
                        relativistic,
                    )?);
                }
                if response.poisson || response.xc.is_some() {
                    response.samples = Some(mo_samples(integrals, &response.orbitals)?);
                }
                if coupling == Coupling::Adiabatic {
                    if let Some(screening) = screening.take() {
                        // The short-range exact exchange in A and B, on the screened factors the
                        // SCF's exchange uses (their PAW compensation and one-center pieces too).
                        let reserve = product(&[m, m, 8, 4])?;
                        let _memory = ctx.reserve(reserve)?;
                        response.factors =
                            Some(Factors::new(&screening.eri, &response.orbitals, o, true)?);
                        response.exchange = screening.fraction;
                        if !singlet {
                            if let Some(terms) = screening.spin.as_ref() {
                                let d = density.map(|z| z.re);
                                response.polarization =
                                    Some(terms.polarization_kernel(ctx, &d, &d)?);
                            }
                        }
                    }
                }
            }
            (None, _) => {
                let exchange = coupling == Coupling::Adiabatic;
                let eri = integrals.eri()?;
                let reserve = product(&[eri.m(), eri.m(), 8, 4])?;
                let _memory = ctx.reserve(reserve)?;
                response.factors = Some(Factors::new(&eri, &response.orbitals, o, exchange)?);
                response.factor_coulomb = true;
                response.exchange = if exchange { 1. } else { 0. };
            }
            _ => return Err(invalid("a Kohn-Sham reference returned no grid fields")),
        }
        Ok(response)
    }
    pub fn dimension(&self) -> usize {
        self.o * self.v
    }
    pub fn occupied(&self) -> usize {
        self.o
    }
    pub fn virtual_orbitals(&self) -> usize {
        self.v
    }
    /// Canonical orbitals of the reference in the orthonormal frame (columns, ascending energy).
    pub fn orbitals(&self) -> &DMatrix<f64> {
        &self.orbitals
    }
    pub fn orbital_energies(&self) -> &[f64] {
        &self.energies
    }
    /// `e_a - e_i`, index `i * v + a`.
    pub fn delta(&self) -> &[f64] {
        &self.delta
    }
    pub fn spin(&self) -> Spin {
        self.spin
    }
    /// The trial vector `y` (index `i * v + a`) as an occupied x virtual matrix.
    pub fn matrix(&self, y: &[f64]) -> DMatrix<f64> {
        DMatrix::from_row_slice(self.o, self.v, y)
    }
    fn flat(&self, k: &DMatrix<f64>) -> Vec<f64> {
        (0..self.o)
            .flat_map(|i| (0..self.v).map(move |a| (i, a)))
            .map(|(i, a)| k[(i, a)])
            .collect()
    }
    /// `<i| F1[dD_sym(y)] |a>` of every column: `(A + B) y - (e_a - e_i) y`.
    pub fn symmetric_kernel(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let count = block.ncols();
        let mut out = DMatrix::zeros(self.dimension(), count);
        if block.nrows() != self.dimension() {
            return Err(invalid(
                "a trial block does not match the response dimension",
            ));
        }
        let singlet = self.spin == Spin::Singlet;
        let ys: Vec<DMatrix<f64>> = (0..count)
            .map(|c| self.matrix(block.column(c).as_slice()))
            .collect();
        let mut kernels: Vec<DMatrix<f64>> = vec![DMatrix::zeros(self.o, self.v); count];
        if self.samples.is_some() {
            let ctx = &self.integrals.context;
            let lanes = ctx.lanes(Level::Threads, count);
            let mut states = (0..lanes)
                .map(|_| self.lane())
                .collect::<Result<Vec<_>>>()?;
            let grid = ctx.map_tasks(Level::Threads, &mut states, count, |lane, c| {
                self.grid_kernel(&ys[c], lane)
            })?;
            for (k, g) in kernels.iter_mut().zip(grid) {
                *k += g;
            }
        }
        for (k, y) in kernels.iter_mut().zip(&ys) {
            if singlet {
                for (w, b) in &self.pairs {
                    *k += b * (4. * w * b.dot(y));
                }
            }
            if let Some(factors) = &self.factors {
                if singlet && self.factor_coulomb {
                    *k += factors.coulomb(y) * 4.;
                }
                if self.exchange != 0. {
                    *k -= factors.exchange(y, 1.) * self.exchange;
                }
            }
            if let Some(kernels) = &self.polarization {
                // dD_up = dD, dD_dn = -dD about D_up = D_dn = D; the polarization's first
                // derivative vanishes there, its second does not.
                let c = &self.orbitals;
                let (co, cv) = (c.columns(0, self.o), c.columns(self.o, self.v));
                let change = co * y * cv.transpose();
                let change = &change + change.transpose();
                let [up, _] = kernels.apply(&change, &(-&change))?;
                *k -= co.transpose() * up * cv * self.exchange;
            }
        }
        for (c, k) in kernels.iter().enumerate() {
            out.set_column(c, &nalgebra::DVector::from_vec(self.flat(k)));
        }
        Ok(out)
    }
    /// `<i| F1[dD_anti(y)] |a>` of every column: `(A - B) y - (e_a - e_i) y`.
    pub fn antisymmetric_kernel(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let mut out = DMatrix::zeros(self.dimension(), block.ncols());
        if block.nrows() != self.dimension() {
            return Err(invalid(
                "a trial block does not match the response dimension",
            ));
        }
        if let (Some(factors), true) = (&self.factors, self.exchange != 0.) {
            for c in 0..block.ncols() {
                let y = self.matrix(block.column(c).as_slice());
                let k = factors.exchange(&y, -1.) * -self.exchange;
                out.set_column(c, &nalgebra::DVector::from_vec(self.flat(&k)));
            }
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
            density: ctx.allocate(n)?,
            spins: ctx.allocate(2 * n)?,
            potentials: ctx.allocate(2 * n)?,
            total: ctx.allocate(if self.poisson { n } else { 0 })?,
            hartree: ctx.allocate(if self.poisson { n } else { 0 })?,
        })
    }
    /// The grid part of the kernel of one trial matrix: Hartree and XC.
    fn grid_kernel(&self, y: &DMatrix<f64>, lane: &mut Lane) -> Result<DMatrix<f64>> {
        let grid = &self.integrals.grid;
        let n = grid.index.points();
        let singlet = self.spin == Spin::Singlet;
        self.density(y, lane.density.as_mut_slice())?;
        let potential = lane.potentials.as_mut_slice();
        potential.fill(0.);
        if let Some(kernel) = &self.xc {
            let (density, spins) = (lane.density.as_slice(), lane.spins.as_mut_slice());
            for k in 0..n {
                spins[k] = density[k];
                spins[n + k] = if singlet { density[k] } else { -density[k] };
            }
            let scratch = lane
                .xc
                .as_mut()
                .ok_or_else(|| invalid("an XC lane has no scratch"))?;
            kernel.apply(grid, lane.spins.as_slice(), potential, scratch)?;
        }
        if let Some(poisson) = lane.poisson.as_mut() {
            let (density, total) = (lane.density.as_slice(), lane.total.as_mut_slice());
            for k in 0..n {
                total[k] = C64::new(2. * density[k], 0.);
            }
            poisson.solve_into(lane.total.as_slice(), lane.hartree.as_mut_slice())?;
            for (p, h) in potential[..n].iter_mut().zip(lane.hartree.as_slice()) {
                *p += h.re;
            }
        }
        self.elements(&potential[..n])
    }
    /// `dn(r) = 2 sum_ia y_ia phi_i(r) phi_a(r)`, the density change of one spin.
    pub(super) fn density(&self, y: &DMatrix<f64>, out: &mut [f64]) -> Result<()> {
        let samples = self
            .samples
            .as_ref()
            .ok_or_else(|| invalid("this response has no orbital samples"))?
            .as_slice();
        pair_density(samples, self.integrals.grid.index.points(), self.o, self.v, y, out)
    }
    /// `<i| w |a> = dv sum_r phi_i(r) w(r) phi_a(r)` of a local potential `w`.
    fn elements(&self, potential: &[f64]) -> Result<DMatrix<f64>> {
        let samples = self
            .samples
            .as_ref()
            .ok_or_else(|| invalid("this response has no orbital samples"))?
            .as_slice();
        let grid = &self.integrals.grid.index;
        local_elements(samples, grid.points(), self.o, self.v, potential, grid.dv())
    }
}

impl super::Operator for Response<'_> {
    fn dimension(&self) -> usize {
        Response::dimension(self)
    }
    fn delta(&self) -> &[f64] {
        Response::delta(self)
    }
    fn sum(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        Response::sum(self, block)
    }
    fn difference(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        Response::difference(self, block)
    }
    fn resonant(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        Response::resonant(self, block)
    }
    fn spin(&self) -> Spin {
        self.spin
    }
}

/// `dn(r) = 2 sum_ia y_ia phi_i(r) phi_a(r)` of one spin, from the orbital samples
/// `[orbital][point]` of `n` points (occupied first).
pub(super) fn pair_density(
    samples: &[f64],
    n: usize,
    o: usize,
    v: usize,
    y: &DMatrix<f64>,
    out: &mut [f64],
) -> Result<()> {
    if out.len() != n || y.shape() != (o, v) {
        return Err(invalid("a transition density does not match the grid"));
    }
    let yt = y.transpose();
    for start in (0..n).step_by(BLOCK) {
        let count = BLOCK.min(n - start);
        let po = view(samples, start, count, o, n);
        let pv = view(samples, o * n + start, count, v, n);
        let t = pv * &yt;
        for i in 0..count {
            out[start + i] = 2. * (0..o).map(|j| po[(i, j)] * t[(i, j)]).sum::<f64>();
        }
    }
    Ok(())
}

/// `<i| w |a> = dv sum_r phi_i(r) w(r) phi_a(r)` of a local potential `w`, from the orbital
/// samples `[orbital][point]` of `n` points (occupied first).
pub(super) fn local_elements(
    samples: &[f64],
    n: usize,
    o: usize,
    v: usize,
    potential: &[f64],
    dv: f64,
) -> Result<DMatrix<f64>> {
    let mut out = DMatrix::zeros(o, v);
    for start in (0..n).step_by(BLOCK) {
        let count = BLOCK.min(n - start);
        let po = view(samples, start, count, o, n);
        let pv = view(samples, o * n + start, count, v, n);
        let weighted = DMatrix::from_fn(count, v, |i, a| pv[(i, a)] * potential[start + i]);
        out += po.transpose() * weighted;
    }
    Ok(out * dv)
}

pub(super) fn view(
    samples: &[f64],
    offset: usize,
    rows: usize,
    columns: usize,
    leading: usize,
) -> DMatrixView<'_, f64> {
    DMatrixView::from_slice_with_strides_generic(
        &samples[offset..],
        Dyn(rows),
        Dyn(columns),
        Const::<1>,
        Dyn(leading),
    )
}

/// The canonical orbitals on the grid, `[orbital][point]`.
pub(super) fn mo_samples(
    integrals: &MolecularIntegrals,
    orbitals: &DMatrix<f64>,
) -> Result<Buffer<f64>> {
    let ctx = &integrals.context;
    let m = integrals.m();
    let n = integrals.grid.index.points();
    let mut out = ctx.allocate::<f64>(product(&[n, m])?)?;
    let mut blocks = SampleBlocks::new(ctx, &integrals.orthonormal_samples, m, n)?;
    let stride = blocks.stride();
    for start in (0..n).step_by(stride) {
        let count = blocks.load(start)?;
        let samples = DMatrixView::from_slice_with_strides_generic(
            blocks.values(),
            Dyn(count),
            Dyn(m),
            Const::<1>,
            Dyn(stride),
        );
        let mo = samples * orbitals;
        for j in 0..m {
            for i in 0..count {
                out.as_mut_slice()[j * n + start + i] = mo[(i, j)];
            }
        }
    }
    Ok(out)
}
