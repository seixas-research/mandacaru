//! Native weighted-k SCF, generalized Hermitian solves and density/moment DIIS.
use super::{
    invalid,
    spectrum::{self, Smearing},
    Crystal, KPoint,
};
use crate::{
    math::{
        fft::Fft3,
        linalg::eigen,
        storage::{MatrixSource, StoredMatrix},
    },
    models::xc,
    product,
    quantum::C64,
    Buffer, Reservation, Result,
};
use nalgebra::{DMatrix, DVector};
type Matrix = DMatrix<C64>;
#[derive(Clone)]
pub struct Options {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub density_tolerance: f64,
    pub smearing: Smearing,
    pub width: f64,
    pub mixing: f64,
    pub history: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_iterations: 300,
            tolerance: 1e-7,
            density_tolerance: 1e-5,
            smearing: Smearing::FermiDirac,
            width: 0.1 / 27.211386245988,
            mixing: 0.25,
            history: 7,
        }
    }
}
#[derive(Clone)]
pub struct Iteration {
    pub iteration: usize,
    pub free_energy: f64,
    pub delta: Option<f64>,
    pub residual: f64,
    pub moment_residual: f64,
    pub magnetization: f64,
    pub kinetic_residual: f64,
}
pub struct PeriodicResult {
    pub energy: f64,
    pub free_energy: f64,
    pub extrapolated_energy: f64,
    pub fermi: f64,
    pub converged: bool,
    pub history: Vec<Iteration>,
    pub eigenvalues: Vec<Vec<f64>>,
    pub occupations: Vec<Vec<f64>>,
    pub coefficients: Vec<StoredMatrix>,
    pub matrices: Vec<StoredMatrix>,
    pub density: Buffer<f64>,
    pub moments: Buffer<C64>,
    pub potential: Buffer<f64>,
    pub moment_potential: Buffer<C64>,
    pub kinetic_density: Option<Buffer<f64>>,
    pub kinetic_potential: Option<Buffer<f64>>,
    pub terms: [f64; 5],
    pub magnetization: f64,
    pub spin_orbit_energy: Option<f64>,
    pub options: Options,
    _memory: Reservation,
}
pub(super) fn sym_field(c: &Crystal, field: &[f64], out: &mut [f64]) {
    if c.permutations.is_empty() {
        out.copy_from_slice(field);
        return;
    }
    for k in 0..field.len() {
        out[k] =
            c.permutations.iter().map(|p| field[p[k]]).sum::<f64>() / c.permutations.len() as f64;
    }
}
fn reciprocal(c: &Crystal, fft: &mut Fft3, input: &[f64]) -> Result<Buffer<C64>> {
    let mut out = c.context.allocate::<C64>(input.len())?;
    for (a, &b) in out.as_mut_slice().iter_mut().zip(input) {
        *a = C64::new(b, 0.);
    }
    fft.transform(out.as_mut_slice(), false)?;
    for (k, v) in out.as_mut_slice().iter_mut().enumerate() {
        *v *= c.phase.as_slice()[k] * c.grid.index.dv();
    }
    Ok(out)
}
fn inverse(c: &Crystal, fft: &mut Fft3, input: &[C64]) -> Result<Buffer<f64>> {
    let mut out = c.context.allocate::<C64>(input.len())?;
    for (k, value) in input.iter().enumerate() {
        out.as_mut_slice()[k] = value * c.phase.as_slice()[k].conj() / c.grid.index.dv();
    }
    fft.transform(out.as_mut_slice(), true)?;
    let mut real = c.context.allocate::<f64>(input.len())?;
    for (r, v) in real.as_mut_slice().iter_mut().zip(out.as_slice()) {
        *r = v.re;
    }
    Ok(real)
}
pub(super) fn derivative(
    c: &Crystal,
    fft: &mut Fft3,
    input: &[f64],
    axis: usize,
) -> Result<Buffer<f64>> {
    let n = c.grid.index.points();
    let mut out = c.context.allocate::<C64>(n)?;
    for (k, value) in input.iter().enumerate() {
        out.as_mut_slice()[k] = C64::new(*value, 0.);
    }
    fft.transform(out.as_mut_slice(), false)?;
    for k in 0..n {
        out.as_mut_slice()[k] *= C64::new(0., c.xc_wave.as_slice()[axis * n + k]);
    }
    fft.transform(out.as_mut_slice(), true)?;
    let mut real = c.context.allocate::<f64>(n)?;
    for k in 0..n {
        real.as_mut_slice()[k] = out.as_slice()[k].re;
    }
    Ok(real)
}
struct Electro {
    energy: f64,
    potential: Buffer<f64>,
    moments: Buffer<C64>,
}
fn electrostatic(c: &Crystal, fft: &mut Fft3, rho: &[f64], q: &[C64]) -> Result<Electro> {
    let n = c.grid.index.points();
    let h = c.channels;
    let volume = n as f64 * c.grid.index.dv();
    let rho = reciprocal(c, fft, rho)?;
    let mut hats = c.context.allocate::<C64>(n)?;
    let mut potential = c.context.allocate::<C64>(n)?;
    let mut moments = c.context.allocate::<C64>(h)?;
    let mut energy = c.eion + c.ion_constant;
    for (a, &charge) in q.iter().enumerate().take(h) {
        let table = c.hats.read_range(&c.context, a * n, n)?;
        for k in 0..n {
            hats.as_mut_slice()[k] += charge * table[k];
        }
        moments.as_mut_slice()[a] = (0..n)
            .map(|k| c.kernel.as_slice()[k] * rho.as_slice()[k].conj() * table[k])
            .sum::<C64>()
            / volume;
    }
    for k in 0..n {
        energy += c.kernel.as_slice()[k]
            * (0.5 * rho.as_slice()[k].norm_sqr()
                + (rho.as_slice()[k].conj() * (hats.as_slice()[k] + c.ion.as_slice()[k])).re)
            / volume;
        potential.as_mut_slice()[k] =
            c.kernel.as_slice()[k] * (rho.as_slice()[k] + hats.as_slice()[k] + c.ion.as_slice()[k]);
    }
    for a in 0..h {
        let uq = (0..h)
            .map(|b| c.u.as_slice()[a * h + b] * q[b])
            .sum::<C64>();
        energy += (0.5 * q[a].conj() * uq
            + q[a].conj() * c.uion.as_slice()[a]
            + q[a] * c.linear.as_slice()[a])
            .re;
        moments.as_mut_slice()[a] += (0..h)
            .map(|b| q[b].conj() * c.u.as_slice()[b * h + a])
            .sum::<C64>()
            + c.uion.as_slice()[a].conj()
            + c.linear.as_slice()[a];
    }
    Ok(Electro {
        energy,
        potential: inverse(c, fft, potential.as_slice())?,
        moments,
    })
}
fn exchange_correlation(
    c: &Crystal,
    fft: &mut Fft3,
    rho: &[f64],
    tau: Option<&[f64]>,
) -> Result<super::fields::Terms> {
    if c.spinor {
        return super::spinor::exchange_correlation(c, fft, rho);
    }
    collinear_xc(c, fft, rho, tau, c.spins)
}
pub(super) fn collinear_xc(
    c: &Crystal,
    fft: &mut Fft3,
    rho: &[f64],
    tau: Option<&[f64]>,
    spins: usize,
) -> Result<super::fields::Terms> {
    if !matches!(c.functional.as_str(), "lda" | "pbe") {
        return super::fields::evaluate(c, fft, rho, tau);
    }
    let n = c.grid.index.points();
    let ctx = &c.context;
    let mut densities = ctx.allocate::<f64>(2 * n)?;
    for spin in 0..2 {
        for k in 0..n {
            densities.as_mut_slice()[spin * n + k] = (if spins == 1 {
                0.5 * rho[k]
            } else {
                rho[spin * n + k]
            } + 0.5 * c.core.as_slice()[k])
                .max(0.);
        }
    }
    let mut gradients = ctx.allocate::<f64>(6 * n)?;
    let mut flux = ctx.allocate::<f64>(6 * n)?;
    let mut potential = ctx.allocate::<f64>(2 * n)?;
    if c.functional == "pbe" {
        for s in 0..2 {
            for a in 0..3 {
                let g = derivative(c, fft, &densities.as_slice()[s * n..(s + 1) * n], a)?;
                gradients.as_mut_slice()[(3 * s + a) * n..(3 * s + a + 1) * n]
                    .copy_from_slice(g.as_slice());
            }
        }
    }
    let mut energy = 0.;
    for k in 0..n {
        let mut sigma = [0.; 3];
        let live = densities.as_slice()[k] + densities.as_slice()[n + k] > 1e-10;
        if live {
            for a in 0..3 {
                let u = gradients.as_slice()[a * n + k];
                let d = gradients.as_slice()[(a + 3) * n + k];
                sigma[0] += u * u;
                sigma[1] += u * d;
                sigma[2] += d * d;
            }
        }
        let (f, df) = xc::partials_relativistic(
            &c.functional,
            [
                densities.as_slice()[k],
                densities.as_slice()[n + k],
                sigma[0],
                sigma[1],
                sigma[2],
            ],
            c.relativistic,
        )?;
        energy += f;
        potential.as_mut_slice()[k] = df[0];
        potential.as_mut_slice()[n + k] = df[1];
        if live && c.functional == "pbe" {
            for a in 0..3 {
                let u = gradients.as_slice()[a * n + k];
                let d = gradients.as_slice()[(a + 3) * n + k];
                flux.as_mut_slice()[a * n + k] = 2. * df[2] * u + df[3] * d;
                flux.as_mut_slice()[(a + 3) * n + k] = 2. * df[4] * d + df[3] * u;
            }
        }
    }
    if c.functional == "pbe" {
        for s in 0..2 {
            for a in 0..3 {
                let divergence = derivative(
                    c,
                    fft,
                    &flux.as_slice()[(s * 3 + a) * n..(s * 3 + a + 1) * n],
                    a,
                )?;
                for k in 0..n {
                    potential.as_mut_slice()[s * n + k] -= divergence.as_slice()[k];
                }
            }
        }
    }
    let mut result = ctx.allocate::<f64>(spins * n)?;
    for s in 0..spins {
        for k in 0..n {
            let raw = if spins == 1 {
                rho[k] + c.core.as_slice()[k]
            } else {
                rho[s * n + k] + 0.5 * c.core.as_slice()[k]
            };
            result.as_mut_slice()[s * n + k] = if raw < 0. {
                0.
            } else {
                potential.as_slice()[s * n + k]
            };
        }
    }
    Ok(super::fields::Terms {
        energy: energy * c.grid.index.dv(),
        potential: result,
        kinetic_potential: None,
    })
}
struct LocalPotential {
    local: Buffer<f64>,
    moments: Buffer<C64>,
    kinetic: Option<Buffer<f64>>,
}
fn local_potential(
    c: &Crystal,
    fft: &mut Fft3,
    rho: &[f64],
    q: &[C64],
    tau: Option<&[f64]>,
) -> Result<LocalPotential> {
    let n = c.grid.index.points();
    let mut total = c.context.allocate::<f64>(n)?;
    for k in 0..n {
        total.as_mut_slice()[k] = if c.spinor {
            rho[k]
        } else {
            (0..c.spins).map(|s| rho[s * n + k]).sum()
        };
    }
    let electro = electrostatic(c, fft, total.as_slice(), q)?;
    let terms = exchange_correlation(c, fft, rho, tau)?;
    let mut xc = terms.potential;
    let mut kinetic = terms.kinetic_potential;
    let mut work = c.context.allocate::<f64>(n)?;
    for s in 0..c.field_channels() {
        for k in 0..n {
            work.as_mut_slice()[k] = xc.as_slice()[s * n + k]
                + if !c.spinor || s == 0 {
                    electro.potential.as_slice()[k]
                } else {
                    0.
                };
        }
        sym_field(
            c,
            work.as_slice(),
            &mut xc.as_mut_slice()[s * n..(s + 1) * n],
        );
        if let Some(v) = &mut kinetic {
            work.as_mut_slice()
                .copy_from_slice(&v.as_slice()[s * n..(s + 1) * n]);
            sym_field(
                c,
                work.as_slice(),
                &mut v.as_mut_slice()[s * n..(s + 1) * n],
            );
        }
    }
    Ok(LocalPotential {
        local: xc,
        moments: electro.moments,
        kinetic,
    })
}
pub fn diagonalize(
    c: &Crystal,
    data: &KPoint,
    v: &[f64],
    w: &[C64],
    kinetic_potential: Option<&[f64]>,
) -> Result<(Vec<f64>, Matrix, Matrix)> {
    if c.spinor {
        return super::spinor::diagonalize(c, data, v, w);
    }
    let m = c.m;
    let p = c.p;
    let n = c.grid.index.points();
    let operator = Matrix::from_fn(p, p, |i, j| {
        (0..c.channels)
            .map(|a| w[a] * c.blocks.as_slice()[(a * p + i) * p + j])
            .sum()
    });
    let mut field = c.context.allocate::<f64>(n)?;
    if v.len() != n {
        return Err(invalid("local periodic potential must match the grid"));
    }
    for (to, from) in field.as_mut_slice().iter_mut().zip(v) {
        *to = from * c.grid.index.dv();
    }
    let samples = data.psi.read(&c.context)?;
    let mut projector = crate::math::projection::ComplexProjector::new(&c.context, &samples, m, n)?;
    let local = projector.project(field.as_slice())?;
    let mut h = &data.fixed
        + &data.projections * operator * data.projections.adjoint()
        + Matrix::from_column_slice(m, m, local);
    drop(projector);
    drop(samples);
    drop(field);
    if let Some(v) = kinetic_potential {
        let operator = super::fields::kinetic_operator(c, data, v)?;
        for a in 0..m {
            for b in 0..m {
                h[(a, b)] += operator.as_slice()[a + b * m];
            }
        }
    }
    let h = (&h + h.adjoint()) * C64::new(0.5, 0.);
    let (eps, y) = eigen(data.lowdin.adjoint() * h * &data.lowdin)?;
    Ok((eps, &data.lowdin * &y, y))
}
fn density(c: &Crystal, matrices: &[StoredMatrix]) -> Result<(Buffer<f64>, Buffer<C64>)> {
    if c.spinor {
        return super::spinor::density(c, matrices);
    }
    let n = c.grid.index.points();
    let p = c.p;
    let mut rho = c.context.allocate::<f64>(c.spins * n)?;
    let mut q = c.context.allocate::<C64>(c.channels)?;
    let mut projected = Matrix::zeros(p, p);
    for s in 0..c.spins {
        let mut raw = c.context.allocate::<f64>(n)?;
        for (k, data) in c.points.iter().enumerate() {
            let d = matrices[s * c.points.len() + k].matrix(&c.context)?;
            projected +=
                data.projections.adjoint() * &*d * &data.projections * C64::new(data.weight, 0.);
            let contribution = crate::math::projection::hermitian_density(
                &c.context,
                &data.psi.read(&c.context)?,
                &d,
                n,
            )?;
            for (to, from) in raw.as_mut_slice().iter_mut().zip(contribution.as_slice()) {
                *to += data.weight * from;
            }
        }
        sym_field(
            c,
            raw.as_slice(),
            &mut rho.as_mut_slice()[s * n..(s + 1) * n],
        );
    }
    let mut raw = c.context.allocate::<C64>(c.channels)?;
    for a in 0..c.channels {
        raw.as_mut_slice()[a] = (0..p)
            .flat_map(|i| (0..p).map(move |j| (i, j)))
            .map(|(i, j)| projected[(i, j)] * c.blocks.as_slice()[(a * p + j) * p + i])
            .sum();
    }
    for a in 0..c.channels {
        q.as_mut_slice()[a] = (0..c.channels)
            .map(|b| c.qsym.as_slice()[a * c.channels + b] * raw.as_slice()[b])
            .sum();
    }
    Ok((rho, q))
}
fn energy(
    c: &Crystal,
    fft: &mut Fft3,
    matrices: &[StoredMatrix],
    rho: &[f64],
    q: &[C64],
    tau: Option<&[f64]>,
) -> Result<[f64; 4]> {
    let n = c.grid.index.points();
    let mut total = c.context.allocate::<f64>(n)?;
    for k in 0..n {
        total.as_mut_slice()[k] = if c.spinor {
            rho[k]
        } else {
            (0..c.spins).map(|s| rho[s * n + k]).sum()
        };
    }
    let band = (0..c.spins)
        .flat_map(|s| c.points.iter().enumerate().map(move |(k, d)| (s, k, d)))
        .map(|(s, k, d)| {
            let matrix = matrices[s * c.points.len() + k].matrix(&c.context)?;
            if c.spinor {
                Ok(d.weight * super::spinor::fixed_energy(d, &matrix, c.m))
            } else {
                Ok(d.weight * (&*matrix * &d.fixed).trace().re)
            }
        })
        .sum::<Result<f64>>()?;
    Ok([
        band,
        electrostatic(c, fft, total.as_slice(), q)?.energy,
        exchange_correlation(c, fft, rho, tau)?.energy,
        c.constant,
    ])
}
fn pack(c: &Crystal, rho: &[f64], q: &[C64], tau: Option<&[f64]>) -> Vec<C64> {
    let n = c.grid.index.points();
    let fields = c.field_channels() * n;
    let end = fields + c.channels;
    let mut x = vec![C64::default(); end + tau.map_or(0, |t| t.len())];
    if c.spinor {
        for (to, &from) in x[..fields].iter_mut().zip(rho) {
            *to = C64::new(from, 0.);
        }
    } else {
        for k in 0..n {
            x[k] = C64::new(
                if c.spins == 1 {
                    rho[k]
                } else {
                    rho[k] + rho[n + k]
                },
                0.,
            );
            if c.spins == 2 {
                x[n + k] = C64::new(rho[k] - rho[n + k], 0.);
            }
        }
    }
    x[fields..end].copy_from_slice(q);
    if let Some(tau) = tau {
        for (to, &from) in x[end..].iter_mut().zip(tau) {
            *to = C64::new(from, 0.);
        }
    }
    x
}
fn unpack(c: &Crystal, x: &[C64], rho: &mut [f64], q: &mut [C64], tau: Option<&mut [f64]>) {
    let n = c.grid.index.points();
    let fields = c.field_channels() * n;
    if c.spinor {
        for (to, from) in rho.iter_mut().zip(&x[..fields]) {
            *to = from.re;
        }
    } else {
        for k in 0..n {
            if c.spins == 1 {
                rho[k] = x[k].re;
            } else {
                rho[k] = 0.5 * (x[k] + x[n + k]).re;
                rho[n + k] = 0.5 * (x[k] - x[n + k]).re;
            }
        }
    }
    let end = fields + c.channels;
    q.copy_from_slice(&x[fields..end]);
    if let Some(tau) = tau {
        for (to, from) in tau.iter_mut().zip(&x[end..]) {
            *to = from.re;
        }
    }
}
struct Mixer {
    history: Vec<(StoredMatrix, StoredMatrix)>,
}
impl Mixer {
    fn next(
        &mut self,
        c: &Crystal,
        fft: &mut Fft3,
        input: Vec<C64>,
        output: Vec<C64>,
        options: &Options,
        warmup: bool,
    ) -> Result<Vec<C64>> {
        let residual = output
            .iter()
            .zip(&input)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        if self.history.len() == options.history {
            self.history.remove(0);
        }
        let (mut x, mut r) = if warmup {
            (input, residual)
        } else {
            let archive =
                product(&[input.len(), 16, 2, options.history])? > c.context.memory_limit() / 8;
            self.history.push((
                StoredMatrix::new(&c.context, Matrix::from_vec(input.len(), 1, input), archive)?,
                StoredMatrix::new(
                    &c.context,
                    Matrix::from_vec(residual.len(), 1, residual),
                    archive,
                )?,
            ));
            let h = self.history.len();
            let mut b = DMatrix::<f64>::zeros(h + 1, h + 1);
            for i in 0..h {
                for j in 0..h {
                    let left = self.history[i].1.matrix(&c.context)?;
                    let right = self.history[j].1.matrix(&c.context)?;
                    b[(i, j)] = left
                        .iter()
                        .zip(right.iter())
                        .enumerate()
                        .map(|(k, (a, b))| {
                            (a.conj() * b).re
                                * if k < c.grid.index.points() * c.field_channels()
                                    || k >= c.grid.index.points() * c.field_channels() + c.channels
                                {
                                    c.grid.index.dv()
                                } else {
                                    1.
                                }
                        })
                        .sum();
                }
                b[(i, h)] = -1.;
                b[(h, i)] = -1.;
            }
            let mut rhs = DVector::zeros(h + 1);
            rhs[h] = -1.;
            let alpha = b
                .lu()
                .solve(&rhs)
                .filter(|a| a.iter().all(|v| v.is_finite() && v.abs() < 1e12));
            let mut x = vec![C64::default(); output.len()];
            let mut r = x.clone();
            for i in 0..h {
                let a = alpha
                    .as_ref()
                    .map_or(if i == h - 1 { 1. } else { 0. }, |a| a[i]);
                let input = self.history[i].0.matrix(&c.context)?;
                let residual = self.history[i].1.matrix(&c.context)?;
                for k in 0..x.len() {
                    x[k] += a * input[k];
                    r[k] += a * residual[k];
                }
            }
            (x, r)
        };
        let n = c.grid.index.points();
        let mut spec = c.context.allocate::<C64>(n)?;
        spec.as_mut_slice().copy_from_slice(&r[..n]);
        fft.transform(spec.as_mut_slice(), false)?;
        for k in 0..n {
            spec.as_mut_slice()[k] *= c.kerker.as_slice()[k];
        }
        fft.transform(spec.as_mut_slice(), true)?;
        for (k, r) in r.iter_mut().enumerate().take(n) {
            *r = C64::new(spec.as_slice()[k].re, 0.);
        }
        for (x, r) in x.iter_mut().zip(&r) {
            *x += options.mixing * r;
        }
        if x.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
            return Err(crate::Error::Kernel(
                "nonfinite periodic density mixing".into(),
            ));
        }
        Ok(x)
    }
}
impl Crystal {
    pub fn solve(&self, options: &Options) -> Result<PeriodicResult> {
        let c = self;
        let ctx = &c.context;
        let n = c.grid.index.points();
        let nk = c.points.len();
        let m = c.bands();
        let meta = crate::models::functional(&c.functional)?.requires_tau;
        if c.spinor && c.functional != "lda" && c.functional != "pbe" {
            return Err(invalid("periodic spinor XC currently requires LDA or PBE"));
        }
        if options.max_iterations == 0
            || options.history == 0
            || options.history > 32
            || !options.tolerance.is_finite()
            || options.tolerance <= 0.
            || !options.density_tolerance.is_finite()
            || options.density_tolerance <= 0.
            || !options.width.is_finite()
            || options.width <= 0.
            || !options.mixing.is_finite()
            || options.mixing <= 0.
            || options.mixing > 1.
        {
            return Err(invalid(
                "invalid periodic SCF iteration, convergence, smearing or mixing options",
            ));
        }
        ctx.run(|| {
            let retained = product(&[options.max_iterations, 128])?;
            let archive_matrices = product(&[c.spins, nk, m, m, 32])? > ctx.memory_limit() / 8;
            let mut memory = ctx.reserve(
                retained
                    .checked_add(product(&[
                        c.field_channels() * n * (1 + usize::from(meta)) + c.channels,
                        16,
                        8,
                    ])?)
                    .ok_or(crate::Error::Overflow)?,
            )?;
            let _projector_work = ctx.reserve(product(&[c.p, c.p, 16, 12])?)?;
            let _matrix_work = ctx.reserve(product(&[m, m, 16, 16])?)?;
            let mut fft = Fft3::new(ctx, c.grid.index.shape())?;
            let mut rho = ctx.allocate::<f64>(c.field_channels() * n)?;
            let mut q = ctx.allocate::<C64>(c.channels)?;
            for k in 0..n {
                if c.spinor {
                    rho.as_mut_slice()[k] = c.initial.as_slice()[k];
                    if !c.time_reversal {
                        for a in 0..3 {
                            rho.as_mut_slice()[(a + 1) * n + k] = c.initial_m.as_slice()[a * n + k];
                        }
                    }
                } else if c.spins == 1 {
                    rho.as_mut_slice()[k] = c.initial.as_slice()[k];
                } else {
                    rho.as_mut_slice()[k] =
                        0.5 * (c.initial.as_slice()[k] + c.initial_m.as_slice()[k]);
                    rho.as_mut_slice()[n + k] =
                        0.5 * (c.initial.as_slice()[k] - c.initial_m.as_slice()[k]);
                }
            }
            q.as_mut_slice().copy_from_slice(c.initial_q.as_slice());
            let mut tau = if meta {
                Some(super::fields::initial_kinetic_density(c, options)?)
            } else {
                None
            };
            let mut previous = None;
            let mut history = vec![];
            let mut mixer = Mixer { history: vec![] };
            for iteration in 1..=options.max_iterations {
                let LocalPotential {
                    local: v,
                    moments: w,
                    kinetic: kinetic_potential,
                } = local_potential(
                    c,
                    &mut fft,
                    rho.as_slice(),
                    q.as_slice(),
                    tau.as_ref().map(|t| t.as_slice()),
                )?;
                let mut eigenvalues = vec![];
                let mut coefficients = vec![];
                for s in 0..c.spins {
                    for data in &c.points {
                        let (eps, coeff, _) = diagonalize(
                            c,
                            data,
                            if c.spinor {
                                v.as_slice()
                            } else {
                                &v.as_slice()[s * n..(s + 1) * n]
                            },
                            w.as_slice(),
                            kinetic_potential
                                .as_ref()
                                .map(|v| &v.as_slice()[s * n..(s + 1) * n]),
                        )?;
                        eigenvalues.push(eps);
                        coefficients.push(StoredMatrix::new(ctx, coeff, archive_matrices)?);
                    }
                }
                let weights: Vec<_> = (0..c.spins)
                    .flat_map(|_| c.points.iter().map(|p| p.weight))
                    .collect();
                let degeneracy = c.degeneracy();
                let mu = spectrum::fermi(
                    &eigenvalues,
                    &weights,
                    c.electrons,
                    options.smearing,
                    options.width,
                    degeneracy,
                )?;
                let occupations: Vec<Vec<f64>> = eigenvalues
                    .iter()
                    .map(|e| {
                        e.iter()
                            .map(|e| {
                                degeneracy * options.smearing.occupation((e - mu) / options.width)
                            })
                            .collect()
                    })
                    .collect();
                let matrices = coefficients
                    .iter()
                    .zip(&occupations)
                    .map(|(c, f)| {
                        let c = c.matrix(ctx)?;
                        let d =
                            &*c * Matrix::from_diagonal(&DVector::from_iterator(
                                m,
                                f.iter().map(|f| C64::new(*f, 0.)),
                            )) * c.adjoint();
                        StoredMatrix::new(ctx, d, archive_matrices)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let (output, qout) = density(c, &matrices)?;
                let tauout = if meta {
                    Some(super::fields::kinetic_density(c, &matrices)?)
                } else {
                    None
                };
                let terms = energy(
                    c,
                    &mut fft,
                    &matrices,
                    output.as_slice(),
                    qout.as_slice(),
                    tauout.as_ref().map(|t| t.as_slice()),
                )?;
                let energy = terms.iter().sum::<f64>();
                let entropy = -options.width
                    * degeneracy
                    * eigenvalues
                        .iter()
                        .zip(&weights)
                        .map(|(eps, w)| {
                            w * eps
                                .iter()
                                .map(|e| options.smearing.entropy((e - mu) / options.width))
                                .sum::<f64>()
                        })
                        .sum::<f64>();
                let free = energy + entropy;
                let x = pack(
                    c,
                    rho.as_slice(),
                    q.as_slice(),
                    tau.as_ref().map(|t| t.as_slice()),
                );
                let y = pack(
                    c,
                    output.as_slice(),
                    qout.as_slice(),
                    tauout.as_ref().map(|t| t.as_slice()),
                );
                let kinetic_residual = tau.as_ref().zip(tauout.as_ref()).map_or(0., |(a, b)| {
                    a.as_slice()
                        .iter()
                        .zip(b.as_slice())
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f64>()
                        * c.grid.index.dv()
                });
                let residual = x[..c.field_channels() * n]
                    .iter()
                    .zip(&y)
                    .map(|(a, b)| (a.re - b.re).abs())
                    .sum::<f64>()
                    * c.grid.index.dv();
                let moment_residual = q
                    .as_slice()
                    .iter()
                    .zip(qout.as_slice())
                    .map(|(a, b)| (a - b).norm())
                    .fold(0f64, f64::max);
                let magnetization = if c.spinor {
                    super::spinor::magnetic_moment(c, &matrices)?
                } else if c.spins == 2 {
                    (0..nk)
                        .map(|k| {
                            c.points[k].weight
                                * (occupations[k].iter().sum::<f64>()
                                    - occupations[nk + k].iter().sum::<f64>())
                        })
                        .sum()
                } else {
                    0.
                };
                let delta = previous.map(|p| free - p);
                history.push(Iteration {
                    iteration,
                    free_energy: free,
                    delta,
                    residual,
                    moment_residual,
                    magnetization,
                    kinetic_residual,
                });
                let converged = delta.is_some_and(|d: f64| d.abs() < options.tolerance)
                    && residual < options.density_tolerance
                    && kinetic_residual < options.density_tolerance
                    && moment_residual < options.density_tolerance;
                if converged || iteration == options.max_iterations {
                    let mut eigenvalues = eigenvalues;
                    for e in eigenvalues.iter_mut().flatten() {
                        *e += c.reference;
                    }
                    memory.shrink_to(retained)?;
                    let spin_orbit_energy = if c.spinor {
                        Some(super::spinor::spin_orbit_energy(c, &matrices)?)
                    } else {
                        None
                    };
                    return Ok(PeriodicResult {
                        energy,
                        free_energy: free,
                        extrapolated_energy: if options.smearing == Smearing::MethfesselPaxton {
                            free
                        } else {
                            0.5 * (energy + free)
                        },
                        fermi: mu + c.reference,
                        converged,
                        history,
                        eigenvalues,
                        occupations,
                        coefficients,
                        matrices,
                        density: output,
                        moments: qout,
                        potential: v,
                        moment_potential: w,
                        kinetic_density: tauout,
                        kinetic_potential,
                        terms: [terms[0], terms[1], terms[2], terms[3], entropy],
                        magnetization,
                        spin_orbit_energy,
                        options: options.clone(),
                        _memory: memory,
                    });
                }
                let next =
                    mixer.next(c, &mut fft, x, y, options, c.spins == 2 && iteration <= 12)?;
                unpack(
                    c,
                    &next,
                    rho.as_mut_slice(),
                    q.as_mut_slice(),
                    tau.as_mut().map(|t| t.as_mut_slice()),
                );
                previous = Some(free);
            }
            unreachable!()
        })
    }
}
