//! Bounded reusable FFT workspaces; no process-global plans or thread pools.
use super::{invalid, SpatialGrid};
use crate::{product, Buffer, Context, Reservation, Result};
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64 as C64;
use rustfft::{Fft, FftPlanner};
use std::{f64::consts::PI, sync::Arc};

pub struct Fft3 {
    context: Context,
    shape: [usize; 3],
    forward: Vec<Arc<dyn Fft<f64>>>,
    backward: Vec<Arc<dyn Fft<f64>>>,
    workers: Vec<FftWorker>,
    _plans: Reservation,
    _metadata: Reservation,
}
struct FftWorker {
    line: Buffer<C64>,
    scratch: Buffer<C64>,
}
impl Fft3 {
    pub fn new(ctx: &Context, shape: [usize; 3]) -> Result<Self> {
        if shape.contains(&0) {
            return Err(invalid("FFT dimensions must be positive"));
        }
        let n = *shape.iter().max().unwrap();
        product(&shape)?;
        // Covers rustfft's factorization, twiddle arrays and planner metadata.
        let plans = ctx.reserve(product(&[n, 4096])?)?;
        let mut planner = FftPlanner::new();
        let forward: Vec<_> = shape.iter().map(|&n| planner.plan_fft_forward(n)).collect();
        let backward: Vec<_> = shape.iter().map(|&n| planner.plan_fft_inverse(n)).collect();
        let work = forward
            .iter()
            .chain(&backward)
            .map(|p| p.get_inplace_scratch_len())
            .max()
            .unwrap();
        let workers = ctx.threads().min(shape[0].max(shape[1] * shape[2]));
        Ok(Self {
            context: ctx.clone(),
            shape,
            forward,
            backward,
            workers: (0..workers)
                .map(|_| {
                    Ok(FftWorker {
                        line: ctx.allocate(n)?,
                        scratch: ctx.allocate(work)?,
                    })
                })
                .collect::<Result<_>>()?,
            _plans: plans,
            _metadata: ctx.reserve(product(&[workers, n, 64])?)?,
        })
    }
    pub fn transform(&mut self, values: &mut [C64], inverse: bool) -> Result<()> {
        let context = self.context.clone();
        context.run(|| self.transform_inner(values, inverse))
    }
    fn transform_inner(&mut self, values: &mut [C64], inverse: bool) -> Result<()> {
        if values.len() != product(&self.shape)? {
            return Err(invalid("FFT input does not match its shape"));
        }
        let [nx, ny, nz] = self.shape;
        let worker_count = if values.len() >= 32768 {
            self.workers.len()
        } else {
            1
        };
        for axis in (0..3).rev() {
            let plan = if inverse {
                &self.backward[axis]
            } else {
                &self.forward[axis]
            };
            if axis == 2 {
                let chunk = (nx * ny).div_ceil(worker_count) * nz;
                std::thread::scope(|scope| {
                    for (values, worker) in values.chunks_mut(chunk).zip(&mut self.workers) {
                        let mut task = move || {
                            // RustFFT accepts a contiguous batch of complete lines.
                            plan.process_with_scratch(values, worker.scratch.as_mut_slice());
                        };
                        if worker_count == 1 {
                            task();
                        } else {
                            mandacaru_kernels::runtime::spawn(scope, task);
                        }
                    }
                });
            } else if axis == 1 {
                let chunk = nx.div_ceil(worker_count) * ny * nz;
                std::thread::scope(|scope| {
                    for (values, worker) in values.chunks_mut(chunk).zip(&mut self.workers) {
                        let mut task = move || {
                            let line = &mut worker.line.as_mut_slice()[..ny];
                            for plane in values.chunks_mut(ny * nz) {
                                for z in 0..nz {
                                    for y in 0..ny {
                                        line[y] = plane[y * nz + z];
                                    }
                                    plan.process_with_scratch(line, worker.scratch.as_mut_slice());
                                    for y in 0..ny {
                                        plane[y * nz + z] = line[y];
                                    }
                                }
                            }
                        };
                        if worker_count == 1 {
                            task();
                        } else {
                            mandacaru_kernels::runtime::spawn(scope, task);
                        }
                    }
                });
            } else {
                let chunk = (ny * nz).div_ceil(worker_count);
                let actual = (ny * nz).div_ceil(chunk);
                let mut stripes: Vec<Vec<&mut [C64]>> =
                    (0..actual).map(|_| Vec::with_capacity(nx)).collect();
                for plane in values.chunks_mut(ny * nz) {
                    for (stripe, slice) in stripes.iter_mut().zip(plane.chunks_mut(chunk)) {
                        stripe.push(slice);
                    }
                }
                std::thread::scope(|scope| {
                    for (mut stripe, worker) in stripes.into_iter().zip(&mut self.workers) {
                        let mut task = move || {
                            let line = &mut worker.line.as_mut_slice()[..nx];
                            for column in 0..stripe[0].len() {
                                for x in 0..nx {
                                    line[x] = stripe[x][column];
                                }
                                plan.process_with_scratch(line, worker.scratch.as_mut_slice());
                                for x in 0..nx {
                                    stripe[x][column] = line[x];
                                }
                            }
                        };
                        if worker_count == 1 {
                            task();
                        } else {
                            mandacaru_kernels::runtime::spawn(scope, task);
                        }
                    }
                });
            }
        }
        if inverse {
            let scale = 1. / values.len() as f64;
            for v in values {
                *v *= scale;
            }
        }
        Ok(())
    }
}
pub fn frequency(index: usize, n: usize) -> f64 {
    if index < n.div_ceil(2) {
        index as f64
    } else {
        index as f64 - n as f64
    }
}
pub fn wave_squared(
    shape: [usize; 3],
    step: &Matrix3<f64>,
    index: [usize; 3],
    nyquist_average: bool,
) -> f64 {
    let cell = step * Matrix3::from_diagonal(&Vector3::from_fn(|i, _| shape[i] as f64));
    let reciprocal = cell.try_inverse().expect("validated grid step").transpose() * (2. * PI);
    let q = reciprocal.transpose() * reciprocal;
    let m: Vec<_> = (0..3).map(|i| frequency(index[i], shape[i])).collect();
    let mut value = 0.;
    for a in 0..3 {
        for b in 0..3 {
            if a != b
                && nyquist_average
                && ((shape[a] % 2 == 0 && index[a] == shape[a] / 2)
                    || (shape[b] % 2 == 0 && index[b] == shape[b] / 2))
            {
                continue;
            }
            value += q[(a, b)] * m[a] * m[b];
        }
    }
    value.max(0.)
}
pub(crate) fn fast_len(mut n: usize) -> Result<usize> {
    loop {
        // scipy.next_fast_len's complex transforms permit radices 2,3,5,7,11.
        let mut rest = n;
        for p in [2, 3, 5, 7, 11] {
            while rest % p == 0 {
                rest /= p;
            }
        }
        if rest == 1 {
            return Ok(n);
        }
        n = n.checked_add(1).ok_or(crate::Error::Overflow)?;
    }
}
pub struct Poisson {
    pub shape: [usize; 3],
    pub padded: [usize; 3],
    pub kernel: Buffer<C64>,
    fft: Fft3,
    workspace: Buffer<C64>,
    dv: f64,
}
impl Poisson {
    /// Isolated erfc(omega R)/R. Subtract the smooth long-range kernel from
    /// the isolated Coulomb discretization on the same non-wrapping box.
    pub fn screened(ctx: &Context, grid: &SpatialGrid, omega: f64) -> Result<Self> {
        if !omega.is_finite() || omega < 0. || grid.periodic {
            return Err(invalid(
                "screened molecular Coulomb requires an isolated grid and nonnegative screening",
            ));
        }
        let mut solver = Self::isolated(ctx, grid)?;
        if omega == 0. {
            return Ok(solver);
        }
        let shape = solver.shape;
        let padded = solver.padded;
        let mut long = ctx.allocate::<C64>(product(&padded)?)?;
        for i in 0..padded[0] {
            for j in 0..padded[1] {
                for k in 0..padded[2] {
                    let index = [i, j, k];
                    let delta = Vector3::from_fn(|a, _| {
                        if index[a] < shape[a] {
                            index[a] as f64
                        } else {
                            index[a] as f64 - padded[a] as f64
                        }
                    });
                    let r = (grid.step * delta).norm();
                    long.as_mut_slice()[(i * padded[1] + j) * padded[2] + k] = C64::new(
                        if r > 0. {
                            libm::erf(omega * r) / r
                        } else {
                            2. * omega / PI.sqrt()
                        },
                        0.,
                    );
                }
            }
        }
        solver.fft.transform(long.as_mut_slice(), false)?;
        for (kernel, long) in solver.kernel.as_mut_slice().iter_mut().zip(long.as_slice()) {
            *kernel -= long;
        }
        Ok(solver)
    }
    pub fn slab(ctx: &Context, grid: &SpatialGrid, axis: usize) -> Result<Self> {
        if axis > 2 {
            return Err(invalid("slab axis must be 0, 1 or 2"));
        }
        let shape = grid.index.shape();
        let cell = grid.step * Matrix3::from_diagonal(&Vector3::from_fn(|i, _| shape[i] as f64));
        let axial = cell.column(axis);
        let length = axial.norm();
        for j in 0..3 {
            if j != axis
                && axial.dot(&cell.column(j)).abs() > 1e-10 * length * cell.column(j).norm()
            {
                return Err(invalid(
                    "slab vacuum axis must be perpendicular to the periodic plane",
                ));
            }
        }
        let reciprocal = cell
            .try_inverse()
            .ok_or_else(|| invalid("singular slab cell"))?
            .transpose()
            * (2. * PI);
        let q = reciprocal.transpose() * reciprocal;
        let mut solver = Self::periodic(ctx, grid)?;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let index = [i, j, k];
                    let modes: Vec<_> = (0..3).map(|a| frequency(index[a], shape[a])).collect();
                    let gz2 = q[(axis, axis)] * modes[axis] * modes[axis];
                    let mut gp2 = 0.;
                    for a in 0..3 {
                        for b in 0..3 {
                            if a == axis || b == axis {
                                continue;
                            }
                            if a != b
                                && ((shape[a] % 2 == 0 && index[a] == shape[a] / 2)
                                    || (shape[b] % 2 == 0 && index[b] == shape[b] / 2))
                            {
                                continue;
                            }
                            gp2 += q[(a, b)] * modes[a] * modes[b];
                        }
                    }
                    let gp = gp2.max(0.).sqrt();
                    let gz = gz2.max(0.).sqrt();
                    let phase = gz * length * 0.5;
                    let value = if gp > 0. {
                        4. * PI
                            * (1.
                                + (-gp * length * 0.5).exp()
                                    * (gz / gp * phase.sin() - phase.cos()))
                            / (gp2 + gz2)
                    } else if gz2 > 0. {
                        2. * PI * (2. - 2. * phase.cos() - gz * length * phase.sin()) / gz2
                    } else {
                        0.
                    };
                    solver.kernel.as_mut_slice()[(i * shape[1] + j) * shape[2] + k] =
                        C64::new(value / grid.index.dv(), 0.);
                }
            }
        }
        Ok(solver)
    }
    pub fn isolated(ctx: &Context, grid: &SpatialGrid) -> Result<Self> {
        let shape = grid.index.shape();
        let dv = grid.index.dv();
        let heights: Vec<_> = (0..3)
            .map(|i| {
                dv / grid
                    .step
                    .column((i + 1) % 3)
                    .cross(&grid.step.column((i + 2) % 3))
                    .norm()
            })
            .collect();
        let split = PI / heights.iter().copied().fold(0f64, f64::max) / (4. * 1e10f64.ln()).sqrt();
        let mut padded = [0; 3];
        for i in 0..3 {
            let reach = (4.572 / (split * heights[i])).ceil() as usize;
            padded[i] = fast_len(
                (2 * shape[i] - 1).max(shape[i].checked_add(reach).ok_or(crate::Error::Overflow)?),
            )?;
        }
        let size = product(&padded)?;
        let mut kernel = ctx.allocate::<C64>(size)?;
        for i in 0..padded[0] {
            for j in 0..padded[1] {
                for k in 0..padded[2] {
                    let xyz = [i, j, k];
                    let offsets = Vector3::from_fn(|a, _| {
                        if xyz[a] < shape[a] {
                            xyz[a] as f64
                        } else {
                            xyz[a] as f64 - padded[a] as f64
                        }
                    });
                    let r = (grid.step * offsets).norm();
                    kernel.as_mut_slice()[(i * padded[1] + j) * padded[2] + k] = C64::new(
                        if r > 0. {
                            libm::erf(split * r) / r
                        } else {
                            2. * split / PI.sqrt()
                        },
                        0.,
                    );
                }
            }
        }
        let mut fft = Fft3::new(ctx, padded)?;
        fft.transform(kernel.as_mut_slice(), false)?;
        for i in 0..padded[0] {
            for j in 0..padded[1] {
                for k in 0..padded[2] {
                    let g = wave_squared(padded, &grid.step, [i, j, k], false);
                    let short = if g > 0. {
                        -4. * PI * (-g / (4. * split * split)).exp_m1() / g
                    } else {
                        PI / (split * split)
                    };
                    kernel.as_mut_slice()[(i * padded[1] + j) * padded[2] + k] += short / dv;
                }
            }
        }
        Ok(Self {
            shape,
            padded,
            kernel,
            fft,
            workspace: ctx.allocate(size)?,
            dv,
        })
    }
    pub fn periodic(ctx: &Context, grid: &SpatialGrid) -> Result<Self> {
        let shape = grid.index.shape();
        let size = grid.index.points();
        let dv = grid.index.dv();
        let mut kernel = ctx.allocate::<C64>(size)?;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let g = wave_squared(shape, &grid.step, [i, j, k], true);
                    kernel.as_mut_slice()[(i * shape[1] + j) * shape[2] + k] =
                        C64::new(if g > 0. { 4. * PI / g / dv } else { 0. }, 0.);
                }
            }
        }
        Ok(Self {
            shape,
            padded: shape,
            kernel,
            fft: Fft3::new(ctx, shape)?,
            workspace: ctx.allocate(size)?,
            dv,
        })
    }
    fn prepare_density(&mut self, rho: &[C64]) -> Result<()> {
        if rho.len() != product(&self.shape)?
            || rho.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(invalid("Poisson density must be finite and match the grid"));
        }
        let [nx, ny, nz] = self.shape;
        let [_, ly, lz] = self.padded;
        self.workspace.as_mut_slice().fill(C64::default());
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    self.workspace.as_mut_slice()[(i * ly + j) * lz + k] =
                        rho[(i * ny + j) * nz + k];
                }
            }
        }
        self.fft.transform(self.workspace.as_mut_slice(), false)?;
        Ok(())
    }
    /// `Re integral rho* V[rho]`, using Parseval on the same discrete kernel.
    /// A Coulomb diagonal needs no inverse transform or potential grid.
    pub fn quadratic_form(&mut self, rho: &[C64]) -> Result<f64> {
        self.prepare_density(rho)?;
        let value = self
            .workspace
            .as_slice()
            .iter()
            .zip(self.kernel.as_slice())
            .map(|(rho, kernel)| rho.norm_sqr() * kernel.re)
            .sum::<f64>()
            * self.dv
            * self.dv
            / self.workspace.as_slice().len() as f64;
        if !value.is_finite() {
            return Err(crate::Error::Kernel(
                "Poisson quadratic form overflowed".into(),
            ));
        }
        Ok(value)
    }
    pub fn solve_into(&mut self, rho: &[C64], out: &mut [C64]) -> Result<()> {
        if out.len() != rho.len() {
            return Err(invalid("Poisson output must match its density"));
        }
        self.prepare_density(rho)?;
        let [nx, ny, nz] = self.shape;
        let [_, ly, lz] = self.padded;
        for (value, kernel) in self
            .workspace
            .as_mut_slice()
            .iter_mut()
            .zip(self.kernel.as_slice())
        {
            *value *= kernel * self.dv;
        }
        self.fft.transform(self.workspace.as_mut_slice(), true)?;
        if self
            .workspace
            .as_slice()
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(crate::Error::Kernel(
                "Poisson convolution overflowed".into(),
            ));
        }
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    out[(i * ny + j) * nz + k] = self.workspace.as_slice()[(i * ly + j) * lz + k];
                }
            }
        }
        Ok(())
    }
}
pub fn spectral_laplacian(
    ctx: &Context,
    grid: &SpatialGrid,
    samples: &[f64],
) -> Result<Buffer<f64>> {
    if samples.len() != grid.index.points() {
        return Err(invalid("Laplacian input does not match the grid"));
    }
    let mut values = ctx.allocate::<C64>(samples.len())?;
    for (v, s) in values.as_mut_slice().iter_mut().zip(samples) {
        *v = C64::new(*s, 0.);
    }
    let mut fft = Fft3::new(ctx, grid.index.shape())?;
    fft.transform(values.as_mut_slice(), false)?;
    let shape = grid.index.shape();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                values.as_mut_slice()[(i * shape[1] + j) * shape[2] + k] *=
                    -wave_squared(shape, &grid.step, [i, j, k], true);
            }
        }
    }
    fft.transform(values.as_mut_slice(), true)?;
    let mut out = ctx.allocate(samples.len())?;
    for (v, s) in out.as_mut_slice().iter_mut().zip(values.as_slice()) {
        *v = s.re;
    }
    Ok(out)
}
