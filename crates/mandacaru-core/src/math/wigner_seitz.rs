//! Wigner-Seitz minimum-image Coulomb kernel on a periodic grid, the voxel
//! self-potential, and the Fourier transform of a real kernel.
//!
//! The grid is the period. Every index offset is one lattice-equivalence class
//! and has exactly one representative inside the Wigner-Seitz cell, its minimum
//! image, so the kernel is `1/|d_min|` at every node and nothing is zeroed.
//! The `d = 0` node carries the voxel average of `1/r`.
//!
//! The sampled kernel integrates `1/r` with an error proportional to `h^2`
//! (a contact term, the same at every wave vector far below the Nyquist one).
//! A caller that needs a quantity to a stated tolerance extrapolates in `h`;
//! the raw value at a modest resolution can fail a 1 % gate. This module owns
//! its transform and does not depend on `Poisson`.
use super::{fft::Fft3, invalid, SpatialGrid};
use crate::{product, Buffer, Context, Result};
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64 as C64;

/// `int_cube d^3r / r = CUBE_CONSTANT * L^2 = (3 ln(2 + sqrt 3) - pi/2) L^2`.
pub const CUBE_CONSTANT: f64 = 2.380_077_363_979_553;
/// Lattice shells searched for the minimum image.
pub const MINIMUM_IMAGE_SHELLS: usize = 2;
/// Most shells accepted: `(2 s + 1)^3 - 1` translations are tried at every node.
pub const MAX_SHELLS: usize = 4;
/// Relative bound on `|K(d) - K(-d)|` for a kernel to have a real transform.
pub const EVEN_TOLERANCE: f64 = 1e-12;

fn triangle_potential(a: Vector3<f64>, b: Vector3<f64>, c: Vector3<f64>) -> f64 {
    // int_T dS/|r| over a planar triangle, field point at the origin: one
    // logarithm per edge and an arctangent pair for the out-of-plane offset.
    let normal = (b - a).cross(&(c - a));
    let norm = normal.norm();
    if norm == 0. {
        return 0.;
    }
    let normal = normal / norm;
    let offset = normal.dot(&a);
    let projected = normal * offset;
    let mut total = 0.;
    for (start, end) in [(a, b), (b, c), (c, a)] {
        let edge = end - start;
        let length = edge.norm();
        if length == 0. {
            continue;
        }
        let along = edge / length;
        let perpendicular = along.cross(&normal).dot(&(start - projected));
        let l_minus = (start - projected).dot(&along);
        let l_plus = (end - projected).dot(&along);
        let (r_minus, r_plus) = (start.norm(), end.norm());
        let r0_squared = perpendicular * perpendicular + offset * offset;
        if perpendicular != 0. && r_plus + l_plus > 0. && r_minus + l_minus > 0. {
            total += perpendicular * ((r_plus + l_plus) / (r_minus + l_minus)).ln();
        }
        total -= offset.abs()
            * ((perpendicular * l_plus).atan2(r0_squared + offset.abs() * r_plus)
                - (perpendicular * l_minus).atan2(r0_squared + offset.abs() * r_minus));
    }
    total
}

fn box_self_potential(dx: f64, dy: f64, dz: f64) -> f64 {
    let (a, b, c) = (0.5 * dx, 0.5 * dy, 0.5 * dz);
    let s = (a * a + b * b + c * c).sqrt();
    8. * (a * b * ((c + s) / a.hypot(b)).ln()
        + b * c * ((a + s) / b.hypot(c)).ln()
        + c * a * ((b + s) / c.hypot(a)).ln()
        - 0.5 * a * a * (b * c / (a * s)).atan()
        - 0.5 * b * b * (c * a / (b * s)).atan()
        - 0.5 * c * c * (a * b / (c * s)).atan())
}

/// `int_voxel d^3r / |r|` over the voxel of `step` (step vectors as columns)
/// centered on the field point, in Bohr^2. Since `laplacian r = 2/r`, the
/// divergence theorem turns it into `1/2 sum_f h_f int_f dS/|r|` over the six
/// faces, each integrated in closed form on two triangles. An orthogonal voxel
/// takes the rectangular closed form; the two agree to rounding.
pub fn voxel_self_potential(step: &Matrix3<f64>) -> Result<f64> {
    if step.iter().any(|v| !v.is_finite()) {
        return Err(invalid("voxel step must be finite"));
    }
    if step.determinant().abs() <= 0. {
        return Err(invalid("the voxel has zero volume (degenerate step)"));
    }
    let diagonal = Vector3::from_fn(|i, _| step[(i, i)].abs());
    let scale = diagonal.max().max(1.);
    let rectangular = (0..3).all(|i| (0..3).all(|j| i == j || step[(i, j)].abs() <= 1e-12 * scale));
    if rectangular {
        return Ok(box_self_potential(diagonal[0], diagonal[1], diagonal[2]));
    }
    let corner = |s: [usize; 3]| step * Vector3::from_fn(|i, _| s[i] as f64 - 0.5);
    let mut total = 0.;
    for axis in 0..3 {
        let (o0, o1) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        for side in 0..2 {
            // The four corners of the face, in cyclic order.
            let v: Vec<_> = [(0, 0), (0, 1), (1, 1), (1, 0)]
                .iter()
                .map(|&(p, q)| {
                    let mut s = [0; 3];
                    s[axis] = side;
                    s[o0] = p;
                    s[o1] = q;
                    corner(s)
                })
                .collect();
            let normal = (v[1] - v[0]).cross(&(v[2] - v[0])).normalize();
            total += normal.dot(&v[0]).abs()
                * (triangle_potential(v[0], v[1], v[2]) + triangle_potential(v[0], v[2], v[3]));
        }
    }
    Ok(0.5 * total)
}

/// Signed FFT-order offset of index `i` on `n` nodes (`0..=n/2`, then negative).
fn signed(i: usize, n: usize) -> f64 {
    if i <= n / 2 {
        i as f64
    } else {
        i as f64 - n as f64
    }
}

/// The bare `1/r` at its minimum image, at every node of a periodic grid, in
/// FFT order (row-major, last index fastest). The `d = 0` node is the voxel
/// average. Memory: `8 N` bytes for the kernel plus a table of
/// `(2 shells + 1)^3 - 1` translations, both reserved from the context.
pub fn minimum_image_kernel(
    ctx: &Context,
    grid: &SpatialGrid,
    shells: usize,
) -> Result<Buffer<f64>> {
    if !grid.periodic {
        return Err(invalid(
            "the Wigner-Seitz kernel needs a periodic grid; an isolated grid has no minimum image",
        ));
    }
    if !(1..=MAX_SHELLS).contains(&shells) {
        return Err(invalid("minimum-image shells must be in 1..=4"));
    }
    let shape = grid.index.shape();
    let [nx, ny, nz] = shape;
    let step = grid.step;
    let cell = step * Matrix3::from_diagonal(&Vector3::new(nx as f64, ny as f64, nz as f64));
    let reach = 2 * shells as i64 + 1;
    let mut table = ctx.allocate::<[f64; 3]>(product(&[reach as usize; 3])? - 1)?;
    let span = -(shells as i64)..=shells as i64;
    let mut count = 0;
    for i in span.clone() {
        for j in span.clone() {
            for k in span.clone() {
                if (i, j, k) != (0, 0, 0) {
                    let v = cell * Vector3::new(i as f64, j as f64, k as f64);
                    table.as_mut_slice()[count] = [v[0], v[1], v[2]];
                    count += 1;
                }
            }
        }
    }
    let mut kernel = ctx.allocate::<f64>(grid.index.points())?;
    let shifts = table.as_slice();
    let selfpot = voxel_self_potential(&step)? / step.determinant().abs();
    let workers = ctx.threads().min(nx);
    let planes = nx.div_ceil(workers);
    ctx.run(|| {
        std::thread::scope(|scope| {
            for (c, block) in kernel
                .as_mut_slice()
                .chunks_mut(planes * ny * nz)
                .enumerate()
            {
                let mut task = move || {
                    for (p, plane) in block.chunks_mut(ny * nz).enumerate() {
                        let sx = signed(c * planes + p, nx);
                        for (q, line) in plane.chunks_mut(nz).enumerate() {
                            let sy = signed(q, ny);
                            for (r, out) in line.iter_mut().enumerate() {
                                let b = step * Vector3::new(sx, sy, signed(r, nz));
                                let mut best = b.norm_squared();
                                for s in shifts {
                                    let (x, y, z) = (b[0] + s[0], b[1] + s[1], b[2] + s[2]);
                                    best = best.min(x * x + y * y + z * z);
                                }
                                *out = if best > 0. { 1. / best.sqrt() } else { 0. };
                            }
                        }
                    }
                };
                if workers == 1 {
                    task();
                } else {
                    mandacaru_kernels::runtime::spawn(scope, task);
                }
            }
        })
    });
    kernel.as_mut_slice()[0] = selfpot;
    Ok(kernel)
}

/// `max |K(d) - K(-d)|` over the grid (absolute).
pub fn asymmetry(kernel: &[f64], shape: [usize; 3]) -> Result<f64> {
    let [nx, ny, nz] = shape;
    if kernel.len() != product(&shape)? {
        return Err(invalid("kernel does not match its shape"));
    }
    let mut worst = 0_f64;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let a = kernel[(i * ny + j) * nz + k];
                let b = kernel[(((nx - i) % nx) * ny + (ny - j) % ny) * nz + (nz - k) % nz];
                worst = worst.max((a - b).abs());
            }
        }
    }
    Ok(worst)
}

/// `max 1/K` over the nodes with a positive kernel: the radius of the
/// Wigner-Seitz cell's farthest corner as sampled.
pub fn minimum_image_radius(kernel: &[f64]) -> f64 {
    1. / kernel
        .iter()
        .copied()
        .filter(|&k| k > 0.)
        .fold(f64::INFINITY, f64::min)
}

/// Transform of an even real kernel and the circular convolution with it.
///
/// `spectrum()[m] = dV * sum_d K(d) exp(-2 pi i m.d / n)` is the continuous
/// transform of the kernel at the reciprocal-lattice node `m` (units of `K`
/// times Bohr^3), so `spectrum()[0]` is the kernel's integral over the cell.
/// Resident memory is `8 N` bytes of spectrum plus `16 N` of workspace, all
/// reserved from the context; building from a kernel holds that kernel too.
pub struct RealKernel {
    shape: [usize; 3],
    dv: f64,
    spectrum: Buffer<f64>,
    work: Buffer<C64>,
    fft: Fft3,
}
impl RealKernel {
    /// `kernel` is in FFT order. It must be finite and even (to
    /// [`EVEN_TOLERANCE`], relative), which is what makes the transform real.
    pub fn new(ctx: &Context, grid: &SpatialGrid, kernel: &[f64]) -> Result<Self> {
        if !grid.periodic {
            return Err(invalid("a kernel transform needs a periodic grid"));
        }
        let shape = grid.index.shape();
        let n = grid.index.points();
        if kernel.len() != n || kernel.iter().any(|v| !v.is_finite()) {
            return Err(invalid("kernel must be finite and match the grid"));
        }
        let [nx, ny, nz] = shape;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let a = kernel[(i * ny + j) * nz + k];
                    let b = kernel[(((nx - i) % nx) * ny + (ny - j) % ny) * nz + (nz - k) % nz];
                    if (a - b).abs() > EVEN_TOLERANCE * a.abs().max(b.abs()) {
                        return Err(invalid(
                            "kernel must be even, K(d) = K(-d), for a real transform",
                        ));
                    }
                }
            }
        }
        let mut fft = Fft3::new(ctx, shape)?;
        let mut work = ctx.allocate::<C64>(n)?;
        let mut spectrum = ctx.allocate::<f64>(n)?;
        for (w, &k) in work.as_mut_slice().iter_mut().zip(kernel) {
            *w = C64::new(k, 0.);
        }
        fft.transform(work.as_mut_slice(), false)?;
        let dv = grid.index.dv();
        for (s, w) in spectrum.as_mut_slice().iter_mut().zip(work.as_slice()) {
            *s = w.re * dv;
        }
        Ok(Self {
            shape,
            dv,
            spectrum,
            work,
            fft,
        })
    }
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }
    pub fn spectrum(&self) -> &[f64] {
        self.spectrum.as_slice()
    }
    /// `out = (K * rho)(r)`, the circular convolution with the volume element.
    pub fn convolve_into(&mut self, rho: &[C64], out: &mut [C64]) -> Result<()> {
        if out.len() != rho.len() {
            return Err(invalid("convolution output must match its density"));
        }
        self.load(rho)?;
        self.fft.transform(self.work.as_mut_slice(), true)?;
        out.copy_from_slice(self.work.as_slice());
        Ok(())
    }
    /// `Re int rho* (K * rho) dV`, from the spectrum alone (no inverse transform).
    pub fn quadratic_form(&mut self, rho: &[C64]) -> Result<f64> {
        self.load_forward(rho)?;
        let n = self.work.as_slice().len() as f64;
        Ok(self.dv / n
            * self
                .work
                .as_slice()
                .iter()
                .zip(self.spectrum.as_slice())
                .map(|(r, s)| s * r.norm_sqr())
                .sum::<f64>())
    }
    fn load_forward(&mut self, rho: &[C64]) -> Result<()> {
        if rho.len() != self.work.as_slice().len()
            || rho.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(invalid("density must be finite and match the grid"));
        }
        self.work.as_mut_slice().copy_from_slice(rho);
        self.fft.transform(self.work.as_mut_slice(), false)
    }
    fn load(&mut self, rho: &[C64]) -> Result<()> {
        self.load_forward(rho)?;
        for (w, s) in self
            .work
            .as_mut_slice()
            .iter_mut()
            .zip(self.spectrum.as_slice())
        {
            *w *= *s;
        }
        Ok(())
    }
}

/// The Wigner-Seitz kernel of `grid` with its transform. Peak memory while
/// building is `8 N` (kernel) `+ 8 N` (spectrum) `+ 16 N` (workspace) bytes.
pub fn wigner_seitz(ctx: &Context, grid: &SpatialGrid, shells: usize) -> Result<RealKernel> {
    let kernel = minimum_image_kernel(ctx, grid, shells)?;
    RealKernel::new(ctx, grid, kernel.as_slice())
}
