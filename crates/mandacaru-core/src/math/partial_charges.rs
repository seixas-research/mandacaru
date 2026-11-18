//! Real-space partial charge partitioning: Bader (Yu-Trinkle near-grid zero-flux),
//! Hirshfeld (stockholder proatomic partitioning), and Voronoi (nearest-atom geometric cells).
//!
//! All methods operate on real-space electron densities sampled on [`SpatialGrid`] (both
//! isolated molecular boxes and arbitrary periodic unit cells) and include any PAW on-site
//! augmentation charge centered on each atom so total electron populations and net charges
//! are conserved identically across DFT, Hartree-Fock, GHF, and variational quantum solvers.
use super::{invalid, lattice::selling_reduce, SpatialGrid};
use crate::{finite, product, Context, Result};
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;

/// Spherically symmetric neutral proatomic radial density table $\rho^0_A(r)$ for Hirshfeld partitioning.
#[derive(Clone, Debug)]
pub struct ProatomicRadial {
    pub r: Vec<f64>,
    pub density: Vec<f64>,
    pub valence_charge: f64,
    pub fallback_scale: f64,
}

impl ProatomicRadial {
    /// Evaluate $\rho^0_A(d)$ at radial distance $d \ge 0$ (Bohr).
    pub fn evaluate(&self, d: f64) -> f64 {
        let d = d.max(0.);
        if self.r.is_empty() || self.density.is_empty() {
            let s = self.fallback_scale.max(0.2);
            return (self.valence_charge / (8. * PI * s * s * s)) * (-d / s).exp();
        }
        let n = self.r.len();
        if d <= self.r[0] {
            return self.density[0].max(0.);
        }
        let last = n - 1;
        if d >= self.r[last] {
            return 0.;
        }
        let idx = self.r.partition_point(|&rk| rk <= d);
        let i0 = idx.saturating_sub(1).min(last - 1);
        let i1 = i0 + 1;
        let span = self.r[i1] - self.r[i0];
        let t = if span > 0. {
            (d - self.r[i0]) / span
        } else {
            0.
        };
        ((1. - t) * self.density[i0] + t * self.density[i1]).max(0.)
    }
}

/// Atomic partial charges and electron populations from real-space density partitioning.
#[derive(Clone, Debug)]
pub struct PartialCharges {
    pub method: String,
    /// Net partial charge $Z_{\text{val}, A} - N_A$ per atom (in elementary charges $e$).
    pub charges: Vec<f64>,
    /// Total electronic population $N_A = N_A^{\text{smooth}} + \Delta Q_A$ per atom.
    pub populations: Vec<f64>,
    /// Sum of atomic electron populations $\sum_A N_A$.
    pub total_electrons: f64,
    /// Total net charge $\sum_A q_A$.
    pub net_charge: f64,
    /// Integrated grid smooth density $\int \tilde{\rho}(\mathbf{r})\,d^3r$.
    pub grid_electrons: f64,
    /// Per-atom PAW on-site augmentation charge $\Delta Q_A$ included in `populations`.
    pub augmentation: Vec<f64>,
    /// Whether the charge density was upsampled by 3D cubic spline interpolation prior to partitioning.
    pub interpolated: bool,
}

const DIST_TIE_TOL: f64 = 1e-12;
const DENSITY_TIE_TOL: f64 = 1e-12;
const PROATOMIC_FLOOR: f64 = 1e-30;

/// Periodic lattice vectors (as columns) and integer image shifts covering Wigner-Seitz Voronoi cells.
struct DistanceKernel {
    periodic: bool,
    inv_cell: Matrix3<f64>,
    cell: Matrix3<f64>,
    shifts: Vec<Vector3<f64>>,
}

impl DistanceKernel {
    fn new(grid: &SpatialGrid) -> Result<Self> {
        if !grid.periodic {
            return Ok(Self {
                periodic: false,
                inv_cell: Matrix3::identity(),
                cell: Matrix3::identity(),
                shifts: vec![Vector3::zeros()],
            });
        }
        let [nx, ny, nz] = grid.index.shape();
        let cell = Matrix3::from_columns(&[
            grid.step.column(0) * nx as f64,
            grid.step.column(1) * ny as f64,
            grid.step.column(2) * nz as f64,
        ]);
        let inv_cell = cell
            .try_inverse()
            .ok_or_else(|| invalid("periodic grid lattice matrix must be nonsingular"))?;
        let (reduced, _) = selling_reduce(&cell)?;
        let reduced_shifts = (-2..=2)
            .flat_map(|i| (-2..=2).flat_map(move |j| (-2..=2).map(move |k| (i, j, k))))
            .map(|(i, j, k)| reduced * Vector3::new(i as f64, j as f64, k as f64))
            .collect::<Vec<_>>();
        Ok(Self {
            periodic: true,
            inv_cell,
            cell,
            shifts: reduced_shifts,
        })
    }

    /// Shortest Cartesian distance from `point` to `center` modulo periodic boundary conditions.
    fn distance(&self, point: Vector3<f64>, center: Vector3<f64>) -> f64 {
        let mut diff = point - center;
        if !self.periodic {
            return diff.norm();
        }
        let f = self.inv_cell * diff;
        let wrapped = Vector3::new(
            f[0] - f[0].round(),
            f[1] - f[1].round(),
            f[2] - f[2].round(),
        );
        diff = self.cell * wrapped;
        self.shifts
            .iter()
            .map(|s| (diff + s).norm_squared())
            .fold(f64::INFINITY, f64::min)
            .sqrt()
    }
}

fn nearest_weights(
    kernel: &DistanceKernel,
    point: Vector3<f64>,
    centers: &[Vector3<f64>],
    out: &mut [(usize, f64)],
) -> usize {
    let mut best = f64::INFINITY;
    let mut count = 0usize;
    for (a, &center) in centers.iter().enumerate() {
        let d = kernel.distance(point, center);
        if d + DIST_TIE_TOL < best {
            best = d;
            out[0] = (a, d);
            count = 1;
        } else if (d - best).abs() <= DIST_TIE_TOL {
            out[count] = (a, d);
            count += 1;
        }
    }
    count
}

fn accumulate_voronoi(
    grid: &SpatialGrid,
    kernel: &DistanceKernel,
    density: &[f64],
    centers: &[Vector3<f64>],
) -> Vec<f64> {
    let n_atoms = centers.len();
    let dv = grid.index.dv();
    let mut pops = vec![0.; n_atoms];
    let mut scratch = vec![(0usize, 0.); n_atoms];
    for (k, &rho) in density.iter().enumerate() {
        let p = Vector3::from(grid.point(k));
        let ties = nearest_weights(kernel, p, centers, &mut scratch);
        let share = rho * dv / ties as f64;
        for &(atom, _) in &scratch[..ties] {
            pops[atom] += share;
        }
    }
    pops
}

fn accumulate_hirshfeld(
    grid: &SpatialGrid,
    kernel: &DistanceKernel,
    density: &[f64],
    centers: &[Vector3<f64>],
    proatomic: &[ProatomicRadial],
) -> Vec<f64> {
    let n_atoms = centers.len();
    let dv = grid.index.dv();
    let mut pops = vec![0.; n_atoms];
    let mut weights = vec![0.; n_atoms];
    let mut scratch = vec![(0usize, 0.); n_atoms];
    for (k, &rho) in density.iter().enumerate() {
        let p = Vector3::from(grid.point(k));
        let mut total = 0.;
        for a in 0..n_atoms {
            let d = kernel.distance(p, centers[a]);
            let w = proatomic[a].evaluate(d);
            weights[a] = w;
            total += w;
        }
        if total > PROATOMIC_FLOOR {
            let scale = rho * dv / total;
            for a in 0..n_atoms {
                pops[a] += weights[a] * scale;
            }
        } else {
            let ties = nearest_weights(kernel, p, centers, &mut scratch);
            let share = rho * dv / ties as f64;
            for &(atom, _) in &scratch[..ties] {
                pops[atom] += share;
            }
        }
    }
    pops
}

/// Convert linear index into `(ix, iy, iz)` coordinates.
#[inline]
fn to_coords(idx: usize, shape: [usize; 3]) -> [isize; 3] {
    let ny = shape[1];
    let nz = shape[2];
    let ix = idx / (ny * nz);
    let rem = idx % (ny * nz);
    let iy = rem / nz;
    let iz = rem % nz;
    [ix as isize, iy as isize, iz as isize]
}

/// Wrap or clamp 3D grid index according to periodicity.
#[inline]
fn resolve_neighbor(
    coords: [isize; 3],
    delta: [isize; 3],
    shape: [usize; 3],
    periodic: bool,
) -> Option<usize> {
    let mut out = [0usize; 3];
    for axis in 0..3 {
        let mut v = coords[axis] + delta[axis];
        let limit = shape[axis] as isize;
        if periodic {
            v = v.rem_euclid(limit);
        } else if v < 0 || v >= limit {
            return None;
        }
        out[axis] = v as usize;
    }
    Some((out[0] * shape[1] + out[1]) * shape[2] + out[2])
}

/// Build tracing density $\rho_{\text{trace}} = \max(\rho, 0) + \sum_A n^{\text{core}}_A$ so
/// PAW smooth densities with hollowed core regions (where valence bond charge exceeds the
/// pseudo-density at the nucleus) still recover nuclear attractors cleanly.
fn bader_tracing_density(
    grid: &SpatialGrid,
    kernel: &DistanceKernel,
    density: &[f64],
    centers: &[Vector3<f64>],
    augmentation: &[f64],
) -> Vec<f64> {
    let n_atoms = centers.len();
    let max_step = (0..3)
        .map(|a| grid.step.column(a).norm())
        .fold(0.0f64, f64::max);
    let mut min_nn = f64::INFINITY;
    for i in 0..n_atoms {
        for j in (i + 1)..n_atoms {
            min_nn = min_nn.min(kernel.distance(centers[i], centers[j]));
        }
    }
    if !min_nn.is_finite() {
        min_nn = 6.0;
    }
    let core_probe = (1.15 * max_step).min(0.42 * min_nn).max(0.25);
    let ring_probe = (0.48 * min_nn).max(core_probe);
    let mut center_max = vec![0.0f64; n_atoms];
    let mut ring_max = vec![0.0f64; n_atoms];
    let mut global_max = 0.0f64;
    for (k, &v) in density.iter().enumerate() {
        let rho = v.max(0.);
        if rho > global_max {
            global_max = rho;
        }
        let p = Vector3::from(grid.point(k));
        for a in 0..n_atoms {
            let d = kernel.distance(p, centers[a]);
            if d <= core_probe && rho > center_max[a] {
                center_max[a] = rho;
            }
            if d <= ring_probe && rho > ring_max[a] {
                ring_max[a] = rho;
            }
        }
    }

    let sigma = (0.35 * max_step.max(0.4)).min(0.22 * min_nn).max(0.18);
    let gauss_cutoff = 5.0 * sigma;
    let r_exp = (0.15 * min_nn).clamp(0.32, 0.65);
    let exp_cutoff = (0.65 * min_nn).max(3.0 * r_exp);
    let hollow: Vec<bool> = (0..n_atoms)
        .map(|a| center_max[a] < 0.9 * ring_max[a])
        .collect();

    let mut trace = density.iter().map(|&v| v.max(0.)).collect::<Vec<_>>();
    for (k, t) in trace.iter_mut().enumerate() {
        let p = Vector3::from(grid.point(k));
        for a in 0..n_atoms {
            let d = kernel.distance(p, centers[a]);
            if hollow[a] {
                if d < exp_cutoff {
                    *t += (20.0 * global_max.max(0.05)) * (-d / r_exp).exp();
                }
            } else if d < gauss_cutoff {
                let amp = if augmentation[a].abs() > 0.05 {
                    0.5 * global_max.max(0.1)
                } else {
                    0.02
                };
                *t += amp * (-0.5 * (d / sigma).powi(2)).exp();
            }
        }
    }
    trace
}

enum BaderStep {
    Unique(usize),
    Tied(Vec<usize>),
    Maximum,
}

/// Choose next neighbor(s) by metric-weighted steepest ascent (`(trace[n] - trace[curr]) / |step|`),
/// keeping all symmetry-tied maxima so zero-flux planes split deterministically and order-independently.
fn bader_step(
    idx: usize,
    shape: [usize; 3],
    periodic: bool,
    step_lengths: &[f64; 27],
    trace: &[f64],
) -> BaderStep {
    let coords = to_coords(idx, shape);
    let center_val = trace[idx];
    let tie_eps = DENSITY_TIE_TOL * center_val.abs().max(1e-12);
    let mut slopes: Vec<(f64, usize)> = Vec::with_capacity(26);
    let mut best_slope = 0.;
    for dx in -1isize..=1 {
        for dy in -1isize..=1 {
            for dz in -1isize..=1 {
                if (dx, dy, dz) == (0, 0, 0) {
                    continue;
                }
                let Some(next_idx) = resolve_neighbor(coords, [dx, dy, dz], shape, periodic) else {
                    continue;
                };
                let dv = trace[next_idx] - center_val;
                if dv > tie_eps {
                    let s_id = ((dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)) as usize;
                    let slope = dv / step_lengths[s_id];
                    if slope > best_slope {
                        best_slope = slope;
                    }
                    slopes.push((slope, next_idx));
                }
            }
        }
    }
    if slopes.is_empty() {
        return BaderStep::Maximum;
    }
    let cutoff = best_slope * (1. - 1e-8);
    let tied: Vec<usize> = slopes
        .into_iter()
        .filter(|&(slope, _)| slope >= cutoff)
        .map(|(_, next_idx)| next_idx)
        .collect();
    if tied.len() == 1 {
        BaderStep::Unique(tied[0])
    } else {
        BaderStep::Tied(tied)
    }
}

fn add_voxel_weights(
    idx: usize,
    weight: f64,
    basin: &[usize],
    tied: &[Option<Vec<usize>>],
    grid: &SpatialGrid,
    kernel: &DistanceKernel,
    centers: &[Vector3<f64>],
    scratch: &mut [(usize, f64)],
    pops: &mut [f64],
    visited: &mut Vec<usize>,
) {
    let target = basin[idx];
    if target == usize::MAX - 1 {
        if visited.contains(&idx) {
            let max_point = Vector3::from(grid.point(idx));
            let ties = nearest_weights(kernel, max_point, centers, scratch);
            let share = weight / ties as f64;
            for &(atom, _) in &scratch[..ties] {
                pops[atom] += share;
            }
            return;
        }
        visited.push(idx);
        if let Some(branches) = &tied[idx] {
            let sub_w = weight / branches.len() as f64;
            for &b in branches {
                add_voxel_weights(
                    b, sub_w, basin, tied, grid, kernel, centers, scratch, pops, visited,
                );
            }
        }
        visited.pop();
        return;
    }
    let max_point = Vector3::from(grid.point(target));
    let ties = nearest_weights(kernel, max_point, centers, scratch);
    let share = weight / ties as f64;
    for &(atom, _) in &scratch[..ties] {
        pops[atom] += share;
    }
}

fn accumulate_bader(
    ctx: &Context,
    grid: &SpatialGrid,
    kernel: &DistanceKernel,
    density: &[f64],
    centers: &[Vector3<f64>],
    augmentation: &[f64],
) -> Result<Vec<f64>> {
    const TIED_MARKER: usize = usize::MAX - 1;
    let shape = grid.index.shape();
    let n_pts = density.len();
    let n_atoms = centers.len();
    let _work = ctx.reserve(product(&[n_pts, 32])?)?;
    let trace = bader_tracing_density(grid, kernel, density, centers, augmentation);
    let mut step_lengths = [1.; 27];
    for dx in -1isize..=1 {
        for dy in -1isize..=1 {
            for dz in -1isize..=1 {
                if (dx, dy, dz) != (0, 0, 0) {
                    let v = grid.step * Vector3::new(dx as f64, dy as f64, dz as f64);
                    let id = ((dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)) as usize;
                    step_lengths[id] = v.norm().max(1e-12);
                }
            }
        }
    }
    // `basin[idx]` holds resolved maximum voxel index (`usize::MAX` = unassigned, `TIED_MARKER` = tied split).
    let mut basin = vec![usize::MAX; n_pts];
    let mut tied: Vec<Option<Vec<usize>>> = vec![None; n_pts];
    let mut path = Vec::with_capacity(64);
    for start in 0..n_pts {
        if basin[start] != usize::MAX {
            continue;
        }
        path.clear();
        let mut curr = start;
        let root = loop {
            if basin[curr] != usize::MAX {
                break basin[curr];
            }
            path.push(curr);
            match bader_step(curr, shape, grid.periodic, &step_lengths, &trace) {
                BaderStep::Unique(next) if !path.contains(&next) => curr = next,
                BaderStep::Tied(branches) => {
                    tied[curr] = Some(branches);
                    basin[curr] = TIED_MARKER;
                    break TIED_MARKER;
                }
                _ => break curr,
            }
        };
        for &node in &path {
            if basin[node] == usize::MAX {
                basin[node] = root;
                if root == TIED_MARKER && tied[node].is_none() {
                    // Forward to `curr` (already resolved to `TIED_MARKER`).
                    if node != curr {
                        tied[node] = Some(vec![curr]);
                    }
                }
            }
        }
    }
    // Map each resolved basin (or symmetrically split saddle branch) to atom populations.
    let dv = grid.index.dv();
    let mut pops = vec![0.; n_atoms];
    let mut scratch = vec![(0usize, 0.); n_atoms];
    let mut visited = Vec::with_capacity(16);
    for (k, &rho) in density.iter().enumerate() {
        visited.clear();
        add_voxel_weights(
            k,
            rho * dv,
            &basin,
            &tied,
            grid,
            kernel,
            centers,
            &mut scratch,
            &mut pops,
            &mut visited,
        );
    }
    Ok(pops)
}

struct SplineScratch {
    rhs: Vec<f64>,
    upper: Vec<f64>,
    z: Vec<f64>,
    q: Vec<f64>,
    second: Vec<f64>,
}

impl SplineScratch {
    fn ensure(&mut self, n: usize) {
        if self.rhs.len() < n {
            self.rhs.resize(n, 0.);
            self.upper.resize(n, 0.);
            self.z.resize(n, 0.);
            self.q.resize(n, 0.);
            self.second.resize(n, 0.);
        }
    }
}

/// Natural ($M_0 = M_{n-1} = 0$) cubic spline second derivatives on unit-spaced knots.
fn natural_cubic_second_derivatives(y: &[f64], s: &mut SplineScratch) {
    let n = y.len();
    s.ensure(n);
    s.second[0] = 0.;
    s.second[n - 1] = 0.;
    if n < 4 {
        for i in 1..n - 1 {
            s.second[i] = 0.;
        }
        return;
    }
    // Interior system m_{i-1} + 4 m_i + m_{i+1} = 6 (y_{i-1} - 2 y_i + y_{i+1}) for i = 1..n-1.
    for i in 1..n - 1 {
        s.rhs[i] = 6. * (y[i - 1] - 2. * y[i] + y[i + 1]);
    }
    s.upper[1] = 0.25;
    s.z[1] = 0.25 * s.rhs[1];
    for i in 2..n - 1 {
        let inv = 1. / (4. - s.upper[i - 1]);
        s.upper[i] = inv;
        s.z[i] = (s.rhs[i] - s.z[i - 1]) * inv;
    }
    s.second[n - 2] = s.z[n - 2];
    for i in (1..n - 2).rev() {
        s.second[i] = s.z[i] - s.upper[i] * s.second[i + 1];
    }
}

/// Cyclic tridiagonal cubic spline second derivatives on unit-spaced periodic knots ($n \ge 3$).
fn periodic_cubic_second_derivatives(y: &[f64], s: &mut SplineScratch) {
    let n = y.len();
    s.ensure(n);
    for i in 0..n {
        let prev = if i == 0 { n - 1 } else { i - 1 };
        let next = if i + 1 == n { 0 } else { i + 1 };
        s.rhs[i] = 6. * (y[prev] - 2. * y[i] + y[next]);
    }
    // Solve A m = rhs where A = B + u v^T with gamma = -2:
    // u = [-2, 0, ..., 0, 1]^T, v = [1, 0, ..., 0, -0.5]^T => B_diag = [6, 4, ..., 4, 4.5].
    let gamma = -2.0;
    s.upper[0] = 1. / (4. - gamma);
    s.z[0] = s.rhs[0] / (4. - gamma);
    s.q[0] = gamma / (4. - gamma);
    for i in 1..n - 1 {
        let inv = 1. / (4. - s.upper[i - 1]);
        s.upper[i] = inv;
        s.z[i] = (s.rhs[i] - s.z[i - 1]) * inv;
        let u_i = 0.0;
        s.q[i] = (u_i - s.q[i - 1]) * inv;
    }
    let last_diag = 4. - 1. / gamma;
    let inv_last = 1. / (last_diag - s.upper[n - 2]);
    s.z[n - 1] = (s.rhs[n - 1] - s.z[n - 2]) * inv_last;
    s.q[n - 1] = (1. - s.q[n - 2]) * inv_last;
    for i in (0..n - 1).rev() {
        s.z[i] -= s.upper[i] * s.z[i + 1];
        s.q[i] -= s.upper[i] * s.q[i + 1];
    }
    let factor = (s.z[0] + s.z[n - 1] / gamma) / (1. + s.q[0] + s.q[n - 1] / gamma);
    for i in 0..n {
        s.second[i] = s.z[i] - factor * s.q[i];
    }
}

fn upsample_line_and_scatter(
    line: &[f64],
    periodic: bool,
    scratch: &mut SplineScratch,
    out_stride: usize,
    out_offset: usize,
    out: &mut [f64],
) {
    let n = line.len();
    if n <= 1 {
        out[out_offset] = line[0];
        return;
    }
    let use_cubic = if periodic { n >= 3 } else { n >= 4 };
    if use_cubic {
        if periodic {
            periodic_cubic_second_derivatives(line, scratch);
        } else {
            natural_cubic_second_derivatives(line, scratch);
        }
    }
    let intervals = if periodic { n } else { n - 1 };
    for k in 0..intervals {
        let next = if k + 1 == n { 0 } else { k + 1 };
        out[out_offset + (2 * k) * out_stride] = line[k];
        let mid = 0.5 * (line[k] + line[next])
            - if use_cubic {
                (scratch.second[k] + scratch.second[next]) / 16.
            } else {
                0.
            };
        out[out_offset + (2 * k + 1) * out_stride] = mid;
    }
    if !periodic {
        out[out_offset + (2 * (n - 1)) * out_stride] = line[n - 1];
    }
}

fn upsample_last_axis_and_rotate(
    src: &[f64],
    shape: [usize; 3],
    periodic: bool,
    scratch: &mut SplineScratch,
) -> (Vec<f64>, [usize; 3]) {
    let [d0, d1, d2] = shape;
    let f2 = if d2 <= 1 {
        d2
    } else if periodic {
        2 * d2
    } else {
        2 * d2 - 1
    };
    let mut dst = vec![0.; d0 * d1 * f2];
    // Upsample contiguous lines along axis 2, scattering directly to rotated shape [f2, d0, d1].
    let stride_rotated = d0 * d1;
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            let src_start = (i0 * d1 + i1) * d2;
            let dst_offset = i0 * d1 + i1;
            upsample_line_and_scatter(
                &src[src_start..src_start + d2],
                periodic,
                scratch,
                stride_rotated,
                dst_offset,
                &mut dst,
            );
        }
    }
    (dst, [f2, d0, d1])
}

/// Upsample a 3D electron density by factor 2 along each grid axis using separable C^2 cubic splines.
fn refine_density_cubic_spline(
    ctx: &Context,
    grid: &SpatialGrid,
    density: &[f64],
) -> Result<(SpatialGrid, Vec<f64>)> {
    let orig_shape = grid.index.shape();
    let fine_shape = orig_shape.map(|n| {
        if n <= 1 {
            n
        } else if grid.periodic {
            2 * n
        } else {
            2 * n - 1
        }
    });
    let fine_pts = product(&fine_shape)?;
    let _budget = ctx.reserve(
        fine_pts
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(crate::Error::Overflow)?,
    )?;
    let mut scratch = SplineScratch {
        rhs: Vec::new(),
        upper: Vec::new(),
        z: Vec::new(),
        q: Vec::new(),
        second: Vec::new(),
    };
    let (p1, s1) = upsample_last_axis_and_rotate(density, orig_shape, grid.periodic, &mut scratch);
    let (p2, s2) = upsample_last_axis_and_rotate(&p1, s1, grid.periodic, &mut scratch);
    drop(p1);
    let (mut fine_rho, final_shape) =
        upsample_last_axis_and_rotate(&p2, s2, grid.periodic, &mut scratch);
    drop(p2);
    debug_assert_eq!(final_shape, fine_shape);

    let mut step_arr = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            let factor = if orig_shape[c] > 1 { 0.5 } else { 1.0 };
            step_arr[3 * r + c] = factor * grid.step[(r, c)];
        }
    }
    let fine_grid = SpatialGrid::new(final_shape, grid.origin, step_arr, grid.periodic)?;

    // Preserve exact grid-integrated smooth electron count after finite-domain spline interpolation.
    if !grid.periodic {
        let q_coarse: f64 = density.iter().sum::<f64>() * grid.index.dv();
        let q_fine: f64 = fine_rho.iter().sum::<f64>() * fine_grid.index.dv();
        if q_fine.abs() > 1e-12 && (q_fine - q_coarse).abs() > 1e-14 {
            let scale = q_coarse / q_fine;
            for v in &mut fine_rho {
                *v *= scale;
            }
        }
    }
    Ok((fine_grid, fine_rho))
}

/// Partition real-space electron density `density` on `grid` among atoms at `positions`
/// using `"bader"`, `"hirshfeld"`, or `"voronoi"`, adding per-atom PAW `augmentation`.
/// When `interpolate` is `true`, refines the grid density via separable 3D cubic splines first.
#[allow(clippy::too_many_arguments)]
pub fn compute_partial_charges(
    ctx: &Context,
    method: &str,
    grid: &SpatialGrid,
    density: &[f64],
    positions: &[[f64; 3]],
    valence_charges: &[f64],
    augmentation: &[f64],
    proatomic: Option<&[ProatomicRadial]>,
    interpolate: bool,
) -> Result<PartialCharges> {
    let method_lc = method.to_ascii_lowercase();
    if !matches!(method_lc.as_str(), "bader" | "hirshfeld" | "voronoi") {
        return Err(invalid(format!(
            "partial charge method must be 'bader', 'hirshfeld', or 'voronoi', got {method:?}"
        )));
    }
    let n_pts = grid.index.points();
    if density.len() != n_pts {
        return Err(invalid(format!(
            "density length ({}) must match grid points ({n_pts})",
            density.len()
        )));
    }
    let n_atoms = positions.len();
    if n_atoms == 0 || valence_charges.len() != n_atoms || augmentation.len() != n_atoms {
        return Err(invalid(
            "positions, valence_charges, and augmentation must be nonempty and match atom count",
        ));
    }
    finite(density, "electron density")?;
    finite(valence_charges, "valence charges")?;
    finite(augmentation, "PAW augmentation charges")?;
    for p in positions {
        finite(p, "atom position")?;
    }
    if let Some(pro) = proatomic {
        if pro.len() != n_atoms {
            return Err(invalid("proatomic densities must match atom count"));
        }
    }
    let refined = if interpolate {
        Some(refine_density_cubic_spline(ctx, grid, density)?)
    } else {
        None
    };
    let (work_grid, work_density): (&SpatialGrid, &[f64]) = match &refined {
        Some((fg, fd)) => (fg, fd.as_slice()),
        None => (grid, density),
    };
    let centers: Vec<Vector3<f64>> = positions.iter().map(|&p| Vector3::from(p)).collect();
    let kernel = DistanceKernel::new(work_grid)?;
    let default_pro;
    let smooth_pops = match method_lc.as_str() {
        "voronoi" => accumulate_voronoi(work_grid, &kernel, work_density, &centers),
        "hirshfeld" => {
            let pro = match proatomic {
                Some(p) => p,
                None => {
                    default_pro = valence_charges
                        .iter()
                        .map(|&q| ProatomicRadial {
                            r: Vec::new(),
                            density: Vec::new(),
                            valence_charge: q,
                            fallback_scale: 0.75,
                        })
                        .collect::<Vec<_>>();
                    &default_pro
                }
            };
            accumulate_hirshfeld(work_grid, &kernel, work_density, &centers, pro)
        }
        _ => accumulate_bader(
            ctx,
            work_grid,
            &kernel,
            work_density,
            &centers,
            augmentation,
        )?,
    };
    let populations: Vec<f64> = smooth_pops
        .into_iter()
        .zip(augmentation)
        .map(|(s, &aug)| s + aug)
        .collect();
    let charges: Vec<f64> = valence_charges
        .iter()
        .zip(&populations)
        .map(|(&z, &pop)| z - pop)
        .collect();
    let grid_electrons = work_density.iter().sum::<f64>() * work_grid.index.dv();
    let total_electrons = populations.iter().sum();
    let net_charge = charges.iter().sum();
    Ok(PartialCharges {
        method: method_lc,
        charges,
        populations,
        total_electrons,
        net_charge,
        grid_electrons,
        augmentation: augmentation.to_vec(),
        interpolated: interpolate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_and_natural_cubic_splines_reproduce_polynomials_and_conserve_periodic_integral() {
        let y_periodic: Vec<f64> = (0..16)
            .map(|k| {
                let x = 2.0 * PI * (k as f64) / 16.0;
                2.5 + 1.2 * x.cos() - 0.6 * (2.0 * x).sin()
            })
            .collect();
        let mut s = SplineScratch {
            rhs: Vec::new(),
            upper: Vec::new(),
            z: Vec::new(),
            q: Vec::new(),
            second: Vec::new(),
        };
        periodic_cubic_second_derivatives(&y_periodic, &mut s);
        for i in 0..16 {
            let prev = if i == 0 { 15 } else { i - 1 };
            let next = (i + 1) % 16;
            let residual = s.second[prev] + 4.0 * s.second[i] + s.second[next] - s.rhs[i];
            assert!(residual.abs() < 1e-13, "i={i}: residual={residual}");
        }
        let mut out = vec![0.0; 32];
        upsample_line_and_scatter(&y_periodic, true, &mut s, 1, 0, &mut out);
        assert!((out.iter().sum::<f64>() * 0.5 - y_periodic.iter().sum::<f64>()).abs() < 1e-12);

        // Linear ramp x -> natural cubic spline reproduces all midpoints identically.
        let y_lin: Vec<f64> = (0..10).map(|k| 1.5 + 0.7 * (k as f64)).collect();
        let mut out_lin = vec![0.0; 19];
        upsample_line_and_scatter(&y_lin, false, &mut s, 1, 0, &mut out_lin);
        for k in 0..19 {
            let expected = 1.5 + 0.7 * (0.5 * k as f64);
            assert!((out_lin[k] - expected).abs() < 1e-13);
        }
    }
}
