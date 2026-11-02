//! Strain tangents of the Bloch sums on the grid: `A_ab(r_i) = d chi(r_i) / d eps_ab`.
//!
//! Under a strain `x -> (1 + eps) x` of the cell the grid nodes and the image centers move with
//! it, the radial tables, the harmonics and the Fourier filter do not, and the fractional k-point
//! is held, so the Bloch phase `e^{i k.T}` is invariant. A function `phi(y)` of the argument
//! `y = r_i - c - T` therefore changes by `(d_a phi)(y) y_b`, per image, inside the image sum
//! (`d_a` is the Cartesian gradient, the sign is PLUS):
//!
//! ```text
//! A_ab,mu(r_i) = sum_T e^{i k.T} (d_a phi_mu)(y) y_b,          y = r_i - c_mu - T
//! ```
//!
//! The weight `y_b` differs from image to image, so the tangent cannot be formed from the summed
//! derivatives of [`super::pulay::KSamples`]: it is one pass of the Bloch kernel with a weight axis
//! ([`crate::kernels::bloch_shell_weighted`]). A tangent exists for every ordered pair `(a, b)`
//! (nine, not six: `A_ab` is not symmetric), each `M x N` complex and formed on demand, so a
//! caller that holds the values and one tangent at a time stays at three times the values.
//! The sphere form (`A_ab - s_b G_a`, the quadrature points rigid about their atom) is in
//! [`super::strain_spheres`].
use super::{
    images::Images,
    pulay::Mesh,
    shells::{DenseRows, RowSink, Sampling, Shells},
};
use crate::{
    kernels::{self, ShellImages},
    product, Buffer, Context, Error, Result,
};
use nalgebra::Matrix3;
use num_complex::Complex64;

/// Bytes of one tangent (`functions x points` complex): what [`StrainSamples::tangent`] reserves.
pub fn tangent_bytes(functions: usize, points: usize) -> Result<usize> {
    product(&[functions, points, 16])
}

/// The 1-Bohr bin of point `j`, ordered by `x`, then `y`, then `z` (the order `Shells` samples in,
/// so the kernel can cull images per block of consecutive points).
fn bin(points: [&[f64]; 3], j: usize) -> (i64, i64, i64) {
    (
        points[0][j].floor() as i64,
        points[1][j].floor() as i64,
        points[2][j].floor() as i64,
    )
}

/// `sum_T e^{i k.T} (d_axis phi_mu)(y) y_weight` of every function of `shells` at `points`, one
/// `(k, function)` row at a time into `sink`, in the caller's point order. The image sum and the
/// orchestration are those of [`Shells::derivatives`]; only the kernel call differs.
pub fn weighted_derivatives(
    ctx: &Context,
    shells: &Shells,
    sampling: &Sampling<'_>,
    points: [&[f64]; 3],
    axis: usize,
    weight: usize,
    sink: &mut dyn RowSink,
) -> Result<()> {
    if axis > 2 || weight > 2 {
        return Err(Error::Invalid(
            "a strain tangent needs a derivative axis and a weight component in 0..=2".into(),
        ));
    }
    let n = points[0].len();
    if n == 0 || points.iter().any(|p| p.len() != n) {
        return Err(Error::Invalid(
            "Bloch sums need points of equal, positive count".into(),
        ));
    }
    let nk = sampling.kpoints.len();
    if nk == 0 {
        return Err(Error::Invalid("Bloch sums need a k-point".into()));
    }
    let _order_bytes = ctx.reserve(product(&[n, 32])?)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&j| bin(points, j));
    let sorted: Vec<Vec<f64>> = points
        .iter()
        .map(|axis| order.iter().map(|&j| axis[j]).collect())
        .collect();
    let widest = shells
        .shells()
        .iter()
        .map(|s| s.ms.len())
        .max()
        .unwrap_or(0);
    let mut scratch = ctx.allocate::<Complex64>(product(&[nk, widest, n])?)?;
    let mut row = ctx.allocate::<Complex64>(n)?;
    for shell in shells.shells() {
        let nm = shell.ms.len();
        let images = Images::around(
            ctx,
            sampling.lattice,
            &sampling.region,
            shell.center,
            shell.radial.support(),
        )?;
        if images.is_empty() {
            continue;
        }
        let phases = images.phases(ctx, sampling.kpoints, 1.)?;
        let spline = shell.radial.spline(shell.l)?;
        let local: Vec<i64> = (0..nm as i64).collect();
        let out = &mut scratch.as_mut_slice()[..nk * nm * n];
        out.fill(Complex64::new(0., 0.));
        kernels::bloch_shell_weighted(
            ctx,
            &ShellImages {
                xyz: [&sorted[0], &sorted[1], &sorted[2]],
                center: shell.center,
                images: images.vectors(),
                phases: phases.as_slice(),
                support: shell.radial.support(),
                spline: &spline,
                ms: &shell.ms,
                derivative: axis as i32,
            },
            weight,
            &local,
            nm,
            out,
        )?;
        for k in 0..nk {
            for (i, &destination) in shell.rows.iter().enumerate() {
                let source = &out[(k * nm + i) * n..(k * nm + i + 1) * n];
                for (value, &j) in source.iter().zip(&order) {
                    row.as_mut_slice()[j] = *value;
                }
                sink.put(k, destination, row.as_slice())?;
            }
        }
    }
    Ok(())
}

/// The strain tangents of every function at ONE k-point on the mesh. It holds no sample: each
/// [`StrainSamples::tangent`] call forms one component and the caller drops it before the next.
pub struct StrainSamples<'a> {
    shells: &'a Shells,
    lattice: &'a Matrix3<f64>,
    mesh: &'a Mesh,
    k: [f64; 3],
}

impl<'a> StrainSamples<'a> {
    /// `k` is the Cartesian wave vector of the unstrained cell; the fractional k-point is the one
    /// held under strain.
    pub fn new(shells: &'a Shells, lattice: &'a Matrix3<f64>, mesh: &'a Mesh, k: [f64; 3]) -> Self {
        Self {
            shells,
            lattice,
            mesh,
            k,
        }
    }
    pub fn functions(&self) -> usize {
        self.shells.functions()
    }
    /// `A_ab,mu(r_i)`, `(function, node)` row-major; the buffer is charged to the `Context` until
    /// it is dropped.
    pub fn tangent(&self, ctx: &Context, a: usize, b: usize) -> Result<Buffer<Complex64>> {
        let (m, n) = (self.shells.functions(), self.mesh.len());
        let mut out = ctx.allocate::<Complex64>(product(&[m, n])?)?;
        weighted_derivatives(
            ctx,
            self.shells,
            &Sampling {
                lattice: self.lattice,
                region: self.mesh.region(),
                kpoints: &[self.k],
            },
            self.mesh.points(),
            a,
            b,
            &mut DenseRows::new(out.as_mut_slice(), m, n),
        )?;
        Ok(out)
    }
}
