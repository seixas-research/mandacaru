//! Bloch sums of atom-centered shells.
//!
//! A shell is one radial table at one center with every magnetic component `m` it carries;
//! its Bloch sum at the Cartesian wave vector `k`,
//!
//! ```text
//! chi_{m,k}(r) = sum_R e^{i k.R} R_l(|r - c - R|) Y_lm(r - c - R)      (complex Condon-Shortley Y)
//! ```
//!
//! is formed one shell at a time into a buffer of its own rows, `(k, m, point)`, and handed to a
//! [`RowSink`] row by row in the caller's point order. A shell owns disjoint rows of the basis,
//! so the sink rounds each stored row once: the float32 path never holds a float64 copy of all
//! `M` rows, and no kernel exists for single-precision complex values.
//!
//! Only the images that reach the region are summed ([`Images`]); the points are visited in
//! 1-Bohr bins, which does not change a value but lets the kernel cull images per block of 256
//! consecutive points.
use super::images::{reaching, Images, Region};
use crate::{basis::table, kernels, kernels::ShellImages, product, Buffer, Context, Error, Result};
use nalgebra::Matrix3;
use num_complex::Complex64;
use std::sync::Arc;

/// A tabulated radial function as the Bloch kernels read it: the breakpoints, the piecewise
/// cubic of the not-a-knot spline of the table (interval-major, highest power first), and the
/// radius beyond which the function counts as zero.
#[derive(Debug)]
pub struct Radial {
    breaks: Buffer<f64>,
    coefficients: Buffer<f64>,
    cutoff: f64,
    support: f64,
}

/// A table is zero beyond the last point where it exceeds this fraction of its peak.
pub const SUPPORT_TOLERANCE: f64 = 1e-12;

impl Radial {
    /// `r`, `values`: the table (strictly increasing `r`, at least four nodes). `confinement` is
    /// the radius at which a confined orbital is cut (`None`: the end of the table).
    pub fn new(ctx: &Context, r: &[f64], values: &[f64], confinement: Option<f64>) -> Result<Self> {
        let by_power = table::coefficients(ctx, r, values)?;
        let intervals = r.len() - 1;
        let mut coefficients = ctx.allocate::<f64>(4 * intervals)?;
        for (i, row) in coefficients.as_mut_slice().chunks_mut(4).enumerate() {
            for (p, c) in row.iter_mut().enumerate() {
                *c = by_power[p * intervals + i];
            }
        }
        let peak = values.iter().fold(0f64, |a, v| a.max(v.abs()));
        let support = match values
            .iter()
            .rposition(|v| v.abs() > SUPPORT_TOLERANCE * peak)
        {
            Some(last) => r[(last + 1).min(r.len() - 1)],
            None => r[0],
        };
        let cutoff = confinement.unwrap_or(r[r.len() - 1]);
        if !(cutoff.is_finite() && cutoff > 0.) {
            return Err(Error::Invalid(
                "the confinement radius must be positive and finite".into(),
            ));
        }
        Ok(Self {
            breaks: ctx.copy_slice(r)?,
            coefficients,
            cutoff,
            support,
        })
    }
    /// Radius beyond which the function is taken as zero: the last point above
    /// [`SUPPORT_TOLERANCE`] times the peak, plus one node.
    pub fn support(&self) -> f64 {
        self.support
    }
    /// The last node of the table.
    pub fn extent(&self) -> f64 {
        self.breaks.as_slice()[self.breaks.as_slice().len() - 1]
    }
    /// The kernel's view of the spline for angular momentum `l`.
    pub fn spline(&self, l: usize) -> Result<kernels::Spline<'_>> {
        kernels::Spline::new(
            self.breaks.as_slice(),
            self.coefficients.as_slice(),
            self.cutoff,
            i32::try_from(l).map_err(|_| Error::Overflow)?,
        )
    }
}

/// The components `ms` of one radial table at one center; `rows[i]` is where `ms[i]` sits in the
/// basis.
#[derive(Clone, Debug)]
pub struct Shell {
    pub center: [f64; 3],
    pub l: usize,
    pub ms: Vec<i32>,
    pub rows: Vec<usize>,
    pub radial: Arc<Radial>,
}

/// The cell the Bloch sums are taken in: the lattice (columns), the ball that bounds the points
/// and the Cartesian k-points.
#[derive(Clone, Copy, Debug)]
pub struct Sampling<'a> {
    pub lattice: &'a Matrix3<f64>,
    pub region: Region,
    pub kpoints: &'a [[f64; 3]],
}

/// Where the rows of a shell go. `values` is one `(k, row)` of `points` Bloch-sum values in the
/// order the points were given; a row no image reaches is never put and stays what the sink
/// started as (zero).
pub trait RowSink {
    fn put(&mut self, k: usize, row: usize, values: &[Complex64]) -> Result<()>;
}

/// A float64 `(k, function, point)` array.
pub struct DenseRows<'a> {
    out: &'a mut [Complex64],
    functions: usize,
    points: usize,
}
impl<'a> DenseRows<'a> {
    pub fn new(out: &'a mut [Complex64], functions: usize, points: usize) -> Self {
        Self {
            out,
            functions,
            points,
        }
    }
}
impl RowSink for DenseRows<'_> {
    fn put(&mut self, k: usize, row: usize, values: &[Complex64]) -> Result<()> {
        let start = (k * self.functions + row) * self.points;
        self.out[start..start + self.points].copy_from_slice(values);
        Ok(())
    }
}

/// The basis of a cell as shells whose rows cover `0..functions` exactly once.
#[derive(Clone, Debug)]
pub struct Shells {
    shells: Vec<Shell>,
    functions: usize,
}

/// The functions of a [`Shells`] that reach a sphere, renumbered `0..rows.len()`; `rows` are
/// their positions in the full basis.
#[derive(Debug)]
pub struct Reached {
    pub rows: Vec<usize>,
    pub shells: Shells,
}

impl Shells {
    pub fn new(shells: Vec<Shell>) -> Result<Self> {
        let functions: usize = shells.iter().map(|s| s.ms.len()).sum();
        let mut seen = vec![false; functions];
        for s in &shells {
            if s.ms.is_empty()
                || s.ms.len() != s.rows.len()
                || s.ms.iter().any(|m| m.unsigned_abs() as usize > s.l)
                || s.center.iter().any(|x| !x.is_finite())
            {
                return Err(Error::Invalid(
                    "a shell needs components |m| <= l, one row each and a finite center".into(),
                ));
            }
            for &row in &s.rows {
                if row >= functions || std::mem::replace(&mut seen[row], true) {
                    return Err(Error::Invalid(
                        "shell rows must be distinct and cover the basis".into(),
                    ));
                }
            }
        }
        Ok(Self { shells, functions })
    }
    pub fn functions(&self) -> usize {
        self.functions
    }
    pub fn shells(&self) -> &[Shell] {
        &self.shells
    }
    /// The shells with an image that reaches `region`, renumbered, and their old rows.
    pub fn reaching(
        &self,
        ctx: &Context,
        lattice: &Matrix3<f64>,
        region: &Region,
    ) -> Result<Reached> {
        let bodies: Vec<_> = self
            .shells
            .iter()
            .map(|s| (s.center, s.radial.support()))
            .collect();
        let verdict = reaching(ctx, lattice, region, &bodies)?;
        let mut kept: Vec<(usize, usize)> = Vec::new();
        for (shell, keep) in self.shells.iter().zip(&verdict) {
            if *keep {
                kept.extend(shell.rows.iter().map(|&row| (row, 0)));
            }
        }
        kept.sort_unstable();
        let rows: Vec<usize> = kept.iter().map(|k| k.0).collect();
        let position = |row: usize| rows.binary_search(&row).expect("a kept row");
        let shells = self
            .shells
            .iter()
            .zip(&verdict)
            .filter(|(_, keep)| **keep)
            .map(|(s, _)| Shell {
                rows: s.rows.iter().map(|&row| position(row)).collect(),
                ..s.clone()
            })
            .collect();
        Ok(Reached {
            shells: Shells::new(shells)?,
            rows,
        })
    }
    /// Bloch sums of every shell at `points`, row by row into `sink`.
    pub fn values(
        &self,
        ctx: &Context,
        sampling: &Sampling<'_>,
        points: [&[f64]; 3],
        sink: &mut dyn RowSink,
    ) -> Result<()> {
        self.sample(ctx, sampling, points, None, sink)
    }
    /// The Cartesian derivative `d chi / d x_axis` instead of the values (`axis` 0, 1, 2).
    pub fn derivatives(
        &self,
        ctx: &Context,
        sampling: &Sampling<'_>,
        points: [&[f64]; 3],
        axis: usize,
        sink: &mut dyn RowSink,
    ) -> Result<()> {
        if axis > 2 {
            return Err(Error::Invalid("the derivative axis is 0, 1 or 2".into()));
        }
        self.sample(ctx, sampling, points, Some(axis), sink)
    }
    fn sample(
        &self,
        ctx: &Context,
        sampling: &Sampling<'_>,
        points: [&[f64]; 3],
        axis: Option<usize>,
        sink: &mut dyn RowSink,
    ) -> Result<()> {
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
        // The points in 1-Bohr bins (stable), and the way back.
        let _order_bytes = ctx.reserve(product(&[n, 32])?)?;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&j| bin(points, j));
        let sorted: Vec<Vec<f64>> = points
            .iter()
            .map(|axis| order.iter().map(|&j| axis[j]).collect())
            .collect();
        let widest = self.shells.iter().map(|s| s.ms.len()).max().unwrap_or(0);
        let mut scratch = ctx.allocate::<Complex64>(product(&[nk, widest, n])?)?;
        let mut row = ctx.allocate::<Complex64>(n)?;
        for shell in &self.shells {
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
            kernels::bloch_shell(
                ctx,
                &ShellImages {
                    xyz: [&sorted[0], &sorted[1], &sorted[2]],
                    center: shell.center,
                    images: images.vectors(),
                    phases: phases.as_slice(),
                    support: shell.radial.support(),
                    spline: &spline,
                    ms: &shell.ms,
                    derivative: axis.map_or(-1, |a| a as i32),
                },
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
}

/// The 1-Bohr bin of point `j`, ordered by `x`, then `y`, then `z`.
fn bin(points: [&[f64]; 3], j: usize) -> (i64, i64, i64) {
    (
        points[0][j].floor() as i64,
        points[1][j].floor() as i64,
        points[2][j].floor() as i64,
    )
}
